import ast
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import pickle
import re
import warnings
from pathlib import Path
from textwrap import dedent

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter1d
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
        "ROD_QZ_NONLINEAR_REFINEMENT_ENABLED": True,
        "ROD_QZ_NONLINEAR_MAX_COMPONENTS": 8,
        "ROD_QZ_NONLINEAR_CENTER_BOUND_BINS": 4.0,
        "ROD_QZ_NONLINEAR_TAIL_POWER_BOUNDS": (0.55, 12.0),
        "ROD_QZ_NONLINEAR_MAX_NFEV": 1200,
        "ROD_QZ_NONLINEAR_LOG_RESIDUAL_WEIGHT": 1.0,
        "ROD_QZ_NONLINEAR_LOG_FLOOR_FRACTION": 0.05,
        "ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED": True,
    }
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def _script_functions(*names: str) -> dict[str, object]:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    wanted = set(names)
    if "show_qr_rod_peak_marker_popup" in wanted:
        wanted.add("hk0_qz_marker_count")
        wanted.add("qr_rod_l_bounds_for_group")
        wanted.add("qr_rod_editor_phase_matches")
        wanted.add("qr_rod_editor_phase_title")
        wanted.add("merge_qr_rod_editor_phase_table")
        wanted.add("qr_rod_editor_profile_rows_from_markers")
        wanted.add("l_reference_rows")
        wanted.add("qz_to_l_linear_coeff")
        wanted.add("rod_profile_marker_l_mapping_is_valid")
    if wanted & {"l_reference_rows", "qz_values_to_l_axis", "qz_to_l_linear_coeff"}:
        wanted.add("rod_profile_marker_l_mapping_is_valid")
    if wanted & {"qr_rod_pre_editor_stage_is_valid", "qr_rod_profile_cache_has_final_fit"}:
        wanted.add("hk0_qz_marker_count")
        wanted.add("rod_profile_has_real_hk0_qz_rows")
    if wanted & {
        "apply_radial_background_subtraction_to_detector",
        "caked_background_plane_from_image",
        "radial_background_model_from_detector",
        "radial_background_subtraction_policy_signature",
        "show_radial_background_subtraction_popup",
    }:
        wanted.add("radial_background_subtraction_config_from_state")
        wanted.add("_radial_background_subtraction_config_from_mapping")
        wanted.add("_radial_background_phi_delta_deg")
        wanted.add("caked_background_plane_from_image")
        wanted.add("radial_background_model_from_detector")
        wanted.add("radial_background_profile_from_detector")
    tree = ast.parse(
        PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(PARALLEL_SCRIPT_PATH)
    )
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
        "pickle": pickle,
        "replace": replace,
        "re": re,
        "plt": plt,
        "LogNorm": LogNorm,
        "binary_dilation": binary_dilation,
        "gaussian_filter1d": gaussian_filter1d,
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
        "ROD_QZ_NONLINEAR_LOG_RESIDUAL_WEIGHT": 1.0,
        "ROD_QZ_NONLINEAR_LOG_FLOOR_FRACTION": 0.05,
        "ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED": True,
        "QR_ROD_FINAL_FIT_CACHE_SIGNATURE": "joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane",
        "PRE_EDITOR_CACHE_SCHEMA": "ra_sim.background_pre_editor_cache.v1",
        "PRE_EDITOR_CACHE_SIGNATURE": "pre_qr_rod_marker_editor_inputs_v1",
        "PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE": "background_peak_fit_results_v5_caked_phi_m90_90_plane",
        "PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE": "profile_fit_cache_v1",
        "PRE_EDITOR_QR_ROD_STAGE_SIGNATURE": "qr_rod_pre_marker_profiles_hk0_roi_v17_caked_phi_m90_90_plane",
        "QR_ROD_BG_SIDE_BAND_INNER_SCALE": 1.30,
        "QR_ROD_BG_SIDE_BAND_OUTER_SCALE": 2.80,
        "QR_ROD_BG_MIN_SIDE_PIXELS": 8,
        "QR_ROD_BG_PERCENTILE": 50.0,
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


def test_detector_region_labels_use_m_not_hk_or_material() -> None:
    namespace = _script_functions(
        "detector_rod_label",
        "detector_specular_label",
    )
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    rod_label = namespace["detector_rod_label"]
    specular_label = namespace["detector_specular_label"]

    assert rod_label(7, "+") == "m = 7 +"
    assert rod_label(7, "-") == "m = 7 -"
    assert specular_label() == "m = 0"
    assert "detector_rod_material_label(" not in source
    assert "detector_specular_material_label(" not in source


def test_detector_region_delta_q_boundaries_use_dashed_contours() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "detector_region_boundary_linestyle" in source
    assert "ax.contour(" in source
    assert "linestyles=detector_region_boundary_linestyle" in source
    assert "linewidths=detector_region_boundary_lw" in source
    assert "draw_detector_mask_layer(\n            boundary_mask" not in source


def test_detector_region_note_describes_white_m_labels_and_dashed_boundaries() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "dashed boundary strokes" in source
    assert "solid-white m labels" in source
    assert "material/source labels" not in source
    assert "intensity scale is saved as a separate file" in source
    assert "dashed edge pixels" not in source


def test_detector_region_saves_intensity_scale_as_separate_file() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_start = source.index(
        "fig, ax = plt.subplots(figsize=(JOURNAL_FULL_WIDTH_IN, fig_height)"
    )
    detector_figure_end = source.index(
        "detector_region_png, detector_region_pdf = save_manuscript_figure",
        detector_figure_start,
    )
    detector_figure_source = source[detector_figure_start:detector_figure_end]

    assert "fig.colorbar(" not in detector_figure_source
    assert "detector_region_cbar" not in detector_figure_source
    assert "def save_detector_region_intensity_scale(" in source
    assert 'stem=f"{ROD_PROFILE_REGION_STEM}_intensity_scale"' in source
    assert "detector_region_scale_png, detector_region_scale_pdf" in source


def test_parallel_script_single_background_figures_keep_row_column_axes() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    detector_start = source.index(
        "fig, axes = plt.subplots(\n    len(ordered_backgrounds),\n    2,"
    )
    detector_setup = source[
        detector_start : source.index(
            "for row, bg in enumerate(ordered_backgrounds):", detector_start
        )
    ]
    assert (
        "axes = np.asarray(axes, dtype=object).reshape(len(ordered_backgrounds), 2)"
        in detector_setup
    )

    profile_start = source.index(
        "fig, axes = plt.subplots(\n    len(ordered_backgrounds),\n    len(columns),"
    )
    profile_setup = source[
        profile_start : source.index(
            "for row, bg in enumerate(ordered_backgrounds):", profile_start
        )
    ]
    assert (
        "axes = np.asarray(axes, dtype=object).reshape(len(ordered_backgrounds), len(columns))"
        in profile_setup
    )


def test_parallel_script_active_lattice_prefers_saved_state_variables() -> None:
    namespace = _script_functions(
        "as_float",
        "positive_float_or_nan",
        "active_lattice_constants_from_state",
    )
    active_lattice = namespace["active_lattice_constants_from_state"]

    lattice = active_lattice(
        {
            "variables": {
                "a_var": "4.59",
                "c_var": 6.78,
            },
            "files": {"primary_cif_path": "ignored.cif"},
        }
    )

    assert lattice == {
        "a": pytest.approx(4.59),
        "c": pytest.approx(6.78),
        "source": "state.variables",
        "primary_cif_path": "ignored.cif",
    }


def test_parallel_script_active_lattice_falls_back_to_primary_cif(tmp_path: Path) -> None:
    namespace = _script_functions(
        "as_float",
        "positive_float_or_nan",
        "active_lattice_constants_from_cif_path",
        "active_lattice_constants_from_state",
    )
    active_lattice = namespace["active_lattice_constants_from_state"]
    cif_path = tmp_path / "PbI2.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_pbi2",
                "_cell_length_a    '4.590(2)'",
                '_cell_length_c    "6.780(3)"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lattice = active_lattice(
        {
            "variables": {"a_var": "", "c_var": ""},
            "files": {"primary_cif_path": str(cif_path)},
        }
    )

    assert lattice["a"] == pytest.approx(4.59)
    assert lattice["c"] == pytest.approx(6.78)
    assert lattice["source"] == "primary_cif_path"
    assert lattice["primary_cif_path"] == str(cif_path)


def test_parallel_script_active_lattice_rejects_missing_values() -> None:
    namespace = _script_functions(
        "as_float",
        "positive_float_or_nan",
        "active_lattice_constants_from_cif_path",
        "active_lattice_constants_from_state",
    )
    active_lattice = namespace["active_lattice_constants_from_state"]

    with pytest.raises(ValueError, match="active lattice constants unavailable"):
        active_lattice({"variables": {"a_var": "0", "c_var": "nan"}, "files": {}})


def test_parallel_script_active_lattice_cache_signature_changes_with_lattice_values() -> None:
    namespace = _script_functions(
        "_cache_normalize_value",
        "as_float",
        "positive_float_or_nan",
        "active_lattice_constants_from_state",
        "file_cache_signature",
        "active_lattice_cache_signature",
    )
    signature = namespace["active_lattice_cache_signature"]

    base = {
        "variables": {"a_var": "4.59", "c_var": "6.78", "unrelated": "before"},
        "files": {"primary_cif_path": "PbI2.cif"},
    }
    same_lattice = {
        "variables": {"a_var": "4.59", "c_var": "6.78", "unrelated": "after"},
        "files": {"primary_cif_path": "PbI2.cif"},
    }
    changed_a = {
        "variables": {"a_var": "5.0", "c_var": "6.78", "unrelated": "before"},
        "files": {"primary_cif_path": "PbI2.cif"},
    }
    changed_c = {
        "variables": {"a_var": "4.59", "c_var": "7.0", "unrelated": "before"},
        "files": {"primary_cif_path": "PbI2.cif"},
    }

    assert signature(base) == signature(same_lattice)
    assert signature(base) != signature(changed_a)
    assert signature(base) != signature(changed_c)


def test_parallel_script_q_group_rows_cache_signature_changes_with_q_values() -> None:
    namespace = _script_functions("_cache_normalize_value", "q_group_rows_cache_signature")
    signature = namespace["q_group_rows_cache_signature"]
    base = {
        "geometry": {
            "q_group_rows": [
                {
                    "key": ["q_group", "primary", 1, 0],
                    "source_label": "primary",
                    "included": True,
                    "qr": 1.58,
                    "qz": 0.0,
                }
            ]
        }
    }
    changed_qr = {
        "geometry": {
            "q_group_rows": [
                {
                    "key": ["q_group", "primary", 1, 0],
                    "source_label": "primary",
                    "included": True,
                    "qr": 1.7,
                    "qz": 0.0,
                }
            ],
            "unrelated": "ignored",
        }
    }

    assert signature(base) != signature(changed_qr)


def test_parallel_script_qr_rod_final_cache_key_accepts_reference_signature() -> None:
    namespace = _script_functions(
        "_cache_normalize_value",
        "clean_marker_title",
        "qr_rod_peak_edit_cache_key",
    )
    cache_key = namespace["qr_rod_peak_edit_cache_key"]

    first = cache_key(
        None,
        lattice_signature={"a": 4.59, "c": 6.78},
        q_group_signature=[{"key": ["q_group", "primary", 1, 0], "qr": 1.58}],
    )
    second = cache_key(
        None,
        lattice_signature={"a": 5.0, "c": 6.78},
        q_group_signature=[{"key": ["q_group", "primary", 1, 0], "qr": 1.58}],
    )
    third = cache_key(
        None,
        lattice_signature={"a": 4.59, "c": 6.78},
        q_group_signature=[{"key": ["q_group", "primary", 1, 0], "qr": 1.7}],
    )

    assert first != second
    assert first != third
    assert first != cache_key(
        None,
        lattice_signature={"a": 4.59, "c": 6.78},
        q_group_signature=[{"key": ["q_group", "primary", 1, 0], "qr": 1.58}],
        rod_reference_policy={"allow_generated": True},
    )


def test_parallel_script_wires_lattice_and_q_groups_into_cache_keys() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "ACTIVE_LATTICE = active_lattice_constants_from_state(state)" in source
    assert '"active_lattice": ACTIVE_LATTICE_CACHE_SIGNATURE' in source
    assert '"q_group_rows": Q_GROUP_ROWS_CACHE_SIGNATURE' in source
    assert '"rod_reference_policy": ROD_REFERENCE_POLICY_SIGNATURE' in source
    assert "lattice_signature=ACTIVE_LATTICE_CACHE_SIGNATURE" in source
    assert "q_group_signature=Q_GROUP_ROWS_CACHE_SIGNATURE" in source
    assert "rod_reference_policy=ROD_REFERENCE_POLICY_SIGNATURE" in source


def test_parallel_script_primary_rod_entries_use_saved_q_groups_by_default() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qr_value_for_m",
        "derived_primary_rod_entry_for_m",
        "profile_rod_entries_from_q_group_rows",
    )
    build_entries = namespace["profile_rod_entries_from_q_group_rows"]

    rods = build_entries(
        [
            {
                "key": ["q_group", "primary", 1, 0],
                "source_label": "primary",
                "included": True,
                "qr": 9.0,
                "qz": 0.0,
            }
        ],
        candidate_m_values=[1, 3],
        lattice_a=4.59,
    )

    by_m = {int(row["m"]): row for row in rods}
    assert sorted(by_m) == [1]
    assert by_m[1]["qr"] == 9.0
    assert by_m[1]["qr_source"] == "saved_q_group_rows"
    assert by_m[1]["generated"] is False
    assert by_m[1]["source_label"] == "primary"
    assert by_m[1]["q_group_key"] == ("q_group", "primary", 1, 0)


def test_parallel_script_generated_primary_rod_entries_require_explicit_gate() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qr_value_for_m",
        "derived_primary_rod_entry_for_m",
        "profile_rod_entries_from_q_group_rows",
    )
    build_entries = namespace["profile_rod_entries_from_q_group_rows"]

    rods = build_entries(
        [
            {
                "key": ["q_group", "primary", 1, 0],
                "source_label": "primary",
                "included": True,
                "qr": 9.0,
                "qz": 0.0,
            }
        ],
        candidate_m_values=[1, 3],
        lattice_a=4.59,
        allow_generated=True,
    )

    by_m = {int(row["m"]): row for row in rods}
    assert sorted(by_m) == [1, 3]
    assert by_m[3]["qr"] == pytest.approx((2.0 * np.pi / 4.59) * np.sqrt(4.0))
    assert by_m[3]["qr_source"] == "active_lattice"
    assert by_m[3]["generated"] is True
    assert "q_group_key" not in by_m[3]


def test_parallel_script_build_profile_rods_disables_generated_fallback_by_default() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "ALLOW_GENERATED_ROD_REFERENCES = _truthy_setting" in source
    assert "allow_generated=ALLOW_GENERATED_ROD_REFERENCES" in source


def test_parallel_script_detector_overlay_filters_generated_rods_by_default() -> None:
    namespace = _script_functions(
        "detector_complete_branch_rod_entries",
        "detector_overlay_rod_entries",
    )
    overlay_rods = namespace["detector_overlay_rod_entries"]
    rods = [
        {"m": 1, "qr": 1.0, "generated": False},
        {"m": 3, "qr": 2.0, "generated": True},
        {"m": 4, "qr": 3.0},
        {"m": 7, "qr": 4.0},
    ]
    region_overlays = [
        {"m": 1, "source": "", "branch": "-", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 1, "source": "", "branch": "+", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 4, "source": "", "branch": "-", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 4, "source": "", "branch": "+", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 7, "source": "", "branch": "-", "qz_min": 0.2, "qz_max": 1.2},
    ]

    saved_only = overlay_rods(rods, region_overlays=region_overlays)
    assert [int(row["m"]) for row in saved_only] == [1, 4]
    assert [
        int(row["m"])
        for row in overlay_rods(rods, region_overlays=region_overlays, allow_generated=True)
    ] == [1, 4]

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("rod_label_entries: list[dict[str, object]] = []") : source.index(
            "specular_color = OKABE_ITO"
        )
    ]
    assert "detector_overlay_rods = detector_overlay_rod_entries(" in detector_figure_source
    assert "region_overlays=region_overlays" in detector_figure_source


def test_parallel_script_final_rod_plot_filters_incomplete_detector_branch_support() -> None:
    namespace = _script_functions("detector_complete_branch_rod_entries")
    complete_rods = namespace["detector_complete_branch_rod_entries"]
    rods = [
        {"m": 1, "source": "primary"},
        {"m": 7, "source": "primary"},
    ]
    region_overlays = [
        {"m": 1, "source": "primary", "branch": "-", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 1, "source": "primary", "branch": "+", "qz_min": 0.2, "qz_max": 1.2},
        {"m": 7, "source": "primary", "branch": "-", "qz_min": 0.2, "qz_max": 1.2},
    ]

    assert [int(row["m"]) for row in complete_rods(rods, region_overlays)] == [1]

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_profile_source = source[
        source.index("drawable_profile_keys = drawable_rod_profile_keys(") : source.index(
            "\nif not plot_rod_entries:"
        )
    ]
    assert (
        "detector_plot_rod_entries = detector_complete_branch_rod_entries(" in final_profile_source
    )
    assert "skipped_incomplete_detector_hk" in final_profile_source


def test_parallel_script_pbi2_hidden_rods_are_removed_before_artifacts() -> None:
    namespace = _script_functions(
        "qr_rod_hidden_hk_values_for_sample",
        "filter_hidden_qr_rod_entries",
        "filter_hidden_qr_rod_table",
    )
    hidden_values = namespace["qr_rod_hidden_hk_values_for_sample"]
    filter_entries = namespace["filter_hidden_qr_rod_entries"]
    filter_table = namespace["filter_hidden_qr_rod_table"]

    rods = [
        {"m": 0, "source": "specular"},
        {"m": 4, "source": "primary"},
        {"m": 7, "source": "primary"},
    ]
    table = pd.DataFrame(
        {
            "m": [0, 4, 7],
            "branch": ["qz", "-", "+"],
            "qz_center": [0.5, 1.0, 1.5],
        }
    )

    assert hidden_values("pbi2", hidden_hk=(7,)) == {7}
    assert hidden_values("Bi2Se3", hidden_hk=(7,)) == set()
    assert [int(row["m"]) for row in filter_entries(rods, "pbi2", hidden_hk=(7,))] == [0, 4]
    assert [int(row["m"]) for row in filter_entries(rods, "Bi2Se3", hidden_hk=(7,))] == [
        0,
        4,
        7,
    ]
    assert filter_table(table, "pbi2", hidden_hk=(7,))["m"].astype(int).tolist() == [0, 4]
    assert filter_table(table, "Bi2Se3", hidden_hk=(7,))["m"].astype(int).tolist() == [
        0,
        4,
        7,
    ]

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"pbi2_hidden_hk":' in source
    assert "rod_entries = filter_hidden_qr_rod_entries(" in source
    assert "rod_profile_table = filter_hidden_qr_rod_table(" in source


def test_parallel_script_reports_rod_reference_sources_and_skipped_generated_rods() -> None:
    namespace = _script_functions("rod_reference_source_summary")
    summarize = namespace["rod_reference_source_summary"]

    summary = summarize(
        [
            {"m": 1, "generated": False},
            {"m": 3, "generated": True},
        ],
        candidate_m_values=[1, 3, 4],
        allow_generated=False,
    )

    assert summary == {
        "saved": 1,
        "generated": 1,
        "skipped_generated": 1,
        "allow_generated": False,
    }

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "rod references: saved=" in source
    assert "Rod reference policy:" in source
    assert "| HK | source | generated | saved Qr |" in source


def test_parallel_script_requires_broad_anchor_coverage_before_detector_qr_refit() -> None:
    namespace = _script_functions(
        "detector_rotation_anchor_summary",
        "detector_rotation_fit_has_anchor_coverage",
    )
    summarize = namespace["detector_rotation_anchor_summary"]
    has_coverage = namespace["detector_rotation_fit_has_anchor_coverage"]
    pbi2_like_points = [
        {"target_source_label": "primary", "m": 1},
        {"target_source_label": "primary", "m": 1},
    ]
    broad_points = [
        {"target_source_label": "primary", "m": 1},
        {"target_source_label": "primary", "m": 1},
        {"target_source_label": "primary", "m": 3},
        {"target_source_label": "primary", "m": 3},
    ]

    assert summarize(pbi2_like_points) == {"anchor_count": 2, "anchor_m_group_count": 1}
    assert has_coverage(pbi2_like_points) is False
    assert has_coverage(broad_points) is True
    assert has_coverage(broad_points, active_mask=[True, True, False, False]) is False

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "insufficient detector-rotation anchor coverage" in source
    assert 'if bool(rod_qspace_calibration.get("success", False)):' in source


def test_parallel_script_qr_sideband_helpers_bin_same_qz_background() -> None:
    namespace = _script_functions("_binned_nanpercentile", "_sideband_qr_mask")
    binned_nanpercentile = namespace["_binned_nanpercentile"]
    sideband_qr_mask = namespace["_sideband_qr_mask"]

    qz = np.asarray([0.2, 0.3, 1.2, 1.3, 1.4], dtype=np.float64)
    values = np.asarray([10.0, 12.0, 20.0, 22.0, 24.0], dtype=np.float64)
    background, counts = binned_nanpercentile(
        qz,
        values,
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        percentile=50.0,
        min_count=2,
    )

    np.testing.assert_allclose(background, np.asarray([11.0, 22.0], dtype=np.float64))
    np.testing.assert_array_equal(counts, np.asarray([2, 3], dtype=np.int64))

    qr_map = np.asarray([[0.0, 0.7, 0.8, 1.0, 1.13, 1.2, 1.3, 1.6]], dtype=np.float64)
    mask, inner, outer = sideband_qr_mask(qr_map, qr0=1.0, delta_qr=0.1)

    assert inner == pytest.approx(0.13)
    assert outer == pytest.approx(0.28)
    assert mask.tolist() == [[False, False, True, False, True, True, False, False]]


def test_parallel_script_qr_sideband_profiles_bypass_fast_accumulators_and_keep_raw_columns() -> (
    None
):
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    profile_source = source[
        source.index("def profile_from_detector_qr_qz(") : source.index(
            "\n# Rod-profile baselines are added"
        )
    ]

    assert "QR_ROD_TRANSVERSE_BACKGROUND_ENABLED" in source
    assert "not bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED)" in profile_source
    assert "background_density_raw" in profile_source
    assert "qr_sideband_background_density" in profile_source
    assert "qr_sideband_pixel_count" in profile_source
    assert "detector_qr_band_per_qz_bin_with_qr_sideband_background" in profile_source


def test_parallel_script_background_image_debug_flag_disables_qr_sideband_subtraction() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    settings_source = source[
        source.index("BACKGROUND_IMAGE_SUBTRACTION_DISABLED =") : source.index(
            "\nQR_ROD_BG_SIDE_BAND_INNER_SCALE"
        )
    ]
    signature_source = source[
        source.index("PRE_EDITOR_CACHE_INPUT_SIGNATURE =") : source.index(
            "\nPRE_EDITOR_CACHE_KEY ="
        )
    ]

    assert 'BACKGROUND_IMAGE_SUBTRACTION_DISABLED_OVERRIDE = "1"' in source
    assert "RA_SIM_BACKGROUND_IMAGE_SUBTRACTION_DISABLED" in source
    assert 'PBI2_DISABLE_BACKGROUND_SUBTRACTION_OVERRIDE = ""' in source
    assert "RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION" in source
    assert "BACKGROUND_IMAGE_SUBTRACTION_DISABLED" in settings_source
    assert "not bool(BACKGROUND_IMAGE_SUBTRACTION_DISABLED)" in settings_source
    assert "PBI2_DISABLE_BACKGROUND_SUBTRACTION" in settings_source
    assert "not bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION)" in settings_source
    assert '"background_image_subtraction_disabled"' in signature_source
    assert '"pbi2_disable_background_subtraction"' in signature_source
    assert "PbI2 no-background debug mode" in source


def test_parallel_script_background_image_subtraction_outputs_raw_images_when_disabled() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    assembly_source = source[
        source.index("detector_peak_model = render_detector_peak_model(") : source.index(
            '\n    np.save(OUT_DIR / f"background_{bg_idx:02d}_caked_peak_fit_model.npy"'
        )
    ]
    print_source = source[
        source.index('f"tth_max=') : source.index(
            "\n\npost_elapsed = time.perf_counter() - post_start"
        )
    ]

    assert "if bool(BACKGROUND_IMAGE_SUBTRACTION_DISABLED):" in assembly_source
    assert "caked_peak_subtracted_image = caked_density_image.copy()" in assembly_source
    assert "detector_peak_subtracted_image = detector_image.copy()" in assembly_source
    assert "caked_peak_subtracted_image = caked_density_image - caked_peak_model" in assembly_source
    assert (
        "detector_peak_subtracted_image = detector_image - detector_peak_model" in assembly_source
    )
    assert "background_image_subtraction_disabled=" in print_source


def test_radial_background_config_uses_legacy_state_only_as_defaults() -> None:
    namespace = _script_functions("radial_background_subtraction_config_from_state")
    build_config = namespace["radial_background_subtraction_config_from_state"]

    saved = {
        "background_subtraction_enabled_var": True,
        "background_subtraction_mode_var": "radial",
        "background_subtraction_scale_var": 0.75,
    }

    default_config = build_config(saved)
    override_config = build_config(saved, mode_override="off", scale_override="0")

    assert default_config["mode"] == "radial"
    assert default_config["scale"] == pytest.approx(0.75)
    assert default_config["percentile"] == pytest.approx(20.0)
    assert default_config["smooth_sigma"] == pytest.approx(6.0)
    assert default_config["clip_zero"] is False
    assert override_config["mode"] == "off"
    assert override_config["scale"] == pytest.approx(0.0)


def test_radial_background_profile_uses_lower_percentile_and_ignores_bright_peak() -> None:
    namespace = _script_functions("radial_background_profile_from_detector")
    profile_from_detector = namespace["radial_background_profile_from_detector"]
    theta_map = np.tile(np.arange(5, dtype=np.float64), (20, 1))
    image = 10.0 + theta_map * 10.0
    image[0, 2] = 1000.0

    profile = profile_from_detector(
        image,
        theta_map,
        n_bins=5,
        percentile=20.0,
        min_pixels=2,
        smooth_sigma=0.0,
    )
    bin_index = int(np.argmin(np.abs(profile["theta_centers"] - 2.0)))

    assert profile["radial_profile"][bin_index] == pytest.approx(30.0)
    assert profile["smoothed_profile"][bin_index] == pytest.approx(30.0)


def test_apply_radial_background_subtraction_scale_zero_is_noop() -> None:
    namespace = _script_functions("apply_radial_background_subtraction_to_detector")
    apply_subtraction = namespace["apply_radial_background_subtraction_to_detector"]
    detector_image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    theta_map = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)

    payload = apply_subtraction(
        detector_image,
        theta_map,
        {
            "mode": "radial",
            "scale": 0.0,
            "percentile": 50.0,
            "smooth_sigma": 0.0,
            "clip_zero": False,
        },
    )

    assert payload["enabled"] is False
    np.testing.assert_allclose(payload["corrected_image"], detector_image)
    np.testing.assert_allclose(payload["background_model"], np.zeros_like(detector_image))
    assert not np.shares_memory(payload["corrected_image"], detector_image)


def test_apply_radial_background_subtraction_scale_one_reduces_radial_halo() -> None:
    namespace = _script_functions("apply_radial_background_subtraction_to_detector")
    apply_subtraction = namespace["apply_radial_background_subtraction_to_detector"]
    theta_map = np.tile(np.arange(5, dtype=np.float64), (30, 1))
    detector_image = 100.0 + theta_map * 25.0
    detector_image[0, 2] = 800.0

    payload = apply_subtraction(
        detector_image,
        theta_map,
        {
            "mode": "radial",
            "scale": 1.0,
            "percentile": 20.0,
            "smooth_sigma": 0.0,
            "clip_zero": False,
        },
    )

    assert payload["enabled"] is True
    assert float(np.nanmedian(np.abs(payload["corrected_image"]))) < 1.0
    assert float(np.nanmax(payload["background_model"])) > 0.0
    assert payload["corrected_image"][0, 2] > 0.0


def test_apply_radial_background_subtraction_clip_zero_is_optional() -> None:
    namespace = _script_functions("apply_radial_background_subtraction_to_detector")
    apply_subtraction = namespace["apply_radial_background_subtraction_to_detector"]
    detector_image = np.asarray([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    theta_map = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    config = {
        "mode": "radial",
        "scale": 2.0,
        "percentile": 50.0,
        "smooth_sigma": 0.0,
    }

    unclipped = apply_subtraction(detector_image, theta_map, {**config, "clip_zero": False})
    clipped = apply_subtraction(detector_image, theta_map, {**config, "clip_zero": True})

    assert float(np.nanmin(unclipped["corrected_image"])) < 0.0
    assert float(np.nanmin(clipped["corrected_image"])) == pytest.approx(0.0)


def test_caked_background_plane_fit_uses_only_phi_between_minus90_and90() -> None:
    namespace = _script_functions("caked_background_plane_from_image")
    fit_plane = namespace["caked_background_plane_from_image"]
    theta_axis = np.linspace(0.0, 4.0, 6, dtype=np.float64)
    phi_axis = np.asarray([-120.0, -80.0, 0.0, 80.0, 120.0], dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    caked_image = 12.0 + 3.0 * theta_grid + 0.25 * phi_grid
    caked_image[[0, 4], :] += 500.0

    payload = fit_plane(caked_image, theta_axis, phi_axis)

    fit_mask = np.asarray(payload["fit_mask"], dtype=bool)
    model = np.asarray(payload["background_model"], dtype=np.float64)
    fit_surface = np.asarray(payload["fit_surface"], dtype=np.float64)
    model_surface = np.asarray(payload["model_surface"], dtype=np.float64)
    assert not np.any(fit_mask[[0, 4], :])
    assert np.all(fit_mask[1:4, :])
    assert fit_surface.shape == caked_image.shape
    assert model_surface.shape == caked_image.shape
    np.testing.assert_allclose(fit_surface[fit_mask], caked_image[fit_mask], atol=1.0e-9)
    np.testing.assert_allclose(model_surface[fit_mask], caked_image[fit_mask], atol=1.0e-9)
    assert np.all(~np.isfinite(fit_surface[~fit_mask]))
    assert np.all(~np.isfinite(model_surface[~fit_mask]))
    assert not np.allclose(model_surface[1, :], model_surface[2, :])
    np.testing.assert_allclose(model[1:4, :], caked_image[1:4, :], atol=1.0e-9)
    assert payload["model_kind"] == "caked_phi_m90_90_plane"


def test_caked_background_plane_fallback_uses_only_minus90_to90_rows() -> None:
    namespace = _script_functions("caked_background_plane_from_image")
    fit_plane = namespace["caked_background_plane_from_image"]
    theta_axis = np.asarray([0.0, 1.0], dtype=np.float64)
    phi_axis = np.asarray([-120.0, 0.0, 120.0], dtype=np.float64)
    caked_image = np.asarray(
        [
            [500.0, 500.0],
            [9.0, 11.0],
            [700.0, 700.0],
        ],
        dtype=np.float64,
    )

    payload = fit_plane(caked_image, theta_axis, phi_axis)

    fit_mask = np.asarray(payload["fit_mask"], dtype=bool)
    model = np.asarray(payload["background_model"], dtype=np.float64)
    fit_surface = np.asarray(payload["fit_surface"], dtype=np.float64)
    model_surface = np.asarray(payload["model_surface"], dtype=np.float64)
    assert not np.any(fit_mask[[0, 2], :])
    assert np.all(fit_mask[1, :])
    np.testing.assert_allclose(fit_surface[fit_mask], caked_image[fit_mask], atol=1.0e-9)
    assert np.all(~np.isfinite(fit_surface[~fit_mask]))
    np.testing.assert_allclose(model_surface[fit_mask], np.full(2, 10.0), atol=1.0e-9)
    assert np.all(~np.isfinite(model_surface[~fit_mask]))
    np.testing.assert_allclose(model, np.full_like(caked_image, 10.0), atol=1.0e-9)


def test_apply_radial_background_subtraction_uses_caked_phi_plane_when_available() -> None:
    namespace = _script_functions("apply_radial_background_subtraction_to_detector")
    apply_subtraction = namespace["apply_radial_background_subtraction_to_detector"]
    theta_axis = np.linspace(0.0, 4.0, 5, dtype=np.float64)
    phi_axis = np.asarray([-80.0, 0.0, 80.0], dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    caked_image = 20.0 + 2.0 * theta_grid + 0.1 * phi_grid
    detector_theta = np.asarray([[0.0, 2.0], [4.0, 1.0]], dtype=np.float64)
    detector_phi = np.asarray([[-80.0, 0.0], [80.0, 40.0]], dtype=np.float64)
    detector_image = 20.0 + 2.0 * detector_theta + 0.1 * detector_phi

    payload = apply_subtraction(
        detector_image,
        detector_theta,
        {
            "mode": "radial",
            "scale": 1.0,
            "percentile": 20.0,
            "smooth_sigma": 0.0,
            "clip_zero": False,
        },
        phi_map=detector_phi,
        caked_image=caked_image,
        theta_axis=theta_axis,
        phi_axis=phi_axis,
    )

    assert payload["enabled"] is True
    assert payload["model_kind"] == "caked_phi_m90_90_plane"
    assert "caked_fit_surface" in payload
    assert "caked_model_surface" in payload
    np.testing.assert_allclose(payload["background_model"], detector_image, atol=1.0e-9)
    np.testing.assert_allclose(
        payload["corrected_image"], np.zeros_like(detector_image), atol=1.0e-9
    )


def test_parallel_script_radial_background_popup_is_before_background_preps() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    popup_call = "radial_background_subtraction_config = show_radial_background_subtraction_popup("
    assert popup_call in source
    assert source.index(popup_call) < source.index("background_preps: list[dict[str, object]] = []")
    popup_source = source[
        source.index("def show_radial_background_subtraction_popup(") : source.index(
            "\n@njit(fastmath=True, nogil=True)"
        )
    ]
    assert "import tkinter as tk" in popup_source
    assert "from tkinter import ttk" in popup_source
    assert "from PIL import Image, ImageDraw, ImageTk" in popup_source
    assert "tk.Canvas(" in popup_source
    assert "ttk.Scale(" in popup_source
    assert "ttk.Checkbutton(" in popup_source
    assert "ttk.Button(" in popup_source
    assert "ImageTk.PhotoImage(" in popup_source
    assert "image_refs" in popup_source
    assert "def detector_preview_image(" in popup_source
    assert 'x_label="2theta (deg)"' in popup_source
    assert 'y_label="phi (deg)"' in popup_source
    assert 'update_panel_image("fit_surface", fit_surface_caked_image)' in popup_source
    assert "profile_canvas" not in popup_source
    assert "profile_canvas.create_line" not in popup_source
    assert "root.mainloop()" in popup_source
    assert "from matplotlib.widgets" not in popup_source
    assert "plt.subplots(" not in popup_source
    assert "Subtract scale" in popup_source
    assert '"Percentile"' not in popup_source
    assert '"Smooth sigma"' not in popup_source
    assert '"Enabled"' in popup_source
    assert '"Clip < 0"' in popup_source
    for label in ('"Accept"', '"Off"', '"Cancel"'):
        assert label in popup_source
    assert "RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_EDIT_MODE" in source
    assert "RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_MODE" in source
    assert "RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_SCALE" in source


def test_parallel_script_radial_background_popup_displays_caked_not_raw_detector() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    popup_source = source[
        source.index("def show_radial_background_subtraction_popup(") : source.index(
            "\n@njit(fastmath=True, nogil=True)"
        )
    ]

    assert "ai.integrate2d(" in popup_source
    assert "caked_density_image_from_result(preview_result)" in popup_source
    assert "raw_caked_image" in popup_source
    assert "raw_caked_preview" in popup_source
    assert "model_caked_image" in popup_source
    assert "corrected_caked_image" in popup_source
    assert "caked_theta_axis" in popup_source
    assert "visible_phi_mask" in popup_source
    assert "caked_phi_axis >= -90.0" in popup_source
    assert "caked_phi_axis <= 90.0" in popup_source
    assert "raw_caked_image = raw_caked_image[visible_phi_mask, :]" in popup_source
    assert "corrected_caked_full = corrected_caked_full[visible_phi_mask, :]" in popup_source
    assert "caked_image=raw_caked_image" in popup_source
    assert "Raw caked image" in popup_source
    assert "Caked background plane (-90 <= phi <= 90)" in popup_source
    assert "Corrected caked image" in popup_source
    assert "Plane fit surface (-90 <= phi <= 90)" in popup_source
    assert "fit_surface_caked_image" in popup_source
    assert "Raw detector image" not in popup_source
    assert "Corrected detector image" not in popup_source


def test_parallel_script_radial_background_popup_reuses_model_on_scale_updates() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    popup_source = source[
        source.index("def show_radial_background_subtraction_popup(") : source.index(
            "\n@njit(fastmath=True, nogil=True)"
        )
    ]
    scale_source = popup_source[
        popup_source.index("def set_scale(") : popup_source.index("\n    def toggle_check(")
    ]

    assert "cached_radial_payload" in popup_source
    assert "def recompute_radial_payload(" in popup_source
    assert "def update_preview(*, recompute_model: bool = False)" in popup_source
    assert "def set_percentile(" not in popup_source
    assert "def set_smooth_sigma(" not in popup_source
    assert "recompute_model=True" not in scale_source


def test_parallel_script_applies_radial_subtraction_before_integrate2d() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    prep_loop_source = source[
        source.index("for bg_idx, bg_path_text in enumerate(background_files):") : source.index(
            "\nprep_elapsed = time.perf_counter() - prep_total_start"
        )
    ]

    assert prep_loop_source.index(
        "raw_detector_image = detector_image.copy()"
    ) < prep_loop_source.index("apply_radial_background_subtraction_to_detector(")
    assert prep_loop_source.index(
        "apply_radial_background_subtraction_to_detector("
    ) < prep_loop_source.index("\n    caked_result = ai.integrate2d(")
    assert "raw_plane_caked_result = ai.integrate2d(" in prep_loop_source
    assert "caked_density_image_from_result(raw_plane_caked_result)" in prep_loop_source
    assert "phi_map=phi_map" in prep_loop_source
    assert "caked_image=raw_plane_caked_image" in prep_loop_source
    assert '"radial_background_model"' in prep_loop_source
    assert '"radial_background_subtraction"' in prep_loop_source


def test_parallel_script_fit_one_peak_uses_corrected_caked_image() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    fit_source = source[
        source.index("def _fit_global_peak_job(") : source.index("\ndef _run_local_peak_jobs(")
    ]

    assert 'np.asarray(prep["caked_image"], dtype=np.float64)' in fit_source
    assert source.index(
        "detector_image = np.asarray(\n        radial_subtraction_payload"
    ) < source.index("caked_image = caked_density_image")


def test_parallel_script_qr_rod_profiles_use_corrected_detector_image() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    profile_source = source[
        source.index("def profile_from_detector_qr_qz(") : source.index(
            "\n# Rod-profile baselines are added"
        )
    ]

    assert 'image = np.asarray(bg["detector_image"], dtype=np.float64)' in profile_source
    assert source.index('"detector_image": detector_image') < source.index(
        "def profile_from_detector_qr_qz("
    )


def test_parallel_script_raw_detector_image_is_preserved() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "raw_detector_image = detector_image.copy()" in source
    assert '"raw_detector_image": raw_detector_image' in source
    assert '"raw_detector_image": prep["raw_detector_image"]' in source
    assert "background_{bg_idx:02d}_detector_radial_background_model.npy" in source
    assert "background_{bg_idx:02d}_detector_radial_background_corrected.npy" in source


def test_parallel_script_radial_background_policy_changes_pre_editor_cache_key() -> None:
    namespace = _script_functions("_cache_normalize_value", "pre_editor_cache_key")
    cache_key = namespace["pre_editor_cache_key"]

    base = {"fit_settings": {"radial_background_subtraction": {"mode": "off", "scale": 0.0}}}
    changed = {"fit_settings": {"radial_background_subtraction": {"mode": "radial", "scale": 0.5}}}

    assert cache_key("state.json", input_signature=base) != cache_key(
        "state.json", input_signature=changed
    )

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    signature_source = source[
        source.index("PRE_EDITOR_CACHE_INPUT_SIGNATURE =") : source.index(
            "\nPRE_EDITOR_CACHE_KEY ="
        )
    ]
    assert '"radial_background_subtraction": RADIAL_BACKGROUND_SUBTRACTION_POLICY_SIGNATURE' in (
        signature_source
    )


def test_parallel_script_radial_background_policy_changes_final_qr_rod_cache_key() -> None:
    namespace = _script_functions("_cache_normalize_value", "qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]

    enabled_key = cache_key(
        None,
        mode="last_cached",
        rod_profile_policy={"radial_background_subtraction": {"mode": "radial", "scale": 0.5}},
    )
    disabled_key = cache_key(
        None,
        mode="last_cached",
        rod_profile_policy={"radial_background_subtraction": {"mode": "off", "scale": 0.0}},
    )

    assert enabled_key != disabled_key
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    signature_source = source[
        source.index("ROD_PROFILE_BACKGROUND_POLICY_SIGNATURE =") : source.index(
            "\nrod_phi_samples ="
        )
    ]
    assert '"radial_background_subtraction": RADIAL_BACKGROUND_SUBTRACTION_POLICY_SIGNATURE' in (
        signature_source
    )
    assert "RADIAL_BACKGROUND_SUBTRACTION_POLICY_SIGNATURE" in source


def test_parallel_script_qr_rod_peak_edits_validate_background_policy(tmp_path: Path) -> None:
    namespace = _script_functions(
        "_cache_normalize_value",
        "clean_marker_title",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    write_edits = namespace["write_qr_rod_peak_edits"]
    load_edits = namespace["load_qr_rod_peak_edits"]
    edit_path = tmp_path / "qr_rod_peak_edits.json"
    markers = pd.DataFrame(
        [{"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0, "display_l": 2.0}]
    )

    write_edits(
        edit_path,
        markers,
        metadata={"radial_background_subtraction": {"mode": "off", "scale": 0.0}},
        parameters={
            "nonzero": {
                "delta_qr": 0.04,
                "l_min": 0.6,
                "l_max": 4.2,
                "theta_initial_deg": 12.5,
            }
        },
    )

    loaded = load_edits(
        edit_path,
        expected_metadata={"radial_background_subtraction": {"mode": "off", "scale": 0.0}},
    )
    with pytest.raises(ValueError, match="different fitting policy"):
        load_edits(
            edit_path,
            expected_metadata={"radial_background_subtraction": {"mode": "radial", "scale": 1.0}},
        )
    legacy_path = tmp_path / "legacy_qr_rod_peak_edits.json"
    legacy_path.write_text(
        json.dumps(
            {"schema": "ra_sim.qr_rod_peak_edits.v1", "markers": markers.to_dict("records")}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different fitting policy"):
        load_edits(
            legacy_path,
            expected_metadata={"radial_background_subtraction": {"mode": "off", "scale": 0.0}},
        )

    assert loaded[["m", "branch", "qz_marker"]].to_dict("records") == [
        {"m": 1, "branch": "+", "qz_marker": 1.0}
    ]
    assert loaded.attrs["qr_rod_peak_edit_parameters"]["nonzero"] == pytest.approx(
        {"delta_qr": 0.04, "l_min": 0.6, "l_max": 4.2, "theta_initial_deg": 12.5}
    )


def test_parallel_script_qr_rod_peak_editor_rejects_stale_background_policy_edits() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    load_section = source[
        source.index("if qr_rod_peak_edits_path:") : source.index(
            "\nqr_rod_nonzero_editor_result = edit_qr_rod_region_editor("
        )
    ]
    save_section = source[
        source.index(
            'if bool(qr_rod_specular_editor_result.get("accepted", False))'
        ) : source.index("\nrod_profile_table = rod_profile_table_for_l_window(")
    ]

    assert "QR_ROD_PEAK_EDIT_CONTEXT_SIGNATURE" in source
    assert "expected_metadata=QR_ROD_PEAK_EDIT_CONTEXT_SIGNATURE" in load_section
    assert "metadata=QR_ROD_PEAK_EDIT_CONTEXT_SIGNATURE" in save_section
    assert "parameters=qr_rod_peak_edit_parameters" in save_section


def test_parallel_script_radial_background_cache_signatures_reject_old_prefit_caches() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "QR_ROD_FINAL_FIT_CACHE_SIGNATURE = (" in source
    assert "joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane" in source
    assert (
        'PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE = "background_peak_fit_results_v5_caked_phi_m90_90_plane"'
        in source
    )
    assert (
        'PRE_EDITOR_QR_ROD_STAGE_SIGNATURE = "qr_rod_pre_marker_profiles_hk0_roi_v17_caked_phi_m90_90_plane"'
        in source
    )


def test_parallel_script_qr_rod_final_cache_key_changes_with_background_debug_policy() -> None:
    namespace = _script_functions("_cache_normalize_value", "qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]

    enabled_key = cache_key(
        None,
        mode="last_cached",
        rod_profile_policy={"pbi2_disable_background_subtraction": True},
    )
    disabled_key = cache_key(
        None,
        mode="last_cached",
        rod_profile_policy={"pbi2_disable_background_subtraction": False},
    )

    assert enabled_key != disabled_key
    assert enabled_key["rod_reference_signature"]["rod_profile_policy"] == {
        "pbi2_disable_background_subtraction": True
    }
    assert disabled_key["rod_reference_signature"]["rod_profile_policy"] == {
        "pbi2_disable_background_subtraction": False
    }


def test_parallel_script_qr_sideband_plot_data_can_use_explicit_data_column() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    helper_source = source[
        source.index("def rod_profile_normalized_payload_for_plot_decision(") : source.index(
            "\ndef positive_log_plot_values("
        )
    ]
    y_limit_source = source[
        source.index("def shared_nonzero_rod_profile_y_axis_limits(") : source.index(
            "\nsupport_diagnostic_stem"
        )
    ]
    final_profile_source = source[
        source.index("for row, rod in enumerate(plot_rod_entries):") : source.index(
            "\nrod_profile_png"
        )
    ]

    assert "transverse_background_enabled=bool(" in y_limit_source
    assert (
        "transverse_background_enabled=bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED)"
        in final_profile_source
    )
    assert "rod_profile_normalized_payload_for_plot_decision(" in y_limit_source
    assert "rod_profile_normalized_payload_for_plot_decision(" in final_profile_source
    assert (
        'data_column = str(plot_decision.get("data_column") or "background_density")'
        in helper_source
    )
    assert (
        'data_density = table[data_column] if data_column in table else table["background_density"]'
        in helper_source
    )
    assert (
        "baseline_density = table[baseline_column] if baseline_column in table else None"
        in helper_source
    )
    assert "Qr-sideband transverse background subtraction enabled" in source


def test_parallel_script_pbi2_plot_policy_keeps_baseline_cancelled_fit_as_diagnostic() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
    )
    plot_decision = namespace["rod_profile_plot_model_decision"]
    sub = pd.DataFrame(
        {
            "background_density": [3.5, 4.0, 3.8, 4.2],
            "background_density_raw": [7.5, 8.0, 7.8, 8.2],
            "qr_sideband_background_density": [4.0, 4.0, 4.0, 4.0],
            "joint_peak_density": [42.0, 45.0, 46.0, 43.0],
            "joint_linear_baseline_density": [-39.0, -41.5, -42.0, -40.0],
            "joint_fit_density": [3.0, 3.5, 4.0, 3.0],
            "fit_density": [3.0, 3.5, 4.0, 3.0],
        }
    )
    markers = pd.DataFrame(
        [
            {"m": 4, "branch": "-", "qz_marker": 1.0, "fit_l": 2.0},
            {"m": 4, "branch": "-", "qz_marker": 2.0, "fit_l": 3.0},
        ]
    )

    decision = plot_decision(
        "pbi2",
        4,
        "-",
        sub,
        markers,
        transverse_background_enabled=True,
    )

    assert decision["plot_model"] is True
    assert decision["data_column"] == "background_density_raw"
    assert decision["density_column"] == "joint_fit_density"
    assert decision["baseline_column"] == "qr_sideband_background_density"
    assert decision["subtract_baseline_from_data"] is False
    assert decision["label"] == "Fit"
    assert decision["reason"] == "pbi2_raw_with_background_fit"
    assert decision["metrics"]["valid_l_mapping"] is True
    assert decision["metrics"]["baseline_cancellation_suspected"] is True


def test_parallel_script_pbi2_plot_policy_uses_total_fit_when_safe() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
    )
    plot_decision = namespace["rod_profile_plot_model_decision"]
    sub = pd.DataFrame(
        {
            "background_density": [8.0, 9.5, 8.8, 9.0],
            "background_density_raw": [9.0, 10.5, 9.8, 10.0],
            "qr_sideband_background_density": [1.0, 1.0, 1.0, 1.0],
            "joint_peak_density": [6.0, 6.5, 6.1, 6.3],
            "joint_linear_baseline_density": [0.5, 0.4, 0.5, 0.4],
            "joint_fit_density": [6.5, 6.9, 6.6, 6.7],
            "fit_density": [6.5, 6.9, 6.6, 6.7],
        }
    )
    markers = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 3.0},
        ]
    )

    decision = plot_decision(
        "PbI2",
        1,
        "+",
        sub,
        markers,
        transverse_background_enabled=True,
    )

    assert decision["plot_model"] is True
    assert decision["data_column"] == "background_density_raw"
    assert decision["density_column"] == "joint_fit_density"
    assert decision["baseline_column"] == "qr_sideband_background_density"
    assert decision["subtract_baseline_from_data"] is False
    assert decision["label"] == "Fit"
    assert decision["reason"] == "pbi2_raw_with_background_fit"
    assert decision["metrics"]["valid_l_mapping"] is True
    assert decision["metrics"]["baseline_cancellation_suspected"] is False


def test_parallel_script_pbi2_plot_policy_keeps_invalid_l_mapping_fit_as_diagnostic() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
    )
    plot_decision = namespace["rod_profile_plot_model_decision"]
    sub = pd.DataFrame(
        {
            "background_density": [8.0, 9.5, 8.8],
            "background_density_raw": [9.0, 10.5, 9.8],
            "qr_sideband_background_density": [1.0, 1.0, 1.0],
            "joint_peak_density": [6.0, 6.5, 6.1],
            "joint_linear_baseline_density": [0.5, 0.4, 0.5],
            "joint_fit_density": [6.5, 6.9, 6.6],
        }
    )
    markers = pd.DataFrame(
        [
            {"m": 3, "branch": "+", "qz_marker": 1.0, "fit_l": np.nan},
            {"m": 3, "branch": "+", "qz_marker": 2.0, "fit_l": np.nan},
        ]
    )

    decision = plot_decision(
        "PbI2",
        3,
        "+",
        sub,
        markers,
        transverse_background_enabled=True,
    )

    assert decision["plot_model"] is True
    assert decision["data_column"] == "background_density_raw"
    assert decision["density_column"] == "joint_fit_density"
    assert decision["baseline_column"] == "qr_sideband_background_density"
    assert decision["subtract_baseline_from_data"] is False
    assert decision["label"] == "Fit"
    assert decision["reason"] == "pbi2_raw_with_background_fit"
    assert decision["metrics"]["valid_l_mapping"] is False


def test_parallel_script_pbi2_debug_plot_policy_uses_raw_data_and_full_fit() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
    )
    plot_decision = namespace["rod_profile_plot_model_decision"]
    sub = pd.DataFrame(
        {
            "background_density": [7.5, 8.0, 7.8],
            "background_density_raw": [7.5, 8.0, 7.8],
            "qr_sideband_background_density": [np.nan, np.nan, np.nan],
            "joint_peak_density": [6.0, 6.5, 6.1],
            "joint_linear_baseline_density": [0.5, 0.4, 0.5],
            "joint_fit_density": [6.5, 6.9, 6.6],
        }
    )

    decision = plot_decision(
        "PbI2",
        3,
        "-",
        sub,
        pd.DataFrame(),
        transverse_background_enabled=False,
        background_subtraction_disabled=True,
    )

    assert decision["plot_model"] is True
    assert decision["data_column"] == "background_density"
    assert decision["density_column"] == "joint_fit_density"
    assert decision["baseline_column"] is None
    assert decision["subtract_baseline_from_data"] is False
    assert decision["label"] == "Fit"
    assert decision["reason"] == "pbi2_no_background_subtraction_debug"


def test_parallel_script_pbi2_marker_l_mapping_allows_duplicate_same_l_rows() -> None:
    namespace = _script_functions("rod_profile_marker_l_mapping_is_valid")
    mapping_is_valid = namespace["rod_profile_marker_l_mapping_is_valid"]
    markers = pd.DataFrame(
        [
            {"m": 1, "branch": "-", "qz_marker": 0.9, "fit_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 1.7, "fit_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 1.7, "fit_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 1.95, "fit_l": 2.0},
            {"m": 1, "branch": "-", "qz_marker": 2.7, "fit_l": 3.0},
        ]
    )
    conflicting = pd.concat(
        [
            markers,
            pd.DataFrame([{"m": 1, "branch": "-", "qz_marker": 1.95, "fit_l": 4.0}]),
        ],
        ignore_index=True,
    )

    assert mapping_is_valid(markers, m_value=1, branch_value="-") is True
    assert mapping_is_valid(conflicting, m_value=1, branch_value="-") is False


def test_parallel_script_invalid_or_missing_marker_l_mapping_uses_lattice_l_axis() -> None:
    namespace = _script_functions(
        "l_reference_rows",
        "rod_profile_marker_l_mapping_is_valid",
        "qz_values_to_l_axis",
    )
    namespace["ACTIVE_LATTICE_C"] = 10.0
    qz_values_to_l_axis = namespace["qz_values_to_l_axis"]
    markers = pd.DataFrame(
        [
            {"m": 4, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0},
            {"m": 4, "branch": "-", "qz_marker": 1.0, "fit_l": 4.0},
            {"m": 4, "branch": "-", "qz_marker": 2.0, "fit_l": 2.0},
            {"m": 3, "branch": "-", "qz_marker": 0.4, "fit_l": 3.0},
            {"m": 3, "branch": "-", "qz_marker": 2.0, "fit_l": 1.0},
            {"m": 3, "branch": "-", "qz_marker": 3.0, "fit_l": 2.0},
        ]
    )
    qz_values = np.asarray([1.0, 2.0], dtype=np.float64)
    expected_lattice_l = qz_values * 10.0 / (2.0 * np.pi)

    invalid_branch_l = qz_values_to_l_axis(
        qz_values, m_value=4, branch_value="-", marker_source=markers
    )
    missing_branch_l = qz_values_to_l_axis(
        qz_values, m_value=4, branch_value="+", marker_source=markers
    )
    nonmonotonic_branch_l = qz_values_to_l_axis(
        qz_values, m_value=3, branch_value="-", marker_source=markers
    )

    np.testing.assert_allclose(invalid_branch_l, expected_lattice_l)
    np.testing.assert_allclose(missing_branch_l, expected_lattice_l)
    np.testing.assert_allclose(nonmonotonic_branch_l, expected_lattice_l)


def test_parallel_script_pbi2_compressed_nonzero_marker_l_mapping_uses_lattice_l_axis() -> None:
    namespace = _script_functions(
        "l_reference_rows",
        "rod_profile_marker_l_mapping_is_valid",
        "qz_values_to_l_axis",
    )
    namespace["ACTIVE_LATTICE_C"] = 2.0 * np.pi
    namespace["SAMPLE_STEM"] = "pbi2"
    qz_values_to_l_axis = namespace["qz_values_to_l_axis"]
    mapping_is_valid = namespace["rod_profile_marker_l_mapping_is_valid"]
    markers = pd.DataFrame(
        [
            {"m": 4, "branch": "-", "qz_marker": 0.478292, "fit_l": 2.166717},
            {"m": 4, "branch": "-", "qz_marker": 1.358040, "fit_l": 2.342622},
            {"m": 4, "branch": "-", "qz_marker": 2.681896, "fit_l": 2.607324},
        ]
    )
    qz_values = np.asarray([0.5, 3.0], dtype=np.float64)

    assert mapping_is_valid(markers, m_value=4, branch_value="-") is False
    np.testing.assert_allclose(
        qz_values_to_l_axis(qz_values, m_value=4, branch_value="-", marker_source=markers),
        qz_values,
    )


def test_parallel_script_final_profile_l_filter_uses_specular_editor_bounds() -> None:
    namespace = _script_functions(
        "as_float",
        "l_reference_rows",
        "rod_profile_marker_l_mapping_is_valid",
        "qz_values_to_l_axis",
        "qz_to_l_linear_coeff",
        "rod_profile_table_for_l_window",
    )
    filter_profile = namespace["rod_profile_table_for_l_window"]
    profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.0, "background_density": 1.0},
            {"m": 0, "branch": "qz", "qz_center": 2.0, "background_density": 2.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 3.0},
            {"m": 3, "branch": "-", "qz_center": 1.0, "background_density": 4.0},
            {"m": 3, "branch": "-", "qz_center": 2.0, "background_density": 5.0},
            {"m": 3, "branch": "-", "qz_center": 3.0, "background_density": 6.0},
        ]
    )
    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0},
            {"m": 3, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0},
            {"m": 3, "branch": "-", "qz_marker": 3.0, "fit_l": 3.0},
        ]
    )

    filtered = filter_profile(
        profile_table,
        marker_table,
        0.5,
        3.0,
        specular_l_min=1.5,
        specular_l_max=2.5,
    )

    hk0 = filtered[filtered["m"] == 0]
    nonzero = filtered[filtered["m"] == 3]
    assert hk0["qz_center"].tolist() == [2.0]
    assert nonzero["qz_center"].tolist() == [1.0, 2.0, 3.0]


def test_parallel_script_plot_policy_keeps_non_pbi2_existing_model_selection() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
    )
    plot_decision = namespace["rod_profile_plot_model_decision"]
    sub = pd.DataFrame(
        {
            "background_density": [4.0, 4.5],
            "joint_peak_density": [1.0, 1.1],
            "joint_linear_baseline_density": [0.2, 0.2],
            "joint_fit_density": [1.2, 1.3],
            "fit_density": [1.2, 1.3],
        }
    )

    decision = plot_decision(
        "Bi2Se3",
        4,
        "-",
        sub,
        pd.DataFrame(),
        transverse_background_enabled=False,
    )

    assert decision["plot_model"] is True
    assert decision["density_column"] == "joint_peak_density"
    assert decision["baseline_column"] == "joint_linear_baseline_density"
    assert decision["subtract_baseline_from_data"] is True
    assert decision["label"] == "Simulation"
    assert decision["reason"] == "default_peak_model"


def test_parallel_script_final_profile_plot_uses_model_decisions() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    helper_source = source[
        source.index("def rod_profile_normalized_payload_for_plot_decision(") : source.index(
            "\ndef positive_log_plot_values("
        )
    ]
    final_profile_source = source[
        source.index("for row, rod in enumerate(plot_rod_entries):") : source.index(
            "\nrod_profile_png"
        )
    ]
    nonzero_profile_source = final_profile_source[final_profile_source.index("for col,") :]

    assert "plot_model_decision_by_key" in final_profile_source
    assert "rod_profile_plot_model_decision(" in source
    assert 'sub.get("joint_peak_density", sub["fit_density"])' not in nonzero_profile_source
    assert "rod_profile_normalized_payload_for_plot_decision(" in nonzero_profile_source
    assert (
        'data_column = str(plot_decision.get("data_column") or "background_density")'
        in helper_source
    )
    assert 'label=str(plot_decision.get("label", "Fit"))' in nonzero_profile_source


def test_parallel_script_pbi2_rod_profile_l_axis_limits_cap_at_three() -> None:
    namespace = _script_functions(
        "sample_uses_pbi2_rod_plot_policy",
        "rod_profile_l_axis_limits_for_sample",
    )
    uses_pbi2_policy = namespace["sample_uses_pbi2_rod_plot_policy"]
    plot_limits = namespace["rod_profile_l_axis_limits_for_sample"]

    assert uses_pbi2_policy("PbI2") is True
    assert uses_pbi2_policy("pbi2_state") is True
    assert uses_pbi2_policy("Bi2Se3") is False
    assert plot_limits("PbI2", (2.0, 3.85), pbi2_l_max=3.0) == (2.0, 3.0)
    assert plot_limits("PbI2", (0.0, 8.0), pbi2_l_max=3.0) == (0.0, 3.0)
    assert plot_limits("Bi2Se3", (2.0, 3.85), pbi2_l_max=3.0) == (2.0, 3.85)


def test_parallel_script_pbi2_final_profile_plots_log_only_hk0_and_cap_l() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    figure_source = source[
        source.index(
            "rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits("
        ) : source.index("\nplot_model_labels = sorted(")
    ]
    hk0_profile_source = figure_source[
        figure_source.index('if int(rod["m"]) == 0:') : figure_source.index("for col,")
    ]
    nonzero_profile_source = figure_source[figure_source.index("for col,") :]

    assert "PBI2_ROD_PROFILE_L_AXIS_MAX = 3.0" in source
    assert "NONZERO_QR_ROD_L_MIN = 0.5" in source
    assert "NONZERO_QR_ROD_L_MAX = 3.0" in source
    assert "min_l=float(NONZERO_QR_ROD_L_MIN)" in figure_source
    assert "max_l=float(NONZERO_QR_ROD_L_MAX)" in figure_source
    assert "rod_profile_hk0_l_axis_limits = rod_profile_l_axis_limits_for_sample(" in figure_source
    assert (
        "pbi2_rod_profile_figure = sample_uses_pbi2_rod_plot_policy(SAMPLE_STEM)" in figure_source
    )
    assert "positive_log_plot_values(data_norm)" in hk0_profile_source
    assert 'ax.set_yscale("log")' in hk0_profile_source
    assert "positive_log_plot_values(data_norm)" not in nonzero_profile_source
    assert "apply_positive_log_y_axis(" not in nonzero_profile_source
    assert "data_plot = data_norm" in nonzero_profile_source
    assert "simulation_plot = simulation_norm" in nonzero_profile_source
    assert "ax.set_ylim(*rod_profile_nonzero_y_axis_limits)" in nonzero_profile_source
    assert (
        "if not pbi2_rod_profile_figure:\n            ax.set_ylim(*rod_profile_nonzero_y_axis_limits)"
        not in nonzero_profile_source
    )
    assert 'ax.axhline(0.0, color="0.80", linewidth=0.45)' in nonzero_profile_source


def test_parallel_script_pbi2_specular_hk0_uses_l15_and_caked_roi() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    specular_profile_source = source[
        source.index("initial_specular_profile = build_specular_qr_rod_profile(") : source.index(
            "\nfor rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:"
        )
    ]

    assert "PBI2_SPECULAR_QR_ROD_L_MIN = 1.5" in source
    assert (
        "SPECULAR_QR_ROD_L_MIN = PBI2_SPECULAR_QR_ROD_L_MIN if IS_PBI2_SAMPLE_STEM else 0.0"
        in source
    )
    assert (
        "specular_l_marker_rows_for_background(\n"
        "        profile_bg,\n"
        "        specular_detector_qz_map,\n"
        "    )"
    ) in source
    assert "initial_specular_profile = build_specular_qr_rod_profile()" in (specular_profile_source)
    assert "specular_qz_edges_for_l_window" not in source
    assert "specular_profile_l_min" not in specular_profile_source
    assert "specular_profile_l_max" not in specular_profile_source
    assert "build_specular_qr_rod_profile(" in specular_profile_source
    assert '"specular_qr_rod_l_min"' in source
    assert '"specular_qr_rod_l_max"' in source
    assert '"specular_roi": SPECULAR_ROI_SIGNATURE' in source
    assert "theta_i_deg" not in specular_profile_source


def test_parallel_script_pbi2_qr_rod_delta_qr_defaults_to_013() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    delta_qr_source = source[
        source.index("DEFAULT_QR_ROD_DELTA_QR_SOURCE =") : source.index(
            "\nPBI2_DISABLE_BACKGROUND_SUBTRACTION ="
        )
    ]

    assert "PBI2_QR_ROD_DELTA_QR_DEFAULT = 0.13" in source
    assert "QR_ROD_DELTA_QR_SCALE = 0.85" in delta_qr_source
    assert "qr_rod_delta_qr_source_default = PBI2_QR_ROD_DELTA_QR_DEFAULT / float(" in (
        delta_qr_source
    )
    assert "np.isclose(" in delta_qr_source
    assert "float(qr_rod_delta_qr_source)" in delta_qr_source
    assert "DEFAULT_QR_ROD_DELTA_QR_SOURCE" in delta_qr_source


def test_parallel_script_pbi2_nonzero_branches_share_qz_bounds() -> None:
    namespace = _script_functions(
        "pbi2_shared_nonzero_branch_qz_bounds",
        "profile_qz_bounds_for_branch",
        "qz_bounds_for_phi_window",
        "positive_qz_bounds",
        "sample_uses_pbi2_rod_plot_policy",
    )
    namespace["POSITIVE_QZ_MIN"] = 0.0
    shared_bounds = namespace["pbi2_shared_nonzero_branch_qz_bounds"]
    branch_bounds = namespace["profile_qz_bounds_for_branch"]
    minus_bounds = (2.0, 3.0)
    plus_bounds = (0.8, 4.2)
    shared = shared_bounds(
        "PbI2",
        3,
        {"-": minus_bounds, "+": plus_bounds}.values(),
    )

    assert shared == pytest.approx((0.8, 4.2))
    assert branch_bounds(
        "PbI2",
        3,
        minus_bounds,
        shared_qz_bounds=shared,
    ) == pytest.approx((0.8, 4.2))
    assert branch_bounds(
        "Bi2Se3",
        3,
        minus_bounds,
        shared_qz_bounds=shared,
    ) == pytest.approx(minus_bounds)
    assert branch_bounds("PbI2", 0, minus_bounds, shared_qz_bounds=shared) == pytest.approx(
        minus_bounds
    )


def test_parallel_script_nonzero_profile_loop_uses_pbi2_shared_branch_bounds() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    profile_loop_source = source[
        source.index(
            "for rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:"
        ) : source.index("\nif not profile_rows:")
    ]
    signature_source = source[
        source.index("ROD_PROFILE_BACKGROUND_POLICY_SIGNATURE =") : source.index(
            "\nrod_phi_samples ="
        )
    ]

    assert "branch_payloads: list[dict[str, object]] = []" in profile_loop_source
    assert "shared_branch_qz_bounds = pbi2_shared_nonzero_branch_qz_bounds(" in profile_loop_source
    assert "profile_qz_bounds_for_branch(" in profile_loop_source
    assert '"pbi2_shared_nonzero_branch_qz_bounds"' in signature_source


def test_parallel_script_dynamic_specular_markers_use_active_lattice_c() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "specular_l_marker_rows_with_lattice_fallback",
    )
    namespace["POSITIVE_QZ_MIN"] = 1.0e-9
    marker_rows = namespace["specular_l_marker_rows_with_lattice_fallback"]

    markers = marker_rows([], lattice_c=6.78, l_max=3.0)

    assert list(markers["l"]) == [1, 2, 3]
    assert float(markers.loc[0, "qz_marker"]) == pytest.approx(2.0 * np.pi / 6.78)
    assert set(markers["marker_source"]) == {"active_lattice"}


def test_parallel_script_dynamic_specular_markers_keep_existing_fitted_rows() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "specular_l_marker_rows_with_lattice_fallback",
    )
    namespace["POSITIVE_QZ_MIN"] = 1.0e-9
    marker_rows = namespace["specular_l_marker_rows_with_lattice_fallback"]

    markers = marker_rows(
        [{"m": 0, "branch": "qz", "l": 2, "fit_l": 2, "display_l": 2.0, "qz_marker": 99.0}],
        lattice_c=6.78,
        l_max=3.0,
    )

    by_l = {int(row["l"]): row for row in markers.to_dict("records")}
    assert by_l[2]["qz_marker"] == 99.0
    assert by_l[2].get("marker_source", "") != "active_lattice"
    assert by_l[1]["qz_marker"] == pytest.approx(2.0 * np.pi / 6.78)
    assert by_l[3]["qz_marker"] == pytest.approx(3.0 * 2.0 * np.pi / 6.78)


def test_parallel_script_pbi2_specular_markers_filter_below_active_l_window() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "specular_l_marker_rows_with_lattice_fallback",
    )
    namespace["POSITIVE_QZ_MIN"] = 1.0e-9
    marker_rows = namespace["specular_l_marker_rows_with_lattice_fallback"]

    markers = marker_rows(
        [
            {"m": 0, "branch": "qz", "l": 1, "fit_l": 1, "display_l": 1.0, "qz_marker": 0.9},
            {"m": 0, "branch": "qz", "l": 2, "fit_l": 2, "display_l": 2.0, "qz_marker": 0.65},
            {"m": 0, "branch": "qz", "l": 3, "fit_l": 3, "display_l": 3.0, "qz_marker": 0.75},
        ],
        lattice_c=6.78,
        l_min=1.5,
        l_max=3.0,
    )

    assert markers["l"].astype(int).tolist() == [2, 3]
    assert markers["qz_marker"].tolist() == pytest.approx([0.65, 0.75])


def test_parallel_script_pbi2_specular_markers_filter_above_active_l_window() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "specular_l_marker_rows_with_lattice_fallback",
    )
    namespace["POSITIVE_QZ_MIN"] = 1.0e-9
    marker_rows = namespace["specular_l_marker_rows_with_lattice_fallback"]

    markers = marker_rows(
        [
            {"m": 0, "branch": "qz", "l": 2, "fit_l": 2, "display_l": 2.0, "qz_marker": 0.65},
            {"m": 0, "branch": "qz", "l": 3, "fit_l": 3, "display_l": 3.0, "qz_marker": 0.75},
            {"m": 0, "branch": "qz", "l": 8, "fit_l": 8, "display_l": 8.0, "qz_marker": 9.0},
        ],
        lattice_c=6.78,
        l_min=1.5,
        l_max=3.0,
    )

    assert markers["l"].astype(int).tolist() == [2, 3]
    assert markers["qz_marker"].tolist() == pytest.approx([0.65, 0.75])


def test_parallel_script_logs_active_lattice_and_records_it_in_notes() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "active lattice:" in source
    assert "ACTIVE_LATTICE_A" in source
    assert "ACTIVE_LATTICE_C" in source
    assert "Active lattice:" in source


def test_detector_region_label_initial_placement_uses_default_geometry() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    low_l_source = source[
        source.index("def place_low_l_rod_label(") : source.index("\ndef place_rod_label(")
    ]
    rod_source = source[
        source.index("def place_rod_label(") : source.index(
            "\ndef show_detector_region_label_position_popup("
        )
    ]
    detector_figure_start = source.index("rod_label_entries: list[dict[str, object]] = []")
    editor_call = source.index("apply_unified_qr_rod_region_editor_labels(", detector_figure_start)
    section = source[detector_figure_start:editor_call]

    assert "choose_detector_label_position(" not in low_l_source
    assert "choose_detector_label_position(" not in rod_source
    assert "high_intensity_mask" not in low_l_source
    assert "high_intensity_mask" not in rod_source
    assert "detector_region_label_high_intensity_mask" not in section
    assert "used_label_positions" not in section
    assert "return base_pos.copy()" in low_l_source
    assert "return base_pos.copy()" in rod_source


def test_detector_qz_values_for_polyline_samples_detector_map() -> None:
    namespace = _script_functions("detector_qz_values_for_polyline")
    qz_values_for_polyline = namespace["detector_qz_values_for_polyline"]

    qz_map = np.arange(12, dtype=np.float64).reshape(3, 4)
    sampled = qz_values_for_polyline(
        np.array([0.2, 2.6, 4.0, np.nan]),
        np.array([1.2, 0.4, 1.0, 1.0]),
        qz_map,
    )

    np.testing.assert_allclose(sampled[:2], np.array([4.0, 3.0]))
    assert np.isnan(sampled[2])
    assert np.isnan(sampled[3])


def test_detector_qspace_config_with_theta_initial_preserves_other_fields() -> None:
    namespace = _script_functions("detector_qspace_config_with_theta_initial")
    with_theta_initial = namespace["detector_qspace_config_with_theta_initial"]

    @dataclass(frozen=True)
    class DummyQConfig:
        theta_initial_deg: float
        distance_cor_to_detector: float

    original = DummyQConfig(theta_initial_deg=6.0, distance_cor_to_detector=0.42)
    updated = with_theta_initial(original, 18.0)

    assert original.theta_initial_deg == pytest.approx(6.0)
    assert updated.theta_initial_deg == pytest.approx(18.0)
    assert updated.distance_cor_to_detector == pytest.approx(original.distance_cor_to_detector)


def test_detector_region_specular_label_defaults_to_low_l_geometry() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "\ndisplay_detector_rotation_fit_debug("
        )
    ]
    specular_label_source = detector_figure_source[
        detector_figure_source.index('"label_id": detector_specular_label_id()') :
    ]

    assert '"label_mode": "low_l_base"' in specular_label_source
    assert '"qz_line": detector_qz_values_for_polyline(' in specular_label_source


def test_detector_region_label_editor_runtime_mode_respects_headless() -> None:
    namespace = _script_functions(
        "qr_rod_peak_edit_runtime_mode", "detector_label_edit_runtime_mode"
    )
    runtime_mode = namespace["detector_label_edit_runtime_mode"]

    assert runtime_mode("auto", backend_name="agg", env={}) == "skip"
    assert runtime_mode("auto", backend_name="TkAgg", env={}) == "popup"
    assert runtime_mode("auto", backend_name="TkAgg", env={"CI": "1"}) == "skip"
    assert runtime_mode("popup", backend_name="agg", env={"CI": "1"}) == "popup"
    assert runtime_mode("skip", backend_name="TkAgg", env={}) == "skip"


def test_detector_region_label_renderer_uses_white_text_from_label_xy() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def draw_detector_region_label_artists(") : source.index(
            "\ndef detector_label_edit_runtime_mode("
        )
    ]

    assert 'entry.get("label_xy"' in function_source
    assert 'color="white"' in function_source
    assert "fontsize=detector_label_fontsize(entry)" in function_source
    assert 'color=str(entry["color"])' not in function_source
    assert "path_effects=" not in function_source


def test_detector_region_label_settings_round_trip_by_label_id() -> None:
    namespace = _script_functions(
        "detector_label_xy",
        "detector_label_id",
        "detector_label_fontsize",
        "detector_label_settings_payload",
        "apply_detector_label_settings",
    )
    settings_payload = namespace["detector_label_settings_payload"]
    apply_settings = namespace["apply_detector_label_settings"]
    namespace["detector_region_label_fontsize"] = 11.0

    exported = settings_payload(
        [
            {
                "label_id": "m=7:+",
                "text": "m = 7 +",
                "label_xy": np.array([12.0, 34.0]),
                "fontsize": 14.5,
            },
            {
                "label_id": "m=0",
                "text": "m = 0",
                "label_xy": np.array([56.0, 78.0]),
                "fontsize": 16.0,
            },
        ]
    )

    assert exported["schema"] == "ra_sim.detector_label_settings.v1"
    assert exported["labels"][0]["label_id"] == "m=7:+"
    assert exported["labels"][0]["label_xy"] == [12.0, 34.0]
    assert exported["labels"][0]["fontsize"] == 14.5

    applied = apply_settings(
        [
            {"label_id": "m=7:+", "text": "m = 7 +", "label_xy": np.array([0.0, 0.0])},
            {"label_id": "m=0", "text": "m = 0", "label_xy": np.array([1.0, 1.0])},
            {"label_id": "m=9:-", "text": "m = 9 -", "label_xy": np.array([2.0, 2.0])},
        ],
        exported,
    )

    np.testing.assert_allclose(applied[0]["label_xy"], np.array([12.0, 34.0]))
    np.testing.assert_allclose(applied[1]["label_xy"], np.array([56.0, 78.0]))
    np.testing.assert_allclose(applied[2]["label_xy"], np.array([2.0, 2.0]))
    assert applied[0]["fontsize"] == 14.5
    assert applied[1]["fontsize"] == 16.0


def test_detector_region_label_editor_uses_tk_canvas_controls() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "import tkinter as tk",
        "from tkinter import ttk",
        "tempfile.NamedTemporaryFile(",
        "tk.PhotoImage(",
        "tk.Canvas(",
        "canvas.create_image(",
        "canvas.create_text(",
        "canvas.tag_bind(",
        'ttk.Button(control_frame, text="Import"',
        'ttk.Button(control_frame, text="Export"',
        'ttk.Button(control_frame, text="Cancel"',
        'ttk.Button(control_frame, text="Accept"',
        "load_detector_label_settings(",
        "save_detector_label_settings(",
        "root.mainloop()",
    ):
        assert token in function_source

    for removed_token in (
        "from matplotlib.widgets import Button, TextBox",
        "TextBox(",
        "fig._ra_sim_detector_label_edit_widgets",
        "fig.canvas.start_event_loop(timeout=-1)",
        'fig.canvas.mpl_connect("motion_notify_event", on_motion)',
    ):
        assert removed_token not in function_source


def test_detector_region_label_editor_selects_labels_with_tk_canvas_items() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "drawable_index_set = set(drawable_indices)",
        "label_canvas_items: dict[int, int] = {}",
        "def select_label(index: int) -> None:",
        "canvas.itemconfigure(",
        "canvas.coords(",
        'item, "<ButtonPress-1>", lambda event, i=int(index): on_label_press(event, i)',
        'canvas.tag_bind(item, "<B1-Motion>"',
        'canvas.tag_bind(item, "<ButtonRelease-1>"',
        "sync_controls()",
    ):
        assert token in function_source
    assert "artist.contains(event)" not in function_source
    assert 'fig.canvas.mpl_connect("button_press_event", on_press)' not in function_source


def test_detector_region_label_editor_tunes_pixel_locations_in_data_space() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "def data_to_canvas_xy(",
        "def canvas_to_data_xy(",
        "ax.transData.transform(",
        "ax.transData.inverted().transform(",
        "image_height -",
        "clamp_detector_label_position(",
        'edited[int(index)]["label_xy"]',
        "def set_selected_label_text(",
        "def step_font(",
        "label_text_var = tk.StringVar(",
        'ttk.Button(control_frame, text="Font -"',
        'ttk.Button(control_frame, text="Font +"',
    ):
        assert token in function_source
    assert "setup_detector_label_editor_blit(" not in source
    assert "redraw_detector_label_editor_blit(" not in source


def test_detector_region_label_editor_drags_labels_on_tk_canvas() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "def on_label_press(event, index: int) -> None:",
        "def on_label_motion(event) -> None:",
        "def on_label_release(_event) -> None:",
        'selected["dragging"] = True',
        'selected["dragging"] = False',
        "canvas.canvasx(event.x)",
        "canvas.canvasy(event.y)",
        "set_label_canvas_position(",
        'selected["drag_offset_x"]',
        'selected["drag_offset_y"]',
    ):
        assert token in function_source


def test_detector_region_label_editor_uses_tk_mainloop_without_matplotlib_event_loop() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "root = tk.Tk()",
        "root.protocol(",
        "root.mainloop()",
        "def finish_editor(",
        "root.destroy()",
        "image_path.unlink()",
    ):
        assert token in function_source
    assert "fig.canvas.start_event_loop(timeout=-1)" not in function_source
    assert "fig.canvas.stop_event_loop()" not in function_source
    assert "plt.show(block=True)" not in function_source
    assert "plt.close(fig)" not in function_source


def test_detector_region_label_editor_removes_temporary_in_figure_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    assert "finally:" in function_source
    assert "image_path.unlink()" in function_source
    assert "root.destroy()" in function_source
    assert "_ra_sim_detector_label_edit_widgets" not in function_source


def test_detector_region_label_editor_wires_before_final_save() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_start = source.index("rod_label_entries: list[dict[str, object]] = []")
    unified_label_call = source.index(
        "apply_unified_qr_rod_region_editor_labels(", detector_figure_start
    )
    editor_call = source.index("edit_detector_region_label_positions(", detector_figure_start)
    draw_call = source.index("draw_detector_region_label_artists(", detector_figure_start)
    save_call = source.index(
        "detector_region_png, detector_region_pdf = save_manuscript_figure",
        detector_figure_start,
    )
    section = source[detector_figure_start:save_call]

    assert unified_label_call < editor_call < draw_call < save_call
    assert "settings_path=detector_label_settings_path" in section
    assert "backend_name=mpl.get_backend()" in section
    assert "env=os.environ" in section
    assert "draw_detector_region_label_artists(" in section
    assert 'label_entry["label_xy"]' in section
    assert "used_label_positions" not in section


def test_detector_region_label_editor_does_not_auto_import_settings_before_popup() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def edit_detector_region_label_positions(") : source.index(
            "\ndetector_region_display_width ="
        )
    ]
    before_popup_source = function_source[
        : function_source.index("show_detector_region_label_position_popup(")
    ]

    assert "load_detector_label_settings(" not in before_popup_source
    assert "apply_detector_label_settings(" not in before_popup_source
    assert "settings_path=path_text" in function_source


def test_detector_region_specular_visual_uses_integrated_qz_region() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    qmap_setup_source = source[
        source.index("profile_detector_q_maps = notebook_detector_qr_qz_maps(") : source.index(
            "\ndef bilinear_sample_detector_map("
        )
    ]
    specular_profile_source = source[
        source.index("specular_region_mask = (") : source.index(
            "for rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:"
        )
    ]
    nonzero_profile_source = source[
        source.index(
            "for rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:"
        ) : source.index("\nif not profile_rows:")
    ]
    detector_figure_source = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "\ndisplay_detector_rotation_fit_debug("
        )
    ]

    assert "specular_detector_q_maps = profile_detector_q_maps" in qmap_setup_source
    assert "specular_detector_region_mask" in specular_profile_source
    assert (
        "specular_detector_qr_map = np.asarray(specular_detector_q_maps[0]"
        in specular_profile_source
    )
    assert (
        "specular_detector_qz_map = np.asarray(specular_detector_q_maps[1]"
        in specular_profile_source
    )
    assert (
        "specular_detector_valid_q = np.asarray(specular_detector_q_maps[2]"
        in specular_profile_source
    )
    assert "SPECULAR_QR_ROD_L_MAX = 8.0" in source
    assert "PBI2_SPECULAR_QR_ROD_L_MIN = 1.5" in source
    assert "build_specular_qr_rod_profile(" in specular_profile_source
    assert "specular_profile_l_min" not in specular_profile_source
    assert "specular_profile_l_max" not in specular_profile_source
    assert "specular_qz_edges_for_l_window" not in specular_profile_source
    assert (
        "specular_detector_region_mask = specular_detector_region_mask &"
        not in specular_profile_source
    )
    assert "profile_from_detector_phi_two_theta_roi(" in specular_profile_source
    assert "specular_fallback_region_mask" not in specular_profile_source
    assert "specular_qz_bounds_need_fallback" not in specular_profile_source
    assert "profile_mask_override=specular_profile_mask_override" not in specular_profile_source
    assert "detector_specular_l_window_mask_per_qz_bin" not in specular_profile_source
    assert "specular_detector_q_maps_for_theta_i" not in specular_profile_source
    assert "theta_initial_deg_used_for_q" not in specular_profile_source
    assert "detector_q_maps=profile_detector_q_maps" in nonzero_profile_source
    assert "profile_from_full_mask(" not in specular_profile_source
    assert 'branch_label="m = 0"' in specular_profile_source
    assert "specular_detector_qz_values = np.asarray(specular_qz_values" in detector_figure_source
    assert '"band_fill_mask": specular_roi_visual_mask' in detector_figure_source
    assert "qr_center=0.0" not in detector_figure_source
    assert "projected_col, projected_row, specular_detector_qz_map" in detector_figure_source
    assert '"label_id": detector_specular_label_id()' in detector_figure_source
    assert "m = 0 ROI uses caked phi/2theta bounds" in source


def _hk0_detector_profile_test_namespace() -> dict[str, object]:
    namespace = _script_functions("as_float", "profile_from_detector_qr_qz")
    namespace.update(
        {
            "SAMPLE_NAME": "unit-test",
            "ROD_PROFILE_MAX_TWO_THETA_DEG": 70.3,
            "POSITIVE_QZ_MIN": 0.0,
            "QR_ROD_TRANSVERSE_BACKGROUND_ENABLED": False,
            "GPU_ACCELERATION_ENABLED": False,
            "NUMBA_AVAILABLE": False,
            "qr_rod_delta_qr": 0.10,
            "positive_l_detector_peak_model_for_display": lambda bg: np.zeros_like(
                np.asarray(bg["detector_image"], dtype=np.float64)
            ),
        }
    )
    return namespace


def _hk0_detector_profile_frame(
    *,
    qr_map: np.ndarray,
    qz_map: np.ndarray,
    delta_qr: float = 0.10,
) -> pd.DataFrame:
    namespace = _hk0_detector_profile_test_namespace()
    profile_from_detector = namespace["profile_from_detector_qr_qz"]
    shape = tuple(int(v) for v in qr_map.shape)
    config = type("QrOverlayConfig", (), {"theta_initial_deg": 0.0})()
    return profile_from_detector(
        {
            "detector_image": np.ones(shape, dtype=np.float64),
            "tilt_deg": 0.0,
            "qr_overlay_config": config,
        },
        {"m": 0, "qr": 0.0},
        branch_name="qz",
        branch_label="m = 0",
        qz_edges=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        phi_min=-5.0,
        phi_max=5.0,
        detector_q_maps=(
            np.asarray(qr_map, dtype=np.float64),
            np.asarray(qz_map, dtype=np.float64),
            np.ones(shape, dtype=bool),
        ),
        detector_phi_map=np.zeros(shape, dtype=np.float64),
        detector_solid_angle=None,
        detector_two_theta_map=np.full(shape, 10.0, dtype=np.float64),
        delta_qr_override=float(delta_qr),
    )


def test_parallel_script_hk0_roi_mask_uses_phi_and_two_theta_bounds() -> None:
    namespace = _script_functions("detector_phi_two_theta_roi_mask")
    roi_mask = namespace["detector_phi_two_theta_roi_mask"]
    phi_map = np.asarray(
        [
            [-12.0, -2.0, 2.0, 12.0],
            [-12.0, -2.0, 2.0, 12.0],
        ],
        dtype=np.float64,
    )
    two_theta_map = np.asarray(
        [
            [8.0, 8.0, 8.0, 8.0],
            [12.0, 12.0, 12.0, 12.0],
        ],
        dtype=np.float64,
    )

    mask = roi_mask(
        phi_map,
        two_theta_map,
        phi_min=-5.0,
        phi_max=5.0,
        two_theta_min=7.0,
        two_theta_max=10.0,
    )
    reversed_bounds_mask = roi_mask(
        phi_map,
        two_theta_map,
        phi_min=5.0,
        phi_max=-5.0,
        two_theta_min=10.0,
        two_theta_max=7.0,
    )

    expected = np.asarray(
        [
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(reversed_bounds_mask, expected)


def _hk0_builder_test_namespace() -> dict[str, object]:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_table_with_specular_l_markers",
        "specular_qz_bounds_for_l_window",
        "specular_l_marker_rows_with_lattice_fallback",
        "active_specular_marker_rows_for_l_window",
        "profile_from_detector_qr_qz",
        "detector_phi_two_theta_roi_mask",
        "profile_from_detector_phi_two_theta_roi",
        "build_specular_qr_rod_profile",
        "qr_rod_editor_phase_matches",
        "rod_profile_marker_l_mapping_is_valid",
        "l_reference_rows",
        "qz_values_to_l_axis",
        "qz_to_l_linear_coeff",
        "rod_profile_table_for_l_window",
        "qz_edges_from_profile_group",
        "recompute_qr_rod_region_profiles",
    )
    qr_map = np.asarray(
        [
            [-0.05, 0.04, 0.16, -0.18],
            [-0.03, 0.05, 0.02, 0.06],
        ],
        dtype=np.float64,
    )
    qz_map = np.asarray(
        [
            [1.10, 1.40, 1.20, 1.80],
            [1.20, 1.60, 1.70, 1.90],
        ],
        dtype=np.float64,
    )
    namespace.update(
        {
            "SAMPLE_NAME": "unit-test",
            "ROD_PROFILE_MAX_TWO_THETA_DEG": 70.3,
            "POSITIVE_QZ_MIN": 0.0,
            "QR_ROD_TRANSVERSE_BACKGROUND_ENABLED": False,
            "GPU_ACCELERATION_ENABLED": False,
            "NUMBA_AVAILABLE": False,
            "qr_rod_delta_qr": 0.10,
            "ROD_PROFILE_QZ_BINS": 2,
            "ACTIVE_LATTICE_C": 2.0 * np.pi,
            "specular_phi_min_deg": -5.0,
            "specular_phi_max_deg": 5.0,
            "specular_theta_min_deg": 7.0,
            "specular_theta_max_deg": 13.0,
            "specular_detector_q_maps": (
                qr_map,
                qz_map,
                np.ones(qr_map.shape, dtype=bool),
            ),
            "profile_detector_phi_map": np.zeros(qr_map.shape, dtype=np.float64),
            "profile_detector_solid_angle": None,
            "profile_detector_two_theta_map": np.full(qr_map.shape, 10.0, dtype=np.float64),
            "positive_l_detector_peak_model_for_display": lambda bg: np.zeros_like(
                np.asarray(bg["detector_image"], dtype=np.float64)
            ),
            "profile_bg": {
                "detector_image": np.ones(qr_map.shape, dtype=np.float64),
                "tilt_deg": 0.0,
                "qr_overlay_config": type("QrOverlayConfig", (), {"theta_initial_deg": 0.0})(),
            },
            "specular_rod_entry": {"m": 0, "qr": 0.0},
            "specular_l_marker_table": pd.DataFrame(
                [
                    {
                        "m": 0,
                        "branch": "qz",
                        "qz_marker": 1.0,
                        "fit_l": 1.0,
                        "display_l": 1.0,
                    },
                    {
                        "m": 0,
                        "branch": "qz",
                        "qz_marker": 2.0,
                        "fit_l": 2.0,
                        "display_l": 2.0,
                    },
                ]
            ),
        }
    )
    return namespace


def test_parallel_script_hk0_profile_keeps_negative_qr_detector_pixels() -> None:
    qr_map = np.asarray([[-0.09, -0.08], [-0.07, -0.06]], dtype=np.float64)
    qz_map = np.asarray([[0.25, 0.75], [1.25, 1.75]], dtype=np.float64)

    profile = _hk0_detector_profile_frame(qr_map=qr_map, qz_map=qz_map)

    assert not profile.empty
    assert profile["m"].astype(int).tolist() == [0, 0]
    assert profile["branch"].tolist() == ["qz", "qz"]
    assert profile["pixel_count"].tolist() == [2, 2]
    assert np.isfinite(np.asarray(profile["pixel_count"], dtype=np.float64)).all()
    assert (np.asarray(profile["pixel_count"], dtype=np.float64) > 0.0).all()


def test_parallel_script_hk0_profile_matches_detector_visual_phi_two_theta_roi() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    profile_source = source[
        source.index("def profile_from_detector_phi_two_theta_roi(") : source.index(
            "\n# Rod-profile baselines are added"
        )
    ]
    specular_source = source[
        source.index("specular_detector_phi_mask = detector_phi_two_theta_roi_mask") : source.index(
            "\n\ndef specular_l_marker_rows_with_lattice_fallback"
        )
    ]
    assert "detector_phi_two_theta_roi_mask(" in profile_source
    assert "profile_mask_override=roi_mask" in profile_source
    assert "delta_qr_override=0.0" in profile_source
    assert "detector_phi_two_theta_roi_mask(" in specular_source
    assert "(specular_detector_qr_map >= -float(qr_rod_delta_qr))" not in specular_source
    assert "(specular_detector_qr_map >= 0.0)" not in specular_source

    roi_mask = _script_functions("detector_phi_two_theta_roi_mask")[
        "detector_phi_two_theta_roi_mask"
    ]
    phi_map = np.asarray(
        [[-6.0, -4.0, 0.0, 4.0, 6.0, 0.0]],
        dtype=np.float64,
    )
    two_theta_map = np.asarray([[8.0, 9.0, 10.0, 11.0, 12.0, 20.0]], dtype=np.float64)
    mask = roi_mask(
        phi_map,
        two_theta_map,
        phi_min=-5.0,
        phi_max=5.0,
        two_theta_min=8.0,
        two_theta_max=12.0,
    )
    np.testing.assert_array_equal(
        mask,
        np.asarray([[False, True, True, True, False, False]], dtype=bool),
    )


def test_parallel_script_builds_hk0_profile_from_active_markers_without_base_edges() -> None:
    namespace = _hk0_builder_test_namespace()
    build_profile = namespace["build_specular_qr_rod_profile"]

    profile = build_profile()

    assert not profile.empty
    assert profile["m"].astype(int).tolist() == [0, 0]
    assert profile["branch"].tolist() == ["qz", "qz"]
    assert profile["pixel_count"].tolist() == [4, 4]
    assert (pd.to_numeric(profile["pixel_count"], errors="coerce") > 0).all()


def test_parallel_script_recompute_creates_hk0_when_base_profiles_lack_hk0() -> None:
    namespace = _hk0_builder_test_namespace()
    recompute = namespace["recompute_qr_rod_region_profiles"]
    namespace["qr_rod_editor_base_profiles"] = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 1.0, "background_density": 4.0},
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 5.0},
        ]
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    namespace["phi_windows"] = [("+", 0.0, 90.0, "+"), ("-", -90.0, 0.0, "-")]
    namespace["rod_entries"] = [{"m": 1, "qr": 1.0}]
    namespace["marker_table"] = pd.DataFrame()

    profile = recompute(
        0.10,
        1.0,
        2.0,
        editor_phase="specular",
        marker_table=pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0},
                {"m": 0, "branch": "qz", "qz_marker": 2.0, "fit_l": 2.0},
            ]
        ),
    )

    assert not profile.empty
    assert set(zip(profile["m"].astype(int), profile["branch"].astype(str))) == {(0, "qz")}
    assert int(pd.to_numeric(profile["pixel_count"], errors="coerce").sum()) == 8


def test_parallel_script_recompute_nonzero_theta_i_changes_q_map_cache_key() -> None:
    namespace = _hk0_builder_test_namespace()
    recompute = namespace["recompute_qr_rod_region_profiles"]
    namespace["qr_rod_editor_base_profiles"] = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 1.0, "background_density": 4.0},
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 5.0},
        ]
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    namespace["phi_windows"] = [("+", 0.0, 90.0, "+"), ("-", -90.0, 0.0, "-")]
    namespace["rod_entries"] = [{"m": 1, "qr": 1.0}]
    namespace["marker_table"] = pd.DataFrame()
    helper_calls: list[float] = []
    profile_calls: list[tuple[float, float]] = []

    def detector_q_maps_for_theta_initial(theta_initial_deg: object) -> tuple[np.ndarray, ...]:
        theta = float(theta_initial_deg)
        helper_calls.append(theta)
        qr_map = np.full((2, 2), 1.0 + theta / 100.0, dtype=np.float64)
        qz_map = np.full((2, 2), theta, dtype=np.float64)
        return qr_map, qz_map, np.ones(qr_map.shape, dtype=bool)

    def profile_from_detector_qr_qz(
        _bg: dict[str, object],
        _rod: dict[str, object],
        *,
        branch_name: str,
        detector_q_maps: tuple[np.ndarray, ...],
        theta_initial_deg_used_for_q: float,
        **_kwargs: object,
    ) -> pd.DataFrame:
        profile_calls.append(
            (
                float(np.asarray(detector_q_maps[1], dtype=np.float64)[0, 0]),
                theta_initial_deg_used_for_q,
            )
        )
        return pd.DataFrame(
            {
                "m": [1],
                "branch": [branch_name],
                "qz_center": [1.5],
                "background_density": [theta_initial_deg_used_for_q],
                "theta_initial_deg_used_for_q": [theta_initial_deg_used_for_q],
            }
        )

    namespace["detector_q_maps_for_theta_initial"] = detector_q_maps_for_theta_initial
    namespace["profile_from_detector_qr_qz"] = profile_from_detector_qr_qz

    first = recompute(0.10, 0.0, 3.0, editor_phase="nonzero", theta_initial_deg=11.25)
    second = recompute(0.10, 0.0, 3.0, editor_phase="nonzero", theta_initial_deg=12.5)

    assert helper_calls == [11.25, 12.5]
    assert profile_calls == [(11.25, 11.25), (12.5, 12.5)]
    assert first["theta_initial_deg_used_for_q"].tolist() == pytest.approx([11.25])
    assert second["theta_initial_deg_used_for_q"].tolist() == pytest.approx([12.5])


def test_parallel_script_recompute_nonzero_branches_use_same_l_start() -> None:
    namespace = _hk0_builder_test_namespace()
    recompute = namespace["recompute_qr_rod_region_profiles"]
    qz_values_to_l_axis = namespace["qz_values_to_l_axis"]
    namespace["ACTIVE_LATTICE_C"] = 2.0 * np.pi
    namespace["ROD_PROFILE_QZ_BINS"] = 4
    namespace["qr_rod_editor_base_profiles"] = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 4.0},
            {"m": 1, "branch": "+", "qz_center": 4.0, "background_density": 5.0},
            {"m": 1, "branch": "-", "qz_center": 10.0, "background_density": 6.0},
            {"m": 1, "branch": "-", "qz_center": 12.0, "background_density": 7.0},
            {"m": 1, "branch": "-", "qz_center": 14.0, "background_density": 8.0},
            {"m": 1, "branch": "-", "qz_center": 16.0, "background_density": 9.0},
            {"m": 1, "branch": "-", "qz_center": 18.0, "background_density": 10.0},
            {"m": 1, "branch": "-", "qz_center": 20.0, "background_density": 11.0},
        ]
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    namespace["phi_windows"] = [("+", 0.0, 90.0, "+"), ("-", -90.0, 0.0, "-")]
    namespace["rod_entries"] = [{"m": 1, "qr": 1.0}]
    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 4.0, "fit_l": 3.0},
            {"m": 1, "branch": "-", "qz_marker": 10.0, "fit_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 20.0, "fit_l": 3.0},
        ]
    )
    namespace["marker_table"] = marker_table

    def detector_q_maps_for_theta_initial(_theta_initial_deg: object) -> tuple[np.ndarray, ...]:
        qr_map = np.ones((2, 2), dtype=np.float64)
        qz_map = np.ones((2, 2), dtype=np.float64)
        return qr_map, qz_map, np.ones(qr_map.shape, dtype=bool)

    def profile_from_detector_qr_qz(
        _bg: dict[str, object],
        _rod: dict[str, object],
        *,
        branch_name: str,
        qz_edges: np.ndarray,
        **_kwargs: object,
    ) -> pd.DataFrame:
        centers = 0.5 * (np.asarray(qz_edges[:-1]) + np.asarray(qz_edges[1:]))
        return pd.DataFrame(
            {
                "m": [1] * int(centers.size),
                "branch": [branch_name] * int(centers.size),
                "qz_center": centers,
                "background_density": np.ones(centers.size, dtype=np.float64),
                "pixel_count": np.ones(centers.size, dtype=np.int64),
            }
        )

    namespace["detector_q_maps_for_theta_initial"] = detector_q_maps_for_theta_initial
    namespace["profile_from_detector_qr_qz"] = profile_from_detector_qr_qz

    profile = recompute(
        0.10,
        1.0,
        3.0,
        editor_phase="nonzero",
        marker_table=marker_table,
    )

    starts: dict[str, float] = {}
    centers_by_branch: dict[str, np.ndarray] = {}
    for branch_name, sub in profile.groupby("branch"):
        l_values = qz_values_to_l_axis(
            sub["qz_center"],
            m_value=1,
            branch_value=str(branch_name),
            marker_source=marker_table,
        )
        starts[str(branch_name)] = float(np.nanmin(l_values))
        centers_by_branch[str(branch_name)] = np.asarray(l_values, dtype=np.float64)
    assert starts["+"] == pytest.approx(starts["-"])
    assert centers_by_branch["+"] == pytest.approx(centers_by_branch["-"])


def test_parallel_script_recompute_nonzero_visible_branch_starts_match_after_empty_bins() -> None:
    namespace = _hk0_builder_test_namespace()
    recompute = namespace["recompute_qr_rod_region_profiles"]
    qz_values_to_l_axis = namespace["qz_values_to_l_axis"]
    namespace["ACTIVE_LATTICE_C"] = 2.0 * np.pi
    namespace["ROD_PROFILE_QZ_BINS"] = 4
    namespace["qr_rod_editor_base_profiles"] = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 4.0},
            {"m": 1, "branch": "+", "qz_center": 4.0, "background_density": 5.0},
            {"m": 1, "branch": "-", "qz_center": 10.0, "background_density": 6.0},
            {"m": 1, "branch": "-", "qz_center": 20.0, "background_density": 7.0},
        ]
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    namespace["phi_windows"] = [("+", 0.0, 90.0, "+"), ("-", -90.0, 0.0, "-")]
    namespace["rod_entries"] = [{"m": 1, "qr": 1.0}]
    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 4.0, "fit_l": 3.0},
            {"m": 1, "branch": "-", "qz_marker": 10.0, "fit_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 20.0, "fit_l": 3.0},
        ]
    )
    namespace["marker_table"] = marker_table

    def detector_q_maps_for_theta_initial(_theta_initial_deg: object) -> tuple[np.ndarray, ...]:
        qr_map = np.ones((2, 2), dtype=np.float64)
        qz_map = np.ones((2, 2), dtype=np.float64)
        return qr_map, qz_map, np.ones(qr_map.shape, dtype=bool)

    def profile_from_detector_qr_qz(
        _bg: dict[str, object],
        _rod: dict[str, object],
        *,
        branch_name: str,
        qz_edges: np.ndarray,
        **_kwargs: object,
    ) -> pd.DataFrame:
        centers = 0.5 * (np.asarray(qz_edges[:-1]) + np.asarray(qz_edges[1:]))
        pixel_count = np.ones(centers.size, dtype=np.int64)
        if branch_name == "-":
            pixel_count[:1] = 0
        return pd.DataFrame(
            {
                "m": [1] * int(centers.size),
                "branch": [branch_name] * int(centers.size),
                "qz_center": centers,
                "background_density": np.ones(centers.size, dtype=np.float64),
                "pixel_count": pixel_count,
            }
        )

    namespace["detector_q_maps_for_theta_initial"] = detector_q_maps_for_theta_initial
    namespace["profile_from_detector_qr_qz"] = profile_from_detector_qr_qz

    profile = recompute(
        0.10,
        1.0,
        3.0,
        editor_phase="nonzero",
        marker_table=marker_table,
    )

    starts: dict[str, float] = {}
    for branch_name, sub in profile.groupby("branch"):
        visible = sub[np.asarray(sub["pixel_count"], dtype=np.float64) > 0.0]
        l_values = qz_values_to_l_axis(
            visible["qz_center"],
            m_value=1,
            branch_value=str(branch_name),
            marker_source=marker_table,
        )
        starts[str(branch_name)] = float(np.nanmin(l_values))
    assert starts["+"] == pytest.approx(starts["-"])


def test_parallel_script_recompute_specular_ignores_l_bounds_and_uses_two_theta_roi() -> None:
    namespace = _hk0_builder_test_namespace()
    recompute = namespace["recompute_qr_rod_region_profiles"]
    qz_map = np.asarray(
        [
            [1.10, 1.20, 1.70, 1.80],
            [1.30, 1.40, 1.90, 2.00],
        ],
        dtype=np.float64,
    )
    qr_map = np.zeros(qz_map.shape, dtype=np.float64)
    theta_map = np.asarray(
        [
            [8.0, 9.0, 14.0, 15.0],
            [8.5, 9.5, 14.5, 15.5],
        ],
        dtype=np.float64,
    )
    namespace["specular_detector_q_maps"] = (qr_map, qz_map, np.ones(qz_map.shape, dtype=bool))
    namespace["profile_detector_phi_map"] = np.zeros(qz_map.shape, dtype=np.float64)
    namespace["profile_detector_two_theta_map"] = theta_map
    namespace["profile_bg"] = {
        "detector_image": np.ones(qz_map.shape, dtype=np.float64),
        "tilt_deg": 0.0,
        "qr_overlay_config": type("QrOverlayConfig", (), {"theta_initial_deg": 0.0})(),
    }
    namespace["phi_windows"] = [("+", 0.0, 90.0, "+"), ("-", -90.0, 0.0, "-")]
    namespace["rod_entries"] = []
    namespace["qr_rod_editor_base_profiles"] = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.2, "background_density": 1.0},
            {"m": 0, "branch": "qz", "qz_center": 1.8, "background_density": 1.0},
        ]
    )
    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0},
            {"m": 0, "branch": "qz", "qz_marker": 2.0, "fit_l": 2.0},
        ]
    )

    namespace["qr_rod_editor_profile_cache"] = {}
    low_l = recompute(
        0.01,
        1.0,
        1.4,
        editor_phase="specular",
        marker_table=marker_table,
        phi_min=-2.0,
        phi_max=2.0,
        two_theta_min=7.0,
        two_theta_max=16.0,
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    high_l = recompute(
        0.01,
        1.6,
        2.0,
        editor_phase="specular",
        marker_table=marker_table,
        phi_min=-2.0,
        phi_max=2.0,
        two_theta_min=7.0,
        two_theta_max=16.0,
    )
    namespace["qr_rod_editor_profile_cache"] = {}
    low_theta = recompute(
        0.01,
        1.0,
        1.4,
        editor_phase="specular",
        marker_table=marker_table,
        phi_min=-2.0,
        phi_max=2.0,
        two_theta_min=7.0,
        two_theta_max=10.0,
    )

    assert low_l["qz_center"].tolist() == pytest.approx(high_l["qz_center"].tolist())
    assert low_l["pixel_count"].tolist() == pytest.approx(high_l["pixel_count"].tolist())
    assert float(np.nansum(low_theta["pixel_count"])) < float(np.nansum(low_l["pixel_count"]))


def test_parallel_script_specular_qz_bounds_for_l_window_limits_to_l8() -> None:
    namespace = _script_functions("qz_l_linear_coeff_from_marker_rows", "qz_bounds_for_l_window")
    qz_bounds = namespace["qz_bounds_for_l_window"]
    marker_table = pd.DataFrame(
        [
            {"qz_marker": 1.0, "fit_l": 4.0},
            {"qz_marker": 2.0, "fit_l": 8.0},
        ]
    )

    bounds = qz_bounds(
        np.linspace(0.25, 4.0, 20),
        marker_table,
        l_min=0.0,
        l_max=8.0,
        positive_qz_min=0.0,
    )

    assert bounds == pytest.approx((0.25, 2.0))
    assert qz_bounds(np.linspace(0.25, 4.0, 20), pd.DataFrame(), l_max=8.0) is None


def test_parallel_script_specular_qz_bounds_falls_back_to_lattice_window() -> None:
    namespace = _script_functions(
        "as_float",
        "active_lattice_qz_value_for_l",
        "qz_l_linear_coeff_from_marker_rows",
        "qz_bounds_for_l_window",
        "specular_qz_bounds_for_l_window",
    )
    bounds = namespace["specular_qz_bounds_for_l_window"](
        np.linspace(0.1, 4.0, 400),
        pd.DataFrame(),
        l_min=1.5,
        l_max=3.0,
        lattice_c=6.978,
        positive_qz_min=0.0,
    )

    expected_lo = namespace["active_lattice_qz_value_for_l"](1.5, 6.978)
    expected_hi = namespace["active_lattice_qz_value_for_l"](3.0, 6.978)
    assert bounds == pytest.approx((expected_lo, expected_hi))

    clipped_support_bounds = namespace["specular_qz_bounds_for_l_window"](
        np.linspace(2.5, 4.0, 20),
        pd.DataFrame(),
        l_min=1.5,
        l_max=3.0,
        lattice_c=6.978,
        positive_qz_min=0.0,
    )
    assert clipped_support_bounds == pytest.approx((expected_lo, expected_hi))

    invalid_marker_bounds = namespace["specular_qz_bounds_for_l_window"](
        np.linspace(2.5, 4.0, 20),
        pd.DataFrame({"qz_marker": [1.0, 1.0], "fit_l": [1.0, 2.0]}),
        l_min=1.5,
        l_max=3.0,
        lattice_c=6.978,
        positive_qz_min=0.0,
    )
    assert invalid_marker_bounds == pytest.approx((expected_lo, expected_hi))

    reversed_marker_bounds = namespace["specular_qz_bounds_for_l_window"](
        np.linspace(2.5, 4.0, 20),
        pd.DataFrame({"qz_marker": [3.64, 3.56], "fit_l": [2.0, 3.0]}),
        l_min=1.5,
        l_max=3.0,
        lattice_c=6.978,
        positive_qz_min=0.0,
    )
    assert reversed_marker_bounds == pytest.approx((expected_lo, expected_hi))


def test_parallel_script_specular_qz_bounds_do_not_use_fallback_decision() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "def specular_qz_bounds_need_fallback(" not in source
    assert "specular_use_fallback_qz_bounds" not in source


def test_parallel_script_specular_profile_uses_strict_qr_band_without_fallback_mask() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    specular_profile_start = source.index(
        "initial_specular_profile = build_specular_qr_rod_profile("
    )
    specular_profile_source = source[
        specular_profile_start : source.index(
            "for rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:",
            specular_profile_start,
        )
    ]

    assert "build_specular_qr_rod_profile(" in specular_profile_source
    assert "delta_qr=qr_rod_delta_qr" not in specular_profile_source
    assert "specular_qz_values.size >= 2" not in specular_profile_source
    assert "specular_fallback_region_mask" not in specular_profile_source
    assert "specular_profile_mask_override" not in specular_profile_source
    assert "specular_qz_edges_for_l_window" not in specular_profile_source
    assert "specular_profile_l_min" not in specular_profile_source
    assert "specular_profile_l_max" not in specular_profile_source


def test_detector_region_centerlines_clip_to_visual_qz_bounds() -> None:
    namespace = _script_functions("clipped_detector_trace_to_qz_bounds")
    clip_trace = namespace["clipped_detector_trace_to_qz_bounds"]

    clipped_x, clipped_y = clip_trace(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([-0.1, 0.2, 0.7, 1.1]),
        (0.0, 0.8),
    )

    assert np.isnan(clipped_x[0])
    assert np.isnan(clipped_y[3])
    np.testing.assert_allclose(clipped_x[1:3], np.array([2.0, 3.0]))
    np.testing.assert_allclose(clipped_y[1:3], np.array([20.0, 30.0]))

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("rod_label_entries: list[dict[str, object]] = []") : source.index(
            "specular_color = OKABE_ITO"
        )
    ]
    assert "detector_overlay_qz_bounds_by_key" in detector_figure_source
    assert "trace_qz_bounds = detector_overlay_qz_bounds_by_key.get(" in detector_figure_source
    assert (
        "if trace_qz_bounds is None:\n            projected_col = raw_projected_col.copy()"
        in detector_figure_source
    )
    assert "label_id = detector_rod_label_id(m_value, branch_suffix)" in detector_figure_source
    assert "append_detector_rod_label_entry(" in detector_figure_source
    assert "clipped_detector_trace_to_qz_bounds(" in detector_figure_source


def test_detector_region_labels_are_not_dropped_when_branch_overlay_bounds_are_missing() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("rod_label_entries: list[dict[str, object]] = []") : source.index(
            "specular_color = OKABE_ITO"
        )
    ]

    assert "if trace_qz_bounds is None:\n            continue" not in detector_figure_source
    assert "should_draw_centerline = trace_qz_bounds is not None" in detector_figure_source
    assert (
        'branch_suffix = "+" if int(payload.get("branch_sign", 0)) > 0 else "-"'
        in detector_figure_source
    )
    assert '"text": detector_rod_label(m_value, branch_suffix)' in detector_figure_source


def test_detector_region_labels_fallback_to_drawn_band_centerlines() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("rod_label_entries: list[dict[str, object]] = []") : source.index(
            "specular_color = OKABE_ITO"
        )
    ]

    assert "detector_label_ids_added: set[str] = set()" in detector_figure_source
    assert "def append_detector_rod_label_entry(" in detector_figure_source
    assert (
        "detector_visual_label_lines = detector_mask_centerline_from_visual(visual)"
        in detector_figure_source
    )
    assert "for visual_col, visual_row in detector_visual_label_lines:" in detector_figure_source
    assert "append_detector_rod_label_entry(" in detector_figure_source
    assert "if label_id in detector_label_ids_added:" in detector_figure_source


def test_detector_region_axis_tick_labels_use_bottom_left_origin() -> None:
    namespace = _script_functions("detector_bottom_left_axis_tick_labels")
    axis_labels = namespace["detector_bottom_left_axis_tick_labels"]

    x_labels, y_labels = axis_labels(
        np.array([9.5, 59.5, 109.5]),
        np.array([20.5, 70.5, 120.5]),
        xlim=(9.5, 109.5),
        ylim=(120.5, 20.5),
    )

    assert x_labels == ["0", "50", "100"]
    assert y_labels == ["100", "50", "0"]

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_region_source = source[
        source.index('ax.set_xlabel("Detector x pixel') : source.index(
            "\nrod_label_entries = apply_unified_qr_rod_region_editor_labels("
        )
    ]
    assert "detector_bottom_left_axis_tick_labels(" in detector_region_source
    assert 'ax.set_xlabel("Detector x pixel (bottom-left origin)")' in detector_region_source
    assert 'ax.set_ylabel("Detector y pixel (bottom-left origin)")' in detector_region_source


def test_detector_region_specular_label_can_use_visual_mask_centerline() -> None:
    namespace = _script_functions("detector_mask_centerline_from_visual")
    namespace["detector_region_shape_mask"] = np.ones((5, 6), dtype=bool)
    centerline_from_visual = namespace["detector_mask_centerline_from_visual"]

    visual = {"band_fill_mask": np.zeros((5, 6), dtype=bool)}
    visual["band_fill_mask"][1, 2:5] = True
    visual["band_fill_mask"][2, 3:5] = True
    lines = centerline_from_visual(visual)

    assert len(lines) == 1
    np.testing.assert_allclose(lines[0][0], np.array([3.0, 3.5]))
    np.testing.assert_allclose(lines[0][1], np.array([1.0, 2.0]))

    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_source = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "\ndisplay_detector_rotation_fit_debug("
        )
    ]
    assert "detector_mask_centerline_from_visual(specular_delta_q_visual)" in detector_figure_source
    assert '"label_id": detector_specular_label_id()' in detector_figure_source


def test_parallel_script_rod_profile_panel_label_uses_m_without_branch_word() -> None:
    namespace = _script_functions("rod_profile_panel_label")
    panel_label = namespace["rod_profile_panel_label"]

    assert panel_label(0, "qz") == "$m = 0$"
    assert panel_label(7, "+") == "$m = 7\\ +$"
    assert panel_label(7, "-") == "$m = 7\\ -$"
    assert "branch" not in panel_label(7, "+").lower()


def test_joint_qz_fit_keeps_close_peak_valley_low() -> None:
    namespace = _script_functions(
        "gaussian_sum_qz_model",
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


def test_parallel_script_joint_qz_fit_keeps_bi2se3_low_l_specular_marker() -> None:
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
    bi2se3_x = 0.0268468453927604 + 0.018013816128405698 * np.arange(96, dtype=np.float64)
    bi2se3_y = np.array(
        [
            1443.4,
            2499.57142857143,
            14085.3333333333,
            6859.64285714286,
            2383.6,
            728.1,
            258.357142857143,
            107.708333333333,
            78.1481481481482,
            70.15,
            68.1212121212121,
            63.4791666666667,
            60.8205128205128,
            61.9285714285714,
            59.2222222222222,
            59.3333333333333,
            58.0588235294118,
            59.0,
            59.6842105263158,
            69.551724137931,
            76.472972972973,
            71.5866666666667,
            73.3823529411765,
            71.1684210526316,
            71.4931506849315,
            72.75,
            72.4050632911392,
            73.8988764044944,
            76.9444444444444,
            82.6704545454545,
            93.0650406504065,
            125.440860215054,
            194.625954198473,
            683.878787878788,
            8644.70161290323,
            6778.25833333333,
            782.1875,
            224.77397260274,
            112.201754385965,
            83.0063291139241,
            68.875,
            62.0180722891566,
            59.0714285714286,
            56.6900584795322,
            54.955223880597,
            53.2080924855491,
            54.0958904109589,
            53.9828571428571,
            52.3417721518987,
            54.0842696629214,
            51.6745562130178,
            51.5028248587571,
            52.8858695652174,
            51.7668711656442,
            50.2857142857143,
            49.0920245398773,
            48.8636363636364,
            50.1179577464789,
            49.3070175438597,
            50.1929824561403,
            49.2022471910112,
            49.3412322274881,
            50.0974358974359,
            50.3039647577093,
            51.121359223301,
            52.6206896551724,
            52.246511627907,
            53.1402714932127,
            58.0207468879668,
            71.0575221238938,
            140.547008547009,
            261.040650406504,
            136.445833333333,
            61.4526748971193,
            53.4365079365079,
            50.3968871595331,
            49.7176470588235,
            48.3667953667954,
            47.2659176029963,
            47.5666666666667,
            47.6557971014493,
            46.054347826087,
            46.1474820143885,
            46.2473498233216,
            45.1003460207612,
            45.7898305084746,
            44.2263513513514,
            44.1196013289037,
            42.4901960784314,
            42.974025974026,
            42.4935897435898,
            41.7476340694006,
            41.5423197492163,
            40.5401234567901,
            40.226586102719,
            40.540059347181,
        ],
        dtype=np.float64,
    )
    bi2se3_markers = np.array(
        [
            0.06287447764957185,
            0.3871231679608744,
            0.5132198808597143,
            0.6393165937585541,
            1.305827790509565,
        ],
        dtype=np.float64,
    )
    bi2te3_x = 0.0263476507894132 + 0.0169190889522689 * np.arange(96, dtype=np.float64)
    bi2te3_y = np.array(
        [
            302.833333333333,
            1198.0,
            11320.0,
            8735.25,
            2900.55,
            974.0,
            359.714285714286,
            155.0,
            87.2903225806452,
            72.5757575757576,
            69.2424242424242,
            65.5454545454545,
            63.7631578947368,
            63.8653846153846,
            63.1818181818182,
            59.7555555555556,
            58.7916666666667,
            58.8235294117647,
            58.112676056338,
            60.0526315789474,
            70.3333333333333,
            75.4590163934426,
            68.9384615384615,
            65.7078651685393,
            66.0138888888889,
            67.5694444444444,
            65.2894736842105,
            65.8846153846154,
            67.1559633027523,
            69.5833333333333,
            72.6279069767442,
            81.7777777777778,
            99.6893203883495,
            381.739130434783,
            3156.94897959184,
            2294.55,
            300.730769230769,
            128.585714285714,
            95.8545454545455,
            92.1801801801802,
            87.4521739130435,
            84.1369863013699,
            71.8692307692308,
            67.2032520325203,
            64.8548387096774,
            61.5660377358491,
            59.2638888888889,
            57.8208955223881,
            56.6544117647059,
            58.0,
            57.4575163398693,
            55.0763888888889,
            54.1987179487179,
            54.4301075268817,
            53.2981366459627,
            54.1602564102564,
            52.6772151898734,
            52.122905027933,
            52.98125,
            53.1298701298701,
            52.9108910891089,
            50.6627118644068,
            51.2416666666667,
            51.8222222222222,
            51.3888888888889,
            51.9562841530055,
            51.2068965517241,
            51.1511111111111,
            51.5492227979275,
            54.3744292237443,
            69.9118942731278,
            76.4146341463415,
            62.0165975103734,
            51.6975609756098,
            50.9752066115703,
            49.8709677419355,
            49.3252032520325,
            50.1294642857143,
            49.4186991869919,
            49.2987012987013,
            49.1254901960784,
            49.5234042553192,
            48.8129770992366,
            49.4177215189873,
            48.58984375,
            48.3372093023256,
            48.1372549019608,
            47.9054545454546,
            46.57421875,
            46.8438661710037,
            46.4822695035461,
            45.4172932330827,
            45.7921146953405,
            45.28125,
            43.8239436619718,
            44.3111888111888,
        ],
        dtype=np.float64,
    )
    bi2te3_markers = np.array(
        [
            0.06699728923054353,
            0.3778623082216827,
            0.615943649473,
            0.7053808103730614,
            1.231121522995,
        ],
        dtype=np.float64,
    )

    bi2se3_payload = fit_joint_qz_peak_sum(bi2se3_x, bi2se3_y, bi2se3_markers)
    bi2te3_payload = fit_joint_qz_peak_sum(bi2te3_x, bi2te3_y, bi2te3_markers)

    def _log_profile_rms(
        x_values: np.ndarray,
        observed: np.ndarray,
        model: np.ndarray,
        *,
        qz_min: float | None = None,
        qz_max: float | None = None,
    ) -> float:
        positive_observed = observed[np.isfinite(observed) & (observed > 0.0)]
        floor = (
            max(float(np.nanpercentile(positive_observed, 5.0)) * 0.05, 1.0e-9)
            if positive_observed.size
            else 1.0e-9
        )
        mask = (
            np.isfinite(x_values)
            & np.isfinite(observed)
            & np.isfinite(model)
            & (observed > 0.0)
            & (model > 0.0)
        )
        if qz_min is not None:
            mask &= x_values >= float(qz_min)
        if qz_max is not None:
            mask &= x_values <= float(qz_max)
        assert np.count_nonzero(mask) >= 3
        log_delta = np.log10(model[mask] + floor) - np.log10(observed[mask] + floor)
        return float(np.sqrt(np.nanmean(log_delta**2)))

    assert bi2se3_payload["success"] is True
    bi2se3_components = bi2se3_payload["components"]
    bi2se3_component_markers = sorted(float(component["marker"]) for component in bi2se3_components)
    assert len(bi2se3_component_markers) == 4
    assert any(abs(marker - 0.3871231679608744) <= 0.03 for marker in bi2se3_component_markers)
    assert not any(abs(marker - 0.5132198808597143) <= 0.03 for marker in bi2se3_component_markers)
    weak_bi2se3 = min(
        bi2se3_components,
        key=lambda component: abs(float(component["marker"]) - 0.3871231679608744),
    )
    assert abs(float(weak_bi2se3["marker"]) - 0.3871231679608744) <= 0.03
    assert np.nanmax(np.asarray(weak_bi2se3["density"], dtype=np.float64)) > 1.0
    bi2se3_model = np.asarray(bi2se3_payload["model_density"], dtype=np.float64)
    assert _log_profile_rms(bi2se3_x, bi2se3_y, bi2se3_model) < 0.075
    assert (
        _log_profile_rms(
            bi2se3_x,
            bi2se3_y,
            bi2se3_model,
            qz_min=0.12,
            qz_max=0.36,
        )
        < 0.16
    )
    for strong_marker in (0.06287447764957185, 0.6393165937585541, 1.305827790509565):
        assert any(abs(marker - strong_marker) <= 0.03 for marker in bi2se3_component_markers)

    assert bi2te3_payload["success"] is True
    bi2te3_component_markers = sorted(
        float(component["marker"]) for component in bi2te3_payload["components"]
    )
    assert len(bi2te3_component_markers) == 5
    assert any(abs(marker - 0.3778623082216827) <= 0.03 for marker in bi2te3_component_markers)
    bi2te3_model = np.asarray(bi2te3_payload["model_density"], dtype=np.float64)
    assert _log_profile_rms(bi2te3_x, bi2te3_y, bi2te3_model) < 0.08


def test_parallel_script_tail_component_aggregation_rejects_shape_mismatch() -> None:
    namespace = _script_functions("_aggregate_tail_components")
    aggregate_tail_components = namespace["_aggregate_tail_components"]

    components = aggregate_tail_components(
        np.ones((3, 1), dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        [{"peak_index": 0, "hwhm": 0.1, "tail_power": 2.0}],
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([True, True], dtype=bool),
        1.0,
    )

    assert components == []


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
        "SAMPLE_NAME_OVERRIDE_TEXT or _sample_name_from_path(primary_cif_path)",
    ):
        assert token in source
    assert "SAMPLE_NAME_OVERRIDE_TEXT = _setting_text(" in source
    assert '"SAMPLE_NAME_OVERRIDE", "RA_SIM_ALL_BACKGROUND_SAMPLE_NAME", ""' in source


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


def test_parallel_script_routes_pbi2_figures_to_manuscript_pbi2_directory() -> None:
    namespace = _script_functions("figure_output_dir_for_sample")
    figure_output_dir_for_sample = namespace["figure_output_dir_for_sample"]
    repo_root = Path(r"C:\repo")

    assert figure_output_dir_for_sample(
        sample_stem="pbi2",
        root=repo_root,
        override="",
    ) == Path(
        r"C:\Users\Kenpo\OneDrive\Documents\GitHub\PhD Work\2D-Manuscript-Draft\figures\results_pbi2"
    )


def test_parallel_script_keeps_non_pbi2_figures_in_ordered_manuscript_directory() -> None:
    namespace = _script_functions("figure_output_dir_for_sample")
    figure_output_dir_for_sample = namespace["figure_output_dir_for_sample"]
    repo_root = Path(r"C:\repo")

    assert figure_output_dir_for_sample(
        sample_stem="bi2se3",
        root=repo_root,
        override="",
    ) == Path(
        r"C:\Users\Kenpo\OneDrive\Documents\GitHub\PhD Work\2D-Manuscript-Draft\figures\results_ordered"
    )


def test_parallel_script_figure_output_override_wins_for_all_samples() -> None:
    namespace = _script_functions("figure_output_dir_for_sample")
    figure_output_dir_for_sample = namespace["figure_output_dir_for_sample"]
    repo_root = Path(r"C:\repo")

    assert (
        figure_output_dir_for_sample(
            sample_stem="pbi2",
            root=repo_root,
            override="custom/figures",
        )
        == repo_root / "custom" / "figures"
    )


def test_parallel_script_refreshes_figure_output_after_state_sample_detection() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    sample_detection = source.index("detected_sample_name = (")
    refresh_stems = source.index("refresh_figure_stems()", sample_detection)
    refresh_output = source.index("refresh_figure_output_dir()", refresh_stems)
    active_lattice = source.index("ACTIVE_LATTICE = active_lattice_constants_from_state(state)")

    assert refresh_stems < refresh_output < active_lattice


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


def test_parallel_script_direct_windows_process_backend_reenters_guarded_runner() -> None:
    namespace = _script_functions("should_reenter_guarded_process_runner")
    should_reenter = namespace["should_reenter_guarded_process_runner"]

    assert should_reenter(
        "process",
        workers=28,
        platform_name="nt",
        process_guard_enabled=False,
    )
    assert should_reenter(
        "auto",
        workers=28,
        platform_name="nt",
        process_guard_enabled=False,
    )
    assert not should_reenter(
        "process",
        workers=28,
        platform_name="nt",
        process_guard_enabled=True,
    )
    assert not should_reenter(
        "thread",
        workers=28,
        platform_name="nt",
        process_guard_enabled=False,
    )
    assert not should_reenter(
        "process",
        workers=1,
        platform_name="nt",
        process_guard_enabled=False,
    )
    assert not should_reenter(
        "process",
        workers=28,
        platform_name="posix",
        process_guard_enabled=False,
    )


def test_parallel_script_guarded_process_runner_command_forwards_run_contract() -> None:
    namespace = _script_functions("guarded_process_runner_command")
    command_for_runner = namespace["guarded_process_runner_command"]

    command = command_for_runner(
        python_executable=r"C:\Python313\python.exe",
        runner_path=r"C:\repo\scripts\diagnostics\run_all_background_peak_fits.py",
        diagnostic_path=r"C:\repo\scripts\diagnostics\all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py",
        state_path=r"C:\Users\Kenpo\.local\share\ra_sim\Bi2Se3.json",
        run_name="Bi2Se3",
        fit_workers=28,
        numba_threads=24,
        process_numba_threads=1,
    )

    assert command == [
        r"C:\Python313\python.exe",
        r"C:\repo\scripts\diagnostics\run_all_background_peak_fits.py",
        "--notebook",
        r"C:\repo\scripts\diagnostics\all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py",
        "--run-name",
        "Bi2Se3",
        "--fit-backend",
        "process",
        "--fit-workers",
        "28",
        "--numba-threads",
        "24",
        "--process-numba-threads",
        "1",
        r"C:\Users\Kenpo\.local\share\ra_sim\Bi2Se3.json",
    ]


def test_parallel_script_direct_process_reentry_is_before_backend_normalization_and_prep() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    reentry_call = source.index("if should_reenter_guarded_process_runner(")
    normalize_call = source.index("FIT_BACKEND = normalize_fit_backend(")
    prep_start = source.index("prep_total_start = time.perf_counter()")

    assert reentry_call < normalize_call < prep_start
    assert "direct Windows process backend: launching guarded runner" in source
    assert "subprocess.run(" in source


def test_parallel_script_00l_region_crop_bounds_spans_mask_top_to_beam_center() -> None:
    namespace = _script_functions("hk0_00l_region_crop_bounds")
    crop_bounds = namespace["hk0_00l_region_crop_bounds"]
    region_mask = np.zeros((200, 300), dtype=bool)
    region_mask[60:125, 145:156] = True

    bounds = crop_bounds(
        (200, 300),
        (150.0, 160.0),
        region_mask,
        lateral_half_width_px=48,
    )

    assert bounds is not None
    row_slice, col_slice = bounds
    assert row_slice.start == 60
    assert row_slice.stop == 161
    assert col_slice.start == 102
    assert col_slice.stop == 199


def test_parallel_script_00l_region_crop_bounds_clips_and_rejects_invalid() -> None:
    namespace = _script_functions("hk0_00l_region_crop_bounds")
    crop_bounds = namespace["hk0_00l_region_crop_bounds"]
    region_mask = np.zeros((80, 70), dtype=bool)
    region_mask[1:60, 2:8] = True

    row_slice, col_slice = crop_bounds(
        (80, 70),
        (4.0, 78.0),
        region_mask,
        lateral_half_width_px=10,
    )

    assert row_slice.start == 1
    assert row_slice.stop == 79
    assert col_slice.start == 0
    assert col_slice.stop == 15
    assert crop_bounds((80, 70), (np.nan, 78.0), region_mask) is None
    assert crop_bounds((80, 70), (4.0, 78.0), None) is None
    assert crop_bounds((80, 70), (4.0, 78.0), np.zeros((79, 70), dtype=bool)) is None
    assert crop_bounds((80, 70), (4.0, 0.0), region_mask) is None


def test_parallel_script_00l_region_save_writes_vertical_and_horizontal_png(tmp_path) -> None:
    namespace = _script_functions(
        "detector_display_cmap",
        "detector_intensity_display",
        "detector_log_norm",
        "hk0_00l_region_crop_bounds",
        "save_hk0_00l_region_crop",
    )
    save_crop = namespace["save_hk0_00l_region_crop"]
    image = np.arange(10_000, dtype=np.float64).reshape(100, 100)
    region_mask = np.zeros(image.shape, dtype=bool)
    region_mask[28:75, 45:55] = True
    out_path = tmp_path / "00L_region.png"
    horizontal_path = tmp_path / "00L_region_horizontal.png"

    result = save_crop(
        image,
        out_path,
        horizontal_output_path=horizontal_path,
        beam_center=(50.0, 82.0),
        region_mask=region_mask,
        lateral_half_width_px=9,
    )

    assert result == (out_path, horizontal_path)
    assert out_path.exists()
    assert horizontal_path.exists()
    assert out_path.stat().st_size > 0
    assert horizontal_path.stat().st_size > 0
    rendered = plt.imread(out_path)
    horizontal = plt.imread(horizontal_path)
    assert rendered.shape[-1] in {3, 4}
    assert horizontal.shape[-1] in {3, 4}
    assert horizontal.shape[1] == rendered.shape[0]
    assert not np.allclose(rendered[..., 0], rendered[..., 1])


def test_parallel_script_00l_region_save_uses_detector_log_color() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def save_hk0_00l_region_crop(") : source.index("\ndef label_from_entry(")
    ]

    assert "detector_display_cmap()" in function_source
    assert "detector_intensity_display(crop)" in function_source
    assert "detector_log_norm([display_crop]" in function_source
    assert "np.rot90(rgba_crop, k=1)" in function_source
    assert 'cmap="gray"' not in function_source


def test_parallel_script_00l_region_output_is_wired() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    for token in (
        '"00L_region.png"',
        '"00L_region_horizontal.png"',
        "hk0_00l_region_mask =",
        '"band_fill_mask"',
        "save_hk0_00l_region_crop(",
        "region_mask=hk0_00l_region_mask",
        "skipped 00L_region.png",
    ):
        assert token in source
    export_block = source[
        source.index("beam_center = (") : source.index("for label_entry in rod_label_entries:")
    ]
    assert "hk0_00l_region_endpoint_from_line" not in export_block
    assert "qz_values_to_l_axis(" not in export_block
    assert "detector_qz_values_for_polyline(" not in export_block
    assert "target_l=16.0" not in export_block
    assert "marker_source=marker_table" not in export_block
    assert "hk0_l3_star" not in source


def test_parallel_script_00l_region_uses_specular_two_theta_roi_for_visual() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_export = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "hk0_00l_region_path = FIGURE_OUT_DIR"
        )
    ]

    assert "specular_visual_qz_bounds = None" in detector_export
    assert "specular_visual_qz_bounds = specular_qz_bounds_for_l_window(" not in detector_export
    assert "l_min=qr_rod_specular_editor_l_min" not in detector_export
    assert "l_max=qr_rod_specular_editor_l_max" not in detector_export
    assert "(specular_detector_qz_map >= float(specular_visual_qz_bounds[0]))" in detector_export
    assert "(specular_detector_qz_map <= float(specular_visual_qz_bounds[1]))" in detector_export
    assert "specular_detector_qz_values.size >= 2" in detector_export


def test_parallel_script_00l_visual_uses_strict_hk0_roi_not_override_mask() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_export = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "hk0_00l_region_path = FIGURE_OUT_DIR"
        )
    ]

    assert "specular_delta_q_visual = {" in detector_export
    assert "specular_roi_visual_mask = (" in detector_export
    assert "& specular_detector_phi_mask" in detector_export
    assert '"band_fill_mask": specular_roi_visual_mask' in detector_export
    assert "build_detector_selected_qr_rod_band_visual_payload(" not in detector_export
    assert "specular_override_visual = qr_rod_profile_mask_visual_payload(" not in detector_export
    assert "if specular_profile_mask_override is not None:" not in detector_export


def test_parallel_script_detector_hk0_region_uses_prominent_specular_style() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'specular_color = OKABE_ITO["sky"]' in source
    assert "detector_region_specular_centerline_lw = detector_region_centerline_lw" in source
    assert "detector_region_specular_band_alpha = 0.42" in source
    assert "detector_region_specular_boundary_alpha = 1.0" in source
    assert "detector_region_specular_boundary_expand_px = 2" in source
    assert 'specular_color = OKABE_ITO["purple"]' not in source
    assert "path_effects=detector_region_specular_path_effects" not in source


def test_parallel_script_detector_hk0_delta_q_draw_uses_specular_style() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    specular_draw_start = source.index(
        "draw_detector_delta_q_region(\n        specular_delta_q_visual"
    )
    specular_draw = source[
        specular_draw_start : source.index(
            "specular_lines = detector_mask_centerline_from_visual", specular_draw_start
        )
    ]

    assert "fill_alpha=detector_region_specular_band_alpha" in specular_draw
    assert "boundary_alpha=detector_region_specular_boundary_alpha" in specular_draw
    assert "boundary_expand_px=detector_region_specular_boundary_expand_px" in specular_draw
    assert (
        "specular_roi_visual_mask = ("
        in source[source.index("specular_delta_q_visual = None") : specular_draw_start]
    )


def test_parallel_script_detector_export_clips_nonzero_rods_to_accepted_l_window() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_export = source[
        source.index("detector_overlay_rods = detector_overlay_rod_entries(") : source.index(
            "specular_color = OKABE_ITO"
        )
    ]

    for token in (
        "l_reference_rows(",
        "marker_source=plot_marker_table",
        "qz_bounds_for_l_window(",
        "l_min=qr_rod_editor_l_min",
        "l_max=qr_rod_editor_l_max",
        "positive_qz_min=POSITIVE_QZ_MIN",
    ):
        assert token in detector_export


def test_parallel_script_hk0_delta_qr_does_not_replace_nonzero_delta_qr() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    specular_result = source.index("qr_rod_specular_editor_result = edit_qr_rod_region_editor(")
    combined_result = source.index("qr_rod_region_editor_result = {", specular_result)
    specular_section = source[specular_result:combined_result]

    assert "qr_rod_specular_delta_qr" not in specular_section
    assert "qr_rod_delta_qr = max(" not in specular_section
    assert '"delta_qr": float(qr_rod_delta_qr)' in source[combined_result:]
    assert '"specular_delta_qr"' not in source[combined_result:]
    assert '"specular_editor_delta_qr"' not in source


def test_parallel_script_detector_mask_expansion_thickens_boundary_only() -> None:
    namespace = _script_functions("expanded_detector_mask")
    expand_mask = namespace["expanded_detector_mask"]
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    expanded = expand_mask(mask, radius_px=1)

    assert int(np.count_nonzero(expanded)) == 9
    assert np.array_equal(expand_mask(mask, radius_px=0), mask)
    assert not np.shares_memory(expand_mask(mask, radius_px=0), mask)


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
    markers = pd.DataFrame([{"m": 1, "branch": "+", "hkl": "-1,0,2", "qz_marker": 1.0}])
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


def test_parallel_script_qr_rod_marker_table_skips_specular_qz_duplicates() -> None:
    namespace = _script_functions("marker_table_with_specular_l_markers")
    include_specular = namespace["marker_table_with_specular_l_markers"]
    markers = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "hkl": "manual",
                "qz_marker": 6.487064476439,
                "fit_l": 6.875686263121,
            },
        ]
    )
    specular = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "hkl": "0,0,7",
                "qz_marker": 6.487064476439101,
                "fit_l": 7.0,
            },
            {"m": 0, "branch": "qz", "hkl": "0,0,8", "qz_marker": 7.413787973073, "fit_l": 8.0},
        ]
    )

    merged = include_specular(markers, specular)

    hk0 = merged[merged["m"] == 0]
    assert hk0["qz_marker"].tolist() == pytest.approx([6.487064476439, 7.413787973073])
    assert hk0["fit_l"].tolist() == pytest.approx([6.875686263121, 8.0])


def test_parallel_script_qr_rod_marker_table_preserves_manual_duplicate_hkl_rows() -> None:
    namespace = _script_functions("marker_table_with_specular_l_markers")
    include_specular = namespace["marker_table_with_specular_l_markers"]
    markers = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "hkl": "0,0,3", "qz_marker": 1.0},
            {"m": 0, "branch": "qz", "hkl": "0,0,3", "qz_marker": 1.4},
        ]
    )
    specular = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "hkl": "0,0,3", "qz_marker": 1.0},
            {"m": 0, "branch": "qz", "hkl": "0,0,6", "qz_marker": 2.0},
        ]
    )

    merged = include_specular(markers, specular)

    assert merged[merged["m"] == 0]["qz_marker"].tolist() == [1.0, 1.4, 2.0]


def test_parallel_script_qr_rod_marker_table_avoids_empty_concat_warning() -> None:
    namespace = _script_functions("marker_table_with_specular_l_markers")
    include_specular = namespace["marker_table_with_specular_l_markers"]
    markers = pd.DataFrame([{"m": 1, "branch": "+", "hkl": "1,0,1", "qz_marker": 1.0}])
    specular = pd.DataFrame(columns=markers.columns)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", FutureWarning)
        merged = include_specular(markers, specular)

    assert merged["qz_marker"].tolist() == [1.0]
    assert not [
        item
        for item in captured
        if issubclass(item.category, FutureWarning)
        and "DataFrame concatenation with empty or all-NA entries" in str(item.message)
    ]


def test_parallel_script_final_rod_profile_entries_exclude_empty_hk_rows() -> None:
    namespace = _script_functions(
        "qz_l_linear_coeff_from_marker_rows",
        "drawable_rod_profile_keys",
    )
    drawable_keys = namespace["drawable_rod_profile_keys"]
    profile_table = pd.DataFrame(
        [
            {
                "m": 1,
                "branch": "+",
                "qz_center": 1.0,
                "pixel_count": 5,
                "background_density": 2.0,
                "joint_peak_density": 0.5,
            },
            {
                "m": 1,
                "branch": "+",
                "qz_center": 1.5,
                "pixel_count": 6,
                "background_density": 2.5,
                "joint_peak_density": 0.6,
            },
            {
                "m": 7,
                "branch": "+",
                "qz_center": 1.0,
                "pixel_count": 0,
                "background_density": np.nan,
                "joint_peak_density": np.nan,
            },
            {
                "m": 7,
                "branch": "-",
                "qz_center": 1.4,
                "pixel_count": 0,
                "background_density": np.nan,
                "joint_peak_density": np.nan,
            },
            {
                "m": 3,
                "branch": "-",
                "qz_center": 1.0,
                "pixel_count": 5,
                "background_density": 1.0,
                "joint_peak_density": 0.2,
            },
            {
                "m": 3,
                "branch": "-",
                "qz_center": 1.5,
                "pixel_count": 5,
                "background_density": 1.2,
                "joint_peak_density": 0.3,
            },
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 0.8,
                "pixel_count": 4,
                "background_density": 3.0,
                "joint_peak_density": 0.7,
            },
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 1.1,
                "pixel_count": 4,
                "background_density": 3.2,
                "joint_peak_density": 0.8,
            },
        ]
    )
    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 1.5, "fit_l": 3.0},
            {"m": 3, "branch": "-", "qz_marker": 1.0, "fit_l": -2.0},
            {"m": 3, "branch": "-", "qz_marker": 1.5, "fit_l": -1.0},
            {"m": 0, "branch": "qz", "qz_marker": 0.8, "fit_l": 3.0},
            {"m": 0, "branch": "qz", "qz_marker": 1.1, "fit_l": 4.0},
        ]
    )

    assert drawable_keys(profile_table, marker_table) == {(0, "qz"), (1, "+")}


def test_parallel_script_final_rod_profile_figure_filters_empty_entries() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    helper_def = source.index("def drawable_rod_profile_keys(")
    plot_keys = source.index("drawable_profile_keys = drawable_rod_profile_keys(", helper_def)
    plot_entries = source.index("plot_rod_entries =", plot_keys)
    figure = source.index("fig = plt.figure(", plot_entries)

    assert helper_def < plot_keys < plot_entries < figure
    assert "plot_rod_entries = rod_entries + [specular_rod_entry]" not in source
    assert 'if (0, "qz") in drawable_profile_keys:' in source
    assert "skipped HK=0 final figure row: no drawable real m=0/qz profile rows" in source


def test_parallel_script_shared_rod_profile_l_axis_limits_span_all_profile_axes() -> None:
    namespace = _script_functions(
        "l_reference_rows",
        "qz_values_to_l_axis",
        "shared_rod_profile_l_axis_limits",
    )
    shared_limits = namespace["shared_rod_profile_l_axis_limits"]
    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0, "l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 4.0, "l": 4.0},
            {"m": 1, "branch": "-", "qz_marker": 1.0, "fit_l": 2.0, "l": 2.0},
            {"m": 1, "branch": "-", "qz_marker": 2.0, "fit_l": 4.0, "l": 4.0},
            {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 3.0, "l": 3.0},
        ]
    )
    namespace["plot_marker_table"] = marker_table
    profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 1.0, "pixel_count": 5},
            {"m": 1, "branch": "+", "qz_center": 2.5, "pixel_count": 5},
            {"m": 1, "branch": "-", "qz_center": 3.0, "pixel_count": 0},
            {"m": 0, "branch": "qz", "qz_center": 1.2, "pixel_count": 0},
        ]
    )
    plot_entries = [{"m": 1}, {"m": 0}]
    phi_windows = [("-", -90.0, 0.0, "-"), ("+", 0.0, 90.0, "+")]

    np.testing.assert_allclose(
        shared_limits(
            profile_table,
            marker_table,
            plot_entries,
            phi_windows,
            min_l=2.0,
            fallback_span=1.0,
        ),
        (2.0, 6.0),
    )
    np.testing.assert_allclose(
        shared_limits(
            profile_table,
            marker_table,
            plot_entries,
            phi_windows,
            min_l=0.5,
            max_l=3.0,
            fallback_span=1.0,
        ),
        (0.5, 3.0),
    )


def test_parallel_script_shared_nonzero_rod_profile_y_axis_limits_ignore_hk0() -> None:
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
        "normalized_data_simulation_payload",
        "rod_profile_normalized_payload_for_plot_decision",
        "l_reference_rows",
        "qz_values_to_l_axis",
        "shared_nonzero_rod_profile_y_axis_limits",
    )
    shared_limits = namespace["shared_nonzero_rod_profile_y_axis_limits"]
    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0, "l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 4.0, "l": 4.0},
            {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 3.0, "l": 3.0},
        ]
    )
    profile_table = pd.DataFrame(
        [
            {
                "m": 1,
                "branch": "+",
                "qz_center": 1.5,
                "pixel_count": 5,
                "background_density": 4.0,
                "joint_peak_density": 1.5,
                "joint_linear_baseline_density": 1.0,
                "fit_density": 1.5,
            },
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 1.5,
                "pixel_count": 5,
                "background_density": 1.0e6,
                "joint_peak_density": 1.0e6,
                "joint_linear_baseline_density": 0.0,
                "fit_density": 1.0e6,
            },
        ]
    )

    y_min, y_max = shared_limits(
        profile_table,
        marker_table,
        [{"m": 1}, {"m": 0}],
        [("-", -90.0, 0.0, "-"), ("+", 0.0, 90.0, "+")],
        fallback_limits=(-1.0, 1.0),
    )

    assert y_min < y_max
    assert -0.2 < y_min < 0.0
    assert 1.0 < y_max < 1.2


def test_parallel_script_shared_nonzero_rod_profile_y_axis_limits_fallback_without_nonzero_data() -> (
    None
):
    namespace = _script_functions(
        "_finite_abs_percentile",
        "rod_profile_marker_l_mapping_is_valid",
        "rod_profile_plot_model_decision",
        "normalized_data_simulation_payload",
        "rod_profile_normalized_payload_for_plot_decision",
        "l_reference_rows",
        "qz_values_to_l_axis",
        "shared_nonzero_rod_profile_y_axis_limits",
    )
    shared_limits = namespace["shared_nonzero_rod_profile_y_axis_limits"]
    profile_table = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 1.5,
                "pixel_count": 5,
                "background_density": 1.0e6,
                "joint_peak_density": 1.0e6,
                "joint_linear_baseline_density": 0.0,
                "fit_density": 1.0e6,
            },
        ]
    )

    assert shared_limits(
        profile_table,
        pd.DataFrame(),
        [{"m": 1}, {"m": 0}],
        [("-", -90.0, 0.0, "-"), ("+", 0.0, 90.0, "+")],
        fallback_limits=(-1.0, 1.0),
    ) == (-1.0, 1.0)


def test_parallel_script_final_rod_profile_axes_use_shared_l_limits_except_hk0() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    helper_def = source.index("def shared_rod_profile_l_axis_limits(")
    nonzero_entries = source.index("nonzero_plot_rod_entries =", helper_def)
    limits_assign = source.index("rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits(")
    hk0_limits = source.index(
        "rod_profile_hk0_l_axis_limits = rod_profile_l_axis_limits_for_sample("
    )
    figure_loop = source.index("for row, rod in enumerate(plot_rod_entries):", limits_assign)

    assert helper_def < nonzero_entries < limits_assign < hk0_limits < figure_loop
    assert "NONZERO_QR_ROD_L_MIN = 0.5" in source
    assert "NONZERO_QR_ROD_L_MAX = 3.0" in source
    assert 'int(rod["m"]) != 0' in source[nonzero_entries:limits_assign]
    assert "nonzero_plot_rod_entries" in source[limits_assign:hk0_limits]
    assert "min_l=float(NONZERO_QR_ROD_L_MIN)" in source[limits_assign:hk0_limits]
    assert "max_l=float(NONZERO_QR_ROD_L_MAX)" in source[limits_assign:hk0_limits]
    assert "(SPECULAR_QR_ROD_L_MIN, SPECULAR_QR_ROD_L_MAX)" in source[hk0_limits:figure_loop]
    assert "ax.set_xlim(*rod_profile_hk0_l_axis_limits)" in source[figure_loop:]
    assert "rod_profile_nonzero_y_axis_limits = shared_nonzero_rod_profile_y_axis_limits(" in source
    assert "ax.set_ylim(*rod_profile_nonzero_y_axis_limits)" in source[figure_loop:]
    assert source.count("ax.set_xlim(*rod_profile_l_axis_limits)") == 1
    assert "ax.set_xlim(2.0, x_max)" not in source


def test_parallel_script_specular_export_markers_follow_edited_qz_positions() -> None:
    namespace = _script_functions("specular_export_marker_table_from_final_markers")
    export_markers = namespace["specular_export_marker_table_from_final_markers"]
    final_markers = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "hkl": "0,0,3",
                "qz_marker": 2.0,
                "fit_l": 3.0,
                "display_l": 3.0,
                "refined_two_theta_deg": 10.0,
                "refined_phi_deg": -5.0,
                "marker_title": "L=3 moved",
            }
        ]
    )
    original_specular = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "hkl": "0,0,3",
                "qz_marker": 1.0,
                "fit_l": 3.0,
                "display_l": 3.0,
                "refined_two_theta_deg": 10.0,
                "refined_phi_deg": -5.0,
            }
        ]
    )
    qz_map = np.asarray(
        [
            [0.5, 1.0, 1.5],
            [0.7, 1.3, 2.0],
        ],
        dtype=np.float64,
    )
    region_mask = np.ones(qz_map.shape, dtype=bool)
    theta_axis = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    phi_axis = np.asarray([-5.0, 5.0], dtype=np.float64)

    exported = export_markers(
        final_markers,
        original_specular,
        qz_map=qz_map,
        region_mask=region_mask,
        theta_axis=theta_axis,
        phi_axis=phi_axis,
    )

    row = exported.iloc[0]
    assert row["qz_marker"] == pytest.approx(2.0)
    assert row["refined_two_theta_deg"] == pytest.approx(30.0)
    assert row["refined_phi_deg"] == pytest.approx(5.0)
    assert row["marker_title"] == "L=3 moved"


def test_parallel_script_specular_exports_are_rebuilt_from_final_markers_before_saving() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_cache_marker_merge = source.index(
        "marker_table = marker_table_with_specular_l_markers(marker_table, specular_l_marker_table)",
        source.index("saved final Qr-rod fit cache="),
    )
    rebuild_call = source.index(
        "specular_l_marker_table = specular_export_marker_table_from_final_markers(",
        final_cache_marker_merge,
    )
    replace_final_markers = source.index(
        'marker_table = marker_table.loc[~((marker_m == 0.0) & (marker_branch == "qz"))].copy()',
        rebuild_call,
    )
    marker_csv = source.index("marker_table.to_csv(marker_csv", replace_final_markers)
    detector_export = source.index(
        "specular_detector_qz_values = np.asarray(specular_qz_values", marker_csv
    )
    star_export = source.index("hk0_00l_region_mask =", detector_export)

    assert final_cache_marker_merge < rebuild_call < replace_final_markers < marker_csv
    assert marker_csv < detector_export < star_export


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

    conflicting_same_qz = pd.DataFrame(
        [
            {"qz_marker": 0.5, "fit_l": 1.0},
            {"qz_marker": 0.5, "fit_l": 4.0},
            {"qz_marker": 2.0, "fit_l": 2.0},
        ]
    )
    decreasing_l = pd.DataFrame(
        [
            {"qz_marker": 0.5, "fit_l": 3.0},
            {"qz_marker": 2.0, "fit_l": 1.0},
        ]
    )

    assert qz_l_coeff(conflicting_same_qz) == pytest.approx((1.0, 0.0))
    assert qz_l_coeff(decreasing_l) == pytest.approx((1.0, 0.0))


def test_parallel_script_qr_rod_snap_moves_all_panel_markers_to_local_peaks() -> None:
    namespace = _script_functions("snap_qr_rod_markers_to_profile_peaks")
    snap_markers = namespace["snap_qr_rod_markers_to_profile_peaks"]
    x = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    y = np.exp(-0.5 * ((x - 0.95) / 0.025) ** 2) + 0.7 * np.exp(-0.5 * ((x - 2.05) / 0.030) ** 2)

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
        "fit_signature": "joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane",
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


def test_parallel_script_hk0_roi_changes_final_cache_key() -> None:
    namespace = _script_functions("_cache_normalize_value", "qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]
    base_policy = {
        "specular_roi": {
            "phi_min": -5.0,
            "phi_max": 5.0,
            "two_theta_min": 8.0,
            "two_theta_max": 18.0,
        }
    }
    shifted_policy = {
        "specular_roi": {
            "phi_min": -5.0,
            "phi_max": 6.0,
            "two_theta_min": 8.0,
            "two_theta_max": 18.0,
        }
    }

    assert cache_key(None, rod_profile_policy=base_policy) != cache_key(
        None, rod_profile_policy=shifted_policy
    )


def test_parallel_script_hk0_profile_rows_change_final_cache_key() -> None:
    namespace = _script_functions("_cache_normalize_value", "qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]
    markers = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 0, "branch": "qz", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    base_profile = pd.DataFrame(
        [
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 1.0,
                "pixel_count": 8,
                "background_density": 12.0,
            },
            {
                "m": 0,
                "branch": "qz",
                "qz_center": 2.0,
                "pixel_count": 9,
                "background_density": 15.0,
            },
            {
                "m": 1,
                "branch": "+",
                "qz_center": 1.0,
                "pixel_count": 10,
                "background_density": 20.0,
            },
        ]
    )
    shifted_profile = base_profile.copy()
    shifted_profile.loc[1, "qz_center"] = 2.5

    assert cache_key(
        None,
        marker_table=markers,
        mode="popup",
        rod_profile_table=base_profile,
    ) != cache_key(
        None,
        marker_table=markers,
        mode="popup",
        rod_profile_table=shifted_profile,
    )


def test_parallel_script_qr_rod_final_cache_requires_fit_signature() -> None:
    namespace = _script_functions("qr_rod_profile_cache_has_final_fit")
    cache_has_final_fit = namespace["qr_rod_profile_cache_has_final_fit"]
    final_fit_signature = "joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane"
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
        {"mode": "last_cached", "fit_signature": final_fit_signature},
    )
    payload["final_peak_edit_cache_key"] = {
        "mode": "last_cached",
        "fit_signature": final_fit_signature,
    }
    assert cache_has_final_fit(
        payload,
        {"mode": "last_cached", "fit_signature": final_fit_signature},
    )


def test_parallel_script_qr_rod_final_cache_rejects_missing_hk0_profile() -> None:
    namespace = _script_functions("qr_rod_profile_cache_has_final_fit")
    cache_has_final_fit = namespace["qr_rod_profile_cache_has_final_fit"]
    final_fit_signature = "joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane"
    payload = {
        "final_rod_profile_table": pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "pixel_count": [8, 9],
                "joint_fit_density": [0.1, 0.2],
            }
        ),
        "final_marker_table": pd.DataFrame(
            {
                "m": [0],
                "branch": ["qz"],
                "qz_marker": [1.5],
                "fit_l": [2.0],
                "display_l": [2.0],
            }
        ),
        "final_rod_component_table": pd.DataFrame(),
        "final_peak_edit_cache_key": {
            "mode": "last_cached",
            "fit_signature": final_fit_signature,
        },
    }

    assert not cache_has_final_fit(
        payload,
        {"mode": "last_cached", "fit_signature": final_fit_signature},
    )


def test_parallel_script_pre_editor_cache_rejects_missing_hk0_profile() -> None:
    namespace = _script_functions("qr_rod_pre_editor_stage_is_valid")
    stage_is_valid = namespace["qr_rod_pre_editor_stage_is_valid"]
    stage_payload = {
        "rod_profile_table": pd.DataFrame(
            {
                "m": [1],
                "branch": ["+"],
                "qz_center": [1.0],
                "background_density": [2.0],
                "pixel_count": [5],
            }
        ),
        "marker_table": pd.DataFrame({"m": [0], "branch": ["qz"], "qz_marker": [1.0]}),
        "specular_l_marker_table": pd.DataFrame(
            {"m": [0], "branch": ["qz"], "qz_marker": [1.0], "fit_l": [2.0]}
        ),
        "region_overlays": [],
        "rod_entries": [{"m": 0, "qr": 0.0}],
        "rod_qspace_calibration": {"success": True},
        "rod_profile_max_two_theta_deg": 70.3,
    }

    assert not stage_is_valid(stage_payload)


def test_parallel_script_pre_editor_cache_key_uses_state_filename_and_inputs() -> None:
    namespace = _script_functions(
        "_cache_normalize_value",
        "pre_editor_cache_key",
    )
    cache_key = namespace["pre_editor_cache_key"]
    inputs = {
        "background_files": [r"C:\data\Bi2Se3_5m_5d.osc"],
        "fit_jobs": [{"background_index": 0, "label": "0,0,6"}],
        "fit_settings": {"fit_model": "model-a", "workers": 28},
    }

    key_a = cache_key(r"C:\one\Bi2Se3.json", input_signature=inputs)
    key_b = cache_key(r"D:\other\Bi2Se3.json", input_signature=inputs)
    changed_inputs = dict(inputs)
    changed_inputs["background_files"] = [r"C:\data\Bi2Te3_5m_5d.osc"]
    key_c = cache_key(r"C:\one\Bi2Se3.json", input_signature=changed_inputs)

    assert key_a == key_b
    assert key_a["state_name"] == "Bi2Se3.json"
    assert key_a["signature"] == "pre_qr_rod_marker_editor_inputs_v1"
    assert key_a["input_sha256"] != key_c["input_sha256"]


def test_parallel_script_pre_editor_cache_round_trips_stages(tmp_path: Path) -> None:
    namespace = _script_functions(
        "_cache_normalize_value",
        "_safe_run_name",
        "pre_editor_cache_path",
        "pre_editor_cache_key",
        "load_pre_editor_cache",
        "write_pre_editor_cache",
        "reset_pre_editor_cache",
        "pre_editor_cache_get_stage",
        "pre_editor_cache_with_stage",
        "background_peak_fit_stage_is_valid",
        "profile_fit_stage_is_valid",
        "qr_rod_pre_editor_stage_is_valid",
    )
    path_for_cache = namespace["pre_editor_cache_path"]
    cache_key = namespace["pre_editor_cache_key"]
    load_cache = namespace["load_pre_editor_cache"]
    write_cache = namespace["write_pre_editor_cache"]
    reset_cache = namespace["reset_pre_editor_cache"]
    get_stage = namespace["pre_editor_cache_get_stage"]
    with_stage = namespace["pre_editor_cache_with_stage"]

    cache_path = path_for_cache(tmp_path, r"C:\states\Bi2Se3.json")
    key = cache_key("Bi2Se3.json", input_signature={"background_files": ["a.osc"]})
    payload: dict[str, object] = {}
    payload = with_stage(
        payload,
        "background_peak_fits",
        "background_peak_fit_results_v2_radial_prefit",
        {
            "fit_job_count": 1,
            "fit_results_by_bg": {0: [{"label": "0,0,6", "params": np.array([1.0])}]},
            "fit_failures_by_bg": {0: []},
        },
    )
    payload = with_stage(
        payload,
        "profile_fits",
        "profile_fit_cache_v1",
        {"profile_target_count": 1, "profile_fit_records": [{"accepted": True}]},
    )
    payload = with_stage(
        payload,
        "qr_rod_pre_editor",
        "qr_rod_pre_marker_profiles_v1",
        {
            "rod_profile_table": pd.DataFrame({"qz_center": [1.0], "background_density": [2.0]}),
            "marker_table": pd.DataFrame({"m": [0], "branch": ["qz"], "qz_marker": [1.0]}),
            "region_overlays": [],
            "rod_entries": [{"m": 0, "qr": 0.0}],
            "rod_qspace_calibration": {"success": True},
            "rod_profile_max_two_theta_deg": 70.3,
        },
    )

    write_cache(cache_path, key, payload)
    loaded = load_cache(cache_path, key)

    assert loaded is not None
    assert namespace["background_peak_fit_stage_is_valid"](
        get_stage(loaded, "background_peak_fits", "background_peak_fit_results_v2_radial_prefit"),
        expected_fit_count=1,
    )
    assert namespace["profile_fit_stage_is_valid"](
        get_stage(loaded, "profile_fits", "profile_fit_cache_v1"),
        expected_profile_count=1,
    )
    assert namespace["qr_rod_pre_editor_stage_is_valid"](
        get_stage(loaded, "qr_rod_pre_editor", "qr_rod_pre_marker_profiles_v1")
    )
    assert (
        load_cache(
            cache_path,
            cache_key("Bi2Se3.json", input_signature={"background_files": ["b.osc"]}),
        )
        is None
    )
    assert reset_cache(cache_path)
    assert not cache_path.exists()


def test_parallel_script_qr_rod_peak_edit_runtime_mode_respects_headless() -> None:
    namespace = _script_functions("qr_rod_peak_edit_runtime_mode")
    runtime_mode = namespace["qr_rod_peak_edit_runtime_mode"]

    assert runtime_mode("auto", backend_name="agg", env={}) == "skip"
    assert runtime_mode("auto", backend_name="TkAgg", env={}) == "popup"
    assert runtime_mode("auto", backend_name="TkAgg", env={"CI": "1"}) == "skip"
    assert runtime_mode("popup", backend_name="agg", env={"CI": "1"}) == "popup"
    assert runtime_mode("skip", backend_name="TkAgg", env={}) == "skip"


def test_parallel_script_collects_manual_background_peak_entries() -> None:
    namespace = _script_functions(
        "as_float",
        "angle_key",
        "detector_xy_from_entry",
        "format_angle_value",
        "label_from_entry",
        "branch_from_phi",
        "background_peak_entries_from_manual_pairs",
    )
    collect_entries = namespace["background_peak_entries_from_manual_pairs"]
    state = {
        "geometry": {
            "manual_pairs": [
                {
                    "background_index": 0,
                    "entries": [
                        {
                            "hkl": [1, 0, 2],
                            "background_two_theta_deg": 17.5,
                            "background_phi_deg": -32.0,
                            "q_group_key": ["q_group", "primary", 1, 2],
                        },
                        {"hkl": [2, 0, 3]},
                    ],
                },
                {
                    "background_index": 9,
                    "entries": [
                        {
                            "hkl": [3, 0, 4],
                            "background_two_theta_deg": 21.0,
                            "background_phi_deg": 12.0,
                        }
                    ],
                },
            ]
        }
    }

    entries = collect_entries(
        state,
        background_files=["bg0.osc"],
        background_tilt_deg={0: -5.0},
        sample_name="Bi2Se3",
        excluded_peaks_by_tilt_normalized=set(),
    )

    assert list(entries) == [0]
    assert len(entries[0]) == 1
    entry = entries[0][0]
    assert entry["_background_index"] == 0
    assert entry["_background_name"] == "bg0.osc"
    assert entry["_display_label"] == "Bi2Se3 -5 deg"
    assert entry["_label"] == "1,0,2"
    assert entry["_branch"] == "-"
    assert entry["q_group_key"] == ["q_group", "primary", 1, 2]


def test_parallel_script_collects_caked_peak_record_entries_for_all_backgrounds() -> None:
    namespace = _script_functions(
        "as_float",
        "angle_key",
        "detector_xy_from_entry",
        "format_angle_value",
        "label_from_entry",
        "branch_from_phi",
        "background_peak_entries_from_peak_records",
    )
    collect_entries = namespace["background_peak_entries_from_peak_records"]
    state = {
        "geometry": {
            "peak_records": [
                {
                    "hkl": [1, 0, 2],
                    "two_theta_deg": 17.5,
                    "phi_deg": 32.0,
                    "q_group_key": ["q_group", "primary", 1, 2],
                    "source_table_index": 4,
                },
                {"hkl": [2, 0, 3], "two_theta_deg": float("nan"), "phi_deg": 12.0},
            ]
        }
    }

    entries = collect_entries(
        state,
        background_files=["bg0.osc", "bg1.osc"],
        background_tilt_deg={0: -5.0, 1: 0.0},
        sample_name="Bi2Se3",
        excluded_peaks_by_tilt_normalized=set(),
    )

    assert [len(entries[idx]) for idx in (0, 1)] == [1, 1]
    for bg_idx in (0, 1):
        entry = entries[bg_idx][0]
        assert entry["_background_index"] == bg_idx
        assert entry["_label"] == "1,0,2"
        assert entry["_branch"] == "+"
        assert entry["background_two_theta_deg"] == 17.5
        assert entry["background_phi_deg"] == 32.0
        assert entry["selection_reason"] == "peak_records_fallback"
        assert entry["q_group_key"] == ["q_group", "primary", 1, 2]
        assert entry["source_table_index"] == 4


def test_parallel_script_peak_record_detector_seed_requires_background_index() -> None:
    namespace = _script_functions(
        "as_float",
        "angle_key",
        "detector_xy_from_entry",
        "format_angle_value",
        "label_from_entry",
        "branch_from_phi",
        "background_peak_entries_from_peak_records",
    )
    collect_entries = namespace["background_peak_entries_from_peak_records"]
    state = {
        "geometry": {
            "peak_records": [
                {
                    "label": "2,0,3",
                    "background_index": 1,
                    "native_col": 42.0,
                    "native_row": 84.0,
                },
                {
                    "label": "3,0,4",
                    "native_col": 50.0,
                    "native_row": 90.0,
                },
            ]
        }
    }

    entries = collect_entries(
        state,
        background_files=["bg0.osc", "bg1.osc"],
        background_tilt_deg={0: -5.0, 1: 0.0},
        sample_name="Bi2Se3",
        excluded_peaks_by_tilt_normalized=set(),
    )

    assert entries[0] == []
    assert len(entries[1]) == 1
    entry = entries[1][0]
    assert entry["_label"] == "2,0,3"
    assert entry["detector_col"] == 42.0
    assert entry["detector_row"] == 84.0
    assert entry["_branch"] is None


def test_parallel_script_background_entries_prefer_manual_pairs_over_peak_records() -> None:
    namespace = _script_functions(
        "as_float",
        "angle_key",
        "detector_xy_from_entry",
        "format_angle_value",
        "label_from_entry",
        "branch_from_phi",
        "background_peak_entries_from_manual_pairs",
        "background_peak_entries_from_peak_records",
        "background_peak_entries_from_state",
    )
    collect_entries = namespace["background_peak_entries_from_state"]
    state = {
        "geometry": {
            "manual_pairs": [
                {
                    "background_index": 0,
                    "entries": [
                        {
                            "hkl": [1, 0, 2],
                            "background_two_theta_deg": 17.5,
                            "background_phi_deg": -32.0,
                        }
                    ],
                }
            ],
            "peak_records": [
                {
                    "hkl": [2, 0, 3],
                    "two_theta_deg": 22.0,
                    "phi_deg": 14.0,
                }
            ],
        }
    }

    entries, source = collect_entries(
        state,
        background_files=["bg0.osc"],
        background_tilt_deg={0: -5.0},
        sample_name="Bi2Se3",
        excluded_peaks_by_tilt_normalized=set(),
    )

    assert source == "manual_pairs"
    assert [entry["_label"] for entry in entries[0]] == ["1,0,2"]


def test_parallel_script_background_entries_fallback_filters_excluded_records() -> None:
    namespace = _script_functions(
        "as_float",
        "angle_key",
        "detector_xy_from_entry",
        "format_angle_value",
        "label_from_entry",
        "branch_from_phi",
        "background_peak_entries_from_manual_pairs",
        "background_peak_entries_from_peak_records",
        "background_peak_entries_from_state",
    )
    collect_entries = namespace["background_peak_entries_from_state"]
    angle_key = namespace["angle_key"]
    state = {
        "geometry": {
            "manual_pairs": [],
            "peak_records": [
                {
                    "hkl": [1, 0, 2],
                    "two_theta_deg": 17.5,
                    "phi_deg": 32.0,
                }
            ],
        }
    }

    entries, source = collect_entries(
        state,
        background_files=["bg0.osc", "bg1.osc"],
        background_tilt_deg={0: -5.0, 1: 0.0},
        sample_name="Bi2Se3",
        excluded_peaks_by_tilt_normalized={(angle_key(-5.0), "1,0,2")},
    )

    assert source == "peak_records_fallback"
    assert entries[0] == []
    assert [entry["_label"] for entry in entries[1]] == ["1,0,2"]


def test_parallel_script_empty_entry_message_mentions_peak_record_fallback() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PARALLEL_SCRIPT_PATH))
    message = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "NO_BACKGROUND_PEAK_ENTRIES_MESSAGE"
            for target in node.targets
        ):
            message = ast.literal_eval(node.value)
            break

    assert "geometry.manual_pairs or usable geometry.peak_records" in message
    assert "entry_source={background_peak_entry_source}" in source


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

    assert loaded[["m", "branch", "qz_marker", "fit_l", "display_l", "marker_title"]].to_dict(
        "records"
    ) == [
        {
            "m": 1,
            "branch": "+",
            "qz_marker": 1.0,
            "fit_l": 2.0,
            "display_l": 2.0,
            "marker_title": "L=2",
        }
    ]


def test_parallel_script_marker_row_title_keeps_peak_editor_labels() -> None:
    namespace = _script_functions(
        "clean_marker_title",
        "marker_row_title",
        "l_tick_label",
        "hk_display_label",
    )
    marker_label = namespace["marker_row_title"]

    assert marker_label({"display_l": 2.0}, 1) == "L=2"
    assert marker_label({"display_l": 2.49}, 1) == "L=2"
    assert marker_label({"display_l": 2.51}, 1) == "L=3"
    assert marker_label({"display_l": 2.0, "marker_title": "L=2 shoulder"}, 1) == "L=2 shoulder"
    assert marker_label({"display_l": 2.51, "marker_title": "L=2.51 custom"}, 1) == "L=2.51 custom"
    assert marker_label({"display_l": 2.0, "marker_title": "  "}, 1) == "L=2"


def test_parallel_script_final_rod_profiles_do_not_draw_peak_l_labels_or_arrows() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_profile_source = source[
        source.index(
            "rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits("
        ) : source.index('maybe_suptitle(fig, rf"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles"')
    ]

    assert "def annotate_rod_profile_hk_locations(" not in source
    assert "def rod_marker_annotation_label(" not in source
    assert "annotate_rod_profile_hk_locations(" not in final_profile_source
    assert "rod_marker_annotation_label(" not in final_profile_source
    assert '"arrowstyle": "->"' not in final_profile_source


def test_parallel_script_rod_profile_panels_use_centered_m_labels() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_profile_source = source[
        source.index(
            "rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits("
        ) : source.index('maybe_suptitle(fig, rf"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles"')
    ]

    assert 'rod_profile_panel_label(0, "qz")' in final_profile_source
    assert 'rod_profile_panel_label(int(rod["m"]), branch_name)' in final_profile_source
    assert "transform=ax.transAxes" in final_profile_source
    assert 'ha="center"' in final_profile_source
    assert 'va="top"' in final_profile_source
    assert "labelleft=col == 0" in final_profile_source
    assert "labelbottom=row == last_nonzero_plot_row" in final_profile_source
    assert 'if col == 0:\n            ax.set_ylabel("Intensity (a.u.)")' in final_profile_source
    assert "branch_label" not in final_profile_source
    assert '"- branch"' not in source
    assert '"+ branch"' not in source
    assert 'ax.set_title(branch_label if row == 0 else "")' not in final_profile_source
    assert 'ax.set_title(r"$HK = 0$")' not in final_profile_source
    assert "ax.set_ylabel(f\"$m = {int(rod['m'])}$\")" not in final_profile_source
    assert "ax.set_ylabel(f\"$HK = {int(rod['m'])}$\")" not in final_profile_source


def test_parallel_script_nonzero_rod_profile_grid_removes_inner_spacing() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_profile_source = source[
        source.index(
            "rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits("
        ) : source.index('maybe_suptitle(fig, rf"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles"')
    ]

    assert "nonzero_profile_grid = profile_grid" in final_profile_source
    assert ".subgridspec(" in final_profile_source
    assert "wspace=0.0" in final_profile_source
    assert "hspace=0.0" in final_profile_source


def test_parallel_script_qr_rod_peak_editor_is_wired_before_joint_fit_cache() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    profile_rows_index = source.index("profile_rows = []")
    marker_table_index = source.index(
        "\nmarker_table = pd.DataFrame(marker_rows)",
        source.index("if not profile_rows:"),
    )
    specular_call = source.index(
        "specular_l_marker_table = specular_l_marker_rows_for_background",
        profile_rows_index,
    )
    specular_merge = source.index(
        "marker_table = marker_table_with_specular_l_markers(",
        marker_table_index,
    )
    editor_call = source.index("edit_qr_rod_region_editor(", marker_table_index)
    cache_key_call = source.index("qr_rod_peak_edit_cache_key(", marker_table_index)
    final_fit_call = source.index("add_joint_qz_fit_columns(", marker_table_index)

    assert (
        specular_call
        < marker_table_index
        < specular_merge
        < editor_call
        < cache_key_call
        < final_fit_call
    )
    assert "RA_SIM_QR_ROD_PEAK_EDIT_MODE" in source
    assert '"QR_ROD_PEAK_EDIT_MODE_OVERRIDE", "RA_SIM_QR_ROD_PEAK_EDIT_MODE", "auto"' in source
    assert "TextBox(" in source
    assert "marker_title" in source
    assert 'event_axes = getattr(event, "inaxes", None)' in source
    assert 'event_axes is getattr(box, "ax", None)' in source


def test_parallel_script_uses_unified_qr_rod_region_editor_before_final_fit() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    marker_table_index = source.index(
        "\nmarker_table = pd.DataFrame(marker_rows)",
        source.index("if not profile_rows:"),
    )
    editor_call = source.index("edit_qr_rod_region_editor(", marker_table_index)
    cache_key_call = source.index("qr_rod_peak_edit_cache_key(", marker_table_index)
    final_fit_call = source.index("add_joint_qz_fit_columns(", marker_table_index)

    assert editor_call < cache_key_call < final_fit_call
    assert "show_qr_rod_peak_marker_popup(" in source
    assert "detector_label_entries" in source
    assert "delta_qr" in source
    assert "l_min" in source
    assert "l_max" in source


def test_parallel_script_unified_editor_restores_late_detector_label_popup() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_start = source.index("rod_label_entries: list[dict[str, object]] = []")
    save_call = source.index(
        "detector_region_png, detector_region_pdf = save_manuscript_figure",
        detector_figure_start,
    )
    detector_save_section = source[detector_figure_start:save_call]

    assert "apply_unified_qr_rod_region_editor_labels(" in detector_save_section
    assert "edit_detector_region_label_positions(" in detector_save_section
    assert "settings_path=detector_label_settings_path" in detector_save_section
    assert "backend_name=mpl.get_backend()" in detector_save_section
    assert "env=os.environ" in detector_save_section


def test_parallel_script_unified_editor_has_region_controls() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_region_editor("
        )
    ]
    wrapper_source = source[
        source.index("def edit_qr_rod_region_editor(") : source.index(
            "\ndef apply_unified_qr_rod_region_editor_labels("
        )
    ]

    for token in (
        "from matplotlib.widgets import Button, Slider, TextBox",
        "delta_qr_slider = Slider(",
        "delta_qr_ax,",
        '"Delta Qr (+/- A^-1)"',
        "l_min_box = TextBox(",
        "l_min_ax,",
        '"L Min"',
        "l_max_box = TextBox(",
        "l_max_ax,",
        '"L Max"',
        "theta_i_box = TextBox(",
        "theta_i_ax,",
        '"theta_i"',
        "set_region_theta_initial",
        "theta_initial_callback_kwargs(",
        "profile_update_callback",
        'region_control_state["rod_profile_table"]',
        "refresh_region_profile_table()",
        "fig._ra_sim_qr_rod_peak_edit_widgets",
        'fig.canvas.mpl_connect("close_event", close_linked_qr_rod_figures)',
        'companion_fig.canvas.mpl_connect("close_event", close_linked_qr_rod_figures)',
    ):
        assert token in function_source
    assert "region_state=region_control_state" in wrapper_source
    assert "profile_update_callback=profile_update_callback" in wrapper_source
    assert "theta_initial_deg" in wrapper_source


def test_parallel_script_qr_rod_peak_editor_draws_companion_figures_before_show(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    calls: list[str] = []

    class FakeCanvas:
        def draw_idle(self) -> None:
            calls.append("companion_draw")

    class FakeFigure:
        canvas = FakeCanvas()

        def show(self, *, warn: bool = True) -> None:
            calls.append(f"companion_show_warn={warn}")

    def fake_show(*_args, **_kwargs) -> None:
        calls.append("blocking_show")
        plt.close(plt.gcf())

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        companion_figures=[FakeFigure()],
    )

    assert accepted is True
    assert calls == ["companion_show_warn=False", "companion_draw", "blocking_show"]


def test_parallel_script_qr_rod_peak_editor_close_closes_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import CloseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    companion_fig = plt.figure()
    closed: list[object] = []
    original_close = plt.close

    def recording_close(target=None) -> None:
        closed.append(target)
        original_close(target)

    def fake_show(*_args, **_kwargs) -> None:
        editor_fig = plt.gcf()
        editor_fig.canvas.callbacks.process(
            "close_event",
            CloseEvent("close_event", editor_fig.canvas),
        )
        original_close(editor_fig)

    marker_table = pd.DataFrame(
        [{"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0}]
    )
    rod_profile_table = pd.DataFrame(
        [{"m": 0, "branch": "qz", "qz_center": 1.0, "background_density": 1.0}]
    )

    monkeypatch.setattr(plt, "close", recording_close)
    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        companion_figures=[companion_fig],
        editor_phase="specular",
    )

    assert accepted is True
    assert companion_fig in closed


def test_parallel_script_qr_rod_preview_close_closes_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import CloseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    companion_fig = plt.figure()
    closed: list[object] = []
    original_close = plt.close

    def recording_close(target=None) -> None:
        closed.append(target)
        original_close(target)

    def fake_show(*_args, **_kwargs) -> None:
        editor_fig = plt.gcf()
        companion_fig.canvas.callbacks.process(
            "close_event",
            CloseEvent("close_event", companion_fig.canvas),
        )
        original_close(companion_fig)
        original_close(editor_fig)

    marker_table = pd.DataFrame(
        [{"m": 0, "branch": "qz", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0}]
    )
    rod_profile_table = pd.DataFrame(
        [{"m": 0, "branch": "qz", "qz_center": 1.0, "background_density": 1.0}]
    )

    monkeypatch.setattr(plt, "close", recording_close)
    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        companion_figures=[companion_fig],
        editor_phase="specular",
    )

    assert accepted is True
    assert any(target is not None and target is not companion_fig for target in closed)


def test_parallel_script_unified_editor_region_controls_update_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        widgets = list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
        for widget in widgets:
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "Slider":
                widget.set_val(0.02)
            elif widget_type == "TextBox" and label == "L Min":
                widget.set_val("0.5")
            elif widget_type == "TextBox" and label == "L Max":
                widget.set_val("2.5")
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1, 1],
            "branch": ["+", "+", "+"],
            "qz_center": [0.5, 1.5, 2.5],
            "background_density": [1.0, 2.0, 1.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
    )

    assert accepted is True
    assert {
        "delta_qr": region_state["delta_qr"],
        "l_min": region_state["l_min"],
        "l_max": region_state["l_max"],
    } == {"delta_qr": 0.02, "l_min": 0.5, "l_max": 2.5}


def test_parallel_script_unified_editor_region_controls_update_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    preview_calls: list[tuple[float, float, float, list[float]]] = []
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def update_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
    ) -> None:
        preview_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                pd.DataFrame(marker_table)["qz_marker"].tolist(),
            )
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        widgets = list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
        for widget in widgets:
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "Slider":
                widget.set_val(0.02)
            elif widget_type == "TextBox" and label == "L Min":
                widget.set_val("0.5")
            elif widget_type == "TextBox" and label == "L Max":
                widget.set_val("2.5")
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1, 1],
            "branch": ["+", "+", "+"],
            "qz_center": [0.5, 1.5, 2.5],
            "background_density": [1.0, 2.0, 1.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=update_preview,
    )

    assert accepted is True
    assert [call[:3] for call in preview_calls] == [
        (0.02, 0.5, 3.0),
        (0.02, 0.5, 2.5),
    ]
    assert update_calls == []
    assert region_state["profile_refresh_pending"] is True
    assert preview_calls[-1][3] == [1.0, 2.0]


def test_parallel_script_nonzero_editor_theta_i_textbox_updates_profile_and_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {
        "delta_qr": 0.01,
        "l_min": 0.0,
        "l_max": 3.0,
        "theta_initial_deg": 6.0,
    }
    preview_calls: list[tuple[float, float, float, float | None]] = []
    update_calls: list[tuple[float, float, float, float | None]] = []

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        theta_initial_deg: float | None = None,
    ) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max), theta_initial_deg))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def update_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
        theta_initial_deg: float | None = None,
    ) -> None:
        preview_calls.append((float(delta_qr), float(l_min), float(l_max), theta_initial_deg))

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "theta_i" in boxes
        boxes["theta_i"].set_val("11.25")
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=update_preview,
        editor_phase="nonzero",
    )

    assert accepted is True
    assert region_state["theta_initial_deg"] == pytest.approx(11.25)
    assert update_calls == [(0.01, 0.0, 3.0, 11.25)]
    assert preview_calls
    assert preview_calls[-1] == (0.01, 0.0, 3.0, 11.25)


def test_parallel_script_unified_editor_l_bounds_reject_invalid_profile_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [np.nan],
                "branch": ["+"],
                "qz_center": [1.5],
                "background_density": [10.0],
            }
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "TextBox" and label == "L Min":
                widget.set_val("0.5")
            elif widget_type == "TextBox" and label == "L Max":
                widget.set_val("2.5")
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
    )

    assert accepted is True
    assert update_calls == [(0.01, 0.5, 2.5)]
    assert "profile_update_error" in region_state
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["m"].tolist() == [1, 1]
    assert refreshed["background_density"].tolist() == [1.0, 2.0]


def test_parallel_script_unified_editor_roi_refresh_keeps_specular_rows_when_refresh_lacks_hk0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 1.5, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [0.5, 3.0],
                "background_density": [10.0, 20.0],
            }
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "L Min" not in boxes
        assert "L Max" not in boxes
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "TextBox" and label == "2theta Min":
                widget.set_val("8")
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
            {"m": 1, "branch": "+", "qz_marker": 0.5, "fit_l": 0.5, "display_l": 0.5},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 4.0},
            {"m": 0, "branch": "qz", "qz_center": 2.2, "background_density": 8.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        editor_phase="specular",
    )

    assert accepted is True
    assert update_calls == [(0.01, 1.5, 3.0), (0.01, 1.5, 3.0), (0.01, 1.5, 3.0)]
    assert "profile_update_error" in region_state
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["m"].astype(int).tolist() == [0, 0, 0]
    assert refreshed["background_density"].tolist() == [4.0, 8.0, 5.0]


def test_parallel_script_hk0_two_theta_submit_refreshes_without_l_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {
        "delta_qr": 0.01,
        "l_min": 1.5,
        "l_max": 3.0,
        "two_theta_min": 7.0,
        "two_theta_max": 18.0,
    }
    update_calls: list[tuple[float, float, float, float, float]] = []
    preview_calls: list[tuple[float, float, float, float, float]] = []
    observed: dict[str, object] = {}

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        two_theta_min: float,
        two_theta_max: float,
    ) -> pd.DataFrame:
        update_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(two_theta_min),
                float(two_theta_max),
            )
        )
        return pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_center": 1.8, "background_density": 6.0},
                {"m": 0, "branch": "qz", "qz_center": 2.4, "background_density": 9.0},
                {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
            ]
        )

    def update_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        two_theta_min: float,
        two_theta_max: float,
        marker_table: pd.DataFrame,
    ) -> None:
        preview_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(two_theta_min),
                float(two_theta_max),
            )
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        hk0 = panels["HK=0 qz"]
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "L Min" not in boxes
        assert "L Max" not in boxes
        boxes["2theta Min"].set_val("9")
        observed["after_submit_xlim"] = hk0.get_xlim()
        observed["update_calls_after_submit"] = list(update_calls)
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 4.0},
            {"m": 0, "branch": "qz", "qz_center": 2.2, "background_density": 8.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=update_preview,
        editor_phase="specular",
    )

    assert accepted is True
    assert region_state["l_min"] == pytest.approx(1.5)
    assert region_state["two_theta_min"] == pytest.approx(9.0)
    assert preview_calls == [(0.01, 1.5, 3.0, 7.0, 18.0), (0.01, 1.5, 3.0, 9.0, 18.0)]
    submitted_xmin, submitted_xmax = observed["after_submit_xlim"]
    assert 1.5 < submitted_xmin < 1.8
    assert 3.0 <= submitted_xmax < 3.2
    assert observed["update_calls_after_submit"] == [
        (0.01, 1.5, 3.0, 7.0, 18.0),
        (0.01, 1.5, 3.0, 9.0, 18.0),
    ]
    assert update_calls == [
        (0.01, 1.5, 3.0, 7.0, 18.0),
        (0.01, 1.5, 3.0, 9.0, 18.0),
    ]
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["qz_center"].tolist() == pytest.approx([1.8, 2.4, 3.0])


def test_parallel_script_unified_editor_l_bounds_keep_popup_on_redraw_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()

        def fail_draw_idle() -> None:
            raise RuntimeError("draw failed")

        fig.canvas.draw_idle = fail_draw_idle
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "TextBox" and label == "L Min":
                widget.set_val("0.5")
            elif widget_type == "TextBox" and label == "L Max":
                widget.set_val("2.5")
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
    )

    assert accepted is True
    assert update_calls == []
    assert region_state["redraw_error"] == "draw failed"


def test_parallel_script_unified_editor_l_bound_enter_does_not_accept_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []
    close_calls_during_l_enter: list[str] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    real_close = plt.close
    in_l_enter = {"label": ""}

    def record_close(*args, **kwargs) -> None:
        if in_l_enter["label"]:
            close_calls_during_l_enter.append(in_l_enter["label"])
        real_close(*args, **kwargs)

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        for label, value in (("L Min", "0.5"), ("L Max", "2.5")):
            box = boxes[label]
            box.set_val(value)
            box.begin_typing()
            event = KeyEvent("key_press_event", fig.canvas, key="enter")
            event.inaxes = box.ax
            in_l_enter["label"] = label
            try:
                fig.canvas.callbacks.process("key_press_event", event)
            finally:
                in_l_enter["label"] = ""
                box.stop_typing()
        real_close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "close", record_close)
    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
    )

    assert accepted is True
    assert close_calls_during_l_enter == []
    assert update_calls == []
    assert region_state["profile_refresh_pending"] is True
    assert region_state["l_min"] == pytest.approx(0.5)
    assert region_state["l_max"] == pytest.approx(2.5)


def test_parallel_script_unified_editor_delta_qr_refreshes_profile_table_on_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []
    preview_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
    ) -> None:
        preview_calls.append((float(delta_qr), float(l_min), float(l_max)))

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            if type(widget).__name__ == "Slider":
                widget.set_val(0.02)
                widget.set_val(0.03)
        assert update_calls == []
        assert preview_calls == []
        fig.canvas.callbacks.process(
            "button_release_event",
            MouseEvent("button_release_event", fig.canvas, 0, 0, button=1),
        )
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
    )

    assert accepted is True
    assert preview_calls == [(0.03, 0.0, 3.0)]
    assert update_calls == [(0.03, 0.0, 3.0)]
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["background_density"].tolist() == [10.0, 20.0]


def test_parallel_script_qr_rod_peak_editor_preview_error_does_not_close_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent, MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
    ) -> None:
        raise RuntimeError("preview failed")

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            if type(widget).__name__ == "Slider":
                widget.set_val(0.02)
        assert "region_preview_update_error" not in region_state
        fig.canvas.callbacks.process(
            "button_release_event",
            MouseEvent("button_release_event", fig.canvas, 0, 0, button=1),
        )
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
    )

    assert accepted is True
    assert update_calls == [(0.02, 0.0, 3.0)]
    assert region_state["region_preview_update_error"] == "preview failed"


def test_parallel_script_hk0_roi_editor_has_no_delta_qr_slider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 1.5, "l_max": 3.0}
    update_calls: list[tuple[float, float, float, str]] = []
    preview_calls: list[tuple[float, float, float, str]] = []

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        editor_phase: object,
    ) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max), str(editor_phase)))
        assert str(editor_phase) == "specular"
        return pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_center": 1.6, "background_density": 30.0},
                {"m": 0, "branch": "qz", "qz_center": 2.5, "background_density": 40.0},
            ]
        )

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
        editor_phase: object,
    ) -> None:
        preview_calls.append((float(delta_qr), float(l_min), float(l_max), str(editor_phase)))

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        assert not any(
            type(widget).__name__ == "Slider"
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
        )
        assert update_calls == [(0.01, 1.5, 3.0, "specular")]
        assert preview_calls == [(0.01, 1.5, 3.0, "specular")]
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 3.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 4.0},
            {"m": 1, "branch": "+", "qz_center": 1.0, "background_density": 10.0},
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 20.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        editor_phase="specular",
    )

    assert accepted is True
    assert preview_calls == [(0.01, 1.5, 3.0, "specular")]
    assert update_calls == [(0.01, 1.5, 3.0, "specular")]
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["m"].astype(int).tolist() == [1, 1, 0, 0]
    assert refreshed["background_density"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_parallel_script_hk0_roi_textboxes_update_specular_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {
        "delta_qr": 0.01,
        "l_min": 1.5,
        "l_max": 3.0,
        "phi_min": -5.0,
        "phi_max": 5.0,
        "two_theta_min": 8.0,
        "two_theta_max": 18.0,
    }
    update_calls: list[tuple[float, float, float, float, float, float, float, str]] = []
    preview_calls: list[tuple[float, float, float, float, float, float, float, str]] = []

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        phi_min: float,
        phi_max: float,
        two_theta_min: float,
        two_theta_max: float,
        editor_phase: object,
    ) -> pd.DataFrame:
        update_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(phi_min),
                float(phi_max),
                float(two_theta_min),
                float(two_theta_max),
                str(editor_phase),
            )
        )
        return pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_center": 1.6, "background_density": 30.0},
                {"m": 0, "branch": "qz", "qz_center": 2.5, "background_density": 40.0},
            ]
        )

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        phi_min: float,
        phi_max: float,
        two_theta_min: float,
        two_theta_max: float,
        marker_table: pd.DataFrame,
        editor_phase: object,
    ) -> None:
        preview_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(phi_min),
                float(phi_max),
                float(two_theta_min),
                float(two_theta_max),
                str(editor_phase),
            )
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert not any(
            type(widget).__name__ == "Slider"
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
        )
        assert "Theta i" not in boxes
        for label in ("L Min", "L Max"):
            assert label not in boxes
        for label in ("Phi Min", "Phi Max", "2theta Min", "2theta Max"):
            assert label in boxes
        boxes["Phi Min"].set_val("-8")
        boxes["Phi Max"].set_val("9")
        boxes["2theta Min"].set_val("7")
        boxes["2theta Max"].set_val("19")
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 3.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 4.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        editor_phase="specular",
    )

    assert accepted is True
    assert region_state["phi_min"] == pytest.approx(-8.0)
    assert region_state["phi_max"] == pytest.approx(9.0)
    assert region_state["two_theta_min"] == pytest.approx(7.0)
    assert region_state["two_theta_max"] == pytest.approx(19.0)
    assert update_calls == [
        (0.01, 1.5, 3.0, -5.0, 5.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 5.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 7.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 7.0, 19.0, "specular"),
    ]
    assert preview_calls == [
        (0.01, 1.5, 3.0, -5.0, 5.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 5.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 8.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 7.0, 18.0, "specular"),
        (0.01, 1.5, 3.0, -8.0, 9.0, 7.0, 19.0, "specular"),
    ]


def test_parallel_script_nonzero_picker_does_not_show_hk0_roi_textboxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed_labels: list[str] = []

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        observed_labels.extend(
            getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        )
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 1.0, "background_density": 10.0},
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 20.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={
            "delta_qr": 0.01,
            "l_min": 0.0,
            "l_max": 3.0,
            "phi_min": -5.0,
            "phi_max": 5.0,
            "two_theta_min": 8.0,
            "two_theta_max": 18.0,
        },
        editor_phase="nonzero",
    )

    assert accepted is True
    assert "Theta i" not in observed_labels
    for label in ("Phi Min", "Phi Max", "2theta Min", "2theta Max"):
        assert label not in observed_labels


def test_parallel_script_qr_rod_peak_editor_passes_marker_table_to_profile_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    received_marker_tables: list[pd.DataFrame] = []
    received_l_bounds: list[tuple[float, float]] = []
    received_theta_mins: list[float] = []

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        marker_table: pd.DataFrame,
        two_theta_min: float,
        editor_phase: object,
    ) -> pd.DataFrame:
        received_marker_tables.append(pd.DataFrame(marker_table).copy())
        received_l_bounds.append((float(l_min), float(l_max)))
        received_theta_mins.append(float(two_theta_min))
        assert float(delta_qr) == pytest.approx(0.01)
        assert str(editor_phase) == "specular"
        return pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_center": 1.8, "background_density": 6.0},
                {"m": 0, "branch": "qz", "qz_center": 2.4, "background_density": 9.0},
            ]
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "L Min" not in boxes
        assert "L Max" not in boxes
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            widget_type = type(widget).__name__
            label = getattr(getattr(widget, "label", None), "get_text", lambda: "")()
            if widget_type == "TextBox" and label == "2theta Min":
                widget.set_val("8")
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 4.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 1.5, "l_max": 3.0},
        profile_update_callback=refresh_profiles,
        editor_phase="specular",
    )

    assert accepted is True
    assert received_l_bounds == [(1.5, 3.0), (1.5, 3.0)]
    assert received_theta_mins == pytest.approx([5.0, 8.0])
    assert len(received_marker_tables) == 2
    assert all(table is not marker_table for table in received_marker_tables)
    assert received_marker_tables[-1]["qz_marker"].tolist() == pytest.approx([1.5, 3.0])


def test_parallel_script_qr_rod_peak_editor_import_refreshes_nonzero_preview_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent
    import tkinter.filedialog as filedialog

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    import_path = tmp_path / "nonzero_peak_edits.json"
    imported_rows = [
        {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
        {"m": 1, "branch": "-", "qz_marker": 1.1, "fit_l": 1.1, "display_l": 1.1},
        {"m": 3, "branch": "+", "qz_marker": 1.2, "fit_l": 1.2, "display_l": 1.2},
        {"m": 3, "branch": "-", "qz_marker": 1.3, "fit_l": 1.3, "display_l": 1.3},
    ]
    import_path.write_text(json.dumps(imported_rows), encoding="utf-8")
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **_kwargs: str(import_path))
    preview_calls: list[pd.DataFrame] = []
    update_calls: list[pd.DataFrame] = []

    def refresh_preview(**kwargs: object) -> None:
        assert kwargs["editor_phase"] == "nonzero"
        preview_calls.append(pd.DataFrame(kwargs["marker_table"]).copy())

    def refresh_profiles(
        _delta_qr: float,
        _l_min: float,
        _l_max: float,
        *,
        marker_table: pd.DataFrame,
        editor_phase: object,
    ) -> pd.DataFrame:
        assert str(editor_phase) == "nonzero"
        update_calls.append(pd.DataFrame(marker_table).copy())
        return pd.DataFrame(
            [
                {
                    "m": row["m"],
                    "branch": row["branch"],
                    "qz_center": row["qz_marker"],
                    "background_density": 10.0,
                }
                for row in imported_rows
            ]
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        import_button = next(
            widget
            for widget in getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")
            if type(widget).__name__ == "Button"
            and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "Import"
        )
        import_button._observers.process("clicked", None)
        assert len(preview_calls) == 1
        assert update_calls == []
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 0.9, "fit_l": 0.9, "display_l": 0.9},
            {"m": 1, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 3, "branch": "+", "qz_marker": 1.1, "fit_l": 1.1, "display_l": 1.1},
            {"m": 3, "branch": "-", "qz_marker": 1.2, "fit_l": 1.2, "display_l": 1.2},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {
                "m": row["m"],
                "branch": row["branch"],
                "qz_center": row["qz_marker"],
                "background_density": 1.0,
            }
            for row in marker_table.to_dict("records")
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0},
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        editor_phase="nonzero",
    )

    assert accepted is True
    assert len(preview_calls) == 1
    assert len(update_calls) == 1
    assert update_calls[0]["qz_marker"].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3])


def test_parallel_script_qr_rod_peak_editor_nonzero_phase_keeps_imported_hk0_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.2, "fit_l": 1.2, "display_l": 1.2},
            {"m": 0, "branch": "qz", "qz_marker": 1.8, "fit_l": 1.8, "display_l": 1.8},
        ]
    )
    required_marker_table = pd.DataFrame(
        [{"m": 0, "branch": "qz", "qz_marker": 9.9, "fit_l": 9.9, "display_l": 9.9}]
    )
    rod_profile_table = pd.DataFrame(
        [{"m": 1, "branch": "+", "qz_center": 1.2, "background_density": 1.0}]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0},
        required_marker_table=required_marker_table,
        editor_phase="nonzero",
    )

    assert accepted is True
    hk0_markers = edited.loc[
        (edited["m"].astype(int) == 0) & (edited["branch"].astype(str) == "qz"), "qz_marker"
    ].tolist()
    assert hk0_markers == pytest.approx([1.8])


def test_parallel_script_qr_rod_peak_editor_import_applies_nonzero_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent
    import tkinter.filedialog as filedialog

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    import_path = tmp_path / "nonzero_peak_edits_with_parameters.json"
    imported_rows = [
        {"m": 1, "branch": "+", "qz_marker": 1.2, "fit_l": 1.2, "display_l": 1.2},
        {"m": 1, "branch": "-", "qz_marker": 1.4, "fit_l": 1.4, "display_l": 1.4},
    ]
    import_path.write_text(
        json.dumps(
            {
                "schema": "ra_sim.qr_rod_peak_edits.v1",
                "markers": imported_rows,
                "parameters": {
                    "nonzero": {
                        "delta_qr": 0.04,
                        "l_min": 0.6,
                        "l_max": 4.2,
                        "theta_initial_deg": 12.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **_kwargs: str(import_path))
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0, "theta_initial_deg": 6.0}
    preview_calls: list[tuple[float, float, float, float, pd.DataFrame]] = []
    update_calls: list[tuple[float, float, float, float, pd.DataFrame]] = []

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
        theta_initial_deg: float,
        editor_phase: object,
    ) -> None:
        assert str(editor_phase) == "nonzero"
        preview_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(theta_initial_deg),
                pd.DataFrame(marker_table).copy(),
            )
        )

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        marker_table: pd.DataFrame,
        theta_initial_deg: float,
        editor_phase: object,
    ) -> pd.DataFrame:
        assert str(editor_phase) == "nonzero"
        update_calls.append(
            (
                float(delta_qr),
                float(l_min),
                float(l_max),
                float(theta_initial_deg),
                pd.DataFrame(marker_table).copy(),
            )
        )
        return pd.DataFrame(
            [
                {
                    "m": row["m"],
                    "branch": row["branch"],
                    "qz_center": row["qz_marker"],
                    "background_density": 10.0,
                }
                for row in imported_rows
            ]
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        import_button = next(
            widget
            for widget in getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")
            if type(widget).__name__ == "Button"
            and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "Import"
        )
        update_count_before_import = len(update_calls)
        import_button._observers.process("clicked", None)
        assert len(update_calls) == update_count_before_import
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 0.9, "fit_l": 0.9, "display_l": 0.9},
            {"m": 1, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 0.9, "background_density": 1.0},
            {"m": 1, "branch": "-", "qz_center": 1.0, "background_density": 1.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        editor_phase="nonzero",
    )

    assert accepted is True
    assert {
        "delta_qr": region_state["delta_qr"],
        "l_min": region_state["l_min"],
        "l_max": region_state["l_max"],
        "theta_initial_deg": region_state["theta_initial_deg"],
    } == pytest.approx({"delta_qr": 0.04, "l_min": 0.6, "l_max": 4.2, "theta_initial_deg": 12.5})
    assert [(call[0], call[1], call[2], call[3]) for call in preview_calls] == pytest.approx(
        [(0.04, 0.6, 4.2, 12.5)]
    )
    assert [(call[0], call[1], call[2], call[3]) for call in update_calls] == pytest.approx(
        [(0.04, 0.6, 4.2, 12.5)]
    )
    assert update_calls[0][4]["qz_marker"].tolist() == pytest.approx([1.2, 1.4])


def test_parallel_script_qr_rod_peak_editor_import_applies_hk0_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent
    import tkinter.filedialog as filedialog

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    import_path = tmp_path / "hk0_peak_edits_with_parameters.json"
    imported_rows = [
        {"m": 0, "branch": "qz", "qz_marker": 1.8, "fit_l": 1.8, "display_l": 1.8},
        {"m": 0, "branch": "qz", "qz_marker": 2.4, "fit_l": 2.4, "display_l": 2.4},
    ]
    import_path.write_text(
        json.dumps(
            {
                "schema": "ra_sim.qr_rod_peak_edits.v1",
                "markers": imported_rows,
                "parameters": {
                    "specular": {
                        "phi_min": -4.5,
                        "phi_max": 6.5,
                        "two_theta_min": 8.5,
                        "two_theta_max": 18.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(filedialog, "askopenfilename", lambda **_kwargs: str(import_path))
    region_state = {
        "delta_qr": 0.01,
        "l_min": 1.0,
        "l_max": 3.0,
        "phi_min": -2.0,
        "phi_max": 2.0,
        "two_theta_min": 7.0,
        "two_theta_max": 12.0,
    }
    preview_calls: list[tuple[float, float, float, float, pd.DataFrame]] = []
    update_calls: list[tuple[float, float, float, float, pd.DataFrame]] = []

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
        phi_min: float,
        phi_max: float,
        two_theta_min: float,
        two_theta_max: float,
        editor_phase: object,
    ) -> None:
        assert str(editor_phase) == "specular"
        del delta_qr, l_min, l_max
        preview_calls.append(
            (
                float(phi_min),
                float(phi_max),
                float(two_theta_min),
                float(two_theta_max),
                pd.DataFrame(marker_table).copy(),
            )
        )

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        marker_table: pd.DataFrame,
        phi_min: float,
        phi_max: float,
        two_theta_min: float,
        two_theta_max: float,
        editor_phase: object,
    ) -> pd.DataFrame:
        assert str(editor_phase) == "specular"
        del delta_qr, l_min, l_max
        update_calls.append(
            (
                float(phi_min),
                float(phi_max),
                float(two_theta_min),
                float(two_theta_max),
                pd.DataFrame(marker_table).copy(),
            )
        )
        return pd.DataFrame(
            [
                {
                    "m": row["m"],
                    "branch": row["branch"],
                    "qz_center": row["qz_marker"],
                    "background_density": 10.0,
                }
                for row in imported_rows
            ]
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "L Min" not in boxes
        assert "L Max" not in boxes
        import_button = next(
            widget
            for widget in getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")
            if type(widget).__name__ == "Button"
            and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "Import"
        )
        update_count_before_import = len(update_calls)
        import_button._observers.process("clicked", None)
        assert len(update_calls) == update_count_before_import
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 1.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 1.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        required_marker_table=marker_table,
        editor_phase="specular",
    )

    assert accepted is True
    assert {
        "phi_min": region_state["phi_min"],
        "phi_max": region_state["phi_max"],
        "two_theta_min": region_state["two_theta_min"],
        "two_theta_max": region_state["two_theta_max"],
    } == pytest.approx(
        {"phi_min": -4.5, "phi_max": 6.5, "two_theta_min": 8.5, "two_theta_max": 18.5}
    )
    assert (
        preview_calls[-1][0],
        preview_calls[-1][1],
        preview_calls[-1][2],
        preview_calls[-1][3],
    ) == pytest.approx((-4.5, 6.5, 8.5, 18.5))
    assert (
        update_calls[-1][0],
        update_calls[-1][1],
        update_calls[-1][2],
        update_calls[-1][3],
    ) == pytest.approx((-4.5, 6.5, 8.5, 18.5))
    assert update_calls[-1][4]["qz_marker"].tolist() == pytest.approx([1.8, 2.4])


def test_parallel_script_qr_rod_peak_editor_export_writes_phase_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent
    import tkinter.filedialog as filedialog

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]

    def export_phase(
        *,
        marker_table: pd.DataFrame,
        rod_profile_table: pd.DataFrame,
        region_state: dict[str, object],
        editor_phase: str,
        export_name: str,
    ) -> dict[str, object]:
        export_path = tmp_path / export_name
        monkeypatch.setattr(filedialog, "asksaveasfilename", lambda **_kwargs: str(export_path))

        def fake_show(*_args, **_kwargs) -> None:
            fig = plt.gcf()
            export_button = next(
                widget
                for widget in getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")
                if type(widget).__name__ == "Button"
                and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "Export"
            )
            export_button._observers.process("clicked", None)
            fig.canvas.callbacks.process(
                "key_press_event",
                KeyEvent("key_press_event", fig.canvas, key="enter"),
            )

        monkeypatch.setattr(plt, "show", fake_show)
        _edited, accepted = show_editor(
            marker_table,
            rod_profile_table,
            backend_name="TkAgg",
            region_state=region_state,
            editor_phase=editor_phase,
        )

        assert accepted is True
        return json.loads(export_path.read_text(encoding="utf-8"))

    nonzero_payload = export_phase(
        marker_table=pd.DataFrame(
            [
                {"m": 1, "branch": "+", "qz_marker": 1.2, "fit_l": 1.2, "display_l": 1.2},
                {"m": 1, "branch": "-", "qz_marker": 1.4, "fit_l": 1.4, "display_l": 1.4},
            ]
        ),
        rod_profile_table=pd.DataFrame(
            [
                {"m": 1, "branch": "+", "qz_center": 1.2, "background_density": 1.0},
                {"m": 1, "branch": "-", "qz_center": 1.4, "background_density": 1.0},
            ]
        ),
        region_state={"delta_qr": 0.04, "l_min": 0.6, "l_max": 4.2, "theta_initial_deg": 12.5},
        editor_phase="nonzero",
        export_name="nonzero_export.json",
    )
    assert nonzero_payload["parameters"]["nonzero"] == pytest.approx(
        {"delta_qr": 0.04, "l_min": 0.6, "l_max": 4.2, "theta_initial_deg": 12.5}
    )
    assert "specular" not in nonzero_payload["parameters"]

    hk0_payload = export_phase(
        marker_table=pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_marker": 1.8, "fit_l": 1.8, "display_l": 1.8},
                {"m": 0, "branch": "qz", "qz_marker": 2.4, "fit_l": 2.4, "display_l": 2.4},
            ]
        ),
        rod_profile_table=pd.DataFrame(
            [
                {"m": 0, "branch": "qz", "qz_center": 1.8, "background_density": 1.0},
                {"m": 0, "branch": "qz", "qz_center": 2.4, "background_density": 1.0},
            ]
        ),
        region_state={
            "delta_qr": 0.01,
            "l_min": 1.0,
            "l_max": 3.0,
            "phi_min": -4.5,
            "phi_max": 6.5,
            "two_theta_min": 8.5,
            "two_theta_max": 18.5,
        },
        editor_phase="specular",
        export_name="hk0_export.json",
    )
    assert hk0_payload["parameters"]["specular"] == pytest.approx(
        {"phi_min": -4.5, "phi_max": 6.5, "two_theta_min": 8.5, "two_theta_max": 18.5}
    )
    assert "nonzero" not in hk0_payload["parameters"]


def test_parallel_script_qr_rod_peak_editor_marks_hk0_marker_changes_dirty() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_region_editor("
        )
    ]

    assert "def mark_marker_table_changed(" in function_source
    assert "refresh_preview: bool = True" in function_source
    for token in (
        "mark_marker_table_changed(group)",
        "mark_marker_table_changed((int(m_value), str(branch_value)))",
        "mark_marker_table_changed(imported_group, refresh_preview=False)",
        "if needs_preview_refresh:\n                refresh_detector_region_preview()",
    ):
        assert token in function_source


def test_parallel_script_qr_rod_peak_editor_freezes_panel_l_axis_mapping() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_region_editor("
        )
    ]
    import_section = function_source[
        function_source.index("def import_peak_edits(") : function_source.index(
            "\n    def export_peak_edits("
        )
    ]

    assert "group_l_coeff_cache: dict[tuple[int, str], tuple[float, float]] = {}" in (
        function_source
    )
    assert "def clear_group_l_coeff_cache(" in function_source
    assert "marker_source=edited.copy()" in function_source
    assert 'for column in ("fit_l", "display_l", "l"):' in function_source
    assert "clear_group_l_coeff_cache()" in import_section


def test_parallel_script_qr_rod_peak_marker_move_delete_defers_callbacks_until_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent, MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[pd.DataFrame] = []
    preview_calls: list[pd.DataFrame] = []
    observed: dict[str, int] = {}

    def refresh_profiles(
        delta_qr: float,
        l_min: float,
        l_max: float,
        *,
        marker_table: pd.DataFrame,
    ) -> pd.DataFrame:
        table = pd.DataFrame(marker_table).copy()
        update_calls.append(table)
        return pd.DataFrame(
            [
                {
                    "m": row["m"],
                    "branch": row["branch"],
                    "qz_center": row["qz_marker"],
                    "background_density": 1.0,
                }
                for row in table.to_dict("records")
            ]
        )

    def refresh_preview(
        *,
        delta_qr: float,
        l_min: float,
        l_max: float,
        marker_table: pd.DataFrame,
    ) -> None:
        preview_calls.append(pd.DataFrame(marker_table).copy())

    def emit_event(fig, ax, name: str, x_value: float, y_value: float) -> None:
        x_pixel, y_pixel = ax.transData.transform((float(x_value), float(y_value)))
        event = MouseEvent(name, fig.canvas, x_pixel, y_pixel, button=1)
        event.inaxes = ax
        fig.canvas.callbacks.process(name, event)

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        target = panels["HK=1 +"]
        fig.canvas.draw()
        emit_event(fig, target, "button_press_event", 1.0, 10.0)
        emit_event(fig, target, "motion_notify_event", 1.2, 10.0)
        emit_event(fig, target, "button_release_event", 1.2, 10.0)
        observed["updates_after_move"] = len(update_calls)
        observed["previews_after_move"] = len(preview_calls)
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="delete"),
        )
        observed["updates_after_delete"] = len(update_calls)
        observed["previews_after_delete"] = len(preview_calls)
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [10.0, 20.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
        region_preview_update_callback=refresh_preview,
        editor_phase="nonzero",
    )

    assert accepted is True
    assert observed == {
        "updates_after_move": 0,
        "previews_after_move": 0,
        "updates_after_delete": 0,
        "previews_after_delete": 0,
    }
    assert len(update_calls) == 1
    assert len(preview_calls) == 1
    assert update_calls[0]["qz_marker"].tolist() == pytest.approx([2.0])
    assert preview_calls[0]["qz_marker"].tolist() == pytest.approx([2.0])


def test_parallel_script_unified_editor_accept_flushes_pending_profile_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import KeyEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.0, "l_max": 3.0}
    update_calls: list[tuple[float, float, float]] = []

    def refresh_profiles(delta_qr: float, l_min: float, l_max: float) -> pd.DataFrame:
        update_calls.append((float(delta_qr), float(l_min), float(l_max)))
        return pd.DataFrame(
            {
                "m": [1, 1],
                "branch": ["+", "+"],
                "qz_center": [1.0, 2.0],
                "background_density": [10.0, 20.0],
            }
        )

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")):
            if type(widget).__name__ == "Slider":
                widget.set_val(0.04)
        assert update_calls == []
        fig.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", fig.canvas, key="enter"),
        )

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        {
            "m": [1, 1],
            "branch": ["+", "+"],
            "qz_center": [1.0, 2.0],
            "background_density": [1.0, 2.0],
        }
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        profile_update_callback=refresh_profiles,
    )

    assert accepted is True
    assert update_calls == [(0.04, 0.0, 3.0)]
    refreshed = pd.DataFrame(region_state["rod_profile_table"])
    assert refreshed["background_density"].tolist() == [10.0, 20.0]


def test_parallel_script_unified_editor_result_updates_final_profile_table() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    nonzero_call = source.index("qr_rod_nonzero_editor_result = edit_qr_rod_region_editor(")
    specular_call = source.index("qr_rod_specular_editor_result = edit_qr_rod_region_editor(")
    combined_result = source.index("qr_rod_region_editor_result = {", specular_call)
    cache_key_call = source.index("qr_rod_peak_edit_cache_key(", combined_result)
    final_fit_call = source.index("add_joint_qz_fit_columns(", combined_result)
    section = source[nonzero_call:final_fit_call]

    assert "def recompute_qr_rod_region_profiles(" in source
    assert "profile_update_callback=recompute_qr_rod_region_profiles" in section
    assert 'qr_rod_nonzero_editor_result.get("rod_profile_table", rod_profile_table)' in section
    assert 'qr_rod_specular_editor_result.get("rod_profile_table", rod_profile_table)' in section
    assert "merge_qr_rod_editor_phase_table(" in section
    assert "rod_profile_table_for_l_window(" in section
    assert "rod_profile_table=rod_profile_table" in section
    assert "specular_l_min=qr_rod_specular_editor_l_min" not in section
    assert '"theta_initial_deg": float(qr_rod_editor_theta_initial_deg)' in section
    assert "editor_theta_initial_deg" in section
    assert "specular_editor_l_min" not in section
    assert "specular_editor_l_max" not in section
    assert (
        "qr_map, qz_map, valid_q_map = detector_q_maps_for_theta_initial("
        "qr_rod_editor_theta_initial_deg)"
    ) in source
    assert "delta_qr_override=delta_qr_value" in source
    assert nonzero_call < specular_call < combined_result < cache_key_call < final_fit_call


def test_parallel_script_specular_phase_reuses_pre_editor_hk0_profiles() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    nonzero_result = source.index(
        'rod_profile_table = pd.DataFrame(\n    qr_rod_nonzero_editor_result.get("rod_profile_table", rod_profile_table)'
    )
    specular_seed = source.index("qr_rod_specular_editor_profile_table =", nonzero_result)
    specular_call = source.index("qr_rod_specular_editor_result = edit_qr_rod_region_editor(")
    specular_section = source[
        specular_call : source.index(")\nmarker_table = pd.DataFrame", specular_call)
    ]

    assert nonzero_result < specular_seed < specular_call
    assert (
        'qr_rod_editor_base_profiles,\n    editor_phase="specular"'
        in source[specular_seed:specular_call]
    )
    assert "qr_rod_specular_editor_profile_table" in specular_section


def test_parallel_script_qr_rod_marker_editor_runs_nonzero_then_specular_phases() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    nonzero_call = source.index("qr_rod_nonzero_editor_result = edit_qr_rod_region_editor(")
    specular_call = source.index("qr_rod_specular_editor_result = edit_qr_rod_region_editor(")
    label_editor_call = source.index("rod_label_entries = edit_detector_region_label_positions(")
    nonzero_section = source[
        nonzero_call : source.index(")\nmarker_table = pd.DataFrame", nonzero_call)
    ]
    specular_section = source[
        specular_call : source.index(")\nmarker_table = pd.DataFrame", specular_call)
    ]

    assert nonzero_call < specular_call < label_editor_call
    assert 'editor_phase="nonzero"' in nonzero_section
    assert 'editor_phase="specular"' in specular_section
    assert "edit_path=None" in nonzero_section
    assert "l_min=qr_rod_editor_initial_l_min" in nonzero_section
    assert "l_min=qr_rod_specular_editor_initial_l_min" in specular_section
    assert "l_max=qr_rod_specular_editor_initial_l_max" in specular_section
    assert "edit_path=None" in specular_section
    assert "edit_path=qr_rod_peak_edits_path" not in source
    assert 'if bool(qr_rod_specular_editor_result.get("accepted", False))' in source
    assert "saved final Qr-rod peak edits" in source


def test_parallel_script_qr_rod_peak_editor_titles_include_phase() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_region_editor("
        )
    ]

    assert "qr_rod_editor_phase_title(editor_phase)" in function_source
    assert "num=window_title" in function_source
    assert "fig.canvas.manager.set_window_title(window_title)" in function_source
    assert "click Accept to continue" in function_source


def test_parallel_script_detector_qr_preview_is_passed_to_unified_editor() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    helper_def = source.index("def build_qr_rod_detector_region_preview_figure(")
    preview_setup = source.index("qr_rod_detector_region_preview_figures = []")
    preview_create = source.index("build_qr_rod_detector_region_preview_figure(", preview_setup)
    nonzero_editor_call = source.index("qr_rod_nonzero_editor_result = edit_qr_rod_region_editor(")
    editor_call = source.index("qr_rod_specular_editor_result = edit_qr_rod_region_editor(")
    final_detector_save = source.index(
        "detector_region_png, detector_region_pdf = save_manuscript_figure"
    )
    editor_call_section = source[
        editor_call : source.index(")\nmarker_table = pd.DataFrame", editor_call)
    ]
    nonzero_editor_call_section = source[
        nonzero_editor_call : source.index(")\nmarker_table = pd.DataFrame", nonzero_editor_call)
    ]
    preview_helper_section = source[helper_def:preview_setup]

    assert helper_def < preview_setup < preview_create < editor_call < final_detector_save
    assert "return preview_fig, update_preview" in preview_helper_section
    assert (
        "qr_rod_detector_region_preview_update_callback = None" in source[preview_setup:editor_call]
    )
    assert (
        "qr_rod_detector_region_preview_figure,\n"
        "            qr_rod_detector_region_preview_update_callback,"
    ) in source[preview_setup:editor_call]
    assert "companion_figures=qr_rod_detector_region_preview_figures" in editor_call_section
    assert "companion_figures=qr_rod_detector_region_preview_figures" in nonzero_editor_call_section
    assert (
        "region_preview_update_callback=qr_rod_detector_region_preview_update_callback"
        in editor_call_section
    )
    assert 'num="Qr rod detector region preview"' in source
    assert "build_detector_selected_qr_rod_band_visual_payload(" in source[helper_def:editor_call]
    assert "preview_overlay_artists" in preview_helper_section
    assert ".remove()" in preview_helper_section
    assert "preview_fig.canvas.draw_idle()" in preview_helper_section
    assert "l_bounds=qr_rod_l_bounds_for_group(" not in preview_helper_section
    assert "prefer_fallback=l_bounds is not None" not in preview_helper_section
    assert "specular_preview_roi_mask = detector_phi_two_theta_roi_mask(" in preview_helper_section
    assert "profile_mask_override=specular_profile_mask_override" not in preview_helper_section
    assert "override_visual = qr_rod_profile_mask_visual_payload(" not in preview_helper_section


def test_parallel_script_qr_rod_refresh_callbacks_filter_editor_phase() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    recompute_source = source[
        source.index("def recompute_qr_rod_region_profiles(") : source.index(
            "\ndef build_qr_rod_detector_region_preview_figure("
        )
    ]
    preview_helper_section = source[
        source.index("def build_qr_rod_detector_region_preview_figure(") : source.index(
            "\nQR_ROD_PROFILE_CACHE_PATH ="
        )
    ]

    assert "editor_phase: object = None" in recompute_source
    assert "marker_table: object = None" in recompute_source
    builder_call = recompute_source.index("specular_profile = build_specular_qr_rod_profile(")
    base_loop = recompute_source.index("for (m_value, branch_value), sub in")
    assert builder_call < base_loop
    assert "if m_int == 0:" in recompute_source
    assert "qr_rod_editor_phase_matches(m_int, editor_phase)" in recompute_source
    assert 'delta_qr_cache_value = "specular_roi" if specular_phase' in recompute_source
    assert "else round(float(delta_qr), 12)" in recompute_source
    assert "profile_mask_override=specular_profile_mask_override" not in recompute_source
    assert "qr_integration_source_override=(" not in recompute_source
    assert "editor_phase: object = None" in preview_helper_section
    assert "draw_nonzero_preview = qr_rod_editor_phase_matches(1, editor_phase)" in (
        preview_helper_section
    )
    assert "draw_specular_preview = qr_rod_editor_phase_matches(0, editor_phase)" in (
        preview_helper_section
    )
    assert "if draw_nonzero_preview:" in preview_helper_section
    assert "if draw_specular_preview" in preview_helper_section


def test_parallel_script_pre_editor_cache_is_checked_before_expensive_stages() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    background_cache_lookup = source.index(
        '"background_peak_fits", PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE'
    )
    process_fit_call = source.index("_run_process_peak_jobs()", background_cache_lookup)
    profile_cache_lookup = source.index('"profile_fits", PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE')
    profile_fit_call = source.index("_fit_profile_cache_item(bg, item)", profile_cache_lookup)
    qr_rod_pre_cache_lookup = source.index('"qr_rod_pre_editor", PRE_EDITOR_QR_ROD_STAGE_SIGNATURE')
    marker_editor_call = source.index("edit_qr_rod_region_editor(", qr_rod_pre_cache_lookup)

    assert background_cache_lookup < process_fit_call
    assert profile_cache_lookup < profile_fit_call
    assert qr_rod_pre_cache_lookup < marker_editor_call
    assert "RA_SIM_RESET_PRE_EDITOR_CACHE" in source
    signature_block = source[
        source.index("PRE_EDITOR_CACHE_INPUT_SIGNATURE = {") : source.index(
            "PRE_EDITOR_CACHE_KEY = pre_editor_cache_key("
        )
    ]
    assert '"sample_name": SAMPLE_NAME' not in signature_block


def test_parallel_script_qz_l_axis_helper_is_defined_before_editor_l_window_setup() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    helper_def = source.index("def qz_values_to_l_axis(")
    editor_l_window_setup = source.index("qr_rod_editor_initial_l_min =")

    assert source.count("def qz_values_to_l_axis(") == 1
    assert helper_def < editor_l_window_setup
    assert "def qr_rod_l_bounds_for_group(" in source[:editor_l_window_setup]
    assert "def editor_l_bounds_for_group(m_value: int)" in source
    assert "group_l_bounds = editor_l_bounds_for_group(int(m_value))" in source
    assert "qr_rod_editor_initial_l_min = float(NONZERO_QR_ROD_L_MIN)" in source
    assert "qr_rod_editor_initial_l_max = float(NONZERO_QR_ROD_L_MAX)" in source
    assert (
        'PRE_EDITOR_QR_ROD_STAGE_SIGNATURE = "qr_rod_pre_marker_profiles_hk0_roi_v17_caked_phi_m90_90_plane"'
        in source
    )


def test_parallel_script_qr_rod_peak_editor_uses_l_axis() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef qr_rod_profile_cache_with_final_fit("
        )
    ]

    assert "qz_to_l_linear_coeff(" in function_source
    assert "marker_source=edited" in function_source
    assert "qz_to_editor_l(" in function_source
    assert "editor_l_to_qz(" in function_source
    assert 'ax.set_xlabel("L"' in function_source
    assert "MaxNLocator(integer=True)" in function_source
    assert 'ax.set_xlabel("Qz"' not in function_source


def test_parallel_script_qr_rod_peak_editor_caches_plot_inputs() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef qr_rod_profile_cache_with_final_fit("
        )
    ]

    assert "group_plot_cache:" in function_source
    assert "clear_group_plot_cache(" in function_source
    assert "group_plot_cache[cache_key]" in function_source
    assert "clear_group_plot_cache()" in function_source


def test_parallel_script_qr_rod_peak_editor_has_import_export_buttons() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef qr_rod_profile_cache_with_final_fit("
        )
    ]

    for token in (
        'Button(button_axes[1], "Import")',
        'Button(button_axes[2], "Export")',
        "filedialog.askopenfilename(",
        "filedialog.asksaveasfilename(",
        "load_qr_rod_peak_edits(import_path)",
        "write_qr_rod_peak_edits(",
        "parameters=current_phase_peak_edit_parameters()",
    ):
        assert token in function_source


def test_parallel_script_qr_rod_peak_editor_shows_hk0_in_log_view() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef qr_rod_profile_cache_with_final_fit("
        )
    ]

    assert "has_hk0_markers = hk0_qz_marker_count(edited) > 0" in function_source
    assert "if required_marker_table is not None and not has_hk0_markers:" in function_source
    assert (
        "edited = marker_table_with_specular_l_markers(edited, required_marker_table)"
        in function_source
    )
    assert "original = edited.copy()" in function_source
    assert "editor_log_y = int(m_value) == 0" in function_source
    assert "y_plot = positive_log_plot_values(y) if editor_log_y else y" in function_source
    assert (
        "y_markers_plot = positive_log_plot_values(y_markers) if editor_log_y else y_markers"
        in function_source
    )
    assert 'ax.set_ylabel("Intensity (log)" if editor_log_y else "Intensity", fontsize=8)' in (
        function_source
    )
    assert "if editor_log_y:\n                apply_positive_log_y_axis(ax, y, y_markers)" in (
        function_source
    )
    assert 'ax.set_yscale("linear")' in function_source


def test_parallel_script_qr_rod_peak_editor_hk0_uses_specular_rows_without_l_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    namespace["PBI2_ROD_PROFILE_L_AXIS_MAX"] = 3.0
    namespace["IS_PBI2_SAMPLE_STEM"] = True
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed: dict[str, object] = {}

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        boxes = {
            getattr(getattr(widget, "label", None), "get_text", lambda: "")(): widget
            for widget in list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
            if type(widget).__name__ == "TextBox"
        }
        assert "L Min" not in boxes
        assert "L Max" not in boxes
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        hk0 = panels["HK=0 qz"]
        observed["hk0_xlim"] = hk0.get_xlim()
        observed["hk0_yscale"] = hk0.get_yscale()
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
            {"m": 1, "branch": "+", "qz_marker": 0.5, "fit_l": 0.5, "display_l": 0.5},
            {"m": 1, "branch": "+", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 4.0},
            {"m": 0, "branch": "qz", "qz_center": 2.2, "background_density": 8.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
            {"m": 1, "branch": "+", "qz_center": 0.5, "background_density": 1.0},
            {"m": 1, "branch": "+", "qz_center": 1.5, "background_density": 2.0},
            {"m": 1, "branch": "+", "qz_center": 3.0, "background_density": 1.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 0.5, "l_max": 3.0},
        editor_phase="specular",
    )

    assert accepted is True
    hk0_xmin, hk0_xmax = observed["hk0_xlim"]
    assert 1.0 < hk0_xmin < 1.5
    assert 3.0 <= hk0_xmax < 3.2
    assert observed["hk0_yscale"] == "log"


def test_parallel_script_qr_rod_peak_editor_filters_phases_and_preserves_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed_panels: list[list[str]] = []

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        observed_panels.append(sorted(ax.get_title() for ax in fig.axes if ax.get_xlabel() == "L"))
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
            {"m": 1, "branch": "+", "qz_marker": 0.5, "fit_l": 0.5, "display_l": 0.5},
            {"m": 1, "branch": "+", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_center": 1.5, "background_density": 4.0},
            {"m": 0, "branch": "qz", "qz_center": 3.0, "background_density": 5.0},
            {"m": 1, "branch": "+", "qz_center": 0.5, "background_density": 1.0},
            {"m": 1, "branch": "+", "qz_center": 3.0, "background_density": 1.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    edited_nonzero, accepted_nonzero = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        editor_phase="nonzero",
    )
    edited_specular, accepted_specular = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        editor_phase="specular",
    )

    assert accepted_nonzero is True
    assert accepted_specular is True
    assert observed_panels == [["HK=1 +"], ["HK=0 qz"]]
    assert sorted(pd.DataFrame(edited_nonzero)["m"].astype(int).tolist()) == [0, 0, 1, 1]
    assert sorted(pd.DataFrame(edited_specular)["m"].astype(int).tolist()) == [0, 0, 1, 1]


def test_parallel_script_qr_rod_peak_editor_specular_phase_opens_without_profile_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed_panels: list[list[str]] = []

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        observed_panels.append(sorted(ax.get_title() for ax in fig.axes if ax.get_xlabel() == "L"))
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "qz_marker": 1.5, "fit_l": 1.5, "display_l": 1.5},
            {"m": 0, "branch": "qz", "qz_marker": 3.0, "fit_l": 3.0, "display_l": 3.0},
            {"m": 1, "branch": "+", "qz_marker": 0.5, "fit_l": 0.5, "display_l": 0.5},
        ]
    )
    nonzero_only_profiles = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 0.5, "background_density": 1.0},
            {"m": 1, "branch": "+", "qz_center": 3.0, "background_density": 1.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    edited, accepted = show_editor(
        marker_table,
        nonzero_only_profiles,
        backend_name="TkAgg",
        editor_phase="specular",
    )

    assert accepted is True
    assert observed_panels == [["HK=0 qz"]]
    assert sorted(pd.DataFrame(edited)["m"].astype(int).tolist()) == [0, 0, 1]


def test_parallel_script_qr_rod_peak_editor_click_preserves_panel_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed: dict[str, object] = {}

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        target = panels["HK=3 -"]
        target.set_xlim(0.25, 3.25)
        target.set_ylim(0.0, 100.0)
        before_xlim = target.get_xlim()
        before_ylim = target.get_ylim()
        fig.canvas.draw()
        x_pixel, y_pixel = target.transData.transform((2.0, 25.0))
        event = MouseEvent("button_press_event", fig.canvas, x_pixel, y_pixel, button=1)
        fig.canvas.callbacks.process("button_press_event", event)
        observed["before_xlim"] = before_xlim
        observed["before_ylim"] = before_ylim
        observed["after_xlim"] = target.get_xlim()
        observed["after_ylim"] = target.get_ylim()
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 3, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 3, "branch": "-", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 3, "branch": "-", "qz_center": 0.5, "background_density": 10.0},
            {"m": 3, "branch": "-", "qz_center": 1.5, "background_density": 20.0},
            {"m": 3, "branch": "-", "qz_center": 2.5, "background_density": 35.0},
            {"m": 3, "branch": "-", "qz_center": 3.0, "background_density": 30.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 0.5, "l_max": 3.0},
    )

    assert accepted is True
    assert observed["after_xlim"] == pytest.approx(observed["before_xlim"])
    assert observed["after_ylim"] == pytest.approx(observed["before_ylim"])


def test_parallel_script_qr_rod_peak_editor_hk1_minus_click_does_not_submit_l_min(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    region_state = {"delta_qr": 0.01, "l_min": 0.5, "l_max": 3.0}
    observed: dict[str, object] = {}

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        widgets = list(getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets"))
        l_min_box = next(
            widget
            for widget in widgets
            if type(widget).__name__ == "TextBox"
            and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "L Min"
        )
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        target = panels["HK=1 -"]
        l_min_box.begin_typing()
        l_min_box.text_disp.set_text("1.25")
        l_min_box.cursor_index = len("1.25")
        fig.canvas.draw()
        x_pixel, y_pixel = target.transData.transform((1.0, 1.0))
        event = MouseEvent("button_press_event", fig.canvas, x_pixel, y_pixel, button=1)
        event.inaxes = target
        fig.canvas.callbacks.process("button_press_event", event)
        observed["l_min_after_click"] = region_state["l_min"]
        observed["l_min_box_active_after_click"] = bool(l_min_box.capturekeystrokes)
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "-", "qz_marker": 1.0, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 2.0, "fit_l": 2.0, "display_l": 2.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "-", "qz_center": 1.0, "background_density": 1.0},
            {"m": 1, "branch": "-", "qz_center": 2.0, "background_density": 2.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
        editor_phase="nonzero",
    )

    assert accepted is True
    assert observed["l_min_after_click"] == pytest.approx(0.5)
    assert observed["l_min_box_active_after_click"] is False


def test_parallel_script_qr_rod_peak_editor_hk1_minus_drag_keeps_other_markers_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.backend_bases import MouseEvent

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed: dict[str, object] = {}

    def emit_event(fig, ax, name: str, x_value: float, y_value: float) -> None:
        x_pixel, y_pixel = ax.transData.transform((float(x_value), float(y_value)))
        event = MouseEvent(name, fig.canvas, x_pixel, y_pixel, button=1)
        event.inaxes = ax
        fig.canvas.callbacks.process(name, event)

    def marker_offsets(ax) -> np.ndarray:
        assert ax.collections
        offsets = ax.collections[0].get_offsets()
        return np.asarray(np.ma.filled(offsets, np.nan), dtype=np.float64)

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        target = panels["HK=1 -"]
        fig.canvas.draw()
        before_xlim = target.get_xlim()
        initial_offsets = marker_offsets(target)
        selected_index = 2
        selected_x = float(initial_offsets[selected_index, 0])
        selected_y = float(initial_offsets[selected_index, 1])
        emit_event(fig, target, "button_press_event", selected_x, selected_y)
        emit_event(fig, target, "motion_notify_event", selected_x + 0.2, selected_y)
        moved_offsets = marker_offsets(target)
        observed["before_xlim"] = before_xlim
        observed["after_xlim"] = target.get_xlim()
        observed["initial_marker_x"] = initial_offsets[:, 0].copy()
        observed["moved_marker_x"] = moved_offsets[:, 0].copy()
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 1, "branch": "-", "qz_marker": 0.9, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 1.7, "fit_l": 1.0, "display_l": 1.0},
            {"m": 1, "branch": "-", "qz_marker": 1.95, "fit_l": 2.0, "display_l": 2.0},
            {"m": 1, "branch": "-", "qz_marker": 2.7, "fit_l": 3.0, "display_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "-", "qz_center": 0.9, "background_density": 8.0},
            {"m": 1, "branch": "-", "qz_center": 1.7, "background_density": 18.0},
            {"m": 1, "branch": "-", "qz_center": 1.95, "background_density": 26.0},
            {"m": 1, "branch": "-", "qz_center": 2.7, "background_density": 12.0},
        ]
    )

    monkeypatch.setattr(plt, "show", fake_show)
    _edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state={"delta_qr": 0.01, "l_min": 0.5, "l_max": 3.0},
        editor_phase="nonzero",
    )

    assert accepted is True
    assert observed["after_xlim"] == pytest.approx(observed["before_xlim"])
    initial_x = np.asarray(observed["initial_marker_x"], dtype=np.float64)
    moved_x = np.asarray(observed["moved_marker_x"], dtype=np.float64)
    assert moved_x[[0, 1, 3]].tolist() == pytest.approx(initial_x[[0, 1, 3]].tolist())
    assert moved_x[2] == pytest.approx(float(initial_x[2]) + 0.2)


def test_parallel_script_qr_rod_peak_editor_hk4_minus_drag_preserves_panel_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.widgets import TextBox

    namespace = _script_functions(
        "as_float",
        "qr_rod_peak_edit_runtime_mode",
        "show_qr_rod_peak_marker_popup",
        "marker_table_with_specular_l_markers",
        "qz_l_linear_coeff_from_marker_rows",
        "marker_row_title",
        "clean_marker_title",
        "replace_qr_rod_marker_group_qz",
        "positive_log_plot_values",
        "apply_positive_log_y_axis",
        "snap_qr_rod_markers_to_profile_peaks",
        "l_tick_label",
        "_safe_run_name",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    namespace["ACTIVE_LATTICE_C"] = 2.0 * np.pi
    namespace["SAMPLE_STEM"] = "pbi2"
    show_editor = namespace["show_qr_rod_peak_marker_popup"]
    observed: dict[str, object] = {}
    clear_calls: list[object] = []
    draw_calls: list[object] = []
    text_box_set_val_calls: list[object] = []
    target_set_xlim_calls: list[object] = []
    target_axis: dict[str, object] = {"ax": None, "record": False}
    original_clear = Axes.clear
    original_draw = FigureCanvasAgg.draw
    original_set_xlim = Axes.set_xlim
    original_text_box_set_val = TextBox.set_val

    def counting_clear(self, *args, **kwargs):
        clear_calls.append(self)
        return original_clear(self, *args, **kwargs)

    def counting_set_xlim(self, *args, **kwargs):
        if bool(target_axis["record"]) and self is target_axis["ax"]:
            target_set_xlim_calls.append((args, kwargs))
        return original_set_xlim(self, *args, **kwargs)

    def counting_draw(self, *args, **kwargs):
        draw_calls.append(self)
        return original_draw(self, *args, **kwargs)

    def counting_text_box_set_val(self, *args, **kwargs):
        text_box_set_val_calls.append(self)
        return original_text_box_set_val(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "clear", counting_clear)
    monkeypatch.setattr(Axes, "set_xlim", counting_set_xlim)
    monkeypatch.setattr(FigureCanvasAgg, "draw", counting_draw)
    monkeypatch.setattr(TextBox, "set_val", counting_text_box_set_val)

    def emit_event(fig, ax, name: str, x_value: float, y_value: float) -> None:
        x_pixel, y_pixel = ax.transData.transform((float(x_value), float(y_value)))
        event = MouseEvent(name, fig.canvas, x_pixel, y_pixel, button=1)
        event.inaxes = ax
        fig.canvas.callbacks.process(name, event)

    def fake_show(*_args, **_kwargs) -> None:
        fig = plt.gcf()
        panels = {ax.get_title(): ax for ax in fig.axes if ax.get_xlabel() == "L"}
        target = panels["HK=4 -"]
        target_axis["ax"] = target
        observed["initial_profile_x"] = np.asarray(target.lines[0].get_xdata(), dtype=np.float64)
        target.set_xlim(2.0, 2.8)
        target.set_ylim(0.0, 100.0)
        before_xlim = target.get_xlim()
        before_ylim = target.get_ylim()
        title_box = next(
            widget
            for widget in getattr(fig, "_ra_sim_qr_rod_peak_edit_widgets")
            if type(widget).__name__ == "TextBox"
            and getattr(getattr(widget, "label", None), "get_text", lambda: "")() == "Label"
        )
        fig.canvas.draw()
        observed["initial_clear_count"] = len(clear_calls)
        observed["initial_draw_count"] = len(draw_calls)
        observed["initial_text_box_set_val_count"] = len(text_box_set_val_calls)
        target_axis["record"] = True
        emit_event(fig, target, "button_press_event", 2.68, 25.0)
        observed["after_press_xlim"] = target.get_xlim()
        observed["after_press_clear_count"] = len(clear_calls)
        observed["after_press_draw_count"] = len(draw_calls)
        observed["after_press_text_box_set_val_count"] = len(text_box_set_val_calls)
        observed["after_press_label_text"] = title_box.text
        motion_xlims = []
        for x_value in (2.69, 2.70, 2.71, 2.72):
            emit_event(fig, target, "motion_notify_event", x_value, 26.0)
            motion_xlims.append(target.get_xlim())
        observed["after_motion_xlims"] = motion_xlims
        observed["after_motion_clear_count"] = len(clear_calls)
        observed["after_motion_draw_count"] = len(draw_calls)
        observed["after_motion_text_box_set_val_count"] = len(text_box_set_val_calls)
        emit_event(fig, target, "button_release_event", 2.72, 26.0)
        observed["target_set_xlim_calls"] = list(target_set_xlim_calls)
        observed["after_release_clear_count"] = len(clear_calls)
        observed["after_release_draw_count"] = len(draw_calls)
        observed["after_release_text_box_set_val_count"] = len(text_box_set_val_calls)
        observed["before_xlim"] = before_xlim
        observed["before_ylim"] = before_ylim
        observed["after_xlim"] = target.get_xlim()
        observed["after_ylim"] = target.get_ylim()
        plt.close(fig)

    marker_table = pd.DataFrame(
        [
            {"m": 4, "branch": "-", "qz_marker": 0.478292, "fit_l": 2.166717},
            {"m": 4, "branch": "-", "qz_marker": 1.358040, "fit_l": 2.342622},
            {"m": 4, "branch": "-", "qz_marker": 2.681896, "fit_l": 2.607324},
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 3.0},
        ]
    )
    rod_profile_table = pd.DataFrame(
        [
            {"m": 4, "branch": "-", "qz_center": 0.5, "background_density": 12.0},
            {"m": 4, "branch": "-", "qz_center": 1.5, "background_density": 28.0},
            {"m": 4, "branch": "-", "qz_center": 2.7, "background_density": 18.0},
            {"m": 1, "branch": "+", "qz_center": 1.0, "background_density": 5.0},
            {"m": 1, "branch": "+", "qz_center": 2.0, "background_density": 8.0},
        ]
    )
    region_state = {"delta_qr": 0.01, "l_min": 0.5, "l_max": 3.0}

    monkeypatch.setattr(plt, "show", fake_show)
    edited, accepted = show_editor(
        marker_table,
        rod_profile_table,
        backend_name="TkAgg",
        region_state=region_state,
    )

    assert accepted is True
    assert observed["initial_profile_x"] == pytest.approx([0.5, 1.5, 2.7])
    assert observed["after_press_xlim"] == pytest.approx(observed["before_xlim"])
    for motion_xlim in observed["after_motion_xlims"]:
        assert motion_xlim == pytest.approx(observed["before_xlim"])
    assert observed["after_xlim"] == pytest.approx(observed["before_xlim"])
    assert observed["after_ylim"] == pytest.approx(observed["before_ylim"])
    assert observed["target_set_xlim_calls"] == []
    assert observed["after_press_clear_count"] == observed["initial_clear_count"]
    assert observed["after_motion_clear_count"] == observed["after_press_clear_count"]
    assert observed["after_release_clear_count"] == observed["after_motion_clear_count"]
    assert observed["after_press_draw_count"] == observed["initial_draw_count"]
    assert observed["after_motion_draw_count"] == observed["after_press_draw_count"]
    assert observed["after_release_draw_count"] == observed["after_motion_draw_count"]
    assert (
        observed["after_press_text_box_set_val_count"] == observed["initial_text_box_set_val_count"]
    )
    assert (
        observed["after_motion_text_box_set_val_count"]
        == observed["after_press_text_box_set_val_count"]
    )
    assert (
        observed["after_release_text_box_set_val_count"]
        == observed["after_motion_text_box_set_val_count"]
    )
    assert observed["after_press_label_text"]
    assert region_state["delta_qr"] == pytest.approx(0.01)
    assert region_state["l_min"] == pytest.approx(0.5)
    assert region_state["l_max"] == pytest.approx(3.0)
    edited_table = pd.DataFrame(edited)
    unchanged = edited_table[(edited_table["m"] == 1) & (edited_table["branch"] == "+")]
    assert unchanged["qz_marker"].tolist() == pytest.approx([1.0, 2.0])


def test_parallel_script_qr_rod_plots_hide_hk7() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    editor_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef qr_rod_profile_cache_with_final_fit("
        )
    ]
    final_profile_source = source[
        source.index("drawable_profile_keys = drawable_rod_profile_keys(") : source.index(
            "\nif not plot_rod_entries:"
        )
    ]

    assert "QR_ROD_HIDDEN_PLOT_HK = (7,)" in source
    assert 'globals().get("QR_ROD_HIDDEN_PLOT_HK", ())' in editor_source
    assert "detector_plot_rod_entries = detector_complete_branch_rod_entries(" in (
        final_profile_source
    )
    assert "detector_plot_rod_entries_all = list(detector_plot_rod_entries)" in (
        final_profile_source
    )
    assert (
        'rod for rod in detector_plot_rod_entries_all if int(rod["m"]) not in hidden_plot_hk'
        in (final_profile_source)
    )
    assert "skipped configured-hidden Qr-rod final figure rows" in source


def test_parallel_script_saved_figures_do_not_include_panel_letters() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    panel_labels_source = source[
        source.index("def add_panel_labels(") : source.index("\ndef maybe_suptitle(")
    ]
    detector_region_source = source[
        source.index(
            "rod_label_entries = apply_unified_qr_rod_region_editor_labels("
        ) : source.index("\ndetector_region_png, detector_region_pdf = save_manuscript_figure")
    ]

    assert "return\n    flat_axes" in panel_labels_source
    assert 'add_panel_label(ax, "(a)"' not in source
    assert 'add_panel_label(ax, "(b)"' not in source
    assert 'add_panel_label(ax, "(c)"' not in source
    assert "maybe_suptitle(fig," not in detector_region_source


def test_parallel_script_has_no_final_rod_marker_label_helper() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    tree = ast.parse(
        PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(PARALLEL_SCRIPT_PATH)
    )
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    call_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "rod_marker_annotation_label" not in names
    assert "rod_marker_annotation_label" not in call_names


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
        'baseline_equation": "density = b + m*(two_theta_deg - seed_two_theta_deg)',
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
