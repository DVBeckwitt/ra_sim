import ast
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path
from textwrap import dedent

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation
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
        "pickle": pickle,
        "replace": replace,
        "re": re,
        "plt": plt,
        "LogNorm": LogNorm,
        "binary_dilation": binary_dilation,
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
        "QR_ROD_FINAL_FIT_CACHE_SIGNATURE": "joint_qz_labeled_marker_fit_specular_theta_i0_l8_v4",
        "PRE_EDITOR_CACHE_SCHEMA": "ra_sim.background_pre_editor_cache.v1",
        "PRE_EDITOR_CACHE_SIGNATURE": "pre_qr_rod_marker_editor_inputs_v1",
        "PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE": "background_peak_fit_results_v1",
        "PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE": "profile_fit_cache_v1",
        "PRE_EDITOR_QR_ROD_STAGE_SIGNATURE": "qr_rod_pre_marker_profiles_specular_theta_i0_l8_v3",
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
    detector_figure_start = source.index("fig, ax = plt.subplots(figsize=(JOURNAL_FULL_WIDTH_IN, fig_height)")
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
    editor_call = source.index("edit_detector_region_label_positions(", detector_figure_start)
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
    namespace = _script_functions("qr_rod_peak_edit_runtime_mode", "detector_label_edit_runtime_mode")
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
    assert 'fontsize=detector_label_fontsize(entry)' in function_source
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


def test_detector_region_label_editor_uses_tkinter_coordinate_controls() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "import tkinter as tk",
        "from tkinter import ttk",
        "root = tk.Tk()",
        "canvas = tk.Canvas(",
        "canvas.create_image(",
        "canvas.create_text(",
        "canvas.tag_bind(",
        "label_var = tk.StringVar",
        "x_var = tk.StringVar",
        "y_var = tk.StringVar",
        "font_var = tk.StringVar",
        "ttk.Combobox(",
        "ttk.Entry(",
        "ttk.Button(",
        'text="X -"',
        'text="X +"',
        'text="Y -"',
        'text="Y +"',
        'text="Import"',
        'text="Export"',
        'text="Cancel"',
        'text="Accept"',
        "load_detector_label_settings(",
        "save_detector_label_settings(",
    ):
        assert token in function_source

    assert "from matplotlib.widgets" not in function_source
    assert "TextBox(" not in function_source
    assert "Button(fig.add_axes" not in function_source


def test_detector_region_label_editor_selects_labels_without_matplotlib_events() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "def select_label(index: int)",
        "label_selector.bind(",
        'canvas.tag_bind(item, "<Button-1>"',
        "update_canvas_label(",
        "sync_controls()",
    ):
        assert token in function_source
    assert "mpl_connect(" not in function_source
    assert "def on_motion(event)" not in function_source
    assert "def on_release(event)" not in function_source
    assert "def on_press(event)" not in function_source
    assert '"dragging": False' not in function_source
    assert ".get_lines()" not in function_source
    assert ".images" not in function_source


def test_detector_region_label_editor_tunes_pixel_locations_below_image() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "def set_label_position(",
        "def apply_position_fields(",
        "def nudge_label(",
        'x_entry.bind("<Return>",',
        'y_entry.bind("<Return>",',
        'x_entry.bind("<FocusOut>",',
        'y_entry.bind("<FocusOut>",',
        'command=lambda: nudge_label(-1.0, 0.0)',
        'command=lambda: nudge_label(1.0, 0.0)',
        'command=lambda: nudge_label(0.0, -1.0)',
        'command=lambda: nudge_label(0.0, 1.0)',
        "canvas.coords(",
        "x_var.set(",
        "y_var.set(",
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
        'canvas.tag_bind(item, "<Button-1>"',
        'canvas.tag_bind(item, "<B1-Motion>"',
        'canvas.tag_bind(item, "<ButtonRelease-1>"',
        "def canvas_event_xy(",
        "canvas.canvasx(event.x)",
        "canvas.canvasy(event.y)",
        "def canvas_xy_to_data(",
        "ax.transData.inverted().transform(",
        "def start_label_drag(",
        "def drag_label_motion(",
        "def finish_label_drag(",
        "def drag_active_label_motion(",
        "def finish_active_label_drag(",
        'canvas.bind("<B1-Motion>", drag_active_label_motion)',
        'canvas.bind("<ButtonRelease-1>", finish_active_label_drag)',
        '"dragging_index": None',
    ):
        assert token in function_source


def test_detector_region_label_editor_renders_static_matplotlib_image_for_tk_canvas() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_detector_region_label_position_popup(") : source.index(
            "\ndef edit_detector_region_label_positions("
        )
    ]

    for token in (
        "fig.savefig(",
        "tempfile.NamedTemporaryFile(",
        "photo = tk.PhotoImage(",
        "display_width, display_height = fig.canvas.get_width_height()",
        "def data_to_canvas_xy(",
        "canvas_y = float(display_height) - float(display_y)",
    ):
        assert token in function_source


def test_detector_region_label_editor_wires_before_final_save() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    detector_figure_start = source.index("rod_label_entries: list[dict[str, object]] = []")
    editor_call = source.index("edit_detector_region_label_positions(", detector_figure_start)
    save_call = source.index(
        "detector_region_png, detector_region_pdf = save_manuscript_figure",
        detector_figure_start,
    )
    section = source[detector_figure_start:save_call]

    assert editor_call < save_call
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
            "for rod in ([] if qr_rod_pre_editor_cache_hit else rod_entries):"
        )
    ]
    nonzero_profile_source = source[
        source.index("for rod in ([] if qr_rod_pre_editor_cache_hit else rod_entries):") : source.index(
            "\nif not profile_rows:"
        )
    ]
    detector_figure_source = source[
        source.index("specular_color = OKABE_ITO") : source.index(
            "\ndisplay_detector_rotation_fit_debug("
        )
    ]

    assert "specular_detector_theta_initial_deg = 3.0 * float(ROD_PROFILE_TILT_DEG)" in source
    assert "specular_detector_q_config = detector_qspace_config_with_theta_initial(" in qmap_setup_source
    assert "specular_detector_q_maps = notebook_detector_qr_qz_maps(" in qmap_setup_source
    assert "config=specular_detector_q_config" in qmap_setup_source
    assert "specular_detector_region_mask" in specular_profile_source
    assert "specular_detector_qr_map = np.asarray(specular_detector_q_maps[0]" in specular_profile_source
    assert "specular_detector_qz_map = np.asarray(specular_detector_q_maps[1]" in specular_profile_source
    assert "specular_detector_valid_q = np.asarray(specular_detector_q_maps[2]" in specular_profile_source
    assert "SPECULAR_QR_ROD_L_MAX = 8.0" in source
    assert "specular_qz_bounds = qz_bounds_for_l_window(" in specular_profile_source
    assert "l_max=SPECULAR_QR_ROD_L_MAX" in specular_profile_source
    assert "specular_detector_region_mask = specular_detector_region_mask &" in specular_profile_source
    assert "profile_from_detector_qr_qz(" in specular_profile_source
    assert "detector_q_maps=specular_detector_q_maps" in specular_profile_source
    assert "theta_initial_deg_used_for_q=specular_detector_theta_initial_deg" in specular_profile_source
    assert "detector_q_maps=profile_detector_q_maps" in nonzero_profile_source
    assert "profile_from_full_mask(" not in specular_profile_source
    assert 'branch_label="m = 0"' in specular_profile_source
    assert "specular_detector_qz_values = np.asarray(specular_qz_values" in detector_figure_source
    assert "qr_map=specular_detector_qr_map" in detector_figure_source
    assert "qz_map=specular_detector_qz_map" in detector_figure_source
    assert "valid_q=specular_detector_valid_q" in detector_figure_source
    assert "projected_col, projected_row, specular_detector_qz_map" in detector_figure_source
    assert '"label_id": detector_specular_label_id()' in detector_figure_source
    assert "theta_i0 = 3*theta_i" in source


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
    assert "if trace_qz_bounds is None:\n            projected_col = raw_projected_col.copy()" in detector_figure_source
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
    assert 'branch_suffix = "+" if int(payload.get("branch_sign", 0)) > 0 else "-"' in detector_figure_source
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
    assert "detector_visual_label_lines = detector_mask_centerline_from_visual(visual)" in detector_figure_source
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
        source.index("ax.set_xlabel(\"Detector x pixel") : source.index(
            "\nrod_label_entries = edit_detector_region_label_positions("
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
            "specular_lines = specular_detector_lines_from_markers", specular_draw_start
        )
    ]

    assert "fill_alpha=detector_region_specular_band_alpha" in specular_draw
    assert "boundary_alpha=detector_region_specular_boundary_alpha" in specular_draw
    assert "boundary_expand_px=detector_region_specular_boundary_expand_px" in specular_draw
    assert (
        "delta_qr=float(qr_rod_delta_qr)"
        in source[source.index("specular_delta_q_visual = (") : specular_draw_start]
    )


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


def test_parallel_script_final_rod_profile_entries_exclude_empty_hk_rows() -> None:
    namespace = _script_functions(
        "qz_l_linear_coeff_from_marker_rows",
        "drawable_rod_profile_keys",
    )
    drawable_keys = namespace["drawable_rod_profile_keys"]
    profile_table = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_center": 1.0, "pixel_count": 5, "background_density": 2.0, "joint_peak_density": 0.5},
            {"m": 1, "branch": "+", "qz_center": 1.5, "pixel_count": 6, "background_density": 2.5, "joint_peak_density": 0.6},
            {"m": 7, "branch": "+", "qz_center": 1.0, "pixel_count": 0, "background_density": np.nan, "joint_peak_density": np.nan},
            {"m": 7, "branch": "-", "qz_center": 1.4, "pixel_count": 0, "background_density": np.nan, "joint_peak_density": np.nan},
            {"m": 3, "branch": "-", "qz_center": 1.0, "pixel_count": 5, "background_density": 1.0, "joint_peak_density": 0.2},
            {"m": 3, "branch": "-", "qz_center": 1.5, "pixel_count": 5, "background_density": 1.2, "joint_peak_density": 0.3},
            {"m": 0, "branch": "qz", "qz_center": 0.8, "pixel_count": 4, "background_density": 3.0, "joint_peak_density": 0.7},
            {"m": 0, "branch": "qz", "qz_center": 1.1, "pixel_count": 4, "background_density": 3.2, "joint_peak_density": 0.8},
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


def test_parallel_script_shared_nonzero_rod_profile_y_axis_limits_ignore_hk0() -> None:
    namespace = _script_functions(
        "normalized_data_simulation_payload",
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


def test_parallel_script_shared_nonzero_rod_profile_y_axis_limits_fallback_without_nonzero_data() -> None:
    namespace = _script_functions(
        "normalized_data_simulation_payload",
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
    hk0_limits = source.index("rod_profile_hk0_l_axis_limits = (0.0, SPECULAR_QR_ROD_L_MAX)")
    figure_loop = source.index("for row, rod in enumerate(plot_rod_entries):", limits_assign)

    assert helper_def < nonzero_entries < limits_assign < hk0_limits < figure_loop
    assert "int(rod[\"m\"]) != 0" in source[nonzero_entries:limits_assign]
    assert "nonzero_plot_rod_entries" in source[limits_assign:hk0_limits]
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
        "marker_table = marker_table.loc[~((marker_m == 0.0) & (marker_branch == \"qz\"))].copy()",
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
    final_fit_signature = "joint_qz_labeled_marker_fit_specular_theta_i0_l8_v4"
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
        "background_peak_fit_results_v1",
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
        get_stage(loaded, "background_peak_fits", "background_peak_fit_results_v1"),
        expected_fit_count=1,
    )
    assert namespace["profile_fit_stage_is_valid"](
        get_stage(loaded, "profile_fits", "profile_fit_cache_v1"),
        expected_profile_count=1,
    )
    assert namespace["qr_rod_pre_editor_stage_is_valid"](
        get_stage(loaded, "qr_rod_pre_editor", "qr_rod_pre_marker_profiles_v1")
    )
    assert load_cache(
        cache_path,
        cache_key("Bi2Se3.json", input_signature={"background_files": ["b.osc"]}),
    ) is None
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
        source.index("rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits(") : source.index(
            "maybe_suptitle(fig, rf\"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles\""
        )
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
        source.index("rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits(") : source.index(
            "maybe_suptitle(fig, rf\"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles\""
        )
    ]

    assert "rod_profile_panel_label(0, \"qz\")" in final_profile_source
    assert "rod_profile_panel_label(int(rod[\"m\"]), branch_name)" in final_profile_source
    assert "transform=ax.transAxes" in final_profile_source
    assert 'ha="center"' in final_profile_source
    assert 'va="top"' in final_profile_source
    assert "labelleft=col == 0" in final_profile_source
    assert "labelbottom=row == last_nonzero_plot_row" in final_profile_source
    assert "branch_label" not in final_profile_source
    assert '"- branch"' not in source
    assert '"+ branch"' not in source
    assert 'ax.set_title(branch_label if row == 0 else "")' not in final_profile_source
    assert 'ax.set_title(r"$HK = 0$")' not in final_profile_source
    assert 'ax.set_ylabel(f"$m = {int(rod[\'m\'])}$")' not in final_profile_source
    assert 'ax.set_ylabel(f"$HK = {int(rod[\'m\'])}$")' not in final_profile_source


def test_parallel_script_nonzero_rod_profile_grid_removes_inner_spacing() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    final_profile_source = source[
        source.index("rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits(") : source.index(
            "maybe_suptitle(fig, rf\"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles\""
        )
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
    editor_call = source.index("edit_qr_rod_peak_markers(", marker_table_index)
    cache_key_call = source.index("qr_rod_peak_edit_cache_key(", marker_table_index)
    final_fit_call = source.index("add_joint_qz_fit_columns(", marker_table_index)

    assert specular_call < marker_table_index < specular_merge < editor_call < cache_key_call < final_fit_call
    assert "RA_SIM_QR_ROD_PEAK_EDIT_MODE" in source
    assert '"QR_ROD_PEAK_EDIT_MODE_OVERRIDE", "RA_SIM_QR_ROD_PEAK_EDIT_MODE", "popup"' in source
    assert "TextBox(" in source
    assert "marker_title" in source
    assert 'getattr(event, "inaxes", None) is getattr(box, "ax", None)' in source


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
    qr_rod_pre_cache_lookup = source.index(
        '"qr_rod_pre_editor", PRE_EDITOR_QR_ROD_STAGE_SIGNATURE'
    )
    marker_editor_call = source.index("edit_qr_rod_peak_markers(", qr_rod_pre_cache_lookup)

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


def test_parallel_script_qr_rod_peak_editor_has_import_export_buttons() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_peak_markers("
        )
    ]

    for token in (
        'Button(button_axes[1], "Import")',
        'Button(button_axes[2], "Export")',
        "filedialog.askopenfilename(",
        "filedialog.asksaveasfilename(",
        "load_qr_rod_peak_edits(import_path)",
        "write_qr_rod_peak_edits(export_path, edited)",
    ):
        assert token in function_source


def test_parallel_script_qr_rod_peak_editor_shows_hk0_in_log_view() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_peak_markers("
        )
    ]

    assert "if required_marker_table is not None:" in function_source
    assert "edited = marker_table_with_specular_l_markers(edited, required_marker_table)" in function_source
    assert "original = edited.copy()" in function_source
    assert "y_plot = positive_log_plot_values(y)" in function_source
    assert "y_markers_plot = positive_log_plot_values(y_markers)" in function_source
    assert "apply_positive_log_y_axis(ax, y, y_markers)" in function_source
    assert 'ax.set_ylabel("Intensity (log)", fontsize=8)' in function_source


def test_parallel_script_saved_figures_do_not_include_panel_letters() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    panel_labels_source = source[
        source.index("def add_panel_labels(") : source.index("\ndef maybe_suptitle(")
    ]
    detector_region_source = source[
        source.index("rod_label_entries = edit_detector_region_label_positions(") : source.index(
            "\ndetector_region_png, detector_region_pdf = save_manuscript_figure"
        )
    ]

    assert "return\n    flat_axes" in panel_labels_source
    assert 'add_panel_label(ax, "(a)"' not in source
    assert 'add_panel_label(ax, "(b)"' not in source
    assert 'add_panel_label(ax, "(c)"' not in source
    assert "maybe_suptitle(fig," not in detector_region_source


def test_parallel_script_has_no_final_rod_marker_label_helper() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    tree = ast.parse(PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(PARALLEL_SCRIPT_PATH))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
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
