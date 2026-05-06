import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import pickle
import py_compile
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares, nnls

from scripts.diagnostics import background_peak_fit_worker as peak_fit_worker
from scripts.diagnostics.run_all_background_peak_fits import _output_dir_for_state, _safe_run_name


NOTEBOOK_PATH = Path("scripts/diagnostics/all_background_peak_fits.ipynb")
PARALLEL_NOTEBOOK_PATH = Path(
    "scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.ipynb"
)
PARALLEL_SCRIPT_PATH = Path(
    "scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py"
)
RUNNER_PATH = Path("scripts/diagnostics/run_all_background_peak_fits.py")


def _identity_njit(*args: object, **_kwargs: object) -> object:
    if args and callable(args[0]) and len(args) == 1:
        return args[0]

    def _decorator(func: object) -> object:
        return func

    return _decorator


def _notebook_source(notebook_path: Path = NOTEBOOK_PATH) -> str:
    if not notebook_path.exists():
        pytest.skip(f"{notebook_path} is not present in this checkout")
    if notebook_path.suffix == ".py":
        return notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _notebook_functions(*names: str, notebook_path: Path = NOTEBOOK_PATH) -> dict[str, object]:
    wanted = set(names)
    selected: list[ast.FunctionDef] = []
    if notebook_path.suffix == ".py":
        tree = ast.parse(notebook_path.read_text(encoding="utf-8"), filename=str(notebook_path))
        selected.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
        )
    else:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            tree = ast.parse(source, filename=f"{notebook_path}:cell{index}")
            selected.extend(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted
            )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "np": np,
        "pd": pd,
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
        "ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED": True,
        "ROD_QZ_NONLINEAR_REFINEMENT_ENABLED": True,
        "ROD_QZ_NONLINEAR_MAX_COMPONENTS": 14,
        "ROD_QZ_NONLINEAR_CENTER_BOUND_BINS": 4.0,
        "ROD_QZ_NONLINEAR_TAIL_POWER_BOUNDS": (0.55, 12.0),
        "ROD_QZ_NONLINEAR_MAX_NFEV": 1200,
        "QR_ROD_PROFILE_CACHE_SCHEMA": "ra_sim.qr_rod_profile_cache.v1",
        "Path": Path,
        "hashlib": hashlib,
        "njit": _identity_njit,
        "pickle": pickle,
        "POSITIVE_QZ_MIN": 0.0,
        "THETA_HALF_WINDOW_DEG": 1.8,
        "PHI_HALF_WINDOW_DEG": 6.0,
        "re": re,
    }
    exec(compile(module, str(notebook_path), "exec"), namespace)
    return namespace


def test_parallel_background_peak_fits_script_exists_and_compiles() -> None:
    assert PARALLEL_SCRIPT_PATH.exists()
    py_compile.compile(str(PARALLEL_SCRIPT_PATH), doraise=True)


def test_parallel_background_peak_fits_limits_qz_profiles_to_configured_two_theta() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    assert "ROD_PROFILE_MAX_TWO_THETA_DEG = 70.3" in source
    assert (
        "ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG = float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    )
    assert "ROD_PROFILE_MAX_TWO_THETA_DEG = rod_profile_two_theta_limit_for_background(" in source
    assert "profile_bg, ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG" in source
    assert "detector_two_theta_map=profile_detector_two_theta_map" in source
    assert "theta_map <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "two_theta_values <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "branch_mask = (" in source
    assert "& theta_region_within_profile_limit[None, :]" in source
    assert "theta_region_within_profile_limit[None, :]" in source


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


def test_parallel_background_peak_fits_script_uses_tail_aware_peak_fits() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    for token in (
        'FIT_MODEL_NAME = "rotated_gaussian_core_lorentzian_tail_shared_center"',
        "GAUSSIAN_TAIL_DISTANCE_WEIGHT = 1.25",
        "GAUSSIAN_CORE_SIGNAL_DOWNSCALE = 0.06",
        "GAUSSIAN_TAIL_OVERPREDICTION_START = 0.55",
        "GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT = 1.75",
        "def _rotated_gaussian_value_numba",
        "gaussian_core = math.exp(-0.5 * gaussian_r2)",
        "lorentzian_tail = 1.0 /",
        "(u / gamma_l_u) * (u / gamma_l_u)",
        "_rotated_gaussian_residual_points_numba",
        "tail_overprediction_weight",
        "if residual > 0.0:",
        "residual *= tail_overprediction_weight[idx]",
        '"fit_model": FIT_MODEL_NAME',
        "positive_l_caked_peak_model_for_display",
        "def fit_joint_qz_peak_sum",
        "def gaussian_sum_qz_model",
        "def add_joint_qz_fit_columns",
        '"joint_fit_density"',
        '"joint_fit_peak_count"',
        'sub.get("joint_peak_density", sub["fit_density"])',
    ):
        assert token in source

    for removed in ("PSEUDO_VOIGT", "_pseudo_voigt", "Fitted pseudo-Voigt"):
        assert removed not in source


def test_joint_qz_fit_keeps_close_peak_valley_low() -> None:
    namespace = _notebook_functions(
        "_center_search_profile",
        "_empty_joint_qz_payload",
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
        "_shared_linear_qz_baseline",
        "_weighted_nonnegative_amplitudes",
        "_aggregate_tail_components",
        "_pearson_vii_component_sum",
        "_component_density_to_sorted",
        "_refine_pearson_vii_components",
        "fit_joint_qz_peak_sum",
        notebook_path=PARALLEL_SCRIPT_PATH,
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
        "BACKGROUND_FIT_BACKEND",
        "BACKGROUND_PROCESS_NUMBA_THREADS",
        "--fit-backend",
        "--process-numba-threads",
        "notebook={notebook_path}",
        "--keep-going",
        "state_count > 1",
    ):
        assert token in source


def test_parallel_background_peak_fits_notebook_uses_process_pool_worker() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    for token in (
        "def normalize_fit_backend(",
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


def test_parallel_background_peak_fits_avoids_windows_spawn_pool_for_top_level_script() -> None:
    namespace = _notebook_functions(
        "normalize_fit_backend",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    normalize_fit_backend = namespace["normalize_fit_backend"]

    assert normalize_fit_backend("process", workers=28, platform_name="nt") == "thread"
    assert normalize_fit_backend("auto", workers=28, platform_name="nt") == "thread"
    assert normalize_fit_backend("process", workers=1, platform_name="nt") == "serial"
    assert normalize_fit_backend("process", workers=28, platform_name="posix") == "process"
    assert normalize_fit_backend("bad", workers=28, platform_name="posix") == "process"


def test_parallel_background_peak_fits_notebook_uses_gaussian_core_lorentzian_tail_model() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

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


def test_parallel_qr_rod_profiles_annotate_hk_locations_with_arrows() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    for token in (
        '"fit_l": int(l_value)',
        '"display_l": float(l_value)',
        "def draw_rod_profile_fit_markers(",
        "def annotate_rod_profile_hk_locations(",
        "line_x: object,",
        "line_y: object,",
        "source = plot_marker_table if marker_source is None else marker_source",
        '"qz_marker"',
        '"branch"',
        '"fit_l"',
        '"display_l"',
        "qz_values_to_l_axis(",
        "ax.scatter(",
        'label="Fit marker"',
        "rod_marker_annotation_label(",
        "line_y_at_markers = np.interp(",
        "ax.annotate(",
        '"arrowstyle": "->"',
        "annotation_clip=True",
        "branch_value=branch_name",
    ):
        assert token in source


def test_parallel_detector_region_final_figure_omits_placed_peak_stars() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    assert "detector_region_cmap = mpl.colormaps[JOURNAL_DETECTOR_CMAP].copy()" in source
    assert 'marker="*"' not in source
    assert '"Placed peak"' not in source
    assert "stars mark accepted placed peaks" not in source


def test_parallel_qr_rod_profile_cache_reuses_same_state_filename(tmp_path: Path) -> None:
    namespace = _notebook_functions(
        "_safe_run_name",
        "qr_rod_profile_cache_path",
        "qr_rod_profile_cache_key",
        "load_qr_rod_profile_cache",
        "write_qr_rod_profile_cache",
        "reset_qr_rod_profile_cache",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    state_path = tmp_path / "sample_state.json"
    state_path.write_text('{"sample": "a"}\n', encoding="utf-8")
    cache_path = namespace["qr_rod_profile_cache_path"](tmp_path, state_path)
    payload = {
        "final_rod_profile_table": pd.DataFrame({"qz_center": [1.0]}),
        "final_marker_table": pd.DataFrame({"qz_marker": [1.0]}),
        "final_rod_component_table": pd.DataFrame({"component_density": [2.0]}),
        "final_peak_edit_cache_key": {"mode": "last_cached"},
    }

    namespace["write_qr_rod_profile_cache"](cache_path, state_path, payload)
    same_name_dir = tmp_path / "rerun"
    same_name_dir.mkdir()
    same_name_state_path = same_name_dir / state_path.name
    same_name_state_path.write_text('{"sample": "changed"}\n', encoding="utf-8")
    loaded = namespace["load_qr_rod_profile_cache"](cache_path, same_name_state_path)

    assert cache_path.name == "sample_state_qr_rod_profile_cache.pkl"
    assert loaded is not None
    assert list(loaded["final_rod_profile_table"]["qz_center"]) == [1.0]

    other_state_path = tmp_path / "other_state.json"
    other_state_path.write_text('{"sample": "a"}\n', encoding="utf-8")
    assert namespace["load_qr_rod_profile_cache"](cache_path, other_state_path) is None
    assert namespace["reset_qr_rod_profile_cache"](cache_path) is True
    assert namespace["reset_qr_rod_profile_cache"](cache_path) is False


def test_parallel_qr_rod_final_fit_cache_hit_skips_joint_refinement() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    cache_path_pos = source.find(
        "QR_ROD_PROFILE_CACHE_PATH = qr_rod_profile_cache_path(OUT_DIR, STATE_PATH)"
    )
    load_pos = source.find(
        "qr_rod_profile_cache = load_qr_rod_profile_cache(QR_ROD_PROFILE_CACHE_PATH, STATE_PATH)"
    )
    hit_pos = source.find(
        "if qr_rod_profile_cache_has_final_fit(qr_rod_profile_cache or {}, qr_rod_peak_edit_key):"
    )
    refine_pos = source.find("rod_profile_table = add_joint_qz_fit_columns(")
    save_pos = source.find(
        "write_qr_rod_profile_cache(QR_ROD_PROFILE_CACHE_PATH, STATE_PATH, qr_rod_profile_cache)"
    )

    assert cache_path_pos >= 0
    assert cache_path_pos < load_pos < hit_pos < refine_pos < save_pos
    assert "RA_SIM_RESET_QR_ROD_PROFILE_CACHE" in source
    assert "RA_SIM_QR_ROD_PEAK_EDITS" in source
    assert "reused final Qr-rod fit cache=" in source
    assert "saved final Qr-rod fit cache=" in source


def test_parallel_qr_rod_final_fit_cache_requires_marker_display_columns() -> None:
    namespace = _notebook_functions(
        "qr_rod_profile_cache_has_final_fit",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    cache_has_final_fit = namespace["qr_rod_profile_cache_has_final_fit"]
    edit_key = {"mode": "last_cached"}
    payload = {
        "final_rod_profile_table": pd.DataFrame({"qz_center": [1.0], "joint_fit_density": [2.0]}),
        "final_marker_table": pd.DataFrame(
            {"qz_marker": [1.0], "fit_l": [1.0], "display_l": [9.5]}
        ),
        "final_rod_component_table": pd.DataFrame({"component_density": [2.0]}),
        "final_peak_edit_cache_key": edit_key,
    }

    assert cache_has_final_fit(payload, edit_key) is True

    stale_payload = dict(payload)
    stale_payload["final_marker_table"] = pd.DataFrame({"qz_marker": [1.0], "l": [1.0]})
    assert cache_has_final_fit(stale_payload, edit_key) is False


def test_parallel_qr_rod_detector_region_specular_support_is_cache_safe() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    assert 'np.asarray(specular_l_marker_table["qz_marker"], dtype=np.float64)' in source
    assert "specular_detector_qz_values = np.asarray(specular_qz_values" not in source


def test_parallel_qr_rod_profile_hk_zero_uses_log_y_axis() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)

    assert "hk0_positive_y = np.asarray([], dtype=np.float64)" in source
    assert 'ax.set_yscale("log")\n        if hk0_positive_y.size:' in source
    assert 'ax.set_title(r"$HK = 0$")' in source


def test_parallel_qr_rod_profile_hk_zero_adds_baseline_to_simulation_only() -> None:
    source = _notebook_source(PARALLEL_SCRIPT_PATH)
    assert "subtract_baseline_from_data=False" in source
    assert (
        "For HK=0 only, plotted `Data` is raw `background_density` and plotted `Simulation` "
        "is `joint_peak_density + joint_linear_baseline_density`."
    ) in source

    namespace = _notebook_functions(
        "normalized_data_simulation_payload",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    normalize = namespace["normalized_data_simulation_payload"]
    measured = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    peak = np.asarray([2.0, 4.0, 6.0], dtype=np.float64)
    baseline = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)

    default_payload = normalize(measured, peak, baseline)
    np.testing.assert_allclose(
        np.asarray(default_payload["data"], dtype=np.float64) * float(default_payload["scale"]),
        measured - baseline,
    )
    np.testing.assert_allclose(
        np.asarray(default_payload["simulation"], dtype=np.float64)
        * float(default_payload["scale"]),
        peak,
    )

    hk0_payload = normalize(
        measured,
        peak,
        baseline,
        subtract_baseline_from_data=False,
    )
    np.testing.assert_allclose(
        np.asarray(hk0_payload["data"], dtype=np.float64) * float(hk0_payload["scale"]),
        measured,
    )
    np.testing.assert_allclose(
        np.asarray(hk0_payload["simulation"], dtype=np.float64) * float(hk0_payload["scale"]),
        peak + baseline,
    )


def test_parallel_qr_rod_profile_hk_arrow_helper_uses_l_axis_markers() -> None:
    namespace = _notebook_functions(
        "hk_display_label",
        "l_tick_label",
        "hk_l_tick_label",
        "rod_marker_annotation_label",
        "l_reference_rows",
        "qz_values_to_l_axis",
        "annotate_rod_profile_hk_locations",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    namespace["plot_marker_table"] = pd.DataFrame(
        [
            {"m": 3, "branch": "+", "qz_marker": 0.5, "l": 1, "fit_l": 1, "display_l": 9.5},
            {"m": 3, "branch": "+", "qz_marker": 1.0, "l": 2, "fit_l": 2},
            {"m": 4, "branch": "+", "qz_marker": 1.5, "l": 3, "fit_l": 3, "display_l": 3},
        ]
    )
    annotate_rod_profile_hk_locations = namespace["annotate_rod_profile_hk_locations"]

    class _Axis:
        def __init__(self) -> None:
            self.annotations: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def get_xlim(self) -> tuple[float, float]:
            return 0.0, 2.5

        def get_ylim(self) -> tuple[float, float]:
            return -0.2, 1.2

        def annotate(self, *args: object, **kwargs: object) -> None:
            self.annotations.append((args, kwargs))

    axis = _Axis()

    annotate_rod_profile_hk_locations(
        axis,
        m_value=3,
        branch_value="+",
        line_x=np.asarray([0.5, 1.0, 2.0], dtype=np.float64),
        line_y=np.asarray([0.1, 0.4, 0.9], dtype=np.float64),
    )

    assert len(axis.annotations) == 2
    labels = [args[0] for args, _kwargs in axis.annotations]
    assert labels == ["(3,9.50)", "(3,2)"]
    xy_positions = [kwargs["xy"] for _args, kwargs in axis.annotations]
    np.testing.assert_allclose(
        np.asarray(xy_positions, dtype=np.float64),
        np.asarray([(1.0, 0.4), (2.0, 0.9)], dtype=np.float64),
    )
    assert all(kwargs["arrowprops"]["arrowstyle"] == "->" for _args, kwargs in axis.annotations)
    assert all(kwargs["annotation_clip"] is True for _args, kwargs in axis.annotations)


def test_parallel_qr_rod_profile_fit_marker_helper_draws_used_positions() -> None:
    namespace = _notebook_functions(
        "l_reference_rows",
        "qz_values_to_l_axis",
        "draw_rod_profile_fit_markers",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    draw_rod_profile_fit_markers = namespace["draw_rod_profile_fit_markers"]
    marker_source = pd.DataFrame(
        [
            {"m": 3, "branch": "+", "qz_marker": 0.5, "l": 1, "fit_l": 1, "display_l": 7.25},
            {"m": 3, "branch": "+", "qz_marker": 1.0, "l": 2, "fit_l": 2, "display_l": 8.25},
        ]
    )

    class _Axis:
        def __init__(self) -> None:
            self.scatters: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def scatter(self, *args: object, **kwargs: object) -> None:
            self.scatters.append((args, kwargs))

    axis = _Axis()
    draw_rod_profile_fit_markers(
        axis,
        m_value=3,
        branch_value="+",
        line_x=np.asarray([1.0, 2.0], dtype=np.float64),
        line_y=np.asarray([0.25, 0.75], dtype=np.float64),
        marker_source=marker_source,
    )

    assert len(axis.scatters) == 1
    args, kwargs = axis.scatters[0]
    np.testing.assert_allclose(np.asarray(args[0], dtype=np.float64), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(args[1], dtype=np.float64), np.asarray([0.25, 0.75]))
    assert kwargs["label"] == "Fit marker"
    assert kwargs["zorder"] > 6


def test_parallel_marker_refinement_uses_local_window_and_keeps_unmatched_markers() -> None:
    namespace = _notebook_functions(
        "_wrapped_delta_deg_scalar_numba",
        "_is_local_peak_top_numba",
        "_find_marker_peak_in_window_numba",
        "_refine_marker_to_local_peak_numba",
        "refine_marker_to_local_peak",
        notebook_path=PARALLEL_SCRIPT_PATH,
    )
    refine_marker_to_local_peak = namespace["refine_marker_to_local_peak"]

    theta_axis = np.linspace(0.0, 8.0, 81, dtype=np.float64)
    phi_axis = np.linspace(-2.0, 2.0, 41, dtype=np.float64)
    theta_grid = np.broadcast_to(theta_axis[None, :], (phi_axis.size, theta_axis.size))
    qz_map = theta_grid + 0.25
    branch_mask = np.ones(qz_map.shape, dtype=bool)
    local_row = int(np.argmin(np.abs(phi_axis - 0.0)))
    local_col = int(np.argmin(np.abs(theta_axis - 1.1)))
    far_col = int(np.argmin(np.abs(theta_axis - 6.0)))

    model = np.zeros(qz_map.shape, dtype=np.float64)
    image = np.ones(qz_map.shape, dtype=np.float64)
    model[local_row, local_col] = 12.0
    model[local_row, far_col] = 200.0
    item = {"params": np.asarray([1.0, 1.0, 0.0, 0.08, 0.08], dtype=np.float64)}
    bg = {
        "theta_axis": theta_axis,
        "phi_axis": phi_axis,
        "caked_peak_model": model,
        "caked_image": image,
    }

    refined = refine_marker_to_local_peak(bg, item, branch_mask, qz_map)

    assert refined is not None
    assert refined[1] == pytest.approx(theta_axis[local_col])
    assert refined[1] != pytest.approx(theta_axis[far_col])

    flat_bg = dict(bg)
    flat_bg["caked_peak_model"] = np.zeros(qz_map.shape, dtype=np.float64)
    flat_bg["caked_image"] = np.ones(qz_map.shape, dtype=np.float64)
    assert refine_marker_to_local_peak(flat_bg, item, branch_mask, qz_map) is None


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
