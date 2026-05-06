import ast
from concurrent.futures import ProcessPoolExecutor
import json
import os
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
RUNNER_PATH = Path("scripts/diagnostics/run_all_background_peak_fits.py")


def _notebook_source(notebook_path: Path = NOTEBOOK_PATH) -> str:
    if not notebook_path.exists():
        pytest.skip(f"{notebook_path} is not present in this checkout")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _notebook_functions(*names: str, notebook_path: Path = NOTEBOOK_PATH) -> dict[str, object]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    wanted = set(names)
    selected: list[ast.FunctionDef] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=f"{notebook_path}:cell{index}")
        selected.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
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
    }
    exec(compile(module, str(notebook_path), "exec"), namespace)
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
    source = _notebook_source(PARALLEL_NOTEBOOK_PATH)

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
    source = _notebook_source(PARALLEL_NOTEBOOK_PATH)

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


def test_parallel_qr_rod_profiles_annotate_hk_locations_with_arrows() -> None:
    source = _notebook_source(PARALLEL_NOTEBOOK_PATH)

    for token in (
        "def annotate_rod_profile_hk_locations(",
        "line_x: object,",
        "line_y: object,",
        "source = plot_marker_table if marker_source is None else marker_source",
        '"qz_marker"',
        '"branch"',
        "qz_values_to_l_axis(",
        "line_y_at_markers = np.interp(",
        "hk_display_label(m_value)",
        "ax.annotate(",
        '"arrowstyle": "->"',
        "annotation_clip=True",
        "branch_value=branch_name",
    ):
        assert token in source


def test_parallel_detector_region_final_figure_omits_placed_peak_stars() -> None:
    source = _notebook_source(PARALLEL_NOTEBOOK_PATH)

    assert "detector_region_cmap = mpl.colormaps[JOURNAL_DETECTOR_CMAP].copy()" in source
    assert 'marker="*"' not in source
    assert '"Placed peak"' not in source
    assert "stars mark accepted placed peaks" not in source


def test_parallel_qr_rod_profile_hk_zero_uses_log_y_axis() -> None:
    source = _notebook_source(PARALLEL_NOTEBOOK_PATH)

    assert "hk0_positive_y = np.asarray([], dtype=np.float64)" in source
    assert "ax.set_yscale(\"log\")\n        if hk0_positive_y.size:" in source
    assert "ax.set_title(r\"$HK = 0$\")" in source


def test_parallel_qr_rod_profile_hk_arrow_helper_uses_l_axis_markers() -> None:
    namespace = _notebook_functions(
        "hk_display_label",
        "l_reference_rows",
        "qz_values_to_l_axis",
        "annotate_rod_profile_hk_locations",
        notebook_path=PARALLEL_NOTEBOOK_PATH,
    )
    namespace["plot_marker_table"] = pd.DataFrame(
        [
            {"m": 3, "branch": "+", "qz_marker": 0.5, "l": 1},
            {"m": 3, "branch": "+", "qz_marker": 1.0, "l": 2},
            {"m": 4, "branch": "+", "qz_marker": 1.5, "l": 3},
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
    assert labels == [r"$HK = 3$", r"$HK = 3$"]
    xy_positions = [kwargs["xy"] for _args, kwargs in axis.annotations]
    np.testing.assert_allclose(
        np.asarray(xy_positions, dtype=np.float64),
        np.asarray([(1.0, 0.4), (2.0, 0.9)], dtype=np.float64),
    )
    assert all(kwargs["arrowprops"]["arrowstyle"] == "->" for _args, kwargs in axis.annotations)
    assert all(kwargs["annotation_clip"] is True for _args, kwargs in axis.annotations)


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
