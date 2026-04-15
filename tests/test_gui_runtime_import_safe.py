import ast
import importlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


RUNTIME_MODULE_NAME = "ra_sim.gui.runtime"
RUNTIME_IMPL_MODULE_NAME = "ra_sim.gui._runtime_impl"
PACKAGE_RUNTIME_IMPL_MODULE_NAME = "ra_sim.gui._runtime.runtime_impl"
RUNTIME_IMPL_WRAPPER_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "_runtime" / "runtime_impl.py"
)
RUNTIME_SESSION_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "_runtime" / "runtime_session.py"
)
FILE_PARSING_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "io" / "file_parsing.py"
)
CLI_SOURCE_PATH = Path(__file__).resolve().parent.parent / "ra_sim" / "cli.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_SOURCE_ROOT = REPO_ROOT / "ra_sim" / "gui"
RA_SIM_SOURCE_ROOT = REPO_ROOT / "ra_sim"
OPTIMIZATION_MOSAIC_PROFILES_SOURCE_PATH = (
    REPO_ROOT / "ra_sim" / "fitting" / "optimization_mosaic_profiles.py"
)
RAW_SOURCE_PEAK_READ_ALLOWLIST = {
    GUI_SOURCE_ROOT / "_runtime" / "runtime_session.py",
    GUI_SOURCE_ROOT / "geometry_fit.py",
    GUI_SOURCE_ROOT / "manual_geometry.py",
}
TRUST_FIELD_ASSIGNMENT_ALLOWLIST = {
    GUI_SOURCE_ROOT / "manual_geometry.py",
}


def test_runtime_import_is_lazy() -> None:
    previous_runtime = sys.modules.pop(RUNTIME_MODULE_NAME, None)
    previous_impl = sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)

    try:
        runtime = importlib.import_module(RUNTIME_MODULE_NAME)

        assert runtime.__name__ == RUNTIME_MODULE_NAME
        assert RUNTIME_IMPL_MODULE_NAME not in sys.modules
        assert callable(runtime.main)
    finally:
        sys.modules.pop(RUNTIME_MODULE_NAME, None)
        sys.modules.pop(RUNTIME_IMPL_MODULE_NAME, None)
        if previous_runtime is not None:
            sys.modules[RUNTIME_MODULE_NAME] = previous_runtime
        if previous_impl is not None:
            sys.modules[RUNTIME_IMPL_MODULE_NAME] = previous_impl


def test_runtime_main_loads_impl_and_forwards_arguments(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)
    calls: list[dict[str, object]] = []

    fake_impl = SimpleNamespace(
        write_excel=None,
        main=lambda **kwargs: calls.append(dict(kwargs)),
    )

    monkeypatch.setattr(runtime, "_load_runtime_module", lambda: fake_impl)
    runtime.write_excel = False

    runtime.main(
        write_excel_flag=True,
        startup_mode="simulation",
        calibrant_bundle="bundle.npz",
    )

    assert fake_impl.write_excel is True
    assert calls == [
        {
            "write_excel_flag": True,
            "startup_mode": "simulation",
            "calibrant_bundle": "bundle.npz",
        }
    ]


def test_runtime_dunder_attribute_raises_without_loading(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)

    monkeypatch.setattr(
        runtime,
        "_load_runtime_module",
        lambda: (_ for _ in ()).throw(AssertionError("impl should not be imported")),
    )

    with pytest.raises(AttributeError):
        _ = runtime.__runtime_test_guard__


def test_runtime_dir_is_lazy() -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)

    available = dir(runtime)

    assert "main" in available
    assert "write_excel" in available


def test_runtime_unknown_attr_forwards_to_impl(monkeypatch) -> None:
    runtime = importlib.import_module(RUNTIME_MODULE_NAME)
    fake_impl = SimpleNamespace(test_value=123)

    monkeypatch.setattr(runtime, "_load_runtime_module", lambda: fake_impl)

    assert runtime.test_value == 123


def test_runtime_impl_source_compiles() -> None:
    py_compile.compile(str(RUNTIME_IMPL_WRAPPER_SOURCE_PATH), doraise=True)
    py_compile.compile(str(RUNTIME_SESSION_SOURCE_PATH), doraise=True)


def test_runtime_impl_wrapper_import_is_lazy() -> None:
    previous_wrapper = sys.modules.pop(PACKAGE_RUNTIME_IMPL_MODULE_NAME, None)
    previous_session = sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

    try:
        wrapper = importlib.import_module(PACKAGE_RUNTIME_IMPL_MODULE_NAME)

        assert wrapper.__name__ == PACKAGE_RUNTIME_IMPL_MODULE_NAME
        assert "ra_sim.gui._runtime.runtime_session" not in sys.modules
        assert callable(wrapper.main)
    finally:
        sys.modules.pop(PACKAGE_RUNTIME_IMPL_MODULE_NAME, None)
        sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)
        if previous_wrapper is not None:
            sys.modules[PACKAGE_RUNTIME_IMPL_MODULE_NAME] = previous_wrapper
        if previous_session is not None:
            sys.modules["ra_sim.gui._runtime.runtime_session"] = previous_session


def test_runtime_session_defers_scipy_gui_modules_until_first_use() -> None:
    script = """
import importlib
import json
import sys
from types import SimpleNamespace

lazy_modules = [
    "ra_sim.gui.analysis_peak_tools",
    "ra_sim.gui.ordered_structure_fit",
]
for name in lazy_modules:
    sys.modules.pop(name, None)
sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
after_import = {name: (name in sys.modules) for name in lazy_modules}

runtime_session.analysis_peak_selection_state = SimpleNamespace(
    radial_fit_results=[],
    azimuth_fit_results=[],
)
runtime_session._ANALYSIS_PEAK_EMPTY_RESULTS_TEXT = "Fit results will appear here."
original_peak_getter = runtime_session._get_analysis_peak_tools_module
runtime_session._get_analysis_peak_tools_module = (
    lambda: (_ for _ in ()).throw(AssertionError("peak tools imported too early"))
)
empty_text = runtime_session._analysis_peak_fit_results_text()
runtime_session._get_analysis_peak_tools_module = original_peak_getter
after_empty_text = {name: (name in sys.modules) for name in lazy_modules}

peak_module = runtime_session._get_analysis_peak_tools_module()
ordered_module = runtime_session._get_ordered_structure_fit_module()
after_getters = {name: (name in sys.modules) for name in lazy_modules}

print(
    json.dumps(
        {
            "after_import": after_import,
            "empty_text": empty_text,
            "after_empty_text": after_empty_text,
            "peak_module": getattr(peak_module, "__name__", ""),
            "ordered_module": getattr(ordered_module, "__name__", ""),
            "after_getters": after_getters,
        }
    )
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
    )
    payload = json.loads(result.stdout.strip())

    assert payload["after_import"] == {
        "ra_sim.gui.analysis_peak_tools": False,
        "ra_sim.gui.ordered_structure_fit": False,
    }
    assert payload["empty_text"] == "Fit results will appear here."
    assert payload["after_empty_text"] == {
        "ra_sim.gui.analysis_peak_tools": False,
        "ra_sim.gui.ordered_structure_fit": False,
    }
    assert payload["peak_module"] == "ra_sim.gui.analysis_peak_tools"
    assert payload["ordered_module"] == "ra_sim.gui.ordered_structure_fit"
    assert payload["after_getters"] == {
        "ra_sim.gui.analysis_peak_tools": True,
        "ra_sim.gui.ordered_structure_fit": True,
    }


def test_runtime_impl_prompts_from_root_only_before_full_runtime_bootstrap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    main_start = source.index('def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):')
    prompt_line = "resolved_mode = gui_bootstrap.choose_startup_mode_dialog(root)"
    prompt_index = source.index(prompt_line, main_start)
    ensure_root_index = source.index("ensure_runtime_root_initialized()", main_start)

    assert ensure_root_index < prompt_index


def test_runtime_impl_runtime_context_builders_gate_staged_initializers() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    state_start = source.index("def build_runtime_state_context() -> RuntimeContext:")
    window_start = source.index("def build_runtime_window_context(context: RuntimeContext) -> RuntimeContext:")
    plot_start = source.index("def build_runtime_plot_context(context: RuntimeContext) -> RuntimeContext:")
    controls_start = source.index("def build_runtime_controls_context(context: RuntimeContext) -> RuntimeContext:")
    benchmark_start = source.index("_STARTUP_BENCHMARK_ENABLED = str(")

    state_block = source[state_start:window_start]
    window_block = source[window_start:plot_start]
    plot_block = source[plot_start:controls_start]
    controls_block = source[controls_start:benchmark_start]

    assert "ensure_runtime_state_initialized()" in state_block
    assert "ensure_runtime_shell_initialized()" in window_block
    assert "ensure_runtime_plot_initialized()" in plot_block
    assert "ensure_runtime_controls_initialized()" in controls_block


def test_runtime_impl_keeps_default_bound_constants_eager() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert source.index('QR_CYLINDER_DISPLAY_MODE_OFF = "Off"') < source.index(
        "def _qr_cylinder_display_mode("
    )
    assert source.index("DEFAULT_ANALYSIS_RADIAL_BINS = 1000") < source.index("def caking(")
    assert source.index("INITIAL_PREVIEW_MAX_SAMPLES = 24") < source.index(
        "def _build_preview_simulation_job("
    )


def test_runtime_impl_gates_raster_projection_helpers_until_controls_stage() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _sync_primary_raster_geometry(")
    helper_end = source.index("def _maybe_refresh_run_status_bar()", helper_start)
    helper_block = source[helper_start:helper_end]

    assert '_RUNTIME_CONTROLS_INITIALIZED", False' in helper_block
    assert "if controls_ready" in helper_block
    assert "apply_projection(artist)" in helper_block


def test_runtime_impl_uses_tiny_startup_placeholder_rasters() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "global_image_buffer = np.zeros((1, 1), dtype=np.float32)" in source
    assert "np.zeros_like(global_image_buffer)" in source
    assert "_ensure_global_image_buffer_shape(simulation_runtime_state.unscaled_image)" in source


def test_runtime_impl_expands_global_image_buffer_for_first_real_frame() -> None:
    resize_buffer = _load_runtime_session_function("_ensure_global_image_buffer_shape")
    resize_buffer.__globals__["global_image_buffer"] = np.zeros((1, 1), dtype=np.float32)
    source = np.arange(12, dtype=np.float64).reshape(3, 4)

    resized = resize_buffer(source)

    assert resized.shape == source.shape
    assert resized.dtype == np.float64
    assert resize_buffer.__globals__["global_image_buffer"] is resized
    assert resize_buffer(source) is resized


def test_runtime_impl_lazy_allocates_job_result_images() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    block_start = source.index("def _run_simulation_generation_job(")
    block_end = source.index("def _submit_async_simulation_job(", block_start)
    block = source[block_start:block_end]

    primary_run_index = block.index('if bool(job["run_primary"]):')
    secondary_run_index = block.index('if bool(job["run_secondary"]):')
    img1_fallback_index = block.index("if img1 is None:")
    img2_fallback_index = block.index("if img2 is None:")

    assert block.index("img1 = None") < primary_run_index < img1_fallback_index
    assert block.index("img2 = None") < secondary_run_index < img2_fallback_index
    assert "if img1 is None:" in block
    assert "if img2 is None:" in block


def test_runtime_impl_routes_legacy_fit_logs_through_debug_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("RA_SIM_DEBUG", "0")' not in source
    assert "if gui_geometry_fit.geometry_fit_log_files_enabled():" in source
    assert "if mosaic_fit_log_files_enabled():" in source
    assert "log_file={log_path if log_path is not None else 'disabled'}" in source


def test_runtime_impl_uses_fast_exact_cake_integrator_for_analysis() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.simulation.exact_cake import start_exact_cake_numba_warmup_in_background" in source
    assert "start_exact_cake_geometry_warmup_in_background" in source
    assert "from ra_sim.simulation.exact_cake_portable import (" in source
    assert "FastAzimuthalIntegrator," in source
    assert "start_exact_cake_geometry_warmup_in_background," in source
    assert "_AZIMUTHAL_INTEGRATOR_CLS = FastAzimuthalIntegrator" in source
    assert "start_forward_simulation_numba_warmup_in_background" in source
    assert "def _schedule_exact_cake_numba_warmup_once() -> None:" in source
    assert "def _schedule_forward_simulation_numba_warmup_once() -> None:" in source
    assert "simulation_runtime_state.exact_cake_numba_warmup_scheduled" in source
    assert "simulation_runtime_state.forward_simulation_numba_warmup_scheduled" in source
    assert "root.after_idle(start_exact_cake_numba_warmup_in_background)" in source
    assert (
        "root.after_idle(start_forward_simulation_numba_warmup_in_background)" in source
    )


def test_runtime_impl_exact_cake_cache_signature_uses_distance_center_and_wavelength() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    signature_comment = (
        "distance, detector center, or the fundamental wavelength changes. Tilt\n"
        "    # updates intentionally do not flush the detector-map or LUT caches"
    )
    signature_block = """sig = _caked_geometry_cache_signature(
        corto_det_up,
        center_x_up,
        center_y_up,
        wave_m,
    )"""

    assert signature_comment in source
    assert signature_block in source
    assert "Gamma_updated" not in signature_block
    assert "gamma_updated" not in signature_block
    assert "center_x_up" in signature_block
    assert "center_y_up" in signature_block
    assert "wave_m" in signature_block
    assert "wavelength=wave_m," in source


def test_runtime_impl_warms_live_exact_cake_geometry_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "start_exact_cake_geometry_warmup_in_background(" in source


def test_runtime_impl_geometry_fit_caking_reuses_signature_by_distance_center_and_wavelength() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "simulation_runtime_state.geometry_fit_caking_ai_cache = {}" in source
    assert "worker_geometry_fit_caking_sig" in source
    assert "requested_sig = _caked_geometry_cache_signature(" in source
    assert 'params_local.get("corto_detector", np.nan),' in source
    assert 'center_value[0] if center_value.size > 0 else np.nan,' in source
    assert 'center_value[1] if center_value.size > 1 else np.nan,' in source
    assert 'params_local.get("lambda", np.nan),' in source
    assert 'persistent_cache = getattr(' in source
    assert '"geometry_fit_caking_ai_cache",' in source
    assert 'persistent_cache.get("sig") == requested_sig' in source
    assert 'simulation_runtime_state.geometry_fit_caking_ai_cache = {' in source
    assert "and worker_geometry_fit_caking_sig == requested_sig" in source
    assert "np.asarray(simulation_runtime_state.unscaled_image).shape" in source
    assert "npt_rad=DEFAULT_ANALYSIS_RADIAL_BINS" in source
    assert "npt_azim=DEFAULT_ANALYSIS_AZIMUTH_BINS" in source


def test_runtime_caking_does_not_call_exact_integrator(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    exact_cake = importlib.import_module("ra_sim.simulation.exact_cake")
    exact_cake_portable = importlib.import_module("ra_sim.simulation.exact_cake_portable")

    def _fail(*_args, **_kwargs):
        raise AssertionError("live caking should not call exact integration")

    monkeypatch.setattr(
        runtime_session,
        "integrate_detector_to_cake_exact",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(exact_cake, "integrate_detector_to_cake_exact", _fail)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fail)

    calls: list[dict[str, object]] = []

    class _FakeAI:
        def integrate2d(self, data, **kwargs):
            calls.append(
                {
                    "data_shape": tuple(np.asarray(data).shape),
                    "kwargs": dict(kwargs),
                }
            )
            return "lut-result"

    result = runtime_session.caking(
        np.ones((3, 4), dtype=float),
        _FakeAI(),
        npt_rad=32,
        npt_azim=64,
        rows=np.array([0, 1], dtype=np.int32),
        cols=np.array([2, 3], dtype=np.int32),
    )

    assert result == "lut-result"
    assert len(calls) == 1
    assert calls[0]["data_shape"] == (3, 4)
    assert calls[0]["kwargs"]["npt_rad"] == 32
    assert calls[0]["kwargs"]["npt_azim"] == 64
    assert calls[0]["kwargs"]["correctSolidAngle"] is True
    assert calls[0]["kwargs"]["method"] == "lut"
    assert calls[0]["kwargs"]["unit"] == "2th_deg"
    np.testing.assert_array_equal(
        calls[0]["kwargs"]["rows"],
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        calls[0]["kwargs"]["cols"],
        np.array([2, 3], dtype=np.int32),
    )


def test_cli_routes_mosaic_fit_logs_through_debug_controls() -> None:
    source = CLI_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.debug_controls import mosaic_fit_log_files_enabled" in source
    assert "if mosaic_fit_log_files_enabled():" in source
    assert '"log_path": str(mosaic_log_path) if mosaic_log_path is not None else None' in source


def test_runtime_impl_attaches_background_theta_trace_after_theta_var_assignment() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    theta_assignment = (
        "theta_initial_var = beam_mosaic_parameter_sliders_view_state.theta_initial_var"
    )
    trace_call = "_attach_live_theta_background_theta_trace(theta_initial_var)"

    assert theta_assignment in source
    assert trace_call in source
    assert source.index(theta_assignment) < source.index(trace_call)


def test_runtime_impl_uses_cached_caking_results_for_range_refreshes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "range_refresh_requires_pending_analysis_result(" in source


def test_runtime_impl_disables_live_drag_preview_degradation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "LIVE_DRAG_PREVIEW_ENABLED = False" in source
    assert "if _live_interaction_active() and LIVE_DRAG_PREVIEW_ENABLED:" in source
    assert "LIVE_DRAG_PREVIEW_ENABLED\n        and PREVIEW_CALCULATIONS_ENABLED" in source


def test_runtime_impl_preserves_wrapped_phi_ranges_for_detector_drags() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "draw_idle_factory=lambda: (" in source
    assert "_request_main_canvas_redraw" in source
    assert "phi_min = float(phi_min_var.get())" in source
    assert "phi_max = float(phi_max_var.get())" in source
    assert "gui_integration_range_drag.detector_phi_mask(" in source
    assert "if phi_max < phi_min and azimuth_sub.size:" in source


def test_runtime_impl_tracks_detector_source_geometry_for_projected_rasters() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '_MAIN_RASTER_SOURCE_EXTENT_ATTR = "_ra_sim_source_extent"' in source
    assert '_MAIN_RASTER_SOURCE_ORIGIN_ATTR = "_ra_sim_source_origin"' in source
    assert 'image_geometry = ("upper", detector_extent)' in source
    assert 'overlay_geometry = ("upper", detector_extent)' in source
    assert "setattr(artist, _MAIN_RASTER_SOURCE_EXTENT_ATTR, normalized_extent)" in source
    assert "setattr(artist, _MAIN_RASTER_SOURCE_ORIGIN_ATTR, str(origin))" in source
    assert "store_geometry(artist, origin=origin, extent=extent)" in source


def test_runtime_impl_overlay_refresh_copies_image_source_geometry() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "_image_source, image_extent, image_origin = _primary_raster_source_payload(" in source
    assert "if image_extent is not None:" in source
    assert "origin=image_origin," in source
    assert "extent=image_extent," in source
    assert "_apply_projected_primary_raster_to_artist(overlay_artist)" in source


def test_runtime_impl_routes_main_figure_chrome_through_shared_helper() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.gui import main_figure_chrome as gui_main_figure_chrome" in source
    assert "gui_main_figure_chrome.configure_matplotlib_canvas_widget(" in source
    assert "gui_main_figure_chrome.configure_main_figure_layout(" in source
    assert "gui_main_figure_chrome.apply_main_figure_axes_chrome(" in source
    assert "axes_visible=bool(analysis_space_display_available)" in source
    assert "axes_visible=bool(show_caked_image)" not in source
    assert "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=True)" in source
    assert "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=False)" in source
    assert "ax.set_title('Simulated Diffraction Pattern')" not in source


def test_runtime_impl_falls_back_to_detector_image_when_caked_cache_is_missing() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "active_view_mode = _resolved_primary_analysis_display_mode()" in source
    assert "show_analysis_image = show_caked_image or show_q_space_image" in source
    assert "if not show_analysis_image:" in source
    assert "_store_primary_raster_source(image_display, global_image_buffer)" in source
    assert '_sync_primary_raster_geometry(view_mode="detector")' in source


def test_runtime_impl_restores_caked_payload_when_view_returns_to_caked() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _restore_caked_display_payload_from_cached_results(" in source
    assert "missing_caked_payload = bool(" in source
    assert "_restore_caked_display_payload_from_cached_results(" in source
    assert "simulation_runtime_state.last_caked_image_unscaled is None" in source
    assert "simulation_runtime_state.last_caked_extent is None" in source


def test_runtime_impl_preserves_primary_axis_limits_across_same_mode_redraws() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "previous_primary_view_mode = _current_primary_figure_mode()" in source
    assert "preserved_primary_limits = gui_canvas_interactions.capture_axis_limits(ax)" in source
    assert "gui_canvas_interactions.restore_axis_view(" in source
    assert "preserve=(previous_primary_view_mode == analysis_space_display_mode)" in source
    assert 'preserve=(previous_primary_view_mode == "detector")' in source


def test_runtime_impl_reuses_cached_hit_tables_for_optics_and_mosaic_only_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _cached_hit_tables_reusable(" in source
    assert "include_mosaic_shape: bool = True" in source
    assert "if include_mosaic_shape:" in source
    assert "requested_hit_table_sig = _simulation_signature_base(" in source
    assert "optics_mode_component=0," in source
    assert "include_mosaic_shape=False," in source
    assert "collect_hit_tables_for_job = bool(" in source
    assert "and not _cached_hit_tables_reusable(" in source
    assert "\"collected_hit_tables\": bool(job[\"collect_hit_tables\"])," in source
    assert "intersection_cache_to_hit_tables," in source
    assert "def _resolved_peak_table_payload(" in source
    assert "primary_peak_tables = _resolved_peak_table_payload(cache1, raw_hit_tables1)" in source
    assert "\"primary_max_positions\": list(primary_peak_tables)," in source
    assert "stored_primary_intersection_cache = _copy_intersection_cache_tables(" in source
    assert "stored_secondary_intersection_cache = _copy_intersection_cache_tables(" in source
    assert "stored_primary_peak_table_lattice = list(" in source
    assert "stored_secondary_peak_table_lattice = list(" in source
    assert "or \"primary_intersection_cache\" in result" in source
    assert "simulation_runtime_state.last_sim_signature = new_sim_image_sig" in source
    assert "need_hit_table_refresh = bool(" in source
    assert "do_update_refresh_hit_tables_in_background" in source
    assert 'progress_label.config(text="Refreshing peak tables in background...")' in source


def test_runtime_impl_supports_incremental_sf_prune_primary_cache_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "primary_contribution_cache_signature" in source
    assert "primary_requested_contribution_keys" in source
    assert "resolve_incremental_sf_prune_action(" in source
    assert 'job_kind="primary_fill"' in source
    assert "_rematerialize_primary_cache_artifacts(" in source
    assert "do_update_extend_primary_cache_in_background" in source
    assert 'text="Extending primary cache in background..."' in source


def test_runtime_impl_routes_optional_runtime_caches_through_retention_policy() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.debug_controls import (" in source
    assert "retain_optional_cache," in source
    assert '"primary_contribution"' in source
    assert '"source_snapshots"' in source
    assert '"caking"' in source
    assert '"manual_pick"' in source
    assert '"geometry_fit_dataset"' in source
    assert '"peak_overlay"' in source


def test_runtime_impl_uses_geometry_manual_state_name_for_manual_pairs() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "manual_geometry_state" not in source
    assert 'globals().get("geometry_manual_state")' in source


def test_runtime_impl_keeps_manual_pick_cache_restores_cache_only() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'allow_source_snapshot_rebuild = bool(lookup_context == "geometry_fit_dataset")' in source
    assert 'if allow_source_snapshot_rebuild:' in source


def test_runtime_impl_worker_geometry_fit_rebuilds_source_rows_on_demand() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prebuild_required_background_caches() -> None:")
    helper_end = source.index("worker_manual_dataset_bindings =", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "def _rebuild_source_rows_for_background_worker(" in source
    assert "geometry_manual_rebuild_source_rows_for_background=(" in source
    assert "_rebuild_source_rows_for_background_worker" in source
    assert "raise RuntimeError(" not in helper_source


def test_runtime_impl_keeps_qr_overlay_live_during_background_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _clear_deferred_overlays(*, clear_qr_overlay: bool = True) -> None:" in source
    assert "if _live_interaction_active():" in source
    assert "qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)" in source
    assert "_clear_deferred_overlays(clear_qr_overlay=False)" in source


def test_runtime_impl_uses_roi_preview_display_sources_for_main_redraw() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "display_primary_source, display_secondary_source = (" in source
    assert "_geometry_fit_caked_roi_preview_display_sources(" in source
    assert "simulation_image=scaled_analysis_for_limits," in source
    assert "simulation_image=global_image_buffer," in source


def test_runtime_impl_hides_aux_artists_while_roi_preview_is_enabled() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if _current_geometry_fit_caked_roi_preview_enabled():" in source
    assert "_hide_geometry_fit_caked_roi_preview_aux_artists()" in source
    assert "_request_overlay_canvas_redraw(force=True)" in source
    assert "visible and _current_geometry_fit_caked_roi_preview_enabled()" in source


def test_runtime_impl_invalidates_qr_overlay_before_view_mode_toggles() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _invalidate_qr_cylinder_overlay_view_state(*, clear_artists: bool) -> None:" in source
    assert "gui_qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(" in source
    assert "def toggle_caked_2d() -> None:" in source
    assert "_invalidate_qr_cylinder_overlay_view_state(clear_artists=True)" in source
    assert "def _apply_main_caked_view_toggle() -> None:" in source
    assert "_apply_main_caked_view_toggle()" in source


def test_runtime_impl_keeps_detector_and_caked_intersection_caches_separate() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    restore_start = source.index("def _restore_caked_display_payload_from_cached_results(")
    restore_end = source.index("def _run_analysis_job(", restore_start)
    restore_source = source[restore_start:restore_end]
    apply_start = source.index("def _apply_ready_analysis_result(result: dict[str, object]) -> None:")
    apply_end = source.index("def schedule_update():", apply_start)
    apply_source = source[apply_start:apply_end]

    assert "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache" in restore_source
    assert "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache" not in restore_source
    assert "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache" in apply_source
    assert "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache" not in apply_source


def test_runtime_live_caked_projection_helper_uses_bound_callback(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    def _fail(*_args, **_kwargs):
        raise AssertionError("live caked helper should use bound callback only")

    callback_calls: list[tuple[float, float]] = []

    def _record(col: float, row: float):
        callback_calls.append((float(col), float(row)))
        return (12.0, 34.0)

    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_caked_display_coords",
        _record,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_detector_angular_maps",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )

    assert runtime_session._native_detector_coords_to_live_caked_coords(1.25, 2.5) == (
        12.0,
        34.0,
    )
    assert callback_calls == [(1.25, 2.5)]


def test_runtime_geometry_fit_roi_projector_does_not_call_legacy_projection_paths(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 6),
        radial_deg=np.array([1.0, 2.0], dtype=float),
        raw_azimuth_deg=np.array([-10.0, 10.0], dtype=float),
        gui_azimuth_deg=np.array([-10.0, 10.0], dtype=float),
        lut=SimpleNamespace(),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("ROI projector should not call legacy projection paths")

    bundle_calls: list[object] = []

    def _record_caked_point_to_detector_pixel(
        ai,
        detector_shape,
        radial_deg,
        azimuth_deg,
        two_theta_deg,
        phi_deg,
        *,
        transform_bundle=None,
    ):
        del ai, detector_shape, radial_deg, azimuth_deg, two_theta_deg, phi_deg
        bundle_calls.append(transform_bundle)
        return (12.0, 34.0)

    monkeypatch.setattr(
        runtime_session,
        "caked_point_to_detector_pixel",
        _record_caked_point_to_detector_pixel,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_detector_angular_maps",
        _fail,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )

    projector = runtime_session._geometry_fit_caked_roi_fit_space_to_detector_point(
        detector_shape=(6, 6),
        radial_axis=None,
        azimuth_axis=None,
        ai=None,
        transform_bundle=bundle,
    )

    assert projector is not None
    assert projector(1.5, -10.0) == (12.0, 34.0)
    assert bundle_calls == [bundle]


def test_runtime_impl_worker_roi_precomputes_projection_context_before_selection() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    store_start = source.index("def _store_worker_caked_view_for_background(")
    store_end = source.index("def _build_source_rows_for_rebuild(", store_start)
    store_source = source[store_start:store_end]

    helper_idx = store_source.index("_geometry_fit_worker_caked_projection_view(")
    selection_idx = store_source.index(
        "roi_selection = gui_geometry_fit.build_geometry_fit_caked_roi_selection("
    )

    assert helper_idx < selection_idx
    assert "background_caked_view.update(precomputed_caked_view)" in store_source
    assert 'analysis_bins = job_data.get("analysis_bins")' in store_source
    assert "npt_rad=precompute_npt_rad" in store_source
    assert "npt_azim=precompute_npt_azim" in store_source


def test_runtime_worker_caked_projection_view_builds_same_axes_metadata(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 8),
        radial_deg=np.array([0.5, 1.5], dtype=float),
        raw_azimuth_deg=np.array([-135.0, 45.0], dtype=float),
        gui_azimuth_deg=np.array([-135.0, 45.0], dtype=float),
        lut=SimpleNamespace(),
    )
    calls: list[tuple[object, ...]] = []

    def _fake_two_theta_max(detector_shape, geometry):
        calls.append(("two_theta_max", tuple(detector_shape), geometry))
        return 12.0

    def _fake_build_angle_axes(
        *,
        npt_rad,
        npt_azim,
        tth_min_deg,
        tth_max_deg,
        azimuth_min_deg,
        azimuth_max_deg,
    ):
        calls.append(
            (
                "axes",
                npt_rad,
                npt_azim,
                tth_min_deg,
                tth_max_deg,
                azimuth_min_deg,
                azimuth_max_deg,
            )
        )
        return (
            np.array([0.5, 1.5], dtype=float),
            np.array([-135.0, 45.0], dtype=float),
        )

    def _fake_build_bundle(ai_arg, detector_shape, radial_deg, raw_azimuth_deg):
        calls.append(
            (
                "bundle",
                ai_arg,
                tuple(detector_shape),
                tuple(np.asarray(radial_deg, dtype=float)),
                tuple(np.asarray(raw_azimuth_deg, dtype=float)),
            )
        )
        return bundle

    monkeypatch.setattr(runtime_session, "detector_two_theta_max_deg", _fake_two_theta_max)
    monkeypatch.setattr(runtime_session, "build_angle_axes", _fake_build_angle_axes)
    monkeypatch.setattr(runtime_session, "build_cake_transform_bundle", _fake_build_bundle)

    projection_view = runtime_session._geometry_fit_worker_caked_projection_view(
        detector_shape=(6, 8),
        ai=ai,
        npt_rad=17,
        npt_azim=19,
    )

    assert projection_view is not None
    np.testing.assert_array_equal(
        projection_view["radial_axis"],
        np.array([0.5, 1.5], dtype=float),
    )
    np.testing.assert_array_equal(
        projection_view["raw_azimuth_axis"],
        np.array([-135.0, 45.0], dtype=float),
    )
    expected_gui = np.asarray(
        runtime_session.raw_phi_to_gui_phi(np.array([-135.0, 45.0], dtype=float)),
        dtype=float,
    )
    expected_order = np.asarray(np.argsort(expected_gui, kind="stable"), dtype=np.int32)
    np.testing.assert_array_equal(
        projection_view["raw_to_gui_row_permutation"],
        expected_order,
    )
    np.testing.assert_allclose(
        projection_view["azimuth_axis"],
        expected_gui[expected_order],
    )
    assert projection_view["transform_bundle"] is bundle
    assert calls[0][:2] == ("two_theta_max", (6, 8))
    assert calls[1] == (
        "axes",
        17,
        19,
        0.0,
        12.0,
        -180.0,
        180.0,
    )
    assert calls[2] == (
        "bundle",
        ai,
        (6, 8),
        (0.5, 1.5),
        (-135.0, 45.0),
    )


def test_runtime_impl_geometry_fit_async_job_captures_analysis_bins() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _build_geometry_fit_async_job(")
    helper_end = source.index("def _handle_geometry_fit_worker_event(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert '"analysis_bins": analysis_bins' in helper_source
    assert '"npt_rad": int(analysis_bins[0])' in helper_source
    assert '"npt_azim": int(analysis_bins[1])' in helper_source


def test_runtime_impl_prepare_caked_payload_keeps_canonical_transform_metadata() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prepare_caked_display_payload(")
    helper_end = source.index("def _prepare_q_space_display_payload(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert '"raw_azimuth_axis": np.asarray(raw_azimuth_axis, dtype=float)' in helper_source
    assert '"raw_to_gui_row_permutation": np.asarray(' in helper_source
    assert '"transform_bundle": transform_bundle' in helper_source


def test_runtime_impl_hkl_pick_disarms_manual_geometry_and_preview_modes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "deactivate_conflicting_modes_factory=lambda: (" in source
    assert "_set_geometry_manual_pick_mode(False)" in source
    assert "_set_geometry_preview_exclude_mode(False)" in source


def test_runtime_impl_hkl_pick_pauses_fast_viewer_runtime() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "hkl_pick_armed_factory=lambda: getattr(" in source
    assert 'globals().get("peak_selection_state")' in source
    assert 'def _handle_hkl_pick_mode_changed(_armed: bool) -> None:' in source
    assert 'globals().get("_refresh_fast_viewer_runtime_mode")' in source
    assert "refresh_fast_viewer(announce=False)" in source
    assert "on_hkl_pick_mode_changed_factory=lambda: _handle_hkl_pick_mode_changed" in source


def test_runtime_impl_defaults_primary_viewport_to_matplotlib_with_safe_activation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'RA_SIM_PRIMARY_VIEWPORT", "matplotlib"' in source
    assert "activate_runtime_primary_viewport(" in source
    assert "PRIMARY_VIEWPORT_BACKEND = primary_viewport_selection.active_backend" in source
    assert '"key": "primary_viewport_backend"' not in source
    assert "set_background_alpha=background_display.set_alpha" in source


def test_runtime_impl_shares_pick_hkl_live_cache_with_manual_qr_picker() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "geometry_manual_cache_workflow = (" in source
    assert (
        "simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params"
        in source
    )


def test_runtime_impl_allows_caked_preview_without_detector_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "one_d_analysis_requested = bool(" in source
    assert "caked_analysis_requested = bool(" in source
    assert "accumulate_image_requested" in source
    assert "or one_d_analysis_requested" in source
    assert "ax.set_title('2D Caked Position Preview')" not in source
    assert "ax.set_title('2D Caked Position Preview (1D paused)')" not in source
    assert 'ax.set_title("")' in source


def test_runtime_impl_enables_async_preview_calculations_in_runtime_update_paths() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "FULL_UPDATE_DEBOUNCE_MS = 90" in source
    assert "PREVIEW_UPDATE_DEBOUNCE_MS = 24" in source
    assert "LIVE_DRAG_SETTLE_MS = 80" in source
    assert "PREVIEW_CALCULATIONS_ENABLED = True" in source
    assert "if not PREVIEW_CALCULATIONS_ENABLED:" in source
    assert "def _current_update_debounce_ms() -> int:" in source
    assert "_current_update_debounce_ms()," in source
    assert "simulation_runtime_state.interaction_drag_requires_settled_update = True" in source
    assert "simulation_runtime_state.interaction_settle_token = root.after(" in source
    assert 'preview_job["job_kind"] = "preview"' in source
    assert "request_status = _request_async_simulation_job(preview_job)" in source
    assert 'text="Computing preview simulation in background..."' in source
    assert "do_update_queue_live_preview" in source
    assert "preview_result = _run_simulation_generation_job" not in source
    assert "desired_analysis_preview = bool(" in source
    assert "PREVIEW_CALCULATIONS_ENABLED\n        and analysis_requested" in source
    assert "PREVIEW_CALCULATIONS_ENABLED\n                and bool(live_geometry_preview_var.get())" in source


def test_runtime_impl_blocks_startup_on_initial_simulation_with_overlay() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    block_start = source.index("has_cached_simulation = (")
    block_end = source.index("elif ready_simulation_result is None and need_hit_table_refresh:")
    block = source[block_start:block_end]
    startup_start = source.index("def _run_initial_startup_work():")
    startup_end = source.index("root.after_idle(_run_initial_startup_work)")
    startup_block = source[startup_start:startup_end]

    assert "def _show_initial_simulation_loading_overlay() -> None:" in source
    assert '"Loading first simulation may take longer"' in source
    assert "_show_initial_simulation_loading_overlay()" in startup_block
    assert "matplotlib_canvas.draw()" in startup_block
    assert "root.update_idletasks()" in startup_block
    assert "start_exact_cake_numba_warmup_in_background()" not in startup_block
    assert "_run_simulation_generation_job(" in block
    assert 'progress_label.config(text="Computing initial simulation...")' in block


def test_runtime_impl_defers_exact_cake_numba_warmup_until_after_startup_work() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if analysis_sig is not None:" in source
    assert "if simulation_runtime_state.stored_sim_image is not None:" in source
    assert source.count("_schedule_exact_cake_numba_warmup_once()") >= 2
    assert source.count("_schedule_forward_simulation_numba_warmup_once()") >= 2


def test_runtime_impl_distinguishes_preview_and_full_worker_jobs() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'str(payload.get("job_kind", "full")),' in source
    assert "def _simulation_result_matches_signature(" in source
    assert "def _simulation_result_superseded_by_queued_job(" in source
    assert "superseded = _simulation_result_superseded_by_queued_job(result, queued_job)" in source
    assert "if isinstance(ready_result, dict) and not _simulation_result_matches_signature(" in source
    assert '_promote_queued_simulation_job(reason="previous_ready_result_consumed")' in source


def test_runtime_impl_keeps_1d_updates_gated_on_intensity_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if (\n                one_d_analysis_requested" in source


def test_runtime_impl_places_qr_cylinder_mode_in_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '\"key\": \"qr_cylinder_mode\"' in source
    assert '\"key\": \"display_raster_size\"' in source
    assert '\"label\": \"display px\"' in source
    assert "_refresh_main_display_raster_projection" in source
    assert "QR_CYLINDER_DISPLAY_MODE_REPLACE" in source
    assert 'parent=app_shell_view_state.match_peak_tools_frame' in source


def test_runtime_impl_moves_analysis_view_options_and_auto_match_to_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '\"key\": \"fast_viewer\"' not in source
    assert '\"key\": \"log_display\"' in source
    assert '\"key\": \"clear_integration_region\"' in source
    assert '\"key\": \"auto_match_scale\"' in source
    assert "control_type\": \"check\"" in source
    assert "control_type\": \"button\"" in source
    assert "parent=None" in source
    assert 'RA_SIM_FAST_VIEWER", "1"' in source
    assert "control_locked=True" in source
    assert "display_controls_view_state.fast_viewer_checkbutton = (" not in source
    assert "display_controls_view_state.simulation_controls_frame" not in source[
        source.index("def _auto_match_scale_factor_to_radial_peak("):
        source.index("def _update_chi_square_display(")
    ]


def test_runtime_impl_keeps_runtime_executors_in_latest_request_wins_mode() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert source.count("max_workers=default_reserved_cpu_worker_count(),") >= 2
    assert "Keep GUI simulation updates in latest-request-wins mode" in source
    assert "Keep GUI analysis updates in latest-request-wins mode" in source
    assert "def _replace_queued_simulation_job(job: dict[str, object]) -> None:" in source
    assert "def _replace_queued_analysis_job(job: dict[str, object]) -> None:" in source
    assert "simulation_runtime_state.worker_ready_result = None" in source
    assert "simulation_runtime_state.analysis_ready_result = None" in source
    assert '"job_requested"' in source
    assert '"job_submitted"' in source
    assert '"job_queued"' in source
    assert '"job_replaced"' in source
    assert '"job_started"' in source
    assert '"job_finished"' in source
    assert '"job_promoted_from_queued"' in source
    assert '"job_discarded_as_stale"' in source


def test_runtime_impl_removes_display_intensity_toggle_from_ui() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "accumulate_intensity_enabled=True" not in source


def test_runtime_impl_gates_1d_analysis_on_analyze_visibility() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _analysis_integration_outputs_visible() -> bool:" in source
    assert "app_shell_view_state.control_tab_var" in source
    assert "gui_views.analysis_popout_window_open(analysis_popout_view_state)" in source
    assert "and _analysis_integration_outputs_visible()" in source
    assert "if not _analysis_integration_outputs_visible():" in source
    assert '_analysis_tab_trace_add("write", _handle_analysis_integration_visibility_change)' in source


def test_runtime_impl_stages_structure_bootstrap_after_first_paint() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "build_lightweight_structure_model_state(" in source
    assert "def _bootstrap_structure_model_state_for_startup() -> None:" in source
    assert 'progress_label.config(text="Initializing structure model...")' in source
    assert "_bootstrap_structure_model_state_for_startup()" in source
    assert "_set_structure_bootstrap_controls_enabled(True)" in source
    assert "root.after_idle(_run_initial_startup_work)" in source


def test_runtime_impl_builds_shell_before_deferred_runtime_state_load() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    main_start = source.index('def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):')
    startup_start = source.index("def _run_initial_startup_work():", main_start)
    startup_end = source.index("root.after_idle(_run_initial_startup_work)", startup_start)
    pre_startup_block = source[main_start:startup_start]
    startup_block = source[startup_start:startup_end]

    assert "runtime_context = RuntimeContext(" in pre_startup_block
    assert "runtime_context = build_runtime_window_context(runtime_context)" in pre_startup_block
    assert "build_runtime_state_context()" not in pre_startup_block
    assert "root.deiconify()" in pre_startup_block
    assert "runtime_context = build_runtime_state_context()" in startup_block
    assert "runtime_context = build_runtime_plot_context(runtime_context)" in startup_block
    assert "runtime_context = build_runtime_controls_context(runtime_context)" in startup_block


def test_runtime_impl_lazy_builds_excel_dataframes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    apply_start = source.index("def _apply_structure_model_runtime_cache_state() -> None:")
    bootstrap_start = source.index("def _bootstrap_structure_model_state_for_startup() -> None:")
    apply_block = source[apply_start:bootstrap_start]
    export_start = source.index("def export_initial_excel():")
    export_end = source.index("app_shell_view_state = app_state.app_shell_view")
    export_block = source[export_start:export_end]

    assert "def _ensure_structure_model_dataframes() -> tuple[object, object]:" in source
    assert "def _build_intensity_dataframes(" in source
    assert "build_intensity_dataframes(" not in apply_block
    assert "df_summary, df_details = _ensure_structure_model_dataframes()" in export_block


def test_runtime_impl_lazy_mounts_analysis_surfaces() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    lazy_block_start = source.index("analysis_surfaces_initialized = False")
    lazy_helper_start = source.index("def _ensure_analysis_figure() -> None:")
    lazy_block = source[lazy_block_start:lazy_helper_start]

    assert "analysis_surfaces_initialized = False" in source
    assert "fig_1d = None" in source
    assert "def _ensure_analysis_figure() -> None:" in source
    assert "def ensure_analysis_surfaces_initialized() -> None:" in source
    assert "_show_analysis_tab_lazy_placeholders()" in source
    assert "plt.subplots(2, 1, figsize=(5, 8))" not in lazy_block
    assert "_mount_analysis_figure(app_shell_view_state.plot_frame_1d)" not in source
    assert "ensure_analysis_surfaces_initialized()" in source


def test_runtime_impl_trims_tools_imports_from_startup_path() -> None:
    runtime_source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    file_parsing_source = FILE_PARSING_SOURCE_PATH.read_text(encoding="utf-8")

    import_block_start = runtime_source.index("from ra_sim.io.file_parsing import parse_poni_file, Open_ASC")
    import_block_end = runtime_source.index("from ra_sim.io.data_loading import (", import_block_start)
    import_block = runtime_source[import_block_start:import_block_end]
    dataframe_helper_start = runtime_source.index("def _build_intensity_dataframes(")
    dataframe_helper_end = runtime_source.index("def _current_primary_cif_path()", dataframe_helper_start)
    dataframe_helper = runtime_source[dataframe_helper_start:dataframe_helper_end]
    azimuthal_helper_start = runtime_source.index("def _show_azimuthal_radial_plot_demo() -> None:")
    azimuthal_helper_end = runtime_source.index("app_shell_view_state = app_state.app_shell_view")
    azimuthal_helper = runtime_source[azimuthal_helper_start:azimuthal_helper_end]

    assert "from ra_sim.utils.tools import (" not in import_block
    assert "from ra_sim.utils.diffraction_tools import (" in import_block
    assert "build_intensity_dataframes," not in import_block
    assert "view_azimuthal_radial," not in import_block
    assert "detect_blobs," not in import_block
    assert "from ra_sim.utils.diffraction_tools import build_intensity_dataframes" in dataframe_helper
    assert "from ra_sim.utils.diffraction_tools import view_azimuthal_radial" in azimuthal_helper
    assert "from ra_sim.utils.tools import detect_blobs" not in file_parsing_source


def _mapping_field_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value)
    return None


def _subscript_field_name(node: ast.Subscript) -> str | None:
    return _mapping_field_name(node.slice)


def _mapping_call_field_name(node: ast.Call) -> tuple[str | None, str | None]:
    if not isinstance(node.func, ast.Attribute) or not node.args:
        return None, None
    return _mapping_field_name(node.args[0]), str(node.func.attr)


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _load_runtime_session_function(function_name: str):
    tree = ast.parse(RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {"np": np}
            exec(compile(module, str(RUNTIME_SESSION_SOURCE_PATH), "exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name!r} not found in runtime_session.py")


def _raw_field_reads(path: Path, *, field_name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    line_numbers: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            mapped_field, method_name = _mapping_call_field_name(node)
            if mapped_field == field_name and method_name == "get":
                line_numbers.add(int(node.lineno))
        elif isinstance(node, ast.Subscript):
            mapped_field = _subscript_field_name(node)
            if mapped_field == field_name and isinstance(node.ctx, ast.Load):
                line_numbers.add(int(node.lineno))
    return sorted(line_numbers)


def _expr_contains_len_range(expr: ast.AST | None) -> bool:
    if expr is None:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            for arg in node.args:
                for nested in ast.walk(arg):
                    if (
                        isinstance(nested, ast.Call)
                        and isinstance(nested.func, ast.Name)
                        and nested.func.id == "len"
                    ):
                        return True
    return False


def _assigned_names(node: ast.AST | None) -> list[str]:
    if isinstance(node, ast.Name):
        return [str(node.id)]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_assigned_names(element))
        return names
    return []


def _trust_field_assignment_lines(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    field_names = {
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
    }
    hits: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        target_nodes: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            target_nodes = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            target_nodes = [node.target]
        for target in target_nodes:
            if isinstance(target, ast.Subscript):
                mapped_field = _subscript_field_name(target)
                if mapped_field in field_names and isinstance(target.ctx, ast.Store):
                    hits.add((str(mapped_field), int(target.lineno)))
    return sorted(hits, key=lambda item: (item[1], item[0]))


def test_runtime_source_peak_alias_reads_stay_in_boundary_modules() -> None:
    offenders: list[str] = []
    inspected_paths = _python_sources(GUI_SOURCE_ROOT) + [OPTIMIZATION_MOSAIC_PROFILES_SOURCE_PATH]
    for path in inspected_paths:
        if path in RAW_SOURCE_PEAK_READ_ALLOWLIST:
            continue
        line_numbers = _raw_field_reads(path, field_name="source_peak_index")
        if line_numbers:
            rel_path = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel_path}:{','.join(str(line) for line in line_numbers)}")

    assert offenders == []


def test_source_reflection_index_arrays_are_not_minted_from_len_ranges() -> None:
    offenders: list[str] = []
    for path in _python_sources(RA_SIM_SOURCE_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            if not _expr_contains_len_range(value):
                continue
            assigned_names = {name for target in targets for name in _assigned_names(target)}
            if not any("source_reflection_indices" in name for name in assigned_names):
                continue
            rel_path = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel_path}:{int(node.lineno)}")

    assert offenders == []


def test_runtime_trust_field_assignments_stay_in_manual_geometry() -> None:
    offenders: list[str] = []
    for path in _python_sources(GUI_SOURCE_ROOT):
        if path in TRUST_FIELD_ASSIGNMENT_ALLOWLIST:
            continue
        hits = _trust_field_assignment_lines(path)
        if hits:
            rel_path = path.relative_to(REPO_ROOT)
            offenders.extend(
                f"{rel_path}:{line}:{field_name}" for field_name, line in hits
            )

    assert offenders == []
