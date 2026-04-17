import ast
import contextlib
import importlib
import json
import os
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
STRUCTURE_MODEL_SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "ra_sim" / "gui" / "structure_model.py"
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


def _top_level_import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets.extend(str(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "" if node.module is None else str(node.module)
            targets.extend(f"{module}:{alias.name}" for alias in node.names)
    return targets


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


def test_runtime_session_import_stays_headless_until_tk_backend_is_needed() -> None:
    script = """
import importlib
import json
import sys

sys.modules.pop("matplotlib.backends.backend_tkagg", None)
sys.modules.pop("ra_sim.gui._runtime.runtime_session", None)

runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

import matplotlib

print(
    json.dumps(
        {
            "backend": matplotlib.get_backend(),
            "tk_backend_loaded": "matplotlib.backends.backend_tkagg" in sys.modules,
            "tk_canvas_cls_cached": runtime_session._TK_FIGURE_CANVAS_CLS is not None,
        }
    )
)
"""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout.strip())

    assert payload["backend"].lower() == "agg"
    assert payload["tk_backend_loaded"] is False
    assert payload["tk_canvas_cls_cached"] is False


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


def test_runtime_session_defers_top_level_pycifrw_and_structure_model_imports() -> None:
    import_targets = _top_level_import_targets(RUNTIME_SESSION_SOURCE_PATH)

    assert "CifFile" not in import_targets
    assert "ra_sim.gui:structure_model" not in import_targets


def test_structure_model_defers_top_level_optional_dependency_imports() -> None:
    import_targets = _top_level_import_targets(STRUCTURE_MODEL_SOURCE_PATH)

    assert "CifFile" not in import_targets
    assert "Dans_Diffraction" not in import_targets


def test_runtime_session_uses_lazy_structure_model_and_cif_reader_helpers() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "_STRUCTURE_MODEL_MODULE = None" in source
    assert "def _get_structure_model_module():" in source
    assert "gui_structure_model = _LazyModuleProxy(_get_structure_model_module)" in source

    cif_helper_start = source.index("def _read_cif_block(cif_path):")
    cif_helper_end = source.index("def _normalize_ordered_structure_scale(", cif_helper_start)
    cif_helper_block = source[cif_helper_start:cif_helper_end]
    assert "import CifFile" in cif_helper_block
    assert "CifFile.ReadCif" in cif_helper_block

    block_01_start = source.index("def _initialize_runtime_state_block_01() -> None:")
    block_01_end = source.index("def _normalize_occupancy_label(", block_01_start)
    block_01 = source[block_01_start:block_01_end]
    assert "cf, blk = _read_cif_block(cif_file)" in block_01
    assert "CifFile.ReadCif" not in block_01

    block_05_start = source.index("def _initialize_runtime_state_block_05() -> None:")
    block_05_end = source.index("def _initialize_runtime_state_block_06() -> None:", block_05_start)
    block_05 = source[block_05_start:block_05_end]
    assert "cf2, blk2 = _read_cif_block(cif_file2)" in block_05
    assert "CifFile.ReadCif" not in block_05


def test_runtime_impl_prompts_from_root_only_before_full_runtime_bootstrap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    main_start = source.index(
        'def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):'
    )
    prompt_line = "resolved_mode = gui_bootstrap.choose_startup_mode_dialog(root)"
    prompt_index = source.index(prompt_line, main_start)
    ensure_root_index = source.index("ensure_runtime_root_initialized()", main_start)

    assert ensure_root_index < prompt_index


def test_runtime_impl_runtime_context_builders_gate_staged_initializers() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    state_start = source.index("def build_runtime_state_context() -> RuntimeContext:")
    window_start = source.index(
        "def build_runtime_window_context(context: RuntimeContext) -> RuntimeContext:"
    )
    plot_start = source.index(
        "def build_runtime_plot_context(context: RuntimeContext) -> RuntimeContext:"
    )
    controls_start = source.index(
        "def build_runtime_controls_context(context: RuntimeContext) -> RuntimeContext:"
    )
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


def test_runtime_impl_removes_fast_viewer_and_viewport_selector_symbols() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "RA_SIM_FAST_VIEWER" not in source
    assert "RA_SIM_PRIMARY_VIEWPORT" not in source
    assert "fast_viewer_workflow" not in source
    assert "_fast_viewer_active" not in source
    assert "runtime_display_acceleration" not in source
    assert "fast_plot_viewer" not in source
    assert "runtime_primary_viewport" not in source
    assert "tk_canvas" not in source


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

    assert (
        "from ra_sim.simulation.exact_cake import start_exact_cake_numba_warmup_in_background"
        in source
    )
    assert "start_exact_cake_geometry_warmup_in_background" in source
    assert "from ra_sim.simulation.exact_cake_portable import (" in source
    assert "FastAzimuthalIntegrator," in source
    assert "start_exact_cake_geometry_warmup_in_background," in source
    assert "_AZIMUTHAL_INTEGRATOR_CLS = FastAzimuthalIntegrator" in source
    assert "start_forward_simulation_numba_warmup_in_background" in source
    assert "start_qr_rod_simulation_numba_warmup_in_background" in source
    assert "process_peaks_parallel_safe," in source
    assert "def _process_peaks_parallel_safe_prefer_python_runner(*args, **kwargs):" in source
    assert 'kwargs.setdefault("prefer_python_runner", True)' in source
    assert "def _schedule_exact_cake_numba_warmup_once() -> None:" in source
    assert "def _schedule_forward_simulation_numba_warmup_once() -> None:" in source
    assert "def _schedule_qr_rod_simulation_numba_warmup_once() -> None:" in source
    assert "simulation_runtime_state.exact_cake_numba_warmup_scheduled" in source
    assert "simulation_runtime_state.forward_simulation_numba_warmup_scheduled" in source
    assert "simulation_runtime_state.qr_rod_simulation_numba_warmup_scheduled" in source
    assert "process_peaks_parallel=_process_peaks_parallel_safe_prefer_python_runner," in source
    assert source.count("process_peaks_parallel=process_peaks_parallel_safe,") == 1
    assert "source_entry=resolved_source_entry," in source
    assert "source_entry=source_entry," in source
    assert "source_entry=raw_record," in source
    assert "root.after_idle(start_exact_cake_numba_warmup_in_background)" in source
    assert "root.after_idle(start_forward_simulation_numba_warmup_in_background)" in source
    assert "root.after_idle(start_qr_rod_simulation_numba_warmup_in_background)" in source


def test_runtime_session_selected_peak_probe_runner_prefers_python(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    captured = {}

    monkeypatch.setattr(
        runtime_session,
        "process_peaks_parallel_safe",
        lambda *args, **kwargs: (
            captured.update({"args": args, "kwargs": dict(kwargs)}) or "runner-result"
        ),
        raising=False,
    )

    result = runtime_session._process_peaks_parallel_safe_prefer_python_runner(
        "sentinel",
        sample="value",
    )

    assert result == "runner-result"
    assert captured["args"] == ("sentinel",)
    assert captured["kwargs"]["sample"] == "value"
    assert captured["kwargs"]["prefer_python_runner"] is True


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


def test_runtime_impl_geometry_fit_caking_reuses_signature_by_distance_center_and_wavelength() -> (
    None
):
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "simulation_runtime_state.geometry_fit_caking_ai_cache = {}" in source
    assert "worker_geometry_fit_caking_sig" in source
    assert "requested_sig = _caked_geometry_cache_signature(" in source
    assert 'params_local.get("corto_detector", np.nan),' in source
    assert "center_value[0] if center_value.size > 0 else np.nan," in source
    assert "center_value[1] if center_value.size > 1 else np.nan," in source
    assert 'params_local.get("lambda", np.nan),' in source
    assert "persistent_cache = getattr(" in source
    assert '"geometry_fit_caking_ai_cache",' in source
    assert 'persistent_cache.get("sig") == requested_sig' in source
    assert "simulation_runtime_state.geometry_fit_caking_ai_cache = {" in source
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


def test_runtime_session_live_manual_peak_cache_update_uses_branch_aware_keys(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    legacy_saved_right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    saved_pairs = [
        {
            **left_entry,
            "x": 181.0,
            "y": 95.0,
        },
        {
            **legacy_saved_right_entry,
            "x": 190.0,
            "y": 96.0,
        },
    ]
    peak_records = [dict(left_entry), dict(right_entry)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_entry), dict(right_entry)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    captured_saved_pairs: dict[int, list[dict[str, object]]] = {}
    side_effects: list[tuple[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(entry) for entry in saved_pairs] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: captured_saved_pairs.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: side_effects.append(("invalidate", None)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(("render", dict(kwargs))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: side_effects.append(("schedule", None)),
        raising=False,
    )

    runtime_session._geometry_fit_live_update_manual_peak_cache(
        {
            "live_cache_records": [
                {
                    "dataset_index": 0,
                    "source_table_index": 9,
                    "source_row_index": 0,
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "simulated_detector_x": 191.0,
                    "simulated_detector_y": 97.0,
                    "simulated_two_theta_deg": 30.25,
                    "simulated_phi_deg": -57.5,
                }
            ]
        }
    )

    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 191.0
    assert peak_records[1]["display_row"] == 97.0
    assert peak_positions == [(181.0, 95.0), (191.0, 97.0)]
    assert peak_overlay_cache["records"][0]["display_col"] == 181.0
    assert peak_overlay_cache["records"][1]["display_col"] == 191.0
    assert peak_overlay_cache["positions"] == [(181.0, 95.0), (191.0, 97.0)]
    assert captured_saved_pairs[0][0]["source_branch_index"] == 0
    assert "refined_sim_x" not in captured_saved_pairs[0][0]
    assert captured_saved_pairs[0][1]["source_branch_index"] == 1
    assert captured_saved_pairs[0][1]["refined_sim_x"] == 191.0
    assert captured_saved_pairs[0][1]["refined_sim_y"] == 97.0
    assert captured_saved_pairs[0][1]["refined_sim_caked_x"] == 30.25
    assert captured_saved_pairs[0][1]["refined_sim_caked_y"] == -57.5
    assert side_effects == [
        ("invalidate", None),
        ("render", {"update_status": False}),
        ("schedule", None),
    ]


def test_runtime_session_live_manual_peak_cache_update_matches_branchless_saved_entry_by_hkl(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "1,0,5",
        "hkl": (1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "display_col": 181.0,
        "display_row": 95.0,
    }
    right_entry = {
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 190.0,
        "display_row": 96.0,
    }
    saved_pairs = [
        {
            "label": "1,0,5",
            "hkl": (1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 181.0,
            "y": 95.0,
        },
        {
            "label": "-1,0,5",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "x": 190.0,
            "y": 96.0,
        },
    ]
    peak_records = [dict(left_entry), dict(right_entry)]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(left_entry), dict(right_entry)],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    captured_saved_pairs: dict[int, list[dict[str, object]]] = {}

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(entry) for entry in saved_pairs] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: captured_saved_pairs.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: None,
        raising=False,
    )

    runtime_session._geometry_fit_live_update_manual_peak_cache(
        {
            "live_cache_records": [
                {
                    "dataset_index": 0,
                    "label": "-1,0,5",
                    "hkl": (-1, 0, 5),
                    "source_table_index": 9,
                    "source_row_index": 0,
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "simulated_detector_x": 191.0,
                    "simulated_detector_y": 97.0,
                    "simulated_two_theta_deg": 30.25,
                    "simulated_phi_deg": -57.5,
                }
            ]
        }
    )

    assert "refined_sim_x" not in captured_saved_pairs[0][0]
    assert captured_saved_pairs[0][1]["refined_sim_x"] == 191.0
    assert captured_saved_pairs[0][1]["refined_sim_y"] == 97.0
    assert captured_saved_pairs[0][1]["refined_sim_caked_x"] == 30.25
    assert captured_saved_pairs[0][1]["refined_sim_caked_y"] == -57.5


def test_runtime_session_replace_gui_state_peak_cache_reprojects_detector_view_from_native_fields(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[],
        peak_positions=[],
        peak_millers=[],
        peak_intensities=[],
        selected_peak_record={"stale": True},
        peak_overlay_cache={},
    )
    projection_calls: list[list[dict[str, object]]] = []
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )

    def _project_to_detector(records):
        projection_calls.append([dict(record) for record in records or ()])
        projected: list[dict[str, object]] = []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            native_col = float(record["native_col"])
            native_row = float(record["native_row"])
            projected.append(
                {
                    **dict(record),
                    "sim_col": native_col,
                    "sim_row": native_row,
                    "display_col": native_col,
                    "display_row": native_row,
                }
            )
        return projected

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_to_detector,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 190.0,
                "native_row": 96.0,
                "hkl": [-1, 0, 5],
                "intensity": 3.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
            }
        ]
    )

    assert projection_calls[0][0]["display_col"] == 30.25
    assert projection_calls[0][0]["display_row"] == -57.5
    assert projection_calls[0][0]["native_col"] == 190.0
    assert projection_calls[0][0]["native_row"] == 96.0
    assert simulation_runtime_state.peak_positions == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [3.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["display_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["native_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["native_row"] == 96.0
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_overlay_cache["records"][0]["display_row"] == 96.0
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert invalidated == [True]


def test_runtime_session_replace_gui_state_peak_cache_skips_bad_rows_when_projection_raises(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    simulation_runtime_state = SimpleNamespace(
        peak_records=[{"stale": True}],
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(9, 9, 9)],
        peak_intensities=[4.0],
        selected_peak_record={"stale": True},
        peak_overlay_cache={"records": [{"stale": True}], "positions": [(1.0, 2.0)]},
    )
    invalidated: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        simulation_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_retain_runtime_optional_cache",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.append(True),
        raising=False,
    )

    projection_calls: list[int] = []

    def _project_per_row(records):
        record = dict(records[0])
        projection_calls.append(int(record["source_row_index"]))
        if int(record["source_row_index"]) == 1:
            raise RuntimeError("boom")
        native_col = float(record["native_col"])
        native_row = float(record["native_row"])
        return [
            {
                **record,
                "sim_col": native_col,
                "sim_row": native_row,
                "display_col": native_col,
                "display_row": native_row,
            }
        ]

    monkeypatch.setattr(
        runtime_session,
        "_project_geometry_manual_peaks_to_current_view",
        _project_per_row,
        raising=False,
    )

    runtime_session._replace_gui_state_peak_cache(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "native_col": 190.0,
                "native_row": 96.0,
                "hkl": [-1, 0, 5],
                "intensity": 3.0,
                "q_group_key": ["q_group", "primary", 1.0, 5],
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
            },
            {
                "display_col": 31.25,
                "display_row": -56.5,
                "native_col": 191.0,
                "native_row": 97.0,
                "hkl": [-2, 0, 5],
                "intensity": 4.0,
                "q_group_key": ["q_group", "primary", 2.0, 5],
                "source_table_index": 9,
                "source_row_index": 1,
                "source_branch_index": 0,
            },
        ]
    )

    assert projection_calls == [0, 1]
    assert len(simulation_runtime_state.peak_records) == 1
    assert simulation_runtime_state.peak_positions == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_millers == [(-1, 0, 5)]
    assert simulation_runtime_state.peak_intensities == [3.0]
    assert simulation_runtime_state.selected_peak_record is None
    assert simulation_runtime_state.peak_records[0]["sim_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["sim_row"] == 96.0
    assert simulation_runtime_state.peak_records[0]["display_col"] == 190.0
    assert simulation_runtime_state.peak_records[0]["display_row"] == 96.0
    assert len(simulation_runtime_state.peak_overlay_cache["records"]) == 1
    assert simulation_runtime_state.peak_overlay_cache["positions"] == [(190.0, 96.0)]
    assert simulation_runtime_state.peak_overlay_cache["restored_from_gui_state"] is True
    assert invalidated == [True]


def test_runtime_session_refine_manual_pair_entry_from_cache_uses_branch_aware_lookup(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_source = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
    }
    right_source = {
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 30.0,
        "caked_y": -57.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 29.5,
        "caked_y": -58.0,
    }
    peak_records = [
        {
            **left_source,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    seen_source_branches: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                runtime_session._geometry_manual_candidate_source_key(left_source): dict(
                    left_source
                ),
                runtime_session._geometry_manual_candidate_source_key(right_source): dict(
                    right_source
                ),
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            seen_source_branches.append(int(source_entry["source_branch_index"])) or (30.25, -57.5)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )

    refined_entry = runtime_session._refine_geometry_manual_pair_entry_from_cache(dict(right_entry))

    assert seen_source_branches == [1]
    assert refined_entry is not None
    assert refined_entry["refined_sim_caked_x"] == 30.25
    assert refined_entry["refined_sim_caked_y"] == -57.5
    assert refined_entry["refined_sim_native_x"] == 30.25
    assert refined_entry["refined_sim_native_y"] == -57.5
    assert refined_entry["refined_sim_x"] == 30.25
    assert refined_entry["refined_sim_y"] == -57.5
    assert peak_records[0]["display_col"] == 181.0
    assert peak_records[0]["display_row"] == 95.0
    assert peak_records[1]["display_col"] == 30.25
    assert peak_records[1]["display_row"] == -57.5
    assert peak_positions == [(181.0, 95.0), (30.25, -57.5)]
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}


def test_runtime_session_refine_current_manual_pairs_uses_branch_aware_lookup(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    left_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "x": 181.0,
        "y": 95.0,
    }
    right_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 29.5,
        "caked_y": -58.0,
        "x": 190.0,
        "y": 96.0,
    }
    left_source_entry = {
        **left_entry,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
    }
    right_source_entry = {
        **right_entry,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
    }
    source_lookup = {
        runtime_session._geometry_manual_candidate_source_key(left_source_entry): {
            **left_source_entry,
            "caked_x": 29.0,
            "caked_y": -58.5,
        },
        runtime_session._geometry_manual_candidate_source_key(right_source_entry): {
            **right_source_entry,
            "caked_x": 30.0,
            "caked_y": -57.0,
        },
    }
    peak_records = [
        {
            **left_source_entry,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source_entry,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    saved_after_refine: dict[int, list[dict[str, object]]] = {}
    progress_messages: list[str] = []
    side_effects: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_state",
        SimpleNamespace(pick_session={}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_pick_session_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(left_entry), dict(right_entry)] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                key: ([dict(entry) for entry in value] if isinstance(value, list) else dict(value))
                for key, value in source_lookup.items()
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            (29.0, -58.5) if int(source_entry["source_branch_index"]) == 0 else (30.25, -57.5)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_push_geometry_manual_undo_state",
        lambda: side_effects.append("undo"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: saved_after_refine.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_fit_dataset_cache",
        lambda: side_effects.append("clear-fit-cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_manual_preview_artists",
        lambda **kwargs: side_effects.append(f"clear-preview:{kwargs.get('redraw')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(f"render:{kwargs.get('update_status')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_geometry_manual_pick_button_label",
        lambda: side_effects.append("button"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_background_status",
        lambda: side_effects.append("status"),
        raising=False,
    )

    runtime_session._refine_current_geometry_manual_pairs()

    assert saved_after_refine[0][0]["source_branch_index"] == 0
    assert saved_after_refine[0][0]["refined_sim_caked_x"] == 29.0
    assert saved_after_refine[0][0]["refined_sim_caked_y"] == -58.5
    assert saved_after_refine[0][1]["source_branch_index"] == 1
    assert saved_after_refine[0][1]["refined_sim_caked_x"] == 30.25
    assert saved_after_refine[0][1]["refined_sim_caked_y"] == -57.5
    assert peak_records[0]["display_col"] == 29.0
    assert peak_records[0]["display_row"] == -58.5
    assert peak_records[1]["display_col"] == 30.25
    assert peak_records[1]["display_row"] == -57.5
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}
    assert side_effects == [
        "undo",
        "clear-fit-cache",
        "clear-preview:False",
        "render:False",
        "button",
        "status",
    ]
    assert progress_messages[-1] == (
        "Refined 2 Qr/Qz simulation points on background 1 (1 moved, 0 skipped)."
    )


def test_runtime_session_refine_current_manual_pairs_ignores_stale_caked_coords_for_branchless_entry(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    saved_entry = {
        "label": "",
        "source_table_index": 9,
        "source_row_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "stale_caked_fields": True,
        "x": 190.0,
        "y": 96.0,
    }
    left_source_entry = {
        "label": "left",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "caked_x": 29.0,
        "caked_y": -58.5,
        "x": 181.0,
        "y": 95.0,
    }
    right_source_entry = {
        "label": "right",
        "source_table_index": 9,
        "source_row_index": 0,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "caked_x": 30.0,
        "caked_y": -57.0,
        "x": 190.0,
        "y": 96.0,
    }
    source_lookup = {
        runtime_session._geometry_manual_candidate_source_key(left_source_entry): dict(
            left_source_entry
        ),
        runtime_session._geometry_manual_candidate_source_key(right_source_entry): dict(
            right_source_entry
        ),
        ("source", 9, 0): [dict(left_source_entry), dict(right_source_entry)],
        (9, 0): [dict(left_source_entry), dict(right_source_entry)],
    }
    peak_records = [
        {
            **left_source_entry,
            "display_col": 181.0,
            "display_row": 95.0,
        },
        {
            **right_source_entry,
            "display_col": 190.0,
            "display_row": 96.0,
        },
    ]
    peak_positions = [(181.0, 95.0), (190.0, 96.0)]
    peak_overlay_cache = {
        "records": [dict(peak_records[0]), dict(peak_records[1])],
        "positions": list(peak_positions),
        "click_spatial_index": {"position_count": 2},
    }
    saved_after_refine: dict[int, list[dict[str, object]]] = {}
    progress_messages: list[str] = []
    side_effects: list[str] = []
    seen_source_branches: list[int] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_manual_state",
        SimpleNamespace(pick_session={}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "progress_label_geometry",
        SimpleNamespace(
            config=lambda **kwargs: progress_messages.append(str(kwargs.get("text", "")))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.zeros((8, 8), dtype=float),
            last_caked_radial_values=np.linspace(10.0, 17.0, 8, dtype=float),
            last_caked_azimuth_values=np.linspace(-4.0, 3.0, 8, dtype=float),
            stored_sim_image=np.zeros((8, 8), dtype=float),
            peak_records=peak_records,
            peak_positions=peak_positions,
            peak_overlay_cache=peak_overlay_cache,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_signature=("sig",),
            manual_pick_cache_data={"old": True},
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 8, raising=False)
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_pick_session_active",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_lookup_source_entry",
        lambda _lookup, entry, **_kwargs: (
            dict(right_source_entry)
            if int(entry.get("source_table_index", -1)) == 9
            and int(entry.get("source_row_index", -1)) == 0
            else None
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: [dict(saved_entry)] if int(index) == 0 else [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_geometry_manual_pick_cache",
        lambda **_kwargs: {
            "simulated_lookup": {
                key: ([dict(entry) for entry in value] if isinstance(value, list) else dict(value))
                for key, value in source_lookup.items()
            },
            "match_config": {"search_radius_px": 6.0},
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_params",
        lambda: {"a": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_manual_pick_background_image",
        lambda: np.zeros((8, 8), dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_match_background_context",
        lambda image, cfg: (dict(cfg), {"image_shape": tuple(np.asarray(image).shape)}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        lambda source_entry, *_args, **_kwargs: (
            seen_source_branches.append(int(source_entry["source_branch_index"]))
            or ((29.0, -58.5) if int(source_entry["source_branch_index"]) == 0 else (30.25, -57.5))
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_caked_angles_to_background_display_coords",
        lambda two_theta, phi: (float(two_theta), float(phi)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_background_display_to_native_detector_coords",
        lambda col, row: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_sim_to_display_coords",
        lambda col, row, _shape: (float(col), float(row)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_push_geometry_manual_undo_state",
        lambda: side_effects.append("undo"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_geometry_manual_pairs_for_index",
        lambda index, entries: saved_after_refine.__setitem__(
            int(index),
            [dict(entry) for entry in entries or ()],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_fit_dataset_cache",
        lambda: side_effects.append("clear-fit-cache"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_manual_preview_artists",
        lambda **kwargs: side_effects.append(f"clear-preview:{kwargs.get('redraw')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_render_current_geometry_manual_pairs",
        lambda **kwargs: side_effects.append(f"render:{kwargs.get('update_status')}"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_geometry_manual_pick_button_label",
        lambda: side_effects.append("button"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_background_status",
        lambda: side_effects.append("status"),
        raising=False,
    )

    runtime_session._refine_current_geometry_manual_pairs()

    peak_records_by_branch = {
        int(record["source_branch_index"]): dict(record) for record in peak_records
    }

    assert seen_source_branches == [1]
    assert saved_after_refine[0][0]["stale_caked_fields"] is True
    assert saved_after_refine[0][0]["refined_sim_caked_x"] == 30.25
    assert saved_after_refine[0][0]["refined_sim_caked_y"] == -57.5
    assert peak_records_by_branch[0]["display_col"] == 181.0
    assert peak_records_by_branch[0]["display_row"] == 95.0
    assert peak_records_by_branch[1]["display_col"] == 30.25
    assert peak_records_by_branch[1]["display_row"] == -57.5
    assert runtime_session.geometry_runtime_state.manual_pick_cache_signature is None
    assert runtime_session.geometry_runtime_state.manual_pick_cache_data == {}
    assert side_effects == [
        "undo",
        "clear-fit-cache",
        "clear-preview:False",
        "render:False",
        "button",
        "status",
    ]
    assert progress_messages[-1] == (
        "Refined 1 Qr/Qz simulation points on background 1 (1 moved, 0 skipped)."
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

    assert "_primary_raster_source_payload(" in source
    assert "_image_source" in source
    assert "image_source_signature" in source
    assert "if image_extent is not None:" in source
    assert "origin=image_origin," in source
    assert "extent=image_extent," in source
    assert "_store_primary_raster_source(" in source
    assert "_integration_overlay_raster_source_signature(" in source
    assert "parent_source_signature=image_source_signature," in source
    assert "source_signature=overlay_source_signature," in source
    assert "_apply_projected_primary_raster_to_artist(overlay_artist)" in source


def test_runtime_impl_wires_caked_custom_mask_signature_factory() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "caked_custom_mask_signature_factory=lambda: (" in source
    assert 'geometry_runtime_state.qr_cylinder_band_cache or {}).get("signature")' in source


def test_runtime_impl_wires_detector_geometry_signature_factory() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert (
        'detector_geometry_signature_factory=lambda: simulation_runtime_state.ai_cache.get("sig")'
        in source
    )


def test_runtime_impl_routes_main_figure_chrome_through_shared_helper() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "from ra_sim.gui import main_figure_chrome as gui_main_figure_chrome" in source
    assert "gui_main_figure_chrome.configure_matplotlib_canvas_widget(" in source
    assert "gui_main_figure_chrome.configure_main_figure_layout(" in source
    assert "gui_main_figure_chrome.apply_main_figure_axes_chrome(" in source
    assert "axes_visible=bool(analysis_space_display_available)" in source
    assert "axes_visible=bool(show_caked_image)" not in source
    assert "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=True)" in source
    assert (
        "gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=False)" in source
    )
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

    assert "def _analysis_display_payload_ready(" in source
    assert "def _analysis_payload_ready_after_restore_attempt(" in source
    assert "def _restore_caked_display_payload_from_cached_results(" in source
    assert "missing_analysis_payload = bool(" in source
    assert "and not analysis_payload_ready" in source
    assert "analysis_payload_ready = _analysis_payload_ready_after_restore_attempt(" in source
    assert "_restore_caked_display_payload_from_cached_results(" in source
    assert "caked_background_image is not None" in source
    assert "q_space_background_image is not None" in source
    assert "_get_current_background_native()" in source
    assert source.index(
        "analysis_payload_ready = _analysis_payload_ready_after_restore_attempt("
    ) < source.index("analysis_result_matches_target = bool(")
    assert source.index("analysis_result_matches_target = bool(") < source.index(
        "_request_async_analysis_job(",
        source.index("analysis_result_matches_target = bool("),
    )


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
    assert '"collected_hit_tables": bool(job["collect_hit_tables"]),' in source
    assert "intersection_cache_to_hit_tables," in source
    assert "def _resolved_peak_table_payload(" in source
    assert "primary_peak_tables = _resolved_peak_table_payload(cache1, raw_hit_tables1)" in source
    assert '"primary_max_positions": list(primary_peak_tables),' in source
    assert "stored_primary_intersection_cache = _copy_intersection_cache_tables(" in source
    assert "stored_secondary_intersection_cache = _copy_intersection_cache_tables(" in source
    assert "stored_primary_peak_table_lattice = list(" in source
    assert "stored_secondary_peak_table_lattice = list(" in source
    assert 'or "primary_intersection_cache" in result' in source
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

    assert (
        'allow_source_snapshot_rebuild = bool(lookup_context == "geometry_fit_dataset")' in source
    )
    assert "if allow_source_snapshot_rebuild:" in source


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

    assert (
        "def _invalidate_qr_cylinder_overlay_view_state(*, clear_artists: bool) -> None:" in source
    )
    assert "gui_qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(" in source
    assert "def toggle_caked_2d(requested_mode: str | None = None) -> None:" in source
    assert "_invalidate_qr_cylinder_overlay_view_state(clear_artists=True)" in source
    assert "def _apply_main_caked_view_toggle() -> None:" in source
    assert "_apply_main_caked_view_toggle()" in source


def test_runtime_impl_keeps_detector_and_caked_intersection_caches_separate() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    restore_start = source.index("def _restore_caked_display_payload_from_cached_results(")
    restore_end = source.index("def _run_analysis_job(", restore_start)
    restore_source = source[restore_start:restore_end]
    apply_start = source.index(
        "def _apply_ready_analysis_result(result: dict[str, object]) -> None:"
    )
    apply_end = source.index("def schedule_update():", apply_start)
    apply_source = source[apply_start:apply_end]

    assert (
        "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache"
        in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_transform_bundle = "
        in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_source_signature = "
        in restore_source
    )
    assert (
        "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache"
        not in restore_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache"
        in apply_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_transform_bundle = " in apply_source
    )
    assert (
        "simulation_runtime_state.last_caked_intersection_cache_source_signature = "
        in apply_source
    )
    assert (
        "simulation_runtime_state.stored_intersection_cache = caked_intersection_cache"
        not in apply_source
    )
    assert "simulation_runtime_state.last_caked_transform_bundle = (" in source


def test_runtime_impl_threads_primary_raster_source_signature_into_projection() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '_MAIN_RASTER_SOURCE_SIGNATURE_ATTR = "_ra_sim_source_signature"' in source
    assert "def _detector_display_raster_source_signature() -> object | None:" in source
    assert "def _resolved_primary_raster_source_signature(" in source
    assert "_MAIN_RASTER_SOURCE_SIGNATURE_ATTR," in source
    assert "source, extent, origin, source_signature" in source
    assert "_primary_raster_source_payload(artist)" in source
    assert "source_signature=source_signature," in source


def test_runtime_impl_preserves_no_redraw_preview_cancel_contract_for_overlay_restore() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    restore_start = source.index("def _restore_legacy_main_matplotlib_overlays(")
    restore_end = source.index("def _preview_legacy_main_matplotlib_view_limits(", restore_start)
    restore_source = source[restore_start:restore_end]
    reset_start = source.index("def _reset_main_figure_live_interaction_state(")
    configure_start = source.index("def _configure_primary_viewport_redraw_helpers()", reset_start)
    reset_source = source[reset_start:configure_start]

    assert (
        "def _restore_legacy_main_matplotlib_overlays(*, redraw: bool = True) -> bool:"
        in restore_source
    )
    assert "if bool(redraw)" in restore_source
    assert "else None" in restore_source
    assert "_request_legacy_main_matplotlib_redraw(force=True)" in restore_source
    assert (
        "def _reset_main_figure_live_interaction_state(*, redraw: bool = True) -> None:"
        in reset_source
    )
    assert "_restore_legacy_main_matplotlib_overlays(redraw=bool(redraw))" in reset_source
    assert (
        "preview_cleared = bool(_clear_legacy_main_matplotlib_preview_view(redraw=False))"
        in reset_source
    )
    assert "if dropped_preview_state and not preview_cleared:" in reset_source
    assert "_reset_main_figure_live_interaction_state(redraw=bool(redraw))" in reset_source
    assert "_reset_main_figure_live_interaction_state(redraw=False)" in reset_source


def test_apply_projected_primary_raster_to_artist_passes_stored_source_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    recorded_signatures: list[object] = []
    set_data_calls: list[np.ndarray] = []

    class _Artist:
        def __init__(self) -> None:
            self._extent = None

        def set_extent(self, extent):
            self._extent = list(extent)

        def set_data(self, image):
            set_data_calls.append(np.asarray(image, dtype=float).copy())

    monkeypatch.setattr(
        runtime_session,
        "_legacy_main_matplotlib_interaction_active",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 100,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_axes_image_origin",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_display_projection,
        "project_raster_to_view",
        lambda image, *, source_signature=None, **_kwargs: (
            recorded_signatures.append(source_signature)
            or runtime_session.gui_display_projection.RasterProjection(
                image=np.asarray(image, dtype=float).copy(),
                extent=(0.0, 2.0, 2.0, 0.0),
            )
        ),
        raising=False,
    )

    artist = _Artist()
    source_image = np.arange(4, dtype=float).reshape(2, 2)
    runtime_session._store_primary_raster_source(
        artist,
        source_image,
        source_signature=("sig", 7),
    )
    runtime_session._store_primary_raster_geometry(
        artist,
        origin="upper",
        extent=(0.0, 2.0, 2.0, 0.0),
    )

    assert runtime_session._apply_projected_primary_raster_to_artist(artist) is True
    assert recorded_signatures == [("sig", 7)]
    assert len(set_data_calls) == 1
    np.testing.assert_array_equal(set_data_calls[0], source_image)


def test_store_primary_raster_source_uses_detector_buffer_signature_for_global_image_buffer(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    artist = SimpleNamespace()
    detector_buffer = np.arange(4, dtype=float).reshape(2, 2)
    detector_signature = ("detector-cache", 3)

    monkeypatch.setattr(runtime_session, "global_image_buffer", detector_buffer, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_unscaled_image_signature=detector_signature),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: 2.5,
        raising=False,
    )

    runtime_session._store_primary_raster_source(artist, detector_buffer)
    source, _extent, _origin, source_signature = runtime_session._primary_raster_source_payload(
        artist
    )

    assert source is detector_buffer
    assert source_signature == (detector_signature, 2.5)


def test_set_primary_integration_overlay_image_uses_distinct_overlay_source_signature(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    image_artist = SimpleNamespace()
    overlay_artist = SimpleNamespace()
    parent_signature = ("image", 5)
    explicit_overlay_signature = ("raw_drag_preview", (2, 2), 3)
    overlay_image = np.full((2, 2), 7.0, dtype=float)
    applied_artists: list[object] = []

    monkeypatch.setattr(runtime_session, "image_display", image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "integration_region_overlay",
        overlay_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_projected_primary_raster_to_artist",
        lambda artist: applied_artists.append(artist) or True,
        raising=False,
    )

    runtime_session._store_primary_raster_source(
        image_artist,
        np.arange(4, dtype=float).reshape(2, 2),
        source_signature=parent_signature,
    )
    runtime_session._store_primary_raster_geometry(
        image_artist,
        origin="upper",
        extent=(0.0, 1.0, 1.0, 0.0),
    )

    runtime_session._set_primary_integration_overlay_image(
        overlay_image,
        source_signature=explicit_overlay_signature,
    )
    stored_source, extent, origin, overlay_signature = (
        runtime_session._primary_raster_source_payload(overlay_artist)
    )

    np.testing.assert_array_equal(stored_source, overlay_image)
    assert tuple(extent) == (0.0, 1.0, 1.0, 0.0)
    assert origin == "upper"
    assert overlay_signature != parent_signature
    assert overlay_signature[0] == "integration_region_overlay"
    assert overlay_signature[1] == parent_signature
    assert overlay_signature[2] == explicit_overlay_signature
    assert applied_artists == [overlay_artist]


def test_integration_overlay_raster_source_signature_prefers_explicit_overlay_signature() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    parent_signature = ("image", 5)
    explicit_overlay_signature = ("raw_drag_preview", (2, 2), 4)
    overlay_image = np.full((2, 2), 7.0, dtype=float)

    overlay_signature = runtime_session._integration_overlay_raster_source_signature(
        parent_source_signature=parent_signature,
        overlay_source=overlay_image,
        overlay_source_signature=explicit_overlay_signature,
    )

    assert overlay_signature == (
        "integration_region_overlay",
        parent_signature,
        explicit_overlay_signature,
    )


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
        raising=False,
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


def _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("preview mask should not call legacy analytic helpers")

    monkeypatch.setattr(runtime_session, "_detector_angular_maps_for_shape", _fail)
    monkeypatch.setattr(runtime_session, "_get_detector_angular_maps", _fail)
    monkeypatch.setattr(
        runtime_session,
        "_detector_pixel_to_scattering_angles",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_scattering_angles_to_detector_pixel",
        _fail,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_angular_maps",
        _fail,
        raising=False,
    )


def _preview_mask_regression_inputs() -> dict[str, object]:
    return {
        "selection": {
            "valid": True,
            "rows": np.array([0, 2, 9], dtype=np.int64),
            "cols": np.array([1, 1, 0], dtype=np.int64),
        },
        "native_shape": (3, 3),
        "radial_axis": np.array([10.0, 20.0], dtype=float),
        "azimuth_axis": np.array([-5.0, 5.0], dtype=float),
        "expected_mask": np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        ),
    }


def test_runtime_geometry_fit_roi_projector_does_not_call_legacy_projection_paths(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 6),
        radial_deg=np.array([1.0, 2.0], dtype=float),
        raw_azimuth_deg=np.array([-100.0, -80.0], dtype=float),
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


def test_runtime_current_geometry_fit_roi_selection_does_not_call_legacy_projection_paths(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    bundle = runtime_session.CakeTransformBundle(
        detector_shape=(6, 6),
        radial_deg=np.array([1.0, 2.0], dtype=float),
        raw_azimuth_deg=np.array([-100.0, -80.0], dtype=float),
        gui_azimuth_deg=np.array([-10.0, 10.0], dtype=float),
        lut=SimpleNamespace(),
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("ROI selection should not call legacy projection paths")

    bundle_calls: list[tuple[float, float, object]] = []

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
        del ai, detector_shape, radial_deg, azimuth_deg
        bundle_calls.append(
            (
                float(two_theta_deg),
                float(phi_deg),
                transform_bundle,
            )
        )
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
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            source_row_snapshots={
                0: {
                    "rows": [
                        {
                            "label": "1,0,0",
                            "q_group_key": ("q_group", "primary", 1, 0),
                            "source_table_index": 1,
                            "source_row_index": 2,
                            "sim_col": 3.0,
                            "sim_row": 4.0,
                        }
                    ]
                }
            },
            last_caked_radial_values=np.array([1.0, 2.0], dtype=float),
            last_caked_azimuth_values=np.array([-10.0, 10.0], dtype=float),
            ai_cache={"ai": None},
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_pairs_for_index",
        lambda index: (
            [
                {
                    "label": "1,0,0",
                    "background_two_theta_deg": 1.5,
                    "background_phi_deg": -10.0,
                }
            ]
            if int(index) == 0
            else []
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_current_background_native",
        lambda: np.ones((6, 6), dtype=float),
    )
    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: bundle,
    )
    monkeypatch.setattr(runtime_session, "fit_config", {"geometry": {"roi": True}})

    def _fake_build_geometry_fit_caked_roi_selection(
        source_rows,
        *,
        required_pairs,
        image_shape,
        fit_config,
        enabled_override,
        fit_space_to_detector_point,
    ):
        del fit_config
        return {
            "enabled": bool(enabled_override),
            "image_shape": tuple(int(v) for v in image_shape),
            "source_row_count": len(source_rows),
            "required_pair_count": len(required_pairs),
            "projected_point": fit_space_to_detector_point(1.5, -10.0),
        }

    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "build_geometry_fit_caked_roi_selection",
        _fake_build_geometry_fit_caked_roi_selection,
    )

    selection = runtime_session._current_geometry_fit_caked_roi_selection(force_enabled=True)

    assert selection == {
        "enabled": True,
        "image_shape": (6, 6),
        "source_row_count": 1,
        "required_pair_count": 1,
        "projected_point": (12.0, 34.0),
    }
    assert bundle_calls == [(1.5, -10.0, bundle)]


def test_runtime_geometry_fit_caked_roi_defaults_to_full_image_when_toggle_missing(
    monkeypatch,
) -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    assert "geometry_fit_caked_roi_enabled_var = tk.BooleanVar(value=False)" in source

    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    monkeypatch.delattr(
        runtime_session,
        "geometry_fit_caked_roi_enabled_var",
        raising=False,
    )

    assert runtime_session._current_geometry_fit_caked_roi_enabled() is False


def test_runtime_geometry_fit_caked_roi_preview_mask_uses_live_exact_bundle_only(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    live_bundle = runtime_session.CakeTransformBundle(
        detector_shape=inputs["native_shape"],
        radial_deg=np.array([10.0, 20.0], dtype=float),
        raw_azimuth_deg=np.array([-95.0, -85.0], dtype=float),
        gui_azimuth_deg=np.array([-5.0, 5.0], dtype=float),
        lut=SimpleNamespace(),
    )
    projection_calls: list[tuple[object, float, float]] = []

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    def _fail_integrator():
        raise AssertionError("preview mask should reuse matching live bundle")

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: live_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        _fail_integrator,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    assert projection_calls == [
        (live_bundle, 1.0, 0.0),
        (live_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_rebuilds_exact_bundle_without_legacy_helpers(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    expected_raw_azimuth = np.asarray(
        runtime_session.gui_phi_to_raw_phi(inputs["azimuth_axis"]),
        dtype=float,
    )
    expected_raw_azimuth = np.sort(expected_raw_azimuth, kind="stable")
    original_resolve_bundle = runtime_session.resolve_cake_transform_bundle
    resolve_calls: list[
        tuple[object, tuple[int, int], tuple[float, ...], tuple[float, ...] | None, bool]
    ] = []
    projection_calls: list[tuple[object, float, float]] = []

    def _record_resolve_bundle(
        ai_arg,
        detector_shape,
        radial_deg,
        *,
        gui_azimuth_deg=None,
        raw_azimuth_deg=None,
        transform_bundle=None,
        require_gui_display_match=False,
        engine="auto",
        workers="auto",
    ):
        del transform_bundle
        resolve_calls.append(
            (
                ai_arg,
                tuple(detector_shape),
                tuple(np.asarray(radial_deg, dtype=float)),
                (
                    tuple(np.asarray(raw_azimuth_deg, dtype=float))
                    if raw_azimuth_deg is not None
                    else None
                ),
                bool(require_gui_display_match),
            )
        )
        return original_resolve_bundle(
            ai_arg,
            detector_shape,
            radial_deg,
            gui_azimuth_deg=gui_azimuth_deg,
            raw_azimuth_deg=raw_azimuth_deg,
            require_gui_display_match=require_gui_display_match,
            engine=engine,
            workers=workers,
        )

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session,
        "resolve_cake_transform_bundle",
        _record_resolve_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    assert resolve_calls[0] == (
        None,
        inputs["native_shape"],
        (10.0, 20.0),
        tuple(expected_raw_azimuth),
        True,
    )
    assert resolve_calls[1] == (
        ai,
        inputs["native_shape"],
        (10.0, 20.0),
        tuple(expected_raw_azimuth),
        True,
    )
    rebuilt_bundle = projection_calls[0][0]
    assert isinstance(rebuilt_bundle, runtime_session.CakeTransformBundle)
    assert projection_calls == [
        (rebuilt_bundle, 1.0, 0.0),
        (rebuilt_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_rebuilds_when_live_bundle_raw_axis_is_stale(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    stale_bundle = runtime_session.CakeTransformBundle(
        detector_shape=inputs["native_shape"],
        radial_deg=np.array([10.0, 20.0], dtype=float),
        raw_azimuth_deg=np.array([70.0, 110.0], dtype=float),
        gui_azimuth_deg=np.array([-20.0, 20.0], dtype=float),
        lut=SimpleNamespace(),
    )
    expected_raw_azimuth = np.asarray(
        runtime_session.gui_phi_to_raw_phi(inputs["azimuth_axis"]),
        dtype=float,
    )
    expected_raw_azimuth = np.sort(expected_raw_azimuth, kind="stable")
    projection_calls: list[tuple[object, float, float]] = []

    def _record_detector_pixel_to_caked_bin(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        np.testing.assert_array_equal(
            np.asarray(bundle.raw_azimuth_deg, dtype=float),
            expected_raw_azimuth,
        )
        if (float(col), float(row)) == (1.0, 0.0):
            return (10.2, -4.0)
        if (float(col), float(row)) == (1.0, 2.0):
            return (19.6, 4.0)
        return (None, None)

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: stale_bundle,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _record_detector_pixel_to_caked_bin,
    )

    mask = runtime_session._geometry_fit_caked_roi_preview_mask(
        selection=inputs["selection"],
        native_shape=inputs["native_shape"],
        radial_axis=inputs["radial_axis"],
        azimuth_axis=inputs["azimuth_axis"],
    )

    np.testing.assert_array_equal(mask, inputs["expected_mask"])
    rebuilt_bundle = projection_calls[0][0]
    assert isinstance(rebuilt_bundle, runtime_session.CakeTransformBundle)
    assert rebuilt_bundle is not stale_bundle
    np.testing.assert_array_equal(
        np.asarray(rebuilt_bundle.raw_azimuth_deg, dtype=float),
        expected_raw_azimuth,
    )
    assert projection_calls == [
        (rebuilt_bundle, 1.0, 0.0),
        (rebuilt_bundle, 1.0, 2.0),
    ]


def test_runtime_geometry_fit_caked_roi_preview_mask_returns_none_without_valid_bundle(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_preview_mask_legacy_helper_guards(monkeypatch, runtime_session)
    inputs = _preview_mask_regression_inputs()
    ai = runtime_session.FastAzimuthalIntegrator(
        dist=0.25,
        poni1=0.0015,
        poni2=0.0020,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    def _fail_projection(*_args, **_kwargs):
        raise AssertionError("preview mask should return None before pixel projection")

    monkeypatch.setattr(
        runtime_session,
        "_current_live_caked_transform_bundle",
        lambda: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_preview_integrator",
        lambda: ai,
    )
    monkeypatch.setattr(
        runtime_session,
        "resolve_cake_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        _fail_projection,
    )

    assert (
        runtime_session._geometry_fit_caked_roi_preview_mask(
            selection=inputs["selection"],
            native_shape=inputs["native_shape"],
            radial_axis=inputs["radial_axis"],
            azimuth_axis=inputs["azimuth_axis"],
        )
        is None
    )


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
    assert projection_view["detector_shape"] == (6, 8)
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

    assert "_normalize_geometry_fit_caked_view_payload(" in helper_source
    assert '"raw_azimuth_axis": np.asarray(' in helper_source
    assert '"transform_bundle": normalized_payload.get("transform_bundle")' in helper_source
    assert '"detector_shape": tuple(normalized_payload.get("detector_shape", ()))' in helper_source


def test_prepare_caked_intersection_cache_uses_exact_detector_projector(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (64, 96)

    bundle = FakeBundle()
    rotate_calls: list[tuple[float, float, tuple[int, int], int]] = []
    projector_calls: list[tuple[object, float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "_rotate_point_for_display",
        lambda col, row, shape, k: (
            rotate_calls.append((float(col), float(row), tuple(shape), int(k)))
            or (41.0, 49.0)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row)))
            or (float(col) + 0.5, float(row) - 0.25)
        ),
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [
            np.asarray(
                [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]],
                dtype=float,
            )
        ],
        transform_bundle=bundle,
    )

    out = np.asarray(transformed[0], dtype=float)
    assert rotate_calls == [(40.0, 50.0, (64, 96), runtime_session.DISPLAY_ROTATE_K)]
    assert projector_calls == [(bundle, 41.0, 49.0)]
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :9], [1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0])
    assert out[0, 14] == 41.5
    assert out[0, 15] == 48.75


def test_prepare_caked_intersection_cache_blanks_prefilled_caked_cols_without_valid_bundle() -> (
    None
):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [table],
        transform_bundle=None,
    )

    out = np.asarray(transformed[0], dtype=float)
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :14], table[0, :14])
    assert np.isnan(out[0, 14])
    assert np.isnan(out[0, 15])


def test_prepare_caked_intersection_cache_blanks_prefilled_caked_cols_when_projector_returns_none(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (64, 96)

    bundle = FakeBundle()
    projector_calls: list[tuple[object, float, float]] = []
    table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "_rotate_point_for_display",
        lambda col, row, shape, k: (float(col) + 1.0, float(row) - 1.0),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row))) or (None, None)
        ),
    )

    transformed = runtime_session._prepare_caked_intersection_cache(
        [table],
        transform_bundle=bundle,
    )

    out = np.asarray(transformed[0], dtype=float)
    assert projector_calls == [(bundle, 41.0, 49.0)]
    assert out.shape == (1, 16)
    np.testing.assert_allclose(out[0, :14], table[0, :14])
    assert np.isnan(out[0, 14])
    assert np.isnan(out[0, 15])


def test_analysis_cache_overlay_coords_reuses_current_prepared_caked_cache_columns(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        detector_shape = (32, 48)

    bundle = FakeBundle()
    detector_cache = [
        np.asarray([[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]], dtype=float)
    ]
    projector_calls: list[tuple[object, float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session,
        "_rotate_point_for_display",
        lambda col, row, shape, k: (float(col) + 1.0, float(row) - 1.0),
    )
    monkeypatch.setattr(
        runtime_session,
        "detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row)))
            or (12.25, -33.5)
        ),
    )

    prepared_cache = runtime_session._prepare_caked_intersection_cache(
        detector_cache,
        transform_bundle=bundle,
    )

    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        prepared_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "stored_intersection_cache",
        detector_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_source_signature",
        (id(detector_cache), len(detector_cache)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current caked cache should reuse prepared caked columns")
        ),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        prepared_cache[0],
        show_caked=True,
    )

    assert projector_calls == [(bundle, 41.0, 49.0)]
    np.testing.assert_allclose(x_vals, [12.25])
    np.testing.assert_allclose(y_vals, [-33.5])


def test_analysis_cache_overlay_coords_ignores_stale_cached_caked_columns(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    stale_cache_bundle = FakeBundle()
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        stale_cache_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (91.0, -44.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        np.asarray(
            [
                [
                    1.5,
                    2.5,
                    40.0,
                    50.0,
                    8.0,
                    0.375,
                    1.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.5,
                    -32.0,
                ]
            ],
            dtype=float,
        ),
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [91.0])
    np.testing.assert_allclose(y_vals, [-44.0])


def test_analysis_cache_overlay_coords_ignores_stale_cached_caked_columns_when_source_sig_mismatches(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    caked_cache_table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )
    detector_cache = [np.asarray([[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]], dtype=float)]
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        [caked_cache_table],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "stored_intersection_cache",
        detector_cache,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_source_signature",
        (999, 1),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (93.0, -46.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        caked_cache_table,
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [93.0])
    np.testing.assert_allclose(y_vals, [-46.0])


def test_analysis_cache_overlay_coords_ignores_detector_cache_caked_columns_even_when_bundle_matches(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        pass

    live_bundle = FakeBundle()
    detector_cache_table = np.asarray(
        [[1.5, 2.5, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.5, -32.0]],
        dtype=float,
    )
    projector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        live_bundle,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.simulation_runtime_state,
        "last_caked_intersection_cache",
        [],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_native_detector_coords_to_live_caked_coords",
        lambda col, row: projector_calls.append((float(col), float(row))) or (82.0, -41.0),
    )

    x_vals, y_vals = runtime_session._analysis_cache_overlay_coords(
        detector_cache_table,
        show_caked=True,
    )

    assert projector_calls == [(40.0, 50.0)]
    np.testing.assert_allclose(x_vals, [82.0])
    np.testing.assert_allclose(y_vals, [-41.0])


def test_invalidate_cached_analysis_space_payloads_clears_caked_cache_provenance(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    live_bundle_calls: list[object] = []
    q_space_payload_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, -1.0, 1.0],
            last_caked_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
            last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
            last_caked_intersection_cache=("cache",),
            last_caked_intersection_cache_transform_bundle="bundle",
            last_caked_intersection_cache_source_signature=(1, 1),
            last_q_space_payload_signature="q-space-sig",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda bundle: live_bundle_calls.append(bundle),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda *, sim_payload=None, bg_payload=None: q_space_payload_calls.append(
            (sim_payload, bg_payload)
        ),
        raising=False,
    )

    runtime_session._invalidate_cached_analysis_space_payloads(
        clear_caked=True,
        clear_q_space=False,
    )

    assert runtime_session.simulation_runtime_state.last_caked_image_unscaled is None
    assert runtime_session.simulation_runtime_state.last_caked_extent is None
    assert runtime_session.simulation_runtime_state.last_caked_background_image_unscaled is None
    assert runtime_session.simulation_runtime_state.last_caked_radial_values is None
    assert runtime_session.simulation_runtime_state.last_caked_azimuth_values is None
    assert runtime_session.simulation_runtime_state.last_caked_intersection_cache is None
    assert (
        runtime_session.simulation_runtime_state.last_caked_intersection_cache_transform_bundle
        is None
    )
    assert (
        runtime_session.simulation_runtime_state.last_caked_intersection_cache_source_signature
        is None
    )
    assert live_bundle_calls == [None]
    assert q_space_payload_calls == []


def test_runtime_impl_combined_detector_cache_recomposition_invalidates_caked_intersection_cache() -> (
    None
):
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    update_start = source.index(
        "simulation_runtime_state.stored_intersection_cache = intersection_cache_local"
    )
    update_end = source.index(
        "simulation_runtime_state.stored_sim_image = updated_image",
        update_start,
    )
    update_source = source[update_start:update_end]

    assert "_clear_caked_intersection_cache()" in update_source


def test_runtime_impl_full_reset_invalidates_caked_intersection_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    reset_start = source.index("def _initialize_runtime_controls_block_28() -> None:")
    reset_end = source.index(
        "###############################################################################",
        reset_start,
    )
    reset_source = source[reset_start:reset_end]

    clear_start = reset_source.index("simulation_runtime_state.stored_intersection_cache = None")
    clear_end = reset_source.index(
        'simulation_runtime_state.last_unscaled_image_signature = None',
        clear_start,
    )
    clear_source = reset_source[clear_start:clear_end]

    assert "_clear_caked_intersection_cache()" in clear_source


def test_runtime_impl_manual_rebuild_invalidates_caked_intersection_cache() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    rebuild_start = source.index(
        "simulation_runtime_state.stored_intersection_cache = _copy_intersection_cache_tables("
    )
    rebuild_end = source.index(
        "simulation_runtime_state.last_simulation_signature = rebuild_result.requested_signature",
        rebuild_start,
    )
    rebuild_source = source[rebuild_start:rebuild_end]

    assert "_clear_caked_intersection_cache()" in rebuild_source


class _RuntimeVar:
    def __init__(self, value: float | bool) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _install_idle_main_figure_preview_state(monkeypatch, runtime_session) -> None:
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            _canvas_preview_limits=None,
            _canvas_pan_session=None,
        ),
        raising=False,
    )


def _install_matching_hidden_analysis_payload_state(
    monkeypatch,
    runtime_session,
    *,
    include_do_update_state: bool,
) -> dict[str, object]:
    sim_signature = ("sim-sig",)
    caked_geometry_sig = ("caked-geom",)
    q_space_payload_geometry_sig = ("q-geom",)
    sim_caking_sig = (sim_signature, caked_geometry_sig, 1.0)
    current_analysis_cache_sig = (
        (
            sim_caking_sig,
            int(runtime_session.DEFAULT_ANALYSIS_RADIAL_BINS),
            int(runtime_session.DEFAULT_ANALYSIS_AZIMUTH_BINS),
        ),
        None,
    )
    caked_image = np.ones((2, 2), dtype=np.float64)
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)
    state_fields = {
        "last_analysis_cache_sig": current_analysis_cache_sig,
        "last_q_space_payload_signature": (
            current_analysis_cache_sig,
            q_space_payload_geometry_sig,
        ),
        "last_caked_image_unscaled": caked_image.copy(),
        "last_caked_extent": [0.0, 1.0, -1.0, 1.0],
        "last_caked_background_image_unscaled": np.full((2, 2), 2.0, dtype=np.float64),
        "last_caked_radial_values": np.array([1.0, 2.0], dtype=np.float64),
        "last_caked_azimuth_values": np.array([-1.0, 1.0], dtype=np.float64),
        "last_caked_intersection_cache": ("cache",),
        "last_caked_intersection_cache_transform_bundle": "bundle",
        "last_q_space_image_unscaled": q_space_image.copy(),
        "last_q_space_qr_values": np.array([0.1, 0.2], dtype=np.float64),
        "last_q_space_qz_values": np.array([0.3, 0.4], dtype=np.float64),
        "last_q_space_extent": [-1.0, 1.0, -2.0, 2.0],
        "last_q_space_background_image_unscaled": np.full((2, 2), 4.0, dtype=np.float64),
        "caked_limits_user_override": False,
    }
    if include_do_update_state:
        state_fields.update(
            update_trace_counter=0,
            update_pending=None,
            update_running=False,
            update_phase="idle",
            worker_active_job=None,
            worker_queued_job=None,
            analysis_active_job=None,
            analysis_queued_job=None,
            preview_active=False,
            num_samples=0,
            sim_primary_qr={},
            sim_miller1=np.zeros((0, 3), dtype=np.float64),
            sim_intens1=np.zeros((0,), dtype=np.float64),
            sim_miller2=np.zeros((0, 3), dtype=np.float64),
            sim_intens2=np.zeros((0,), dtype=np.float64),
            primary_requested_source_mode="",
            primary_requested_contribution_keys=(),
            primary_requested_filter_signature=None,
            sf_prune_stats={},
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            stored_primary_sim_image=np.ones((2, 2), dtype=np.float64),
            stored_secondary_sim_image=None,
            last_sim_signature=sim_signature,
            last_simulation_signature=sim_signature + (0,),
            stored_primary_max_positions=None,
            stored_secondary_max_positions=None,
            stored_primary_source_reflection_indices=None,
            stored_secondary_source_reflection_indices=None,
            stored_primary_peak_table_lattice=None,
            stored_secondary_peak_table_lattice=None,
            stored_primary_intersection_cache=None,
            stored_secondary_intersection_cache=None,
            normalization_scale_cache={"sig": None, "value": 1.0},
            peak_intensities=[],
            peak_records=[],
            last_1d_integration_data={},
            ai_cache={"sig": caked_geometry_sig, "ai": object()},
            stored_hit_table_signature=None,
            stored_sim_image=None,
            stored_max_positions_local=None,
            stored_source_reflection_indices_local=None,
            stored_peak_table_lattice=None,
            stored_intersection_cache=None,
            last_unscaled_image_signature=None,
        )
    state = SimpleNamespace(**state_fields)
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", state, raising=False)
    return {
        "simulation_runtime_state": state,
        "sim_signature": sim_signature,
        "caked_geometry_sig": caked_geometry_sig,
        "q_space_payload_geometry_sig": q_space_payload_geometry_sig,
        "current_analysis_cache_sig": current_analysis_cache_sig,
        "caked_image": caked_image,
        "q_space_image": q_space_image,
    }


def _patch_do_update_detector_cache_prereqs(monkeypatch, runtime_session, fixture) -> tuple:
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            current_background_index=0,
            current_background_image=None,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            visible=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(scale_factor_user_override=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(1.0),
            simulation_min_var=_RuntimeVar(0.0),
            simulation_max_var=_RuntimeVar(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=_RuntimeVar(False), show_1d_var=_RuntimeVar(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "gamma_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_width_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_length_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "sample_depth_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "debye_x_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "debye_y_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _RuntimeVar(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "center_x_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _RuntimeVar(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "wave_m", 1.24e-10, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.0, raising=False)
    monkeypatch.setattr(runtime_session, "av2", None, raising=False)
    monkeypatch.setattr(runtime_session, "cv2", None, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_q_group_state", object(), raising=False)
    monkeypatch.setattr(
        runtime_session, "peak_selection_runtime_maintenance", object(), raising=False
    )
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 2.0), raising=False)
    monkeypatch.setattr(runtime_session, "_last_a_for_ht", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_c_for_ht", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_iodine_z_for_ht", 0.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_phi_l_divisor", 1.0, raising=False)
    monkeypatch.setattr(runtime_session, "_last_phase_delta_expression", "", raising=False)
    monkeypatch.setattr(runtime_session, "_last_atom_site_fractional_signature", (), raising=False)
    monkeypatch.setattr(runtime_session, "_ensure_runtime_update_trace_hooks", lambda: None)
    monkeypatch.setattr(
        runtime_session, "_append_runtime_update_trace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runtime_session, "_refresh_run_status_bar", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_initial_simulation_loading_overlay",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "detector_two_theta_max", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(runtime_session, "_current_phase_delta_expression", lambda: "")
    monkeypatch.setattr(runtime_session, "_current_phi_l_divisor", lambda: 1.0)
    monkeypatch.setattr(runtime_session, "_current_iodine_z", lambda: 0.0)
    monkeypatch.setattr(runtime_session, "_current_atom_site_fractional_values", lambda: ())
    monkeypatch.setattr(runtime_session, "_atom_site_fractional_signature", lambda _values: ())
    monkeypatch.setattr(
        runtime_session, "_current_effective_theta_initial", lambda strict_count=False: 0.0
    )
    monkeypatch.setattr(runtime_session, "_current_ordered_structure_scale", lambda: 1.0)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "build_mosaic_params",
        lambda: {
            "sigma_mosaic_deg": 0.0,
            "gamma_mosaic_deg": 0.0,
            "eta": 0.0,
        },
    )
    monkeypatch.setattr(runtime_session, "_current_optics_mode_flag", lambda: 0)
    monkeypatch.setattr(
        runtime_session,
        "_qr_cylinder_replace_simulation_enabled",
        lambda: False,
    )
    monkeypatch.setattr(runtime_session, "_should_collect_hit_tables_for_update", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_from_params",
        lambda *_args, **_kwargs: fixture["sim_signature"],
    )
    monkeypatch.setattr(runtime_session, "current_sf_prune_bias", lambda: 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_cached_hit_tables_reusable", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(runtime_session, "_consume_ready_simulation_result", lambda _sig: None)
    monkeypatch.setattr(
        runtime_session, "_copy_intersection_cache_tables", lambda tables: list(tables)
    )
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "consume_geometry_q_group_refresh_request",
        lambda _state: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_runtime_geometry_interaction,
        "refresh_runtime_peak_selection_after_update",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_live_geometry_preview_enabled",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_get_current_background_backend", lambda: None)
    monkeypatch.setattr(runtime_session, "_trace_live_cache_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_live_cache_signature_summary", lambda sig: sig)
    monkeypatch.setattr(runtime_session, "_set_scale_factor_value", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime_session,
        "_caked_geometry_cache_signature",
        lambda *_args, **_kwargs: fixture["caked_geometry_sig"],
    )
    monkeypatch.setattr(
        runtime_session,
        "_q_space_geometry_cache_signature",
        lambda **_kwargs: fixture["q_space_payload_geometry_sig"],
    )
    monkeypatch.setattr(
        runtime_session,
        "start_exact_cake_geometry_warmup_in_background",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "detector")
    monkeypatch.setattr(
        runtime_session,
        "_current_primary_figure_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "LIVE_DRAG_PREVIEW_ENABLED", False, raising=False)
    monkeypatch.setattr(runtime_session, "PREVIEW_CALCULATIONS_ENABLED", True, raising=False)
    monkeypatch.setattr(runtime_session, "_analysis_integration_outputs_visible", lambda: False)
    monkeypatch.setattr(runtime_session, "_live_interaction_active", lambda: False)
    monkeypatch.setattr(
        runtime_session,
        "_clear_pending_main_figure_preview_interaction",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_defer_nonessential_redraw",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "ax", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "capture_axis_limits",
        lambda *_args, **_kwargs: ((0.0, 2.0), (2.0, 0.0)),
        raising=False,
    )
    return ((0.0, 2.0), (2.0, 0.0))


def test_do_update_routes_hidden_payload_guard_into_primary_display_helper(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _StopAtPrimaryDisplayHelper(RuntimeError):
        pass

    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=True,
    )
    preserved_primary_limits = _patch_do_update_detector_cache_prereqs(
        monkeypatch,
        runtime_session,
        fixture,
    )
    helper_calls: list[dict[str, object]] = []
    display_helper_calls: list[dict[str, object]] = []
    real_invalidate_hidden_payloads = (
        runtime_session._invalidate_hidden_analysis_space_payloads_if_stale
    )

    def _record_hidden_payload_guard(**kwargs) -> None:
        helper_calls.append(dict(kwargs))
        real_invalidate_hidden_payloads(**kwargs)

    def _stop_at_primary_display_helper(
        previous_primary_view_mode: str | None,
        preserved_primary_limits_arg: object,
    ) -> str:
        display_helper_calls.append(
            {
                "previous_primary_view_mode": previous_primary_view_mode,
                "preserved_primary_limits": preserved_primary_limits_arg,
                "caked_image": np.asarray(
                    runtime_session.simulation_runtime_state.last_caked_image_unscaled,
                    dtype=np.float64,
                ).copy(),
                "q_space_image": np.asarray(
                    runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
                    dtype=np.float64,
                ).copy(),
            }
        )
        raise _StopAtPrimaryDisplayHelper()

    monkeypatch.setattr(
        runtime_session,
        "_invalidate_hidden_analysis_space_payloads_if_stale",
        _record_hidden_payload_guard,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_primary_figure_display_from_cached_results",
        _stop_at_primary_display_helper,
        raising=False,
    )

    with pytest.raises(_StopAtPrimaryDisplayHelper):
        runtime_session.do_update()

    assert helper_calls == [
        {
            "caked_analysis_requested": False,
            "q_space_requested": False,
            "current_analysis_cache_sig": fixture["current_analysis_cache_sig"],
            "q_space_payload_geometry_sig": fixture["q_space_payload_geometry_sig"],
        }
    ]
    assert len(display_helper_calls) == 1
    assert display_helper_calls[0]["previous_primary_view_mode"] == "detector"
    assert display_helper_calls[0]["preserved_primary_limits"] == preserved_primary_limits
    np.testing.assert_array_equal(
        display_helper_calls[0]["caked_image"],
        fixture["caked_image"],
    )
    np.testing.assert_array_equal(
        display_helper_calls[0]["q_space_image"],
        fixture["q_space_image"],
    )


def test_apply_primary_figure_display_from_cached_results_preserves_hidden_analysis_payloads_during_detector_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _StopAtDetectorRedrawBoundary(RuntimeError):
        pass

    fixture = _install_matching_hidden_analysis_payload_state(
        monkeypatch,
        runtime_session,
        include_do_update_state=False,
    )
    detector_redraw_states: list[dict[str, np.ndarray | str]] = []

    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=False, current_background_display=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(
            set_visible=lambda *_args, **_kwargs: None,
            set_clim=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "apply_main_figure_axes_chrome",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    def _stop_at_detector_redraw(**kwargs) -> None:
        if kwargs.get("view_mode") == "detector":
            detector_redraw_states.append(
                {
                    "view_mode": str(kwargs["view_mode"]),
                    "caked_image": np.asarray(
                        runtime_session.simulation_runtime_state.last_caked_image_unscaled,
                        dtype=np.float64,
                    ).copy(),
                    "q_space_image": np.asarray(
                        runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
                        dtype=np.float64,
                    ).copy(),
                }
            )
            raise _StopAtDetectorRedrawBoundary()

    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        _stop_at_detector_redraw,
        raising=False,
    )

    with pytest.raises(_StopAtDetectorRedrawBoundary):
        runtime_session._apply_primary_figure_display_from_cached_results(
            "detector",
            ((0.0, 2.0), (2.0, 0.0)),
        )

    assert len(detector_redraw_states) == 1
    assert detector_redraw_states[0]["view_mode"] == "detector"
    np.testing.assert_array_equal(
        detector_redraw_states[0]["caked_image"],
        fixture["caked_image"],
    )
    np.testing.assert_array_equal(
        detector_redraw_states[0]["q_space_image"],
        fixture["q_space_image"],
    )


@pytest.mark.parametrize(
    ("view_mode", "payload_signature"),
    [
        ("caked", ("analysis", 3)),
        ("q_space", ("q-space", 7)),
    ],
)
def test_apply_primary_figure_display_from_cached_results_scales_cached_background_payload(
    monkeypatch,
    view_mode: str,
    payload_signature: tuple[str, int],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    stored_sources: list[tuple[str, np.ndarray, object | None]] = []
    preview_calls: list[dict[str, np.ndarray | None]] = []
    scale = 2.5
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)
    q_space_background = np.full((2, 2), 4.0, dtype=np.float64)
    caked_image = np.full((2, 2), 2.0, dtype=np.float64)
    caked_background = np.full((2, 2), 5.0, dtype=np.float64)
    image_artist = SimpleNamespace()
    background_artist = SimpleNamespace(
        set_visible=lambda *_args, **_kwargs: None,
        set_clim=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_q_space_image_unscaled=q_space_image,
            last_q_space_background_image_unscaled=q_space_background,
            last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
            last_q_space_payload_signature=(
                payload_signature if view_mode == "q_space" else ("q-space", 99)
            ),
            last_caked_image_unscaled=caked_image,
            last_caked_background_image_unscaled=caked_background,
            last_caked_extent=[0.0, 90.0, -180.0, 180.0],
            last_analysis_cache_sig=(
                payload_signature if view_mode == "caked" else ("analysis", 99)
            ),
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(visible=True),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(simulation_limits_user_override=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_RuntimeVar(0.0),
            simulation_max_var=_RuntimeVar(20.0),
            background_min_var=_RuntimeVar(0.0),
            background_max_var=_RuntimeVar(20.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _RuntimeVar(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "image_display", image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            set_aspect=lambda *_args, **_kwargs: None,
            set_xlabel=lambda *_args, **_kwargs: None,
            set_ylabel=lambda *_args, **_kwargs: None,
            set_title=lambda *_args, **_kwargs: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: view_mode,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: scale,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_auto_caked_limits",
        lambda image: (float(np.min(image)), float(np.max(image))),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda artist, source, **kwargs: stored_sources.append(
            (
                "primary" if artist is image_artist else "background",
                np.asarray(source, dtype=float).copy(),
                kwargs.get("source_signature"),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_canvas_interactions,
        "restore_axis_view",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "set_main_figure_axes_axis_visibility",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_main_figure_chrome,
        "apply_main_figure_axes_chrome",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    def _record_preview_sources(**kwargs):
        preview_calls.append(
            {
                "simulation_image": np.asarray(kwargs["simulation_image"], dtype=float).copy(),
                "background_image": (
                    None
                    if kwargs["background_image"] is None
                    else np.asarray(kwargs["background_image"], dtype=float).copy()
                ),
            }
        )
        return (kwargs["simulation_image"], kwargs["background_image"])

    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        _record_preview_sources,
        raising=False,
    )

    target_mode = runtime_session._apply_primary_figure_display_from_cached_results(
        "detector",
        ((0.0, 2.0), (2.0, 0.0)),
    )

    expected_primary = caked_image * scale if view_mode == "caked" else q_space_image * scale
    expected_background = (
        caked_background * scale if view_mode == "caked" else q_space_background * scale
    )

    assert target_mode == view_mode
    assert len(stored_sources) == 2
    assert stored_sources[0][0] == "primary"
    np.testing.assert_array_equal(stored_sources[0][1], expected_primary)
    assert stored_sources[0][2] == (view_mode, "primary", payload_signature, scale)
    assert stored_sources[1][0] == "background"
    np.testing.assert_array_equal(stored_sources[1][1], expected_background)
    assert stored_sources[1][2] == (view_mode, "background", payload_signature, scale)
    if view_mode == "caked":
        assert len(preview_calls) == 1
        np.testing.assert_array_equal(
            preview_calls[0]["simulation_image"],
            expected_primary,
        )
        np.testing.assert_array_equal(
            preview_calls[0]["background_image"],
            expected_background,
        )
    else:
        assert preview_calls == []


@pytest.mark.parametrize("stale_target", ["caked", "q_space"])
def test_invalidate_hidden_analysis_space_payloads_if_stale_clears_only_stale_hidden_payload(
    monkeypatch,
    stale_target: str,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    current_analysis_cache_sig = (("sim-cache", 1), ("bg-cache", 1))
    q_space_payload_geometry_sig = ("geom", 1)
    matching_q_space_signature = (current_analysis_cache_sig, q_space_payload_geometry_sig)
    caked_image = np.ones((2, 2), dtype=np.float64)
    q_space_image = np.full((2, 2), 3.0, dtype=np.float64)

    if stale_target == "caked":
        last_analysis_cache_sig = (("stale-sim", 0), ("stale-bg", 0))
        last_q_space_payload_signature = matching_q_space_signature
    else:
        last_analysis_cache_sig = current_analysis_cache_sig
        last_q_space_payload_signature = (current_analysis_cache_sig, ("old-geom", 0))

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_analysis_cache_sig=last_analysis_cache_sig,
            last_q_space_payload_signature=last_q_space_payload_signature,
            last_caked_image_unscaled=caked_image.copy(),
            last_caked_extent=[0.0, 1.0, -1.0, 1.0],
            last_caked_background_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
            last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
            last_caked_intersection_cache=("cache",),
            last_caked_intersection_cache_transform_bundle="bundle",
            last_q_space_image_unscaled=q_space_image.copy(),
            last_q_space_qr_values=np.array([0.1, 0.2], dtype=np.float64),
            last_q_space_qz_values=np.array([0.3, 0.4], dtype=np.float64),
            last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
            last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )

    if stale_target == "caked":
        assert runtime_session.simulation_runtime_state.last_caked_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_caked_extent is None
        assert runtime_session.simulation_runtime_state.last_caked_background_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_caked_radial_values is None
        assert runtime_session.simulation_runtime_state.last_caked_azimuth_values is None
        assert runtime_session.simulation_runtime_state.last_caked_intersection_cache is None
        assert (
            runtime_session.simulation_runtime_state.last_caked_intersection_cache_transform_bundle
            is None
        )
        np.testing.assert_array_equal(
            runtime_session.simulation_runtime_state.last_q_space_image_unscaled,
            q_space_image,
        )
        assert runtime_session.simulation_runtime_state.last_q_space_payload_signature == (
            current_analysis_cache_sig,
            q_space_payload_geometry_sig,
        )
    else:
        np.testing.assert_array_equal(
            runtime_session.simulation_runtime_state.last_caked_image_unscaled,
            caked_image,
        )
        assert runtime_session.simulation_runtime_state.last_q_space_image_unscaled is None
        assert runtime_session.simulation_runtime_state.last_q_space_qr_values is None
        assert runtime_session.simulation_runtime_state.last_q_space_qz_values is None
        assert runtime_session.simulation_runtime_state.last_q_space_extent is None
        assert (
            runtime_session.simulation_runtime_state.last_q_space_background_image_unscaled is None
        )
        assert runtime_session.simulation_runtime_state.last_q_space_payload_signature is None


@pytest.mark.parametrize(
    ("stale_target", "expected_calls"),
    [
        ("caked", [(True, False)]),
        ("q_space", [(False, True)]),
    ],
)
def test_invalidate_hidden_analysis_space_payloads_if_stale_clears_each_payload_once(
    monkeypatch,
    stale_target: str,
    expected_calls: list[tuple[bool, bool]],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    current_analysis_cache_sig = (("sim-cache", 1), ("bg-cache", 1))
    q_space_payload_geometry_sig = ("geom", 1)
    matching_q_space_signature = (current_analysis_cache_sig, q_space_payload_geometry_sig)
    invalidation_calls: list[tuple[bool, bool]] = []

    if stale_target == "caked":
        last_analysis_cache_sig = (("stale-sim", 0), ("stale-bg", 0))
        last_q_space_payload_signature = matching_q_space_signature
    else:
        last_analysis_cache_sig = current_analysis_cache_sig
        last_q_space_payload_signature = (current_analysis_cache_sig, ("old-geom", 0))

    state = SimpleNamespace(
        last_analysis_cache_sig=last_analysis_cache_sig,
        last_q_space_payload_signature=last_q_space_payload_signature,
        last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
        last_caked_extent=[0.0, 1.0, -1.0, 1.0],
        last_caked_background_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
        last_caked_radial_values=np.array([1.0, 2.0], dtype=np.float64),
        last_caked_azimuth_values=np.array([-1.0, 1.0], dtype=np.float64),
        last_caked_intersection_cache=("cache",),
        last_caked_intersection_cache_transform_bundle="bundle",
        last_q_space_image_unscaled=np.full((2, 2), 3.0, dtype=np.float64),
        last_q_space_qr_values=np.array([0.1, 0.2], dtype=np.float64),
        last_q_space_qz_values=np.array([0.3, 0.4], dtype=np.float64),
        last_q_space_extent=[-1.0, 1.0, -2.0, 2.0],
        last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        state,
        raising=False,
    )

    def _record_invalidation(*, clear_caked=False, clear_q_space=False) -> None:
        invalidation_calls.append((bool(clear_caked), bool(clear_q_space)))
        if clear_caked:
            state.last_caked_image_unscaled = None
            state.last_caked_extent = None
            state.last_caked_background_image_unscaled = None
            state.last_caked_radial_values = None
            state.last_caked_azimuth_values = None
            state.last_caked_intersection_cache = None
            state.last_caked_intersection_cache_transform_bundle = None
        if clear_q_space:
            state.last_q_space_image_unscaled = None
            state.last_q_space_qr_values = None
            state.last_q_space_qz_values = None
            state.last_q_space_extent = None
            state.last_q_space_background_image_unscaled = None
            state.last_q_space_payload_signature = None

    monkeypatch.setattr(
        runtime_session,
        "_invalidate_cached_analysis_space_payloads",
        _record_invalidation,
        raising=False,
    )

    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )
    runtime_session._invalidate_hidden_analysis_space_payloads_if_stale(
        caked_analysis_requested=False,
        q_space_requested=False,
        current_analysis_cache_sig=current_analysis_cache_sig,
        q_space_payload_geometry_sig=q_space_payload_geometry_sig,
    )

    assert invalidation_calls == expected_calls


def test_runtime_impl_forces_canvas_redraw_when_primary_view_mode_changes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "force_canvas_redraw=False," in source
    assert "_request_main_canvas_redraw(force_matplotlib=bool(force_canvas_redraw))" in source
    assert "target_primary_view_mode = (" in source
    assert "force_canvas_redraw=(previous_primary_view_mode != target_primary_view_mode)," in source


def test_runtime_impl_prepare_q_space_payload_uses_direct_detector_remap() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _prepare_q_space_display_payload(")
    helper_end = source.index("def _store_q_space_display_payload(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "detector_image: np.ndarray | None" in helper_source
    assert "convert_image_to_q_space(" in helper_source
    assert "caked_image_to_q_space_payload" not in helper_source
    assert "center_row_px=float(center_array[0])" in helper_source
    assert "center_col_px=float(center_array[1])" in helper_source


def test_runtime_impl_run_analysis_job_builds_q_space_from_detector_images() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _run_analysis_job(")
    helper_end = source.index("def _submit_async_analysis_job(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert 'q_space_requested = bool(job.get("q_space_requested", False))' in helper_source
    assert "if q_space_requested:" in helper_source
    assert (
        "sim_q_space = _prepare_q_space_display_payload(\n            sim_image," in helper_source
    )
    assert "bg_q_space = _prepare_q_space_display_payload(\n            bg_array," in helper_source
    assert (
        "sim_q_space = _prepare_q_space_display_payload(\n        sim_caked," not in helper_source
    )
    assert "bg_q_space = _prepare_q_space_display_payload(\n        bg_caked," not in helper_source


def test_runtime_impl_analysis_job_payload_tracks_q_space_intent() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"q_space_requested": q_space_requested,' in source


def test_runtime_impl_resolves_live_view_mode_before_q_space_restore() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "requested_view_mode = _current_app_shell_view_mode()" in source
    assert 'and requested_view_mode == "q_space"' in source
    assert "q_space_requested=q_space_requested," in source
    assert 'live_q_space_requested = bool(_current_app_shell_view_mode() == "q_space")' in source


def test_runtime_impl_q_space_view_mode_is_not_gated_on_caked_toggle() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _current_app_shell_view_mode() -> str:")
    helper_end = source.index("def _active_caked_primary_view()", helper_start)
    helper_source = source[helper_start:helper_end]

    assert 'if selected_mode == "q_space":\n        return "q_space"' in helper_source
    assert helper_source.index('if selected_mode == "q_space":') < helper_source.index(
        "if show_caked:"
    )


def test_runtime_impl_single_viewport_startup_uses_embedded_matplotlib_tk() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    assert "figure_canvas_cls = _get_tk_figure_canvas_cls()" in source
    assert "matplotlib_canvas = figure_canvas_cls(" in source
    assert "master=app_shell_view_state.canvas_frame" in source
    assert "_set_runtime_canvas(matplotlib_canvas)" in source
    assert "_initialize_runtime_plot_block_03()" not in source


def test_initialize_runtime_plot_block_02_projects_startup_rasters_without_viewport_workflow(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _FakeArtist:
        def __init__(self, array: np.ndarray, extent: tuple[float, float, float, float]) -> None:
            self._array = np.asarray(array, dtype=float)
            self._extent = tuple(float(value) for value in extent)
            self.origin = "upper"
            self.data = None

        def get_array(self):
            return self._array

        def get_extent(self):
            return self._extent

        def set_extent(self, extent):
            self._extent = tuple(float(value) for value in extent)

        def set_data(self, data):
            self.data = np.asarray(data, dtype=float)

    projection_calls: list[dict[str, object]] = []

    def _project_raster_to_view(source, **kwargs):
        projection_calls.append(dict(kwargs))
        return SimpleNamespace(
            extent=tuple(float(value) for value in kwargs["extent"]),
            image=np.asarray(source, dtype=float),
        )

    image_display = _FakeArtist(np.arange(4, dtype=float).reshape(2, 2), (0.0, 1.0, 1.0, 0.0))
    background_display = _FakeArtist(np.ones((2, 2), dtype=float), (0.0, 1.0, 1.0, 0.0))
    overlay_display = _FakeArtist(np.zeros((2, 2), dtype=float), (0.0, 1.0, 1.0, 0.0))

    monkeypatch.setattr(runtime_session, "_RUNTIME_CONTROLS_INITIALIZED", True, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 2, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(last_caked_extent=None, last_q_space_extent=None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "ax",
        SimpleNamespace(
            get_xlim=lambda: (0.0, 2.0),
            get_ylim=lambda: (2.0, 0.0),
            bbox=SimpleNamespace(width=200.0, height=200.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_display_projection",
        SimpleNamespace(project_raster_to_view=_project_raster_to_view),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_main_display_raster_size_limit",
        lambda: 64,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_display", image_display, raising=False)
    monkeypatch.setattr(runtime_session, "background_display", background_display, raising=False)
    monkeypatch.setattr(
        runtime_session, "integration_region_overlay", overlay_display, raising=False
    )

    runtime_session._initialize_runtime_plot_block_02()

    assert len(projection_calls) == 3
    assert image_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert background_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert overlay_display.get_extent() == (0.0, 2.0, 2.0, 0.0)
    assert image_display.data is not None
    assert background_display.data is not None
    assert overlay_display.data is not None


def test_run_analysis_job_can_emit_sim_q_space_without_caked_payload(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 7,
            "signature": ("analysis", 1),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 1),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 1),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": True,
        }
    )

    assert result["sim_caked"] is None
    assert result["bg_caked"] is None
    assert isinstance(result["sim_q_space"], dict)
    assert result["bg_q_space"] is None
    assert np.asarray(result["sim_q_space"]["image"], dtype=float).size > 0
    assert np.asarray(result["sim_q_space"]["qr"], dtype=float).size == 48
    assert np.asarray(result["sim_q_space"]["qz"], dtype=float).size == 40


def test_toggle_caked_2d_legacy_path_overrides_stale_q_space_shell_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("q_space")
    show_caked_var = _Var(True)
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=show_caked_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"


def test_toggle_caked_2d_skips_qr_overlay_invalidation_when_overlays_are_off(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    invalidations: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **kwargs: invalidations.append(dict(kwargs)),
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert invalidations == []


def test_toggle_caked_2d_reselects_current_hkl_peak_for_new_view(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    reselect_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_runtime_callbacks",
        SimpleNamespace(
            reselect_current_peak=lambda: reselect_calls.append("reselect") or True
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert reselect_calls == ["reselect"]


def test_toggle_caked_2d_hides_selected_peak_when_reselect_fails(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    class _Marker:
        def __init__(self) -> None:
            self.visible = True

        def set_visible(self, value: object) -> None:
            self.visible = bool(value)

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(True)
    marker = _Marker()
    redraw_calls: list[bool] = []

    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(
            show_caked_2d_var=show_caked_var,
            show_qz_rods_var=_Var(False),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_overlay_actions_view_state",
        SimpleNamespace(show_geometry_overlays_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "peak_selection_runtime_callbacks",
        SimpleNamespace(reselect_current_peak=lambda: False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "selected_peak_marker",
        marker,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_calls.append(bool(force_matplotlib)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_analysis_range_values",
        lambda: {"integrate_qz_rods": False, "qr_half_width": 0.0},
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_apply_main_caked_view_toggle", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)

    runtime_session.toggle_caked_2d()

    assert shell_mode_var.get() == "caked"
    assert marker.visible is False
    assert redraw_calls == [False]


def test_set_persistent_view_mode_keeps_q_space_when_enabling_caked_data(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: object) -> None:
            self._value = value

        def get(self) -> object:
            return self._value

        def set(self, value: object) -> None:
            self._value = value

    shell_mode_var = _Var("detector")
    show_caked_var = _Var(False)
    schedule_calls: list[str] = []
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_caked_2d_var=show_caked_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "app_shell_view_state",
        SimpleNamespace(view_mode_var=shell_mode_var),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_peak_selection_state",
        SimpleNamespace(pick_armed=False),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(unscaled_image=np.ones((1, 1), dtype=np.float64)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "gui_views",
        SimpleNamespace(
            set_app_shell_view_mode=lambda view_state, mode: view_state.view_mode_var.set(mode)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_qr_cylinder_overlay_view_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runtime_session, "_refresh_run_status_bar", lambda: None)
    monkeypatch.setattr(runtime_session, "_sync_center_marker", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_session, "_render_analysis_peak_overlays", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime_session, "schedule_update", lambda: schedule_calls.append("schedule")
    )

    runtime_session._set_persistent_view_mode("q_space")

    assert show_caked_var.get() is True
    assert shell_mode_var.get() == "q_space"
    assert schedule_calls == ["schedule"]


def test_apply_scale_factor_to_existing_results_can_force_canvas_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    redraw_calls: list[bool] = []
    synced_modes: list[str | None] = []
    stored_sources: list[np.ndarray] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    fake_image_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    fake_background_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runtime_session, "image_display", fake_image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        fake_background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: float(default),
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda _artist, source, **_kwargs: stored_sources.append(
            np.asarray(source, dtype=float).copy()
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **kwargs: synced_modes.append(kwargs.get("view_mode")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_calls.append(bool(force_matplotlib)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=True,
        update_chi_square=False,
    )

    assert synced_modes == ["caked"]
    assert redraw_calls == [True]
    assert len(stored_sources) >= 1


@pytest.mark.parametrize(
    ("view_mode", "payload_signature"),
    [
        ("caked", ("analysis", 3)),
        ("q_space", ("q-space", 7)),
    ],
)
def test_apply_scale_factor_to_existing_results_uses_semantic_analysis_source_signatures(
    monkeypatch,
    view_mode: str,
    payload_signature: tuple[str, int],
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    stored_signatures: list[tuple[str, object | None]] = []
    fake_image_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    fake_background_artist = SimpleNamespace(set_visible=lambda *_args, **_kwargs: None)
    scale = 2.0

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=np.full((2, 2), 3.0, dtype=np.float64),
            last_q_space_background_image_unscaled=np.full((2, 2), 4.0, dtype=np.float64),
            last_q_space_payload_signature=(
                payload_signature if view_mode == "q_space" else ("q-space", 99)
            ),
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=np.full((2, 2), 5.0, dtype=np.float64),
            last_analysis_cache_sig=(
                payload_signature if view_mode == "caked" else ("analysis", 99)
            ),
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=True,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(10.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "image_display", fake_image_artist, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        fake_background_artist,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: view_mode,
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: scale,
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda artist, _source, **kwargs: stored_signatures.append(
            (
                "primary" if artist is fake_image_artist else "background",
                kwargs.get("source_signature"),
            )
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=False,
        update_chi_square=False,
    )

    assert stored_signatures == [
        ("primary", (view_mode, "primary", payload_signature, scale)),
        ("background", (view_mode, "background", payload_signature, scale)),
    ]


def test_apply_scale_factor_to_existing_results_uses_runtime_image_signature_for_chi_square(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _install_idle_main_figure_preview_state(monkeypatch, runtime_session)

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    chi_state = {"buffer_sig": ("stale",)}
    dirty_calls: list[str] = []
    runtime_signature = ("runtime-sig", 42)

    monkeypatch.setattr(
        runtime_session, "last_unscaled_image_signature", ("wrong", 999), raising=False
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=runtime_signature,
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=None,
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "detector",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: 1.5,
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)
    monkeypatch.setattr(runtime_session, "chi_square_state", chi_state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_mark_chi_square_dirty",
        lambda: dirty_calls.append("dirty"),
        raising=False,
    )

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=False,
        update_chi_square=False,
    )

    assert chi_state["buffer_sig"][0] == runtime_signature
    assert chi_state["buffer_sig"][1] == 1.5
    assert dirty_calls == ["dirty"]


def test_apply_scale_factor_to_existing_results_clears_pending_preview_before_sync_and_redraw(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: float | bool) -> None:
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    geometry_runtime_state = SimpleNamespace(
        _canvas_preview_limits=((4.0, 8.0), (9.0, 2.0)),
        _canvas_pan_session={"drag": True},
    )
    sync_states: list[tuple[object, object, str | None]] = []
    redraw_states: list[tuple[object, object, bool]] = []
    cleared_tokens: list[object] = []

    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        geometry_runtime_state,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_clear_legacy_main_matplotlib_preview_view",
        lambda *, redraw=True: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            unscaled_image=np.ones((2, 2), dtype=np.float64),
            last_unscaled_image_signature=("sim", 1),
            last_q_space_image_unscaled=None,
            last_q_space_background_image_unscaled=None,
            last_caked_image_unscaled=np.full((2, 2), 2.0, dtype=np.float64),
            last_caked_background_image_unscaled=None,
            last_1d_integration_data={
                "radials_sim": None,
                "intensities_2theta_sim": None,
                "azimuths_sim": None,
                "intensities_azimuth_sim": None,
                "radials_bg": None,
                "intensities_2theta_bg": None,
                "azimuths_bg": None,
                "intensities_azimuth_bg": None,
            },
            caked_limits_user_override=False,
            interaction_drag_active=True,
            interaction_drag_requires_settled_update=True,
            interaction_settle_token="pending-preview",
            main_matplotlib_overlays_suspended=True,
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "root", object(), raising=False)
    monkeypatch.setattr(
        runtime_session.gui_controllers,
        "clear_tk_after_token",
        lambda _root, token: cleared_tokens.append(token),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_refresh_run_status_bar",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(
            visible=False,
            current_background_index=0,
            backend_rotation_k=0,
            backend_flip_x=False,
            backend_flip_y=False,
            current_background_display=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_view_state",
        SimpleNamespace(
            simulation_min_var=_Var(0.0),
            simulation_max_var=_Var(10.0),
            background_min_var=_Var(0.0),
            background_max_var=_Var(1.0),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "display_controls_state",
        SimpleNamespace(
            simulation_limits_user_override=False,
            background_limits_user_override=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "analysis_view_controls_view_state",
        SimpleNamespace(show_1d_var=_Var(False)),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "vmin_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "vmax_caked_var", _Var(0.0), raising=False)
    monkeypatch.setattr(
        runtime_session,
        "colorbar_main",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "caked_colorbar",
        SimpleNamespace(update_normal=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "image_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "background_display",
        SimpleNamespace(set_visible=lambda *_args, **_kwargs: None),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_resolved_primary_analysis_display_mode",
        lambda: "caked",
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_scale_factor_value",
        lambda default=1.0: float(default),
    )
    monkeypatch.setattr(
        runtime_session,
        "_ensure_global_image_buffer_shape",
        lambda _image: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "global_image_buffer",
        np.zeros((2, 2), dtype=np.float64),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_caked_roi_preview_display_sources",
        lambda **kwargs: (kwargs["simulation_image"], kwargs["background_image"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_update_simulation_sliders_from_image",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_primary_raster_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_sync_primary_raster_geometry",
        lambda **kwargs: sync_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                kwargs.get("view_mode"),
            )
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_apply_intensity_display_range",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_request_main_canvas_redraw",
        lambda *, force_matplotlib=False: redraw_states.append(
            (
                geometry_runtime_state._canvas_preview_limits,
                geometry_runtime_state._canvas_pan_session,
                bool(force_matplotlib),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_update_chi_square_display", lambda: None)

    runtime_session.apply_scale_factor_to_existing_results(
        update_1d=False,
        force_canvas_redraw=True,
        update_chi_square=False,
    )

    assert cleared_tokens == ["pending-preview"]
    assert geometry_runtime_state._canvas_preview_limits is None
    assert geometry_runtime_state._canvas_pan_session is None
    assert sync_states == [(None, None, "caked")]
    assert redraw_states == [(None, None, True)]


def test_run_analysis_job_skips_q_space_for_caked_only_requests(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    q_space_calls: list[np.ndarray | None] = []

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda res2, **_kwargs: (
            None
            if res2 is None
            else {
                "image": np.ones((2, 2), dtype=np.float64),
                "radial": np.array([1.0, 2.0], dtype=np.float64),
                "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
                "transform_bundle": None,
                "detector_shape": None,
                "extent": [0.0, 1.0, -1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda detector_image, **_kwargs: q_space_calls.append(
            None if detector_image is None else np.asarray(detector_image, dtype=np.float64)
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 8,
            "signature": ("analysis", 2),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 2),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 2),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": False,
        }
    )

    assert isinstance(result["sim_caked"], dict)
    assert result["bg_caked"] is None
    assert result["sim_q_space"] is None
    assert result["bg_q_space"] is None
    assert q_space_calls == []


def test_run_analysis_job_caked_only_request_ignores_q_space_failure(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda image, _ai, **_kwargs: {
            "shape": tuple(np.asarray(image).shape),
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda res2, **_kwargs: (
            None
            if res2 is None
            else {
                "image": np.ones((2, 2), dtype=np.float64),
                "radial": np.array([1.0, 2.0], dtype=np.float64),
                "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
                "transform_bundle": None,
                "detector_shape": None,
                "extent": [0.0, 1.0, -1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("q-space failure")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)

    result = runtime_session._run_analysis_job(
        {
            "job_id": 9,
            "signature": ("analysis", 3),
            "epoch": 3,
            "image": np.eye(8, dtype=np.float64),
            "background_image": None,
            "npt_rad": 48,
            "npt_azim": 40,
            "is_preview": False,
            "cached_bg_res2": None,
            "cached_bg_caked": None,
            "intersection_cache": None,
            "sim_cache_sig": ("sim", 3),
            "bg_cache_sig": None,
            "sim_caking_sig": ("cake", 3),
            "bg_caking_sig": None,
            "distance_m": 0.5,
            "center": np.array([4.0, 4.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.0,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "q_space_requested": False,
        }
    )

    assert isinstance(result["sim_caked"], dict)
    assert result["sim_q_space"] is None
    assert result["bg_q_space"] is None


def test_restore_caked_payload_rebuilds_q_space_from_live_q_space_mode(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class _Var:
        def __init__(self, value: float) -> None:
            self._value = value

        def get(self) -> float:
            return float(self._value)

    background_native = np.arange(16, dtype=np.float64).reshape(4, 4)
    q_space_inputs: list[np.ndarray] = []
    stored_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            analysis_preview_bins=(8, 6),
            last_res2_sim=object(),
            last_res2_background=object(),
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
            "transform_bundle": None,
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda detector_image, **_kwargs: (
            q_space_inputs.append(np.asarray(detector_image, dtype=np.float64).copy())
            or {
                "image": np.asarray(detector_image, dtype=np.float64),
                "qr": np.array([0.0, 1.0], dtype=np.float64),
                "qz": np.array([0.0, 1.0], dtype=np.float64),
                "sum_signal": np.ones((2, 2), dtype=np.float64),
                "sum_normalization": np.ones((2, 2), dtype=np.float64),
                "count": np.ones((2, 2), dtype=np.float64),
                "extent": [0.0, 1.0, 0.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        runtime_session,
        "_get_current_background_backend",
        lambda: (_ for _ in ()).throw(AssertionError("restore should use native background")),
    )
    monkeypatch.setattr(
        runtime_session, "_get_current_background_native", lambda: background_native
    )
    monkeypatch.setattr(runtime_session, "center_x_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", _Var(2.0), raising=False)
    monkeypatch.setattr(runtime_session, "corto_detector_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "Gamma_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "chi_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "psi_z_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "cor_angle_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zs_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "zb_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.24, raising=False)
    monkeypatch.setattr(runtime_session, "psi", 0.0, raising=False)
    monkeypatch.setattr(
        runtime_session, "_current_effective_theta_initial", lambda strict_count=False: 0.0
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "q_space")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=True,
        q_space_requested=False,
    )

    assert restored is True
    np.testing.assert_array_equal(
        runtime_session.simulation_runtime_state.last_caked_background_image_unscaled,
        np.ones((2, 2), dtype=np.float64),
    )
    assert len(q_space_inputs) == 2
    np.testing.assert_array_equal(q_space_inputs[1], background_native)
    assert len(stored_payloads) == 1


def test_analysis_display_payload_ready_requires_visible_background_payload() -> None:
    payload_ready = _load_runtime_session_function("_analysis_display_payload_ready")

    assert (
        payload_ready(
            show_caked_2d=True,
            q_space_requested=False,
            background_visible=True,
            caked_image=np.ones((2, 2), dtype=np.float64),
            caked_extent=[0.0, 1.0, 0.0, 1.0],
            caked_background_image=None,
            q_space_image=None,
            q_space_extent=None,
            q_space_background_image=None,
        )
        is False
    )
    assert (
        payload_ready(
            show_caked_2d=True,
            q_space_requested=False,
            background_visible=True,
            caked_image=np.ones((2, 2), dtype=np.float64),
            caked_extent=[0.0, 1.0, 0.0, 1.0],
            caked_background_image=np.ones((2, 2), dtype=np.float64),
            q_space_image=None,
            q_space_extent=None,
            q_space_background_image=None,
        )
        is True
    )
    assert (
        payload_ready(
            show_caked_2d=False,
            q_space_requested=True,
            background_visible=True,
            caked_image=None,
            caked_extent=None,
            caked_background_image=None,
            q_space_image=np.ones((2, 2), dtype=np.float64),
            q_space_extent=[0.0, 1.0, 0.0, 1.0],
            q_space_background_image=None,
        )
        is False
    )


def test_analysis_payload_ready_after_restore_attempt_restores_once_when_needed(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    def _restore() -> None:
        restore_calls.append("restore")
        runtime_session.simulation_runtime_state.last_caked_background_image_unscaled = np.ones(
            (2, 2), dtype=np.float64
        )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=_restore,
    )

    assert ready is True
    assert restore_calls == ["restore"]


def test_analysis_payload_ready_after_restore_attempt_stays_false_when_restore_noops(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=lambda: restore_calls.append("restore"),
    )

    assert ready is False
    assert restore_calls == ["restore"]


def test_analysis_payload_ready_after_restore_attempt_skips_restore_when_ready(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    restore_calls: list[str] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_caked_extent=[0.0, 1.0, 0.0, 1.0],
            last_caked_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_image_unscaled=None,
            last_q_space_extent=None,
            last_q_space_background_image_unscaled=None,
        ),
    )

    ready = runtime_session._analysis_payload_ready_after_restore_attempt(
        analysis_result_current=True,
        show_caked_2d=True,
        q_space_requested=False,
        background_visible=True,
        restore_payload=lambda: restore_calls.append("restore"),
    )

    assert ready is True
    assert restore_calls == []


def test_restore_caked_payload_clears_stale_q_space_outside_live_q_space_mode(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    stored_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_res2_sim=object(),
            last_res2_background=object(),
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=np.ones((2, 2), dtype=np.float64),
            last_q_space_background_image_unscaled=np.ones((2, 2), dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth": np.array([-1.0, 1.0], dtype=np.float64),
            "transform_bundle": None,
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_set_live_caked_transform_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restore should skip q-space rebuild outside q-space mode")
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "detector")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=True,
        q_space_requested=True,
    )

    assert restored is True
    assert stored_payloads == [{"sim_payload": None, "bg_payload": None}]


def test_runtime_impl_hkl_pick_disarms_manual_geometry_and_preview_modes() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "deactivate_conflicting_modes_factory=lambda: (" in source
    assert "_set_geometry_manual_pick_mode(False)" in source
    assert "_set_geometry_preview_exclude_mode(False)" in source


def test_runtime_impl_hkl_pick_refreshes_mode_banner_without_viewport_hooks() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _handle_hkl_pick_mode_changed(_armed: bool) -> None:" in source
    assert 'globals().get("_refresh_fast_viewer_runtime_mode")' not in source
    assert "on_hkl_pick_mode_changed_factory=lambda: _handle_hkl_pick_mode_changed" in source


def test_runtime_impl_initializes_only_embedded_matplotlib_tk_primary_viewport() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "figure_canvas_cls = _get_tk_figure_canvas_cls()" in source
    assert "matplotlib_canvas = figure_canvas_cls(" in source
    assert "canvas = matplotlib_canvas" in source
    assert "_set_runtime_canvas(matplotlib_canvas)" in source
    assert "activate_runtime_primary_viewport(" not in source
    assert '"key": "primary_viewport_backend"' not in source
    assert "set_background_alpha=background_display.set_alpha" in source


def test_runtime_impl_shares_pick_hkl_live_cache_with_manual_qr_picker() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "geometry_manual_cache_workflow = (" in source
    assert "simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params" in source


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
    assert (
        "PREVIEW_CALCULATIONS_ENABLED\n                and bool(live_geometry_preview_var.get())"
        in source
    )


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
    assert "_request_main_canvas_redraw(force_matplotlib=True)" in startup_block
    assert startup_block.index("do_update()") < startup_block.index(
        "_request_main_canvas_redraw(force_matplotlib=True)"
    )
    assert startup_block.rindex("root.update_idletasks()") > startup_block.index(
        "_request_main_canvas_redraw(force_matplotlib=True)"
    )
    assert "start_exact_cake_numba_warmup_in_background()" not in startup_block
    assert "_run_simulation_generation_job(" in block
    assert 'progress_label.config(text="Computing initial simulation...")' in block


def test_runtime_impl_defers_exact_cake_numba_warmup_until_after_startup_work() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if analysis_sig is not None:" in source
    assert "if simulation_runtime_state.stored_sim_image is not None:" in source
    assert source.count("_schedule_exact_cake_numba_warmup_once()") >= 2
    assert source.count("_schedule_forward_simulation_numba_warmup_once()") >= 2
    assert source.count("_schedule_qr_rod_simulation_numba_warmup_once()") >= 2


def test_runtime_impl_distinguishes_preview_and_full_worker_jobs() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'str(payload.get("job_kind", "full")),' in source
    assert "def _simulation_result_matches_signature(" in source
    assert "def _simulation_result_superseded_by_queued_job(" in source
    assert "superseded = _simulation_result_superseded_by_queued_job(result, queued_job)" in source
    assert (
        "if isinstance(ready_result, dict) and not _simulation_result_matches_signature(" in source
    )
    assert '_promote_queued_simulation_job(reason="previous_ready_result_consumed")' in source


def test_runtime_impl_keeps_1d_updates_gated_on_intensity_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if (\n                one_d_analysis_requested" in source


def test_runtime_impl_places_qr_cylinder_mode_in_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"key": "qr_cylinder_mode"' in source
    assert '"key": "display_raster_size"' in source
    assert '"label": "display px"' in source
    assert "_refresh_main_display_raster_projection" in source
    assert "QR_CYLINDER_DISPLAY_MODE_REPLACE" in source
    assert "parent=app_shell_view_state.match_peak_tools_frame" in source


def test_runtime_impl_moves_analysis_view_options_and_auto_match_to_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '"key": "fast_viewer"' not in source
    assert '"key": "log_display"' in source
    assert '"key": "show_qz_rods"' in source
    assert '"key": "clear_integration_region"' in source
    assert '"key": "auto_match_scale"' in source
    assert 'control_type": "check"' in source
    assert 'control_type": "button"' in source
    assert "parent=None" in source
    assert "RA_SIM_FAST_VIEWER" not in source
    assert "control_locked=True" not in source
    assert "display_controls_view_state.fast_viewer_checkbutton = (" not in source
    assert (
        "display_controls_view_state.simulation_controls_frame"
        not in source[
            source.index("def _auto_match_scale_factor_to_radial_peak(") : source.index(
                "def _update_chi_square_display("
            )
        ]
    )
    assert "def _current_qr_cylinder_caked_band_masks() -> dict[str, object] | None:" in source
    assert "def _invalidate_qr_cylinder_band_cache() -> None:" in source


def test_runtime_session_current_analysis_range_values_preserve_rod_controls(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            tth_min_value=1.5,
            tth_max_value=55.0,
            phi_min_value=-12.0,
            phi_max_value=18.0,
            integrate_qz_rods_value=True,
            qr_half_width_value=0.03125,
        ),
        raising=False,
    )
    for name in (
        "tth_min_var",
        "tth_max_var",
        "phi_min_var",
        "phi_max_var",
        "integrate_qz_rods_var",
        "qr_half_width_var",
    ):
        monkeypatch.setitem(runtime_session.__dict__, name, None)

    values = runtime_session._current_analysis_range_values()

    assert values == {
        "tth_min": 1.5,
        "tth_max": 55.0,
        "phi_min": -12.0,
        "phi_max": 18.0,
        "integrate_qz_rods": True,
        "qr_half_width": 0.03125,
    }


def test_runtime_session_current_qr_cylinder_caked_band_masks_uses_cached_rod_controls(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    build_calls: list[float] = []

    monkeypatch.setattr(
        runtime_session,
        "integration_range_controls_view_state",
        SimpleNamespace(
            integrate_qz_rods_value=True,
            qr_half_width_value=0.03125,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.asarray([10.0, 20.0], dtype=float),
            last_caked_azimuth_values=np.asarray([-5.0, 5.0], dtype=float),
            ai_cache={},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(qr_cylinder_band_cache={"signature": None, "result": None}),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_qr_cylinder_caked_projection_context",
        lambda: {"projection": "ok"},
    )
    monkeypatch.setattr(
        runtime_session,
        "active_qr_cylinder_overlay_entries_factory",
        lambda: [{"key": "rod-a", "qr": 1.25}],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "qr_cylinder_overlay_render_config_factory",
        lambda: runtime_session.gui_qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
            render_in_caked_space=False,
            image_size=64,
            display_rotate_k=-1,
            center_col=10.0,
            center_row=11.0,
            distance_cor_to_detector=123.0,
            gamma_deg=1.5,
            Gamma_deg=2.5,
            chi_deg=3.5,
            psi_deg=4.5,
            psi_z_deg=5.5,
            zs=6.5,
            zb=7.5,
            theta_initial_deg=8.5,
            cor_angle_deg=9.5,
            pixel_size_m=1.0e-4,
            wavelength=1.54,
            n2=1.1 + 0.0j,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "build_qr_cylinder_overlay_signature",
        lambda entries, **kwargs: (
            tuple(entry["key"] for entry in entries),
            bool(kwargs["config"].render_in_caked_space),
        ),
    )
    monkeypatch.setattr(
        runtime_session.gui_qr_cylinder_overlay,
        "build_qr_cylinder_caked_band_masks",
        lambda entries, **kwargs: (
            build_calls.append(float(kwargs["delta_qr"]))
            or {"union_mask": np.ones((2, 2), dtype=bool)}
        ),
    )
    for name in ("integrate_qz_rods_var", "qr_half_width_var"):
        monkeypatch.setitem(runtime_session.__dict__, name, None)

    result = runtime_session._current_qr_cylinder_caked_band_masks()

    assert build_calls == [0.03125]
    np.testing.assert_array_equal(result["union_mask"], np.ones((2, 2), dtype=bool))


def test_runtime_session_current_qr_cylinder_caked_projection_context_prefers_live_bundle_shape(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    class FakeBundle:
        def __init__(self):
            self.detector_shape = (9, 7)
            self.raw_azimuth_deg = np.asarray([-90.0, 0.0], dtype=float)

    bundle = FakeBundle()

    monkeypatch.setattr(runtime_session, "CakeTransformBundle", FakeBundle)
    monkeypatch.setattr(runtime_session, "_current_live_caked_transform_bundle", lambda: bundle)
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            last_caked_radial_values=np.asarray([10.0, 20.0], dtype=float),
            last_caked_azimuth_values=np.asarray([-5.0, 5.0], dtype=float),
            ai_cache={"ai": None, "detector_shape": (99, 88)},
            unscaled_image=np.ones((4, 4), dtype=np.float64),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_normalize_geometry_fit_caked_view_payload",
        lambda payload, **kwargs: dict(payload),
    )

    context = runtime_session._current_qr_cylinder_caked_projection_context()

    assert context["detector_shape"] == (9, 7)
    np.testing.assert_allclose(context["raw_azimuth_axis"], [-90.0, 0.0])


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
    assert (
        '_analysis_tab_trace_add("write", _handle_analysis_integration_visibility_change)' in source
    )


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
    main_start = source.index(
        'def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):'
    )
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

    import_block_start = runtime_source.index(
        "from ra_sim.io.file_parsing import parse_poni_file, Open_ASC"
    )
    import_block_end = runtime_source.index(
        "from ra_sim.io.data_loading import (", import_block_start
    )
    import_block = runtime_source[import_block_start:import_block_end]
    dataframe_helper_start = runtime_source.index("def _build_intensity_dataframes(")
    dataframe_helper_end = runtime_source.index(
        "def _current_primary_cif_path()", dataframe_helper_start
    )
    dataframe_helper = runtime_source[dataframe_helper_start:dataframe_helper_end]
    azimuthal_helper_start = runtime_source.index("def _show_azimuthal_radial_plot_demo() -> None:")
    azimuthal_helper_end = runtime_source.index("app_shell_view_state = app_state.app_shell_view")
    azimuthal_helper = runtime_source[azimuthal_helper_start:azimuthal_helper_end]

    assert "from ra_sim.utils.tools import (" not in import_block
    assert "from ra_sim.utils.diffraction_tools import (" in import_block
    assert "build_intensity_dataframes," not in import_block
    assert "view_azimuthal_radial," not in import_block
    assert "detect_blobs," not in import_block
    assert (
        "from ra_sim.utils.diffraction_tools import build_intensity_dataframes" in dataframe_helper
    )
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
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
        ):
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
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        pending.extend(ast.iter_child_nodes(node))
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
            offenders.extend(f"{rel_path}:{line}:{field_name}" for field_name, line in hits)

    assert offenders == []
