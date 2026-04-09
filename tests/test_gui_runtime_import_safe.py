import importlib
import py_compile
import sys
from pathlib import Path
from types import SimpleNamespace

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
CLI_SOURCE_PATH = Path(__file__).resolve().parent.parent / "ra_sim" / "cli.py"


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


def test_runtime_impl_routes_legacy_fit_logs_through_debug_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert 'os.environ.setdefault("RA_SIM_DEBUG", "0")' not in source
    assert "if gui_geometry_fit.geometry_fit_log_files_enabled():" in source
    assert "if mosaic_fit_log_files_enabled():" in source
    assert "log_file={log_path if log_path is not None else 'disabled'}" in source


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


def test_runtime_impl_preserves_wrapped_phi_ranges_for_detector_drags() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "draw_idle_factory=lambda: (" in source
    assert "_request_main_canvas_redraw" in source
    assert "phi_min = float(phi_min_var.get())" in source
    assert "phi_max = float(phi_max_var.get())" in source
    assert "gui_integration_range_drag.detector_phi_mask(" in source
    assert "if phi_max < phi_min and azimuth_sub.size:" in source


def test_runtime_impl_falls_back_to_detector_image_when_caked_cache_is_missing() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    show_caked_image_assignment = (
        "show_caked_image = bool(show_caked_requested and caked_payload_available)"
    )

    assert "show_caked_requested = bool(" in source
    assert "caked_payload_available = (" in source
    assert show_caked_image_assignment in source
    assert "if not show_caked_image:" in source
    assert "image_display.set_data(global_image_buffer)" in source
    assert "background_display.set_extent([0, image_size, image_size, 0])" in source


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
    assert 'preserve=(previous_primary_view_mode == "caked")' in source
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


def test_runtime_impl_keeps_qr_overlay_live_during_background_updates() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _clear_deferred_overlays(*, clear_qr_overlay: bool = True) -> None:" in source
    assert "if _live_interaction_active():" in source
    assert "qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)" in source
    assert "_clear_deferred_overlays(clear_qr_overlay=False)" in source


def test_runtime_impl_invalidates_qr_overlay_before_view_mode_toggles() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "def _invalidate_qr_cylinder_overlay_view_state(*, clear_artists: bool) -> None:" in source
    assert "gui_qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(" in source
    assert "def toggle_caked_2d() -> None:" in source
    assert "_invalidate_qr_cylinder_overlay_view_state(clear_artists=True)" in source
    assert "_toggle_caked_2d_impl()" in source


def test_runtime_impl_uses_bound_caked_projection_callback_for_live_overlay_coords() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _native_detector_coords_to_live_caked_coords(")
    helper_end = source.index("def _scattering_angles_to_detector_pixel(", helper_start)
    helper_source = source[helper_start:helper_end]

    assert "return _native_detector_coords_to_caked_display_coords(\n        col,\n        row,\n    )" in helper_source
    assert "ai=" not in helper_source


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
    assert "ax.set_title('2D Caked Position Preview')" in source
    assert "ax.set_title('2D Caked Position Preview (1D paused)')" in source
    assert "Detector Preview While Caked Position Preview Loads" in source


def test_runtime_impl_disables_preview_calculations_in_runtime_update_paths() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "PREVIEW_CALCULATIONS_ENABLED = False" in source
    assert "if not PREVIEW_CALCULATIONS_ENABLED:" in source
    assert "desired_analysis_preview = bool(" in source
    assert "PREVIEW_CALCULATIONS_ENABLED\n        and analysis_requested" in source
    assert "PREVIEW_CALCULATIONS_ENABLED\n                and bool(live_geometry_preview_var.get())" in source


def test_runtime_impl_keeps_1d_updates_gated_on_intensity_accumulation() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "if (\n                one_d_analysis_requested" in source


def test_runtime_impl_places_qr_cylinder_mode_in_quick_controls() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert '\"key\": \"qr_cylinder_mode\"' in source
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


def test_runtime_impl_lazy_mounts_analysis_surfaces() -> None:
    source = RUNTIME_SESSION_SOURCE_PATH.read_text(encoding="utf-8")

    assert "analysis_surfaces_initialized = False" in source
    assert "def ensure_analysis_surfaces_initialized() -> None:" in source
    assert "_show_analysis_tab_lazy_placeholders()" in source
    assert "_mount_analysis_figure(app_shell_view_state.plot_frame_1d)" not in source
    assert "ensure_analysis_surfaces_initialized()" in source
