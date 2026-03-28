import os
from types import SimpleNamespace

import pytest

from ra_sim.gui import bootstrap


def test_extract_bundle_arg_supports_inline_and_separate_values() -> None:
    assert bootstrap.extract_bundle_arg(["--bundle=test.npz"]) == "test.npz"
    assert bootstrap.extract_bundle_arg(["--bundle", "test.npz"]) == "test.npz"
    assert bootstrap.extract_bundle_arg(["--bundle"]) is None


def test_should_forward_to_cli_only_for_non_launcher_commands() -> None:
    assert bootstrap.should_forward_to_cli(["simulate", "--out", "output.png"])
    assert not bootstrap.should_forward_to_cli(["gui"])
    assert not bootstrap.should_forward_to_cli(["calibrant"])
    assert not bootstrap.should_forward_to_cli(["--help"])


def test_parse_launch_args_round_trip() -> None:
    args = bootstrap.parse_launch_args(["gui", "--no-excel", "--bundle", "bundle.npz"])

    assert args.command == "gui"
    assert args.no_excel is True
    assert args.bundle == "bundle.npz"


def test_resolve_startup_mode_prefers_explicit_command() -> None:
    assert bootstrap.resolve_startup_mode("gui") == "simulation"
    assert bootstrap.resolve_startup_mode("calibrant", early_mode="simulation") == "calibrant"
    assert bootstrap.resolve_startup_mode(None, early_mode="simulation") == "simulation"
    assert bootstrap.resolve_startup_mode(None) == "prompt"


def test_early_main_bootstrap_marks_gui_mode(monkeypatch) -> None:
    monkeypatch.delenv("RA_SIM_EARLY_STARTUP_MODE", raising=False)

    bootstrap.early_main_bootstrap("__main__", ["gui"])

    assert os.environ["RA_SIM_EARLY_STARTUP_MODE"] == "simulation"


def test_early_main_bootstrap_forwards_cli(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.delenv("RA_SIM_EARLY_STARTUP_MODE", raising=False)

    def _fake_cli_main(argv: list[str]) -> None:
        calls.append(list(argv))

    with pytest.raises(SystemExit) as exc_info:
        bootstrap.early_main_bootstrap(
            "__main__",
            ["simulate", "--out", "output.png"],
            cli_main=_fake_cli_main,
        )

    assert exc_info.value.code == 0
    assert calls == [["simulate", "--out", "output.png"]]


def test_build_runtime_structure_factor_pruning_bootstrap_delegates_callbacks() -> None:
    calls: list[tuple[object, ...]] = []

    class _FakePruningModule:
        def make_runtime_structure_factor_pruning_bindings_factory(self, **kwargs):
            calls.append(("bindings", kwargs))
            return "bindings-factory"

        def make_runtime_current_sf_prune_bias_callback(self, bindings_factory):
            calls.append(("bias", bindings_factory))
            return "bias-callback"

        def make_runtime_current_solve_q_values_callback(
            self,
            bindings_factory,
            *,
            uniform_flag,
            adaptive_flag,
        ):
            calls.append(
                ("solve_q_values", bindings_factory, uniform_flag, adaptive_flag)
            )
            return "solve-q-callback"

        def make_runtime_structure_factor_pruning_status_callback(
            self, bindings_factory
        ):
            calls.append(("status", bindings_factory))
            return "status-callback"

        def make_runtime_bragg_qr_filter_callback(self, bindings_factory):
            calls.append(("filters", bindings_factory))
            return "filter-callback"

        def make_runtime_sf_prune_bias_change_callback(self, bindings_factory):
            calls.append(("bias_change", bindings_factory))
            return "bias-change-callback"

        def make_runtime_solve_q_steps_change_callback(self, bindings_factory):
            calls.append(("steps_change", bindings_factory))
            return "steps-change-callback"

        def make_runtime_solve_q_rel_tol_change_callback(self, bindings_factory):
            calls.append(("rel_tol_change", bindings_factory))
            return "rel-tol-change-callback"

        def make_runtime_solve_q_control_states_callback(self, bindings_factory):
            calls.append(("control_states", bindings_factory))
            return "control-states-callback"

        def make_runtime_solve_q_mode_change_callback(self, bindings_factory):
            calls.append(("mode_change", bindings_factory))
            return "mode-change-callback"

    bundle = bootstrap.build_runtime_structure_factor_pruning_bootstrap(
        structure_factor_pruning_module=_FakePruningModule(),
        uniform_flag=11,
        adaptive_flag=22,
        view_state_factory="view-state",
    )

    assert bundle.bindings_factory == "bindings-factory"
    assert bundle.current_sf_prune_bias == "bias-callback"
    assert bundle.current_solve_q_values == "solve-q-callback"
    assert bundle.update_status_label == "status-callback"
    assert bundle.apply_filters == "filter-callback"
    assert bundle.on_sf_prune_bias_change == "bias-change-callback"
    assert bundle.on_solve_q_steps_change == "steps-change-callback"
    assert bundle.on_solve_q_rel_tol_change == "rel-tol-change-callback"
    assert bundle.set_solve_q_control_states == "control-states-callback"
    assert bundle.on_solve_q_mode_change == "mode-change-callback"
    assert calls[0] == ("bindings", {"view_state_factory": "view-state"})
    assert calls[2] == ("solve_q_values", "bindings-factory", 11, 22)


def test_build_runtime_qr_cylinder_overlay_bootstrap_delegates_callbacks() -> None:
    calls: list[tuple[object, ...]] = []

    module = SimpleNamespace(
        make_runtime_qr_cylinder_overlay_bindings_factory=(
            lambda **kwargs: calls.append(("bindings", kwargs)) or "overlay-bindings"
        ),
        make_runtime_qr_cylinder_overlay_refresh_callback=(
            lambda bindings_factory: (
                calls.append(("refresh", bindings_factory)) or "overlay-refresh"
            )
        ),
        make_runtime_qr_cylinder_overlay_toggle_callback=(
            lambda bindings_factory: (
                calls.append(("toggle", bindings_factory)) or "overlay-toggle"
            )
        ),
    )

    bundle = bootstrap.build_runtime_qr_cylinder_overlay_bootstrap(
        qr_cylinder_overlay_module=module,
        ax="axis",
    )

    assert bundle.bindings_factory == "overlay-bindings"
    assert bundle.refresh == "overlay-refresh"
    assert bundle.toggle == "overlay-toggle"
    assert calls == [
        ("bindings", {"ax": "axis"}),
        ("refresh", "overlay-bindings"),
        ("toggle", "overlay-bindings"),
    ]


def test_build_runtime_bragg_qr_bootstrap_delegates_callbacks() -> None:
    calls: list[tuple[object, ...]] = []

    module = SimpleNamespace(
        make_runtime_bragg_qr_bindings_factory=(
            lambda **kwargs: calls.append(("bindings", kwargs)) or "bragg-bindings"
        ),
        make_runtime_bragg_qr_refresh_callback=(
            lambda bindings_factory: (
                calls.append(("refresh", bindings_factory)) or "bragg-refresh"
            )
        ),
        make_runtime_bragg_qr_open_callback=(
            lambda **kwargs: calls.append(("open", kwargs)) or "bragg-open"
        ),
    )

    bundle = bootstrap.build_runtime_bragg_qr_bootstrap(
        bragg_qr_manager_module=module,
        root="root-window",
        view_state="view-state",
    )

    assert bundle.bindings_factory == "bragg-bindings"
    assert bundle.refresh_window == "bragg-refresh"
    assert bundle.open_window == "bragg-open"
    assert calls == [
        ("bindings", {"view_state": "view-state"}),
        ("refresh", "bragg-bindings"),
        (
            "open",
            {"root": "root-window", "bindings_factory": "bragg-bindings"},
        ),
    ]


def test_runtime_callback_bootstrap_helpers_delegate_to_feature_modules() -> None:
    peak_calls: list[tuple[object, ...]] = []
    peak_module = SimpleNamespace(
        make_runtime_peak_selection_bindings_factory=(
            lambda **kwargs: peak_calls.append(("bindings", kwargs)) or "peak-bindings"
        ),
        make_runtime_peak_selection_callbacks=(
            lambda bindings_factory: (
                peak_calls.append(("callbacks", bindings_factory)) or "peak-callbacks"
            )
        ),
    )
    peak_bundle = bootstrap.build_runtime_peak_selection_bootstrap(
        peak_selection_module=peak_module,
        peak_state="state",
    )
    assert peak_bundle.bindings_factory == "peak-bindings"
    assert peak_bundle.callbacks == "peak-callbacks"
    assert peak_calls == [
        ("bindings", {"peak_state": "state"}),
        ("callbacks", "peak-bindings"),
    ]

    drag_calls: list[tuple[object, ...]] = []
    drag_module = SimpleNamespace(
        make_runtime_integration_range_drag_bindings_factory=(
            lambda **kwargs: drag_calls.append(("bindings", kwargs)) or "drag-bindings"
        ),
        make_runtime_integration_range_drag_callbacks=(
            lambda bindings_factory: (
                drag_calls.append(("callbacks", bindings_factory)) or "drag-callbacks"
            )
        ),
    )
    drag_bundle = bootstrap.build_runtime_integration_range_drag_bootstrap(
        integration_range_drag_module=drag_module,
        image_display="image",
    )
    assert drag_bundle.bindings_factory == "drag-bindings"
    assert drag_bundle.callbacks == "drag-callbacks"
    assert drag_calls == [
        ("bindings", {"image_display": "image"}),
        ("callbacks", "drag-bindings"),
    ]

    canvas_calls: list[tuple[object, ...]] = []
    canvas_module = SimpleNamespace(
        make_runtime_canvas_interaction_bindings_factory=(
            lambda **kwargs: (
                canvas_calls.append(("bindings", kwargs)) or "canvas-bindings"
            )
        ),
        make_runtime_canvas_interaction_callbacks=(
            lambda bindings_factory: (
                canvas_calls.append(("callbacks", bindings_factory))
                or "canvas-callbacks"
            )
        ),
    )
    canvas_bundle = bootstrap.build_runtime_canvas_interaction_bootstrap(
        canvas_interactions_module=canvas_module,
        axis="ax",
    )
    assert canvas_bundle.bindings_factory == "canvas-bindings"
    assert canvas_bundle.callbacks == "canvas-callbacks"
    assert canvas_calls == [
        ("bindings", {"axis": "ax"}),
        ("callbacks", "canvas-bindings"),
    ]

    background_calls: list[tuple[object, ...]] = []
    background_module = SimpleNamespace(
        make_runtime_background_bindings_factory=(
            lambda **kwargs: (
                background_calls.append(("bindings", kwargs))
                or "background-bindings"
            )
        ),
        make_runtime_background_callbacks=(
            lambda bindings_factory: (
                background_calls.append(("callbacks", bindings_factory))
                or "background-callbacks"
            )
        ),
    )
    background_bundle = bootstrap.build_runtime_background_bootstrap(
        background_manager_module=background_module,
        view_state="workspace",
    )
    assert background_bundle.bindings_factory == "background-bindings"
    assert background_bundle.callbacks == "background-callbacks"
    assert background_calls == [
        ("bindings", {"view_state": "workspace"}),
        ("callbacks", "background-bindings"),
    ]


def test_build_runtime_selected_peak_bootstrap_composes_feature_setup(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    peak_module = SimpleNamespace(
        make_runtime_selected_peak_config_factories=(
            lambda **kwargs: calls.append(("config", kwargs))
            or SimpleNamespace(
                canvas_pick="canvas-config",
                intersection="intersection-config",
                ideal_center="ideal-center",
            )
        ),
        make_runtime_peak_overlay_data_callback=(
            lambda **kwargs: calls.append(("overlay", kwargs)) or "overlay-callback"
        ),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_runtime_peak_selection_bootstrap",
        lambda **kwargs: calls.append(("runtime", kwargs))
        or SimpleNamespace(
            bindings_factory="bindings-factory",
            callbacks="callbacks",
        ),
    )

    bundle = bootstrap.build_runtime_selected_peak_bootstrap(
        peak_selection_module=peak_module,
        simulation_runtime_state="runtime-state",
        peak_selection_state="peak-state",
        hkl_lookup_view_state_factory="view-factory",
        selected_peak_marker_factory="marker-factory",
        current_primary_a_factory="current-primary-a",
        caked_view_enabled_factory="caked-flag",
        sync_peak_selection_state="sync-state",
        image_size=64,
        primary_a_factory="primary-a",
        primary_c_factory="primary-c",
        max_distance_px=12.0,
        min_separation_px=2.0,
        image_shape_factory="image-shape",
        center_col_factory="center-col",
        center_row_factory="center-row",
        distance_cor_to_detector_factory="distance",
        gamma_deg_factory="gamma",
        Gamma_deg_factory="Gamma",
        chi_deg_factory="chi",
        psi_deg_factory="psi",
        psi_z_deg_factory="psi-z",
        zs_factory="zs",
        zb_factory="zb",
        theta_initial_deg_factory="theta-initial",
        cor_angle_deg_factory="cor-angle",
        sigma_mosaic_deg_factory="sigma-mosaic",
        gamma_mosaic_deg_factory="gamma-mosaic",
        eta_factory="eta",
        wavelength_factory="wavelength",
        debye_x_factory="debye-x",
        debye_y_factory="debye-y",
        detector_center_factory="detector-center",
        optics_mode_factory="optics-mode",
        solve_q_values_factory="solve-q",
        overlay_primary_a_factory="overlay-primary-a",
        overlay_primary_c_factory="overlay-primary-c",
        native_sim_to_display_coords="display-coords",
        reflection_q_group_metadata="q-group",
        max_hits_per_reflection="max-hits",
        schedule_update_factory="schedule-update",
        set_status_text_factory="status-text",
        draw_idle_factory="draw-idle",
        display_to_native_sim_coords="native-coords",
        deactivate_conflicting_modes_factory="deactivate-modes",
        n2="n2",
        process_peaks_parallel="process-peaks",
        tcl_error_types=(RuntimeError,),
    )

    assert bundle.bindings_factory == "bindings-factory"
    assert bundle.callbacks == "callbacks"
    assert bundle.ensure_peak_overlay_data == "overlay-callback"
    assert calls == [
        (
            "config",
            {
                "simulation_runtime_state": "runtime-state",
                "image_size": 64,
                "primary_a_factory": "primary-a",
                "primary_c_factory": "primary-c",
                "max_distance_px": 12.0,
                "min_separation_px": 2.0,
                "image_shape_factory": "image-shape",
                "center_col_factory": "center-col",
                "center_row_factory": "center-row",
                "distance_cor_to_detector_factory": "distance",
                "gamma_deg_factory": "gamma",
                "Gamma_deg_factory": "Gamma",
                "chi_deg_factory": "chi",
                "psi_deg_factory": "psi",
                "psi_z_deg_factory": "psi-z",
                "zs_factory": "zs",
                "zb_factory": "zb",
                "theta_initial_deg_factory": "theta-initial",
                "cor_angle_deg_factory": "cor-angle",
                "sigma_mosaic_deg_factory": "sigma-mosaic",
                "gamma_mosaic_deg_factory": "gamma-mosaic",
                "eta_factory": "eta",
                "wavelength_factory": "wavelength",
                "debye_x_factory": "debye-x",
                "debye_y_factory": "debye-y",
                "detector_center_factory": "detector-center",
                "optics_mode_factory": "optics-mode",
                "solve_q_values_factory": "solve-q",
                "n2": "n2",
                "process_peaks_parallel": "process-peaks",
            },
        ),
        (
            "overlay",
            {
                "simulation_runtime_state": "runtime-state",
                "primary_a_factory": "overlay-primary-a",
                "primary_c_factory": "overlay-primary-c",
                "native_sim_to_display_coords": "display-coords",
                "reflection_q_group_metadata": "q-group",
                "max_hits_per_reflection": "max-hits",
                "min_separation_px": 2.0,
            },
        ),
        (
            "runtime",
            {
                "peak_selection_module": peak_module,
                "simulation_runtime_state": "runtime-state",
                "peak_selection_state": "peak-state",
                "hkl_lookup_view_state_factory": "view-factory",
                "selected_peak_marker_factory": "marker-factory",
                "current_primary_a_factory": "current-primary-a",
                "caked_view_enabled_factory": "caked-flag",
                "current_canvas_pick_config_factory": "canvas-config",
                "current_intersection_config_factory": "intersection-config",
                "ensure_peak_overlay_data": "overlay-callback",
                "sync_peak_selection_state": "sync-state",
                "schedule_update_factory": "schedule-update",
                "set_status_text_factory": "status-text",
                "draw_idle_factory": "draw-idle",
                "display_to_native_sim_coords": "native-coords",
                "native_sim_to_display_coords": "display-coords",
                "simulate_ideal_hkl_native_center": "ideal-center",
                "deactivate_conflicting_modes_factory": "deactivate-modes",
                "n2": "n2",
                "tcl_error_types": (RuntimeError,),
            },
        ),
    ]


def test_build_runtime_integration_range_workflow_bootstrap_composes_setup(
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    drag_module = SimpleNamespace(
        create_integration_region_rectangle=(
            lambda ax: calls.append(("region-rect", ax)) or "region-rect"
        ),
        create_drag_select_rectangle=(
            lambda ax: calls.append(("drag-rect", ax)) or "drag-rect"
        ),
        make_runtime_integration_region_visuals_callback=(
            lambda bindings_factory: (
                calls.append(("refresh", bindings_factory)) or "refresh-callback"
            )
        ),
    )

    monkeypatch.setattr(
        bootstrap,
        "build_runtime_integration_range_drag_bootstrap",
        lambda **kwargs: calls.append(("runtime", kwargs))
        or SimpleNamespace(
            bindings_factory="bindings-factory",
            callbacks="callbacks",
        ),
    )

    bundle = bootstrap.build_runtime_integration_range_workflow_bootstrap(
        integration_range_drag_module=drag_module,
        ax="axis",
        integration_region_overlay="overlay",
        image_display="image-display",
        range_visible_factory="range-visible",
    )

    assert bundle.bindings_factory == "bindings-factory"
    assert bundle.callbacks == "callbacks"
    assert bundle.refresh_visuals == "refresh-callback"
    assert calls == [
        ("region-rect", "axis"),
        ("drag-rect", "axis"),
        (
            "runtime",
            {
                "integration_range_drag_module": drag_module,
                "ax": "axis",
                "drag_select_rect": "drag-rect",
                "integration_region_rect": "region-rect",
                "integration_region_overlay": "overlay",
                "image_display": "image-display",
                "range_visible_factory": "range-visible",
            },
        ),
        ("refresh", "bindings-factory"),
    ]

    geometry_calls: list[tuple[object, ...]] = []
    geometry_module = SimpleNamespace(
        make_runtime_geometry_q_group_bindings_factory=(
            lambda **kwargs: (
                geometry_calls.append(("bindings", kwargs)) or "geometry-bindings"
            )
        ),
        make_runtime_geometry_q_group_callbacks=(
            lambda **kwargs: (
                geometry_calls.append(("callbacks", kwargs)) or "geometry-callbacks"
            )
        ),
    )
    geometry_bundle = bootstrap.build_runtime_geometry_q_group_bootstrap(
        geometry_q_group_manager_module=geometry_module,
        root="root-window",
        fit_config="fit-config",
    )
    assert geometry_bundle.bindings_factory == "geometry-bindings"
    assert geometry_bundle.callbacks == "geometry-callbacks"
    assert geometry_calls == [
        ("bindings", {"fit_config": "fit-config"}),
        (
            "callbacks",
            {"root": "root-window", "bindings_factory": "geometry-bindings"},
        ),
    ]
