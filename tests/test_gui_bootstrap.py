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


def test_build_runtime_structure_factor_pruning_controls_bootstrap_wires_creation() -> None:
    calls: list[tuple[object, object]] = []
    view_state = SimpleNamespace()

    def _create_controls(**kwargs):
        calls.append(("create", kwargs))
        view_state.sf_prune_bias_var = "bias-var"
        view_state.solve_q_steps_var = "steps-var"
        view_state.solve_q_rel_tol_var = "rel-tol-var"
        view_state.solve_q_mode_var = "mode-var"

    module = SimpleNamespace(
        build_runtime_structure_factor_pruning_defaults=(
            lambda *args, **kwargs: (
                calls.append(("defaults", (args, kwargs)))
                or SimpleNamespace(
                    prune_bias=0.25,
                    solve_q=SimpleNamespace(
                        mode_label="adaptive",
                        steps=21,
                        rel_tol=5.0e-4,
                    ),
                )
            )
        ),
        initialize_runtime_structure_factor_pruning_controls=(
            lambda view_state_arg, **kwargs: (
                calls.append(("initialize", (view_state_arg, kwargs))) or None
            )
        ),
    )

    bundle = bootstrap.build_runtime_structure_factor_pruning_controls_bootstrap(
        views_module=SimpleNamespace(create_structure_factor_pruning_controls=_create_controls),
        structure_factor_pruning_module=module,
        view_state=view_state,
        raw_prune_bias="0.25",
        raw_solve_q_steps="20.6",
        raw_solve_q_rel_tol="5e-4",
        raw_solve_q_mode="robust",
        prune_bias_fallback=0.0,
        prune_bias_minimum=-2.0,
        prune_bias_maximum=2.0,
        steps_fallback=16,
        steps_minimum=4,
        steps_maximum=64,
        rel_tol_fallback=1.0e-3,
        rel_tol_minimum=1.0e-6,
        rel_tol_maximum=1.0e-2,
        uniform_flag=1,
        adaptive_flag=2,
        on_sf_prune_bias_change="bias-change",
        update_status_label="update-status",
        on_solve_q_steps_change="steps-change",
        on_solve_q_rel_tol_change="rel-tol-change",
        on_solve_q_mode_change="mode-change",
        set_solve_q_control_states="set-controls",
    )

    bundle.create_controls(parent="parent-frame")

    assert calls == [
        (
            "defaults",
            (
                ("0.25", "20.6", "5e-4", "robust"),
                {
                    "prune_bias_fallback": 0.0,
                    "prune_bias_minimum": -2.0,
                    "prune_bias_maximum": 2.0,
                    "steps_fallback": 16,
                    "steps_minimum": 4,
                    "steps_maximum": 64,
                    "rel_tol_fallback": 1.0e-3,
                    "rel_tol_minimum": 1.0e-6,
                    "rel_tol_maximum": 1.0e-2,
                    "uniform_flag": 1,
                    "adaptive_flag": 2,
                },
            ),
        ),
        (
            "create",
            {
                "parent": "parent-frame",
                "view_state": view_state,
                "sf_prune_bias_range": (-2.0, 2.0),
                "sf_prune_bias_value": 0.25,
                "solve_q_mode": "adaptive",
                "solve_q_steps_range": (4.0, 64.0),
                "solve_q_steps_value": 21.0,
                "solve_q_rel_tol_range": (1.0e-6, 1.0e-2),
                "solve_q_rel_tol_value": 5.0e-4,
                "status_text": "",
            },
        ),
        (
            "initialize",
            (
                view_state,
                {
                    "on_sf_prune_bias_change": "bias-change",
                    "update_status_label": "update-status",
                    "on_solve_q_steps_change": "steps-change",
                    "on_solve_q_rel_tol_change": "rel-tol-change",
                    "on_solve_q_mode_change": "mode-change",
                    "set_solve_q_control_states": "set-controls",
                },
            ),
        ),
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


def test_build_runtime_geometry_fit_action_bootstrap_delegates_callbacks() -> None:
    calls: list[tuple[object, object]] = []

    before_run = lambda: None
    module = SimpleNamespace(
        make_runtime_geometry_fit_action_bindings_factory=(
            lambda **kwargs: calls.append(("bindings", kwargs)) or "fit-bindings"
        ),
        make_runtime_geometry_fit_action_callback=(
            lambda **kwargs: calls.append(("callback", kwargs)) or "fit-callback"
        ),
    )

    bundle = bootstrap.build_runtime_geometry_fit_action_bootstrap(
        geometry_fit_module=module,
        before_run=before_run,
        live_values="value-callbacks",
    )

    assert bundle.bindings_factory == "fit-bindings"
    assert bundle.callback == "fit-callback"
    assert calls == [
        ("bindings", {"live_values": "value-callbacks"}),
        (
            "callback",
            {
                "bindings_factory": "fit-bindings",
                "before_run": before_run,
            },
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

    background_control_calls: list[tuple[str, object]] = []
    background_callbacks = SimpleNamespace(
        refresh_status=lambda: background_control_calls.append(("refresh-status",))
        or "status",
        refresh_backend_status=(
            lambda: background_control_calls.append(("refresh-backend-status",))
            or "backend-status"
        ),
        toggle_visibility=lambda: background_control_calls.append(("toggle",)) or True,
        switch_background="switch-background",
        browse_files="browse-files",
        rotate_backend_minus_90="rotate--90",
        rotate_backend_plus_90="rotate-90",
        flip_backend_x="flip-x",
        flip_backend_y="flip-y",
        reset_backend_orientation="reset",
    )
    workspace_view_state = SimpleNamespace(
        workspace_actions_frame="actions-frame",
        workspace_backgrounds_frame="backgrounds-frame",
    )
    background_controls_bundle = bootstrap.build_runtime_background_controls_bootstrap(
        views_module=SimpleNamespace(
            populate_stacked_button_group=lambda parent, button_specs: (
                background_control_calls.append(
                    ("buttons", parent, list(button_specs))
                )
            ),
            create_background_file_controls=lambda **kwargs: (
                background_control_calls.append(("workspace-controls", kwargs))
            ),
            create_background_backend_debug_controls=lambda **kwargs: (
                background_control_calls.append(("backend-controls", kwargs))
            ),
        ),
        workspace_view_state=workspace_view_state,
        background_backend_debug_view_state="backend-view",
        background_callbacks=background_callbacks,
    )

    assert background_controls_bundle.toggle_visibility() is True
    background_controls_bundle.create_workspace_controls()
    background_controls_bundle.create_backend_debug_controls("fit-body")
    assert background_controls_bundle.refresh_status() == "status"
    assert background_controls_bundle.refresh_backend_status() == "backend-status"
    assert background_control_calls == [
        ("toggle",),
        (
            "buttons",
            "actions-frame",
            [
                ("Toggle Background", background_controls_bundle.toggle_visibility),
                ("Switch Background", "switch-background"),
            ],
        ),
        (
            "workspace-controls",
            {
                "parent": "backgrounds-frame",
                "view_state": workspace_view_state,
                "on_load_backgrounds": "browse-files",
                "status_text": "",
            },
        ),
        ("refresh-status",),
        (
            "backend-controls",
            {
                "parent": "fit-body",
                "view_state": "backend-view",
                "status_text": "",
                "on_rotate_minus_90": "rotate--90",
                "on_rotate_plus_90": "rotate-90",
                "on_flip_x": "flip-x",
                "on_flip_y": "flip-y",
                "on_reset": "reset",
            },
        ),
        ("refresh-backend-status",),
        ("refresh-status",),
        ("refresh-backend-status",),
    ]

    background_theta_calls: list[tuple[object, ...]] = []
    background_theta_module = SimpleNamespace(
        make_runtime_background_theta_bindings_factory=(
            lambda **kwargs: (
                background_theta_calls.append(("bindings", kwargs))
                or "background-theta-bindings"
            )
        ),
        make_runtime_background_theta_callbacks=(
            lambda bindings_factory: (
                background_theta_calls.append(("callbacks", bindings_factory))
                or "background-theta-callbacks"
            )
        ),
    )
    background_theta_bundle = bootstrap.build_runtime_background_theta_bootstrap(
        background_theta_module=background_theta_module,
        defaults={"theta_initial": 6.0},
    )
    assert background_theta_bundle.bindings_factory == "background-theta-bindings"
    assert background_theta_bundle.callbacks == "background-theta-callbacks"
    assert background_theta_calls == [
        ("bindings", {"defaults": {"theta_initial": 6.0}}),
        ("callbacks", "background-theta-bindings"),
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
        make_runtime_peak_selection_maintenance_callbacks=(
            lambda bindings_factory: (
                calls.append(("maintenance", bindings_factory))
                or "maintenance-callbacks"
            )
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
    assert bundle.maintenance_callbacks == "maintenance-callbacks"
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
        ("maintenance", "bindings-factory"),
    ]


def test_build_runtime_geometry_manual_bootstrap_wraps_callback_bundle() -> None:
    calls: list[tuple[object, ...]] = []

    bundle = bootstrap.build_runtime_geometry_manual_bootstrap(
        manual_geometry_module=SimpleNamespace(
            make_runtime_geometry_manual_callbacks=(
                lambda **kwargs: calls.append(("manual", kwargs))
                or "manual-callbacks"
            )
        ),
        background_visible="visible",
        current_background_index="background-index",
    )

    assert bundle.callbacks == "manual-callbacks"
    assert calls == [
        (
            "manual",
            {
                "background_visible": "visible",
                "current_background_index": "background-index",
            },
        )
    ]


def test_build_runtime_geometry_tool_action_callbacks_bootstrap_wraps_bundle() -> None:
    calls: list[tuple[object, ...]] = []

    bundle = bootstrap.build_runtime_geometry_tool_action_callbacks_bootstrap(
        geometry_fit_module=SimpleNamespace(
            make_runtime_geometry_tool_action_callbacks=(
                lambda **kwargs: calls.append(("tool-actions", kwargs))
                or "tool-action-callbacks"
            )
        ),
        geometry_fit_history_state="history-state",
        manual_pick_armed="armed",
    )

    assert bundle.callbacks == "tool-action-callbacks"
    assert calls == [
        (
            "tool-actions",
            {
                "geometry_fit_history_state": "history-state",
                "manual_pick_armed": "armed",
            },
        )
    ]


def test_build_runtime_hkl_lookup_controls_bootstrap_wires_peak_and_bragg_callbacks() -> None:
    view_calls: list[dict[str, object]] = []
    callback_calls: list[tuple[object, ...]] = []

    def _select_hkl() -> None:
        callback_calls.append(("select",))

    def _toggle_hkl_pick() -> None:
        callback_calls.append(("toggle",))

    def _clear_selected_peak() -> None:
        callback_calls.append(("clear",))

    def _open_bragg_ewald() -> None:
        callback_calls.append(("bragg-ewald",))

    def _refresh_controls() -> None:
        callback_calls.append(("refresh",))

    def _set_hkl_pick_mode(enabled: bool, message: str | None = None) -> None:
        callback_calls.append(("set", bool(enabled), message))

    def _open_bragg_qr_groups() -> None:
        callback_calls.append(("bragg-qr",))

    bundle = bootstrap.build_runtime_hkl_lookup_controls_bootstrap(
        views_module=SimpleNamespace(
            create_hkl_lookup_controls=lambda **kwargs: view_calls.append(kwargs)
        ),
        view_state="lookup-view",
        peak_selection_callbacks=SimpleNamespace(
            select_peak_from_hkl_controls=_select_hkl,
            toggle_hkl_pick_mode=_toggle_hkl_pick,
            clear_selected_peak=_clear_selected_peak,
            open_selected_peak_intersection_figure=_open_bragg_ewald,
            update_hkl_pick_button_label=_refresh_controls,
            set_hkl_pick_mode=_set_hkl_pick_mode,
        ),
        open_bragg_qr_groups=_open_bragg_qr_groups,
    )

    bundle.create_controls(parent="parent-frame")

    assert view_calls == [
        {
            "parent": "parent-frame",
            "view_state": "lookup-view",
            "on_select_hkl": view_calls[0]["on_select_hkl"],
            "on_toggle_hkl_pick": view_calls[0]["on_toggle_hkl_pick"],
            "on_clear_selected_peak": view_calls[0]["on_clear_selected_peak"],
            "on_show_bragg_ewald": view_calls[0]["on_show_bragg_ewald"],
            "on_open_bragg_qr_groups": view_calls[0]["on_open_bragg_qr_groups"],
        }
    ]
    assert callback_calls == [("refresh",)]

    view_calls[0]["on_select_hkl"]()
    view_calls[0]["on_toggle_hkl_pick"]()
    view_calls[0]["on_clear_selected_peak"]()
    view_calls[0]["on_show_bragg_ewald"]()
    view_calls[0]["on_open_bragg_qr_groups"]()
    bundle.refresh_controls()
    bundle.set_hkl_pick_mode(True, "armed")

    assert callback_calls == [
        ("refresh",),
        ("select",),
        ("toggle",),
        ("clear",),
        ("bragg-ewald",),
        ("bragg-qr",),
        ("refresh",),
        ("set", True, "armed"),
    ]


def test_build_runtime_geometry_tool_action_controls_bootstrap_wires_callbacks() -> None:
    view_calls: list[dict[str, object]] = []
    events: list[str] = []

    bundle = bootstrap.build_runtime_geometry_tool_action_controls_bootstrap(
        views_module=SimpleNamespace(
            create_geometry_tool_action_controls=lambda **kwargs: view_calls.append(kwargs)
        ),
        view_state="geometry-view",
        on_undo_fit=lambda: events.append("undo-fit"),
        on_redo_fit=lambda: events.append("redo-fit"),
        on_toggle_manual_pick=lambda: events.append("toggle-pick"),
        on_undo_manual_placement=lambda: events.append("undo-placement"),
        on_export_manual_pairs=lambda: events.append("export"),
        on_import_manual_pairs=lambda: events.append("import"),
        on_toggle_preview_exclude=lambda: events.append("preview"),
        on_clear_manual_pairs=lambda: events.append("clear"),
    )

    bundle.create_controls(parent="parent-frame")

    assert view_calls == [
        {
            "parent": "parent-frame",
            "view_state": "geometry-view",
            "on_undo_fit": view_calls[0]["on_undo_fit"],
            "on_redo_fit": view_calls[0]["on_redo_fit"],
            "on_toggle_manual_pick": view_calls[0]["on_toggle_manual_pick"],
            "on_undo_manual_placement": view_calls[0]["on_undo_manual_placement"],
            "on_export_manual_pairs": view_calls[0]["on_export_manual_pairs"],
            "on_import_manual_pairs": view_calls[0]["on_import_manual_pairs"],
            "on_toggle_preview_exclude": view_calls[0]["on_toggle_preview_exclude"],
            "on_clear_manual_pairs": view_calls[0]["on_clear_manual_pairs"],
        }
    ]

    for key in (
        "on_undo_fit",
        "on_redo_fit",
        "on_toggle_manual_pick",
        "on_undo_manual_placement",
        "on_export_manual_pairs",
        "on_import_manual_pairs",
        "on_toggle_preview_exclude",
        "on_clear_manual_pairs",
    ):
        view_calls[0][key]()

    assert events == [
        "undo-fit",
        "redo-fit",
        "toggle-pick",
        "undo-placement",
        "export",
        "import",
        "preview",
        "clear",
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


def test_build_runtime_integration_range_update_bootstrap_composes_controls_and_callbacks(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    drag_module = SimpleNamespace(
        make_runtime_integration_range_update_bindings_factory=(
            lambda **kwargs: calls.append(("bindings", kwargs)) or "bindings-factory"
        ),
        make_runtime_integration_range_update_callbacks=(
            lambda bindings_factory: calls.append(("callbacks", bindings_factory))
            or SimpleNamespace(
                schedule_range_update="schedule-range-update",
                toggle_1d_plots="toggle-1d",
                toggle_caked_2d="toggle-caked",
                toggle_log_radial="toggle-log-radial",
                toggle_log_azimuth="toggle-log-azimuth",
            )
        ),
        create_runtime_integration_range_controls=(
            lambda **kwargs: calls.append(("range-controls", kwargs))
        ),
    )
    views_module = SimpleNamespace(
        create_analysis_view_controls=lambda **kwargs: calls.append(
            ("analysis-controls", kwargs)
        )
    )

    bundle = bootstrap.build_runtime_integration_range_update_bootstrap(
        views_module=views_module,
        integration_range_drag_module=drag_module,
        range_view_state="range-view-state",
        analysis_view_state="analysis-view-state",
        root="root",
        simulation_runtime_state="sim-state",
        display_controls_state="display-state",
    )

    assert bundle.bindings_factory == "bindings-factory"
    assert bundle.callbacks.schedule_range_update == "schedule-range-update"
    bundle.create_range_controls("range-parent")
    bundle.create_analysis_controls("analysis-parent")

    assert calls == [
        (
            "bindings",
            {
                "root": "root",
                "simulation_runtime_state": "sim-state",
                "display_controls_state": "display-state",
            },
        ),
        ("callbacks", "bindings-factory"),
        (
            "range-controls",
            {
                "parent": "range-parent",
                "views_module": views_module,
                "view_state": "range-view-state",
                "schedule_range_update": "schedule-range-update",
                "tth_min": 0.0,
                "tth_max": 60.0,
                "phi_min": -15.0,
                "phi_max": 15.0,
            },
        ),
        (
            "analysis-controls",
            {
                "parent": "analysis-parent",
                "view_state": "analysis-view-state",
                "on_toggle_1d_plots": "toggle-1d",
                "on_toggle_caked_2d": "toggle-caked",
                "on_toggle_log_radial": "toggle-log-radial",
                "on_toggle_log_azimuth": "toggle-log-azimuth",
                "show_1d": False,
                "show_caked_2d": False,
                "log_radial": False,
                "log_azimuth": False,
            },
        ),
    ]


def test_build_runtime_bragg_qr_workflow_bootstrap_composes_pruning_and_manager(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    filter_calls: list[bool] = []

    monkeypatch.setattr(
        bootstrap,
        "build_runtime_structure_factor_pruning_bootstrap",
        lambda **kwargs: calls.append(("pruning", kwargs))
        or SimpleNamespace(
            bindings_factory="pruning-bindings",
            current_sf_prune_bias="current-bias",
            current_solve_q_values="current-solve-q",
            update_status_label="update-status",
            apply_filters=(
                lambda *, trigger_update=True: (
                    filter_calls.append(bool(trigger_update))
                    or {"trigger_update": bool(trigger_update)}
                )
            ),
            on_sf_prune_bias_change="on-bias",
            on_solve_q_steps_change="on-steps",
            on_solve_q_rel_tol_change="on-rel-tol",
            set_solve_q_control_states="set-controls",
            on_solve_q_mode_change="on-mode",
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "build_runtime_bragg_qr_bootstrap",
        lambda **kwargs: calls.append(("manager", kwargs))
        or SimpleNamespace(
            bindings_factory="manager-bindings",
            refresh_window="refresh-window",
            open_window="open-window",
        ),
    )

    bundle = bootstrap.build_runtime_bragg_qr_workflow_bootstrap(
        structure_factor_pruning_module="pruning-module",
        bragg_qr_manager_module="manager-module",
        root="root-window",
        uniform_flag=1,
        adaptive_flag=2,
        structure_factor_pruning_view_state_factory="pruning-view",
        bragg_qr_view_state="manager-view",
        simulation_runtime_state="runtime-state",
        bragg_qr_manager_state="manager-state",
        clip_prune_bias="clip-bias",
        clip_solve_q_steps="clip-steps",
        clip_solve_q_rel_tol="clip-rel-tol",
        normalize_solve_q_mode_label="normalize-mode",
        schedule_update_factory="schedule-update",
        primary_candidate="primary-candidate",
        primary_fallback="primary-fallback",
        secondary_candidate="secondary-candidate",
        set_progress_text_factory="progress-factory",
        invalid_key=-99,
        tcl_error_types=(RuntimeError,),
    )

    assert bundle.pruning_bindings_factory == "pruning-bindings"
    assert bundle.manager_bindings_factory == "manager-bindings"
    assert bundle.current_sf_prune_bias == "current-bias"
    assert bundle.current_solve_q_values == "current-solve-q"
    assert bundle.update_status_label == "update-status"
    assert bundle.on_sf_prune_bias_change == "on-bias"
    assert bundle.on_solve_q_steps_change == "on-steps"
    assert bundle.on_solve_q_rel_tol_change == "on-rel-tol"
    assert bundle.set_solve_q_control_states == "set-controls"
    assert bundle.on_solve_q_mode_change == "on-mode"
    assert bundle.refresh_window == "refresh-window"
    assert bundle.open_window == "open-window"
    assert bundle.apply_filters(trigger_update=False) == {"trigger_update": False}
    assert filter_calls == [False]

    pruning_call = calls[0]
    manager_call = calls[1]
    assert pruning_call == (
        "pruning",
        {
            "structure_factor_pruning_module": "pruning-module",
            "uniform_flag": 1,
            "adaptive_flag": 2,
            "view_state_factory": "pruning-view",
            "simulation_runtime_state": "runtime-state",
            "bragg_qr_manager_state": "manager-state",
            "clip_prune_bias": "clip-bias",
            "clip_solve_q_steps": "clip-steps",
            "clip_solve_q_rel_tol": "clip-rel-tol",
            "normalize_solve_q_mode_label": "normalize-mode",
            "schedule_update_factory": "schedule-update",
            "refresh_window_factory": pruning_call[1]["refresh_window_factory"],
        },
    )
    assert manager_call == (
        "manager",
        {
            "bragg_qr_manager_module": "manager-module",
            "root": "root-window",
            "view_state": "manager-view",
            "manager_state": "manager-state",
            "simulation_runtime_state": "runtime-state",
            "primary_candidate": "primary-candidate",
            "primary_fallback": "primary-fallback",
            "secondary_candidate": "secondary-candidate",
            "apply_filters": manager_call[1]["apply_filters"],
            "set_progress_text_factory": "progress-factory",
            "invalid_key": -99,
            "tcl_error_types": (RuntimeError,),
        },
    )
    assert pruning_call[1]["refresh_window_factory"]() == "refresh-window"
    assert manager_call[1]["apply_filters"]() == {"trigger_update": True}
    assert filter_calls == [False, True]

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
