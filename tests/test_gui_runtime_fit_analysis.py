from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_fit_analysis


def test_build_runtime_pruning_workflow_composes_controls_and_initial_filter_refresh() -> None:
    calls: list[tuple[str, object]] = []
    filter_calls: list[bool] = []
    runtime_obj = SimpleNamespace(
        current_sf_prune_bias="current-bias",
        current_solve_q_values="current-solve-q",
        update_status_label="update-status",
        apply_filters=lambda *, trigger_update=True: (
            filter_calls.append(bool(trigger_update)) or {"trigger_update": bool(trigger_update)}
        ),
        on_sf_prune_bias_change="on-bias",
        on_solve_q_steps_change="on-steps",
        on_solve_q_rel_tol_change="on-rel-tol",
        set_solve_q_control_states="set-controls",
        on_solve_q_mode_change="on-mode",
    )
    controls_runtime = SimpleNamespace(name="controls-runtime")
    bootstrap_module = SimpleNamespace(
        build_runtime_bragg_qr_workflow_bootstrap=lambda **kwargs: (
            calls.append(("workflow", kwargs)) or runtime_obj
        ),
        build_runtime_structure_factor_pruning_controls_bootstrap=lambda **kwargs: (
            calls.append(("controls", kwargs)) or controls_runtime
        ),
    )

    workflow = runtime_fit_analysis.build_runtime_pruning_workflow(
        bootstrap_module=bootstrap_module,
        views_module="views-module",
        structure_factor_pruning_module="pruning-module",
        view_state="pruning-view-state",
        bragg_qr_bootstrap_kwargs={"root": "root-window", "uniform_flag": 1},
        pruning_controls_bootstrap_kwargs={"raw_prune_bias": 0.25},
        initialize_filters=True,
    )

    assert workflow.runtime is runtime_obj
    assert workflow.controls_runtime is controls_runtime
    assert workflow.current_sf_prune_bias == "current-bias"
    assert workflow.current_solve_q_values == "current-solve-q"
    assert workflow.update_status_label == "update-status"
    assert workflow.on_sf_prune_bias_change == "on-bias"
    assert workflow.on_solve_q_steps_change == "on-steps"
    assert workflow.on_solve_q_rel_tol_change == "on-rel-tol"
    assert workflow.set_solve_q_control_states == "set-controls"
    assert workflow.on_solve_q_mode_change == "on-mode"
    assert workflow.apply_filters(trigger_update=True) == {"trigger_update": True}
    assert calls == [
        (
            "workflow",
            {
                "structure_factor_pruning_module": "pruning-module",
                "root": "root-window",
                "uniform_flag": 1,
            },
        ),
        (
            "controls",
            {
                "views_module": "views-module",
                "structure_factor_pruning_module": "pruning-module",
                "view_state": "pruning-view-state",
                "on_sf_prune_bias_change": "on-bias",
                "update_status_label": "update-status",
                "on_solve_q_steps_change": "on-steps",
                "on_solve_q_rel_tol_change": "on-rel-tol",
                "on_solve_q_mode_change": "on-mode",
                "set_solve_q_control_states": "set-controls",
                "raw_prune_bias": 0.25,
            },
        ),
    ]
    assert filter_calls == [False, True]


def test_build_runtime_integration_range_workflow_exposes_callback_aliases() -> None:
    calls: list[tuple[str, object]] = []
    callbacks = SimpleNamespace(
        schedule_range_update="schedule-range-update",
        toggle_1d_plots="toggle-1d-plots",
        toggle_caked_2d="toggle-caked-2d",
        toggle_log_radial="toggle-log-radial",
        toggle_log_azimuth="toggle-log-azimuth",
    )
    update_runtime = SimpleNamespace(callbacks=callbacks)
    bootstrap_module = SimpleNamespace(
        build_runtime_integration_range_update_bootstrap=lambda **kwargs: (
            calls.append(("update", kwargs)) or update_runtime
        )
    )

    workflow = runtime_fit_analysis.build_runtime_integration_range_workflow(
        bootstrap_module=bootstrap_module,
        views_module="views-module",
        integration_range_drag_module="integration-range-drag-module",
        integration_range_update_bootstrap_kwargs={
            "root": "root-window",
            "range_view_state": "range-view-state",
        },
    )

    assert workflow.update_runtime is update_runtime
    assert workflow.callbacks is callbacks
    assert workflow.schedule_range_update == "schedule-range-update"
    assert workflow.toggle_1d_plots == "toggle-1d-plots"
    assert workflow.toggle_caked_2d == "toggle-caked-2d"
    assert workflow.toggle_log_radial == "toggle-log-radial"
    assert workflow.toggle_log_azimuth == "toggle-log-azimuth"
    assert calls == [
        (
            "update",
            {
                "views_module": "views-module",
                "integration_range_drag_module": "integration-range-drag-module",
                "root": "root-window",
                "range_view_state": "range-view-state",
            },
        )
    ]


def test_build_runtime_geometry_fit_action_workflow_exposes_runtime_aliases() -> None:
    calls: list[tuple[str, object]] = []
    runtime_obj = SimpleNamespace(
        bindings_factory="fit-bindings-factory",
        callback="fit-callback",
    )
    bootstrap_module = SimpleNamespace(
        build_runtime_geometry_fit_action_bootstrap=lambda **kwargs: (
            calls.append(("fit-action", kwargs)) or runtime_obj
        )
    )

    workflow = runtime_fit_analysis.build_runtime_geometry_fit_action_workflow(
        bootstrap_module=bootstrap_module,
        geometry_fit_module="geometry-fit-module",
        geometry_fit_action_bootstrap_kwargs={
            "value_callbacks_factory": "value-callbacks-factory",
            "schedule_update": "schedule-update",
        },
    )

    assert workflow.runtime is runtime_obj
    assert workflow.bindings_factory == "fit-bindings-factory"
    assert workflow.callback == "fit-callback"
    assert calls == [
        (
            "fit-action",
            {
                "geometry_fit_module": "geometry-fit-module",
                "value_callbacks_factory": "value-callbacks-factory",
                "schedule_update": "schedule-update",
            },
        )
    ]


def test_resolve_runtime_pruning_control_defaults_delegates_to_shared_module() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    module = SimpleNamespace(
        build_runtime_structure_factor_pruning_defaults=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or "pruning-defaults"
        )
    )

    defaults = runtime_fit_analysis.resolve_runtime_pruning_control_defaults(
        structure_factor_pruning_module=module,
        raw_prune_bias="0.25",
        raw_solve_q_steps="33",
        raw_solve_q_rel_tol="5e-4",
        raw_solve_q_mode="adaptive",
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
    )

    assert defaults == "pruning-defaults"
    assert calls == [
        (
            ("0.25", "33", "5e-4", "adaptive"),
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
        )
    ]
