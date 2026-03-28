from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import runtime_geometry_fit


def test_build_runtime_geometry_fit_workflow_composes_value_dataset_config_and_action_workflow() -> None:
    calls: list[tuple[str, object]] = []
    value_callbacks = SimpleNamespace(var_map={"theta_initial": "theta-var"})
    action_workflow = SimpleNamespace(
        runtime="action-runtime",
        bindings_factory="action-bindings-factory",
        callback="on-fit-geometry-click",
    )

    geometry_fit_module = SimpleNamespace(
        build_runtime_geometry_fit_value_callbacks=(
            lambda bindings: calls.append(("value", bindings)) or value_callbacks
        ),
        make_runtime_geometry_fit_manual_dataset_bindings_factory=(
            lambda **kwargs: calls.append(("manual-dataset", kwargs))
            or "manual-dataset-factory"
        ),
        build_runtime_geometry_fit_config_factory=(
            lambda **kwargs: calls.append(("runtime-config", kwargs))
            or "runtime-config-factory"
        ),
    )
    runtime_fit_analysis_module = SimpleNamespace(
        build_runtime_geometry_fit_action_workflow=lambda **kwargs: (
            calls.append(("action-workflow", kwargs)) or action_workflow
        )
    )

    workflow = runtime_geometry_fit.build_runtime_geometry_fit_workflow(
        geometry_fit_module=geometry_fit_module,
        runtime_fit_analysis_module=runtime_fit_analysis_module,
        bootstrap_module="bootstrap-module",
        value_bindings="value-bindings",
        manual_dataset_bindings_factory_kwargs={
            "osc_files_factory": "osc-files-factory",
            "image_size": 2048,
        },
        runtime_config_factory_kwargs={
            "base_config": {"solver": {"loss": "soft_l1"}},
            "current_constraint_state": "constraint-state",
        },
        action_bootstrap_kwargs={
            "value_callbacks_factory": "value-callbacks-factory",
            "schedule_update": "schedule-update",
        },
    )

    assert workflow.value_callbacks is value_callbacks
    assert workflow.var_map == {"theta_initial": "theta-var"}
    assert workflow.manual_dataset_bindings_factory == "manual-dataset-factory"
    assert workflow.runtime_config_factory == "runtime-config-factory"
    assert workflow.action_workflow is action_workflow
    assert workflow.action_runtime == "action-runtime"
    assert workflow.action_bindings_factory == "action-bindings-factory"
    assert workflow.on_fit_geometry_click == "on-fit-geometry-click"
    assert calls == [
        ("value", "value-bindings"),
        (
            "manual-dataset",
            {
                "osc_files_factory": "osc-files-factory",
                "image_size": 2048,
            },
        ),
        (
            "runtime-config",
            {
                "base_config": {"solver": {"loss": "soft_l1"}},
                "current_constraint_state": "constraint-state",
            },
        ),
        (
            "action-workflow",
            {
                "bootstrap_module": "bootstrap-module",
                "geometry_fit_module": geometry_fit_module,
                "geometry_fit_action_bootstrap_kwargs": {
                    "value_callbacks_factory": "value-callbacks-factory",
                    "schedule_update": "schedule-update",
                    "manual_dataset_bindings_factory": "manual-dataset-factory",
                    "build_runtime_config_factory": "runtime-config-factory",
                    "var_map": {"theta_initial": "theta-var"},
                },
            },
        ),
    ]
