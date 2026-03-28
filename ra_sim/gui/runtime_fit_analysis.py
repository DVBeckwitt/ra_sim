"""Import-safe helpers for assembling runtime fit and analysis workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimePruningWorkflow:
    """Bootstrapped pruning workflow plus its control-cluster runtime."""

    runtime: object
    controls_runtime: object
    current_sf_prune_bias: Callable[..., Any]
    current_solve_q_values: Callable[..., Any]
    update_status_label: Callable[..., Any]
    apply_filters: Callable[..., Any]
    on_sf_prune_bias_change: Callable[..., Any]
    on_solve_q_steps_change: Callable[..., Any]
    on_solve_q_rel_tol_change: Callable[..., Any]
    set_solve_q_control_states: Callable[..., Any]
    on_solve_q_mode_change: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeIntegrationRangeWorkflow:
    """Bootstrapped integration-range update runtime plus callback aliases."""

    update_runtime: object
    callbacks: object
    schedule_range_update: Callable[..., Any]
    toggle_1d_plots: Callable[..., Any]
    toggle_caked_2d: Callable[..., Any]
    toggle_log_radial: Callable[..., Any]
    toggle_log_azimuth: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeGeometryFitActionWorkflow:
    """Bootstrapped geometry-fit action runtime plus callback aliases."""

    runtime: object
    bindings_factory: Callable[[], Any]
    callback: Callable[..., Any]


def build_runtime_pruning_workflow(
    *,
    bootstrap_module,
    views_module,
    structure_factor_pruning_module,
    view_state,
    bragg_qr_bootstrap_kwargs: Mapping[str, Any] | None = None,
    pruning_controls_bootstrap_kwargs: Mapping[str, Any] | None = None,
    initialize_filters: bool = False,
) -> RuntimePruningWorkflow:
    """Assemble the live pruning workflow and its control-cluster runtime."""

    runtime = bootstrap_module.build_runtime_bragg_qr_workflow_bootstrap(
        structure_factor_pruning_module=structure_factor_pruning_module,
        **dict(bragg_qr_bootstrap_kwargs or {}),
    )
    controls_runtime = bootstrap_module.build_runtime_structure_factor_pruning_controls_bootstrap(
        views_module=views_module,
        structure_factor_pruning_module=structure_factor_pruning_module,
        view_state=view_state,
        on_sf_prune_bias_change=runtime.on_sf_prune_bias_change,
        update_status_label=runtime.update_status_label,
        on_solve_q_steps_change=runtime.on_solve_q_steps_change,
        on_solve_q_rel_tol_change=runtime.on_solve_q_rel_tol_change,
        on_solve_q_mode_change=runtime.on_solve_q_mode_change,
        set_solve_q_control_states=runtime.set_solve_q_control_states,
        **dict(pruning_controls_bootstrap_kwargs or {}),
    )
    apply_filters = runtime.apply_filters
    if initialize_filters and callable(apply_filters):
        apply_filters(trigger_update=False)
    return RuntimePruningWorkflow(
        runtime=runtime,
        controls_runtime=controls_runtime,
        current_sf_prune_bias=runtime.current_sf_prune_bias,
        current_solve_q_values=runtime.current_solve_q_values,
        update_status_label=runtime.update_status_label,
        apply_filters=apply_filters,
        on_sf_prune_bias_change=runtime.on_sf_prune_bias_change,
        on_solve_q_steps_change=runtime.on_solve_q_steps_change,
        on_solve_q_rel_tol_change=runtime.on_solve_q_rel_tol_change,
        set_solve_q_control_states=runtime.set_solve_q_control_states,
        on_solve_q_mode_change=runtime.on_solve_q_mode_change,
    )


def build_runtime_integration_range_workflow(
    *,
    bootstrap_module,
    views_module,
    integration_range_drag_module,
    integration_range_update_bootstrap_kwargs: Mapping[str, Any] | None = None,
) -> RuntimeIntegrationRangeWorkflow:
    """Assemble the integration-range update runtime and callback aliases."""

    update_runtime = bootstrap_module.build_runtime_integration_range_update_bootstrap(
        views_module=views_module,
        integration_range_drag_module=integration_range_drag_module,
        **dict(integration_range_update_bootstrap_kwargs or {}),
    )
    callbacks = update_runtime.callbacks
    return RuntimeIntegrationRangeWorkflow(
        update_runtime=update_runtime,
        callbacks=callbacks,
        schedule_range_update=callbacks.schedule_range_update,
        toggle_1d_plots=callbacks.toggle_1d_plots,
        toggle_caked_2d=callbacks.toggle_caked_2d,
        toggle_log_radial=callbacks.toggle_log_radial,
        toggle_log_azimuth=callbacks.toggle_log_azimuth,
    )


def build_runtime_geometry_fit_action_workflow(
    *,
    bootstrap_module,
    geometry_fit_module,
    geometry_fit_action_bootstrap_kwargs: Mapping[str, Any] | None = None,
) -> RuntimeGeometryFitActionWorkflow:
    """Assemble the geometry-fit action runtime and callback aliases."""

    runtime = bootstrap_module.build_runtime_geometry_fit_action_bootstrap(
        geometry_fit_module=geometry_fit_module,
        **dict(geometry_fit_action_bootstrap_kwargs or {}),
    )
    return RuntimeGeometryFitActionWorkflow(
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        callback=runtime.callback,
    )
