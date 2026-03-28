"""Import-safe helpers for assembling runtime background workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeBackgroundStatusRefreshers:
    """Late-bound status refresh callbacks for background controls."""

    refresh_status: Callable[[], str]
    refresh_backend_status: Callable[[], str]


@dataclass(frozen=True)
class RuntimeBackgroundThetaWorkflow:
    """Bootstrapped background-theta runtime plus bound callback aliases."""

    runtime: object
    callbacks: object
    current_geometry_fit_background_indices: Callable[..., Any]
    geometry_fit_uses_shared_theta_offset: Callable[..., Any]
    current_geometry_theta_offset: Callable[..., Any]
    current_background_theta_values: Callable[..., Any]
    background_theta_for_index: Callable[..., Any]
    sync_background_theta_controls: Callable[..., Any]
    apply_background_theta_metadata: Callable[..., Any]
    apply_geometry_fit_background_selection: Callable[..., Any]
    sync_geometry_fit_background_selection: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeBackgroundWorkflow:
    """Bootstrapped background runtime and control wiring."""

    runtime: object
    bindings_factory: Callable[[], Any]
    callbacks: object
    controls_runtime: object
    toggle_visibility: Callable[..., Any]


def _resolve_runtime_value(value_or_factory: object) -> object:
    if callable(value_or_factory):
        try:
            return value_or_factory()
        except Exception:
            return None
    return value_or_factory


def _refresh_runtime_background_attr(
    attr_name: str,
    *,
    background_controls_runtime_factory: object,
    background_runtime_callbacks_factory: object,
) -> str:
    runtime = _resolve_runtime_value(background_controls_runtime_factory)
    refresh = getattr(runtime, attr_name, None)
    if callable(refresh):
        return str(refresh())

    callbacks = _resolve_runtime_value(background_runtime_callbacks_factory)
    refresh = getattr(callbacks, attr_name, None)
    if callable(refresh):
        return str(refresh())
    return ""


def build_runtime_background_status_refreshers(
    *,
    background_controls_runtime_factory: object,
    background_runtime_callbacks_factory: object,
) -> RuntimeBackgroundStatusRefreshers:
    """Build late-bound status refresh helpers for background workflows."""

    return RuntimeBackgroundStatusRefreshers(
        refresh_status=lambda: _refresh_runtime_background_attr(
            "refresh_status",
            background_controls_runtime_factory=background_controls_runtime_factory,
            background_runtime_callbacks_factory=background_runtime_callbacks_factory,
        ),
        refresh_backend_status=lambda: _refresh_runtime_background_attr(
            "refresh_backend_status",
            background_controls_runtime_factory=background_controls_runtime_factory,
            background_runtime_callbacks_factory=background_runtime_callbacks_factory,
        ),
    )


def build_runtime_background_theta_workflow(
    *,
    bootstrap_module,
    background_theta_module,
    **kwargs,
) -> RuntimeBackgroundThetaWorkflow:
    """Assemble the live background-theta runtime bundle and aliases."""

    runtime = bootstrap_module.build_runtime_background_theta_bootstrap(
        background_theta_module=background_theta_module,
        **kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeBackgroundThetaWorkflow(
        runtime=runtime,
        callbacks=callbacks,
        current_geometry_fit_background_indices=(
            callbacks.current_geometry_fit_background_indices
        ),
        geometry_fit_uses_shared_theta_offset=(
            callbacks.geometry_fit_uses_shared_theta_offset
        ),
        current_geometry_theta_offset=callbacks.current_geometry_theta_offset,
        current_background_theta_values=callbacks.current_background_theta_values,
        background_theta_for_index=callbacks.background_theta_for_index,
        sync_background_theta_controls=callbacks.sync_background_theta_controls,
        apply_background_theta_metadata=callbacks.apply_background_theta_metadata,
        apply_geometry_fit_background_selection=(
            callbacks.apply_geometry_fit_background_selection
        ),
        sync_geometry_fit_background_selection=(
            callbacks.sync_geometry_fit_background_selection
        ),
    )


def build_runtime_background_workflow(
    *,
    bootstrap_module,
    background_manager_module,
    views_module,
    workspace_view_state,
    background_backend_debug_view_state,
    **kwargs,
) -> RuntimeBackgroundWorkflow:
    """Assemble the live background runtime and its control wiring."""

    runtime = bootstrap_module.build_runtime_background_bootstrap(
        background_manager_module=background_manager_module,
        view_state=workspace_view_state,
        **kwargs,
    )
    callbacks = runtime.callbacks
    controls_runtime = bootstrap_module.build_runtime_background_controls_bootstrap(
        views_module=views_module,
        workspace_view_state=workspace_view_state,
        background_backend_debug_view_state=background_backend_debug_view_state,
        background_callbacks=callbacks,
    )
    return RuntimeBackgroundWorkflow(
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        callbacks=callbacks,
        controls_runtime=controls_runtime,
        toggle_visibility=controls_runtime.toggle_visibility,
    )
