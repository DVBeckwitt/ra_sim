"""Import-safe helpers for assembling runtime geometry-preview workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeGeometryQGroupWorkflow:
    """Bootstrapped geometry Q-group runtime plus callback aliases."""

    runtime: object
    bindings_factory: Callable[[], Any]
    callbacks: object
    live_preview_enabled: Callable[..., Any]
    render_live_preview_state: Callable[..., Any]
    set_preview_exclude_mode: Callable[..., Any]
    clear_preview_exclusions: Callable[..., Any]
    toggle_preview_exclusion_at: Callable[..., Any]
    toggle_live_preview: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeCanvasInteractionWorkflow:
    """Bootstrapped canvas-interaction runtime plus callback aliases."""

    runtime: object
    bindings_factory: Callable[[], Any]
    callbacks: object


def _resolve_runtime_value(value_or_factory: object) -> object:
    if callable(value_or_factory):
        try:
            return value_or_factory()
        except Exception:
            return None
    return value_or_factory


def build_runtime_geometry_q_group_workflow(
    *,
    bootstrap_module,
    geometry_q_group_manager_module,
    root,
    schedule_update_resolver: object = None,
    **bindings_kwargs,
) -> RuntimeGeometryQGroupWorkflow:
    """Assemble geometry Q-group runtime wiring and aliases."""

    kwargs = dict(bindings_kwargs)
    if schedule_update_resolver is not None and "schedule_update_factory" not in kwargs:
        kwargs["schedule_update_factory"] = lambda: _resolve_runtime_value(
            schedule_update_resolver
        )
    runtime = bootstrap_module.build_runtime_geometry_q_group_bootstrap(
        geometry_q_group_manager_module=geometry_q_group_manager_module,
        root=root,
        **kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeGeometryQGroupWorkflow(
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        callbacks=callbacks,
        live_preview_enabled=callbacks.live_preview_enabled,
        render_live_preview_state=callbacks.render_live_preview_state,
        set_preview_exclude_mode=callbacks.set_preview_exclude_mode,
        clear_preview_exclusions=callbacks.clear_preview_exclusions,
        toggle_preview_exclusion_at=callbacks.toggle_preview_exclusion_at,
        toggle_live_preview=callbacks.toggle_live_preview,
    )


def build_runtime_canvas_interaction_workflow(
    *,
    bootstrap_module,
    canvas_interactions_module,
    geometry_q_group_manager_module,
    geometry_q_group_runtime_bindings_factory_resolver: object,
    **bindings_kwargs,
) -> RuntimeCanvasInteractionWorkflow:
    """Assemble canvas-interaction runtime wiring with late-bound preview hooks."""

    def _set_geometry_preview_exclude_mode(
        enabled: bool,
        message: str | None = None,
    ) -> object:
        bindings_factory = _resolve_runtime_value(
            geometry_q_group_runtime_bindings_factory_resolver
        )
        if not callable(bindings_factory):
            return False
        return geometry_q_group_manager_module.set_runtime_geometry_preview_exclude_mode(
            bindings_factory(),
            enabled,
            message=message,
        )

    def _toggle_live_geometry_preview_exclusion_at(
        col: float,
        row: float,
    ) -> object:
        bindings_factory = _resolve_runtime_value(
            geometry_q_group_runtime_bindings_factory_resolver
        )
        if not callable(bindings_factory):
            return False
        return (
            geometry_q_group_manager_module.toggle_runtime_live_geometry_preview_exclusion_at(
                bindings_factory(),
                col,
                row,
            )
        )

    runtime = bootstrap_module.build_runtime_canvas_interaction_bootstrap(
        canvas_interactions_module=canvas_interactions_module,
        set_geometry_preview_exclude_mode=_set_geometry_preview_exclude_mode,
        toggle_live_geometry_preview_exclusion_at=(
            _toggle_live_geometry_preview_exclusion_at
        ),
        **dict(bindings_kwargs),
    )
    return RuntimeCanvasInteractionWorkflow(
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        callbacks=runtime.callbacks,
    )


def initialize_runtime_canvas_interaction_bindings(
    *,
    canvas,
    callbacks,
) -> tuple[object, ...]:
    """Bind the shared canvas-interaction callbacks to the matplotlib canvas."""

    return (
        canvas.mpl_connect("button_press_event", callbacks.on_click),
        canvas.mpl_connect("button_press_event", callbacks.on_press),
        canvas.mpl_connect("motion_notify_event", callbacks.on_motion),
        canvas.mpl_connect("button_release_event", callbacks.on_release),
        canvas.mpl_connect("scroll_event", callbacks.on_scroll),
    )
