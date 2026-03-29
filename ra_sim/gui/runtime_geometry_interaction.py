"""Import-safe helpers for assembling runtime geometry interaction workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimePeakSelectionWorkflow:
    """Bootstrapped selected-peak runtime plus HKL lookup control wiring."""

    runtime: object
    bindings_factory: Callable[[], Any]
    callbacks: object
    maintenance_callbacks: object
    ensure_peak_overlay_data: Callable[..., Any]
    hkl_lookup_controls_runtime: object


@dataclass(frozen=True)
class RuntimeGeometryManualProjectionWorkflow:
    """Bootstrapped manual-geometry projection runtime plus callback aliases."""

    runtime: object
    callbacks: object
    pick_uses_caked_space: Callable[..., Any]
    current_background_image: Callable[..., Any]
    entry_display_coords: Callable[..., Any]
    caked_angles_to_background_display_coords: Callable[..., Any]
    native_detector_coords_to_caked_display_coords: Callable[..., Any]
    project_peaks_to_current_view: Callable[..., Any]
    simulated_peaks_for_params: Callable[..., Any]
    pick_candidates: Callable[..., Any]
    simulated_lookup: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeGeometryManualCacheWorkflow:
    """Bootstrapped manual-geometry cache runtime plus callback aliases."""

    runtime: object
    callbacks: object
    current_match_config: Callable[..., Any]
    pick_cache_signature: Callable[..., Any]
    get_pick_cache: Callable[..., Any]
    build_initial_pairs_display: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeGeometryManualWorkflow:
    """Bootstrapped manual-geometry interaction runtime plus callback aliases."""

    runtime: object
    callbacks: object
    render_current_pairs: Callable[..., Any]
    toggle_selection_at: Callable[..., Any]
    place_selection_at: Callable[..., Any]
    update_pick_preview: Callable[..., Any]
    cancel_pick_session: Callable[..., Any]


@dataclass(frozen=True)
class RuntimeGeometryToolActionWorkflow:
    """Bootstrapped geometry-tool action runtime plus callback aliases."""

    runtime: object
    callbacks: object
    update_fit_history_button_state: Callable[..., Any]
    update_manual_pick_button_label: Callable[..., Any]
    set_manual_pick_mode: Callable[..., Any]
    toggle_manual_pick_mode: Callable[..., Any]
    clear_current_manual_pairs: Callable[..., Any]


def refresh_runtime_peak_selection_after_update(
    *,
    maintenance_callbacks,
    live_geometry_preview_enabled: bool,
) -> bool:
    """Refresh selected-peak maintenance through the bound maintenance bundle."""

    refresh = getattr(maintenance_callbacks, "refresh_after_simulation_update", None)
    if callable(refresh):
        return bool(refresh(bool(live_geometry_preview_enabled)))
    return False


def apply_restored_runtime_selected_hkl_target(
    *,
    maintenance_callbacks,
    selected_hkl_target: object,
):
    """Reapply one restored selected-HKL target through the maintenance bundle."""

    apply_target = getattr(
        maintenance_callbacks,
        "apply_restored_selected_hkl_target",
        None,
    )
    if callable(apply_target):
        return apply_target(selected_hkl_target)
    return None


def build_runtime_peak_selection_workflow(
    *,
    bootstrap_module,
    peak_selection_module,
    views_module,
    hkl_lookup_view_state,
    open_bragg_qr_groups,
    **selected_peak_kwargs,
) -> RuntimePeakSelectionWorkflow:
    """Assemble selected-peak runtime wiring plus HKL lookup controls."""

    runtime = bootstrap_module.build_runtime_selected_peak_bootstrap(
        peak_selection_module=peak_selection_module,
        **selected_peak_kwargs,
    )
    hkl_lookup_controls_runtime = (
        bootstrap_module.build_runtime_hkl_lookup_controls_bootstrap(
            views_module=views_module,
            view_state=hkl_lookup_view_state,
            peak_selection_callbacks=runtime.callbacks,
            open_bragg_qr_groups=open_bragg_qr_groups,
        )
    )
    return RuntimePeakSelectionWorkflow(
        runtime=runtime,
        bindings_factory=runtime.bindings_factory,
        callbacks=runtime.callbacks,
        maintenance_callbacks=runtime.maintenance_callbacks,
        ensure_peak_overlay_data=runtime.ensure_peak_overlay_data,
        hkl_lookup_controls_runtime=hkl_lookup_controls_runtime,
    )


def build_runtime_geometry_manual_projection_workflow(
    *,
    bootstrap_module,
    manual_geometry_module,
    **projection_kwargs,
) -> RuntimeGeometryManualProjectionWorkflow:
    """Assemble manual-geometry projection runtime wiring and aliases."""

    runtime = bootstrap_module.build_runtime_geometry_manual_projection_bootstrap(
        manual_geometry_module=manual_geometry_module,
        **projection_kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeGeometryManualProjectionWorkflow(
        runtime=runtime,
        callbacks=callbacks,
        pick_uses_caked_space=callbacks.pick_uses_caked_space,
        current_background_image=callbacks.current_background_image,
        entry_display_coords=callbacks.entry_display_coords,
        caked_angles_to_background_display_coords=(
            callbacks.caked_angles_to_background_display_coords
        ),
        native_detector_coords_to_caked_display_coords=(
            callbacks.native_detector_coords_to_caked_display_coords
        ),
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        simulated_peaks_for_params=callbacks.simulated_peaks_for_params,
        pick_candidates=callbacks.pick_candidates,
        simulated_lookup=callbacks.simulated_lookup,
    )


def build_runtime_geometry_manual_cache_workflow(
    *,
    bootstrap_module,
    manual_geometry_module,
    **cache_kwargs,
) -> RuntimeGeometryManualCacheWorkflow:
    """Assemble manual-geometry cache runtime wiring and aliases."""

    runtime = bootstrap_module.build_runtime_geometry_manual_cache_bootstrap(
        manual_geometry_module=manual_geometry_module,
        **cache_kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeGeometryManualCacheWorkflow(
        runtime=runtime,
        callbacks=callbacks,
        current_match_config=callbacks.current_match_config,
        pick_cache_signature=callbacks.pick_cache_signature,
        get_pick_cache=callbacks.get_pick_cache,
        build_initial_pairs_display=callbacks.build_initial_pairs_display,
    )


def build_runtime_geometry_manual_workflow(
    *,
    bootstrap_module,
    manual_geometry_module,
    **manual_kwargs,
) -> RuntimeGeometryManualWorkflow:
    """Assemble manual-geometry runtime wiring and aliases."""

    runtime = bootstrap_module.build_runtime_geometry_manual_bootstrap(
        manual_geometry_module=manual_geometry_module,
        **manual_kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeGeometryManualWorkflow(
        runtime=runtime,
        callbacks=callbacks,
        render_current_pairs=callbacks.render_current_pairs,
        toggle_selection_at=callbacks.toggle_selection_at,
        place_selection_at=callbacks.place_selection_at,
        update_pick_preview=callbacks.update_pick_preview,
        cancel_pick_session=callbacks.cancel_pick_session,
    )


def build_runtime_geometry_tool_action_workflow(
    *,
    bootstrap_module,
    geometry_fit_module,
    **tool_action_kwargs,
) -> RuntimeGeometryToolActionWorkflow:
    """Assemble geometry-tool action runtime wiring and aliases."""

    runtime = bootstrap_module.build_runtime_geometry_tool_action_callbacks_bootstrap(
        geometry_fit_module=geometry_fit_module,
        **tool_action_kwargs,
    )
    callbacks = runtime.callbacks
    return RuntimeGeometryToolActionWorkflow(
        runtime=runtime,
        callbacks=callbacks,
        update_fit_history_button_state=callbacks.update_fit_history_button_state,
        update_manual_pick_button_label=callbacks.update_manual_pick_button_label,
        set_manual_pick_mode=callbacks.set_manual_pick_mode,
        toggle_manual_pick_mode=callbacks.toggle_manual_pick_mode,
        clear_current_manual_pairs=callbacks.clear_current_manual_pairs,
    )


def build_runtime_geometry_tool_action_controls_runtime(
    *,
    bootstrap_module,
    views_module,
    **controls_kwargs,
):
    """Assemble the geometry-tool action control runtime through bootstrap helpers."""

    return bootstrap_module.build_runtime_geometry_tool_action_controls_bootstrap(
        views_module=views_module,
        **controls_kwargs,
    )


def initialize_runtime_geometry_interaction_controls(
    *,
    fit_actions_parent,
    geometry_tool_actions_runtime,
    hkl_lookup_controls_runtime,
    update_fit_history_button_state: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    update_preview_exclude_button_label: Callable[[], None] | None = None,
) -> None:
    """Create geometry-interaction controls and refresh their initial labels."""

    geometry_tool_actions_runtime.create_controls(parent=fit_actions_parent)
    if callable(update_fit_history_button_state):
        update_fit_history_button_state()
    if callable(update_manual_pick_button_label):
        update_manual_pick_button_label()
    if callable(update_preview_exclude_button_label):
        update_preview_exclude_button_label()
    hkl_lookup_controls_runtime.create_controls(parent=fit_actions_parent)
