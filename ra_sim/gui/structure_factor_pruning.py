"""Workflow helpers for structure-factor pruning and solve-q controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import controllers as gui_controllers
from . import views as gui_views


@dataclass
class StructureFactorPruningRuntimeBindings:
    """Runtime callbacks and state used by the pruning / solve-q controls."""

    view_state: Any
    simulation_runtime_state: Any
    bragg_qr_manager_state: Any
    clip_prune_bias: Callable[[object], float]
    clip_solve_q_steps: Callable[[object], int]
    clip_solve_q_rel_tol: Callable[[object], float]
    normalize_solve_q_mode_label: Callable[[object], str]
    apply_filters: Callable[..., Mapping[str, object]] = gui_controllers.apply_bragg_qr_filters
    schedule_update: Callable[[], None] | None = None
    refresh_window: Callable[[], None] | None = None


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def make_runtime_structure_factor_pruning_bindings_factory(
    *,
    view_state_factory: object,
    simulation_runtime_state,
    bragg_qr_manager_state,
    clip_prune_bias: Callable[[object], float],
    clip_solve_q_steps: Callable[[object], int],
    clip_solve_q_rel_tol: Callable[[object], float],
    normalize_solve_q_mode_label: Callable[[object], str],
    apply_filters: Callable[..., Mapping[str, object]] = gui_controllers.apply_bragg_qr_filters,
    schedule_update_factory: object | None = None,
    refresh_window_factory: object | None = None,
) -> Callable[[], StructureFactorPruningRuntimeBindings]:
    """Return a zero-arg factory for live pruning / solve-q runtime bindings."""

    def _build_bindings() -> StructureFactorPruningRuntimeBindings:
        return StructureFactorPruningRuntimeBindings(
            view_state=_resolve_runtime_value(view_state_factory),
            simulation_runtime_state=simulation_runtime_state,
            bragg_qr_manager_state=bragg_qr_manager_state,
            clip_prune_bias=clip_prune_bias,
            clip_solve_q_steps=clip_solve_q_steps,
            clip_solve_q_rel_tol=clip_solve_q_rel_tol,
            normalize_solve_q_mode_label=normalize_solve_q_mode_label,
            apply_filters=apply_filters,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            refresh_window=_resolve_runtime_value(refresh_window_factory),
        )

    return _build_bindings


def make_runtime_structure_factor_pruning_status_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[[], str]:
    """Return a zero-arg callback that refreshes pruning status from live bindings."""

    return lambda: update_runtime_structure_factor_pruning_status(bindings_factory())


def make_runtime_bragg_qr_filter_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., Mapping[str, object]]:
    """Return a callback that applies Bragg-Qr filters from live bindings."""

    def _apply_filters(*, trigger_update: bool = True) -> Mapping[str, object]:
        return apply_runtime_bragg_qr_filters(
            bindings_factory(),
            trigger_update=trigger_update,
        )

    return _apply_filters


def make_runtime_sf_prune_bias_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime SF-pruning bias updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_sf_prune_bias_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_steps_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q interval updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_steps_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_rel_tol_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q tolerance updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_rel_tol_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_control_states_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[[], str]:
    """Return a zero-arg callback that syncs runtime solve-q control state."""

    return lambda: set_runtime_solve_q_control_states(bindings_factory())


def make_runtime_solve_q_mode_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q mode updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_mode_change(bindings_factory())

    return _on_change


def _safe_var_get(var: object) -> object:
    getter = getattr(var, "get", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _safe_var_set(var: object, value: object) -> None:
    setter = getattr(var, "set", None)
    if callable(setter):
        setter(value)


def current_runtime_sf_prune_bias(
    bindings: StructureFactorPruningRuntimeBindings,
) -> float:
    """Return the current clipped SF-pruning bias from the bound view state."""

    return float(
        bindings.clip_prune_bias(
            _safe_var_get(getattr(bindings.view_state, "sf_prune_bias_var", None))
        )
    )


def current_runtime_solve_q_steps(
    bindings: StructureFactorPruningRuntimeBindings,
) -> int:
    """Return the current clipped solve-q interval count from the bound view state."""

    return int(
        bindings.clip_solve_q_steps(
            _safe_var_get(getattr(bindings.view_state, "solve_q_steps_var", None))
        )
    )


def current_runtime_solve_q_rel_tol(
    bindings: StructureFactorPruningRuntimeBindings,
) -> float:
    """Return the current clipped solve-q relative tolerance from the bound view state."""

    return float(
        bindings.clip_solve_q_rel_tol(
            _safe_var_get(getattr(bindings.view_state, "solve_q_rel_tol_var", None))
        )
    )


def current_runtime_solve_q_mode_label(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Return the normalized solve-q mode label from the bound view state."""

    return str(
        bindings.normalize_solve_q_mode_label(
            _safe_var_get(getattr(bindings.view_state, "solve_q_mode_var", None))
        )
    )


def update_runtime_structure_factor_pruning_status(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Refresh the pruning-status label from the current runtime state."""

    text = gui_controllers.format_structure_factor_pruning_status(
        bindings.simulation_runtime_state.sf_prune_stats,
        prune_bias=current_runtime_sf_prune_bias(bindings),
    )
    gui_views.set_structure_factor_pruning_status_text(bindings.view_state, text)
    return str(text)


def invalidate_runtime_bragg_qr_filter_results(
    bindings: StructureFactorPruningRuntimeBindings,
) -> None:
    """Clear cached simulation artifacts invalidated by Bragg-Qr filter changes."""

    bindings.simulation_runtime_state.last_simulation_signature = None
    bindings.simulation_runtime_state.stored_max_positions_local = None
    bindings.simulation_runtime_state.stored_sim_image = None
    bindings.simulation_runtime_state.stored_peak_table_lattice = None
    bindings.simulation_runtime_state.selected_peak_record = None


def apply_runtime_bragg_qr_filters(
    bindings: StructureFactorPruningRuntimeBindings,
    *,
    trigger_update: bool = True,
) -> Mapping[str, object]:
    """Apply Bragg-Qr / SF-pruning filters and runtime side effects."""

    stats = bindings.apply_filters(
        bindings.simulation_runtime_state,
        bindings.bragg_qr_manager_state,
        prune_bias=current_runtime_sf_prune_bias(bindings),
    )
    update_runtime_structure_factor_pruning_status(bindings)
    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.refresh_window):
        bindings.refresh_window()
    if trigger_update and callable(bindings.schedule_update):
        bindings.schedule_update()
    return stats


def set_runtime_solve_q_control_states(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Sync the adaptive relative-tolerance enabled state from the current mode."""

    mode = current_runtime_solve_q_mode_label(bindings)
    gui_views.set_structure_factor_pruning_rel_tol_enabled(
        bindings.view_state,
        enabled=(mode != "uniform"),
    )
    return mode


def on_runtime_sf_prune_bias_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one SF-pruning bias variable change from the runtime GUI."""

    var = getattr(bindings.view_state, "sf_prune_bias_var", None)
    raw_value = _safe_var_get(var)
    clipped_value = current_runtime_sf_prune_bias(bindings)
    try:
        raw_float = float(raw_value)
    except (TypeError, ValueError):
        raw_float = None
    if raw_float is None or not np.isclose(
        raw_float,
        clipped_value,
        rtol=0.0,
        atol=1e-12,
    ):
        _safe_var_set(var, clipped_value)
        return False

    apply_runtime_bragg_qr_filters(bindings, trigger_update=True)
    return True


def on_runtime_solve_q_steps_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q interval-count change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_steps_var", None)
    raw_value = _safe_var_get(var)
    clipped_value = current_runtime_solve_q_steps(bindings)
    try:
        raw_float = float(raw_value)
    except (TypeError, ValueError):
        raw_float = None
    if raw_float is None or not np.isclose(
        raw_float,
        float(clipped_value),
        rtol=0.0,
        atol=1e-12,
    ):
        _safe_var_set(var, float(clipped_value))
        return False

    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True


def on_runtime_solve_q_rel_tol_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q relative-tolerance change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_rel_tol_var", None)
    raw_value = _safe_var_get(var)
    clipped_value = current_runtime_solve_q_rel_tol(bindings)
    try:
        raw_float = float(raw_value)
    except (TypeError, ValueError):
        raw_float = None
    if raw_float is None or not np.isclose(
        raw_float,
        clipped_value,
        rtol=0.0,
        atol=1e-12,
    ):
        _safe_var_set(var, float(clipped_value))
        return False

    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True


def on_runtime_solve_q_mode_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q mode change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_mode_var", None)
    raw_value = _safe_var_get(var)
    normalized = current_runtime_solve_q_mode_label(bindings)
    if raw_value != normalized:
        _safe_var_set(var, normalized)
        return False

    set_runtime_solve_q_control_states(bindings)
    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True
