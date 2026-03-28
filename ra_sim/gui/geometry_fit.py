"""Pure helpers for GUI geometry-fit state and configuration."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GEOMETRY_FIT_PARAM_ORDER = [
    "zb",
    "zs",
    "theta_initial",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "a",
    "c",
    "center_x",
    "center_y",
]


@dataclass(frozen=True)
class GeometryFitPreparedRun:
    """Prepared inputs and metadata for one manual-pair geometry-fit run."""

    fit_params: dict[str, object]
    selected_background_indices: list[int]
    background_theta_values: list[float]
    joint_background_mode: bool
    current_dataset: dict[str, object]
    dataset_infos: list[dict[str, object]]
    dataset_specs: list[dict[str, object]]
    start_cmd_line: str
    start_log_sections: list[tuple[str, list[str]]]
    max_display_markers: int
    geometry_runtime_cfg: dict[str, object]


@dataclass(frozen=True)
class GeometryFitRuntimePreparationBindings:
    """Runtime values and callbacks used to prepare one geometry-fit run."""

    fit_config: Mapping[str, object] | None
    osc_files: Sequence[object]
    current_background_index: int
    theta_initial: object
    image_size: int
    display_rotate_k: int
    apply_geometry_fit_background_selection: Callable[..., bool]
    current_geometry_fit_background_indices: Callable[..., list[int]]
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    apply_background_theta_metadata: Callable[..., bool]
    current_background_theta_values: Callable[..., list[float]]
    current_geometry_theta_offset: Callable[..., float]
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]]
    ensure_geometry_fit_caked_view: Callable[[], None]
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]]
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None]
    geometry_manual_simulated_peaks_for_params: Callable[..., object]
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]]
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ]
    unrotate_display_peaks: Callable[..., list[dict[str, object]]]
    display_to_native_sim_coords: Callable[..., tuple[float, float]]
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]]
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]]
    orient_image_for_fit: Callable[..., object]
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]]


@dataclass(frozen=True)
class GeometryFitRuntimeValueBindings:
    """Live Tk/runtime value sources used by geometry-fit runtime helpers."""

    fit_zb_var: object
    fit_zs_var: object
    fit_theta_var: object
    fit_psi_z_var: object
    fit_chi_var: object
    fit_cor_var: object
    fit_gamma_var: object
    fit_Gamma_var: object
    fit_dist_var: object
    fit_a_var: object
    fit_c_var: object
    fit_center_x_var: object
    fit_center_y_var: object
    zb_var: object
    zs_var: object
    theta_initial_var: object
    psi_z_var: object
    chi_var: object
    cor_angle_var: object
    gamma_var: object
    Gamma_var: object
    corto_detector_var: object
    a_var: object
    c_var: object
    center_x_var: object
    center_y_var: object
    debye_x_var: object
    debye_y_var: object
    geometry_theta_offset_var: object | None
    current_background_index: object
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    current_geometry_theta_offset: Callable[..., float]
    background_theta_for_index: Callable[..., object]
    build_mosaic_params: Callable[[], Mapping[str, object] | None]
    current_optics_mode_flag: Callable[[], object]
    lambda_value: object
    psi: object
    n2: object


@dataclass(frozen=True)
class GeometryFitRuntimeSolverInputs:
    """Simulation inputs needed to invoke the live geometry-fit solver."""

    miller: object
    intensities: object
    image_size: int


@dataclass(frozen=True)
class GeometryFitSolverRequest:
    """One concrete geometry-fit solver request."""

    miller: object
    intensities: object
    image_size: int
    params: dict[str, object]
    measured_peaks: object
    var_names: list[str]
    experimental_image: object
    dataset_specs: list[dict[str, object]] | None
    refinement_config: dict[str, object]


@dataclass(frozen=True)
class GeometryFitPreparationResult:
    """One geometry-fit preflight result."""

    prepared_run: GeometryFitPreparedRun | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryFitPostprocessResult:
    """Pure post-solver geometry-fit analysis results."""

    fitted_params: dict[str, object]
    point_match_summary_lines: list[str]
    pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]]
    overlay_records: list[dict[str, object]]
    overlay_state: dict[str, object]
    overlay_diagnostic_lines: list[str]
    frame_warning: str | None
    export_records: list[dict[str, object]]
    save_path: Path
    fit_summary_lines: list[str]
    progress_text: str


@dataclass(frozen=True)
class GeometryFitRuntimeResultBindings:
    """Runtime callback bundle for applying one successful geometry fit."""

    log_section: Callable[[str, Sequence[str]], None]
    capture_undo_state: Callable[[], dict[str, object]]
    apply_result_values: Callable[[Sequence[object], Sequence[object]], None]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    build_profile_cache: Callable[[], dict[str, object]]
    replace_profile_cache: Callable[[dict[str, object]], None]
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    build_fitted_params: Callable[[], dict[str, object]]
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]


@dataclass(frozen=True)
class GeometryFitRuntimeUiBindings:
    """Runtime callbacks and UI state sources used during one geometry fit."""

    fit_params: Mapping[str, object] | None
    base_profile_cache: Mapping[str, object] | None
    mosaic_params: Mapping[str, object] | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    geometry_theta_offset_var: object | None
    capture_undo_state: Callable[[], dict[str, object]]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    replace_profile_cache: Callable[[dict[str, object]], None]
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]


@dataclass(frozen=True)
class GeometryFitRuntimePostprocessConfig:
    """Post-solver inputs needed to analyze and persist one geometry fit."""

    current_background_index: int
    downloads_dir: Path | str
    stamp: str
    log_path: Path | str
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionSetup:
    """Prepared runtime execution inputs for one geometry-fit run."""

    ui_bindings: GeometryFitRuntimeUiBindings
    postprocess_config: GeometryFitRuntimePostprocessConfig


@dataclass(frozen=True)
class GeometryFitRuntimeValueCallbacks:
    """Bound callbacks that expose live geometry-fit values from runtime."""

    current_var_names: Callable[[], list[str]]
    current_params: Callable[[], dict[str, object]]
    current_ui_params: Callable[[], dict[str, object]]
    var_map: Mapping[str, object]


@dataclass(frozen=True)
class GeometryFitRuntimeActionExecutionBindings:
    """Live runtime sources needed to build and run one geometry-fit action."""

    downloads_dir: Path | str
    simulation_runtime_state: Any
    background_runtime_state: Any
    theta_initial_var: Any
    geometry_theta_offset_var: Any | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    background_theta_for_index: Callable[..., object]
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    capture_undo_state: Callable[[], dict[str, object]]
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]


@dataclass(frozen=True)
class GeometryFitRuntimeActionBindings:
    """Bound runtime callbacks that drive one top-level geometry-fit action."""

    value_callbacks: GeometryFitRuntimeValueCallbacks
    prepare_bindings_factory: Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]
    execution_bindings: GeometryFitRuntimeActionExecutionBindings
    solve_fit: Callable[..., object]
    stamp_factory: Callable[[], str]
    flush_ui: Callable[[], None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionResult:
    """Result metadata for one top-level geometry-fit action invocation."""

    params: Mapping[str, object]
    var_names: list[str]
    preserve_live_theta: bool
    prepare_result: GeometryFitPreparationResult | None = None
    execution_result: GeometryFitRuntimeExecutionResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeApplyResult:
    """Result metadata returned after applying one successful geometry fit."""

    rms: float
    fitted_params: dict[str, object]
    postprocess: GeometryFitPostprocessResult


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionResult:
    """Result metadata for one full runtime geometry-fit execution."""

    log_path: Path
    solver_request: GeometryFitSolverRequest | None = None
    solver_result: object | None = None
    apply_result: GeometryFitRuntimeApplyResult | None = None
    error_text: str | None = None


def make_runtime_geometry_fit_action_prepare_bindings_factory(
    *,
    fit_config: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial: object,
    image_size: int,
    display_rotate_k: int,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    ensure_geometry_fit_caked_view: Callable[[], None],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
) -> Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]:
    """Build the live prepare-bundle factory for one geometry-fit action."""

    def _build(var_names: Sequence[str]) -> GeometryFitRuntimePreparationBindings:
        return GeometryFitRuntimePreparationBindings(
            fit_config=fit_config,
            osc_files=osc_files,
            current_background_index=int(current_background_index),
            theta_initial=theta_initial,
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            apply_geometry_fit_background_selection=(
                apply_geometry_fit_background_selection
            ),
            current_geometry_fit_background_indices=(
                current_geometry_fit_background_indices
            ),
            geometry_fit_uses_shared_theta_offset=(
                geometry_fit_uses_shared_theta_offset
            ),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            load_background_by_index=load_background_by_index,
            apply_background_backend_orientation=apply_background_backend_orientation,
            geometry_manual_simulated_peaks_for_params=(
                geometry_manual_simulated_peaks_for_params
            ),
            geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
            geometry_manual_entry_display_coords=(
                geometry_manual_entry_display_coords
            ),
            unrotate_display_peaks=unrotate_display_peaks,
            display_to_native_sim_coords=display_to_native_sim_coords,
            select_fit_orientation=select_fit_orientation,
            apply_orientation_to_entries=apply_orientation_to_entries,
            orient_image_for_fit=orient_image_for_fit,
            build_runtime_config=(
                lambda fit_params: build_runtime_config_factory(
                    list(var_names),
                    fit_params,
                )
            ),
        )

    return _build


def build_runtime_geometry_fit_action_execution_bindings(
    *,
    downloads_dir: Path | str,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
) -> GeometryFitRuntimeActionExecutionBindings:
    """Build the live execution-bundle for one geometry-fit action."""

    return GeometryFitRuntimeActionExecutionBindings(
        downloads_dir=downloads_dir,
        simulation_runtime_state=simulation_runtime_state,
        background_runtime_state=background_runtime_state,
        theta_initial_var=theta_initial_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        current_ui_params=current_ui_params,
        var_map=var_map,
        background_theta_for_index=background_theta_for_index,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        capture_undo_state=capture_undo_state,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
    )


def build_runtime_geometry_fit_action_bindings(
    *,
    value_callbacks: GeometryFitRuntimeValueCallbacks,
    fit_config: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial: object,
    image_size: int,
    display_rotate_k: int,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    ensure_geometry_fit_caked_view: Callable[[], None],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
    downloads_dir: Path | str,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeActionBindings:
    """Build the top-level live geometry-fit action bindings."""

    return GeometryFitRuntimeActionBindings(
        value_callbacks=value_callbacks,
        prepare_bindings_factory=make_runtime_geometry_fit_action_prepare_bindings_factory(
            fit_config=fit_config,
            osc_files=osc_files,
            current_background_index=int(current_background_index),
            theta_initial=theta_initial,
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            apply_geometry_fit_background_selection=(
                apply_geometry_fit_background_selection
            ),
            current_geometry_fit_background_indices=(
                current_geometry_fit_background_indices
            ),
            geometry_fit_uses_shared_theta_offset=(
                geometry_fit_uses_shared_theta_offset
            ),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            load_background_by_index=load_background_by_index,
            apply_background_backend_orientation=apply_background_backend_orientation,
            geometry_manual_simulated_peaks_for_params=(
                geometry_manual_simulated_peaks_for_params
            ),
            geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
            geometry_manual_entry_display_coords=(
                geometry_manual_entry_display_coords
            ),
            unrotate_display_peaks=unrotate_display_peaks,
            display_to_native_sim_coords=display_to_native_sim_coords,
            select_fit_orientation=select_fit_orientation,
            apply_orientation_to_entries=apply_orientation_to_entries,
            orient_image_for_fit=orient_image_for_fit,
            build_runtime_config_factory=build_runtime_config_factory,
        ),
        execution_bindings=build_runtime_geometry_fit_action_execution_bindings(
            downloads_dir=downloads_dir,
            simulation_runtime_state=simulation_runtime_state,
            background_runtime_state=background_runtime_state,
            theta_initial_var=theta_initial_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_ui_params=current_ui_params,
            var_map=var_map,
            background_theta_for_index=background_theta_for_index,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            capture_undo_state=capture_undo_state,
            push_undo_state=push_undo_state,
            request_preview_skip_once=request_preview_skip_once,
            schedule_update=schedule_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            set_last_overlay_state=set_last_overlay_state,
            set_progress_text=set_progress_text,
            cmd_line=cmd_line,
            solver_inputs=solver_inputs,
            sim_display_rotate_k=int(sim_display_rotate_k),
            background_display_rotate_k=int(background_display_rotate_k),
            simulate_and_compare_hkl=simulate_and_compare_hkl,
            aggregate_match_centers=aggregate_match_centers,
            build_overlay_records=build_overlay_records,
            compute_frame_diagnostics=compute_frame_diagnostics,
        ),
        solve_fit=solve_fit,
        stamp_factory=stamp_factory,
        flush_ui=flush_ui,
    )


def copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: copy_geometry_fit_state_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(copy_geometry_fit_state_value(val) for val in value)
    return value


def current_geometry_fit_ui_params(
    *,
    zb: float,
    zs: float,
    theta_initial: float,
    psi_z: float,
    chi: float,
    cor_angle: float,
    gamma: float,
    Gamma: float,
    corto_detector: float,
    a: float,
    c: float,
    center_x: float,
    center_y: float,
    theta_offset: float | None = None,
) -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    params = {
        "zb": float(zb),
        "zs": float(zs),
        "theta_initial": float(theta_initial),
        "psi_z": float(psi_z),
        "chi": float(chi),
        "cor_angle": float(cor_angle),
        "gamma": float(gamma),
        "Gamma": float(Gamma),
        "corto_detector": float(corto_detector),
        "a": float(a),
        "c": float(c),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center": [float(center_x), float(center_y)],
    }
    if theta_offset is not None:
        params["theta_offset"] = float(theta_offset)
    return params


def current_geometry_fit_var_names(
    *,
    fit_zb: bool,
    fit_zs: bool,
    fit_theta: bool,
    fit_psi_z: bool,
    fit_chi: bool,
    fit_cor: bool,
    fit_gamma: bool,
    fit_Gamma: bool,
    fit_dist: bool,
    fit_a: bool,
    fit_c: bool,
    fit_center_x: bool,
    fit_center_y: bool,
    use_shared_theta_offset: bool = False,
) -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    var_names: list[str] = []
    if fit_zb:
        var_names.append("zb")
    if fit_zs:
        var_names.append("zs")
    if fit_theta:
        var_names.append("theta_offset" if use_shared_theta_offset else "theta_initial")
    if fit_psi_z:
        var_names.append("psi_z")
    if fit_chi:
        var_names.append("chi")
    if fit_cor:
        var_names.append("cor_angle")
    if fit_gamma:
        var_names.append("gamma")
    if fit_Gamma:
        var_names.append("Gamma")
    if fit_dist:
        var_names.append("corto_detector")
    if fit_a:
        var_names.append("a")
    if fit_c:
        var_names.append("c")
    if fit_center_x:
        var_names.append("center_x")
    if fit_center_y:
        var_names.append("center_y")
    return var_names


def build_runtime_geometry_fit_value_callbacks(
    bindings: GeometryFitRuntimeValueBindings,
) -> GeometryFitRuntimeValueCallbacks:
    """Build bound live geometry-fit value readers for the runtime."""

    var_map = {
        "zb": bindings.zb_var,
        "zs": bindings.zs_var,
        "theta_initial": bindings.theta_initial_var,
        "psi_z": bindings.psi_z_var,
        "chi": bindings.chi_var,
        "cor_angle": bindings.cor_angle_var,
        "gamma": bindings.gamma_var,
        "Gamma": bindings.Gamma_var,
        "corto_detector": bindings.corto_detector_var,
        "a": bindings.a_var,
        "c": bindings.c_var,
        "center_x": bindings.center_x_var,
        "center_y": bindings.center_y_var,
    }

    def _current_var_names() -> list[str]:
        return current_geometry_fit_var_names(
            fit_zb=bool(bindings.fit_zb_var.get()),
            fit_zs=bool(bindings.fit_zs_var.get()),
            fit_theta=bool(bindings.fit_theta_var.get()),
            fit_psi_z=bool(bindings.fit_psi_z_var.get()),
            fit_chi=bool(bindings.fit_chi_var.get()),
            fit_cor=bool(bindings.fit_cor_var.get()),
            fit_gamma=bool(bindings.fit_gamma_var.get()),
            fit_Gamma=bool(bindings.fit_Gamma_var.get()),
            fit_dist=bool(bindings.fit_dist_var.get()),
            fit_a=bool(bindings.fit_a_var.get()),
            fit_c=bool(bindings.fit_c_var.get()),
            fit_center_x=bool(bindings.fit_center_x_var.get()),
            fit_center_y=bool(bindings.fit_center_y_var.get()),
            use_shared_theta_offset=bool(
                bindings.geometry_fit_uses_shared_theta_offset()
            ),
        )

    def _current_params() -> dict[str, object]:
        use_theta_offset = bool(bindings.geometry_fit_uses_shared_theta_offset())
        theta_offset_current = (
            float(bindings.current_geometry_theta_offset(strict=False))
            if use_theta_offset
            else 0.0
        )
        current_background_index_value = (
            bindings.current_background_index()
            if callable(bindings.current_background_index)
            else bindings.current_background_index
        )
        current_background_index = int(current_background_index_value)
        theta_current = (
            bindings.background_theta_for_index(
                current_background_index,
                strict_count=False,
            )
            if use_theta_offset
            else bindings.theta_initial_var.get()
        )
        return {
            "a": bindings.a_var.get(),
            "c": bindings.c_var.get(),
            "lambda": bindings.lambda_value,
            "psi": bindings.psi,
            "psi_z": bindings.psi_z_var.get(),
            "zs": bindings.zs_var.get(),
            "zb": bindings.zb_var.get(),
            "chi": bindings.chi_var.get(),
            "n2": bindings.n2,
            "mosaic_params": dict(bindings.build_mosaic_params() or {}),
            "debye_x": bindings.debye_x_var.get(),
            "debye_y": bindings.debye_y_var.get(),
            "center": [bindings.center_x_var.get(), bindings.center_y_var.get()],
            "center_x": bindings.center_x_var.get(),
            "center_y": bindings.center_y_var.get(),
            "theta_initial": theta_current,
            "theta_offset": theta_offset_current,
            "uv1": np.array([1.0, 0.0, 0.0]),
            "uv2": np.array([0.0, 1.0, 0.0]),
            "corto_detector": bindings.corto_detector_var.get(),
            "gamma": bindings.gamma_var.get(),
            "Gamma": bindings.Gamma_var.get(),
            "cor_angle": bindings.cor_angle_var.get(),
            "optics_mode": bindings.current_optics_mode_flag(),
        }

    def _current_ui_params() -> dict[str, object]:
        theta_offset = None
        if bindings.geometry_theta_offset_var is not None:
            theta_offset = float(bindings.current_geometry_theta_offset(strict=False))
        return current_geometry_fit_ui_params(
            zb=float(bindings.zb_var.get()),
            zs=float(bindings.zs_var.get()),
            theta_initial=float(bindings.theta_initial_var.get()),
            psi_z=float(bindings.psi_z_var.get()),
            chi=float(bindings.chi_var.get()),
            cor_angle=float(bindings.cor_angle_var.get()),
            gamma=float(bindings.gamma_var.get()),
            Gamma=float(bindings.Gamma_var.get()),
            corto_detector=float(bindings.corto_detector_var.get()),
            a=float(bindings.a_var.get()),
            c=float(bindings.c_var.get()),
            center_x=float(bindings.center_x_var.get()),
            center_y=float(bindings.center_y_var.get()),
            theta_offset=theta_offset,
        )

    return GeometryFitRuntimeValueCallbacks(
        current_var_names=_current_var_names,
        current_params=_current_params,
        current_ui_params=_current_ui_params,
        var_map=var_map,
    )


def build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    runtime_cfg = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    bounds_cfg = runtime_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    runtime_cfg["bounds"] = bounds_cfg

    priors_cfg = runtime_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    runtime_cfg["priors"] = priors_cfg

    for name, current in (current_params or {}).items():
        try:
            current_value = float(current)
        except Exception:
            continue
        if not np.isfinite(current_value):
            continue

        control = (control_settings or {}).get(name, {}) or {}
        try:
            window = float(control.get("window", 0.0))
        except Exception:
            window = 0.0
        if not np.isfinite(window):
            window = 0.0
        window = max(0.0, float(window))

        lo = float(current_value - window)
        hi = float(current_value + window)

        domain = (parameter_domains or {}).get(name)
        if isinstance(domain, (list, tuple)) and len(domain) >= 2:
            try:
                domain_lo = float(domain[0])
                domain_hi = float(domain[1])
            except Exception:
                domain_lo = float("nan")
                domain_hi = float("nan")
            if np.isfinite(domain_lo):
                lo = max(lo, float(domain_lo))
            if np.isfinite(domain_hi):
                hi = min(hi, float(domain_hi))

        if hi < lo:
            lo = hi = min(max(current_value, lo), hi)

        bounds_cfg[str(name)] = [float(lo), float(hi)]

        try:
            pull = float(control.get("pull", 0.0))
        except Exception:
            pull = 0.0
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        if pull > 0.0 and window > 0.0:
            sigma_scale = max(0.05, 1.0 - 0.95 * pull)
            priors_cfg[str(name)] = {
                "center": float(current_value),
                "sigma": float(max(window * sigma_scale, 1.0e-6)),
            }
        else:
            priors_cfg.pop(str(name), None)

    return runtime_cfg


def apply_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
):
    """Apply saved UI values and return copied cache/overlay state."""

    if not isinstance(state, dict):
        return {
            "profile_cache": {},
            "overlay_state": None,
        }

    ui_params = state.get("ui_params", {}) or {}
    for name, var in var_map.items():
        try:
            value = float(ui_params.get(name))
        except Exception:
            continue
        if np.isfinite(value):
            var.set(value)

    if geometry_theta_offset_var is not None:
        try:
            theta_offset = float(ui_params.get("theta_offset", 0.0))
        except Exception:
            theta_offset = 0.0
        if np.isfinite(theta_offset):
            geometry_theta_offset_var.set(f"{theta_offset:.6g}")

    overlay_state = copy_geometry_fit_state_value(state.get("overlay_state"))
    if not overlay_state:
        overlay_state = None

    return {
        "profile_cache": copy_geometry_fit_state_value(state.get("profile_cache", {})) or {},
        "overlay_state": overlay_state,
    }


def build_geometry_manual_fit_dataset(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    image_size: int,
    display_rotate_k: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    orientation_cfg: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one saved-manual-pair geometry dataset for the optimizer."""

    background_idx = int(background_index)
    selected_entries = list(geometry_manual_pairs_for_index(background_idx) or ())
    if not selected_entries:
        raise RuntimeError(
            f"background {background_idx + 1} has no saved manual geometry pairs"
        )

    native_background, display_background = load_background_by_index(background_idx)
    backend_background = apply_background_backend_orientation(native_background)
    if backend_background is None:
        backend_background = native_background

    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    params_i["theta_initial"] = float(theta_base + theta_offset)
    simulated_peaks = geometry_manual_simulated_peaks_for_params(
        params_i,
        prefer_cache=(background_idx == int(current_background_index)),
    )
    simulated_lookup = geometry_manual_simulated_lookup(simulated_peaks)

    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, entry in enumerate(selected_entries):
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        measured_display.append(measured_entry)

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": entry.get("hkl", entry.get("label")),
        }
        bg_coords = geometry_manual_entry_display_coords(entry)
        if bg_coords is not None and len(bg_coords) >= 2:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        try:
            source_key = (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            )
        except Exception:
            source_key = None
        if source_key is not None:
            sim_entry = simulated_lookup.get(source_key)
            if isinstance(sim_entry, Mapping):
                try:
                    sim_col = float(sim_entry.get("sim_col"))
                    sim_row = float(sim_entry.get("sim_row"))
                except Exception:
                    sim_col = float("nan")
                    sim_row = float("nan")
                if np.isfinite(sim_col) and np.isfinite(sim_row):
                    initial_entry["sim_display"] = (float(sim_col), float(sim_row))
        initial_pairs_display.append(initial_entry)

    measured_native = unrotate_display_peaks(
        measured_display,
        display_background.shape,
        k=display_rotate_k,
    )

    sim_orientation_points: list[tuple[float, float]] = []
    meas_orientation_points: list[tuple[float, float]] = []
    sim_native_shape = (int(image_size), int(image_size))
    for initial_entry, measured_entry in zip(initial_pairs_display, measured_native):
        if not isinstance(measured_entry, Mapping):
            continue
        sim_display = initial_entry.get("sim_display")
        if (
            not isinstance(sim_display, (list, tuple, np.ndarray))
            or len(sim_display) < 2
        ):
            continue
        try:
            sim_native = display_to_native_sim_coords(
                float(sim_display[0]),
                float(sim_display[1]),
                sim_native_shape,
            )
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if not (
            np.isfinite(sim_native[0])
            and np.isfinite(sim_native[1])
            and np.isfinite(mx)
            and np.isfinite(my)
        ):
            continue
        sim_orientation_points.append((float(sim_native[0]), float(sim_native[1])))
        meas_orientation_points.append((float(mx), float(my)))

    orientation_choice, orientation_diag = select_fit_orientation(
        sim_orientation_points,
        meas_orientation_points,
        tuple(int(v) for v in native_background.shape[:2]),
        cfg=orientation_cfg or {},
    )
    measured_for_fit = apply_orientation_to_entries(
        measured_native,
        native_background.shape,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    experimental_image_for_fit = orient_image_for_fit(
        backend_background,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )

    label = (
        Path(str(osc_files[background_idx])).name
        if 0 <= background_idx < len(osc_files)
        else f"background_{background_idx}"
    )
    group_count = len(
        {
            entry.get("q_group_key")
            for entry in selected_entries
            if entry.get("q_group_key") is not None
        }
    )

    return {
        "dataset_index": int(background_idx),
        "label": label,
        "theta_base": float(theta_base),
        "theta_effective": float(theta_base + theta_offset),
        "group_count": int(group_count),
        "pair_count": int(len(measured_display)),
        "measured_display": measured_display,
        "measured_native": measured_native,
        "measured_for_fit": measured_for_fit,
        "initial_pairs_display": initial_pairs_display,
        "experimental_image_for_fit": experimental_image_for_fit,
        "native_background": native_background,
        "orientation_choice": orientation_choice,
        "orientation_diag": orientation_diag,
        "summary_line": (
            "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
            "groups={groups} points={points} orientation={orientation}"
        ).format(
            idx=background_idx,
            name=label,
            theta_base=float(theta_base),
            theta_eff=float(theta_base + theta_offset),
            groups=int(group_count),
            points=int(len(measured_display)),
            orientation=orientation_choice.get("label", "identity"),
        ),
        "spec": {
            "dataset_index": int(background_idx),
            "label": label,
            "theta_initial": float(theta_base),
            "measured_peaks": measured_for_fit,
            "experimental_image": experimental_image_for_fit,
        },
    }


def prepare_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    fit_config: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial: object,
    preserve_live_theta: bool,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    ensure_geometry_fit_caked_view: Callable[[], None],
    build_dataset: Callable[..., dict[str, object]],
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]],
) -> GeometryFitPreparationResult:
    """Validate and assemble the manual-pair geometry-fit runtime inputs."""

    selected_var_names = [str(name) for name in (var_names or ())]
    if not selected_var_names:
        return GeometryFitPreparationResult(
            error_text="No geometry parameters are selected for fitting."
        )

    fit_params = dict(params or {})
    current_index = int(current_background_index)

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, Mapping):
        orientation_cfg = {}
    overlay_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(overlay_cfg, Mapping):
        overlay_cfg = {}
    max_display_markers = max(1, int(overlay_cfg.get("max_display_markers", 120)))

    if not osc_files:
        return GeometryFitPreparationResult(
            error_text="Geometry fit unavailable: no background image is loaded."
        )

    if not apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return GeometryFitPreparationResult()

    try:
        selected_background_indices = current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        return GeometryFitPreparationResult(
            error_text=(
                "Geometry fit unavailable: invalid fit background selection "
                f"({exc})."
            )
        )

    if current_index not in {int(idx) for idx in selected_background_indices}:
        return GeometryFitPreparationResult(
            error_text=(
                "Geometry fit unavailable: the active background must be part of "
                "the fit selection so the overlay can be drawn on the current image."
            )
        )

    joint_background_mode = False
    background_theta_values: list[float] = []
    if geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            return GeometryFitPreparationResult()
        try:
            background_theta_values = list(
                current_background_theta_values(strict_count=True)
            )
            fit_params["theta_offset"] = float(current_geometry_theta_offset(strict=True))
        except Exception as exc:
            return GeometryFitPreparationResult(
                error_text=(
                    "Geometry fit unavailable: failed to parse background theta "
                    f"settings ({exc})."
                )
            )
        joint_background_mode = len(selected_background_indices) > 1
        if background_theta_values:
            fit_params["theta_initial"] = float(
                background_theta_values[current_index]
                + float(fit_params.get("theta_offset", 0.0))
            )
    else:
        fit_params["theta_offset"] = 0.0
        fit_params["theta_initial"] = float(
            fit_params.get("theta_initial", theta_initial)
        )
        background_theta_values = [float(fit_params["theta_initial"])]

    required_indices = (
        list(selected_background_indices)
        if joint_background_mode
        else [int(current_index)]
    )
    missing_indices = [
        idx
        for idx in required_indices
        if not geometry_manual_pairs_for_index(int(idx))
    ]
    if missing_indices:
        missing_names = [
            Path(str(osc_files[idx])).name
            for idx in missing_indices
            if 0 <= idx < len(osc_files)
        ]
        return GeometryFitPreparationResult(
            error_text=(
                "Geometry fit unavailable: save manual Qr/Qz pairs first for "
                + ", ".join(
                    missing_names
                    or [f"background {idx + 1}" for idx in missing_indices]
                )
                + "."
            )
        )

    try:
        ensure_geometry_fit_caked_view()
    except Exception as exc:
        return GeometryFitPreparationResult(
            error_text=(
                "Geometry fit unavailable: failed to prepare the 2D caked view "
                f"({exc})."
            )
        )

    current_theta_base = (
        float(background_theta_values[current_index])
        if joint_background_mode
        else float(fit_params.get("theta_initial", theta_initial))
    )
    current_dataset = build_dataset(
        int(current_index),
        theta_base=float(current_theta_base),
        base_fit_params=fit_params,
        orientation_cfg=dict(orientation_cfg),
    )
    dataset_infos = [current_dataset]
    if joint_background_mode:
        for bg_idx in selected_background_indices:
            idx = int(bg_idx)
            if idx == current_index:
                continue
            dataset_infos.append(
                build_dataset(
                    idx,
                    theta_base=float(background_theta_values[idx]),
                    base_fit_params=fit_params,
                    orientation_cfg=dict(orientation_cfg),
                )
            )

    dataset_specs = build_geometry_fit_dataset_specs(dataset_infos)
    return GeometryFitPreparationResult(
        prepared_run=GeometryFitPreparedRun(
            fit_params=fit_params,
            selected_background_indices=[int(idx) for idx in selected_background_indices],
            background_theta_values=[float(value) for value in background_theta_values],
            joint_background_mode=bool(joint_background_mode),
            current_dataset=current_dataset,
            dataset_infos=dataset_infos,
            dataset_specs=dataset_specs,
            start_cmd_line=build_geometry_fit_start_cmd_line(
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
            ),
            start_log_sections=build_geometry_fit_start_log_sections(
                params=fit_params,
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
            ),
            max_display_markers=int(max_display_markers),
            geometry_runtime_cfg=dict(build_runtime_config(fit_params) or {}),
        )
    )


def prepare_runtime_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    preserve_live_theta: bool,
    bindings: GeometryFitRuntimePreparationBindings,
) -> GeometryFitPreparationResult:
    """Prepare one geometry fit from the live runtime value/callback sources."""

    fit_config = bindings.fit_config if isinstance(bindings.fit_config, Mapping) else {}

    return prepare_geometry_fit_run(
        params=params,
        var_names=var_names,
        fit_config=fit_config,
        osc_files=bindings.osc_files,
        current_background_index=int(bindings.current_background_index),
        theta_initial=bindings.theta_initial,
        preserve_live_theta=preserve_live_theta,
        apply_geometry_fit_background_selection=(
            bindings.apply_geometry_fit_background_selection
        ),
        current_geometry_fit_background_indices=(
            bindings.current_geometry_fit_background_indices
        ),
        geometry_fit_uses_shared_theta_offset=(
            bindings.geometry_fit_uses_shared_theta_offset
        ),
        apply_background_theta_metadata=bindings.apply_background_theta_metadata,
        current_background_theta_values=bindings.current_background_theta_values,
        current_geometry_theta_offset=bindings.current_geometry_theta_offset,
        geometry_manual_pairs_for_index=bindings.geometry_manual_pairs_for_index,
        ensure_geometry_fit_caked_view=bindings.ensure_geometry_fit_caked_view,
        build_dataset=(
            lambda background_index, *, theta_base, base_fit_params, orientation_cfg: (
                build_geometry_manual_fit_dataset(
                    background_index,
                    theta_base=theta_base,
                    base_fit_params=base_fit_params,
                    osc_files=bindings.osc_files,
                    current_background_index=int(bindings.current_background_index),
                    image_size=int(bindings.image_size),
                    display_rotate_k=int(bindings.display_rotate_k),
                    geometry_manual_pairs_for_index=(
                        bindings.geometry_manual_pairs_for_index
                    ),
                    load_background_by_index=bindings.load_background_by_index,
                    apply_background_backend_orientation=(
                        bindings.apply_background_backend_orientation
                    ),
                    geometry_manual_simulated_peaks_for_params=(
                        bindings.geometry_manual_simulated_peaks_for_params
                    ),
                    geometry_manual_simulated_lookup=(
                        bindings.geometry_manual_simulated_lookup
                    ),
                    geometry_manual_entry_display_coords=(
                        bindings.geometry_manual_entry_display_coords
                    ),
                    unrotate_display_peaks=bindings.unrotate_display_peaks,
                    display_to_native_sim_coords=(
                        bindings.display_to_native_sim_coords
                    ),
                    select_fit_orientation=bindings.select_fit_orientation,
                    apply_orientation_to_entries=(
                        bindings.apply_orientation_to_entries
                    ),
                    orient_image_for_fit=bindings.orient_image_for_fit,
                    orientation_cfg=orientation_cfg,
                )
            )
        ),
        build_runtime_config=(
            lambda fit_params: bindings.build_runtime_config(
                dict(fit_params or {})
            )
        ),
    )


def build_geometry_fit_dataset_specs(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Copy optimizer dataset specs from the prepared dataset metadata."""

    dataset_specs: list[dict[str, object]] = []
    for info in dataset_infos or ():
        if not isinstance(info, Mapping):
            continue
        spec = info.get("spec")
        if isinstance(spec, Mapping):
            dataset_specs.append(dict(spec))
    return dataset_specs


def build_geometry_fit_start_cmd_line(
    *,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> str:
    """Build the console start line for one geometry-fit run."""

    dataset_list = list(dataset_infos or ())
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    return (
        "start: "
        f"vars={','.join(str(name) for name in var_names)} "
        f"datasets={len(dataset_list)} "
        f"current_groups={int(dataset.get('group_count', 0) or 0)} "
        f"current_points={int(dataset.get('pair_count', 0) or 0)}"
    )


def build_geometry_fit_start_log_sections(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> list[tuple[str, list[str]]]:
    """Build the start-log sections for one geometry-fit run."""

    fit_params = params if isinstance(params, Mapping) else {}
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    orientation_diag = dataset.get("orientation_diag") or {}
    if not isinstance(orientation_diag, Mapping):
        orientation_diag = {}
    orientation_choice = dataset.get("orientation_choice") or {}
    if not isinstance(orientation_choice, Mapping):
        orientation_choice = {}
    dataset_lines = [
        str(info.get("summary_line", ""))
        for info in (dataset_infos or ())
        if isinstance(info, Mapping)
    ] or ["<none>"]
    return [
        (
            "Fitting variables (start values):",
            [
                f"{name}={float(fit_params.get(str(name), np.nan)):.6f}"
                for name in var_names
            ],
        ),
        (
            "Manual geometry datasets:",
            dataset_lines,
        ),
        (
            "Current orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"chosen={orientation_choice.get('label', 'identity')}",
                f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                f"reason={orientation_diag.get('reason', 'n/a')}",
            ],
        ),
    ]


def write_geometry_fit_run_start_log(
    *,
    stamp: str,
    prepared_run: GeometryFitPreparedRun,
    cmd_line: Callable[[str], None],
    log_line: Callable[[str], None],
    log_section: Callable[[str, Sequence[str]], None],
) -> None:
    """Emit the runtime console/log prelude for one prepared geometry-fit run."""

    cmd_line(str(prepared_run.start_cmd_line))
    log_line(f"Geometry fit started: {stamp}")
    log_line("")
    for title, lines in prepared_run.start_log_sections:
        log_section(title, list(lines))


def build_geometry_fit_solver_request(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> GeometryFitSolverRequest:
    """Build one concrete solver request from a prepared geometry-fit run."""

    return GeometryFitSolverRequest(
        miller=solver_inputs.miller,
        intensities=solver_inputs.intensities,
        image_size=int(solver_inputs.image_size),
        params=dict(prepared_run.fit_params),
        measured_peaks=prepared_run.current_dataset["measured_for_fit"],
        var_names=[str(name) for name in var_names],
        experimental_image=prepared_run.current_dataset["experimental_image_for_fit"],
        dataset_specs=(
            list(prepared_run.dataset_specs)
            if prepared_run.joint_background_mode
            else None
        ),
        refinement_config=dict(prepared_run.geometry_runtime_cfg),
    )


def solve_geometry_fit_request(
    request: GeometryFitSolverRequest,
    *,
    solve_fit: Callable[..., object],
) -> object:
    """Invoke the live geometry-fit solver for one prepared request."""

    return solve_fit(
        request.miller,
        request.intensities,
        request.image_size,
        request.params,
        request.measured_peaks,
        request.var_names,
        pixel_tol=float("inf"),
        experimental_image=request.experimental_image,
        dataset_specs=request.dataset_specs,
        refinement_config=request.refinement_config,
    )


def apply_geometry_fit_result_values(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var: object | None = None,
) -> None:
    """Apply fitted geometry values back to the live UI variables."""

    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(value):
            continue

        param_name = str(name)
        if param_name == "theta_offset":
            if geometry_theta_offset_var is not None:
                geometry_theta_offset_var.set(f"{value:.6g}")
            continue

        var = var_map.get(param_name)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            continue


def build_geometry_fit_export_records(
    agg_millers: Sequence[Sequence[object]],
    agg_sim_coords: Sequence[Sequence[object]],
    agg_meas_coords: Sequence[Sequence[object]],
    pixel_offsets: Sequence[Sequence[object]],
) -> list[dict[str, object]]:
    """Build the matched-peak export rows saved after one geometry fit."""

    export_recs: list[dict[str, object]] = []
    for source_label, coords in (
        ("sim", agg_sim_coords),
        ("meas", agg_meas_coords),
    ):
        for hkl, (x, y), offset in zip(agg_millers, coords, pixel_offsets):
            try:
                hkl_triplet = tuple(int(v) for v in hkl[:3])
            except Exception:
                continue
            try:
                dist = float(offset[3])
            except Exception:
                dist = float("nan")
            export_recs.append(
                {
                    "source": str(source_label),
                    "hkl": hkl_triplet,
                    "x": int(x),
                    "y": int(y),
                    "dist_px": dist,
                }
            )
    return export_recs


def build_geometry_fit_optimizer_diagnostics_lines(result: object) -> list[str]:
    """Format optimizer diagnostics for one geometry-fit result."""

    lines = [
        f"success={getattr(result, 'success', False)}",
        f"status={getattr(result, 'status', '')}",
        f"message={(getattr(result, 'message', '') or '').strip()}",
        f"nfev={getattr(result, 'nfev', '<unknown>')}",
        f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
        f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
        f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
        f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
        f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
        f"active_mask={list(getattr(result, 'active_mask', []))}",
    ]
    for entry in getattr(result, "restart_history", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                idx=int(entry.get("restart", -1)),
                cost=float(entry.get("cost", np.nan)),
                success=bool(entry.get("success", False)),
                msg=str(entry.get("message", "")).strip(),
            )
        )
    return lines


def geometry_fit_result_rms(result: object) -> float:
    """Resolve the displayed RMS residual from one geometry-fit result."""

    try:
        direct_rms = float(getattr(result, "rms_px", np.nan))
    except Exception:
        direct_rms = float("nan")
    if np.isfinite(direct_rms):
        return float(direct_rms)

    fun = getattr(result, "fun", None)
    if fun is None:
        return 0.0
    try:
        residuals = np.asarray(fun, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    finite_residuals = residuals[np.isfinite(residuals)]
    if finite_residuals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite_residuals**2)))


def build_geometry_fit_result_lines(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    rms: float,
) -> list[str]:
    """Format fitted parameter values plus the RMS residual line."""

    lines = []
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        lines.append(f"{name} = {value:.6f}")
    lines.append(f"RMS residual = {float(rms):.6f} px")
    return lines


def build_geometry_fit_fitted_params(
    base_params: Mapping[str, object] | None,
    *,
    zb: object,
    zs: object,
    theta_initial: object,
    theta_offset: object,
    chi: object,
    cor_angle: object,
    psi_z: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Merge the current fitted UI values into one simulation parameter dict."""

    fitted = dict(base_params or {})
    fitted.update(
        current_geometry_fit_ui_params(
            zb=float(zb),
            zs=float(zs),
            theta_initial=float(theta_initial),
            theta_offset=float(theta_offset),
            psi_z=float(psi_z),
            chi=float(chi),
            cor_angle=float(cor_angle),
            gamma=float(gamma),
            Gamma=float(Gamma),
            corto_detector=float(corto_detector),
            a=float(a),
            c=float(c),
            center_x=float(center_x),
            center_y=float(center_y),
        )
    )
    return fitted


def build_geometry_fit_fitted_params_from_ui(
    base_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build fitted simulation params from one current UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_fitted_params(
        base_params,
        zb=params.get("zb", np.nan),
        zs=params.get("zs", np.nan),
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        chi=params.get("chi", np.nan),
        cor_angle=params.get("cor_angle", np.nan),
        psi_z=params.get("psi_z", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_profile_cache(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    *,
    theta_initial: object,
    theta_offset: object,
    cor_angle: object,
    chi: object,
    zs: object,
    zb: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Build the geometry-fit profile-cache payload after one successful fit."""

    profile_cache = dict(base_cache or {})
    profile_cache.update(dict(mosaic_params or {}))
    profile_cache.update(
        {
            "theta_initial": theta_initial,
            "theta_offset": theta_offset,
            "cor_angle": cor_angle,
            "chi": chi,
            "zs": zs,
            "zb": zb,
            "gamma": gamma,
            "Gamma": Gamma,
            "corto_detector": corto_detector,
            "a": a,
            "c": c,
            "center_x": center_x,
            "center_y": center_y,
        }
    )
    return profile_cache


def build_geometry_fit_profile_cache_from_ui(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the post-fit profile-cache payload from one UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_profile_cache(
        base_cache,
        mosaic_params,
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        cor_angle=params.get("cor_angle", np.nan),
        chi=params.get("chi", np.nan),
        zs=params.get("zs", np.nan),
        zb=params.get("zb", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_pixel_offsets(
    agg_millers: Sequence[Sequence[object]],
    agg_sim_coords: Sequence[Sequence[object]],
    agg_meas_coords: Sequence[Sequence[object]],
) -> list[tuple[tuple[int, int, int], float, float, float]]:
    """Build per-HKL pixel offset tuples from aggregated match centers."""

    pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
    for hkl_key, sim_center, meas_center in zip(
        agg_millers,
        agg_sim_coords,
        agg_meas_coords,
    ):
        try:
            hkl = tuple(int(v) for v in hkl_key[:3])
            dx = float(sim_center[0]) - float(meas_center[0])
            dy = float(sim_center[1]) - float(meas_center[1])
        except Exception:
            continue
        pixel_offsets.append((hkl, dx, dy, float(np.hypot(dx, dy))))
    return pixel_offsets


def filter_geometry_fit_overlay_point_match_diagnostics(
    point_match_diagnostics: object,
    *,
    joint_background_mode: bool,
    current_background_index: int,
) -> list[dict[str, object]] | object:
    """Keep only the current-background overlay diagnostics in joint mode."""

    if not (joint_background_mode and isinstance(point_match_diagnostics, list)):
        return point_match_diagnostics
    filtered: list[dict[str, object]] = []
    for entry in point_match_diagnostics:
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("dataset_index", -1)) != int(current_background_index):
            continue
        filtered.append(dict(entry))
    return filtered


def build_geometry_fit_point_match_summary_lines(
    point_match_summary: Mapping[str, object] | None,
) -> list[str]:
    """Format the optional point-match summary section."""

    if not isinstance(point_match_summary, Mapping):
        return []
    return [f"{key}={value}" for key, value in sorted(point_match_summary.items())]


def build_geometry_fit_overlay_diagnostic_lines(
    frame_diag: Mapping[str, object] | None,
    *,
    overlay_record_count: int,
) -> list[str]:
    """Format overlay frame diagnostics for the geometry-fit log."""

    diag = frame_diag if isinstance(frame_diag, Mapping) else {}
    return [
        "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
        f"overlay_records={int(overlay_record_count)}",
        f"paired_records={int(diag.get('paired_records', 0))}",
        f"sim_display_med_px={float(diag.get('sim_display_med_px', np.nan)):.3f}",
        f"bg_display_med_px={float(diag.get('bg_display_med_px', np.nan)):.3f}",
        f"sim_display_p90_px={float(diag.get('sim_display_p90_px', np.nan)):.3f}",
        f"bg_display_p90_px={float(diag.get('bg_display_p90_px', np.nan)):.3f}",
    ]


def build_geometry_fit_pixel_offset_lines(
    pixel_offsets: Sequence[Sequence[object]],
) -> list[str]:
    """Format one pixel-offset section for the geometry-fit log/status."""

    lines = []
    for entry in pixel_offsets:
        try:
            hkl = tuple(int(v) for v in entry[0][:3])
            dx = float(entry[1])
            dy = float(entry[2])
            dist = float(entry[3])
        except Exception:
            continue
        lines.append(f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px")
    return lines or ["No matched peaks"]


def build_geometry_fit_summary_lines(
    *,
    current_dataset: Mapping[str, object],
    overlay_record_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    save_path: object,
) -> list[str]:
    """Format the fit-summary section written to the geometry-fit log."""

    lines = [
        f"manual_groups={int(current_dataset.get('group_count', 0) or 0)}",
        f"manual_points={int(current_dataset.get('pair_count', 0) or 0)}",
        f"overlay_records={int(overlay_record_count)}",
        "orientation={orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        ),
    ]
    lines.extend(
        build_geometry_fit_result_lines(
            var_names,
            values,
            rms=rms,
        )[:-1]
    )
    lines.append(f"RMS residual = {float(rms):.6f} px")
    lines.append(f"Matched peaks saved to: {save_path}")
    return lines


def build_geometry_fit_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    pixel_offsets: Sequence[Sequence[object]],
    export_record_count: int,
    save_path: object,
    log_path: object,
    frame_warning: str | None,
) -> str:
    """Build the final geometry-fit status text shown in the GUI."""

    base_summary_lines = ["Manual geometry fit complete:"]
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        base_summary_lines.append(f"{name} = {value:.4f}")
    base_summary_lines.append(f"RMS residual = {float(rms):.2f} px")
    base_summary_lines.append(
        "Orientation = {orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        )
    )
    base_summary = "\n".join(base_summary_lines)
    overlay_hint = (
        "Overlay: blue squares=selected simulated points, amber triangles=saved "
        "background points, green circles=fitted simulated peaks, dashed "
        "arrows=initial->fitted sim shifts."
    )
    dist_report = (
        "\n".join(
            f"HKL={tuple(int(v) for v in entry[0][:3])}: |Δ|={float(entry[3]):.2f}px "
            f"(dx={float(entry[1]):.2f}, dy={float(entry[2]):.2f})"
            for entry in pixel_offsets
        )
        if pixel_offsets
        else "No matched peaks to report distances."
    )
    return (
        f"{base_summary}\n"
        "Manual pairs: {points} points across {groups} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
        + "\n"
        + overlay_hint
        + (f"\n{frame_warning}" if frame_warning else "")
        + f"\nSaved {int(export_record_count)} peak records → {save_path}"
        + f"\nPixel offsets:\n{dist_report}"
        + f"\nFit log → {log_path}"
    )


def postprocess_geometry_fit_result(
    *,
    fitted_params: Mapping[str, object],
    result: object,
    current_dataset: Mapping[str, object],
    joint_background_mode: bool,
    current_background_index: int,
    dataset_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    miller: object,
    intensities: object,
    image_size: int,
    max_display_markers: int,
    downloads_dir: Path | str,
    stamp: str,
    log_path: Path | str,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
) -> GeometryFitPostprocessResult:
    """Build the post-fit analysis artifacts used by the live GUI."""

    (
        _,
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    ) = simulate_and_compare_hkl(
        miller,
        intensities,
        int(image_size),
        dict(fitted_params),
        current_dataset["measured_for_fit"],
        pixel_tol=float("inf"),
    )
    agg_sim_coords, agg_meas_coords, agg_millers = aggregate_match_centers(
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    )
    pixel_offsets = build_geometry_fit_pixel_offsets(
        agg_millers,
        agg_sim_coords,
        agg_meas_coords,
    )

    overlay_point_match_diagnostics = filter_geometry_fit_overlay_point_match_diagnostics(
        getattr(result, "point_match_diagnostics", None),
        joint_background_mode=joint_background_mode,
        current_background_index=int(current_background_index),
    )
    overlay_records = build_overlay_records(
        current_dataset["initial_pairs_display"],
        overlay_point_match_diagnostics,
        native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
        orientation_choice=current_dataset["orientation_choice"],
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
    )
    frame_diag, frame_warning = compute_frame_diagnostics(overlay_records)
    overlay_diagnostic_lines = build_geometry_fit_overlay_diagnostic_lines(
        frame_diag,
        overlay_record_count=len(overlay_records),
    )

    export_records = build_geometry_fit_export_records(
        agg_millers,
        agg_sim_coords,
        agg_meas_coords,
        pixel_offsets,
    )
    save_path = Path(downloads_dir) / f"matched_peaks_{stamp}.npy"
    fit_summary_lines = build_geometry_fit_summary_lines(
        current_dataset=current_dataset,
        overlay_record_count=len(overlay_records),
        var_names=var_names,
        values=values,
        rms=rms,
        save_path=save_path,
    )
    progress_text = build_geometry_fit_progress_text(
        current_dataset=current_dataset,
        dataset_count=int(dataset_count),
        joint_background_mode=joint_background_mode,
        var_names=var_names,
        values=values,
        rms=rms,
        pixel_offsets=pixel_offsets,
        export_record_count=len(export_records),
        save_path=save_path,
        log_path=log_path,
        frame_warning=frame_warning,
    )
    overlay_state = {
        "overlay_records": copy_geometry_fit_state_value(overlay_records),
        "initial_pairs_display": copy_geometry_fit_state_value(
            current_dataset["initial_pairs_display"]
        ),
        "max_display_markers": int(max_display_markers),
    }

    return GeometryFitPostprocessResult(
        fitted_params=fitted_params,
        point_match_summary_lines=build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        ),
        pixel_offsets=pixel_offsets,
        overlay_records=overlay_records,
        overlay_state=overlay_state,
        overlay_diagnostic_lines=overlay_diagnostic_lines,
        frame_warning=frame_warning,
        export_records=export_records,
        save_path=save_path,
        fit_summary_lines=fit_summary_lines,
        progress_text=progress_text,
    )


def apply_runtime_geometry_fit_result(
    *,
    result: object,
    var_names: Sequence[object],
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    preserve_live_theta: bool,
    max_display_markers: int,
    bindings: GeometryFitRuntimeResultBindings,
) -> GeometryFitRuntimeApplyResult:
    """Apply one successful geometry-fit result through runtime callbacks."""

    bindings.log_section(
        "Optimizer diagnostics:",
        build_geometry_fit_optimizer_diagnostics_lines(result),
    )

    undo_state = bindings.capture_undo_state()
    result_values = list(getattr(result, "x", []) or [])
    bindings.apply_result_values(var_names, result_values)

    if joint_background_mode and not preserve_live_theta:
        sync_theta = bindings.sync_joint_background_theta
        if sync_theta is not None:
            sync_theta()

    bindings.refresh_status()
    bindings.update_manual_pick_button_label()
    bindings.replace_profile_cache(bindings.build_profile_cache())
    bindings.push_undo_state(undo_state)
    bindings.request_preview_skip_once()
    bindings.mark_last_simulation_dirty()
    bindings.schedule_update()

    rms = geometry_fit_result_rms(result)
    bindings.log_section(
        "Optimization result:",
        build_geometry_fit_result_lines(
            var_names,
            result_values,
            rms=rms,
        ),
    )

    fitted_params = bindings.build_fitted_params()
    postprocess = bindings.postprocess_result(fitted_params, rms)

    if postprocess.point_match_summary_lines:
        bindings.log_section(
            "Point-match summary:",
            postprocess.point_match_summary_lines,
        )

    bindings.log_section(
        "Overlay frame diagnostics:",
        postprocess.overlay_diagnostic_lines,
    )

    if postprocess.overlay_records:
        bindings.draw_overlay_records(
            postprocess.overlay_records,
            int(max_display_markers),
        )
    else:
        bindings.draw_initial_pairs_overlay(
            current_dataset["initial_pairs_display"],
            int(max_display_markers),
        )

    bindings.set_last_overlay_state(postprocess.overlay_state)
    bindings.save_export_records(postprocess.save_path, postprocess.export_records)
    bindings.log_section(
        "Pixel offsets (native frame):",
        build_geometry_fit_pixel_offset_lines(postprocess.pixel_offsets),
    )
    bindings.log_section(
        "Fit summary:",
        postprocess.fit_summary_lines,
    )
    bindings.set_progress_text(postprocess.progress_text)
    bindings.cmd_line(
        "done: "
        f"datasets={int(dataset_count)} "
        f"groups={int(current_dataset.get('group_count', 0) or 0)} "
        f"points={int(current_dataset.get('pair_count', 0) or 0)} "
        f"rms={float(rms):.4f}px"
    )

    return GeometryFitRuntimeApplyResult(
        rms=float(rms),
        fitted_params=fitted_params,
        postprocess=postprocess,
    )


def build_runtime_geometry_fit_result_bindings(
    *,
    fit_params: Mapping[str, object] | None,
    base_profile_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
    log_section: Callable[[str, Sequence[str]], None],
    capture_undo_state: Callable[[], dict[str, object]],
    sync_joint_background_theta: Callable[[], None] | None,
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    replace_profile_cache: Callable[[dict[str, object]], None],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    mark_last_simulation_dirty: Callable[[], None],
    schedule_update: Callable[[], None],
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime success-path callback bundle for one geometry fit."""

    def _snapshot_ui_params() -> dict[str, object]:
        try:
            params = current_ui_params()
        except Exception:
            params = {}
        return dict(params) if isinstance(params, Mapping) else {}

    return GeometryFitRuntimeResultBindings(
        log_section=log_section,
        capture_undo_state=capture_undo_state,
        apply_result_values=(
            lambda names, values: apply_geometry_fit_result_values(
                names,
                values,
                var_map=var_map,
                geometry_theta_offset_var=geometry_theta_offset_var,
            )
        ),
        sync_joint_background_theta=sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        build_profile_cache=(
            lambda: build_geometry_fit_profile_cache_from_ui(
                base_profile_cache,
                mosaic_params,
                _snapshot_ui_params(),
            )
        ),
        replace_profile_cache=replace_profile_cache,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=mark_last_simulation_dirty,
        schedule_update=schedule_update,
        build_fitted_params=(
            lambda: build_geometry_fit_fitted_params_from_ui(
                fit_params,
                _snapshot_ui_params(),
            )
        ),
        postprocess_result=postprocess_result,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=save_export_records,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
    )


def build_runtime_geometry_fit_execution_result_bindings(
    *,
    result: object,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    ui_bindings: GeometryFitRuntimeUiBindings,
    postprocess_config: GeometryFitRuntimePostprocessConfig,
    log_section: Callable[[str, Sequence[str]], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime apply-bindings for one executed geometry-fit result."""

    solver_inputs = postprocess_config.solver_inputs
    return build_runtime_geometry_fit_result_bindings(
        fit_params=ui_bindings.fit_params,
        base_profile_cache=ui_bindings.base_profile_cache,
        mosaic_params=ui_bindings.mosaic_params,
        current_ui_params=ui_bindings.current_ui_params,
        var_map=ui_bindings.var_map,
        geometry_theta_offset_var=ui_bindings.geometry_theta_offset_var,
        log_section=log_section,
        capture_undo_state=ui_bindings.capture_undo_state,
        sync_joint_background_theta=ui_bindings.sync_joint_background_theta,
        refresh_status=ui_bindings.refresh_status,
        update_manual_pick_button_label=ui_bindings.update_manual_pick_button_label,
        replace_profile_cache=ui_bindings.replace_profile_cache,
        push_undo_state=ui_bindings.push_undo_state,
        request_preview_skip_once=ui_bindings.request_preview_skip_once,
        mark_last_simulation_dirty=ui_bindings.mark_last_simulation_dirty,
        schedule_update=ui_bindings.schedule_update,
        postprocess_result=(
            lambda fitted_params, rms: postprocess_geometry_fit_result(
                fitted_params=fitted_params,
                result=result,
                current_dataset=prepared_run.current_dataset,
                joint_background_mode=prepared_run.joint_background_mode,
                current_background_index=int(postprocess_config.current_background_index),
                dataset_count=len(prepared_run.dataset_infos),
                var_names=var_names,
                values=getattr(result, "x", []),
                rms=rms,
                miller=solver_inputs.miller,
                intensities=solver_inputs.intensities,
                image_size=int(solver_inputs.image_size),
                max_display_markers=int(prepared_run.max_display_markers),
                downloads_dir=postprocess_config.downloads_dir,
                stamp=postprocess_config.stamp,
                log_path=postprocess_config.log_path,
                sim_display_rotate_k=int(postprocess_config.sim_display_rotate_k),
                background_display_rotate_k=int(
                    postprocess_config.background_display_rotate_k
                ),
                simulate_and_compare_hkl=postprocess_config.simulate_and_compare_hkl,
                aggregate_match_centers=postprocess_config.aggregate_match_centers,
                build_overlay_records=postprocess_config.build_overlay_records,
                compute_frame_diagnostics=postprocess_config.compute_frame_diagnostics,
            )
        ),
        draw_overlay_records=ui_bindings.draw_overlay_records,
        draw_initial_pairs_overlay=ui_bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=ui_bindings.set_last_overlay_state,
        save_export_records=ui_bindings.save_export_records,
        set_progress_text=ui_bindings.set_progress_text,
        cmd_line=ui_bindings.cmd_line,
    )


def build_runtime_geometry_fit_execution_setup(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    downloads_dir: Path | str,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
) -> GeometryFitRuntimeExecutionSetup:
    """Build the runtime execution setup for one prepared geometry-fit run."""

    ui_bindings = GeometryFitRuntimeUiBindings(
        fit_params=prepared_run.fit_params,
        base_profile_cache=getattr(simulation_runtime_state, "profile_cache", {}),
        mosaic_params=mosaic_params,
        current_ui_params=current_ui_params,
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
        capture_undo_state=capture_undo_state,
        sync_joint_background_theta=(
            lambda: theta_initial_var.set(
                background_theta_for_index(
                    getattr(background_runtime_state, "current_background_index", 0),
                    strict_count=False,
                )
            )
        ),
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        replace_profile_cache=(
            lambda profile_cache: setattr(
                simulation_runtime_state,
                "profile_cache",
                dict(profile_cache),
            )
        ),
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=(
            lambda: setattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        ),
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=(
            lambda save_path, export_records: np.save(
                save_path,
                np.array(export_records, dtype=object),
                allow_pickle=True,
            )
        ),
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
    )

    postprocess_config = GeometryFitRuntimePostprocessConfig(
        current_background_index=int(
            getattr(background_runtime_state, "current_background_index", 0)
        ),
        downloads_dir=downloads_dir,
        stamp=stamp,
        log_path=Path(downloads_dir) / f"geometry_fit_log_{stamp}.txt",
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
    )

    return GeometryFitRuntimeExecutionSetup(
        ui_bindings=ui_bindings,
        postprocess_config=postprocess_config,
    )


def build_runtime_geometry_fit_execution_setup_from_bindings(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    bindings: GeometryFitRuntimeActionExecutionBindings,
) -> GeometryFitRuntimeExecutionSetup:
    """Build one runtime execution setup from a bound action bundle."""

    return build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params=mosaic_params,
        stamp=stamp,
        downloads_dir=bindings.downloads_dir,
        simulation_runtime_state=bindings.simulation_runtime_state,
        background_runtime_state=bindings.background_runtime_state,
        theta_initial_var=bindings.theta_initial_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        current_ui_params=bindings.current_ui_params,
        var_map=bindings.var_map,
        background_theta_for_index=bindings.background_theta_for_index,
        refresh_status=bindings.refresh_status,
        update_manual_pick_button_label=bindings.update_manual_pick_button_label,
        capture_undo_state=bindings.capture_undo_state,
        push_undo_state=bindings.push_undo_state,
        request_preview_skip_once=bindings.request_preview_skip_once,
        schedule_update=bindings.schedule_update,
        draw_overlay_records=bindings.draw_overlay_records,
        draw_initial_pairs_overlay=bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=bindings.set_last_overlay_state,
        set_progress_text=bindings.set_progress_text,
        cmd_line=bindings.cmd_line,
        solver_inputs=bindings.solver_inputs,
        sim_display_rotate_k=bindings.sim_display_rotate_k,
        background_display_rotate_k=bindings.background_display_rotate_k,
        simulate_and_compare_hkl=bindings.simulate_and_compare_hkl,
        aggregate_match_centers=bindings.aggregate_match_centers,
        build_overlay_records=bindings.build_overlay_records,
        compute_frame_diagnostics=bindings.compute_frame_diagnostics,
    )


def run_runtime_geometry_fit_action(
    *,
    bindings: GeometryFitRuntimeActionBindings,
    prepare_run: Callable[..., GeometryFitPreparationResult] | None = None,
    build_execution_setup: Callable[..., GeometryFitRuntimeExecutionSetup] | None = None,
    execute_run: Callable[..., GeometryFitRuntimeExecutionResult] | None = None,
) -> GeometryFitRuntimeActionResult:
    """Run the full top-level geometry-fit action from live runtime bindings."""

    if prepare_run is None:
        prepare_run = prepare_runtime_geometry_fit_run
    if build_execution_setup is None:
        build_execution_setup = build_runtime_geometry_fit_execution_setup_from_bindings
    if execute_run is None:
        execute_run = execute_runtime_geometry_fit

    params = bindings.value_callbacks.current_params()
    mosaic_params = dict(params.get("mosaic_params", {}))
    var_names = list(bindings.value_callbacks.current_var_names())
    preserve_live_theta = (
        "theta_initial" not in var_names and "theta_offset" not in var_names
    )
    prepare_bindings = bindings.prepare_bindings_factory(var_names)

    try:
        prepare_result = prepare_run(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            bindings=prepare_bindings,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        bindings.execution_bindings.cmd_line(f"failed: {exc}")
        bindings.execution_bindings.set_progress_text(error_text)
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            error_text=error_text,
        )

    if prepare_result.prepared_run is None:
        if prepare_result.error_text:
            bindings.execution_bindings.set_progress_text(str(prepare_result.error_text))
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            prepare_result=prepare_result,
            error_text=prepare_result.error_text,
        )

    execution_setup = build_execution_setup(
        prepared_run=prepare_result.prepared_run,
        mosaic_params=mosaic_params,
        stamp=str(bindings.stamp_factory()),
        bindings=bindings.execution_bindings,
    )
    execution_result = execute_run(
        prepared_run=prepare_result.prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        solve_fit=bindings.solve_fit,
        setup=execution_setup,
        flush_ui=bindings.flush_ui,
    )
    return GeometryFitRuntimeActionResult(
        params=params,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=execution_result.error_text,
    )


def execute_runtime_geometry_fit(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    preserve_live_theta: bool,
    solve_fit: Callable[..., object],
    setup: GeometryFitRuntimeExecutionSetup,
    start_progress_text: str = "Running geometry fit from saved manual Qr/Qz pairs…",
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeExecutionResult:
    """Run one prepared geometry fit through the live solver and callbacks."""

    ui_bindings = setup.ui_bindings
    postprocess_config = setup.postprocess_config
    log_path = Path(postprocess_config.log_path)
    log_file = None

    def _log_line(text: str = "") -> None:
        if log_file is None:
            return
        try:
            log_file.write(text + "\n")
            log_file.flush()
        except Exception:
            pass

    def _log_section(title: str, lines: Sequence[str]) -> None:
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    try:
        log_file = log_path.open("w", encoding="utf-8")
        write_geometry_fit_run_start_log(
            stamp=str(postprocess_config.stamp),
            prepared_run=prepared_run,
            cmd_line=ui_bindings.cmd_line,
            log_line=_log_line,
            log_section=_log_section,
        )

        ui_bindings.set_progress_text(str(start_progress_text))
        if callable(flush_ui):
            flush_ui()

        solver_request = build_geometry_fit_solver_request(
            prepared_run=prepared_run,
            var_names=var_names,
            solver_inputs=postprocess_config.solver_inputs,
        )
        result = solve_geometry_fit_request(
            solver_request,
            solve_fit=solve_fit,
        )
        apply_result = apply_runtime_geometry_fit_result(
            result=result,
            var_names=var_names,
            current_dataset=prepared_run.current_dataset,
            dataset_count=len(prepared_run.dataset_infos),
            joint_background_mode=prepared_run.joint_background_mode,
            preserve_live_theta=preserve_live_theta,
            max_display_markers=prepared_run.max_display_markers,
            bindings=build_runtime_geometry_fit_execution_result_bindings(
                result=result,
                prepared_run=prepared_run,
                var_names=var_names,
                ui_bindings=ui_bindings,
                postprocess_config=postprocess_config,
                log_section=_log_section,
            ),
        )
        return GeometryFitRuntimeExecutionResult(
            log_path=log_path,
            solver_request=solver_request,
            solver_result=result,
            apply_result=apply_result,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        ui_bindings.cmd_line(f"failed: {exc}")
        _log_line(error_text)
        ui_bindings.set_progress_text(error_text)
        return GeometryFitRuntimeExecutionResult(
            log_path=log_path,
            error_text=error_text,
        )
    finally:
        try:
            if log_file is not None:
                log_file.close()
        except Exception:
            pass
