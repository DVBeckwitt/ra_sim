"""Pure helpers for GUI geometry-fit state and configuration."""

from __future__ import annotations

import copy
import os
import sys
import inspect
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

GEOMETRY_FIT_ACCEPT_MAX_RMS_PX = 100.0
GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX = 150.0


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
class GeometryFitRuntimeManualDatasetBindings:
    """Live manual-pair dataset callbacks and values reused by geometry-fit prep."""

    osc_files: Sequence[object]
    current_background_index: int
    image_size: int
    display_rotate_k: int
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]]
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


@dataclass(frozen=True)
class GeometryFitRuntimePreparationBindings:
    """Runtime values and callbacks used to prepare one geometry-fit run."""

    fit_config: Mapping[str, object] | None
    theta_initial: object
    apply_geometry_fit_background_selection: Callable[..., bool]
    current_geometry_fit_background_indices: Callable[..., list[int]]
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    apply_background_theta_metadata: Callable[..., bool]
    current_background_theta_values: Callable[..., list[float]]
    current_geometry_theta_offset: Callable[..., float]
    ensure_geometry_fit_caked_view: Callable[[], None]
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings
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
    sample_width_var: object
    sample_length_var: object
    sample_depth_var: object
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
    pixel_size_value: object


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
    runtime_safety_note: str | None = None


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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
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
class GeometryFitActionNotice:
    """User-facing notice derived from one top-level geometry-fit action."""

    level: str
    title: str
    message: str


@dataclass(frozen=True)
class GeometryFitRuntimeApplyResult:
    """Result metadata returned after applying one successful geometry fit."""

    accepted: bool
    rejection_reason: str | None
    rms: float
    fitted_params: dict[str, object] | None
    postprocess: GeometryFitPostprocessResult | None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionResult:
    """Result metadata for one full runtime geometry-fit execution."""

    log_path: Path
    solver_request: GeometryFitSolverRequest | None = None
    solver_result: object | None = None
    apply_result: GeometryFitRuntimeApplyResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryToolActionRuntimeCallbacks:
    """Bound runtime callbacks for the geometry tool action control cluster."""

    update_fit_history_button_state: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    set_manual_pick_mode: Callable[[bool, str | None], None]
    toggle_manual_pick_mode: Callable[[], None]
    clear_current_manual_pairs: Callable[[], None]


@dataclass(frozen=True)
class GeometryFitRuntimeHistoryCallbacks:
    """Bound runtime callbacks for geometry-fit undo/redo history transitions."""

    undo: Callable[[], bool]
    redo: Callable[[], bool]


def build_runtime_geometry_fit_manual_dataset_bindings(
    *,
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
) -> GeometryFitRuntimeManualDatasetBindings:
    """Build the live manual-pair dataset bundle used during geometry-fit prep."""

    return GeometryFitRuntimeManualDatasetBindings(
        osc_files=osc_files,
        current_background_index=int(current_background_index),
        image_size=int(image_size),
        display_rotate_k=int(display_rotate_k),
        geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
        load_background_by_index=load_background_by_index,
        apply_background_backend_orientation=apply_background_backend_orientation,
        geometry_manual_simulated_peaks_for_params=(
            geometry_manual_simulated_peaks_for_params
        ),
        geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
        geometry_manual_entry_display_coords=geometry_manual_entry_display_coords,
        unrotate_display_peaks=unrotate_display_peaks,
        display_to_native_sim_coords=display_to_native_sim_coords,
        select_fit_orientation=select_fit_orientation,
        apply_orientation_to_entries=apply_orientation_to_entries,
        orient_image_for_fit=orient_image_for_fit,
    )


def make_runtime_geometry_fit_manual_dataset_bindings_factory(
    *,
    osc_files_factory: Callable[[], Sequence[object]],
    current_background_index_factory: Callable[[], object],
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
) -> Callable[[], GeometryFitRuntimeManualDatasetBindings]:
    """Build a factory that resolves the live manual-pair dataset bundle on demand."""

    def _build() -> GeometryFitRuntimeManualDatasetBindings:
        return build_runtime_geometry_fit_manual_dataset_bindings(
            osc_files=osc_files_factory(),
            current_background_index=int(current_background_index_factory()),
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            load_background_by_index=load_background_by_index,
            apply_background_backend_orientation=(
                apply_background_backend_orientation
            ),
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
        )

    return _build


def make_runtime_geometry_fit_action_prepare_bindings_factory(
    *,
    fit_config: Mapping[str, object] | None,
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
) -> Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]:
    """Build the live prepare-bundle factory for one geometry-fit action."""

    def _build(var_names: Sequence[str]) -> GeometryFitRuntimePreparationBindings:
        return GeometryFitRuntimePreparationBindings(
            fit_config=fit_config,
            theta_initial=theta_initial,
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
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
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
        replace_dataset_cache=replace_dataset_cache,
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
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
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
            theta_initial=theta_initial,
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
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
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
            replace_dataset_cache=replace_dataset_cache,
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


def make_runtime_geometry_fit_action_bindings_factory(
    *,
    value_callbacks_factory: Callable[[], GeometryFitRuntimeValueCallbacks],
    fit_config: Mapping[str, object] | None,
    theta_initial_factory: Callable[[], object],
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings_factory: Callable[
        [],
        GeometryFitRuntimeManualDatasetBindings,
    ],
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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs_factory: Callable[[], GeometryFitRuntimeSolverInputs],
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> Callable[[], GeometryFitRuntimeActionBindings]:
    """Build a factory that resolves live geometry-fit action bindings on demand."""

    def _build() -> GeometryFitRuntimeActionBindings:
        return build_runtime_geometry_fit_action_bindings(
            value_callbacks=value_callbacks_factory(),
            fit_config=fit_config,
            theta_initial=theta_initial_factory(),
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
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings_factory(),
            build_runtime_config_factory=build_runtime_config_factory,
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
            replace_dataset_cache=replace_dataset_cache,
            request_preview_skip_once=request_preview_skip_once,
            schedule_update=schedule_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            set_last_overlay_state=set_last_overlay_state,
            set_progress_text=set_progress_text,
            cmd_line=cmd_line,
            solver_inputs=solver_inputs_factory(),
            sim_display_rotate_k=int(sim_display_rotate_k),
            background_display_rotate_k=int(background_display_rotate_k),
            simulate_and_compare_hkl=simulate_and_compare_hkl,
            aggregate_match_centers=aggregate_match_centers,
            build_overlay_records=build_overlay_records,
            compute_frame_diagnostics=compute_frame_diagnostics,
            solve_fit=solve_fit,
            stamp_factory=stamp_factory,
            flush_ui=flush_ui,
        )

    return _build


def make_runtime_geometry_fit_action_callback(
    bindings_factory: Callable[[], GeometryFitRuntimeActionBindings],
    *,
    before_run: Callable[[], None] | None = None,
    run_action: Callable[..., GeometryFitRuntimeActionResult] | None = None,
    after_run: Callable[[GeometryFitRuntimeActionResult], None] | None = None,
) -> Callable[[], GeometryFitRuntimeActionResult]:
    """Build the zero-arg runtime callback for the top-level geometry-fit action."""

    def _run() -> GeometryFitRuntimeActionResult:
        if callable(before_run):
            before_run()
        action = run_action if callable(run_action) else run_runtime_geometry_fit_action
        result = action(bindings=bindings_factory())
        if callable(after_run):
            after_run(result)
        return result

    return _run


def build_geometry_fit_action_notice(
    action_result: GeometryFitRuntimeActionResult | None,
) -> GeometryFitActionNotice | None:
    """Return a user-facing notice for one failed or rejected geometry fit."""

    if action_result is None:
        return None

    error_text = str(action_result.error_text or "").strip()
    if error_text:
        return GeometryFitActionNotice(
            level="error",
            title="Geometry Fit Failed",
            message=error_text,
        )

    execution_result = action_result.execution_result
    if execution_result is None:
        return None

    apply_result = execution_result.apply_result
    if apply_result is None or bool(apply_result.accepted):
        return None

    lines = ["Geometry fit finished but the solution was rejected."]
    rejection_reason = str(apply_result.rejection_reason or "").strip()
    if rejection_reason:
        lines.append(rejection_reason)
    lines.append("The live geometry state was left unchanged.")
    log_path = getattr(execution_result, "log_path", None)
    if log_path is not None:
        lines.append(f"Fit log: {Path(log_path)}")
    return GeometryFitActionNotice(
        level="warning",
        title="Geometry Fit Rejected",
        message="\n".join(lines),
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


def geometry_fit_constraint_source_name(name: str) -> str:
    """Map fitted parameter names back to the UI constraint control names."""

    if str(name) == "theta_offset":
        return "theta_initial"
    return str(name)


def geometry_fit_constraint_parameter_name(
    name: str,
    *,
    use_shared_theta_offset: bool = False,
) -> str:
    """Map UI constraint row names to the active fitted parameter names."""

    if str(name) == "theta_initial" and bool(use_shared_theta_offset):
        return "theta_offset"
    return str(name)


def read_runtime_geometry_fit_constraint_state(
    *,
    controls: Mapping[str, object],
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, dict[str, float]]:
    """Read the live geometry-fit constraint controls into normalized settings."""

    selected_names = list(names) if names is not None else list(controls)
    state: dict[str, dict[str, float]] = {}
    for name in selected_names:
        control = controls.get(
            geometry_fit_constraint_source_name(
                geometry_fit_constraint_parameter_name(
                    str(name),
                    use_shared_theta_offset=use_shared_theta_offset,
                )
            )
        )
        if not isinstance(control, dict):
            continue
        try:
            window = float(control["window_var"].get())
        except Exception:
            window = float("nan")
        try:
            pull = float(control["pull_var"].get())
        except Exception:
            pull = 0.0
        if not np.isfinite(window):
            continue
        window = max(0.0, float(window))
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        state[str(name)] = {
            "window": float(window),
            "pull": float(pull),
        }
    return state


def _geometry_fit_config_section(
    fit_config: Mapping[str, object] | None,
    section: str,
) -> Mapping[str, object]:
    """Return one normalized geometry-fit config mapping section."""

    fit_geometry_cfg = (
        fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    )
    if not isinstance(fit_geometry_cfg, Mapping):
        return {}
    section_cfg = fit_geometry_cfg.get(section, {}) or {}
    if not isinstance(section_cfg, Mapping):
        return {}
    return section_cfg


def read_runtime_geometry_fit_parameter_domains(
    *,
    parameter_specs: Mapping[str, object],
    image_size: object,
    fit_config: Mapping[str, object] | None,
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, tuple[float, float]]:
    """Read live parameter domains from bounds config and slider ranges."""

    selected_names = list(names) if names is not None else list(parameter_specs)
    domains: dict[str, tuple[float, float]] = {}
    bounds_cfg = _geometry_fit_config_section(fit_config, "bounds")
    try:
        image_size_value = float(image_size)
    except Exception:
        image_size_value = 0.0

    for name in selected_names:
        parameter_name = geometry_fit_constraint_parameter_name(
            str(name),
            use_shared_theta_offset=use_shared_theta_offset,
        )
        control_name = geometry_fit_constraint_source_name(parameter_name)

        if parameter_name == "center_x" or parameter_name == "center_y":
            domains[str(name)] = (0.0, max(image_size_value - 1.0, 0.0))
            continue

        if parameter_name == "theta_offset":
            entry = bounds_cfg.get("theta_offset")
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    lo = float(entry[0])
                    hi = float(entry[1])
                except Exception:
                    lo = float("nan")
                    hi = float("nan")
                if np.isfinite(lo) and np.isfinite(hi):
                    if lo > hi:
                        lo, hi = hi, lo
                    domains[str(name)] = (float(lo), float(hi))
                    continue
            elif isinstance(entry, Mapping):
                try:
                    lo = float(entry.get("min"))
                    hi = float(entry.get("max"))
                except Exception:
                    lo = float("nan")
                    hi = float("nan")
                if np.isfinite(lo) and np.isfinite(hi):
                    if lo > hi:
                        lo, hi = hi, lo
                    domains[str(name)] = (float(lo), float(hi))
                    continue

        spec = parameter_specs.get(control_name)
        if not isinstance(spec, Mapping):
            continue
        slider_widget = spec.get("value_slider")
        if slider_widget is None:
            continue
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            continue
        if parameter_name == "theta_offset":
            span = max(abs(lo), abs(hi), 1.0)
            domains[str(name)] = (-float(span), float(span))
            continue
        if lo > hi:
            lo, hi = hi, lo
        domains[str(name)] = (float(lo), float(hi))
    return domains


def default_runtime_geometry_fit_constraint_window(
    *,
    name: str,
    parameter_specs: Mapping[str, object],
    fit_config: Mapping[str, object] | None,
    parameter_domains: Mapping[str, tuple[float, float]] | None = None,
    current_theta_offset: object = 0.0,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint window for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    control_name = geometry_fit_constraint_source_name(parameter_name)
    spec = parameter_specs.get(control_name, {})
    if parameter_name == "theta_offset":
        try:
            current_value = float(current_theta_offset)
        except Exception:
            current_value = 0.0
    else:
        try:
            current_value = float(spec["value_var"].get())
        except Exception:
            current_value = 0.0
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    resolved_domains = parameter_domains or {}
    domain = resolved_domains.get(parameter_name)
    if domain is None:
        domain = resolved_domains.get(str(name))
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))

    bounds_cfg = _geometry_fit_config_section(fit_config, "bounds")
    entry = bounds_cfg.get(parameter_name)

    default_window = float("nan")
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            lo = float(entry[0])
            hi = float(entry[1])
            default_window = max(abs(current_value - lo), abs(hi - current_value))
        except Exception:
            default_window = float("nan")
    elif isinstance(entry, Mapping):
        mode = str(entry.get("mode", "absolute")).strip().lower()
        try:
            min_raw = (
                float(entry.get("min"))
                if entry.get("min") is not None
                else float("nan")
            )
        except Exception:
            min_raw = float("nan")
        try:
            max_raw = (
                float(entry.get("max"))
                if entry.get("max") is not None
                else float("nan")
            )
        except Exception:
            max_raw = float("nan")
        if mode in {"relative", "rel", "relative_min0", "rel_min0"}:
            candidates = [abs(v) for v in (min_raw, max_raw) if np.isfinite(v)]
            if candidates:
                default_window = max(candidates)
        else:
            candidates = [
                abs(current_value - v)
                for v in (min_raw, max_raw)
                if np.isfinite(v)
            ]
            if candidates:
                default_window = max(candidates)

    if not np.isfinite(default_window) or default_window <= 0.0:
        default_window = max(
            step * 10.0,
            0.02 * domain_span,
            0.1 * max(abs(current_value), 1.0),
        )

    if domain_span > 0.0:
        default_window = min(default_window, domain_span)

    return max(float(default_window), step)


def default_runtime_geometry_fit_constraint_pull(
    *,
    name: str,
    fit_config: Mapping[str, object] | None,
    window: float,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint pull for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    priors_cfg = _geometry_fit_config_section(fit_config, "priors")
    entry = priors_cfg.get(parameter_name)
    if not isinstance(entry, Mapping):
        return 0.0
    try:
        sigma = float(entry.get("sigma"))
    except Exception:
        sigma = float("nan")
    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
        or not np.isfinite(window)
        or window <= 0.0
    ):
        return 0.0
    inferred = (1.0 - min(max(sigma / window, 0.05), 1.0)) / 0.95
    if not np.isfinite(inferred):
        return 0.0
    return min(max(float(inferred), 0.0), 1.0)


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
        n2_value = bindings.n2() if callable(bindings.n2) else bindings.n2
        pixel_size_value = (
            bindings.pixel_size_value()
            if callable(bindings.pixel_size_value)
            else bindings.pixel_size_value
        )
        return {
            "a": bindings.a_var.get(),
            "c": bindings.c_var.get(),
            "lambda": bindings.lambda_value,
            "psi": bindings.psi,
            "psi_z": bindings.psi_z_var.get(),
            "zs": bindings.zs_var.get(),
            "zb": bindings.zb_var.get(),
            "sample_width_m": bindings.sample_width_var.get(),
            "sample_length_m": bindings.sample_length_var.get(),
            "sample_depth_m": bindings.sample_depth_var.get(),
            "chi": bindings.chi_var.get(),
            "n2": n2_value,
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
            "pixel_size": pixel_size_value,
            "pixel_size_m": pixel_size_value,
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


def build_runtime_geometry_fit_config_factory(
    *,
    base_config: Mapping[str, object] | None,
    current_constraint_state: Callable[[Sequence[str] | None], Mapping[str, object]],
    current_parameter_domains: Callable[[Sequence[str] | None], Mapping[str, object]],
) -> Callable[[Sequence[str], Mapping[str, object]], dict[str, object]]:
    """Build the live geometry-fit refinement-config factory from runtime readers."""

    def _build(
        var_names: Sequence[str],
        fit_params: Mapping[str, object],
    ) -> dict[str, object]:
        selected_names = [str(name) for name in var_names]
        current_params = {
            name: fit_params.get(name)
            for name in selected_names
        }
        return build_geometry_fit_runtime_config(
            base_config,
            current_params,
            current_constraint_state(selected_names),
            current_parameter_domains(selected_names),
        )

    return _build


def build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    runtime_cfg = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    # The live GUI fit should favor stability over throughput. Native/parallel
    # fitting paths have triggered hard system failures on some machines, so the
    # runtime config clamps geometry fits to a conservative execution profile by
    # default. Advanced users can opt back in through gui_* overrides in config.
    solver_cfg = runtime_cfg.get("solver", {}) or {}
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}
    runtime_cfg["solver"] = solver_cfg

    gui_use_numba = runtime_cfg.pop("gui_use_numba", None)
    runtime_cfg["use_numba"] = (
        bool(gui_use_numba) if gui_use_numba is not None else False
    )

    gui_workers = solver_cfg.pop("gui_workers", None)
    solver_cfg["workers"] = gui_workers if gui_workers is not None else 1

    gui_parallel_mode = solver_cfg.pop("gui_parallel_mode", None)
    solver_cfg["parallel_mode"] = (
        str(gui_parallel_mode).strip()
        if gui_parallel_mode is not None
        else "off"
    )

    gui_worker_numba_threads = solver_cfg.pop("gui_worker_numba_threads", None)
    solver_cfg["worker_numba_threads"] = (
        gui_worker_numba_threads if gui_worker_numba_threads is not None else 1
    )

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


def set_runtime_geometry_fit_history_button_state(
    *,
    can_undo: bool,
    can_redo: bool,
    set_button_state: Callable[[bool, bool], None] | None = None,
) -> None:
    """Apply the current geometry-fit undo/redo availability to the UI."""

    if callable(set_button_state):
        set_button_state(bool(can_undo), bool(can_redo))


def _resolve_runtime_value(value_or_factory):
    """Return one runtime value, calling it first when it is a factory."""

    if callable(value_or_factory):
        return value_or_factory()
    return value_or_factory


def _geometry_fit_sequence_has_items(value: object) -> bool:
    """Return whether one saved sequence-like payload contains any entries."""

    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).size) > 0
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except Exception:
        return True


def _geometry_fit_sequence_list(value: object) -> list[object]:
    """Normalize one saved sequence-like payload into a plain list."""

    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            item = arr.item()
            if item is None:
                return []
            if isinstance(item, list):
                return list(item)
            if isinstance(item, tuple):
                return list(item)
            return [item]
        return list(arr.tolist())
    try:
        return list(value)  # type: ignore[arg-type]
    except Exception:
        return [value]


def redraw_runtime_geometry_fit_overlay_state(
    overlay_state: Mapping[str, object] | None,
    *,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
) -> bool:
    """Redraw one saved geometry-fit overlay payload when it contains display data."""

    overlay_payload = dict(overlay_state) if isinstance(overlay_state, Mapping) else {}
    overlay_records = _geometry_fit_sequence_list(
        overlay_payload.get("overlay_records")
    )
    initial_pairs_display = _geometry_fit_sequence_list(
        overlay_payload.get("initial_pairs_display")
    )
    try:
        max_display_markers = int(overlay_payload.get("max_display_markers", 120))
    except Exception:
        max_display_markers = 120
    max_display_markers = max(1, max_display_markers)

    if overlay_records and callable(draw_overlay_records):
        draw_overlay_records(overlay_records, int(max_display_markers))
        return True
    if initial_pairs_display and callable(draw_initial_pairs_overlay):
        draw_initial_pairs_overlay(initial_pairs_display, int(max_display_markers))
        return True
    return False


def capture_runtime_geometry_fit_undo_state(
    *,
    current_ui_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    current_profile_cache: Callable[[], object] | object,
    copy_state_value: Callable[[object], object],
    last_overlay_state: Callable[[], Mapping[str, object] | None] | Mapping[str, object] | None,
    build_initial_pairs_display: Callable[..., tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]],
    current_background_index: Callable[[], object] | object,
    current_fit_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    pending_pairs_display: Callable[[], Sequence[dict[str, object]]] | Sequence[dict[str, object]],
) -> dict[str, object]:
    """Capture the current geometry-fit UI/profile/overlay state for undo."""

    overlay_state = copy_state_value(_resolve_runtime_value(last_overlay_state))
    overlay_records = (
        overlay_state.get("overlay_records")
        if isinstance(overlay_state, dict)
        else None
    )
    initial_pairs_display = (
        overlay_state.get("initial_pairs_display")
        if isinstance(overlay_state, dict)
        else None
    )
    if not (
        isinstance(overlay_state, dict)
        and (
            _geometry_fit_sequence_has_items(overlay_records)
            or _geometry_fit_sequence_has_items(initial_pairs_display)
        )
    ):
        try:
            _, initial_pairs_display = build_initial_pairs_display(
                int(_resolve_runtime_value(current_background_index)),
                param_set=_resolve_runtime_value(current_fit_params),
                prefer_cache=True,
            )
            pending_display = _resolve_runtime_value(pending_pairs_display)
            combined_pairs_display = list(initial_pairs_display) + list(pending_display)
            if combined_pairs_display:
                overlay_state = {
                    "overlay_records": [],
                    "initial_pairs_display": copy_state_value(combined_pairs_display),
                    "max_display_markers": max(1, len(combined_pairs_display)),
                }
        except Exception:
            pass

    return {
        "ui_params": copy_state_value(_resolve_runtime_value(current_ui_params)),
        "profile_cache": copy_state_value(_resolve_runtime_value(current_profile_cache)),
        "overlay_state": overlay_state,
    }


def restore_runtime_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Mapping[str, object]:
    """Apply one saved geometry-fit history state back onto the live runtime."""

    ui_params_raw = state.get("ui_params", {}) if isinstance(state, dict) else {}
    ui_params = dict(ui_params_raw) if isinstance(ui_params_raw, Mapping) else {}
    overlay_state_raw = (
        copy_geometry_fit_state_value(state.get("overlay_state"))
        if isinstance(state, dict)
        else None
    )
    overlay_state = (
        dict(overlay_state_raw)
        if isinstance(overlay_state_raw, Mapping)
        else None
    )
    profile_cache_raw = (
        copy_geometry_fit_state_value(state.get("profile_cache", {}))
        if isinstance(state, dict)
        else {}
    )
    profile_cache = (
        dict(profile_cache_raw)
        if isinstance(profile_cache_raw, Mapping)
        else {}
    )
    restored = apply_undo_state(
        {
            **(dict(state) if isinstance(state, dict) else {}),
            "ui_params": ui_params,
            "overlay_state": overlay_state,
            "profile_cache": profile_cache,
        },
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
    )
    restored_profile_cache = restored.get("profile_cache", {})
    replace_profile_cache(
        dict(restored_profile_cache)
        if isinstance(restored_profile_cache, Mapping)
        else {}
    )
    overlay_state = restored.get("overlay_state")
    set_last_overlay_state(
        dict(overlay_state) if isinstance(overlay_state, dict) else None
    )

    if callable(request_preview_skip_once):
        request_preview_skip_once()
    if callable(mark_last_simulation_dirty):
        mark_last_simulation_dirty()
    if callable(cancel_pending_update):
        cancel_pending_update()
    if callable(run_update):
        run_update()

    redraw_runtime_geometry_fit_overlay_state(
        overlay_state if isinstance(overlay_state, Mapping) else None,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
    )

    if callable(refresh_status):
        refresh_status()
    if callable(update_manual_pick_button_label):
        update_manual_pick_button_label()
    return restored


def build_runtime_geometry_fit_undo_restore_callback(
    *,
    var_map_factory: Mapping[str, object] | Callable[[], Mapping[str, object]],
    geometry_theta_offset_var_factory: object | Callable[[], object] | None = None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Callable[[dict[str, object]], Mapping[str, object] | None]:
    """Build one undo-restore callback that resolves live runtime hooks lazily."""

    def _restore(state: dict[str, object]) -> Mapping[str, object] | None:
        if not isinstance(state, dict):
            return None
        return restore_runtime_geometry_fit_undo_state(
            state,
            var_map=_resolve_runtime_value(var_map_factory),
            geometry_theta_offset_var=_resolve_runtime_value(
                geometry_theta_offset_var_factory
            ),
            replace_profile_cache=replace_profile_cache,
            set_last_overlay_state=set_last_overlay_state,
            request_preview_skip_once=request_preview_skip_once,
            mark_last_simulation_dirty=mark_last_simulation_dirty,
            cancel_pending_update=cancel_pending_update,
            run_update=run_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            apply_undo_state=apply_undo_state,
        )

    return _restore


def _run_runtime_geometry_fit_history_transition(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_transition: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
    empty_text: str,
    failure_prefix: str,
    success_text: str,
) -> bool:
    """Run one geometry-fit history transition against the live runtime."""

    history_available = (
        bool(has_history()) if callable(has_history) else bool(has_history)
    )
    if not history_available:
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    current_state = capture_current_state()
    state = read_state()
    if not isinstance(state, dict):
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    try:
        restore_state(state)
    except Exception as exc:
        if callable(set_progress_text):
            set_progress_text(f"{failure_prefix}: {exc}")
        return False

    commit_transition(current_state)
    if callable(update_button_state):
        update_button_state()
    if callable(set_progress_text):
        set_progress_text(success_text)
    return True


def undo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_undo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_undo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Restore the previous geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_undo_state,
        restore_state=restore_state,
        commit_transition=commit_undo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to undo.",
        failure_prefix="Failed to undo geometry fit",
        success_text="Restored the previous geometry-fit state.",
    )


def redo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_redo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_redo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Reapply the next geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_redo_state,
        restore_state=restore_state,
        commit_transition=commit_redo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to redo.",
        failure_prefix="Failed to redo geometry fit",
        success_text="Reapplied the next geometry-fit state.",
    )


def build_runtime_geometry_fit_history_callbacks(
    *,
    history_state: Any,
    capture_current_state: Callable[[], dict[str, object]],
    restore_state: Callable[[dict[str, object]], object],
    copy_state_value: Callable[[object], object],
    history_limit: Callable[[], object] | object,
    peek_last_undo_state: Callable[..., dict[str, object] | None],
    peek_last_redo_state: Callable[..., dict[str, object] | None],
    commit_undo: Callable[..., None],
    commit_redo: Callable[..., None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryFitRuntimeHistoryCallbacks:
    """Build shared runtime undo/redo callbacks around one geometry-fit history store."""

    def _undo() -> bool:
        return undo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "undo_stack", [])),
            capture_current_state=capture_current_state,
            read_undo_state=(
                lambda: peek_last_undo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_undo=(
                lambda current_state: commit_undo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    def _redo() -> bool:
        return redo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "redo_stack", [])),
            capture_current_state=capture_current_state,
            read_redo_state=(
                lambda: peek_last_redo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_redo=(
                lambda current_state: commit_redo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    return GeometryFitRuntimeHistoryCallbacks(
        undo=_undo,
        redo=_redo,
    )


def make_runtime_geometry_tool_action_callbacks(
    *,
    geometry_fit_history_state: Any,
    manual_pick_armed: Callable[[], bool] | bool,
    set_manual_pick_armed: Callable[[bool], None],
    current_background_index: Callable[[], object] | object,
    current_pick_session: Callable[[], object] | object,
    manual_pick_session_active: Callable[[], bool] | bool,
    build_manual_pick_button_label: Callable[..., str],
    pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    pair_group_count: Callable[[int], int],
    set_manual_pick_text: Callable[[str], None] | None = None,
    set_history_button_state: Callable[[bool, bool], None] | None = None,
    show_caked_2d_var: Any = None,
    toggle_caked_2d: Callable[[], None] | None = None,
    ensure_geometry_fit_caked_view: Callable[[], None] | None = None,
    set_hkl_pick_mode: Callable[..., None] | None = None,
    set_geometry_preview_exclude_mode: Callable[..., None] | None = None,
    cancel_manual_pick_session: Callable[..., None] | None = None,
    canvas_widget: Callable[[], Any] | Any = None,
    push_manual_undo_state: Callable[[], None] | None = None,
    clear_pairs_for_current_background: Callable[[int], None] | None = None,
    clear_geometry_pick_artists: Callable[[], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryToolActionRuntimeCallbacks:
    """Build the live geometry tool action callbacks around shared helpers."""

    def _current_background() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _manual_pick_armed() -> bool:
        return bool(_resolve_runtime_value(manual_pick_armed))

    def _manual_pick_session_is_active() -> bool:
        active = (
            manual_pick_session_active()
            if callable(manual_pick_session_active)
            else manual_pick_session_active
        )
        return bool(active)

    def _update_fit_history_button_state() -> None:
        set_runtime_geometry_fit_history_button_state(
            can_undo=bool(getattr(geometry_fit_history_state, "undo_stack", [])),
            can_redo=bool(getattr(geometry_fit_history_state, "redo_stack", [])),
            set_button_state=set_history_button_state,
        )

    def _update_manual_pick_button_label() -> None:
        label = build_manual_pick_button_label(
            armed=_manual_pick_armed(),
            current_background_index=_current_background(),
            pick_session=_resolve_runtime_value(current_pick_session),
            pairs_for_index=pairs_for_index,
            pair_group_count=pair_group_count,
        )
        if callable(set_manual_pick_text):
            set_manual_pick_text(str(label))

    def _set_manual_pick_mode(enabled: bool, message: str | None = None) -> None:
        armed = bool(enabled)
        set_manual_pick_armed(armed)

        if armed:
            if callable(ensure_geometry_fit_caked_view):
                ensure_geometry_fit_caked_view()
            else:
                show_caked_2d_var_local = _resolve_runtime_value(show_caked_2d_var)
                show_caked_2d = False
                getter = getattr(show_caked_2d_var_local, "get", None)
                if callable(getter):
                    try:
                        show_caked_2d = bool(getter())
                    except Exception:
                        show_caked_2d = False
                if not show_caked_2d:
                    setter = getattr(show_caked_2d_var_local, "set", None)
                    if callable(setter):
                        try:
                            setter(True)
                        except Exception:
                            pass
                    if callable(toggle_caked_2d):
                        toggle_caked_2d()
            if callable(set_hkl_pick_mode):
                set_hkl_pick_mode(False)
            if callable(set_geometry_preview_exclude_mode):
                set_geometry_preview_exclude_mode(False)
        elif callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=True)

        _update_manual_pick_button_label()

        widget = _resolve_runtime_value(canvas_widget)
        configure = getattr(widget, "configure", None)
        if callable(configure):
            try:
                configure(cursor="crosshair" if armed else "")
            except Exception:
                pass

        if message and callable(set_progress_text):
            set_progress_text(message)

    def _toggle_manual_pick_mode() -> None:
        armed = _manual_pick_armed()
        _set_manual_pick_mode(
            not armed,
            message=(
                (
                    "Manual geometry picking armed in 2D caked phi-vs-2theta view. "
                    "Click a Qr/Qz set once, then click the matching background peaks "
                    "for each simulated member of that set."
                )
                if not armed
                else "Manual geometry picking disabled."
            ),
        )

    def _clear_current_manual_pairs() -> None:
        background_index = _current_background()
        if pairs_for_index(background_index) or _manual_pick_session_is_active():
            if callable(push_manual_undo_state):
                push_manual_undo_state()
        if callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=False)
        if callable(clear_pairs_for_current_background):
            clear_pairs_for_current_background(background_index)
        if callable(clear_geometry_pick_artists):
            clear_geometry_pick_artists()
        _update_manual_pick_button_label()
        if callable(refresh_status):
            refresh_status()
        if callable(set_progress_text):
            set_progress_text(
                "Cleared saved geometry pairs for the current background image."
            )

    return GeometryToolActionRuntimeCallbacks(
        update_fit_history_button_state=_update_fit_history_button_state,
        update_manual_pick_button_label=_update_manual_pick_button_label,
        set_manual_pick_mode=_set_manual_pick_mode,
        toggle_manual_pick_mode=_toggle_manual_pick_mode,
        clear_current_manual_pairs=_clear_current_manual_pairs,
    )


def build_geometry_manual_fit_dataset(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    orientation_cfg: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one saved-manual-pair geometry dataset for the optimizer."""

    background_idx = int(background_index)
    selected_entries = list(
        manual_dataset_bindings.geometry_manual_pairs_for_index(background_idx) or ()
    )
    if not selected_entries:
        raise RuntimeError(
            f"background {background_idx + 1} has no saved manual geometry pairs"
        )

    native_background, display_background = (
        manual_dataset_bindings.load_background_by_index(background_idx)
    )
    backend_background = (
        manual_dataset_bindings.apply_background_backend_orientation(
            native_background
        )
    )
    if backend_background is None:
        backend_background = native_background

    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    params_i["theta_initial"] = float(theta_base + theta_offset)
    simulated_peaks = manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
        params_i,
        prefer_cache=(
            background_idx == int(manual_dataset_bindings.current_background_index)
        ),
    )
    simulated_lookup = manual_dataset_bindings.geometry_manual_simulated_lookup(
        simulated_peaks
    )

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
        bg_coords = manual_dataset_bindings.geometry_manual_entry_display_coords(entry)
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
                try:
                    sim_col_raw = float(sim_entry.get("sim_col_raw", sim_col))
                    sim_row_raw = float(sim_entry.get("sim_row_raw", sim_row))
                except Exception:
                    sim_col_raw = float("nan")
                    sim_row_raw = float("nan")
                if np.isfinite(sim_col_raw) and np.isfinite(sim_row_raw):
                    try:
                        sim_native = manual_dataset_bindings.display_to_native_sim_coords(
                            float(sim_col_raw),
                            float(sim_row_raw),
                            (
                                int(manual_dataset_bindings.image_size),
                                int(manual_dataset_bindings.image_size),
                            ),
                        )
                    except Exception:
                        sim_native = None
                    if (
                        isinstance(sim_native, tuple)
                        and len(sim_native) >= 2
                        and np.isfinite(float(sim_native[0]))
                        and np.isfinite(float(sim_native[1]))
                    ):
                        initial_entry["sim_native"] = (
                            float(sim_native[0]),
                            float(sim_native[1]),
                        )
        initial_pairs_display.append(initial_entry)

    measured_native = manual_dataset_bindings.unrotate_display_peaks(
        measured_display,
        display_background.shape,
        k=manual_dataset_bindings.display_rotate_k,
    )
    for initial_entry, measured_entry in zip(initial_pairs_display, measured_native):
        if not isinstance(measured_entry, Mapping):
            continue
        try:
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if np.isfinite(mx) and np.isfinite(my):
            initial_entry["bg_native"] = (float(mx), float(my))

    sim_orientation_points: list[tuple[float, float]] = []
    meas_orientation_points: list[tuple[float, float]] = []
    sim_native_shape = (
        int(manual_dataset_bindings.image_size),
        int(manual_dataset_bindings.image_size),
    )
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
            sim_native_raw = initial_entry.get("sim_native")
            if (
                isinstance(sim_native_raw, (list, tuple, np.ndarray))
                and len(sim_native_raw) >= 2
            ):
                sim_native = (
                    float(sim_native_raw[0]),
                    float(sim_native_raw[1]),
                )
            else:
                sim_native = manual_dataset_bindings.display_to_native_sim_coords(
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

    orientation_choice, orientation_diag = (
        manual_dataset_bindings.select_fit_orientation(
            sim_orientation_points,
            meas_orientation_points,
            tuple(int(v) for v in native_background.shape[:2]),
            cfg=orientation_cfg or {},
        )
    )
    measured_for_fit = manual_dataset_bindings.apply_orientation_to_entries(
        measured_native,
        native_background.shape,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    experimental_image_for_fit = manual_dataset_bindings.orient_image_for_fit(
        backend_background,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )

    label = (
        Path(str(manual_dataset_bindings.osc_files[background_idx])).name
        if 0 <= background_idx < len(manual_dataset_bindings.osc_files)
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
            fit_params["theta_initial"] = float(background_theta_values[current_index])
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
            geometry_runtime_cfg=apply_joint_geometry_fit_runtime_safety_overrides(
                build_runtime_config(fit_params),
                joint_background_mode=joint_background_mode,
            ),
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
    manual_dataset_bindings = bindings.manual_dataset_bindings

    return prepare_geometry_fit_run(
        params=params,
        var_names=var_names,
        fit_config=fit_config,
        osc_files=manual_dataset_bindings.osc_files,
        current_background_index=int(manual_dataset_bindings.current_background_index),
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
        geometry_manual_pairs_for_index=(
            manual_dataset_bindings.geometry_manual_pairs_for_index
        ),
        ensure_geometry_fit_caked_view=bindings.ensure_geometry_fit_caked_view,
        build_dataset=(
            lambda background_index, *, theta_base, base_fit_params, orientation_cfg: (
                build_geometry_manual_fit_dataset(
                    background_index,
                    theta_base=theta_base,
                    base_fit_params=base_fit_params,
                    manual_dataset_bindings=manual_dataset_bindings,
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


def apply_joint_geometry_fit_runtime_safety_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Clamp interactive multi-background fits to a conservative solver profile."""

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    if not joint_background_mode:
        return cfg

    solver_cfg_raw = cfg.get("solver", {})
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}
    solver_cfg["workers"] = 1
    solver_cfg["parallel_mode"] = "off"
    solver_cfg["worker_numba_threads"] = 1
    solver_cfg["stagnation_probe"] = False
    solver_cfg["stagnation_probe_pairwise"] = False
    solver_cfg["stagnation_probe_random_directions"] = 0
    if "restarts" in solver_cfg:
        try:
            solver_cfg["restarts"] = min(max(int(solver_cfg["restarts"]), 0), 1)
        except Exception:
            solver_cfg["restarts"] = 1
    cfg["solver"] = solver_cfg
    cfg["use_numba"] = False

    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        dict(identifiability_cfg_raw)
        if isinstance(identifiability_cfg_raw, Mapping)
        else {}
    )
    identifiability_cfg["enabled"] = False
    cfg["identifiability"] = identifiability_cfg
    return cfg


def build_geometry_fit_dataset_cache_payload(
    prepared_run: GeometryFitPreparedRun,
    *,
    current_background_index: int | None = None,
) -> dict[str, object]:
    """Copy the reusable dataset bundle from a successful geometry fit."""

    dataset_index = current_background_index
    if dataset_index is None:
        current_dataset = (
            prepared_run.current_dataset
            if isinstance(prepared_run.current_dataset, Mapping)
            else {}
        )
        try:
            dataset_index = int(current_dataset.get("dataset_index", 0))
        except Exception:
            dataset_index = 0

    return {
        "selected_background_indices": [
            int(idx) for idx in prepared_run.selected_background_indices
        ],
        "current_background_index": int(dataset_index),
        "joint_background_mode": bool(prepared_run.joint_background_mode),
        "background_theta_values": [
            float(value) for value in prepared_run.background_theta_values
        ],
        "dataset_specs": [
            copy_geometry_fit_state_value(dict(spec))
            for spec in (prepared_run.dataset_specs or [])
            if isinstance(spec, Mapping)
        ],
    }


def geometry_fit_dataset_cache_stale_reason(
    cache_payload: Mapping[str, object] | None,
    *,
    selected_background_indices: Sequence[object],
    current_background_index: int,
    joint_background_mode: bool,
    background_theta_values: Sequence[object],
) -> str | None:
    """Return a human-readable stale-cache reason, or ``None`` when valid."""

    if not isinstance(cache_payload, Mapping):
        return "Run geometry fit first."

    raw_specs = cache_payload.get("dataset_specs")
    if not isinstance(raw_specs, Sequence) or len(raw_specs) <= 0:
        return "Run geometry fit first."

    try:
        cached_selected = [
            int(idx) for idx in cache_payload.get("selected_background_indices", [])
        ]
    except Exception:
        return "Run geometry fit first."
    current_selected = [int(idx) for idx in selected_background_indices]
    if cached_selected != current_selected:
        return "Geometry-fit background selection changed. Rerun geometry fit."

    try:
        cached_index = int(cache_payload.get("current_background_index", -1))
    except Exception:
        return "Run geometry fit first."
    if int(cached_index) != int(current_background_index):
        return "Active background changed since geometry fit. Rerun geometry fit."

    cached_joint = bool(cache_payload.get("joint_background_mode", False))
    if cached_joint != bool(joint_background_mode):
        return "Shared-theta mode changed since geometry fit. Rerun geometry fit."

    try:
        cached_theta_values = [
            float(value) for value in cache_payload.get("background_theta_values", [])
        ]
        current_theta_values = [float(value) for value in background_theta_values]
    except Exception:
        return "Background theta values are unavailable. Rerun geometry fit."
    if len(cached_theta_values) != len(current_theta_values):
        return "Background theta values changed since geometry fit. Rerun geometry fit."
    for cached_value, current_value in zip(
        cached_theta_values,
        current_theta_values,
    ):
        if not np.isfinite(cached_value) or not np.isfinite(current_value):
            return "Background theta values changed since geometry fit. Rerun geometry fit."
        if not np.isclose(
            float(cached_value),
            float(current_value),
            rtol=0.0,
            atol=1.0e-9,
        ):
            return "Background theta values changed since geometry fit. Rerun geometry fit."

    return None


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


def should_apply_geometry_fit_runtime_safety_overrides(
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether GUI geometry fitting should force the safe serial runtime."""

    if platform_name is None:
        platform_name = os.name
    if version_info is None:
        version_info = sys.version_info
    if env is None:
        env = os.environ

    opt_out = str(
        env.get("RA_SIM_ALLOW_UNSAFE_GEOMETRY_FIT_RUNTIME", "")
    ).strip().lower()
    if opt_out in {"1", "true", "yes", "on"}:
        return False

    if str(platform_name).strip().lower() != "nt":
        return False

    try:
        major = int(version_info[0])
        minor = int(version_info[1])
    except Exception:
        return False
    return (major, minor) >= (3, 13)


def apply_geometry_fit_runtime_safety_overrides(
    refinement_config: Mapping[str, object] | None,
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str | None]:
    """Return one copied refinement config with GUI runtime safety overrides."""

    if isinstance(refinement_config, Mapping):
        resolved = copy.deepcopy(dict(refinement_config))
    else:
        resolved = {}

    if not should_apply_geometry_fit_runtime_safety_overrides(
        platform_name=platform_name,
        version_info=version_info,
        env=env,
    ):
        return resolved, None

    solver_cfg_raw = resolved.get("solver", {})
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}

    changed = False
    if bool(resolved.get("use_numba", True)):
        resolved["use_numba"] = False
        changed = True
    if solver_cfg.get("parallel_mode") != "off":
        solver_cfg["parallel_mode"] = "off"
        changed = True
    if solver_cfg.get("workers") != 1:
        solver_cfg["workers"] = 1
        changed = True
    if solver_cfg.get("worker_numba_threads") != 1:
        solver_cfg["worker_numba_threads"] = 1
        changed = True

    resolved["solver"] = solver_cfg
    if not changed:
        return resolved, None

    return (
        resolved,
        (
            "Windows/Python 3.13 runtime guard enabled: "
            "geometry fit is running serially with Numba disabled."
        ),
    )


def build_geometry_fit_solver_request(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> GeometryFitSolverRequest:
    """Build one concrete solver request from a prepared geometry-fit run."""

    refinement_config, runtime_safety_note = (
        apply_geometry_fit_runtime_safety_overrides(
            prepared_run.geometry_runtime_cfg,
        )
    )

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
        refinement_config=refinement_config,
        runtime_safety_note=runtime_safety_note,
    )


def solve_geometry_fit_request(
    request: GeometryFitSolverRequest,
    *,
    solve_fit: Callable[..., object],
    status_callback: Callable[[str], None] | None = None,
) -> object:
    """Invoke the live geometry-fit solver for one prepared request."""

    solve_kwargs: dict[str, object] = {
        "pixel_tol": float("inf"),
        "experimental_image": request.experimental_image,
        "dataset_specs": request.dataset_specs,
        "refinement_config": request.refinement_config,
    }
    if callable(status_callback):
        try:
            signature = inspect.signature(solve_fit)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            accepts_status_callback = (
                "status_callback" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_status_callback:
                solve_kwargs["status_callback"] = status_callback

    return solve_fit(
        request.miller,
        request.intensities,
        request.image_size,
        request.params,
        request.measured_peaks,
        request.var_names,
        **solve_kwargs,
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


def _geometry_fit_metric_float(value: object, *, default: float = np.nan) -> float:
    """Return one finite float metric, or ``default`` when unavailable."""

    try:
        resolved = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(resolved):
        return float(default)
    return float(resolved)


def _geometry_fit_debug_value_text(
    value: object,
    *,
    float_digits: int = 6,
) -> str:
    """Format one geometry-fit debug value for log output."""

    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.{int(float_digits)}f}"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key}={_geometry_fit_debug_value_text(val, float_digits=float_digits)}"
            for key, val in value.items()
        ) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [
            _geometry_fit_debug_value_text(item, float_digits=float_digits)
            for item in items[:6]
        ]
        if len(items) > 6:
            preview.append("...")
        return "[" + ", ".join(preview) + "]"
    return str(value)


def build_geometry_fit_debug_lines(result: object) -> list[str]:
    """Format solver setup/progress details for one geometry-fit result."""

    debug_summary = getattr(result, "geometry_fit_debug_summary", None)
    if not isinstance(debug_summary, Mapping):
        return []

    lines: list[str] = [
        "point_match_mode={mode} datasets={datasets} vars={vars}".format(
            mode=bool(debug_summary.get("point_match_mode", False)),
            datasets=int(debug_summary.get("dataset_count", 0) or 0),
            vars=",".join(str(name) for name in debug_summary.get("var_names", []) or ()),
        )
    ]

    solver = debug_summary.get("solver", None)
    if isinstance(solver, Mapping):
        lines.append(
            "solver loss={loss} f_scale_px={f_scale} max_nfev={max_nfev} "
            "restarts={restarts} weighted_matching={weighted} "
            "missing_pair_penalty_px={missing} measurement_uncertainty={unc} "
            "anisotropic_uncertainty={anisotropic}".format(
                loss=str(solver.get("loss", "<unknown>")),
                f_scale=_geometry_fit_debug_value_text(solver.get("f_scale_px", np.nan)),
                max_nfev=_geometry_fit_debug_value_text(solver.get("max_nfev", "<unknown>"), float_digits=0),
                restarts=_geometry_fit_debug_value_text(solver.get("restarts", "<unknown>"), float_digits=0),
                weighted=_geometry_fit_debug_value_text(solver.get("weighted_matching", False), float_digits=0),
                missing=_geometry_fit_debug_value_text(solver.get("missing_pair_penalty_px", np.nan)),
                unc=_geometry_fit_debug_value_text(solver.get("use_measurement_uncertainty", False), float_digits=0),
                anisotropic=_geometry_fit_debug_value_text(
                    solver.get("anisotropic_measurement_uncertainty", False),
                    float_digits=0,
                ),
            )
        )

    parallel = debug_summary.get("parallelization", None)
    if isinstance(parallel, Mapping):
        lines.append(
            "parallel mode={mode} configured_workers={configured} "
            "dataset_workers={datasets} restart_workers={restarts} "
            "worker_numba_threads={threads} thread_budget={budget}".format(
                mode=str(parallel.get("mode", "<unknown>")),
                configured=_geometry_fit_debug_value_text(parallel.get("configured_workers", "<unknown>"), float_digits=0),
                datasets=_geometry_fit_debug_value_text(parallel.get("dataset_workers", "<unknown>"), float_digits=0),
                restarts=_geometry_fit_debug_value_text(parallel.get("restart_workers", "<unknown>"), float_digits=0),
                threads=_geometry_fit_debug_value_text(parallel.get("worker_numba_threads", "None"), float_digits=0),
                budget=_geometry_fit_debug_value_text(parallel.get("numba_thread_budget", "<unknown>"), float_digits=0),
            )
        )

    main_seed = debug_summary.get("main_solve_seed", None)
    if isinstance(main_seed, Mapping):
        lines.append(
            "main_seed cost={cost} weighted_rms_px={rms}".format(
                cost=_geometry_fit_debug_value_text(main_seed.get("cost", np.nan)),
                rms=_geometry_fit_debug_value_text(main_seed.get("weighted_rms_px", np.nan)),
            )
        )
        point_seed = main_seed.get("point_match_summary", None)
        if isinstance(point_seed, Mapping):
            lines.append(
                "main_seed_point_match matched={matched} missing={missing} "
                "peak_rms_px={peak_rms} peak_max_px={peak_max}".format(
                    matched=_geometry_fit_debug_value_text(point_seed.get("matched_pair_count", 0), float_digits=0),
                    missing=_geometry_fit_debug_value_text(point_seed.get("missing_pair_count", 0), float_digits=0),
                    peak_rms=_geometry_fit_debug_value_text(point_seed.get("unweighted_peak_rms_px", np.nan)),
                    peak_max=_geometry_fit_debug_value_text(point_seed.get("unweighted_peak_max_px", np.nan)),
                )
            )

    for entry in debug_summary.get("dataset_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "dataset[{idx}] label={label} theta_initial_deg={theta} measured={measured} "
            "subset_reflections={subset}/{total} fixed_source_reflections={fixed} "
            "fallback_hkls={fallback} reduced={reduced}".format(
                idx=_geometry_fit_debug_value_text(entry.get("dataset_index", -1), float_digits=0),
                label=str(entry.get("label", "")),
                theta=_geometry_fit_debug_value_text(entry.get("theta_initial_deg", np.nan)),
                measured=_geometry_fit_debug_value_text(entry.get("measured_count", 0), float_digits=0),
                subset=_geometry_fit_debug_value_text(entry.get("subset_reflection_count", 0), float_digits=0),
                total=_geometry_fit_debug_value_text(entry.get("total_reflection_count", 0), float_digits=0),
                fixed=_geometry_fit_debug_value_text(entry.get("fixed_source_reflection_count", 0), float_digits=0),
                fallback=_geometry_fit_debug_value_text(entry.get("fallback_hkl_count", 0), float_digits=0),
                reduced=_geometry_fit_debug_value_text(entry.get("subset_reduced", False), float_digits=0),
            )
        )

    for entry in debug_summary.get("parameter_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        line = (
            "param[{name}] group={group} start={start} final={final} delta={delta} "
            "bounds=[{lower}, {upper}] scale={scale}".format(
                name=str(entry.get("name", "")),
                group=str(entry.get("group", "other")),
                start=_geometry_fit_debug_value_text(entry.get("start", np.nan)),
                final=_geometry_fit_debug_value_text(entry.get("final", np.nan)),
                delta=_geometry_fit_debug_value_text(entry.get("delta", np.nan)),
                lower=_geometry_fit_debug_value_text(entry.get("lower_bound", np.nan)),
                upper=_geometry_fit_debug_value_text(entry.get("upper_bound", np.nan)),
                scale=_geometry_fit_debug_value_text(entry.get("scale", np.nan)),
            )
        )
        if bool(entry.get("prior_enabled", False)):
            line += (
                " prior_center={center} prior_sigma={sigma}".format(
                    center=_geometry_fit_debug_value_text(entry.get("prior_center", np.nan)),
                    sigma=_geometry_fit_debug_value_text(entry.get("prior_sigma", np.nan)),
                )
            )
        lines.append(line)

    final_summary = debug_summary.get("final", None)
    if isinstance(final_summary, Mapping):
        lines.append(
            "final cost={cost} robust_cost={robust} weighted_rms_px={weighted_rms} "
            "display_rms_px={display_rms}".format(
                cost=_geometry_fit_debug_value_text(final_summary.get("cost", np.nan)),
                robust=_geometry_fit_debug_value_text(final_summary.get("robust_cost", np.nan)),
                weighted_rms=_geometry_fit_debug_value_text(final_summary.get("weighted_rms_px", np.nan)),
                display_rms=_geometry_fit_debug_value_text(final_summary.get("display_rms_px", np.nan)),
            )
        )

    solve_progress = debug_summary.get("solve_progress", None)
    if isinstance(solve_progress, Mapping):
        lines.append(
            "solve_progress label={label} evaluations={evals} best_cost={best_cost} "
            "best_weighted_rms_px={best_rms} status_updates={updates}".format(
                label=str(solve_progress.get("label", "")),
                evals=_geometry_fit_debug_value_text(solve_progress.get("evaluation_count", 0), float_digits=0),
                best_cost=_geometry_fit_debug_value_text(solve_progress.get("best_cost_seen", np.nan)),
                best_rms=_geometry_fit_debug_value_text(solve_progress.get("best_weighted_rms_px", np.nan)),
                updates=_geometry_fit_debug_value_text(solve_progress.get("status_emit_count", 0), float_digits=0),
            )
        )
        for idx, event in enumerate(solve_progress.get("trace", []) or []):
            if not isinstance(event, Mapping):
                continue
            lines.append(
                "solve_progress[{idx}] eval={eval} reason={reason} cost={cost} "
                "best_cost={best_cost} weighted_rms_px={rms}".format(
                    idx=int(idx),
                    eval=_geometry_fit_debug_value_text(event.get("eval", 0), float_digits=0),
                    reason=str(event.get("reason", "")),
                    cost=_geometry_fit_debug_value_text(event.get("current_cost", np.nan)),
                    best_cost=_geometry_fit_debug_value_text(event.get("best_cost", np.nan)),
                    rms=_geometry_fit_debug_value_text(event.get("weighted_rms_px", np.nan)),
                )
            )

    return lines


def build_geometry_fit_stage_summary_lines(result: object) -> list[str]:
    """Format the stage-by-stage geometry-fit workflow summaries."""

    stage_specs = [
        ("reparameterization", getattr(result, "reparameterization_summary", None)),
        ("staged_release", getattr(result, "staged_release_summary", None)),
        ("adaptive_regularization", getattr(result, "adaptive_regularization_summary", None)),
        ("single_ray_polish", getattr(result, "single_ray_polish_summary", None)),
        ("ridge_refinement", getattr(result, "ridge_refinement_summary", None)),
        ("image_refinement", getattr(result, "image_refinement_summary", None)),
        ("auto_freeze", getattr(result, "auto_freeze_summary", None)),
        ("selective_thaw", getattr(result, "selective_thaw_summary", None)),
    ]
    preferred_keys = (
        "status",
        "reason",
        "accepted",
        "success",
        "start_cost",
        "final_cost",
        "regularized_cost",
        "release_cost",
        "matched_pair_count_before",
        "matched_pair_count_after",
        "accepted_stage_count",
        "release_accepted",
        "coarse_single_ray_enabled",
        "coarse_single_ray_accepted",
        "fixed_parameters",
        "thawed_parameters",
        "applied_parameters",
        "remaining_fixed_parameters",
        "nfev",
        "max_nfev",
    )

    lines: list[str] = []
    for stage_name, summary in stage_specs:
        if not isinstance(summary, Mapping):
            continue
        parts = [f"{stage_name}:"]
        for key in preferred_keys:
            if key not in summary:
                continue
            value = summary.get(key)
            if value in ("", None):
                continue
            parts.append(
                f"{key}={_geometry_fit_debug_value_text(value)}"
            )
        if len(parts) == 1:
            parts.append("summary=<empty>")
        lines.append(" ".join(parts))
    return lines


def build_geometry_fit_rejection_reason_lines(
    result: object,
    *,
    rms: float,
) -> list[str]:
    """Return human-readable rejection reasons for one geometry-fit result."""

    reasons: list[str] = []

    if not bool(getattr(result, "success", True)):
        reasons.append("Optimizer did not report success.")

    if not np.isfinite(rms):
        reasons.append("RMS residual is not finite.")
    elif float(rms) > GEOMETRY_FIT_ACCEPT_MAX_RMS_PX:
        reasons.append(
            "RMS residual {rms:.2f} px exceeds the acceptance limit of "
            "{limit:.2f} px.".format(
                rms=float(rms),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_RMS_PX),
            )
        )

    point_match_summary = getattr(result, "point_match_summary", None)
    matched_pair_count = 0
    has_matched_pair_count = False
    max_offset = float("nan")
    if isinstance(point_match_summary, Mapping):
        if "matched_pair_count" in point_match_summary:
            has_matched_pair_count = True
            try:
                matched_pair_count = int(point_match_summary.get("matched_pair_count", 0))
            except Exception:
                matched_pair_count = 0
        max_offset = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_max_px", np.nan)
        )

    if has_matched_pair_count and matched_pair_count <= 0:
        reasons.append("No matched peak pairs were available for the fitted solution.")
    if np.isfinite(max_offset) and float(max_offset) > GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX:
        reasons.append(
            "Largest matched-peak offset {offset:.2f} px exceeds the acceptance "
            "limit of {limit:.2f} px.".format(
                offset=float(max_offset),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX),
            )
        )

    return reasons


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


def build_geometry_fit_rejected_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    rms: float,
    rejection_reasons: Sequence[object],
) -> str:
    """Build the GUI status text for one rejected manual geometry fit."""

    lines = ["Manual geometry fit rejected:"]
    for reason in rejection_reasons:
        text = str(reason).strip()
        if text:
            lines.append(text)
    if np.isfinite(rms):
        lines.append(f"RMS residual = {float(rms):.2f} px")
    lines.append(
        "Manual pairs: {points} points across {groups} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
    )
    lines.append("Add more manual points or remove outliers before rerunning the fit.")
    return "\n".join(lines)


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

    simulated_params = dict(fitted_params)
    try:
        theta_base = float(simulated_params.get("theta_initial", np.nan))
        theta_offset = float(simulated_params.get("theta_offset", 0.0))
    except Exception:
        theta_base = float("nan")
        theta_offset = 0.0
    if np.isfinite(theta_base) and np.isfinite(theta_offset):
        simulated_params["theta_initial"] = float(theta_base + theta_offset)

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
        simulated_params,
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
    debug_lines = build_geometry_fit_debug_lines(result)
    if debug_lines:
        bindings.log_section(
            "Fit mechanics:",
            debug_lines,
        )
    stage_summary_lines = build_geometry_fit_stage_summary_lines(result)
    if stage_summary_lines:
        bindings.log_section(
            "Solver stages:",
            stage_summary_lines,
        )

    result_vector = getattr(result, "x", None)
    result_values = [] if result_vector is None else list(result_vector)
    rms = geometry_fit_result_rms(result)
    rejection_reasons = build_geometry_fit_rejection_reason_lines(
        result,
        rms=rms,
    )
    if rejection_reasons:
        bindings.log_section(
            "Optimization result:",
            build_geometry_fit_result_lines(
                var_names,
                result_values,
                rms=rms,
            ),
        )
        point_match_summary_lines = build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        )
        if point_match_summary_lines:
            bindings.log_section(
                "Point-match summary:",
                point_match_summary_lines,
            )
        bindings.log_section("Fit rejected:", rejection_reasons)
        bindings.set_progress_text(
            build_geometry_fit_rejected_progress_text(
                current_dataset=current_dataset,
                dataset_count=dataset_count,
                joint_background_mode=joint_background_mode,
                rms=rms,
                rejection_reasons=rejection_reasons,
            )
        )
        bindings.cmd_line(
            "rejected: "
            f"datasets={int(dataset_count)} "
            f"groups={int(current_dataset.get('group_count', 0) or 0)} "
            f"points={int(current_dataset.get('pair_count', 0) or 0)} "
            f"rms={float(rms):.4f}px "
            f"reason={rejection_reasons[0]}"
        )
        return GeometryFitRuntimeApplyResult(
            accepted=False,
            rejection_reason=" ".join(str(reason) for reason in rejection_reasons),
            rms=float(rms),
            fitted_params=None,
            postprocess=None,
        )

    undo_state = bindings.capture_undo_state()
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
        accepted=True,
        rejection_reason=None,
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
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
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

    def _sync_joint_background_theta() -> None:
        background_index = int(
            getattr(background_runtime_state, "current_background_index", 0)
        )
        if 0 <= background_index < len(prepared_run.background_theta_values):
            theta_initial_var.set(
                float(prepared_run.background_theta_values[background_index])
            )
            return
        theta_initial_var.set(
            background_theta_for_index(
                background_index,
                strict_count=False,
            )
        )

    ui_bindings = GeometryFitRuntimeUiBindings(
        fit_params=prepared_run.fit_params,
        base_profile_cache=getattr(simulation_runtime_state, "profile_cache", {}),
        mosaic_params=mosaic_params,
        current_ui_params=current_ui_params,
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
        capture_undo_state=capture_undo_state,
        sync_joint_background_theta=_sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        replace_profile_cache=(
            lambda profile_cache: setattr(
                simulation_runtime_state,
                "profile_cache",
                dict(profile_cache),
            )
        ),
        replace_dataset_cache=replace_dataset_cache,
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
        replace_dataset_cache=bindings.replace_dataset_cache,
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

    def _solver_status_callback(text: str) -> None:
        status_text = str(text).strip()
        if not status_text:
            return
        _log_line(status_text)
        ui_bindings.cmd_line(status_text)
        ui_bindings.set_progress_text(status_text)
        if callable(flush_ui):
            flush_ui()

    try:
        log_file = log_path.open("w", encoding="utf-8")
        write_geometry_fit_run_start_log(
            stamp=str(postprocess_config.stamp),
            prepared_run=prepared_run,
            cmd_line=ui_bindings.cmd_line,
            log_line=_log_line,
            log_section=_log_section,
        )
        ui_bindings.cmd_line(f"log: {log_path}")

        ui_bindings.set_progress_text(str(start_progress_text))
        if callable(flush_ui):
            flush_ui()

        solver_request = build_geometry_fit_solver_request(
            prepared_run=prepared_run,
            var_names=var_names,
            solver_inputs=postprocess_config.solver_inputs,
        )
        if solver_request.runtime_safety_note:
            _log_line(f"Runtime safety: {solver_request.runtime_safety_note}")
            _log_line()
        result = solve_geometry_fit_request(
            solver_request,
            solve_fit=solve_fit,
            status_callback=_solver_status_callback,
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
        replace_dataset_cache = ui_bindings.replace_dataset_cache
        if apply_result.accepted and callable(replace_dataset_cache):
            replace_dataset_cache(
                build_geometry_fit_dataset_cache_payload(
                    prepared_run,
                    current_background_index=int(
                        postprocess_config.current_background_index
                    ),
                )
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
