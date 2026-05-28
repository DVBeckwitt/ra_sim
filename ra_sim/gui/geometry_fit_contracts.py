"""Geometry-fit contract types shared by GUI runtime helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GeometryFitStageCallback = Callable[[str, Mapping[str, object]], None]


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
    stage_timing_s: dict[str, float] | None = None


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
    geometry_manual_project_peaks_to_current_view: (
        Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None
    geometry_manual_project_peaks_for_background_view: (
        Callable[..., list[dict[str, object]]] | None
    ) = None
    backend_detector_coords_to_native_detector_coords: (
        Callable[
            [float, float, Sequence[object] | None],
            tuple[float | None, float | None],
        ]
        | None
    ) = None
    native_detector_coords_to_bundle_detector_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ) = None
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None
    native_detector_coords_to_detector_display_coords_for_background: (
        Callable[
            [int],
            Callable[[float, float], tuple[float | None, float | None] | None] | None,
        ]
        | None
    ) = None
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None
    geometry_manual_rebuild_source_rows_for_background: Callable[..., object] | None = None
    geometry_manual_last_source_snapshot_diagnostics: Callable[[], Mapping[str, object]] | None = (
        None
    )
    geometry_manual_last_simulation_diagnostics: Callable[[], Mapping[str, object]] | None = None
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None
    pick_uses_caked_space: Callable[[], bool] | None = None
    geometry_manual_caked_view_for_index: Callable[[int], object] | None = None
    geometry_manual_refresh_pair_entry: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None] | None
    ) = None
    geometry_manual_caked_projection_for_index: Callable[[int], object] | None = None
    simulation_native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], object] | None
    ) = None


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
    build_mosaic_params: Callable[..., Mapping[str, object] | None]
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
    candidate_param_names: list[str] | None
    dataset_specs: list[dict[str, object]] | None
    refinement_config: dict[str, object]
    runtime_safety_note: str | None = None


@dataclass(frozen=True)
class GeometryFitPreparationResult:
    """One geometry-fit preflight result."""

    prepared_run: GeometryFitPreparedRun | None = None
    error_text: str | None = None
    failure_log_sections: list[tuple[str, list[str]]] | None = None
    log_path: Path | None = None


@dataclass(frozen=True)
class GeometryFitSourceRowRebuildResult:
    """Pure source-row rebuild payload returned before any runtime-state commit."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    rebuild_source: str | None
    rebuild_attempts: list[str]
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    source_reflection_indices: list[int] | None = None
    intersection_cache: list[object] | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitBackgroundCacheBundle:
    """Job-scoped geometry-fit cache bundle for one prepared background."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    background_label: str
    theta_base: float
    theta_initial: float
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    cache_source: str | None
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    intersection_cache: list[object] | None = None
    cache_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitPostprocessResult:
    """Pure post-solver geometry-fit analysis results."""

    fitted_params: dict[str, object]
    point_match_summary_lines: list[str]
    pixel_offsets: list[dict[str, object]]
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
    geometry_runtime_cfg: Mapping[str, object] | None = None
    preview_fitted_params: (
        Callable[[Sequence[object], Sequence[object]], dict[str, object]] | None
    ) = None


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
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None


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
    log_dir: Path | str | None = None


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
    build_mosaic_params: Callable[..., Mapping[str, object] | None] | None = None


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
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None
    log_dir: Path | str | None = None


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
class GeometryFitSweepApplyResult:
    """Result metadata for applying one dry-run parameter-sweep combo."""

    applied: bool
    combo_result_path: Path
    active_vars: list[str]
    applied_values: dict[str, float]
    accepted_overlay_path: Path | None = None
    rebuild_overlay_result: object | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionResult:
    """Result metadata for one full runtime geometry-fit execution."""

    log_path: Path
    trace_path: Path | None = None
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
