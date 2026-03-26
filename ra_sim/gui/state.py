"""Shared GUI state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ManualGeometryUndoSnapshot:
    """Snapshot of manual-geometry state for undo restoration."""

    pairs_by_background: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    pick_session: dict[str, object] = field(default_factory=dict)


@dataclass
class ManualGeometryState:
    """Mutable state owned by the manual-geometry GUI workflow."""

    pairs_by_background: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    pick_session: dict[str, object] = field(default_factory=dict)
    undo_stack: list[ManualGeometryUndoSnapshot] = field(default_factory=list)


@dataclass
class GeometryFitHistoryState:
    """Mutable state for geometry-fit undo/redo history."""

    undo_stack: list[dict[str, object]] = field(default_factory=list)
    redo_stack: list[dict[str, object]] = field(default_factory=list)
    last_overlay_state: dict[str, object] | None = None


@dataclass
class GeometryFitConstraintsViewState:
    """Widget references for the geometry-fit constraints panel."""

    panel: Any = None
    canvas: Any = None
    body: Any = None
    body_window: Any = None
    controls: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class GeometryFitParameterControlsViewState:
    """Widget references and vars for the fit-geometry parameter checklist."""

    frame: Any = None
    fit_zb_var: Any = None
    fit_zb_checkbutton: Any = None
    fit_zs_var: Any = None
    fit_zs_checkbutton: Any = None
    fit_theta_var: Any = None
    fit_theta_checkbutton: Any = None
    fit_psi_z_var: Any = None
    fit_psi_z_checkbutton: Any = None
    fit_chi_var: Any = None
    fit_chi_checkbutton: Any = None
    fit_cor_var: Any = None
    fit_cor_checkbutton: Any = None
    fit_gamma_var: Any = None
    fit_gamma_checkbutton: Any = None
    fit_Gamma_var: Any = None
    fit_Gamma_checkbutton: Any = None
    fit_dist_var: Any = None
    fit_dist_checkbutton: Any = None
    fit_a_var: Any = None
    fit_a_checkbutton: Any = None
    fit_c_var: Any = None
    fit_c_checkbutton: Any = None
    fit_center_x_var: Any = None
    fit_center_x_checkbutton: Any = None
    fit_center_y_var: Any = None
    fit_center_y_checkbutton: Any = None
    toggle_vars: dict[str, Any] = field(default_factory=dict)
    toggle_checkbuttons: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackgroundThetaControlsViewState:
    """Widget references and vars for background-theta control surfaces."""

    background_theta_controls: Any = None
    background_theta_list_var: Any = None
    background_theta_entry: Any = None
    geometry_theta_offset_var: Any = None
    background_theta_offset_entry: Any = None
    geometry_fit_background_controls: Any = None
    geometry_fit_background_selection_var: Any = None
    geometry_fit_background_entry: Any = None


@dataclass
class WorkspacePanelsViewState:
    """Widget references for the workspace action/background/session panels."""

    workspace_actions_frame: Any = None
    workspace_backgrounds_frame: Any = None
    workspace_session_frame: Any = None
    background_file_status_var: Any = None
    background_file_status_label: Any = None


@dataclass
class BackgroundBackendDebugViewState:
    """Widget references and vars for background backend/orientation debug UI."""

    background_backend_frame: Any = None
    background_backend_status_label: Any = None
    backend_orientation_frame: Any = None
    backend_rotation_var: Any = None
    backend_flip_y_axis_var: Any = None
    backend_flip_x_axis_var: Any = None
    backend_flip_order_var: Any = None


@dataclass
class PrimaryCifControlsViewState:
    """Widget references and vars for the primary-CIF / diffuse-HT controls."""

    cif_frame: Any = None
    cif_file_var: Any = None
    cif_entry: Any = None
    cif_actions_frame: Any = None
    browse_button: Any = None
    apply_button: Any = None
    diffuse_ht_button: Any = None
    export_diffuse_ht_button: Any = None


@dataclass
class CifWeightControlsViewState:
    """Widget references and vars for the optional secondary-CIF weight controls."""

    frame: Any = None
    weight1_var: Any = None
    weight1_scale: Any = None
    weight2_var: Any = None
    weight2_scale: Any = None


@dataclass
class DisplayControlsState:
    """Mutable state for display-control override and callback bookkeeping."""

    background_limits_user_override: bool = False
    simulation_limits_user_override: bool = False
    scale_factor_user_override: bool = False
    suppress_background_limit_callback: bool = False
    suppress_simulation_limit_callback: bool = False
    suppress_scale_factor_callback: bool = False


@dataclass
class DisplayControlsViewState:
    """Widget references and vars for background/simulation display controls."""

    frame: Any = None
    background_controls_frame: Any = None
    simulation_controls_frame: Any = None
    background_min_var: Any = None
    background_max_var: Any = None
    background_transparency_var: Any = None
    background_min_slider: Any = None
    background_max_slider: Any = None
    background_transparency_slider: Any = None
    simulation_min_var: Any = None
    simulation_max_var: Any = None
    simulation_scale_factor_var: Any = None
    simulation_min_slider: Any = None
    simulation_max_slider: Any = None
    scale_factor_slider: Any = None
    scale_factor_entry: Any = None


@dataclass
class StructureFactorPruningControlsViewState:
    """Widget references and vars for SF-pruning / arc-integration controls."""

    frame: Any = None
    sf_prune_bias_var: Any = None
    sf_prune_bias_scale: Any = None
    sf_prune_status_var: Any = None
    sf_prune_status_label: Any = None
    solve_q_mode_row: Any = None
    solve_q_mode_var: Any = None
    solve_q_uniform_button: Any = None
    solve_q_adaptive_button: Any = None
    solve_q_steps_var: Any = None
    solve_q_steps_scale: Any = None
    solve_q_rel_tol_var: Any = None
    solve_q_rel_tol_scale: Any = None


@dataclass
class BeamMosaicParameterSlidersViewState:
    """Widget references and vars for beam/geometry/mosaic parameter sliders."""

    theta_initial_var: Any = None
    theta_initial_scale: Any = None
    cor_angle_var: Any = None
    cor_angle_scale: Any = None
    gamma_var: Any = None
    gamma_scale: Any = None
    Gamma_var: Any = None
    Gamma_scale: Any = None
    chi_var: Any = None
    chi_scale: Any = None
    psi_z_var: Any = None
    psi_z_scale: Any = None
    zs_var: Any = None
    zs_scale: Any = None
    zb_var: Any = None
    zb_scale: Any = None
    debye_x_var: Any = None
    debye_x_scale: Any = None
    debye_y_var: Any = None
    debye_y_scale: Any = None
    corto_detector_var: Any = None
    corto_detector_scale: Any = None
    a_var: Any = None
    a_scale: Any = None
    c_var: Any = None
    c_scale: Any = None
    sigma_mosaic_var: Any = None
    sigma_mosaic_scale: Any = None
    gamma_mosaic_var: Any = None
    gamma_mosaic_scale: Any = None
    eta_var: Any = None
    eta_scale: Any = None
    center_x_var: Any = None
    center_x_scale: Any = None
    bandwidth_percent_var: Any = None
    bandwidth_percent_scale: Any = None
    center_y_var: Any = None
    center_y_scale: Any = None


@dataclass
class SamplingOpticsControlsViewState:
    """Widget references and vars for sampling-resolution / optics controls."""

    resolution_selector_frame: Any = None
    resolution_var: Any = None
    resolution_menu: Any = None
    resolution_count_var: Any = None
    resolution_count_label: Any = None
    custom_samples_var: Any = None
    custom_samples_row: Any = None
    custom_samples_entry: Any = None
    custom_samples_apply_button: Any = None
    optics_mode_frame: Any = None
    optics_mode_var: Any = None
    fast_optics_button: Any = None
    exact_optics_button: Any = None


@dataclass
class FiniteStackControlsViewState:
    """Widget references and vars for finite-stack HT controls."""

    frame: Any = None
    finite_stack_var: Any = None
    finite_stack_checkbutton: Any = None
    layers_row: Any = None
    stack_layers_var: Any = None
    layers_scale: Any = None
    layers_entry_var: Any = None
    layers_entry: Any = None
    phi_l_divisor_var: Any = None
    phi_l_divisor_entry_var: Any = None
    phi_l_divisor_entry: Any = None
    phase_delta_expr_var: Any = None
    phase_delta_entry_var: Any = None
    phase_delta_entry: Any = None


@dataclass
class StackingParameterControlsViewState:
    """Widget references and vars for stacking/occupancy/atom-site controls."""

    stack_frame: Any = None
    p0_var: Any = None
    p0_scale: Any = None
    w0_var: Any = None
    w0_scale: Any = None
    p1_var: Any = None
    p1_scale: Any = None
    w1_var: Any = None
    w1_scale: Any = None
    p2_var: Any = None
    p2_scale: Any = None
    w2_var: Any = None
    w2_scale: Any = None
    occupancy_frame: Any = None
    occ_slider_frame: Any = None
    occ_entry_frame: Any = None
    occ_scale_widgets: list[Any] = field(default_factory=list)
    occ_label_widgets: list[Any] = field(default_factory=list)
    occ_entry_widgets: list[Any] = field(default_factory=list)
    occ_entry_label_widgets: list[Any] = field(default_factory=list)
    atom_site_frame: Any = None
    atom_site_table_frame: Any = None
    atom_site_coord_entry_widgets: list[Any] = field(default_factory=list)


@dataclass
class GeometryToolActionsViewState:
    """Widget references and vars for fit-history/manual-geometry action controls."""

    undo_geometry_fit_button: Any = None
    redo_geometry_fit_button: Any = None
    geometry_manual_pick_button_var: Any = None
    geometry_manual_pick_button: Any = None
    geometry_manual_undo_button: Any = None
    geometry_manual_export_button: Any = None
    geometry_manual_import_button: Any = None
    geometry_preview_exclude_button_var: Any = None
    geometry_preview_exclude_button: Any = None
    clear_geometry_preview_exclusions_button: Any = None


@dataclass
class HklLookupViewState:
    """Widget references and vars for the HKL lookup / peak-selection controls."""

    frame: Any = None
    selected_h_var: Any = None
    selected_k_var: Any = None
    selected_l_var: Any = None
    h_entry: Any = None
    k_entry: Any = None
    l_entry: Any = None
    select_button: Any = None
    hkl_pick_button_var: Any = None
    hkl_pick_button: Any = None
    clear_button: Any = None
    show_bragg_ewald_button: Any = None
    bragg_qr_groups_button: Any = None


@dataclass
class GeometryOverlayActionsViewState:
    """Widget references and vars for overlay/mosaic action controls."""

    show_qr_cylinder_overlay_var: Any = None
    show_qr_cylinder_overlay_checkbutton: Any = None
    clear_geometry_markers_button: Any = None
    fit_button_mosaic: Any = None


@dataclass
class AnalysisViewControlsViewState:
    """Widget references and vars for 1D/caked/log analysis view controls."""

    show_1d_var: Any = None
    check_1d: Any = None
    show_caked_2d_var: Any = None
    check_2d: Any = None
    log_radial_var: Any = None
    check_log_radial: Any = None
    log_azimuth_var: Any = None
    check_log_azimuth: Any = None


@dataclass
class AnalysisExportControlsViewState:
    """Widget references for analysis export controls."""

    snapshot_button: Any = None
    save_q_button: Any = None
    save_1d_grid_button: Any = None


@dataclass
class GeometryPreviewOverlayState:
    """Cached live-preview overlay data and summary metrics."""

    signature: object = None
    pairs: list[dict[str, object]] = field(default_factory=list)
    simulated_count: int = 0
    min_matches: int = 0
    best_radius: float = float("nan")
    mean_dist: float = float("nan")
    p90_dist: float = float("nan")
    quality_fail: bool = False
    max_display_markers: int = 120
    auto_match_attempts: list[dict[str, object]] = field(default_factory=list)
    q_group_total: int = 0
    q_group_excluded: int = 0
    excluded_q_peaks: int = 0
    collapsed_degenerate_peaks: int = 0


@dataclass
class GeometryPreviewState:
    """Mutable state for live geometry preview filters and caches."""

    excluded_keys: set[tuple[object, ...]] = field(default_factory=set)
    excluded_q_groups: set[tuple[object, ...]] = field(default_factory=set)
    exclude_armed: bool = False
    overlay: GeometryPreviewOverlayState = field(
        default_factory=GeometryPreviewOverlayState
    )
    skip_once: bool = False
    auto_match_background_cache_key: object = None
    auto_match_background_cache_data: dict[str, object] | None = None


@dataclass
class GeometryQGroupState:
    """Mutable state for the geometry-fit Qr/Qz selector workflow."""

    row_vars: dict[tuple[object, ...], Any] = field(default_factory=dict)
    cached_entries: list[dict[str, object]] = field(default_factory=list)
    refresh_requested: bool = False


@dataclass
class GeometryQGroupViewState:
    """Widget references for the Qr/Qz selector window."""

    window: Any = None
    canvas: Any = None
    body: Any = None
    status_label: Any = None


@dataclass
class BraggQrManagerState:
    """Selection/index bookkeeping for the Bragg Qr manager window."""

    qr_index_keys: list[tuple[str, int]] = field(default_factory=list)
    l_index_keys: list[int] = field(default_factory=list)
    selected_group_key: tuple[str, int] | None = None


@dataclass
class BraggQrManagerViewState:
    """Widget references for the Bragg Qr manager window."""

    window: Any = None
    qr_listbox: Any = None
    qr_status_label: Any = None
    l_listbox: Any = None
    l_status_label: Any = None


@dataclass
class HbnGeometryDebugViewState:
    """Widget references for the hBN geometry debug viewer window."""

    window: Any = None
    text_widget: Any = None


@dataclass
class AppState:
    """Minimal mutable state container for GUI controller/view coordination."""

    instrument_config: dict[str, Any] = field(default_factory=dict)
    file_paths: dict[str, Any] = field(default_factory=dict)
    image_size: int = 3000
    background_images_native: list[np.ndarray] = field(default_factory=list)
    background_images_display: list[np.ndarray] = field(default_factory=list)
    current_background_index: int = 0
    manual_geometry: ManualGeometryState = field(default_factory=ManualGeometryState)
    geometry_fit_history: GeometryFitHistoryState = field(
        default_factory=GeometryFitHistoryState
    )
    geometry_fit_parameter_controls_view: GeometryFitParameterControlsViewState = field(
        default_factory=GeometryFitParameterControlsViewState
    )
    geometry_fit_constraints_view: GeometryFitConstraintsViewState = field(
        default_factory=GeometryFitConstraintsViewState
    )
    background_theta_controls_view: BackgroundThetaControlsViewState = field(
        default_factory=BackgroundThetaControlsViewState
    )
    workspace_panels_view: WorkspacePanelsViewState = field(
        default_factory=WorkspacePanelsViewState
    )
    background_backend_debug_view: BackgroundBackendDebugViewState = field(
        default_factory=BackgroundBackendDebugViewState
    )
    primary_cif_controls_view: PrimaryCifControlsViewState = field(
        default_factory=PrimaryCifControlsViewState
    )
    cif_weight_controls_view: CifWeightControlsViewState = field(
        default_factory=CifWeightControlsViewState
    )
    display_controls_state: DisplayControlsState = field(
        default_factory=DisplayControlsState
    )
    display_controls_view: DisplayControlsViewState = field(
        default_factory=DisplayControlsViewState
    )
    structure_factor_pruning_controls_view: StructureFactorPruningControlsViewState = field(
        default_factory=StructureFactorPruningControlsViewState
    )
    beam_mosaic_parameter_sliders_view: BeamMosaicParameterSlidersViewState = field(
        default_factory=BeamMosaicParameterSlidersViewState
    )
    sampling_optics_controls_view: SamplingOpticsControlsViewState = field(
        default_factory=SamplingOpticsControlsViewState
    )
    finite_stack_controls_view: FiniteStackControlsViewState = field(
        default_factory=FiniteStackControlsViewState
    )
    stacking_parameter_controls_view: StackingParameterControlsViewState = field(
        default_factory=StackingParameterControlsViewState
    )
    geometry_tool_actions_view: GeometryToolActionsViewState = field(
        default_factory=GeometryToolActionsViewState
    )
    hkl_lookup_view: HklLookupViewState = field(
        default_factory=HklLookupViewState
    )
    geometry_overlay_actions_view: GeometryOverlayActionsViewState = field(
        default_factory=GeometryOverlayActionsViewState
    )
    analysis_view_controls_view: AnalysisViewControlsViewState = field(
        default_factory=AnalysisViewControlsViewState
    )
    analysis_export_controls_view: AnalysisExportControlsViewState = field(
        default_factory=AnalysisExportControlsViewState
    )
    geometry_preview: GeometryPreviewState = field(default_factory=GeometryPreviewState)
    geometry_q_groups: GeometryQGroupState = field(default_factory=GeometryQGroupState)
    geometry_q_group_view: GeometryQGroupViewState = field(
        default_factory=GeometryQGroupViewState
    )
    bragg_qr_manager: BraggQrManagerState = field(default_factory=BraggQrManagerState)
    bragg_qr_manager_view: BraggQrManagerViewState = field(
        default_factory=BraggQrManagerViewState
    )
    hbn_geometry_debug_view: HbnGeometryDebugViewState = field(
        default_factory=HbnGeometryDebugViewState
    )
