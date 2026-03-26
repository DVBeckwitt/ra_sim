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
    sampling_optics_controls_view: SamplingOpticsControlsViewState = field(
        default_factory=SamplingOpticsControlsViewState
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
