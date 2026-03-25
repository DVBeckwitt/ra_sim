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
