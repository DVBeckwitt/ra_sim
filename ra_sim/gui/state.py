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
class GeometryQGroupState:
    """Mutable state for the geometry-fit Qr/Qz selector workflow."""

    row_vars: dict[tuple[object, ...], Any] = field(default_factory=dict)
    cached_entries: list[dict[str, object]] = field(default_factory=list)
    refresh_requested: bool = False


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
    geometry_q_groups: GeometryQGroupState = field(default_factory=GeometryQGroupState)
