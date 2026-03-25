"""GUI controller helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ra_sim.path_config import get_instrument_config

from .state import AppState, ManualGeometryState, ManualGeometryUndoSnapshot


def build_initial_state() -> AppState:
    """Build the initial GUI state snapshot from current configuration."""

    instrument_cfg = get_instrument_config()
    detector_cfg = instrument_cfg.get("instrument", {}).get("detector", {})
    image_size = int(detector_cfg.get("image_size", 3000))
    return AppState(
        instrument_config=instrument_cfg,
        image_size=image_size,
    )


def launch_gui(*, write_excel_flag: bool | None = None) -> Any:
    """Launch the full GUI application."""

    from . import app

    return app.main(write_excel_flag=write_excel_flag)


def replace_manual_geometry_pairs_by_background(
    state: ManualGeometryState,
    pairs_by_background: Mapping[object, Sequence[dict[str, object]]] | None,
) -> dict[int, list[dict[str, object]]]:
    """Replace the stored manual-geometry pair map in place."""

    normalized: dict[int, list[dict[str, object]]] = {}
    for raw_index, raw_entries in (pairs_by_background or {}).items():
        try:
            index = int(raw_index)
        except Exception:
            continue
        entries = [
            copy.deepcopy(dict(entry))
            for entry in raw_entries or ()
            if isinstance(entry, dict)
        ]
        if entries:
            normalized[index] = entries

    state.pairs_by_background.clear()
    state.pairs_by_background.update(normalized)
    return state.pairs_by_background


def replace_manual_geometry_pick_session(
    state: ManualGeometryState,
    pick_session: Mapping[str, object] | None,
) -> dict[str, object]:
    """Replace the active manual-geometry pick session in place."""

    normalized = (
        copy.deepcopy(dict(pick_session))
        if isinstance(pick_session, Mapping)
        else {}
    )
    state.pick_session.clear()
    state.pick_session.update(normalized)
    return state.pick_session


def clear_manual_geometry_undo_stack(state: ManualGeometryState) -> None:
    """Discard all saved manual-geometry undo history."""

    state.undo_stack.clear()


def build_manual_geometry_undo_snapshot(
    state: ManualGeometryState,
) -> ManualGeometryUndoSnapshot:
    """Return a deep copy of manual-geometry state for undo restoration."""

    return ManualGeometryUndoSnapshot(
        pairs_by_background=copy.deepcopy(dict(state.pairs_by_background)),
        pick_session=copy.deepcopy(dict(state.pick_session)),
    )


def push_manual_geometry_undo_state(
    state: ManualGeometryState,
    *,
    limit: int,
) -> None:
    """Push the current manual-geometry state onto the undo stack."""

    state.undo_stack.append(build_manual_geometry_undo_snapshot(state))
    max_items = max(1, int(limit))
    if len(state.undo_stack) > max_items:
        del state.undo_stack[:-max_items]


def restore_last_manual_geometry_undo_state(
    state: ManualGeometryState,
) -> ManualGeometryUndoSnapshot | None:
    """Restore the most recent manual-geometry undo snapshot in place."""

    if not state.undo_stack:
        return None

    snapshot = state.undo_stack.pop()
    replace_manual_geometry_pairs_by_background(
        state,
        snapshot.pairs_by_background,
    )
    replace_manual_geometry_pick_session(
        state,
        snapshot.pick_session,
    )
    return snapshot
