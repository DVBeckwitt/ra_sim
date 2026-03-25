"""GUI controller helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ra_sim.path_config import get_instrument_config

from .state import (
    AppState,
    GeometryFitHistoryState,
    GeometryQGroupState,
    ManualGeometryState,
    ManualGeometryUndoSnapshot,
)


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


def clear_geometry_fit_history(state: GeometryFitHistoryState) -> None:
    """Discard all geometry-fit undo/redo history and overlay state."""

    state.undo_stack.clear()
    state.redo_stack.clear()
    state.last_overlay_state = None


def replace_geometry_fit_last_overlay_state(
    state: GeometryFitHistoryState,
    overlay_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Replace the remembered geometry-fit overlay state."""

    if overlay_state is None:
        state.last_overlay_state = None
    else:
        state.last_overlay_state = copy_state_value(dict(overlay_state))
    return state.last_overlay_state


def push_geometry_fit_undo_state(
    state: GeometryFitHistoryState,
    fit_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Push one state onto the geometry-fit undo stack and clear redo history."""

    if not isinstance(fit_state, Mapping):
        return
    state.undo_stack.append(copy_state_value(dict(fit_state)))
    max_items = max(1, int(limit))
    if len(state.undo_stack) > max_items:
        del state.undo_stack[:-max_items]
    state.redo_stack.clear()


def push_geometry_fit_redo_state(
    state: GeometryFitHistoryState,
    fit_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Push one state onto the geometry-fit redo stack."""

    if not isinstance(fit_state, Mapping):
        return
    state.redo_stack.append(copy_state_value(dict(fit_state)))
    max_items = max(1, int(limit))
    if len(state.redo_stack) > max_items:
        del state.redo_stack[:-max_items]


def peek_last_geometry_fit_undo_state(
    state: GeometryFitHistoryState,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Return a detached copy of the most recent geometry-fit undo state."""

    if not state.undo_stack:
        return None
    return copy_state_value(state.undo_stack[-1])


def peek_last_geometry_fit_redo_state(
    state: GeometryFitHistoryState,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Return a detached copy of the most recent geometry-fit redo state."""

    if not state.redo_stack:
        return None
    return copy_state_value(state.redo_stack[-1])


def commit_geometry_fit_undo(
    state: GeometryFitHistoryState,
    current_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Finalize a successful geometry-fit undo transition."""

    if state.undo_stack:
        state.undo_stack.pop()
    push_geometry_fit_redo_state(
        state,
        current_state,
        copy_state_value=copy_state_value,
        limit=limit,
    )


def commit_geometry_fit_redo(
    state: GeometryFitHistoryState,
    current_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Finalize a successful geometry-fit redo transition."""

    if state.redo_stack:
        state.redo_stack.pop()
    if isinstance(current_state, Mapping):
        state.undo_stack.append(copy_state_value(dict(current_state)))
        max_items = max(1, int(limit))
        if len(state.undo_stack) > max_items:
            del state.undo_stack[:-max_items]


def clone_geometry_q_group_entries(
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Return detached copies of Qr/Qz selector entries."""

    cloned: list[dict[str, object]] = []
    for raw_entry in entries or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        entry["hkl_preview"] = list(raw_entry.get("hkl_preview", []))
        cloned.append(entry)
    return cloned


def listed_geometry_q_group_entries(
    state: GeometryQGroupState,
) -> list[dict[str, object]]:
    """Return detached copies of the stored Qr/Qz selector entries."""

    return clone_geometry_q_group_entries(state.cached_entries)


def listed_geometry_q_group_keys(
    state: GeometryQGroupState,
    entries: Sequence[dict[str, object]] | None = None,
) -> set[tuple[object, ...]]:
    """Return the stable keys for the stored Qr/Qz selector entries."""

    keys: set[tuple[object, ...]] = set()
    source_entries = entries if entries is not None else state.cached_entries
    for entry in source_entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if key is not None:
            keys.add(key)
    return keys


def replace_geometry_q_group_cached_entries(
    state: GeometryQGroupState,
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Replace the stored Qr/Qz selector entry snapshot in place."""

    cloned = clone_geometry_q_group_entries(entries)
    state.cached_entries.clear()
    state.cached_entries.extend(cloned)
    return listed_geometry_q_group_entries(state)


def clear_geometry_q_group_row_vars(state: GeometryQGroupState) -> None:
    """Discard the current Qr/Qz selector row-var map."""

    state.row_vars.clear()


def set_geometry_q_group_row_var(
    state: GeometryQGroupState,
    group_key: tuple[object, ...] | None,
    row_var: Any,
) -> None:
    """Store one Qr/Qz selector row-var binding."""

    if group_key is None:
        return
    state.row_vars[group_key] = row_var


def request_geometry_q_group_refresh(state: GeometryQGroupState) -> None:
    """Mark the Qr/Qz selector listing for refresh on the next update."""

    state.refresh_requested = True


def consume_geometry_q_group_refresh_request(
    state: GeometryQGroupState,
) -> bool:
    """Return and clear the pending Qr/Qz selector refresh flag."""

    requested = bool(state.refresh_requested)
    state.refresh_requested = False
    return requested
