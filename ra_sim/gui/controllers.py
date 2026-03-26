"""GUI controller helpers."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ra_sim.path_config import get_instrument_config
from ra_sim.utils.stacking_fault import (
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)

from .state import (
    AppState,
    BraggQrManagerState,
    GeometryFitHistoryState,
    GeometryPreviewOverlayState,
    GeometryPreviewState,
    GeometryQGroupState,
    ManualGeometryState,
    ManualGeometryUndoSnapshot,
)


def parse_sampling_count(
    raw_value: object,
    fallback: object,
    *,
    minimum: int = 1,
) -> int:
    """Coerce one sampling-count value to a positive integer."""

    try:
        parsed = int(round(float(str(raw_value).strip().replace(",", ""))))
    except (TypeError, ValueError):
        try:
            parsed = int(round(float(str(fallback).strip().replace(",", ""))))
        except (TypeError, ValueError):
            parsed = int(minimum)
    return max(int(minimum), parsed)


def normalize_sampling_resolution_choice(
    resolution_value: object,
    *,
    allowed_options: Sequence[object],
    fallback: object,
) -> str:
    """Normalize one sampling-resolution choice against the allowed labels."""

    allowed = [str(option) for option in allowed_options]
    value_text = str(resolution_value) if resolution_value is not None else ""
    if value_text in allowed:
        return value_text

    fallback_text = str(fallback)
    if fallback_text in allowed:
        return fallback_text
    if allowed:
        return allowed[0]
    return ""


def resolve_sampling_count(
    resolution_value: object,
    *,
    custom_option: object,
    custom_value: object,
    preset_counts: Mapping[object, object],
    fallback_resolution: object,
    fallback_count: object,
) -> int:
    """Return the effective sample count for one resolution selection."""

    normalized = str(resolution_value) if resolution_value is not None else ""
    custom_label = str(custom_option)
    if normalized == custom_label:
        return parse_sampling_count(custom_value, fallback_count)

    fallback_label = str(fallback_resolution)
    preset_value = preset_counts.get(normalized)
    if preset_value is None:
        preset_value = preset_counts.get(fallback_label, fallback_count)
    return parse_sampling_count(preset_value, fallback_count)


def format_sampling_resolution_summary(
    resolution_value: object,
    *,
    custom_option: object,
    custom_value: object,
    preset_counts: Mapping[object, object],
    fallback_resolution: object,
    fallback_count: object,
) -> str:
    """Format the GUI sampling summary label for the current selection."""

    custom_label = str(custom_option)
    normalized = str(resolution_value) if resolution_value is not None else ""
    count = resolve_sampling_count(
        normalized,
        custom_option=custom_label,
        custom_value=custom_value,
        preset_counts=preset_counts,
        fallback_resolution=fallback_resolution,
        fallback_count=fallback_count,
    )
    suffix = " (custom)" if normalized == custom_label else ""
    return f"{count:,} samples{suffix}" if count >= 1000 else f"{count} samples{suffix}"


def normalize_finite_stack_layer_count(
    raw_value: object,
    fallback: object,
) -> int:
    """Normalize one finite-stack layer-count input to a positive integer."""

    return parse_sampling_count(raw_value, fallback)


def format_finite_stack_layer_count(value: object) -> str:
    """Format one finite-stack layer count for entry display."""

    return str(normalize_finite_stack_layer_count(value, 1))


def normalize_finite_stack_phase_delta_expression(
    raw_value: object,
    *,
    fallback: object,
) -> str:
    """Normalize and validate one finite-stack phase-delta expression."""

    normalized = normalize_phase_delta_expression(raw_value, fallback=str(fallback))
    return validate_phase_delta_expression(normalized)


def normalize_finite_stack_phi_l_divisor(
    raw_value: object,
    *,
    fallback: object,
) -> float:
    """Normalize one finite-stack phi-L divisor to a positive finite float."""

    return normalize_phi_l_divisor(raw_value, fallback=float(fallback))


def format_finite_stack_phi_l_divisor(value: object) -> str:
    """Format one finite-stack phi-L divisor for entry display."""

    normalized = normalize_finite_stack_phi_l_divisor(value, fallback=1.0)
    return f"{normalized:.6g}"


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


def replace_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    excluded_q_groups: Sequence[object] | None,
) -> set[tuple[object, ...]]:
    """Replace the excluded Qr/Qz group key set in place."""

    normalized: set[tuple[object, ...]] = set()
    for raw_key in excluded_q_groups or ():
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))

    state.excluded_q_groups.clear()
    state.excluded_q_groups.update(normalized)
    return state.excluded_q_groups


def retain_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    allowed_keys: Sequence[object] | set[tuple[object, ...]] | None,
) -> set[tuple[object, ...]]:
    """Keep only excluded Qr/Qz keys that still exist in the listed snapshot."""

    normalized: set[tuple[object, ...]] = set()
    for raw_key in allowed_keys or ():
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))

    if normalized:
        state.excluded_q_groups.intersection_update(normalized)
    else:
        state.excluded_q_groups.clear()
    return state.excluded_q_groups


def clear_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
) -> None:
    """Clear all excluded Qr/Qz group keys."""

    state.excluded_q_groups.clear()


def clear_geometry_preview_excluded_keys(
    state: GeometryPreviewState,
) -> None:
    """Clear all excluded live-preview pair keys."""

    state.excluded_keys.clear()


def set_geometry_preview_match_included(
    state: GeometryPreviewState,
    match_key: tuple[object, ...] | None,
    *,
    included: bool,
) -> bool:
    """Toggle one live-preview match in the exclusion set."""

    if match_key is None:
        return False
    if included:
        state.excluded_keys.discard(match_key)
    else:
        state.excluded_keys.add(match_key)
    return True


def set_geometry_preview_exclude_mode(
    state: GeometryPreviewState,
    enabled: bool,
) -> bool:
    """Arm or disarm live-preview exclusion editing."""

    state.exclude_armed = bool(enabled)
    return state.exclude_armed


def _preview_int(value: object, default: int = 0, *, minimum: int | None = None) -> int:
    """Coerce one preview-state value to int with an optional lower bound."""

    try:
        numeric = int(value)
    except Exception:
        numeric = int(default)
    if minimum is not None:
        numeric = max(int(minimum), numeric)
    return numeric


def _preview_float(value: object, default: float) -> float:
    """Coerce one preview-state value to float."""

    try:
        return float(value)
    except Exception:
        return float(default)


def replace_geometry_preview_overlay_state(
    state: GeometryPreviewState,
    overlay_state: Mapping[str, object] | None,
) -> GeometryPreviewOverlayState:
    """Replace cached live-preview overlay data and summary metrics in place."""

    overlay = state.overlay
    source = overlay_state if isinstance(overlay_state, Mapping) else {}

    overlay.signature = source.get("signature")
    overlay.pairs.clear()
    overlay.pairs.extend(
        copy.deepcopy(dict(entry))
        for entry in source.get("pairs", [])
        if isinstance(entry, Mapping)
    )
    overlay.simulated_count = _preview_int(source.get("simulated_count"), 0)
    overlay.min_matches = _preview_int(source.get("min_matches"), 0)
    overlay.best_radius = _preview_float(source.get("best_radius"), float("nan"))
    overlay.mean_dist = _preview_float(source.get("mean_dist"), float("nan"))
    overlay.p90_dist = _preview_float(source.get("p90_dist"), float("nan"))
    overlay.quality_fail = bool(source.get("quality_fail", False))
    overlay.max_display_markers = _preview_int(
        source.get("max_display_markers"),
        120,
        minimum=1,
    )
    overlay.auto_match_attempts.clear()
    overlay.auto_match_attempts.extend(
        copy.deepcopy(dict(entry))
        for entry in source.get("auto_match_attempts", [])
        if isinstance(entry, Mapping)
    )
    overlay.q_group_total = _preview_int(source.get("q_group_total"), 0)
    overlay.q_group_excluded = _preview_int(source.get("q_group_excluded"), 0)
    overlay.excluded_q_peaks = _preview_int(source.get("excluded_q_peaks"), 0)
    overlay.collapsed_degenerate_peaks = _preview_int(
        source.get("collapsed_degenerate_peaks"),
        0,
    )
    return overlay


def set_geometry_preview_q_group_included(
    state: GeometryPreviewState,
    group_key: tuple[object, ...] | None,
    *,
    included: bool,
) -> bool:
    """Toggle one Qr/Qz group in the preview exclusion set."""

    if group_key is None:
        return False
    if included:
        state.excluded_q_groups.discard(group_key)
    else:
        state.excluded_q_groups.add(group_key)
    return True


def count_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    keys: Sequence[object] | None = None,
) -> int:
    """Count excluded Qr/Qz keys, optionally restricted to one listing."""

    if keys is None:
        return int(len(state.excluded_q_groups))

    normalized: set[tuple[object, ...]] = set()
    for raw_key in keys:
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))
    return int(sum(1 for key in normalized if key in state.excluded_q_groups))


def request_geometry_preview_skip_once(state: GeometryPreviewState) -> None:
    """Skip one live-preview refresh on the next update cycle."""

    state.skip_once = True


def consume_geometry_preview_skip_once(
    state: GeometryPreviewState,
) -> bool:
    """Return and clear the one-shot live-preview skip flag."""

    requested = bool(state.skip_once)
    state.skip_once = False
    return requested


def clear_geometry_preview_skip_once(state: GeometryPreviewState) -> None:
    """Clear the one-shot live-preview skip flag."""

    state.skip_once = False


def get_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
    cache_key: object,
) -> dict[str, object] | None:
    """Return the cached auto-match background context when the key matches."""

    if state.auto_match_background_cache_key != cache_key:
        return None
    if not isinstance(state.auto_match_background_cache_data, dict):
        return None
    return state.auto_match_background_cache_data


def replace_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
    cache_key: object,
    cache_data: dict[str, object] | None,
) -> dict[str, object] | None:
    """Replace the cached auto-match background context."""

    state.auto_match_background_cache_key = cache_key
    state.auto_match_background_cache_data = cache_data
    return state.auto_match_background_cache_data


def clear_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
) -> None:
    """Discard the cached auto-match background context."""

    state.auto_match_background_cache_key = None
    state.auto_match_background_cache_data = None


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


def clear_bragg_qr_manager_state(state: BraggQrManagerState) -> None:
    """Discard all Bragg-Qr manager list bookkeeping."""

    state.qr_index_keys.clear()
    state.l_index_keys.clear()
    state.selected_group_key = None


def replace_bragg_qr_index_keys(
    state: BraggQrManagerState,
    keys: Sequence[object] | None,
) -> list[tuple[str, int]]:
    """Replace the stored Bragg-Qr group index-key list in place."""

    normalized: list[tuple[str, int]] = []
    for raw_key in keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            normalized.append((str(raw_key[0]), int(raw_key[1])))
        except Exception:
            continue
    state.qr_index_keys.clear()
    state.qr_index_keys.extend(normalized)
    return list(state.qr_index_keys)


def replace_bragg_qr_l_index_keys(
    state: BraggQrManagerState,
    keys: Sequence[object] | None,
) -> list[int]:
    """Replace the stored Bragg-Qr L-index list in place."""

    normalized: list[int] = []
    for raw_key in keys or ():
        try:
            normalized.append(int(raw_key))
        except Exception:
            continue
    state.l_index_keys.clear()
    state.l_index_keys.extend(normalized)
    return list(state.l_index_keys)


def set_bragg_qr_selected_group_key(
    state: BraggQrManagerState,
    group_key: tuple[str, int] | Sequence[object] | None,
) -> tuple[str, int] | None:
    """Store the currently selected Bragg-Qr group key."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        state.selected_group_key = None
    else:
        try:
            state.selected_group_key = (str(group_key[0]), int(group_key[1]))
        except Exception:
            state.selected_group_key = None
    return state.selected_group_key


def selected_bragg_qr_keys(
    state: BraggQrManagerState,
    selected_indices: Sequence[object] | None,
) -> list[tuple[str, int]]:
    """Map selected listbox indices to Bragg-Qr group keys."""

    selected_keys: list[tuple[str, int]] = []
    for raw_idx in selected_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < len(state.qr_index_keys):
            selected_keys.append(state.qr_index_keys[idx])
    return selected_keys


def selected_bragg_qr_l_keys(
    state: BraggQrManagerState,
    selected_indices: Sequence[object] | None,
) -> list[int]:
    """Map selected listbox indices to Bragg-Qr L keys."""

    selected_keys: list[int] = []
    for raw_idx in selected_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < len(state.l_index_keys):
            selected_keys.append(int(state.l_index_keys[idx]))
    return selected_keys


def set_bragg_qr_groups_enabled(
    disabled_groups: set[tuple[str, int]],
    group_keys: Sequence[tuple[str, int]] | None,
    *,
    enabled: bool,
) -> int:
    """Enable or disable the provided normalized Bragg-Qr groups in place."""

    changed = 0
    for raw_key in group_keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            key = (str(raw_key[0]), int(raw_key[1]))
        except Exception:
            continue
        if enabled:
            if key in disabled_groups:
                disabled_groups.remove(key)
                changed += 1
        else:
            if key not in disabled_groups:
                disabled_groups.add(key)
                changed += 1
    return changed


def toggle_bragg_qr_groups(
    disabled_groups: set[tuple[str, int]],
    group_keys: Sequence[tuple[str, int]] | None,
) -> int:
    """Toggle the provided normalized Bragg-Qr groups in place."""

    changed = 0
    for raw_key in group_keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            key = (str(raw_key[0]), int(raw_key[1]))
        except Exception:
            continue
        if key in disabled_groups:
            disabled_groups.remove(key)
        else:
            disabled_groups.add(key)
        changed += 1
    return changed


def set_bragg_qr_l_values_enabled(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
    l_keys: Sequence[object] | None,
    *,
    enabled: bool,
    invalid_key: int,
) -> int:
    """Enable or disable the provided normalized Bragg-Qr L values in place."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return 0
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return 0

    changed = 0
    for raw_l_key in l_keys or ():
        try:
            l_key = int(raw_l_key)
        except Exception:
            continue
        if l_key == int(invalid_key):
            continue
        key = (source_label, m_idx, l_key)
        if enabled:
            if key in disabled_l_values:
                disabled_l_values.remove(key)
                changed += 1
        else:
            if key not in disabled_l_values:
                disabled_l_values.add(key)
                changed += 1
    return changed


def toggle_bragg_qr_l_values(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
    l_keys: Sequence[object] | None,
    *,
    invalid_key: int,
) -> int:
    """Toggle the provided normalized Bragg-Qr L values in place."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return 0
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return 0

    changed = 0
    for raw_l_key in l_keys or ():
        try:
            l_key = int(raw_l_key)
        except Exception:
            continue
        if l_key == int(invalid_key):
            continue
        key = (source_label, m_idx, l_key)
        if key in disabled_l_values:
            disabled_l_values.remove(key)
        else:
            disabled_l_values.add(key)
        changed += 1
    return changed


def clear_bragg_qr_l_values_for_group(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
) -> bool:
    """Enable every L value for one normalized Bragg-Qr group."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return False
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return False

    filtered = {
        (src, mm, lk)
        for src, mm, lk in disabled_l_values
        if not (str(src) == source_label and int(mm) == m_idx)
    }
    if len(filtered) == len(disabled_l_values):
        return False
    disabled_l_values.clear()
    disabled_l_values.update(filtered)
    return True
