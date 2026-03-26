"""Workflow helpers for the geometry Q-group selector window."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import controllers as gui_controllers
from . import manual_geometry as gui_manual_geometry
from . import views as gui_views


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _row_var_value(row_var: object) -> bool:
    if row_var is None:
        return False
    try:
        return bool(row_var.get())
    except Exception:
        return bool(row_var)


def format_geometry_q_group_line(entry: Mapping[str, object]) -> str:
    """Return a compact label for one Qr/Qz selector row."""

    qr_val = _coerce_float(entry.get("qr", np.nan), float("nan"))
    qz_val = _coerce_float(entry.get("qz", np.nan), float("nan"))
    total_intensity = _coerce_float(entry.get("total_intensity", 0.0), 0.0)
    peak_count = _coerce_int(entry.get("peak_count", 0), 0)
    source_label = str(entry.get("source_label", ""))
    gz_index = entry.get("gz_index")
    if gz_index is None:
        key = entry.get("key")
        if isinstance(key, tuple) and len(key) >= 4:
            gz_index = key[3]
    hkl_items = entry.get("hkl_preview", [])
    if isinstance(hkl_items, np.ndarray):
        hkl_items = list(hkl_items)
    elif not isinstance(hkl_items, Sequence) or isinstance(hkl_items, str | bytes):
        hkl_items = []
    else:
        hkl_items = list(hkl_items)
    hkl_preview = ", ".join(str(hkl) for hkl in hkl_items[:3])
    if len(hkl_items) > 3:
        hkl_preview += ", ..."
    gz_text = f"{int(gz_index):4d}" if gz_index is not None else " n/a"
    return (
        f"{source_label:<9}  "
        f"Qr={qr_val:8.4f}  "
        f"Gz={gz_text}  "
        f"Qz={qz_val:8.4f}  "
        f"I={total_intensity:10.3f}  "
        f"hits={peak_count:4d}"
        + (f"  HKL={hkl_preview}" if hkl_preview else "")
    )


def current_geometry_auto_match_min_matches(
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
) -> int:
    """Return the current geometry auto-match minimum peak count."""

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(
        fit_config,
        Mapping,
    ) else {}
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, Mapping):
        auto_match_cfg = {}
    default_min_matches = max(6, len(list(current_geometry_fit_var_names or ())) + 2)
    try:
        min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    except Exception:
        min_matches = int(default_min_matches)
    return max(1, int(min_matches))


def geometry_q_group_excluded_count(
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> int:
    """Count excluded Qr/Qz rows, optionally scoped to one entry list."""

    keys = gui_controllers.listed_geometry_q_group_keys(q_group_state, entries)
    if not keys:
        return gui_controllers.count_geometry_preview_excluded_q_groups(preview_state)
    return gui_controllers.count_geometry_preview_excluded_q_groups(
        preview_state,
        keys,
    )


def build_geometry_q_group_window_status_text(
    *,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Build the summary text shown above the Qr/Qz selector rows."""

    rows = list(entries) if entries is not None else gui_controllers.listed_geometry_q_group_entries(
        q_group_state
    )
    excluded_q_groups = getattr(preview_state, "excluded_q_groups", set())
    total_count = len(rows)
    included_rows = [
        entry
        for entry in rows
        if entry.get("key") not in excluded_q_groups
    ]
    selected_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in included_rows)
    )
    total_peak_count = int(
        sum(_coerce_int(entry.get("peak_count", 0), 0) for entry in rows)
    )
    min_matches = current_geometry_auto_match_min_matches(
        fit_config,
        current_geometry_fit_var_names,
    )
    shortfall = max(0, int(min_matches - selected_peak_count))
    selected_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in included_rows)
    )
    total_intensity = float(
        sum(_coerce_float(entry.get("total_intensity", 0.0), 0.0) for entry in rows)
    )
    return (
        f"Included Qr/Qz groups: {len(included_rows)}/{total_count}  "
        f"Selected peaks: {selected_peak_count}/{total_peak_count}  "
        f"Need >= {min_matches}"
        + (f"  short {shortfall}" if shortfall > 0 else "  ready")
        + "\n"
        + f"Intensity={selected_intensity:.3f}/{total_intensity:.3f}  "
        + 'Listed peaks stay fixed until you press "Update Listed Peaks".'
    )


def update_geometry_q_group_window_status(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    entries: Sequence[dict[str, object]] | None = None,
) -> None:
    """Refresh the summary label shown at the top of the selector window."""

    gui_views.set_geometry_q_group_status_text(
        view_state,
        build_geometry_q_group_window_status_text(
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
    )


def refresh_geometry_q_group_window(
    *,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
) -> bool:
    """Redraw the Qr/Qz selector window from the stored manual snapshot."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    return gui_views.refresh_geometry_q_group_window(
        view_state=view_state,
        entries=entries,
        excluded_q_groups=getattr(preview_state, "excluded_q_groups", set()),
        status_text=build_geometry_q_group_window_status_text(
            preview_state=preview_state,
            q_group_state=q_group_state,
            fit_config=fit_config,
            current_geometry_fit_var_names=current_geometry_fit_var_names,
            entries=entries,
        ),
        format_line=format_geometry_q_group_line,
        on_toggle=on_toggle,
        clear_row_vars=lambda: gui_controllers.clear_geometry_q_group_row_vars(
            q_group_state,
        ),
        register_row_var=lambda group_key, row_var: gui_controllers.set_geometry_q_group_row_var(
            q_group_state,
            group_key,
            row_var,
        ),
    )


def apply_geometry_q_group_checkbox_change(
    preview_state,
    group_key: tuple[object, ...] | None,
    row_var: object,
) -> str | None:
    """Apply one Qr/Qz include/exclude toggle from the selector window."""

    if group_key is None:
        return None
    included = _row_var_value(row_var)
    gui_controllers.set_geometry_preview_q_group_included(
        preview_state,
        group_key,
        included=included,
    )
    return "Included" if included else "Excluded"


def set_all_geometry_q_groups_enabled(
    preview_state,
    q_group_state,
    *,
    enabled: bool,
) -> tuple[str, int]:
    """Enable or disable every currently listed Qr/Qz group."""

    entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    if enabled:
        gui_controllers.clear_geometry_preview_excluded_q_groups(preview_state)
        action = "Included"
    else:
        gui_controllers.replace_geometry_preview_excluded_q_groups(
            preview_state,
            [
                entry["key"]
                for entry in entries
                if entry.get("key") is not None
            ],
        )
        action = "Excluded"
    return action, len(entries)


def request_geometry_q_group_window_update(q_group_state) -> None:
    """Mark the Qr/Qz selector listing for refresh on the next update."""

    gui_controllers.request_geometry_q_group_refresh(q_group_state)


def close_geometry_q_group_window(view_state, q_group_state) -> None:
    """Destroy the Qr/Qz selector window and clear its row-var map."""

    gui_views.close_geometry_q_group_window(view_state)
    gui_controllers.clear_geometry_q_group_row_vars(q_group_state)


def open_geometry_q_group_window(
    *,
    root,
    view_state,
    preview_state,
    q_group_state,
    fit_config: Mapping[str, object] | None,
    current_geometry_fit_var_names: Sequence[str] | None,
    on_toggle: Callable[[tuple[object, ...] | None, object], None],
    on_include_all: Callable[[], None],
    on_exclude_all: Callable[[], None],
    on_update_listed_peaks: Callable[[], None],
    on_save: Callable[[], None],
    on_load: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open and refresh the geometry Q-group selector window."""

    opened = gui_views.open_geometry_q_group_window(
        root=root,
        view_state=view_state,
        on_include_all=on_include_all,
        on_exclude_all=on_exclude_all,
        on_update_listed_peaks=on_update_listed_peaks,
        on_save=on_save,
        on_load=on_load,
        on_close=on_close,
    )
    refresh_geometry_q_group_window(
        view_state=view_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=current_geometry_fit_var_names,
        on_toggle=on_toggle,
    )
    return opened


def geometry_q_group_key_to_jsonable(group_key: object) -> list[object] | None:
    """Convert one stable Qr/Qz group key into a JSON-safe list."""

    return gui_manual_geometry.geometry_q_group_key_to_jsonable(group_key)


def geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""

    return gui_manual_geometry.geometry_q_group_key_from_jsonable(value)


def geometry_q_group_float_for_json(value: object) -> float | None:
    """Return a finite float for JSON export, or ``None`` when unavailable."""

    numeric = _coerce_float(value, float("nan"))
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_geometry_q_group_export_rows(
    *,
    preview_state,
    q_group_state,
    entries: Sequence[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build JSON export rows for the current Qr/Qz selector listing."""

    rows: list[dict[str, object]] = []
    source_entries = list(entries) if entries is not None else gui_controllers.listed_geometry_q_group_entries(
        q_group_state
    )
    excluded_q_groups = getattr(preview_state, "excluded_q_groups", set())
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            continue
        group_key = entry.get("key")
        serialized_key = geometry_q_group_key_to_jsonable(group_key)
        if serialized_key is None:
            continue
        hkl_preview = []
        for hkl_value in entry.get("hkl_preview", [])[:8]:
            if (
                not isinstance(hkl_value, (list, tuple, np.ndarray))
                or len(hkl_value) < 3
            ):
                continue
            try:
                hkl_preview.append(
                    [int(hkl_value[0]), int(hkl_value[1]), int(hkl_value[2])]
                )
            except Exception:
                continue
        rows.append(
            {
                "key": serialized_key,
                "included": bool(group_key not in excluded_q_groups),
                "source_label": str(entry.get("source_label", "")),
                "qr": geometry_q_group_float_for_json(entry.get("qr", np.nan)),
                "qz": geometry_q_group_float_for_json(entry.get("qz", np.nan)),
                "gz_index": int(entry.get("gz_index", serialized_key[3])),
                "total_intensity": geometry_q_group_float_for_json(
                    entry.get("total_intensity", np.nan)
                ),
                "peak_count": int(entry.get("peak_count", 0)),
                "hkl_preview": hkl_preview,
                "display_label": format_geometry_q_group_line(entry),
            }
        )
    return rows


def build_geometry_q_group_save_payload(
    export_rows: Sequence[Mapping[str, object]],
    *,
    saved_at: str,
) -> dict[str, object]:
    """Build the JSON payload written by the Qr/Qz selector save action."""

    return {
        "type": "ra_sim.geometry_q_group_selection",
        "version": 1,
        "saved_at": str(saved_at),
        "row_count": int(len(export_rows)),
        "included_count": int(
            sum(1 for row in export_rows if bool(row.get("included", False)))
        ),
        "rows": [dict(row) for row in export_rows],
    }


def load_geometry_q_group_saved_state(
    payload: object,
) -> tuple[dict[tuple[object, ...], bool] | None, str | None]:
    """Validate one saved selector payload and rebuild its inclusion map."""

    if not isinstance(payload, Mapping):
        return None, "Invalid Qr/Qz peak list file: expected a JSON object."
    if str(payload.get("type", "")) != "ra_sim.geometry_q_group_selection":
        return None, "Invalid Qr/Qz peak list file type."

    saved_rows = payload.get("rows", [])
    if not isinstance(saved_rows, list) or not saved_rows:
        return None, "Loaded Qr/Qz peak list is empty."

    saved_state: dict[tuple[object, ...], bool] = {}
    for row in saved_rows:
        if not isinstance(row, Mapping):
            continue
        group_key = geometry_q_group_key_from_jsonable(row.get("key"))
        if group_key is None:
            continue
        saved_state[group_key] = bool(row.get("included", True))

    if not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."
    return saved_state, None


def apply_loaded_geometry_q_group_saved_state(
    *,
    preview_state,
    q_group_state,
    saved_state: Mapping[tuple[object, ...], bool] | None,
) -> tuple[dict[str, int] | None, str | None]:
    """Apply one loaded selector inclusion map to the current listed rows."""

    if not isinstance(saved_state, Mapping) or not saved_state:
        return None, "Loaded Qr/Qz peak list does not contain any valid rows."

    current_entries = gui_controllers.listed_geometry_q_group_entries(q_group_state)
    current_keys = [
        entry.get("key")
        for entry in current_entries
        if isinstance(entry, Mapping) and entry.get("key") is not None
    ]
    if not current_keys:
        return None, (
            "No listed Qr/Qz groups are available to match against the saved list. "
            'Press "Update Listed Peaks" first.'
        )

    current_key_set = set(current_keys)
    matched_keys = current_key_set.intersection(saved_state.keys())
    if not matched_keys:
        return None, "Loaded Qr/Qz peak list does not match any currently listed groups."

    gui_controllers.replace_geometry_preview_excluded_q_groups(
        preview_state,
        [
            key
            for key in current_keys
            if (key not in saved_state) or (not bool(saved_state.get(key, False)))
        ],
    )
    return {
        "matched_total": int(len(matched_keys)),
        "included_total": int(
            sum(1 for key in matched_keys if bool(saved_state.get(key, False)))
        ),
        "current_only": int(sum(1 for key in current_keys if key not in saved_state)),
        "saved_only": int(sum(1 for key in saved_state if key not in current_key_set)),
    }, None
