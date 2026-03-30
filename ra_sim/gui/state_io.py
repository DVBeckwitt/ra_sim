"""Helpers for GUI state persistence and background image switching."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


GUI_STATE_EXCLUDED_VAR_NAMES = {
    "background_file_status_var",
    "cif_file_var",
    "geometry_preview_exclude_button_var",
    "hkl_pick_button_var",
    "resolution_count_var",
}


def canonicalize_gui_state_background_path(path: object) -> str:
    """Return a stable normalized path key for background files."""

    return os.path.normcase(str(Path(str(path)).expanduser().resolve(strict=False)))


def _coerce_gui_state_bool(value: object, *, fallback: bool) -> bool:
    """Return a compatibility-safe boolean restored from saved GUI state."""

    if value is None:
        return bool(fallback)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return bool(fallback)
        return bool(int(round(numeric)))

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(fallback)


def _coerce_gui_state_rotation_k(value: object, *, fallback: int) -> int:
    """Return one restored background rotation quarter-turn count."""

    numeric_value: int | None = None
    if value is None:
        numeric_value = None
    elif isinstance(value, (int, np.integer)):
        numeric_value = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            numeric_value = int(round(numeric))
    else:
        text = str(value).strip()
        if text:
            try:
                numeric_value = int(text)
            except Exception:
                try:
                    parsed = float(text)
                except Exception:
                    numeric_value = None
                else:
                    if np.isfinite(parsed):
                        numeric_value = int(round(parsed))

    if numeric_value is None:
        numeric_value = int(fallback)
    return int(numeric_value) % 4


def background_files_match_loaded_state(
    file_paths: list[str],
    *,
    osc_files: list[str],
    background_images_native: list[object],
    background_images_display: list[object],
) -> bool:
    """Return whether the requested backgrounds already match the loaded state."""

    if not file_paths:
        return False
    if len(file_paths) != len(osc_files):
        return False
    if len(background_images_native) != len(file_paths):
        return False
    if len(background_images_display) != len(file_paths):
        return False
    requested_paths = [
        canonicalize_gui_state_background_path(path) for path in file_paths
    ]
    current_paths = [
        canonicalize_gui_state_background_path(path) for path in osc_files
    ]
    return requested_paths == current_paths


def load_background_files_for_state(
    file_paths: list[str],
    *,
    osc_files: list[str],
    background_images: list[object],
    background_images_native: list[object],
    background_images_display: list[object],
    select_index: int = 0,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    set_background_display: Callable[[object], None] | None = None,
) -> dict[str, object] | None:
    """Load or reuse background images and return the updated background state."""

    normalized_paths = [
        str(Path(str(path)).expanduser())
        for path in file_paths
        if path is not None
    ]
    if not normalized_paths:
        return None

    if background_files_match_loaded_state(
        normalized_paths,
        osc_files=osc_files,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
    ):
        index = max(0, min(int(select_index), len(background_images_native) - 1))
        current_background_image = background_images_native[index]
        current_background_display = background_images_display[index]
        if set_background_display is not None:
            set_background_display(current_background_display)
        return {
            "osc_files": list(normalized_paths),
            "background_images": list(background_images),
            "background_images_native": list(background_images_native),
            "background_images_display": list(background_images_display),
            "current_background_index": index,
            "current_background_image": current_background_image,
            "current_background_display": current_background_display,
        }

    loaded_native = [np.asarray(read_osc(path)) for path in normalized_paths]
    if not loaded_native:
        return None

    new_background_images = [np.array(img) for img in loaded_native]
    new_background_images_native = [np.array(img) for img in loaded_native]
    new_background_images_display = [
        np.rot90(img, display_rotate_k) for img in new_background_images_native
    ]
    index = max(0, min(int(select_index), len(new_background_images_native) - 1))
    current_background_image = new_background_images_native[index]
    current_background_display = new_background_images_display[index]
    if set_background_display is not None:
        set_background_display(current_background_display)
    return {
        "osc_files": list(normalized_paths),
        "background_images": new_background_images,
        "background_images_native": new_background_images_native,
        "background_images_display": new_background_images_display,
        "current_background_index": index,
        "current_background_image": current_background_image,
        "current_background_display": current_background_display,
    }


def is_persistable_gui_var(
    name: str,
    value: object,
    *,
    tk_variable_type: type[object],
) -> bool:
    """Return whether a Tk variable should be included in GUI-state snapshots."""

    if not isinstance(value, tk_variable_type):
        return False
    if name.startswith("_") or name in GUI_STATE_EXCLUDED_VAR_NAMES:
        return False
    if name.endswith(("_button_var", "_entry_var", "_label_var", "_status_var")):
        return False
    return True


def collect_full_gui_state_snapshot(
    *,
    global_items: Mapping[str, object],
    tk_variable_type: type[object],
    occ_vars: Sequence[object],
    atom_site_fract_vars: Sequence[object],
    geometry_q_group_rows: Sequence[dict[str, object]],
    geometry_manual_pairs: Sequence[dict[str, object]],
    selected_hkl_target: object,
    primary_cif_path: object,
    secondary_cif_path: object | None,
    osc_files: Sequence[object],
    current_background_index: int,
    background_visible: bool,
    background_backend_rotation_k: int,
    background_backend_flip_x: bool,
    background_backend_flip_y: bool,
    background_limits_user_override: bool,
    simulation_limits_user_override: bool,
    scale_factor_user_override: bool,
) -> dict[str, object]:
    """Build a portable snapshot of the current GUI state."""

    variables: dict[str, object] = {}
    for name, value in global_items.items():
        if not is_persistable_gui_var(
            str(name),
            value,
            tk_variable_type=tk_variable_type,
        ):
            continue
        try:
            variables[str(name)] = value.get()
        except Exception:
            continue

    occupancy_values: list[float | None] = []
    for occ_var in occ_vars:
        try:
            occupancy_values.append(float(occ_var.get()))
        except Exception:
            occupancy_values.append(None)

    atom_site_values: list[dict[str, float]] = []
    for row in atom_site_fract_vars:
        if not isinstance(row, Mapping):
            continue
        atom_site_values.append(
            {
                "x": float(row["x"].get()),
                "y": float(row["y"].get()),
                "z": float(row["z"].get()),
            }
        )

    geometry_state: dict[str, object] = {
        "q_group_rows": list(geometry_q_group_rows),
        "manual_pairs": list(geometry_manual_pairs),
    }
    if isinstance(selected_hkl_target, tuple) and len(selected_hkl_target) == 3:
        geometry_state["selected_hkl_target"] = [
            int(selected_hkl_target[0]),
            int(selected_hkl_target[1]),
            int(selected_hkl_target[2]),
        ]

    return {
        "variables": variables,
        "dynamic_lists": {
            "occupancy_values": occupancy_values,
            "atom_site_fractional_values": atom_site_values,
        },
        "files": {
            "primary_cif_path": str(primary_cif_path),
            "secondary_cif_path": (
                str(secondary_cif_path) if secondary_cif_path else None
            ),
            "background_files": [
                str(Path(str(path)).expanduser()) for path in osc_files
            ],
            "current_background_index": int(current_background_index),
        },
        "flags": {
            "background_visible": bool(background_visible),
            "background_backend_rotation_k": int(background_backend_rotation_k),
            "background_backend_flip_x": bool(background_backend_flip_x),
            "background_backend_flip_y": bool(background_backend_flip_y),
            "background_limits_user_override": bool(
                background_limits_user_override
            ),
            "simulation_limits_user_override": bool(
                simulation_limits_user_override
            ),
            "scale_factor_user_override": bool(scale_factor_user_override),
        },
        "geometry": geometry_state,
    }


def apply_gui_state_background_theta_compatibility(
    saved_variables: Mapping[str, object] | None,
    *,
    osc_files: Sequence[object],
    theta_initial_var: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    format_background_theta_values: Callable[[Sequence[float]], str],
    default_geometry_fit_background_selection: Callable[[], str],
) -> None:
    """Backfill newer background-theta GUI state from legacy snapshots."""

    if not isinstance(saved_variables, Mapping):
        saved_variables = {}

    if (
        "background_theta_list_var" not in saved_variables
        and "theta_initial_var" in saved_variables
        and background_theta_list_var is not None
        and len(osc_files) > 0
    ):
        try:
            restored_theta = float(theta_initial_var.get())
        except Exception:
            restored_theta = None
        if restored_theta is not None and np.isfinite(restored_theta):
            background_theta_list_var.set(
                format_background_theta_values(
                    [float(restored_theta)] * int(len(osc_files))
                )
            )

    if (
        geometry_theta_offset_var is not None
        and "geometry_theta_offset_var" not in saved_variables
    ):
        geometry_theta_offset_var.set("0.0")

    if (
        geometry_fit_background_selection_var is not None
        and "geometry_fit_background_selection_var" not in saved_variables
    ):
        geometry_fit_background_selection_var.set(
            default_geometry_fit_background_selection()
        )


def apply_gui_state_files(
    files: Mapping[str, object] | None,
    *,
    apply_primary_cif_path: Callable[[str], None],
    load_background_files: Callable[[list[str], int], None],
) -> list[str]:
    """Apply file-path portions of a GUI-state snapshot and return warnings."""

    warnings: list[str] = []
    if not isinstance(files, Mapping):
        return warnings

    cif_path = files.get("primary_cif_path")
    if cif_path:
        candidate = Path(str(cif_path)).expanduser()
        if candidate.is_file():
            try:
                apply_primary_cif_path(str(candidate))
            except Exception as exc:
                warnings.append(f"primary CIF: {exc}")
        else:
            warnings.append(f"primary CIF missing: {candidate}")

    raw_background_paths = files.get("background_files", [])
    background_paths: list[str] = []
    if isinstance(raw_background_paths, list):
        for raw_path in raw_background_paths:
            if raw_path is None:
                continue
            candidate = Path(str(raw_path)).expanduser()
            if candidate.is_file():
                background_paths.append(str(candidate))
            else:
                warnings.append(f"background missing: {candidate}")
    if background_paths:
        try:
            load_background_files(
                background_paths,
                int(files.get("current_background_index", 0)),
            )
        except Exception as exc:
            warnings.append(f"backgrounds: {exc}")

    return warnings


def apply_gui_state_variables(
    saved_variables: Mapping[str, object] | None,
    *,
    global_items: Mapping[str, object],
    tk_variable_type: type[object],
) -> list[str]:
    """Apply saved Tk variable values and return any set warnings."""

    warnings: list[str] = []
    if not isinstance(saved_variables, Mapping):
        return warnings

    for name, stored_value in saved_variables.items():
        target_var = global_items.get(str(name))
        if not isinstance(target_var, tk_variable_type):
            continue
        try:
            target_var.set(stored_value)
        except Exception as exc:
            warnings.append(f"{name}: {exc}")

    return warnings


def apply_dynamic_gui_state_lists(
    dynamic_lists: Mapping[str, object] | None,
    *,
    occ_vars: Sequence[object],
    atom_site_fract_vars: Sequence[object],
) -> None:
    """Apply saved occupancy and atom-site list values."""

    if not isinstance(dynamic_lists, Mapping):
        return

    saved_occ = dynamic_lists.get("occupancy_values", [])
    if isinstance(saved_occ, list):
        for occ_var, stored_value in zip(occ_vars, saved_occ):
            try:
                occ_var.set(float(stored_value))
            except Exception:
                continue

    saved_atom_sites = dynamic_lists.get("atom_site_fractional_values", [])
    if isinstance(saved_atom_sites, list):
        for axis_vars, stored_row in zip(atom_site_fract_vars, saved_atom_sites):
            if not isinstance(axis_vars, Mapping) or not isinstance(stored_row, Mapping):
                continue
            for axis_name in ("x", "y", "z"):
                axis_var = axis_vars.get(axis_name)
                if axis_var is None or axis_name not in stored_row:
                    continue
                try:
                    axis_var.set(float(stored_row[axis_name]))
                except Exception:
                    continue


def apply_gui_state_flags(
    flags: Mapping[str, object] | None,
    *,
    current_flags: Mapping[str, object],
    toggle_background: Callable[[], None],
) -> dict[str, object]:
    """Apply saved non-Tk GUI flags and return the updated flag state."""

    updated_flags = {
        "background_visible": _coerce_gui_state_bool(
            current_flags.get("background_visible", False),
            fallback=False,
        ),
        "background_backend_rotation_k": _coerce_gui_state_rotation_k(
            current_flags.get("background_backend_rotation_k", 0),
            fallback=0,
        ),
        "background_backend_flip_x": _coerce_gui_state_bool(
            current_flags.get("background_backend_flip_x", False),
            fallback=False,
        ),
        "background_backend_flip_y": _coerce_gui_state_bool(
            current_flags.get("background_backend_flip_y", False),
            fallback=False,
        ),
        "background_limits_user_override": _coerce_gui_state_bool(
            current_flags.get("background_limits_user_override", False),
            fallback=False,
        ),
        "simulation_limits_user_override": _coerce_gui_state_bool(
            current_flags.get("simulation_limits_user_override", False),
            fallback=False,
        ),
        "scale_factor_user_override": _coerce_gui_state_bool(
            current_flags.get("scale_factor_user_override", False),
            fallback=False,
        ),
    }
    if not isinstance(flags, Mapping):
        return updated_flags

    desired_background_visible = _coerce_gui_state_bool(
        flags.get("background_visible", updated_flags["background_visible"]),
        fallback=bool(updated_flags["background_visible"]),
    )
    if desired_background_visible != bool(updated_flags["background_visible"]):
        toggle_background()
        updated_flags["background_visible"] = desired_background_visible

    updated_flags["background_backend_rotation_k"] = _coerce_gui_state_rotation_k(
        flags.get(
            "background_backend_rotation_k",
            updated_flags["background_backend_rotation_k"],
        ),
        fallback=int(updated_flags["background_backend_rotation_k"]),
    )
    updated_flags["background_backend_flip_x"] = _coerce_gui_state_bool(
        flags.get("background_backend_flip_x", updated_flags["background_backend_flip_x"]),
        fallback=bool(updated_flags["background_backend_flip_x"]),
    )
    updated_flags["background_backend_flip_y"] = _coerce_gui_state_bool(
        flags.get("background_backend_flip_y", updated_flags["background_backend_flip_y"]),
        fallback=bool(updated_flags["background_backend_flip_y"]),
    )
    updated_flags["background_limits_user_override"] = _coerce_gui_state_bool(
        flags.get(
            "background_limits_user_override",
            updated_flags["background_limits_user_override"],
        ),
        fallback=bool(updated_flags["background_limits_user_override"]),
    )
    updated_flags["simulation_limits_user_override"] = _coerce_gui_state_bool(
        flags.get(
            "simulation_limits_user_override",
            updated_flags["simulation_limits_user_override"],
        ),
        fallback=bool(updated_flags["simulation_limits_user_override"]),
    )
    updated_flags["scale_factor_user_override"] = _coerce_gui_state_bool(
        flags.get("scale_factor_user_override", updated_flags["scale_factor_user_override"]),
        fallback=bool(updated_flags["scale_factor_user_override"]),
    )
    return updated_flags


def apply_gui_state_geometry(
    geometry_state: Mapping[str, object] | None,
    *,
    geometry_preview_excluded_q_groups: set[tuple[object, ...]],
    geometry_q_group_key_from_jsonable: Callable[
        [object], tuple[object, ...] | None
    ],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    apply_geometry_manual_pairs_snapshot: Callable[..., object],
    current_background_index: int,
    selected_hkl_target: tuple[int, int, int] | None,
) -> dict[str, Any]:
    """Apply saved geometry selectors/manual placements."""

    updated_selected_hkl_target = selected_hkl_target
    warnings: list[str] = []
    if not isinstance(geometry_state, Mapping):
        return {
            "selected_hkl_target": updated_selected_hkl_target,
            "warnings": warnings,
        }

    saved_rows = geometry_state.get("q_group_rows", [])
    if isinstance(saved_rows, list):
        geometry_preview_excluded_q_groups.clear()
        for row in saved_rows:
            if not isinstance(row, Mapping):
                continue
            group_key = geometry_q_group_key_from_jsonable(row.get("key"))
            if group_key is None:
                continue
            if not bool(row.get("included", True)):
                geometry_preview_excluded_q_groups.add(group_key)
        invalidate_geometry_manual_pick_cache()

    target_hkl = geometry_state.get("selected_hkl_target")
    if isinstance(target_hkl, (list, tuple)) and len(target_hkl) >= 3:
        try:
            updated_selected_hkl_target = (
                int(target_hkl[0]),
                int(target_hkl[1]),
                int(target_hkl[2]),
            )
        except Exception:
            updated_selected_hkl_target = None

    try:
        _ = apply_geometry_manual_pairs_snapshot(
            {
                "manual_pairs": geometry_state.get("manual_pairs", []),
                "background_files": [],
                "current_background_index": int(current_background_index),
            },
            allow_background_reload=False,
        )
    except Exception as exc:
        warnings.append(f"manual placements: {exc}")

    return {
        "selected_hkl_target": updated_selected_hkl_target,
        "warnings": warnings,
    }


def geometry_state_requires_visible_background(
    geometry_state: Mapping[str, object] | None,
    *,
    geometry_q_group_key_from_jsonable: Callable[
        [object], tuple[object, ...] | None
    ],
) -> bool:
    """Return whether restored geometry state should force the background visible."""

    if not isinstance(geometry_state, Mapping):
        return False

    saved_rows = geometry_state.get("q_group_rows", [])
    if not isinstance(saved_rows, list):
        return False

    for row in saved_rows:
        if not isinstance(row, Mapping):
            continue
        if not bool(row.get("included", True)):
            continue
        if geometry_q_group_key_from_jsonable(row.get("key")) is None:
            continue
        return True
    return False


def apply_geometry_state_background_view_compatibility(
    geometry_state: Mapping[str, object] | None,
    *,
    geometry_q_group_key_from_jsonable: Callable[
        [object], tuple[object, ...] | None
    ],
    show_caked_2d_var: object | None,
    background_visible: bool,
    toggle_background: Callable[[], None],
) -> bool:
    """Ensure imported Qr/Qz selections reopen the detector-background 2D view."""

    if not geometry_state_requires_visible_background(
        geometry_state,
        geometry_q_group_key_from_jsonable=geometry_q_group_key_from_jsonable,
    ):
        return False

    if show_caked_2d_var is not None:
        try:
            show_caked_2d_var.set(False)
        except Exception:
            pass
    if not bool(background_visible):
        toggle_background()
    return True


def build_gui_state_import_summary(warnings: Sequence[str]) -> str:
    """Return the user-facing import summary for a GUI-state restore."""

    summary = "Imported GUI state snapshot."
    if warnings:
        summary += " Warnings: " + "; ".join(list(warnings)[:4])
        if len(warnings) > 4:
            summary += f"; +{len(warnings) - 4} more"
    return summary
