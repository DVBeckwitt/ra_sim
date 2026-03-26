"""Workflow helpers for GUI background-file management."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from . import background as gui_background
from . import views as gui_views


def _replace_list(target: list, values: Sequence[object] | None) -> list:
    target.clear()
    target.extend(list(values or ()))
    return target


def apply_background_state_update(
    background_state,
    payload: dict[str, object],
):
    """Apply one background payload to the shared runtime state in place."""

    if "osc_files" in payload:
        _replace_list(background_state.osc_files, payload.get("osc_files"))
    if "background_images" in payload:
        _replace_list(
            background_state.background_images,
            payload.get("background_images"),
        )
    if "background_images_native" in payload:
        _replace_list(
            background_state.background_images_native,
            payload.get("background_images_native"),
        )
    if "background_images_display" in payload:
        _replace_list(
            background_state.background_images_display,
            payload.get("background_images_display"),
        )
    if "current_background_index" in payload:
        background_state.current_background_index = int(
            payload.get("current_background_index", 0)
        )
    if "current_background_image" in payload:
        background_state.current_background_image = payload.get(
            "current_background_image"
        )
    if "current_background_display" in payload:
        background_state.current_background_display = payload.get(
            "current_background_display"
        )
    return background_state


def load_background_files(
    background_state,
    file_paths: Sequence[object],
    *,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    select_index: int = 0,
) -> dict[str, object]:
    """Load background files and apply the resulting state payload in place."""

    updated = gui_background.load_background_files(
        file_paths,
        image_size=image_size,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
        select_index=select_index,
    )
    apply_background_state_update(background_state, updated)
    return updated


def switch_background(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
) -> dict[str, object]:
    """Advance to the next background image and apply it in place."""

    updated = gui_background.switch_background(
        osc_files=background_state.osc_files,
        background_images=background_state.background_images,
        background_images_native=background_state.background_images_native,
        background_images_display=background_state.background_images_display,
        current_background_index=background_state.current_background_index,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    apply_background_state_update(background_state, updated)
    return updated


def background_file_dialog_initial_dir(
    osc_files: Sequence[object],
    current_background_index: int,
    default_dir: object,
) -> str:
    """Return the preferred starting directory for the background file dialog."""

    try:
        if osc_files:
            current_path = Path(
                str(osc_files[int(current_background_index)])
            ).expanduser()
            return str(current_path.parent)
    except Exception:
        pass
    return str(default_dir)


def build_background_file_status_text(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_base: float | None = None,
    theta_effective: float | None = None,
    use_shared_theta_offset: bool = False,
    pair_count: int = 0,
    group_count: int = 0,
    sigma_values: Sequence[object] | None = None,
    fit_indices: Sequence[object] | None = None,
) -> str:
    """Build the background-file status line shown in the GUI."""

    if not osc_files:
        return "Background: no files loaded"

    idx = int(current_background_index) % len(osc_files)
    current_name = Path(str(osc_files[idx])).name
    status_text = f"Background {idx + 1}/{len(osc_files)}: {current_name}"

    theta_base_finite = theta_base is not None and np.isfinite(float(theta_base))
    theta_effective_finite = theta_effective is not None and np.isfinite(
        float(theta_effective)
    )
    if theta_base_finite:
        if use_shared_theta_offset and theta_effective_finite:
            status_text += (
                f" | theta_i={float(theta_base):.4f}°"
                f" | theta={float(theta_effective):.4f}°"
            )
        else:
            status_text += f" | theta={float(theta_base):.4f}°"

    pair_count_i = max(0, int(pair_count))
    group_count_i = max(0, int(group_count))
    if pair_count_i > 0:
        status_text += f" | manual={group_count_i} groups/{pair_count_i} pts"
        finite_sigma_values = [
            float(value)
            for value in sigma_values or ()
            if np.isfinite(float(value))
        ]
        if finite_sigma_values:
            status_text += (
                f" | sigma~{float(np.median(np.asarray(finite_sigma_values, dtype=float))):.2f}px"
            )

    normalized_fit_indices: list[int] = []
    for raw_idx in fit_indices or ():
        try:
            normalized_fit_indices.append(int(raw_idx))
        except Exception:
            continue
    if normalized_fit_indices:
        if len(normalized_fit_indices) > 1:
            status_text += f" | fit={len(normalized_fit_indices)} backgrounds"
        elif len(osc_files) > 1:
            status_text += f" | fit=bg {normalized_fit_indices[0] + 1}"

    return status_text


def set_background_file_status_from_state(
    *,
    view_state,
    background_state,
    current_background_theta_values: Callable[[], Sequence[object]],
    background_theta_for_index: Callable[[int], object],
    geometry_fit_uses_shared_theta_offset: Callable[[], bool],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[object]],
    geometry_manual_pair_group_count: Callable[[int], int],
    current_geometry_fit_background_indices: Callable[[], Sequence[object]],
) -> str:
    """Refresh the GUI background-file status line from shared runtime state."""

    if getattr(view_state, "background_file_status_var", None) is None:
        return ""

    theta_base = None
    theta_effective = None
    use_shared_theta_offset = False
    try:
        theta_values = list(current_background_theta_values())
        if theta_values:
            idx = int(background_state.current_background_index) % max(
                1,
                len(background_state.osc_files),
            )
            theta_base = float(theta_values[idx])
            theta_effective = float(background_theta_for_index(idx))
            use_shared_theta_offset = bool(geometry_fit_uses_shared_theta_offset())
    except Exception:
        pass

    pair_count = 0
    group_count = 0
    sigma_values: list[float] = []
    try:
        idx = int(background_state.current_background_index) % max(
            1,
            len(background_state.osc_files),
        )
        pair_rows = list(geometry_manual_pairs_for_index(idx) or ())
        pair_count = len(pair_rows)
        group_count = int(geometry_manual_pair_group_count(idx))
        for entry in pair_rows:
            if not isinstance(entry, Mapping):
                continue
            try:
                sigma_value = float(entry.get("sigma_px", np.nan))
            except Exception:
                continue
            if np.isfinite(sigma_value):
                sigma_values.append(sigma_value)
    except Exception:
        pass

    fit_indices: list[int] = []
    try:
        fit_indices = [
            int(raw_idx)
            for raw_idx in current_geometry_fit_background_indices() or ()
        ]
    except Exception:
        fit_indices = []

    status_text = build_background_file_status_text(
        osc_files=background_state.osc_files,
        current_background_index=background_state.current_background_index,
        theta_base=theta_base,
        theta_effective=theta_effective,
        use_shared_theta_offset=use_shared_theta_offset,
        pair_count=pair_count,
        group_count=group_count,
        sigma_values=sigma_values,
        fit_indices=fit_indices,
    )
    gui_views.set_background_file_status_text(view_state, status_text)
    return status_text


def load_background_files_with_side_effects(
    background_state,
    file_paths: Sequence[object],
    *,
    image_size: int,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    sync_background_runtime_state: Callable[[], None],
    replace_geometry_manual_pairs_by_background: Callable[[dict], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    set_geometry_manual_pick_mode: Callable[[bool], None],
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    sync_background_theta_controls: Callable[[], None],
    sync_geometry_fit_background_selection: Callable[[], None],
    clear_geometry_pick_artists: Callable[[], None],
    refresh_background_file_status: Callable[[], None],
    schedule_update: Callable[[], None],
    select_index: int = 0,
) -> dict[str, object]:
    """Load background files and apply the dependent GUI side effects."""

    updated = load_background_files(
        background_state,
        file_paths,
        image_size=image_size,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
        select_index=select_index,
    )
    sync_background_runtime_state()
    replace_geometry_manual_pairs_by_background({})
    invalidate_geometry_manual_pick_cache()
    clear_geometry_manual_undo_stack()
    clear_geometry_fit_undo_stack()
    set_geometry_manual_pick_mode(False)

    current_display = background_state.current_background_display
    set_background_display_data(current_display)
    update_background_slider_defaults(current_display)
    sync_background_theta_controls()
    sync_geometry_fit_background_selection()
    clear_geometry_pick_artists()
    refresh_background_file_status()
    schedule_update()
    return updated


def switch_background_with_side_effects(
    background_state,
    *,
    display_rotate_k: int,
    read_osc: Callable[[str], object],
    sync_background_runtime_state: Callable[[], None],
    invalidate_geometry_manual_pick_cache: Callable[[], None],
    clear_geometry_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    sync_theta_initial_to_background: Callable[[int], None] | None,
    set_background_display_data: Callable[[object], None],
    update_background_slider_defaults: Callable[[object], None],
    refresh_background_file_status: Callable[[], None],
    render_current_geometry_manual_pairs: Callable[[], None],
    schedule_update: Callable[[], None],
) -> dict[str, object]:
    """Switch backgrounds and apply the dependent GUI side effects."""

    updated = switch_background(
        background_state,
        display_rotate_k=display_rotate_k,
        read_osc=read_osc,
    )
    sync_background_runtime_state()
    invalidate_geometry_manual_pick_cache()
    clear_geometry_manual_undo_stack()
    clear_geometry_fit_undo_stack()
    if sync_theta_initial_to_background is not None:
        sync_theta_initial_to_background(int(background_state.current_background_index))

    current_display = background_state.current_background_display
    set_background_display_data(current_display)
    update_background_slider_defaults(current_display)
    refresh_background_file_status()
    render_current_geometry_manual_pairs()
    schedule_update()
    return updated
