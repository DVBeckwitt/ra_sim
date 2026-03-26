"""Workflow helpers for the Bragg-Qr manager window."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import controllers as gui_controllers
from . import views as gui_views


def _safe_listbox_curselection(
    listbox: object,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> list[int]:
    if listbox is None:
        return []
    try:
        return [int(idx) for idx in list(listbox.curselection())]
    except Exception as exc:
        if tcl_error_types and isinstance(exc, tcl_error_types):
            return []
        raise


def selected_bragg_qr_window_keys(
    manager_state,
    qr_listbox: object,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> list[tuple[str, int]]:
    """Return normalized Bragg-Qr group keys selected in the manager listbox."""

    return gui_controllers.selected_bragg_qr_keys(
        manager_state,
        _safe_listbox_curselection(
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )


def selected_primary_bragg_qr_window_key(
    manager_state,
    *,
    selected_keys: Sequence[tuple[str, int]] | None = None,
) -> tuple[str, int] | None:
    """Return the primary selected Bragg-Qr group key or the stored fallback."""

    if selected_keys:
        source_label, m_idx = selected_keys[0]
        return (
            gui_controllers.normalize_bragg_qr_source_label(source_label),
            int(m_idx),
        )

    group_key = getattr(manager_state, "selected_group_key", None)
    if not isinstance(group_key, tuple) or len(group_key) < 2:
        return None
    return (
        gui_controllers.normalize_bragg_qr_source_label(group_key[0]),
        int(group_key[1]),
    )


def selected_bragg_qr_l_window_keys(
    manager_state,
    l_listbox: object,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> list[int]:
    """Return Bragg-Qr L keys selected in the manager listbox."""

    return gui_controllers.selected_bragg_qr_l_keys(
        manager_state,
        _safe_listbox_curselection(
            l_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )


def refresh_bragg_qr_l_toggle_listbox(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    l_listbox: object,
    build_l_value_map: Callable[[str, int], Mapping[int, object]],
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Rebuild the Bragg-Qr L-value listbox from current manager state."""

    if not gui_views.bragg_qr_manager_window_open(view_state):
        return False
    if l_listbox is None:
        return False

    selected_l_keys = selected_bragg_qr_l_window_keys(
        manager_state,
        l_listbox,
        tcl_error_types=tcl_error_types,
    )
    selected_group = selected_primary_bragg_qr_window_key(
        manager_state,
        selected_keys=selected_bragg_qr_window_keys(
            manager_state,
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )
    l_model = gui_controllers.build_bragg_qr_l_list_model(
        manager_state,
        group_key=selected_group,
        l_value_map=(
            build_l_value_map(selected_group[0], int(selected_group[1]))
            if selected_group is not None
            else None
        ),
        selected_l_keys=selected_l_keys,
    )
    gui_controllers.set_bragg_qr_selected_group_key(
        manager_state,
        l_model["selected_group_key"],
    )
    gui_controllers.replace_bragg_qr_l_index_keys(
        manager_state,
        l_model["index_keys"],
    )
    gui_views.refresh_bragg_qr_manager_l_list(
        view_state=view_state,
        lines=l_model["lines"],
        selected_indices=l_model["selected_indices"],
        status_text=l_model["status_text"],
    )
    return True


def refresh_bragg_qr_toggle_window(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    entries: Sequence[Mapping[str, object]] | None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Rebuild the Bragg-Qr group listbox from current manager state."""

    if not gui_views.bragg_qr_manager_window_open(view_state):
        return False
    if qr_listbox is None:
        return False

    qr_model = gui_controllers.build_bragg_qr_qr_list_model(
        manager_state,
        entries,
        selected_keys=selected_bragg_qr_window_keys(
            manager_state,
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )
    gui_controllers.replace_bragg_qr_index_keys(
        manager_state,
        qr_model["index_keys"],
    )
    gui_views.refresh_bragg_qr_manager_qr_list(
        view_state=view_state,
        lines=qr_model["lines"],
        selected_indices=qr_model["selected_indices"],
        status_text=qr_model["status_text"],
        see_index=qr_model["see_index"],
    )
    return True
