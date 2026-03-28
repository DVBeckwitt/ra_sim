"""Workflow helpers for the Bragg-Qr manager window."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import controllers as gui_controllers
from . import views as gui_views


@dataclass
class BraggQrRuntimeBindings:
    """Runtime callbacks and shared state used by the Bragg-Qr manager GUI."""

    view_state: Any
    manager_state: Any
    get_entries: Callable[[], Sequence[Mapping[str, object]]]
    build_l_value_map: Callable[[str, int], Mapping[int, object]]
    apply_filters: Callable[[], None]
    set_progress_text: Callable[[str], None] | None = None
    invalid_key: int = 0
    tcl_error_types: tuple[type[BaseException], ...] = ()


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def current_runtime_bragg_qr_lattice_values(
    *,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
) -> tuple[float, float]:
    """Return the current primary/secondary lattice constants for Bragg-Qr helpers."""

    try:
        primary_a = float(_resolve_runtime_value(primary_candidate))
    except Exception:
        primary_a = float(_resolve_runtime_value(primary_fallback))

    secondary_a = float(primary_a)
    try:
        secondary_value = _resolve_runtime_value(secondary_candidate)
        if secondary_value is not None:
            secondary_candidate_value = float(secondary_value)
            if (
                np.isfinite(secondary_candidate_value)
                and secondary_candidate_value > 0.0
            ):
                secondary_a = float(secondary_candidate_value)
    except Exception:
        secondary_a = float(primary_a)

    return float(primary_a), float(secondary_a)


def build_runtime_bragg_qr_entries(
    simulation_runtime_state,
    *,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
) -> list[dict[str, object]]:
    """Build Bragg-Qr manager entries from the current runtime lattice values."""

    primary_a, secondary_a = current_runtime_bragg_qr_lattice_values(
        primary_candidate=primary_candidate,
        primary_fallback=primary_fallback,
        secondary_candidate=secondary_candidate,
    )
    return gui_controllers.build_bragg_qr_entries(
        simulation_runtime_state,
        primary_a=primary_a,
        secondary_a=secondary_a,
    )


def build_runtime_bragg_qr_l_value_map(
    simulation_runtime_state,
    source_label: str,
    m_idx: int,
) -> dict[int, float]:
    """Build one Bragg-Qr L-value map from the runtime simulation state."""

    return gui_controllers.build_bragg_qr_l_value_map(
        simulation_runtime_state,
        source_label,
        int(m_idx),
    )


def _m_values_from_miller_array(arr_like: object) -> set[int]:
    arr = np.asarray(_resolve_runtime_value(arr_like), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return set()
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    if not np.any(finite):
        return set()
    rows = arr[finite, :2]
    h_vals = np.rint(rows[:, 0]).astype(np.int64, copy=False)
    k_vals = np.rint(rows[:, 1]).astype(np.int64, copy=False)
    return {
        int(v) for v in (h_vals * h_vals + h_vals * k_vals + k_vals * k_vals)
    }


def build_active_qr_cylinder_overlay_entries(
    simulation_runtime_state,
    *,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
    primary_miller_all: object = None,
    secondary_miller_all: object = None,
) -> list[dict[str, object]]:
    """Return active Bragg-Qr groups used by the analytic Qr-cylinder overlay."""

    primary_a, secondary_a = current_runtime_bragg_qr_lattice_values(
        primary_candidate=primary_candidate,
        primary_fallback=primary_fallback,
        secondary_candidate=secondary_candidate,
    )

    primary_m: set[int] = set()
    if isinstance(simulation_runtime_state.sim_primary_qr, Mapping):
        for m_raw in simulation_runtime_state.sim_primary_qr.keys():
            try:
                primary_m.add(int(m_raw))
            except (TypeError, ValueError):
                continue
    primary_m.update(_m_values_from_miller_array(primary_miller_all))

    secondary_m = _m_values_from_miller_array(secondary_miller_all)

    entries: list[dict[str, object]] = []
    for source_label, lattice_a, m_values in (
        ("primary", primary_a, primary_m),
        ("secondary", secondary_a, secondary_m),
    ):
        for m_idx in sorted(m_values):
            qr_val = gui_controllers.qr_value_for_m(int(m_idx), lattice_a)
            if not np.isfinite(qr_val) or qr_val < 0.0:
                continue
            entries.append(
                {
                    "key": (str(source_label), int(m_idx)),
                    "source": str(source_label),
                    "m": int(m_idx),
                    "qr": float(qr_val),
                }
            )
    return entries


def make_runtime_active_qr_cylinder_overlay_entries_factory(
    *,
    simulation_runtime_state,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
    primary_miller_all: object = None,
    secondary_miller_all: object = None,
) -> Callable[[], list[dict[str, object]]]:
    """Return a zero-arg factory for the live Qr-cylinder overlay entries."""

    return lambda: build_active_qr_cylinder_overlay_entries(
        simulation_runtime_state,
        primary_candidate=primary_candidate,
        primary_fallback=primary_fallback,
        secondary_candidate=secondary_candidate,
        primary_miller_all=primary_miller_all,
        secondary_miller_all=secondary_miller_all,
    )


def build_runtime_bragg_qr_bindings(
    *,
    view_state,
    manager_state,
    simulation_runtime_state,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    invalid_key: int = 0,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> BraggQrRuntimeBindings:
    """Build shared runtime bindings for the Bragg-Qr manager from live values."""

    return BraggQrRuntimeBindings(
        view_state=view_state,
        manager_state=manager_state,
        get_entries=lambda: build_runtime_bragg_qr_entries(
            simulation_runtime_state,
            primary_candidate=primary_candidate,
            primary_fallback=primary_fallback,
            secondary_candidate=secondary_candidate,
        ),
        build_l_value_map=lambda source_label, m_idx: build_runtime_bragg_qr_l_value_map(
            simulation_runtime_state,
            source_label,
            int(m_idx),
        ),
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
        invalid_key=int(invalid_key),
        tcl_error_types=tcl_error_types,
    )


def make_runtime_bragg_qr_bindings_factory(
    *,
    view_state,
    manager_state,
    simulation_runtime_state,
    primary_candidate: object,
    primary_fallback: object,
    secondary_candidate: object | None = None,
    apply_filters: Callable[[], None],
    set_progress_text_factory: Callable[[], Callable[[str], None] | None] | None = None,
    invalid_key: int = 0,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> Callable[[], BraggQrRuntimeBindings]:
    """Return a zero-arg factory for live Bragg-Qr runtime bindings."""

    def _build_bindings() -> BraggQrRuntimeBindings:
        set_progress_text = (
            set_progress_text_factory() if callable(set_progress_text_factory) else None
        )
        return build_runtime_bragg_qr_bindings(
            view_state=view_state,
            manager_state=manager_state,
            simulation_runtime_state=simulation_runtime_state,
            primary_candidate=primary_candidate,
            primary_fallback=primary_fallback,
            secondary_candidate=secondary_candidate,
            apply_filters=apply_filters,
            set_progress_text=set_progress_text,
            invalid_key=invalid_key,
            tcl_error_types=tcl_error_types,
        )

    return _build_bindings


def make_runtime_bragg_qr_refresh_callback(
    bindings_factory: Callable[[], BraggQrRuntimeBindings],
) -> Callable[[], bool]:
    """Return a zero-arg callback that refreshes the Bragg-Qr runtime window."""

    return lambda: refresh_runtime_bragg_qr_toggle_window(bindings_factory())


def make_runtime_bragg_qr_open_callback(
    *,
    root,
    bindings_factory: Callable[[], BraggQrRuntimeBindings],
) -> Callable[[], bool]:
    """Return a zero-arg callback that opens the Bragg-Qr runtime window."""

    return lambda: open_runtime_bragg_qr_toggle_window(
        root=root,
        bindings=bindings_factory(),
    )


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


def on_bragg_qr_selection_changed(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    l_listbox: object,
    build_l_value_map: Callable[[str, int], Mapping[int, object]],
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Store the active Bragg-Qr group selection and refresh the L list."""

    gui_controllers.set_bragg_qr_selected_group_key(
        manager_state,
        selected_primary_bragg_qr_window_key(
            manager_state,
            selected_keys=selected_bragg_qr_window_keys(
                manager_state,
                qr_listbox,
                tcl_error_types=tcl_error_types,
            ),
        ),
    )
    return refresh_bragg_qr_l_toggle_listbox(
        view_state=view_state,
        manager_state=manager_state,
        qr_listbox=qr_listbox,
        l_listbox=l_listbox,
        build_l_value_map=build_l_value_map,
        tcl_error_types=tcl_error_types,
    )


def _normalize_group_keys(
    group_keys: Sequence[tuple[str, int]] | None,
) -> list[tuple[str, int]]:
    normalized_keys: list[tuple[str, int]] = []
    for source_label, m_idx in group_keys or ():
        normalized_keys.append(
            (
                gui_controllers.normalize_bragg_qr_source_label(source_label),
                int(m_idx),
            )
        )
    return normalized_keys


def set_bragg_qr_groups_enabled(
    *,
    manager_state,
    group_keys: Sequence[tuple[str, int]] | None,
    enabled: bool,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Enable or disable the provided Bragg-Qr groups and apply side effects."""

    normalized_keys = _normalize_group_keys(group_keys)
    if not normalized_keys:
        return False

    changed_count = gui_controllers.set_bragg_qr_groups_enabled(
        manager_state.disabled_groups,
        normalized_keys,
        enabled=enabled,
    )
    if changed_count <= 0:
        refresh_window()
        return False

    apply_filters()
    if set_progress_text is not None:
        action = "Enabled" if enabled else "Disabled"
        set_progress_text(f"{action} {len(normalized_keys)} Bragg Qr group(s).")
    return True


def set_bragg_qr_l_values_enabled(
    *,
    view_state,
    manager_state,
    group_key: tuple[str, int] | None,
    l_keys: Sequence[int] | None,
    enabled: bool,
    invalid_key: int,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Enable or disable Bragg-Qr L values for one selected group."""

    if group_key is None:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            l_text="Select a Qr group first.",
        )
        return False
    if not l_keys:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            l_text="Select one or more L values.",
        )
        return False

    source_label = gui_controllers.normalize_bragg_qr_source_label(group_key[0])
    m_idx = int(group_key[1])
    changed_count = gui_controllers.set_bragg_qr_l_values_enabled(
        manager_state.disabled_l_values,
        (source_label, m_idx),
        l_keys,
        enabled=enabled,
        invalid_key=invalid_key,
    )
    if changed_count <= 0:
        refresh_window()
        return False

    apply_filters()
    if set_progress_text is not None:
        action = "Enabled" if enabled else "Disabled"
        set_progress_text(
            f"{action} {len(list(l_keys))} L value(s) for {source_label} m={m_idx}."
        )
    return True


def disable_selected_bragg_qr_groups(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Disable the selected Bragg-Qr groups from the manager list."""

    keys = selected_bragg_qr_window_keys(
        manager_state,
        qr_listbox,
        tcl_error_types=tcl_error_types,
    )
    if not keys:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            qr_text="Select one or more Qr groups.",
        )
        return False
    return set_bragg_qr_groups_enabled(
        manager_state=manager_state,
        group_keys=keys,
        enabled=False,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def enable_selected_bragg_qr_groups(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Enable the selected Bragg-Qr groups from the manager list."""

    keys = selected_bragg_qr_window_keys(
        manager_state,
        qr_listbox,
        tcl_error_types=tcl_error_types,
    )
    if not keys:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            qr_text="Select one or more Qr groups.",
        )
        return False
    return set_bragg_qr_groups_enabled(
        manager_state=manager_state,
        group_keys=keys,
        enabled=True,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def toggle_selected_bragg_qr_groups(
    *,
    manager_state,
    qr_listbox: object,
    apply_filters: Callable[[], None],
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Toggle the selected Bragg-Qr groups from the manager list."""

    keys = selected_bragg_qr_window_keys(
        manager_state,
        qr_listbox,
        tcl_error_types=tcl_error_types,
    )
    if not keys:
        return False
    changed_count = gui_controllers.toggle_bragg_qr_groups(
        manager_state.disabled_groups,
        _normalize_group_keys(keys),
    )
    if changed_count > 0:
        apply_filters()
        return True
    return False


def disable_selected_bragg_qr_l_values(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    l_listbox: object,
    invalid_key: int,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Disable the selected Bragg-Qr L values for the selected group."""

    return set_bragg_qr_l_values_enabled(
        view_state=view_state,
        manager_state=manager_state,
        group_key=selected_primary_bragg_qr_window_key(
            manager_state,
            selected_keys=selected_bragg_qr_window_keys(
                manager_state,
                qr_listbox,
                tcl_error_types=tcl_error_types,
            ),
        ),
        l_keys=selected_bragg_qr_l_window_keys(
            manager_state,
            l_listbox,
            tcl_error_types=tcl_error_types,
        ),
        enabled=False,
        invalid_key=invalid_key,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def enable_selected_bragg_qr_l_values(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    l_listbox: object,
    invalid_key: int,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Enable the selected Bragg-Qr L values for the selected group."""

    return set_bragg_qr_l_values_enabled(
        view_state=view_state,
        manager_state=manager_state,
        group_key=selected_primary_bragg_qr_window_key(
            manager_state,
            selected_keys=selected_bragg_qr_window_keys(
                manager_state,
                qr_listbox,
                tcl_error_types=tcl_error_types,
            ),
        ),
        l_keys=selected_bragg_qr_l_window_keys(
            manager_state,
            l_listbox,
            tcl_error_types=tcl_error_types,
        ),
        enabled=True,
        invalid_key=invalid_key,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def toggle_selected_bragg_qr_l_values(
    *,
    manager_state,
    qr_listbox: object,
    l_listbox: object,
    invalid_key: int,
    apply_filters: Callable[[], None],
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Toggle the selected Bragg-Qr L values for the selected group."""

    group_key = selected_primary_bragg_qr_window_key(
        manager_state,
        selected_keys=selected_bragg_qr_window_keys(
            manager_state,
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )
    if group_key is None:
        return False
    l_keys = selected_bragg_qr_l_window_keys(
        manager_state,
        l_listbox,
        tcl_error_types=tcl_error_types,
    )
    if not l_keys:
        return False

    source_label = gui_controllers.normalize_bragg_qr_source_label(group_key[0])
    m_idx = int(group_key[1])
    changed_count = gui_controllers.toggle_bragg_qr_l_values(
        manager_state.disabled_l_values,
        (source_label, m_idx),
        l_keys,
        invalid_key=invalid_key,
    )
    if changed_count > 0:
        apply_filters()
        return True
    return False


def disable_all_bragg_qr_groups(
    *,
    manager_state,
    entries: Sequence[Mapping[str, object]] | None,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Disable every Bragg-Qr group listed in the manager model."""

    keys = [
        entry["key"]
        for entry in entries or ()
        if isinstance(entry, Mapping) and entry.get("key") is not None
    ]
    if not keys:
        return False
    return set_bragg_qr_groups_enabled(
        manager_state=manager_state,
        group_keys=keys,
        enabled=False,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def enable_all_bragg_qr_groups(
    *,
    manager_state,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Enable every Bragg-Qr group currently disabled in the manager state."""

    if not manager_state.disabled_groups:
        refresh_window()
        return False

    manager_state.disabled_groups.clear()
    apply_filters()
    if set_progress_text is not None:
        set_progress_text("Enabled all Bragg Qr groups.")
    return True


def disable_all_bragg_qr_l_values_for_selected_qr(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    build_l_value_map: Callable[[str, int], Mapping[int, object]],
    invalid_key: int,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Disable every available L value for the selected Bragg-Qr group."""

    group_key = selected_primary_bragg_qr_window_key(
        manager_state,
        selected_keys=selected_bragg_qr_window_keys(
            manager_state,
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )
    if group_key is None:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            l_text="Select a Qr group first.",
        )
        return False

    l_value_map = build_l_value_map(group_key[0], int(group_key[1]))
    l_keys = [int(key) for key in l_value_map.keys()]
    if not l_keys:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            l_text="No L values available.",
        )
        return False
    return set_bragg_qr_l_values_enabled(
        view_state=view_state,
        manager_state=manager_state,
        group_key=group_key,
        l_keys=l_keys,
        enabled=False,
        invalid_key=invalid_key,
        refresh_window=refresh_window,
        apply_filters=apply_filters,
        set_progress_text=set_progress_text,
    )


def enable_all_bragg_qr_l_values_for_selected_qr(
    *,
    view_state,
    manager_state,
    qr_listbox: object,
    refresh_window: Callable[[], None],
    apply_filters: Callable[[], None],
    set_progress_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Enable every L value for the selected Bragg-Qr group."""

    group_key = selected_primary_bragg_qr_window_key(
        manager_state,
        selected_keys=selected_bragg_qr_window_keys(
            manager_state,
            qr_listbox,
            tcl_error_types=tcl_error_types,
        ),
    )
    if group_key is None:
        gui_views.set_bragg_qr_manager_status_text(
            view_state,
            l_text="Select a Qr group first.",
        )
        return False

    source_label = gui_controllers.normalize_bragg_qr_source_label(group_key[0])
    m_idx = int(group_key[1])
    changed = gui_controllers.clear_bragg_qr_l_values_for_group(
        manager_state.disabled_l_values,
        (source_label, m_idx),
    )
    if not changed:
        refresh_window()
        return False

    apply_filters()
    if set_progress_text is not None:
        set_progress_text(f"Enabled all L values for {source_label} m={m_idx}.")
    return True


def close_bragg_qr_toggle_window(view_state, manager_state) -> None:
    """Destroy the Bragg-Qr manager window and clear stored selection state."""

    gui_views.close_bragg_qr_manager_window(view_state)
    gui_controllers.clear_bragg_qr_manager_state(manager_state)


def open_bragg_qr_toggle_window(
    *,
    root,
    view_state,
    on_qr_selection_changed: Callable[[object], None],
    on_toggle_qr: Callable[[object], None],
    on_toggle_l: Callable[[object], None],
    on_enable_selected_qr: Callable[[], None],
    on_disable_selected_qr: Callable[[], None],
    on_enable_all_qr: Callable[[], None],
    on_disable_all_qr: Callable[[], None],
    on_enable_selected_l: Callable[[], None],
    on_disable_selected_l: Callable[[], None],
    on_enable_all_l: Callable[[], None],
    on_disable_all_l: Callable[[], None],
    on_refresh: Callable[[], None],
    on_close: Callable[[], None],
) -> bool:
    """Open the Bragg-Qr manager window and trigger an immediate refresh."""

    opened = gui_views.open_bragg_qr_manager_window(
        root=root,
        view_state=view_state,
        on_qr_selection_changed=on_qr_selection_changed,
        on_toggle_qr=on_toggle_qr,
        on_toggle_l=on_toggle_l,
        on_enable_selected_qr=on_enable_selected_qr,
        on_disable_selected_qr=on_disable_selected_qr,
        on_enable_all_qr=on_enable_all_qr,
        on_disable_all_qr=on_disable_all_qr,
        on_enable_selected_l=on_enable_selected_l,
        on_disable_selected_l=on_disable_selected_l,
        on_enable_all_l=on_enable_all_l,
        on_disable_all_l=on_disable_all_l,
        on_refresh=on_refresh,
        on_close=on_close,
    )
    on_refresh()
    return opened


def refresh_runtime_bragg_qr_toggle_window(bindings: BraggQrRuntimeBindings) -> bool:
    """Refresh the runtime Bragg-Qr window using the shared runtime bindings."""

    refreshed = refresh_bragg_qr_toggle_window(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        entries=list(bindings.get_entries() or ()),
        tcl_error_types=bindings.tcl_error_types,
    )
    if not refreshed:
        return False
    on_bragg_qr_selection_changed(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        l_listbox=getattr(bindings.view_state, "l_listbox", None),
        build_l_value_map=bindings.build_l_value_map,
        tcl_error_types=bindings.tcl_error_types,
    )
    return True


def on_runtime_bragg_qr_selection_changed(bindings: BraggQrRuntimeBindings) -> bool:
    """Apply one runtime Bragg-Qr selection-change callback."""

    return on_bragg_qr_selection_changed(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        l_listbox=getattr(bindings.view_state, "l_listbox", None),
        build_l_value_map=bindings.build_l_value_map,
        tcl_error_types=bindings.tcl_error_types,
    )


def _runtime_refresh_window(bindings: BraggQrRuntimeBindings) -> None:
    refresh_runtime_bragg_qr_toggle_window(bindings)


def disable_selected_bragg_qr_groups_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Disable the selected Bragg-Qr groups through the shared runtime bindings."""

    return disable_selected_bragg_qr_groups(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def enable_selected_bragg_qr_groups_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Enable the selected Bragg-Qr groups through the shared runtime bindings."""

    return enable_selected_bragg_qr_groups(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def toggle_selected_bragg_qr_groups_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Toggle the selected Bragg-Qr groups through the shared runtime bindings."""

    return toggle_selected_bragg_qr_groups(
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        apply_filters=bindings.apply_filters,
        tcl_error_types=bindings.tcl_error_types,
    )


def disable_selected_bragg_qr_l_values_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Disable selected Bragg-Qr L values through the shared runtime bindings."""

    return disable_selected_bragg_qr_l_values(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        l_listbox=getattr(bindings.view_state, "l_listbox", None),
        invalid_key=int(bindings.invalid_key),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def enable_selected_bragg_qr_l_values_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Enable selected Bragg-Qr L values through the shared runtime bindings."""

    return enable_selected_bragg_qr_l_values(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        l_listbox=getattr(bindings.view_state, "l_listbox", None),
        invalid_key=int(bindings.invalid_key),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def toggle_selected_bragg_qr_l_values_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Toggle selected Bragg-Qr L values through the shared runtime bindings."""

    return toggle_selected_bragg_qr_l_values(
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        l_listbox=getattr(bindings.view_state, "l_listbox", None),
        invalid_key=int(bindings.invalid_key),
        apply_filters=bindings.apply_filters,
        tcl_error_types=bindings.tcl_error_types,
    )


def disable_all_bragg_qr_groups_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Disable all listed Bragg-Qr groups through the shared runtime bindings."""

    return disable_all_bragg_qr_groups(
        manager_state=bindings.manager_state,
        entries=list(bindings.get_entries() or ()),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
    )


def enable_all_bragg_qr_groups_runtime(bindings: BraggQrRuntimeBindings) -> bool:
    """Enable all Bragg-Qr groups through the shared runtime bindings."""

    return enable_all_bragg_qr_groups(
        manager_state=bindings.manager_state,
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
    )


def disable_all_bragg_qr_l_values_for_selected_qr_runtime(
    bindings: BraggQrRuntimeBindings,
) -> bool:
    """Disable all L values for the selected Bragg-Qr group using runtime bindings."""

    return disable_all_bragg_qr_l_values_for_selected_qr(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        build_l_value_map=bindings.build_l_value_map,
        invalid_key=int(bindings.invalid_key),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def enable_all_bragg_qr_l_values_for_selected_qr_runtime(
    bindings: BraggQrRuntimeBindings,
) -> bool:
    """Enable all L values for the selected Bragg-Qr group using runtime bindings."""

    return enable_all_bragg_qr_l_values_for_selected_qr(
        view_state=bindings.view_state,
        manager_state=bindings.manager_state,
        qr_listbox=getattr(bindings.view_state, "qr_listbox", None),
        refresh_window=lambda: _runtime_refresh_window(bindings),
        apply_filters=bindings.apply_filters,
        set_progress_text=bindings.set_progress_text,
        tcl_error_types=bindings.tcl_error_types,
    )


def close_runtime_bragg_qr_toggle_window(bindings: BraggQrRuntimeBindings) -> None:
    """Close the runtime Bragg-Qr window using the shared runtime bindings."""

    close_bragg_qr_toggle_window(bindings.view_state, bindings.manager_state)


def open_runtime_bragg_qr_toggle_window(
    *,
    root,
    bindings: BraggQrRuntimeBindings,
) -> bool:
    """Open the Bragg-Qr runtime window and wire callbacks from one context."""

    return open_bragg_qr_toggle_window(
        root=root,
        view_state=bindings.view_state,
        on_qr_selection_changed=lambda _event=None: on_runtime_bragg_qr_selection_changed(bindings),
        on_toggle_qr=lambda _event=None: toggle_selected_bragg_qr_groups_runtime(bindings),
        on_toggle_l=lambda _event=None: toggle_selected_bragg_qr_l_values_runtime(bindings),
        on_enable_selected_qr=lambda: enable_selected_bragg_qr_groups_runtime(bindings),
        on_disable_selected_qr=lambda: disable_selected_bragg_qr_groups_runtime(bindings),
        on_enable_all_qr=lambda: enable_all_bragg_qr_groups_runtime(bindings),
        on_disable_all_qr=lambda: disable_all_bragg_qr_groups_runtime(bindings),
        on_enable_selected_l=lambda: enable_selected_bragg_qr_l_values_runtime(bindings),
        on_disable_selected_l=lambda: disable_selected_bragg_qr_l_values_runtime(bindings),
        on_enable_all_l=lambda: enable_all_bragg_qr_l_values_for_selected_qr_runtime(bindings),
        on_disable_all_l=lambda: disable_all_bragg_qr_l_values_for_selected_qr_runtime(bindings),
        on_refresh=lambda: refresh_runtime_bragg_qr_toggle_window(bindings),
        on_close=lambda: close_runtime_bragg_qr_toggle_window(bindings),
    )
