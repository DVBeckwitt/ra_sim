"""Workflow helpers for HKL lookup and selected-peak state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import views as gui_views


def hkl_pick_button_text(armed: bool) -> str:
    """Return the current HKL image-pick button label."""

    return "Pick HKL on Image (Armed)" if bool(armed) else "Pick HKL on Image"


def nearest_integer_hkl(h: float, k: float, l: float) -> tuple[int, int, int]:
    """Round one HKL triplet to the nearest integer indices."""

    return (
        int(np.rint(float(h))),
        int(np.rint(float(k))),
        int(np.rint(float(l))),
    )


def format_hkl_triplet(h: int, k: int, l: int) -> str:
    """Return a compact string for one integer HKL triplet."""

    return f"({int(h)} {int(k)} {int(l)})"


def source_miller_for_label(simulation_runtime_state, source_label: str | None) -> np.ndarray:
    """Return the active reflection table for one source label."""

    label = str(source_label or "primary").lower()
    if (
        label == "secondary"
        and isinstance(simulation_runtime_state.sim_miller2, np.ndarray)
        and simulation_runtime_state.sim_miller2.size
    ):
        return np.asarray(simulation_runtime_state.sim_miller2, dtype=float)
    if (
        isinstance(simulation_runtime_state.sim_miller1, np.ndarray)
        and simulation_runtime_state.sim_miller1.size
    ):
        return np.asarray(simulation_runtime_state.sim_miller1, dtype=float)
    return np.empty((0, 3), dtype=float)


def degenerate_hkls_for_qr(
    simulation_runtime_state,
    h: int,
    k: int,
    l: int,
    *,
    source_label: str | None,
) -> list[tuple[int, int, int]]:
    """Return all integer HKLs that share the same Qr rod and nearest L."""

    source_miller = source_miller_for_label(simulation_runtime_state, source_label)
    if (
        source_miller.ndim != 2
        or source_miller.shape[1] < 3
        or source_miller.shape[0] == 0
    ):
        return []

    finite = (
        np.isfinite(source_miller[:, 0])
        & np.isfinite(source_miller[:, 1])
        & np.isfinite(source_miller[:, 2])
    )
    if not np.any(finite):
        return []
    source_int = np.rint(source_miller[finite, :3]).astype(np.int64, copy=False)

    h_vals = source_int[:, 0]
    k_vals = source_int[:, 1]
    l_vals = source_int[:, 2]
    m_vals = h_vals * h_vals + h_vals * k_vals + k_vals * k_vals
    m_target = int(h * h + h * k + k * k)

    mask = (l_vals == int(l)) & (m_vals == m_target)
    if not np.any(mask):
        rod_mask = m_vals == m_target
        if not np.any(rod_mask):
            return []
        nearest_l = int(
            l_vals[rod_mask][np.argmin(np.abs(l_vals[rod_mask] - int(l)))]
        )
        mask = rod_mask & (l_vals == nearest_l)

    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for hh, kk, ll in source_int[mask]:
        key = (int(hh), int(kk), int(ll))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    out.sort(key=lambda vals: (vals[0], vals[1], vals[2]))
    return out


def selected_peak_qr_and_degenerates(
    simulation_runtime_state,
    H: float,
    K: float,
    L: float,
    selected_peak: Mapping[str, object] | None,
    *,
    primary_a: float,
) -> tuple[float, list[tuple[int, int, int]]]:
    """Return the selected peak's Qr value and degenerate HKL set."""

    h_raw, k_raw, l_raw = float(H), float(K), float(L)
    source_label = "primary"
    a_used = float(primary_a)
    if isinstance(selected_peak, Mapping):
        raw_hkl = selected_peak.get("hkl_raw")
        if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
            h_raw = float(raw_hkl[0])
            k_raw = float(raw_hkl[1])
            l_raw = float(raw_hkl[2])
        source_label = str(selected_peak.get("source_label", "primary"))
        try:
            a_used = float(selected_peak.get("av", primary_a))
        except (TypeError, ValueError):
            a_used = float(primary_a)

    h_int, k_int, l_int = nearest_integer_hkl(h_raw, k_raw, l_raw)
    m_val = float(h_int * h_int + h_int * k_int + k_int * k_int)
    if a_used > 0.0 and np.isfinite(a_used) and m_val >= 0.0:
        qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
    else:
        qr_val = float("nan")

    deg_hkls = degenerate_hkls_for_qr(
        simulation_runtime_state,
        h_int,
        k_int,
        l_int,
        source_label=source_label,
    )
    if not deg_hkls:
        deg_hkls = [(h_int, k_int, l_int)]
    return float(qr_val), deg_hkls


def build_selected_peak_status_text(
    *,
    prefix: str,
    h: int,
    k: int,
    l: int,
    display_col: float,
    display_row: float,
    intensity: float,
    qr_val: float,
    deg_hkls: list[tuple[int, int, int]],
) -> str:
    """Build the GUI status text for one selected Bragg peak."""

    shown_deg = deg_hkls[:12]
    deg_text = ", ".join(
        format_hkl_triplet(hv, kv, lv) for hv, kv, lv in shown_deg
    )
    if len(deg_hkls) > len(shown_deg):
        deg_text += f", ... (+{len(deg_hkls) - len(shown_deg)} more)"
    qr_text = f"  Qr={qr_val:.4f} A^-1" if np.isfinite(qr_val) else ""
    return (
        f"{prefix}: HKL=({int(h)} {int(k)} {int(l)})  "
        f"pixel=({float(display_col):.1f},{float(display_row):.1f})  "
        f"I={float(intensity):.2g}{qr_text}  HKLs@same Qr,L: {deg_text}"
    )


def _copy_selected_peak_record(
    simulation_runtime_state,
    idx: int,
) -> dict[str, object] | None:
    if idx >= len(simulation_runtime_state.peak_records):
        return None
    raw_record = simulation_runtime_state.peak_records[idx]
    if not isinstance(raw_record, Mapping):
        return None
    return dict(raw_record)


def _apply_selected_peak_record_coordinates(
    record: dict[str, object] | None,
    *,
    clicked_display: tuple[float, float] | None = None,
    clicked_native: tuple[float, float] | None = None,
    selected_display: tuple[float, float] | None = None,
    selected_native: tuple[float, float] | None = None,
) -> dict[str, object] | None:
    if record is None:
        return None
    if clicked_display is not None:
        record["clicked_display_col"] = float(clicked_display[0])
        record["clicked_display_row"] = float(clicked_display[1])
    if clicked_native is not None:
        record["clicked_native_col"] = float(clicked_native[0])
        record["clicked_native_row"] = float(clicked_native[1])
    if selected_display is not None:
        record["selected_display_col"] = float(selected_display[0])
        record["selected_display_row"] = float(selected_display[1])
    if selected_native is not None:
        record["selected_native_col"] = float(selected_native[0])
        record["selected_native_row"] = float(selected_native[1])
    elif clicked_native is not None:
        record["selected_native_col"] = float(clicked_native[0])
        record["selected_native_row"] = float(clicked_native[1])
    else:
        record["selected_native_col"] = float(record["native_col"])
        record["selected_native_row"] = float(record["native_row"])
    return record


def select_peak_by_index(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    idx: int,
    *,
    primary_a: float,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    prefix: str = "Selected peak",
    sync_hkl_vars: bool = True,
    clicked_display: tuple[float, float] | None = None,
    clicked_native: tuple[float, float] | None = None,
    selected_display: tuple[float, float] | None = None,
    selected_native: tuple[float, float] | None = None,
) -> bool:
    """Select one simulated peak by cached index and update GUI state."""

    if idx < 0 or idx >= len(simulation_runtime_state.peak_positions):
        return False

    px, py = simulation_runtime_state.peak_positions[idx]
    H, K, L = simulation_runtime_state.peak_millers[idx]
    intensity = simulation_runtime_state.peak_intensities[idx]
    disp_col, disp_row = (
        (float(selected_display[0]), float(selected_display[1]))
        if selected_display is not None
        else (float(px), float(py))
    )

    selected_peak_marker.set_data([disp_col], [disp_row])
    selected_peak_marker.set_visible(True)

    peak_selection_state.selected_hkl_target = (int(H), int(K), int(L))
    sync_peak_selection_state()

    selected_record = _copy_selected_peak_record(simulation_runtime_state, idx)
    selected_record = _apply_selected_peak_record_coordinates(
        selected_record,
        clicked_display=clicked_display,
        clicked_native=clicked_native,
        selected_display=selected_display,
        selected_native=selected_native,
    )
    simulation_runtime_state.selected_peak_record = selected_record

    if sync_hkl_vars:
        gui_views.set_hkl_lookup_values(
            hkl_lookup_view_state,
            h_text=str(int(H)),
            k_text=str(int(K)),
            l_text=str(int(L)),
        )

    qr_val, deg_hkls = selected_peak_qr_and_degenerates(
        simulation_runtime_state,
        H,
        K,
        L,
        simulation_runtime_state.selected_peak_record,
        primary_a=primary_a,
    )
    if simulation_runtime_state.selected_peak_record is not None:
        simulation_runtime_state.selected_peak_record["qr"] = float(qr_val)
        simulation_runtime_state.selected_peak_record["degenerate_hkls"] = [
            (int(hv), int(kv), int(lv)) for hv, kv, lv in deg_hkls
        ]

    set_status_text(
        build_selected_peak_status_text(
            prefix=prefix,
            h=int(H),
            k=int(K),
            l=int(L),
            display_col=disp_col,
            display_row=disp_row,
            intensity=float(intensity),
            qr_val=float(qr_val),
            deg_hkls=deg_hkls,
        )
    )
    draw_idle()
    return True


def _m_index(hkl_triplet: tuple[int, int, int]) -> int:
    h0, k0, _l0 = hkl_triplet
    return int(h0 * h0 + h0 * k0 + k0 * k0)


def select_peak_by_hkl(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    h: int,
    k: int,
    l: int,
    *,
    primary_a: float,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
) -> bool:
    """Select the simulated peak matching one requested integer HKL."""

    ensure_peak_overlay_data(force=False)
    target = (int(h), int(k), int(l))

    if not simulation_runtime_state.peak_positions:
        if (
            not silent_if_missing
            and simulation_runtime_state.unscaled_image is not None
        ):
            schedule_update()
        if not silent_if_missing:
            set_status_text("Preparing simulated peak map... try again after update.")
        return False

    matches = [
        idx
        for idx, hkl_triplet in enumerate(simulation_runtime_state.peak_millers)
        if (
            tuple(int(np.rint(v)) for v in hkl_triplet) == target
            and simulation_runtime_state.peak_positions[idx][0] >= 0
        )
    ]

    if not matches:
        m_target = _m_index(target)
        l_target = int(target[2])
        matches = [
            idx
            for idx, hkl_triplet in enumerate(simulation_runtime_state.peak_millers)
            if (
                simulation_runtime_state.peak_positions[idx][0] >= 0
                and int(np.rint(hkl_triplet[2])) == l_target
                and _m_index(
                    (
                        int(np.rint(hkl_triplet[0])),
                        int(np.rint(hkl_triplet[1])),
                        int(np.rint(hkl_triplet[2])),
                    )
                )
                == m_target
            )
        ]

    if not matches:
        if not silent_if_missing:
            set_status_text(
                f"HKL ({target[0]} {target[1]} {target[2]}) not found in current simulation."
            )
        peak_selection_state.selected_hkl_target = target
        sync_peak_selection_state()
        simulation_runtime_state.selected_peak_record = None
        return False

    def _score(idx_value: int) -> float:
        val = simulation_runtime_state.peak_intensities[idx_value]
        return float(val) if np.isfinite(val) else float("-inf")

    best_idx = max(matches, key=_score)
    return select_peak_by_index(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        best_idx,
        primary_a=primary_a,
        sync_peak_selection_state=sync_peak_selection_state,
        set_status_text=set_status_text,
        draw_idle=draw_idle,
        prefix="Selected peak",
        sync_hkl_vars=sync_hkl_vars,
    )


def selected_hkl_from_lookup_controls(
    hkl_lookup_view_state,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> tuple[int, int, int] | None:
    """Read one integer HKL target from the lookup entry vars."""

    try:
        h = int(round(float(hkl_lookup_view_state.selected_h_var.get().strip())))
        k = int(round(float(hkl_lookup_view_state.selected_k_var.get().strip())))
        l = int(round(float(hkl_lookup_view_state.selected_l_var.get().strip())))
    except Exception as exc:
        if tcl_error_types and isinstance(exc, tcl_error_types):
            return None
        if isinstance(exc, (ValueError, AttributeError)):
            return None
        raise
    return (h, k, l)


def select_peak_from_hkl_controls(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    *,
    primary_a: float,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Select one peak from the HKL lookup entry controls."""

    target = selected_hkl_from_lookup_controls(
        hkl_lookup_view_state,
        tcl_error_types=tcl_error_types,
    )
    if target is None:
        set_status_text("Enter numeric H, K, L values.")
        return False

    peak_selection_state.selected_hkl_target = target
    sync_peak_selection_state()
    return select_peak_by_hkl(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        int(target[0]),
        int(target[1]),
        int(target[2]),
        primary_a=primary_a,
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        schedule_update=schedule_update,
        sync_peak_selection_state=sync_peak_selection_state,
        set_status_text=set_status_text,
        draw_idle=draw_idle,
        sync_hkl_vars=True,
        silent_if_missing=False,
    )


def clear_selected_peak(
    simulation_runtime_state,
    peak_selection_state,
    selected_peak_marker: object,
    *,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
) -> None:
    """Clear the current selected-peak state and marker."""

    peak_selection_state.selected_hkl_target = None
    sync_peak_selection_state()
    simulation_runtime_state.selected_peak_record = None
    selected_peak_marker.set_visible(False)
    set_status_text("Peak selection cleared.")
    draw_idle()
