"""Workflow helpers for HKL lookup and selected-peak state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ra_sim.simulation.intersection_analysis import (
    BeamSamples as IntersectionBeamSamples,
    IntersectionGeometry,
    MosaicParams as IntersectionMosaicParams,
    analyze_reflection_intersection,
    plot_intersection_analysis,
)

from . import views as gui_views


@dataclass(frozen=True)
class SelectedPeakIntersectionConfig:
    """Scalar GUI inputs needed to inspect one selected peak intersection."""

    image_size: int
    center_col: float
    center_row: float
    distance_cor_to_detector: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    theta_initial_deg: float
    cor_angle_deg: float
    sigma_mosaic_deg: float
    gamma_mosaic_deg: float
    eta: float
    solve_q_steps: int
    solve_q_rel_tol: float
    solve_q_mode: int
    pixel_size_m: float = 100e-6


@dataclass(frozen=True)
class SelectedPeakCanvasPickConfig:
    """Inputs needed to resolve one HKL image-pick click."""

    image_size: int
    primary_a: float
    primary_c: float
    max_distance_px: float
    min_separation_px: float
    image_shape: tuple[int, ...] | None = None


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


def toggle_hkl_pick_mode(
    simulation_runtime_state,
    peak_selection_state,
    *,
    caked_view_enabled: bool,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    set_pick_mode: Any,
    set_status_text: Any,
) -> None:
    """Arm or disarm HKL image-pick mode based on current GUI state."""

    if peak_selection_state.hkl_pick_armed:
        set_pick_mode(False, message="HKL image-pick canceled.")
        return

    if bool(caked_view_enabled):
        set_status_text("Switch off 2D caked view before picking HKL in the image.")
        return

    if simulation_runtime_state.unscaled_image is None:
        set_status_text("Run a simulation first.")
        return

    if (
        not ensure_peak_overlay_data(force=False)
        or not simulation_runtime_state.peak_positions
    ):
        set_pick_mode(
            True,
            message=(
                "Preparing simulated peak map for HKL picking... "
                "wait for the next update."
            ),
        )
        schedule_update()
        return

    set_pick_mode(
        True,
        message="HKL image-pick armed: click near a Bragg peak in raw camera view.",
    )


def _nearest_peak_index_for_click(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
) -> tuple[int, float]:
    best_i = -1
    best_d2 = float("inf")
    best_i_val = float("-inf")
    for i, (px, py) in enumerate(simulation_runtime_state.peak_positions):
        if float(px) < 0.0:
            continue
        d2 = (float(px) - float(click_col)) ** 2 + (float(py) - float(click_row)) ** 2
        val = simulation_runtime_state.peak_intensities[i]
        score_val = float(val) if np.isfinite(val) else float("-inf")
        if d2 < best_d2 - 1e-9 or (
            abs(d2 - best_d2) <= 1e-9 and score_val > best_i_val
        ):
            best_i = int(i)
            best_d2 = float(d2)
            best_i_val = float(score_val)
    return best_i, best_d2


def _resolve_selected_peak_click_coordinates(
    simulation_runtime_state,
    idx: int,
    *,
    click_col: float,
    click_row: float,
    clicked_native_col: float,
    clicked_native_row: float,
    config: SelectedPeakCanvasPickConfig,
    native_sim_to_display_coords: Any,
    simulate_ideal_hkl_native_center: Any,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    selected_display = None
    selected_native = None

    if idx >= len(simulation_runtime_state.peak_records):
        return selected_display, selected_native

    peak_record = simulation_runtime_state.peak_records[idx]
    if not isinstance(peak_record, Mapping):
        return selected_display, selected_native

    image_shape = (
        tuple(int(v) for v in config.image_shape)
        if config.image_shape is not None
        else (int(config.image_size), int(config.image_size))
    )
    try:
        raw_hkl = peak_record.get("hkl_raw")
        if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
            rec_h = float(raw_hkl[0])
            rec_k = float(raw_hkl[1])
            rec_l = float(raw_hkl[2])
        else:
            rec_h, rec_k, rec_l = tuple(
                float(v) for v in simulation_runtime_state.peak_millers[idx]
            )
        rec_av = float(peak_record.get("av", float(config.primary_a)))
        rec_cv = float(peak_record.get("cv", float(config.primary_c)))
        ideal_native = simulate_ideal_hkl_native_center(
            rec_h,
            rec_k,
            rec_l,
            rec_av,
            rec_cv,
        )
        if ideal_native is not None:
            ideal_display = native_sim_to_display_coords(
                ideal_native[0],
                ideal_native[1],
                image_shape,
            )
            base_display = simulation_runtime_state.peak_positions[idx]
            snap_delta = float(
                np.hypot(
                    float(ideal_display[0]) - float(base_display[0]),
                    float(ideal_display[1]) - float(base_display[1]),
                )
            )
            snap_limit = max(4.0, float(config.min_separation_px) * 2.0)
            if snap_delta <= snap_limit:
                selected_native = (
                    float(ideal_native[0]),
                    float(ideal_native[1]),
                )
                selected_display = (
                    float(ideal_display[0]),
                    float(ideal_display[1]),
                )
    except Exception:
        selected_display = None
        selected_native = None

    if selected_native is None:
        selected_native = (
            float(peak_record.get("native_col", clicked_native_col)),
            float(peak_record.get("native_row", clicked_native_row)),
        )

    return selected_display, selected_native


def select_peak_from_canvas_click(
    simulation_runtime_state,
    peak_selection_state,
    click_col: float,
    click_row: float,
    *,
    config: SelectedPeakCanvasPickConfig,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    display_to_native_sim_coords: Any,
    native_sim_to_display_coords: Any,
    simulate_ideal_hkl_native_center: Any,
    select_peak_by_index: Any,
    set_pick_mode: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
) -> bool:
    """Select the nearest visible peak from one raw-image click."""

    ensure_peak_overlay_data(force=False)
    if not simulation_runtime_state.peak_positions:
        schedule_update()
        set_status_text("Preparing simulated peak map... click again after update.")
        return False

    best_i, best_d2 = _nearest_peak_index_for_click(
        simulation_runtime_state,
        float(click_col),
        float(click_row),
    )
    if best_i == -1:
        set_status_text("No peaks on screen.")
        return False
    if best_d2 > float(config.max_distance_px) ** 2:
        set_status_text(
            f"No simulated peak within {float(config.max_distance_px):.0f}px "
            f"(nearest is {best_d2**0.5:.1f}px away)."
        )
        return False

    cx = int(round(float(click_col)))
    cy = int(round(float(click_row)))
    image_shape = (
        tuple(int(v) for v in config.image_shape)
        if config.image_shape is not None
        else (int(config.image_size), int(config.image_size))
    )
    clicked_native_col, clicked_native_row = display_to_native_sim_coords(
        cx,
        cy,
        image_shape,
    )
    selected_display, selected_native = _resolve_selected_peak_click_coordinates(
        simulation_runtime_state,
        best_i,
        click_col=float(click_col),
        click_row=float(click_row),
        clicked_native_col=float(clicked_native_col),
        clicked_native_row=float(clicked_native_row),
        config=config,
        native_sim_to_display_coords=native_sim_to_display_coords,
        simulate_ideal_hkl_native_center=simulate_ideal_hkl_native_center,
    )

    prefix = f"Nearest peak (Δ={best_d2**0.5:.1f}px)"
    if selected_display is not None:
        snapped_dist = float(
            np.hypot(
                float(selected_display[0]) - float(click_col),
                float(selected_display[1]) - float(click_row),
            )
        )
        prefix = f"Ideal HKL center (click Δ={snapped_dist:.1f}px)"

    picked = bool(
        select_peak_by_index(
            best_i,
            prefix=prefix,
            sync_hkl_vars=True,
            clicked_display=(float(click_col), float(click_row)),
            clicked_native=(float(clicked_native_col), float(clicked_native_row)),
            selected_display=selected_display,
            selected_native=selected_native,
        )
    )
    if picked:
        set_pick_mode(False)
        peak_selection_state.suppress_drag_press_once = True
        sync_peak_selection_state()
    return picked


def open_selected_peak_intersection_figure(
    simulation_runtime_state,
    *,
    config: SelectedPeakIntersectionConfig,
    n2: Any,
    set_status_text: Any,
    geometry_factory: Any = IntersectionGeometry,
    beam_factory: Any = IntersectionBeamSamples,
    mosaic_factory: Any = IntersectionMosaicParams,
    analyze_intersection: Any = analyze_reflection_intersection,
    plot_intersection: Any = plot_intersection_analysis,
) -> bool:
    """Open a Bragg/Ewald intersection analysis plot for the selected peak."""

    selected_peak = simulation_runtime_state.selected_peak_record
    if not isinstance(selected_peak, Mapping):
        set_status_text(
            "Select a Bragg peak first (arm Pick HKL on Image or use HKL controls)."
        )
        return False

    try:
        h, k, l = tuple(int(v) for v in selected_peak["hkl"])
        native_col = float(
            selected_peak.get(
                "selected_native_col",
                selected_peak.get("native_col"),
            )
        )
        native_row = float(
            selected_peak.get(
                "selected_native_row",
                selected_peak.get("native_row"),
            )
        )
        lattice_a = float(selected_peak["av"])
        lattice_c = float(selected_peak["cv"])

        geometry = geometry_factory(
            image_size=int(config.image_size),
            center_col=float(config.center_col),
            center_row=float(config.center_row),
            distance_cor_to_detector=float(config.distance_cor_to_detector),
            gamma_deg=float(config.gamma_deg),
            Gamma_deg=float(config.Gamma_deg),
            chi_deg=float(config.chi_deg),
            psi_deg=float(config.psi_deg),
            psi_z_deg=float(config.psi_z_deg),
            zs=float(config.zs),
            zb=float(config.zb),
            theta_initial_deg=float(config.theta_initial_deg),
            cor_angle_deg=float(config.cor_angle_deg),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            pixel_size_m=float(config.pixel_size_m),
        )

        profile_cache = simulation_runtime_state.profile_cache
        beam = beam_factory(
            beam_x_array=np.asarray(profile_cache["beam_x_array"], dtype=np.float64),
            beam_y_array=np.asarray(profile_cache["beam_y_array"], dtype=np.float64),
            theta_array=np.asarray(profile_cache["theta_array"], dtype=np.float64),
            phi_array=np.asarray(profile_cache["phi_array"], dtype=np.float64),
            wavelength_array=np.asarray(
                profile_cache["wavelength_array"],
                dtype=np.float64,
            ),
        )
        mosaic = mosaic_factory(
            sigma_mosaic_deg=float(config.sigma_mosaic_deg),
            gamma_mosaic_deg=float(config.gamma_mosaic_deg),
            eta=float(config.eta),
            solve_q_steps=int(config.solve_q_steps),
            solve_q_rel_tol=float(config.solve_q_rel_tol),
            solve_q_mode=int(config.solve_q_mode),
        )

        analysis = analyze_intersection(
            h=h,
            k=k,
            l=l,
            lattice_a=lattice_a,
            lattice_c=lattice_c,
            selected_native_col=native_col,
            selected_native_row=native_row,
            geometry=geometry,
            beam=beam,
            mosaic=mosaic,
            n2=n2,
        )
        fig_analysis = plot_intersection(analysis)
        manager = getattr(getattr(fig_analysis, "canvas", None), "manager", None)
        if manager is not None:
            manager.set_window_title(f"Bragg/Ewald HKL=({h},{k},{l})")
            manager.show()
        else:
            fig_analysis.show()

        set_status_text(
            f"Opened Bragg/Ewald analysis for HKL=({h} {k} {l}) "
            f"from source={selected_peak.get('source_label', 'unknown')}."
        )
        return True
    except Exception as exc:
        set_status_text(f"Intersection analysis failed for selected peak: {exc}")
        return False
