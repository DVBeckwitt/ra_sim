"""Overlay artist helpers for the GUI runtime."""

from __future__ import annotations

import math
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .geometry_overlay import normalize_initial_geometry_pairs_display

LOGGER = logging.getLogger(__name__)
GEOMETRY_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT_ENV = (
    "RA_SIM_GEOM_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT"
)


def clear_artists(
    artists: list[object],
    *,
    draw_idle: Callable[[], None] | None = None,
    redraw: bool = True,
) -> None:
    """Remove artists from the plot and clear the backing list."""

    for artist in list(artists):
        try:
            artist.remove()
        except ValueError:
            pass
    artists.clear()
    if redraw and draw_idle is not None:
        draw_idle()


def _parse_point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        col = float(value[0])
        row = float(value[1])
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _overlay_env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _overlay_value_is_true(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "dynamic"}


def _arrow_semantics_status(entry: Mapping[str, object]) -> str:
    final_native_source = str(entry.get("final_sim_native_source", "") or "")
    final_display_source = str(entry.get("final_sim_display_source", "") or "")
    final_prediction_source = str(entry.get("final_prediction_source", "") or "")
    prediction_source = str(entry.get("fit_prediction_source", "") or "")
    is_dynamic = _overlay_value_is_true(entry.get("fit_prediction_is_dynamic"))
    dynamic_source = bool(
        "dynamic_final_forward_simulation" in final_native_source
        or "dynamic_final_forward_simulation" in final_display_source
        or "dynamic_final_forward_simulation" in final_prediction_source
        or final_display_source.startswith("dynamic_final")
        or final_native_source.startswith("dynamic_final")
    )
    if is_dynamic or dynamic_source:
        return "dynamic_final_prediction"
    if _overlay_value_is_true(entry.get("stale_final_sim")):
        return "stale_prediction"
    prediction_source_lower = prediction_source.lower()
    if "locked_manual_qr" in prediction_source_lower or "saved" in prediction_source_lower:
        return "locked_saved_prediction"
    return "missing_dynamic_source"


def _normalize_q_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return None


def _normalize_hkl_key(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    try:
        return (
            int(value[0]),
            int(value[1]),
            int(value[2]),
        )
    except Exception:
        return None


def _line_group_ids_for_entry(entry: dict[str, object]) -> tuple[tuple[object, ...], ...]:
    group_ids: list[tuple[object, ...]] = []
    q_group_key = _normalize_q_group_key(entry.get("q_group_key"))
    if q_group_key is not None:
        group_ids.append(("q_group_line",) + q_group_key)

    hkl_key = _normalize_hkl_key(entry.get("hkl"))
    if hkl_key is not None and hkl_key[0] == 0 and hkl_key[1] == 0:
        source_label = (
            str(q_group_key[1])
            if isinstance(q_group_key, tuple) and len(q_group_key) >= 2
            else "primary"
        )
        group_ids.append(("qz_axis_line", str(source_label)))
    return tuple(group_ids)


def _fit_overlay_line_segment(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 2:
        return None
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] < 2:
        return None

    centroid = np.mean(arr, axis=0)
    centered = arr - centroid
    if arr.shape[0] == 2:
        direction = centered[1] - centered[0]
    else:
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if vh.size < 2:
            return None
        direction = vh[0]
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1.0e-12:
        return None
    direction = direction / direction_norm
    projections = centered @ direction
    proj_min = float(np.min(projections))
    proj_max = float(np.max(projections))
    if not (np.isfinite(proj_min) and np.isfinite(proj_max)):
        return None
    if abs(proj_max - proj_min) <= 1.0e-9:
        return None
    start = centroid + proj_min * direction
    end = centroid + proj_max * direction
    return (
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1])),
    )


def _draw_initial_q_group_lines(
    ax: Any,
    initial_pairs: Sequence[dict[str, object]],
    *,
    geometry_pick_artists: list[object],
) -> None:
    sim_points_by_group: dict[tuple[object, ...], list[tuple[float, float]]] = {}
    bg_points_by_group: dict[tuple[object, ...], list[tuple[float, float]]] = {}

    for entry in initial_pairs:
        if not isinstance(entry, dict):
            continue
        group_ids = _line_group_ids_for_entry(entry)
        if not group_ids:
            continue
        sim_display = _parse_point(entry.get("sim_display"))
        bg_display = _parse_point(entry.get("bg_display"))
        for group_id in group_ids:
            if sim_display is not None:
                sim_points_by_group.setdefault(group_id, []).append(sim_display)
            if bg_display is not None:
                bg_points_by_group.setdefault(group_id, []).append(bg_display)

    for grouped_points, color, linestyle in (
        (sim_points_by_group, "#74b9ff", "--"),
        (bg_points_by_group, "#fdcb6e", "-."),
    ):
        for points in grouped_points.values():
            segment = _fit_overlay_line_segment(points)
            if segment is None:
                continue
            (line,) = ax.plot(
                [float(segment[0][0]), float(segment[1][0])],
                [float(segment[0][1]), float(segment[1][1])],
                color=color,
                linestyle=linestyle,
                linewidth=1.15,
                alpha=0.72,
                zorder=5,
                solid_capstyle="round",
            )
            geometry_pick_artists.append(line)


def _plot_pair_label_text(
    ax: Any,
    label: str,
    *,
    start_xy: tuple[float, float] | None,
    end_xy: tuple[float, float] | None,
    color: str,
    fontsize: float = 8.0,
    alpha: float = 1.0,
    bbox_alpha: float = 0.82,
    zorder: int = 6,
) -> object | None:
    if start_xy is not None and end_xy is not None:
        anchor = (
            0.5 * float(start_xy[0] + end_xy[0]),
            0.5 * float(start_xy[1] + end_xy[1]),
        )
    elif end_xy is not None:
        anchor = (float(end_xy[0]), float(end_xy[1]))
    elif start_xy is not None:
        anchor = (float(start_xy[0]), float(start_xy[1]))
    else:
        return None
    return ax.text(
        float(anchor[0]),
        float(anchor[1]),
        label,
        color=color,
        fontsize=float(fontsize),
        alpha=float(alpha),
        ha="center",
        va="center",
        zorder=zorder,
        bbox=dict(
            facecolor="white",
            alpha=float(bbox_alpha),
            edgecolor="none",
            pad=1.0,
        ),
    )


def _format_pair_identity_label(
    entry: Mapping[str, object] | None,
    fallback_label: str,
) -> str:
    """Return the visible pair label, appending cached Qr/Qz when available."""

    label = str(fallback_label)
    if not isinstance(entry, Mapping):
        return label

    q_parts: list[str] = []
    for field_name, prefix in (("qr", "Qr"), ("qz", "Qz")):
        try:
            value = float(entry.get(field_name, np.nan))
        except Exception:
            value = float("nan")
        if np.isfinite(value):
            q_parts.append(f"{prefix}={value:.2f}")
    if not q_parts:
        return label
    q_text = "  ".join(q_parts)
    if not label:
        return q_text
    return f"{label}\n{q_text}"


def _format_q_group_identity_label(entry: Mapping[str, object] | None) -> str:
    """Return a compact one-per-Qr-set label without per-HKL text."""

    label = "Qr set"
    q_group_key = _normalize_q_group_key(
        entry.get("q_group_key") if isinstance(entry, Mapping) else None
    )
    if isinstance(q_group_key, tuple) and len(q_group_key) >= 4 and q_group_key[0] == "q_group":
        label = f"Qr set {q_group_key[2]}/{q_group_key[3]}"
    return _format_pair_identity_label(entry, label)


def draw_geometry_fit_overlay(
    ax: Any,
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    geometry_pick_artists: list[object],
    clear_geometry_pick_artists: Callable[..., None],
    draw_idle: Callable[[], None],
    max_display_markers: int = 120,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    visual_probe_records: list[dict[str, object]] | None = None,
    draw_audit_records: list[dict[str, object]] | None = None,
) -> None:
    """Draw one fixed-background/fitted-simulation overlay record per match."""

    clear_geometry_pick_artists(redraw=False)

    def _resolve_display_point(
        entry: dict[str, object],
        *,
        display_key: str,
        native_key: str,
        caked_projection_audit: dict[str, dict[str, object]] | None = None,
    ) -> tuple[float, float] | None:
        if show_caked_2d:
            caked_display_key = display_key.replace("_display", "_caked_display")
            caked_point = _parse_point(entry.get(caked_display_key))
            if caked_point is not None:
                if caked_projection_audit is not None:
                    caked_projection_audit[display_key] = {
                        "caked_projection_input_native": None,
                        "caked_projection_input_source": caked_display_key,
                        "caked_projection_output": caked_point,
                        "caked_projection_valid": True,
                    }
                return caked_point
            native_point = _parse_point(entry.get(native_key))
            if (
                native_point is not None
                and native_detector_coords_to_caked_display_coords is not None
            ):
                try:
                    projected = native_detector_coords_to_caked_display_coords(
                        float(native_point[0]),
                        float(native_point[1]),
                    )
                except Exception:
                    projected = None
                projected_point = _parse_point(projected)
                if caked_projection_audit is not None:
                    caked_projection_audit[display_key] = {
                        "caked_projection_input_native": native_point,
                        "caked_projection_input_source": native_key,
                        "caked_projection_output": projected_point,
                        "caked_projection_valid": projected_point is not None,
                    }
                if projected_point is not None:
                    return projected_point
            elif caked_projection_audit is not None:
                caked_projection_audit[display_key] = {
                    "caked_projection_input_native": native_point,
                    "caked_projection_input_source": native_key,
                    "caked_projection_output": None,
                    "caked_projection_valid": False,
                }
        return _parse_point(entry.get(display_key))

    def _plot_marker(
        col: float,
        row: float,
        label: str | None,
        color: str,
        marker: str,
        *,
        zorder: int = 7,
    ) -> object:
        (point,) = ax.plot(
            [float(col)],
            [float(row)],
            marker,
            color=color,
            markersize=8,
            markerfacecolor="none",
            zorder=zorder,
            linestyle="None",
        )
        geometry_pick_artists.append(point)
        if label:
            text = ax.text(
                float(col),
                float(row),
                label,
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
                zorder=zorder + 1,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
            )
            geometry_pick_artists.append(text)
        return point

    def _artist_point(marker_artist: object) -> tuple[float, float] | None:
        try:
            xdata = list(marker_artist.get_xdata())
            ydata = list(marker_artist.get_ydata())
        except Exception:
            return None
        if not xdata or not ydata:
            return None
        try:
            return (float(xdata[0]), float(ydata[0]))
        except Exception:
            return None

    def _plot_arrow(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        color: str,
        linestyle: str = "--",
        lw: float = 1.0,
        alpha: float = 0.8,
        annotate: str | None = None,
    ) -> None:
        arrow = ax.annotate(
            annotate or "",
            xy=end_xy,
            xytext=start_xy,
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                linestyle=linestyle,
                alpha=alpha,
            ),
            bbox=(
                dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0) if annotate else None
            ),
            zorder=6,
        )
        geometry_pick_artists.append(arrow)

    def _plot_initial_link(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        label: str,
    ) -> None:
        link = ax.annotate(
            label,
            xy=end_xy,
            xytext=start_xy,
            color="#636e72",
            fontsize=8,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color="#636e72",
                lw=1.0,
                linestyle=":",
                alpha=0.8,
            ),
            zorder=6,
        )
        geometry_pick_artists.append(link)

    limit = max(1, int(max_display_markers))
    for idx, entry in enumerate(overlay_records or []):
        if idx >= limit or not isinstance(entry, dict):
            break

        caked_projection_audit: dict[str, dict[str, object]] = {}
        initial_sim_display = _resolve_display_point(
            entry,
            display_key="initial_sim_display",
            native_key="initial_sim_native",
            caked_projection_audit=caked_projection_audit,
        )
        initial_bg_display = _resolve_display_point(
            entry,
            display_key="initial_bg_display",
            native_key="initial_bg_native",
            caked_projection_audit=caked_projection_audit,
        )
        final_sim_display = _resolve_display_point(
            entry,
            display_key="final_sim_display",
            native_key="final_sim_native",
            caked_projection_audit=caked_projection_audit,
        )
        final_bg_display = _resolve_display_point(
            entry,
            display_key="final_bg_display",
            native_key="final_bg_native",
            caked_projection_audit=caked_projection_audit,
        )

        label = _format_pair_identity_label(
            entry,
            str(entry.get("hkl", entry.get("label", "match"))),
        )
        if initial_sim_display is not None:
            _plot_marker(
                initial_sim_display[0],
                initial_sim_display[1],
                None,
                "#0984e3",
                "s",
                zorder=7,
            )

        if initial_bg_display is not None:
            _plot_marker(
                initial_bg_display[0],
                initial_bg_display[1],
                None,
                "#f39c12",
                "^",
                zorder=7,
            )

        status = str(entry.get("match_status", "matched")).strip().lower()
        matched = (
            status == "matched" and final_sim_display is not None and final_bg_display is not None
        )
        arrow_semantics_status = _arrow_semantics_status(entry)
        draw_audit_record: dict[str, object] | None = None
        if draw_audit_records is not None:
            draw_audit_record = {
                "overlay_match_index": entry.get("overlay_match_index", idx),
                "blue_square_point": initial_sim_display,
                "blue_square_source": str(
                    entry.get("chosen_initial_sim_display_source", "initial_sim_display")
                ),
                "amber_triangle_point": initial_bg_display,
                "green_circle_point": final_sim_display,
                "green_circle_source": str(
                    entry.get(
                        "final_sim_caked_display_source"
                        if show_caked_2d
                        else "final_sim_display_source",
                        "",
                    )
                ),
                "dashed_arrow_start": None,
                "dashed_arrow_end": None,
                "dashed_arrow_length": float("nan"),
                "match_status": status,
                "fit_prediction_source": str(entry.get("fit_prediction_source", "")),
                "fit_prediction_is_dynamic": entry.get("fit_prediction_is_dynamic"),
                "final_sim_display_source": str(entry.get("final_sim_display_source", "")),
                "final_sim_native_source": str(entry.get("final_sim_native_source", "")),
                "arrow_semantics_status": arrow_semantics_status,
                "green_circle_semantics_status": arrow_semantics_status,
                "dashed_arrow_suppressed_by_diagnostic_flag": False,
                "suppressed_stale_arrow": False,
                "stale_arrow_drawn": False,
                "caked_projection_audit": dict(caked_projection_audit),
            }
        if not matched:
            if initial_sim_display is not None and initial_bg_display is not None:
                _plot_initial_link(
                    initial_sim_display,
                    initial_bg_display,
                    label=label,
                )
            if draw_audit_record is not None:
                draw_audit_records.append(draw_audit_record)
            continue

        sim_shift = (
            math.hypot(
                final_sim_display[0] - initial_sim_display[0],
                final_sim_display[1] - initial_sim_display[1],
            )
            if initial_sim_display is not None
            else 0.0
        )
        draw_fitted_shift_arrow = bool(
            initial_sim_display is not None
            and sim_shift > 0.25
            and arrow_semantics_status == "dynamic_final_prediction"
        )
        suppress_stale_arrow = bool(
            initial_sim_display is not None
            and sim_shift > 0.25
            and arrow_semantics_status != "dynamic_final_prediction"
        )
        diagnostic_flag_suppressed_arrow = bool(
            suppress_stale_arrow
            and _overlay_env_flag_enabled(GEOMETRY_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT_ENV)
        )
        if suppress_stale_arrow:
            if diagnostic_flag_suppressed_arrow:
                LOGGER.info("%s active", GEOMETRY_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT_ENV)
            if draw_audit_record is not None:
                draw_audit_record["dashed_arrow_suppressed_by_diagnostic_flag"] = (
                    diagnostic_flag_suppressed_arrow
                )
                draw_audit_record["suppressed_stale_arrow"] = True
        if draw_fitted_shift_arrow:
            if draw_audit_record is not None and initial_sim_display is not None:
                draw_audit_record["dashed_arrow_start"] = initial_sim_display
                draw_audit_record["dashed_arrow_end"] = final_sim_display
                draw_audit_record["dashed_arrow_length"] = float(sim_shift)
            _plot_arrow(
                initial_sim_display,
                final_sim_display,
                color="#0984e3",
                linestyle="--",
                lw=1.0,
                alpha=0.85,
            )
        elif suppress_stale_arrow and draw_audit_record is not None:
            draw_audit_record["stale_arrow_drawn"] = False

        final_sim_label_suffix = (
            "fit sim"
            if arrow_semantics_status == "dynamic_final_prediction"
            else "diagnostic sim"
        )
        fit_sim_artist = _plot_marker(
            final_sim_display[0],
            final_sim_display[1],
            f"{label} {final_sim_label_suffix}",
            "#00b894",
            "o",
            zorder=8,
        )
        if visual_probe_records is not None:
            identity_fields = (
                "dataset_index",
                "pair_id",
                "q_group_key",
                "hkl",
                "source_branch_index",
                "source_table_index",
                "source_row_index",
                "source_peak_index",
            )
            visual_probe_record = {field: entry.get(field) for field in identity_fields}
            visual_probe_record.update(
                {
                    "overlay_match_index": entry.get("overlay_match_index", idx),
                    "display_mode": "caked" if show_caked_2d else "detector",
                    "record_point": (
                        float(final_sim_display[0]),
                        float(final_sim_display[1]),
                    ),
                    "artist_point": _artist_point(fit_sim_artist),
                    "record_source": str(
                        entry.get(
                            "final_sim_caked_display_source"
                            if show_caked_2d
                            else "final_sim_display_source",
                            "",
                        )
                    ),
                    "fit_prediction_source": str(entry.get("fit_prediction_source", "")),
                }
            )
            visual_probe_records.append(visual_probe_record)
        if draw_audit_record is not None:
            draw_audit_records.append(draw_audit_record)

        residual_dist = math.hypot(
            final_sim_display[0] - final_bg_display[0],
            final_sim_display[1] - final_bg_display[1],
        )
        residual_units = "deg" if show_caked_2d else "px"
        _plot_arrow(
            final_sim_display,
            final_bg_display,
            color="#2d3436",
            linestyle="-",
            lw=1.1,
            alpha=0.9,
            annotate=f"|Δ|={residual_dist:.1f}{residual_units}",
        )

    draw_idle()


def draw_initial_geometry_pairs_overlay(
    ax: Any,
    initial_pairs_display: Sequence[dict[str, object]] | None,
    *,
    geometry_pick_artists: list[object],
    clear_geometry_pick_artists: Callable[..., None],
    draw_idle: Callable[[], None],
    max_display_markers: int = 120,
    show_pair_connectors: bool = True,
) -> None:
    """Draw only the initially selected simulation/background peak pairs."""

    initial_pairs = normalize_initial_geometry_pairs_display(initial_pairs_display)
    clear_geometry_pick_artists(redraw=False)
    _draw_initial_q_group_lines(
        ax,
        initial_pairs,
        geometry_pick_artists=geometry_pick_artists,
    )

    limit = max(1, int(max_display_markers))
    labeled_q_group_keys: set[tuple[object, ...]] = set()
    for idx, entry in enumerate(initial_pairs):
        if idx >= limit:
            break
        sim_display = entry.get("sim_display")
        bg_display = entry.get("bg_display")
        if sim_display is None and bg_display is None:
            continue
        q_group_key = _normalize_q_group_key(entry.get("q_group_key"))
        pair_label = _format_pair_identity_label(
            entry,
            str(entry.get("hkl", entry.get("label", idx))),
        )
        q_group_label = _format_q_group_identity_label(entry) if q_group_key is not None else ""

        if sim_display is not None:
            (sim_pt,) = ax.plot(
                [float(sim_display[0])],
                [float(sim_display[1])],
                "s",
                color="#0984e3",
                markersize=8,
                markerfacecolor="none",
                linestyle="None",
                zorder=7,
            )
            geometry_pick_artists.append(sim_pt)
        if bg_display is not None:
            (bg_pt,) = ax.plot(
                [float(bg_display[0])],
                [float(bg_display[1])],
                "^",
                color="#f39c12",
                markersize=8,
                markerfacecolor="none",
                linestyle="None",
                zorder=7,
            )
            geometry_pick_artists.append(bg_pt)

        if sim_display is not None and bg_display is not None and show_pair_connectors:
            label_text = "" if q_group_key is not None else pair_label
            link = ax.annotate(
                label_text,
                xy=(float(bg_display[0]), float(bg_display[1])),
                xytext=(float(sim_display[0]), float(sim_display[1])),
                color="#636e72",
                fontsize=8,
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#636e72",
                    lw=1.0,
                    linestyle=":",
                    alpha=0.8,
                ),
                zorder=6,
            )
            geometry_pick_artists.append(link)
        elif q_group_key is None:
            label = _plot_pair_label_text(
                ax,
                pair_label,
                start_xy=sim_display,
                end_xy=bg_display,
                color="#636e72",
                zorder=6,
            )
            if label is not None:
                geometry_pick_artists.append(label)

        if q_group_key is not None and q_group_key not in labeled_q_group_keys:
            labeled_q_group_keys.add(q_group_key)
            label = _plot_pair_label_text(
                ax,
                q_group_label,
                start_xy=sim_display,
                end_xy=bg_display,
                color="#2d3436",
                fontsize=6.5,
                alpha=0.58,
                bbox_alpha=0.3,
                zorder=6,
            )
            if label is not None:
                geometry_pick_artists.append(label)

    draw_idle()


def draw_live_geometry_preview_overlay(
    ax: Any,
    matched_pairs: Sequence[dict[str, object]] | None,
    *,
    geometry_preview_artists: list[object],
    clear_geometry_preview_artists: Callable[..., None],
    draw_idle: Callable[[], None],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None],
    live_preview_match_is_excluded: Callable[[dict[str, object]], bool],
    max_display_markers: int = 120,
    show_pair_connectors: bool = True,
) -> None:
    """Draw the current live auto-match preview without disturbing fit markers."""

    clear_geometry_preview_artists(redraw=False)

    limit = max(1, int(max_display_markers))
    for idx, entry in enumerate(matched_pairs or []):
        if idx >= limit:
            break
        hkl_key = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl_key is None:
            continue
        try:
            sim_col = float(entry["sim_x"])
            sim_row = float(entry["sim_y"])
            bg_col = float(entry["x"])
            bg_row = float(entry["y"])
        except Exception:
            continue
        if not all(np.isfinite(v) for v in (sim_col, sim_row, bg_col, bg_row)):
            continue
        excluded = live_preview_match_is_excluded(entry)
        sim_color = "#b2bec3" if excluded else "#0984e3"
        bg_color = "#636e72" if excluded else "#f39c12"
        line_color = "#95a5a6" if excluded else "#636e72"
        line_alpha = 0.35 if excluded else 0.85
        label_text = _format_pair_identity_label(
            entry,
            f"{hkl_key} excluded" if excluded else f"{hkl_key}",
        )

        (sim_pt,) = ax.plot(
            [float(sim_col)],
            [float(sim_row)],
            "s",
            color=sim_color,
            markersize=8,
            markerfacecolor="none",
            linestyle="None",
            zorder=5,
            alpha=line_alpha,
        )
        (bg_pt,) = ax.plot(
            [float(bg_col)],
            [float(bg_row)],
            "^",
            color=bg_color,
            markersize=8,
            markerfacecolor="none",
            linestyle="None",
            zorder=5,
            alpha=line_alpha,
        )
        geometry_preview_artists.extend([sim_pt, bg_pt])
        if show_pair_connectors:
            (link,) = ax.plot(
                [float(sim_col), float(bg_col)],
                [float(sim_row), float(bg_row)],
                color=line_color,
                linestyle="--" if excluded else ":",
                linewidth=1.0,
                alpha=line_alpha,
                zorder=4,
            )
            geometry_preview_artists.append(link)
            label = ax.annotate(
                label_text,
                xy=(float(bg_col), float(bg_row)),
                xytext=(float(sim_col), float(sim_row)),
                color=line_color,
                fontsize=8,
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color=line_color,
                    lw=1.0,
                    linestyle="--" if excluded else ":",
                    alpha=0.45 if excluded else 0.8,
                ),
                zorder=6,
            )
        else:
            label = _plot_pair_label_text(
                ax,
                label_text,
                start_xy=(float(sim_col), float(sim_row)),
                end_xy=(float(bg_col), float(bg_row)),
                color=line_color,
                zorder=6,
            )
        geometry_preview_artists.append(label)

    draw_idle()


def draw_qr_cylinder_overlay_paths(
    ax: Any,
    paths: Sequence[dict[str, object]],
    *,
    qr_cylinder_overlay_artists: list[object],
    clear_qr_cylinder_overlay_artists: Callable[..., None],
    draw_idle: Callable[[], None],
    redraw: bool = True,
) -> None:
    """Draw precomputed analytic Qr-cylinder overlay paths."""

    clear_qr_cylinder_overlay_artists(redraw=False)
    for path in paths:
        color = "#fff06a" if path.get("source") == "primary" else "#78d7ff"
        (line,) = ax.plot(
            path["cols"],
            path["rows"],
            color=color,
            linewidth=0.9,
            alpha=0.58,
            zorder=4.6,
            solid_capstyle="round",
        )
        qr_cylinder_overlay_artists.append(line)
    if redraw:
        draw_idle()
