"""Pure helpers for geometry-fit overlay rendering and diagnostics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rotate_point_for_display(
    col: float,
    row: float,
    shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Rotate one ``(col, row)`` point using the same rule as ``np.rot90``."""

    height, width = int(shape[0]), int(shape[1])
    col_new = float(col)
    row_new = float(row)

    for _ in range(int(k) % 4):
        row_new, col_new, height, width = (
            width - 1.0 - col_new,
            row_new,
            width,
            height,
        )

    return float(col_new), float(row_new)


def transform_points_orientation(
    points: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
) -> list[tuple[float, float]]:
    """Apply the discrete fit orientation transform to point coordinates."""

    base_height, base_width = int(shape[0]), int(shape[1])
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        height, width = base_width, base_height
    else:
        height, width = base_height, base_width

    order = (flip_order or "yx").lower()
    transformed: list[tuple[float, float]] = []

    def _flip_xy(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_x:
            col_t = width - 1.0 - col_t
        if flip_y:
            row_t = height - 1.0 - row_t
        return float(col_t), float(row_t)

    def _flip_yx(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_y:
            row_t = height - 1.0 - row_t
        if flip_x:
            col_t = width - 1.0 - col_t
        return float(col_t), float(row_t)

    flipper = _flip_xy if order == "xy" else _flip_yx

    for col, row in points:
        col_t = float(col)
        row_t = float(row)
        if mode == "yx":
            col_t, row_t = row_t, col_t
        col_t, row_t = flipper(col_t, row_t)
        transformed.append(
            rotate_point_for_display(col_t, row_t, (height, width), int(k))
        )

    return transformed


def iter_orientation_transform_candidates():
    """Yield all discrete 90° rotation / flip transform candidates."""

    for indexing_mode in ("xy", "yx"):
        for flip_order in ("yx", "xy"):
            for k in range(4):
                for flip_x in (False, True):
                    for flip_y in (False, True):
                        yield {
                            "indexing_mode": indexing_mode,
                            "k": int(k),
                            "flip_x": bool(flip_x),
                            "flip_y": bool(flip_y),
                            "flip_order": flip_order,
                        }


def inverse_orientation_transform(
    shape: tuple[int, int],
    orientation_choice: dict[str, object] | None,
) -> dict[str, object]:
    """Return the inverse of one discrete orientation-choice transform."""

    if not isinstance(orientation_choice, dict):
        return {
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        }

    forward = {
        "indexing_mode": str(orientation_choice.get("indexing_mode", "xy")),
        "k": int(orientation_choice.get("k", 0)),
        "flip_x": bool(orientation_choice.get("flip_x", False)),
        "flip_y": bool(orientation_choice.get("flip_y", False)),
        "flip_order": str(orientation_choice.get("flip_order", "yx")),
    }

    height, width = int(shape[0]), int(shape[1])
    refs = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (0.0, float(height - 1)),
        (float(width - 1), float(height - 1)),
        (0.5 * float(width - 1), 0.5 * float(height - 1)),
    ]
    mapped = transform_points_orientation(refs, shape, **forward)

    best = None
    best_err = float("inf")
    for candidate in iter_orientation_transform_candidates():
        unmapped = transform_points_orientation(mapped, shape, **candidate)
        err = 0.0
        for (x_ref, y_ref), (x_back, y_back) in zip(refs, unmapped):
            err = max(err, float(math.hypot(x_back - x_ref, y_back - y_ref)))
        if err < best_err:
            best_err = err
            best = dict(candidate)
        if err <= 1e-6:
            break

    if best is None:
        return {
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        }
    return best


def normalize_overlay_match_index(value: object, fallback: int) -> int:
    """Return a non-negative per-match overlay index."""

    try:
        out = int(value)
    except Exception:
        out = int(fallback)
    if out < 0:
        return int(fallback)
    return int(out)


def normalize_initial_geometry_pairs_display(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Normalize initial display-frame match records with stable overlay indices."""

    normalized: list[dict[str, object]] = []

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

    for fallback_index, raw_entry in enumerate(initial_pairs_display or []):
        if not isinstance(raw_entry, dict):
            continue
        normalized_entry = dict(raw_entry)
        normalized_entry["overlay_match_index"] = normalize_overlay_match_index(
            raw_entry.get("overlay_match_index"),
            fallback_index,
        )
        sim_display = _parse_point(raw_entry.get("sim_display"))
        bg_display = _parse_point(raw_entry.get("bg_display"))
        if sim_display is not None:
            normalized_entry["sim_display"] = sim_display
        if bg_display is not None:
            normalized_entry["bg_display"] = bg_display
        normalized.append(normalized_entry)

    return normalized


def build_geometry_fit_overlay_records(
    initial_pairs_display: Sequence[dict[str, object]] | None,
    point_match_diagnostics: Sequence[dict[str, object]] | None,
    *,
    native_shape: tuple[int, int],
    orientation_choice: dict[str, object] | None = None,
    sim_display_rotate_k: int = 0,
    background_display_rotate_k: int = 0,
) -> list[dict[str, object]]:
    """Build one overlay record per matched peak from optimizer diagnostics."""

    native_frame_shape = (int(native_shape[0]), int(native_shape[1]))
    inverse_orientation = inverse_orientation_transform(
        native_frame_shape,
        orientation_choice,
    )
    initial_by_index = {
        int(entry["overlay_match_index"]): entry
        for entry in normalize_initial_geometry_pairs_display(initial_pairs_display)
    }

    records: list[dict[str, object]] = []
    for fallback_index, raw_entry in enumerate(point_match_diagnostics or []):
        if not isinstance(raw_entry, dict):
            continue
        status = str(raw_entry.get("match_status", "matched")).strip().lower()
        if status != "matched":
            continue
        try:
            sim_fit = (
                float(raw_entry.get("simulated_x", np.nan)),
                float(raw_entry.get("simulated_y", np.nan)),
            )
            bg_fit = (
                float(raw_entry.get("measured_x", np.nan)),
                float(raw_entry.get("measured_y", np.nan)),
            )
        except Exception:
            continue
        if not all(np.isfinite(v) for v in (*sim_fit, *bg_fit)):
            continue

        overlay_match_index = normalize_overlay_match_index(
            raw_entry.get("overlay_match_index", raw_entry.get("match_input_index")),
            fallback_index,
        )
        initial_entry = initial_by_index.get(overlay_match_index, {})

        sim_native = transform_points_orientation(
            [sim_fit],
            native_frame_shape,
            indexing_mode=str(inverse_orientation.get("indexing_mode", "xy")),
            k=int(inverse_orientation.get("k", 0)),
            flip_x=bool(inverse_orientation.get("flip_x", False)),
            flip_y=bool(inverse_orientation.get("flip_y", False)),
            flip_order=str(inverse_orientation.get("flip_order", "yx")),
        )[0]
        bg_native = transform_points_orientation(
            [bg_fit],
            native_frame_shape,
            indexing_mode=str(inverse_orientation.get("indexing_mode", "xy")),
            k=int(inverse_orientation.get("k", 0)),
            flip_x=bool(inverse_orientation.get("flip_x", False)),
            flip_y=bool(inverse_orientation.get("flip_y", False)),
            flip_order=str(inverse_orientation.get("flip_order", "yx")),
        )[0]

        final_sim_display = rotate_point_for_display(
            float(sim_native[0]),
            float(sim_native[1]),
            native_frame_shape,
            sim_display_rotate_k,
        )
        final_bg_display = rotate_point_for_display(
            float(bg_native[0]),
            float(bg_native[1]),
            native_frame_shape,
            background_display_rotate_k,
        )

        record = dict(raw_entry)
        record["overlay_match_index"] = int(overlay_match_index)
        record["initial_sim_display"] = initial_entry.get("sim_display")
        record["initial_bg_display"] = initial_entry.get("bg_display")
        record["final_sim_fit"] = (float(sim_fit[0]), float(sim_fit[1]))
        record["final_bg_fit"] = (float(bg_fit[0]), float(bg_fit[1]))
        record["final_sim_native"] = (float(sim_native[0]), float(sim_native[1]))
        record["final_bg_native"] = (float(bg_native[0]), float(bg_native[1]))
        record["final_sim_display"] = (
            float(final_sim_display[0]),
            float(final_sim_display[1]),
        )
        record["final_bg_display"] = (
            float(final_bg_display[0]),
            float(final_bg_display[1]),
        )
        if "hkl" not in record and initial_entry.get("hkl") is not None:
            record["hkl"] = initial_entry.get("hkl")
        if "label" not in record and initial_entry.get("hkl") is not None:
            record["label"] = str(initial_entry.get("hkl"))
        try:
            distance_px = float(record.get("distance_px", np.nan))
        except Exception:
            distance_px = float("nan")
        if not np.isfinite(distance_px):
            distance_px = float(math.hypot(sim_fit[0] - bg_fit[0], sim_fit[1] - bg_fit[1]))
        record["overlay_distance_px"] = float(distance_px)
        records.append(record)

    return records


def compute_geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, float], str]:
    """Summarize per-match display-frame agreement for the final overlay."""

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

    sim_frame_dists: list[float] = []
    bg_frame_dists: list[float] = []
    paired_records = 0

    for raw_entry in overlay_records or []:
        if not isinstance(raw_entry, dict):
            continue
        initial_sim = _parse_point(raw_entry.get("initial_sim_display"))
        final_sim = _parse_point(raw_entry.get("final_sim_display"))
        initial_bg = _parse_point(raw_entry.get("initial_bg_display"))
        final_bg = _parse_point(raw_entry.get("final_bg_display"))
        if (
            initial_sim is not None
            and final_sim is not None
            and initial_bg is not None
            and final_bg is not None
        ):
            paired_records += 1
        if initial_sim is not None and final_sim is not None:
            sim_frame_dists.append(
                float(math.hypot(final_sim[0] - initial_sim[0], final_sim[1] - initial_sim[1]))
            )
        if initial_bg is not None and final_bg is not None:
            bg_frame_dists.append(
                float(math.hypot(final_bg[0] - initial_bg[0], final_bg[1] - initial_bg[1]))
            )

    stats: dict[str, float] = {
        "overlay_record_count": float(len(list(overlay_records or []))),
        "paired_records": float(paired_records),
        "sim_display_med_px": float(np.median(sim_frame_dists))
        if sim_frame_dists
        else float("nan"),
        "bg_display_med_px": float(np.median(bg_frame_dists))
        if bg_frame_dists
        else float("nan"),
        "sim_display_p90_px": float(np.percentile(sim_frame_dists, 90.0))
        if sim_frame_dists
        else float("nan"),
        "bg_display_p90_px": float(np.percentile(bg_frame_dists, 90.0))
        if bg_frame_dists
        else float("nan"),
    }

    warning = ""
    sim_med = float(stats["sim_display_med_px"])
    bg_med = float(stats["bg_display_med_px"])
    if (
        len(sim_frame_dists) >= 3
        and np.isfinite(sim_med)
        and np.isfinite(bg_med)
        and sim_med - bg_med > 40.0
        and bg_med <= 0.6 * sim_med
    ):
        warning = (
            "Frame mismatch suspect: fitted simulation overlay points do not land in the "
            "same display frame as the fixed background picks."
        )

    return stats, warning
