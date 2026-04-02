"""Pure helpers for geometry-fit overlay rendering and diagnostics."""

from __future__ import annotations

import math
from typing import Callable, Sequence

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


def compose_orientation_transforms(
    shape: tuple[int, int],
    first: dict[str, object] | None,
    second: dict[str, object] | None,
) -> dict[str, object]:
    """Return one discrete transform equivalent to applying ``first`` then ``second``."""

    def _normalize(choice: dict[str, object] | None) -> dict[str, object]:
        if not isinstance(choice, dict):
            return {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
            }
        return {
            "indexing_mode": str(choice.get("indexing_mode", "xy")),
            "k": int(choice.get("k", 0)),
            "flip_x": bool(choice.get("flip_x", False)),
            "flip_y": bool(choice.get("flip_y", False)),
            "flip_order": str(choice.get("flip_order", "yx")),
        }

    first_norm = _normalize(first)
    second_norm = _normalize(second)

    height, width = int(shape[0]), int(shape[1])
    refs = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (0.0, float(height - 1)),
        (float(width - 1), float(height - 1)),
        (0.5 * float(width - 1), 0.5 * float(height - 1)),
    ]
    mapped_once = transform_points_orientation(refs, shape, **first_norm)
    mapped_twice = transform_points_orientation(mapped_once, shape, **second_norm)

    best = None
    best_err = float("inf")
    for candidate in iter_orientation_transform_candidates():
        remapped = transform_points_orientation(refs, shape, **candidate)
        err = 0.0
        for (x_ref, y_ref), (x_back, y_back) in zip(mapped_twice, remapped):
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


def rotate_measured_peaks_for_display(
    measured,
    rotated_shape,
    *,
    display_rotate_k: int = -1,
):
    """Rotate measured-peak coordinates to match the displayed background."""

    if measured is None:
        return []

    rotated_entries = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = rotate_point_for_display(
                    updated["x"],
                    updated["y"],
                    rotated_shape,
                    display_rotate_k,
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = rotate_point_for_display(
                    updated["x_pix"],
                    updated["y_pix"],
                    rotated_shape,
                    display_rotate_k,
                )
            rotated_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = rotate_point_for_display(
                seq[3],
                seq[4],
                rotated_shape,
                display_rotate_k,
            )
            rotated_entries.append(type(entry)(seq))
        else:
            rotated_entries.append(entry)

    return rotated_entries


def unrotate_display_peaks(
    measured,
    rotated_shape,
    *,
    k: int | None = None,
    default_display_rotate_k: int = -1,
):
    """Undo a display rotation on peak coordinates."""

    if measured is None:
        return []

    rotate_k = default_display_rotate_k if k is None else k
    inv_k = -int(rotate_k)

    unrotated = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = rotate_point_for_display(
                    updated["x"],
                    updated["y"],
                    rotated_shape,
                    inv_k,
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = rotate_point_for_display(
                    updated["x_pix"],
                    updated["y_pix"],
                    rotated_shape,
                    inv_k,
                )
            unrotated.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = rotate_point_for_display(
                seq[3],
                seq[4],
                rotated_shape,
                inv_k,
            )
            unrotated.append(type(entry)(seq))
        else:
            unrotated.append(entry)

    return unrotated


def apply_indexing_mode_to_entries(
    measured,
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
):
    """Swap x/y coordinates when using alternate indexing modes."""

    if measured is None:
        return []

    _ = shape
    mode = (indexing_mode or "xy").lower()
    if mode == "xy":
        return list(measured)

    swapped_entries = []

    def _swap_pair(col: float, row: float) -> tuple[float, float]:
        return float(row), float(col)

    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _swap_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _swap_pair(
                    updated["x_pix"],
                    updated["y_pix"],
                )
            swapped_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _swap_pair(seq[3], seq[4])
            swapped_entries.append(type(entry)(seq))
        else:
            swapped_entries.append(entry)

    return swapped_entries


def apply_orientation_to_entries(
    measured,
    rotated_shape,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Apply backend-only rotations/flips to measured peak entries."""

    if measured is None:
        return []

    indexed = apply_indexing_mode_to_entries(
        measured,
        rotated_shape,
        indexing_mode=indexing_mode,
    )

    k_mod = int(k) % 4
    if k_mod == 0 and not flip_x and not flip_y:
        return list(indexed)

    mode = (indexing_mode or "xy").lower()
    oriented_shape = (
        rotated_shape
        if mode == "xy"
        else (rotated_shape[1], rotated_shape[0])
    )

    def _apply_pair(x_val: float, y_val: float) -> tuple[float, float]:
        return transform_points_orientation(
            [(x_val, y_val)],
            oriented_shape,
            indexing_mode="xy",
            k=k_mod,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_order=flip_order,
        )[0]

    oriented_entries = []
    for entry in indexed:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _apply_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _apply_pair(
                    updated["x_pix"],
                    updated["y_pix"],
                )
            oriented_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _apply_pair(seq[3], seq[4])
            oriented_entries.append(type(entry)(seq))
        else:
            oriented_entries.append(entry)

    return oriented_entries


def orient_image_for_fit(
    image: np.ndarray | None,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Return a rotated/flipped copy of ``image`` for backend fitting only."""

    if image is None:
        return None

    oriented = np.asarray(image)
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        oriented = np.swapaxes(oriented, 0, 1)
    order = (flip_order or "yx").lower()
    if order == "xy":
        if flip_x:
            oriented = np.flip(oriented, axis=1)
        if flip_y:
            oriented = np.flip(oriented, axis=0)
    else:
        if flip_y:
            oriented = np.flip(oriented, axis=0)
        if flip_x:
            oriented = np.flip(oriented, axis=1)
    k_mod = int(k) % 4
    if k_mod:
        oriented = np.rot90(oriented, k_mod)
    return oriented


def native_sim_to_display_coords(
    col: float,
    row: float,
    image_shape: tuple[int, ...],
    *,
    sim_display_rotate_k: int = 0,
) -> tuple[float, float]:
    """Rotate native simulation coordinates into the displayed frame."""

    return rotate_point_for_display(col, row, image_shape, sim_display_rotate_k)


def display_to_native_sim_coords(
    col: float,
    row: float,
    image_shape: tuple[int, ...],
    *,
    sim_display_rotate_k: int = 0,
) -> tuple[float, float]:
    """Map displayed simulation coordinates back into native simulation frame."""

    return rotate_point_for_display(col, row, image_shape, -sim_display_rotate_k)


def best_orientation_alignment(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
):
    """Search over 90° rotations and axis flips to minimize RMS distance."""

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        return None

    def _describe(
        k: int,
        flip_x: bool,
        flip_y: bool,
        flip_order: str,
        indexing_mode: str,
    ) -> str:
        parts: list[str] = []
        if k % 4:
            parts.append(f"rot{(k % 4) * 90}° CCW")
        if flip_x:
            parts.append("flip_x")
        if flip_y:
            parts.append("flip_y")
        parts.append(f"order={flip_order}")
        parts.append(f"indexing={indexing_mode}")
        return " + ".join(parts)

    best = None
    for candidate in iter_orientation_transform_candidates():
        transformed = transform_points_orientation(
            meas_coords,
            shape,
            indexing_mode=str(candidate["indexing_mode"]),
            k=int(candidate["k"]),
            flip_x=bool(candidate["flip_x"]),
            flip_y=bool(candidate["flip_y"]),
            flip_order=str(candidate["flip_order"]),
        )
        deltas = [
            math.hypot(sx - mx, sy - my)
            for (sx, sy), (mx, my) in zip(sim_coords, transformed)
        ]
        if not deltas:
            continue
        rms = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
        mean = sum(deltas) / len(deltas)
        candidate_result = {
            "k": int(candidate["k"]),
            "flip_x": bool(candidate["flip_x"]),
            "flip_y": bool(candidate["flip_y"]),
            "flip_order": str(candidate["flip_order"]),
            "indexing_mode": str(candidate["indexing_mode"]),
            "rms": float(rms),
            "mean": float(mean),
            "label": _describe(
                int(candidate["k"]),
                bool(candidate["flip_x"]),
                bool(candidate["flip_y"]),
                str(candidate["flip_order"]),
                str(candidate["indexing_mode"]),
            ),
        }
        if best is None or candidate_result["rms"] < best["rms"]:
            best = candidate_result

    return best


def orientation_metrics(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str,
    k: int,
    flip_x: bool,
    flip_y: bool,
    flip_order: str,
):
    """Return RMS/mean/max distance after transforming measured coordinates."""

    transformed = transform_points_orientation(
        meas_coords,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )
    deltas = [
        math.hypot(sx - mx, sy - my)
        for (sx, sy), (mx, my) in zip(sim_coords, transformed)
    ]
    if not deltas:
        return {
            "rms": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    arr = np.asarray(deltas, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def select_fit_orientation(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    cfg: dict[str, object] | None = None,
):
    """Choose a measured-peak orientation transform that best aligns to simulation."""

    identity = {
        "k": 0,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
        "indexing_mode": "xy",
        "label": "identity",
    }
    config = cfg if isinstance(cfg, dict) else {}
    enabled = bool(config.get("enabled", True))
    min_improvement = max(0.0, float(config.get("min_improvement_px", 0.25)))
    max_rms = float(config.get("max_rms_px", np.inf))

    diagnostics = {
        "enabled": bool(enabled),
        "pairs": int(min(len(sim_coords), len(meas_coords))),
        "identity_rms_px": float("nan"),
        "best_rms_px": float("nan"),
        "best_label": "identity",
        "chosen_label": "identity",
        "improvement_px": float("nan"),
        "reason": "identity_fallback",
    }

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        diagnostics["reason"] = "insufficient_pairs"
        return identity, diagnostics

    identity_metrics = orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode="xy",
        k=0,
        flip_x=False,
        flip_y=False,
        flip_order="yx",
    )
    diagnostics["identity_rms_px"] = float(identity_metrics["rms"])

    best = best_orientation_alignment(sim_coords, meas_coords, shape)
    if best is None:
        diagnostics["reason"] = "no_candidate"
        return identity, diagnostics

    best_metrics = orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode=str(best.get("indexing_mode", "xy")),
        k=int(best.get("k", 0)),
        flip_x=bool(best.get("flip_x", False)),
        flip_y=bool(best.get("flip_y", False)),
        flip_order=str(best.get("flip_order", "yx")),
    )
    best_rms = float(best_metrics["rms"])
    identity_rms = float(identity_metrics["rms"])
    improvement = identity_rms - best_rms

    diagnostics.update(
        {
            "best_rms_px": best_rms,
            "best_label": str(best.get("label", "candidate")),
            "improvement_px": float(improvement),
        }
    )

    if not enabled:
        diagnostics["reason"] = "disabled_by_config"
        return identity, diagnostics

    if not np.isfinite(best_rms):
        diagnostics["reason"] = "best_rms_not_finite"
        return identity, diagnostics

    if np.isfinite(max_rms) and best_rms > max_rms:
        diagnostics["reason"] = "best_rms_above_threshold"
        return identity, diagnostics

    if not np.isfinite(improvement) or improvement < min_improvement:
        diagnostics["reason"] = "insufficient_improvement"
        return identity, diagnostics

    chosen = {
        "k": int(best.get("k", 0)),
        "flip_x": bool(best.get("flip_x", False)),
        "flip_y": bool(best.get("flip_y", False)),
        "flip_order": str(best.get("flip_order", "yx")),
        "indexing_mode": str(best.get("indexing_mode", "xy")),
        "label": str(best.get("label", "candidate")),
    }
    diagnostics["chosen_label"] = str(chosen["label"])
    diagnostics["reason"] = "selected_best"
    return chosen, diagnostics


def aggregate_match_centers(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    sim_millers: list[tuple[int, int, int]],
    meas_millers: list[tuple[int, int, int]],
):
    """Collapse matched peaks by HKL and return centroid pairs."""

    aggregated: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for hkl_sim, hkl_meas, sim_xy, meas_xy in zip(
        sim_millers,
        meas_millers,
        sim_coords,
        meas_coords,
    ):
        hkl_key = tuple(int(v) for v in (hkl_sim or hkl_meas))
        entry = aggregated.setdefault(hkl_key, {"sim": [], "meas": []})
        entry["sim"].append(sim_xy)
        entry["meas"].append(meas_xy)

    agg_sim_coords: list[tuple[float, float]] = []
    agg_meas_coords: list[tuple[float, float]] = []
    agg_millers: list[tuple[int, int, int]] = []

    for hkl_key in sorted(aggregated):
        sim_arr = np.asarray(aggregated[hkl_key]["sim"], dtype=float)
        meas_arr = np.asarray(aggregated[hkl_key]["meas"], dtype=float)

        agg_sim_coords.append(
            (float(sim_arr[:, 0].mean()), float(sim_arr[:, 1].mean()))
        )
        agg_meas_coords.append(
            (float(meas_arr[:, 0].mean()), float(meas_arr[:, 1].mean()))
        )
        agg_millers.append(hkl_key)

    return agg_sim_coords, agg_meas_coords, agg_millers


def normalize_hkl_key(
    value: object,
) -> tuple[int, int, int] | None:
    """Return a rounded integer HKL tuple when *value* looks like one."""

    if isinstance(value, str):
        parts = (
            value.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .split(",")
        )
        if len(parts) < 3:
            return None
        try:
            return tuple(int(np.rint(float(parts[i].strip()))) for i in range(3))
        except Exception:
            return None

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        try:
            return tuple(int(np.rint(float(value[i]))) for i in range(3))
        except Exception:
            return None

    return None


def aggregate_initial_geometry_display_pairs(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> dict[tuple[int, int, int], dict[str, tuple[float, float]]]:
    """Aggregate initial display-frame picks by HKL."""

    grouped: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}

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

    for entry in initial_pairs_display or []:
        if not isinstance(entry, dict):
            continue
        hkl_key = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl_key is None:
            continue
        bucket = grouped.setdefault(hkl_key, {"sim": [], "bg": []})
        sim_pt = _parse_point(entry.get("sim_display"))
        bg_pt = _parse_point(entry.get("bg_display"))
        if sim_pt is not None:
            bucket["sim"].append(sim_pt)
        if bg_pt is not None:
            bucket["bg"].append(bg_pt)

    aggregated: dict[tuple[int, int, int], dict[str, tuple[float, float]]] = {}
    for hkl_key, bucket in grouped.items():
        item: dict[str, tuple[float, float]] = {}
        if bucket["sim"]:
            sim_arr = np.asarray(bucket["sim"], dtype=float)
            item["sim_display"] = (
                float(sim_arr[:, 0].mean()),
                float(sim_arr[:, 1].mean()),
            )
        if bucket["bg"]:
            bg_arr = np.asarray(bucket["bg"], dtype=float)
            item["bg_display"] = (
                float(bg_arr[:, 0].mean()),
                float(bg_arr[:, 1].mean()),
            )
        if item:
            aggregated[hkl_key] = item

    return aggregated


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
        sim_caked_display = _parse_point(raw_entry.get("sim_caked_display"))
        bg_caked_display = _parse_point(raw_entry.get("bg_caked_display"))
        if sim_caked_display is not None:
            normalized_entry["sim_caked_display"] = sim_caked_display
        if bg_caked_display is not None:
            normalized_entry["bg_caked_display"] = bg_caked_display
        raw_group_key = raw_entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            normalized_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            normalized_entry["q_group_key"] = tuple(raw_group_key)
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
    """Build one overlay record per matched peak from optimizer diagnostics.

    Coordinate-frame contract for ``point_match_diagnostics``:

    - ``simulated_x/y`` are simulation-native coordinates straight from the
      solver / hit-table path. They should only receive the simulation display
      rotation.
    - ``measured_x/y`` are fit-oriented coordinates already transformed to
      align with the simulation frame used by the solver. They must be inverse-
      oriented back to native background space before the background display
      rotation is applied.
    """

    native_frame_shape = (int(native_shape[0]), int(native_shape[1]))

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

    inverse_orientation = inverse_orientation_transform(
        native_frame_shape,
        orientation_choice,
    )
    initial_by_index = {
        int(entry["overlay_match_index"]): entry
        for entry in normalize_initial_geometry_pairs_display(initial_pairs_display)
    }

    diagnostics_by_index: dict[int, dict[str, object]] = {}
    diagnostic_order: list[int] = []
    for fallback_index, raw_entry in enumerate(point_match_diagnostics or []):
        if not isinstance(raw_entry, dict):
            continue
        overlay_match_index = normalize_overlay_match_index(
            raw_entry.get("overlay_match_index", raw_entry.get("match_input_index")),
            fallback_index,
        )
        status = str(raw_entry.get("match_status", "matched")).strip().lower()
        existing = diagnostics_by_index.get(overlay_match_index)
        if existing is None:
            diagnostics_by_index[overlay_match_index] = dict(raw_entry)
            diagnostic_order.append(int(overlay_match_index))
            continue
        existing_status = str(existing.get("match_status", "matched")).strip().lower()
        if existing_status != "matched" and status == "matched":
            diagnostics_by_index[overlay_match_index] = dict(raw_entry)

    ordered_indices: list[int] = []
    seen_indices: set[int] = set()
    for entry in normalize_initial_geometry_pairs_display(initial_pairs_display):
        index = int(entry["overlay_match_index"])
        if index in seen_indices:
            continue
        seen_indices.add(index)
        ordered_indices.append(index)
    for index in diagnostic_order:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        ordered_indices.append(int(index))

    records: list[dict[str, object]] = []
    for fallback_index, overlay_match_index in enumerate(ordered_indices):
        initial_entry = initial_by_index.get(int(overlay_match_index), {})
        raw_entry = diagnostics_by_index.get(int(overlay_match_index), {})
        status = str(raw_entry.get("match_status", "missing_pair")).strip().lower()

        record = dict(raw_entry)
        record["overlay_match_index"] = int(overlay_match_index)
        if not status:
            status = "missing_pair"
        record["match_status"] = status

        initial_sim_native = _parse_point(initial_entry.get("sim_native"))
        initial_bg_native = _parse_point(initial_entry.get("bg_native"))
        initial_sim_caked_display = _parse_point(
            initial_entry.get("sim_caked_display")
        )
        initial_bg_caked_display = _parse_point(
            initial_entry.get("bg_caked_display")
        )
        initial_sim_display = None
        initial_bg_display = None
        # Saved fits can be redrawn in a different view than the one that
        # produced the initial overlay snapshot, so prefer native coordinates
        # when available and rebuild the detector-frame display positions.
        if initial_sim_native is not None:
            rotated = rotate_point_for_display(
                float(initial_sim_native[0]),
                float(initial_sim_native[1]),
                native_frame_shape,
                sim_display_rotate_k,
            )
            initial_sim_display = (float(rotated[0]), float(rotated[1]))
        else:
            initial_sim_display = _parse_point(initial_entry.get("sim_display"))
        if initial_bg_native is not None:
            rotated = rotate_point_for_display(
                float(initial_bg_native[0]),
                float(initial_bg_native[1]),
                native_frame_shape,
                background_display_rotate_k,
            )
            initial_bg_display = (float(rotated[0]), float(rotated[1]))
        else:
            initial_bg_display = _parse_point(initial_entry.get("bg_display"))
        record["initial_sim_display"] = initial_sim_display
        record["initial_bg_display"] = initial_bg_display
        if initial_sim_caked_display is not None:
            record["initial_sim_caked_display"] = (
                float(initial_sim_caked_display[0]),
                float(initial_sim_caked_display[1]),
            )
        if initial_bg_caked_display is not None:
            record["initial_bg_caked_display"] = (
                float(initial_bg_caked_display[0]),
                float(initial_bg_caked_display[1]),
            )
        if initial_sim_native is not None:
            record["initial_sim_native"] = (
                float(initial_sim_native[0]),
                float(initial_sim_native[1]),
            )
        if initial_bg_native is not None:
            record["initial_bg_native"] = (
                float(initial_bg_native[0]),
                float(initial_bg_native[1]),
            )
        if "hkl" not in record and initial_entry.get("hkl") is not None:
            record["hkl"] = initial_entry.get("hkl")
        if "label" not in record and initial_entry.get("hkl") is not None:
            record["label"] = str(initial_entry.get("hkl"))

        if status == "matched":
            try:
                simulated_native = (
                    float(raw_entry.get("simulated_x", np.nan)),
                    float(raw_entry.get("simulated_y", np.nan)),
                )
                measured_fit_oriented = (
                    float(raw_entry.get("measured_x", np.nan)),
                    float(raw_entry.get("measured_y", np.nan)),
                )
            except Exception:
                continue
            if not all(
                np.isfinite(v) for v in (*simulated_native, *measured_fit_oriented)
            ):
                continue

            measured_native = transform_points_orientation(
                [measured_fit_oriented],
                native_frame_shape,
                indexing_mode=str(inverse_orientation.get("indexing_mode", "xy")),
                k=int(inverse_orientation.get("k", 0)),
                flip_x=bool(inverse_orientation.get("flip_x", False)),
                flip_y=bool(inverse_orientation.get("flip_y", False)),
                flip_order=str(inverse_orientation.get("flip_order", "yx")),
            )[0]

            final_sim_display = rotate_point_for_display(
                float(simulated_native[0]),
                float(simulated_native[1]),
                native_frame_shape,
                sim_display_rotate_k,
            )
            final_bg_display = rotate_point_for_display(
                float(measured_native[0]),
                float(measured_native[1]),
                native_frame_shape,
                background_display_rotate_k,
            )
            record["final_sim_fit"] = (
                float(simulated_native[0]),
                float(simulated_native[1]),
            )
            record["final_bg_fit"] = (
                float(measured_fit_oriented[0]),
                float(measured_fit_oriented[1]),
            )
            record["final_sim_native"] = (
                float(simulated_native[0]),
                float(simulated_native[1]),
            )
            record["final_bg_native"] = (
                float(measured_native[0]),
                float(measured_native[1]),
            )
            record["final_sim_display"] = (
                float(final_sim_display[0]),
                float(final_sim_display[1]),
            )
            record["final_bg_display"] = (
                float(final_bg_display[0]),
                float(final_bg_display[1]),
            )
            if initial_bg_caked_display is not None:
                record["final_bg_caked_display"] = (
                    float(initial_bg_caked_display[0]),
                    float(initial_bg_caked_display[1]),
                )
            record["simulated_frame"] = "sim_native"
            record["measured_frame"] = "fit_oriented"
            try:
                distance_px = float(record.get("distance_px", np.nan))
            except Exception:
                distance_px = float("nan")
            if not np.isfinite(distance_px):
                distance_px = float(
                    math.hypot(
                        simulated_native[0] - measured_fit_oriented[0],
                        simulated_native[1] - measured_fit_oriented[1],
                    )
                )
            record["overlay_distance_px"] = float(distance_px)

        records.append(record)

    return records


def compute_geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
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

    def _resolve_display_point(
        entry: dict[str, object],
        *,
        display_key: str,
        native_key: str,
    ) -> tuple[float, float] | None:
        if show_caked_2d and native_detector_coords_to_caked_display_coords is not None:
            caked_display_key = display_key.replace("_display", "_caked_display")
            caked_point = _parse_point(entry.get(caked_display_key))
            if caked_point is not None:
                return caked_point
            native_point = _parse_point(entry.get(native_key))
            if native_point is not None:
                try:
                    projected = native_detector_coords_to_caked_display_coords(
                        float(native_point[0]),
                        float(native_point[1]),
                    )
                except Exception:
                    projected = None
                if projected is not None:
                    return _parse_point(projected)
        return _parse_point(entry.get(display_key))

    sim_frame_dists: list[float] = []
    bg_frame_dists: list[float] = []
    paired_records = 0

    for raw_entry in overlay_records or []:
        if not isinstance(raw_entry, dict):
            continue
        initial_sim = _resolve_display_point(
            raw_entry,
            display_key="initial_sim_display",
            native_key="initial_sim_native",
        )
        final_sim = _resolve_display_point(
            raw_entry,
            display_key="final_sim_display",
            native_key="final_sim_native",
        )
        initial_bg = _resolve_display_point(
            raw_entry,
            display_key="initial_bg_display",
            native_key="initial_bg_native",
        )
        final_bg = _resolve_display_point(
            raw_entry,
            display_key="final_bg_display",
            native_key="final_bg_native",
        )
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
