"""Seed-local background peak detection and matching helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


Logger = Callable[[str], None] | None


def _log(logger: Logger, text: str) -> None:
    if logger is None:
        return
    try:
        logger(text)
    except Exception:
        pass


def _refine_peak_center(
    peakness: np.ndarray,
    fine_image: np.ndarray,
    row_idx: int,
    col_idx: int,
) -> tuple[float, float]:
    """Refine a summit center with a quadratic peak fit and centroid fallback."""

    if peakness.ndim != 2:
        return float(col_idx), float(row_idx)
    height, width = peakness.shape
    r0 = max(0, int(row_idx) - 1)
    r1 = min(height, int(row_idx) + 2)
    c0 = max(0, int(col_idx) - 1)
    c1 = min(width, int(col_idx) + 2)
    peak_patch = peakness[r0:r1, c0:c1]
    weight_patch = np.clip(peak_patch, 0.0, None)
    if not np.any(weight_patch > 0.0) and fine_image.ndim == 2:
        fine_patch = fine_image[r0:r1, c0:c1]
        weight_patch = np.clip(fine_patch - np.min(fine_patch), 0.0, None)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    total_weight = float(np.sum(weight_patch))
    center_col = float(col_idx)
    center_row = float(row_idx)
    if total_weight > 0.0 and np.isfinite(total_weight):
        center_col = float(np.sum(weight_patch * cc) / total_weight)
        center_row = float(np.sum(weight_patch * rr) / total_weight)

    fit_patch = peak_patch
    if not np.any(np.isfinite(fit_patch)) and fine_image.ndim == 2:
        fit_patch = fine_image[r0:r1, c0:c1]
    fit_mask = np.isfinite(fit_patch)
    if np.count_nonzero(fit_mask) >= 6:
        x = cc[fit_mask].astype(np.float64) - center_col
        y = rr[fit_mask].astype(np.float64) - center_row
        z = fit_patch[fit_mask].astype(np.float64)
        sample_weights = weight_patch[fit_mask].astype(np.float64)
        if not np.any(sample_weights > 0.0):
            sample_weights = np.ones_like(z, dtype=np.float64)
        design = np.column_stack(
            (
                x * x,
                y * y,
                x * y,
                x,
                y,
                np.ones_like(x),
            )
        )
        root_weights = np.sqrt(np.clip(sample_weights, 1.0e-6, None))
        try:
            coeffs, *_ = np.linalg.lstsq(
                design * root_weights[:, None],
                z * root_weights,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            coeffs = None
        if coeffs is not None and coeffs.shape[0] == 6 and np.all(np.isfinite(coeffs)):
            a, b, c, d, e, _ = coeffs
            hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=np.float64)
            det = float(np.linalg.det(hessian))
            if np.isfinite(det) and det > 1.0e-12 and a < -1.0e-12 and b < -1.0e-12:
                try:
                    offset = -np.linalg.solve(hessian, np.array([d, e], dtype=np.float64))
                except np.linalg.LinAlgError:
                    offset = None
                if offset is not None and offset.shape[0] == 2 and np.all(np.isfinite(offset)):
                    offset_col = float(offset[0])
                    offset_row = float(offset[1])
                    if abs(offset_col) <= 1.25 and abs(offset_row) <= 1.25:
                        center_col = float(center_col + offset_col)
                        center_row = float(center_row + offset_row)

    return center_col, center_row


def build_background_peak_context(
    background_image: object,
    cfg: dict[str, object] | None = None,
    *,
    logger: Logger = None,
) -> dict[str, object]:
    """Build reusable background peakness data for seed-local peak matching."""

    config = cfg if isinstance(cfg, dict) else {}
    local_max_size = max(3, int(config.get("local_max_size_px", 5)))
    if local_max_size % 2 == 0:
        local_max_size += 1
    smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
    climb_sigma = max(
        0.0,
        float(
            config.get(
                "climb_sigma_px",
                min(1.0, 0.5 * smooth_sigma) if smooth_sigma > 0.0 else 0.8,
            )
        ),
    )
    min_prominence_sigma = float(config.get("min_prominence_sigma", 2.0))
    fallback_percentile = float(config.get("fallback_percentile", 99.5))
    fallback_percentile = min(100.0, max(50.0, fallback_percentile))

    img = np.asarray(background_image, dtype=np.float32)
    valid_mask = np.isfinite(img)
    if img.ndim != 2 or not np.any(valid_mask):
        return {"img_valid": False}

    height, width = img.shape
    _log(logger, f"context build start shape=({height},{width})")
    baseline = np.float32(np.median(img[valid_mask]))
    work = np.where(valid_mask, img, baseline).astype(np.float32, copy=False)
    fine = gaussian_filter(work, sigma=climb_sigma, mode="nearest") if climb_sigma > 0.0 else work
    broad_sigma = smooth_sigma if smooth_sigma > 0.0 else max(2.0, climb_sigma + 1.5)
    broad = gaussian_filter(work, sigma=broad_sigma, mode="nearest")
    peakness = fine - broad
    local_max = peakness == maximum_filter(peakness, size=local_max_size, mode="nearest")

    prom_vals = peakness[valid_mask]
    prom_center = float(np.median(prom_vals))
    mad = float(np.median(np.abs(prom_vals - prom_center)))
    sigma_est = 1.4826 * mad
    if not np.isfinite(sigma_est) or sigma_est <= 1e-12:
        sigma_est = float(np.std(prom_vals))
    if not np.isfinite(sigma_est) or sigma_est <= 1e-12:
        sigma_est = 1.0

    candidate_floor = prom_center + min_prominence_sigma * sigma_est
    candidate_mask = local_max & valid_mask & (peakness >= candidate_floor)
    if not np.any(candidate_mask):
        fallback_floor = float(np.percentile(prom_vals, fallback_percentile))
        candidate_mask = local_max & valid_mask & (peakness >= fallback_floor)
    _log(logger, f"context candidates nonzero={int(np.count_nonzero(candidate_mask))}")

    summit_records: list[dict[str, object]] = []
    if np.any(candidate_mask):
        candidate_labels, candidate_count_raw = label(
            candidate_mask.astype(np.uint8),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        _log(logger, f"context label done candidate_components={int(candidate_count_raw)}")
        rr, cc = np.nonzero(candidate_labels > 0)
        if rr.size > 0:
            label_vals = candidate_labels[rr, cc].astype(np.int32, copy=False)
            peak_vals = peakness[rr, cc]
            intensity_vals = work[rr, cc]
            order = np.lexsort((intensity_vals, peak_vals, label_vals))
            labels_sorted = label_vals[order]
            group_end = np.ones(labels_sorted.shape, dtype=bool)
            if labels_sorted.size > 1:
                group_end[:-1] = labels_sorted[:-1] != labels_sorted[1:]
            best_order = order[group_end]
            best_rows = rr[best_order]
            best_cols = cc[best_order]
            best_labels = label_vals[best_order]
            best_peak_vals = peakness[best_rows, best_cols]
            best_intensity_vals = work[best_rows, best_cols]
            for idx, summit_id in enumerate(best_labels):
                peak_row = int(best_rows[idx])
                peak_col = int(best_cols[idx])
                prom_sigma = float(
                    (best_peak_vals[idx] - prom_center) / (sigma_est + 1e-12)
                )
                bg_intensity = float(best_intensity_vals[idx])
                summit_records.append(
                    {
                        "summit_id": int(summit_id),
                        "row": float(peak_row),
                        "col": float(peak_col),
                        "center_row": float(peak_row),
                        "center_col": float(peak_col),
                        "center_refined": False,
                        "prominence_sigma": float(prom_sigma),
                        "background_intensity": float(bg_intensity),
                    }
                )
        summit_records.sort(
            key=lambda info: (
                float(info.get("prominence_sigma", -np.inf)),
                float(info.get("background_intensity", -np.inf)),
            ),
            reverse=True,
        )
    else:
        candidate_labels = np.zeros_like(work, dtype=np.int32)

    return {
        "img_valid": True,
        "work": work,
        "fine": fine,
        "peakness": peakness,
        "valid_mask": valid_mask,
        "candidate_labels": candidate_labels,
        "summit_records": summit_records,
        "sigma_est": float(sigma_est),
        "prominence_center": float(prom_center),
        "height": int(height),
        "width": int(width),
    }


def _assignment_score(
    dist_px: float,
    prom_sigma: float,
    seed_weight: float,
    match_radius_px: float,
    config: dict[str, object],
) -> float:
    radius = max(1.0, float(match_radius_px))
    distance_weight = max(0.0, float(config.get("distance_score_weight", 2.5)))
    prominence_weight = max(0.0, float(config.get("prominence_score_weight", 0.05)))
    seed_weight_scale = max(0.0, float(config.get("seed_weight_score_scale", 1e-12)))
    distance_term = max(0.0, 1.0 - float(dist_px) / radius)
    prominence_term = max(0.0, float(prom_sigma))
    return (
        distance_weight * distance_term
        + prominence_weight * prominence_term
        + seed_weight_scale * max(0.0, float(seed_weight))
    )


def _build_match_stats(
    *,
    simulated_count: int,
    search_radius_px: float,
    distance_sigma_clip: float,
    sigma_est: float = float("nan"),
    prominence_center: float = float("nan"),
    candidate_count: float = 0.0,
    qualified_summit_count: float = 0.0,
    within_radius_count: float = 0.0,
    unambiguous_count: float = 0.0,
    ownership_filtered_count: float = 0.0,
    claimed_summit_count: float = 0.0,
    conflicted_match_count: float = 0.0,
    matched_pre_clip_count: float = 0.0,
    matched_count: float = 0.0,
    clipped_count: float = 0.0,
    distance_clip_limit_px: float = float("nan"),
    distance_median_pre_clip_px: float = float("nan"),
    distance_sigma_pre_clip_px: float = float("nan"),
    mean_match_distance_px: float = float("nan"),
    p90_match_distance_px: float = float("nan"),
    median_match_confidence: float = float("nan"),
    mean_walk_steps: float = 0.0,
    mean_net_ascent_sigma: float = float("nan"),
) -> dict[str, float]:
    return {
        "simulated_count": float(simulated_count),
        "candidate_count": float(candidate_count),
        "qualified_summit_count": float(qualified_summit_count),
        "within_radius_count": float(within_radius_count),
        "unambiguous_count": float(unambiguous_count),
        "ownership_filtered_count": float(ownership_filtered_count),
        "claimed_summit_count": float(claimed_summit_count),
        "conflicted_match_count": float(conflicted_match_count),
        "matched_pre_clip_count": float(matched_pre_clip_count),
        "matched_count": float(matched_count),
        "clipped_count": float(clipped_count),
        "sigma_est": float(sigma_est),
        "prominence_center": float(prominence_center),
        "search_radius_px": float(search_radius_px),
        "distance_sigma_clip": float(distance_sigma_clip),
        "distance_clip_limit_px": float(distance_clip_limit_px),
        "distance_median_pre_clip_px": float(distance_median_pre_clip_px),
        "distance_sigma_pre_clip_px": float(distance_sigma_pre_clip_px),
        "mean_match_distance_px": float(mean_match_distance_px),
        "p90_match_distance_px": float(p90_match_distance_px),
        "median_match_confidence": float(median_match_confidence),
        "mean_walk_steps": float(mean_walk_steps),
        "mean_net_ascent_sigma": float(mean_net_ascent_sigma),
    }


def match_simulated_peaks_to_peak_context(
    simulated_peaks: Sequence[dict[str, object]],
    peak_context: dict[str, object],
    cfg: dict[str, object] | None = None,
    *,
    context_offset_col: float = 0.0,
    context_offset_row: float = 0.0,
    logger: Logger = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Assign each simulated seed to one nearby measured background peak."""

    config = dict(cfg) if isinstance(cfg, dict) else {}
    search_radius = max(1.0, float(config.get("search_radius_px", 24.0)))
    min_match_prominence_sigma = float(
        config.get(
            "min_match_prominence_sigma",
            config.get("min_prominence_sigma", 2.0),
        )
    )
    min_confidence = float(config.get("min_confidence", 0.0))
    max_candidate_peaks = max(50, int(config.get("max_candidate_peaks", 1200)))
    k_neighbors = max(1, int(config.get("k_neighbors", 12)))
    distance_sigma_clip = max(0.0, float(config.get("distance_sigma_clip", 3.5)))
    ambiguity_ratio_min = max(1.0, float(config.get("ambiguity_ratio_min", 1.15)))
    ambiguity_margin_px = max(0.0, float(config.get("ambiguity_margin_px", 2.0)))
    require_candidate_ownership = bool(config.get("require_candidate_ownership", True))
    max_match_distance_px = float(config.get("max_match_distance_px", search_radius))
    if not np.isfinite(max_match_distance_px) or max_match_distance_px <= 0.0:
        max_match_distance_px = search_radius
    match_radius = min(search_radius, max_match_distance_px)

    if not bool(peak_context.get("img_valid", False)):
        return [], _build_match_stats(
            simulated_count=len(simulated_peaks),
            search_radius_px=search_radius,
            distance_sigma_clip=distance_sigma_clip,
        )

    work = np.asarray(peak_context.get("work", []), dtype=float)
    fine = np.asarray(peak_context.get("fine", []), dtype=float)
    peakness = np.asarray(peak_context.get("peakness", []), dtype=float)
    sigma_est = float(peak_context.get("sigma_est", np.nan))
    prom_center = float(peak_context.get("prominence_center", np.nan))
    height = int(peak_context.get("height", work.shape[0] if work.ndim == 2 else 0))
    width = int(peak_context.get("width", work.shape[1] if work.ndim == 2 else 0))

    summit_records = list(peak_context.get("summit_records", []))
    if not summit_records:
        return [], _build_match_stats(
            simulated_count=len(simulated_peaks),
            search_radius_px=search_radius,
            distance_sigma_clip=distance_sigma_clip,
            sigma_est=sigma_est,
            prominence_center=prom_center,
        )

    qualified_summits = [
        dict(record)
        for record in summit_records
        if float(record.get("prominence_sigma", -np.inf)) >= min_match_prominence_sigma
    ]
    if len(qualified_summits) > max_candidate_peaks:
        qualified_summits.sort(
            key=lambda info: (
                float(info.get("prominence_sigma", -np.inf)),
                float(info.get("background_intensity", -np.inf)),
            ),
            reverse=True,
        )
        qualified_summits = qualified_summits[:max_candidate_peaks]
    _log(
        logger,
        f"candidate peaks total={len(summit_records)} qualified={len(qualified_summits)} radius={match_radius:.2f}",
    )
    if not qualified_summits:
        return [], _build_match_stats(
            simulated_count=len(simulated_peaks),
            search_radius_px=search_radius,
            distance_sigma_clip=distance_sigma_clip,
            sigma_est=sigma_est,
            prominence_center=prom_center,
            candidate_count=len(summit_records),
        )

    for info in qualified_summits:
        if bool(info.get("center_refined", False)):
            continue
        peak_row = int(round(float(info.get("row", 0.0))))
        peak_col = int(round(float(info.get("col", 0.0))))
        center_col, center_row = _refine_peak_center(peakness, fine, peak_row, peak_col)
        info["center_col"] = float(center_col)
        info["center_row"] = float(center_row)
        info["center_refined"] = True

    candidate_coords = np.asarray(
        [
            [float(info.get("center_col", info.get("col", np.nan))), float(info.get("center_row", info.get("row", np.nan)))]
            for info in qualified_summits
        ],
        dtype=float,
    )
    finite_candidate_mask = np.all(np.isfinite(candidate_coords), axis=1)
    if not np.all(finite_candidate_mask):
        filtered = []
        filtered_coords = []
        for keep, info, coord in zip(finite_candidate_mask, qualified_summits, candidate_coords):
            if keep:
                filtered.append(info)
                filtered_coords.append(coord)
        qualified_summits = filtered
        candidate_coords = np.asarray(filtered_coords, dtype=float)
    if candidate_coords.size == 0:
        return [], _build_match_stats(
            simulated_count=len(simulated_peaks),
            search_radius_px=search_radius,
            distance_sigma_clip=distance_sigma_clip,
            sigma_est=sigma_est,
            prominence_center=prom_center,
            candidate_count=len(summit_records),
        )

    ordered_simulated = [
        dict(entry)
        for entry in sorted(
            simulated_peaks,
            key=lambda entry: float(entry.get("weight", 0.0)),
            reverse=True,
        )
    ]
    tree = cKDTree(candidate_coords)

    owner_seed_by_candidate = np.full(candidate_coords.shape[0], -1, dtype=np.int64)
    owner_dist_by_candidate = np.full(candidate_coords.shape[0], np.inf, dtype=float)
    competitor_dist_by_candidate = np.full(candidate_coords.shape[0], np.inf, dtype=float)
    within_radius_mask = np.zeros(candidate_coords.shape[0], dtype=bool)
    unambiguous_mask = np.ones(candidate_coords.shape[0], dtype=bool)
    ownership_filtered_count = 0
    if require_candidate_ownership and candidate_coords.shape[0] > 0:
        seed_coords_local: list[list[float]] = []
        seed_tree_indices: list[int] = []
        for seed_idx, entry in enumerate(ordered_simulated):
            sim_col_local = float(entry.get("sim_col_local", entry.get("sim_col", np.nan)))
            sim_row_local = float(entry.get("sim_row_local", entry.get("sim_row", np.nan)))
            if not (np.isfinite(sim_col_local) and np.isfinite(sim_row_local)):
                continue
            seed_coords_local.append([sim_col_local, sim_row_local])
            seed_tree_indices.append(int(seed_idx))

        if seed_coords_local:
            seed_coords_arr = np.asarray(seed_coords_local, dtype=float)
            seed_tree = cKDTree(seed_coords_arr)
            k_query = 1 if len(seed_tree_indices) <= 1 else 2
            owner_dists_raw, owner_idx_raw = seed_tree.query(candidate_coords, k=k_query)
            owner_dists = np.asarray(owner_dists_raw, dtype=float)
            owner_idx = np.asarray(owner_idx_raw, dtype=np.int64)
            if owner_dists.ndim == 1:
                owner_dists = owner_dists[:, np.newaxis]
                owner_idx = owner_idx[:, np.newaxis]

            owner_dist_by_candidate = owner_dists[:, 0]
            owner_seed_by_candidate = np.asarray(
                [
                    int(seed_tree_indices[idx]) if 0 <= idx < len(seed_tree_indices) else -1
                    for idx in owner_idx[:, 0].tolist()
                ],
                dtype=np.int64,
            )
            within_radius_mask = np.isfinite(owner_dist_by_candidate) & (
                owner_dist_by_candidate <= match_radius + 1.0e-9
            )
            if owner_dists.shape[1] >= 2:
                competitor_dist_by_candidate = owner_dists[:, 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = competitor_dist_by_candidate / np.maximum(
                        owner_dist_by_candidate,
                        1.0e-9,
                    )
                unambiguous_mask = (
                    ~np.isfinite(competitor_dist_by_candidate)
                    | ((competitor_dist_by_candidate - owner_dist_by_candidate) >= ambiguity_margin_px)
                    | (ratio >= ambiguity_ratio_min)
                )
            ownership_filtered_count = int(
                np.count_nonzero(within_radius_mask & ~unambiguous_mask)
            )
        else:
            unambiguous_mask = np.zeros(candidate_coords.shape[0], dtype=bool)

    within_radius_count = int(np.count_nonzero(within_radius_mask))
    unambiguous_count = int(np.count_nonzero(within_radius_mask & unambiguous_mask))

    edge_lookup: dict[tuple[int, int], dict[str, object]] = {}
    viable_by_summit: dict[int, list[dict[str, object]]] = defaultdict(list)
    net_ascent_values: list[float] = []

    for seed_idx, entry in enumerate(ordered_simulated):
        sim_col_local = float(entry.get("sim_col_local", entry.get("sim_col", np.nan)))
        sim_row_local = float(entry.get("sim_row_local", entry.get("sim_row", np.nan)))
        if not (np.isfinite(sim_col_local) and np.isfinite(sim_row_local)):
            continue
        sim_col = float(entry.get("sim_col_global", entry.get("sim_col", np.nan)))
        sim_row = float(entry.get("sim_row_global", entry.get("sim_row", np.nan)))
        if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
            sim_col = float(sim_col_local + context_offset_col)
            sim_row = float(sim_row_local + context_offset_row)

        cand_indices = tree.query_ball_point([sim_col_local, sim_row_local], r=match_radius + 1e-9)
        if not cand_indices:
            continue

        if len(cand_indices) > k_neighbors:
            local_coords = candidate_coords[np.asarray(cand_indices, dtype=int)]
            local_dists = np.hypot(
                local_coords[:, 0] - sim_col_local,
                local_coords[:, 1] - sim_row_local,
            )
            order = np.argsort(local_dists)
            cand_indices = [cand_indices[int(i)] for i in order[:k_neighbors]]

        seed_col_px = int(np.clip(round(sim_col_local), 0, max(width - 1, 0)))
        seed_row_px = int(np.clip(round(sim_row_local), 0, max(height - 1, 0)))
        seed_peakness = float(peakness[seed_row_px, seed_col_px]) if peakness.ndim == 2 else 0.0

        for cand_idx in cand_indices:
            cand_idx = int(cand_idx)
            if require_candidate_ownership:
                if cand_idx < 0 or cand_idx >= owner_seed_by_candidate.shape[0]:
                    continue
                if not bool(within_radius_mask[cand_idx]) or not bool(unambiguous_mask[cand_idx]):
                    continue
                if int(owner_seed_by_candidate[cand_idx]) != int(seed_idx):
                    continue
            info = qualified_summits[cand_idx]
            center_col_local = float(candidate_coords[cand_idx, 0])
            center_row_local = float(candidate_coords[cand_idx, 1])
            dist_px = float(
                np.hypot(center_col_local - sim_col_local, center_row_local - sim_row_local)
            )
            if dist_px > match_radius:
                continue

            prom_sigma = float(info.get("prominence_sigma", np.nan))
            bg_intensity = float(info.get("background_intensity", np.nan))
            confidence = max(0.0, prom_sigma) / (1.0 + max(0.0, dist_px))
            if confidence < min_confidence:
                continue

            raw_col_local = float(info.get("col", center_col_local))
            raw_row_local = float(info.get("row", center_row_local))
            raw_distance = float(
                np.hypot(raw_col_local - sim_col_local, raw_row_local - sim_row_local)
            )
            summit_row_px = int(np.clip(round(raw_row_local), 0, max(height - 1, 0)))
            summit_col_px = int(np.clip(round(raw_col_local), 0, max(width - 1, 0)))
            net_ascent_sigma = float(
                (peakness[summit_row_px, summit_col_px] - seed_peakness)
                / (sigma_est + 1e-12)
            )
            net_ascent_values.append(net_ascent_sigma)

            assignment_score = _assignment_score(
                dist_px,
                prom_sigma,
                float(entry.get("weight", 0.0)),
                match_radius,
                config,
            )
            summit_id = int(info["summit_id"])
            competitor_dist = (
                float(competitor_dist_by_candidate[cand_idx])
                if 0 <= cand_idx < competitor_dist_by_candidate.shape[0]
                else float("nan")
            )
            owner_dist = (
                float(owner_dist_by_candidate[cand_idx])
                if 0 <= cand_idx < owner_dist_by_candidate.shape[0]
                else float("nan")
            )
            ownership_margin = (
                competitor_dist - owner_dist
                if np.isfinite(competitor_dist) and np.isfinite(owner_dist)
                else float("inf")
            )
            ownership_ratio = (
                competitor_dist / max(owner_dist, 1.0e-9)
                if np.isfinite(competitor_dist) and np.isfinite(owner_dist)
                else float("inf")
            )
            match_entry = {
                "seed_index": int(seed_idx),
                "summit_id": summit_id,
                "hkl": tuple(int(v) for v in entry["hkl"]),
                "label": str(entry["label"]),
                "x": float(center_col_local + context_offset_col),
                "y": float(center_row_local + context_offset_row),
                "sim_x": float(sim_col),
                "sim_y": float(sim_row),
                "background_intensity": float(bg_intensity),
                "distance_px": float(dist_px),
                "prominence_sigma": float(prom_sigma),
                "confidence": float(confidence),
                "assignment_score": float(assignment_score),
                "weight": float(entry.get("weight", 0.0)),
                "walk_steps": 0,
                "walk_distance_px": float(raw_distance),
                "net_ascent_sigma": float(net_ascent_sigma),
                "owner_distance_px": float(owner_dist),
                "competitor_distance_px": float(competitor_dist),
                "ownership_margin_px": float(ownership_margin),
                "ownership_ratio": float(ownership_ratio),
                "source_peak_index": entry.get("source_peak_index"),
                "source_label": entry.get("source_label"),
                "source_table_index": entry.get("source_table_index"),
                "source_row_index": entry.get("source_row_index"),
                "hkl_raw": entry.get("hkl_raw"),
                "qr": entry.get("qr"),
                "qz": entry.get("qz"),
                "q_group_key": entry.get("q_group_key"),
            }
            edge_lookup[(int(seed_idx), summit_id)] = match_entry
            viable_by_summit[summit_id].append(match_entry)

    if not edge_lookup:
        return [], _build_match_stats(
            simulated_count=len(simulated_peaks),
            search_radius_px=search_radius,
            distance_sigma_clip=distance_sigma_clip,
            sigma_est=sigma_est,
            prominence_center=prom_center,
            candidate_count=len(summit_records),
            qualified_summit_count=len(qualified_summits),
            within_radius_count=within_radius_count,
            unambiguous_count=unambiguous_count,
            ownership_filtered_count=ownership_filtered_count,
            mean_net_ascent_sigma=(
                float(np.mean(net_ascent_values)) if net_ascent_values else float("nan")
            ),
        )

    seed_keys = sorted({key[0] for key in edge_lookup})
    summit_keys = sorted({key[1] for key in edge_lookup})
    seed_index_map = {seed_key: idx for idx, seed_key in enumerate(seed_keys)}
    summit_index_map = {summit_key: idx for idx, summit_key in enumerate(summit_keys)}

    score_matrix = np.full((len(seed_keys), len(summit_keys)), -1e9, dtype=float)
    for (seed_key, summit_key), match_entry in edge_lookup.items():
        score_matrix[seed_index_map[seed_key], summit_index_map[summit_key]] = float(
            match_entry["assignment_score"]
        )

    valid_mask_matrix = score_matrix > -1e8
    valid_scores = score_matrix[valid_mask_matrix]
    max_score = float(np.max(valid_scores))
    cost_matrix = np.full_like(score_matrix, max_score + 1e6)
    cost_matrix[valid_mask_matrix] = max_score - score_matrix[valid_mask_matrix]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pre_clip_matches: list[dict[str, object]] = []
    for row_idx, col_idx in zip(row_ind, col_ind):
        if row_idx >= len(seed_keys) or col_idx >= len(summit_keys):
            continue
        if not bool(valid_mask_matrix[row_idx, col_idx]):
            continue
        seed_key = seed_keys[row_idx]
        summit_key = summit_keys[col_idx]
        match_entry = edge_lookup.get((seed_key, summit_key))
        if match_entry is None:
            continue
        clean_entry = dict(match_entry)
        clean_entry.pop("seed_index", None)
        pre_clip_matches.append(clean_entry)

    conflicted_match_count = sum(max(0, len(contenders) - 1) for contenders in viable_by_summit.values())

    matches = list(pre_clip_matches)
    clipped_count = 0
    clip_limit = match_radius
    if matches:
        dists = np.asarray([float(m["distance_px"]) for m in matches], dtype=float)
        dist_med = float(np.median(dists))
        dist_mad = float(np.median(np.abs(dists - dist_med)))
        dist_sigma = 1.4826 * dist_mad
        if np.isfinite(distance_sigma_clip):
            if np.isfinite(dist_sigma) and dist_sigma > 1e-9:
                clip_limit = min(clip_limit, dist_med + distance_sigma_clip * dist_sigma)
            else:
                clip_limit = min(clip_limit, dist_med + max(1.0, distance_sigma_clip))
        matches = [
            m for m in matches if float(m["distance_px"]) <= float(clip_limit) + 1e-9
        ]
        clipped_count = max(0, len(pre_clip_matches) - len(matches))
    else:
        dist_med = float("nan")
        dist_sigma = float("nan")

    match_dists = np.asarray([float(m["distance_px"]) for m in matches], dtype=float)
    match_conf = np.asarray([float(m["confidence"]) for m in matches], dtype=float)
    match_walk_steps = np.asarray([float(m.get("walk_steps", 0.0)) for m in matches], dtype=float)
    match_net_ascent = np.asarray([float(m.get("net_ascent_sigma", np.nan)) for m in matches], dtype=float)
    finite_net_ascent = match_net_ascent[np.isfinite(match_net_ascent)]

    return matches, _build_match_stats(
        simulated_count=len(simulated_peaks),
        search_radius_px=search_radius,
        distance_sigma_clip=distance_sigma_clip,
        sigma_est=sigma_est,
        prominence_center=prom_center,
        candidate_count=len(summit_records),
        qualified_summit_count=len(qualified_summits),
        within_radius_count=within_radius_count,
        unambiguous_count=unambiguous_count,
        ownership_filtered_count=ownership_filtered_count,
        claimed_summit_count=len(viable_by_summit),
        conflicted_match_count=conflicted_match_count,
        matched_pre_clip_count=len(pre_clip_matches),
        matched_count=len(matches),
        clipped_count=clipped_count,
        distance_clip_limit_px=(
            float(clip_limit) if np.isfinite(clip_limit) else float("nan")
        ),
        distance_median_pre_clip_px=(
            float(dist_med) if np.isfinite(dist_med) else float("nan")
        ),
        distance_sigma_pre_clip_px=(
            float(dist_sigma) if np.isfinite(dist_sigma) else float("nan")
        ),
        mean_match_distance_px=(
            float(np.mean(match_dists)) if match_dists.size else float("nan")
        ),
        p90_match_distance_px=(
            float(np.percentile(match_dists, 90.0)) if match_dists.size else float("nan")
        ),
        median_match_confidence=(
            float(np.median(match_conf)) if match_conf.size else float("nan")
        ),
        mean_walk_steps=(
            float(np.mean(match_walk_steps)) if match_walk_steps.size else float(0.0)
        ),
        mean_net_ascent_sigma=(
            float(np.mean(finite_net_ascent)) if finite_net_ascent.size else float("nan")
        ),
    )


def match_simulated_peaks_to_background(
    simulated_peaks: Sequence[dict[str, object]],
    background_image: object,
    cfg: dict[str, object] | None = None,
    *,
    logger: Logger = None,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, object]]:
    """Convenience wrapper that builds context and matches in one call."""

    context = build_background_peak_context(background_image, cfg, logger=logger)
    matches, stats = match_simulated_peaks_to_peak_context(
        simulated_peaks,
        context,
        cfg,
        logger=logger,
    )
    return matches, stats, context
