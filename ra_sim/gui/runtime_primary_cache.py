"""Helpers for incremental primary-source cache updates in the GUI runtime."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ra_sim.gui import controllers as gui_controllers
from ra_sim.simulation.diffraction import (
    build_intersection_cache,
    intersection_cache_to_hit_tables,
)


@dataclass(frozen=True)
class IncrementalSfPruneAction:
    """Describe how the runtime should respond to one SF-prune-only change."""

    mode: str
    added_keys: tuple[object, ...]
    removed_keys: tuple[object, ...]
    missing_keys: tuple[object, ...]
    reason: str | None = None


def _job_compatible_with_primary_cache(
    job: object,
    *,
    cache_signature: object,
    source_mode: str,
) -> bool:
    if not isinstance(job, dict):
        return True
    if not bool(job.get("run_primary", False)):
        return True
    return job.get("primary_contribution_cache_signature") == cache_signature and str(
        job.get("primary_source_mode", "")
    ) == str(source_mode)


def resolve_incremental_sf_prune_action(
    *,
    cache_signature: object,
    cached_signature: object,
    source_mode: str,
    cached_source_mode: str,
    active_keys: Sequence[object],
    previous_active_keys: Sequence[object],
    primary_hit_table_cache: Mapping[object, np.ndarray] | None,
    active_job: object = None,
    queued_job: object = None,
) -> IncrementalSfPruneAction:
    """Return whether one bias-only change can reuse the primary contribution cache."""

    current_keys = tuple(active_keys or ())
    previous_keys = tuple(previous_active_keys or ())
    previous_key_set = set(previous_keys)
    current_key_set = set(current_keys)

    if cache_signature != cached_signature:
        return IncrementalSfPruneAction(
            mode="full",
            added_keys=tuple(),
            removed_keys=tuple(),
            missing_keys=tuple(),
            reason="cache_signature_changed",
        )

    if str(source_mode) != str(cached_source_mode):
        return IncrementalSfPruneAction(
            mode="full",
            added_keys=tuple(),
            removed_keys=tuple(),
            missing_keys=tuple(),
            reason="source_mode_changed",
        )

    if not isinstance(primary_hit_table_cache, Mapping) or len(primary_hit_table_cache) <= 0:
        return IncrementalSfPruneAction(
            mode="full",
            added_keys=tuple(),
            removed_keys=tuple(),
            missing_keys=tuple(),
            reason="missing_primary_cache",
        )

    if not _job_compatible_with_primary_cache(
        active_job,
        cache_signature=cache_signature,
        source_mode=source_mode,
    ) or not _job_compatible_with_primary_cache(
        queued_job,
        cache_signature=cache_signature,
        source_mode=source_mode,
    ):
        return IncrementalSfPruneAction(
            mode="full",
            added_keys=tuple(),
            removed_keys=tuple(),
            missing_keys=tuple(),
            reason="worker_incompatible",
        )

    added_keys = tuple(key for key in current_keys if key not in previous_key_set)
    removed_keys = tuple(key for key in previous_keys if key not in current_key_set)
    missing_keys = tuple(key for key in current_keys if key not in primary_hit_table_cache)
    return IncrementalSfPruneAction(
        mode=("fill" if missing_keys else "reuse"),
        added_keys=added_keys,
        removed_keys=removed_keys,
        missing_keys=missing_keys,
        reason=("fill_missing_keys" if missing_keys else "all_keys_cached"),
    )


def build_primary_subset_payload(
    *,
    source_mode: str,
    all_primary_qr: Mapping[object, object] | None,
    all_miller: np.ndarray,
    all_intensities: np.ndarray,
    requested_keys: Sequence[object],
) -> dict[str, object]:
    """Build one primary-only simulation subset payload from stable contribution keys."""

    ordered_keys = tuple(requested_keys or ())
    if str(source_mode) == "qr":
        source_dict = gui_controllers.copy_bragg_qr_dict(dict(all_primary_qr or {}))
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for raw_key in ordered_keys:
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                continue
            try:
                m_idx = int(raw_key[0])
                local_idx = int(raw_key[1])
            except (TypeError, ValueError):
                continue
            grouped_indices[m_idx].append(local_idx)

        primary_data: dict[int, dict[str, object]] = {}
        normalized_keys: list[tuple[int, int]] = []
        for m_idx in sorted(grouped_indices):
            entry = source_dict.get(int(m_idx))
            if not isinstance(entry, dict):
                continue
            l_vals = np.asarray(entry.get("L", []), dtype=np.float64).reshape(-1)
            i_vals = np.asarray(entry.get("I", []), dtype=np.float64).reshape(-1)
            row_count = min(l_vals.shape[0], i_vals.shape[0])
            if row_count <= 0:
                continue
            valid_local_indices = [
                idx for idx in grouped_indices[m_idx] if 0 <= idx < int(row_count)
            ]
            if not valid_local_indices:
                continue
            hk_value = entry.get("hk", (0, 0))
            if isinstance(hk_value, (list, tuple)):
                hk_tuple = tuple(hk_value)
            else:
                hk_tuple = (0, 0)
            try:
                deg_value = int(str(entry.get("deg", 1)))
            except Exception:
                deg_value = 1
            primary_data[int(m_idx)] = {
                "L": l_vals[valid_local_indices].copy(),
                "I": i_vals[valid_local_indices].copy(),
                "hk": hk_tuple,
                "deg": int(deg_value),
            }
            normalized_keys.extend((int(m_idx), int(idx)) for idx in valid_local_indices)

        return {
            "primary_data": primary_data,
            "primary_intensities": np.empty((0,), dtype=np.float64),
            "primary_contribution_keys": normalized_keys,
        }

    miller_arr = np.asarray(all_miller, dtype=np.float64)
    intens = np.asarray(all_intensities, dtype=np.float64).reshape(-1)
    row_count = min(
        int(miller_arr.shape[0]) if miller_arr.ndim == 2 else 0,
        int(intens.shape[0]),
    )
    valid_indices = [
        int(raw_key)
        for raw_key in ordered_keys
        if isinstance(raw_key, (int, np.integer)) and 0 <= int(raw_key) < row_count
    ]
    miller_primary_data: np.ndarray
    miller_primary_intensities: np.ndarray
    if row_count <= 0 or miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
        valid_indices = []
        miller_primary_data = np.empty((0, 3), dtype=np.float64)
        miller_primary_intensities = np.empty((0,), dtype=np.float64)
    else:
        miller_primary_data = miller_arr[valid_indices, :].copy()
        miller_primary_intensities = intens[valid_indices].copy()

    return {
        "primary_data": miller_primary_data,
        "primary_intensities": miller_primary_intensities,
        "primary_contribution_keys": valid_indices,
    }


def rasterize_hit_tables_to_image(
    hit_tables: Sequence[np.ndarray] | None,
    *,
    image_size: int,
) -> np.ndarray:
    """Rasterize raw hit tables into one detector image using bilinear splats."""

    image = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    if hit_tables is None:
        return image

    for table in hit_tables:
        try:
            hits = np.asarray(table, dtype=np.float64)
        except Exception:
            continue
        if hits.size <= 0:
            continue
        if hits.ndim == 1:
            hits = hits.reshape(1, -1)
        if hits.ndim != 2 or hits.shape[1] < 3:
            continue

        for intensity, col_f, row_f in hits[:, :3]:
            if not (np.isfinite(intensity) and np.isfinite(col_f) and np.isfinite(row_f)):
                continue
            row0 = int(np.floor(row_f))
            col0 = int(np.floor(col_f))
            d_row = float(row_f - row0)
            d_col = float(col_f - col0)
            for row_offset in range(2):
                rr = row0 + row_offset
                if rr < 0 or rr >= int(image_size):
                    continue
                w_row = 1.0 - d_row if row_offset == 0 else d_row
                if w_row <= 0.0:
                    continue
                for col_offset in range(2):
                    cc = col0 + col_offset
                    if cc < 0 or cc >= int(image_size):
                        continue
                    w_col = 1.0 - d_col if col_offset == 0 else d_col
                    if w_col <= 0.0:
                        continue
                    image[rr, cc] += float(intensity) * w_row * w_col
    return image


def rematerialize_primary_artifacts(
    *,
    primary_hit_table_cache: Mapping[object, np.ndarray],
    active_keys: Sequence[object],
    image_size: int,
    a_primary: float,
    c_primary: float,
    beam_x_array: np.ndarray | None = None,
    beam_y_array: np.ndarray | None = None,
    theta_array: np.ndarray | None = None,
    phi_array: np.ndarray | None = None,
    wavelength_array: np.ndarray | None = None,
    lattice_label: str = "primary",
) -> dict[str, object]:
    """Rebuild primary simulation artifacts from cached raw hit tables."""

    raw_hit_tables = [
        np.asarray(primary_hit_table_cache[key], dtype=np.float64).copy()
        for key in active_keys
        if key in primary_hit_table_cache
    ]
    image = rasterize_hit_tables_to_image(raw_hit_tables, image_size=int(image_size))
    intersection_cache = build_intersection_cache(
        raw_hit_tables,
        float(a_primary),
        float(c_primary),
        beam_x_array=beam_x_array,
        beam_y_array=beam_y_array,
        theta_array=theta_array,
        phi_array=phi_array,
        wavelength_array=wavelength_array,
    )
    peak_tables = intersection_cache_to_hit_tables(intersection_cache)
    if not peak_tables:
        peak_tables = [table.copy() for table in raw_hit_tables]
    return {
        "image": image,
        "raw_hit_tables": raw_hit_tables,
        "intersection_cache": [
            np.asarray(table, dtype=np.float64).copy() for table in intersection_cache
        ],
        "peak_tables": [np.asarray(table, dtype=np.float64).copy() for table in peak_tables],
        "peak_table_lattice": [
            (float(a_primary), float(c_primary), str(lattice_label))
            for _ in range(len(peak_tables))
        ],
    }
