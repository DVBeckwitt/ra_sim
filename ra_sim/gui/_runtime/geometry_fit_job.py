"""Pure geometry-fit async job snapshot helpers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeometryFitSelectionSnapshot:
    params: dict[str, object]
    var_names: list[str]
    preserve_live_theta: bool
    selected_background_indices: list[int]
    required_indices: list[int]
    skipped_empty_indices: set[int]
    selection_applied: bool
    selection_error: str | None
    uses_shared_theta: bool
    theta_metadata_applied: bool
    background_theta_values: list[float]
    background_theta_error: str | None
    theta_offset: float
    fit_params_snapshot: dict[str, object]
    primary_index: int
    joint_background_mode: bool
    build_all_selected_backgrounds: bool


@dataclass(frozen=True)
class GeometryFitBackgroundInputSnapshot:
    background_images: dict[int, dict[str, object]]
    manual_pairs_by_background: dict[int, list[dict[str, object]]]
    source_snapshots: dict[int, dict[str, object]]
    background_labels: dict[int, str]
    requested_signatures: dict[int, object]
    requested_signature_summaries: dict[int, object]
    theta_base_by_background: dict[int, float]
    theta_initial_by_background: dict[int, float]
    manual_fit_space_by_background: dict[int, str]
    required_indices: list[int]
    selected_background_indices: list[int]
    skipped_manual_pair_backgrounds: dict[int, str]
    fit_params_snapshot: dict[str, object]
    primary_index: int
    uses_shared_theta: bool
    joint_background_mode: bool
    build_all_selected_backgrounds: bool


@dataclass(frozen=True)
class GeometryFitLiveRowsSnapshot:
    live_rows_by_background: dict[int, list[dict[str, object]]]
    live_rows_cache_metadata_by_background: dict[int, dict[str, object]]
    live_rows_signature_by_background: dict[int, object]
    live_rows_handoff_diagnostics: dict[str, object]


@dataclass(frozen=True)
class GeometryFitRuntimeConfigSnapshot:
    geometry_runtime_cfg: dict[str, object]
    fit_solver_mosaic_params: object
    fit_sample_count: object


@dataclass(frozen=True)
class GeometryFitCakedProjectionSnapshot:
    caked_views_by_background: dict[int, dict[str, object]]
    projection_payload_by_background: dict[int, dict[str, object]]


@dataclass(frozen=True)
class GeometryFitProjectionViewSignatures:
    analysis_bins: tuple[int, int]
    projection_view_mode: str
    projection_view_signature_by_background: dict[int, dict[str, object]]


def resolve_geometry_fit_selection(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    current_background_index: int,
    theta_initial_value: float,
    total_background_count: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., Sequence[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., Sequence[float]],
    current_geometry_theta_offset: Callable[..., float],
    effective_geometry_fit_background_indices: Callable[..., list[int]],
) -> GeometryFitSelectionSnapshot:
    params_snapshot = dict(params or {})
    var_names_snapshot = [str(name) for name in (var_names or ())]
    preserve_live_theta = (
        "theta_initial" not in var_names_snapshot and "theta_offset" not in var_names_snapshot
    )

    selection_applied = bool(
        apply_geometry_fit_background_selection(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        )
    )
    selected_background_indices: list[int] = []
    requested_indices: list[int] = []
    skipped_empty_indices: set[int] = set()
    selection_error: str | None = None
    if selection_applied:
        try:
            requested_indices = [
                int(idx) for idx in current_geometry_fit_background_indices(strict=True)
            ]
            selected_background_indices = effective_geometry_fit_background_indices(
                requested_indices,
                total_count=int(total_background_count),
                geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            )
            active_set = {int(idx) for idx in selected_background_indices}
            skipped_empty_indices = {int(idx) for idx in requested_indices if idx not in active_set}
        except Exception as exc:
            selection_error = str(exc)

    uses_shared_theta = False
    if selection_error is None:
        try:
            uses_shared_theta = bool(
                geometry_fit_uses_shared_theta_offset(list(selected_background_indices))
            )
        except Exception:
            uses_shared_theta = False

    theta_metadata_applied = True
    background_theta_values: list[float] = []
    background_theta_error: str | None = None
    theta_offset_value = 0.0
    if uses_shared_theta:
        theta_metadata_applied = bool(
            apply_background_theta_metadata(
                trigger_update=False,
                sync_live_theta=not preserve_live_theta,
            )
        )
        if theta_metadata_applied:
            try:
                background_theta_values = [
                    float(value) for value in current_background_theta_values(strict_count=True)
                ]
                theta_offset_value = float(current_geometry_theta_offset(strict=True))
            except Exception as exc:
                background_theta_error = str(exc)

    fit_params_snapshot = dict(params_snapshot)
    active_in_selection = int(current_background_index) in set(selected_background_indices)
    primary_index = (
        int(current_background_index)
        if active_in_selection
        else (
            int(selected_background_indices[0])
            if selected_background_indices
            else int(current_background_index)
        )
    )
    joint_background_mode = bool(uses_shared_theta and len(selected_background_indices) > 1)
    build_all_selected_backgrounds = bool(joint_background_mode)
    if uses_shared_theta:
        fit_params_snapshot["theta_offset"] = float(theta_offset_value)
        if 0 <= int(primary_index) < len(background_theta_values):
            fit_params_snapshot["theta_initial"] = float(background_theta_values[int(primary_index)])
    else:
        fit_params_snapshot["theta_offset"] = 0.0
        theta_default = float(fit_params_snapshot.get("theta_initial", theta_initial_value))
        fit_params_snapshot["theta_initial"] = float(theta_default)
        background_theta_values = [float(theta_default)]

    if not selection_applied or selection_error is not None or not active_in_selection:
        required_indices: list[int] = []
    elif build_all_selected_backgrounds:
        required_indices = [int(idx) for idx in selected_background_indices]
    else:
        required_indices = [int(primary_index)]

    return GeometryFitSelectionSnapshot(
        params=params_snapshot,
        var_names=var_names_snapshot,
        preserve_live_theta=bool(preserve_live_theta),
        selected_background_indices=[int(idx) for idx in selected_background_indices],
        required_indices=required_indices,
        skipped_empty_indices=skipped_empty_indices,
        selection_applied=bool(selection_applied),
        selection_error=selection_error,
        uses_shared_theta=bool(uses_shared_theta),
        theta_metadata_applied=bool(theta_metadata_applied),
        background_theta_values=[float(value) for value in background_theta_values],
        background_theta_error=background_theta_error,
        theta_offset=float(theta_offset_value),
        fit_params_snapshot=fit_params_snapshot,
        primary_index=int(primary_index),
        joint_background_mode=bool(joint_background_mode),
        build_all_selected_backgrounds=bool(build_all_selected_backgrounds),
    )


def snapshot_geometry_fit_background_inputs(
    *,
    required_indices: Sequence[int],
    selected_background_indices: Sequence[int],
    skipped_empty_indices: set[int],
    current_background_index: int,
    theta_initial_value: float,
    fit_params_snapshot: Mapping[str, object],
    background_theta_values: Sequence[float],
    uses_shared_theta: bool,
    joint_background_mode: bool,
    build_all_selected_backgrounds: bool,
    primary_index: int,
    osc_files: Sequence[object],
    load_background_by_index: Callable[[int], tuple[object, object]],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    source_row_snapshots: Mapping[int, Mapping[str, object]],
    background_label_for_index: Callable[[int], str],
    source_snapshot_signature_for_background: Callable[[int, Mapping[str, object]], object],
    signature_summary: Callable[[object], object],
) -> GeometryFitBackgroundInputSnapshot:
    background_images: dict[int, dict[str, object]] = {}
    manual_pairs_by_background: dict[int, list[dict[str, object]]] = {}
    source_snapshots: dict[int, dict[str, object]] = {}
    requested_signatures: dict[int, object] = {}
    requested_signature_summaries: dict[int, object] = {}
    background_labels: dict[int, str] = {}
    theta_base_by_background: dict[int, float] = {}
    theta_initial_by_background: dict[int, float] = {}
    fit_params = dict(fit_params_snapshot)

    def _manual_dataset_background_label(idx: int) -> str:
        try:
            osc_path = osc_files[int(idx)]
        except Exception:
            osc_path = None
        if osc_path is not None:
            try:
                name = Path(str(osc_path)).name
            except Exception:
                name = ""
            if str(name).strip():
                return str(name)
        return str(background_label_for_index(int(idx)))

    def _theta_base_for_index(dataset_index: int) -> float:
        if build_all_selected_backgrounds or joint_background_mode:
            if 0 <= int(dataset_index) < len(background_theta_values):
                return float(background_theta_values[int(dataset_index)])
        return float(fit_params.get("theta_initial", theta_initial_value))

    def _record_background_fit_inputs(idx: int) -> None:
        theta_base_value = float(_theta_base_for_index(int(idx)))
        theta_initial_value_i = float(theta_base_value + float(fit_params.get("theta_offset", 0.0)))
        theta_base_by_background[int(idx)] = float(theta_base_value)
        theta_initial_by_background[int(idx)] = float(theta_initial_value_i)
        params_i = dict(fit_params)
        params_i["theta_initial"] = float(theta_initial_value_i)
        requested_signature = source_snapshot_signature_for_background(int(idx), params_i)
        requested_signatures[int(idx)] = requested_signature
        requested_signature_summaries[int(idx)] = signature_summary(requested_signature)

    for idx in required_indices:
        native_background, display_background = load_background_by_index(int(idx))
        background_images[int(idx)] = {
            "native": np.asarray(native_background, dtype=np.float64).copy(),
            "display": np.asarray(display_background, dtype=np.float64).copy(),
        }
        manual_pairs_by_background[int(idx)] = [
            dict(entry)
            for entry in (geometry_manual_pairs_for_index(int(idx)) or ())
            if isinstance(entry, Mapping)
        ]
        source_snapshots[int(idx)] = copy.deepcopy(source_row_snapshots.get(int(idx)) or {})
        background_labels[int(idx)] = _manual_dataset_background_label(int(idx))
        _record_background_fit_inputs(int(idx))

    return GeometryFitBackgroundInputSnapshot(
        background_images=background_images,
        manual_pairs_by_background=manual_pairs_by_background,
        source_snapshots=source_snapshots,
        background_labels=background_labels,
        requested_signatures=requested_signatures,
        requested_signature_summaries=requested_signature_summaries,
        theta_base_by_background=theta_base_by_background,
        theta_initial_by_background=theta_initial_by_background,
        manual_fit_space_by_background={},
        required_indices=[int(idx) for idx in required_indices],
        selected_background_indices=[int(idx) for idx in selected_background_indices],
        skipped_manual_pair_backgrounds={
            int(idx): _manual_dataset_background_label(int(idx))
            for idx in sorted(skipped_empty_indices)
        },
        fit_params_snapshot=fit_params,
        primary_index=int(primary_index),
        uses_shared_theta=bool(uses_shared_theta),
        joint_background_mode=bool(joint_background_mode),
        build_all_selected_backgrounds=bool(build_all_selected_backgrounds),
    )


def filter_geometry_fit_background_inputs_for_manual_pair_space(
    snapshot: GeometryFitBackgroundInputSnapshot,
    *,
    pick_uses_caked_space: bool,
    current_background_index: int,
    theta_initial_value: float,
    background_theta_values: Sequence[float],
    geometry_manual_fit_space_by_background: Callable[..., Mapping[int, object]],
    source_snapshot_signature_for_background: Callable[[int, Mapping[str, object]], object],
    signature_summary: Callable[[object], object],
) -> GeometryFitBackgroundInputSnapshot:
    manual_fit_space_by_background = {
        int(idx): str(kind)
        for idx, kind in geometry_manual_fit_space_by_background(
            list(snapshot.required_indices),
            snapshot.manual_pairs_by_background,
            pick_uses_caked_space=bool(pick_uses_caked_space),
            current_background_index=int(current_background_index),
        ).items()
    }
    has_usable_manual_pair_background = any(
        str(kind).strip().lower() != "missing" for kind in manual_fit_space_by_background.values()
    )
    if snapshot.skipped_manual_pair_backgrounds and not has_usable_manual_pair_background:
        manual_fit_space_by_background.update(
            {int(idx): "missing" for idx in snapshot.skipped_manual_pair_backgrounds}
        )
    missing_manual_pair_indices = [
        int(idx)
        for idx, kind in manual_fit_space_by_background.items()
        if str(kind).strip().lower() == "missing"
    ]
    missing_manual_pair_set = {int(idx) for idx in missing_manual_pair_indices}
    usable_manual_pair_indices = [
        int(idx)
        for idx in snapshot.required_indices
        if int(idx) not in missing_manual_pair_set
    ]
    if not (missing_manual_pair_indices and usable_manual_pair_indices):
        return replace(
            snapshot,
            manual_fit_space_by_background={
                int(idx): str(kind) for idx, kind in manual_fit_space_by_background.items()
            },
        )

    usable_set = {int(idx) for idx in usable_manual_pair_indices}
    skipped_manual_pair_backgrounds = dict(snapshot.skipped_manual_pair_backgrounds)
    skipped_manual_pair_backgrounds.update(
        {
            int(idx): str(snapshot.background_labels.get(int(idx), f"background {int(idx) + 1}"))
            for idx in sorted(missing_manual_pair_indices)
        }
    )
    required_indices = [
        int(idx) for idx in snapshot.required_indices if int(idx) in usable_set
    ]
    selected_background_indices = [
        int(idx) for idx in snapshot.selected_background_indices if int(idx) in usable_set
    ]
    manual_fit_space_by_background = {
        int(idx): str(manual_fit_space_by_background[int(idx)]) for idx in required_indices
    }
    background_images = {
        int(idx): payload
        for idx, payload in snapshot.background_images.items()
        if int(idx) in usable_set
    }
    manual_pairs_by_background = {
        int(idx): rows
        for idx, rows in snapshot.manual_pairs_by_background.items()
        if int(idx) in usable_set
    }
    source_snapshots = {
        int(idx): payload
        for idx, payload in snapshot.source_snapshots.items()
        if int(idx) in usable_set
    }
    background_labels = {
        int(idx): label
        for idx, label in snapshot.background_labels.items()
        if int(idx) in usable_set
    }
    primary_index = (
        int(current_background_index)
        if int(current_background_index) in usable_set
        else int(required_indices[0])
    )
    uses_shared_theta = bool(snapshot.uses_shared_theta and len(selected_background_indices) > 1)
    joint_background_mode = bool(uses_shared_theta)
    build_all_selected_backgrounds = bool(joint_background_mode)
    fit_params = dict(snapshot.fit_params_snapshot)
    if not uses_shared_theta:
        fit_params["theta_offset"] = 0.0
        if 0 <= int(primary_index) < len(background_theta_values):
            fit_params["theta_initial"] = float(background_theta_values[int(primary_index)])

    theta_base_by_background: dict[int, float] = {}
    theta_initial_by_background: dict[int, float] = {}
    requested_signatures: dict[int, object] = {}
    requested_signature_summaries: dict[int, object] = {}

    def _theta_base_for_index(dataset_index: int) -> float:
        if build_all_selected_backgrounds or joint_background_mode:
            if 0 <= int(dataset_index) < len(background_theta_values):
                return float(background_theta_values[int(dataset_index)])
        return float(fit_params.get("theta_initial", theta_initial_value))

    for idx in required_indices:
        theta_base_value = float(_theta_base_for_index(int(idx)))
        theta_initial_value_i = float(theta_base_value + float(fit_params.get("theta_offset", 0.0)))
        theta_base_by_background[int(idx)] = float(theta_base_value)
        theta_initial_by_background[int(idx)] = float(theta_initial_value_i)
        params_i = dict(fit_params)
        params_i["theta_initial"] = float(theta_initial_value_i)
        requested_signature = source_snapshot_signature_for_background(int(idx), params_i)
        requested_signatures[int(idx)] = requested_signature
        requested_signature_summaries[int(idx)] = signature_summary(requested_signature)

    return replace(
        snapshot,
        background_images=background_images,
        manual_pairs_by_background=manual_pairs_by_background,
        source_snapshots=source_snapshots,
        background_labels=background_labels,
        requested_signatures=requested_signatures,
        requested_signature_summaries=requested_signature_summaries,
        theta_base_by_background=theta_base_by_background,
        theta_initial_by_background=theta_initial_by_background,
        manual_fit_space_by_background=manual_fit_space_by_background,
        required_indices=required_indices,
        selected_background_indices=selected_background_indices,
        skipped_manual_pair_backgrounds=skipped_manual_pair_backgrounds,
        fit_params_snapshot=fit_params,
        primary_index=int(primary_index),
        uses_shared_theta=bool(uses_shared_theta),
        joint_background_mode=bool(joint_background_mode),
        build_all_selected_backgrounds=bool(build_all_selected_backgrounds),
    )


def snapshot_live_rows_handoff(
    *,
    current_background_index: int,
    required_indices: Sequence[int],
    background_images: Mapping[int, Mapping[str, object]],
    manual_pairs_by_background: Mapping[int, Sequence[Mapping[str, object]]],
    requested_signatures: Mapping[int, object],
    theta_initial_value: float,
    fit_params_snapshot: Mapping[str, object],
    theta_initial_by_background: Mapping[int, float],
    q_group_cached_entry_count: int,
    current_live_preview_rows: Callable[[], Sequence[Mapping[str, object]]],
    current_live_preview_cache_metadata: Callable[[], Mapping[str, object]],
    q_group_fallback_rows: Callable[..., tuple[Sequence[Mapping[str, object]], Mapping[str, object]]],
    merge_key: Callable[[Mapping[str, object]], tuple[object, ...] | None],
    source_counts: Callable[[Sequence[Mapping[str, object]]], Mapping[str, int]],
    trace_callback: Callable[..., object],
    handoff_patch_marker: str,
) -> GeometryFitLiveRowsSnapshot:
    live_rows_by_background: dict[int, list[dict[str, object]]] = {}
    live_rows_cache_metadata_by_background: dict[int, dict[str, object]] = {}
    live_rows_signature_by_background: dict[int, object] = {}
    live_rows_handoff_diagnostics: dict[str, object] = {
        "geometry_fit_live_handoff_patch_marker": str(handoff_patch_marker),
        "current_background": int(current_background_index),
        "q_group_cached_entries": int(q_group_cached_entry_count),
        "manual_picker_candidates": 0,
        "live_preview_rows_count": 0,
        "live_rows_by_background_keys": [],
        "live_rows_by_background_current_count": 0,
        "requested_signature_keys": sorted(int(key) for key in requested_signatures),
        "requested_signature_by_background_keys": sorted(int(key) for key in requested_signatures),
        "live_rows_signature_by_background_keys": [],
    }
    if int(current_background_index) in set(int(idx) for idx in required_indices):
        current_background_idx = int(current_background_index)
        live_preview_rows = [
            dict(entry) for entry in (current_live_preview_rows() or ()) if isinstance(entry, Mapping)
        ]
        live_rows_by_background[current_background_idx] = live_preview_rows
        live_rows_cache_metadata_by_background[current_background_idx] = dict(
            current_live_preview_cache_metadata() or {}
        )
        live_rows_signature_by_background[current_background_idx] = requested_signatures.get(
            current_background_idx
        )
        live_rows_cache_metadata_by_background[current_background_idx].update(
            {
                "live_rows_raw_count": int(
                    len(live_rows_by_background.get(current_background_idx, ()))
                ),
                "live_rows_payload_count": int(
                    len(live_rows_by_background.get(current_background_idx, ()))
                ),
                "live_rows_signature_match": True,
                "live_rows_signature_reason": "matched_at_job_build",
                "live_rows_cache_source": str(
                    live_rows_cache_metadata_by_background[current_background_idx].get(
                        "cache_source",
                        "live_preview_cache",
                    )
                    or "live_preview_cache"
                ),
                "live_rows_source_counts": source_counts(
                    live_rows_by_background.get(current_background_idx, ())
                ),
                "geometry_fit_live_handoff_patch_marker": str(handoff_patch_marker),
            }
        )
        live_rows_handoff_diagnostics["live_preview_rows_count"] = int(len(live_preview_rows))
        current_manual_pairs = list(
            manual_pairs_by_background.get(int(current_background_idx), ()) or ()
        )
        if current_manual_pairs or not live_preview_rows:
            background_payload = dict(background_images.get(current_background_idx) or {})
            fallback_params = dict(fit_params_snapshot)
            fallback_params["theta_initial"] = float(
                theta_initial_by_background.get(
                    current_background_idx,
                    fit_params_snapshot.get("theta_initial", theta_initial_value),
                )
            )
            fallback_rows, fallback_diag = q_group_fallback_rows(
                background_index=current_background_idx,
                params_local=fallback_params,
                display_background=background_payload.get("display"),
                manual_pairs=current_manual_pairs,
            )
            live_rows_handoff_diagnostics.update(dict(fallback_diag))
            if fallback_rows:
                current_live_rows = [
                    dict(row)
                    for row in live_rows_by_background.get(current_background_idx, ())
                    if isinstance(row, Mapping)
                ]
                fallback_keys = {
                    key for row in fallback_rows if (key := merge_key(row)) is not None
                }
                current_keys = {
                    key for row in current_live_rows if (key := merge_key(row)) is not None
                }
                if (
                    current_manual_pairs
                    and str(fallback_diag.get("job_local_fallback_source") or "")
                    == "saved_manual_pairs"
                    and not (fallback_keys & current_keys)
                ):
                    live_rows_by_background[current_background_idx] = [
                        dict(row) for row in fallback_rows
                    ]
                elif current_live_rows:
                    merged_rows = list(current_live_rows)
                    merged_key_indices: dict[tuple[object, ...], int] = {}
                    for row_index, row in enumerate(merged_rows):
                        key = merge_key(row)
                        if key is not None:
                            merged_key_indices.setdefault(key, int(row_index))
                    for fallback_row in fallback_rows:
                        fallback_copy = dict(fallback_row)
                        key = merge_key(fallback_copy)
                        if key is not None and key in merged_key_indices:
                            merged_rows[int(merged_key_indices[key])] = fallback_copy
                        else:
                            if key is not None:
                                merged_key_indices[key] = int(len(merged_rows))
                            merged_rows.append(fallback_copy)
                    live_rows_by_background[current_background_idx] = merged_rows
                else:
                    live_rows_by_background[current_background_idx] = [
                        dict(row) for row in fallback_rows
                    ]
                payload_rows = live_rows_by_background.get(current_background_idx, ())
                live_rows_cache_metadata_by_background[current_background_idx].update(
                    {
                        "cache_source": str(
                            fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                        ),
                        "live_rows_cache_source": str(
                            fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                        ),
                        "live_rows_raw_count": int(len(payload_rows)),
                        "live_rows_payload_count": int(len(payload_rows)),
                        "live_rows_signature_match": True,
                        "live_rows_signature_reason": "matched_job_local_snapshot",
                        "live_rows_source_counts": source_counts(payload_rows),
                        "geometry_fit_live_handoff_patch_marker": str(handoff_patch_marker),
                        "job_local_fallback_rows": int(len(fallback_rows)),
                        "job_local_fallback_source": str(
                            fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                        ),
                    }
                )
                trace_callback(
                    "geometry_fit_job_live_rows_from_q_group_snapshot",
                    geometry_fit_live_handoff_patch_marker=str(handoff_patch_marker),
                    background_index=int(current_background_idx),
                    rows=int(len(fallback_rows)),
                    sources=source_counts(fallback_rows),
                    job_local_fallback_source=str(
                        fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                    ),
                )
        live_rows_handoff_diagnostics.update(
            {
                "live_rows_by_background_keys": sorted(int(key) for key in live_rows_by_background),
                "live_rows_by_background_current_count": int(
                    len(live_rows_by_background.get(current_background_idx, ()))
                ),
                "live_rows_current_background_count": int(
                    len(live_rows_by_background.get(current_background_idx, ()))
                ),
                "live_rows_signature_by_background_keys": sorted(
                    int(key) for key in live_rows_signature_by_background
                ),
            }
        )
        live_rows_cache_metadata_by_background[current_background_idx].update(
            {
                "geometry_fit_live_handoff_patch_marker": str(handoff_patch_marker),
                "live_rows_handoff_diagnostics": dict(live_rows_handoff_diagnostics),
                "q_group_cached_entries": int(
                    live_rows_handoff_diagnostics.get("q_group_cached_entries", 0) or 0
                ),
                "manual_picker_candidates": int(
                    live_rows_handoff_diagnostics.get("manual_picker_candidates", 0) or 0
                ),
                "live_preview_rows_count": int(
                    live_rows_handoff_diagnostics.get("live_preview_rows_count", 0) or 0
                ),
            }
        )
    for required_background_idx in required_indices:
        background_idx = int(required_background_idx)
        if live_rows_by_background.get(background_idx):
            continue
        background_payload = dict(background_images.get(background_idx) or {})
        fallback_params = dict(fit_params_snapshot)
        fallback_params["theta_initial"] = float(
            theta_initial_by_background.get(
                background_idx,
                fit_params_snapshot.get("theta_initial", theta_initial_value),
            )
        )
        fallback_rows, fallback_diag = q_group_fallback_rows(
            background_index=background_idx,
            params_local=fallback_params,
            display_background=background_payload.get("display"),
            manual_pairs=manual_pairs_by_background.get(int(background_idx), ()),
        )
        live_rows_handoff_diagnostics.update(dict(fallback_diag))
        if not fallback_rows:
            continue
        live_rows_by_background[background_idx] = [dict(row) for row in fallback_rows]
        metadata = dict(live_rows_cache_metadata_by_background.get(background_idx) or {})
        metadata.update(
            {
                "background_index": int(background_idx),
                "cache_source": str(
                    fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                ),
                "live_rows_cache_source": str(
                    fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                ),
                "live_rows_raw_count": int(len(fallback_rows)),
                "live_rows_payload_count": int(len(fallback_rows)),
                "live_rows_signature_match": True,
                "live_rows_signature_reason": "matched_job_local_snapshot",
                "live_rows_source_counts": source_counts(fallback_rows),
                "geometry_fit_live_handoff_patch_marker": str(handoff_patch_marker),
                "job_local_fallback_rows": int(len(fallback_rows)),
                "job_local_fallback_source": str(
                    fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
                ),
            }
        )
        live_rows_cache_metadata_by_background[background_idx] = metadata
        live_rows_signature_by_background[background_idx] = requested_signatures.get(background_idx)
        trace_callback(
            "geometry_fit_job_live_rows_from_q_group_snapshot",
            geometry_fit_live_handoff_patch_marker=str(handoff_patch_marker),
            background_index=int(background_idx),
            rows=int(len(fallback_rows)),
            sources=source_counts(fallback_rows),
            job_local_fallback_source=str(
                fallback_diag.get("job_local_fallback_source") or "q_group_snapshot"
            ),
        )
    live_rows_handoff_diagnostics.update(
        {
            "live_rows_by_background_keys": sorted(int(key) for key in live_rows_by_background),
            "live_rows_by_background_current_count": int(
                len(live_rows_by_background.get(int(current_background_index), ()))
            ),
            "live_rows_current_background_count": int(
                len(live_rows_by_background.get(int(current_background_index), ()))
            ),
            "live_rows_signature_by_background_keys": sorted(
                int(key) for key in live_rows_signature_by_background
            ),
        }
    )
    for background_idx, metadata in list(live_rows_cache_metadata_by_background.items()):
        metadata["live_rows_handoff_diagnostics"] = dict(live_rows_handoff_diagnostics)
        live_rows_cache_metadata_by_background[int(background_idx)] = metadata
    return GeometryFitLiveRowsSnapshot(
        live_rows_by_background=live_rows_by_background,
        live_rows_cache_metadata_by_background=live_rows_cache_metadata_by_background,
        live_rows_signature_by_background=live_rows_signature_by_background,
        live_rows_handoff_diagnostics=live_rows_handoff_diagnostics,
    )


def build_geometry_fit_runtime_config_snapshot(
    *,
    fit_params_snapshot: Mapping[str, object],
    manual_pairs_by_background: Mapping[int, Sequence[Mapping[str, object]]],
    manual_fit_requires_caked_space: bool,
    joint_background_mode: bool,
    var_names: Sequence[str],
    build_runtime_config: Callable[[dict[str, object]], Mapping[str, object]],
    apply_joint_runtime_safety_overrides: Callable[..., Mapping[str, object]],
    manual_pair_enabled_for_geometry_fit: Callable[[Mapping[str, object]], bool],
    apply_manual_caked_runtime_overrides: Callable[..., Mapping[str, object]],
    apply_manual_point_runtime_overrides: Callable[..., Mapping[str, object]],
    build_solver_mosaic_params: Callable[..., tuple[object, object]],
    build_mosaic_params: Callable[..., object],
) -> GeometryFitRuntimeConfigSnapshot:
    base_geometry_runtime_cfg = copy.deepcopy(build_runtime_config(dict(fit_params_snapshot)))
    geometry_runtime_cfg = dict(
        apply_joint_runtime_safety_overrides(
            base_geometry_runtime_cfg,
            joint_background_mode=joint_background_mode,
        )
    )
    if manual_fit_requires_caked_space:
        selected_manual_pair_rows = [
            dict(entry)
            for rows in manual_pairs_by_background.values()
            for entry in (rows or ())
            if isinstance(entry, Mapping) and manual_pair_enabled_for_geometry_fit(entry)
        ]
        geometry_runtime_cfg = dict(
            apply_manual_caked_runtime_overrides(
                geometry_runtime_cfg,
                joint_background_mode=joint_background_mode,
                active_var_names=list(var_names),
                manual_pair_rows=selected_manual_pair_rows,
            )
        )
    else:
        geometry_runtime_cfg = dict(
            apply_manual_point_runtime_overrides(
                geometry_runtime_cfg,
                joint_background_mode=joint_background_mode,
            )
        )
    fit_solver_mosaic_params, fit_sample_count = build_solver_mosaic_params(
        params=dict(fit_params_snapshot),
        geometry_runtime_cfg=geometry_runtime_cfg,
        build_mosaic_params=build_mosaic_params,
    )
    return GeometryFitRuntimeConfigSnapshot(
        geometry_runtime_cfg=geometry_runtime_cfg,
        fit_solver_mosaic_params=fit_solver_mosaic_params,
        fit_sample_count=fit_sample_count,
    )


def snapshot_caked_projection_payloads(
    *,
    manual_fit_requires_caked_space: bool,
    current_background_index: int,
    required_indices: Sequence[int],
    caked_view_for_index: Callable[[int], object],
    caked_projection_for_index: Callable[[int], object],
    caked_projection_payload_from_view: Callable[[object], object],
    projection_payload_storage_copy: Callable[[Mapping[str, object]], object],
    sanitize_caked_display_payload: Callable[[object], tuple[object, object]],
) -> GeometryFitCakedProjectionSnapshot:
    caked_views_by_background: dict[int, dict[str, object]] = {}
    projection_payload_by_background: dict[int, dict[str, object]] = {}
    if manual_fit_requires_caked_space and int(current_background_index) in set(
        int(idx) for idx in required_indices
    ):
        caked_view_payload = caked_view_for_index(int(current_background_index))
        if isinstance(caked_view_payload, Mapping):
            try:
                projection_payload = caked_projection_for_index(int(current_background_index))
                if not isinstance(projection_payload, Mapping):
                    projection_payload = caked_projection_payload_from_view(caked_view_payload)
                if isinstance(projection_payload, Mapping):
                    stored_projection_payload = projection_payload_storage_copy(projection_payload)
                    if isinstance(stored_projection_payload, Mapping):
                        projection_payload_by_background[int(current_background_index)] = dict(
                            stored_projection_payload
                        )
                sanitized_payload, _sanitize_diag = sanitize_caked_display_payload(
                    caked_view_payload
                )
                if isinstance(sanitized_payload, Mapping):
                    caked_views_by_background[int(current_background_index)] = {
                        "background": np.asarray(
                            sanitized_payload.get("background"),
                            dtype=np.float64,
                        ).copy(),
                        "radial_axis": np.asarray(
                            sanitized_payload.get("radial_axis"),
                            dtype=np.float64,
                        ).copy(),
                        "azimuth_axis": np.asarray(
                            sanitized_payload.get("azimuth_axis"),
                            dtype=np.float64,
                        ).copy(),
                        "raw_azimuth_axis": np.asarray(
                            sanitized_payload.get("raw_azimuth_axis"),
                            dtype=np.float64,
                        ).copy(),
                        "raw_to_gui_row_permutation": np.asarray(
                            sanitized_payload.get("raw_to_gui_row_permutation"),
                            dtype=np.int32,
                        ).copy(),
                        "transform_bundle": sanitized_payload.get("transform_bundle"),
                        "detector_shape": tuple(
                            int(v) for v in tuple(sanitized_payload.get("detector_shape", ()))[:2]
                        ),
                    }
            except Exception:
                pass
    return GeometryFitCakedProjectionSnapshot(
        caked_views_by_background=caked_views_by_background,
        projection_payload_by_background=projection_payload_by_background,
    )


def build_projection_view_signatures(
    *,
    current_background_index: int,
    required_indices: Sequence[int],
    background_images: Mapping[int, Mapping[str, object]],
    caked_views_by_background: Mapping[int, Mapping[str, object]],
    projection_payload_by_background: Mapping[int, Mapping[str, object]],
    manual_fit_requires_caked_space: bool,
    targeted_projection_view_mode: str,
    analysis_preview_bins: object,
    default_radial_bins: int,
    default_azimuth_bins: int,
    targeted_projection_view_signature: Callable[..., Mapping[str, object]],
    normalize_projection_view_signature: Callable[..., dict[str, object]],
) -> GeometryFitProjectionViewSignatures:
    current_caked_view = caked_views_by_background.get(int(current_background_index))
    try:
        current_radial_axis = np.asarray(
            dict(current_caked_view or {}).get("radial_axis"),
            dtype=np.float64,
        )
        current_azimuth_axis = np.asarray(
            dict(current_caked_view or {}).get("azimuth_axis"),
            dtype=np.float64,
        )
    except Exception:
        current_radial_axis = None
        current_azimuth_axis = None
    if (
        isinstance(current_radial_axis, np.ndarray)
        and current_radial_axis.size > 0
        and isinstance(current_azimuth_axis, np.ndarray)
        and current_azimuth_axis.size > 0
    ):
        analysis_bins = (
            int(current_radial_axis.size),
            int(current_azimuth_axis.size),
        )
    elif (
        isinstance(analysis_preview_bins, tuple)
        and len(analysis_preview_bins) == 2
        and all(int(value) > 0 for value in analysis_preview_bins)
    ):
        analysis_bins = (int(analysis_preview_bins[0]), int(analysis_preview_bins[1]))
    else:
        analysis_bins = (int(default_radial_bins), int(default_azimuth_bins))

    projection_view_mode = "caked" if manual_fit_requires_caked_space else targeted_projection_view_mode
    projection_view_signature_by_background: dict[int, dict[str, object]] = {}
    for idx in required_indices:
        background_payload = dict(background_images.get(int(idx)) or {})
        native_background = background_payload.get("native")
        try:
            detector_shape = tuple(
                int(v) for v in np.asarray(native_background, dtype=np.float64).shape[:2]
            )
        except Exception:
            detector_shape = None
        projection_view_signature_by_background[int(idx)] = normalize_projection_view_signature(
            targeted_projection_view_signature(
                int(idx),
                mode_override=projection_view_mode,
                caked_payload=(dict(projection_payload_by_background.get(int(idx)) or {}) or None),
                detector_shape=detector_shape,
                analysis_preview_bins=analysis_bins,
            ),
            int(idx),
        )
    return GeometryFitProjectionViewSignatures(
        analysis_bins=analysis_bins,
        projection_view_mode=str(projection_view_mode),
        projection_view_signature_by_background=projection_view_signature_by_background,
    )


def build_current_hit_table_cache_payload(
    *,
    current_background_index: int,
    required_indices: Sequence[int],
    requested_signatures: Mapping[int, object],
    requested_signature_summaries: Mapping[int, object],
    background_labels: Mapping[int, str],
    projection_view_mode: str,
    manual_fit_space_by_background: Mapping[int, object],
    projection_view_signature_by_background: Mapping[int, Mapping[str, object]],
    last_simulation_signature: object,
    stored_max_positions_local: object,
    source_snapshot_base_signature: Callable[[object], object],
    copy_hit_tables: Callable[[object], list[object]],
    cache_jsonable: Callable[[object], object],
    digest_payload: Callable[[object], object],
) -> dict[int, dict[str, object]]:
    current_hit_table_cache_by_background: dict[int, dict[str, object]] = {}
    if int(current_background_index) not in set(int(idx) for idx in required_indices):
        return current_hit_table_cache_by_background

    current_background_idx = int(current_background_index)
    requested_signature = requested_signatures.get(current_background_idx)
    requested_base_signature = source_snapshot_base_signature(requested_signature)
    current_base_signature = last_simulation_signature
    source_signature_match = bool(
        current_base_signature is not None and requested_base_signature == current_base_signature
    )
    current_source_tables: list[object] = []
    current_hit_table_reason = "base_simulation_signature_mismatch"
    if source_signature_match:
        try:
            current_source_tables = copy_hit_tables(stored_max_positions_local or [])
            current_hit_table_reason = (
                "matched" if current_source_tables else "missing_stored_max_positions_local"
            )
        except Exception:
            current_source_tables = []
            current_hit_table_reason = "stored_max_positions_local_unavailable"
    requested_base_signature_json = cache_jsonable(requested_base_signature)
    current_base_signature_json = cache_jsonable(current_base_signature)
    current_hit_table_cache_by_background[current_background_idx] = {
        "hit_tables": copy.deepcopy(list(current_source_tables or ())),
        "cache_metadata": {
            "background_index": int(current_background_idx),
            "background_label": background_labels.get(
                int(current_background_idx),
                f"background {int(current_background_idx) + 1}",
            ),
            "projection_view_mode": str(projection_view_mode),
            "fit_space": str(
                manual_fit_space_by_background.get(int(current_background_idx), "")
                or projection_view_mode
            ),
            "requested_signature": cache_jsonable(requested_signature),
            "requested_signature_summary": requested_signature_summaries.get(
                int(current_background_idx)
            ),
            "base_simulation_signature_match": bool(source_signature_match),
            "source_signature_match": bool(source_signature_match),
            "table_source_kind": "stored_max_positions_local",
            "table_base_signature": current_base_signature_json,
            "requested_base_signature": requested_base_signature_json,
            "requested_base_signature_digest": digest_payload(requested_base_signature_json),
            "table_base_signature_digest": digest_payload(current_base_signature_json),
            "current_base_signature_digest": digest_payload(current_base_signature_json),
            "projection_view_signature": copy.deepcopy(
                projection_view_signature_by_background.get(int(current_background_idx))
            ),
            "table_kind": "hit_tables" if current_source_tables else "unavailable",
            "hit_table_count": int(len(current_source_tables or ())),
            "cache_source": "current_hit_table_cache",
            "current_hit_table_cache_reason": str(current_hit_table_reason),
        },
    }
    return current_hit_table_cache_by_background


def assemble_geometry_fit_worker_job(
    *,
    core_fields: Mapping[str, object],
    selection_fields: Mapping[str, object],
    background_snapshot: GeometryFitBackgroundInputSnapshot,
    runtime_config_snapshot: GeometryFitRuntimeConfigSnapshot,
    caked_projection_snapshot: GeometryFitCakedProjectionSnapshot,
    projection_signatures: GeometryFitProjectionViewSignatures,
    live_rows_snapshot: GeometryFitLiveRowsSnapshot,
    cache_fields: Mapping[str, object],
    diagnostic_fields: Mapping[str, object],
    manual_fields: Mapping[str, object],
    callback_fields: Mapping[str, object],
) -> dict[str, object]:
    fit_sample_count = runtime_config_snapshot.fit_sample_count
    current_background_index = int(core_fields["current_background_index"])
    projection_view_signature = projection_signatures.projection_view_signature_by_background.get(
        int(current_background_index)
    )
    if not projection_view_signature:
        fallback = core_fields.get("projection_view_signature_fallback")
        projection_view_signature = fallback() if callable(fallback) else fallback
    projection_view_signature = copy.deepcopy(projection_view_signature)
    return {
        "job_id": int(core_fields["job_id"]),
        "stamp": str(core_fields["stamp"]),
        "log_path": Path(str(core_fields["log_path"])),
        "params": dict(core_fields["params"]),  # type: ignore[arg-type]
        "var_names": list(core_fields["var_names"]),  # type: ignore[arg-type]
        "preserve_live_theta": bool(core_fields["preserve_live_theta"]),
        "mosaic_params": dict(core_fields["mosaic_params"]),  # type: ignore[arg-type]
        "fit_solver_mosaic_params": (
            dict(runtime_config_snapshot.fit_solver_mosaic_params)
            if fit_sample_count is not None
            else None
        ),
        "fit_sample_count": fit_sample_count,
        "fit_config": core_fields["fit_config"],
        "geometry_runtime_cfg": runtime_config_snapshot.geometry_runtime_cfg,
        "theta_initial": float(core_fields["theta_initial"]),
        "current_background_index": current_background_index,
        "selection_applied": bool(selection_fields["selection_applied"]),
        "selected_background_indices": [
            int(idx) for idx in selection_fields["selected_background_indices"]  # type: ignore[index]
        ],
        "selection_error": selection_fields["selection_error"],
        "uses_shared_theta": bool(selection_fields["uses_shared_theta"]),
        "theta_metadata_applied": bool(selection_fields["theta_metadata_applied"]),
        "background_theta_values": [
            float(value) for value in selection_fields["background_theta_values"]  # type: ignore[index]
        ],
        "background_theta_error": selection_fields["background_theta_error"],
        "theta_offset": float(selection_fields["theta_offset"]),
        "joint_background_mode": bool(selection_fields["joint_background_mode"]),
        "required_indices": [
            int(idx) for idx in selection_fields["required_indices"]  # type: ignore[index]
        ],
        "primary_index": int(selection_fields["primary_index"]),
        "osc_files": [str(path) for path in core_fields["osc_files"]],  # type: ignore[index]
        "image_size": int(core_fields["image_size"]),
        "display_rotate_k": int(core_fields["display_rotate_k"]),
        "analysis_bins": projection_signatures.analysis_bins,
        "npt_rad": int(projection_signatures.analysis_bins[0]),
        "npt_azim": int(projection_signatures.analysis_bins[1]),
        "background_images": background_snapshot.background_images,
        "caked_views_by_background": caked_projection_snapshot.caked_views_by_background,
        "projection_payload_by_background": (
            caked_projection_snapshot.projection_payload_by_background
        ),
        "manual_pairs_by_background": background_snapshot.manual_pairs_by_background,
        "source_snapshots": background_snapshot.source_snapshots,
        "background_cache_by_index": {},
        "requested_signatures": background_snapshot.requested_signatures,
        "requested_signature_summaries": background_snapshot.requested_signature_summaries,
        "background_labels": background_snapshot.background_labels,
        "projection_view_mode": str(projection_signatures.projection_view_mode),
        "projection_view_signature": projection_view_signature,
        "projection_view_signature_by_background": copy.deepcopy(
            projection_signatures.projection_view_signature_by_background
        ),
        "theta_base_by_background": background_snapshot.theta_base_by_background,
        "theta_initial_by_background": background_snapshot.theta_initial_by_background,
        "source_snapshot_diagnostics": diagnostic_fields["source_snapshot_diagnostics"],
        "simulation_diagnostics": diagnostic_fields["simulation_diagnostics"],
        "live_cache_inventory": diagnostic_fields["live_cache_inventory"],
        "memory_intersection_cache": cache_fields["memory_intersection_cache"],
        "memory_intersection_cache_signature": cache_fields["memory_intersection_cache_signature"],
        "current_hit_table_cache_by_background": cache_fields[
            "current_hit_table_cache_by_background"
        ],
        "live_rows_by_background": live_rows_snapshot.live_rows_by_background,
        "live_rows_cache_metadata_by_background": (
            live_rows_snapshot.live_rows_cache_metadata_by_background
        ),
        "live_rows_signature_by_background": live_rows_snapshot.live_rows_signature_by_background,
        "live_rows_signature": cache_fields["memory_intersection_cache_signature"],
        "live_rows_handoff_diagnostics": live_rows_snapshot.live_rows_handoff_diagnostics,
        "manual_match_config": manual_fields["manual_match_config"],
        "manual_fit_space_by_background": dict(background_snapshot.manual_fit_space_by_background),
        "manual_caked_fit_space_required_by_background": dict(
            manual_fields["manual_caked_fit_space_required_by_background"]  # type: ignore[arg-type]
        ),
        "disable_auto_caked_route_for_gamma_gamma": bool(
            manual_fields["disable_auto_caked_route_for_gamma_gamma"]
        ),
        "skipped_manual_pair_backgrounds": dict(
            background_snapshot.skipped_manual_pair_backgrounds
        ),
        "pick_uses_caked_space": bool(manual_fields["pick_uses_caked_space"]),
        "geometry_manual_simulated_lookup": callback_fields["geometry_manual_simulated_lookup"],
        "geometry_manual_entry_display_coords": callback_fields[
            "geometry_manual_entry_display_coords"
        ],
        "geometry_manual_project_peaks_to_current_view": callback_fields[
            "geometry_manual_project_peaks_to_current_view"
        ],
        "unrotate_display_peaks": callback_fields["unrotate_display_peaks"],
        "display_to_native_sim_coords": callback_fields["display_to_native_sim_coords"],
        "native_detector_coords_to_detector_display_coords": callback_fields[
            "native_detector_coords_to_detector_display_coords"
        ],
        "select_fit_orientation": callback_fields["select_fit_orientation"],
        "apply_orientation_to_entries": callback_fields["apply_orientation_to_entries"],
        "orient_image_for_fit": callback_fields["orient_image_for_fit"],
        "apply_background_backend_orientation": callback_fields[
            "apply_background_backend_orientation"
        ],
        "solver_inputs": core_fields["solver_inputs"],
        "solve_fit": core_fields["solve_fit"],
        "execution_bindings": core_fields["execution_bindings"],
        "event_queue": core_fields["event_queue"],
        "enable_live_update_events": bool(core_fields["enable_live_update_events"]),
    }
