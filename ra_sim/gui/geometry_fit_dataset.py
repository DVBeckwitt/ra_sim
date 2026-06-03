"""Internal helpers for geometry-fit manual dataset assembly."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryManualDatasetInputSnapshot:
    """Normalized manual-pair inputs consumed by dataset assembly."""

    background_idx: int
    raw_selected_entries: list[object]
    selected_entry_inputs: list[dict[str, object]]
    selected_entries: list[dict[str, object]]
    manual_picker_truth_pairs: object
    manual_picker_truth_by_order: dict[object, dict[str, object]]
    use_caked_display: bool
    baseline_fit_params_i: dict[str, object]
    params_i: dict[str, object]
    theta_offset: float
    theta_initial: float
    reference_a: float | None
    reference_c: float | None
    reference_lambda: float | None

    def required_pairs_callback_payload(self) -> list[dict[str, object]]:
        callback_pairs: list[dict[str, object]] = []
        for entry in self.selected_entries:
            copied = dict(entry)
            copied.pop("source_label", None)
            callback_pairs.append(copied)
        return callback_pairs


@dataclass(frozen=True)
class GeometryProviderCoverageDeps:
    normalize_source_coverage_key: Callable[[object], object | None]
    source_coverage_alias_keys: Callable[[Mapping[str, object] | None], set[object]]
    source_coverage_key_payload: Callable[[object], object | None]
    source_coverage_filter_diagnostics: Callable[[Sequence[object] | None], Mapping[str, object]]
    cache_jsonable: Callable[[object], object]
    point_list: Callable[[object], Sequence[float] | None]
    normalize_point_frame: Callable[[object], str]
    put_simulated_point_fields: Callable[[dict[str, object], Sequence[float], object], None]
    coerce_nonnegative_index: Callable[[object], int | None]
    normalized_hkl: Callable[[object], tuple[int, int, int] | None]
    source_branch_index: Callable[[Mapping[str, object] | None], int | None]
    stable_group_identity: Callable[[object], object | None]
    group_identity: Callable[[Mapping[str, object] | None], object | None]
    group_identity_is_q_group: Callable[[object], bool]
    source_row_reuses_manual_caked_target: (
        Callable[[Mapping[str, object], Mapping[str, object]], bool]
    )
    provider_backed_source_row_for_target: Callable[..., Mapping[str, object] | None]
    pairs_use_caked_fit_space: Callable[[Sequence[Mapping[str, object]]], bool]
    zero_qr_coverage_branch_slot: object


@dataclass(frozen=True)
class GeometrySourceCandidateDeps:
    coerce_nonnegative_index: Callable[[object], int | None]
    trusted_full_reflection_identity: Callable[[Mapping[str, object] | None], bool]
    source_branch_index: Callable[[Mapping[str, object] | None], int | None]
    normalized_hkl: Callable[[object], tuple[int, int, int] | None]
    source_entry_hkl_matches: (
        Callable[[Mapping[str, object] | None, Mapping[str, object] | None], bool]
    )
    entry_source_label: Callable[[Mapping[str, object] | None], str]
    stable_group_identity: Callable[[object], object | None]
    is_zero_qr_00l: Callable[[Mapping[str, object] | None], bool]
    entry_display_point: Callable[[Mapping[str, object] | None], tuple[float, float] | None]
    entry_saved_simulated_current_view_point: (
        Callable[[Mapping[str, object] | None], tuple[float, float] | None]
    )
    legacy_saved_simulated_detector_hint: Callable[
        [Mapping[str, object] | None],
        object,
    ]
    candidate_current_view_point: (
        Callable[[Mapping[str, object] | None], tuple[float, float] | None]
    )
    candidate_current_view_frame: Callable[[Mapping[str, object] | None], str | None]
    candidate_point_for_frame: (
        Callable[..., tuple[float, float] | None]
    )
    compact_source_resolution_entry_payload: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None]
    )
    trace_candidate_inventory: Callable[[Sequence[Mapping[str, object]] | None], list[dict[str, object]]]
    legacy_dense_working_entry: (
        Callable[
            [Mapping[str, object] | None, Mapping[str, object] | None],
            dict[str, object] | None,
        ]
    )
    resolve_source_entry_candidate_pool: (
        Callable[[Mapping[str, object] | None], tuple[list[dict[str, object]], str | None]]
    )
    legacy_branch_hint_resolution: Callable[
        [Mapping[str, object] | None],
        tuple[int | None, str | None, str | None],
    ]
    legacy_geometry_hint: Callable[
        [Mapping[str, object] | None, Sequence[Mapping[str, object]] | None],
        tuple[str | None, tuple[float, float] | None, float],
    ]
    canonicalize_live_source_entry: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None]
    )
    is_canonical_live_source_entry: (
        Callable[[Mapping[str, object] | None], tuple[bool, str | None]]
    )
    apply_refined_simulated_override: Callable[..., dict[str, object] | None]
    cache_jsonable: Callable[[object], object]
    background_current_view_frame: Callable[[Mapping[str, object] | None], str | None]


def _source_row_key(
    entry: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    try:
        return (
            int(entry.get("source_table_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        return None


def _source_reflection_row_key(
    entry: Mapping[str, object] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> tuple[int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    if not deps.trusted_full_reflection_identity(entry):
        return None
    try:
        return (
            int(entry.get("source_reflection_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        return None

def _source_locator_payload(
    entry: Mapping[str, object] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        return {
            "source_reflection_index": None,
            "source_reflection_namespace": None,
            "source_reflection_is_full": None,
            "source_table_index": None,
            "source_row_index": None,
            "source_branch_index": None,
            "source_peak_index": None,
        }
    return {
        "source_reflection_index": entry.get("source_reflection_index"),
        "source_reflection_namespace": entry.get("source_reflection_namespace"),
        "source_reflection_is_full": entry.get("source_reflection_is_full"),
        "source_table_index": entry.get("source_table_index"),
        "source_row_index": entry.get("source_row_index"),
        "source_branch_index": deps.source_branch_index(entry),
        "source_peak_index": entry.get("source_peak_index"),
    }


def _source_locator_identity_match(
    saved_entry: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> bool:
    saved_locator = _source_locator_payload(saved_entry, deps=deps)
    candidate_locator = _source_locator_payload(candidate, deps=deps)
    comparable_keys = [key for key, value in saved_locator.items() if value is not None]
    if not comparable_keys:
        return False
    return all(candidate_locator.get(key) == saved_locator.get(key) for key in comparable_keys)


def _filter_hkl_candidates(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    filtered: list[dict[str, object]] = []
    excluded_missing_hkl: list[dict[str, object]] = []
    excluded_mismatched_hkl: list[dict[str, object]] = []
    target_hkl = deps.normalized_hkl(entry.get("hkl") if isinstance(entry, Mapping) else None)
    for raw_candidate in candidates or ():
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        candidate_hkl = deps.normalized_hkl(candidate.get("hkl"))
        if candidate_hkl is None:
            excluded_missing_hkl.append(candidate)
            continue
        if (
            target_hkl is not None
            and tuple(candidate_hkl) != tuple(target_hkl)
            and not deps.source_entry_hkl_matches(entry, candidate)
        ):
            excluded_mismatched_hkl.append(candidate)
            continue
        filtered.append(candidate)
    return filtered, excluded_missing_hkl, excluded_mismatched_hkl


def _candidate_matches_group_constraints(
    entry: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> bool:
    if not isinstance(entry, Mapping):
        return True
    if not isinstance(candidate, Mapping):
        return False
    if deps.entry_source_label(candidate) != deps.entry_source_label(entry):
        return False
    required_branch_group = deps.stable_group_identity(entry.get("branch_group_key"))
    if required_branch_group is not None:
        candidate_branch_group = deps.stable_group_identity(candidate.get("branch_group_key"))
        if candidate_branch_group != required_branch_group:
            return False

    required_q_groups: list[object] = []
    for key in ("q_group_key", "source_q_group_key"):
        required_q_group = deps.stable_group_identity(entry.get(key))
        if required_q_group is not None and not any(
            existing == required_q_group for existing in required_q_groups
        ):
            required_q_groups.append(required_q_group)
    if required_q_groups:
        candidate_q_groups = [
            deps.stable_group_identity(candidate.get(key))
            for key in ("q_group_key", "source_q_group_key")
        ]
        candidate_q_groups = [value for value in candidate_q_groups if value is not None]
        if not candidate_q_groups:
            return False
        for required_q_group in required_q_groups:
            if not any(candidate_q_group == required_q_group for candidate_q_group in candidate_q_groups):
                return False
    return True


def _filter_group_candidates(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
) -> list[dict[str, object]]:
    return [
        dict(candidate)
        for candidate in (candidates or ())
        if isinstance(candidate, Mapping)
        and _candidate_matches_group_constraints(entry, candidate, deps=deps)
    ]


def _filter_source_branch_candidates(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
    required_branch_index: int | None = None,
) -> list[dict[str, object]]:
    candidate_pool = [
        dict(candidate) for candidate in (candidates or ()) if isinstance(candidate, Mapping)
    ]
    if not candidate_pool:
        return []
    if deps.is_zero_qr_00l(entry):
        return candidate_pool
    branch_idx = (
        int(required_branch_index)
        if required_branch_index in {0, 1}
        else deps.source_branch_index(entry)
    )
    if branch_idx in {0, 1}:
        return [
            dict(candidate)
            for candidate in candidate_pool
            if deps.source_branch_index(candidate) == int(branch_idx)
        ]
    return candidate_pool


def _source_candidate_filter_inventory(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
    required_branch_index: int | None = None,
) -> dict[str, object]:
    candidate_pool = [
        dict(candidate) for candidate in (candidates or ()) if isinstance(candidate, Mapping)
    ]
    hkl_candidates, missing_hkl_candidates, mismatched_hkl_candidates = _filter_hkl_candidates(
        entry,
        candidate_pool,
        deps=deps,
    )
    group_candidates = _filter_group_candidates(entry, hkl_candidates, deps=deps)
    branch_candidates = _filter_source_branch_candidates(
        entry,
        group_candidates,
        deps=deps,
        required_branch_index=required_branch_index,
    )
    return {
        "candidate_pool": candidate_pool,
        "hkl_candidates": hkl_candidates,
        "missing_hkl_candidates": missing_hkl_candidates,
        "mismatched_hkl_candidates": mismatched_hkl_candidates,
        "group_candidates": group_candidates,
        "branch_candidates": branch_candidates,
    }


def _candidate_sort_identity(
    candidate: Mapping[str, object],
    *,
    deps: GeometrySourceCandidateDeps,
) -> tuple[int, int, int, int]:
    sentinel = 1 << 30
    reflection_idx = deps.coerce_nonnegative_index(
        candidate.get("source_reflection_index") if isinstance(candidate, Mapping) else None
    )
    table_idx = deps.coerce_nonnegative_index(
        candidate.get("source_table_index") if isinstance(candidate, Mapping) else None
    )
    row_idx = deps.coerce_nonnegative_index(
        candidate.get("source_row_index") if isinstance(candidate, Mapping) else None
    )
    branch_idx = deps.source_branch_index(candidate)
    return (
        int(reflection_idx) if reflection_idx is not None else sentinel,
        int(table_idx) if table_idx is not None else sentinel,
        int(row_idx) if row_idx is not None else sentinel,
        int(branch_idx) if branch_idx in {0, 1} else sentinel,
    )


def _dedupe_geometry_tied_candidates(
    candidates: Sequence[Mapping[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
    frame_name: str,
) -> list[dict[str, object]]:
    unique_candidates: dict[tuple[object, ...], dict[str, object]] = {}
    for raw_candidate in candidates or ():
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        dedupe_key = (
            deps.normalized_hkl(candidate.get("hkl")),
            deps.source_branch_index(candidate),
            str(frame_name),
            deps.candidate_point_for_frame(candidate, frame_name=str(frame_name)),
            tuple(sorted(_source_locator_payload(candidate, deps=deps).items())),
        )
        existing = unique_candidates.get(dedupe_key)
        if existing is None or _candidate_sort_identity(
            candidate,
            deps=deps,
        ) < _candidate_sort_identity(existing, deps=deps):
            unique_candidates[dedupe_key] = candidate
    return [
        dict(candidate)
        for _key, candidate in sorted(
            unique_candidates.items(),
            key=lambda item: _candidate_sort_identity(item[1], deps=deps),
        )
    ]


def _select_source_candidate(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
    *,
    deps: GeometrySourceCandidateDeps,
    saved_identity_entry: Mapping[str, object] | None = None,
    target_point: tuple[float, float] | None = None,
    frame_name: str | None = None,
    tie_tolerance: float = 0.0,
    required_branch_index: int | None = None,
    fail_on_ambiguous_tie: bool = False,
    require_all_candidate_points: bool = False,
    allow_missing_target_selection: bool = True,
    allow_single_candidate_without_score: bool = False,
) -> dict[str, object]:
    inventory = _source_candidate_filter_inventory(
        entry,
        candidates,
        deps=deps,
        required_branch_index=required_branch_index,
    )
    branch_candidates = [
        dict(candidate)
        for candidate in inventory.get("branch_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    result: dict[str, object] = {
        **inventory,
        "selected": None,
        "selected_score": None,
        "selected_point": None,
        "selected_frame": frame_name,
        "scored_candidates": [],
        "score_inventory": [],
        "best_score": None,
        "second_best_score": None,
        "tied_candidates": [],
        "identity_tied_candidates": [],
        "selection_tie_breaker": None,
        "failure_reason": None,
    }
    if not inventory.get("hkl_candidates"):
        result["failure_reason"] = "missing_candidate_pool"
        return result
    if not inventory.get("group_candidates"):
        result["failure_reason"] = "group_constraint_no_match"
        return result
    if not branch_candidates:
        result["failure_reason"] = "branch_constraint_no_match"
        return result

    resolved_target_point = target_point
    if resolved_target_point is None:
        resolved_target_point = deps.entry_display_point(entry)
    if resolved_target_point is None:
        resolved_target_point = deps.entry_saved_simulated_current_view_point(entry)
    if resolved_target_point is None:
        if allow_missing_target_selection or (
            allow_single_candidate_without_score and len(branch_candidates) == 1
        ):
            selected = dict(branch_candidates[0])
            result["selected"] = selected
            result["selection_tie_breaker"] = "missing_target_point_first_candidate"
            return result
        result["failure_reason"] = "missing_target_point"
        return result

    scored_candidates: list[
        tuple[float, dict[str, object], tuple[float, float], str | None]
    ] = []
    for candidate in branch_candidates:
        candidate_point = (
            deps.candidate_point_for_frame(candidate, frame_name=str(frame_name))
            if frame_name is not None
            else deps.candidate_current_view_point(candidate)
        )
        if candidate_point is None:
            if require_all_candidate_points:
                result["failure_reason"] = "missing_candidate_point"
                return result
            continue
        score = float(
            math.hypot(
                float(candidate_point[0]) - float(resolved_target_point[0]),
                float(candidate_point[1]) - float(resolved_target_point[1]),
            )
        )
        if not math.isfinite(float(score)):
            result["failure_reason"] = "nonfinite_score"
            return result
        candidate_frame = (
            str(frame_name)
            if frame_name is not None
            else deps.candidate_current_view_frame(candidate)
        )
        scored_candidates.append(
            (
                float(score),
                dict(candidate),
                (float(candidate_point[0]), float(candidate_point[1])),
                candidate_frame,
            )
        )

    if not scored_candidates:
        if allow_single_candidate_without_score and len(branch_candidates) == 1:
            selected = dict(branch_candidates[0])
            result["selected"] = selected
            result["selection_tie_breaker"] = "single_candidate_without_geometry"
            return result
        result["failure_reason"] = "missing_candidate_point"
        return result

    scored_candidates.sort(
        key=lambda item: (
            float(item[0]),
            *_candidate_sort_identity(item[1], deps=deps),
        )
    )
    score_inventory = [
        {
            **(deps.compact_source_resolution_entry_payload(candidate) or {}),
            "score": float(score),
            "frame_name": str(candidate_frame or ""),
            "candidate_point": [
                float(candidate_point[0]),
                float(candidate_point[1]),
            ],
            "saved_target_point": [
                float(resolved_target_point[0]),
                float(resolved_target_point[1]),
            ],
        }
        for score, candidate, candidate_point, candidate_frame in scored_candidates
    ]
    best_score = float(scored_candidates[0][0])
    second_best_score = float(scored_candidates[1][0]) if len(scored_candidates) > 1 else None
    try:
        tie_window = float(tie_tolerance)
    except Exception:
        tie_window = 0.0
    if not math.isfinite(float(tie_window)) or tie_window < 0.0:
        tie_window = 0.0
    tied_scored = [
        item
        for item in scored_candidates
        if abs(float(item[0]) - float(best_score)) <= float(tie_window)
    ]
    identity_tied_scored = [
        item
        for item in tied_scored
        if _source_locator_identity_match(saved_identity_entry, item[1], deps=deps)
    ]
    chosen = tied_scored[0]
    selection_tie_breaker: str | None = None
    tied_candidates = [dict(item[1]) for item in tied_scored]
    identity_tied_candidates = [dict(item[1]) for item in identity_tied_scored]
    if fail_on_ambiguous_tie and len(tied_scored) > 1:
        deduped_tied_candidates = _dedupe_geometry_tied_candidates(
            tied_candidates,
            deps=deps,
            frame_name=str(frame_name or ""),
        )
        tied_candidates = [dict(candidate) for candidate in deduped_tied_candidates]
        identity_tied_candidates = [
            dict(candidate)
            for candidate in deduped_tied_candidates
            if _source_locator_identity_match(saved_identity_entry, candidate, deps=deps)
        ]
        if len(deduped_tied_candidates) == 1:
            selection_tie_breaker = "duplicate_live_rows_canonicalized"
            chosen_candidate = dict(deduped_tied_candidates[0])
            chosen = next(
                (item for item in scored_candidates if dict(item[1]) == chosen_candidate),
                scored_candidates[0],
            )
        elif len(identity_tied_candidates) == 1:
            selection_tie_breaker = "saved_source_identity"
            chosen_candidate = dict(identity_tied_candidates[0])
            chosen = next(
                (item for item in scored_candidates if dict(item[1]) == chosen_candidate),
                scored_candidates[0],
            )
        else:
            result.update(
                {
                    "scored_candidates": scored_candidates,
                    "score_inventory": score_inventory,
                    "best_score": best_score,
                    "second_best_score": second_best_score,
                    "tied_candidates": tied_candidates,
                    "identity_tied_candidates": identity_tied_candidates,
                    "failure_reason": "ambiguous_geometry_tie",
                }
            )
            return result
    elif len(identity_tied_scored) == 1:
        chosen = identity_tied_scored[0]
        if len(tied_scored) > 1:
            selection_tie_breaker = "saved_source_identity"

    result.update(
        {
            "selected": dict(chosen[1]),
            "selected_score": float(chosen[0]),
            "selected_point": [
                float(chosen[2][0]),
                float(chosen[2][1]),
            ],
            "selected_frame": chosen[3],
            "scored_candidates": scored_candidates,
            "score_inventory": score_inventory,
            "best_score": best_score,
            "second_best_score": second_best_score,
            "tied_candidates": tied_candidates,
            "identity_tied_candidates": identity_tied_candidates,
            "selection_tie_breaker": selection_tie_breaker,
        }
    )
    return result


def _resolve_legacy_dense_source_entry(
    entry: Mapping[str, object] | None,
    *,
    raw_saved_entry: Mapping[str, object] | None,
    deps: GeometrySourceCandidateDeps,
    use_caked_display: bool = False,
) -> tuple[dict[str, object] | None, str | None, dict[str, object]]:
    diagnostics: dict[str, object] = {
        "legacy_raw_saved_entry": deps.compact_source_resolution_entry_payload(
            raw_saved_entry
        ),
        "legacy_normalized_saved_entry": deps.compact_source_resolution_entry_payload(entry),
    }
    working_entry = deps.legacy_dense_working_entry(entry, raw_saved_entry)
    saved_identity_entry = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
    diagnostics["legacy_working_entry"] = deps.compact_source_resolution_entry_payload(
        working_entry
    )
    if not isinstance(working_entry, Mapping):
        return None, None, diagnostics

    candidate_pool, candidate_pool_source = deps.resolve_source_entry_candidate_pool(working_entry)
    candidate_inventory = _source_candidate_filter_inventory(
        working_entry,
        candidate_pool,
        deps=deps,
    )
    hkl_candidates = [
        dict(candidate)
        for candidate in candidate_inventory.get("hkl_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    group_candidates = [
        dict(candidate)
        for candidate in candidate_inventory.get("group_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    missing_hkl_candidates = [
        dict(candidate)
        for candidate in candidate_inventory.get("missing_hkl_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    mismatched_hkl_candidates = [
        dict(candidate)
        for candidate in candidate_inventory.get("mismatched_hkl_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    diagnostics["legacy_candidate_pool_source"] = candidate_pool_source
    diagnostics["legacy_candidate_count_initial"] = int(len(hkl_candidates))
    diagnostics["legacy_candidate_inventory"] = deps.trace_candidate_inventory(hkl_candidates)
    diagnostics["legacy_candidate_count_after_hkl"] = int(len(hkl_candidates))
    diagnostics["legacy_excluded_missing_hkl_candidates"] = deps.trace_candidate_inventory(
        missing_hkl_candidates
    )
    diagnostics["legacy_excluded_mismatched_hkl_candidates"] = deps.trace_candidate_inventory(
        mismatched_hkl_candidates
    )
    diagnostics["legacy_candidate_count_after_group"] = int(len(group_candidates))
    diagnostics["legacy_candidate_inventory_after_group"] = deps.trace_candidate_inventory(
        group_candidates
    )
    if not hkl_candidates or not group_candidates:
        diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_candidate_pool"
        return None, None, diagnostics

    branch_idx, branch_source, branch_reason = deps.legacy_branch_hint_resolution(working_entry)
    diagnostics["legacy_branch_hint_source"] = branch_source
    diagnostics["legacy_branch_hint_reason"] = branch_reason
    if branch_idx not in {0, 1}:
        diagnostics["legacy_failure_reason"] = (
            f"legacy_rebind_{branch_reason or 'missing_branch_hint'}"
        )
        return None, None, diagnostics

    branch_inventory = _source_candidate_filter_inventory(
        working_entry,
        candidate_pool,
        deps=deps,
        required_branch_index=int(branch_idx),
    )
    branch_candidates = [
        dict(candidate)
        for candidate in branch_inventory.get("branch_candidates", [])
        if isinstance(candidate, Mapping)
    ]
    diagnostics["legacy_candidate_count_after_branch"] = int(len(branch_candidates))
    diagnostics["legacy_candidate_inventory_after_branch"] = deps.trace_candidate_inventory(
        branch_candidates
    )
    if not branch_candidates:
        diagnostics["legacy_failure_reason"] = "legacy_rebind_no_candidate_on_branch"
        return None, None, diagnostics

    geometry_hint_source, geometry_target_point, tie_tolerance = deps.legacy_geometry_hint(
        working_entry,
        branch_candidates,
    )
    diagnostics["legacy_geometry_hint_source"] = geometry_hint_source
    diagnostics["legacy_tie_tolerance"] = (
        float(tie_tolerance) if math.isfinite(float(tie_tolerance)) else None
    )

    selection = _select_source_candidate(
        working_entry,
        candidate_pool,
        deps=deps,
        saved_identity_entry=saved_identity_entry,
        target_point=geometry_target_point,
        frame_name=geometry_hint_source,
        tie_tolerance=tie_tolerance,
        required_branch_index=int(branch_idx),
        fail_on_ambiguous_tie=True,
        require_all_candidate_points=True,
        allow_missing_target_selection=False,
        allow_single_candidate_without_score=True,
    )
    score_inventory = [
        dict(item) for item in selection.get("score_inventory", []) if isinstance(item, Mapping)
    ]
    selected_geometry_count = 1 if isinstance(selection.get("selected"), Mapping) else 0
    diagnostics["legacy_candidate_count_after_geometry"] = int(
        len(score_inventory) if score_inventory else selected_geometry_count
    )
    if score_inventory:
        diagnostics["legacy_geometry_candidate_scores"] = score_inventory
    if selection.get("best_score") is not None:
        diagnostics["legacy_best_score"] = float(selection["best_score"])
    if selection.get("second_best_score") is not None:
        diagnostics["legacy_second_best_score"] = float(selection["second_best_score"])
    else:
        diagnostics["legacy_second_best_score"] = None
    if selection.get("selection_tie_breaker"):
        diagnostics["legacy_selection_tie_breaker"] = str(
            selection.get("selection_tie_breaker")
        )
    chosen_live_row = (
        dict(selection.get("selected")) if isinstance(selection.get("selected"), Mapping) else None
    )
    if chosen_live_row is None:
        failure_reason = str(selection.get("failure_reason") or "")
        if failure_reason == "missing_target_point":
            diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_geometry_hint"
        elif failure_reason == "missing_candidate_point":
            diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_geometry_candidate"
        elif failure_reason == "nonfinite_score":
            diagnostics["legacy_failure_reason"] = "legacy_rebind_nonfinite_geometry_score"
        elif failure_reason == "branch_constraint_no_match":
            diagnostics["legacy_failure_reason"] = "legacy_rebind_no_candidate_on_branch"
        elif failure_reason == "ambiguous_geometry_tie":
            diagnostics["legacy_failure_reason"] = "legacy_rebind_ambiguous_geometry_tie"
            diagnostics["legacy_geometry_tied_candidates"] = deps.trace_candidate_inventory(
                selection.get("tied_candidates", [])
            )
            diagnostics["legacy_geometry_identity_tied_candidates"] = (
                deps.trace_candidate_inventory(selection.get("identity_tied_candidates", []))
            )
        else:
            diagnostics["legacy_failure_reason"] = (
                "legacy_rebind_missing_candidate_pool"
                if failure_reason in {"missing_candidate_pool", "group_constraint_no_match"}
                else "legacy_rebind_ambiguous_geometry_tie"
            )
        return None, None, diagnostics

    canonical_live_row = deps.canonicalize_live_source_entry(chosen_live_row)
    canonical_ok, canonical_reason = deps.is_canonical_live_source_entry(canonical_live_row)
    if not canonical_ok or not isinstance(canonical_live_row, Mapping):
        diagnostics["legacy_failure_reason"] = (
            f"legacy_rebind_noncanonical_live_row:{canonical_reason or 'unknown'}"
        )
        return None, None, diagnostics

    fit_bound_entry = deps.apply_refined_simulated_override(
        dict(working_entry),
        dict(canonical_live_row),
        prefer_caked_display=bool(use_caked_display),
    )
    canonical_fit_ok, canonical_fit_reason = deps.is_canonical_live_source_entry(
        fit_bound_entry
    )
    if not canonical_fit_ok or not isinstance(fit_bound_entry, Mapping):
        diagnostics["legacy_failure_reason"] = (
            "legacy_rebind_fit_bound_identity_loss"
            if canonical_fit_reason is None
            else f"legacy_rebind_fit_bound_identity_loss:{canonical_fit_reason}"
        )
        return None, None, diagnostics

    diagnostics["legacy_chosen_live_row"] = deps.compact_source_resolution_entry_payload(
        canonical_live_row
    )
    diagnostics["legacy_selected_source_identity_fields"] = _source_locator_payload(
        canonical_live_row,
        deps=deps,
    )
    diagnostics["legacy_saved_background_current_view_point"] = deps.cache_jsonable(
        deps.entry_display_point(entry)
    )
    diagnostics["legacy_saved_background_current_view_frame"] = deps.background_current_view_frame(
        entry
    )
    diagnostics["legacy_selected_live_simulated_current_view_point"] = deps.cache_jsonable(
        deps.candidate_current_view_point(canonical_live_row)
    )
    diagnostics["legacy_selected_live_simulated_current_view_frame"] = (
        deps.candidate_current_view_frame(canonical_live_row)
    )
    candidate_frame = deps.candidate_current_view_frame(canonical_live_row)
    background_frame = deps.background_current_view_frame(entry)
    candidate_point = deps.candidate_current_view_point(canonical_live_row)
    background_point = deps.entry_display_point(entry)
    diagnostics["legacy_selected_to_background_distance_px"] = (
        float(
            math.hypot(
                float(candidate_point[0]) - float(background_point[0]),
                float(candidate_point[1]) - float(background_point[1]),
            )
        )
        if candidate_frame == background_frame
        and candidate_point is not None
        and background_point is not None
        else None
    )
    diagnostics["legacy_saved_simulated_detector_hint"] = deps.cache_jsonable(
        deps.legacy_saved_simulated_detector_hint(saved_identity_entry)
    )
    diagnostics["legacy_fit_bound_entry"] = deps.compact_source_resolution_entry_payload(
        fit_bound_entry
    )
    resolution_kind = (
        "legacy_dense_q_group_rebind"
        if candidate_pool_source == "q_group"
        else "legacy_dense_hkl_rebind"
    )
    return dict(fit_bound_entry), resolution_kind, diagnostics


def collect_geometry_manual_dataset_inputs(
    *,
    background_idx: int,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: Any,
    manual_fit_requires_caked_space: bool,
    pair_enabled: Callable[[object], bool],
    background_detector_pair_for_frame: Callable[[Mapping[str, object], str], object | None],
    entry_source_label: Callable[[Mapping[str, object] | None], str],
    build_manual_picker_truth_pairs: Callable[..., object],
    truth_by_order_key: Callable[[object], dict[object, dict[str, object]]],
    finite_float: Callable[[object], float | None],
    raw_selected_entries: Sequence[object] | None = None,
    use_caked_display: bool | None = None,
) -> GeometryManualDatasetInputSnapshot:
    if use_caked_display is None:
        use_caked_display = bool(manual_fit_requires_caked_space)
        pick_uses_caked_space = getattr(manual_dataset_bindings, "pick_uses_caked_space", None)
        if callable(pick_uses_caked_space):
            try:
                use_caked_display = bool(use_caked_display or pick_uses_caked_space())
            except Exception:
                pass

    selected_raw_entries = (
        list(raw_selected_entries)
        if raw_selected_entries is not None
        else [
            entry
            for entry in (manual_dataset_bindings.geometry_manual_pairs_for_index(background_idx) or ())
            if pair_enabled(entry)
        ]
    )
    if not selected_raw_entries:
        raise RuntimeError(f"background {background_idx + 1} has no saved manual geometry pairs")

    selected_entry_inputs: list[dict[str, object]] = []
    refresh_pair_entry = getattr(
        manual_dataset_bindings,
        "geometry_manual_refresh_pair_entry",
        None,
    )
    if callable(refresh_pair_entry):
        for raw_entry in selected_raw_entries:
            raw_saved_entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else None
            refreshed = refresh_pair_entry(raw_entry)
            normalized_entry = (
                dict(refreshed)
                if isinstance(refreshed, Mapping)
                else (dict(raw_entry) if isinstance(raw_entry, Mapping) else None)
            )
            if normalized_entry is None:
                continue
            if background_detector_pair_for_frame(normalized_entry, "native_detector") is not None:
                normalized_entry.setdefault(
                    "background_detector_frame_provenance",
                    "geometry_manual_refresh_pair_entry",
                )
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": (
                        raw_saved_entry
                        if isinstance(raw_saved_entry, dict)
                        else dict(normalized_entry)
                    ),
                    "entry": dict(normalized_entry),
                }
            )
    else:
        for raw_entry in selected_raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": dict(raw_entry),
                    "entry": dict(raw_entry),
                }
            )

    selected_entries = [
        {
            **dict(item["entry"]),
            "source_label": entry_source_label(
                item["entry"] if isinstance(item.get("entry"), Mapping) else None
            ),
        }
        for item in selected_entry_inputs
        if isinstance(item.get("entry"), Mapping)
    ]

    manual_picker_truth_pairs = build_manual_picker_truth_pairs(
        background_idx,
        [
            dict(item["raw_saved_entry"])
            for item in selected_entry_inputs
            if isinstance(item.get("raw_saved_entry"), Mapping)
        ],
        refresh_pair_entry=refresh_pair_entry,
    )
    manual_picker_truth_by_order = truth_by_order_key(manual_picker_truth_pairs)

    baseline_fit_params_i = dict(base_fit_params or {})
    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    theta_initial = float(theta_base + theta_offset)
    params_i["theta_initial"] = float(theta_initial)

    return GeometryManualDatasetInputSnapshot(
        background_idx=int(background_idx),
        raw_selected_entries=list(selected_raw_entries),
        selected_entry_inputs=selected_entry_inputs,
        selected_entries=selected_entries,
        manual_picker_truth_pairs=manual_picker_truth_pairs,
        manual_picker_truth_by_order=manual_picker_truth_by_order,
        use_caked_display=bool(use_caked_display),
        baseline_fit_params_i=baseline_fit_params_i,
        params_i=params_i,
        theta_offset=float(theta_offset),
        theta_initial=float(theta_initial),
        reference_a=finite_float(params_i.get("a")),
        reference_c=finite_float(params_i.get("c")),
        reference_lambda=finite_float(params_i.get("lambda")),
    )


def augment_source_rows_with_provider_coverage(
    rows: Sequence[object] | None,
    diagnostics: Mapping[str, object] | None,
    *,
    selected_entry_inputs: Sequence[Mapping[str, object]],
    selected_entries: Sequence[Mapping[str, object]],
    manual_picker_truth_by_order: Mapping[object, Mapping[str, object]],
    background_idx: int,
    use_caked_display: bool,
    deps: GeometryProviderCoverageDeps,
    require_provider_backed_rows: bool = False,
    allow_saved_coordinate_materialization: bool = True,
    include_coverage_diagnostics: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    augmented_rows = [dict(item) for item in (rows or ()) if isinstance(item, Mapping)]
    augmented_diagnostics = dict(diagnostics or {})
    coverage_keys = {
        key for row in augmented_rows for key in deps.source_coverage_alias_keys(row)
    }

    def _row_is_provider_backed_source_coverage(row: Mapping[str, object]) -> bool:
        return bool(row.get("provider_backed_live_source_row", False)) or (
            str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage"
        )

    provider_backed_keys = {
        key
        for row in augmented_rows
        if _row_is_provider_backed_source_coverage(row)
        for key in deps.source_coverage_alias_keys(row)
    }
    materialized_rows: list[dict[str, object]] = []
    promoted_rows: list[dict[str, object]] = []
    point_missing: list[dict[str, object]] = []

    def _source_identity_value(
        item: Mapping[str, object],
        *keys: str,
    ) -> int | None:
        for key in keys:
            value = deps.coerce_nonnegative_index(item.get(key))
            if value is not None and value >= 0:
                return int(value)
        return None

    def _finite_point_for_keys(
        item: Mapping[str, object],
        x_key: str,
        y_key: str,
    ) -> tuple[float, float] | None:
        try:
            point = (float(item.get(x_key)), float(item.get(y_key)))
        except Exception:
            return None
        if not (math.isfinite(point[0]) and math.isfinite(point[1])):
            return None
        return float(point[0]), float(point[1])

    def _points_for_tuple_keys(
        item: Mapping[str, object],
        keys: Sequence[str],
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for key in keys:
            point = deps.point_list(item.get(key))
            if point is None:
                continue
            points.append((float(point[0]), float(point[1])))
        return points

    def _points_for_coordinate_keys(
        item: Mapping[str, object],
        keys: Sequence[tuple[str, str]],
    ) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for x_key, y_key in keys:
            point = _finite_point_for_keys(item, x_key, y_key)
            if point is None:
                continue
            points.append((float(point[0]), float(point[1])))
        return points

    def _target_match_points(item: Mapping[str, object]) -> list[tuple[float, float]]:
        return _points_for_tuple_keys(
            item,
            (
                "manual_selected_simulated_point",
                "provider_selected_simulated_point",
                "selected_live_simulated_current_view_point",
                "simulated_point",
                "sim_display",
                "sim_native",
                "sim_caked_display",
            ),
        ) + _points_for_coordinate_keys(
            item,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("refined_sim_x", "refined_sim_y"),
                ("refined_sim_caked_x", "refined_sim_caked_y"),
                ("simulated_two_theta_deg", "simulated_phi_deg"),
            ),
        )

    def _row_match_points(item: Mapping[str, object]) -> list[tuple[float, float]]:
        return _points_for_tuple_keys(
            item,
            (
                "sim_display",
                "sim_native",
                "sim_caked_display",
            ),
        ) + _points_for_coordinate_keys(
            item,
            (
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
                ("sim_col_raw", "sim_row_raw"),
                ("sim_col", "sim_row"),
                ("display_col", "display_row"),
                ("caked_x", "caked_y"),
                ("two_theta_deg", "phi_deg"),
                ("simulated_two_theta_deg", "simulated_phi_deg"),
            ),
        )

    def _rows_use_caked_fit_space() -> bool:
        return bool(use_caked_display or deps.pairs_use_caked_fit_space(selected_entries))

    def _row_is_current_dynamic_provider_backed(
        row: Mapping[str, object],
    ) -> bool:
        if not _row_is_provider_backed_source_coverage(row):
            return False
        if not _rows_use_caked_fit_space():
            return True
        if str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage":
            return False
        actual_source = str(row.get("actual_source") or "").strip()
        source_kind = str(row.get("source_kind") or "").strip()
        projection_frame = str(row.get("projection_frame") or "").strip()
        provenance = str(row.get("coordinate_provenance") or "").strip()
        if actual_source != "sim_visual_caked_deg" or source_kind != "sim_visual_caked_deg":
            return False
        if projection_frame != "caked_display":
            return False
        if provenance and provenance != "trial_geometry_projection":
            return False
        if "is_dynamic_trial_row" in row and row.get("is_dynamic_trial_row") is not True:
            return False
        return True

    def _current_provider_backed_source_row_exists(target_key: object) -> bool:
        for row in augmented_rows:
            if target_key not in deps.source_coverage_alias_keys(row):
                continue
            if _row_is_current_dynamic_provider_backed(row):
                return True
        return False

    def _fresh_row_match_distance(
        target: Mapping[str, object],
        row: Mapping[str, object],
    ) -> float | None:
        target_points = _target_match_points(target)
        row_points = _row_match_points(row)
        if not target_points or not row_points:
            return None
        best_distance: float | None = None
        for target_x, target_y in target_points:
            for row_x, row_y in row_points:
                distance = math.hypot(float(row_x) - float(target_x), float(row_y) - float(target_y))
                if not math.isfinite(distance):
                    continue
                if best_distance is None or float(distance) < float(best_distance):
                    best_distance = float(distance)
        return best_distance

    def _best_unique_match_by_score_and_distance(
        target: Mapping[str, object],
        scored_rows: Sequence[tuple[int, int, dict[str, object]]],
    ) -> list[tuple[int, dict[str, object]]]:
        if not scored_rows:
            return []
        best_score = max(score for score, _row_idx, _row in scored_rows)
        best_scored = [
            (row_idx, row)
            for score, row_idx, row in scored_rows
            if int(score) == int(best_score)
        ]
        if len(best_scored) <= 1:
            return best_scored
        distanced: list[tuple[float, int, dict[str, object]]] = []
        for row_idx, row in best_scored:
            distance = _fresh_row_match_distance(target, row)
            if distance is None:
                continue
            distanced.append((float(distance), int(row_idx), row))
        if not distanced:
            return []
        best_distance = min(distance for distance, _row_idx, _row in distanced)
        best = [
            (row_idx, row)
            for distance, row_idx, row in distanced
            if abs(float(distance) - float(best_distance)) <= 1.0e-9
        ]
        return best if len(best) == 1 else []

    def _fresh_row_match_score(
        target: Mapping[str, object],
        target_key: object,
        row: Mapping[str, object],
    ) -> int | None:
        if _row_is_provider_backed_source_coverage(row):
            return None
        if deps.source_row_reuses_manual_caked_target(target, row):
            return None
        if not (
            isinstance(target_key, tuple)
            and len(target_key) >= 3
            and isinstance(target_key[0], tuple)
        ):
            return None
        row_hkl = deps.normalized_hkl(
            row.get("normalized_hkl", row.get("hkl", row.get("source_hkl")))
        )
        target_hkl = tuple(int(v) for v in target_key[0])
        target_table = _source_identity_value(
            target,
            "source_table_index",
            "resolved_table_index",
        )
        target_row = _source_identity_value(
            target,
            "source_row_index",
            "resolved_source_row_index",
            "resolved_source_row_position",
        )
        target_peak = _source_identity_value(
            target,
            "source_peak_index",
            "resolved_peak_index",
        )
        target_reflection = _source_identity_value(
            target,
            "source_reflection_index",
            "legacy_source_reflection_index",
        )
        target_branch_slot = target_key[1]
        row_branch = deps.source_branch_index(row)
        row_table = _source_identity_value(
            row,
            "source_table_index",
            "resolved_table_index",
        )
        row_index = _source_identity_value(
            row,
            "source_row_index",
            "resolved_source_row_index",
            "resolved_source_row_position",
        )
        row_peak = _source_identity_value(
            row,
            "source_peak_index",
            "resolved_peak_index",
        )
        row_reflection = _source_identity_value(
            row,
            "source_reflection_index",
            "legacy_source_reflection_index",
        )
        identity_score = 0
        target_group = (
            deps.stable_group_identity(target_key[2])
            if isinstance(target_key, tuple) and len(target_key) >= 3
            else None
        )
        row_group = deps.group_identity(row)
        group_matches = (
            deps.group_identity_is_q_group(target_group)
            and target_group is not None
            and row_group == target_group
        )
        strong_identity_score = 0
        if (
            target_table is not None
            and row_table is not None
            and int(row_table) == int(target_table)
        ):
            identity_score += 4
            strong_identity_score += 4
        if (
            target_row is not None
            and row_index is not None
            and int(row_index) == int(target_row)
        ):
            identity_score += 4
            strong_identity_score += 4
        if (
            target_peak is not None
            and row_peak is not None
            and int(row_peak) == int(target_peak)
        ):
            identity_score += 2
        if (
            target_reflection is not None
            and row_reflection is not None
            and int(row_reflection) == int(target_reflection)
        ):
            identity_score += 6
            strong_identity_score += 6
        hkl_matches = row_hkl == target_hkl
        if not hkl_matches and not group_matches and strong_identity_score <= 0:
            return None

        score = int(identity_score)
        if hkl_matches:
            score += 8
        if group_matches:
            score += 6
        if target_branch_slot == deps.zero_qr_coverage_branch_slot:
            score += 2
        elif target_branch_slot in {0, 1}:
            branch_matches = row_branch == int(target_branch_slot)
            fixed_source_identity_matches = bool(
                strong_identity_score > 0 and hkl_matches and group_matches
            )
            if not branch_matches and not fixed_source_identity_matches:
                return None
            score += 4 if branch_matches else 1
        else:
            return None
        return int(score)

    def _matching_fresh_rows(
        target: Mapping[str, object],
        target_key: object,
    ) -> list[tuple[int, dict[str, object]]]:
        matches: list[tuple[int, dict[str, object]]] = []
        for row_idx, row in enumerate(augmented_rows):
            if target_key not in deps.source_coverage_alias_keys(row):
                continue
            if _row_is_provider_backed_source_coverage(row):
                continue
            if deps.source_row_reuses_manual_caked_target(target, row):
                continue
            matches.append((int(row_idx), row))
        if len(matches) <= 1:
            if matches:
                return matches
            scored_matches: list[tuple[int, int, dict[str, object]]] = []
            for row_idx, row in enumerate(augmented_rows):
                score = _fresh_row_match_score(target, target_key, row)
                if score is not None:
                    scored_matches.append((int(score), int(row_idx), row))
            return _best_unique_match_by_score_and_distance(target, scored_matches)

        target_table = _source_identity_value(
            target,
            "source_table_index",
            "resolved_table_index",
        )
        target_row = _source_identity_value(
            target,
            "source_row_index",
            "resolved_source_row_index",
            "resolved_source_row_position",
        )
        target_peak = _source_identity_value(
            target,
            "source_peak_index",
            "resolved_peak_index",
        )
        target_reflection = _source_identity_value(
            target,
            "source_reflection_index",
            "legacy_source_reflection_index",
        )

        scored: list[tuple[int, int, dict[str, object]]] = []
        for row_idx, row in matches:
            score = 0
            row_table = _source_identity_value(
                row,
                "source_table_index",
                "resolved_table_index",
            )
            row_index = _source_identity_value(
                row,
                "source_row_index",
                "resolved_source_row_index",
                "resolved_source_row_position",
            )
            row_peak = _source_identity_value(
                row,
                "source_peak_index",
                "resolved_peak_index",
            )
            row_reflection = _source_identity_value(
                row,
                "source_reflection_index",
                "legacy_source_reflection_index",
            )
            if (
                target_table is not None
                and row_table is not None
                and int(row_table) == int(target_table)
            ):
                score += 4
            if (
                target_row is not None
                and row_index is not None
                and int(row_index) == int(target_row)
            ):
                score += 4
            if (
                target_peak is not None
                and row_peak is not None
                and int(row_peak) == int(target_peak)
            ):
                score += 2
            if (
                target_reflection is not None
                and row_reflection is not None
                and int(row_reflection) == int(target_reflection)
            ):
                score += 2
            scored.append((int(score), int(row_idx), row))
        if not scored:
            return matches
        best_score = max(score for score, _row_idx, _row in scored)
        if best_score <= 0:
            return _best_unique_match_by_score_and_distance(
                target,
                [(1, row_idx, row) for row_idx, row in matches],
            )
        return _best_unique_match_by_score_and_distance(target, scored)

    def _promote_fresh_source_row(
        *,
        pair_idx: int,
        target_key: object,
        entry: Mapping[str, object],
    ) -> bool:
        matches = _matching_fresh_rows(entry, target_key)
        if len(matches) != 1:
            return False
        row_index, row = matches[0]
        coverage_payload = deps.source_coverage_key_payload(target_key)
        promoted = dict(row)
        aliases = list(promoted.get("source_coverage_aliases") or [])
        if coverage_payload is not None and coverage_payload not in aliases:
            aliases.append(coverage_payload)
        if aliases:
            promoted["source_coverage_aliases"] = aliases
        for key in (
            "hkl",
            "normalized_hkl",
            "q_group_key",
            "source_q_group_key",
            "branch_group_key",
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "legacy_source_reflection_index",
            "legacy_source_peak_index",
            "branch_id",
            "best_sample_index",
            "mosaic_top_rank_key",
            "selection_reason",
            "selected_source_identity_canonical",
        ):
            locked_value = entry.get(key)
            if locked_value is None:
                continue
            if promoted.get(key) is not None and promoted.get(key) != locked_value:
                promoted.setdefault(f"trial_{key}", promoted.get(key))
            promoted[key] = locked_value
        promoted["provider_backed_live_source_row"] = True
        promoted["provider_backed_live_source_row_reason"] = (
            "geometry_fit_dataset_required_source_coverage"
        )
        if _rows_use_caked_fit_space():
            promoted["source_kind"] = "sim_visual_caked_deg"
            promoted["actual_source"] = "sim_visual_caked_deg"
            promoted["expected_source"] = "sim_visual_caked_deg"
            promoted["projection_frame"] = "caked_display"
            promoted["coordinate_provenance"] = "trial_geometry_projection"
            promoted["is_dynamic_trial_row"] = True
        promoted.setdefault(
            "row_origin",
            str(row.get("row_origin") or "geometry_fit_dataset_required_source_row"),
        )
        promoted["physical_branch_slot"] = (
            coverage_payload.get("branch_slot") if isinstance(coverage_payload, Mapping) else None
        )
        promoted["fit_qr_branch_key"] = {
            "q_group_key": deps.cache_jsonable(promoted.get("q_group_key")),
            "hkl": deps.cache_jsonable(
                promoted.get("normalized_hkl", promoted.get("hkl"))
            ),
            "physical_branch_slot": promoted.get("physical_branch_slot"),
            "source_branch_index": promoted.get("source_branch_index"),
            "source_peak_index": promoted.get("source_peak_index"),
        }
        promoted["background_index"] = int(background_idx)
        promoted["overlay_match_index"] = int(pair_idx)
        promoted["pair_id"] = str(
            entry.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}"
        )
        augmented_rows[int(row_index)] = promoted
        promoted_rows.append(promoted)
        provider_backed_keys.update(deps.source_coverage_alias_keys(promoted))
        return True

    for pair_idx, selected_input in enumerate(selected_entry_inputs):
        raw_entry = selected_input.get("entry")
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        raw_saved_entry = (
            dict(selected_input.get("raw_saved_entry"))
            if isinstance(selected_input.get("raw_saved_entry"), Mapping)
            else None
        )
        truth_pair = manual_picker_truth_by_order.get((int(background_idx), int(pair_idx)), {})
        simulated_point = deps.point_list(truth_pair.get("manual_selected_simulated_point"))
        simulated_frame = deps.normalize_point_frame(
            truth_pair.get("manual_selected_simulated_frame")
        )
        if simulated_point is not None and simulated_frame != "unknown":
            entry["manual_selected_simulated_point"] = [
                float(simulated_point[0]),
                float(simulated_point[1]),
            ]
            entry["manual_selected_simulated_frame"] = simulated_frame
            entry["manual_simulated_point_source"] = str(
                truth_pair.get("manual_simulated_point_source") or ""
            )
            deps.put_simulated_point_fields(
                entry,
                simulated_point,
                simulated_frame,
            )
        target_key = deps.normalize_source_coverage_key(entry)
        if target_key is None:
            continue
        if target_key in coverage_keys:
            if not require_provider_backed_rows:
                continue
            if (
                target_key in provider_backed_keys
                and _current_provider_backed_source_row_exists(target_key)
            ):
                continue
        if require_provider_backed_rows:
            if _promote_fresh_source_row(
                pair_idx=int(pair_idx),
                target_key=target_key,
                entry=entry,
            ):
                continue
        if not allow_saved_coordinate_materialization:
            point_missing.append(
                {
                    "pair_index": int(pair_idx),
                    "target_key": deps.source_coverage_key_payload(target_key),
                    "reason": "missing_dynamic_trial_source_row",
                }
            )
            continue
        provider_row = deps.provider_backed_source_row_for_target(
            pair_idx=int(pair_idx),
            entry=entry,
            raw_saved_entry=raw_saved_entry,
        )
        if provider_row is None:
            point_missing.append(
                {
                    "pair_index": int(pair_idx),
                    "target_key": deps.source_coverage_key_payload(target_key),
                    "reason": "coverage_source_present_point_missing",
                }
            )
            continue
        materialized_rows.append(dict(provider_row))
        coverage_keys.update(deps.source_coverage_alias_keys(provider_row))
        provider_backed_keys.update(deps.source_coverage_alias_keys(provider_row))
    if materialized_rows:
        augmented_rows.extend(materialized_rows)
    if materialized_rows or promoted_rows or point_missing:
        augmented_diagnostics["source_coverage_materialization"] = {
            "provider_backed_row_count": int(len(materialized_rows)),
            "provider_backed_fresh_row_count": int(len(promoted_rows)),
            "point_missing_count": int(len(point_missing)),
            "saved_coordinate_materialization_allowed": bool(
                allow_saved_coordinate_materialization
            ),
            "provider_backed_keys": [
                deps.source_coverage_key_payload(deps.normalize_source_coverage_key(row))
                for row in materialized_rows
            ],
            "point_missing": point_missing,
        }
        augmented_diagnostics["provider_backed_fresh_source_coverage_row_count"] = int(
            len(promoted_rows)
        )
        augmented_diagnostics["provider_backed_source_coverage_row_count"] = int(
            len(materialized_rows)
        )
        augmented_diagnostics["coverage_source_present_point_missing_count"] = int(
            len(point_missing)
        )
        augmented_diagnostics["missing_dynamic_trial_source_row_count"] = int(
            sum(
                1
                for item in point_missing
                if str(item.get("reason") or "") == "missing_dynamic_trial_source_row"
            )
        )
    if include_coverage_diagnostics:
        coverage_diagnostics = deps.source_coverage_filter_diagnostics(augmented_rows)
        if coverage_diagnostics:
            augmented_diagnostics.update(copy.deepcopy(dict(coverage_diagnostics)))
            targeted_gate = dict(augmented_diagnostics.get("targeted_performance_gate") or {})
            targeted_gate.update(copy.deepcopy(dict(coverage_diagnostics)))
            augmented_diagnostics["targeted_performance_gate"] = targeted_gate
    return augmented_rows, augmented_diagnostics
