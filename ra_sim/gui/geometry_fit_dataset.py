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
