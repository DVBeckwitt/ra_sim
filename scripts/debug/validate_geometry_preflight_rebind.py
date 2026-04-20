"""Validate fresh grouped-pick emission and canonical preflight rebinding."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ra_sim.fitting import optimization as opt
from ra_sim import headless_geometry_fit as hgf
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.io.data_loading import load_gui_state_file, save_gui_state_file


class _CapturedPreflight(RuntimeError):
    """Internal stop used to abort headless execution after preflight capture."""


class _CapturedExecutionSetup(RuntimeError):
    """Internal stop used to abort headless execution after setup capture."""


CANONICAL_IDENTITY_FIELDS = (
    "hkl",
    "source_reflection_index",
    "source_reflection_namespace",
    "source_reflection_is_full",
    "source_branch_index",
    "source_peak_index",
)

DOWNSTREAM_IDENTITY_STAGE_ORDER = (
    "input_contract",
    "preflight_normalized_pairs",
    "solver_request_measured_peaks",
    "subset_measured_entries",
    "seed_correspondence_records",
    "full_beam_identity_coverage",
    "full_beam_fixed_correspondence",
)


def _finalize_cli_result(
    result: Mapping[str, object] | None,
    *,
    requested_mode: str,
    effective_mode: str,
) -> dict[str, object]:
    payload = dict(result or {})
    payload["requested_mode"] = str(requested_mode)
    payload["effective_mode"] = str(effective_mode)
    if (
        str(requested_mode).strip().lower() == "full"
        and str(effective_mode).strip().lower() == "fresh-all"
    ):
        payload["mode_note"] = "full now aliases fresh-all new4 detector-alignment preflight gate"
    return payload


def _stable_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _try_save_gui_state_file(
    path: Path,
    state: Mapping[str, object],
    *,
    failure_stage: str,
    create_parent: bool = False,
    remove_partial_target: bool = False,
) -> dict[str, object] | None:
    target_path = Path(path).expanduser().resolve()
    target_exists_before = target_path.exists()
    temp_save_path = None
    try:
        if create_parent:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            dir=target_path.parent,
            prefix=f"{target_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
        ) as handle:
            temp_save_path = Path(handle.name).resolve()
        save_gui_state_file(temp_save_path, dict(state))
        temp_save_path.replace(target_path)
    except Exception as exc:
        partial_target_removed = False
        try:
            if temp_save_path is not None and temp_save_path.exists():
                temp_save_path.unlink()
            if (
                remove_partial_target
                and not target_exists_before
                and target_path.exists()
            ):
                target_path.unlink()
            partial_target_removed = bool(
                (temp_save_path is None or not temp_save_path.exists())
                and (target_exists_before or not target_path.exists())
            )
        except Exception:
            partial_target_removed = False
        return {
            "failure_stage": str(failure_stage),
            "save_path": str(target_path),
            "error_type": type(exc).__name__,
            "error_text": str(exc),
            "partial_target_removed": partial_target_removed,
        }
    return None


def _normalize_hkl(value: object) -> tuple[int, int, int] | None:
    return gui_geometry_fit._geometry_fit_normalized_hkl(value)


def _normalize_q_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return None


def _normalize_key_tuple(value: object) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def _identity_payload(
    entry: Mapping[str, object] | None,
    *,
    pair_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        return {
            "pair_id": pair_id,
            "hkl": None,
            "source_reflection_index": None,
            "source_reflection_namespace": None,
            "source_reflection_is_full": None,
            "source_branch_index": None,
            "source_peak_index": None,
        }
    return {
        "pair_id": pair_id if pair_id is not None else entry.get("pair_id"),
        "hkl": _normalize_hkl(entry.get("hkl")),
        "source_reflection_index": entry.get("source_reflection_index"),
        "source_reflection_namespace": entry.get("source_reflection_namespace"),
        "source_reflection_is_full": entry.get("source_reflection_is_full"),
        "source_branch_index": entry.get("source_branch_index"),
        "source_peak_index": entry.get("source_peak_index"),
    }


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return float(number) if np.isfinite(number) else None


def _canonical_identity(entry: Mapping[str, object] | None) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    return (
        entry.get("source_reflection_index"),
        entry.get("source_reflection_namespace"),
        entry.get("source_reflection_is_full"),
        entry.get("source_branch_index"),
        entry.get("source_peak_index"),
    )


def _finite_point(
    entry: Mapping[str, object] | None,
    *keys: str,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping) or len(keys) < 2:
        return None
    try:
        col = float(entry.get(keys[0], np.nan))
        row = float(entry.get(keys[1], np.nan))
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _normalized_branch_index(entry: Mapping[str, object] | None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("source_branch_index", "source_peak_index"):
        try:
            value = int(entry.get(key))
        except Exception:
            continue
        if value in {0, 1}:
            return int(value)
    return None


def _entry_dataset_index(
    entry: Mapping[str, object] | None,
    *,
    default: int = 0,
) -> int:
    if not isinstance(entry, Mapping):
        return int(default)
    try:
        return int(entry.get("dataset_index", default))
    except Exception:
        return int(default)


def _downstream_identity_key(
    entry: Mapping[str, object] | None,
    *,
    default_dataset_index: int = 0,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    return (
        _entry_dataset_index(entry, default=default_dataset_index),
        _normalize_hkl(entry.get("hkl")),
        entry.get("source_reflection_index"),
        entry.get("source_branch_index"),
        entry.get("source_peak_index"),
    )


def _saved_entries_for_background(
    saved_state: Mapping[str, object] | None,
    *,
    background_index: int,
) -> list[dict[str, object]]:
    geometry_state = (
        saved_state.get("geometry", {})
        if isinstance(saved_state, Mapping) and isinstance(saved_state.get("geometry"), Mapping)
        else {}
    )
    manual_pairs = geometry_state.get("manual_pairs", []) or ()
    for raw_entry in manual_pairs:
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            entry_background_index = int(raw_entry.get("background_index", -1))
        except Exception:
            entry_background_index = -1
        if entry_background_index != int(background_index):
            continue
        return [
            dict(entry)
            for entry in (raw_entry.get("entries", []) or ())
            if isinstance(entry, Mapping)
        ]
    return []


def _all_saved_manual_pair_entries(
    saved_state: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    geometry_state = (
        saved_state.get("geometry", {})
        if isinstance(saved_state, Mapping) and isinstance(saved_state.get("geometry"), Mapping)
        else {}
    )
    entries: list[dict[str, object]] = []
    for raw_entry in geometry_state.get("manual_pairs", []) or ():
        if not isinstance(raw_entry, Mapping):
            continue
        background_index = _int_or_none(raw_entry.get("background_index"))
        for slot_index, raw_pair in enumerate(raw_entry.get("entries", []) or ()):
            if not isinstance(raw_pair, Mapping):
                continue
            pair = dict(raw_pair)
            if background_index is not None:
                pair.setdefault("background_index", int(background_index))
            pair.setdefault("slot_index", int(slot_index))
            entries.append(pair)
    return entries


def _empty_saved_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 0


def _point_payload(point: tuple[float, float] | None) -> list[float] | None:
    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _detector_distance(
    reference_point: tuple[float, float] | None,
    candidate_point: tuple[float, float] | None,
) -> float:
    if reference_point is None or candidate_point is None:
        return float("nan")
    return float(
        np.hypot(
            float(candidate_point[0]) - float(reference_point[0]),
            float(candidate_point[1]) - float(reference_point[1]),
        )
    )


def _saved_simulated_display_hint(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for keys in (
        ("refined_sim_x", "refined_sim_y"),
        ("x", "y"),
        ("raw_x", "raw_y"),
    ):
        hint = _finite_point(entry, *keys)
        if hint is not None:
            return hint
    return None


def _saved_background_current_view_point_frame(
    entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> tuple[tuple[float, float] | None, str | None]:
    if bool(use_caked_space):
        for keys in (("raw_caked_x", "raw_caked_y"), ("caked_x", "caked_y")):
            point = _finite_point(entry, *keys)
            if point is not None:
                return point, "caked_display"
        return None, None
    for keys in (("x", "y"), ("raw_x", "raw_y"), ("display_col", "display_row")):
        point = _finite_point(entry, *keys)
        if point is not None:
            return point, "current_view_display"
    return None, None


def _saved_background_current_view_point(
    entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> tuple[float, float] | None:
    point, _frame = _saved_background_current_view_point_frame(
        entry,
        use_caked_space=use_caked_space,
    )
    return point


def _saved_background_current_view_frame(
    entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> str | None:
    _point, frame = _saved_background_current_view_point_frame(
        entry,
        use_caked_space=use_caked_space,
    )
    return frame


def _saved_simulated_detector_hint(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for keys in (
        ("refined_sim_native_x", "refined_sim_native_y"),
        ("simulated_detector_x", "simulated_detector_y"),
        ("sim_col_raw", "sim_row_raw"),
        ("sim_col", "sim_row"),
    ):
        hint = _finite_point(entry, *keys)
        if hint is not None:
            return hint
    for keys in (
        ("refined_sim_x", "refined_sim_y"),
        ("raw_x", "raw_y"),
        ("x", "y"),
    ):
        hint = _finite_point(entry, *keys)
        if hint is not None:
            return hint
    return None


def _saved_measured_detector_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for keys in (
        ("native_col", "native_row"),
        ("background_detector_x", "background_detector_y"),
        ("detector_x", "detector_y"),
        ("x", "y"),
    ):
        point = _finite_point(entry, *keys)
        if point is not None:
            return point
    return None


def _candidate_detector_native_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for keys in (
        ("native_col", "native_row"),
        ("sim_col_raw", "sim_row_raw"),
        ("sim_col", "sim_row"),
    ):
        point = _finite_point(entry, *keys)
        if point is not None:
            return point
    return None


def _looks_like_live_simulated_candidate(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    has_measured_background_hint = any(
        key in entry
        for key in (
            "x",
            "y",
            "raw_x",
            "raw_y",
            "detector_x",
            "detector_y",
            "background_detector_x",
            "background_detector_y",
        )
    )
    has_live_simulated_shape = any(
        key in entry
        for key in (
            "source_reflection_index",
            "source_table_index",
            "source_row_index",
            "source_peak_index",
            "sim_col",
            "sim_row",
            "sim_col_raw",
            "sim_row_raw",
            "native_col",
            "native_row",
        )
    )
    return bool(has_live_simulated_shape and not has_measured_background_hint)


def _candidate_current_view_point_frame(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, str | None]:
    caked_point = _finite_point(entry, "caked_x", "caked_y")
    if caked_point is None:
        caked_point = _finite_point(entry, "raw_caked_x", "raw_caked_y")
    if caked_point is None:
        caked_point = _finite_point(entry, "two_theta_deg", "phi_deg")
    if caked_point is not None:
        return None, None

    sim_point = _finite_point(entry, "sim_col", "sim_row")
    if sim_point is not None:
        return sim_point, "current_view_display"
    display_point = _finite_point(entry, "display_col", "display_row")
    if display_point is not None and _looks_like_live_simulated_candidate(entry):
        return display_point, "current_view_display"
    return None, None


def _candidate_current_view_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    point, _frame = _candidate_current_view_point_frame(entry)
    return point


def _candidate_current_view_frame(
    entry: Mapping[str, object] | None,
) -> str | None:
    _point, frame = _candidate_current_view_point_frame(entry)
    return frame


def _candidate_detector_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _candidate_current_view_point(entry)


def _candidate_detector_native_distance(
    detector_hint: tuple[float, float] | None,
    candidate: Mapping[str, object] | None,
) -> float:
    return _detector_distance(detector_hint, _candidate_detector_native_point(candidate))


def _candidate_detector_display_distance(
    detector_hint: tuple[float, float] | None,
    candidate: Mapping[str, object] | None,
) -> float:
    return _detector_distance(detector_hint, _candidate_detector_display_point(candidate))


def _candidate_background_distance(
    background_point: tuple[float, float] | None,
    *,
    background_frame: str | None,
    candidate: Mapping[str, object] | None,
) -> float:
    candidate_point, candidate_frame = _candidate_current_view_point_frame(candidate)
    if background_frame is None or candidate_frame is None:
        return float("nan")
    if str(background_frame) != str(candidate_frame):
        return float("nan")
    return _detector_distance(background_point, candidate_point)


def _candidate_selection_distance(
    *,
    background_point: tuple[float, float] | None,
    background_frame: str | None,
    detector_hint: tuple[float, float] | None,
    candidate: Mapping[str, object] | None,
    use_current_view_distance: bool,
) -> float:
    if bool(use_current_view_distance):
        return _candidate_background_distance(
            background_point,
            background_frame=background_frame,
            candidate=candidate,
        )
    return _candidate_detector_native_distance(detector_hint, candidate)


def _display_hint(
    entry: Mapping[str, object] | None,
    *,
    entry_display_coords,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    coords = entry_display_coords(entry)
    if coords is None or len(coords) < 2:
        return None
    try:
        col = float(coords[0])
        row = float(coords[1])
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _compact_entry(
    entry: Mapping[str, object] | None,
    *,
    display_hint: tuple[float, float] | None = None,
    detector_hint: tuple[float, float] | None = None,
    measured_detector_point: tuple[float, float] | None = None,
    background_point: tuple[float, float] | None = None,
    selection_coordinate_frame: str | None = None,
    selection_reference_frame: str | None = None,
) -> dict[str, object] | None:
    if not isinstance(entry, Mapping):
        return None
    detector_native = _candidate_detector_native_point(entry)
    current_view_point = _candidate_current_view_point(entry)
    current_view_frame = _candidate_current_view_frame(entry)
    payload = {
        "candidate_source_key": gui_manual_geometry.geometry_manual_candidate_source_key(
            dict(entry)
        ),
        "q_group_key": _normalize_q_group_key(entry.get("q_group_key")),
        "hkl": _normalize_hkl(entry.get("hkl")),
        "source_table_index": entry.get("source_table_index"),
        "source_row_index": entry.get("source_row_index"),
        "source_reflection_index": entry.get("source_reflection_index"),
        "source_reflection_namespace": entry.get("source_reflection_namespace"),
        "source_reflection_is_full": entry.get("source_reflection_is_full"),
        "source_branch_index": entry.get("source_branch_index"),
        "source_peak_index": entry.get("source_peak_index"),
        "native_col": entry.get("native_col"),
        "native_row": entry.get("native_row"),
        "sim_col": entry.get("sim_col"),
        "sim_row": entry.get("sim_row"),
        "sim_col_raw": entry.get("sim_col_raw"),
        "sim_row_raw": entry.get("sim_row_raw"),
        "selected_live_simulated_current_view_point": _point_payload(current_view_point),
        "selected_live_simulated_current_view_frame": current_view_frame,
        "selected_live_simulated_detector_point": _point_payload(detector_native),
        "sim_detector_display": _point_payload(detector_native),
        "caked_x": entry.get("caked_x"),
        "caked_y": entry.get("caked_y"),
    }
    if selection_coordinate_frame is not None:
        payload["coordinate_frame"] = selection_coordinate_frame
    if display_hint is not None:
        payload["distance_to_saved_display_hint_px"] = (
            gui_manual_geometry.geometry_manual_candidate_distance_to_point(
                float(display_hint[0]),
                float(display_hint[1]),
                dict(entry),
            )
        )
    if detector_hint is not None:
        distance_to_saved_sim = _candidate_detector_native_distance(detector_hint, entry)
        payload["distance_to_saved_sim_detector_hint_px"] = distance_to_saved_sim
        payload["distance_to_saved_sim_hint_px"] = distance_to_saved_sim
    if measured_detector_point is not None:
        payload["distance_to_saved_measured_detector_point_px"] = _detector_distance(
            measured_detector_point,
            detector_native,
        )
    if background_point is not None:
        distance_to_saved_background = _candidate_background_distance(
            background_point,
            background_frame=selection_reference_frame,
            candidate=entry,
        )
        payload["saved_background_current_view_frame"] = selection_reference_frame
        payload["distance_to_saved_background_reference_frame"] = selection_reference_frame
        payload["distance_to_saved_background_candidate_frame"] = current_view_frame
        payload["distance_to_saved_background_current_view_point_px"] = (
            distance_to_saved_background
        )
        payload["distance_to_saved_display_hint_px"] = distance_to_saved_background
    return payload


def _source_locator_payload(
    entry: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        return {
            "source_reflection_namespace": None,
            "source_reflection_index": None,
            "source_table_index": None,
            "source_row_index": None,
            "source_peak_index": None,
            "source_branch_index": None,
        }
    return {
        "source_reflection_namespace": entry.get("source_reflection_namespace"),
        "source_reflection_index": entry.get("source_reflection_index"),
        "source_table_index": entry.get("source_table_index"),
        "source_row_index": entry.get("source_row_index"),
        "source_peak_index": entry.get("source_peak_index"),
        "source_branch_index": entry.get("source_branch_index"),
    }


def _source_locator_identity_match(
    saved_entry: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
) -> bool:
    saved_locator = _source_locator_payload(saved_entry)
    candidate_locator = _source_locator_payload(candidate)
    comparable_keys = [
        key for key, value in saved_locator.items() if value is not None
    ]
    if not comparable_keys:
        return False
    return all(candidate_locator.get(key) == saved_locator.get(key) for key in comparable_keys)


def _candidate_duplicate_tie_key(
    entry: Mapping[str, object] | None,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    return (
        _normalize_hkl(entry.get("hkl")),
        _normalized_branch_index(entry),
        _candidate_current_view_frame(entry),
        _candidate_current_view_point(entry),
        tuple(sorted(_source_locator_payload(entry).items())),
    )


def _dedupe_tied_candidates(
    candidates: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    unique_candidates: dict[tuple[object, ...], dict[str, object]] = {}
    for raw_candidate in candidates or ():
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        dedupe_key = _candidate_duplicate_tie_key(candidate)
        if dedupe_key is None:
            dedupe_key = (
                gui_manual_geometry.geometry_manual_candidate_source_key(candidate),
            )
        existing = unique_candidates.get(dedupe_key)
        if existing is None:
            unique_candidates[dedupe_key] = candidate
            continue
        existing_key = repr(gui_manual_geometry.geometry_manual_candidate_source_key(existing))
        candidate_key = repr(gui_manual_geometry.geometry_manual_candidate_source_key(candidate))
        if candidate_key < existing_key:
            unique_candidates[dedupe_key] = candidate
    return [
        dict(candidate)
        for _key, candidate in sorted(
            unique_candidates.items(),
            key=lambda item: repr(gui_manual_geometry.geometry_manual_candidate_source_key(item[1])),
        )
    ]


def _slot_detector_diagnostic(
    *,
    pair_id: str,
    saved_entry: Mapping[str, object] | None,
    selected_candidate: Mapping[str, object] | None,
    saved_background_current_view_point: tuple[float, float] | None = None,
    saved_background_current_view_frame: str | None = None,
) -> dict[str, object]:
    saved_simulated_detector_hint = _saved_simulated_detector_hint(saved_entry)
    saved_measured_detector_point = _saved_measured_detector_point(saved_entry)
    selection_coordinate_frame = (
        "current_view_px"
        if saved_background_current_view_point is not None
        else "detector_native_px"
    )
    selected_live_simulated_current_view_point = _candidate_current_view_point(
        selected_candidate
    )
    selected_live_simulated_current_view_frame = _candidate_current_view_frame(
        selected_candidate
    )
    selected_live_simulated_detector_point = _candidate_detector_native_point(selected_candidate)
    selected_locator = _source_locator_payload(selected_candidate)
    return {
        "pair_id": pair_id,
        "hkl": _normalize_hkl(
            (selected_candidate or {}).get("hkl")
            if isinstance(selected_candidate, Mapping)
            else (saved_entry or {}).get("hkl")
        ),
        "q_group_key": _normalize_q_group_key(
            (selected_candidate or {}).get("q_group_key")
            if isinstance(selected_candidate, Mapping)
            and _normalize_q_group_key(selected_candidate.get("q_group_key")) is not None
            else (saved_entry or {}).get("q_group_key")
        ),
        "saved_source_identity_fields": _source_locator_payload(saved_entry),
        "selected_live_source_identity_fields": selected_locator,
        "saved_measured_detector_point": _point_payload(saved_measured_detector_point),
        "saved_background_current_view_point": _point_payload(
            saved_background_current_view_point
        ),
        "saved_background_current_view_frame": saved_background_current_view_frame,
        "saved_simulated_detector_hint": _point_payload(saved_simulated_detector_hint),
        "selected_live_simulated_current_view_point": _point_payload(
            selected_live_simulated_current_view_point
        ),
        "selected_live_simulated_current_view_frame": (
            selected_live_simulated_current_view_frame
        ),
        "selected_live_simulated_detector_point": _point_payload(
            selected_live_simulated_detector_point
        ),
        "selected_to_background_distance_px": _candidate_background_distance(
            saved_background_current_view_point,
            background_frame=saved_background_current_view_frame,
            candidate=selected_candidate,
        ),
        "selected_to_saved_sim_distance_px": _detector_distance(
            saved_simulated_detector_hint,
            selected_live_simulated_detector_point,
        ),
        "selected_to_measured_distance_px": _detector_distance(
            saved_measured_detector_point,
            selected_live_simulated_detector_point,
        ),
        "branch_index": (
            _normalized_branch_index(selected_candidate)
            if isinstance(selected_candidate, Mapping)
            else _normalized_branch_index(saved_entry)
        ),
        "source_reflection_namespace": selected_locator.get("source_reflection_namespace"),
        "source_reflection_index": selected_locator.get("source_reflection_index"),
        "source_table_index": selected_locator.get("source_table_index"),
        "source_row_index": selected_locator.get("source_row_index"),
        "source_peak_index": selected_locator.get("source_peak_index"),
        "source_branch_index": selected_locator.get("source_branch_index"),
        "coordinate_frame": selection_coordinate_frame,
    }


def _merged_saved_entry_for_slot(
    *,
    raw_saved_entries: Sequence[Mapping[str, object]],
    saved_entries: Sequence[Mapping[str, object]],
    slot_index: int,
) -> dict[str, object]:
    saved_entry = (
        dict(raw_saved_entries[int(slot_index)])
        if int(slot_index) < len(raw_saved_entries)
        and isinstance(raw_saved_entries[int(slot_index)], Mapping)
        else {}
    )
    if int(slot_index) < len(saved_entries) and isinstance(saved_entries[int(slot_index)], Mapping):
        saved_entry.update(dict(saved_entries[int(slot_index)]))
    return saved_entry


def _background_distance_threshold_px(
    saved_entries: Sequence[Mapping[str, object]],
) -> float:
    placement_errors = [
        _float_or_none(entry.get("placement_error_px"))
        for entry in saved_entries
        if isinstance(entry, Mapping)
    ]
    finite_placement_errors = [
        float(value)
        for value in placement_errors
        if value is not None and np.isfinite(value)
    ]
    if not finite_placement_errors:
        return 100.0
    return float(max(100.0, 3.0 * max(finite_placement_errors)))


def _group_inventory(
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    if not isinstance(grouped_candidates, Mapping):
        return inventory
    normalized_keys = sorted(
        (_normalize_q_group_key(key) for key in grouped_candidates.keys()),
        key=repr,
    )
    for group_key in normalized_keys:
        entries = [
            dict(entry)
            for entry in grouped_candidates.get(group_key, ())
            if isinstance(entry, Mapping)
        ]
        inventory.append(
            {
                "group_key": group_key,
                "candidate_count": len(entries),
                "candidate_source_keys": [
                    gui_manual_geometry.geometry_manual_candidate_source_key(entry)
                    for entry in entries
                ],
                "hkls": sorted(
                    {
                        _normalize_hkl(entry.get("hkl"))
                        for entry in entries
                        if _normalize_hkl(entry.get("hkl")) is not None
                    },
                    key=repr,
                ),
            }
        )
    return inventory


def _select_live_candidate_for_saved_entry(
    *,
    saved_entry: Mapping[str, object],
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
    saved_background_current_view_point: tuple[float, float] | None = None,
    saved_background_current_view_frame: str | None = None,
    tie_tolerance_px: float = gui_geometry_fit.GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX,
) -> dict[str, object]:
    background_point = saved_background_current_view_point
    use_current_view_distance = background_point is not None
    selection_coordinate_frame = (
        "current_view_px" if use_current_view_distance else "detector_native_px"
    )
    selection_reference_frame = (
        saved_background_current_view_frame or "current_view_display"
        if use_current_view_distance
        else "native_detector"
    )
    detector_hint = _saved_simulated_detector_hint(saved_entry)
    measured_detector_point = _saved_measured_detector_point(saved_entry)
    target_hkl = _normalize_hkl(saved_entry.get("hkl"))
    target_branch = _normalized_branch_index(saved_entry)
    inventory = _group_inventory(grouped_candidates)
    matching_candidates: list[dict[str, object]] = []
    excluded_missing_hkl_inventory: list[dict[str, object]] = []
    if isinstance(grouped_candidates, Mapping):
        for group_key, raw_entries in grouped_candidates.items():
            normalized_group_key = _normalize_q_group_key(group_key)
            for raw_entry in raw_entries or ():
                if not isinstance(raw_entry, Mapping):
                    continue
                entry = dict(raw_entry)
                candidate_hkl = _normalize_hkl(entry.get("hkl"))
                if target_hkl is not None and candidate_hkl is None:
                    entry["q_group_key"] = normalized_group_key
                    excluded_missing_hkl_inventory.append(
                        {
                            **(
                                _compact_entry(
                                    entry,
                                    background_point=background_point,
                                    selection_reference_frame=(
                                        saved_background_current_view_frame
                                    ),
                                )
                                or {}
                            ),
                            "candidate_hkl_status": "missing_or_invalid",
                        }
                    )
                    continue
                if target_hkl is not None and candidate_hkl != target_hkl:
                    continue
                if target_branch is not None and _normalized_branch_index(entry) != target_branch:
                    continue
                entry["q_group_key"] = normalized_group_key
                matching_candidates.append(entry)

    scored_candidates = [
        (
            entry,
            _candidate_selection_distance(
                background_point=background_point,
                background_frame=saved_background_current_view_frame,
                detector_hint=detector_hint,
                candidate=entry,
                use_current_view_distance=use_current_view_distance,
            ),
        )
        for entry in matching_candidates
    ]
    scored_candidates.sort(
        key=lambda item: (
            float(item[1]) if np.isfinite(float(item[1])) else float("inf"),
            repr(gui_manual_geometry.geometry_manual_candidate_source_key(item[0])),
        )
    )
    candidate_inventory = [
        _compact_entry(
            entry,
            detector_hint=detector_hint,
            measured_detector_point=measured_detector_point,
            background_point=background_point if use_current_view_distance else None,
            selection_coordinate_frame=selection_coordinate_frame,
            selection_reference_frame=selection_reference_frame,
        )
        for entry, _distance in scored_candidates
    ]
    finite_candidates = [
        (entry, float(distance))
        for entry, distance in scored_candidates
        if np.isfinite(float(distance))
    ]
    if not finite_candidates:
        return {
            "ok": False,
            "failure_stage": "grouped_candidate_regeneration",
            "selection_status": "missing_live_candidate",
            "saved_background_current_view_point": background_point,
            "saved_simulated_detector_hint": detector_hint,
            "saved_measured_detector_point": measured_detector_point,
            "saved_target_hkl": target_hkl,
            "saved_target_branch_index": target_branch,
            "selection_coordinate_frame": selection_coordinate_frame,
            "group_inventory": inventory,
            "excluded_missing_hkl_candidate_inventory": excluded_missing_hkl_inventory,
            "matching_branch_candidate_inventory": candidate_inventory,
        }

    best_entry, best_distance = finite_candidates[0]
    tied_entries = [
        dict(entry)
        for entry, distance in finite_candidates
        if abs(float(distance) - float(best_distance)) <= float(tie_tolerance_px)
    ]
    deduped_tied_entries = _dedupe_tied_candidates(tied_entries)
    if len(deduped_tied_entries) == 1:
        selected_entry = dict(deduped_tied_entries[0])
        selection_tie_breaker = (
            "duplicate_live_rows_canonicalized"
            if len(tied_entries) > 1
            else None
        )
        result = {
            "ok": True,
            "selection_status": "selected",
            "saved_background_current_view_point": background_point,
            "saved_background_current_view_frame": saved_background_current_view_frame,
            "saved_simulated_detector_hint": detector_hint,
            "saved_measured_detector_point": measured_detector_point,
            "saved_target_hkl": target_hkl,
            "saved_target_branch_index": target_branch,
            "background_distance_px": float(best_distance),
            "best_distance_px": float(best_distance),
            "selection_coordinate_frame": selection_coordinate_frame,
            "background_distance_reference_frame": selection_reference_frame,
            "background_distance_candidate_frame": (
                _candidate_current_view_frame(selected_entry)
                if use_current_view_distance
                else "native_detector"
            ),
            "selected_candidate": selected_entry,
            "group_key": _normalize_q_group_key(selected_entry.get("q_group_key")),
            "group_inventory": inventory,
            "excluded_missing_hkl_candidate_inventory": excluded_missing_hkl_inventory,
            "matching_branch_candidate_inventory": candidate_inventory,
        }
        if selection_tie_breaker is not None:
            result["selection_tie_breaker"] = selection_tie_breaker
            result["tied_candidate_inventory"] = [
                _compact_entry(
                        entry,
                        detector_hint=detector_hint,
                        measured_detector_point=measured_detector_point,
                        background_point=(
                            background_point if use_current_view_distance else None
                        ),
                        selection_coordinate_frame=selection_coordinate_frame,
                        selection_reference_frame=selection_reference_frame,
                    )
                    for entry in tied_entries
                ]
        return result
    if len(deduped_tied_entries) > 1:
        identity_tied_entries = [
            dict(entry)
            for entry in deduped_tied_entries
            if _source_locator_identity_match(saved_entry, entry)
        ]
        if len(identity_tied_entries) == 1:
            selected_entry = dict(identity_tied_entries[0])
            return {
                "ok": True,
                "selection_status": "selected",
                "selection_tie_breaker": "saved_source_identity",
                "saved_background_current_view_point": background_point,
                "saved_background_current_view_frame": saved_background_current_view_frame,
                "saved_simulated_detector_hint": detector_hint,
                "saved_measured_detector_point": measured_detector_point,
                "saved_target_hkl": target_hkl,
                "saved_target_branch_index": target_branch,
                "background_distance_px": float(best_distance),
                "best_distance_px": float(best_distance),
                "selection_coordinate_frame": selection_coordinate_frame,
                "background_distance_reference_frame": selection_reference_frame,
                "background_distance_candidate_frame": (
                    _candidate_current_view_frame(selected_entry)
                    if use_current_view_distance
                    else "native_detector"
                ),
                "selected_candidate": selected_entry,
                "group_key": _normalize_q_group_key(selected_entry.get("q_group_key")),
                "group_inventory": inventory,
                "excluded_missing_hkl_candidate_inventory": excluded_missing_hkl_inventory,
                "matching_branch_candidate_inventory": candidate_inventory,
                "tied_candidate_inventory": [
                    _compact_entry(
                        entry,
                        detector_hint=detector_hint,
                        measured_detector_point=measured_detector_point,
                        background_point=(
                            background_point if use_current_view_distance else None
                        ),
                        selection_coordinate_frame=selection_coordinate_frame,
                        selection_reference_frame=selection_reference_frame,
                    )
                    for entry in tied_entries
                ],
            }
        return {
            "ok": False,
            "failure_stage": "resolved_live_row_selection",
            "selection_status": "ambiguous_live_row_selection",
            "saved_background_current_view_point": background_point,
            "saved_background_current_view_frame": saved_background_current_view_frame,
            "saved_simulated_detector_hint": detector_hint,
            "saved_measured_detector_point": measured_detector_point,
            "saved_target_hkl": target_hkl,
            "saved_target_branch_index": target_branch,
            "tie_tolerance_px": float(tie_tolerance_px),
            "background_distance_px": float(best_distance),
            "best_distance_px": float(best_distance),
            "selection_coordinate_frame": selection_coordinate_frame,
            "background_distance_reference_frame": selection_reference_frame,
            "group_inventory": inventory,
            "excluded_missing_hkl_candidate_inventory": excluded_missing_hkl_inventory,
            "matching_branch_candidate_inventory": candidate_inventory,
            "identity_tied_candidate_inventory": [
                _compact_entry(
                    entry,
                    detector_hint=detector_hint,
                    measured_detector_point=measured_detector_point,
                    background_point=(
                        background_point if use_current_view_distance else None
                    ),
                    selection_coordinate_frame=selection_coordinate_frame,
                    selection_reference_frame=selection_reference_frame,
                )
                for entry in identity_tied_entries
            ],
            "tied_candidate_inventory": [
                _compact_entry(
                    entry,
                    detector_hint=detector_hint,
                    measured_detector_point=measured_detector_point,
                    background_point=(
                        background_point if use_current_view_distance else None
                    ),
                    selection_coordinate_frame=selection_coordinate_frame,
                    selection_reference_frame=selection_reference_frame,
                )
                for entry in tied_entries
            ],
        }

    return {
        "ok": True,
        "selection_status": "selected",
        "saved_background_current_view_point": background_point,
        "saved_background_current_view_frame": saved_background_current_view_frame,
        "saved_simulated_detector_hint": detector_hint,
        "saved_measured_detector_point": measured_detector_point,
        "saved_target_hkl": target_hkl,
        "saved_target_branch_index": target_branch,
        "background_distance_px": float(best_distance),
        "best_distance_px": float(best_distance),
        "selection_coordinate_frame": selection_coordinate_frame,
        "background_distance_reference_frame": selection_reference_frame,
        "background_distance_candidate_frame": (
            _candidate_current_view_frame(best_entry)
            if use_current_view_distance
            else "native_detector"
        ),
        "selected_candidate": dict(best_entry),
        "group_key": _normalize_q_group_key(best_entry.get("q_group_key")),
        "group_inventory": inventory,
        "excluded_missing_hkl_candidate_inventory": excluded_missing_hkl_inventory,
        "matching_branch_candidate_inventory": candidate_inventory,
    }


def _capture_preflight(
    *,
    saved_state: dict[str, object],
    state_path: Path,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    original_prepare = hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run
    original_projection = hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks

    def _projection_wrapper(*args, **kwargs):
        callbacks = original_projection(*args, **kwargs)
        captured["projection_callbacks"] = callbacks
        return callbacks

    def _prepare_wrapper(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        captured["prepare_kwargs"] = dict(kwargs)
        captured["prepare_result"] = result
        raise _CapturedPreflight()

    hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks = _projection_wrapper
    hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = _prepare_wrapper
    try:
        hgf.run_headless_geometry_fit(
            saved_state,
            state_path=state_path,
            downloads_dir=state_path.parent,
            stamp=f"{state_path.stem}_preflight_probe",
        )
    except _CapturedPreflight:
        pass
    finally:
        hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks = original_projection
        hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = original_prepare
    if "prepare_kwargs" not in captured or "prepare_result" not in captured:
        raise RuntimeError("Failed to capture headless preflight context.")
    return captured


def _capture_execution_setup(
    *,
    saved_state: dict[str, object],
    state_path: Path,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    original_prepare = hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run
    original_execute = hgf.gui_geometry_fit.execute_runtime_geometry_fit

    def _prepare_wrapper(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        captured["prepare_kwargs"] = dict(kwargs)
        captured["prepare_result"] = result
        return result

    def _execute_wrapper(*args, **kwargs):
        captured["execute_kwargs"] = dict(kwargs)
        raise _CapturedExecutionSetup()

    hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = _prepare_wrapper
    hgf.gui_geometry_fit.execute_runtime_geometry_fit = _execute_wrapper
    try:
        hgf.run_headless_geometry_fit(
            saved_state,
            state_path=state_path,
            downloads_dir=state_path.parent,
            stamp=f"{state_path.stem}_downstream_probe",
        )
    except _CapturedExecutionSetup:
        pass
    except Exception as exc:
        captured["execution_error_text"] = str(exc)
    finally:
        hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = original_prepare
        hgf.gui_geometry_fit.execute_runtime_geometry_fit = original_execute
    if "prepare_kwargs" not in captured or "prepare_result" not in captured:
        raise RuntimeError("Failed to capture headless execution setup context.")
    return captured


def _failed_stage_result(
    stage: str,
    *,
    failure_reason: str,
    failed_pair: Mapping[str, object] | None = None,
    **extra: object,
) -> dict[str, object]:
    result = {
        "stage": str(stage),
        "ok": False,
        "failure_reason": str(failure_reason),
    }
    if isinstance(failed_pair, Mapping):
        result["failed_pair"] = dict(failed_pair)
    result.update(extra)
    return result


def _deterministic_identity_sort_key(record: Mapping[str, object]) -> tuple[str, str]:
    return (
        repr(record.get("identity_key")),
        repr(record.get("pair_id")),
    )


def _downstream_stage_record(
    entry: Mapping[str, object] | None,
    *,
    default_dataset_index: int = 0,
    slot_index: int | None = None,
) -> dict[str, object]:
    payload = _comparable_identity_payload(entry)
    record = {
        "slot_index": int(slot_index) if slot_index is not None else None,
        "dataset_index": _entry_dataset_index(entry, default=default_dataset_index),
        "pair_id": (
            str(entry.get("pair_id"))
            if isinstance(entry, Mapping) and entry.get("pair_id") is not None
            else None
        ),
        "overlay_match_index": (
            _int_or_none(entry.get("overlay_match_index"))
            if isinstance(entry, Mapping)
            else None
        ),
        "match_input_index": (
            _int_or_none(entry.get("match_input_index"))
            if isinstance(entry, Mapping)
            else None
        ),
        "match_status": (
            str(entry.get("match_status"))
            if isinstance(entry, Mapping) and entry.get("match_status") is not None
            else None
        ),
        "match_kind": (
            str(entry.get("match_kind"))
            if isinstance(entry, Mapping) and entry.get("match_kind") is not None
            else None
        ),
        "resolution_kind": (
            str(entry.get("resolution_kind"))
            if isinstance(entry, Mapping) and entry.get("resolution_kind") is not None
            else None
        ),
        "identity": payload,
        "identity_key": _downstream_identity_key(
            entry,
            default_dataset_index=default_dataset_index,
        ),
    }
    return record


def _ordered_stage_records(
    entries: Sequence[object] | None,
    *,
    default_dataset_index: int = 0,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for slot_index, raw_entry in enumerate(entries or ()):
        if not isinstance(raw_entry, Mapping):
            continue
        records.append(
            _downstream_stage_record(
                raw_entry,
                default_dataset_index=default_dataset_index,
                slot_index=int(slot_index),
            )
        )
    return records


def _ordered_stage_result(
    stage: str,
    *,
    expected_entries: Sequence[object] | None,
    actual_entries: Sequence[object] | None,
    default_dataset_index: int = 0,
) -> dict[str, object]:
    expected_records = _ordered_stage_records(
        expected_entries,
        default_dataset_index=default_dataset_index,
    )
    actual_records = _ordered_stage_records(
        actual_entries,
        default_dataset_index=default_dataset_index,
    )
    if len(expected_records) != len(actual_records):
        return _failed_stage_result(
            stage,
            failure_reason="pair_count_drift",
            expected_count=len(expected_records),
            actual_count=len(actual_records),
            expected_identity_keys=[
                record.get("identity_key") for record in expected_records
            ],
            actual_identity_keys=[
                record.get("identity_key") for record in actual_records
            ],
        )
    for index, (expected_record, actual_record) in enumerate(
        zip(expected_records, actual_records)
    ):
        if (
            expected_record.get("identity_key") != actual_record.get("identity_key")
            or expected_record.get("identity") != actual_record.get("identity")
        ):
            return _failed_stage_result(
                stage,
                failure_reason="identity_or_order_drift",
                failed_pair={
                    "slot_index": int(index),
                    "expected": expected_record,
                    "actual": actual_record,
                },
                expected_identity_keys=[
                    record.get("identity_key") for record in expected_records
                ],
                actual_identity_keys=[
                    record.get("identity_key") for record in actual_records
                ],
            )
    return {
        "stage": str(stage),
        "ok": True,
        "pair_count": len(actual_records),
        "identity_keys": [record.get("identity_key") for record in actual_records],
    }


def _coverage_stage_result(
    stage: str,
    *,
    expected_entries: Sequence[object] | None,
    actual_entries: Sequence[object] | None,
    default_dataset_index: int = 0,
) -> dict[str, object]:
    def _coverage_failure(
        failure_reason: str,
        *,
        failed_pair: Mapping[str, object] | None = None,
        **extra: object,
    ) -> dict[str, object]:
        return _failed_stage_result(
            stage,
            failure_reason=str(failure_reason),
            failed_pair=failed_pair,
            coverage_mismatch_classification="identity_key_coverage_mismatch",
            **extra,
        )

    expected_records = _ordered_stage_records(
        expected_entries,
        default_dataset_index=default_dataset_index,
    )
    actual_records = _ordered_stage_records(
        actual_entries,
        default_dataset_index=default_dataset_index,
    )
    expected_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for record in expected_records:
        identity_key = record.get("identity_key")
        if identity_key is None:
            return _coverage_failure(
                failure_reason="missing_expected_identity_key",
                failed_pair=record,
            )
        expected_by_key[identity_key] = record
    actual_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for record in actual_records:
        identity_key = record.get("identity_key")
        if identity_key is None:
            return _coverage_failure(
                failure_reason="missing_actual_identity_key",
                failed_pair=record,
            )
        if identity_key in actual_by_key:
            return _coverage_failure(
                failure_reason="duplicate_actual_identity_key",
                failed_pair={
                    "identity_key": identity_key,
                    "first": actual_by_key[identity_key],
                    "second": record,
                },
            )
        if str(record.get("match_status", "") or "").strip().lower() != "matched":
            return _coverage_failure(
                failure_reason="unresolved_full_beam_correspondence",
                failed_pair=record,
            )
        actual_by_key[identity_key] = record
    expected_keys = set(expected_by_key.keys())
    actual_keys = set(actual_by_key.keys())
    if expected_keys != actual_keys:
        return _coverage_failure(
            failure_reason="identity_coverage_drift",
            expected_identity_keys=sorted(expected_keys, key=repr),
            actual_identity_keys=sorted(actual_keys, key=repr),
        )
    for identity_key in sorted(expected_keys, key=repr):
        expected_record = expected_by_key[identity_key]
        actual_record = actual_by_key[identity_key]
        if expected_record.get("identity") != actual_record.get("identity"):
            return _failed_stage_result(
                stage,
                failure_reason="canonical_identity_drift",
                failed_pair={
                    "identity_key": identity_key,
                    "expected": expected_record,
                    "actual": actual_record,
                },
            )
    return {
        "stage": str(stage),
        "ok": True,
        "pair_count": len(actual_by_key),
        "identity_keys": sorted(expected_keys, key=repr),
    }


def _flatten_fixed_correspondence_groups(
    grouped: Mapping[int, Sequence[object]] | None,
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    if not isinstance(grouped, Mapping):
        return flattened
    for dataset_index in sorted(grouped.keys()):
        for raw_entry in grouped.get(dataset_index, ()) or ():
            if isinstance(raw_entry, Mapping):
                flattened.append(dict(raw_entry))
    return flattened


def _full_beam_record_is_fixed(entry: Mapping[str, object] | None) -> bool:
    if not isinstance(entry, Mapping):
        return False
    match_status = str(entry.get("match_status", "") or "").strip().lower()
    resolution_kind = str(entry.get("resolution_kind", "") or "").strip().lower()
    match_kind = str(entry.get("match_kind", "") or "").strip().lower()
    return (
        match_status == "matched"
        and resolution_kind == "fixed_source"
        and match_kind in {"", "fixed_correspondence", "full_beam_fixed"}
    )


def _fixed_correspondence_stage_result(
    stage: str,
    *,
    expected_entries: Sequence[object] | None,
    actual_entries: Sequence[object] | None,
    final_metric_name: str,
    full_beam_polish_summary: Mapping[str, object] | None = None,
    default_dataset_index: int = 0,
) -> dict[str, object]:
    if str(final_metric_name).strip() != "full_beam_fixed_correspondence":
        result = _failed_stage_result(
            stage,
            failure_reason="unexpected_final_metric_name",
            final_metric_name=str(final_metric_name),
        )
        if isinstance(full_beam_polish_summary, Mapping):
            result["full_beam_polish_summary"] = copy.deepcopy(
                dict(full_beam_polish_summary)
            )
            result["full_beam_point_match_comparison"] = _build_full_beam_rejection_comparison(
                full_beam_polish_summary,
                dataset_index=int(default_dataset_index),
            )
        return result
    grouped = opt._build_geometry_fit_fixed_correspondence_groups(actual_entries)
    actual_records = [
        _downstream_stage_record(
            entry,
            default_dataset_index=default_dataset_index,
        )
        for entry in _flatten_fixed_correspondence_groups(grouped)
    ]
    expected_records = [
        _downstream_stage_record(
            entry,
            default_dataset_index=default_dataset_index,
        )
        for entry in expected_entries or ()
        if isinstance(entry, Mapping)
    ]
    for record in actual_records:
        if not _full_beam_record_is_fixed(record):
            return _failed_stage_result(
                stage,
                failure_reason="non_fixed_full_beam_correspondence",
                failed_pair=record,
                final_metric_name=str(final_metric_name),
            )
    expected_sorted = sorted(expected_records, key=_deterministic_identity_sort_key)
    actual_sorted = sorted(actual_records, key=_deterministic_identity_sort_key)
    if len(expected_sorted) != len(actual_sorted):
        return _failed_stage_result(
            stage,
            failure_reason="pair_count_drift",
            expected_count=len(expected_sorted),
            actual_count=len(actual_sorted),
            final_metric_name=str(final_metric_name),
        )
    for index, (expected_record, actual_record) in enumerate(
        zip(expected_sorted, actual_sorted)
    ):
        if (
            expected_record.get("identity_key") != actual_record.get("identity_key")
            or expected_record.get("identity") != actual_record.get("identity")
        ):
            return _failed_stage_result(
                stage,
                failure_reason="canonical_identity_drift",
                failed_pair={
                    "slot_index": int(index),
                    "expected": expected_record,
                    "actual": actual_record,
                },
                final_metric_name=str(final_metric_name),
            )
    return {
        "stage": str(stage),
        "ok": True,
        "pair_count": len(actual_sorted),
        "identity_keys": [
            record.get("identity_key") for record in actual_sorted
        ],
        "final_metric_name": str(final_metric_name),
    }


def _filter_stage_entries_for_dataset(
    entries: Sequence[object] | None,
    *,
    dataset_index: int,
) -> list[dict[str, object]]:
    return [
        dict(entry)
        for entry in entries or ()
        if isinstance(entry, Mapping)
        and _entry_dataset_index(entry, default=int(dataset_index)) == int(dataset_index)
    ]


def _point_match_entries_from_mapping(
    payload: Mapping[str, object] | None,
    key: str,
) -> list[dict[str, object]]:
    if not isinstance(payload, Mapping):
        return []
    raw_entries = payload.get(str(key), ())
    if not isinstance(raw_entries, Sequence) or isinstance(
        raw_entries,
        (str, bytes, bytearray),
    ):
        return []
    return [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]


def _full_beam_point_match_entries(result: object) -> list[dict[str, object]]:
    full_beam_summary = getattr(result, "full_beam_polish_summary", None)
    final_metric_name = str(getattr(result, "final_metric_name", "") or "").strip()
    if final_metric_name != "full_beam_fixed_correspondence":
        summary_entries = _point_match_entries_from_mapping(
            full_beam_summary,
            "point_match_diagnostics",
        )
        if summary_entries:
            return summary_entries
    raw_entries = getattr(result, "point_match_diagnostics", None)
    if isinstance(raw_entries, Sequence) and not isinstance(
        raw_entries,
        (str, bytes, bytearray),
    ):
        direct_entries = [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
        if direct_entries:
            return direct_entries
    if isinstance(full_beam_summary, Mapping):
        summary_entries = _point_match_entries_from_mapping(
            full_beam_summary,
            "point_match_diagnostics",
        )
        if summary_entries:
            return summary_entries
    return []


def _full_beam_comparison_record(
    entry: Mapping[str, object] | None,
    *,
    default_dataset_index: int = 0,
    slot_index: int | None = None,
) -> dict[str, object]:
    record = _downstream_stage_record(
        entry,
        default_dataset_index=default_dataset_index,
        slot_index=slot_index,
    )
    if isinstance(entry, Mapping):
        record.update(
            {
                "match_status": (
                    str(entry.get("match_status"))
                    if entry.get("match_status") is not None
                    else None
                ),
                "resolved_table_index": _int_or_none(entry.get("resolved_table_index")),
                "resolved_peak_index": _int_or_none(entry.get("resolved_peak_index")),
                "distance_px": _float_or_none(entry.get("distance_px")),
                "dx_px": _float_or_none(entry.get("dx_px")),
                "dy_px": _float_or_none(entry.get("dy_px")),
                "weighted_dx_px": _float_or_none(entry.get("weighted_dx_px")),
                "weighted_dy_px": _float_or_none(entry.get("weighted_dy_px")),
                "distance_weight": _float_or_none(entry.get("distance_weight")),
                "sigma_weight": _float_or_none(entry.get("sigma_weight")),
                "priority_weight": _float_or_none(entry.get("priority_weight")),
                "weight": _float_or_none(entry.get("weight")),
                "match_radius_exceeded": bool(entry.get("match_radius_exceeded", False)),
                "resolution_reason": (
                    str(entry.get("resolution_reason"))
                    if entry.get("resolution_reason") is not None
                    else None
                ),
            }
        )
    return record


def _full_beam_records_by_identity_key(
    records: Sequence[Mapping[str, object]] | None,
) -> tuple[
    dict[tuple[object, ...], dict[str, object]],
    list[tuple[object, ...] | None],
    list[dict[str, object]],
]:
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    ordered_keys: list[tuple[object, ...] | None] = []
    duplicates: list[dict[str, object]] = []
    for record in records or ():
        key = record.get("identity_key")
        ordered_keys.append(key)
        if key is None:
            continue
        if key in by_key:
            duplicates.append(
                {
                    "identity_key": key,
                    "first": dict(by_key[key]),
                    "second": dict(record),
                }
            )
            continue
        by_key[key] = dict(record)
    return by_key, ordered_keys, duplicates


def _build_full_beam_rejection_comparison(
    full_beam_polish_summary: Mapping[str, object] | None,
    *,
    dataset_index: int,
) -> dict[str, object]:
    def _pair_delta_value(
        start_value: object,
        candidate_value: object,
    ) -> float | None:
        start_number = _float_or_none(start_value)
        candidate_number = _float_or_none(candidate_value)
        if start_number is None or candidate_number is None:
            return None
        return float(candidate_number - start_number)

    def _pair_delta_sq_value(
        start_value: object,
        candidate_value: object,
    ) -> float | None:
        start_number = _float_or_none(start_value)
        candidate_number = _float_or_none(candidate_value)
        if start_number is None or candidate_number is None:
            return None
        return float(candidate_number * candidate_number - start_number * start_number)

    start_entries = _filter_stage_entries_for_dataset(
        _point_match_entries_from_mapping(
            full_beam_polish_summary,
            "start_point_match_diagnostics",
        ),
        dataset_index=int(dataset_index),
    )
    candidate_entries = _filter_stage_entries_for_dataset(
        _point_match_entries_from_mapping(
            full_beam_polish_summary,
            "candidate_point_match_diagnostics",
        ),
        dataset_index=int(dataset_index),
    )
    start_records = [
        _full_beam_comparison_record(
            entry,
            default_dataset_index=int(dataset_index),
            slot_index=int(slot_index),
        )
        for slot_index, entry in enumerate(start_entries)
    ]
    candidate_records = [
        _full_beam_comparison_record(
            entry,
            default_dataset_index=int(dataset_index),
            slot_index=int(slot_index),
        )
        for slot_index, entry in enumerate(candidate_entries)
    ]
    start_by_key, start_ordered_keys, start_duplicates = _full_beam_records_by_identity_key(
        start_records
    )
    candidate_by_key, candidate_ordered_keys, candidate_duplicates = (
        _full_beam_records_by_identity_key(candidate_records)
    )
    start_identity_keys = sorted(start_by_key.keys(), key=repr)
    candidate_identity_keys = sorted(candidate_by_key.keys(), key=repr)
    coverage_mismatch = bool(
        len(start_records) != len(candidate_records)
        or start_identity_keys != candidate_identity_keys
        or start_duplicates
        or candidate_duplicates
    )
    paired_entries: list[dict[str, object]] = []
    identity_drift_count = 0
    coverage_drift_count = 0
    resolved_correspondence_drift_count = 0
    all_identity_keys = sorted(
        set(start_by_key.keys()) | set(candidate_by_key.keys()),
        key=repr,
    )
    for identity_key in all_identity_keys:
        start_record = start_by_key.get(identity_key)
        candidate_record = candidate_by_key.get(identity_key)
        coverage_drift = start_record is None or candidate_record is None
        identity_drift = bool(
            coverage_drift
            or start_record.get("identity_key") != candidate_record.get("identity_key")
            or start_record.get("identity") != candidate_record.get("identity")
        )
        resolved_correspondence_drift = bool(
            not coverage_drift
            and (
                start_record.get("resolved_table_index")
                != candidate_record.get("resolved_table_index")
                or start_record.get("resolved_peak_index")
                != candidate_record.get("resolved_peak_index")
            )
        )
        identity_drift_count += int(identity_drift)
        coverage_drift_count += int(coverage_drift)
        resolved_correspondence_drift_count += int(resolved_correspondence_drift)
        pair_id = None
        if isinstance(start_record, Mapping):
            pair_id = start_record.get("pair_id")
        if pair_id is None and isinstance(candidate_record, Mapping):
            pair_id = candidate_record.get("pair_id")
        identity_payload = (
            start_record.get("identity")
            if isinstance(start_record, Mapping)
            else (
                candidate_record.get("identity")
                if isinstance(candidate_record, Mapping)
                else {}
            )
        )
        start_distance_px = (
            start_record.get("distance_px") if isinstance(start_record, Mapping) else None
        )
        candidate_distance_px = (
            candidate_record.get("distance_px") if isinstance(candidate_record, Mapping) else None
        )
        delta_px = _pair_delta_value(start_distance_px, candidate_distance_px)
        delta_sq = _pair_delta_sq_value(start_distance_px, candidate_distance_px)
        paired_entries.append(
            {
                "pair_id": pair_id,
                "hkl": (
                    identity_payload.get("hkl") if isinstance(identity_payload, Mapping) else None
                ),
                "source_reflection_index": (
                    identity_payload.get("source_reflection_index")
                    if isinstance(identity_payload, Mapping)
                    else None
                ),
                "source_branch_index": (
                    identity_payload.get("source_branch_index")
                    if isinstance(identity_payload, Mapping)
                    else None
                ),
                "source_peak_index": (
                    identity_payload.get("source_peak_index")
                    if isinstance(identity_payload, Mapping)
                    else None
                ),
                "identity_key": identity_key,
                "start_match_status": (
                    start_record.get("match_status") if isinstance(start_record, Mapping) else None
                ),
                "candidate_match_status": (
                    candidate_record.get("match_status")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_resolved_table_index": (
                    start_record.get("resolved_table_index")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_resolved_table_index": (
                    candidate_record.get("resolved_table_index")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_resolved_peak_index": (
                    start_record.get("resolved_peak_index")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_resolved_peak_index": (
                    candidate_record.get("resolved_peak_index")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_distance_px": (
                    start_distance_px
                ),
                "candidate_distance_px": (
                    candidate_distance_px
                ),
                "delta_px": delta_px,
                "delta_sq_px": delta_sq,
                "start_match_radius_exceeded": (
                    start_record.get("match_radius_exceeded")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_match_radius_exceeded": (
                    candidate_record.get("match_radius_exceeded")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_resolution_reason": (
                    start_record.get("resolution_reason")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_resolution_reason": (
                    candidate_record.get("resolution_reason")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_weighted_dx_px": (
                    start_record.get("weighted_dx_px")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_weighted_dx_px": (
                    candidate_record.get("weighted_dx_px")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_weighted_dy_px": (
                    start_record.get("weighted_dy_px")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_weighted_dy_px": (
                    candidate_record.get("weighted_dy_px")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_distance_weight": (
                    start_record.get("distance_weight")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_distance_weight": (
                    candidate_record.get("distance_weight")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_sigma_weight": (
                    start_record.get("sigma_weight")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_sigma_weight": (
                    candidate_record.get("sigma_weight")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_priority_weight": (
                    start_record.get("priority_weight")
                    if isinstance(start_record, Mapping)
                    else None
                ),
                "candidate_priority_weight": (
                    candidate_record.get("priority_weight")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "start_weight": (
                    start_record.get("weight") if isinstance(start_record, Mapping) else None
                ),
                "candidate_weight": (
                    candidate_record.get("weight")
                    if isinstance(candidate_record, Mapping)
                    else None
                ),
                "identity_drift": bool(identity_drift),
                "coverage_drift": bool(coverage_drift),
                "resolved_correspondence_drift": bool(resolved_correspondence_drift),
            }
        )
    if coverage_mismatch:
        comparison_classification = "identity_key_coverage_mismatch"
    elif identity_drift_count > 0:
        comparison_classification = "canonical_identity_drift"
    elif resolved_correspondence_drift_count > 0:
        comparison_classification = "resolved_correspondence_drift"
    else:
        comparison_classification = "objective_acceptance_mismatch"
    ranked_by_delta_px = sorted(
        paired_entries,
        key=lambda entry: (
            _float_or_none(entry.get("delta_px"))
            if _float_or_none(entry.get("delta_px")) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    ranked_by_candidate_distance_px = sorted(
        paired_entries,
        key=lambda entry: (
            _float_or_none(entry.get("candidate_distance_px"))
            if _float_or_none(entry.get("candidate_distance_px")) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    ranked_by_delta_sq_px = sorted(
        paired_entries,
        key=lambda entry: (
            _float_or_none(entry.get("delta_sq_px"))
            if _float_or_none(entry.get("delta_sq_px")) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    return {
        "comparison_classification": str(comparison_classification),
        "start_identity_key_count": int(len(start_records)),
        "candidate_identity_key_count": int(len(candidate_records)),
        "start_identity_key_digest": _stable_digest(start_identity_keys),
        "candidate_identity_key_digest": _stable_digest(candidate_identity_keys),
        "start_identity_keys": start_identity_keys,
        "candidate_identity_keys": candidate_identity_keys,
        "start_identity_key_order_digest": _stable_digest(start_ordered_keys),
        "candidate_identity_key_order_digest": _stable_digest(candidate_ordered_keys),
        "start_duplicate_identity_keys": start_duplicates,
        "candidate_duplicate_identity_keys": candidate_duplicates,
        "identity_drift_count": int(identity_drift_count),
        "coverage_drift_count": int(coverage_drift_count),
        "resolved_correspondence_drift_count": int(resolved_correspondence_drift_count),
        "paired_entries": paired_entries,
        "ranked_by_delta_px": ranked_by_delta_px,
        "ranked_by_candidate_distance_px": ranked_by_candidate_distance_px,
        "ranked_by_delta_sq_px": ranked_by_delta_sq_px,
    }


def _validate_downstream_identity_input(
    saved_state: Mapping[str, object] | None,
    *,
    background_index: int,
) -> dict[str, object]:
    geometry_state = (
        saved_state.get("geometry", {})
        if isinstance(saved_state, Mapping) and isinstance(saved_state.get("geometry"), Mapping)
        else {}
    )
    peak_records = geometry_state.get("peak_records")
    q_group_rows = geometry_state.get("q_group_rows")
    all_entries = _all_saved_manual_pair_entries(saved_state)
    background_entries = _saved_entries_for_background(
        saved_state,
        background_index=int(background_index),
    )
    if not _empty_saved_sequence(peak_records):
        return _failed_stage_result(
            "input_contract",
            failure_reason="peak_records_not_empty",
            peak_records_count=(
                len(peak_records)
                if isinstance(peak_records, Sequence)
                and not isinstance(peak_records, (str, bytes, bytearray))
                else None
            ),
        )
    if not _empty_saved_sequence(q_group_rows):
        return _failed_stage_result(
            "input_contract",
            failure_reason="q_group_rows_not_empty",
            q_group_rows_count=(
                len(q_group_rows)
                if isinstance(q_group_rows, Sequence)
                and not isinstance(q_group_rows, (str, bytes, bytearray))
                else None
            ),
        )
    if not background_entries:
        return _failed_stage_result(
            "input_contract",
            failure_reason="missing_manual_pairs_for_background",
            background_index=int(background_index),
        )
    for entry in all_entries:
        if (
            entry.get("source_reflection_namespace") != "full_reflection"
            or bool(entry.get("source_reflection_is_full", False)) is not True
        ):
            return _failed_stage_result(
                "input_contract",
                failure_reason="non_canonical_manual_pair",
                failed_pair=_identity_payload(entry),
            )
    return {
        "stage": "input_contract",
        "ok": True,
        "pair_count": len(background_entries),
        "all_manual_pair_count": len(all_entries),
    }

def _build_group_cache(
    *,
    background_index: int,
    params: Mapping[str, object],
    dataset: Mapping[str, object],
    manual_dataset_bindings,
    projection_callbacks,
) -> dict[str, object]:
    _native_bg, display_bg = manual_dataset_bindings.load_background_by_index(
        int(background_index)
    )
    requested_signature = None
    simulation_diagnostics = dataset.get("simulation_diagnostics")
    if isinstance(simulation_diagnostics, Mapping):
        requested_signature = simulation_diagnostics.get("requested_signature")
    cache_data, _next_signature, _next_state = gui_manual_geometry.build_geometry_manual_pick_cache(
        param_set=dict(params),
        prefer_cache=False,
        background_index=int(background_index),
        current_background_index=int(manual_dataset_bindings.current_background_index),
        background_image=display_bg,
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **kwargs: gui_manual_geometry.geometry_manual_pick_cache_signature(
            placed_cache_signature=(
                gui_manual_geometry.geometry_manual_pick_placed_cache_signature(
                    source_snapshot_signature=requested_signature,
                    background_index=int(kwargs["background_index"]),
                    background_image=kwargs["background_image"],
                    use_caked_space=False,
                )
            ),
            disabled_qr_sets=(),
            disabled_qz_sections=(),
        ),
        source_rows_for_background=manual_dataset_bindings.geometry_manual_source_rows_for_background,
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        peak_records=[],
        build_grouped_candidates=projection_callbacks.pick_candidates,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        current_match_config=manual_dataset_bindings.geometry_manual_match_config,
    )
    return cache_data


def _find_resolved_live_row(
    *,
    live_rows: Sequence[Mapping[str, object]],
    resolution_diag: Mapping[str, object],
) -> dict[str, object] | None:
    row_key = _normalize_key_tuple(resolution_diag.get("fit_source_row_key"))
    peak_key = _normalize_key_tuple(resolution_diag.get("fit_source_peak_key"))
    if row_key is not None:
        for row in live_rows:
            if gui_geometry_fit._geometry_fit_source_row_key(row) == row_key:
                return dict(row)
    if peak_key is not None:
        for row in live_rows:
            if gui_geometry_fit._geometry_fit_source_peak_key(row) == peak_key:
                return dict(row)
    target_identity = (
        resolution_diag.get("fit_source_reflection_index"),
        resolution_diag.get("fit_source_branch_index"),
    )
    for row in live_rows:
        row_identity = (
            row.get("source_reflection_index"),
            row.get("source_branch_index"),
        )
        if row_identity == target_identity:
            return dict(row)
    return None


def _validate_harness(
    *,
    saved_state: Mapping[str, object],
    dataset: Mapping[str, object],
    background_index: int,
    params: Mapping[str, object],
    manual_dataset_bindings,
    group_cache: Mapping[str, object],
) -> dict[str, object]:
    files_state = saved_state.get("files", {}) if isinstance(saved_state.get("files"), Mapping) else {}
    saved_background_files = [
        str(path) for path in (files_state.get("background_files", []) or ()) if str(path).strip()
    ]
    expected_background_path = (
        saved_background_files[int(background_index)]
        if len(saved_background_files) > int(background_index)
        else None
    )
    actual_background_path = (
        str(manual_dataset_bindings.osc_files[int(background_index)])
        if len(manual_dataset_bindings.osc_files) > int(background_index)
        else None
    )
    simulation_diagnostics = dataset.get("simulation_diagnostics")
    requested_signature = (
        simulation_diagnostics.get("requested_signature")
        if isinstance(simulation_diagnostics, Mapping)
        else None
    )
    cache_signature = group_cache.get("signature")
    placed_cache_signature = (
        cache_signature[0]
        if isinstance(cache_signature, tuple) and len(cache_signature) > 0
        else None
    )
    source_snapshot_signature_match = bool(
        requested_signature is not None
        and (
            cache_signature == requested_signature
            or (
                isinstance(placed_cache_signature, tuple)
                and len(placed_cache_signature) > 0
                and placed_cache_signature[0] == requested_signature
            )
        )
    )
    return {
        "valid": bool(
            expected_background_path == actual_background_path
            and source_snapshot_signature_match
        ),
        "background_index": int(background_index),
        "expected_background_path": expected_background_path,
        "actual_background_path": actual_background_path,
        "param_digest": _stable_digest(dict(params)),
        "requested_signature_digest": _stable_digest(requested_signature),
        "placed_cache_signature_digest": _stable_digest(placed_cache_signature),
        "group_cache_signature_digest": _stable_digest(cache_signature),
        "source_snapshot_signature_match": source_snapshot_signature_match,
    }


def _validate_pair(
    *,
    background_index: int,
    slot_index: int,
    expected_pair_id: str,
    saved_entries: Sequence[Mapping[str, object]],
    dataset: Mapping[str, object],
    group_cache: Mapping[str, object],
    entry_display_coords,
) -> dict[str, object]:
    saved_entry = dict(saved_entries[int(slot_index)])
    saved_entry["q_group_key"] = _normalize_q_group_key(saved_entry.get("q_group_key"))
    expected_identity = _identity_payload(saved_entry, pair_id=expected_pair_id)

    measured_for_fit = list(dataset.get("measured_for_fit", []) or ())
    source_resolution_diagnostics = list(dataset.get("source_resolution_diagnostics", []) or ())
    if int(slot_index) >= len(measured_for_fit) or int(slot_index) >= len(source_resolution_diagnostics):
        return {
            "ok": False,
            "failure_stage": "preflight_copyback",
            "message": "requested slot missing from preflight dataset",
            "slot_index": int(slot_index),
        }

    preflight_pair = dict(measured_for_fit[int(slot_index)])
    resolution_diag = dict(source_resolution_diagnostics[int(slot_index)])
    q_group_key = _normalize_q_group_key(saved_entry.get("q_group_key"))
    live_rows_all = list(dataset.get("source_rows_for_trace", []) or ())
    live_rows = [
        dict(entry)
        for entry in live_rows_all
        if _normalize_q_group_key(entry.get("q_group_key")) == q_group_key
        and _normalize_hkl(entry.get("hkl")) == _normalize_hkl(saved_entry.get("hkl"))
    ]
    grouped_candidates = list((group_cache.get("grouped_candidates", {}) or {}).get(q_group_key, []) or ())
    resolved_live_row = _find_resolved_live_row(
        live_rows=live_rows_all,
        resolution_diag=resolution_diag,
    )
    display_hint = _display_hint(saved_entry, entry_display_coords=entry_display_coords)
    fit_kind = resolution_diag.get("fit_resolution_kind")
    overlay_kind = resolution_diag.get("overlay_resolution_kind")
    ok = bool(
        _identity_payload(preflight_pair) == expected_identity
        and _identity_payload(resolved_live_row, pair_id=expected_pair_id) == expected_identity
        and preflight_pair.get("pair_id") == expected_pair_id
        and fit_kind in {"source_row", "source_peak"}
        and overlay_kind in {"source_row", "source_peak"}
    )
    failure_stage = None
    if not grouped_candidates:
        failure_stage = "grouped_candidate_regeneration"
    elif not live_rows:
        failure_stage = "live_source_row_regeneration"
    elif _identity_payload(resolved_live_row, pair_id=expected_pair_id) != expected_identity:
        failure_stage = "resolved_live_row_selection"
    elif _identity_payload(preflight_pair) != expected_identity or preflight_pair.get("pair_id") != expected_pair_id:
        failure_stage = "preflight_copyback"
    elif fit_kind not in {"source_row", "source_peak"} or overlay_kind not in {"source_row", "source_peak"}:
        failure_stage = "resolved_live_row_selection"
    return {
        "ok": ok,
        "background_index": int(background_index),
        "slot_index": int(slot_index),
        "expected_pair_id": expected_pair_id,
        "expected_identity": expected_identity,
        "group_cache_metadata": dict(group_cache.get("cache_metadata", {}) or {}),
        "saved_entry": {
            "pair_id": None,
            "hkl": _normalize_hkl(saved_entry.get("hkl")),
            "q_group_key": q_group_key,
            "source_reflection_index": saved_entry.get("source_reflection_index"),
            "source_reflection_namespace": saved_entry.get("source_reflection_namespace"),
            "source_reflection_is_full": saved_entry.get("source_reflection_is_full"),
            "source_branch_index": saved_entry.get("source_branch_index"),
            "source_peak_index": saved_entry.get("source_peak_index"),
        },
        "fit_resolution_kind": fit_kind,
        "overlay_resolution_kind": overlay_kind,
        "row_candidate_status": resolution_diag.get("row_candidate_status"),
        "peak_candidate_status": resolution_diag.get("peak_candidate_status"),
        "failure_stage": failure_stage,
        "chosen_resolved_live_row": _compact_entry(
            resolved_live_row,
            display_hint=display_hint,
        ),
        "emitted_preflight_normalized_pair": dict(preflight_pair),
        "source_resolution_diagnostic": resolution_diag,
        "grouped_candidate_inventory": [
            _compact_entry(entry, display_hint=display_hint)
            for entry in grouped_candidates
        ],
        "live_source_row_inventory": [
            _compact_entry(entry, display_hint=display_hint)
            for entry in live_rows
        ],
    }


def _theta_base_for_background(
    *,
    background_index: int,
    bindings,
    params: Mapping[str, object],
) -> float:
    try:
        bindings.apply_geometry_fit_background_selection(
            trigger_update=False,
            sync_live_theta=False,
        )
    except Exception:
        pass
    theta_default = float(params.get("theta_initial", bindings.theta_initial))
    try:
        selected_background_indices = list(
            bindings.current_geometry_fit_background_indices(strict=True)
        )
    except Exception:
        selected_background_indices = [int(background_index)]
    try:
        uses_shared_theta = bool(
            bindings.geometry_fit_uses_shared_theta_offset(selected_background_indices)
        )
    except Exception:
        uses_shared_theta = False
    if uses_shared_theta:
        try:
            bindings.apply_background_theta_metadata(
                trigger_update=False,
                sync_live_theta=False,
            )
        except Exception:
            pass
    for strict_count in (True, False):
        try:
            theta_values = list(bindings.current_background_theta_values(strict_count=strict_count))
        except Exception:
            continue
        if len(theta_values) > int(background_index):
            try:
                return float(theta_values[int(background_index)])
            except Exception:
                continue
    return float(theta_default)


def _raw_entry_display_coords(entry: Mapping[str, object] | None) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    for x_key, y_key in (("x", "y"), ("raw_x", "raw_y")):
        try:
            col = float(entry.get(x_key, np.nan))
            row = float(entry.get(y_key, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    return None


def _isolated_manual_dataset_bindings(bindings):
    return replace(
        bindings.manual_dataset_bindings,
        geometry_manual_refresh_pair_entry=None,
        geometry_manual_entry_display_coords=_raw_entry_display_coords,
    )


def _build_single_background_dataset(
    *,
    background_index: int,
    params: Mapping[str, object],
    bindings,
) -> dict[str, object]:
    geometry_runtime_cfg = bindings.build_runtime_config(dict(params or {}))
    orientation_cfg = (
        geometry_runtime_cfg.get("orientation", {})
        if isinstance(geometry_runtime_cfg, Mapping)
        else {}
    )
    if not isinstance(orientation_cfg, Mapping):
        orientation_cfg = {}
    isolated_bindings = _isolated_manual_dataset_bindings(bindings)
    return gui_geometry_fit.build_geometry_manual_fit_dataset(
        int(background_index),
        theta_base=_theta_base_for_background(
            background_index=int(background_index),
            bindings=bindings,
            params=params,
        ),
        base_fit_params=dict(params or {}),
        manual_dataset_bindings=isolated_bindings,
        orientation_cfg=dict(orientation_cfg),
    )


def _prepare_validation_context(state_path: Path, background_index: int) -> dict[str, object]:
    payload = load_gui_state_file(state_path)
    saved_state = dict(payload["state"])
    raw_saved_entries = _saved_entries_for_background(
        saved_state,
        background_index=int(background_index),
    )
    captured = _capture_preflight(saved_state=saved_state, state_path=state_path)
    prepare_kwargs = dict(captured["prepare_kwargs"])
    prepare_result = captured["prepare_result"]
    projection_callbacks = captured["projection_callbacks"]
    prepared_run = prepare_result.prepared_run
    bindings = prepare_kwargs["bindings"]
    params = dict(prepare_kwargs["params"] or {})
    manual_dataset_bindings = bindings.manual_dataset_bindings
    saved_entries = list(
        manual_dataset_bindings.geometry_manual_pairs_for_index(int(background_index)) or ()
    )
    runtime_prepare_ok = bool(prepared_run is not None and not prepare_result.error_text)
    result: dict[str, object] = {
        "state_path": str(state_path),
        "background_index": int(background_index),
        "saved_pair_count": int(len(saved_entries)),
        "captured_preflight_error_text": prepare_result.error_text,
        "used_isolated_background_dataset": prepared_run is None,
        "runtime_prepare_ok": runtime_prepare_ok,
    }
    try:
        if prepared_run is not None:
            dataset = dict(prepared_run.current_dataset or {})
            manual_dataset_bindings = bindings.manual_dataset_bindings
        else:
            dataset = _build_single_background_dataset(
                background_index=int(background_index),
                params=params,
                bindings=bindings,
            )
            manual_dataset_bindings = _isolated_manual_dataset_bindings(bindings)
    except Exception as exc:
        result["ok"] = False
        result["classification"] = "source_snapshot_unavailable"
        result["failure_stage"] = "live_source_row_regeneration"
        result["error_text"] = str(exc)
        return result

    result["dataset_pair_count"] = int(dataset.get("pair_count", 0) or 0)
    result["dataset_resolved_source_pair_count"] = int(
        dataset.get("resolved_source_pair_count", 0) or 0
    )
    try:
        group_cache = _build_group_cache(
            background_index=int(background_index),
            params=params,
            dataset=dataset,
            manual_dataset_bindings=manual_dataset_bindings,
            projection_callbacks=projection_callbacks,
        )
    except Exception as exc:
        result["ok"] = False
        result["classification"] = "group_cache_unavailable"
        result["failure_stage"] = "grouped_candidate_regeneration"
        result["error_text"] = str(exc)
        return result
    harness_validation = _validate_harness(
        saved_state=saved_state,
        dataset=dataset,
        background_index=int(background_index),
        params=params,
        manual_dataset_bindings=manual_dataset_bindings,
        group_cache=group_cache,
    )
    result["harness_validation"] = harness_validation
    if not bool(harness_validation.get("valid", False)):
        result["ok"] = False
        result["classification"] = "harness_invalid"
        return result
    result["ok"] = True
    result["saved_state"] = saved_state
    result["params"] = params
    result["bindings"] = bindings
    result["projection_callbacks"] = projection_callbacks
    result["manual_dataset_bindings"] = manual_dataset_bindings
    result["dataset"] = dataset
    result["group_cache"] = group_cache
    result["saved_entries"] = saved_entries
    result["raw_saved_entries"] = raw_saved_entries
    return result


def _run_saved_state_compatibility_validation(
    state_path: Path,
    background_index: int,
) -> dict[str, object]:
    context = _prepare_validation_context(state_path, background_index)
    if not bool(context.get("ok", False)):
        return context
    saved_entries = list(context["saved_entries"])
    dataset = dict(context["dataset"])
    group_cache = dict(context["group_cache"])
    manual_dataset_bindings = context["manual_dataset_bindings"]
    result = {
        key: value
        for key, value in context.items()
        if key
        not in {
            "saved_state",
            "params",
            "bindings",
            "projection_callbacks",
            "manual_dataset_bindings",
            "dataset",
            "group_cache",
            "saved_entries",
            "raw_saved_entries",
        }
    }

    checked_slot_indices = _compatibility_probe_slot_indices(saved_entries)
    result["checked_slot_indices"] = list(checked_slot_indices)
    if not checked_slot_indices:
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "saved_entry_missing",
            "message": "no saved entries available for compatibility validation",
        }
        return result

    checked_pairs: list[dict[str, object]] = []
    for slot_index in checked_slot_indices:
        pair_result = _validate_pair(
            background_index=int(background_index),
            slot_index=int(slot_index),
            expected_pair_id=f"bg{int(background_index)}:pair{int(slot_index)}",
            saved_entries=saved_entries,
            dataset=dataset,
            group_cache=group_cache,
            entry_display_coords=manual_dataset_bindings.geometry_manual_entry_display_coords,
        )
        checked_pairs.append(pair_result)
        if not bool(pair_result.get("ok", False)):
            result["ok"] = False
            result["classification"] = "seam_failure"
            result["checked_pairs"] = checked_pairs
            result["failed_pair"] = pair_result
            return result

    if len(checked_pairs) >= 2 and _canonical_identity(
        checked_pairs[0]["chosen_resolved_live_row"]
    ) == _canonical_identity(checked_pairs[1]["chosen_resolved_live_row"]):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["checked_pairs"] = checked_pairs
        result["failed_pair"] = {
            "failure_stage": "resolved_live_row_selection",
            "message": "mirrored pair collapsed to same canonical identity",
            "pair1_identity": checked_pairs[0]["chosen_resolved_live_row"],
            "pair2_identity": checked_pairs[1]["chosen_resolved_live_row"],
        }
        return result

    full_sweep: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for slot_index in range(len(saved_entries)):
        pair_result = _validate_pair(
            background_index=int(background_index),
            slot_index=int(slot_index),
            expected_pair_id=f"bg{int(background_index)}:pair{int(slot_index)}",
            saved_entries=saved_entries,
            dataset=dataset,
            group_cache=group_cache,
            entry_display_coords=manual_dataset_bindings.geometry_manual_entry_display_coords,
        )
        full_sweep.append(
            {
                "slot_index": int(slot_index),
                "expected_pair_id": pair_result.get("expected_pair_id"),
                "ok": bool(pair_result.get("ok", False)),
                "identity": _identity_payload(
                    pair_result.get("emitted_preflight_normalized_pair"),
                ),
                "fit_resolution_kind": pair_result.get("fit_resolution_kind"),
                "overlay_resolution_kind": pair_result.get("overlay_resolution_kind"),
            }
        )
        if not bool(pair_result.get("ok", False)):
            result["ok"] = False
            result["classification"] = "seam_failure"
            result["checked_pairs"] = checked_pairs
            result["full_sweep"] = full_sweep
            result["failed_pair"] = pair_result
            return result
        identity = _canonical_identity(pair_result.get("emitted_preflight_normalized_pair"))
        if identity in seen:
            result["ok"] = False
            result["classification"] = "seam_failure"
            result["checked_pairs"] = checked_pairs
            result["full_sweep"] = full_sweep
            result["failed_pair"] = {
                "failure_stage": "preflight_copyback",
                "message": "duplicate canonical identity collision in full sweep",
                "identity": identity,
                "slot_index": int(slot_index),
            }
            return result
        seen.add(identity)

    result["ok"] = True
    result["classification"] = "pass"
    result["checked_pairs"] = checked_pairs
    result["full_sweep"] = full_sweep
    return result


def _compatibility_probe_slot_indices(
    saved_entries: Sequence[Mapping[str, object]],
) -> list[int]:
    if not saved_entries:
        return []

    grouped_slots: dict[tuple[object, object], list[int]] = {}
    for slot_index, entry in enumerate(saved_entries):
        group_key = (
            _normalize_q_group_key(entry.get("q_group_key")),
            _normalize_hkl(entry.get("hkl")),
        )
        grouped_slots.setdefault(group_key, []).append(int(slot_index))

    for slot_indices in grouped_slots.values():
        if len(slot_indices) >= 2:
            return list(slot_indices[:2])

    return list(range(min(2, len(saved_entries))))


def _public_context_fields(context: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in context.items()
        if key
        not in {
            "saved_state",
            "params",
            "bindings",
            "projection_callbacks",
            "manual_dataset_bindings",
            "dataset",
            "group_cache",
            "saved_entries",
        }
    }


def _current_source_rows_for_background(
    *,
    background_index: int,
    context: Mapping[str, object],
    consumer: str,
) -> list[dict[str, object]]:
    manual_dataset_bindings = context["manual_dataset_bindings"]
    params = dict(context["params"])
    source_rows_for_background = getattr(
        manual_dataset_bindings,
        "geometry_manual_source_rows_for_background",
        None,
    )
    if callable(source_rows_for_background):
        rows = source_rows_for_background(
            int(background_index),
            params,
            consumer=str(consumer),
        )
        return [dict(entry) for entry in rows or () if isinstance(entry, Mapping)]
    dataset = context["dataset"]
    return [
        dict(entry)
        for entry in dataset.get("source_rows_for_trace", ()) or ()
        if isinstance(entry, Mapping)
    ]


def _entry_caked_display_coords(
    entry: Mapping[str, object] | None,
    *,
    projection_callbacks,
) -> tuple[float, float] | None:
    for keys in (("caked_x", "caked_y"), ("raw_caked_x", "raw_caked_y")):
        point = _finite_point(entry, *keys)
        if point is not None:
            return point
    detector_point = _finite_point(entry, "detector_x", "detector_y")
    if detector_point is not None:
        converted = projection_callbacks.native_detector_coords_to_caked_display_coords(
            float(detector_point[0]),
            float(detector_point[1]),
        )
        if converted is not None:
            try:
                return float(converted[0]), float(converted[1])
            except Exception:
                return None
    display_point = _finite_point(entry, "x", "y")
    if display_point is not None:
        native_point = projection_callbacks.background_display_to_native_detector_coords(
            float(display_point[0]),
            float(display_point[1]),
        )
        if native_point is not None:
            converted = projection_callbacks.native_detector_coords_to_caked_display_coords(
                float(native_point[0]),
                float(native_point[1]),
            )
            if converted is not None:
                try:
                    return float(converted[0]), float(converted[1])
                except Exception:
                    return None
    return None


def _run_single_slot_preflight_validation(
    state_path: Path,
    *,
    background_index: int,
    slot_index: int,
) -> dict[str, object]:
    context = _prepare_validation_context(state_path, background_index)
    if not bool(context.get("ok", False)):
        return context
    saved_entries = list(context["saved_entries"])
    dataset = dict(context["dataset"])
    group_cache = dict(context["group_cache"])
    manual_dataset_bindings = context["manual_dataset_bindings"]
    pair_result = _validate_pair(
        background_index=int(background_index),
        slot_index=int(slot_index),
        expected_pair_id=f"bg{int(background_index)}:pair{int(slot_index)}",
        saved_entries=saved_entries,
        dataset=dataset,
        group_cache=group_cache,
        entry_display_coords=manual_dataset_bindings.geometry_manual_entry_display_coords,
    )
    result = _public_context_fields(context)
    result["slot_index"] = int(slot_index)
    result["pair_result"] = pair_result
    result["ok"] = bool(pair_result.get("ok", False))
    result["classification"] = "pass" if result["ok"] else "seam_failure"
    if not result["ok"]:
        result["failed_pair"] = pair_result
    return result


def _session_refresh_summary(session: Mapping[str, object] | None) -> dict[str, object]:
    tagged_candidate = session.get("tagged_candidate") if isinstance(session, Mapping) else None
    return {
        "group_key": (
            _normalize_q_group_key(session.get("group_key"))
            if isinstance(session, Mapping)
            else None
        ),
        "target_count": session.get("target_count") if isinstance(session, Mapping) else None,
        "tagged_candidate_key": (
            session.get("tagged_candidate_key")
            if isinstance(session, Mapping)
            else None
        ),
        "tagged_candidate_identity": _identity_payload(tagged_candidate),
    }


def _compatibility_summary(result: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(result, Mapping):
        return {"ok": False, "classification": "missing"}
    failed_pair = result.get("failed_pair") if isinstance(result.get("failed_pair"), Mapping) else {}
    return {
        "ok": bool(result.get("ok", False)),
        "classification": result.get("classification"),
        "failure_stage": failed_pair.get("failure_stage"),
        "fit_resolution_kind": failed_pair.get("fit_resolution_kind"),
        "overlay_resolution_kind": failed_pair.get("overlay_resolution_kind"),
    }


def _comparable_identity_payload(
    entry: Mapping[str, object] | None,
) -> dict[str, object]:
    payload = _identity_payload(entry)
    payload.pop("pair_id", None)
    return payload


def _saved_to_selected_identity_delta(
    saved_entry: Mapping[str, object] | None,
    selected_entry: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    saved_identity = _comparable_identity_payload(saved_entry)
    selected_identity = _comparable_identity_payload(selected_entry)
    return {
        key: {
            "saved": saved_identity.get(key),
            "selected": selected_identity.get(key),
        }
        for key in saved_identity.keys()
        if saved_identity.get(key) != selected_identity.get(key)
    }


def _trusted_full_reflection_identity_payload(
    payload: Mapping[str, object] | None,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("source_reflection_namespace") == "full_reflection"
        and bool(payload.get("source_reflection_is_full", False))
        and payload.get("source_reflection_index") is not None
    )


def _legacy_untrusted_reflection_identity_payload(
    payload: Mapping[str, object] | None,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("source_reflection_index") is None
        and bool(payload.get("source_reflection_is_full", False)) is not True
    )


def _legacy_reflection_identity_alias_payload(
    saved_entry: Mapping[str, object] | None,
    payload: Mapping[str, object] | None,
    selected_entry: Mapping[str, object] | None,
) -> bool:
    if (
        not isinstance(saved_entry, Mapping)
        or not isinstance(payload, Mapping)
        or not isinstance(selected_entry, Mapping)
    ):
        return False
    reflection_index = payload.get("source_reflection_index")
    source_table_index = saved_entry.get("source_table_index")
    source_row_index = saved_entry.get("source_row_index")
    if (
        reflection_index is None
        or source_table_index is None
        or source_row_index is None
    ):
        return False
    if reflection_index != source_table_index:
        return False
    selected_table_index = selected_entry.get("source_table_index")
    selected_row_index = selected_entry.get("source_row_index")
    if selected_table_index is None or selected_row_index is None:
        return False
    return (
        selected_entry.get("source_branch_index") == saved_entry.get("source_branch_index")
        and selected_entry.get("source_peak_index") == saved_entry.get("source_peak_index")
    )


def _classify_saved_to_selected_identity_delta(
    saved_entry: Mapping[str, object] | None,
    selected_entry: Mapping[str, object] | None,
) -> tuple[dict[str, dict[str, object]], str]:
    delta = _saved_to_selected_identity_delta(saved_entry, selected_entry)
    saved_identity = _comparable_identity_payload(saved_entry)
    selected_identity = _comparable_identity_payload(selected_entry)
    if not delta:
        return delta, "saved_identity_already_canonical"
    if "hkl" in delta:
        return delta, "hkl_drift"
    if "source_branch_index" in delta or "source_peak_index" in delta:
        return delta, "branch_drift"
    if set(delta.keys()).issubset(
        {"source_reflection_index", "source_reflection_namespace", "source_reflection_is_full"}
    ) and _trusted_full_reflection_identity_payload(selected_identity) and (
        _legacy_untrusted_reflection_identity_payload(saved_identity)
        or _legacy_reflection_identity_alias_payload(
            saved_entry,
            saved_identity,
            selected_entry,
        )
    ):
        return delta, "legacy_saved_identity_canonicalized"
    if "source_reflection_namespace" in delta:
        return delta, "namespace_drift"
    if "source_reflection_is_full" in delta:
        return delta, "full_trust_drift"
    return delta, "identity_drift"


def _build_fresh_pair_state(
    *,
    saved_state: Mapping[str, object],
    background_index: int,
    emitted_pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    next_state = copy.deepcopy(dict(saved_state))
    files_state = dict(next_state.get("files", {}))
    variables_state = dict(next_state.get("variables", {}))
    geometry_state = dict(next_state.get("geometry", {}))
    background_files = [
        str(path) for path in (files_state.get("background_files", []) or ()) if str(path).strip()
    ]
    selection_text = gui_background_theta.serialize_geometry_fit_background_selection(
        selected_indices=[int(background_index)],
        total_count=len(background_files),
        current_index=int(background_index),
    )
    variables_state["geometry_fit_background_selection_var"] = str(selection_text)
    files_state["current_background_index"] = int(background_index)

    existing_bg_entry = {}
    manual_pairs = []
    replaced_target_background = False
    target_bg_entry = None
    for raw_entry in geometry_state.get("manual_pairs", []) or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        entry_background_index = int(entry.get("background_index", -1))
        if entry_background_index == int(background_index):
            existing_bg_entry = dict(entry)
            if not replaced_target_background:
                target_bg_entry = {
                    "background_index": int(background_index),
                    "background_name": existing_bg_entry.get("background_name"),
                    "background_path": existing_bg_entry.get("background_path"),
                    "entries": [dict(pair) for pair in emitted_pairs],
                }
                manual_pairs.append(target_bg_entry)
                replaced_target_background = True
            continue
        manual_pairs.append(entry)
    if not replaced_target_background:
        manual_pairs.append(
            {
                "background_index": int(background_index),
                "background_name": existing_bg_entry.get("background_name"),
                "background_path": existing_bg_entry.get("background_path"),
                "entries": [dict(pair) for pair in emitted_pairs],
            }
        )
    geometry_state["manual_pairs"] = manual_pairs
    geometry_state["peak_records"] = []
    geometry_state["q_group_rows"] = []

    next_state["files"] = files_state
    next_state["variables"] = variables_state
    next_state["geometry"] = geometry_state
    return next_state


def _build_fresh_one_pair_state(
    *,
    saved_state: Mapping[str, object],
    background_index: int,
    emitted_pair: Mapping[str, object],
) -> dict[str, object]:
    return _build_fresh_pair_state(
        saved_state=saved_state,
        background_index=int(background_index),
        emitted_pairs=[dict(emitted_pair)],
    )


def _prepare_fresh_slot_runtime(
    *,
    context: Mapping[str, object],
    background_index: int,
) -> dict[str, object]:
    projection_callbacks = context["projection_callbacks"]
    group_cache = dict(context["group_cache"])
    current_source_rows = _current_source_rows_for_background(
        background_index=int(background_index),
        context=context,
        consumer="manual_pick_group_probe",
    )
    grouped_candidates = dict(group_cache.get("grouped_candidates", {}) or {})
    grouped_candidate_source = "pick_cache"
    if not grouped_candidates and current_source_rows:
        try:
            grouped_candidates = projection_callbacks.pick_candidates(current_source_rows)
        except Exception:
            grouped_candidates = {}
        grouped_candidate_source = "current_live_source_rows"
    return {
        "projection_callbacks": projection_callbacks,
        "group_cache": group_cache,
        "current_source_rows": current_source_rows,
        "grouped_candidates": grouped_candidates,
        "grouped_candidate_source": grouped_candidate_source,
    }


def _run_fresh_slot_validation(
    *,
    context: Mapping[str, object],
    background_index: int,
    slot_index: int,
    runtime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    saved_entries = list(context["saved_entries"])
    raw_saved_entries = list(context.get("raw_saved_entries", ()) or ())
    result = {
        "slot_index": int(slot_index),
    }
    if int(slot_index) >= len(saved_entries):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "saved_entry_missing",
            "selection_status": "missing_saved_entry",
            "slot_index": int(slot_index),
        }
        return result

    prepared_runtime = (
        dict(runtime)
        if isinstance(runtime, Mapping)
        else _prepare_fresh_slot_runtime(
            context=context,
            background_index=int(background_index),
        )
    )
    projection_callbacks = prepared_runtime["projection_callbacks"]
    group_cache = dict(prepared_runtime["group_cache"])
    current_source_rows = [
        dict(entry)
        for entry in prepared_runtime.get("current_source_rows", ()) or ()
        if isinstance(entry, Mapping)
    ]
    grouped_candidates = dict(prepared_runtime.get("grouped_candidates", {}) or {})
    grouped_candidate_source = prepared_runtime.get("grouped_candidate_source")

    saved_entry = _merged_saved_entry_for_slot(
        raw_saved_entries=raw_saved_entries,
        saved_entries=saved_entries,
        slot_index=int(slot_index),
    )
    pair_id = f"bg{int(background_index)}:pair{int(slot_index)}"
    use_caked_space = bool(projection_callbacks.pick_uses_caked_space())
    saved_simulated_detector_hint = _saved_simulated_detector_hint(saved_entry)
    saved_measured_detector_point = _saved_measured_detector_point(saved_entry)
    saved_background_current_view_point = _saved_background_current_view_point(
        saved_entry,
        use_caked_space=use_caked_space,
    )
    saved_background_current_view_frame = _saved_background_current_view_frame(
        saved_entry,
        use_caked_space=use_caked_space,
    )
    selected_candidate_result = _select_live_candidate_for_saved_entry(
        saved_entry=saved_entry,
        grouped_candidates=grouped_candidates,
        saved_background_current_view_point=saved_background_current_view_point,
        saved_background_current_view_frame=saved_background_current_view_frame,
    )
    result["saved_entry"] = {
        **dict(saved_entry),
        **_identity_payload(saved_entry, pair_id=pair_id),
        "q_group_key": _normalize_q_group_key(saved_entry.get("q_group_key")),
        "saved_source_identity_fields": _source_locator_payload(saved_entry),
        "saved_measured_detector_point": _point_payload(saved_measured_detector_point),
        "saved_background_current_view_point": _point_payload(
            saved_background_current_view_point
        ),
        "saved_background_current_view_frame": saved_background_current_view_frame,
        "saved_simulated_detector_hint": _point_payload(saved_simulated_detector_hint),
    }
    result["grouped_candidate_source"] = grouped_candidate_source
    result["rebind_ok"] = False
    current_live_candidates = [
        dict(entry)
        for entry in current_source_rows
        if _normalize_hkl(entry.get("hkl")) == _normalize_hkl(saved_entry.get("hkl"))
    ]
    result["current_live_source_row_inventory"] = [
        _compact_entry(
            entry,
            detector_hint=saved_simulated_detector_hint,
            measured_detector_point=saved_measured_detector_point,
            background_point=saved_background_current_view_point,
            selection_reference_frame=saved_background_current_view_frame,
        )
        for entry in current_live_candidates
    ]
    result["candidate_selection"] = selected_candidate_result
    if not bool(selected_candidate_result.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = selected_candidate_result
        return result

    selected_candidate = dict(selected_candidate_result["selected_candidate"])
    saved_to_selected_delta, delta_classification = (
        _classify_saved_to_selected_identity_delta(
            saved_entry,
            selected_candidate,
        )
    )
    result["saved_to_selected_identity_delta"] = saved_to_selected_delta
    result["saved_to_selected_identity_delta_classification"] = delta_classification
    if delta_classification not in {
        "saved_identity_already_canonical",
        "legacy_saved_identity_canonicalized",
    }:
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "saved_to_selected_identity_alignment",
            "selection_status": "identity_drift",
            "saved_to_selected_identity_delta_classification": delta_classification,
            "saved_to_selected_identity_delta": saved_to_selected_delta,
            "saved_identity": _comparable_identity_payload(saved_entry),
            "selected_identity": _comparable_identity_payload(selected_candidate),
        }
        return result
    result["rebind_ok"] = True
    result["detector_space_diagnostic"] = _slot_detector_diagnostic(
        pair_id=pair_id,
        saved_entry=saved_entry,
        selected_candidate=selected_candidate,
        saved_background_current_view_point=saved_background_current_view_point,
        saved_background_current_view_frame=saved_background_current_view_frame,
    )

    selected_group_key = _normalize_q_group_key(selected_candidate.get("q_group_key"))
    group_entries = [
        dict(entry)
        for entry in grouped_candidates.get(selected_group_key, ())
        if isinstance(entry, Mapping)
    ]
    display_background = projection_callbacks.current_background_image()
    click_hint = _saved_background_current_view_point(
        saved_entry,
        use_caked_space=use_caked_space,
    )
    if click_hint is None:
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "fresh_emitted_pair",
            "selection_status": "missing_saved_click_hint",
        }
        return result

    saved_entry_sets: list[list[dict[str, object]]] = []
    set_sessions: list[dict[str, object]] = []
    refine_preview_point = (
        lambda candidate, raw_col, raw_row, **kwargs: gui_manual_geometry.geometry_manual_refine_preview_point(
            candidate,
            raw_col,
            raw_row,
            use_caked_space=use_caked_space,
            **kwargs,
        )
    )
    handled, next_session = gui_manual_geometry.geometry_manual_place_selection_at(
        float(click_hint[0]),
        float(click_hint[1]),
        pick_session={
            "group_key": selected_group_key,
            "group_entries": [dict(entry) for entry in group_entries],
            "pending_entries": [],
            "target_count": int(
                gui_manual_geometry.geometry_manual_group_target_count(
                    selected_group_key,
                    group_entries,
                )
            ),
            "base_entries": [],
            "q_label": repr(selected_group_key),
            "background_index": int(background_index),
            "tagged_candidate_key": gui_manual_geometry.geometry_manual_candidate_source_key(
                selected_candidate
            ),
            "tagged_candidate": dict(selected_candidate),
            "zoom_active": False,
            "zoom_center": None,
            "saved_xlim": None,
            "saved_ylim": None,
        },
        current_background_index=int(background_index),
        display_background=display_background,
        get_cache_data=lambda **_kwargs: dict(group_cache),
        refine_preview_point=refine_preview_point,
        set_pairs_for_index_fn=(
            lambda _idx, entries: saved_entry_sets.append(list(entries or []))
            or list(entries or [])
        ),
        set_pick_session_fn=lambda session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=None,
        push_undo_state_fn=lambda: None,
        use_caked_space=use_caked_space,
        caked_angles_to_background_display_coords=(
            projection_callbacks.caked_angles_to_background_display_coords
        ),
        background_display_to_native_detector_coords=(
            projection_callbacks.background_display_to_native_detector_coords
        ),
        refine_saved_pair_entry_fn=projection_callbacks.refresh_entry_geometry,
    )
    emitted_pair = None
    if isinstance(next_session, Mapping):
        pending_entries = next_session.get("pending_entries")
        if isinstance(pending_entries, list) and pending_entries:
            emitted_pair = dict(pending_entries[-1])
    if emitted_pair is None and saved_entry_sets:
        latest_entries = saved_entry_sets[-1]
        if latest_entries:
            emitted_pair = dict(latest_entries[-1])
    emitted_identity = _comparable_identity_payload(emitted_pair)
    expected_identity = _comparable_identity_payload(selected_candidate)
    fresh_emission = {
        "handled": bool(handled),
        "use_caked_space": bool(use_caked_space),
        "saved_click_hint": click_hint,
        "group_key": selected_group_key,
        "target_count": int(
            gui_manual_geometry.geometry_manual_group_target_count(
                selected_group_key,
                group_entries,
            )
        ),
        "selected_candidate": _compact_entry(
            selected_candidate,
            detector_hint=saved_simulated_detector_hint,
            measured_detector_point=saved_measured_detector_point,
            background_point=saved_background_current_view_point,
            selection_reference_frame=saved_background_current_view_frame,
        ),
        "emitted_pair": dict(emitted_pair) if isinstance(emitted_pair, Mapping) else None,
        "expected_identity": expected_identity,
        "actual_identity": emitted_identity,
        "detector_space_diagnostic": dict(result["detector_space_diagnostic"]),
    }
    result["fresh_emission"] = fresh_emission
    if (
        not handled
        or not isinstance(emitted_pair, Mapping)
        or emitted_identity != expected_identity
    ):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "fresh_emitted_pair",
            "expected_identity": expected_identity,
            "actual_identity": emitted_identity,
        }
        return result

    detector_measured, detector_pairs = gui_manual_geometry.build_geometry_manual_initial_pairs_display(
        int(background_index),
        param_set=dict(context["params"]),
        current_background_index=int(background_index),
        prefer_cache=True,
        use_caked_display=False,
        pairs_for_index=lambda _idx: [dict(emitted_pair)],
        get_cache_data=lambda **_kwargs: dict(group_cache),
        source_rows_for_background=lambda *_args, **_kwargs: [dict(entry) for entry in current_source_rows],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        entry_display_coords=lambda entry: _finite_point(entry, "x", "y"),
    )
    caked_measured, caked_pairs = gui_manual_geometry.build_geometry_manual_initial_pairs_display(
        int(background_index),
        param_set=dict(context["params"]),
        current_background_index=int(background_index),
        prefer_cache=True,
        use_caked_display=True,
        pairs_for_index=lambda _idx: [dict(emitted_pair)],
        get_cache_data=lambda **_kwargs: dict(group_cache),
        source_rows_for_background=lambda *_args, **_kwargs: [dict(entry) for entry in current_source_rows],
        simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        build_simulated_lookup=projection_callbacks.simulated_lookup,
        entry_display_coords=lambda entry: _entry_caked_display_coords(
            entry,
            projection_callbacks=projection_callbacks,
        ),
    )
    redraw_result = {
        "detector_measured": detector_measured,
        "detector_pairs": detector_pairs,
        "caked_measured": caked_measured,
        "caked_pairs": caked_pairs,
    }
    result["no_fit_redraw"] = redraw_result
    if (
        len(detector_measured) != 1
        or len(caked_measured) != 1
        or _comparable_identity_payload(detector_measured[0]) != emitted_identity
        or _comparable_identity_payload(caked_measured[0]) != emitted_identity
        or len(detector_pairs) != 1
        or len(caked_pairs) != 1
    ):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = {
            "failure_stage": "no_fit_overlay_redraw",
            "emitted_identity": emitted_identity,
        }
        return result

    refresh_source_session = (
        dict(next_session)
        if isinstance(next_session, Mapping) and next_session
        else {
            "group_key": selected_group_key,
            "group_entries": [dict(entry) for entry in group_entries],
            "pending_entries": [dict(emitted_pair)],
            "target_count": int(
                gui_manual_geometry.geometry_manual_group_target_count(
                    selected_group_key,
                    group_entries,
                )
            ),
            "background_index": int(background_index),
            "tagged_candidate_key": gui_manual_geometry.geometry_manual_candidate_source_key(
                selected_candidate
            ),
            "tagged_candidate": dict(selected_candidate),
        }
    )
    reversed_grouped_candidates = dict(grouped_candidates)
    reversed_grouped_candidates[selected_group_key] = list(reversed(group_entries))
    refreshed = gui_manual_geometry.refresh_geometry_manual_pick_session_candidates(
        refresh_source_session,
        grouped_candidates=grouped_candidates,
        cache_signature=group_cache.get("signature"),
    )
    refreshed_reversed = gui_manual_geometry.refresh_geometry_manual_pick_session_candidates(
        refresh_source_session,
        grouped_candidates=reversed_grouped_candidates,
        cache_signature=group_cache.get("signature"),
    )
    refresh_result = {
        "forward": _session_refresh_summary(refreshed),
        "reversed": _session_refresh_summary(refreshed_reversed),
    }
    result["session_refresh"] = refresh_result
    selected_candidate_key = gui_manual_geometry.geometry_manual_candidate_source_key(
        selected_candidate
    )
    for refreshed_session in (refreshed, refreshed_reversed):
        tagged_candidate = refreshed_session.get("tagged_candidate")
        tagged_identity = _comparable_identity_payload(tagged_candidate)
        if (
            refreshed_session.get("group_key") != selected_group_key
            or refreshed_session.get("tagged_candidate_key") != selected_candidate_key
            or tagged_identity != expected_identity
        ):
            result["ok"] = False
            result["classification"] = "seam_failure"
            result["failed_pair"] = {
                "failure_stage": "session_refresh_rebind",
                "expected_identity": expected_identity,
                "actual_identity": tagged_identity,
            }
            return result

    fresh_state = _build_fresh_pair_state(
        saved_state=context["saved_state"],
        background_index=int(background_index),
        emitted_pairs=[dict(emitted_pair)],
    )
    with tempfile.TemporaryDirectory(prefix="geometry_fresh_probe_") as tmp_dir:
        temp_state_path = Path(tmp_dir) / "fresh_one_pair_state.json"
        temp_save_error = _try_save_gui_state_file(
            temp_state_path,
            fresh_state,
            failure_stage="fresh_one_pair_temp_state_save",
        )
        if temp_save_error is None:
            fresh_preflight = _run_single_slot_preflight_validation(
                temp_state_path,
                background_index=int(background_index),
                slot_index=0,
            )
        else:
            fresh_preflight = {
                "ok": False,
                "classification": "temp_state_save_failed",
                "failed_pair": temp_save_error,
            }
    result["fresh_one_pair_preflight"] = fresh_preflight
    if not bool(fresh_preflight.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_pair"] = fresh_preflight.get("failed_pair", fresh_preflight)
        return result

    result["emitted_pair"] = dict(emitted_pair)
    result["ok"] = True
    result["classification"] = "pass"
    return result


def _run_fresh_contract_validation(
    state_path: Path,
    *,
    background_index: int,
    sentinel_slot_index: int = 1,
    export_fresh_state_path: Path | None = None,
) -> dict[str, object]:
    context = _prepare_validation_context(state_path, background_index)
    if not bool(context.get("ok", False)):
        return context

    result = _public_context_fields(context)
    result["sentinel_slot_index"] = int(sentinel_slot_index)
    slot_failed_pair = None
    skipped_compatibility = {
        "ok": False,
        "classification": "skipped_due_to_slot_failure",
    }
    result["fresh_state_export_written"] = False
    result["fresh_export_ok"] = bool(export_fresh_state_path is None)
    result["compatibility_ok"] = False
    result["exported_state_compatibility"] = skipped_compatibility
    result["state_compatibility"] = _compatibility_summary(skipped_compatibility)
    temp_save_error = None
    runtime = _prepare_fresh_slot_runtime(
        context=context,
        background_index=int(background_index),
    )
    slot_result = _run_fresh_slot_validation(
        context=context,
        background_index=int(background_index),
        slot_index=int(sentinel_slot_index),
        runtime=runtime,
    )
    result.update(slot_result)
    if isinstance(slot_result.get("failed_pair"), Mapping):
        slot_failed_pair = dict(slot_result["failed_pair"])
        skipped_compatibility["failed_pair"] = slot_failed_pair
        result["exported_state_compatibility"] = skipped_compatibility
        result["state_compatibility"] = _compatibility_summary(skipped_compatibility)
    if not bool(slot_result.get("ok", False)):
        result["ok"] = False
        return result

    emitted_pair = slot_result.get("emitted_pair")
    fresh_state = _build_fresh_pair_state(
        saved_state=context["saved_state"],
        background_index=int(background_index),
        emitted_pairs=[dict(emitted_pair)] if isinstance(emitted_pair, Mapping) else [],
    )
    with tempfile.TemporaryDirectory(prefix="geometry_fresh_probe_") as tmp_dir:
        temp_state_path = Path(tmp_dir) / "fresh_state.json"
        temp_save_error = _try_save_gui_state_file(
            temp_state_path,
            fresh_state,
            failure_stage="fresh_contract_temp_state_save",
        )
        if temp_save_error is None:
            compatibility = _run_saved_state_compatibility_validation(
                temp_state_path,
                int(background_index),
            )
        else:
            compatibility = {
                "ok": False,
                "classification": "temp_state_save_failed",
                "failed_pair": temp_save_error,
            }

    compatibility_ok = bool(compatibility.get("ok", False))
    if compatibility_ok and export_fresh_state_path is not None:
        export_path = export_fresh_state_path.expanduser().resolve()
        export_save_error = _try_save_gui_state_file(
            export_path,
            fresh_state,
            failure_stage="fresh_contract_export",
            create_parent=True,
            remove_partial_target=True,
        )
        if export_save_error is None:
            result["exported_fresh_state_path"] = str(export_path)
            result["fresh_state_export_written"] = bool(export_path.is_file())
        else:
            export_save_error.setdefault("compatibility_ok", compatibility_ok)
            export_save_error.setdefault("fresh_export_ok", False)
            result["failed_pair"] = export_save_error
    fresh_export_ok = bool(
        export_fresh_state_path is None or result.get("fresh_state_export_written", False)
    )
    result["fresh_export_ok"] = fresh_export_ok
    result["compatibility_ok"] = compatibility_ok
    result["exported_state_compatibility"] = compatibility
    result["state_compatibility"] = _compatibility_summary(compatibility)
    result["ok"] = bool(compatibility_ok and fresh_export_ok)
    if temp_save_error is not None:
        result["classification"] = "fresh_contract_temp_save_fail"
        result["failed_pair"] = temp_save_error
    elif not compatibility_ok:
        result["classification"] = "fresh_contract_state_compatibility_fail"
        result["failed_pair"] = compatibility.get("failed_pair", compatibility)
    elif not fresh_export_ok:
        result["classification"] = "fresh_contract_export_fail"
        failed_pair = (
            dict(result["failed_pair"])
            if isinstance(result.get("failed_pair"), Mapping)
            else {}
        )
        failed_pair.setdefault("failure_stage", "fresh_contract_export")
        failed_pair.setdefault("compatibility_ok", compatibility_ok)
        failed_pair.setdefault("fresh_export_ok", fresh_export_ok)
        result["failed_pair"] = failed_pair
    else:
        result["classification"] = "pass"
    return result


def _run_fresh_all_contract_validation(
    state_path: Path,
    *,
    background_index: int,
    export_fresh_state_path: Path | None = None,
) -> dict[str, object]:
    context = _prepare_validation_context(state_path, background_index)
    if not bool(context.get("ok", False)):
        return context

    saved_entries = list(context["saved_entries"])
    raw_saved_entries = [
        dict(entry)
        for entry in context.get("raw_saved_entries", ()) or ()
        if isinstance(entry, Mapping)
    ]
    result = _public_context_fields(context)
    runtime = _prepare_fresh_slot_runtime(
        context=context,
        background_index=int(background_index),
    )
    slot_results: list[dict[str, object]] = []
    emitted_pairs: list[dict[str, object]] = []
    failed_slot_index: int | None = None
    failed_pair: Mapping[str, object] | None = None
    for slot_index in range(len(saved_entries)):
        slot_result = _run_fresh_slot_validation(
            context=context,
            background_index=int(background_index),
            slot_index=int(slot_index),
            runtime=runtime,
        )
        slot_results.append(slot_result)
        emitted_pair = slot_result.get("emitted_pair")
        if isinstance(emitted_pair, Mapping):
            emitted_pairs.append(dict(emitted_pair))
        if failed_slot_index is None and not bool(slot_result.get("ok", False)):
            failed_slot_index = int(slot_index)
            failed_pair = (
                slot_result.get("failed_pair")
                if isinstance(slot_result.get("failed_pair"), Mapping)
                else slot_result
            )

    result["slot_results"] = slot_results
    processed_manual_entry_count = int(len(saved_entries))
    bound_manual_entry_count = int(
        sum(1 for slot_result in slot_results if bool(slot_result.get("rebind_ok", False)))
    )
    missing_manual_entry_count = int(
        max(0, processed_manual_entry_count - bound_manual_entry_count)
    )
    background_distances_px = [
        _float_or_none(
            dict(slot_result.get("candidate_selection", {})).get("background_distance_px")
        )
        if _float_or_none(
            dict(slot_result.get("candidate_selection", {})).get("background_distance_px")
        )
        is not None
        else _float_or_none(
            dict(slot_result.get("candidate_selection", {})).get("best_distance_px")
        )
        for slot_result in slot_results
    ]
    finite_background_distances_px = [
        float(distance)
        for distance in background_distances_px
        if distance is not None and np.isfinite(distance)
    ]
    merged_saved_entries = [
        _merged_saved_entry_for_slot(
            raw_saved_entries=raw_saved_entries,
            saved_entries=saved_entries,
            slot_index=slot_index,
        )
        for slot_index in range(len(saved_entries))
    ]
    background_distance_gate_threshold_px = _background_distance_threshold_px(
        merged_saved_entries
    )
    max_background_distance_px = (
        float(max(finite_background_distances_px))
        if finite_background_distances_px
        else None
    )
    branch_mismatch_count = 0
    for slot_result in slot_results:
        saved_entry = (
            dict(slot_result.get("saved_entry", {}))
            if isinstance(slot_result.get("saved_entry"), Mapping)
            else {}
        )
        selected_candidate = (
            dict(dict(slot_result.get("candidate_selection", {})).get("selected_candidate", {}))
            if isinstance(dict(slot_result.get("candidate_selection", {})).get("selected_candidate"), Mapping)
            else {}
        )
        saved_branch_index = _normalized_branch_index(saved_entry)
        selected_branch_index = _normalized_branch_index(selected_candidate)
        if (
            saved_branch_index is not None
            and selected_branch_index is not None
            and int(saved_branch_index) != int(selected_branch_index)
        ):
            branch_mismatch_count += 1
    runtime_prepare_ok = bool(context.get("runtime_prepare_ok", False))
    background_distances_all_finite = bool(
        processed_manual_entry_count > 0
        and len(background_distances_px) == processed_manual_entry_count
        and len(finite_background_distances_px) == processed_manual_entry_count
    )
    background_distance_gate_ok = bool(
        background_distances_all_finite
        and max_background_distance_px is not None
        and float(max_background_distance_px)
        <= float(background_distance_gate_threshold_px)
    )
    isolated_rebind_ok = bool(
        processed_manual_entry_count > 0
        and bound_manual_entry_count == processed_manual_entry_count
        and missing_manual_entry_count == 0
        and int(branch_mismatch_count) == 0
    )
    result["processed_manual_entry_count"] = processed_manual_entry_count
    result["bound_manual_entry_count"] = bound_manual_entry_count
    result["missing_manual_entry_count"] = missing_manual_entry_count
    result["branch_mismatch_count"] = int(branch_mismatch_count)
    result["background_distance_px"] = list(background_distances_px)
    result["background_distances_all_finite"] = background_distances_all_finite
    result["background_distance_gate_threshold_px"] = float(
        background_distance_gate_threshold_px
    )
    result["max_background_distance_px"] = max_background_distance_px
    result["background_distance_gate_ok"] = background_distance_gate_ok
    result["deprecated_aliases_present"] = True
    result["candidate_distance_px"] = list(result["background_distance_px"])
    result["candidate_distances_all_finite"] = bool(
        result["background_distances_all_finite"]
    )
    result["candidate_distance_gate_threshold_px"] = float(
        result["background_distance_gate_threshold_px"]
    )
    result["max_candidate_distance_px"] = result["max_background_distance_px"]
    result["detector_distance_gate_ok"] = bool(result["background_distance_gate_ok"])
    result["candidate_distance_gate_ok"] = bool(result["background_distance_gate_ok"])
    detector_distance_gate_ok = bool(result["detector_distance_gate_ok"])
    candidate_distances_all_finite = bool(result["candidate_distances_all_finite"])
    max_candidate_distance_px = result["max_candidate_distance_px"]
    candidate_distance_gate_threshold_px = float(
        result["candidate_distance_gate_threshold_px"]
    )
    result["runtime_prepare_ok"] = runtime_prepare_ok
    result["isolated_rebind_ok"] = isolated_rebind_ok
    result["fresh_state_export_written"] = False
    result["fresh_export_ok"] = bool(export_fresh_state_path is None)
    result["compatibility_ok"] = False
    result["exported_state_compatibility"] = {
        "ok": False,
        "classification": "missing",
    }
    result["state_compatibility"] = _compatibility_summary(None)
    temp_save_error = None

    if failed_slot_index is None:
        fresh_state = _build_fresh_pair_state(
            saved_state=context["saved_state"],
            background_index=int(background_index),
            emitted_pairs=emitted_pairs,
        )
        with tempfile.TemporaryDirectory(prefix="geometry_fresh_all_probe_") as tmp_dir:
            temp_state_path = Path(tmp_dir) / "fresh_all_state.json"
            temp_save_error = _try_save_gui_state_file(
                temp_state_path,
                fresh_state,
                failure_stage="fresh_all_contract_temp_state_save",
            )
            if temp_save_error is None:
                compatibility = _run_saved_state_compatibility_validation(
                    temp_state_path,
                    int(background_index),
                )
            else:
                compatibility = {
                    "ok": False,
                    "classification": "temp_state_save_failed",
                    "failed_pair": temp_save_error,
                }
    else:
        fresh_state = None
        compatibility = {
            "ok": False,
            "classification": "skipped_due_to_slot_failure",
            "failed_pair": failed_pair,
        }
        result["failed_slot_index"] = int(failed_slot_index)

    compatibility_ok = bool(compatibility.get("ok", False))
    provisional_ok = bool(
        runtime_prepare_ok
        and isolated_rebind_ok
        and compatibility_ok
        and background_distance_gate_ok
    )
    if provisional_ok and export_fresh_state_path is not None and fresh_state is not None:
        export_path = export_fresh_state_path.expanduser().resolve()
        export_save_error = _try_save_gui_state_file(
            export_path,
            fresh_state,
            failure_stage="fresh_all_contract_export",
            create_parent=True,
            remove_partial_target=True,
        )
        if export_save_error is None:
            result["exported_fresh_state_path"] = str(export_path)
            result["fresh_state_export_written"] = bool(export_path.is_file())
        else:
            export_save_error.setdefault("compatibility_ok", compatibility_ok)
            export_save_error.setdefault("fresh_export_ok", False)
            result["failed_pair"] = export_save_error
    fresh_export_ok = bool(
        export_fresh_state_path is None or result.get("fresh_state_export_written", False)
    )
    result["fresh_export_ok"] = fresh_export_ok
    result["compatibility_ok"] = compatibility_ok
    result["exported_state_compatibility"] = compatibility
    result["state_compatibility"] = _compatibility_summary(compatibility)
    result["ok"] = bool(provisional_ok and fresh_export_ok)
    if temp_save_error is not None:
        result["classification"] = "fresh_all_contract_temp_save_fail"
        result["failed_pair"] = temp_save_error
    elif not fresh_export_ok and provisional_ok:
        result["classification"] = "fresh_all_contract_export_fail"
    elif not runtime_prepare_ok and isolated_rebind_ok:
        result["classification"] = "runtime_prepare_failed_but_isolated_rebind_ok"
    else:
        result["classification"] = "pass" if result["ok"] else "seam_failure"
    if not result["ok"] and not isinstance(result.get("failed_pair"), Mapping):
        result["failed_pair"] = (
            failed_pair
            if failed_slot_index is not None and isinstance(failed_pair, Mapping)
            else compatibility.get("failed_pair", compatibility)
            if not compatibility_ok
            else {
                "failure_stage": "fresh_all_summary_gate",
                "runtime_prepare_ok": runtime_prepare_ok,
                "isolated_rebind_ok": isolated_rebind_ok,
                "fresh_export_ok": fresh_export_ok,
                "compatibility_ok": compatibility_ok,
                "detector_distance_gate_ok": detector_distance_gate_ok,
                "processed_manual_entry_count": processed_manual_entry_count,
                "bound_manual_entry_count": bound_manual_entry_count,
                "missing_manual_entry_count": missing_manual_entry_count,
                "branch_mismatch_count": int(branch_mismatch_count),
                "candidate_distances_all_finite": candidate_distances_all_finite,
                "max_candidate_distance_px": max_candidate_distance_px,
                "candidate_distance_gate_threshold_px": float(
                    candidate_distance_gate_threshold_px
                ),
            }
        )
    return result


def _run_downstream_identity_validation(
    state_path: Path,
    *,
    background_index: int,
) -> dict[str, object]:
    payload = load_gui_state_file(state_path)
    saved_state = dict(payload["state"])
    saved_entries = _saved_entries_for_background(
        saved_state,
        background_index=int(background_index),
    )
    result = {
        "state_path": str(state_path),
        "background_index": int(background_index),
        "saved_pair_count": int(len(saved_entries)),
        "identity_fields_checked": list(CANONICAL_IDENTITY_FIELDS),
        "stage_order": list(DOWNSTREAM_IDENTITY_STAGE_ORDER),
        "stage_results": [],
        "final_metric_name": None,
    }

    input_contract = _validate_downstream_identity_input(
        saved_state,
        background_index=int(background_index),
    )
    result["stage_results"].append(input_contract)
    if not bool(input_contract.get("ok", False)):
        result["ok"] = False
        result["classification"] = "invalid_downstream_identity_input"
        result["failed_stage"] = "input_contract"
        result["failed_pair"] = input_contract.get("failed_pair", input_contract)
        return result

    captured = _capture_execution_setup(
        saved_state=saved_state,
        state_path=state_path,
    )
    prepare_result = captured["prepare_result"]
    prepared_run = getattr(prepare_result, "prepared_run", None)
    if prepared_run is None:
        stage_result = _failed_stage_result(
            "preflight_normalized_pairs",
            failure_reason="prepare_runtime_geometry_fit_run_failed",
            error_text=str(
                getattr(prepare_result, "error_text", None)
                or captured.get("execution_error_text")
                or "Geometry fit preflight failed."
            ),
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "preflight_normalized_pairs"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result

    prepare_kwargs = (
        dict(captured.get("prepare_kwargs", {}))
        if isinstance(captured.get("prepare_kwargs"), Mapping)
        else {}
    )
    bindings = prepare_kwargs.get("bindings")
    params = (
        dict(prepare_kwargs.get("params", {}) or {})
        if isinstance(prepare_kwargs.get("params", {}) or {}, Mapping)
        else {}
    )
    downstream_runtime_cfg = prepared_run.geometry_runtime_cfg
    if bindings is not None and callable(getattr(bindings, "build_runtime_config", None)):
        try:
            downstream_runtime_cfg = gui_geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
                bindings.build_runtime_config(dict(params or {})),
                joint_background_mode=bool(
                    getattr(prepared_run, "joint_background_mode", False)
                ),
            )
        except Exception:
            downstream_runtime_cfg = prepared_run.geometry_runtime_cfg
    downstream_prepared_run = replace(
        prepared_run,
        geometry_runtime_cfg=(
            dict(downstream_runtime_cfg)
            if isinstance(downstream_runtime_cfg, Mapping)
            else downstream_runtime_cfg
        ),
    )

    current_dataset = (
        dict(downstream_prepared_run.current_dataset)
        if isinstance(downstream_prepared_run.current_dataset, Mapping)
        else {}
    )
    current_dataset_index = _entry_dataset_index(current_dataset, default=0)
    preflight_entries = [
        dict(entry)
        for entry in current_dataset.get("measured_for_fit", ()) or ()
        if isinstance(entry, Mapping)
    ]
    preflight_stage = _ordered_stage_result(
        "preflight_normalized_pairs",
        expected_entries=saved_entries,
        actual_entries=preflight_entries,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(preflight_stage)
    if not bool(preflight_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "preflight_normalized_pairs"
        result["failed_pair"] = preflight_stage.get("failed_pair", preflight_stage)
        return result

    execute_kwargs = (
        dict(captured.get("execute_kwargs", {}))
        if isinstance(captured.get("execute_kwargs"), Mapping)
        else {}
    )
    setup = execute_kwargs.get("setup")
    postprocess_config = getattr(setup, "postprocess_config", None)
    solver_inputs = getattr(postprocess_config, "solver_inputs", None)
    var_names = list(
        execute_kwargs.get(
            "var_names",
            (
                prepare_kwargs.get("var_names", ())
                if isinstance(prepare_kwargs, Mapping)
                else ()
            ),
        )
        or ()
    )
    if solver_inputs is None:
        stage_result = _failed_stage_result(
            "solver_request_measured_peaks",
            failure_reason="missing_execution_setup",
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "solver_request_measured_peaks"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result

    try:
        request = gui_geometry_fit.build_geometry_fit_solver_request(
            prepared_run=downstream_prepared_run,
            var_names=var_names,
            solver_inputs=solver_inputs,
        )
    except Exception as exc:
        stage_result = _failed_stage_result(
            "solver_request_measured_peaks",
            failure_reason="solver_request_build_failed",
            error_text=str(exc),
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "solver_request_measured_peaks"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result
    solver_entries = [
        dict(entry)
        for entry in request.measured_peaks or ()
        if isinstance(entry, Mapping)
    ]
    solver_stage = _ordered_stage_result(
        "solver_request_measured_peaks",
        expected_entries=preflight_entries,
        actual_entries=solver_entries,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(solver_stage)
    if not bool(solver_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "solver_request_measured_peaks"
        result["failed_pair"] = solver_stage.get("failed_pair", solver_stage)
        return result

    try:
        dataset_contexts = opt._build_geometry_fit_dataset_contexts(
            request.miller,
            request.intensities,
            request.params,
            request.measured_peaks,
            None,
            request.dataset_specs,
        )
    except Exception as exc:
        stage_result = _failed_stage_result(
            "subset_measured_entries",
            failure_reason="subset_context_build_failed",
            error_text=str(exc),
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "subset_measured_entries"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result
    if current_dataset_index >= len(dataset_contexts):
        stage_result = _failed_stage_result(
            "subset_measured_entries",
            failure_reason="missing_dataset_context",
            dataset_index=int(current_dataset_index),
            dataset_context_count=int(len(dataset_contexts)),
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "subset_measured_entries"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result
    subset_entries = [
        dict(entry)
        for entry in dataset_contexts[int(current_dataset_index)].subset.measured_entries
        if isinstance(entry, Mapping)
    ]
    subset_stage = _ordered_stage_result(
        "subset_measured_entries",
        expected_entries=solver_entries,
        actual_entries=subset_entries,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(subset_stage)
    if not bool(subset_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "subset_measured_entries"
        result["failed_pair"] = subset_stage.get("failed_pair", subset_stage)
        return result

    try:
        solve_result = gui_geometry_fit.solve_geometry_fit_request(
            request,
            solve_fit=opt.fit_geometry_parameters,
        )
    except Exception as exc:
        stage_result = _failed_stage_result(
            "seed_correspondence_records",
            failure_reason="solve_geometry_fit_request_failed",
            error_text=str(exc),
        )
        result["stage_results"].append(stage_result)
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "seed_correspondence_records"
        result["failed_pair"] = stage_result.get("failed_pair", stage_result)
        return result

    full_beam_summary = (
        getattr(solve_result, "full_beam_polish_summary", None)
        if isinstance(getattr(solve_result, "full_beam_polish_summary", None), Mapping)
        else {}
    )
    result["final_metric_name"] = str(
        getattr(solve_result, "final_metric_name", "") or ""
    )
    seed_entries = _filter_stage_entries_for_dataset(
        full_beam_summary.get("seed_correspondence_records", ()),
        dataset_index=int(current_dataset_index),
    )
    seed_stage = _ordered_stage_result(
        "seed_correspondence_records",
        expected_entries=subset_entries,
        actual_entries=seed_entries,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(seed_stage)
    if not bool(seed_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "seed_correspondence_records"
        result["failed_pair"] = seed_stage.get("failed_pair", seed_stage)
        return result

    full_beam_entries = _filter_stage_entries_for_dataset(
        _full_beam_point_match_entries(solve_result),
        dataset_index=int(current_dataset_index),
    )
    coverage_stage = _coverage_stage_result(
        "full_beam_identity_coverage",
        expected_entries=seed_entries,
        actual_entries=full_beam_entries,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(coverage_stage)
    if not bool(coverage_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = str(
            coverage_stage.get("coverage_mismatch_classification", "seam_failure")
        )
        result["failed_stage"] = "full_beam_identity_coverage"
        result["failed_pair"] = coverage_stage.get("failed_pair", coverage_stage)
        if isinstance(full_beam_summary, Mapping):
            result["full_beam_polish_summary"] = copy.deepcopy(dict(full_beam_summary))
            result["full_beam_point_match_comparison"] = _build_full_beam_rejection_comparison(
                full_beam_summary,
                dataset_index=int(current_dataset_index),
            )
        return result
    fixed_stage = _fixed_correspondence_stage_result(
        "full_beam_fixed_correspondence",
        expected_entries=seed_entries,
        actual_entries=full_beam_entries,
        final_metric_name=result["final_metric_name"],
        full_beam_polish_summary=full_beam_summary,
        default_dataset_index=int(current_dataset_index),
    )
    result["stage_results"].append(fixed_stage)
    if not bool(fixed_stage.get("ok", False)):
        result["ok"] = False
        result["classification"] = "seam_failure"
        result["failed_stage"] = "full_beam_fixed_correspondence"
        result["failed_pair"] = fixed_stage.get("failed_pair", fixed_stage)
        if isinstance(fixed_stage.get("full_beam_polish_summary"), Mapping):
            result["full_beam_polish_summary"] = copy.deepcopy(
                dict(fixed_stage["full_beam_polish_summary"])
            )
        if isinstance(fixed_stage.get("full_beam_point_match_comparison"), Mapping):
            result["full_beam_point_match_comparison"] = copy.deepcopy(
                dict(fixed_stage["full_beam_point_match_comparison"])
            )
        return result

    result["ok"] = True
    result["classification"] = "pass"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate fresh grouped-pick emission and saved-state preflight rebinding.",
    )
    parser.add_argument("--state", required=True, help="Path to the saved GUI state.")
    parser.add_argument(
        "--background-index",
        type=int,
        default=0,
        help="Background index to validate. Default: 0.",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "fresh", "fresh-all", "compatibility", "downstream-identity"),
        default="full",
        help="Validation mode. Default: full (aliases fresh-all milestone-6 gate).",
    )
    parser.add_argument(
        "--sentinel-slot-index",
        type=int,
        default=1,
        help="Saved entry slot to use for fresh emission validation. Default: 1.",
    )
    parser.add_argument(
        "--export-fresh-state",
        help="Optional path to write the validated fresh one-pair state.",
    )
    args = parser.parse_args()

    resolved_state_path = Path(args.state).expanduser().resolve()
    export_fresh_state_path = (
        Path(args.export_fresh_state) if args.export_fresh_state else None
    )
    requested_mode = str(args.mode)
    effective_mode = "fresh-all" if requested_mode == "full" else requested_mode
    if (
        export_fresh_state_path is not None
        and effective_mode not in {"fresh", "fresh-all"}
    ):
        parser.error(
            "--export-fresh-state requires --mode fresh, --mode fresh-all, or --mode full."
        )
    if effective_mode == "fresh":
        result = _run_fresh_contract_validation(
            resolved_state_path,
            background_index=int(args.background_index),
            sentinel_slot_index=int(args.sentinel_slot_index),
            export_fresh_state_path=export_fresh_state_path,
        )
    elif effective_mode == "fresh-all":
        result = _run_fresh_all_contract_validation(
            resolved_state_path,
            background_index=int(args.background_index),
            export_fresh_state_path=export_fresh_state_path,
        )
    elif effective_mode == "downstream-identity":
        result = _run_downstream_identity_validation(
            resolved_state_path,
            background_index=int(args.background_index),
        )
    elif effective_mode == "compatibility":
        result = _run_saved_state_compatibility_validation(
            resolved_state_path,
            int(args.background_index),
        )
    else:
        result = _run_fresh_contract_validation(
            resolved_state_path,
            background_index=int(args.background_index),
            sentinel_slot_index=int(args.sentinel_slot_index),
            export_fresh_state_path=export_fresh_state_path,
        )
    result = _finalize_cli_result(
        result,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if bool(result.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
