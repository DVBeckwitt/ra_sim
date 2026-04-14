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

from ra_sim import headless_geometry_fit as hgf
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.io.data_loading import load_gui_state_file, save_gui_state_file


class _CapturedPreflight(RuntimeError):
    """Internal stop used to abort headless execution after preflight capture."""


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
        payload["mode_note"] = "full now aliases fresh-all milestone-6 gate"
    return payload


def _stable_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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


def _saved_click_hint(
    entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> tuple[float, float] | None:
    key_sets = (
        (("raw_caked_x", "raw_caked_y"), ("caked_x", "caked_y"))
        if bool(use_caked_space)
        else (("raw_x", "raw_y"), ("x", "y"))
    )
    for keys in key_sets:
        hint = _finite_point(entry, *keys)
        if hint is not None:
            return hint
    return None


def _candidate_detector_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for keys in (("sim_col_raw", "sim_row_raw"), ("sim_col", "sim_row")):
        point = _finite_point(entry, *keys)
        if point is not None:
            return point
    return None


def _candidate_detector_display_distance(
    detector_hint: tuple[float, float] | None,
    candidate: Mapping[str, object] | None,
) -> float:
    if detector_hint is None:
        return float("nan")
    candidate_point = _candidate_detector_display_point(candidate)
    if candidate_point is None:
        return float("nan")
    return float(
        np.hypot(
            float(candidate_point[0]) - float(detector_hint[0]),
            float(candidate_point[1]) - float(detector_hint[1]),
        )
    )


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
    detector_display_hint: tuple[float, float] | None = None,
) -> dict[str, object] | None:
    if not isinstance(entry, Mapping):
        return None
    detector_display = _candidate_detector_display_point(entry)
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
        "sim_col": entry.get("sim_col"),
        "sim_row": entry.get("sim_row"),
        "sim_col_raw": entry.get("sim_col_raw"),
        "sim_row_raw": entry.get("sim_row_raw"),
        "sim_detector_display": list(detector_display) if detector_display is not None else None,
        "caked_x": entry.get("caked_x"),
        "caked_y": entry.get("caked_y"),
    }
    if display_hint is not None:
        payload["distance_to_saved_display_hint_px"] = (
            gui_manual_geometry.geometry_manual_candidate_distance_to_point(
                float(display_hint[0]),
                float(display_hint[1]),
                dict(entry),
            )
        )
    if detector_display_hint is not None:
        payload["distance_to_saved_sim_hint_px"] = _candidate_detector_display_distance(
            detector_display_hint,
            entry,
        )
    return payload


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
    tie_tolerance_px: float = gui_geometry_fit.GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX,
) -> dict[str, object]:
    detector_hint = _saved_simulated_display_hint(saved_entry)
    target_hkl = _normalize_hkl(saved_entry.get("hkl"))
    target_branch = _normalized_branch_index(saved_entry)
    inventory = _group_inventory(grouped_candidates)
    matching_candidates: list[dict[str, object]] = []
    if isinstance(grouped_candidates, Mapping):
        for group_key, raw_entries in grouped_candidates.items():
            normalized_group_key = _normalize_q_group_key(group_key)
            for raw_entry in raw_entries or ():
                if not isinstance(raw_entry, Mapping):
                    continue
                entry = dict(raw_entry)
                if target_hkl is not None and _normalize_hkl(entry.get("hkl")) != target_hkl:
                    continue
                if target_branch is not None and _normalized_branch_index(entry) != target_branch:
                    continue
                entry["q_group_key"] = normalized_group_key
                matching_candidates.append(entry)

    candidate_inventory = [
        _compact_entry(entry, detector_display_hint=detector_hint)
        for entry in matching_candidates
    ]
    finite_candidates = [
        (entry, _candidate_detector_display_distance(detector_hint, entry))
        for entry in matching_candidates
        if np.isfinite(_candidate_detector_display_distance(detector_hint, entry))
    ]
    if not finite_candidates:
        return {
            "ok": False,
            "failure_stage": "grouped_candidate_regeneration",
            "selection_status": "missing_live_candidate",
            "saved_simulated_display_hint": detector_hint,
            "saved_target_hkl": target_hkl,
            "saved_target_branch_index": target_branch,
            "group_inventory": inventory,
            "matching_branch_candidate_inventory": candidate_inventory,
        }

    finite_candidates.sort(key=lambda item: item[1])
    best_entry, best_distance = finite_candidates[0]
    tied_entries = [
        dict(entry)
        for entry, distance in finite_candidates
        if abs(float(distance) - float(best_distance)) <= float(tie_tolerance_px)
    ]
    if len(tied_entries) > 1:
        return {
            "ok": False,
            "failure_stage": "resolved_live_row_selection",
            "selection_status": "ambiguous_live_row_selection",
            "saved_simulated_display_hint": detector_hint,
            "saved_target_hkl": target_hkl,
            "saved_target_branch_index": target_branch,
            "tie_tolerance_px": float(tie_tolerance_px),
            "best_distance_px": float(best_distance),
            "group_inventory": inventory,
            "matching_branch_candidate_inventory": candidate_inventory,
            "tied_candidate_inventory": [
                _compact_entry(entry, detector_display_hint=detector_hint)
                for entry in tied_entries
            ],
        }

    return {
        "ok": True,
        "selection_status": "selected",
        "saved_simulated_display_hint": detector_hint,
        "saved_target_hkl": target_hkl,
        "saved_target_branch_index": target_branch,
        "best_distance_px": float(best_distance),
        "selected_candidate": dict(best_entry),
        "group_key": _normalize_q_group_key(best_entry.get("q_group_key")),
        "group_inventory": inventory,
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
        cache_signature_fn=lambda **kwargs: (
            gui_manual_geometry.geometry_manual_pick_cache_signature(
                source_snapshot_signature=requested_signature,
                background_index=int(kwargs["background_index"]),
                background_image=kwargs["background_image"],
                use_caked_space=False,
                geometry_preview_excluded_q_groups=(),
                geometry_q_group_cached_entries=(),
            )
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
    return {
        "valid": bool(
            expected_background_path == actual_background_path
            and requested_signature is not None
            and isinstance(cache_signature, tuple)
            and len(cache_signature) > 0
            and cache_signature[0] == requested_signature
        ),
        "background_index": int(background_index),
        "expected_background_path": expected_background_path,
        "actual_background_path": actual_background_path,
        "param_digest": _stable_digest(dict(params)),
        "requested_signature_digest": _stable_digest(requested_signature),
        "group_cache_signature_digest": _stable_digest(cache_signature),
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
    result: dict[str, object] = {
        "state_path": str(state_path),
        "background_index": int(background_index),
        "saved_pair_count": int(len(saved_entries)),
        "captured_preflight_error_text": prepare_result.error_text,
        "used_isolated_background_dataset": prepared_run is None,
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


def _classify_saved_to_selected_identity_delta(
    saved_entry: Mapping[str, object] | None,
    selected_entry: Mapping[str, object] | None,
) -> tuple[dict[str, dict[str, object]], str]:
    delta = _saved_to_selected_identity_delta(saved_entry, selected_entry)
    if not delta:
        return delta, "saved_identity_already_canonical"
    if "hkl" in delta:
        return delta, "hkl_drift"
    if "source_reflection_namespace" in delta:
        return delta, "namespace_drift"
    if "source_reflection_is_full" in delta:
        return delta, "full_trust_drift"
    if "source_branch_index" in delta or "source_peak_index" in delta:
        return delta, "branch_drift"
    if set(delta.keys()) == {"source_reflection_index"}:
        saved_identity = _comparable_identity_payload(saved_entry)
        selected_identity = _comparable_identity_payload(selected_entry)
        if _trusted_full_reflection_identity_payload(
            saved_identity
        ) and _trusted_full_reflection_identity_payload(selected_identity):
            return delta, "legacy_saved_identity_canonicalized"
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

    saved_entry = dict(saved_entries[int(slot_index)])
    selected_candidate_result = _select_live_candidate_for_saved_entry(
        saved_entry=saved_entry,
        grouped_candidates=grouped_candidates,
    )
    result["saved_entry"] = {
        **_identity_payload(
            saved_entry,
            pair_id=f"bg{int(background_index)}:pair{int(slot_index)}",
        ),
        "q_group_key": _normalize_q_group_key(saved_entry.get("q_group_key")),
        "saved_simulated_display_hint": _saved_simulated_display_hint(saved_entry),
    }
    result["grouped_candidate_source"] = grouped_candidate_source
    result["current_live_source_row_inventory"] = [
        _compact_entry(
            entry,
            detector_display_hint=_saved_simulated_display_hint(saved_entry),
        )
        for entry in current_source_rows
        if _normalize_hkl(entry.get("hkl")) == _normalize_hkl(saved_entry.get("hkl"))
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

    selected_group_key = _normalize_q_group_key(selected_candidate.get("q_group_key"))
    group_entries = [
        dict(entry)
        for entry in grouped_candidates.get(selected_group_key, ())
        if isinstance(entry, Mapping)
    ]
    display_background = projection_callbacks.current_background_image()
    use_caked_space = bool(projection_callbacks.pick_uses_caked_space())
    click_hint = _saved_click_hint(saved_entry, use_caked_space=use_caked_space)
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
            detector_display_hint=_saved_simulated_display_hint(saved_entry),
        ),
        "emitted_pair": dict(emitted_pair) if isinstance(emitted_pair, Mapping) else None,
        "expected_identity": expected_identity,
        "actual_identity": emitted_identity,
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
    with tempfile.TemporaryDirectory(prefix="new2_fresh_probe_") as tmp_dir:
        temp_state_path = Path(tmp_dir) / "fresh_one_pair_state.json"
        save_gui_state_file(temp_state_path, fresh_state)
        fresh_preflight = _run_single_slot_preflight_validation(
            temp_state_path,
            background_index=int(background_index),
            slot_index=0,
        )
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
    if not bool(slot_result.get("ok", False)):
        return result

    emitted_pair = slot_result.get("emitted_pair")
    fresh_state = _build_fresh_pair_state(
        saved_state=context["saved_state"],
        background_index=int(background_index),
        emitted_pairs=[dict(emitted_pair)] if isinstance(emitted_pair, Mapping) else [],
    )
    if export_fresh_state_path is not None:
        export_fresh_state_path = export_fresh_state_path.expanduser().resolve()
        export_fresh_state_path.parent.mkdir(parents=True, exist_ok=True)
        save_gui_state_file(export_fresh_state_path, fresh_state)
        result["exported_fresh_state_path"] = str(export_fresh_state_path)

    compatibility = _run_saved_state_compatibility_validation(
        state_path,
        int(background_index),
    )
    result["new2_compatibility"] = _compatibility_summary(compatibility)
    result["classification"] = (
        "fresh_contract_pass"
        if bool(compatibility.get("ok", False))
        else "fresh_contract_pass_new2_compatibility_fail"
    )
    result["ok"] = True
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
    result = _public_context_fields(context)
    runtime = _prepare_fresh_slot_runtime(
        context=context,
        background_index=int(background_index),
    )
    slot_results: list[dict[str, object]] = []
    emitted_pairs: list[dict[str, object]] = []
    for slot_index in range(len(saved_entries)):
        slot_result = _run_fresh_slot_validation(
            context=context,
            background_index=int(background_index),
            slot_index=int(slot_index),
            runtime=runtime,
        )
        slot_results.append(slot_result)
        if not bool(slot_result.get("ok", False)):
            result["ok"] = False
            result["classification"] = "seam_failure"
            result["slot_results"] = slot_results
            result["failed_slot_index"] = int(slot_index)
            result["failed_pair"] = slot_result.get("failed_pair", slot_result)
            return result
        emitted_pair = slot_result.get("emitted_pair")
        if isinstance(emitted_pair, Mapping):
            emitted_pairs.append(dict(emitted_pair))

    result["slot_results"] = slot_results
    fresh_state = _build_fresh_pair_state(
        saved_state=context["saved_state"],
        background_index=int(background_index),
        emitted_pairs=emitted_pairs,
    )

    if export_fresh_state_path is not None:
        export_path = export_fresh_state_path.expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        save_gui_state_file(export_path, fresh_state)
        result["exported_fresh_state_path"] = str(export_path)
        compatibility = _run_saved_state_compatibility_validation(
            export_path,
            int(background_index),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="new2_fresh_all_probe_") as tmp_dir:
            temp_state_path = Path(tmp_dir) / "fresh_all_state.json"
            save_gui_state_file(temp_state_path, fresh_state)
            compatibility = _run_saved_state_compatibility_validation(
                temp_state_path,
                int(background_index),
            )

    result["exported_state_compatibility"] = compatibility
    result["ok"] = bool(compatibility.get("ok", False))
    result["classification"] = "pass" if result["ok"] else "seam_failure"
    if not result["ok"]:
        result["failed_pair"] = compatibility.get("failed_pair", compatibility)
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
        choices=("full", "fresh", "fresh-all", "compatibility"),
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
    if effective_mode == "compatibility" and export_fresh_state_path is not None:
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
