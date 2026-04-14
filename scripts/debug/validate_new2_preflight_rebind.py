"""Validate canonical preflight rebinding on one saved GUI state."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from ra_sim import headless_geometry_fit as hgf
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.io.data_loading import load_gui_state_file


class _CapturedPreflight(RuntimeError):
    """Internal stop used to abort headless execution after preflight capture."""


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
) -> dict[str, object] | None:
    if not isinstance(entry, Mapping):
        return None
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
    return payload


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
 

def _run_validation(state_path: Path, background_index: int) -> dict[str, object]:
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

    checked_pairs: list[dict[str, object]] = []
    for slot_index in (1, 2):
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

    if _canonical_identity(checked_pairs[0]["chosen_resolved_live_row"]) == _canonical_identity(
        checked_pairs[1]["chosen_resolved_live_row"]
    ):
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate canonical preflight rebinding on a saved GUI state.",
    )
    parser.add_argument("--state", required=True, help="Path to the saved GUI state.")
    parser.add_argument(
        "--background-index",
        type=int,
        default=0,
        help="Background index to validate. Default: 0.",
    )
    args = parser.parse_args()

    result = _run_validation(
        Path(args.state).expanduser().resolve(),
        background_index=int(args.background_index),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if bool(result.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
