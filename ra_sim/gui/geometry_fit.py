"""Pure helpers for GUI geometry-fit state and configuration."""

from __future__ import annotations

import copy
import os
import sys
import inspect
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ra_sim.debug_controls import (
    geometry_fit_extra_sections_enabled,
    geometry_fit_log_files_enabled,
    resolve_startup_debug_log_path,
)
from ra_sim.fitting.background_peak_matching import (
    build_background_peak_context,
    match_simulated_peaks_to_peak_context,
)
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.utils.calculations import d_spacing, two_theta
from ra_sim.utils.notifications import play_completion_chime

GeometryFitStageCallback = Callable[[str, Mapping[str, object]], None]


GEOMETRY_FIT_PARAM_ORDER = [
    "zb",
    "zs",
    "theta_initial",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "a",
    "c",
    "center_x",
    "center_y",
]

GEOMETRY_FIT_ACCEPT_MAX_RMS_PX = 100.0
GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX = 150.0


def geometry_fit_all_logging_disabled(
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether all geometry-fit file/debug logging is disabled."""

    return not geometry_fit_log_files_enabled(os.environ if env is None else env)


def build_geometry_fit_log_path(
    *,
    stamp: str,
    log_dir: Path | str | None = None,
    downloads_dir: Path | str | None = None,
) -> Path:
    """Return the shared startup-scoped log path for geometry-fit diagnostics."""

    return resolve_startup_debug_log_path(
        stamp=stamp,
        log_dir=log_dir,
        downloads_dir=downloads_dir,
    )


_GEOMETRY_FIT_LOG_SEPARATOR = "=" * 80


def _build_geometry_fit_log_writers(
    log_path: Path | str,
) -> tuple[Path, Callable[[str], None], Callable[[str, Sequence[str]], None]]:
    """Return append-only writers for one geometry-fit log entry."""

    resolved_log_path = Path(log_path)
    logging_disabled = geometry_fit_all_logging_disabled()
    entry_started = False

    def _ensure_entry_started() -> None:
        nonlocal entry_started
        if entry_started or logging_disabled:
            return
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_separator = (
            resolved_log_path.exists()
            and resolved_log_path.stat().st_size > 0
        )
        with resolved_log_path.open("a", encoding="utf-8") as log_file:
            if needs_separator:
                log_file.write("\n")
            log_file.write(_GEOMETRY_FIT_LOG_SEPARATOR + "\n")
        entry_started = True

    def _log_line(text: str = "") -> None:
        if logging_disabled:
            return
        try:
            _ensure_entry_started()
            with resolved_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(text + "\n")
        except Exception:
            pass

    def _log_section(title: str, lines: Sequence[str]) -> None:
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    return resolved_log_path, _log_line, _log_section


@dataclass(frozen=True)
class GeometryFitPreparedRun:
    """Prepared inputs and metadata for one manual-pair geometry-fit run."""

    fit_params: dict[str, object]
    selected_background_indices: list[int]
    background_theta_values: list[float]
    joint_background_mode: bool
    current_dataset: dict[str, object]
    dataset_infos: list[dict[str, object]]
    dataset_specs: list[dict[str, object]]
    start_cmd_line: str
    start_log_sections: list[tuple[str, list[str]]]
    max_display_markers: int
    geometry_runtime_cfg: dict[str, object]


@dataclass(frozen=True)
class GeometryFitRuntimeManualDatasetBindings:
    """Live manual-pair dataset callbacks and values reused by geometry-fit prep."""

    osc_files: Sequence[object]
    current_background_index: int
    image_size: int
    display_rotate_k: int
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]]
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]]
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None]
    geometry_manual_simulated_peaks_for_params: Callable[..., object]
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]]
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ]
    unrotate_display_peaks: Callable[..., list[dict[str, object]]]
    display_to_native_sim_coords: Callable[..., tuple[float, float]]
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]]
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]]
    orient_image_for_fit: Callable[..., object]
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None
    geometry_manual_rebuild_source_rows_for_background: (
        Callable[..., object]
        | None
    ) = None
    geometry_manual_last_source_snapshot_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None
    geometry_manual_last_simulation_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None
    pick_uses_caked_space: Callable[[], bool] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimePreparationBindings:
    """Runtime values and callbacks used to prepare one geometry-fit run."""

    fit_config: Mapping[str, object] | None
    theta_initial: object
    apply_geometry_fit_background_selection: Callable[..., bool]
    current_geometry_fit_background_indices: Callable[..., list[int]]
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    apply_background_theta_metadata: Callable[..., bool]
    current_background_theta_values: Callable[..., list[float]]
    current_geometry_theta_offset: Callable[..., float]
    ensure_geometry_fit_caked_view: Callable[[], None]
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]]


@dataclass(frozen=True)
class GeometryFitRuntimeValueBindings:
    """Live Tk/runtime value sources used by geometry-fit runtime helpers."""

    fit_zb_var: object
    fit_zs_var: object
    fit_theta_var: object
    fit_psi_z_var: object
    fit_chi_var: object
    fit_cor_var: object
    fit_gamma_var: object
    fit_Gamma_var: object
    fit_dist_var: object
    fit_a_var: object
    fit_c_var: object
    fit_center_x_var: object
    fit_center_y_var: object
    zb_var: object
    zs_var: object
    theta_initial_var: object
    psi_z_var: object
    chi_var: object
    cor_angle_var: object
    sample_width_var: object
    sample_length_var: object
    sample_depth_var: object
    gamma_var: object
    Gamma_var: object
    corto_detector_var: object
    a_var: object
    c_var: object
    center_x_var: object
    center_y_var: object
    debye_x_var: object
    debye_y_var: object
    geometry_theta_offset_var: object | None
    current_background_index: object
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    current_geometry_theta_offset: Callable[..., float]
    background_theta_for_index: Callable[..., object]
    build_mosaic_params: Callable[[], Mapping[str, object] | None]
    current_optics_mode_flag: Callable[[], object]
    lambda_value: object
    psi: object
    n2: object
    pixel_size_value: object


@dataclass(frozen=True)
class GeometryFitRuntimeSolverInputs:
    """Simulation inputs needed to invoke the live geometry-fit solver."""

    miller: object
    intensities: object
    image_size: int


@dataclass(frozen=True)
class GeometryFitSolverRequest:
    """One concrete geometry-fit solver request."""

    miller: object
    intensities: object
    image_size: int
    params: dict[str, object]
    measured_peaks: object
    var_names: list[str]
    candidate_param_names: list[str] | None
    dataset_specs: list[dict[str, object]] | None
    refinement_config: dict[str, object]
    runtime_safety_note: str | None = None


@dataclass(frozen=True)
class GeometryFitPreparationResult:
    """One geometry-fit preflight result."""

    prepared_run: GeometryFitPreparedRun | None = None
    error_text: str | None = None
    failure_log_sections: list[tuple[str, list[str]]] | None = None
    log_path: Path | None = None


@dataclass(frozen=True)
class GeometryFitSourceRowRebuildResult:
    """Pure source-row rebuild payload returned before any runtime-state commit."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    rebuild_source: str | None
    rebuild_attempts: list[str]
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    intersection_cache: list[object] | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitBackgroundCacheBundle:
    """Job-scoped geometry-fit cache bundle for one prepared background."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    background_label: str
    theta_base: float
    theta_initial: float
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    cache_source: str | None
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    intersection_cache: list[object] | None = None
    cache_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitPostprocessResult:
    """Pure post-solver geometry-fit analysis results."""

    fitted_params: dict[str, object]
    point_match_summary_lines: list[str]
    pixel_offsets: list[dict[str, object]]
    overlay_records: list[dict[str, object]]
    overlay_state: dict[str, object]
    overlay_diagnostic_lines: list[str]
    frame_warning: str | None
    export_records: list[dict[str, object]]
    save_path: Path
    fit_summary_lines: list[str]
    progress_text: str


@dataclass(frozen=True)
class GeometryFitRuntimeResultBindings:
    """Runtime callback bundle for applying one successful geometry fit."""

    log_section: Callable[[str, Sequence[str]], None]
    capture_undo_state: Callable[[], dict[str, object]]
    apply_result_values: Callable[[Sequence[object], Sequence[object]], None]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    build_profile_cache: Callable[[], dict[str, object]]
    replace_profile_cache: Callable[[dict[str, object]], None]
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    build_fitted_params: Callable[[], dict[str, object]]
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    geometry_runtime_cfg: Mapping[str, object] | None = None
    preview_fitted_params: (
        Callable[[Sequence[object], Sequence[object]], dict[str, object]]
        | None
    ) = None


@dataclass(frozen=True)
class GeometryFitRuntimeUiBindings:
    """Runtime callbacks and UI state sources used during one geometry fit."""

    fit_params: Mapping[str, object] | None
    base_profile_cache: Mapping[str, object] | None
    mosaic_params: Mapping[str, object] | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    geometry_theta_offset_var: object | None
    capture_undo_state: Callable[[], dict[str, object]]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    replace_profile_cache: Callable[[dict[str, object]], None]
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimePostprocessConfig:
    """Post-solver inputs needed to analyze and persist one geometry fit."""

    current_background_index: int
    downloads_dir: Path | str
    stamp: str
    log_path: Path | str
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]
    log_dir: Path | str | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionSetup:
    """Prepared runtime execution inputs for one geometry-fit run."""

    ui_bindings: GeometryFitRuntimeUiBindings
    postprocess_config: GeometryFitRuntimePostprocessConfig


@dataclass(frozen=True)
class GeometryFitRuntimeValueCallbacks:
    """Bound callbacks that expose live geometry-fit values from runtime."""

    current_var_names: Callable[[], list[str]]
    current_params: Callable[[], dict[str, object]]
    current_ui_params: Callable[[], dict[str, object]]
    var_map: Mapping[str, object]


@dataclass(frozen=True)
class GeometryFitRuntimeActionExecutionBindings:
    """Live runtime sources needed to build and run one geometry-fit action."""

    downloads_dir: Path | str
    simulation_runtime_state: Any
    background_runtime_state: Any
    theta_initial_var: Any
    geometry_theta_offset_var: Any | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    background_theta_for_index: Callable[..., object]
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    capture_undo_state: Callable[[], dict[str, object]]
    push_undo_state: Callable[[dict[str, object] | None], None]
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
    request_preview_skip_once: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None
    log_dir: Path | str | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionBindings:
    """Bound runtime callbacks that drive one top-level geometry-fit action."""

    value_callbacks: GeometryFitRuntimeValueCallbacks
    prepare_bindings_factory: Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]
    execution_bindings: GeometryFitRuntimeActionExecutionBindings
    solve_fit: Callable[..., object]
    stamp_factory: Callable[[], str]
    flush_ui: Callable[[], None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionResult:
    """Result metadata for one top-level geometry-fit action invocation."""

    params: Mapping[str, object]
    var_names: list[str]
    preserve_live_theta: bool
    prepare_result: GeometryFitPreparationResult | None = None
    execution_result: GeometryFitRuntimeExecutionResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryFitActionNotice:
    """User-facing notice derived from one top-level geometry-fit action."""

    level: str
    title: str
    message: str


@dataclass(frozen=True)
class GeometryFitRuntimeApplyResult:
    """Result metadata returned after applying one successful geometry fit."""

    accepted: bool
    rejection_reason: str | None
    rms: float
    fitted_params: dict[str, object] | None
    postprocess: GeometryFitPostprocessResult | None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionResult:
    """Result metadata for one full runtime geometry-fit execution."""

    log_path: Path
    solver_request: GeometryFitSolverRequest | None = None
    solver_result: object | None = None
    apply_result: GeometryFitRuntimeApplyResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryToolActionRuntimeCallbacks:
    """Bound runtime callbacks for the geometry tool action control cluster."""

    update_fit_history_button_state: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    set_manual_pick_mode: Callable[[bool, str | None], None]
    toggle_manual_pick_mode: Callable[[], None]
    clear_current_manual_pairs: Callable[[], None]


@dataclass(frozen=True)
class GeometryFitRuntimeHistoryCallbacks:
    """Bound runtime callbacks for geometry-fit undo/redo history transitions."""

    undo: Callable[[], bool]
    redo: Callable[[], bool]


def _emit_geometry_fit_stage_event(
    stage_callback: GeometryFitStageCallback | None,
    stage: str,
    **payload: object,
) -> None:
    """Best-effort helper for optional geometry-fit preflight progress callbacks."""

    if not callable(stage_callback):
        return
    try:
        stage_callback(str(stage), dict(payload))
    except Exception:
        return


def rebuild_geometry_fit_source_rows(
    *,
    background_index: int,
    background_label: str | None = None,
    params_local: Mapping[str, object],
    consumer: str,
    prior_diagnostics: Mapping[str, object] | None,
    requested_signature: object,
    requested_signature_summary: object,
    can_use_live_runtime_cache: bool,
    build_live_rows: Callable[[], Sequence[object]] | None,
    get_memory_intersection_cache: Callable[[], Sequence[object]] | None,
    memory_cache_signature: object | None = None,
    load_logged_intersection_cache: Callable[
        [],
        tuple[Sequence[object], Mapping[str, object] | None],
    ]
    | None,
    logged_cache_matches_params: Callable[
        [Mapping[str, object] | None, Mapping[str, object]],
        bool,
    ]
    | None,
    build_source_rows_from_hit_tables: Callable[
        [Sequence[object]],
        tuple[Sequence[object], Sequence[object] | None, Sequence[object] | None],
    ],
    simulate_hit_tables: Callable[[Mapping[str, object]], Sequence[object]] | None,
    project_rows: Callable[[Sequence[object] | None], Sequence[object]] | None = None,
    live_cache_inventory: Mapping[str, object]
    | Callable[[], Mapping[str, object]]
    | None = None,
) -> GeometryFitSourceRowRebuildResult:
    """Rebuild source rows without mutating runtime state."""

    background_idx = int(background_index)
    lookup_context = str(consumer or "unspecified")
    resolved_background_label = (
        str(background_label).strip()
        if background_label is not None and str(background_label).strip()
        else f"background {int(background_idx) + 1}"
    )
    normalized_params = dict(params_local or {})
    rebuild_attempts: list[str] = []

    def _copy_rows(raw_rows: Sequence[object] | None) -> list[dict[str, object]]:
        return [
            dict(entry)
            for entry in (raw_rows or ())
            if isinstance(entry, Mapping)
        ]

    def _copy_optional_sequence(
        raw_values: Sequence[object] | None,
    ) -> list[object] | None:
        if raw_values is None:
            return None
        return [
            copy.deepcopy(entry)
            for entry in raw_values
        ]

    def _resolve_live_cache_inventory() -> dict[str, object]:
        if callable(live_cache_inventory):
            try:
                resolved = live_cache_inventory()
            except Exception:
                resolved = {}
        else:
            resolved = live_cache_inventory or {}
        copied = copy.deepcopy(resolved)
        return copied if isinstance(copied, dict) else {}

    def _project_copied_rows(raw_rows: Sequence[object] | None) -> list[dict[str, object]]:
        projected_rows = raw_rows
        if callable(project_rows):
            try:
                projected_rows = project_rows(raw_rows)
            except Exception:
                projected_rows = raw_rows
        return _copy_rows(projected_rows)

    def _success_result(
        raw_rows: Sequence[object] | None,
        *,
        rebuild_source: str,
        peak_table_lattice: Sequence[object] | None = None,
        hit_tables_local: Sequence[object] | None = None,
        intersection_cache_local: Sequence[object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> GeometryFitSourceRowRebuildResult:
        stored_rows = _copy_rows(raw_rows)
        projected_rows = _project_copied_rows(stored_rows)
        cache_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        diagnostics = {
            "source": "source_snapshot",
            "cache_family": "source_snapshot",
            "action": "rebuild",
            "consumer": lookup_context,
            "status": "snapshot_hit",
            "background_index": int(background_idx),
            "background_label": resolved_background_label,
            "requested_signature": requested_signature,
            "requested_signature_summary": requested_signature_summary,
            "snapshot_signature": requested_signature,
            "stored_signature_summary": requested_signature_summary,
            "raw_peak_count": int(len(stored_rows)),
            "projected_peak_count": int(len(projected_rows)),
            "created_from": str(rebuild_source),
            "cache_source": "source_snapshot_rebuild",
            "rebuild_source": str(rebuild_source),
            "signature_match": True,
            "rebuild_attempts": list(rebuild_attempts),
            "cache_metadata": cache_metadata,
            "live_cache_inventory": _resolve_live_cache_inventory(),
        }
        return GeometryFitSourceRowRebuildResult(
            background_index=int(background_idx),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            projected_rows=projected_rows,
            stored_rows=stored_rows,
            rebuild_source=str(rebuild_source),
            rebuild_attempts=list(rebuild_attempts),
            diagnostics=diagnostics,
            peak_table_lattice=_copy_optional_sequence(peak_table_lattice),
            hit_tables=_copy_optional_sequence(hit_tables_local),
            intersection_cache=_copy_optional_sequence(intersection_cache_local),
            metadata=cache_metadata,
        )

    if can_use_live_runtime_cache and callable(build_live_rows):
        rebuild_attempts.append("live_runtime_cache")
        live_rows = _copy_rows(build_live_rows())
        if live_rows:
            return _success_result(
                live_rows,
                rebuild_source="live_runtime_cache",
            )

    if (
        memory_cache_signature is not None
        and memory_cache_signature != requested_signature
    ):
        rebuild_attempts.append("last_intersection_cache_memory_signature_mismatch")
    else:
        rebuild_attempts.append("last_intersection_cache_memory")
        memory_intersection_cache = (
            list(get_memory_intersection_cache() or ())
            if callable(get_memory_intersection_cache)
            else []
        )
        memory_rows: list[dict[str, object]] = []
        memory_lattice: Sequence[object] | None = None
        memory_hit_tables: Sequence[object] | None = None
        if memory_intersection_cache:
            memory_rows, memory_lattice, memory_hit_tables = (
                build_source_rows_from_hit_tables(
                    memory_intersection_cache
                )
            )
        if memory_rows:
            return _success_result(
                memory_rows,
                rebuild_source="last_intersection_cache_memory",
                peak_table_lattice=memory_lattice,
                hit_tables_local=memory_hit_tables,
                intersection_cache_local=memory_intersection_cache,
            )

    rebuild_attempts.append("last_intersection_cache_log")
    logged_intersection_cache: Sequence[object] = []
    logged_metadata: Mapping[str, object] | None = None
    if callable(load_logged_intersection_cache):
        try:
            logged_intersection_cache, logged_metadata = load_logged_intersection_cache()
        except Exception:
            logged_intersection_cache, logged_metadata = [], None
    if (
        logged_intersection_cache
        and callable(logged_cache_matches_params)
        and logged_cache_matches_params(logged_metadata, normalized_params)
    ):
        logged_rows, logged_lattice, logged_hit_tables = build_source_rows_from_hit_tables(
            logged_intersection_cache
        )
        if logged_rows:
            return _success_result(
                logged_rows,
                rebuild_source="last_intersection_cache_log",
                peak_table_lattice=logged_lattice,
                hit_tables_local=logged_hit_tables,
                intersection_cache_local=logged_intersection_cache,
                metadata=logged_metadata,
            )

    rebuild_attempts.append("fresh_simulation")
    fresh_hit_tables: Sequence[object] = []
    if callable(simulate_hit_tables):
        try:
            fresh_hit_tables = simulate_hit_tables(normalized_params)
        except Exception:
            fresh_hit_tables = []
    fresh_rows, fresh_lattice, fresh_hit_tables = build_source_rows_from_hit_tables(
        fresh_hit_tables
    )
    if fresh_rows:
        return _success_result(
            fresh_rows,
            rebuild_source="fresh_simulation",
            peak_table_lattice=fresh_lattice,
            hit_tables_local=fresh_hit_tables,
        )

    diagnostics = {
        "source": "source_snapshot",
        "cache_family": "source_snapshot",
        "action": "rebuild",
        "consumer": lookup_context,
        "status": "snapshot_rebuild_failed",
        "background_index": int(background_idx),
        "background_label": resolved_background_label,
        "requested_signature": requested_signature,
        "requested_signature_summary": requested_signature_summary,
        "raw_peak_count": 0,
        "projected_peak_count": 0,
        "signature_match": False,
        "rebuild_attempts": list(rebuild_attempts),
        "prior_diagnostics": (
            dict(prior_diagnostics) if isinstance(prior_diagnostics, Mapping) else {}
        ),
        "live_cache_inventory": _resolve_live_cache_inventory(),
    }
    return GeometryFitSourceRowRebuildResult(
        background_index=int(background_idx),
        requested_signature=requested_signature,
        requested_signature_summary=requested_signature_summary,
        projected_rows=[],
        stored_rows=[],
        rebuild_source=None,
        rebuild_attempts=list(rebuild_attempts),
        diagnostics=diagnostics,
    )


def build_runtime_geometry_fit_manual_dataset_bindings(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    image_size: int,
    display_rotate_k: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None,
    geometry_manual_rebuild_source_rows_for_background: (
        Callable[..., object]
        | None
    ) = None,
    geometry_manual_last_source_snapshot_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None,
    geometry_manual_last_simulation_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None,
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None,
    pick_uses_caked_space: Callable[[], bool] | None = None,
) -> GeometryFitRuntimeManualDatasetBindings:
    """Build the live manual-pair dataset bundle used during geometry-fit prep."""

    return GeometryFitRuntimeManualDatasetBindings(
        osc_files=osc_files,
        current_background_index=int(current_background_index),
        image_size=int(image_size),
        display_rotate_k=int(display_rotate_k),
        geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
        load_background_by_index=load_background_by_index,
        apply_background_backend_orientation=apply_background_backend_orientation,
        geometry_manual_simulated_peaks_for_params=(
            geometry_manual_simulated_peaks_for_params
        ),
        geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
        geometry_manual_source_rows_for_background=(
            geometry_manual_source_rows_for_background
        ),
        geometry_manual_rebuild_source_rows_for_background=(
            geometry_manual_rebuild_source_rows_for_background
        ),
        geometry_manual_last_source_snapshot_diagnostics=(
            geometry_manual_last_source_snapshot_diagnostics
        ),
        geometry_manual_last_simulation_diagnostics=(
            geometry_manual_last_simulation_diagnostics
        ),
        geometry_manual_match_config=geometry_manual_match_config,
        geometry_manual_entry_display_coords=geometry_manual_entry_display_coords,
        unrotate_display_peaks=unrotate_display_peaks,
        display_to_native_sim_coords=display_to_native_sim_coords,
        select_fit_orientation=select_fit_orientation,
        apply_orientation_to_entries=apply_orientation_to_entries,
        orient_image_for_fit=orient_image_for_fit,
        pick_uses_caked_space=pick_uses_caked_space,
    )


def make_runtime_geometry_fit_manual_dataset_bindings_factory(
    *,
    osc_files_factory: Callable[[], Sequence[object]],
    current_background_index_factory: Callable[[], object],
    image_size: int,
    display_rotate_k: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None,
    geometry_manual_rebuild_source_rows_for_background: (
        Callable[..., object]
        | None
    ) = None,
    geometry_manual_last_source_snapshot_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None,
    geometry_manual_last_simulation_diagnostics: (
        Callable[[], Mapping[str, object]]
        | None
    ) = None,
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None,
    pick_uses_caked_space: Callable[[], bool] | None = None,
) -> Callable[[], GeometryFitRuntimeManualDatasetBindings]:
    """Build a factory that resolves the live manual-pair dataset bundle on demand."""

    def _build() -> GeometryFitRuntimeManualDatasetBindings:
        return build_runtime_geometry_fit_manual_dataset_bindings(
            osc_files=osc_files_factory(),
            current_background_index=int(current_background_index_factory()),
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            load_background_by_index=load_background_by_index,
            apply_background_backend_orientation=(
                apply_background_backend_orientation
            ),
            geometry_manual_simulated_peaks_for_params=(
                geometry_manual_simulated_peaks_for_params
            ),
            geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
            geometry_manual_source_rows_for_background=(
                geometry_manual_source_rows_for_background
            ),
            geometry_manual_rebuild_source_rows_for_background=(
                geometry_manual_rebuild_source_rows_for_background
            ),
            geometry_manual_last_source_snapshot_diagnostics=(
                geometry_manual_last_source_snapshot_diagnostics
            ),
            geometry_manual_last_simulation_diagnostics=(
                geometry_manual_last_simulation_diagnostics
            ),
            geometry_manual_match_config=geometry_manual_match_config,
            geometry_manual_entry_display_coords=(
                geometry_manual_entry_display_coords
            ),
            unrotate_display_peaks=unrotate_display_peaks,
            display_to_native_sim_coords=display_to_native_sim_coords,
            select_fit_orientation=select_fit_orientation,
            apply_orientation_to_entries=apply_orientation_to_entries,
            orient_image_for_fit=orient_image_for_fit,
            pick_uses_caked_space=pick_uses_caked_space,
        )

    return _build


def make_runtime_geometry_fit_action_prepare_bindings_factory(
    *,
    fit_config: Mapping[str, object] | None,
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
) -> Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]:
    """Build the live prepare-bundle factory for one geometry-fit action."""

    def _build(var_names: Sequence[str]) -> GeometryFitRuntimePreparationBindings:
        return GeometryFitRuntimePreparationBindings(
            fit_config=fit_config,
            theta_initial=theta_initial,
            apply_geometry_fit_background_selection=(
                apply_geometry_fit_background_selection
            ),
            current_geometry_fit_background_indices=(
                current_geometry_fit_background_indices
            ),
            geometry_fit_uses_shared_theta_offset=(
                geometry_fit_uses_shared_theta_offset
            ),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config=(
                lambda fit_params: build_runtime_config_factory(
                    list(var_names),
                    fit_params,
                )
            ),
        )

    return _build


def build_runtime_geometry_fit_action_execution_bindings(
    *,
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeActionExecutionBindings:
    """Build the live execution-bundle for one geometry-fit action."""

    return GeometryFitRuntimeActionExecutionBindings(
        downloads_dir=downloads_dir,
        log_dir=log_dir,
        simulation_runtime_state=simulation_runtime_state,
        background_runtime_state=background_runtime_state,
        theta_initial_var=theta_initial_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        current_ui_params=current_ui_params,
        var_map=var_map,
        background_theta_for_index=background_theta_for_index,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        capture_undo_state=capture_undo_state,
        push_undo_state=push_undo_state,
        replace_dataset_cache=replace_dataset_cache,
        request_preview_skip_once=request_preview_skip_once,
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
        live_update_callback=live_update_callback,
    )


def build_runtime_geometry_fit_action_bindings(
    *,
    value_callbacks: GeometryFitRuntimeValueCallbacks,
    fit_config: Mapping[str, object] | None,
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeActionBindings:
    """Build the top-level live geometry-fit action bindings."""

    execution_binding_kwargs: dict[str, object] = {
        "downloads_dir": downloads_dir,
        "simulation_runtime_state": simulation_runtime_state,
        "background_runtime_state": background_runtime_state,
        "theta_initial_var": theta_initial_var,
        "geometry_theta_offset_var": geometry_theta_offset_var,
        "current_ui_params": current_ui_params,
        "var_map": var_map,
        "background_theta_for_index": background_theta_for_index,
        "refresh_status": refresh_status,
        "update_manual_pick_button_label": update_manual_pick_button_label,
        "capture_undo_state": capture_undo_state,
        "push_undo_state": push_undo_state,
        "replace_dataset_cache": replace_dataset_cache,
        "request_preview_skip_once": request_preview_skip_once,
        "schedule_update": schedule_update,
        "draw_overlay_records": draw_overlay_records,
        "draw_initial_pairs_overlay": draw_initial_pairs_overlay,
        "set_last_overlay_state": set_last_overlay_state,
        "set_progress_text": set_progress_text,
        "cmd_line": cmd_line,
        "solver_inputs": solver_inputs,
        "sim_display_rotate_k": int(sim_display_rotate_k),
        "background_display_rotate_k": int(background_display_rotate_k),
        "simulate_and_compare_hkl": simulate_and_compare_hkl,
        "aggregate_match_centers": aggregate_match_centers,
        "build_overlay_records": build_overlay_records,
        "compute_frame_diagnostics": compute_frame_diagnostics,
        "live_update_callback": live_update_callback,
    }
    if log_dir is not None:
        execution_binding_kwargs["log_dir"] = log_dir

    return GeometryFitRuntimeActionBindings(
        value_callbacks=value_callbacks,
        prepare_bindings_factory=make_runtime_geometry_fit_action_prepare_bindings_factory(
            fit_config=fit_config,
            theta_initial=theta_initial,
            apply_geometry_fit_background_selection=(
                apply_geometry_fit_background_selection
            ),
            current_geometry_fit_background_indices=(
                current_geometry_fit_background_indices
            ),
            geometry_fit_uses_shared_theta_offset=(
                geometry_fit_uses_shared_theta_offset
            ),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config_factory=build_runtime_config_factory,
        ),
        execution_bindings=build_runtime_geometry_fit_action_execution_bindings(
            **execution_binding_kwargs
        ),
        solve_fit=solve_fit,
        stamp_factory=stamp_factory,
        flush_ui=flush_ui,
    )


def make_runtime_geometry_fit_action_bindings_factory(
    *,
    value_callbacks_factory: Callable[[], GeometryFitRuntimeValueCallbacks],
    fit_config: Mapping[str, object] | None,
    theta_initial_factory: Callable[[], object],
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings_factory: Callable[
        [],
        GeometryFitRuntimeManualDatasetBindings,
    ],
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs_factory: Callable[[], GeometryFitRuntimeSolverInputs],
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> Callable[[], GeometryFitRuntimeActionBindings]:
    """Build a factory that resolves live geometry-fit action bindings on demand."""

    def _build() -> GeometryFitRuntimeActionBindings:
        return build_runtime_geometry_fit_action_bindings(
            value_callbacks=value_callbacks_factory(),
            fit_config=fit_config,
            theta_initial=theta_initial_factory(),
            apply_geometry_fit_background_selection=(
                apply_geometry_fit_background_selection
            ),
            current_geometry_fit_background_indices=(
                current_geometry_fit_background_indices
            ),
            geometry_fit_uses_shared_theta_offset=(
                geometry_fit_uses_shared_theta_offset
            ),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings_factory(),
            build_runtime_config_factory=build_runtime_config_factory,
            downloads_dir=downloads_dir,
            log_dir=log_dir,
            simulation_runtime_state=simulation_runtime_state,
            background_runtime_state=background_runtime_state,
            theta_initial_var=theta_initial_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_ui_params=current_ui_params,
            var_map=var_map,
            background_theta_for_index=background_theta_for_index,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            capture_undo_state=capture_undo_state,
            push_undo_state=push_undo_state,
            replace_dataset_cache=replace_dataset_cache,
            request_preview_skip_once=request_preview_skip_once,
            schedule_update=schedule_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            set_last_overlay_state=set_last_overlay_state,
            set_progress_text=set_progress_text,
            cmd_line=cmd_line,
            solver_inputs=solver_inputs_factory(),
            sim_display_rotate_k=int(sim_display_rotate_k),
            background_display_rotate_k=int(background_display_rotate_k),
            simulate_and_compare_hkl=simulate_and_compare_hkl,
            aggregate_match_centers=aggregate_match_centers,
            build_overlay_records=build_overlay_records,
            compute_frame_diagnostics=compute_frame_diagnostics,
            live_update_callback=live_update_callback,
            solve_fit=solve_fit,
            stamp_factory=stamp_factory,
            flush_ui=flush_ui,
        )

    return _build


def make_runtime_geometry_fit_action_callback(
    bindings_factory: Callable[[], GeometryFitRuntimeActionBindings],
    *,
    before_run: Callable[[], None] | None = None,
    run_action: Callable[..., GeometryFitRuntimeActionResult] | None = None,
    after_run: Callable[[GeometryFitRuntimeActionResult], None] | None = None,
) -> Callable[[], None]:
    """Build the zero-arg Tk-safe runtime callback for the top-level geometry-fit action."""

    def _run() -> None:
        if callable(before_run):
            before_run()
        action = run_action if callable(run_action) else run_runtime_geometry_fit_action
        result = action(bindings=bindings_factory())
        if callable(after_run):
            after_run(result)
        # Tkinter stringifies callback return values. Returning the rich action
        # result here can force a dataclass repr of SciPy OptimizeResult payloads.
        return None

    return _run


def build_geometry_fit_action_notice(
    action_result: GeometryFitRuntimeActionResult | None,
) -> GeometryFitActionNotice | None:
    """Return a user-facing notice for one failed or rejected geometry fit."""

    if action_result is None:
        return None

    error_text = str(action_result.error_text or "").strip()
    if error_text:
        lines = [error_text]
        log_path = None
        execution_result = action_result.execution_result
        if execution_result is not None:
            log_path = getattr(execution_result, "log_path", None)
        if log_path is None:
            prepare_result = action_result.prepare_result
            if prepare_result is not None:
                log_path = getattr(prepare_result, "log_path", None)
        if log_path is not None:
            lines.append(f"Fit log: {Path(log_path)}")
        return GeometryFitActionNotice(
            level="error",
            title="Geometry Fit Failed",
            message="\n".join(lines),
        )

    execution_result = action_result.execution_result
    if execution_result is None:
        return None

    apply_result = execution_result.apply_result
    if apply_result is None or bool(apply_result.accepted):
        return None

    lines = ["Geometry fit finished but the solution was rejected."]
    rejection_reason = str(apply_result.rejection_reason or "").strip()
    if rejection_reason:
        lines.append(rejection_reason)
    lines.append("The live geometry state was left unchanged.")
    log_path = getattr(execution_result, "log_path", None)
    if log_path is not None:
        lines.append(f"Fit log: {Path(log_path)}")
    return GeometryFitActionNotice(
        level="warning",
        title="Geometry Fit Rejected",
        message="\n".join(lines),
    )


def copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: copy_geometry_fit_state_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(copy_geometry_fit_state_value(val) for val in value)
    return value


def current_geometry_fit_ui_params(
    *,
    zb: float,
    zs: float,
    theta_initial: float,
    psi_z: float,
    chi: float,
    cor_angle: float,
    gamma: float,
    Gamma: float,
    corto_detector: float,
    a: float,
    c: float,
    center_x: float,
    center_y: float,
    theta_offset: float | None = None,
) -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    params = {
        "zb": float(zb),
        "zs": float(zs),
        "theta_initial": float(theta_initial),
        "psi_z": float(psi_z),
        "chi": float(chi),
        "cor_angle": float(cor_angle),
        "gamma": float(gamma),
        "Gamma": float(Gamma),
        "corto_detector": float(corto_detector),
        "a": float(a),
        "c": float(c),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center": [float(center_x), float(center_y)],
    }
    if theta_offset is not None:
        params["theta_offset"] = float(theta_offset)
    return params


def current_geometry_fit_var_names(
    *,
    fit_zb: bool,
    fit_zs: bool,
    fit_theta: bool,
    fit_psi_z: bool,
    fit_chi: bool,
    fit_cor: bool,
    fit_gamma: bool,
    fit_Gamma: bool,
    fit_dist: bool,
    fit_a: bool,
    fit_c: bool,
    fit_center_x: bool,
    fit_center_y: bool,
    use_shared_theta_offset: bool = False,
) -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    var_names: list[str] = []
    if fit_zb:
        var_names.append("zb")
    if fit_zs:
        var_names.append("zs")
    if fit_theta:
        var_names.append("theta_offset" if use_shared_theta_offset else "theta_initial")
    if fit_psi_z:
        var_names.append("psi_z")
    if fit_chi:
        var_names.append("chi")
    if fit_cor:
        var_names.append("cor_angle")
    if fit_gamma:
        var_names.append("gamma")
    if fit_Gamma:
        var_names.append("Gamma")
    if fit_dist:
        var_names.append("corto_detector")
    if fit_a:
        var_names.append("a")
    if fit_c:
        var_names.append("c")
    if fit_center_x:
        var_names.append("center_x")
    if fit_center_y:
        var_names.append("center_y")
    return var_names


def geometry_fit_constraint_source_name(name: str) -> str:
    """Map fitted parameter names back to the UI constraint control names."""

    if str(name) == "theta_offset":
        return "theta_initial"
    return str(name)


def geometry_fit_constraint_parameter_name(
    name: str,
    *,
    use_shared_theta_offset: bool = False,
) -> str:
    """Map UI constraint row names to the active fitted parameter names."""

    if str(name) == "theta_initial" and bool(use_shared_theta_offset):
        return "theta_offset"
    return str(name)


def read_runtime_geometry_fit_constraint_state(
    *,
    controls: Mapping[str, object],
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, dict[str, float]]:
    """Read the live geometry-fit constraint controls into normalized settings."""

    selected_names = list(names) if names is not None else list(controls)
    state: dict[str, dict[str, float]] = {}
    for name in selected_names:
        control = controls.get(
            geometry_fit_constraint_source_name(
                geometry_fit_constraint_parameter_name(
                    str(name),
                    use_shared_theta_offset=use_shared_theta_offset,
                )
            )
        )
        if not isinstance(control, dict):
            continue
        try:
            window = float(control["window_var"].get())
        except Exception:
            window = float("nan")
        try:
            pull = float(control["pull_var"].get())
        except Exception:
            pull = 0.0
        if not np.isfinite(window):
            continue
        window = max(0.0, float(window))
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        state[str(name)] = {
            "window": float(window),
            "pull": float(pull),
        }
    return state


def _geometry_fit_config_section(
    fit_config: Mapping[str, object] | None,
    section: str,
) -> Mapping[str, object]:
    """Return one normalized geometry-fit config mapping section."""

    fit_geometry_cfg = (
        fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    )
    if not isinstance(fit_geometry_cfg, Mapping):
        return {}
    section_cfg = fit_geometry_cfg.get(section, {}) or {}
    if not isinstance(section_cfg, Mapping):
        return {}
    return section_cfg


def read_runtime_geometry_fit_parameter_domains(
    *,
    parameter_specs: Mapping[str, object],
    image_size: object,
    fit_config: Mapping[str, object] | None,
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, tuple[float, float]]:
    """Read live parameter domains from slider ranges and image geometry."""

    selected_names = list(names) if names is not None else list(parameter_specs)
    domains: dict[str, tuple[float, float]] = {}
    try:
        image_size_value = float(image_size)
    except Exception:
        image_size_value = 0.0

    for name in selected_names:
        parameter_name = geometry_fit_constraint_parameter_name(
            str(name),
            use_shared_theta_offset=use_shared_theta_offset,
        )
        control_name = geometry_fit_constraint_source_name(parameter_name)

        if parameter_name == "center_x" or parameter_name == "center_y":
            domains[str(name)] = (0.0, max(image_size_value - 1.0, 0.0))
            continue

        spec = parameter_specs.get(control_name)
        if not isinstance(spec, Mapping):
            continue
        slider_widget = spec.get("value_slider")
        if slider_widget is None:
            continue
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            continue
        if parameter_name == "theta_offset":
            span = max(abs(lo), abs(hi), 1.0)
            domains[str(name)] = (-float(span), float(span))
            continue
        if lo > hi:
            lo, hi = hi, lo
        domains[str(name)] = (float(lo), float(hi))
    return domains


def default_runtime_geometry_fit_constraint_window(
    *,
    name: str,
    parameter_specs: Mapping[str, object],
    fit_config: Mapping[str, object] | None,
    parameter_domains: Mapping[str, tuple[float, float]] | None = None,
    current_theta_offset: object = 0.0,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint window for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    control_name = geometry_fit_constraint_source_name(parameter_name)
    spec = parameter_specs.get(control_name, {})
    if parameter_name == "theta_offset":
        try:
            current_value = float(current_theta_offset)
        except Exception:
            current_value = 0.0
    else:
        try:
            current_value = float(spec["value_var"].get())
        except Exception:
            current_value = 0.0
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    resolved_domains = parameter_domains or {}
    domain = resolved_domains.get(parameter_name)
    if domain is None:
        domain = resolved_domains.get(str(name))
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))

    bounds_cfg = _geometry_fit_config_section(fit_config, "bounds")
    entry = bounds_cfg.get(parameter_name)

    default_window = float("nan")
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            lo = float(entry[0])
            hi = float(entry[1])
            default_window = max(abs(current_value - lo), abs(hi - current_value))
        except Exception:
            default_window = float("nan")
    elif isinstance(entry, Mapping):
        mode = str(entry.get("mode", "absolute")).strip().lower()
        try:
            min_raw = (
                float(entry.get("min"))
                if entry.get("min") is not None
                else float("nan")
            )
        except Exception:
            min_raw = float("nan")
        try:
            max_raw = (
                float(entry.get("max"))
                if entry.get("max") is not None
                else float("nan")
            )
        except Exception:
            max_raw = float("nan")
        if mode in {"relative", "rel", "relative_min0", "rel_min0"}:
            candidates = [abs(v) for v in (min_raw, max_raw) if np.isfinite(v)]
            if candidates:
                default_window = max(candidates)
        else:
            candidates = [
                abs(current_value - v)
                for v in (min_raw, max_raw)
                if np.isfinite(v)
            ]
            if candidates:
                default_window = max(candidates)

    if not np.isfinite(default_window) or default_window <= 0.0:
        default_window = max(
            step * 10.0,
            0.02 * domain_span,
            0.1 * max(abs(current_value), 1.0),
        )

    if domain_span > 0.0:
        default_window = min(default_window, domain_span)

    return max(float(default_window), step)


def default_runtime_geometry_fit_constraint_pull(
    *,
    name: str,
    fit_config: Mapping[str, object] | None,
    window: float,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint pull for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    priors_cfg = _geometry_fit_config_section(fit_config, "priors")
    entry = priors_cfg.get(parameter_name)
    if not isinstance(entry, Mapping):
        return 0.0
    try:
        sigma = float(entry.get("sigma"))
    except Exception:
        sigma = float("nan")
    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
        or not np.isfinite(window)
        or window <= 0.0
    ):
        return 0.0
    inferred = (1.0 - min(max(sigma / window, 0.05), 1.0)) / 0.95
    if not np.isfinite(inferred):
        return 0.0
    return min(max(float(inferred), 0.0), 1.0)


def build_runtime_geometry_fit_value_callbacks(
    bindings: GeometryFitRuntimeValueBindings,
) -> GeometryFitRuntimeValueCallbacks:
    """Build bound live geometry-fit value readers for the runtime."""

    var_map = {
        "zb": bindings.zb_var,
        "zs": bindings.zs_var,
        "theta_initial": bindings.theta_initial_var,
        "psi_z": bindings.psi_z_var,
        "chi": bindings.chi_var,
        "cor_angle": bindings.cor_angle_var,
        "gamma": bindings.gamma_var,
        "Gamma": bindings.Gamma_var,
        "corto_detector": bindings.corto_detector_var,
        "a": bindings.a_var,
        "c": bindings.c_var,
        "center_x": bindings.center_x_var,
        "center_y": bindings.center_y_var,
    }

    def _current_var_names() -> list[str]:
        return current_geometry_fit_var_names(
            fit_zb=bool(bindings.fit_zb_var.get()),
            fit_zs=bool(bindings.fit_zs_var.get()),
            fit_theta=bool(bindings.fit_theta_var.get()),
            fit_psi_z=bool(bindings.fit_psi_z_var.get()),
            fit_chi=bool(bindings.fit_chi_var.get()),
            fit_cor=bool(bindings.fit_cor_var.get()),
            fit_gamma=bool(bindings.fit_gamma_var.get()),
            fit_Gamma=bool(bindings.fit_Gamma_var.get()),
            fit_dist=bool(bindings.fit_dist_var.get()),
            fit_a=bool(bindings.fit_a_var.get()),
            fit_c=bool(bindings.fit_c_var.get()),
            fit_center_x=bool(bindings.fit_center_x_var.get()),
            fit_center_y=bool(bindings.fit_center_y_var.get()),
            use_shared_theta_offset=bool(
                bindings.geometry_fit_uses_shared_theta_offset()
            ),
        )

    def _current_params() -> dict[str, object]:
        use_theta_offset = bool(bindings.geometry_fit_uses_shared_theta_offset())
        theta_offset_current = (
            float(bindings.current_geometry_theta_offset(strict=False))
            if use_theta_offset
            else 0.0
        )
        current_background_index_value = (
            bindings.current_background_index()
            if callable(bindings.current_background_index)
            else bindings.current_background_index
        )
        current_background_index = int(current_background_index_value)
        theta_current = (
            bindings.background_theta_for_index(
                current_background_index,
                strict_count=False,
            )
            if use_theta_offset
            else bindings.theta_initial_var.get()
        )
        n2_value = bindings.n2() if callable(bindings.n2) else bindings.n2
        pixel_size_value = (
            bindings.pixel_size_value()
            if callable(bindings.pixel_size_value)
            else bindings.pixel_size_value
        )
        return {
            "a": bindings.a_var.get(),
            "c": bindings.c_var.get(),
            "lambda": bindings.lambda_value,
            "psi": bindings.psi,
            "psi_z": bindings.psi_z_var.get(),
            "zs": bindings.zs_var.get(),
            "zb": bindings.zb_var.get(),
            "sample_width_m": bindings.sample_width_var.get(),
            "sample_length_m": bindings.sample_length_var.get(),
            "sample_depth_m": bindings.sample_depth_var.get(),
            "chi": bindings.chi_var.get(),
            "n2": n2_value,
            "mosaic_params": dict(bindings.build_mosaic_params() or {}),
            "debye_x": bindings.debye_x_var.get(),
            "debye_y": bindings.debye_y_var.get(),
            "center": [bindings.center_x_var.get(), bindings.center_y_var.get()],
            "center_x": bindings.center_x_var.get(),
            "center_y": bindings.center_y_var.get(),
            "theta_initial": theta_current,
            "theta_offset": theta_offset_current,
            "uv1": np.array([1.0, 0.0, 0.0]),
            "uv2": np.array([0.0, 1.0, 0.0]),
            "corto_detector": bindings.corto_detector_var.get(),
            "gamma": bindings.gamma_var.get(),
            "Gamma": bindings.Gamma_var.get(),
            "cor_angle": bindings.cor_angle_var.get(),
            "optics_mode": bindings.current_optics_mode_flag(),
            "pixel_size": pixel_size_value,
            "pixel_size_m": pixel_size_value,
        }

    def _current_ui_params() -> dict[str, object]:
        theta_offset = None
        if bindings.geometry_theta_offset_var is not None:
            theta_offset = float(bindings.current_geometry_theta_offset(strict=False))
        return current_geometry_fit_ui_params(
            zb=float(bindings.zb_var.get()),
            zs=float(bindings.zs_var.get()),
            theta_initial=float(bindings.theta_initial_var.get()),
            psi_z=float(bindings.psi_z_var.get()),
            chi=float(bindings.chi_var.get()),
            cor_angle=float(bindings.cor_angle_var.get()),
            gamma=float(bindings.gamma_var.get()),
            Gamma=float(bindings.Gamma_var.get()),
            corto_detector=float(bindings.corto_detector_var.get()),
            a=float(bindings.a_var.get()),
            c=float(bindings.c_var.get()),
            center_x=float(bindings.center_x_var.get()),
            center_y=float(bindings.center_y_var.get()),
            theta_offset=theta_offset,
        )

    return GeometryFitRuntimeValueCallbacks(
        current_var_names=_current_var_names,
        current_params=_current_params,
        current_ui_params=_current_ui_params,
        var_map=var_map,
    )


def build_runtime_geometry_fit_config_factory(
    *,
    base_config: Mapping[str, object] | None,
    current_constraint_state: Callable[[Sequence[str] | None], Mapping[str, object]],
    current_parameter_domains: Callable[[Sequence[str] | None], Mapping[str, object]],
    current_candidate_param_names: Callable[[], Sequence[str]] | None = None,
) -> Callable[[Sequence[str], Mapping[str, object]], dict[str, object]]:
    """Build the live geometry-fit refinement-config factory from runtime readers."""

    def _build(
        var_names: Sequence[str],
        fit_params: Mapping[str, object],
    ) -> dict[str, object]:
        selected_names = [str(name) for name in var_names]
        if callable(current_candidate_param_names):
            candidate_names = [str(name) for name in current_candidate_param_names() or ()]
        else:
            candidate_names = []
        if not candidate_names:
            candidate_names = list(selected_names)
        candidate_names = list(dict.fromkeys(candidate_names))
        current_params = {
            name: fit_params.get(name)
            for name in candidate_names
        }
        return build_geometry_fit_runtime_config(
            base_config,
            current_params,
            {},
            current_parameter_domains(candidate_names),
            candidate_param_names=candidate_names,
        )

    return _build


def build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
    *,
    candidate_param_names: Sequence[str] | None = None,
):
    runtime_cfg = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    # GUI geometry fits should preserve the configured solver parallel settings
    # by default so the runtime keeps using the shared "all but two" worker
    # budget. The unsafe-runtime toggle still controls whether Numba stays
    # enabled, while explicit gui_* overrides continue to win when provided.
    optimizer_cfg_raw = runtime_cfg.get("optimizer", runtime_cfg.get("solver", {})) or {}
    optimizer_cfg = (
        dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    )
    runtime_cfg["optimizer"] = optimizer_cfg
    runtime_cfg["solver"] = optimizer_cfg

    gui_use_numba = runtime_cfg.pop("gui_use_numba", None)
    runtime_cfg["use_numba"] = (
        bool(gui_use_numba) if gui_use_numba is not None else False
    )

    gui_allow_unsafe_runtime = runtime_cfg.pop("gui_allow_unsafe_runtime", None)
    allow_unsafe_runtime = bool(gui_allow_unsafe_runtime)
    runtime_cfg["allow_unsafe_runtime"] = allow_unsafe_runtime

    gui_workers = optimizer_cfg.pop("gui_workers", None)
    if gui_workers is None:
        gui_workers = optimizer_cfg.get("workers", "auto")
    optimizer_cfg["workers"] = gui_workers if gui_workers is not None else "auto"

    gui_parallel_mode = optimizer_cfg.pop("gui_parallel_mode", None)
    if gui_parallel_mode is None:
        gui_parallel_mode = optimizer_cfg.get("parallel_mode", "auto")
    optimizer_cfg["parallel_mode"] = (
        str(gui_parallel_mode).strip()
        if gui_parallel_mode is not None
        else "auto"
    )

    gui_worker_numba_threads = optimizer_cfg.pop("gui_worker_numba_threads", None)
    if gui_worker_numba_threads is None:
        gui_worker_numba_threads = optimizer_cfg.get("worker_numba_threads", 0)
    optimizer_cfg["worker_numba_threads"] = (
        gui_worker_numba_threads if gui_worker_numba_threads is not None else 0
    )

    bounds_cfg = runtime_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    runtime_cfg["bounds"] = bounds_cfg

    priors_cfg = runtime_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    runtime_cfg["priors"] = priors_cfg

    active_names = [
        str(name)
        for name in (
            candidate_param_names
            if candidate_param_names is not None
            else list(current_params or {})
        )
    ]
    active_names = list(dict.fromkeys(active_names))

    # GUI geometry fits now expose the full live parameter domains instead of
    # constraining each variable to a current-value window or soft prior.
    for name in active_names:
        priors_cfg.pop(str(name), None)
        domain = (parameter_domains or {}).get(name)
        if not isinstance(domain, (list, tuple)) or len(domain) < 2:
            bounds_cfg.pop(str(name), None)
            continue
        try:
            lo = float(domain[0])
            hi = float(domain[1])
        except Exception:
            bounds_cfg.pop(str(name), None)
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            bounds_cfg.pop(str(name), None)
            continue
        if lo > hi:
            lo, hi = hi, lo
        bounds_cfg[str(name)] = [float(lo), float(hi)]

    runtime_cfg["candidate_param_names"] = active_names

    return runtime_cfg


def apply_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
):
    """Apply saved UI values and return copied cache/overlay state."""

    if not isinstance(state, dict):
        return {
            "profile_cache": {},
            "overlay_state": None,
        }

    ui_params = state.get("ui_params", {}) or {}
    for name, var in var_map.items():
        try:
            value = float(ui_params.get(name))
        except Exception:
            continue
        if np.isfinite(value):
            var.set(value)

    if geometry_theta_offset_var is not None:
        try:
            theta_offset = float(ui_params.get("theta_offset", 0.0))
        except Exception:
            theta_offset = 0.0
        if np.isfinite(theta_offset):
            geometry_theta_offset_var.set(f"{theta_offset:.6g}")

    overlay_state = copy_geometry_fit_state_value(state.get("overlay_state"))
    if not overlay_state:
        overlay_state = None

    return {
        "profile_cache": copy_geometry_fit_state_value(state.get("profile_cache", {})) or {},
        "overlay_state": overlay_state,
    }


def set_runtime_geometry_fit_history_button_state(
    *,
    can_undo: bool,
    can_redo: bool,
    set_button_state: Callable[[bool, bool], None] | None = None,
) -> None:
    """Apply the current geometry-fit undo/redo availability to the UI."""

    if callable(set_button_state):
        set_button_state(bool(can_undo), bool(can_redo))


def _resolve_runtime_value(value_or_factory):
    """Return one runtime value, calling it first when it is a factory."""

    if callable(value_or_factory):
        return value_or_factory()
    return value_or_factory


def _geometry_fit_sequence_has_items(value: object) -> bool:
    """Return whether one saved sequence-like payload contains any entries."""

    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).size) > 0
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except Exception:
        return True


def _geometry_fit_sequence_list(value: object) -> list[object]:
    """Normalize one saved sequence-like payload into a plain list."""

    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            item = arr.item()
            if item is None:
                return []
            if isinstance(item, list):
                return list(item)
            if isinstance(item, tuple):
                return list(item)
            return [item]
        return list(arr.tolist())
    try:
        return list(value)  # type: ignore[arg-type]
    except Exception:
        return [value]


def redraw_runtime_geometry_fit_overlay_state(
    overlay_state: Mapping[str, object] | None,
    *,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
) -> bool:
    """Redraw one saved geometry-fit overlay payload when it contains display data."""

    overlay_payload = dict(overlay_state) if isinstance(overlay_state, Mapping) else {}
    overlay_records = _geometry_fit_sequence_list(
        overlay_payload.get("overlay_records")
    )
    initial_pairs_display = _geometry_fit_sequence_list(
        overlay_payload.get("initial_pairs_display")
    )
    try:
        max_display_markers = int(overlay_payload.get("max_display_markers", 120))
    except Exception:
        max_display_markers = 120
    max_display_markers = max(1, max_display_markers)

    if overlay_records and callable(draw_overlay_records):
        draw_overlay_records(overlay_records, int(max_display_markers))
        return True
    if initial_pairs_display and callable(draw_initial_pairs_overlay):
        draw_initial_pairs_overlay(initial_pairs_display, int(max_display_markers))
        return True
    return False


def capture_runtime_geometry_fit_undo_state(
    *,
    current_ui_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    current_profile_cache: Callable[[], object] | object,
    copy_state_value: Callable[[object], object],
    last_overlay_state: Callable[[], Mapping[str, object] | None] | Mapping[str, object] | None,
    build_initial_pairs_display: Callable[..., tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]],
    current_background_index: Callable[[], object] | object,
    current_fit_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    pending_pairs_display: Callable[[], Sequence[dict[str, object]]] | Sequence[dict[str, object]],
) -> dict[str, object]:
    """Capture the current geometry-fit UI/profile/overlay state for undo."""

    overlay_state = copy_state_value(_resolve_runtime_value(last_overlay_state))
    overlay_records = (
        overlay_state.get("overlay_records")
        if isinstance(overlay_state, dict)
        else None
    )
    initial_pairs_display = (
        overlay_state.get("initial_pairs_display")
        if isinstance(overlay_state, dict)
        else None
    )
    if not (
        isinstance(overlay_state, dict)
        and (
            _geometry_fit_sequence_has_items(overlay_records)
            or _geometry_fit_sequence_has_items(initial_pairs_display)
        )
    ):
        try:
            _, initial_pairs_display = build_initial_pairs_display(
                int(_resolve_runtime_value(current_background_index)),
                param_set=_resolve_runtime_value(current_fit_params),
                prefer_cache=True,
            )
            pending_display = _resolve_runtime_value(pending_pairs_display)
            combined_pairs_display = list(initial_pairs_display) + list(pending_display)
            if combined_pairs_display:
                overlay_state = {
                    "overlay_records": [],
                    "initial_pairs_display": copy_state_value(combined_pairs_display),
                    "max_display_markers": max(1, len(combined_pairs_display)),
                }
        except Exception:
            pass

    return {
        "ui_params": copy_state_value(_resolve_runtime_value(current_ui_params)),
        "profile_cache": copy_state_value(_resolve_runtime_value(current_profile_cache)),
        "overlay_state": overlay_state,
    }


def restore_runtime_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Mapping[str, object]:
    """Apply one saved geometry-fit history state back onto the live runtime."""

    ui_params_raw = state.get("ui_params", {}) if isinstance(state, dict) else {}
    ui_params = dict(ui_params_raw) if isinstance(ui_params_raw, Mapping) else {}
    overlay_state_raw = (
        copy_geometry_fit_state_value(state.get("overlay_state"))
        if isinstance(state, dict)
        else None
    )
    overlay_state = (
        dict(overlay_state_raw)
        if isinstance(overlay_state_raw, Mapping)
        else None
    )
    profile_cache_raw = (
        copy_geometry_fit_state_value(state.get("profile_cache", {}))
        if isinstance(state, dict)
        else {}
    )
    profile_cache = (
        dict(profile_cache_raw)
        if isinstance(profile_cache_raw, Mapping)
        else {}
    )
    restored = apply_undo_state(
        {
            **(dict(state) if isinstance(state, dict) else {}),
            "ui_params": ui_params,
            "overlay_state": overlay_state,
            "profile_cache": profile_cache,
        },
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
    )
    restored_profile_cache = restored.get("profile_cache", {})
    replace_profile_cache(
        dict(restored_profile_cache)
        if isinstance(restored_profile_cache, Mapping)
        else {}
    )
    overlay_state = restored.get("overlay_state")
    set_last_overlay_state(
        dict(overlay_state) if isinstance(overlay_state, dict) else None
    )

    if callable(request_preview_skip_once):
        request_preview_skip_once()
    if callable(mark_last_simulation_dirty):
        mark_last_simulation_dirty()
    if callable(cancel_pending_update):
        cancel_pending_update()
    if callable(run_update):
        run_update()

    redraw_runtime_geometry_fit_overlay_state(
        overlay_state if isinstance(overlay_state, Mapping) else None,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
    )

    if callable(refresh_status):
        refresh_status()
    if callable(update_manual_pick_button_label):
        update_manual_pick_button_label()
    return restored


def build_runtime_geometry_fit_undo_restore_callback(
    *,
    var_map_factory: Mapping[str, object] | Callable[[], Mapping[str, object]],
    geometry_theta_offset_var_factory: object | Callable[[], object] | None = None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Callable[[dict[str, object]], Mapping[str, object] | None]:
    """Build one undo-restore callback that resolves live runtime hooks lazily."""

    def _restore(state: dict[str, object]) -> Mapping[str, object] | None:
        if not isinstance(state, dict):
            return None
        return restore_runtime_geometry_fit_undo_state(
            state,
            var_map=_resolve_runtime_value(var_map_factory),
            geometry_theta_offset_var=_resolve_runtime_value(
                geometry_theta_offset_var_factory
            ),
            replace_profile_cache=replace_profile_cache,
            set_last_overlay_state=set_last_overlay_state,
            request_preview_skip_once=request_preview_skip_once,
            mark_last_simulation_dirty=mark_last_simulation_dirty,
            cancel_pending_update=cancel_pending_update,
            run_update=run_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            apply_undo_state=apply_undo_state,
        )

    return _restore


def _run_runtime_geometry_fit_history_transition(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_transition: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
    empty_text: str,
    failure_prefix: str,
    success_text: str,
) -> bool:
    """Run one geometry-fit history transition against the live runtime."""

    history_available = (
        bool(has_history()) if callable(has_history) else bool(has_history)
    )
    if not history_available:
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    current_state = capture_current_state()
    state = read_state()
    if not isinstance(state, dict):
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    try:
        restore_state(state)
    except Exception as exc:
        if callable(set_progress_text):
            set_progress_text(f"{failure_prefix}: {exc}")
        return False

    commit_transition(current_state)
    if callable(update_button_state):
        update_button_state()
    if callable(set_progress_text):
        set_progress_text(success_text)
    return True


def undo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_undo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_undo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Restore the previous geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_undo_state,
        restore_state=restore_state,
        commit_transition=commit_undo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to undo.",
        failure_prefix="Failed to undo geometry fit",
        success_text="Restored the previous geometry-fit state.",
    )


def redo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_redo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_redo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Reapply the next geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_redo_state,
        restore_state=restore_state,
        commit_transition=commit_redo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to redo.",
        failure_prefix="Failed to redo geometry fit",
        success_text="Reapplied the next geometry-fit state.",
    )


def build_runtime_geometry_fit_history_callbacks(
    *,
    history_state: Any,
    capture_current_state: Callable[[], dict[str, object]],
    restore_state: Callable[[dict[str, object]], object],
    copy_state_value: Callable[[object], object],
    history_limit: Callable[[], object] | object,
    peek_last_undo_state: Callable[..., dict[str, object] | None],
    peek_last_redo_state: Callable[..., dict[str, object] | None],
    commit_undo: Callable[..., None],
    commit_redo: Callable[..., None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryFitRuntimeHistoryCallbacks:
    """Build shared runtime undo/redo callbacks around one geometry-fit history store."""

    def _undo() -> bool:
        return undo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "undo_stack", [])),
            capture_current_state=capture_current_state,
            read_undo_state=(
                lambda: peek_last_undo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_undo=(
                lambda current_state: commit_undo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    def _redo() -> bool:
        return redo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "redo_stack", [])),
            capture_current_state=capture_current_state,
            read_redo_state=(
                lambda: peek_last_redo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_redo=(
                lambda current_state: commit_redo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    return GeometryFitRuntimeHistoryCallbacks(
        undo=_undo,
        redo=_redo,
    )


def make_runtime_geometry_tool_action_callbacks(
    *,
    geometry_fit_history_state: Any,
    manual_pick_armed: Callable[[], bool] | bool,
    set_manual_pick_armed: Callable[[bool], None],
    current_background_index: Callable[[], object] | object,
    current_pick_session: Callable[[], object] | object,
    manual_pick_session_active: Callable[[], bool] | bool,
    build_manual_pick_button_label: Callable[..., str],
    pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    pair_group_count: Callable[[int], int],
    set_manual_pick_text: Callable[[str], None] | None = None,
    set_history_button_state: Callable[[bool, bool], None] | None = None,
    show_caked_2d_var: Any = None,
    toggle_caked_2d: Callable[[], None] | None = None,
    ensure_geometry_fit_caked_view: Callable[[], None] | None = None,
    set_hkl_pick_mode: Callable[..., None] | None = None,
    set_geometry_preview_exclude_mode: Callable[..., None] | None = None,
    cancel_manual_pick_session: Callable[..., None] | None = None,
    canvas_widget: Callable[[], Any] | Any = None,
    push_manual_undo_state: Callable[[], None] | None = None,
    clear_pairs_for_current_background: Callable[[int], None] | None = None,
    clear_geometry_pick_artists: Callable[[], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryToolActionRuntimeCallbacks:
    """Build the live geometry tool action callbacks around shared helpers."""

    def _current_background() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _manual_pick_armed() -> bool:
        return bool(_resolve_runtime_value(manual_pick_armed))

    def _manual_pick_session_is_active() -> bool:
        active = (
            manual_pick_session_active()
            if callable(manual_pick_session_active)
            else manual_pick_session_active
        )
        return bool(active)

    def _update_fit_history_button_state() -> None:
        set_runtime_geometry_fit_history_button_state(
            can_undo=bool(getattr(geometry_fit_history_state, "undo_stack", [])),
            can_redo=bool(getattr(geometry_fit_history_state, "redo_stack", [])),
            set_button_state=set_history_button_state,
        )

    def _update_manual_pick_button_label() -> None:
        label = build_manual_pick_button_label(
            armed=_manual_pick_armed(),
            current_background_index=_current_background(),
            pick_session=_resolve_runtime_value(current_pick_session),
            pairs_for_index=pairs_for_index,
            pair_group_count=pair_group_count,
        )
        if callable(set_manual_pick_text):
            set_manual_pick_text(str(label))

    def _set_manual_pick_mode(enabled: bool, message: str | None = None) -> None:
        armed = bool(enabled)
        set_manual_pick_armed(armed)

        if armed:
            if callable(ensure_geometry_fit_caked_view):
                ensure_geometry_fit_caked_view()
            else:
                show_caked_2d_var_local = _resolve_runtime_value(show_caked_2d_var)
                show_caked_2d = False
                getter = getattr(show_caked_2d_var_local, "get", None)
                if callable(getter):
                    try:
                        show_caked_2d = bool(getter())
                    except Exception:
                        show_caked_2d = False
                if not show_caked_2d:
                    setter = getattr(show_caked_2d_var_local, "set", None)
                    if callable(setter):
                        try:
                            setter(True)
                        except Exception:
                            pass
                    if callable(toggle_caked_2d):
                        toggle_caked_2d()
            if callable(set_hkl_pick_mode):
                set_hkl_pick_mode(False)
            if callable(set_geometry_preview_exclude_mode):
                set_geometry_preview_exclude_mode(False)
        elif callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=True)

        _update_manual_pick_button_label()

        widget = _resolve_runtime_value(canvas_widget)
        configure = getattr(widget, "configure", None)
        if callable(configure):
            try:
                configure(cursor="crosshair" if armed else "")
            except Exception:
                pass

        if message and callable(set_progress_text):
            set_progress_text(message)

    def _toggle_manual_pick_mode() -> None:
        armed = _manual_pick_armed()
        _set_manual_pick_mode(
            not armed,
            message=(
                (
                    "Manual geometry picking armed in 2D caked phi-vs-2theta view. "
                    "Click a Qr/Qz set once, then click the matching background peaks "
                    "for each simulated member of that set."
                )
                if not armed
                else "Manual geometry picking disabled."
            ),
        )

    def _clear_current_manual_pairs() -> None:
        background_index = _current_background()
        if pairs_for_index(background_index) or _manual_pick_session_is_active():
            if callable(push_manual_undo_state):
                push_manual_undo_state()
        if callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=False)
        if callable(clear_pairs_for_current_background):
            clear_pairs_for_current_background(background_index)
        if callable(clear_geometry_pick_artists):
            clear_geometry_pick_artists()
        _update_manual_pick_button_label()
        if callable(refresh_status):
            refresh_status()
        if callable(set_progress_text):
            set_progress_text(
                "Cleared saved geometry pairs for the current background image."
            )

    return GeometryToolActionRuntimeCallbacks(
        update_fit_history_button_state=_update_fit_history_button_state,
        update_manual_pick_button_label=_update_manual_pick_button_label,
        set_manual_pick_mode=_set_manual_pick_mode,
        toggle_manual_pick_mode=_toggle_manual_pick_mode,
        clear_current_manual_pairs=_clear_current_manual_pairs,
    )


def build_geometry_manual_fit_dataset(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    orientation_cfg: Mapping[str, object] | None = None,
    stage_callback: GeometryFitStageCallback | None = None,
) -> dict[str, object]:
    """Build one saved-manual-pair geometry dataset for the optimizer."""

    background_idx = int(background_index)
    _emit_geometry_fit_stage_event(
        stage_callback,
        "dataset_start",
        background_index=int(background_idx),
        message=f"preflight: building dataset for background {int(background_idx) + 1}",
    )

    def _finite_float(value: object) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out):
            return None
        return float(out)

    def _caked_angle_pair(
        entry: Mapping[str, object] | None,
        *,
        x_keys: Sequence[str],
        y_keys: Sequence[str],
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None
        two_theta_value: float | None = None
        phi_value: float | None = None
        for key in x_keys:
            two_theta_value = _finite_float(entry.get(key))
            if two_theta_value is not None:
                break
        for key in y_keys:
            phi_value = _finite_float(entry.get(key))
            if phi_value is not None:
                break
        if two_theta_value is None or phi_value is None:
            return None
        return float(two_theta_value), float(phi_value)

    def _reference_two_theta_deg(
        entry: Mapping[str, object] | None,
        *,
        a_lattice: float | None,
        c_lattice: float | None,
        wavelength: float | None,
    ) -> float | None:
        if not isinstance(entry, Mapping):
            return None
        hkl = entry.get("hkl")
        if (
            isinstance(hkl, (list, tuple, np.ndarray))
            and len(hkl) >= 3
            and a_lattice is not None
            and c_lattice is not None
            and wavelength is not None
        ):
            try:
                spacing = d_spacing(
                    int(hkl[0]),
                    int(hkl[1]),
                    int(hkl[2]),
                    float(a_lattice),
                    float(c_lattice),
                )
                if spacing is not None:
                    value = two_theta(float(spacing), float(wavelength))
                    finite_value = _finite_float(value)
                    if finite_value is not None:
                        return float(finite_value)
            except Exception:
                pass

        qr_value = _finite_float(entry.get("qr"))
        qz_value = _finite_float(entry.get("qz"))
        if qr_value is None or qz_value is None or wavelength is None:
            return None
        q_mag = math.hypot(float(qr_value), float(qz_value))
        if not (np.isfinite(q_mag) and q_mag > 0.0):
            return None
        arg = float(q_mag) * float(wavelength) / (4.0 * np.pi)
        if not np.isfinite(arg) or abs(arg) > 1.0:
            return None
        return float(2.0 * np.degrees(np.arcsin(arg)))

    use_caked_display = False
    if callable(manual_dataset_bindings.pick_uses_caked_space):
        try:
            use_caked_display = bool(manual_dataset_bindings.pick_uses_caked_space())
        except Exception:
            use_caked_display = False
    selected_entries = list(
        manual_dataset_bindings.geometry_manual_pairs_for_index(background_idx) or ()
    )
    if not selected_entries:
        raise RuntimeError(
            f"background {background_idx + 1} has no saved manual geometry pairs"
        )

    native_background, display_background = (
        manual_dataset_bindings.load_background_by_index(background_idx)
    )

    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    params_i["theta_initial"] = float(theta_base + theta_offset)
    reference_a = _finite_float(params_i.get("a"))
    reference_c = _finite_float(params_i.get("c"))
    reference_lambda = _finite_float(params_i.get("lambda"))

    def _current_simulation_diagnostics() -> dict[str, object]:
        if callable(
            manual_dataset_bindings.geometry_manual_last_source_snapshot_diagnostics
        ):
            raw_value = (
                manual_dataset_bindings.geometry_manual_last_source_snapshot_diagnostics()
            )
        elif callable(manual_dataset_bindings.geometry_manual_last_simulation_diagnostics):
            raw_value = (
                manual_dataset_bindings.geometry_manual_last_simulation_diagnostics()
            )
        else:
            raw_value = {}
        copied = copy.deepcopy(raw_value)
        return copied if isinstance(copied, dict) else {}

    if callable(manual_dataset_bindings.geometry_manual_source_rows_for_background):
        simulated_peaks = (
            manual_dataset_bindings.geometry_manual_source_rows_for_background(
                int(background_idx),
                params_i,
                consumer="geometry_fit_dataset",
            )
        )
    else:
        simulated_peaks = (
            manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
                params_i,
                prefer_cache=True,
            )
        )
    simulation_diagnostics = _current_simulation_diagnostics()
    if (
        not simulated_peaks
        and callable(
            manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background
        )
    ):
        _emit_geometry_fit_stage_event(
            stage_callback,
            "source_snapshot_rebuild",
            background_index=int(background_idx),
            message=(
                "preflight: rebuilding simulated source rows for "
                f"background {int(background_idx) + 1}"
            ),
        )
        simulated_peaks = (
            manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background(
                int(background_idx),
                params_i,
                consumer="geometry_fit_dataset",
                prior_diagnostics=dict(simulation_diagnostics),
            )
        )
        simulation_diagnostics = _current_simulation_diagnostics()
    if not simulated_peaks:
        snapshot_status = str(simulation_diagnostics.get("status", "<unknown>"))
        _emit_geometry_fit_stage_event(
            stage_callback,
            "dataset_failed",
            background_index=int(background_idx),
            status=snapshot_status,
            message=(
                "preflight: source snapshot unavailable for "
                f"background {int(background_idx) + 1}"
            ),
        )
        raise RuntimeError(
            "Geometry-fit source snapshot unavailable for "
            f"background {int(background_idx)} (status={snapshot_status})."
        )
    simulated_lookup = manual_dataset_bindings.geometry_manual_simulated_lookup(
        simulated_peaks
    )

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

    def _source_peak_key(
        entry: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        if not isinstance(entry, Mapping):
            return None
        try:
            return (
                int(entry.get("source_table_index")),
                int(entry.get("source_peak_index")),
            )
        except Exception:
            return None

    def _normalized_hkl(
        value: object,
    ) -> tuple[int, int, int] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
            return None
        try:
            return (
                int(value[0]),
                int(value[1]),
                int(value[2]),
            )
        except Exception:
            return None

    simulated_by_peak: dict[tuple[int, int], dict[str, object]] = {}
    simulated_by_q_group: dict[object, list[dict[str, object]]] = {}
    simulated_by_hkl: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        peak_key = _source_peak_key(entry)
        if peak_key is not None:
            simulated_by_peak[peak_key] = entry
        q_group_key = entry.get("q_group_key")
        if q_group_key is not None:
            simulated_by_q_group.setdefault(q_group_key, []).append(entry)
        hkl_key = _normalized_hkl(entry.get("hkl"))
        if hkl_key is not None:
            simulated_by_hkl.setdefault(hkl_key, []).append(entry)

    def _entry_display_point(
        entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        coords = (
            manual_dataset_bindings.geometry_manual_entry_display_coords(entry)
            if isinstance(entry, Mapping)
            else None
        )
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

    def _source_entry_hkl_matches(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        entry_hkl = _normalized_hkl(entry.get("hkl") if isinstance(entry, Mapping) else None)
        candidate_hkl = _normalized_hkl(
            candidate.get("hkl") if isinstance(candidate, Mapping) else None
        )
        return (
            entry_hkl is None
            or candidate_hkl is None
            or tuple(int(v) for v in candidate_hkl) == tuple(int(v) for v in entry_hkl)
        )

    def _resolve_cached_source_entry(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(entry, Mapping):
            return None
        row_key = _source_row_key(entry)
        if row_key is not None:
            candidate = simulated_lookup.get(row_key)
            if isinstance(candidate, Mapping):
                resolved = dict(candidate)
                resolved.setdefault("source_table_index", int(row_key[0]))
                resolved.setdefault("source_row_index", int(row_key[1]))
                if _source_entry_hkl_matches(entry, resolved):
                    return resolved
        peak_key = _source_peak_key(entry)
        if peak_key is not None:
            candidate = simulated_by_peak.get(peak_key)
            if candidate is not None:
                resolved = dict(candidate)
                resolved.setdefault("source_table_index", int(peak_key[0]))
                resolved.setdefault("source_peak_index", int(peak_key[1]))
                if _source_entry_hkl_matches(entry, resolved):
                    return resolved
        return None

    def _resolve_source_entry(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        resolved = _resolve_cached_source_entry(entry)
        if isinstance(resolved, dict):
            return resolved

        candidate_pool: list[dict[str, object]] = []
        q_group_key = entry.get("q_group_key")
        if q_group_key is not None:
            candidate_pool = [dict(item) for item in simulated_by_q_group.get(q_group_key, ())]
        if not candidate_pool:
            hkl_key = _normalized_hkl(entry.get("hkl"))
            if hkl_key is not None:
                candidate_pool = [dict(item) for item in simulated_by_hkl.get(hkl_key, ())]
        if not candidate_pool:
            return None
        if len(candidate_pool) == 1:
            return dict(candidate_pool[0])

        display_point = _entry_display_point(entry)
        if display_point is None:
            return dict(candidate_pool[0])

        def _distance_sq(candidate: Mapping[str, object]) -> float:
            try:
                sim_col = float(candidate.get("sim_col"))
                sim_row = float(candidate.get("sim_row"))
            except Exception:
                return float("inf")
            if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
                return float("inf")
            return float(
                (sim_col - float(display_point[0])) ** 2
                + (sim_row - float(display_point[1])) ** 2
            )

        candidate_pool.sort(key=_distance_sq)
        return dict(candidate_pool[0]) if candidate_pool else None

    def _normalized_q_group_key(
        value: object,
    ) -> tuple[object, ...] | None:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return None

    def _resolved_source_distance_sq(
        entry: Mapping[str, object],
        resolved_source_entry: Mapping[str, object] | None,
    ) -> float:
        display_point = _entry_display_point(entry)
        if display_point is None or not isinstance(resolved_source_entry, Mapping):
            return float("inf")
        try:
            sim_col = float(resolved_source_entry.get("sim_col"))
            sim_row = float(resolved_source_entry.get("sim_row"))
        except Exception:
            return float("inf")
        if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
            return float("inf")
        return float(
            (sim_col - float(display_point[0])) ** 2
            + (sim_row - float(display_point[1])) ** 2
        )

    def _source_candidate_status(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
        *,
        key_kind: str,
    ) -> str:
        if key_kind == "row":
            if _source_row_key(entry) is None:
                return "no_saved_row"
        elif key_kind == "peak":
            if _source_peak_key(entry) is None:
                return "no_saved_peak"
        if not isinstance(candidate, Mapping):
            return "missing"
        if _source_entry_hkl_matches(entry, candidate):
            return "matched"
        return "hkl_mismatch"

    def _resolved_source_kind(
        entry: Mapping[str, object] | None,
        resolved_source_entry: Mapping[str, object] | None,
    ) -> str | None:
        if not (isinstance(entry, Mapping) and isinstance(resolved_source_entry, Mapping)):
            return None
        row_key = _source_row_key(entry)
        if row_key is not None and row_key == _source_row_key(resolved_source_entry):
            return "source_row"
        peak_key = _source_peak_key(entry)
        if peak_key is not None and peak_key == _source_peak_key(resolved_source_entry):
            return "source_peak"
        q_group_key = entry.get("q_group_key")
        if q_group_key is not None and q_group_key == resolved_source_entry.get("q_group_key"):
            return "q_group_fallback"
        if _source_entry_hkl_matches(entry, resolved_source_entry):
            return "hkl_fallback"
        return "override"

    def _uses_legacy_dense_source_identity(
        entry: Mapping[str, object] | None,
    ) -> bool:
        row_key = _source_row_key(entry)
        peak_key = _source_peak_key(entry)
        if row_key is None or peak_key is None:
            return False
        try:
            return (
                int(row_key[1]) == 0
                and int(peak_key[0]) == int(row_key[0])
                and int(peak_key[1]) == int(row_key[0])
            )
        except Exception:
            return False

    def _promoted_source_entry(
        entry: Mapping[str, object] | None,
        overlay_source_entry: Mapping[str, object] | None,
        *,
        overlay_kind: str | None,
    ) -> tuple[dict[str, object] | None, str | None]:
        if not (
            _uses_legacy_dense_source_identity(entry)
            and isinstance(overlay_source_entry, Mapping)
            and overlay_kind in {"q_group_fallback", "hkl_fallback"}
            and _source_entry_hkl_matches(entry, overlay_source_entry)
        ):
            return None, None
        entry_group_key = _normalized_q_group_key(
            entry.get("q_group_key") if isinstance(entry, Mapping) else None
        )
        overlay_group_key = _normalized_q_group_key(
            overlay_source_entry.get("q_group_key")
        )
        if entry_group_key is not None and overlay_group_key != entry_group_key:
            return None, None
        promoted = dict(overlay_source_entry)
        promoted_kind = (
            "legacy_dense_q_group_rebind"
            if overlay_kind == "q_group_fallback"
            else "legacy_dense_hkl_rebind"
        )
        return promoted, promoted_kind

    def _strict_failure_reason(
        *,
        row_status: str,
        peak_status: str,
        overlay_kind: str | None,
    ) -> str:
        reasons: list[str] = []
        if row_status == "no_saved_row" and peak_status == "no_saved_peak":
            reasons.append("saved pair has no cached source row/peak identity")
        if row_status == "missing":
            reasons.append("saved source row is missing from the current simulated rows")
        elif row_status == "hkl_mismatch":
            reasons.append("saved source row HKL no longer matches the current row")
        if peak_status == "missing":
            reasons.append("saved source peak is missing from the current simulated peaks")
        elif peak_status == "hkl_mismatch":
            reasons.append("saved source peak HKL no longer matches the current peak")
        if overlay_kind == "q_group_fallback":
            reasons.append("only a q-group fallback candidate was available")
        elif overlay_kind == "hkl_fallback":
            reasons.append("only an HKL fallback candidate was available")
        elif overlay_kind == "override":
            reasons.append("only an override/refined fallback candidate was available")
        if not reasons:
            reasons.append("strict cached-source lookup returned no acceptable candidate")
        return "; ".join(reasons)

    selected_records: list[dict[str, object]] = []
    for raw_entry in selected_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
        if q_group_key is not None:
            entry["q_group_key"] = q_group_key
        row_key = _source_row_key(entry)
        row_candidate = (
            dict(simulated_lookup.get(row_key))
            if row_key is not None and isinstance(simulated_lookup.get(row_key), Mapping)
            else None
        )
        peak_key = _source_peak_key(entry)
        peak_candidate = (
            dict(simulated_by_peak.get(peak_key))
            if peak_key is not None and isinstance(simulated_by_peak.get(peak_key), Mapping)
            else None
        )
        strict_source_entry = _resolve_cached_source_entry(entry)
        overlay_source_entry = _resolve_source_entry(entry)
        overlay_source_entry = gui_manual_geometry.geometry_manual_apply_refined_simulated_override(
            entry,
            dict(overlay_source_entry)
            if isinstance(overlay_source_entry, Mapping)
            else None,
            prefer_caked_display=use_caked_display,
        )
        overlay_kind = _resolved_source_kind(entry, overlay_source_entry)
        fit_source_entry = (
            dict(strict_source_entry)
            if isinstance(strict_source_entry, Mapping)
            else None
        )
        fit_resolution_kind = _resolved_source_kind(entry, fit_source_entry)
        if fit_source_entry is None:
            promoted_source_entry, promoted_resolution_kind = _promoted_source_entry(
                entry,
                overlay_source_entry,
                overlay_kind=overlay_kind,
            )
            if isinstance(promoted_source_entry, Mapping):
                fit_source_entry = dict(promoted_source_entry)
                fit_resolution_kind = str(
                    promoted_resolution_kind or "legacy_dense_source_rebind"
                )
        selected_records.append(
            {
                "entry": entry,
                "strict_source_entry": dict(strict_source_entry)
                if isinstance(strict_source_entry, Mapping)
                else None,
                "fit_source_entry": dict(fit_source_entry)
                if isinstance(fit_source_entry, Mapping)
                else None,
                "fit_resolution_kind": fit_resolution_kind,
                "overlay_source_entry": dict(overlay_source_entry)
                if isinstance(overlay_source_entry, Mapping)
                else None,
                "overlay_resolution_kind": overlay_kind,
                "row_candidate": dict(row_candidate)
                if isinstance(row_candidate, Mapping)
                else None,
                "peak_candidate": dict(peak_candidate)
                if isinstance(peak_candidate, Mapping)
                else None,
                "row_candidate_status": _source_candidate_status(
                    entry,
                    row_candidate,
                    key_kind="row",
                ),
                "peak_candidate_status": _source_candidate_status(
                    entry,
                    peak_candidate,
                    key_kind="peak",
                ),
            }
        )
    resolved_source_pair_count = int(
        sum(
            1
            for record in selected_records
            if isinstance(record.get("fit_source_entry"), Mapping)
        )
    )

    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    source_resolution_diagnostics: list[dict[str, object]] = []
    selected_entries = [
        dict(record["entry"])
        for record in selected_records
        if isinstance(record.get("entry"), Mapping)
    ]
    for pair_idx, record in enumerate(selected_records):
        entry = (
            dict(record.get("entry"))
            if isinstance(record.get("entry"), Mapping)
            else None
        )
        if entry is None:
            continue
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        strict_source_entry = (
            dict(record.get("strict_source_entry"))
            if isinstance(record.get("strict_source_entry"), Mapping)
            else None
        )
        fit_source_entry = (
            dict(record.get("fit_source_entry"))
            if isinstance(record.get("fit_source_entry"), Mapping)
            else None
        )
        row_candidate = (
            dict(record.get("row_candidate"))
            if isinstance(record.get("row_candidate"), Mapping)
            else None
        )
        peak_candidate = (
            dict(record.get("peak_candidate"))
            if isinstance(record.get("peak_candidate"), Mapping)
            else None
        )
        overlay_source_entry = (
            dict(record.get("overlay_source_entry"))
            if isinstance(record.get("overlay_source_entry"), Mapping)
            else _resolve_source_entry(measured_entry)
        )
        row_candidate_status = str(record.get("row_candidate_status", "") or "")
        peak_candidate_status = str(record.get("peak_candidate_status", "") or "")
        overlay_distance_sq = _resolved_source_distance_sq(entry, overlay_source_entry)
        overlay_distance_px = (
            float(np.sqrt(overlay_distance_sq))
            if np.isfinite(float(overlay_distance_sq))
            else float("nan")
        )
        overlay_kind = str(record.get("overlay_resolution_kind", "") or "") or _resolved_source_kind(
            entry,
            overlay_source_entry,
        )
        source_resolution_diagnostics.append(
            {
                "pair_index": int(pair_idx),
                "saved_source_table_index": entry.get("source_table_index"),
                "saved_source_row_index": entry.get("source_row_index"),
                "saved_source_peak_index": entry.get("source_peak_index"),
                "saved_source_row_key": _source_row_key(entry),
                "saved_source_peak_key": _source_peak_key(entry),
                "saved_hkl": _normalized_hkl(entry.get("hkl")),
                "saved_q_group_key": entry.get("q_group_key"),
                "saved_display_point": _entry_display_point(entry),
                "strict_resolved": isinstance(strict_source_entry, Mapping),
                "fit_resolved": isinstance(fit_source_entry, Mapping),
                "fit_resolution_kind": str(
                    record.get("fit_resolution_kind", "") or ""
                )
                or None,
                "resolution_kind": str(
                    record.get("fit_resolution_kind", "") or ""
                )
                or None,
                "fit_source_row_key": _source_row_key(fit_source_entry),
                "fit_source_peak_key": _source_peak_key(fit_source_entry),
                "strict_resolution_kind": _resolved_source_kind(entry, strict_source_entry),
                "row_candidate_status": row_candidate_status,
                "row_candidate_hkl": _normalized_hkl(
                    row_candidate.get("hkl") if isinstance(row_candidate, Mapping) else None
                ),
                "peak_candidate_status": peak_candidate_status,
                "peak_candidate_hkl": _normalized_hkl(
                    peak_candidate.get("hkl")
                    if isinstance(peak_candidate, Mapping)
                    else None
                ),
                "overlay_resolution_kind": overlay_kind,
                "overlay_source_row_key": _source_row_key(overlay_source_entry),
                "overlay_source_peak_key": _source_peak_key(overlay_source_entry),
                "overlay_hkl": _normalized_hkl(
                    overlay_source_entry.get("hkl")
                    if isinstance(overlay_source_entry, Mapping)
                    else None
                ),
                "overlay_distance_px": (
                    float(overlay_distance_px)
                    if np.isfinite(float(overlay_distance_px))
                    else None
                ),
                "failure_reason": (
                    None
                    if isinstance(fit_source_entry, Mapping)
                    else _strict_failure_reason(
                        row_status=row_candidate_status,
                        peak_status=peak_candidate_status,
                        overlay_kind=overlay_kind,
                    )
                ),
            }
        )
        for key in ("source_table_index", "source_row_index", "source_peak_index"):
            measured_entry.pop(key, None)
        if isinstance(fit_source_entry, Mapping):
            for key in ("source_table_index", "source_row_index", "source_peak_index"):
                if key in fit_source_entry:
                    measured_entry[key] = fit_source_entry.get(key)
        measured_entry["fit_source_identity_only"] = True
        if record.get("fit_resolution_kind") is not None:
            measured_entry["fit_source_resolution_kind"] = record.get(
                "fit_resolution_kind"
            )
        measured_display.append(measured_entry)

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": entry.get("hkl", entry.get("label")),
        }
        raw_group_key = entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            initial_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            initial_entry["q_group_key"] = tuple(raw_group_key)
        bg_coords = manual_dataset_bindings.geometry_manual_entry_display_coords(entry)
        if bg_coords is not None and len(bg_coords) >= 2:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
            if use_caked_display:
                initial_entry["bg_caked_display"] = (
                    float(bg_coords[0]),
                    float(bg_coords[1]),
                )
        background_angles = _caked_angle_pair(
            entry,
            x_keys=(
                "background_two_theta_deg",
                "caked_x",
                "raw_caked_x",
            ),
            y_keys=(
                "background_phi_deg",
                "caked_y",
                "raw_caked_y",
            ),
        )
        if background_angles is not None:
            initial_entry["background_two_theta_deg"] = float(background_angles[0])
            initial_entry["background_phi_deg"] = float(background_angles[1])
        reference_two_theta = (
            _finite_float(entry.get("background_reference_two_theta_deg"))
            or _reference_two_theta_deg(
                entry,
                a_lattice=reference_a,
                c_lattice=reference_c,
                wavelength=reference_lambda,
            )
        )
        if reference_two_theta is not None:
            initial_entry["background_reference_two_theta_deg"] = float(
                reference_two_theta
            )
        if reference_a is not None:
            initial_entry["background_reference_a"] = float(reference_a)
        if reference_c is not None:
            initial_entry["background_reference_c"] = float(reference_c)
        if reference_lambda is not None:
            initial_entry["background_reference_lambda"] = float(reference_lambda)
        for source_key, target_key in (
            ("qr", "background_reference_qr"),
            ("qz", "background_reference_qz"),
        ):
            value = _finite_float(entry.get(source_key))
            if value is not None:
                initial_entry[target_key] = float(value)
        if isinstance(fit_source_entry, Mapping):
            for key in ("source_table_index", "source_row_index", "source_peak_index"):
                if key in fit_source_entry:
                    initial_entry[key] = fit_source_entry.get(key)
        if isinstance(overlay_source_entry, Mapping):
            try:
                sim_col = float(overlay_source_entry.get("sim_col"))
                sim_row = float(overlay_source_entry.get("sim_row"))
            except Exception:
                sim_col = float("nan")
                sim_row = float("nan")
            if np.isfinite(sim_col) and np.isfinite(sim_row):
                initial_entry["sim_display"] = (float(sim_col), float(sim_row))
                if use_caked_display:
                    initial_entry["sim_caked_display"] = (
                        float(sim_col),
                        float(sim_row),
                    )
            simulated_angles = _caked_angle_pair(
                overlay_source_entry,
                x_keys=("two_theta_deg", "caked_x"),
                y_keys=("phi_deg", "caked_y"),
            )
            if simulated_angles is not None:
                initial_entry["simulated_two_theta_deg"] = float(simulated_angles[0])
                initial_entry["simulated_phi_deg"] = float(simulated_angles[1])
            try:
                sim_col_raw = float(overlay_source_entry.get("sim_col_raw", sim_col))
                sim_row_raw = float(overlay_source_entry.get("sim_row_raw", sim_row))
            except Exception:
                sim_col_raw = float("nan")
                sim_row_raw = float("nan")
            if np.isfinite(sim_col_raw) and np.isfinite(sim_row_raw):
                try:
                    sim_native = manual_dataset_bindings.display_to_native_sim_coords(
                        float(sim_col_raw),
                        float(sim_row_raw),
                        (
                            int(manual_dataset_bindings.image_size),
                            int(manual_dataset_bindings.image_size),
                        ),
                    )
                except Exception:
                    sim_native = None
                if (
                    isinstance(sim_native, tuple)
                    and len(sim_native) >= 2
                    and np.isfinite(float(sim_native[0]))
                    and np.isfinite(float(sim_native[1]))
                ):
                    initial_entry["sim_native"] = (
                        float(sim_native[0]),
                        float(sim_native[1]),
                    )
        initial_pairs_display.append(initial_entry)

    measured_native = manual_dataset_bindings.unrotate_display_peaks(
        measured_display,
        display_background.shape,
        k=manual_dataset_bindings.display_rotate_k,
    )
    for original_entry, initial_entry, measured_entry in zip(
        measured_display,
        initial_pairs_display,
        measured_native,
    ):
        if not isinstance(measured_entry, dict):
            continue
        detector_anchor = None
        try:
            detector_anchor = (
                float(original_entry.get("detector_x")),
                float(original_entry.get("detector_y")),
            )
        except Exception:
            detector_anchor = None
        if (
            isinstance(detector_anchor, tuple)
            and len(detector_anchor) >= 2
            and np.isfinite(float(detector_anchor[0]))
            and np.isfinite(float(detector_anchor[1]))
        ):
            measured_entry["x"] = float(detector_anchor[0])
            measured_entry["y"] = float(detector_anchor[1])
        try:
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if np.isfinite(mx) and np.isfinite(my):
            measured_entry["detector_x"] = float(mx)
            measured_entry["detector_y"] = float(my)
            initial_entry["bg_native"] = (float(mx), float(my))

    sim_orientation_points: list[tuple[float, float]] = []
    meas_orientation_points: list[tuple[float, float]] = []
    sim_native_shape = (
        int(manual_dataset_bindings.image_size),
        int(manual_dataset_bindings.image_size),
    )
    for initial_entry, measured_entry in zip(initial_pairs_display, measured_native):
        if not isinstance(measured_entry, Mapping):
            continue
        sim_display = initial_entry.get("sim_display")
        if (
            not isinstance(sim_display, (list, tuple, np.ndarray))
            or len(sim_display) < 2
        ):
            continue
        try:
            sim_native_raw = initial_entry.get("sim_native")
            if (
                isinstance(sim_native_raw, (list, tuple, np.ndarray))
                and len(sim_native_raw) >= 2
            ):
                sim_native = (
                    float(sim_native_raw[0]),
                    float(sim_native_raw[1]),
                )
            else:
                sim_native = manual_dataset_bindings.display_to_native_sim_coords(
                    float(sim_display[0]),
                    float(sim_display[1]),
                    sim_native_shape,
                )
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if not (
            np.isfinite(sim_native[0])
            and np.isfinite(sim_native[1])
            and np.isfinite(mx)
            and np.isfinite(my)
        ):
            continue
        sim_orientation_points.append((float(sim_native[0]), float(sim_native[1])))
        meas_orientation_points.append((float(mx), float(my)))

    orientation_choice, orientation_diag = (
        manual_dataset_bindings.select_fit_orientation(
            sim_orientation_points,
            meas_orientation_points,
            tuple(int(v) for v in native_background.shape[:2]),
            cfg=orientation_cfg or {},
        )
    )
    measured_for_fit = manual_dataset_bindings.apply_orientation_to_entries(
        measured_native,
        native_background.shape,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    for measured_entry in measured_for_fit:
        if not isinstance(measured_entry, dict):
            continue
        try:
            detector_x = float(measured_entry.get("x"))
            detector_y = float(measured_entry.get("y"))
        except Exception:
            continue
        if np.isfinite(detector_x) and np.isfinite(detector_y):
            measured_entry["detector_x"] = float(detector_x)
            measured_entry["detector_y"] = float(detector_y)
        measured_entry["fit_source_identity_only"] = True
    for measured_entry, initial_entry in zip(measured_for_fit, initial_pairs_display):
        if not isinstance(measured_entry, dict) or not isinstance(initial_entry, Mapping):
            continue
        for key in (
            "overlay_match_index",
            "q_group_key",
            "source_table_index",
            "source_row_index",
            "source_peak_index",
            "background_two_theta_deg",
            "background_phi_deg",
            "background_reference_two_theta_deg",
            "background_reference_a",
            "background_reference_c",
            "background_reference_lambda",
            "background_reference_qr",
            "background_reference_qz",
        ):
            if key in initial_entry:
                measured_entry[key] = initial_entry.get(key)
    backend_background = manual_dataset_bindings.apply_background_backend_orientation(
        native_background
    )
    if backend_background is None:
        backend_background = native_background
    experimental_image_for_fit = manual_dataset_bindings.orient_image_for_fit(
        backend_background,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    dynamic_reanchor_callback = None
    dynamic_reanchor_match_cfg: dict[str, object] = {}
    dynamic_reanchor_background_context: dict[str, object] | None = None
    dynamic_reanchor_enabled = (
        isinstance(experimental_image_for_fit, np.ndarray)
        and experimental_image_for_fit.ndim == 2
        and experimental_image_for_fit.size > 0
    )
    if callable(manual_dataset_bindings.geometry_manual_match_config):
        try:
            dynamic_reanchor_match_cfg = dict(
                manual_dataset_bindings.geometry_manual_match_config() or {}
            )
        except Exception:
            dynamic_reanchor_match_cfg = {}
    if dynamic_reanchor_enabled:
        try:
            built_background_context = build_background_peak_context(
                experimental_image_for_fit,
                dict(dynamic_reanchor_match_cfg),
            )
        except Exception:
            built_background_context = None
        if isinstance(built_background_context, Mapping):
            dynamic_reanchor_background_context = dict(built_background_context)

        dynamic_reanchor_cache_data = {
            "match_config": dict(dynamic_reanchor_match_cfg),
            "background_context": dynamic_reanchor_background_context,
        }
        dynamic_reanchor_image = np.asarray(experimental_image_for_fit, dtype=np.float64)

        def _dynamic_reanchor_callback(
            measured_entry: Mapping[str, object] | None,
            simulated_detector_point: object,
            local_params: Mapping[str, object] | None = None,
            dataset_ctx: object = None,
        ) -> dict[str, object] | None:
            del local_params, dataset_ctx
            if not isinstance(measured_entry, Mapping):
                return None
            if not isinstance(
                simulated_detector_point,
                (list, tuple, np.ndarray),
            ) or len(simulated_detector_point) < 2:
                return None
            try:
                sim_col = float(simulated_detector_point[0])
                sim_row = float(simulated_detector_point[1])
            except Exception:
                return None
            if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
                return None

            seed_entry = dict(measured_entry)
            seed_entry["sim_col"] = float(sim_col)
            seed_entry["sim_row"] = float(sim_row)
            try:
                refined_col, refined_row = (
                    gui_manual_geometry.geometry_manual_refine_preview_point(
                        seed_entry,
                        float(sim_col),
                        float(sim_row),
                        display_background=dynamic_reanchor_image,
                        cache_data=dynamic_reanchor_cache_data,
                        use_caked_space=False,
                        match_simulated_peaks_to_peak_context=(
                            match_simulated_peaks_to_peak_context
                        ),
                    )
                )
            except Exception:
                return None
            if not (np.isfinite(refined_col) and np.isfinite(refined_row)):
                return None
            return {
                "x": float(refined_col),
                "y": float(refined_row),
                "detector_x": float(refined_col),
                "detector_y": float(refined_row),
                "background_two_theta_deg": float("nan"),
                "background_phi_deg": float("nan"),
            }

        dynamic_reanchor_callback = _dynamic_reanchor_callback

    label = (
        Path(str(manual_dataset_bindings.osc_files[background_idx])).name
        if 0 <= background_idx < len(manual_dataset_bindings.osc_files)
        else f"background_{background_idx}"
    )
    group_count = len(
        {
            entry.get("q_group_key")
            for entry in selected_entries
            if entry.get("q_group_key") is not None
        }
    )
    cache_metadata = build_geometry_fit_dataset_cache_metadata(
        background_index=int(background_idx),
        current_background_index=int(manual_dataset_bindings.current_background_index),
        simulated_peaks=simulated_peaks,
        source_snapshot_diagnostics=simulation_diagnostics,
        source_resolution_diagnostics=source_resolution_diagnostics,
        pair_count=int(len(measured_display)),
        resolved_source_pair_count=int(resolved_source_pair_count),
    )
    _emit_geometry_fit_stage_event(
        stage_callback,
        "dataset_ready",
        background_index=int(background_idx),
        pair_count=int(len(measured_display)),
        resolved_source_pair_count=int(resolved_source_pair_count),
        message=(
            "preflight: dataset ready for "
            f"background {int(background_idx) + 1} "
            f"({int(resolved_source_pair_count)}/{int(len(selected_entries))} source pairs resolved)"
        ),
    )

    return {
        "dataset_index": int(background_idx),
        "label": label,
        "theta_base": float(theta_base),
        "theta_effective": float(theta_base + theta_offset),
        "group_count": int(group_count),
        "pair_count": int(len(measured_display)),
        "resolved_source_pair_count": int(resolved_source_pair_count),
        "simulated_peak_count": int(
            len([item for item in simulated_peaks or () if isinstance(item, Mapping)])
        ),
        "simulated_lookup_count": int(len(simulated_lookup)),
        "simulation_diagnostics": (
            simulation_diagnostics
            if isinstance(simulation_diagnostics, Mapping)
            else {}
        ),
        "cache_metadata": cache_metadata,
        "source_resolution_diagnostics": source_resolution_diagnostics,
        "measured_display": measured_display,
        "measured_native": measured_native,
        "measured_for_fit": measured_for_fit,
        "initial_pairs_display": initial_pairs_display,
        "native_background": native_background,
        "orientation_choice": orientation_choice,
        "orientation_diag": orientation_diag,
        "summary_line": (
            "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
            "groups={groups} points={points} orientation={orientation}"
        ).format(
            idx=background_idx,
            name=label,
            theta_base=float(theta_base),
            theta_eff=float(theta_base + theta_offset),
            groups=int(group_count),
            points=int(len(measured_display)),
            orientation=orientation_choice.get("label", "identity"),
        ),
        "spec": {
            "dataset_index": int(background_idx),
            "label": label,
            "theta_initial": float(theta_base),
            "measured_peaks": measured_for_fit,
            "experimental_image": experimental_image_for_fit,
            "dynamic_reanchor_callback": dynamic_reanchor_callback,
            "dynamic_reanchor_enabled": bool(dynamic_reanchor_enabled),
        },
    }


def _manual_geometry_fit_preflight_error(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> str | None:
    """Return a preflight error when saved manual pairs are no longer trustworthy."""

    if not dataset_infos:
        return None

    orientation_failures: list[str] = []
    partially_unresolved_sources: list[str] = []
    resolved_source_total = 0
    resolved_source_known = False
    total_pairs = 0
    for idx, dataset in enumerate(dataset_infos):
        if not isinstance(dataset, Mapping):
            continue
        label = str(dataset.get("label", f"background_{idx + 1}"))
        try:
            pair_count = max(0, int(dataset.get("pair_count", 0)))
        except Exception:
            pair_count = 0
        total_pairs += int(pair_count)

        orientation_diag = dataset.get("orientation_diag")
        if isinstance(orientation_diag, Mapping) and pair_count > 0:
            try:
                orientation_pairs = int(orientation_diag.get("pairs", 0))
            except Exception:
                orientation_pairs = 0
            if orientation_pairs <= 0:
                orientation_failures.append(label)

        if "resolved_source_pair_count" in dataset:
            resolved_source_known = True
            try:
                resolved_count = max(
                    0,
                    int(dataset.get("resolved_source_pair_count", 0)),
                )
                resolved_source_total += int(resolved_count)
                if pair_count > 0 and int(resolved_count) < int(pair_count):
                    partially_unresolved_sources.append(
                        "{label} ({resolved}/{total})".format(
                            label=label,
                            resolved=int(resolved_count),
                            total=int(pair_count),
                        )
                    )
            except Exception:
                pass

    if orientation_failures:
        joined = ", ".join(orientation_failures)
        return (
            "Geometry fit unavailable: orientation preflight produced no usable "
            f"simulated/measured anchor pairs for {joined}. Refresh the picks "
            "before fitting."
        )

    if resolved_source_known and total_pairs > 0 and resolved_source_total <= 0:
        return (
            "Geometry fit unavailable: saved manual pairs no longer resolve to "
            "current simulated source rows on any selected background. Refresh "
            "the picks before fitting."
        )
    if partially_unresolved_sources:
        joined = ", ".join(partially_unresolved_sources)
        return (
            "Geometry fit unavailable: some saved manual pairs no longer "
            "resolve to current simulated source rows: "
            f"{joined}. Refresh the picks before fitting."
        )
    return None


def prepare_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    fit_config: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial: object,
    preserve_live_theta: bool,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    ensure_geometry_fit_caked_view: Callable[[], None],
    build_dataset: Callable[..., dict[str, object]],
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]],
    require_selected_var_names: bool = True,
    require_active_background_in_selection: bool = True,
    include_all_selected_backgrounds: bool | None = None,
    stage_callback: GeometryFitStageCallback | None = None,
) -> GeometryFitPreparationResult:
    """Validate and assemble the manual-pair geometry-fit runtime inputs."""

    selected_var_names = [str(name) for name in (var_names or ())]
    fit_params = dict(params or {})
    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}

    current_index = int(current_background_index)
    _emit_geometry_fit_stage_event(
        stage_callback,
        "prepare_start",
        message="preflight: collecting geometry-fit datasets",
        selected_var_names=list(selected_var_names),
        current_background_index=int(current_index),
    )

    def _failure_result(
        error_text: str,
        *,
        dataset_infos: Sequence[Mapping[str, object]] | None = None,
        current_dataset: Mapping[str, object] | None = None,
        selected_background_indices: Sequence[object] = (),
        joint_background_mode: bool = False,
        geometry_runtime_cfg: Mapping[str, object] | None = None,
    ) -> GeometryFitPreparationResult:
        log_runtime_cfg = (
            geometry_runtime_cfg
            if isinstance(geometry_runtime_cfg, Mapping)
            else geometry_refine_cfg
        )
        _emit_geometry_fit_stage_event(
            stage_callback,
            "preflight_failure",
            error_text=str(error_text),
            selected_background_indices=[
                int(idx) for idx in (selected_background_indices or ())
            ],
            joint_background_mode=bool(joint_background_mode),
            message=str(error_text),
        )
        return GeometryFitPreparationResult(
            error_text=str(error_text),
            failure_log_sections=build_geometry_fit_preflight_log_sections(
                error_text=str(error_text),
                params=fit_params,
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
                geometry_runtime_cfg=log_runtime_cfg,
            ),
        )

    if require_selected_var_names and not selected_var_names:
        return _failure_result(
            "No geometry parameters are selected for fitting."
        )
    lattice_refinement_cfg = geometry_refine_cfg.get("lattice_refinement", {}) or {}
    if not isinstance(lattice_refinement_cfg, Mapping):
        lattice_refinement_cfg = {}
    lattice_refinement_enabled = bool(lattice_refinement_cfg.get("enabled", False))
    selected_lattice_vars = [
        name for name in selected_var_names if name in {"a", "c"}
    ]
    if selected_lattice_vars and not lattice_refinement_enabled:
        joined = ", ".join(selected_lattice_vars)
        return _failure_result(
            (
                "Geometry fit unavailable: lattice parameters "
                f"({joined}) are frozen by default. Enable "
                "`fit.geometry.lattice_refinement.enabled` to refine them."
            )
        )
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, Mapping):
        orientation_cfg = {}
    overlay_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(overlay_cfg, Mapping):
        overlay_cfg = {}
    max_display_markers = max(1, int(overlay_cfg.get("max_display_markers", 120)))

    if not osc_files:
        return _failure_result(
            "Geometry fit unavailable: no background image is loaded."
        )

    if not apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return GeometryFitPreparationResult()

    try:
        selected_background_indices = current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        return _failure_result(
            (
                "Geometry fit unavailable: invalid fit background selection "
                f"({exc})."
            )
        )
    selected_background_indices = [int(idx) for idx in selected_background_indices]
    if not selected_background_indices:
        return _failure_result(
            "Geometry fit unavailable: no fit backgrounds are selected."
        )

    if current_index in set(selected_background_indices):
        primary_index = int(current_index)
    elif require_active_background_in_selection:
        return _failure_result(
            (
                "Geometry fit unavailable: the active background must be part of "
                "the fit selection so the overlay can be drawn on the current image."
            ),
            selected_background_indices=selected_background_indices,
        )
    else:
        primary_index = int(selected_background_indices[0])

    joint_background_mode = False
    background_theta_values: list[float] = []
    if geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            return GeometryFitPreparationResult()
        try:
            background_theta_values = list(
                current_background_theta_values(strict_count=True)
            )
            fit_params["theta_offset"] = float(current_geometry_theta_offset(strict=True))
        except Exception as exc:
            return _failure_result(
                (
                    "Geometry fit unavailable: failed to parse background theta "
                    f"settings ({exc})."
                ),
                selected_background_indices=selected_background_indices,
                joint_background_mode=len(selected_background_indices) > 1,
            )
        joint_background_mode = len(selected_background_indices) > 1
        if background_theta_values:
            fit_params["theta_initial"] = float(background_theta_values[primary_index])
    else:
        fit_params["theta_offset"] = 0.0
        theta_default = float(fit_params.get("theta_initial", theta_initial))
        build_all_selected_backgrounds = bool(include_all_selected_backgrounds)
        if build_all_selected_backgrounds:
            try:
                background_theta_values = list(
                    current_background_theta_values(strict_count=True)
                )
            except Exception:
                try:
                    background_theta_values = list(
                        current_background_theta_values(strict_count=False)
                    )
                except Exception:
                    background_theta_values = []
            required_theta_count = max(selected_background_indices) + 1
            if len(background_theta_values) < required_theta_count:
                background_theta_values.extend(
                    [float(theta_default)]
                    * (required_theta_count - len(background_theta_values))
                )
            normalized_theta_values: list[float] = []
            for raw_value in background_theta_values:
                try:
                    theta_value = float(raw_value)
                except Exception:
                    theta_value = float(theta_default)
                if not np.isfinite(theta_value):
                    theta_value = float(theta_default)
                normalized_theta_values.append(float(theta_value))
            background_theta_values = normalized_theta_values
            fit_params["theta_initial"] = float(background_theta_values[primary_index])
        else:
            fit_params["theta_initial"] = float(theta_default)
            background_theta_values = [float(fit_params["theta_initial"])]

    build_all_selected_backgrounds = (
        bool(joint_background_mode)
        if include_all_selected_backgrounds is None
        else bool(include_all_selected_backgrounds)
    )

    required_indices = (
        list(selected_background_indices)
        if build_all_selected_backgrounds
        else [int(primary_index)]
    )
    missing_indices = [
        idx
        for idx in required_indices
        if not geometry_manual_pairs_for_index(int(idx))
    ]
    if missing_indices:
        missing_names = [
            Path(str(osc_files[idx])).name
            for idx in missing_indices
            if 0 <= idx < len(osc_files)
        ]
        return _failure_result(
            (
                "Geometry fit unavailable: save manual Qr/Qz pairs first for "
                + ", ".join(
                    missing_names
                    or [f"background {idx + 1}" for idx in missing_indices]
                )
                + "."
            ),
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
        )

    def _theta_base_for_index(dataset_index: int) -> float:
        if build_all_selected_backgrounds:
            return float(background_theta_values[int(dataset_index)])
        if joint_background_mode:
            return float(background_theta_values[int(dataset_index)])
        return float(fit_params.get("theta_initial", theta_initial))

    def _build_dataset_entry(
        dataset_index: int,
        *,
        theta_base_value: float,
    ) -> dict[str, object]:
        build_kwargs = {
            "theta_base": float(theta_base_value),
            "base_fit_params": fit_params,
            "orientation_cfg": dict(orientation_cfg),
        }
        if callable(stage_callback):
            try:
                signature = inspect.signature(build_dataset)
            except (TypeError, ValueError):
                signature = None
            accepts_var_kwargs = bool(
                signature is not None
                and any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if signature is None or "stage_callback" in signature.parameters or accepts_var_kwargs:
                build_kwargs["stage_callback"] = stage_callback
        return build_dataset(
            int(dataset_index),
            **build_kwargs,
        )

    current_theta_base = (
        _theta_base_for_index(primary_index)
    )
    _emit_geometry_fit_stage_event(
        stage_callback,
        "current_dataset_start",
        background_index=int(primary_index),
        message=(
            "preflight: building active dataset for "
            f"background {int(primary_index) + 1}"
        ),
    )
    current_dataset = _build_dataset_entry(
        int(primary_index),
        theta_base_value=float(current_theta_base),
    )
    dataset_infos = [current_dataset]
    if build_all_selected_backgrounds:
        for bg_idx in selected_background_indices:
            idx = int(bg_idx)
            if idx == primary_index:
                continue
            _emit_geometry_fit_stage_event(
                stage_callback,
                "additional_dataset_start",
                background_index=int(idx),
                message=(
                    "preflight: building additional dataset for "
                    f"background {int(idx) + 1}"
                ),
            )
            dataset_infos.append(
                _build_dataset_entry(
                    idx,
                    theta_base_value=float(_theta_base_for_index(idx)),
                )
            )

    preflight_error = _manual_geometry_fit_preflight_error(dataset_infos)
    if preflight_error:
        return _failure_result(
            preflight_error,
            dataset_infos=dataset_infos,
            current_dataset=current_dataset,
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
        )

    dataset_specs = build_geometry_fit_dataset_specs(dataset_infos)
    geometry_runtime_cfg = apply_manual_point_geometry_fit_runtime_overrides(
        apply_joint_geometry_fit_runtime_safety_overrides(
            build_runtime_config(fit_params),
            joint_background_mode=joint_background_mode,
        ),
        joint_background_mode=joint_background_mode,
    )
    _emit_geometry_fit_stage_event(
        stage_callback,
        "prepare_ready",
        dataset_count=int(len(dataset_infos)),
        joint_background_mode=bool(joint_background_mode),
        message=(
            "preflight: ready to solve geometry fit "
            f"({int(len(dataset_infos))} dataset{'s' if len(dataset_infos) != 1 else ''})"
        ),
    )
    return GeometryFitPreparationResult(
        prepared_run=GeometryFitPreparedRun(
            fit_params=fit_params,
            selected_background_indices=[int(idx) for idx in selected_background_indices],
            background_theta_values=[float(value) for value in background_theta_values],
            joint_background_mode=bool(joint_background_mode),
            current_dataset=current_dataset,
            dataset_infos=dataset_infos,
            dataset_specs=dataset_specs,
            start_cmd_line=build_geometry_fit_start_cmd_line(
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
            ),
            start_log_sections=build_geometry_fit_start_log_sections(
                params=fit_params,
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
                geometry_runtime_cfg=geometry_runtime_cfg,
            ),
            max_display_markers=int(max_display_markers),
            geometry_runtime_cfg=geometry_runtime_cfg,
        )
    )


def prepare_runtime_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    preserve_live_theta: bool,
    bindings: GeometryFitRuntimePreparationBindings,
    stage_callback: GeometryFitStageCallback | None = None,
) -> GeometryFitPreparationResult:
    """Prepare one geometry fit from the live runtime value/callback sources."""

    fit_config = bindings.fit_config if isinstance(bindings.fit_config, Mapping) else {}
    manual_dataset_bindings = bindings.manual_dataset_bindings

    return prepare_geometry_fit_run(
        params=params,
        var_names=var_names,
        fit_config=fit_config,
        osc_files=manual_dataset_bindings.osc_files,
        current_background_index=int(manual_dataset_bindings.current_background_index),
        theta_initial=bindings.theta_initial,
        preserve_live_theta=preserve_live_theta,
        apply_geometry_fit_background_selection=(
            bindings.apply_geometry_fit_background_selection
        ),
        current_geometry_fit_background_indices=(
            bindings.current_geometry_fit_background_indices
        ),
        geometry_fit_uses_shared_theta_offset=(
            bindings.geometry_fit_uses_shared_theta_offset
        ),
        apply_background_theta_metadata=bindings.apply_background_theta_metadata,
        current_background_theta_values=bindings.current_background_theta_values,
        current_geometry_theta_offset=bindings.current_geometry_theta_offset,
        geometry_manual_pairs_for_index=(
            manual_dataset_bindings.geometry_manual_pairs_for_index
        ),
        ensure_geometry_fit_caked_view=bindings.ensure_geometry_fit_caked_view,
        build_dataset=(
            lambda background_index, *, theta_base, base_fit_params, orientation_cfg, stage_callback=None: (
                build_geometry_manual_fit_dataset(
                    background_index,
                    theta_base=theta_base,
                    base_fit_params=base_fit_params,
                    manual_dataset_bindings=manual_dataset_bindings,
                    orientation_cfg=orientation_cfg,
                    stage_callback=stage_callback,
                )
            )
        ),
        build_runtime_config=(
            lambda fit_params: bindings.build_runtime_config(
                dict(fit_params or {})
            )
        ),
        stage_callback=stage_callback,
    )


def build_geometry_fit_dataset_specs(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Copy optimizer dataset specs from the prepared dataset metadata."""

    dataset_specs: list[dict[str, object]] = []
    for info in dataset_infos or ():
        if not isinstance(info, Mapping):
            continue
        spec = info.get("spec")
        if isinstance(spec, Mapping):
            dataset_specs.append(dict(spec))
    return dataset_specs


def apply_joint_geometry_fit_runtime_safety_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Apply interactive multi-background runtime safeguards without serializing."""

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    if not joint_background_mode:
        return cfg
    if bool(cfg.get("allow_unsafe_runtime", False)):
        return cfg

    solver_cfg_raw = cfg.get("solver", {})
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}
    solver_cfg["stagnation_probe"] = False
    solver_cfg["stagnation_probe_pairwise"] = False
    solver_cfg["stagnation_probe_random_directions"] = 0
    if "restarts" in solver_cfg:
        try:
            solver_cfg["restarts"] = min(max(int(solver_cfg["restarts"]), 0), 1)
        except Exception:
            solver_cfg["restarts"] = 1
    cfg["solver"] = solver_cfg
    cfg["use_numba"] = False

    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        dict(identifiability_cfg_raw)
        if isinstance(identifiability_cfg_raw, Mapping)
        else {}
    )
    identifiability_cfg["enabled"] = False
    cfg["identifiability"] = identifiability_cfg
    return cfg


def apply_manual_point_geometry_fit_runtime_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Build the lean runtime profile for raw detector-pixel manual point fitting."""

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    unsafe_runtime_enabled = bool(cfg.get("allow_unsafe_runtime", False))

    optimizer_cfg_raw = cfg.get("optimizer", cfg.get("solver", {}))
    optimizer_cfg = (
        dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    )
    for key in (
        "dynamic_point_geometry_fit",
        "restarts",
        "restart_jitter",
        "stagnation_probe",
        "stagnation_probe_pairwise",
        "stagnation_probe_pair_limit",
        "stagnation_probe_random_directions",
        "staged_release",
        "reparameterize_pairs",
    ):
        optimizer_cfg.pop(key, None)
    optimizer_cfg["manual_point_fit_mode"] = True
    optimizer_cfg["missing_pair_penalty_px"] = float(
        optimizer_cfg.get("missing_pair_penalty_px", 20.0)
    )
    optimizer_cfg["missing_pair_penalty_deg"] = float(
        optimizer_cfg.get("missing_pair_penalty_deg", 5.0)
    )
    optimizer_cfg["q_group_line_constraints"] = bool(
        optimizer_cfg.get("q_group_line_constraints", True)
    )
    optimizer_cfg["q_group_line_angle_weight"] = float(
        optimizer_cfg.get("q_group_line_angle_weight", 0.6)
    )
    optimizer_cfg["q_group_line_offset_weight"] = float(
        optimizer_cfg.get("q_group_line_offset_weight", 1.0)
    )
    optimizer_cfg["q_group_line_missing_penalty_scale"] = float(
        optimizer_cfg.get("q_group_line_missing_penalty_scale", 0.35)
    )
    optimizer_cfg["hk0_peak_priority_weight"] = float(
        optimizer_cfg.get("hk0_peak_priority_weight", 6.0)
    )
    optimizer_cfg["workers"] = optimizer_cfg.get("workers", "auto")
    optimizer_cfg["parallel_mode"] = str(
        optimizer_cfg.get("parallel_mode", "auto")
    ).strip()
    optimizer_cfg["worker_numba_threads"] = optimizer_cfg.get(
        "worker_numba_threads",
        0,
    )
    cfg["optimizer"] = optimizer_cfg
    cfg["solver"] = optimizer_cfg

    seed_search_cfg_raw = cfg.get("seed_search", {})
    seed_search_cfg = (
        dict(seed_search_cfg_raw) if isinstance(seed_search_cfg_raw, Mapping) else {}
    )
    # Manual point fits are interactive; keep the normalized-u solver on one
    # trusted seed instead of inheriting the heavier global multistart budget.
    seed_search_cfg["prescore_top_k"] = 1
    seed_search_cfg["n_global"] = 0
    seed_search_cfg["n_jitter"] = 0
    cfg["seed_search"] = seed_search_cfg
    cfg["use_numba"] = bool(cfg.get("use_numba", True)) if unsafe_runtime_enabled else False
    cfg["allow_unsafe_runtime"] = bool(unsafe_runtime_enabled)

    discrete_modes_cfg_raw = cfg.get("discrete_modes", {})
    discrete_modes_cfg = (
        dict(discrete_modes_cfg_raw)
        if isinstance(discrete_modes_cfg_raw, Mapping)
        else {}
    )
    # Manual GUI fits already resolve detector orientation from the saved
    # point pairs before the solver runs. Repeating the full solver-side
    # rot/flip sweep multiplies the work by up to 16x and has been able to
    # stall or crash interactive runs on large datasets.
    discrete_modes_cfg["enabled"] = False
    cfg["discrete_modes"] = discrete_modes_cfg

    # Manual detector-pixel fits should react to real peak offsets, not to a
    # heavily downweighted residual that can remain small even when the
    # geometry is still hundreds of pixels off.
    optimizer_cfg["loss"] = "linear"
    optimizer_cfg["f_scale_px"] = 1.0
    optimizer_cfg["weighted_matching"] = False
    optimizer_cfg["use_measurement_uncertainty"] = False
    optimizer_cfg["anisotropic_measurement_uncertainty"] = False

    cfg.pop("full_beam_polish", None)
    cfg.pop("ridge_refinement", None)
    cfg.pop("image_refinement", None)

    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        dict(identifiability_cfg_raw)
        if isinstance(identifiability_cfg_raw, Mapping)
        else {}
    )
    identifiability_cfg["enabled"] = True
    identifiability_cfg.pop("auto_freeze", None)
    identifiability_cfg.pop("selective_thaw", None)
    identifiability_cfg.pop("adaptive_regularization", None)
    cfg["identifiability"] = identifiability_cfg
    return cfg


def apply_dynamic_point_geometry_fit_runtime_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Backward-compatible alias for the raw manual point-fit runtime profile."""

    return apply_manual_point_geometry_fit_runtime_overrides(
        runtime_cfg,
        joint_background_mode=joint_background_mode,
    )


def _geometry_fit_cache_normalized_hkl(
    value: object,
) -> tuple[int, int, int] | None:
    """Return one normalized HKL triplet when the value is usable."""

    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except Exception:
        return None


def _geometry_fit_cache_jsonable(value: object) -> object:
    """Convert one cache metadata value into a stable JSON-safe shape."""

    if isinstance(value, Mapping):
        return {
            str(key): _geometry_fit_cache_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_geometry_fit_cache_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_geometry_fit_cache_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_geometry_fit_cache_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _geometry_fit_cache_finite_float(
    value: object,
) -> float | None:
    """Return one finite float cache metadata value when possible."""

    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_geometry_fit_dataset_cache_metadata(
    *,
    background_index: int,
    current_background_index: int,
    simulated_peaks: Sequence[object] | None,
    source_snapshot_diagnostics: Mapping[str, object] | None = None,
    source_resolution_diagnostics: Sequence[object] | None,
    pair_count: int,
    resolved_source_pair_count: int,
) -> dict[str, object]:
    """Build structured cache metadata for one geometry-fit dataset payload."""

    raw_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)
    diag_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)
    raw_by_row: dict[tuple[int, int], dict[str, object]] = {}
    raw_by_peak: dict[tuple[int, int], dict[str, object]] = {}

    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            table_idx = int(raw_entry.get("source_table_index"))
        except Exception:
            continue
        entry = dict(raw_entry)
        raw_by_table[table_idx].append(entry)
        try:
            row_idx = int(entry.get("source_row_index"))
        except Exception:
            row_idx = -1
        if row_idx >= 0:
            raw_by_row[(int(table_idx), int(row_idx))] = entry
        try:
            peak_idx = int(entry.get("source_peak_index"))
        except Exception:
            peak_idx = -1
        if peak_idx >= 0:
            raw_by_peak[(int(table_idx), int(peak_idx))] = entry

    for raw_diag in source_resolution_diagnostics or ():
        if not isinstance(raw_diag, Mapping):
            continue
        diag = dict(raw_diag)
        table_idx = None
        for key_name in (
            "fit_source_row_key",
            "fit_source_peak_key",
            "overlay_source_row_key",
            "overlay_source_peak_key",
            "saved_source_row_key",
            "saved_source_peak_key",
        ):
            key_value = diag.get(key_name)
            if isinstance(key_value, (list, tuple, np.ndarray)) and len(key_value) >= 2:
                try:
                    table_idx = int(key_value[0])
                    break
                except Exception:
                    continue
        if table_idx is None:
            raw_table_idx = diag.get("saved_source_table_index")
            try:
                table_idx = int(raw_table_idx)
            except Exception:
                continue
        diag_by_table[int(table_idx)].append(diag)

    table_summaries: list[dict[str, object]] = []
    for table_idx in sorted(set(raw_by_table) | set(diag_by_table)):
        raw_entries = list(raw_by_table.get(table_idx, ()))
        table_diags = list(diag_by_table.get(table_idx, ()))
        reference_entry: dict[str, object] | None = None
        representative_row_indices: set[int] = set()
        kept_keys: set[tuple[str, int]] = set()
        nominal_hkl_recovery_count = 0

        for diag in table_diags:
            fit_kind = str(diag.get("fit_resolution_kind", "") or "")
            if fit_kind in {"legacy_dense_hkl_rebind", "hkl_fallback"}:
                nominal_hkl_recovery_count += 1
            if not bool(diag.get("fit_resolved", False)):
                continue
            row_key_added = False
            fit_row_key = diag.get("fit_source_row_key")
            if (
                isinstance(fit_row_key, (list, tuple, np.ndarray))
                and len(fit_row_key) >= 2
            ):
                try:
                    row_idx = int(fit_row_key[1])
                except Exception:
                    row_idx = -1
                if row_idx >= 0:
                    representative_row_indices.add(int(row_idx))
                    kept_keys.add(("row", int(row_idx)))
                    row_key_added = True
                    reference_entry = (
                        raw_by_row.get((int(table_idx), int(row_idx)))
                        or reference_entry
                    )
            fit_peak_key = diag.get("fit_source_peak_key")
            if (
                isinstance(fit_peak_key, (list, tuple, np.ndarray))
                and len(fit_peak_key) >= 2
            ):
                try:
                    peak_idx = int(fit_peak_key[1])
                except Exception:
                    peak_idx = -1
                if peak_idx >= 0 and not row_key_added:
                    kept_keys.add(("peak", int(peak_idx)))
                    if reference_entry is None:
                        reference_entry = raw_by_peak.get(
                            (int(table_idx), int(peak_idx))
                        )
            if reference_entry is None:
                overlay_row_key = diag.get("overlay_source_row_key")
                if (
                    isinstance(overlay_row_key, (list, tuple, np.ndarray))
                    and len(overlay_row_key) >= 2
                ):
                    try:
                        overlay_row_idx = int(overlay_row_key[1])
                    except Exception:
                        overlay_row_idx = -1
                    if overlay_row_idx >= 0:
                        reference_entry = raw_by_row.get(
                            (int(table_idx), int(overlay_row_idx))
                        )
            if reference_entry is None:
                overlay_peak_key = diag.get("overlay_source_peak_key")
                if (
                    isinstance(overlay_peak_key, (list, tuple, np.ndarray))
                    and len(overlay_peak_key) >= 2
                ):
                    try:
                        overlay_peak_idx = int(overlay_peak_key[1])
                    except Exception:
                        overlay_peak_idx = -1
                    if overlay_peak_idx >= 0:
                        reference_entry = raw_by_peak.get(
                            (int(table_idx), int(overlay_peak_idx))
                        )

        if reference_entry is None and raw_entries:
            reference_entry = dict(raw_entries[0])

        dropped_nonfinite = 0
        for entry in raw_entries:
            sim_col = _geometry_fit_cache_finite_float(entry.get("sim_col"))
            sim_row = _geometry_fit_cache_finite_float(entry.get("sim_row"))
            if sim_col is None or sim_row is None:
                dropped_nonfinite += 1
        row_count_before = int(len(raw_entries))
        row_count_after = int(len(kept_keys))
        merged_group_count = max(
            0,
            int(row_count_before) - int(row_count_after) - int(dropped_nonfinite),
        )
        reference = dict(reference_entry or {})
        nominal_hkl = _geometry_fit_cache_normalized_hkl(reference.get("hkl"))
        table_summaries.append(
            {
                "source_table_index": int(table_idx),
                "nominal_hkl": (
                    list(nominal_hkl) if isinstance(nominal_hkl, tuple) else None
                ),
                "q_group_key": _geometry_fit_cache_jsonable(
                    reference.get("q_group_key")
                ),
                "qr": _geometry_fit_cache_finite_float(reference.get("qr")),
                "qz": _geometry_fit_cache_finite_float(reference.get("qz")),
                "row_count_before_grouping": int(row_count_before),
                "row_count_after_grouping": int(row_count_after),
                "dropped_nonfinite_row_count": int(dropped_nonfinite),
                "nominal_hkl_recovery_count": int(nominal_hkl_recovery_count),
                "merged_group_count": int(merged_group_count),
                "representative_row_indices_kept": [
                    int(idx) for idx in sorted(representative_row_indices)
                ],
            }
        )

    snapshot_diag = (
        dict(source_snapshot_diagnostics)
        if isinstance(source_snapshot_diagnostics, Mapping)
        else {}
    )
    snapshot_status = str(
        snapshot_diag.get(
            "status",
            "snapshot_hit" if simulated_peaks else "snapshot_empty",
        )
    )
    snapshot_hit = snapshot_status == "snapshot_hit"
    snapshot_cache_source = str(
        snapshot_diag.get(
            "cache_source",
            snapshot_diag.get("source", "source_snapshot"),
        )
        or "source_snapshot"
    )
    snapshot_rebuild_source = str(
        snapshot_diag.get("rebuild_source", snapshot_diag.get("created_from", ""))
        or ""
    )

    return {
        "cache_action": ("reused" if snapshot_hit else "rebuilt"),
        "reused": bool(snapshot_hit),
        "rebuilt": bool(not snapshot_hit),
        "stale_reason": (
            None if snapshot_hit else f"source snapshot status={snapshot_status}"
        ),
        "cache_source": snapshot_cache_source,
        "cache_provenance": [
            f"source_snapshot:{snapshot_status}",
            *([f"rebuild_source:{snapshot_rebuild_source}"] if snapshot_rebuild_source else []),
            "build_geometry_manual_fit_dataset",
        ],
        "background_index": int(background_index),
        "current_background_index": int(current_background_index),
        "prefer_cache": False,
        "pair_count": int(pair_count),
        "resolved_source_pair_count": int(resolved_source_pair_count),
        "source_snapshot_status": snapshot_status,
        "source_snapshot_created_from": snapshot_diag.get("created_from"),
        "source_snapshot_signature_match": bool(
            snapshot_diag.get("signature_match", snapshot_hit)
        ),
        "source_snapshot_consumer": snapshot_diag.get("consumer"),
        "source_snapshot_cache_family": snapshot_diag.get("cache_family"),
        "source_snapshot_action": snapshot_diag.get("action"),
        "source_snapshot_requested_signature_summary": snapshot_diag.get(
            "requested_signature_summary",
        ),
        "source_snapshot_stored_signature_summary": snapshot_diag.get(
            "stored_signature_summary",
        ),
        "source_snapshot_raw_peak_count": int(
            snapshot_diag.get(
                "raw_peak_count",
                sum(len(entries) for entries in raw_by_table.values()),
            )
            or 0
        ),
        "source_snapshot_row_count": int(
            snapshot_diag.get(
                "projected_peak_count",
                sum(len(entries) for entries in raw_by_table.values()),
            )
            or 0
        ),
        "simulated_peak_count": int(
            sum(len(entries) for entries in raw_by_table.values())
        ),
        "table_count": int(len(table_summaries)),
        "table_summaries": table_summaries,
    }


def build_geometry_fit_dataset_cache_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format geometry-fit dataset cache metadata for one debug log section."""

    lines: list[str] = []
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        cache_metadata = raw_dataset.get("cache_metadata")
        if not isinstance(cache_metadata, Mapping):
            continue
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        lines.append(
            (
                "{label}: cache_action={action} reused={reused} rebuilt={rebuilt} "
                "stale_reason={stale} source={source}"
            ).format(
                label=label,
                action=str(cache_metadata.get("cache_action", "<unknown>")),
                reused=_geometry_fit_debug_value_text(
                    cache_metadata.get("reused", False)
                ),
                rebuilt=_geometry_fit_debug_value_text(
                    cache_metadata.get("rebuilt", True)
                ),
                stale=str(cache_metadata.get("stale_reason", "<none>") or "<none>"),
                source=str(cache_metadata.get("cache_source", "<unknown>")),
            )
        )
        provenance = cache_metadata.get("cache_provenance")
        if isinstance(provenance, Sequence) and not isinstance(
            provenance,
            (str, bytes, bytearray),
        ):
            lines.append(
                "{label}: provenance={provenance}".format(
                    label=label,
                    provenance=_geometry_fit_debug_value_text(
                        list(provenance),
                        float_digits=3,
                    ),
                )
            )
        lines.append(
            (
                "{label}: pair_count={pairs} resolved_source_pairs={resolved} "
                "source_rows={peaks} tables={tables}"
            ).format(
                label=label,
                pairs=_geometry_fit_debug_value_text(
                    cache_metadata.get("pair_count", raw_dataset.get("pair_count", 0)),
                    float_digits=0,
                ),
                resolved=_geometry_fit_debug_value_text(
                    cache_metadata.get(
                        "resolved_source_pair_count",
                        raw_dataset.get("resolved_source_pair_count", 0),
                    ),
                    float_digits=0,
                ),
                peaks=_geometry_fit_debug_value_text(
                    cache_metadata.get(
                        "simulated_peak_count",
                        raw_dataset.get("simulated_peak_count", 0),
                    ),
                    float_digits=0,
                ),
                tables=_geometry_fit_debug_value_text(
                    cache_metadata.get("table_count", 0),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: source_snapshot family={family} action={action} "
                "status={status} consumer={consumer} created_from={created_from} "
                "signature_match={match} rows={rows} raw_peaks={raw}"
            ).format(
                label=label,
                family=str(
                    cache_metadata.get("source_snapshot_cache_family", "<unknown>")
                ),
                action=str(
                    cache_metadata.get("source_snapshot_action", "<unknown>")
                ),
                status=str(
                    cache_metadata.get("source_snapshot_status", "<unknown>")
                ),
                consumer=str(
                    cache_metadata.get("source_snapshot_consumer", "<unknown>")
                ),
                created_from=str(
                    cache_metadata.get("source_snapshot_created_from", "<unknown>")
                ),
                match=_geometry_fit_debug_value_text(
                    cache_metadata.get("source_snapshot_signature_match", False)
                ),
                rows=_geometry_fit_debug_value_text(
                    cache_metadata.get("source_snapshot_row_count", 0),
                    float_digits=0,
                ),
                raw=_geometry_fit_debug_value_text(
                    cache_metadata.get("source_snapshot_raw_peak_count", 0),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: source_snapshot requested_signature={requested} "
                "stored_signature={stored}"
            ).format(
                label=label,
                requested=_geometry_fit_debug_value_text(
                    cache_metadata.get("source_snapshot_requested_signature_summary")
                ),
                stored=_geometry_fit_debug_value_text(
                    cache_metadata.get("source_snapshot_stored_signature_summary")
                ),
            )
        )
        table_summaries = cache_metadata.get("table_summaries")
        if not isinstance(table_summaries, Sequence) or isinstance(
            table_summaries,
            (str, bytes, bytearray),
        ):
            continue
        for raw_summary in table_summaries:
            if not isinstance(raw_summary, Mapping):
                continue
            table_index = raw_summary.get(
                "source_table_index",
                raw_summary.get("table_index", 0),
            )
            lines.append(
                (
                    "{label}: table[{table}] nominal_hkl={hkl} q_group_key={q_group} "
                    "qr={qr} qz={qz} rows_before={before} rows_after={after} "
                    "dropped_nonfinite={dropped} nominal_hkl_recovery={recovery} "
                    "merged_groups={merged} representative_rows={rows}"
                ).format(
                    label=label,
                    table=_geometry_fit_debug_value_text(table_index, float_digits=0),
                    hkl=_geometry_fit_debug_value_text(
                        raw_summary.get("nominal_hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        raw_summary.get("q_group_key"),
                        float_digits=3,
                    ),
                    qr=_geometry_fit_debug_value_text(
                        raw_summary.get("qr"),
                        float_digits=6,
                    ),
                    qz=_geometry_fit_debug_value_text(
                        raw_summary.get("qz"),
                        float_digits=6,
                    ),
                    before=_geometry_fit_debug_value_text(
                        raw_summary.get("row_count_before_grouping", 0),
                        float_digits=0,
                    ),
                    after=_geometry_fit_debug_value_text(
                        raw_summary.get("row_count_after_grouping", 0),
                        float_digits=0,
                    ),
                    dropped=_geometry_fit_debug_value_text(
                        raw_summary.get("dropped_nonfinite_row_count", 0),
                        float_digits=0,
                    ),
                    recovery=_geometry_fit_debug_value_text(
                        raw_summary.get("nominal_hkl_recovery_count", 0),
                        float_digits=0,
                    ),
                    merged=_geometry_fit_debug_value_text(
                        raw_summary.get("merged_group_count", 0),
                        float_digits=0,
                    ),
                    rows=_geometry_fit_debug_value_text(
                        raw_summary.get("representative_row_indices_kept"),
                        float_digits=0,
                    ),
                )
            )
    return lines


def build_geometry_fit_live_cache_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format live-cache inventory and source-snapshot behavior for debug logs."""

    lines: list[str] = []
    inventory: Mapping[str, object] | None = None
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("simulation_diagnostics")
        if not isinstance(diagnostics_raw, Mapping):
            continue
        diagnostics = dict(diagnostics_raw)
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        lines.append(
            (
                "{label}: source_snapshot status={status} consumer={consumer} "
                "created_from={created_from} signature_match={match} "
                "rows={rows} raw_peaks={raw}"
            ).format(
                label=label,
                status=str(diagnostics.get("status", "<unknown>")),
                consumer=str(diagnostics.get("consumer", "<unknown>")),
                created_from=str(diagnostics.get("created_from", "<unknown>")),
                match=_geometry_fit_debug_value_text(
                    diagnostics.get("signature_match", False)
                ),
                rows=_geometry_fit_debug_value_text(
                    diagnostics.get(
                        "projected_peak_count",
                        raw_dataset.get("simulated_peak_count", 0),
                    ),
                    float_digits=0,
                ),
                raw=_geometry_fit_debug_value_text(
                    diagnostics.get("raw_peak_count", 0),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: source_snapshot requested_signature={requested} "
                "stored_signature={stored}"
            ).format(
                label=label,
                requested=_geometry_fit_debug_value_text(
                    diagnostics.get("requested_signature_summary")
                ),
                stored=_geometry_fit_debug_value_text(
                    diagnostics.get("stored_signature_summary")
                ),
            )
        )
        inventory_raw = diagnostics.get("live_cache_inventory")
        if inventory is None and isinstance(inventory_raw, Mapping):
            inventory = dict(inventory_raw)
    if not isinstance(inventory, Mapping):
        return lines
    lines.append(
        (
            "inventory: preview_active={preview_active} preview_sample_count={preview_samples} "
            "stored_hit_table_signature_present={stored_present} "
            "stored_hit_table_signature={stored_sig}"
        ).format(
            preview_active=_geometry_fit_debug_value_text(
                inventory.get("preview_active", False)
            ),
            preview_samples=_geometry_fit_debug_value_text(
                inventory.get("preview_sample_count"),
                float_digits=0,
            ),
            stored_present=_geometry_fit_debug_value_text(
                inventory.get("stored_hit_table_signature_present", False)
            ),
            stored_sig=_geometry_fit_debug_value_text(
                inventory.get("stored_hit_table_signature_summary")
            ),
        )
    )
    lines.append(
        (
            "inventory: last_simulation_signature={last_sig} "
            "primary_contribution_signature={primary_sig} primary_source_mode={source_mode} "
            "primary_active_keys={active_keys} primary_hit_table_entries={entries}"
        ).format(
            last_sig=_geometry_fit_debug_value_text(
                inventory.get("last_simulation_signature_summary")
            ),
            primary_sig=_geometry_fit_debug_value_text(
                inventory.get("primary_contribution_cache_signature_summary")
            ),
            source_mode=_geometry_fit_debug_value_text(
                inventory.get("primary_source_mode")
            ),
            active_keys=_geometry_fit_debug_value_text(
                inventory.get("primary_active_contribution_key_count", 0),
                float_digits=0,
            ),
            entries=_geometry_fit_debug_value_text(
                inventory.get("primary_hit_table_cache_entry_count", 0),
                float_digits=0,
            ),
        )
    )
    source_snapshots_raw = inventory.get("source_snapshots")
    if isinstance(source_snapshots_raw, Sequence) and not isinstance(
        source_snapshots_raw,
        (str, bytes, bytearray),
    ):
        source_snapshots = [
            dict(item) for item in source_snapshots_raw if isinstance(item, Mapping)
        ]
    else:
        source_snapshots = []
    if not source_snapshots:
        lines.append("inventory: source_snapshots=<none>")
    else:
        for snapshot in source_snapshots:
            lines.append(
                (
                    "inventory: snapshot bg[{background}] rows={rows} "
                    "created_from={created_from} signature={signature}"
                ).format(
                    background=int(snapshot.get("background_index", -1) or -1),
                    rows=_geometry_fit_debug_value_text(
                        snapshot.get("row_count", 0),
                        float_digits=0,
                    ),
                    created_from=_geometry_fit_debug_value_text(
                        snapshot.get("created_from")
                    ),
                    signature=_geometry_fit_debug_value_text(
                        snapshot.get("signature_summary")
                    ),
                )
            )
    return lines


def build_geometry_fit_dataset_cache_payload(
    prepared_run: GeometryFitPreparedRun,
    *,
    current_background_index: int | None = None,
) -> dict[str, object]:
    """Copy the reusable dataset bundle from a successful geometry fit."""

    dataset_index = current_background_index
    if dataset_index is None:
        current_dataset = (
            prepared_run.current_dataset
            if isinstance(prepared_run.current_dataset, Mapping)
            else {}
        )
        try:
            dataset_index = int(current_dataset.get("dataset_index", 0))
        except Exception:
            dataset_index = 0

    dataset_cache_metadata: list[dict[str, object]] = []
    for info in prepared_run.dataset_infos or []:
        if not isinstance(info, Mapping):
            continue
        raw_metadata = info.get("cache_metadata")
        if not isinstance(raw_metadata, Mapping):
            continue
        copied_metadata = copy_geometry_fit_state_value(dict(raw_metadata))
        if isinstance(copied_metadata, Mapping):
            copied_metadata = dict(copied_metadata)
            copied_metadata.setdefault(
                "dataset_index",
                int(info.get("dataset_index", copied_metadata.get("dataset_index", 0)) or 0),
            )
            label = str(info.get("label", "") or "").strip()
            if label:
                copied_metadata.setdefault("label", label)
            dataset_cache_metadata.append(copied_metadata)

    return {
        "selected_background_indices": [
            int(idx) for idx in prepared_run.selected_background_indices
        ],
        "current_background_index": int(dataset_index),
        "joint_background_mode": bool(prepared_run.joint_background_mode),
        "background_theta_values": [
            float(value) for value in prepared_run.background_theta_values
        ],
        "dataset_specs": [
            copy_geometry_fit_state_value(dict(spec))
            for spec in (prepared_run.dataset_specs or [])
            if isinstance(spec, Mapping)
        ],
        "cache_metadata": {
            "cache_action": "rebuilt",
            "reused": False,
            "rebuilt": True,
            "stale_reason": None,
            "cache_source": "build_geometry_fit_dataset_cache_payload",
            "cache_provenance": [
                "build_geometry_manual_fit_dataset",
                "build_geometry_fit_dataset_cache_payload",
            ],
            "dataset_count": int(len(prepared_run.dataset_infos or [])),
            "dataset_cache_metadata": dataset_cache_metadata,
        },
    }


def geometry_fit_dataset_cache_stale_reason(
    cache_payload: Mapping[str, object] | None,
    *,
    selected_background_indices: Sequence[object],
    current_background_index: int,
    joint_background_mode: bool,
    background_theta_values: Sequence[object],
) -> str | None:
    """Return a human-readable stale-cache reason, or ``None`` when valid."""

    if not isinstance(cache_payload, Mapping):
        return "Run geometry fit first."

    raw_specs = cache_payload.get("dataset_specs")
    if not isinstance(raw_specs, Sequence) or len(raw_specs) <= 0:
        return "Run geometry fit first."

    try:
        cached_selected = [
            int(idx) for idx in cache_payload.get("selected_background_indices", [])
        ]
    except Exception:
        return "Run geometry fit first."
    current_selected = [int(idx) for idx in selected_background_indices]
    if cached_selected != current_selected:
        return "Geometry-fit background selection changed. Rerun geometry fit."

    try:
        cached_index = int(cache_payload.get("current_background_index", -1))
    except Exception:
        return "Run geometry fit first."
    if int(cached_index) != int(current_background_index):
        return "Active background changed since geometry fit. Rerun geometry fit."

    cached_joint = bool(cache_payload.get("joint_background_mode", False))
    if cached_joint != bool(joint_background_mode):
        return "Shared-theta mode changed since geometry fit. Rerun geometry fit."

    try:
        cached_theta_values = [
            float(value) for value in cache_payload.get("background_theta_values", [])
        ]
        current_theta_values = [float(value) for value in background_theta_values]
    except Exception:
        return "Background theta values are unavailable. Rerun geometry fit."
    if len(cached_theta_values) != len(current_theta_values):
        return "Background theta values changed since geometry fit. Rerun geometry fit."
    for cached_value, current_value in zip(
        cached_theta_values,
        current_theta_values,
    ):
        if not np.isfinite(cached_value) or not np.isfinite(current_value):
            return "Background theta values changed since geometry fit. Rerun geometry fit."
        if not np.isclose(
            float(cached_value),
            float(current_value),
            rtol=0.0,
            atol=1.0e-9,
        ):
            return "Background theta values changed since geometry fit. Rerun geometry fit."

    return None


def build_geometry_fit_start_cmd_line(
    *,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> str:
    """Build the console start line for one geometry-fit run."""

    dataset_list = list(dataset_infos or ())
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    return (
        "start: "
        f"vars={','.join(str(name) for name in var_names)} "
        f"datasets={len(dataset_list)} "
        f"current_groups={int(dataset.get('group_count', 0) or 0)} "
        f"current_points={int(dataset.get('pair_count', 0) or 0)}"
    )


def _build_geometry_fit_run_request_lines(
    *,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> list[str]:
    """Summarize the prepared runtime request for one geometry-fit run."""

    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    lines = [
        f"joint_background_mode={bool(joint_background_mode)}",
        "selected_background_indices=[{indices}]".format(
            indices=", ".join(str(int(idx)) for idx in selected_background_indices)
        ),
        f"dataset_count={int(len(list(dataset_infos or ())))}",
        f"current_dataset_index={int(dataset.get('dataset_index', 0) or 0)}",
        f"current_groups={int(dataset.get('group_count', 0) or 0)}",
        f"current_points={int(dataset.get('pair_count', 0) or 0)}",
    ]
    label = str(dataset.get("label", "") or "").strip()
    if label:
        lines.append(f"current_label={label}")
    if "resolved_source_pair_count" in dataset:
        lines.append(
            "resolved_source_pairs={count}".format(
                count=int(dataset.get("resolved_source_pair_count", 0) or 0)
            )
        )
    if "theta_base" in dataset:
        lines.append(
            "theta_base={theta}".format(
                theta=_geometry_fit_debug_value_text(
                    dataset.get("theta_base", np.nan),
                )
            )
        )
    if "theta_effective" in dataset:
        lines.append(
            "theta_effective={theta}".format(
                theta=_geometry_fit_debug_value_text(
                    dataset.get("theta_effective", np.nan),
                )
            )
        )
    return lines


def _geometry_fit_flag_enabled(value: object) -> bool:
    """Return whether one user-facing debug flag value is enabled."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def geometry_fit_debug_logging_enabled(
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> bool:
    """Return whether extra geometry-fit logging is enabled for this run."""

    return geometry_fit_extra_sections_enabled(
        geometry_runtime_cfg=geometry_runtime_cfg,
    )


def _build_geometry_fit_runtime_config_lines(
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> list[str]:
    """Summarize the active runtime/solver configuration for one fit run."""

    cfg = geometry_runtime_cfg if isinstance(geometry_runtime_cfg, Mapping) else {}
    optimizer_cfg_raw = cfg.get("optimizer", cfg.get("solver", {}))
    optimizer_cfg = (
        optimizer_cfg_raw if isinstance(optimizer_cfg_raw, Mapping) else {}
    )
    solver_cfg_raw = cfg.get("solver", optimizer_cfg)
    solver_cfg = solver_cfg_raw if isinstance(solver_cfg_raw, Mapping) else {}
    discrete_cfg_raw = cfg.get("discrete_modes", {})
    discrete_cfg = (
        discrete_cfg_raw if isinstance(discrete_cfg_raw, Mapping) else {}
    )
    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        identifiability_cfg_raw
        if isinstance(identifiability_cfg_raw, Mapping)
        else {}
    )
    if not cfg and not optimizer_cfg and not solver_cfg:
        return []
    lines = [
        "use_numba={use_numba} allow_unsafe_runtime={unsafe}".format(
            use_numba=_geometry_fit_debug_value_text(
                cfg.get("use_numba", "<default>")
            ),
            unsafe=_geometry_fit_debug_value_text(
                cfg.get("allow_unsafe_runtime", False)
            ),
        ),
        (
            "optimizer loss={loss} f_scale_px={f_scale} manual_point_fit_mode={manual} "
            "weighted_matching={weighted} q_group_line_constraints={line_constraints}"
        ).format(
            loss=str(optimizer_cfg.get("loss", "<default>")),
            f_scale=_geometry_fit_debug_value_text(
                optimizer_cfg.get("f_scale_px", np.nan)
            ),
            manual=_geometry_fit_debug_value_text(
                optimizer_cfg.get("manual_point_fit_mode", False)
            ),
            weighted=_geometry_fit_debug_value_text(
                optimizer_cfg.get("weighted_matching", "<default>")
            ),
            line_constraints=_geometry_fit_debug_value_text(
                optimizer_cfg.get("q_group_line_constraints", "<default>")
            ),
        ),
        (
            "solver parallel_mode={mode} workers={workers} "
            "worker_numba_threads={threads} discrete_modes_enabled={discrete} "
            "identifiability_enabled={ident}"
        ).format(
            mode=str(solver_cfg.get("parallel_mode", "<default>")),
            workers=_geometry_fit_debug_value_text(
                solver_cfg.get("workers", "<default>"),
                float_digits=0,
            ),
            threads=_geometry_fit_debug_value_text(
                solver_cfg.get("worker_numba_threads", "<default>"),
                float_digits=0,
            ),
            discrete=_geometry_fit_debug_value_text(
                discrete_cfg.get("enabled", "<default>")
            ),
            ident=_geometry_fit_debug_value_text(
                identifiability_cfg.get("enabled", "<default>")
            ),
        ),
    ]
    return lines


def build_geometry_fit_start_log_sections(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> list[tuple[str, list[str]]]:
    """Build the start-log sections for one geometry-fit run."""

    fit_params = params if isinstance(params, Mapping) else {}
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    debug_logging = geometry_fit_debug_logging_enabled(geometry_runtime_cfg)
    orientation_diag = dataset.get("orientation_diag") or {}
    if not isinstance(orientation_diag, Mapping):
        orientation_diag = {}
    orientation_choice = dataset.get("orientation_choice") or {}
    if not isinstance(orientation_choice, Mapping):
        orientation_choice = {}
    dataset_lines = [
        str(info.get("summary_line", ""))
        for info in (dataset_infos or ())
        if isinstance(info, Mapping)
    ] or ["<none>"]
    sections: list[tuple[str, list[str]]] = []
    if debug_logging:
        sections.append(
            (
                "Run request:",
                _build_geometry_fit_run_request_lines(
                    selected_background_indices=selected_background_indices,
                    joint_background_mode=joint_background_mode,
                    dataset_infos=dataset_infos,
                    current_dataset=current_dataset,
                ),
            )
        )
        runtime_cfg_lines = _build_geometry_fit_runtime_config_lines(
            geometry_runtime_cfg
        )
        if runtime_cfg_lines:
            sections.append(
                (
                    "Runtime configuration:",
                    runtime_cfg_lines,
                )
            )
        dataset_cache_lines = build_geometry_fit_dataset_cache_log_lines(
            dataset_infos
        )
        if dataset_cache_lines:
            sections.append(
                (
                    "Geometry-fit dataset cache diagnostics:",
                    dataset_cache_lines,
                )
            )
        live_cache_lines = build_geometry_fit_live_cache_log_lines(dataset_infos)
        if live_cache_lines:
            sections.append(
                (
                    "Live simulation cache:",
                    live_cache_lines,
                )
            )
    sections.extend(
        [
        (
            "Fitting variables (start values):",
            [
                f"{name}={float(fit_params.get(str(name), np.nan)):.6f}"
                for name in var_names
            ],
        ),
        (
            "Manual geometry datasets:",
            dataset_lines,
        ),
        (
            "Current orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"chosen={orientation_choice.get('label', 'identity')}",
                f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                f"reason={orientation_diag.get('reason', 'n/a')}",
            ],
        ),
        ]
    )
    return sections


def _geometry_fit_source_resolution_key_text(value: object) -> str:
    """Format one cached-source identity tuple for log output."""

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        try:
            return f"({int(value[0])}, {int(value[1])})"
        except Exception:
            return _geometry_fit_debug_value_text(value, float_digits=3)
    return "<none>"


def build_geometry_fit_source_resolution_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
    *,
    max_unresolved_pairs_per_dataset: int = 8,
) -> list[str]:
    """Format cached-source resolution diagnostics for log output."""

    lines: list[str] = []
    max_pairs = max(1, int(max_unresolved_pairs_per_dataset))
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("source_resolution_diagnostics")
        if not isinstance(diagnostics_raw, Sequence) or isinstance(
            diagnostics_raw,
            (str, bytes, bytearray),
        ):
            continue
        diagnostics = [
            dict(item) for item in diagnostics_raw if isinstance(item, Mapping)
        ]
        if not diagnostics:
            continue
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        pair_count = int(raw_dataset.get("pair_count", len(diagnostics)) or 0)
        resolved_count = int(raw_dataset.get("resolved_source_pair_count", 0) or 0)
        simulated_peak_count = int(raw_dataset.get("simulated_peak_count", 0) or 0)
        simulated_lookup_count = int(raw_dataset.get("simulated_lookup_count", 0) or 0)
        lines.append(
            (
                "{label}: resolved_source_pairs={resolved}/{pairs} "
                "source_rows={peaks} source_lookup_rows={rows}"
            ).format(
                label=label,
                resolved=resolved_count,
                pairs=pair_count,
                peaks=simulated_peak_count,
                rows=simulated_lookup_count,
            )
        )
        unresolved = [
            item
            for item in diagnostics
            if not bool(
                item.get(
                    "fit_resolved",
                    item.get("strict_resolved", False),
                )
            )
        ]
        if not unresolved:
            if any(
                bool(item.get("fit_resolved", False))
                and not bool(item.get("strict_resolved", False))
                for item in diagnostics
            ):
                lines.append(
                    f"{label}: all saved manual pairs resolved for fit "
                    "(including legacy source-id remaps)."
                )
            else:
                lines.append(f"{label}: all saved manual pairs strictly resolved.")
            continue
        for item in unresolved[:max_pairs]:
            pair_index = int(item.get("pair_index", 0) or 0)
            lines.append(
                (
                    "{label} pair#{pair}: saved_row={saved_row} saved_peak={saved_peak} "
                    "saved_hkl={saved_hkl} q_group={q_group} display={display}"
                ).format(
                    label=label,
                    pair=pair_index,
                    saved_row=_geometry_fit_source_resolution_key_text(
                        item.get("saved_source_row_key")
                    ),
                    saved_peak=_geometry_fit_source_resolution_key_text(
                        item.get("saved_source_peak_key")
                    ),
                    saved_hkl=_geometry_fit_debug_value_text(
                        item.get("saved_hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        item.get("saved_q_group_key"),
                        float_digits=3,
                    ),
                    display=_geometry_fit_debug_value_text(
                        item.get("saved_display_point"),
                        float_digits=3,
                    ),
                )
            )
            lines.append(
                (
                    "{label} pair#{pair}: strict_row={row_status} strict_peak={peak_status} "
                    "fallback={fallback} fallback_row={fallback_row} "
                    "fallback_peak={fallback_peak} fallback_hkl={fallback_hkl} "
                    "fallback_distance_px={distance}"
                ).format(
                    label=label,
                    pair=pair_index,
                    row_status=str(item.get("row_candidate_status", "<unknown>")),
                    peak_status=str(item.get("peak_candidate_status", "<unknown>")),
                    fallback=str(
                        item.get("overlay_resolution_kind", "<none>") or "<none>"
                    ),
                    fallback_row=_geometry_fit_source_resolution_key_text(
                        item.get("overlay_source_row_key")
                    ),
                    fallback_peak=_geometry_fit_source_resolution_key_text(
                        item.get("overlay_source_peak_key")
                    ),
                    fallback_hkl=_geometry_fit_debug_value_text(
                        item.get("overlay_hkl"),
                        float_digits=3,
                    ),
                    distance=_geometry_fit_debug_value_text(
                        item.get("overlay_distance_px"),
                        float_digits=3,
                    ),
                )
            )
            reason = str(item.get("failure_reason", "") or "").strip()
            if reason:
                lines.append(f"{label} pair#{pair_index}: reason={reason}")
        extra_count = len(unresolved) - max_pairs
        if extra_count > 0:
            lines.append(f"{label}: ... {int(extra_count)} more unresolved pair(s) not shown")
    return lines


def _geometry_fit_simulation_diag_int(
    value: object,
) -> int | None:
    """Return one diagnostics count as an integer when possible."""

    try:
        return int(value)
    except Exception:
        return None


def _geometry_fit_short_count_list(
    values: object,
    *,
    limit: int = 8,
) -> str:
    """Return one compact count-list preview for diagnostics logs."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return "<none>"
    ints: list[str] = []
    for raw_value in values[: max(1, int(limit))]:
        count = _geometry_fit_simulation_diag_int(raw_value)
        ints.append(str(count) if count is not None else "?")
    suffix = ""
    try:
        extra_count = len(values) - max(1, int(limit))
    except Exception:
        extra_count = 0
    if extra_count > 0:
        suffix = f", ... +{int(extra_count)}"
    return "[" + ", ".join(ints) + suffix + "]"


def build_geometry_fit_simulation_diagnostic_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format source-snapshot diagnostics for geometry-fit preflight logs."""

    lines: list[str] = []
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("simulation_diagnostics")
        if not isinstance(diagnostics_raw, Mapping):
            continue
        diagnostics = dict(diagnostics_raw)
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        runtime_diag_raw = diagnostics.get("runtime_simulation")
        runtime_diag = (
            dict(runtime_diag_raw)
            if isinstance(runtime_diag_raw, Mapping)
            else {}
        )
        lines.append(
            (
                "{label}: source={source} status={status} consumer={consumer} "
                "created_from={created_from} signature_match={signature_match} "
                "projected_rows={projected} raw_peaks={raw}"
            ).format(
                label=label,
                source=str(diagnostics.get("source", "<unknown>")),
                status=str(diagnostics.get("status", "<unknown>")),
                consumer=str(diagnostics.get("consumer", "<unknown>")),
                created_from=str(diagnostics.get("created_from", "<unknown>")),
                signature_match=_geometry_fit_debug_value_text(
                    diagnostics.get("signature_match", False)
                ),
                projected=_geometry_fit_debug_value_text(
                    diagnostics.get(
                        "projected_peak_count",
                        raw_dataset.get("simulated_peak_count", np.nan),
                    ),
                    float_digits=0,
                ),
                raw=_geometry_fit_debug_value_text(
                    diagnostics.get("raw_peak_count", np.nan),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: requested_signature={requested} stored_signature={stored}"
            ).format(
                label=label,
                requested=_geometry_fit_debug_value_text(
                    diagnostics.get("requested_signature_summary")
                ),
                stored=_geometry_fit_debug_value_text(
                    diagnostics.get("stored_signature_summary")
                ),
            )
        )
        lines.append(
            (
                "{label}: inputs miller_shape={miller} intensity_shape={intensity} "
                "image_size={image_size}"
            ).format(
                label=label,
                miller=_geometry_fit_debug_value_text(
                    diagnostics.get("miller_shape"),
                    float_digits=0,
                ),
                intensity=_geometry_fit_debug_value_text(
                    diagnostics.get("intensity_shape"),
                    float_digits=0,
                ),
                image_size=_geometry_fit_debug_value_text(
                    diagnostics.get("image_size", np.nan),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: missing_params={params} missing_mosaic={mosaic}"
            ).format(
                label=label,
                params=_geometry_fit_debug_value_text(
                    diagnostics.get("missing_param_keys", []),
                    float_digits=0,
                ),
                mosaic=_geometry_fit_debug_value_text(
                    diagnostics.get("missing_mosaic_keys", []),
                    float_digits=0,
                ),
            )
        )
        param_summary_raw = diagnostics.get("param_summary")
        if isinstance(param_summary_raw, Mapping) and param_summary_raw:
            lines.append(
                (
                    "{label}: params a={a} c={c} lambda={lam} theta_initial={theta} "
                    "center={center} n2={n2}"
                ).format(
                    label=label,
                    a=_geometry_fit_debug_value_text(
                        param_summary_raw.get("a", np.nan)
                    ),
                    c=_geometry_fit_debug_value_text(
                        param_summary_raw.get("c", np.nan)
                    ),
                    lam=_geometry_fit_debug_value_text(
                        param_summary_raw.get("lambda", np.nan)
                    ),
                    theta=_geometry_fit_debug_value_text(
                        param_summary_raw.get("theta_initial", np.nan)
                    ),
                    center=_geometry_fit_debug_value_text(
                        param_summary_raw.get("center", None)
                    ),
                    n2=_geometry_fit_debug_value_text(
                        param_summary_raw.get("n2", None)
                    ),
                )
            )
        mosaic_sizes_raw = diagnostics.get("mosaic_array_sizes")
        if isinstance(mosaic_sizes_raw, Mapping) and mosaic_sizes_raw:
            lines.append(
                (
                    "{label}: mosaic_sizes beam_x={beam_x} beam_y={beam_y} "
                    "theta={theta} phi={phi} wavelength={wave} wavelength_i={wave_i} "
                    "sample_weights={weights}"
                ).format(
                    label=label,
                    beam_x=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("beam_x_array", None),
                        float_digits=0,
                    ),
                    beam_y=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("beam_y_array", None),
                        float_digits=0,
                    ),
                    theta=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("theta_array", None),
                        float_digits=0,
                    ),
                    phi=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("phi_array", None),
                        float_digits=0,
                    ),
                    wave=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("wavelength_array", None),
                        float_digits=0,
                    ),
                    wave_i=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("wavelength_i_array", None),
                        float_digits=0,
                    ),
                    weights=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("sample_weights", None),
                        float_digits=0,
                    ),
                )
            )
        if runtime_diag:
            lines.append(
                (
                    "{label}: runtime stage={stage} status={status} "
                    "hit_tables={tables} nonempty_hit_tables={nonempty} "
                    "finite_hit_rows={rows} peak_count={peaks}"
                ).format(
                    label=label,
                    stage=str(runtime_diag.get("stage", "<unknown>")),
                    status=str(runtime_diag.get("status", "<unknown>")),
                    tables=_geometry_fit_debug_value_text(
                        runtime_diag.get("hit_table_count", None),
                        float_digits=0,
                    ),
                    nonempty=_geometry_fit_debug_value_text(
                        runtime_diag.get("nonempty_hit_table_count", None),
                        float_digits=0,
                    ),
                    rows=_geometry_fit_debug_value_text(
                        runtime_diag.get("finite_hit_row_total", None),
                        float_digits=0,
                    ),
                    peaks=_geometry_fit_debug_value_text(
                        runtime_diag.get(
                            "peak_count",
                            runtime_diag.get("peak_center_count", None),
                        ),
                        float_digits=0,
                    ),
                )
            )
            if "hit_row_counts" in runtime_diag:
                lines.append(
                    (
                        "{label}: runtime hit_row_counts={counts}"
                    ).format(
                        label=label,
                        counts=_geometry_fit_short_count_list(
                            runtime_diag.get("hit_row_counts")
                        ),
                    )
                )
        exception_type = diagnostics.get(
            "exception_type",
            runtime_diag.get("exception_type"),
        )
        exception_message = diagnostics.get(
            "exception_message",
            runtime_diag.get("exception_message"),
        )
        if exception_type or exception_message:
            lines.append(
                (
                    "{label}: exception={exc_type}: {exc_message}"
                ).format(
                    label=label,
                    exc_type=str(exception_type or "<unknown>"),
                    exc_message=str(exception_message or "<no message>"),
                )
            )
    return lines


def build_geometry_fit_preflight_log_sections(
    *,
    error_text: str,
    params: Mapping[str, object] | None,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    geometry_runtime_cfg: Mapping[str, object] | None = None,
) -> list[tuple[str, list[str]]]:
    """Build the log sections for a geometry-fit failure before solver start."""

    debug_logging = geometry_fit_debug_logging_enabled(geometry_runtime_cfg)
    sections: list[tuple[str, list[str]]] = [
        (
            "Failure:",
            [
                str(error_text).strip(),
                "stage=preflight",
            ],
        )
    ]
    sections.extend(
        build_geometry_fit_start_log_sections(
            params=params,
            var_names=var_names,
            dataset_infos=dataset_infos,
            current_dataset=current_dataset,
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
            geometry_runtime_cfg=geometry_runtime_cfg,
        )
    )
    if debug_logging:
        source_resolution_lines = build_geometry_fit_source_resolution_log_lines(
            dataset_infos
        )
        simulation_diagnostic_lines = build_geometry_fit_simulation_diagnostic_log_lines(
            dataset_infos
        )
        if simulation_diagnostic_lines:
            sections.append(
                ("Source snapshot diagnostics:", simulation_diagnostic_lines)
            )
        if source_resolution_lines:
            sections.append(("Cached source-row diagnostics:", source_resolution_lines))
    return sections


def write_geometry_fit_preflight_failure_log(
    *,
    stamp: str,
    error_text: str,
    log_path: Path | str,
    log_sections: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> Path:
    """Persist one geometry-fit preflight failure log."""

    resolved_log_path, _log_line, _log_section = _build_geometry_fit_log_writers(
        log_path
    )
    if geometry_fit_all_logging_disabled():
        return resolved_log_path
    _log_line(f"Geometry fit aborted before solver start: {stamp}")
    _log_line("")
    sections = list(log_sections or ())
    if not sections:
        sections = [("Failure:", [str(error_text).strip(), "stage=preflight"])]
    for title, lines in sections:
        _log_section(str(title), [str(line) for line in (lines or ())])
    return resolved_log_path


def write_geometry_fit_run_start_log(
    *,
    stamp: str,
    prepared_run: GeometryFitPreparedRun,
    cmd_line: Callable[[str], None],
    log_line: Callable[[str], None],
    log_section: Callable[[str, Sequence[str]], None],
) -> None:
    """Emit the runtime console/log prelude for one prepared geometry-fit run."""

    cmd_line(str(prepared_run.start_cmd_line))
    log_line(f"Geometry fit started: {stamp}")
    log_line("")
    for title, lines in prepared_run.start_log_sections:
        log_section(title, list(lines))


def should_apply_geometry_fit_runtime_safety_overrides(
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether GUI geometry fitting should force the safe non-Numba runtime."""

    if platform_name is None:
        platform_name = os.name
    if version_info is None:
        version_info = sys.version_info
    if env is None:
        env = os.environ

    opt_out = str(
        env.get("RA_SIM_ALLOW_UNSAFE_GEOMETRY_FIT_RUNTIME", "")
    ).strip().lower()
    if opt_out in {"1", "true", "yes", "on"}:
        return False

    if str(platform_name).strip().lower() != "nt":
        return False

    try:
        major = int(version_info[0])
        minor = int(version_info[1])
    except Exception:
        return False
    return (major, minor) >= (3, 13)


def apply_geometry_fit_runtime_safety_overrides(
    refinement_config: Mapping[str, object] | None,
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str | None]:
    """Return one copied refinement config with GUI runtime safety overrides."""

    if isinstance(refinement_config, Mapping):
        resolved = copy.deepcopy(dict(refinement_config))
    else:
        resolved = {}

    if bool(resolved.get("allow_unsafe_runtime", False)):
        return resolved, None

    if not should_apply_geometry_fit_runtime_safety_overrides(
        platform_name=platform_name,
        version_info=version_info,
        env=env,
    ):
        return resolved, None

    optimizer_cfg_raw = resolved.get("optimizer", None)
    solver_cfg_raw = resolved.get("solver", {})
    if isinstance(optimizer_cfg_raw, Mapping):
        solver_cfg = dict(optimizer_cfg_raw)
    elif isinstance(solver_cfg_raw, Mapping):
        solver_cfg = dict(solver_cfg_raw)
    else:
        solver_cfg = {}

    changed = False
    if bool(resolved.get("use_numba", True)):
        resolved["use_numba"] = False
        changed = True

    if isinstance(optimizer_cfg_raw, Mapping):
        resolved["optimizer"] = solver_cfg
    resolved["solver"] = solver_cfg
    if not changed:
        return resolved, None

    return (
        resolved,
        (
            "Windows/Python 3.13 runtime guard enabled: "
            "geometry fit keeps its configured parallel workers with Numba disabled."
        ),
    )


def build_geometry_fit_solver_request(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> GeometryFitSolverRequest:
    """Build one concrete solver request from a prepared geometry-fit run."""

    refinement_config, runtime_safety_note = (
        apply_geometry_fit_runtime_safety_overrides(
            prepared_run.geometry_runtime_cfg,
        )
    )
    refinement_config = (
        copy.deepcopy(dict(refinement_config))
        if isinstance(refinement_config, Mapping)
        else {}
    )
    optimizer_cfg_raw = refinement_config.get("optimizer", None)
    solver_cfg_raw = refinement_config.get("solver", optimizer_cfg_raw)
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}
    if solver_cfg.get("workers") is None:
        solver_cfg["workers"] = "auto"
    if solver_cfg.get("parallel_mode") is None:
        solver_cfg["parallel_mode"] = "auto"
    if solver_cfg.get("worker_numba_threads") is None:
        solver_cfg["worker_numba_threads"] = 0
    refinement_config["solver"] = solver_cfg
    if isinstance(optimizer_cfg_raw, Mapping):
        optimizer_cfg = dict(optimizer_cfg_raw)
        if optimizer_cfg.get("workers") is None:
            optimizer_cfg["workers"] = solver_cfg["workers"]
        if optimizer_cfg.get("parallel_mode") is None:
            optimizer_cfg["parallel_mode"] = solver_cfg["parallel_mode"]
        if optimizer_cfg.get("worker_numba_threads") is None:
            optimizer_cfg["worker_numba_threads"] = solver_cfg["worker_numba_threads"]
        refinement_config["optimizer"] = optimizer_cfg

    return GeometryFitSolverRequest(
        miller=solver_inputs.miller,
        intensities=solver_inputs.intensities,
        image_size=int(solver_inputs.image_size),
        params=dict(prepared_run.fit_params),
        measured_peaks=prepared_run.current_dataset["measured_for_fit"],
        var_names=[str(name) for name in var_names],
        candidate_param_names=(
            [
                str(name)
                for name in (
                    refinement_config.get("candidate_param_names", [])
                    if isinstance(refinement_config, Mapping)
                    else []
                )
            ]
            or None
        ),
        dataset_specs=list(prepared_run.dataset_specs),
        refinement_config=refinement_config,
        runtime_safety_note=runtime_safety_note,
    )


def solve_geometry_fit_request(
    request: GeometryFitSolverRequest,
    *,
    solve_fit: Callable[..., object],
    status_callback: Callable[[str], None] | None = None,
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> object:
    """Invoke the live geometry-fit solver for one prepared request."""

    solve_kwargs: dict[str, object] = {
        "pixel_tol": float("inf"),
        "experimental_image": None,
        "dataset_specs": request.dataset_specs,
        "refinement_config": request.refinement_config,
    }
    signature = None
    accepts_var_kwargs = False
    try:
        signature = inspect.signature(solve_fit)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if request.candidate_param_names is not None and (
            "candidate_param_names" in signature.parameters or accepts_var_kwargs
        ):
            solve_kwargs["candidate_param_names"] = request.candidate_param_names
    if callable(status_callback):
        if signature is not None:
            accepts_status_callback = (
                "status_callback" in signature.parameters
                or accepts_var_kwargs
            )
            if accepts_status_callback:
                solve_kwargs["status_callback"] = status_callback
    if callable(live_update_callback):
        if signature is not None:
            accepts_live_update_callback = (
                "live_update_callback" in signature.parameters
                or accepts_var_kwargs
            )
            if accepts_live_update_callback:
                solve_kwargs["live_update_callback"] = live_update_callback

    return solve_fit(
        request.miller,
        request.intensities,
        request.image_size,
        request.params,
        request.measured_peaks,
        request.var_names,
        **solve_kwargs,
    )


def apply_geometry_fit_result_values(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var: object | None = None,
) -> None:
    """Apply fitted geometry values back to the live UI variables."""

    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(value):
            continue

        param_name = str(name)
        if param_name == "theta_offset":
            if geometry_theta_offset_var is not None:
                geometry_theta_offset_var.set(f"{value:.6g}")
            continue

        var = var_map.get(param_name)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            continue


def _geometry_fit_normalize_hkl(raw_hkl: object) -> tuple[int, int, int] | object:
    """Normalize one HKL triplet when possible."""

    if (
        isinstance(raw_hkl, Sequence)
        and not isinstance(raw_hkl, (str, bytes))
        and len(raw_hkl) == 3
    ):
        try:
            return tuple(int(v) for v in raw_hkl)
        except Exception:
            return raw_hkl
    return raw_hkl


def build_geometry_fit_export_records(
    point_match_diagnostics: Sequence[object] | None = None,
    *,
    agg_millers: Sequence[Sequence[object]] | None = None,
    agg_sim_coords: Sequence[Sequence[object]] | None = None,
    agg_meas_coords: Sequence[Sequence[object]] | None = None,
    pixel_offsets: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, object]]:
    """Build one export row per manual point from final diagnostics."""

    if agg_millers is not None and agg_sim_coords is not None and agg_meas_coords is not None and pixel_offsets is not None:
        export_recs: list[dict[str, object]] = []
        for source_label, coords in (("sim", agg_sim_coords), ("meas", agg_meas_coords)):
            for hkl, (x, y), offset in zip(agg_millers, coords, pixel_offsets):
                try:
                    hkl_triplet = tuple(int(v) for v in hkl[:3])
                except Exception:
                    continue
                try:
                    dist = float(offset[3])
                except Exception:
                    dist = float("nan")
                export_recs.append(
                    {
                        "source": str(source_label),
                        "hkl": hkl_triplet,
                        "x": int(x),
                        "y": int(y),
                        "dist_px": dist,
                    }
                )
        return export_recs

    export_recs: list[dict[str, object]] = []
    for raw_entry in point_match_diagnostics or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        resolution_reason = entry.get(
            "correspondence_resolution_reason",
            entry.get("resolution_reason"),
        )
        export_recs.append(
            {
                "dataset_index": int(entry.get("dataset_index", 0)),
                "overlay_match_index": int(entry.get("overlay_match_index", -1)),
                "match_status": str(entry.get("match_status", "")),
                "hkl": _geometry_fit_normalize_hkl(entry.get("hkl")),
                "measured_x": float(entry.get("measured_x", np.nan)),
                "measured_y": float(entry.get("measured_y", np.nan)),
                "simulated_x": float(entry.get("simulated_x", np.nan)),
                "simulated_y": float(entry.get("simulated_y", np.nan)),
                "dx_px": float(entry.get("dx_px", np.nan)),
                "dy_px": float(entry.get("dy_px", np.nan)),
                "distance_px": float(entry.get("distance_px", np.nan)),
                "source_table_index": entry.get("source_table_index"),
                "source_row_index": entry.get("source_row_index"),
                "source_peak_index": entry.get("source_peak_index"),
                "resolution_kind": str(entry.get("resolution_kind", "")),
                "resolution_reason": (
                    None if resolution_reason is None else str(resolution_reason)
                ),
            }
        )
    return export_recs


def build_geometry_fit_optimizer_diagnostics_lines(result: object) -> list[str]:
    """Format optimizer diagnostics for one geometry-fit result."""

    lines = [
        f"success={getattr(result, 'success', False)}",
        f"status={getattr(result, 'status', '')}",
        f"message={(getattr(result, 'message', '') or '').strip()}",
        f"nfev={getattr(result, 'nfev', '<unknown>')}",
        f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
        f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
        f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
        f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
        f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
        f"active_mask={list(getattr(result, 'active_mask', []))}",
    ]
    optimizer_method = str(getattr(result, "optimizer_method", "") or "").strip()
    if optimizer_method:
        lines.append(f"optimizer_method={optimizer_method}")
    weighted_rms = _geometry_fit_metric_float(
        getattr(result, "weighted_residual_rms_px", np.nan)
    )
    if np.isfinite(weighted_rms):
        lines.append(f"weighted_residual_rms_px={weighted_rms:.6f}")
    display_rms = geometry_fit_result_rms(result)
    if np.isfinite(display_rms):
        lines.append(f"display_rms_px={display_rms:.6f}")
    final_metric_name = str(getattr(result, "final_metric_name", "") or "").strip()
    if final_metric_name:
        lines.append(f"final_metric_name={final_metric_name}")
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.append(f"solver_discrete_mode={mode_label}")
    bound_hits = getattr(result, "bound_hits", None)
    if isinstance(bound_hits, Sequence) and not isinstance(
        bound_hits, (str, bytes, bytearray)
    ):
        bound_hit_names = [str(name) for name in bound_hits if str(name).strip()]
        if bound_hit_names:
            lines.append(f"bound_hits=[{', '.join(bound_hit_names)}]")
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()
    if boundary_warning:
        lines.append(f"boundary_warning={boundary_warning}")
    for entry in getattr(result, "restart_history", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                idx=int(entry.get("restart", -1)),
                cost=float(entry.get("cost", np.nan)),
                success=bool(entry.get("success", False)),
                msg=str(entry.get("message", "")).strip(),
            )
        )
    return lines


def geometry_fit_result_rms(result: object) -> float:
    """Resolve the displayed RMS residual from one geometry-fit result."""

    try:
        direct_rms = float(getattr(result, "rms_px", np.nan))
    except Exception:
        direct_rms = float("nan")
    if np.isfinite(direct_rms):
        return float(direct_rms)

    fun = getattr(result, "fun", None)
    if fun is None:
        return 0.0
    try:
        residuals = np.asarray(fun, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    finite_residuals = residuals[np.isfinite(residuals)]
    if finite_residuals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite_residuals**2)))


def _geometry_fit_metric_float(value: object, *, default: float = np.nan) -> float:
    """Return one finite float metric, or ``default`` when unavailable."""

    try:
        resolved = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(resolved):
        return float(default)
    return float(resolved)


def _geometry_fit_debug_value_text(
    value: object,
    *,
    float_digits: int = 6,
) -> str:
    """Format one geometry-fit debug value for log output."""

    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.{int(float_digits)}f}"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key}={_geometry_fit_debug_value_text(val, float_digits=float_digits)}"
            for key, val in value.items()
        ) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [
            _geometry_fit_debug_value_text(item, float_digits=float_digits)
            for item in items[:6]
        ]
        if len(items) > 6:
            preview.append("...")
        return "[" + ", ".join(preview) + "]"
    return str(value)


def build_geometry_fit_debug_lines(result: object) -> list[str]:
    """Format solver setup/progress details for one geometry-fit result."""

    debug_summary = getattr(result, "geometry_fit_debug_summary", None)
    if not isinstance(debug_summary, Mapping):
        return []

    lines: list[str] = [
        "point_match_mode={mode} datasets={datasets} vars={vars}".format(
            mode=bool(debug_summary.get("point_match_mode", False)),
            datasets=int(debug_summary.get("dataset_count", 0) or 0),
            vars=",".join(str(name) for name in debug_summary.get("var_names", []) or ()),
        )
    ]

    solver = debug_summary.get("solver", None)
    if isinstance(solver, Mapping):
        lines.append(
            "solver loss={loss} f_scale_px={f_scale} max_nfev={max_nfev} "
            "restarts={restarts} weighted_matching={weighted} "
            "missing_pair_penalty_px={missing} hk0_peak_priority_weight={hk0} "
            "measurement_uncertainty={unc} "
            "anisotropic_uncertainty={anisotropic} full_beam_polish={full_beam} "
            "full_beam_radius_px={full_beam_radius}".format(
                loss=str(solver.get("loss", "<unknown>")),
                f_scale=_geometry_fit_debug_value_text(solver.get("f_scale_px", np.nan)),
                max_nfev=_geometry_fit_debug_value_text(solver.get("max_nfev", "<unknown>"), float_digits=0),
                restarts=_geometry_fit_debug_value_text(solver.get("restarts", "<unknown>"), float_digits=0),
                weighted=_geometry_fit_debug_value_text(solver.get("weighted_matching", False), float_digits=0),
                missing=_geometry_fit_debug_value_text(solver.get("missing_pair_penalty_px", np.nan)),
                hk0=_geometry_fit_debug_value_text(solver.get("hk0_peak_priority_weight", np.nan)),
                unc=_geometry_fit_debug_value_text(solver.get("use_measurement_uncertainty", False), float_digits=0),
                anisotropic=_geometry_fit_debug_value_text(
                    solver.get("anisotropic_measurement_uncertainty", False),
                    float_digits=0,
                ),
                full_beam=_geometry_fit_debug_value_text(
                    solver.get("full_beam_polish_enabled", False),
                    float_digits=0,
                ),
                full_beam_radius=_geometry_fit_debug_value_text(
                    solver.get("full_beam_polish_match_radius_px", np.nan),
                ),
            )
        )

    parallel = debug_summary.get("parallelization", None)
    if isinstance(parallel, Mapping):
        lines.append(
            "parallel mode={mode} configured_workers={configured} "
            "dataset_workers={datasets} restart_workers={restarts} "
            "worker_numba_threads={threads} thread_budget={budget}".format(
                mode=str(parallel.get("mode", "<unknown>")),
                configured=_geometry_fit_debug_value_text(parallel.get("configured_workers", "<unknown>"), float_digits=0),
                datasets=_geometry_fit_debug_value_text(parallel.get("dataset_workers", "<unknown>"), float_digits=0),
                restarts=_geometry_fit_debug_value_text(parallel.get("restart_workers", "<unknown>"), float_digits=0),
                threads=_geometry_fit_debug_value_text(parallel.get("worker_numba_threads", "None"), float_digits=0),
                budget=_geometry_fit_debug_value_text(parallel.get("numba_thread_budget", "<unknown>"), float_digits=0),
            )
        )

    seed_search = debug_summary.get("seed_search", None)
    if isinstance(seed_search, Mapping):
        lines.append(
            "seed_search prescore_top_k={top_k} n_global={n_global} "
            "n_jitter={n_jitter} jitter_sigma_u={jitter} "
            "min_seed_separation_u={separation} "
            "trusted_prior_fraction_of_span={trusted_prior}".format(
                top_k=_geometry_fit_debug_value_text(
                    seed_search.get("prescore_top_k", 0),
                    float_digits=0,
                ),
                n_global=_geometry_fit_debug_value_text(
                    seed_search.get("n_global", 0),
                    float_digits=0,
                ),
                n_jitter=_geometry_fit_debug_value_text(
                    seed_search.get("n_jitter", 0),
                    float_digits=0,
                ),
                jitter=_geometry_fit_debug_value_text(
                    seed_search.get("jitter_sigma_u", np.nan)
                ),
                separation=_geometry_fit_debug_value_text(
                    seed_search.get("min_seed_separation_u", np.nan)
                ),
                trusted_prior=_geometry_fit_debug_value_text(
                    seed_search.get("trusted_prior_fraction_of_span", np.nan)
                ),
            )
        )

    discrete_modes = debug_summary.get("discrete_modes", None)
    if isinstance(discrete_modes, Sequence) and not isinstance(
        discrete_modes, (str, bytes, bytearray)
    ):
        mode_labels = [
            _geometry_fit_discrete_mode_label(mode)
            for mode in discrete_modes
            if isinstance(mode, Mapping)
        ]
        mode_labels = [label for label in mode_labels if label]
        selected_label = str(debug_summary.get("selected_mode_label", "") or "").strip()
        line = f"discrete_modes count={int(len(list(discrete_modes)))}"
        if selected_label:
            line += f" selected={selected_label}"
        if mode_labels:
            line += " labels=[{labels}]".format(labels=", ".join(mode_labels[:6]))
            if len(mode_labels) > 6:
                line += ", ..."
        lines.append(line)

    for idx, entry in enumerate(debug_summary.get("mode_seed_summary", []) or []):
        if not isinstance(entry, Mapping):
            continue
        mode_label = _geometry_fit_discrete_mode_label(entry.get("mode"))
        selected_seeds = entry.get("selected_seeds", [])
        best_seed_cost = float("nan")
        best_seed_label = ""
        if isinstance(selected_seeds, Sequence) and not isinstance(
            selected_seeds, (str, bytes, bytearray)
        ):
            for raw_seed in selected_seeds:
                if not isinstance(raw_seed, Mapping):
                    continue
                best_seed_cost = _geometry_fit_metric_float(
                    raw_seed.get("cost", np.nan)
                )
                best_seed_label = "{kind}:{label}".format(
                    kind=str(raw_seed.get("seed_kind", "")).strip() or "?",
                    label=str(raw_seed.get("seed_label", "")).strip() or "?",
                )
                break
        lines.append(
            "mode_seed[{idx}] label={label} total_seeds={seed_count} "
            "selected_seeds={selected} best_selected={best_label} "
            "best_selected_cost={best_cost}".format(
                idx=int(idx),
                label=mode_label or "<unknown>",
                seed_count=_geometry_fit_debug_value_text(
                    entry.get("seed_count", 0),
                    float_digits=0,
                ),
                selected=_geometry_fit_debug_value_text(
                    entry.get("selected_seed_count", 0),
                    float_digits=0,
                ),
                best_label=best_seed_label or "<none>",
                best_cost=_geometry_fit_debug_value_text(best_seed_cost),
            )
        )

    main_seed = debug_summary.get("main_solve_seed", None)
    if isinstance(main_seed, Mapping):
        lines.append(
            "main_seed kind={kind} label={label} cost={cost} weighted_rms_px={rms}".format(
                kind=str(main_seed.get("seed_kind", "") or "?"),
                label=str(main_seed.get("seed_label", "") or "?"),
                cost=_geometry_fit_debug_value_text(main_seed.get("cost", np.nan)),
                rms=_geometry_fit_debug_value_text(main_seed.get("weighted_rms_px", np.nan)),
            )
        )
        point_seed = main_seed.get("point_match_summary", None)
        if isinstance(point_seed, Mapping):
            lines.append(
                "main_seed_point_match matched={matched} missing={missing} "
                "seed_rms_px={peak_rms} peak_max_px={peak_max}".format(
                    matched=_geometry_fit_debug_value_text(point_seed.get("matched_pair_count", 0), float_digits=0),
                    missing=_geometry_fit_debug_value_text(point_seed.get("missing_pair_count", 0), float_digits=0),
                    peak_rms=_geometry_fit_debug_value_text(point_seed.get("unweighted_peak_rms_px", np.nan)),
                    peak_max=_geometry_fit_debug_value_text(point_seed.get("unweighted_peak_max_px", np.nan)),
                )
            )

    for entry in debug_summary.get("dataset_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "dataset[{idx}] label={label} theta_initial_deg={theta} measured={measured} "
            "subset_reflections={subset}/{total} fixed_source_reflections={fixed} "
            "fallback_hkls={fallback} reduced={reduced}".format(
                idx=_geometry_fit_debug_value_text(entry.get("dataset_index", -1), float_digits=0),
                label=str(entry.get("label", "")),
                theta=_geometry_fit_debug_value_text(entry.get("theta_initial_deg", np.nan)),
                measured=_geometry_fit_debug_value_text(entry.get("measured_count", 0), float_digits=0),
                subset=_geometry_fit_debug_value_text(entry.get("subset_reflection_count", 0), float_digits=0),
                total=_geometry_fit_debug_value_text(entry.get("total_reflection_count", 0), float_digits=0),
                fixed=_geometry_fit_debug_value_text(entry.get("fixed_source_reflection_count", 0), float_digits=0),
                fallback=_geometry_fit_debug_value_text(entry.get("fallback_hkl_count", 0), float_digits=0),
                reduced=_geometry_fit_debug_value_text(entry.get("subset_reduced", False), float_digits=0),
            )
        )

    for entry in debug_summary.get("parameter_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        line = (
            "param[{name}] group={group} start={start} final={final} delta={delta} "
            "bounds=[{lower}, {upper}] scale={scale}".format(
                name=str(entry.get("name", "")),
                group=str(entry.get("group", "other")),
                start=_geometry_fit_debug_value_text(entry.get("start", np.nan)),
                final=_geometry_fit_debug_value_text(entry.get("final", np.nan)),
                delta=_geometry_fit_debug_value_text(entry.get("delta", np.nan)),
                lower=_geometry_fit_debug_value_text(entry.get("lower_bound", np.nan)),
                upper=_geometry_fit_debug_value_text(entry.get("upper_bound", np.nan)),
                scale=_geometry_fit_debug_value_text(entry.get("scale", np.nan)),
            )
        )
        if bool(entry.get("prior_enabled", False)):
            line += (
                " prior_center={center} prior_sigma={sigma}".format(
                    center=_geometry_fit_debug_value_text(entry.get("prior_center", np.nan)),
                    sigma=_geometry_fit_debug_value_text(entry.get("prior_sigma", np.nan)),
                )
            )
        lines.append(line)

    final_summary = debug_summary.get("final", None)
    if isinstance(final_summary, Mapping):
        lines.append(
            "final metric={metric} cost={cost} robust_cost={robust} "
            "weighted_rms_px={weighted_rms} final_full_beam_rms_px={display_rms}".format(
                metric=str(final_summary.get("metric_name", "") or "<unknown>"),
                cost=_geometry_fit_debug_value_text(final_summary.get("cost", np.nan)),
                robust=_geometry_fit_debug_value_text(final_summary.get("robust_cost", np.nan)),
                weighted_rms=_geometry_fit_debug_value_text(final_summary.get("weighted_rms_px", np.nan)),
                display_rms=_geometry_fit_debug_value_text(
                    final_summary.get(
                        "final_full_beam_rms_px",
                        final_summary.get("display_rms_px", np.nan),
                    )
                ),
            )
        )

    solve_counts = debug_summary.get("solve_counts", None)
    if isinstance(solve_counts, Mapping):
        lines.append(
            "solve_counts prescored={prescored} solved={solved}".format(
                prescored=_geometry_fit_debug_value_text(
                    solve_counts.get("prescored", 0),
                    float_digits=0,
                ),
                solved=_geometry_fit_debug_value_text(
                    solve_counts.get("solved", 0),
                    float_digits=0,
                ),
            )
        )

    solve_progress = debug_summary.get("solve_progress", None)
    if isinstance(solve_progress, Mapping):
        lines.append(
            "solve_progress label={label} evaluations={evals} best_cost={best_cost} "
            "last_cost={last_cost} best_weighted_rms_px={best_rms} "
            "last_weighted_rms_px={last_rms} status_updates={updates} "
            "aborted_early={aborted}".format(
                label=str(solve_progress.get("label", "")),
                evals=_geometry_fit_debug_value_text(solve_progress.get("evaluation_count", 0), float_digits=0),
                best_cost=_geometry_fit_debug_value_text(solve_progress.get("best_cost_seen", np.nan)),
                last_cost=_geometry_fit_debug_value_text(solve_progress.get("last_cost_seen", np.nan)),
                best_rms=_geometry_fit_debug_value_text(solve_progress.get("best_weighted_rms_px", np.nan)),
                last_rms=_geometry_fit_debug_value_text(solve_progress.get("last_weighted_rms_px", np.nan)),
                updates=_geometry_fit_debug_value_text(solve_progress.get("status_emit_count", 0), float_digits=0),
                aborted=_geometry_fit_debug_value_text(
                    solve_progress.get("aborted_early", False),
                    float_digits=0,
                ),
            )
        )
        early_stop_reason = str(
            solve_progress.get("early_stop_reason", "") or ""
        ).strip()
        if early_stop_reason:
            lines.append(f"solve_progress early_stop_reason={early_stop_reason}")
        for idx, event in enumerate(solve_progress.get("trace", []) or []):
            if not isinstance(event, Mapping):
                continue
            lines.append(
                "solve_progress[{idx}] eval={eval} reason={reason} cost={cost} "
                "best_cost={best_cost} weighted_rms_px={rms}".format(
                    idx=int(idx),
                    eval=_geometry_fit_debug_value_text(event.get("eval", 0), float_digits=0),
                    reason=str(event.get("reason", "")),
                    cost=_geometry_fit_debug_value_text(event.get("current_cost", np.nan)),
                    best_cost=_geometry_fit_debug_value_text(event.get("best_cost", np.nan)),
                    rms=_geometry_fit_debug_value_text(event.get("weighted_rms_px", np.nan)),
                )
            )

    return lines


def _geometry_fit_discrete_mode_label(mode: object) -> str:
    """Return one compact discrete-mode label from a mode mapping."""

    if not isinstance(mode, Mapping):
        return ""
    explicit = str(mode.get("label", "") or "").strip()
    if explicit:
        return explicit
    parts: list[str] = []
    try:
        rot90 = int(mode.get("rot90", mode.get("k", 0))) % 4
    except Exception:
        rot90 = 0
    if rot90:
        parts.append(f"rot90={rot90}")
    if bool(mode.get("flip_x", False)):
        parts.append("flip_x")
    if bool(mode.get("flip_y", False)):
        parts.append("flip_y")
    return "+".join(parts) if parts else "identity"


def build_geometry_fit_stage_summary_lines(result: object) -> list[str]:
    """Format the stage-by-stage geometry-fit workflow summaries."""

    stage_specs = [
        ("reparameterization", getattr(result, "reparameterization_summary", None)),
        ("staged_release", getattr(result, "staged_release_summary", None)),
        ("adaptive_regularization", getattr(result, "adaptive_regularization_summary", None)),
        ("full_beam_polish", getattr(result, "full_beam_polish_summary", None)),
        ("ridge_refinement", getattr(result, "ridge_refinement_summary", None)),
        ("image_refinement", getattr(result, "image_refinement_summary", None)),
        ("auto_freeze", getattr(result, "auto_freeze_summary", None)),
        ("selective_thaw", getattr(result, "selective_thaw_summary", None)),
    ]
    preferred_keys = (
        "status",
        "reason",
        "accepted",
        "success",
        "start_cost",
        "final_cost",
        "regularized_cost",
        "release_cost",
        "seed_correspondence_count",
        "matched_pair_count_before",
        "matched_pair_count_after",
        "fixed_source_resolved_count",
        "start_rms_px",
        "final_rms_px",
        "accepted_stage_count",
        "release_accepted",
        "fixed_parameters",
        "thawed_parameters",
        "applied_parameters",
        "remaining_fixed_parameters",
        "nfev",
        "stage_nfev",
        "stage_success",
        "stage_message",
        "max_nfev",
        "point_cost_before",
        "point_cost_after",
        "point_cost_limit",
        "point_rms_before_px",
        "point_rms_after_px",
        "point_rms_limit_px",
        "ridge_cost_before",
        "ridge_cost_after",
        "min_rois",
    )

    lines: list[str] = []
    for stage_name, summary in stage_specs:
        if not isinstance(summary, Mapping):
            continue
        parts = [f"{stage_name}:"]
        for key in preferred_keys:
            if key not in summary:
                continue
            value = summary.get(key)
            if value in ("", None):
                continue
            parts.append(
                f"{key}={_geometry_fit_debug_value_text(value)}"
            )
        if len(parts) == 1:
            parts.append("summary=<empty>")
        matched_before = _geometry_fit_metric_float(
            summary.get("matched_pair_count_before", np.nan)
        )
        matched_after = _geometry_fit_metric_float(
            summary.get("matched_pair_count_after", np.nan)
        )
        if np.isfinite(matched_before) and np.isfinite(matched_after):
            parts.append(
                "matched_pair_delta={delta}".format(
                    delta=int(round(matched_after - matched_before))
                )
            )
        start_rms = _geometry_fit_metric_float(summary.get("start_rms_px", np.nan))
        final_rms = _geometry_fit_metric_float(summary.get("final_rms_px", np.nan))
        if np.isfinite(start_rms) and np.isfinite(final_rms):
            parts.append(
                "rms_delta_px={delta}".format(
                    delta=_geometry_fit_debug_value_text(final_rms - start_rms)
                )
            )
        point_cost_before = _geometry_fit_metric_float(
            summary.get("point_cost_before", np.nan)
        )
        point_cost_after = _geometry_fit_metric_float(
            summary.get("point_cost_after", np.nan)
        )
        if np.isfinite(point_cost_before) and np.isfinite(point_cost_after):
            parts.append(
                "point_cost_delta={delta}".format(
                    delta=_geometry_fit_debug_value_text(
                        point_cost_after - point_cost_before
                    )
                )
            )
        ridge_cost_before = _geometry_fit_metric_float(
            summary.get("ridge_cost_before", np.nan)
        )
        ridge_cost_after = _geometry_fit_metric_float(
            summary.get("ridge_cost_after", np.nan)
        )
        if np.isfinite(ridge_cost_before) and np.isfinite(ridge_cost_after):
            parts.append(
                "ridge_cost_delta={delta}".format(
                    delta=_geometry_fit_debug_value_text(
                        ridge_cost_after - ridge_cost_before
                    )
                )
            )
        lines.append(" ".join(parts))
    return lines


def _geometry_fit_selected_discrete_mode_label(result: object) -> str | None:
    """Return the selected solver-side discrete mode label, if any."""

    discrete_summary = getattr(result, "discrete_mode_summary", None)
    if isinstance(discrete_summary, Mapping):
        label = str(discrete_summary.get("selected_label", "") or "").strip()
        if label:
            return label

    chosen_mode = getattr(result, "chosen_discrete_mode", None)
    label = _geometry_fit_discrete_mode_label(chosen_mode)
    return label or None


def _geometry_fit_combo_text(combo: object) -> str:
    if not isinstance(combo, Mapping):
        return "<none>"
    pieces: list[str] = []
    for name, raw_weight in combo.items():
        try:
            weight = float(raw_weight)
        except Exception:
            continue
        pieces.append(f"{name}={weight:+.3f}")
    return ", ".join(pieces) if pieces else "<none>"


def _geometry_fit_effective_orientation_choice(
    *,
    native_shape: tuple[int, int],
    orientation_choice: Mapping[str, object] | None,
    result: object | None = None,
) -> dict[str, object]:
    """Compose the GUI-selected orientation with any solver-side discrete mode."""

    base_choice = (
        dict(orientation_choice)
        if isinstance(orientation_choice, Mapping)
        else {
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        }
    )
    chosen_mode = getattr(result, "chosen_discrete_mode", None)
    if not isinstance(chosen_mode, Mapping):
        return base_choice

    solver_mode = {
        "indexing_mode": str(chosen_mode.get("indexing_mode", "xy")),
        "k": int(chosen_mode.get("k", 0)),
        "flip_x": bool(chosen_mode.get("flip_x", False)),
        "flip_y": bool(chosen_mode.get("flip_y", False)),
        "flip_order": str(chosen_mode.get("flip_order", "yx")),
    }
    if (
        solver_mode["indexing_mode"] == "xy"
        and int(solver_mode["k"]) % 4 == 0
        and not bool(solver_mode["flip_x"])
        and not bool(solver_mode["flip_y"])
    ):
        return base_choice

    from ra_sim.gui import geometry_overlay as gui_geometry_overlay

    return gui_geometry_overlay.compose_orientation_transforms(
        native_shape,
        base_choice,
        solver_mode,
    )


def _build_geometry_fit_identifiability_scope_lines(
    label: str,
    summary: Mapping[str, object],
) -> list[str]:
    status = str(summary.get("status", "unknown"))
    lines = [
        (
            f"{label}: status={status} "
            f"rank={_geometry_fit_debug_value_text(summary.get('rank', '?'), float_digits=0)}/"
            f"{_geometry_fit_debug_value_text(summary.get('num_parameters', '?'), float_digits=0)} "
            f"residuals={_geometry_fit_debug_value_text(summary.get('residual_count', '?'), float_digits=0)} "
            f"underconstrained={_geometry_fit_debug_value_text(summary.get('underconstrained', False), float_digits=0)} "
            f"priors={_geometry_fit_debug_value_text(summary.get('includes_priors', False), float_digits=0)}"
        )
    ]

    singular_values = summary.get("singular_values", [])
    if isinstance(singular_values, Sequence) and not isinstance(
        singular_values, (str, bytes, bytearray)
    ):
        singular_array = np.asarray(list(singular_values), dtype=float).reshape(-1)
        if singular_array.size:
            preview = ", ".join(
                f"{float(value):.3e}" for value in singular_array[: min(5, singular_array.size)]
            )
            if singular_array.size > 5:
                preview += ", ..."
            lines.append(f"{label}: singular_values=[{preview}]")

    weak_parameters = summary.get("weak_parameters", [])
    if isinstance(weak_parameters, Sequence) and not isinstance(
        weak_parameters, (str, bytes, bytearray)
    ):
        names = [
            str(entry.get("name", "")).strip()
            for entry in weak_parameters
            if isinstance(entry, Mapping) and str(entry.get("name", "")).strip()
        ]
        if names:
            lines.append(f"{label}: weak_parameters={', '.join(names)}")

    weak_combinations = summary.get("weak_combinations", [])
    if isinstance(weak_combinations, Sequence) and not isinstance(
        weak_combinations, (str, bytes, bytearray)
    ):
        for idx, entry in enumerate(list(weak_combinations)[:3]):
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                f"{label}: weak_combo[{idx}] "
                f"sv_rel={_geometry_fit_debug_value_text(entry.get('sv_rel', np.nan))} "
                f"combo={_geometry_fit_combo_text(entry.get('combo', {}))}"
            )

    return lines


def build_geometry_fit_identifiability_lines(result: object) -> list[str]:
    """Format solver-conditioned and data-only identifiability diagnostics."""

    lines: list[str] = []
    solver_summary = getattr(result, "identifiability_summary", None)
    if isinstance(solver_summary, Mapping):
        lines.extend(
            _build_geometry_fit_identifiability_scope_lines(
                "solver_conditioned",
                solver_summary,
            )
        )

    data_summary = getattr(result, "data_only_identifiability_summary", None)
    if isinstance(data_summary, Mapping):
        lines.extend(
            _build_geometry_fit_identifiability_scope_lines(
                "data_only",
                data_summary,
            )
        )

    recommendations = getattr(result, "next_stage_recommendations", None)
    if not isinstance(recommendations, Sequence) or isinstance(
        recommendations, (str, bytes, bytearray)
    ):
        recommendations = []
    if recommendations:
        for idx, entry in enumerate(list(recommendations)[:3]):
            if not isinstance(entry, Mapping):
                continue
            params = [
                str(name)
                for name in entry.get("params", [])
                if str(name).strip()
            ]
            lines.append(
                "next_stage[{idx}] params={params} rank_gain={rank_gain} "
                "min_sv_gain={min_sv_gain} block_independence={indep} "
                "use_soft_prior={use_soft_prior} reason={reason}".format(
                    idx=int(idx),
                    params=",".join(params) if params else "<none>",
                    rank_gain=_geometry_fit_debug_value_text(
                        entry.get("rank_gain", np.nan),
                        float_digits=0,
                    ),
                    min_sv_gain=_geometry_fit_debug_value_text(
                        entry.get("min_sv_gain", np.nan)
                    ),
                    indep=_geometry_fit_debug_value_text(
                        entry.get("block_independence", np.nan)
                    ),
                    use_soft_prior=_geometry_fit_debug_value_text(
                        entry.get("use_soft_prior", False),
                        float_digits=0,
                    ),
                    reason=str(entry.get("reason", "")).strip(),
                )
            )
    elif isinstance(data_summary, Mapping) and bool(data_summary.get("enabled", False)):
        lines.append("next_stage: no thaw recommendation")

    return lines


def build_geometry_fit_rejection_reason_lines(
    result: object,
    *,
    rms: float,
) -> list[str]:
    """Return human-readable rejection reasons for one geometry-fit result."""

    reasons: list[str] = []

    early_stop_reason = getattr(result, "early_stop_reason", None)
    if not early_stop_reason:
        geometry_fit_progress = getattr(result, "geometry_fit_progress", None)
        if isinstance(geometry_fit_progress, Mapping):
            early_stop_reason = geometry_fit_progress.get("early_stop_reason")
    if isinstance(early_stop_reason, str) and early_stop_reason.strip():
        reasons.append(str(early_stop_reason).strip())

    if not bool(getattr(result, "success", True)):
        reasons.append("Optimizer did not report success.")

    if not np.isfinite(rms):
        reasons.append("RMS residual is not finite.")
    elif float(rms) > GEOMETRY_FIT_ACCEPT_MAX_RMS_PX:
        reasons.append(
            "RMS residual {rms:.2f} px exceeds the acceptance limit of "
            "{limit:.2f} px.".format(
                rms=float(rms),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_RMS_PX),
            )
        )

    point_match_summary = getattr(result, "point_match_summary", None)
    matched_pair_count = 0
    has_matched_pair_count = False
    max_offset = float("nan")
    if isinstance(point_match_summary, Mapping):
        if "matched_pair_count" in point_match_summary:
            has_matched_pair_count = True
            try:
                matched_pair_count = int(point_match_summary.get("matched_pair_count", 0))
            except Exception:
                matched_pair_count = 0
        max_offset = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_max_px", np.nan)
        )

    if has_matched_pair_count and matched_pair_count <= 0:
        reasons.append("No matched peak pairs were available for the fitted solution.")
    if np.isfinite(max_offset) and float(max_offset) > GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX:
        reasons.append(
            "Largest matched-peak offset {offset:.2f} px exceeds the acceptance "
            "limit of {limit:.2f} px.".format(
                offset=float(max_offset),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX),
            )
        )

    identifiability_summary = getattr(result, "identifiability_summary", None)
    underconstrained = isinstance(identifiability_summary, Mapping) and bool(
        identifiability_summary.get("underconstrained", False)
    )
    if underconstrained:
        reasons.append(
            "Fit is underconstrained according to the final identifiability diagnostics."
        )

    bound_proximity_summary = getattr(result, "bound_proximity_summary", None)
    if underconstrained and isinstance(bound_proximity_summary, Mapping):
        near_bound_entries = bound_proximity_summary.get("near_bound_parameters", [])
        if isinstance(near_bound_entries, Sequence) and near_bound_entries:
            joined = ", ".join(
                "{name}({side})".format(
                    name=str(entry.get("name", "")),
                    side=str(entry.get("side", "")),
                )
                for entry in near_bound_entries
                if isinstance(entry, Mapping) and entry.get("name")
            )
            if joined:
                reasons.append(
                    "Parameters finished within 1% of a finite bound span from a bound: "
                    + joined
                    + "."
                )

    return reasons


def build_geometry_fit_result_lines(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    rms: float,
) -> list[str]:
    """Format fitted parameter values plus the RMS residual line."""

    lines = []
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        lines.append(f"{name} = {value:.6f}")
    lines.append(f"RMS residual = {float(rms):.6f} px")
    return lines


def build_geometry_fit_fitted_params(
    base_params: Mapping[str, object] | None,
    *,
    zb: object,
    zs: object,
    theta_initial: object,
    theta_offset: object,
    chi: object,
    cor_angle: object,
    psi_z: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Merge the current fitted UI values into one simulation parameter dict."""

    fitted = dict(base_params or {})
    fitted.update(
        current_geometry_fit_ui_params(
            zb=float(zb),
            zs=float(zs),
            theta_initial=float(theta_initial),
            theta_offset=float(theta_offset),
            psi_z=float(psi_z),
            chi=float(chi),
            cor_angle=float(cor_angle),
            gamma=float(gamma),
            Gamma=float(Gamma),
            corto_detector=float(corto_detector),
            a=float(a),
            c=float(c),
            center_x=float(center_x),
            center_y=float(center_y),
        )
    )
    return fitted


def build_geometry_fit_fitted_params_from_ui(
    base_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build fitted simulation params from one current UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_fitted_params(
        base_params,
        zb=params.get("zb", np.nan),
        zs=params.get("zs", np.nan),
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        chi=params.get("chi", np.nan),
        cor_angle=params.get("cor_angle", np.nan),
        psi_z=params.get("psi_z", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_profile_cache(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    *,
    theta_initial: object,
    theta_offset: object,
    cor_angle: object,
    chi: object,
    zs: object,
    zb: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Build the geometry-fit profile-cache payload after one successful fit."""

    profile_cache = dict(base_cache or {})
    profile_cache.update(dict(mosaic_params or {}))
    profile_cache.update(
        {
            "theta_initial": theta_initial,
            "theta_offset": theta_offset,
            "cor_angle": cor_angle,
            "chi": chi,
            "zs": zs,
            "zb": zb,
            "gamma": gamma,
            "Gamma": Gamma,
            "corto_detector": corto_detector,
            "a": a,
            "c": c,
            "center_x": center_x,
            "center_y": center_y,
        }
    )
    return profile_cache


def build_geometry_fit_profile_cache_from_ui(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the post-fit profile-cache payload from one UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_profile_cache(
        base_cache,
        mosaic_params,
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        cor_angle=params.get("cor_angle", np.nan),
        chi=params.get("chi", np.nan),
        zs=params.get("zs", np.nan),
        zb=params.get("zb", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_pixel_offsets(
    point_match_diagnostics: Sequence[object] | None = None,
    agg_millers: Sequence[Sequence[object]] | None = None,
    agg_sim_coords: Sequence[Sequence[object]] | None = None,
    agg_meas_coords: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, object]] | list[tuple[tuple[int, int, int], float, float, float]]:
    """Build one per-point native-frame offset record from final diagnostics."""

    if (
        agg_meas_coords is None
        and point_match_diagnostics is not None
        and agg_millers is not None
        and agg_sim_coords is not None
    ):
        agg_meas_coords = agg_sim_coords
        agg_sim_coords = agg_millers
        agg_millers = point_match_diagnostics  # type: ignore[assignment]
        point_match_diagnostics = None

    if agg_millers is not None and agg_sim_coords is not None and agg_meas_coords is not None:
        legacy_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
        for hkl_key, sim_center, meas_center in zip(
            agg_millers,
            agg_sim_coords,
            agg_meas_coords,
        ):
            try:
                hkl = tuple(int(v) for v in hkl_key[:3])
                dx = float(sim_center[0]) - float(meas_center[0])
                dy = float(sim_center[1]) - float(meas_center[1])
            except Exception:
                continue
            legacy_offsets.append((hkl, dx, dy, float(np.hypot(dx, dy))))
        return legacy_offsets

    pixel_offsets: list[dict[str, object]] = []
    for raw_entry in point_match_diagnostics or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        pixel_offsets.append(
            {
                "dataset_index": int(entry.get("dataset_index", 0)),
                "overlay_match_index": int(entry.get("overlay_match_index", -1)),
                "match_status": str(entry.get("match_status", "")),
                "hkl": _geometry_fit_normalize_hkl(entry.get("hkl")),
                "dx_px": float(entry.get("dx_px", np.nan)),
                "dy_px": float(entry.get("dy_px", np.nan)),
                "distance_px": float(entry.get("distance_px", np.nan)),
            }
        )
    return pixel_offsets


def filter_geometry_fit_overlay_point_match_diagnostics(
    point_match_diagnostics: object,
    *,
    joint_background_mode: bool,
    current_background_index: int,
) -> list[dict[str, object]] | object:
    """Keep only the current-background overlay diagnostics in joint mode."""

    if not (joint_background_mode and isinstance(point_match_diagnostics, list)):
        return point_match_diagnostics
    filtered: list[dict[str, object]] = []
    for entry in point_match_diagnostics:
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("dataset_index", -1)) != int(current_background_index):
            continue
        filtered.append(dict(entry))
    return filtered


def build_geometry_fit_point_match_summary_lines(
    point_match_summary: Mapping[str, object] | None,
) -> list[str]:
    """Format the optional point-match summary section."""

    if not isinstance(point_match_summary, Mapping):
        return []
    lines: list[str] = []
    try:
        fixed_source_count = int(point_match_summary.get("fixed_source_resolved_count", 0))
    except Exception:
        fixed_source_count = 0
    try:
        measured_count = int(point_match_summary.get("measured_count", 0))
    except Exception:
        measured_count = 0
    if measured_count > 0 and fixed_source_count == 0:
        lines.append(
            "WARNING: fit used only HKL-fallback correspondences; no fixed source-row anchors resolved."
        )
    lines.extend(f"{key}={value}" for key, value in sorted(point_match_summary.items()))
    return lines


def _geometry_fit_pixel_size_provenance(
    params: Mapping[str, object] | None,
) -> tuple[str, float, dict[str, float]]:
    """Resolve the pixel-size source together with the raw candidate values."""

    cfg = params if isinstance(params, Mapping) else {}

    def _value(key: str) -> float:
        try:
            return float(cfg.get(key, np.nan))
        except Exception:
            return float("nan")

    raw_values = {
        "pixel_size": _value("pixel_size"),
        "pixel_size_m": _value("pixel_size_m"),
        "debye_x": _value("debye_x"),
        "debye_y": _value("debye_y"),
        "corto_detector": _value("corto_detector"),
    }
    for key in ("pixel_size", "pixel_size_m", "debye_x"):
        value = float(raw_values.get(key, np.nan))
        if np.isfinite(value) and value > 0.0:
            return key, float(value), raw_values
    fallback = raw_values.get("corto_detector", np.nan) / 4096.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0e-6
    return "corto_detector/4096", float(fallback), raw_values


def build_geometry_fit_calibration_lines(
    params: Mapping[str, object] | None,
) -> list[str]:
    """Format pixel-size provenance for debug geometry-fit logs."""

    if not isinstance(params, Mapping):
        return []
    source, pixel_size_value, raw_values = _geometry_fit_pixel_size_provenance(params)
    lines = [
        "pixel_size_source={source} value={value}".format(
            source=source,
            value=_geometry_fit_debug_value_text(pixel_size_value),
        ),
        (
            "pixel_size={pixel_size} pixel_size_m={pixel_size_m} "
            "debye_x={debye_x} debye_y={debye_y}"
        ).format(
            pixel_size=_geometry_fit_debug_value_text(raw_values.get("pixel_size")),
            pixel_size_m=_geometry_fit_debug_value_text(raw_values.get("pixel_size_m")),
            debye_x=_geometry_fit_debug_value_text(raw_values.get("debye_x")),
            debye_y=_geometry_fit_debug_value_text(raw_values.get("debye_y")),
        ),
    ]
    debye_x = float(raw_values.get("debye_x", np.nan))
    if source != "debye_x" and np.isfinite(debye_x) and debye_x <= 0.0:
        lines.append(f"warning=debye_x <= 0; using {source} instead")
    return lines


def _geometry_fit_reason_counter_lines(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> list[str]:
    """Return one count summary for a diagnostics reason/status key."""

    counts: Counter[str] = Counter()
    for item in diagnostics:
        raw_value = item.get(key)
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        counts[text] += 1
    if not counts:
        return []
    summary = ", ".join(
        f"{name}={count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return [f"{key}: {summary}"]


def _geometry_fit_preferred_failure_reason(
    entry: Mapping[str, object],
) -> str:
    """Choose the most actionable failure-reason field for one point match."""

    for key in (
        "correspondence_resolution_reason",
        "detector_resolution_reason",
        "measured_resolution_reason",
        "resolution_reason",
    ):
        value = entry.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def _geometry_fit_overlay_match_index(
    entry: Mapping[str, object],
) -> int:
    """Return one stable overlay-match index for diagnostics output."""

    for key in ("overlay_match_index", "match_input_index"):
        try:
            return int(entry.get(key, -1))
        except Exception:
            continue
    return -1


def _geometry_fit_point_text(
    x_value: object,
    y_value: object,
    *,
    float_digits: int = 3,
) -> str:
    """Format one detector or fit-space point for debug logs."""

    try:
        x_numeric = float(x_value)
    except Exception:
        x_numeric = float("nan")
    try:
        y_numeric = float(y_value)
    except Exception:
        y_numeric = float("nan")
    if not np.isfinite(x_numeric) and not np.isfinite(y_numeric):
        return "<none>"
    return _geometry_fit_debug_value_text(
        [x_numeric, y_numeric],
        float_digits=float_digits,
    )


def build_geometry_fit_point_match_failure_reason_lines(
    point_match_diagnostics: Sequence[object] | None,
    *,
    max_unresolved_pairs_per_dataset: int = 5,
) -> list[str]:
    """Format one post-solve point-match diagnostics summary for debug logs."""

    diagnostics = [
        dict(entry)
        for entry in (point_match_diagnostics or ())
        if isinstance(entry, Mapping)
    ]
    if not diagnostics:
        return []

    lines: list[str] = []
    for key in (
        "match_status",
        "resolution_reason",
        "measured_resolution_reason",
        "detector_resolution_reason",
        "correspondence_resolution_reason",
        "fit_source_resolution_kind",
    ):
        lines.extend(_geometry_fit_reason_counter_lines(diagnostics, key=key))

    unresolved_by_dataset: dict[tuple[int, str], list[dict[str, object]]] = {}
    dataset_order: list[tuple[int, str]] = []
    for entry in diagnostics:
        if str(entry.get("match_status", "")).strip().lower() == "matched":
            continue
        try:
            dataset_index = int(entry.get("dataset_index", 0))
        except Exception:
            dataset_index = 0
        label = str(
            entry.get(
                "dataset_label",
                f"bg[{dataset_index}]",
            )
        )
        dataset_key = (int(dataset_index), label)
        if dataset_key not in unresolved_by_dataset:
            dataset_order.append(dataset_key)
        unresolved_by_dataset.setdefault(dataset_key, []).append(entry)

    max_pairs = max(1, int(max_unresolved_pairs_per_dataset))
    for dataset_index, label in dataset_order:
        unresolved = unresolved_by_dataset.get((dataset_index, label), [])
        if not unresolved:
            continue
        lines.append(
            f"dataset[{dataset_index}] {label}: unresolved_pairs={len(unresolved)}"
        )
        for entry in unresolved[:max_pairs]:
            pair_index = _geometry_fit_overlay_match_index(entry)
            lines.append(
                (
                    "dataset[{dataset_index}] {label} overlay_index={pair} "
                    "hkl={hkl} q_group={q_group} measured_detector={measured_detector} "
                    "measured_angles={measured_angles}"
                ).format(
                    dataset_index=dataset_index,
                    label=label,
                    pair=pair_index,
                    hkl=_geometry_fit_debug_value_text(
                        entry.get("hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        entry.get("q_group_key"),
                        float_digits=3,
                    ),
                    measured_detector=_geometry_fit_point_text(
                        entry.get("measured_x"),
                        entry.get("measured_y"),
                        float_digits=3,
                    ),
                    measured_angles=_geometry_fit_point_text(
                        entry.get("measured_two_theta_deg"),
                        entry.get("measured_phi_deg"),
                        float_digits=3,
                    ),
                )
            )
            lines.append(
                (
                    "dataset[{dataset_index}] {label} overlay_index={pair} "
                    "simulated={simulated} distance_px={distance} "
                    "delta_two_theta_deg={delta_tth} delta_phi_deg={delta_phi} "
                    "resolution_reason={reason}"
                ).format(
                    dataset_index=dataset_index,
                    label=label,
                    pair=pair_index,
                    simulated=_geometry_fit_point_text(
                        entry.get("simulated_x"),
                        entry.get("simulated_y"),
                        float_digits=3,
                    ),
                    distance=_geometry_fit_debug_value_text(
                        entry.get("distance_px"),
                        float_digits=3,
                    ),
                    delta_tth=_geometry_fit_debug_value_text(
                        entry.get("delta_two_theta_deg"),
                        float_digits=3,
                    ),
                    delta_phi=_geometry_fit_debug_value_text(
                        entry.get("delta_phi_deg"),
                        float_digits=3,
                    ),
                    reason=_geometry_fit_preferred_failure_reason(entry),
                )
            )
        extra_count = len(unresolved) - max_pairs
        if extra_count > 0:
            lines.append(
                "dataset[{dataset_index}] {label}: ... {extra_count} more unresolved "
                "pair(s) not shown".format(
                    dataset_index=dataset_index,
                    label=label,
                    extra_count=int(extra_count),
                )
            )

    return lines


def build_geometry_fit_overlay_diagnostic_lines(
    frame_diag: Mapping[str, object] | None,
    *,
    overlay_record_count: int,
    result: object | None = None,
) -> list[str]:
    """Format overlay frame diagnostics for the geometry-fit log."""

    diag = frame_diag if isinstance(frame_diag, Mapping) else {}
    lines = [
        "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
        f"overlay_records={int(overlay_record_count)}",
        f"paired_records={int(diag.get('paired_records', 0))}",
        f"sim_display_med_px={float(diag.get('sim_display_med_px', np.nan)):.3f}",
        f"bg_display_med_px={float(diag.get('bg_display_med_px', np.nan)):.3f}",
        f"sim_display_p90_px={float(diag.get('sim_display_p90_px', np.nan)):.3f}",
        f"bg_display_p90_px={float(diag.get('bg_display_p90_px', np.nan)):.3f}",
    ]
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.insert(1, f"solver_discrete_mode={mode_label}")
    return lines


def count_geometry_fit_matched_overlay_records(
    overlay_records: Sequence[Mapping[str, object]] | None,
) -> int:
    """Return the number of overlay records that represent matched fitted points."""

    matched_count = 0
    for entry in overlay_records or ():
        if not isinstance(entry, Mapping):
            continue
        status = str(entry.get("match_status", "")).strip().lower()
        if status:
            if status == "matched":
                matched_count += 1
            continue
        matched_count += 1
    return int(matched_count)


def build_geometry_fit_pixel_offset_lines(
    pixel_offsets: Sequence[Sequence[object]],
) -> list[str]:
    """Format one pixel-offset section for the geometry-fit log/status."""

    lines = []
    for entry in pixel_offsets:
        if isinstance(entry, Mapping):
            hkl = _geometry_fit_normalize_hkl(entry.get("hkl"))
            dataset_index = int(entry.get("dataset_index", 0))
            overlay_match_index = int(entry.get("overlay_match_index", -1))
            match_status = str(entry.get("match_status", ""))
            dx = float(entry.get("dx_px", np.nan))
            dy = float(entry.get("dy_px", np.nan))
            dist = float(entry.get("distance_px", np.nan))
            prefix = f"dataset={dataset_index} idx={overlay_match_index} HKL={hkl}"
            if match_status.lower() != "matched" or not np.isfinite(dist):
                lines.append(f"{prefix}: status={match_status or 'unknown'}")
                continue
            lines.append(f"{prefix}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px")
            continue
        if not isinstance(entry, Mapping):
            try:
                hkl = tuple(int(v) for v in entry[0][:3])
                dx = float(entry[1])
                dy = float(entry[2])
                dist = float(entry[3])
            except Exception:
                continue
            lines.append(f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px")
    return lines or ["No matched peaks"]


def build_geometry_fit_summary_lines(
    *,
    current_dataset: Mapping[str, object],
    overlay_record_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    save_path: object,
    result: object | None = None,
) -> list[str]:
    """Format the fit-summary section written to the geometry-fit log."""

    lines = [
        f"manual_groups={int(current_dataset.get('group_count', 0) or 0)}",
        f"manual_points={int(current_dataset.get('pair_count', 0) or 0)}",
        f"overlay_records={int(overlay_record_count)}",
        "orientation={orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        ),
    ]
    if "resolved_source_pair_count" in current_dataset:
        lines.append(
            "resolved_source_pairs={count}".format(
                count=int(current_dataset.get("resolved_source_pair_count", 0) or 0)
            )
        )
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.append(f"solver_discrete_mode={mode_label}")
    final_metric_name = str(getattr(result, "final_metric_name", "") or "").strip()
    if final_metric_name:
        lines.append(f"final_metric={final_metric_name}")
    point_match_summary = getattr(result, "point_match_summary", None)
    if isinstance(point_match_summary, Mapping):
        if "matched_pair_count" in point_match_summary:
            lines.append(
                "matched_pairs={count}".format(
                    count=int(point_match_summary.get("matched_pair_count", 0) or 0)
                )
            )
        if "fixed_source_resolved_count" in point_match_summary:
            lines.append(
                "fixed_source_resolved={count}".format(
                    count=int(
                        point_match_summary.get("fixed_source_resolved_count", 0) or 0
                    )
                )
            )
        peak_rms = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_rms_px", np.nan)
        )
        if np.isfinite(peak_rms):
            lines.append(f"matched_peak_rms_px={peak_rms:.6f}")
        peak_max = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_max_px", np.nan)
        )
        if np.isfinite(peak_max):
            lines.append(f"matched_peak_max_px={peak_max:.6f}")
    identifiability_summary = getattr(result, "identifiability_summary", None)
    if isinstance(identifiability_summary, Mapping):
        lines.append(
            "underconstrained={flag}".format(
                flag=bool(identifiability_summary.get("underconstrained", False))
            )
        )
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()
    if boundary_warning:
        lines.append(f"boundary_warning={boundary_warning}")
    next_stage_recommendation = getattr(result, "next_stage_recommendation", None)
    if isinstance(next_stage_recommendation, Mapping):
        next_params = [
            str(name)
            for name in next_stage_recommendation.get("params", [])
            if str(name).strip()
        ]
        if next_params:
            lines.append("next_stage_params={params}".format(params=",".join(next_params)))
    lines.extend(
        build_geometry_fit_result_lines(
            var_names,
            values,
            rms=rms,
        )[:-1]
    )
    lines.append(f"RMS residual = {float(rms):.6f} px")
    lines.append(f"Matched peaks saved to: {save_path}")
    return lines


def build_geometry_fit_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    pixel_offsets: Sequence[Sequence[object]],
    export_record_count: int,
    save_path: object,
    log_path: object,
    frame_warning: str | None,
    result: object | None = None,
) -> str:
    """Build the final geometry-fit status text shown in the GUI."""

    base_summary_lines = ["Manual geometry fit complete:"]
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        base_summary_lines.append(f"{name} = {value:.4f}")
    base_summary_lines.append(f"RMS residual = {float(rms):.2f} px")
    base_summary_lines.append(
        "Orientation = {orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        )
    )
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        base_summary_lines.append(f"Solver discrete mode = {mode_label}")
    base_summary = "\n".join(base_summary_lines)
    overlay_hint = (
        "Overlay: blue squares=selected simulated points, amber triangles=saved "
        "background points, green circles=fitted simulated peaks, dashed "
        "arrows=initial->fitted sim shifts."
    )
    dist_report_lines = build_geometry_fit_pixel_offset_lines(pixel_offsets)
    dist_report = "\n".join(dist_report_lines)
    return (
        f"{base_summary}\n"
        "Manual pairs: {points} points across {groups} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
        + "\n"
        + overlay_hint
        + (f"\n{frame_warning}" if frame_warning else "")
        + f"\nSaved {int(export_record_count)} peak records → {save_path}"
        + f"\nPixel offsets:\n{dist_report}"
        + f"\nFit log → {log_path}"
    )


def build_geometry_fit_rejected_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    rms: float,
    rejection_reasons: Sequence[object],
) -> str:
    """Build the GUI status text for one rejected manual geometry fit."""

    lines = ["Manual geometry fit rejected:"]
    for reason in rejection_reasons:
        text = str(reason).strip()
        if text:
            lines.append(text)
    if np.isfinite(rms):
        lines.append(f"RMS residual = {float(rms):.2f} px")
    lines.append(
        "Manual pairs: {points} points across {groups} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
    )
    lines.append("Add more manual points or remove outliers before rerunning the fit.")
    return "\n".join(lines)


def postprocess_geometry_fit_result(
    *,
    fitted_params: Mapping[str, object],
    result: object,
    current_dataset: Mapping[str, object],
    joint_background_mode: bool,
    current_background_index: int,
    dataset_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    miller: object,
    intensities: object,
    image_size: int,
    max_display_markers: int,
    downloads_dir: Path | str,
    stamp: str,
    log_path: Path | str,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
) -> GeometryFitPostprocessResult:
    """Build the post-fit analysis artifacts used by the live GUI."""

    point_match_diagnostics = getattr(result, "point_match_diagnostics", None)
    if not isinstance(point_match_diagnostics, list):
        point_match_diagnostics = []
    pixel_offsets = build_geometry_fit_pixel_offsets(point_match_diagnostics)

    overlay_point_match_diagnostics = filter_geometry_fit_overlay_point_match_diagnostics(
        point_match_diagnostics,
        joint_background_mode=joint_background_mode,
        current_background_index=int(current_background_index),
    )
    effective_orientation_choice = _geometry_fit_effective_orientation_choice(
        native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
        orientation_choice=(
            current_dataset["orientation_choice"]
            if isinstance(current_dataset, Mapping)
            else None
        ),
        result=result,
    )
    overlay_records = build_overlay_records(
        current_dataset["initial_pairs_display"],
        overlay_point_match_diagnostics,
        native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
        orientation_choice=effective_orientation_choice,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
    )
    matched_overlay_record_count = count_geometry_fit_matched_overlay_records(
        overlay_records
    )
    frame_diag, frame_warning = compute_frame_diagnostics(overlay_records)
    overlay_diagnostic_lines = build_geometry_fit_overlay_diagnostic_lines(
        frame_diag,
        overlay_record_count=int(matched_overlay_record_count),
        result=result,
    )

    export_records = build_geometry_fit_export_records(point_match_diagnostics)
    save_path = Path(downloads_dir) / f"matched_peaks_{stamp}.npy"
    fit_summary_lines = build_geometry_fit_summary_lines(
        current_dataset=current_dataset,
        overlay_record_count=int(matched_overlay_record_count),
        var_names=var_names,
        values=values,
        rms=rms,
        save_path=save_path,
        result=result,
    )
    progress_text = build_geometry_fit_progress_text(
        current_dataset=current_dataset,
        dataset_count=int(dataset_count),
        joint_background_mode=joint_background_mode,
        var_names=var_names,
        values=values,
        rms=rms,
        pixel_offsets=pixel_offsets,
        export_record_count=len(export_records),
        save_path=save_path,
        log_path=log_path,
        frame_warning=frame_warning,
        result=result,
    )
    overlay_state = {
        "overlay_records": copy_geometry_fit_state_value(overlay_records),
        "initial_pairs_display": copy_geometry_fit_state_value(
            current_dataset["initial_pairs_display"]
        ),
        "max_display_markers": int(max_display_markers),
    }

    return GeometryFitPostprocessResult(
        fitted_params=fitted_params,
        point_match_summary_lines=build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        ),
        pixel_offsets=pixel_offsets,
        overlay_records=overlay_records,
        overlay_state=overlay_state,
        overlay_diagnostic_lines=overlay_diagnostic_lines,
        frame_warning=frame_warning,
        export_records=export_records,
        save_path=save_path,
        fit_summary_lines=fit_summary_lines,
        progress_text=progress_text,
    )


def apply_runtime_geometry_fit_result(
    *,
    result: object,
    var_names: Sequence[object],
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    preserve_live_theta: bool,
    max_display_markers: int,
    bindings: GeometryFitRuntimeResultBindings,
) -> GeometryFitRuntimeApplyResult:
    """Apply one successful geometry-fit result through runtime callbacks."""

    debug_logging = geometry_fit_debug_logging_enabled(
        bindings.geometry_runtime_cfg
    )
    bindings.log_section(
        "Optimizer diagnostics:",
        build_geometry_fit_optimizer_diagnostics_lines(result),
    )
    debug_lines = build_geometry_fit_debug_lines(result)
    if debug_lines:
        bindings.log_section(
            "Fit mechanics:",
            debug_lines,
        )
    stage_summary_lines = build_geometry_fit_stage_summary_lines(result)
    if stage_summary_lines:
        bindings.log_section(
            "Solver stages:",
            stage_summary_lines,
        )
    identifiability_lines = build_geometry_fit_identifiability_lines(result)
    if identifiability_lines:
        bindings.log_section(
            "Identifiability diagnostics:",
            identifiability_lines,
        )

    result_vector = getattr(result, "x", None)
    result_values = [] if result_vector is None else list(result_vector)
    preview_fitted_params = (
        bindings.preview_fitted_params(var_names, result_values)
        if callable(bindings.preview_fitted_params)
        else None
    )
    rms = geometry_fit_result_rms(result)
    rejection_reasons = build_geometry_fit_rejection_reason_lines(
        result,
        rms=rms,
    )
    if rejection_reasons:
        bindings.log_section(
            "Optimization result:",
            build_geometry_fit_result_lines(
                var_names,
                result_values,
                rms=rms,
            ),
        )
        point_match_summary_lines = build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        )
        if point_match_summary_lines:
            bindings.log_section(
                "Point-match summary:",
                point_match_summary_lines,
            )
        if debug_logging:
            diagnostic_lines = build_geometry_fit_point_match_failure_reason_lines(
                getattr(result, "point_match_diagnostics", None)
            )
            if diagnostic_lines:
                bindings.log_section(
                    "Point-match diagnostics:",
                    diagnostic_lines,
                )
            calibration_lines = build_geometry_fit_calibration_lines(
                preview_fitted_params
            )
            if calibration_lines:
                bindings.log_section(
                    "Fit-space calibration:",
                    calibration_lines,
                )
        bindings.log_section("Fit rejected:", rejection_reasons)
        bindings.set_progress_text(
            build_geometry_fit_rejected_progress_text(
                current_dataset=current_dataset,
                dataset_count=dataset_count,
                joint_background_mode=joint_background_mode,
                rms=rms,
                rejection_reasons=rejection_reasons,
            )
        )
        bindings.cmd_line(
            "rejected: "
            f"datasets={int(dataset_count)} "
            f"groups={int(current_dataset.get('group_count', 0) or 0)} "
            f"points={int(current_dataset.get('pair_count', 0) or 0)} "
            f"rms={float(rms):.4f}px "
            f"reason={rejection_reasons[0]}"
        )
        return GeometryFitRuntimeApplyResult(
            accepted=False,
            rejection_reason=" ".join(str(reason) for reason in rejection_reasons),
            rms=float(rms),
            fitted_params=None,
            postprocess=None,
        )

    undo_state = bindings.capture_undo_state()
    bindings.apply_result_values(var_names, result_values)

    if joint_background_mode and not preserve_live_theta:
        sync_theta = bindings.sync_joint_background_theta
        if sync_theta is not None:
            sync_theta()

    bindings.refresh_status()
    bindings.update_manual_pick_button_label()
    bindings.replace_profile_cache(bindings.build_profile_cache())
    bindings.push_undo_state(undo_state)
    bindings.request_preview_skip_once()
    bindings.mark_last_simulation_dirty()
    bindings.schedule_update()

    bindings.log_section(
        "Optimization result:",
        build_geometry_fit_result_lines(
            var_names,
            result_values,
            rms=rms,
        ),
    )

    fitted_params = bindings.build_fitted_params()
    postprocess = bindings.postprocess_result(fitted_params, rms)

    if postprocess.point_match_summary_lines:
        bindings.log_section(
            "Point-match summary:",
            postprocess.point_match_summary_lines,
        )
    if debug_logging:
        calibration_lines = build_geometry_fit_calibration_lines(
            preview_fitted_params or fitted_params
        )
        if calibration_lines:
            bindings.log_section(
                "Fit-space calibration:",
                calibration_lines,
            )
        diagnostic_lines = build_geometry_fit_point_match_failure_reason_lines(
            getattr(result, "point_match_diagnostics", None)
        )
        if diagnostic_lines:
            bindings.log_section(
                "Point-match diagnostics:",
                diagnostic_lines,
            )

    bindings.log_section(
        "Overlay frame diagnostics:",
        postprocess.overlay_diagnostic_lines,
    )

    if postprocess.overlay_records:
        bindings.draw_overlay_records(
            postprocess.overlay_records,
            int(max_display_markers),
        )
    else:
        bindings.draw_initial_pairs_overlay(
            current_dataset["initial_pairs_display"],
            int(max_display_markers),
        )

    bindings.set_last_overlay_state(postprocess.overlay_state)
    bindings.save_export_records(postprocess.save_path, postprocess.export_records)
    bindings.log_section(
        "Pixel offsets (native frame):",
        build_geometry_fit_pixel_offset_lines(postprocess.pixel_offsets),
    )
    bindings.log_section(
        "Fit summary:",
        postprocess.fit_summary_lines,
    )
    bindings.set_progress_text(postprocess.progress_text)
    bindings.cmd_line(
        "done: "
        f"datasets={int(dataset_count)} "
        f"groups={int(current_dataset.get('group_count', 0) or 0)} "
        f"points={int(current_dataset.get('pair_count', 0) or 0)} "
        f"rms={float(rms):.4f}px"
    )

    return GeometryFitRuntimeApplyResult(
        accepted=True,
        rejection_reason=None,
        rms=float(rms),
        fitted_params=fitted_params,
        postprocess=postprocess,
    )


def build_runtime_geometry_fit_result_bindings(
    *,
    fit_params: Mapping[str, object] | None,
    base_profile_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    geometry_runtime_cfg: Mapping[str, object] | None = None,
    geometry_theta_offset_var=None,
    log_section: Callable[[str, Sequence[str]], None],
    capture_undo_state: Callable[[], dict[str, object]],
    sync_joint_background_theta: Callable[[], None] | None,
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    replace_profile_cache: Callable[[dict[str, object]], None],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    mark_last_simulation_dirty: Callable[[], None],
    schedule_update: Callable[[], None],
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime success-path callback bundle for one geometry fit."""

    def _snapshot_ui_params() -> dict[str, object]:
        try:
            params = current_ui_params()
        except Exception:
            params = {}
        return dict(params) if isinstance(params, Mapping) else {}

    def _preview_fitted_params(
        names: Sequence[object],
        values: Sequence[object],
    ) -> dict[str, object]:
        ui_params = _snapshot_ui_params()
        for name, value in zip(names, values):
            key = str(name)
            try:
                ui_params[key] = float(value)
            except Exception:
                ui_params[key] = value
        return build_geometry_fit_fitted_params_from_ui(
            fit_params,
            ui_params,
        )

    return GeometryFitRuntimeResultBindings(
        log_section=log_section,
        capture_undo_state=capture_undo_state,
        apply_result_values=(
            lambda names, values: apply_geometry_fit_result_values(
                names,
                values,
                var_map=var_map,
                geometry_theta_offset_var=geometry_theta_offset_var,
            )
        ),
        sync_joint_background_theta=sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        build_profile_cache=(
            lambda: build_geometry_fit_profile_cache_from_ui(
                base_profile_cache,
                mosaic_params,
                _snapshot_ui_params(),
            )
        ),
        replace_profile_cache=replace_profile_cache,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=mark_last_simulation_dirty,
        schedule_update=schedule_update,
        build_fitted_params=(
            lambda: build_geometry_fit_fitted_params_from_ui(
                fit_params,
                _snapshot_ui_params(),
            )
        ),
        postprocess_result=postprocess_result,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=save_export_records,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        geometry_runtime_cfg=(
            dict(geometry_runtime_cfg)
            if isinstance(geometry_runtime_cfg, Mapping)
            else None
        ),
        preview_fitted_params=_preview_fitted_params,
    )


def build_runtime_geometry_fit_execution_result_bindings(
    *,
    result: object,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    ui_bindings: GeometryFitRuntimeUiBindings,
    postprocess_config: GeometryFitRuntimePostprocessConfig,
    log_section: Callable[[str, Sequence[str]], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime apply-bindings for one executed geometry-fit result."""

    solver_inputs = postprocess_config.solver_inputs
    return build_runtime_geometry_fit_result_bindings(
        fit_params=ui_bindings.fit_params,
        base_profile_cache=ui_bindings.base_profile_cache,
        mosaic_params=ui_bindings.mosaic_params,
        current_ui_params=ui_bindings.current_ui_params,
        var_map=ui_bindings.var_map,
        geometry_runtime_cfg=prepared_run.geometry_runtime_cfg,
        geometry_theta_offset_var=ui_bindings.geometry_theta_offset_var,
        log_section=log_section,
        capture_undo_state=ui_bindings.capture_undo_state,
        sync_joint_background_theta=ui_bindings.sync_joint_background_theta,
        refresh_status=ui_bindings.refresh_status,
        update_manual_pick_button_label=ui_bindings.update_manual_pick_button_label,
        replace_profile_cache=ui_bindings.replace_profile_cache,
        push_undo_state=ui_bindings.push_undo_state,
        request_preview_skip_once=ui_bindings.request_preview_skip_once,
        mark_last_simulation_dirty=ui_bindings.mark_last_simulation_dirty,
        schedule_update=ui_bindings.schedule_update,
        postprocess_result=(
            lambda fitted_params, rms: postprocess_geometry_fit_result(
                fitted_params=fitted_params,
                result=result,
                current_dataset=prepared_run.current_dataset,
                joint_background_mode=prepared_run.joint_background_mode,
                current_background_index=int(postprocess_config.current_background_index),
                dataset_count=len(prepared_run.dataset_infos),
                var_names=var_names,
                values=getattr(result, "x", []),
                rms=rms,
                miller=solver_inputs.miller,
                intensities=solver_inputs.intensities,
                image_size=int(solver_inputs.image_size),
                max_display_markers=int(prepared_run.max_display_markers),
                downloads_dir=postprocess_config.downloads_dir,
                stamp=postprocess_config.stamp,
                log_path=postprocess_config.log_path,
                sim_display_rotate_k=int(postprocess_config.sim_display_rotate_k),
                background_display_rotate_k=int(
                    postprocess_config.background_display_rotate_k
                ),
                simulate_and_compare_hkl=postprocess_config.simulate_and_compare_hkl,
                aggregate_match_centers=postprocess_config.aggregate_match_centers,
                build_overlay_records=postprocess_config.build_overlay_records,
                compute_frame_diagnostics=postprocess_config.compute_frame_diagnostics,
            )
        ),
        draw_overlay_records=ui_bindings.draw_overlay_records,
        draw_initial_pairs_overlay=ui_bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=ui_bindings.set_last_overlay_state,
        save_export_records=ui_bindings.save_export_records,
        set_progress_text=ui_bindings.set_progress_text,
        cmd_line=ui_bindings.cmd_line,
    )


def build_runtime_geometry_fit_execution_setup(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeExecutionSetup:
    """Build the runtime execution setup for one prepared geometry-fit run."""

    def _sync_joint_background_theta() -> None:
        background_index = int(
            getattr(background_runtime_state, "current_background_index", 0)
        )
        if 0 <= background_index < len(prepared_run.background_theta_values):
            theta_initial_var.set(
                float(prepared_run.background_theta_values[background_index])
            )
            return
        theta_initial_var.set(
            background_theta_for_index(
                background_index,
                strict_count=False,
            )
        )

    ui_bindings = GeometryFitRuntimeUiBindings(
        fit_params=prepared_run.fit_params,
        base_profile_cache=getattr(simulation_runtime_state, "profile_cache", {}),
        mosaic_params=mosaic_params,
        current_ui_params=current_ui_params,
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
        capture_undo_state=capture_undo_state,
        sync_joint_background_theta=_sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        replace_profile_cache=(
            lambda profile_cache: setattr(
                simulation_runtime_state,
                "profile_cache",
                dict(profile_cache),
            )
        ),
        replace_dataset_cache=replace_dataset_cache,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=(
            lambda: setattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        ),
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=(
            lambda save_path, export_records: np.save(
                save_path,
                np.array(export_records, dtype=object),
                allow_pickle=True,
            )
        ),
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        live_update_callback=live_update_callback,
    )

    postprocess_config = GeometryFitRuntimePostprocessConfig(
        current_background_index=int(
            getattr(background_runtime_state, "current_background_index", 0)
        ),
        downloads_dir=downloads_dir,
        stamp=stamp,
        log_path=build_geometry_fit_log_path(
            stamp=stamp,
            log_dir=log_dir,
            downloads_dir=downloads_dir,
        ),
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
        log_dir=log_dir,
    )

    return GeometryFitRuntimeExecutionSetup(
        ui_bindings=ui_bindings,
        postprocess_config=postprocess_config,
    )


def build_runtime_geometry_fit_execution_setup_from_bindings(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    bindings: GeometryFitRuntimeActionExecutionBindings,
) -> GeometryFitRuntimeExecutionSetup:
    """Build one runtime execution setup from a bound action bundle."""

    return build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params=mosaic_params,
        stamp=stamp,
        downloads_dir=bindings.downloads_dir,
        log_dir=bindings.log_dir,
        simulation_runtime_state=bindings.simulation_runtime_state,
        background_runtime_state=bindings.background_runtime_state,
        theta_initial_var=bindings.theta_initial_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        current_ui_params=bindings.current_ui_params,
        var_map=bindings.var_map,
        background_theta_for_index=bindings.background_theta_for_index,
        refresh_status=bindings.refresh_status,
        update_manual_pick_button_label=bindings.update_manual_pick_button_label,
        capture_undo_state=bindings.capture_undo_state,
        push_undo_state=bindings.push_undo_state,
        replace_dataset_cache=bindings.replace_dataset_cache,
        request_preview_skip_once=bindings.request_preview_skip_once,
        schedule_update=bindings.schedule_update,
        draw_overlay_records=bindings.draw_overlay_records,
        draw_initial_pairs_overlay=bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=bindings.set_last_overlay_state,
        set_progress_text=bindings.set_progress_text,
        cmd_line=bindings.cmd_line,
        solver_inputs=bindings.solver_inputs,
        sim_display_rotate_k=bindings.sim_display_rotate_k,
        background_display_rotate_k=bindings.background_display_rotate_k,
        simulate_and_compare_hkl=bindings.simulate_and_compare_hkl,
        aggregate_match_centers=bindings.aggregate_match_centers,
        build_overlay_records=bindings.build_overlay_records,
        compute_frame_diagnostics=bindings.compute_frame_diagnostics,
        live_update_callback=bindings.live_update_callback,
    )


def execute_runtime_geometry_fit_solver_phase(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solve_fit: Callable[..., object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    stamp: str,
    log_path: Path | str,
    start_progress_text: str = "Running geometry fit from saved manual Qr/Qz pairs...",
    event_callback: GeometryFitStageCallback | None = None,
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeExecutionResult:
    """Run the worker-safe logging and solver phase before any UI-state apply."""

    resolved_log_path = Path(log_path)
    logging_disabled = geometry_fit_all_logging_disabled()
    _, _log_line, _log_section = _build_geometry_fit_log_writers(resolved_log_path)

    def _emit_execution_event(kind: str, **payload: object) -> None:
        _emit_geometry_fit_stage_event(event_callback, kind, **payload)

    def _emit_cmd_line(text: str) -> None:
        text_value = str(text).strip()
        if text_value:
            _emit_execution_event("cmd_line", text=text_value)

    def _emit_progress_text(text: str) -> None:
        text_value = str(text)
        if text_value:
            _emit_execution_event("progress_text", text=text_value)

    def _solver_status_callback(text: str) -> None:
        status_text = str(text).strip()
        if not status_text:
            return
        _log_line(status_text)
        _emit_cmd_line(status_text)
        _emit_progress_text(status_text)

    try:
        write_geometry_fit_run_start_log(
            stamp=str(stamp),
            prepared_run=prepared_run,
            cmd_line=_emit_cmd_line,
            log_line=_log_line,
            log_section=_log_section,
        )
        if not logging_disabled:
            _emit_cmd_line(f"log: {resolved_log_path}")

        _emit_progress_text(str(start_progress_text))

        solver_request = build_geometry_fit_solver_request(
            prepared_run=prepared_run,
            var_names=var_names,
            solver_inputs=solver_inputs,
        )
        if solver_request.runtime_safety_note:
            _log_line(f"Runtime safety: {solver_request.runtime_safety_note}")
            _log_line()
        result = solve_geometry_fit_request(
            solver_request,
            solve_fit=solve_fit,
            status_callback=_solver_status_callback,
            live_update_callback=live_update_callback,
        )
        _emit_execution_event(
            "solver_finished",
            status="success",
            log_path=str(resolved_log_path),
        )
        return GeometryFitRuntimeExecutionResult(
            log_path=resolved_log_path,
            solver_request=solver_request,
            solver_result=result,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        _emit_cmd_line(f"failed: {exc}")
        _log_line(error_text)
        _emit_progress_text(error_text)
        _emit_execution_event(
            "solver_finished",
            status="error",
            error_text=error_text,
            log_path=str(resolved_log_path),
        )
        return GeometryFitRuntimeExecutionResult(
            log_path=resolved_log_path,
            error_text=error_text,
        )


def finalize_runtime_geometry_fit_execution(
    *,
    execution_result: GeometryFitRuntimeExecutionResult,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    preserve_live_theta: bool,
    setup: GeometryFitRuntimeExecutionSetup,
) -> GeometryFitRuntimeExecutionResult:
    """Apply one completed geometry-fit solver result on the runtime/UI thread."""

    if execution_result.error_text or execution_result.solver_result is None:
        return execution_result

    ui_bindings = setup.ui_bindings
    postprocess_config = setup.postprocess_config
    resolved_log_path = Path(execution_result.log_path)
    _, _, _log_section = _build_geometry_fit_log_writers(resolved_log_path)
    result = execution_result.solver_result
    apply_result = apply_runtime_geometry_fit_result(
        result=result,
        var_names=var_names,
        current_dataset=prepared_run.current_dataset,
        dataset_count=len(prepared_run.dataset_infos),
        joint_background_mode=prepared_run.joint_background_mode,
        preserve_live_theta=preserve_live_theta,
        max_display_markers=prepared_run.max_display_markers,
        bindings=build_runtime_geometry_fit_execution_result_bindings(
            result=result,
            prepared_run=prepared_run,
            var_names=var_names,
            ui_bindings=ui_bindings,
            postprocess_config=postprocess_config,
            log_section=_log_section,
        ),
    )
    replace_dataset_cache = ui_bindings.replace_dataset_cache
    if apply_result.accepted:
        if callable(replace_dataset_cache):
            replace_dataset_cache(
                build_geometry_fit_dataset_cache_payload(
                    prepared_run,
                    current_background_index=int(postprocess_config.current_background_index),
                )
            )
        play_completion_chime(
            prepared_run.geometry_runtime_cfg.get("completion_chime")
            if isinstance(prepared_run.geometry_runtime_cfg, Mapping)
            else None
        )
    return GeometryFitRuntimeExecutionResult(
        log_path=resolved_log_path,
        solver_request=execution_result.solver_request,
        solver_result=result,
        apply_result=apply_result,
    )


def run_runtime_geometry_fit_action(
    *,
    bindings: GeometryFitRuntimeActionBindings,
    prepare_run: Callable[..., GeometryFitPreparationResult] | None = None,
    build_execution_setup: Callable[..., GeometryFitRuntimeExecutionSetup] | None = None,
    execute_run: Callable[..., GeometryFitRuntimeExecutionResult] | None = None,
) -> GeometryFitRuntimeActionResult:
    """Run the full top-level geometry-fit action from live runtime bindings."""

    if prepare_run is None:
        prepare_run = prepare_runtime_geometry_fit_run
    if build_execution_setup is None:
        build_execution_setup = build_runtime_geometry_fit_execution_setup_from_bindings
    if execute_run is None:
        execute_run = execute_runtime_geometry_fit

    params = bindings.value_callbacks.current_params()
    mosaic_params = dict(params.get("mosaic_params", {}))
    var_names = list(bindings.value_callbacks.current_var_names())
    preserve_live_theta = (
        "theta_initial" not in var_names and "theta_offset" not in var_names
    )
    prepare_bindings = bindings.prepare_bindings_factory(var_names)
    preflight_runtime_cfg = (
        prepare_bindings.fit_config
        if isinstance(getattr(prepare_bindings, "fit_config", None), Mapping)
        else None
    )

    def _persist_preflight_failure_log(
        error_text: str,
        failure_log_sections: Sequence[tuple[str, Sequence[str]]] | None,
    ) -> Path | None:
        if geometry_fit_all_logging_disabled():
            return None
        try:
            stamp = str(bindings.stamp_factory())
            log_path = build_geometry_fit_log_path(
                stamp=stamp,
                log_dir=bindings.execution_bindings.log_dir,
                downloads_dir=bindings.execution_bindings.downloads_dir,
            )
            return write_geometry_fit_preflight_failure_log(
                stamp=stamp,
                error_text=str(error_text),
                log_path=log_path,
                log_sections=failure_log_sections,
            )
        except Exception:
            return None

    try:
        prepare_result = prepare_run(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            bindings=prepare_bindings,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        failure_log_sections = build_geometry_fit_preflight_log_sections(
            error_text=error_text,
            params=params,
            var_names=var_names,
            dataset_infos=None,
            current_dataset=None,
            selected_background_indices=(),
            joint_background_mode=False,
            geometry_runtime_cfg=preflight_runtime_cfg,
        )
        log_path = _persist_preflight_failure_log(error_text, failure_log_sections)
        prepare_result = GeometryFitPreparationResult(
            error_text=error_text,
            failure_log_sections=failure_log_sections,
            log_path=log_path,
        )
        bindings.execution_bindings.cmd_line(f"failed: {exc}")
        bindings.execution_bindings.set_progress_text(error_text)
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            prepare_result=prepare_result,
            error_text=error_text,
        )

    if prepare_result.prepared_run is None:
        should_force_preflight_log = bool(prepare_result.error_text) or bool(
            geometry_fit_debug_logging_enabled(preflight_runtime_cfg)
        )
        if should_force_preflight_log:
            preflight_error_text = str(
                prepare_result.error_text
                or "Geometry fit aborted before solver start."
            )
            failure_log_sections = (
                list(prepare_result.failure_log_sections or ())
                or build_geometry_fit_preflight_log_sections(
                    error_text=preflight_error_text,
                    params=params,
                    var_names=var_names,
                    dataset_infos=None,
                    current_dataset=None,
                    selected_background_indices=(),
                    joint_background_mode=False,
                    geometry_runtime_cfg=preflight_runtime_cfg,
                )
            )
            log_path = _persist_preflight_failure_log(
                preflight_error_text,
                failure_log_sections,
            )
            prepare_result = GeometryFitPreparationResult(
                prepared_run=prepare_result.prepared_run,
                error_text=prepare_result.error_text,
                failure_log_sections=[
                    (str(title), [str(line) for line in (lines or ())])
                    for title, lines in failure_log_sections
                ],
                log_path=log_path,
            )
            if prepare_result.error_text:
                bindings.execution_bindings.set_progress_text(
                    str(prepare_result.error_text)
                )
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            prepare_result=prepare_result,
            error_text=prepare_result.error_text,
        )

    execution_setup = build_execution_setup(
        prepared_run=prepare_result.prepared_run,
        mosaic_params=mosaic_params,
        stamp=str(bindings.stamp_factory()),
        bindings=bindings.execution_bindings,
    )
    execution_result = execute_run(
        prepared_run=prepare_result.prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        solve_fit=bindings.solve_fit,
        setup=execution_setup,
        flush_ui=bindings.flush_ui,
    )
    return GeometryFitRuntimeActionResult(
        params=params,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=execution_result.error_text,
    )


def execute_runtime_geometry_fit(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    preserve_live_theta: bool,
    solve_fit: Callable[..., object],
    setup: GeometryFitRuntimeExecutionSetup,
    start_progress_text: str = "Running geometry fit from saved manual Qr/Qz pairs…",
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeExecutionResult:
    """Run one prepared geometry fit through the live solver and callbacks."""

    ui_bindings = setup.ui_bindings
    postprocess_config = setup.postprocess_config
    execution_phase = execute_runtime_geometry_fit_solver_phase(
        prepared_run=prepared_run,
        var_names=var_names,
        solve_fit=solve_fit,
        solver_inputs=postprocess_config.solver_inputs,
        stamp=str(postprocess_config.stamp),
        log_path=postprocess_config.log_path,
        start_progress_text=str(start_progress_text),
        event_callback=(
            lambda kind, payload: (
                ui_bindings.cmd_line(str(payload.get("text", "")))
                if str(kind) == "cmd_line"
                else (
                    ui_bindings.set_progress_text(str(payload.get("text", "")))
                    if str(kind) == "progress_text"
                    else None
                ),
                flush_ui() if callable(flush_ui) and str(kind) in {"cmd_line", "progress_text"} else None,
            )
        ),
        live_update_callback=ui_bindings.live_update_callback,
    )
    return finalize_runtime_geometry_fit_execution(
        execution_result=execution_phase,
        prepared_run=prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        setup=setup,
    )
