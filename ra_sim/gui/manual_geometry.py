"""Helpers for manual geometry selection, caching, and serialization."""

from __future__ import annotations
import hashlib
import inspect
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ra_sim.fitting.background_peak_matching import (
    _candidate_summit_id_near_pixel as _background_candidate_summit_id_near_pixel,
)
from ra_sim.fitting.background_peak_matching import (
    _refine_peak_center as _background_refine_peak_center,
)
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import mosaic_top_selection as gui_mosaic_top
from ra_sim.gui.geometry_overlay import normalize_hkl_key as _default_normalize_hkl_key
from ra_sim.gui.geometry_overlay import (
    display_point_to_native_for_rotation as _default_display_point_to_native_for_rotation,
)
from ra_sim.gui.geometry_overlay import rotate_point_for_display as _default_rotate_point
from ra_sim.simulation.exact_cake_portable import (
    CakeTransformBundle,
    caked_point_to_detector_pixel as _caked_point_to_detector_pixel,
    detector_pixel_to_caked_bin as _detector_pixel_to_caked_bin,
)
from ra_sim.utils.calculations import (
    resolve_canonical_branch,
    source_branch_index_from_phi_deg,
)
from ra_sim.utils.parallel import (
    default_reserved_cpu_worker_count,
    make_detached_thread_pool_executor,
)
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


DEFAULT_POSITION_SIGMA_FLOOR_PX = 0.75
DEFAULT_PREVIEW_GOOD_SIGMA_PX = 1.5
DEFAULT_PREVIEW_BAD_SIGMA_PX = 12.0
DEFAULT_CAKED_SEARCH_TTH_DEG = 1.5
DEFAULT_CAKED_SEARCH_PHI_DEG = 10.0
DEFAULT_AUTO_REFINE_SEARCH_RADIUS_PX = 24.0
MIN_AUTO_REFINE_SEARCH_RADIUS_PX = 4.0
MAX_AUTO_REFINE_SEARCH_RADIUS_PX = 128.0
MAX_AUTO_FINAL_REFINE_WORKERS = 8
DEFAULT_PREVIEW_MIN_INTERVAL_S = 0.03
DEFAULT_PREVIEW_MIN_MOVE_PX = 0.8
DEFAULT_GUI_ENTRYPOINT = "python -m ra_sim gui"
MANUAL_GEOMETRY_SOURCE_PATH_MARKER = "manual_preview_visual_distance_patch_v1"
MANUAL_GEOMETRY_TRACE_VERSION = MANUAL_GEOMETRY_SOURCE_PATH_MARKER
MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID = "legacy_unattributed"
_GEOMETRY_MANUAL_TRUSTED_PROJECTION_TOKEN_SCHEMA = "geometry_fit_projection_content_v3"
_GEOMETRY_MANUAL_TRUSTED_PROJECTION_TOKEN_SOURCE = "runtime_projection_storage_copy_v3"
_GEOMETRY_MANUAL_LIVE_CODE_PATH_STAMP_PRINTED = False
_GEOMETRY_MANUAL_RUN_COUNTER = 0
_GEOMETRY_MANUAL_ACTIVE_RUN_ID = "<no-active-manual-geometry-run>"
_GEOMETRY_MANUAL_CAKED_QR_ID_FIELDS = (
    "source_table_index",
    "source_row_index",
    "source_reflection_index",
    "source_branch_index",
    "source_ray_id",
    "branch_id",
)


def _geometry_manual_projection_blake2b_hex(*parts: bytes | str) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, str):
            raw = part.encode("utf-8", "backslashreplace")
        else:
            raw = bytes(part)
        hasher.update(len(raw).to_bytes(8, "little", signed=False))
        hasher.update(raw)
    return hasher.hexdigest()


def geometry_manual_stable_axis_value_token(values: object) -> tuple[object, ...] | None:
    """Return a deterministic content token for projection axes."""

    try:
        arr = np.asarray(values, dtype=np.dtype("<f8")).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0 or not np.all(np.isfinite(arr)):
        return None
    arr = np.ascontiguousarray(arr)
    return (
        "axis_content_v3",
        int(arr.size),
        "<f8",
        _geometry_manual_projection_blake2b_hex(arr.tobytes(order="C")),
    )


def geometry_manual_stable_permutation_value_token(
    values: object,
) -> tuple[object, ...] | None:
    """Return a deterministic content token for row permutations."""

    try:
        arr = np.asarray(values, dtype=np.dtype("<i8")).reshape(-1)
    except Exception:
        return None
    if arr.size <= 0:
        return None
    arr = np.ascontiguousarray(arr)
    return (
        "perm_content_v3",
        int(arr.size),
        "<i8",
        _geometry_manual_projection_blake2b_hex(arr.tobytes(order="C")),
    )


def _geometry_manual_projection_axis_token(values: object) -> tuple[object, ...] | None:
    return geometry_manual_stable_axis_value_token(values)


def _geometry_manual_projection_permutation_token(values: object) -> tuple[object, ...] | None:
    return geometry_manual_stable_permutation_value_token(values)


def _geometry_manual_projection_dense_token(
    values: object,
    *,
    kind: str,
    allow_empty: bool = True,
) -> tuple[object, ...] | None:
    try:
        raw = np.asarray(values)
    except Exception:
        return None
    if raw.dtype.kind == "O":
        return None
    if raw.dtype.kind in {"f", "c"}:
        try:
            arr = np.asarray(values, dtype=np.dtype("<f8"))
        except Exception:
            return None
        if not np.all(np.isfinite(arr)):
            return None
    elif raw.dtype.kind in {"i", "u"}:
        try:
            arr = np.asarray(values, dtype=np.dtype("<i8"))
        except Exception:
            return None
    elif raw.dtype.kind == "b":
        arr = np.asarray(values, dtype=np.dtype("u1"))
    else:
        return None
    if arr.size <= 0 and not bool(allow_empty):
        return None
    arr = np.ascontiguousarray(arr)
    return (
        "manual_projection_dense_content_v3",
        str(kind),
        tuple(int(v) for v in arr.shape),
        str(arr.dtype),
        _geometry_manual_projection_blake2b_hex(
            "manual_projection_dense_content_v3",
            str(kind),
            repr(tuple(int(v) for v in arr.shape)),
            str(arr.dtype),
            arr.tobytes(order="C"),
        ),
    )


def _geometry_manual_projection_sparse_token(
    values: object,
    *,
    kind: str,
) -> tuple[object, ...] | None:
    if not hasattr(values, "tocsr"):
        return None
    try:
        csr = values.tocsr(copy=True)  # type: ignore[attr-defined]
    except TypeError:
        try:
            csr = values.tocsr()  # type: ignore[attr-defined]
        except Exception:
            return None
    except Exception:
        return None
    try:
        csr.sort_indices()
    except Exception:
        pass
    try:
        shape = tuple(int(v) for v in tuple(csr.shape)[:2])
        indptr = np.ascontiguousarray(np.asarray(csr.indptr, dtype=np.dtype("<i8")).reshape(-1))
        indices = np.ascontiguousarray(np.asarray(csr.indices, dtype=np.dtype("<i8")).reshape(-1))
        data_raw = np.asarray(csr.data)
    except Exception:
        return None
    if len(shape) < 2 or shape[0] < 0 or shape[1] < 0 or data_raw.dtype.kind == "O":
        return None
    if data_raw.dtype.kind in {"f", "c"}:
        data = np.asarray(csr.data, dtype=np.dtype("<f8")).reshape(-1)
        if not np.all(np.isfinite(data)):
            return None
    elif data_raw.dtype.kind in {"i", "u", "b"}:
        data = np.asarray(csr.data, dtype=np.dtype("<i8")).reshape(-1)
    else:
        return None
    data = np.ascontiguousarray(data)
    return (
        "manual_projection_sparse_csr_content_v3",
        str(kind),
        shape,
        int(indptr.size),
        int(indices.size),
        int(data.size),
        str(data.dtype),
        _geometry_manual_projection_blake2b_hex(
            "manual_projection_sparse_csr_content_v3",
            str(kind),
            repr(shape),
            indptr.tobytes(order="C"),
            indices.tobytes(order="C"),
            str(data.dtype),
            data.tobytes(order="C"),
        ),
    )


def _geometry_manual_projection_transform_token(
    values: object,
    *,
    kind: str,
) -> tuple[object, ...] | None:
    if values is None:
        return ("manual_projection_transform_none_v3", str(kind))
    if all(
        hasattr(values, attr) for attr in ("image_shape", "n_rad", "n_az", "matrix", "count_flat")
    ):
        try:
            image_shape = tuple(int(v) for v in tuple(values.image_shape)[:2])
            n_rad = int(values.n_rad)
            n_az = int(values.n_az)
        except Exception:
            return None
        matrix_token = _geometry_manual_projection_transform_token(
            values.matrix,
            kind=f"{kind}.matrix",
        )
        count_token = _geometry_manual_projection_dense_token(
            values.count_flat,
            kind=f"{kind}.count_flat",
            allow_empty=True,
        )
        if matrix_token is None or count_token is None:
            return None
        return (
            "manual_projection_detector_cake_lut_content_v3",
            str(kind),
            image_shape,
            int(n_rad),
            int(n_az),
            matrix_token,
            count_token,
        )
    sparse_token = _geometry_manual_projection_sparse_token(values, kind=kind)
    if sparse_token is not None:
        return sparse_token
    return _geometry_manual_projection_dense_token(values, kind=kind, allow_empty=True)


def _geometry_manual_projection_bundle_token(bundle: object) -> tuple[object, ...] | None:
    if not isinstance(bundle, CakeTransformBundle):
        return None
    detector_shape = _geometry_manual_detector_shape_token(getattr(bundle, "detector_shape", None))
    radial_token = _geometry_manual_projection_axis_token(getattr(bundle, "radial_deg", None))
    gui_token = _geometry_manual_projection_axis_token(getattr(bundle, "gui_azimuth_deg", None))
    raw_token = _geometry_manual_projection_axis_token(getattr(bundle, "raw_azimuth_deg", None))
    lut_token = _geometry_manual_projection_transform_token(
        getattr(bundle, "lut", None), kind="lut"
    )
    lut_t_token = _geometry_manual_projection_transform_token(
        getattr(bundle, "lut_t", None),
        kind="lut_t",
    )
    if (
        detector_shape is None
        or radial_token is None
        or gui_token is None
        or raw_token is None
        or lut_token is None
        or lut_t_token is None
    ):
        return None
    return (
        "manual_projection_cake_transform_bundle_content_v3",
        detector_shape,
        radial_token,
        gui_token,
        raw_token,
        lut_token,
        lut_t_token,
    )


def _geometry_manual_detector_shape_token(values: object) -> tuple[int, int] | None:
    try:
        shape = tuple(int(value) for value in tuple(values)[:2])  # type: ignore[arg-type]
    except Exception:
        return None
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        return None
    return int(shape[0]), int(shape[1])


def geometry_manual_normalize_auto_refine_search_radius_px(
    value: object,
    *,
    default: float = DEFAULT_AUTO_REFINE_SEARCH_RADIUS_PX,
    min_px: float = MIN_AUTO_REFINE_SEARCH_RADIUS_PX,
    max_px: float = MAX_AUTO_REFINE_SEARCH_RADIUS_PX,
) -> float:
    """Return a finite auto-refinement search radius in display pixels."""

    try:
        radius = float(value)
    except Exception:
        radius = float(default)
    if not np.isfinite(radius):
        radius = float(default)
    lower = float(min_px)
    upper = float(max_px)
    if not np.isfinite(lower):
        lower = float(MIN_AUTO_REFINE_SEARCH_RADIUS_PX)
    if not np.isfinite(upper) or upper < lower:
        upper = float(MAX_AUTO_REFINE_SEARCH_RADIUS_PX)
    return float(np.clip(radius, lower, upper))


def geometry_manual_caked_search_windows_for_radius(
    search_radius_px: object,
    *,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    default_tth_window_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    default_phi_window_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
) -> tuple[float, float]:
    """Convert a pixel search radius into caked two-theta/phi windows."""

    radius = geometry_manual_normalize_auto_refine_search_radius_px(search_radius_px)

    def _axis_step(axis_values: Sequence[float] | None) -> float:
        try:
            axis = np.asarray(axis_values, dtype=float)
        except Exception:
            return float("nan")
        if axis.size < 2:
            return float("nan")
        diffs = np.abs(np.diff(axis[np.isfinite(axis)]))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size <= 0:
            return float("nan")
        return float(np.median(diffs))

    radial_step = _axis_step(radial_axis)
    azimuth_step = _axis_step(azimuth_axis)
    tth_window = float(default_tth_window_deg)
    phi_window = float(default_phi_window_deg)
    if np.isfinite(radial_step) and radial_step > 0.0:
        tth_window = max(tth_window, float(radius) * float(radial_step))
    if np.isfinite(azimuth_step) and azimuth_step > 0.0:
        phi_window = max(phi_window, float(radius) * float(azimuth_step))
    return float(tth_window), float(phi_window)


def geometry_manual_cache_with_auto_refine_search_radius(
    cache_data: Mapping[str, object] | None,
    *,
    search_radius_px: object | None,
) -> dict[str, object]:
    """Return cache data with an auto-placement-only refinement radius override."""

    output = dict(cache_data) if isinstance(cache_data, Mapping) else {}
    if search_radius_px is None:
        return output
    radius = geometry_manual_normalize_auto_refine_search_radius_px(search_radius_px)
    match_cfg = (
        dict(output.get("match_config")) if isinstance(output.get("match_config"), Mapping) else {}
    )
    match_cfg["search_radius_px"] = float(radius)
    min_margin = max(96.0, 6.0 * float(radius))
    try:
        current_margin = float(match_cfg.get("context_margin_px", float("nan")))
    except Exception:
        current_margin = float("nan")
    if not np.isfinite(current_margin) or current_margin < min_margin:
        match_cfg["context_margin_px"] = float(min_margin)
    output["match_config"] = match_cfg
    output["manual_auto_refine_search_radius_px"] = float(radius)
    return output


def geometry_manual_remove_q_group_peak_entries(
    entries: Sequence[Mapping[str, object]] | None,
    *,
    selected_q_group_entries: Sequence[Mapping[str, object]] | None = None,
    selected_q_group_keys: Sequence[object] | None = None,
) -> tuple[list[dict[str, object]], int, set[tuple[object, ...]]]:
    """Remove saved manual placements whose Qr/Qz group key is selected."""

    remove_keys: set[tuple[object, ...]] = set()

    def _append_key(value: object) -> None:
        group_key = gui_controllers.normalize_geometry_q_group_key(value)
        if group_key is not None:
            remove_keys.add(group_key)

    for selected_entry in selected_q_group_entries or ():
        if not isinstance(selected_entry, Mapping):
            continue
        _append_key(selected_entry.get("key", selected_entry.get("q_group_key")))
    for raw_key in selected_q_group_keys or ():
        _append_key(raw_key)

    kept_entries: list[dict[str, object]] = []
    removed_count = 0
    for raw_entry in entries or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        entry_key = gui_controllers.normalize_geometry_q_group_key(entry.get("q_group_key"))
        if entry_key in remove_keys:
            removed_count += 1
            continue
        kept_entries.append(entry)
    return kept_entries, int(removed_count), set(remove_keys)


def _geometry_manual_git_commit() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return "<unavailable>"
    commit = str(result.stdout or "").strip()
    return commit if result.returncode == 0 and commit else "<unavailable>"


def geometry_manual_live_code_path_stamp() -> str:
    runtime_module = sys.modules.get("ra_sim.gui._runtime.runtime_session")
    runtime_file = getattr(runtime_module, "__file__", None)
    return (
        "[geometry] manual_geometry_live_code_path "
        + geometry_manual_cmd_provenance_text(
            emitter="geometry_manual_live_code_path_stamp",
            event="module_stamp",
        )
        + " "
        f"manual_geometry.__file__={Path(__file__).resolve()} "
        f"runtime_session.__file__={Path(runtime_file).resolve() if runtime_file else '<unavailable>'} "
        f"git_commit={_geometry_manual_git_commit()}"
    )


def print_geometry_manual_live_code_path_stamp(*, force: bool = False) -> str:
    global _GEOMETRY_MANUAL_LIVE_CODE_PATH_STAMP_PRINTED
    stamp = geometry_manual_live_code_path_stamp()
    if bool(force) or not _GEOMETRY_MANUAL_LIVE_CODE_PATH_STAMP_PRINTED:
        print(stamp)
        _GEOMETRY_MANUAL_LIVE_CODE_PATH_STAMP_PRINTED = True
    return stamp


def geometry_manual_start_run_id() -> str:
    """Start one attributed manual-geometry arming session."""

    global _GEOMETRY_MANUAL_ACTIVE_RUN_ID
    global _GEOMETRY_MANUAL_RUN_COUNTER
    _GEOMETRY_MANUAL_RUN_COUNTER += 1
    _GEOMETRY_MANUAL_ACTIVE_RUN_ID = f"manual-{_GEOMETRY_MANUAL_RUN_COUNTER:06d}"
    try:
        _LIVE_CAKED_TRACE_LAST.clear()
    except NameError:
        pass
    return _GEOMETRY_MANUAL_ACTIVE_RUN_ID


def geometry_manual_current_run_id() -> str:
    return str(_GEOMETRY_MANUAL_ACTIVE_RUN_ID)


def _geometry_manual_session_run_id(
    pick_session: Mapping[str, object] | None,
) -> str:
    if isinstance(pick_session, Mapping):
        raw_run_id = pick_session.get("manual_geometry_run_id")
        if raw_run_id is not None and str(raw_run_id):
            return str(raw_run_id)
    return MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID


def _geometry_manual_entry_run_id(
    *entries: Mapping[str, object] | Sequence[Mapping[str, object]] | object,
) -> str | None:
    for entry in entries:
        if isinstance(entry, Mapping):
            raw_run_id = entry.get("manual_geometry_run_id")
            if raw_run_id is not None and str(raw_run_id):
                return str(raw_run_id)
            continue
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            for item in entry:
                if not isinstance(item, Mapping):
                    continue
                raw_run_id = item.get("manual_geometry_run_id")
                if raw_run_id is not None and str(raw_run_id):
                    return str(raw_run_id)
    return None


def geometry_manual_cmd_provenance_text(
    *,
    run_id: object | None = None,
    emitter: str,
    event: str | None = None,
    branch: object | None = None,
    actual_source: object | None = None,
    expected_source: object | None = None,
    **extra_fields: object,
) -> str:
    resolved_run_id = (
        str(run_id) if run_id is not None and str(run_id) else MANUAL_GEOMETRY_UNATTRIBUTED_RUN_ID
    )
    parts = [
        f"manual_geometry_run_id={resolved_run_id}",
        f"manual_trace_version={MANUAL_GEOMETRY_TRACE_VERSION}",
        f"source_path_marker={MANUAL_GEOMETRY_SOURCE_PATH_MARKER}",
        f"emitter={str(emitter)}",
    ]
    if event is not None:
        parts.append(f"event={str(event)}")
    if branch is not None:
        parts.append(f"branch={str(branch)}")
    if actual_source is not None:
        parts.append(f"actual_source={str(actual_source)}")
    if expected_source is not None:
        parts.append(f"expected_source={str(expected_source)}")
    for key, value in extra_fields.items():
        if value is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        parts.append(f"{key_text}={str(value)}")
    return " ".join(parts)


def _geometry_manual_expected_distance_source(use_caked_space: bool) -> str:
    return "sim_visual_caked_deg" if bool(use_caked_space) else "sim_visual_detector_display_px"


_GEOMETRY_MANUAL_QR_SIM_ALIAS_FIELDS = (
    "refined_sim_x",
    "refined_sim_y",
    "refined_sim_caked_x",
    "refined_sim_caked_y",
    "display_col",
    "display_row",
    "sim_col",
    "sim_row",
    "sim_col_local",
    "sim_row_local",
    "sim_col_global",
    "sim_row_global",
    "caked_x",
    "caked_y",
    "raw_caked_x",
    "raw_caked_y",
    "two_theta_deg",
    "phi_deg",
    "simulated_two_theta_deg",
    "simulated_phi_deg",
)
_GEOMETRY_MANUAL_CAKED_VISUAL_PRESERVE_KEYS = (
    "sim_refined_caked_deg",
    "sim_visual_deg",
    "sim_visual_source",
    "sim_caked",
    "sim_refinement_source",
    "sim_refinement_status",
    "sim_refinement_delta_caked_deg",
    "refined_sim_caked_x",
    "refined_sim_caked_y",
)


@dataclass(frozen=True)
class GeometryManualRuntimeCallbacks:
    """Bound runtime callbacks for manual geometry preview and pick actions."""

    render_current_pairs: Callable[..., bool]
    toggle_selection_at: Callable[[float, float], bool]
    place_selection_at: Callable[[float, float], bool]
    update_pick_preview: Callable[..., None]
    cancel_pick_session: Callable[..., None]


@dataclass(frozen=True)
class GeometryManualRuntimeCacheCallbacks:
    """Bound runtime callbacks for manual-geometry cache and overlay state."""

    current_match_config: Callable[[], dict[str, object]]
    pick_cache_signature: Callable[..., tuple[object, ...]]
    get_pick_cache: Callable[..., dict[str, object]]
    prewarm_pick_cache: Callable[..., dict[str, object]]
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ]


@dataclass(frozen=True)
class GeometryManualRuntimeProjectionCallbacks:
    """Bound runtime callbacks for manual-geometry view/projection helpers."""

    pick_uses_caked_space: Callable[[], bool]
    current_background_image: Callable[[], object | None]
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ]
    refresh_entry_geometry: Callable[
        [Mapping[str, object] | None],
        dict[str, object] | None,
    ]
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    project_peaks_to_caked_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    simulated_peaks_for_params: Callable[..., list[dict[str, object]]]
    last_simulation_diagnostics: Callable[[], dict[str, object]]
    pick_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ]
    simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ]
    caked_projection_signature: Callable[[], object]


GeometryManualLookupBucket = dict[str, object] | list[dict[str, object]]
GeometryManualLookupMap = dict[tuple[object, ...], GeometryManualLookupBucket]


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _geometry_manual_callable_accepts_keyword(
    callback: Callable[..., object] | None,
    keyword: str,
) -> bool | None:
    if not callable(callback):
        return False
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return None
    parameters = signature.parameters
    return str(keyword) in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _geometry_manual_type_error_is_keyword_arity_mismatch(
    exc: TypeError,
    keyword: str,
) -> bool:
    message = str(exc).lower()
    keyword_lower = str(keyword).lower()
    return bool(
        keyword_lower in message
        and (
            "unexpected keyword" in message
            or "got an unexpected" in message
            or "invalid keyword" in message
        )
    )


def geometry_manual_position_error_px(
    raw_col: float,
    raw_row: float,
    refined_col: float,
    refined_row: float,
) -> float:
    """Return the click-to-refined placement error in display pixels."""

    try:
        delta = float(
            np.hypot(
                float(refined_col) - float(raw_col),
                float(refined_row) - float(raw_row),
            )
        )
    except Exception:
        return 0.0
    if not np.isfinite(delta):
        return 0.0
    return max(0.0, float(delta))


def geometry_manual_position_sigma_px(
    placement_error_px: object,
    *,
    floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> float:
    """Convert a manual click-placement error into a fit sigma in pixels."""

    try:
        error_px = float(placement_error_px)
    except Exception:
        error_px = 0.0
    if not np.isfinite(error_px):
        error_px = 0.0

    floor_val = max(1.0e-3, float(floor_px))
    return float(np.hypot(float(error_px), floor_val))


def _canonicalize_manual_entry_branch_fields(
    entry: dict[str, object],
    *,
    allow_legacy_peak_fallback: bool,
    preserve_legacy_peak_when_unresolved: bool = False,
    retry_legacy_peak_on_deadband: bool = False,
) -> None:
    """Stamp canonical branch identity onto one saved/manual geometry entry."""

    try:
        legacy_peak_idx = int(entry.get("source_peak_index"))
    except Exception:
        legacy_peak_idx = None

    branch_idx, _branch_source, branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=allow_legacy_peak_fallback,
    )
    if branch_idx in {0, 1}:
        entry["source_branch_index"] = int(branch_idx)
        entry["source_peak_index"] = int(branch_idx)
        return
    if (
        retry_legacy_peak_on_deadband
        and branch_reason == "ambiguous_branch_deadband"
        and legacy_peak_idx in {0, 1}
    ):
        entry["source_branch_index"] = int(legacy_peak_idx)
        entry["source_peak_index"] = int(legacy_peak_idx)
        return

    entry.pop("source_branch_index", None)
    if not preserve_legacy_peak_when_unresolved and (
        legacy_peak_idx in {0, 1} or legacy_peak_idx is None
    ):
        entry.pop("source_peak_index", None)


def _coerce_nonnegative_index(value: object) -> int | None:
    try:
        idx = int(value)
    except Exception:
        return None
    return int(idx) if idx >= 0 else None


def _resolve_legacy_source_peak_index(entry: Mapping[str, object] | None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    return _coerce_nonnegative_index(entry.get("source_peak_index"))


def _entry_has_explicit_trusted_reflection_identity(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    reflection_idx = _coerce_nonnegative_index(entry.get("source_reflection_index"))
    namespace = str(entry.get("source_reflection_namespace", "") or "").strip().lower()
    return (
        reflection_idx is not None
        and namespace == "full_reflection"
        and bool(entry.get("source_reflection_is_full", False))
    )


def _strip_live_source_trust_fields(entry: dict[str, object]) -> None:
    entry.pop("source_reflection_index", None)
    entry.pop("source_reflection_namespace", None)
    entry.pop("source_reflection_is_full", None)


def _set_live_source_trust_fields(
    entry: dict[str, object],
    *,
    reflection_index: int,
) -> None:
    entry["source_reflection_index"] = int(reflection_index)
    entry["source_reflection_namespace"] = "full_reflection"
    entry["source_reflection_is_full"] = True


def geometry_manual_canonicalize_live_source_entry(
    entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[
        [object],
        tuple[int, int, int] | None,
    ] = _default_normalize_hkl_key,
    allow_legacy_peak_fallback: bool = False,
    preserve_existing_trusted_identity: bool = False,
    trusted_reflection_index: object = None,
    source_reflection_indices_local: Sequence[object] | None = None,
    source_row_hkl_lookup: Mapping[tuple[int, int], tuple[int, int, int]] | None = None,
    provenance_signature_matches: bool = False,
    provenance_revision_matches: bool = False,
    expected_table_count: int | None = None,
) -> dict[str, object] | None:
    """Canonicalize one live/source row using single-source branch/trust rules."""

    if not isinstance(entry, Mapping):
        return None

    normalized = dict(entry)
    hkl_key = normalize_hkl_key(
        normalized.get("hkl_raw", normalized.get("hkl", normalized.get("label")))
    )
    if hkl_key is not None:
        normalized["hkl"] = hkl_key
        if not str(normalized.get("label", "") or "").strip():
            normalized["label"] = f"{hkl_key[0]},{hkl_key[1]},{hkl_key[2]}"

    raw_group_key = normalized.get("q_group_key")
    if isinstance(raw_group_key, list):
        normalized["q_group_key"] = tuple(raw_group_key)
    elif not isinstance(raw_group_key, tuple):
        normalized.pop("q_group_key", None)

    _canonicalize_manual_entry_branch_fields(
        normalized,
        allow_legacy_peak_fallback=bool(allow_legacy_peak_fallback),
        preserve_legacy_peak_when_unresolved=True,
    )

    trusted_restore_idx = _coerce_nonnegative_index(trusted_reflection_index)
    if trusted_restore_idx is not None:
        _set_live_source_trust_fields(
            normalized,
            reflection_index=int(trusted_restore_idx),
        )
        return normalized

    if preserve_existing_trusted_identity and _entry_has_explicit_trusted_reflection_identity(
        normalized
    ):
        return normalized

    reflection_index_map = (
        list(source_reflection_indices_local or ())
        if isinstance(source_reflection_indices_local, Sequence)
        and not isinstance(source_reflection_indices_local, (str, bytes))
        else []
    )
    expected_count = (
        int(expected_table_count)
        if isinstance(expected_table_count, int) and expected_table_count >= 0
        else None
    )
    table_idx = _coerce_nonnegative_index(normalized.get("source_table_index"))
    row_idx = _coerce_nonnegative_index(normalized.get("source_row_index"))
    trust_proven = bool(provenance_signature_matches or provenance_revision_matches)
    map_len_matches = bool(reflection_index_map) and (
        expected_count is None or int(len(reflection_index_map)) == int(expected_count)
    )
    if (
        trust_proven
        and map_len_matches
        and table_idx is not None
        and row_idx is not None
        and table_idx < len(reflection_index_map)
        and isinstance(source_row_hkl_lookup, Mapping)
    ):
        reflection_idx = _coerce_nonnegative_index(reflection_index_map[int(table_idx)])
        active_hkl_value = source_row_hkl_lookup.get((int(table_idx), int(row_idx)))
        active_hkl = (
            tuple(int(v) for v in active_hkl_value[:3])
            if isinstance(active_hkl_value, (list, tuple, np.ndarray))
            and len(active_hkl_value) >= 3
            else None
        )
        if (
            reflection_idx is not None
            and hkl_key is not None
            and isinstance(active_hkl, tuple)
            and tuple(int(v) for v in active_hkl) == tuple(int(v) for v in hkl_key)
        ):
            _set_live_source_trust_fields(
                normalized,
                reflection_index=int(reflection_idx),
            )
            return normalized

    _strip_live_source_trust_fields(normalized)
    return normalized


def refresh_geometry_manual_pair_entry(
    entry: Mapping[str, object] | None,
    *,
    background_display_shape: Sequence[object] | None,
    background_display_to_native_detector_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ),
    caked_angles_to_background_display_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ) = None,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None,
    rotate_point_for_display: Callable[
        [float, float, tuple[int, ...], int],
        tuple[float, float],
    ] = _default_rotate_point,
    display_rotate_k: int = 0,
    normalize_hkl_key: Callable[
        [object],
        tuple[int, int, int] | None,
    ] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
    stale_caked_tolerance_px: float = 0.5,
    allow_legacy_peak_fallback: bool = False,
    current_projected_sim_entry: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Refresh cached detector, display, and caked fields for one manual entry."""

    raw_entry = dict(entry) if isinstance(entry, Mapping) else None
    entry_for_normalize = dict(raw_entry) if raw_entry is not None else None

    def _points_match(
        left: tuple[float, float] | None,
        right: tuple[float, float] | None,
        *,
        tol: float = 1.0e-9,
    ) -> bool:
        return bool(
            left is not None
            and right is not None
            and abs(float(left[0]) - float(right[0])) <= float(tol)
            and abs(float(left[1]) - float(right[1])) <= float(tol)
        )

    if isinstance(entry_for_normalize, dict):
        seeded_xy = None
        existing_xy = _geometry_manual_finite_point(
            entry_for_normalize,
            (("x", "y"),),
        )
        if existing_xy is None:
            seeded_xy = _geometry_manual_finite_point(
                entry_for_normalize,
                (("sim_col_raw", "sim_row_raw"),),
            )
        if seeded_xy is None:
            display_point = _geometry_manual_finite_point(
                entry_for_normalize,
                (("display_col", "display_row"),),
            )
            caked_point = _geometry_manual_finite_point(
                entry_for_normalize,
                (
                    ("caked_x", "caked_y"),
                    ("raw_caked_x", "raw_caked_y"),
                    ("background_two_theta_deg", "background_phi_deg"),
                ),
            )
            if display_point is not None and not _points_match(display_point, caked_point):
                seeded_xy = display_point
        if seeded_xy is not None:
            entry_for_normalize["x"] = float(seeded_xy[0])
            entry_for_normalize["y"] = float(seeded_xy[1])
    normalized = normalize_geometry_manual_pair_entry(
        entry_for_normalize,
        normalize_hkl_key=normalize_hkl_key,
        sigma_floor_px=sigma_floor_px,
    )
    if normalized is None:
        return None

    _canonicalize_manual_entry_branch_fields(
        normalized,
        allow_legacy_peak_fallback=allow_legacy_peak_fallback,
        preserve_legacy_peak_when_unresolved=False,
    )
    normalized.pop("stale_caked_fields", None)

    normalized_origin = _geometry_manual_normalized_frame_token(
        normalized.get("manual_background_input_origin")
    )
    if normalized_origin != "caked":
        _geometry_manual_apply_sim_visual_detector_fields(
            normalized,
            detector_display_to_native_coords=background_display_to_native_detector_coords,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
    _geometry_manual_apply_detector_origin_conversion_ledger(
        normalized,
        background_display_to_native_detector_coords=(background_display_to_native_detector_coords),
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
    )

    def _finite_pair(
        x_key: str,
        y_key: str,
        *,
        source: Mapping[str, object] | None = None,
    ) -> tuple[float, float] | None:
        raw_source = source if isinstance(source, Mapping) else normalized
        try:
            col = float(raw_source.get(x_key, np.nan))
            row = float(raw_source.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _finite_tuple_pair(
        value: object,
    ) -> tuple[float, float] | None:
        if not isinstance(value, tuple) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    try:
        shape = tuple(int(v) for v in (background_display_shape or ())[:2])
    except Exception:
        shape = ()

    saved_background_caked_point = _finite_pair(
        "background_two_theta_deg",
        "background_phi_deg",
        source=raw_entry,
    )
    caked_background_is_authoritative = saved_background_caked_point is not None
    saved_detector_point = _finite_pair("detector_x", "detector_y", source=raw_entry)
    if saved_detector_point is None:
        saved_detector_point = _finite_pair(
            "background_detector_x",
            "background_detector_y",
            source=raw_entry,
        )
    detector_fields_are_authoritative = bool(
        saved_background_caked_point is None and saved_detector_point is not None
    )
    caked_point = saved_background_caked_point
    if caked_point is None and not detector_fields_are_authoritative:
        caked_point = _finite_pair("caked_x", "caked_y")
    if caked_point is None and not detector_fields_are_authoritative:
        caked_point = _finite_pair("raw_caked_x", "raw_caked_y")
    if caked_point is not None:
        normalized["background_two_theta_deg"] = float(caked_point[0])
        normalized["background_phi_deg"] = float(caked_point[1])
        normalized["caked_x"] = float(caked_point[0])
        normalized["caked_y"] = float(caked_point[1])
        normalized["two_theta_deg"] = float(caked_point[0])
        normalized["phi_deg"] = float(caked_point[1])
        if _finite_pair("raw_caked_x", "raw_caked_y") is None:
            normalized["raw_caked_x"] = float(caked_point[0])
            normalized["raw_caked_y"] = float(caked_point[1])

    projected_sim_entry = _geometry_manual_caked_qr_projection_entry(current_projected_sim_entry)
    sim_detector_replay = None
    sim_replay_caked_point = _geometry_manual_finite_point(
        projected_sim_entry,
        (
            ("caked_x", "caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
    )
    sim_replay_eligible = bool(
        projected_sim_entry is not None
        and sim_replay_caked_point is not None
        and _geometry_manual_caked_qr_projection_source(
            raw_entry if raw_entry is not None else normalized
        )
        is not None
    )
    if sim_replay_eligible:
        sim_detector_replay = resolve_sim_detector_replay_from_caked_projection(
            raw_entry if raw_entry is not None else normalized,
            projected_sim_entry,
            caked_angles_to_background_display_coords=(caked_angles_to_background_display_coords),
            background_display_to_native_detector_coords=(
                background_display_to_native_detector_coords
            ),
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            stale_caked_tolerance_px=float(stale_caked_tolerance_px),
        )
        if sim_detector_replay is None:
            for stale_key in (
                "x",
                "y",
                "display_col",
                "display_row",
                "detector_x",
                "detector_y",
                "refined_sim_x",
                "refined_sim_y",
                "native_col",
                "native_row",
                "refined_sim_native_x",
                "refined_sim_native_y",
                "sim_native_x",
                "sim_native_y",
                "sim_col",
                "sim_row",
                "sim_col_raw",
                "sim_row_raw",
                "sim_detector_anchor_x",
                "sim_detector_anchor_y",
                "sim_detector_display_col",
                "sim_detector_display_row",
                "sim_detector_frame_provenance",
            ):
                normalized.pop(stale_key, None)
            return normalized

    display_point = _finite_pair("x", "y")
    if display_point is None:
        display_point = _finite_pair("display_col", "display_row", source=raw_entry)
    raw_detector_display = _finite_pair(
        "sim_col_raw",
        "sim_row_raw",
        source=raw_entry,
    )
    detector_point = None
    detector_point_source = "unresolved"
    if sim_detector_replay is not None:
        detector_point = _finite_pair(
            "sim_detector_anchor_x",
            "sim_detector_anchor_y",
            source=sim_detector_replay,
        )
        detector_point_source = str(
            sim_detector_replay.get(
                "sim_detector_frame_provenance",
                "sim_reverse_lut_replay_cache",
            )
        )
        cached_sim_display = _finite_pair(
            "sim_detector_display_col",
            "sim_detector_display_row",
            source=sim_detector_replay,
        )
        if cached_sim_display is not None:
            raw_detector_display = cached_sim_display
            if display_point is None:
                display_point = cached_sim_display
        else:
            raw_detector_display = None
            display_point = None
    elif caked_background_is_authoritative:
        detector_point = _finite_pair(
            "background_detector_x",
            "background_detector_y",
        )
        if detector_point is not None:
            detector_point_source = "background_detector_fields"
        if detector_point is None:
            detector_point = _finite_pair(
                "refined_detector_native_col",
                "refined_detector_native_row",
                source=raw_entry,
            )
            if detector_point is not None:
                detector_point_source = "refined_detector_native_cache"
        if detector_point is None and not callable(caked_angles_to_background_display_coords):
            detector_point = _finite_pair("detector_x", "detector_y")
            if detector_point is not None:
                detector_point_source = "legacy_detector_fields"
    else:
        native_point = _finite_pair("native_col", "native_row", source=raw_entry)
        if native_point is None:
            native_point = _finite_pair("sim_native_x", "sim_native_y", source=raw_entry)
        detector_point = native_point
        detector_point_source = "native_detector_coords"
        if (
            detector_point is None
            and raw_detector_display is not None
            and callable(background_display_to_native_detector_coords)
        ):
            detector_point = _finite_tuple_pair(
                background_display_to_native_detector_coords(
                    float(raw_detector_display[0]),
                    float(raw_detector_display[1]),
                )
            )
            if detector_point is not None:
                detector_point_source = "sim_col_raw_inverse_projection"
            if display_point is None:
                display_point = (
                    float(raw_detector_display[0]),
                    float(raw_detector_display[1]),
                )
        if detector_point is None:
            detector_point = _finite_pair("detector_x", "detector_y")
            if detector_point is not None:
                detector_point_source = "detector_fields"
        if detector_point is None:
            detector_point = _finite_pair(
                "background_detector_x",
                "background_detector_y",
            )
            if detector_point is not None:
                detector_point_source = "background_detector_fields"
        if (
            detector_point is None
            and display_point is not None
            and callable(background_display_to_native_detector_coords)
        ):
            detector_point = _finite_tuple_pair(
                background_display_to_native_detector_coords(
                    float(display_point[0]),
                    float(display_point[1]),
                )
            )
            if detector_point is not None:
                detector_point_source = "display_point_inverse_projection"

    def _mark_detector_anchor_stale() -> dict[str, object]:
        for stale_key in (
            "x",
            "y",
            "detector_x",
            "detector_y",
            "background_detector_x",
            "background_detector_y",
        ):
            normalized.pop(stale_key, None)
        normalized["stale_caked_fields"] = True
        return normalized

    anchor_refresh_needed = False
    anchor_caked_point = sim_replay_caked_point if sim_detector_replay is not None else caked_point
    if anchor_caked_point is not None:
        anchor_refresh_needed = detector_point is None
        if detector_point is not None and callable(native_detector_coords_to_caked_display_coords):
            anchor_caked = _finite_tuple_pair(
                native_detector_coords_to_caked_display_coords(
                    float(detector_point[0]),
                    float(detector_point[1]),
                )
            )
            if anchor_caked is None:
                anchor_refresh_needed = True
            else:
                anchor_error = float(
                    np.hypot(
                        float(anchor_caked[0]) - float(anchor_caked_point[0]),
                        float(anchor_caked[1]) - float(anchor_caked_point[1]),
                    )
                )
                if anchor_error > float(stale_caked_tolerance_px):
                    anchor_refresh_needed = True

    if anchor_refresh_needed:
        if not (
            anchor_caked_point is not None
            and callable(caked_angles_to_background_display_coords)
            and callable(background_display_to_native_detector_coords)
        ):
            return _mark_detector_anchor_stale()
        mapped_display_point = _finite_tuple_pair(
            caked_angles_to_background_display_coords(
                float(anchor_caked_point[0]),
                float(anchor_caked_point[1]),
            )
        )
        mapped_detector_point = (
            _finite_tuple_pair(
                background_display_to_native_detector_coords(
                    float(mapped_display_point[0]),
                    float(mapped_display_point[1]),
                )
            )
            if mapped_display_point is not None
            else None
        )
        if mapped_display_point is None or mapped_detector_point is None:
            return _mark_detector_anchor_stale()
        if callable(native_detector_coords_to_caked_display_coords):
            roundtrip_caked = _finite_tuple_pair(
                native_detector_coords_to_caked_display_coords(
                    float(mapped_detector_point[0]),
                    float(mapped_detector_point[1]),
                )
            )
            if roundtrip_caked is None:
                return _mark_detector_anchor_stale()
            closure_error = float(
                np.hypot(
                    float(roundtrip_caked[0]) - float(anchor_caked_point[0]),
                    float(roundtrip_caked[1]) - float(anchor_caked_point[1]),
                )
            )
            if closure_error > float(stale_caked_tolerance_px):
                return _mark_detector_anchor_stale()
        detector_point = (
            float(mapped_detector_point[0]),
            float(mapped_detector_point[1]),
        )
        detector_point_source = "caked_inverse_projection_refresh"
        display_point = (
            float(mapped_display_point[0]),
            float(mapped_display_point[1]),
        )

    if detector_point is None:
        return normalized

    detector_col = float(detector_point[0])
    detector_row = float(detector_point[1])
    normalized["detector_x"] = float(detector_col)
    normalized["detector_y"] = float(detector_row)
    if sim_detector_replay is None:
        normalized["background_detector_x"] = float(detector_col)
        normalized["background_detector_y"] = float(detector_row)
        normalized["background_detector_input_frame"] = "native_detector"
        normalized["background_detector_frame_provenance"] = str(detector_point_source)
    else:
        normalized["sim_detector_anchor_x"] = float(detector_col)
        normalized["sim_detector_anchor_y"] = float(detector_row)
        normalized["sim_detector_frame_provenance"] = str(detector_point_source)
        for stale_key in (
            "background_detector_x",
            "background_detector_y",
            "background_detector_input_frame",
            "background_detector_frame_provenance",
        ):
            normalized.pop(stale_key, None)
    normalized["native_col"] = float(detector_col)
    normalized["native_row"] = float(detector_row)
    normalized["sim_native_x"] = float(detector_col)
    normalized["sim_native_y"] = float(detector_row)

    projected_display_point = None
    if callable(native_detector_coords_to_detector_display_coords):
        try:
            projected_display_point = _finite_tuple_pair(
                native_detector_coords_to_detector_display_coords(
                    float(detector_col),
                    float(detector_row),
                )
            )
        except Exception:
            projected_display_point = None
    if projected_display_point is not None:
        display_point = (
            float(projected_display_point[0]),
            float(projected_display_point[1]),
        )
    elif len(shape) >= 2:
        try:
            display_point = rotate_point_for_display(
                float(detector_col),
                float(detector_row),
                shape,
                int(display_rotate_k),
            )
        except Exception:
            display_point = None
    if display_point is None and raw_detector_display is not None:
        display_point = (
            float(raw_detector_display[0]),
            float(raw_detector_display[1]),
        )
    if display_point is not None:
        normalized["x"] = float(display_point[0])
        normalized["y"] = float(display_point[1])
        normalized["display_col"] = float(display_point[0])
        normalized["display_row"] = float(display_point[1])
        normalized["sim_col_raw"] = float(display_point[0])
        normalized["sim_row_raw"] = float(display_point[1])
        normalized["sim_col"] = float(display_point[0])
        normalized["sim_row"] = float(display_point[1])
        if sim_detector_replay is not None:
            normalized["sim_detector_display_col"] = float(display_point[0])
            normalized["sim_detector_display_row"] = float(display_point[1])
    elif sim_detector_replay is not None:
        for stale_key in (
            "x",
            "y",
            "display_col",
            "display_row",
            "sim_col_raw",
            "sim_row_raw",
            "sim_col",
            "sim_row",
            "sim_detector_display_col",
            "sim_detector_display_row",
        ):
            normalized.pop(stale_key, None)

    recomputed_caked = None
    if callable(native_detector_coords_to_caked_display_coords):
        recomputed_caked = _finite_tuple_pair(
            native_detector_coords_to_caked_display_coords(
                float(detector_col),
                float(detector_row),
            )
        )
    output_caked = (
        saved_background_caked_point if caked_background_is_authoritative else recomputed_caked
    )
    if output_caked is not None:
        if caked_background_is_authoritative and callable(
            caked_angles_to_background_display_coords
        ):
            projected_output_display = _finite_tuple_pair(
                caked_angles_to_background_display_coords(
                    float(output_caked[0]),
                    float(output_caked[1]),
                )
            )
            if projected_output_display is not None:
                normalized["x"] = float(projected_output_display[0])
                normalized["y"] = float(projected_output_display[1])
                normalized["display_col"] = float(projected_output_display[0])
                normalized["display_row"] = float(projected_output_display[1])
        normalized["background_two_theta_deg"] = float(output_caked[0])
        normalized["background_phi_deg"] = float(output_caked[1])
        normalized["caked_x"] = float(output_caked[0])
        normalized["caked_y"] = float(output_caked[1])
        normalized["raw_caked_x"] = float(output_caked[0])
        normalized["raw_caked_y"] = float(output_caked[1])
        normalized["two_theta_deg"] = float(output_caked[0])
        normalized["phi_deg"] = float(output_caked[1])

    if geometry_manual_live_caked_trace_enabled():
        geometry_manual_trace_live_caked_visual_source_event(
            "caked_to_detector_projection",
            selected_candidate=current_projected_sim_entry,
            placement_entry=normalized,
            saved_entries=[normalized],
        )
    return normalized


def geometry_manual_preview_color(
    sigma_px: object,
    *,
    good_sigma_px: float = DEFAULT_PREVIEW_GOOD_SIGMA_PX,
    bad_sigma_px: float = DEFAULT_PREVIEW_BAD_SIGMA_PX,
) -> str:
    """Return a green-to-red preview color for one manual-pick uncertainty."""

    try:
        sigma_val = float(sigma_px)
    except Exception:
        sigma_val = float("nan")
    if not np.isfinite(sigma_val):
        sigma_val = float(bad_sigma_px)

    good_val = max(1.0e-3, float(good_sigma_px))
    bad_val = max(good_val + 1.0e-3, float(bad_sigma_px))
    ratio = (sigma_val - good_val) / (bad_val - good_val)
    ratio = min(max(float(ratio), 0.0), 1.0)

    return _geometry_manual_preview_gradient_color(float(ratio))


def _geometry_manual_preview_gradient_color(ratio: float) -> str:
    """Return the shared green-yellow-red preview gradient for one normalized ratio."""

    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.5:
        local_ratio = ratio / 0.5
        start_rgb = (0x2E, 0xCC, 0x71)
        end_rgb = (0xF1, 0xC4, 0x0F)
    else:
        local_ratio = (ratio - 0.5) / 0.5
        start_rgb = (0xF1, 0xC4, 0x0F)
        end_rgb = (0xE7, 0x4C, 0x3C)

    rgb = tuple(
        int(round((1.0 - local_ratio) * start + local_ratio * end))
        for start, end in zip(start_rgb, end_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def geometry_manual_preview_confidence_color(
    confidence: object,
    *,
    good_confidence: float = 1.0,
    bad_confidence: float = 0.25,
) -> str:
    """Return a red-to-green preview color for one predicted match confidence."""

    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = float("nan")
    if not np.isfinite(confidence_val):
        confidence_val = float(bad_confidence)

    bad_val = max(0.0, float(bad_confidence))
    good_val = max(bad_val + 1.0e-6, float(good_confidence))
    ratio = (good_val - confidence_val) / (good_val - bad_val)
    return _geometry_manual_preview_gradient_color(float(ratio))


def geometry_manual_preview_quality_label(
    sigma_px: object,
    *,
    good_sigma_px: float = DEFAULT_PREVIEW_GOOD_SIGMA_PX,
    bad_sigma_px: float = DEFAULT_PREVIEW_BAD_SIGMA_PX,
) -> str:
    """Return a coarse quality label for one manual-pick uncertainty."""

    try:
        sigma_val = float(sigma_px)
    except Exception:
        sigma_val = float("nan")
    if not np.isfinite(sigma_val):
        return "bad"

    good_val = max(1.0e-3, float(good_sigma_px))
    bad_val = max(good_val + 1.0e-3, float(bad_sigma_px))
    ratio = (sigma_val - good_val) / (bad_val - good_val)
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.2:
        return "good"
    if ratio >= 0.75:
        return "bad"
    return "warning"


def geometry_manual_preview_confidence_quality_label(
    confidence: object,
    *,
    good_confidence: float = 1.0,
    bad_confidence: float = 0.25,
) -> str:
    """Return a coarse quality label for one predicted match confidence."""

    try:
        confidence_val = float(confidence)
    except Exception:
        confidence_val = float("nan")
    if not np.isfinite(confidence_val):
        return "bad"

    bad_val = max(0.0, float(bad_confidence))
    good_val = max(bad_val + 1.0e-6, float(good_confidence))
    ratio = (good_val - confidence_val) / (good_val - bad_val)
    ratio = min(max(float(ratio), 0.0), 1.0)
    if ratio <= 0.2:
        return "good"
    if ratio >= 0.75:
        return "bad"
    return "warning"


def geometry_manual_preview_match_confidence(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    allow_cache_build: bool = True,
    use_caked_space: bool,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[[float, Sequence[float] | None], float] | None = None,
) -> float:
    """Return the predicted match confidence for one chosen manual-placement peak."""

    if not isinstance(candidate, dict):
        return float("nan")

    state = cache_data if isinstance(cache_data, dict) else {}
    allow_build = bool(allow_cache_build)
    if isinstance(state, dict):
        metadata = state.get("cache_metadata")
        if bool(state.get("manual_no_build_cache", False)):
            allow_build = False
        if isinstance(metadata, Mapping) and metadata.get("allow_cache_build") is False:
            allow_build = False
    if not state and allow_build and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = {}
        if isinstance(built_state, dict):
            state = built_state

    match_cfg = dict(state.get("match_config", {})) if isinstance(state, dict) else {}
    background_context = state.get("background_context") if isinstance(state, dict) else None
    if not isinstance(background_context, dict) or not bool(
        background_context.get("img_valid", False)
    ):
        return float("nan")

    axis_to_image_index = (
        caked_axis_to_image_index
        if caked_axis_to_image_index_fn is None
        else caked_axis_to_image_index_fn
    )

    try:
        sim_col_local = float(candidate.get("sim_col_local", candidate.get("sim_col", np.nan)))
        sim_row_local = float(candidate.get("sim_row_local", candidate.get("sim_row", np.nan)))
    except Exception:
        return float("nan")

    if use_caked_space:
        radial_axis_arr = np.asarray(radial_axis, dtype=float)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float)
        peak_col_local = float(axis_to_image_index(float(peak_col), radial_axis_arr))
        peak_row_local = float(axis_to_image_index(float(peak_row), azimuth_axis_arr))
        if not np.isfinite(sim_col_local):
            sim_col_local = float(
                axis_to_image_index(
                    float(candidate.get("sim_col", np.nan)),
                    radial_axis_arr,
                )
            )
        if not np.isfinite(sim_row_local):
            sim_row_local = float(
                axis_to_image_index(
                    float(candidate.get("sim_row", np.nan)),
                    azimuth_axis_arr,
                )
            )
    else:
        peak_col_local = float(peak_col)
        peak_row_local = float(peak_row)

    if not (
        np.isfinite(sim_col_local)
        and np.isfinite(sim_row_local)
        and np.isfinite(peak_col_local)
        and np.isfinite(peak_row_local)
    ):
        return float("nan")

    candidate_labels = np.asarray(
        background_context.get("candidate_labels", []),
        dtype=np.int32,
    )
    peakness = np.asarray(background_context.get("peakness", []), dtype=float)
    fine = np.asarray(background_context.get("fine", []), dtype=float)
    height = int(
        background_context.get(
            "height",
            candidate_labels.shape[0] if candidate_labels.ndim == 2 else 0,
        )
    )
    width = int(
        background_context.get(
            "width",
            candidate_labels.shape[1] if candidate_labels.ndim == 2 else 0,
        )
    )
    if height <= 0 or width <= 0:
        return float("nan")

    peak_row_px = int(np.clip(round(peak_row_local), 0, max(height - 1, 0)))
    peak_col_px = int(np.clip(round(peak_col_local), 0, max(width - 1, 0)))
    summit_id = _background_candidate_summit_id_near_pixel(
        candidate_labels,
        peakness,
        peak_row_px,
        peak_col_px,
        radius_px=max(
            1,
            int(round(0.5 * float(match_cfg.get("local_max_size_px", 5)))),
        ),
    )
    if summit_id <= 0:
        return float("nan")

    summit_record = None
    for record in background_context.get("summit_records", ()):
        if int(record.get("summit_id", -1)) == summit_id:
            summit_record = dict(record)
            break
    if summit_record is None:
        return float("nan")

    peak_row_seed = int(
        np.clip(
            round(float(summit_record.get("row", peak_row_local))),
            0,
            max(height - 1, 0),
        )
    )
    peak_col_seed = int(
        np.clip(
            round(float(summit_record.get("col", peak_col_local))),
            0,
            max(width - 1, 0),
        )
    )
    center_col_local, center_row_local = _background_refine_peak_center(
        peakness,
        fine,
        peak_row_seed,
        peak_col_seed,
    )
    prom_sigma = float(summit_record.get("prominence_sigma", np.nan))
    if not (
        np.isfinite(center_col_local) and np.isfinite(center_row_local) and np.isfinite(prom_sigma)
    ):
        return float("nan")

    dist_px = float(
        np.hypot(
            float(center_col_local) - float(sim_col_local),
            float(center_row_local) - float(sim_row_local),
        )
    )
    return float(max(0.0, prom_sigma) / (1.0 + max(0.0, dist_px)))


def normalize_geometry_manual_pair_entry(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Normalize one saved manual geometry-pair entry."""

    if not isinstance(entry, dict):
        return None
    try:
        x_val = float(entry.get("x", np.nan))
        y_val = float(entry.get("y", np.nan))
    except Exception:
        return None
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return None

    normalized_hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
    label = str(entry.get("label", "")) if entry.get("label") is not None else ""
    if not label and normalized_hkl is not None:
        label = f"{normalized_hkl[0]},{normalized_hkl[1]},{normalized_hkl[2]}"

    normalized: dict[str, object] = {
        "x": float(x_val),
        "y": float(y_val),
        "label": label,
    }
    if normalized_hkl is not None:
        normalized["hkl"] = normalized_hkl

    for bool_key in (
        "background_qr_set_reference",
        "manual_background_reference",
        "geometry_fit_disabled",
    ):
        if bool_key in entry:
            normalized[bool_key] = bool(entry.get(bool_key, False))

    for x_key, y_key in (
        ("background_two_theta_deg", "background_phi_deg"),
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
    ):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            caked_x_val = float(raw_x_local) if raw_x_local is not None else float("nan")
        except Exception:
            caked_x_val = float("nan")
        try:
            caked_y_val = float(raw_y_local) if raw_y_local is not None else float("nan")
        except Exception:
            caked_y_val = float("nan")
        if np.isfinite(caked_x_val) and np.isfinite(caked_y_val):
            normalized["background_two_theta_deg"] = float(caked_x_val)
            normalized["background_phi_deg"] = float(caked_y_val)
            break

    raw_group_key = entry.get("q_group_key")
    if isinstance(raw_group_key, tuple):
        normalized["q_group_key"] = raw_group_key
    elif isinstance(raw_group_key, list):
        normalized["q_group_key"] = tuple(raw_group_key)

    for key in (
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "source_branch_index",
        "source_peak_index",
    ):
        if key not in entry:
            continue
        try:
            normalized[key] = int(entry.get(key))  # type: ignore[arg-type]
        except Exception:
            pass

    if entry.get("source_reflection_namespace") is not None:
        normalized["source_reflection_namespace"] = str(entry.get("source_reflection_namespace"))
    if "source_reflection_is_full" in entry:
        normalized["source_reflection_is_full"] = bool(
            entry.get("source_reflection_is_full", False)
        )

    if entry.get("source_label") is not None:
        if bool(entry.get("background_qr_set_reference", False)):
            normalized["source_label"] = str(entry.get("source_label"))
        else:
            normalized["source_label"] = normalize_bragg_qr_source_label(
                str(entry.get("source_label"))
            )
    elif normalized.get("q_group_key") is not None:
        normalized["source_label"] = _geometry_manual_q_group_source_label(
            normalized.get("q_group_key")
        )
    elif not bool(entry.get("background_qr_set_reference", False)):
        normalized["source_label"] = "primary"
    for text_key in ("phase_label", "structure_role"):
        raw_text = entry.get(text_key)
        if raw_text is not None and str(raw_text).strip():
            normalized[text_key] = str(raw_text)
    if entry.get("background_qr_set_reference_label_source") is not None:
        normalized["background_qr_set_reference_label_source"] = str(
            entry.get("background_qr_set_reference_label_source")
        )
    if "stale_caked_fields" in entry:
        normalized["stale_caked_fields"] = bool(entry.get("stale_caked_fields", False))
    if entry.get("manual_background_input_frame") is not None:
        normalized["manual_background_input_frame"] = normalize_geometry_point_frame(
            entry.get("manual_background_input_frame")
        )
    if entry.get("manual_background_input_origin") is not None:
        normalized["manual_background_input_origin"] = str(
            entry.get("manual_background_input_origin")
        )
    if entry.get("manual_geometry_run_id") is not None:
        normalized["manual_geometry_run_id"] = str(entry.get("manual_geometry_run_id"))
    if entry.get("manual_trace_version") is not None:
        normalized["manual_trace_version"] = str(entry.get("manual_trace_version"))
    for key in ("branch_id", "branch_source", "selection_reason"):
        if entry.get(key) is not None:
            normalized[key] = str(entry.get(key))
    if entry.get("best_sample_index") is not None:
        try:
            normalized["best_sample_index"] = int(entry.get("best_sample_index"))
        except Exception:
            pass
    if entry.get("mosaic_weight") is not None:
        try:
            mosaic_weight = float(entry.get("mosaic_weight"))
        except Exception:
            mosaic_weight = float("nan")
        if np.isfinite(mosaic_weight):
            normalized["mosaic_weight"] = float(mosaic_weight)
    rank_key = entry.get("mosaic_top_rank_key")
    if isinstance(rank_key, (tuple, list)):
        normalized_rank: list[object] = []
        for item in rank_key:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, (int, np.integer)):
                normalized_rank.append(int(item))
                continue
            if isinstance(item, (float, np.floating)):
                item_float = float(item)
                if np.isfinite(item_float):
                    normalized_rank.append(item_float)
                continue
            if isinstance(item, str):
                normalized_rank.append(str(item))
        if normalized_rank:
            normalized["mosaic_top_rank_key"] = tuple(normalized_rank)

    raw_x = entry.get("raw_x")
    raw_y = entry.get("raw_y")
    try:
        raw_x_val = float(raw_x) if raw_x is not None else float("nan")
    except Exception:
        raw_x_val = float("nan")
    try:
        raw_y_val = float(raw_y) if raw_y is not None else float("nan")
    except Exception:
        raw_y_val = float("nan")
    if np.isfinite(raw_x_val) and np.isfinite(raw_y_val):
        normalized["raw_x"] = float(raw_x_val)
        normalized["raw_y"] = float(raw_y_val)

    for x_key, y_key in (
        ("detector_x", "detector_y"),
        ("background_detector_x", "background_detector_y"),
    ):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            detector_x_val = float(raw_x_local) if raw_x_local is not None else float("nan")
        except Exception:
            detector_x_val = float("nan")
        try:
            detector_y_val = float(raw_y_local) if raw_y_local is not None else float("nan")
        except Exception:
            detector_y_val = float("nan")
        if np.isfinite(detector_x_val) and np.isfinite(detector_y_val):
            normalized[x_key] = float(detector_x_val)
            normalized[y_key] = float(detector_y_val)

    for x_key, y_key in (("caked_x", "caked_y"), ("raw_caked_x", "raw_caked_y")):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            caked_x_val = float(raw_x_local) if raw_x_local is not None else float("nan")
        except Exception:
            caked_x_val = float("nan")
        try:
            caked_y_val = float(raw_y_local) if raw_y_local is not None else float("nan")
        except Exception:
            caked_y_val = float("nan")
        if np.isfinite(caked_x_val) and np.isfinite(caked_y_val):
            normalized[x_key] = float(caked_x_val)
            normalized[y_key] = float(caked_y_val)

    for x_key, y_key in (
        ("refined_sim_x", "refined_sim_y"),
        ("refined_sim_native_x", "refined_sim_native_y"),
        ("refined_sim_caked_x", "refined_sim_caked_y"),
        ("sim_detector_x", "sim_detector_y"),
        ("obs_detector_x", "obs_detector_y"),
    ):
        raw_x_local = entry.get(x_key)
        raw_y_local = entry.get(y_key)
        try:
            refined_x_val = float(raw_x_local) if raw_x_local is not None else float("nan")
        except Exception:
            refined_x_val = float("nan")
        try:
            refined_y_val = float(raw_y_local) if raw_y_local is not None else float("nan")
        except Exception:
            refined_y_val = float("nan")
        if np.isfinite(refined_x_val) and np.isfinite(refined_y_val):
            normalized[x_key] = float(refined_x_val)
            normalized[y_key] = float(refined_y_val)

    for tuple_key in (
        "raw_detector_display_px",
        "raw_detector_native_px",
        "geometry_detector_display_px",
        "geometry_detector_native_px",
        "raw_caked_deg",
        "geometry_caked_deg",
        "sim_nominal_detector_display_px",
        "sim_nominal_detector_native_px",
        "sim_nominal_caked_deg",
        "sim_visual_detector_display_px",
        "sim_visual_detector_native_px",
        "sim_visual_detector_native_existing",
        "sim_visual_caked_deg",
        "sim_refined_detector_display_px",
        "sim_refined_detector_native_px",
        "sim_refined_caked_deg",
        "sim_visual_deg",
        "sim_caked",
    ):
        raw_point = entry.get(tuple_key)
        if not isinstance(raw_point, (list, tuple, np.ndarray)) or len(raw_point) < 2:
            continue
        try:
            point = (float(raw_point[0]), float(raw_point[1]))
        except Exception:
            continue
        if np.isfinite(point[0]) and np.isfinite(point[1]):
            normalized[tuple_key] = (float(point[0]), float(point[1]))

    for float_key in (
        "sim_refinement_delta_detector_px",
        "sim_refinement_delta_caked_deg",
    ):
        try:
            numeric_value = float(entry.get(float_key, np.nan))
        except Exception:
            continue
        if np.isfinite(numeric_value):
            normalized[float_key] = float(numeric_value)

    for text_key in (
        "sim_refinement_source",
        "sim_refinement_status",
        "sim_visual_source",
        "manual_background_input_origin",
        "sim_visual_detector_display_source",
        "sim_visual_detector_native_source",
        "sim_visual_detector_native_unavailable_reason",
        "sim_visual_caked_source",
        "raw_detector_native_source",
        "geometry_detector_native_source",
        "sim_refined_caked_projection_callback",
        "sim_refined_caked_projection_status",
        "background_qr_set_reference_label_source",
    ):
        raw_text = entry.get(text_key)
        if raw_text is not None and str(raw_text).strip():
            normalized[text_key] = str(raw_text)
    if "sim_refined_caked_projection_real_callback" in entry:
        normalized["sim_refined_caked_projection_real_callback"] = bool(
            entry.get("sim_refined_caked_projection_real_callback", False)
        )

    placement_error_value = entry.get("placement_error_px")
    if placement_error_value is None and np.isfinite(raw_x_val) and np.isfinite(raw_y_val):
        placement_error_value = geometry_manual_position_error_px(
            float(raw_x_val),
            float(raw_y_val),
            float(x_val),
            float(y_val),
        )
    try:
        placement_error_px = (
            float(placement_error_value) if placement_error_value is not None else float("nan")
        )
    except Exception:
        placement_error_px = float("nan")
    if np.isfinite(placement_error_px):
        normalized["placement_error_px"] = max(0.0, float(placement_error_px))

    sigma_value = (
        entry.get("sigma_px")
        if entry.get("sigma_px") is not None
        else entry.get("position_sigma_px", entry.get("measurement_sigma_px"))
    )
    if sigma_value is None and np.isfinite(placement_error_px):
        sigma_value = geometry_manual_position_sigma_px(
            float(placement_error_px),
            floor_px=sigma_floor_px,
        )
    try:
        sigma_px = float(sigma_value) if sigma_value is not None else float("nan")
    except Exception:
        sigma_px = float("nan")
    if np.isfinite(sigma_px) and sigma_px > 0.0:
        normalized["sigma_px"] = float(sigma_px)

    _canonicalize_manual_entry_branch_fields(
        normalized,
        allow_legacy_peak_fallback=False,
        preserve_legacy_peak_when_unresolved=True,
    )

    return normalized


def geometry_manual_pairs_for_index(
    index: int,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> list[dict[str, object]]:
    """Return normalized saved manual geometry pairs for one background index."""

    try:
        key = int(index)
    except Exception:
        return []
    raw_entries = pairs_by_background.get(key, [])
    normalized_entries: list[dict[str, object]] = []
    for raw_entry in raw_entries:
        normalized = normalize_geometry_manual_pair_entry(
            raw_entry,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if normalized is not None:
            normalized_entries.append(normalized)
    return normalized_entries


def set_geometry_manual_pairs_for_index(
    index: int,
    entries: Sequence[dict[str, object]] | None,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> list[dict[str, object]]:
    """Replace one background's saved manual geometry-pair list."""

    try:
        key = int(index)
    except Exception:
        return []

    normalized_entries: list[dict[str, object]] = []
    for raw_entry in entries or []:
        normalized = normalize_geometry_manual_pair_entry(
            raw_entry,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if normalized is not None:
            normalized_entries.append(normalized)

    if normalized_entries:
        pairs_by_background[key] = normalized_entries
    else:
        pairs_by_background.pop(key, None)
    return list(normalized_entries)


def geometry_manual_entry_has_finite_caked_coordinates(entry: object) -> bool:
    """Return whether one manual entry already carries usable caked anchors."""

    if not isinstance(entry, Mapping):
        return False
    return (
        _geometry_manual_finite_point(
            entry,
            (
                ("background_two_theta_deg", "background_phi_deg"),
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
            ),
        )
        is not None
    )


def geometry_manual_entry_has_detector_coordinate_anchor(entry: object) -> bool:
    """Return whether one manual entry has detector pixels for caked backfill."""

    if not isinstance(entry, Mapping):
        return False
    if (
        _geometry_manual_finite_point(
            entry,
            (
                ("refined_detector_native_col", "refined_detector_native_row"),
                ("background_detector_x", "background_detector_y"),
                ("detector_x", "detector_y"),
                ("native_col", "native_row"),
                ("refined_detector_display_col", "refined_detector_display_row"),
                ("x", "y"),
                ("display_col", "display_row"),
                ("detector_seed_col", "detector_seed_row"),
                ("raw_x", "raw_y"),
            ),
        )
        is not None
    ):
        return True
    return any(
        _geometry_manual_tuple_point(entry, key) is not None
        for key in (
            "geometry_detector_native_px",
            "raw_detector_native_px",
            "geometry_detector_display_px",
            "raw_detector_display_px",
        )
    )


def geometry_manual_entry_needs_caked_coordinate_backfill(entry: object) -> bool:
    """Return whether one manual entry should get caked anchors from pixels."""

    return (
        isinstance(entry, Mapping)
        and not geometry_manual_entry_has_finite_caked_coordinates(entry)
        and geometry_manual_entry_has_detector_coordinate_anchor(entry)
    )


def _iter_geometry_manual_pair_row_entries(
    rows: Sequence[object] | Mapping[str, object] | None,
):
    if rows is None:
        return
    if isinstance(rows, Mapping):
        row_iterable: Sequence[object] = (rows,)
    elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        row_iterable = rows
    else:
        return
    for raw_row in row_iterable:
        if not isinstance(raw_row, Mapping):
            continue
        raw_entries = raw_row.get("entries")
        if isinstance(raw_entries, Sequence) and not isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            for entry in raw_entries:
                if isinstance(entry, Mapping):
                    yield entry
        else:
            yield raw_row


def geometry_manual_pairs_rows_missing_caked_backfill_count(
    rows: Sequence[object] | Mapping[str, object] | None,
) -> int:
    """Count manual entries that need detector-pixel to caked-coordinate backfill."""

    return sum(
        1
        for entry in _iter_geometry_manual_pair_row_entries(rows)
        if geometry_manual_entry_needs_caked_coordinate_backfill(entry)
    )


def geometry_manual_pairs_rows_need_caked_backfill(
    rows: Sequence[object] | Mapping[str, object] | None,
) -> bool:
    """Return whether any saved manual entry needs caked-coordinate backfill."""

    return geometry_manual_pairs_rows_missing_caked_backfill_count(rows) > 0


def geometry_manual_backfill_missing_caked_coordinates(
    entries: Sequence[Mapping[str, object]] | None,
    *,
    background_display_to_native_detector_coords: Callable[[float, float], object] | None,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object] | None,
) -> tuple[list[dict[str, object]], int]:
    """Fill legacy detector-origin manual pairs with caked 2theta/phi anchors."""

    output: list[dict[str, object]] = []
    changed_count = 0
    for raw_entry in entries or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        before_caked = _geometry_manual_finite_point(
            entry,
            (
                ("background_two_theta_deg", "background_phi_deg"),
                ("caked_x", "caked_y"),
            ),
        )
        if before_caked is None:
            entry.setdefault("manual_background_input_origin", "detector")
        if _geometry_manual_tuple_point(entry, "geometry_detector_display_px") is None:
            display_point = _geometry_manual_finite_point(
                entry,
                (
                    ("refined_detector_display_col", "refined_detector_display_row"),
                    ("x", "y"),
                ),
            )
            if display_point is not None:
                entry["geometry_detector_display_px"] = (
                    float(display_point[0]),
                    float(display_point[1]),
                )
        if _geometry_manual_tuple_point(entry, "raw_detector_display_px") is None:
            raw_display_point = _geometry_manual_finite_point(
                entry,
                (
                    ("detector_seed_col", "detector_seed_row"),
                    ("raw_x", "raw_y"),
                ),
            )
            if raw_display_point is None:
                raw_display_point = _geometry_manual_tuple_point(
                    entry,
                    "geometry_detector_display_px",
                )
            if raw_display_point is not None:
                entry["raw_detector_display_px"] = (
                    float(raw_display_point[0]),
                    float(raw_display_point[1]),
                )
        if _geometry_manual_tuple_point(entry, "geometry_detector_native_px") is None:
            native_point = _geometry_manual_finite_point(
                entry,
                (
                    ("refined_detector_native_col", "refined_detector_native_row"),
                    ("background_detector_x", "background_detector_y"),
                    ("detector_x", "detector_y"),
                ),
            )
            if native_point is not None:
                entry["geometry_detector_native_px"] = (
                    float(native_point[0]),
                    float(native_point[1]),
                )
        _geometry_manual_apply_detector_origin_conversion_ledger(
            entry,
            background_display_to_native_detector_coords=(
                background_display_to_native_detector_coords
            ),
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        after_caked = _geometry_manual_finite_point(
            entry,
            (
                ("background_two_theta_deg", "background_phi_deg"),
                ("caked_x", "caked_y"),
            ),
        )
        if before_caked is None and after_caked is not None:
            changed_count += 1
        output.append(entry)
    return output, int(changed_count)


def geometry_manual_pair_group_count(
    index: int,
    *,
    pairs_by_background: dict[int, list[dict[str, object]]],
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> int:
    """Return how many distinct Qr/Qz groups are saved for one background."""

    group_keys = {
        entry.get("q_group_key")
        for entry in geometry_manual_pairs_for_index(
            index,
            pairs_by_background=pairs_by_background,
            normalize_hkl_key=normalize_hkl_key,
            sigma_floor_px=sigma_floor_px,
        )
        if entry.get("q_group_key") is not None
    }
    return int(len(group_keys))


def peak_maximum_near_in_image(
    image: np.ndarray | None,
    col: float,
    row: float,
    *,
    search_radius: int = 5,
    display_extent: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Return the brightest local pixel near ``(col, row)`` in display coordinates."""

    if image is None:
        return float(col), float(row)
    try:
        image_arr = np.asarray(image, dtype=float)
    except Exception:
        return float(col), float(row)
    if image_arr.ndim < 2 or image_arr.size == 0:
        return float(col), float(row)

    height = int(image_arr.shape[0])
    width = int(image_arr.shape[1])

    extent: tuple[float, float, float, float] | None = None
    if display_extent is not None:
        try:
            normalized_extent = tuple(float(value) for value in display_extent)
        except Exception:
            normalized_extent = ()
        if len(normalized_extent) == 4 and all(np.isfinite(value) for value in normalized_extent):
            extent = normalized_extent

    def _display_axis_to_index(
        value: float, *, axis_start: float, axis_end: float, size: int
    ) -> float:
        span = float(axis_end) - float(axis_start)
        if int(size) <= 0 or not np.isfinite(span) or abs(span) <= 1.0e-12:
            return float(value)
        return float(
            np.clip(
                ((float(value) - float(axis_start)) / span) * float(size) - 0.5,
                0.0,
                float(size - 1),
            )
        )

    def _index_to_display_axis(
        index_value: float, *, axis_start: float, axis_end: float, size: int
    ) -> float:
        span = float(axis_end) - float(axis_start)
        if int(size) <= 0 or not np.isfinite(span) or abs(span) <= 1.0e-12:
            return float(index_value)
        return float(axis_start) + ((float(index_value) + 0.5) / float(size)) * span

    if extent is None:
        c = int(round(float(col)))
        r = int(round(float(row)))
    else:
        c = int(
            round(
                _display_axis_to_index(
                    float(col),
                    axis_start=float(extent[0]),
                    axis_end=float(extent[1]),
                    size=width,
                )
            )
        )
        r = int(
            round(
                _display_axis_to_index(
                    float(row),
                    axis_start=float(extent[2]),
                    axis_end=float(extent[3]),
                    size=height,
                )
            )
        )
    r0 = max(0, r - int(search_radius))
    r1 = min(height, r + int(search_radius) + 1)
    c0 = max(0, c - int(search_radius))
    c1 = min(width, c + int(search_radius) + 1)

    window = image_arr[r0:r1, c0:c1]
    if window.size == 0 or not np.isfinite(window).any():
        return float(col), float(row)

    max_idx = int(np.nanargmax(window))
    win_r, win_c = np.unravel_index(max_idx, window.shape)
    peak_col = float(c0 + win_c)
    peak_row = float(r0 + win_r)
    if extent is None:
        return peak_col, peak_row
    return (
        _index_to_display_axis(
            peak_col,
            axis_start=float(extent[0]),
            axis_end=float(extent[1]),
            size=width,
        ),
        _index_to_display_axis(
            peak_row,
            axis_start=float(extent[2]),
            axis_end=float(extent[3]),
            size=height,
        ),
    )


def geometry_manual_apply_refined_simulated_override(
    entry: dict[str, object] | None,
    resolved_source_entry: dict[str, object] | None,
    *,
    prefer_caked_display: bool | None = None,
) -> dict[str, object] | None:
    """Overlay one saved refined-simulation position onto a resolved source entry."""

    result = dict(resolved_source_entry) if isinstance(resolved_source_entry, dict) else {}
    if not isinstance(entry, dict):
        return result or None

    if not result:
        for key in (
            "hkl",
            "label",
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "source_label",
            "q_group_key",
        ):
            if key in entry:
                result[key] = entry.get(key)

    def _pair(x_key: str, y_key: str) -> tuple[float, float] | None:
        try:
            x_val = float(entry.get(x_key, np.nan))
            y_val = float(entry.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            return None
        return float(x_val), float(y_val)

    refined_raw = _pair("refined_sim_x", "refined_sim_y")
    refined_native = _pair("refined_sim_native_x", "refined_sim_native_y")
    refined_caked = _pair("refined_sim_caked_x", "refined_sim_caked_y")

    if refined_raw is not None:
        result["refined_sim_x"] = float(refined_raw[0])
        result["refined_sim_y"] = float(refined_raw[1])
    if refined_native is not None:
        result["refined_sim_native_x"] = float(refined_native[0])
        result["refined_sim_native_y"] = float(refined_native[1])
    if refined_caked is not None:
        result["refined_sim_caked_x"] = float(refined_caked[0])
        result["refined_sim_caked_y"] = float(refined_caked[1])
    elif refined_raw is not None:
        result.pop("refined_sim_caked_x", None)
        result.pop("refined_sim_caked_y", None)

    if prefer_caked_display is None:
        use_caked_display = False
        try:
            display_col = float(result.get("display_col", np.nan))
            display_row = float(result.get("display_row", np.nan))
            sim_col = float(result.get("sim_col", np.nan))
            sim_row = float(result.get("sim_row", np.nan))
            caked_x = float(result.get("caked_x", np.nan))
            caked_y = float(result.get("caked_y", np.nan))
            if (
                np.isfinite(display_col)
                and np.isfinite(display_row)
                and np.isfinite(caked_x)
                and np.isfinite(caked_y)
            ):
                use_caked_display = (
                    abs(float(display_col) - float(caked_x)) <= 1.0e-9
                    and abs(float(display_row) - float(caked_y)) <= 1.0e-9
                )
            elif (
                np.isfinite(sim_col)
                and np.isfinite(sim_row)
                and np.isfinite(caked_x)
                and np.isfinite(caked_y)
            ):
                use_caked_display = (
                    abs(float(sim_col) - float(caked_x)) <= 1.0e-9
                    and abs(float(sim_row) - float(caked_y)) <= 1.0e-9
                )
        except Exception:
            use_caked_display = False
    else:
        use_caked_display = bool(prefer_caked_display)

    if refined_raw is not None:
        result["sim_col_raw"] = float(refined_raw[0])
        result["sim_row_raw"] = float(refined_raw[1])
        result["sim_col"] = float(refined_raw[0])
        result["sim_row"] = float(refined_raw[1])
        if not use_caked_display:
            result["display_col"] = float(refined_raw[0])
            result["display_row"] = float(refined_raw[1])
            result["x"] = float(refined_raw[0])
            result["y"] = float(refined_raw[1])

    if refined_native is not None:
        result["native_col"] = float(refined_native[0])
        result["native_row"] = float(refined_native[1])
        result["sim_native_x"] = float(refined_native[0])
        result["sim_native_y"] = float(refined_native[1])
    elif refined_raw is not None or refined_caked is not None:
        for stale_native_key in (
            "native_col",
            "native_row",
            "sim_native_x",
            "sim_native_y",
            "detector_x",
            "detector_y",
            "background_detector_x",
            "background_detector_y",
            "simulated_detector_x",
            "simulated_detector_y",
        ):
            result.pop(stale_native_key, None)

    if refined_caked is not None:
        result["caked_x"] = float(refined_caked[0])
        result["caked_y"] = float(refined_caked[1])
        result["raw_caked_x"] = float(refined_caked[0])
        result["raw_caked_y"] = float(refined_caked[1])
        result["two_theta_deg"] = float(refined_caked[0])
        result["phi_deg"] = float(refined_caked[1])
        if use_caked_display:
            result["display_col"] = float(refined_caked[0])
            result["display_row"] = float(refined_caked[1])

    return result or None


def update_geometry_manual_peak_record_cache(
    peak_records: Sequence[object] | None,
    *,
    source_key: tuple[object, ...] | None,
    source_entry: Mapping[str, object] | None = None,
    refined_caked: tuple[float, float] | None = None,
    refined_native: tuple[float, float] | None = None,
    refined_display: tuple[float, float] | None = None,
    peak_positions: list[tuple[float, float]] | None = None,
    peak_overlay_cache: dict[str, object] | None = None,
) -> bool:
    """Update cached simulated peak records after manual caked-space refinement."""

    normalized_source_entry = dict(source_entry) if isinstance(source_entry, Mapping) else None
    if normalized_source_entry is None and not (
        isinstance(source_key, tuple) and len(source_key) >= 2
    ):
        return False

    normalized_source_key: tuple[object, ...] | None = None
    if isinstance(source_key, tuple) and len(source_key) >= 2:
        if len(source_key) >= 3 and str(source_key[0]) in {"source", "source_branch"}:
            normalized_source_key = tuple(source_key)
        else:
            try:
                normalized_source_key = ("source", int(source_key[0]), int(source_key[1]))
            except Exception:
                normalized_source_key = None

    try:
        refined_tth = float(refined_caked[0]) if refined_caked is not None else float("nan")
        refined_phi = float(refined_caked[1]) if refined_caked is not None else float("nan")
    except Exception:
        refined_tth = float("nan")
        refined_phi = float("nan")
    try:
        refined_native_col = (
            float(refined_native[0]) if refined_native is not None else float("nan")
        )
        refined_native_row = (
            float(refined_native[1]) if refined_native is not None else float("nan")
        )
    except Exception:
        refined_native_col = float("nan")
        refined_native_row = float("nan")
    try:
        refined_display_col = (
            float(refined_display[0]) if refined_display is not None else float("nan")
        )
        refined_display_row = (
            float(refined_display[1]) if refined_display is not None else float("nan")
        )
    except Exception:
        refined_display_col = float("nan")
        refined_display_row = float("nan")

    def _collect_matching_indexes(records: Sequence[object] | None) -> list[int]:
        if records is None:
            return []
        if normalized_source_entry is not None:
            identity_matches: list[int] = []
            for idx, raw_record in enumerate(records):
                if not isinstance(raw_record, Mapping):
                    continue
                if geometry_manual_source_entries_share_identity(
                    normalized_source_entry,
                    raw_record,
                ):
                    identity_matches.append(int(idx))
            if len(identity_matches) > 0:
                return identity_matches

        if normalized_source_key is None:
            return []
        key_matches: list[int] = []
        for idx, raw_record in enumerate(records):
            if not isinstance(raw_record, Mapping):
                continue
            if geometry_manual_source_key_matches_entry(
                normalized_source_key,
                raw_record,
            ):
                key_matches.append(int(idx))
        return key_matches

    def _apply_updates(record: dict[str, object]) -> bool:
        updated_local = False
        if np.isfinite(refined_tth) and np.isfinite(refined_phi):
            record["two_theta_deg"] = float(refined_tth)
            record["phi_deg"] = float(refined_phi)
            record["caked_x"] = float(refined_tth)
            record["caked_y"] = float(refined_phi)
            record["raw_caked_x"] = float(refined_tth)
            record["raw_caked_y"] = float(refined_phi)
            updated_local = True
        if np.isfinite(refined_native_col) and np.isfinite(refined_native_row):
            record["native_col"] = float(refined_native_col)
            record["native_row"] = float(refined_native_row)
            record["sim_native_x"] = float(refined_native_col)
            record["sim_native_y"] = float(refined_native_row)
            updated_local = True
        if np.isfinite(refined_display_col) and np.isfinite(refined_display_row):
            record["display_col"] = float(refined_display_col)
            record["display_row"] = float(refined_display_row)
            record["sim_col"] = float(refined_display_col)
            record["sim_row"] = float(refined_display_row)
            record["sim_col_raw"] = float(refined_display_col)
            record["sim_row_raw"] = float(refined_display_row)
            updated_local = True
        return updated_local

    matched_peak_indexes: list[int] = _collect_matching_indexes(peak_records)
    updated = False
    if len(matched_peak_indexes) == 1:
        raw_record = peak_records[matched_peak_indexes[0]]
        if isinstance(raw_record, dict) and _apply_updates(raw_record):
            updated = True

    if (
        len(matched_peak_indexes) == 1
        and isinstance(peak_positions, list)
        and np.isfinite(refined_display_col)
        and np.isfinite(refined_display_row)
    ):
        idx = int(matched_peak_indexes[0])
        if 0 <= idx < len(peak_positions):
            peak_positions[idx] = (
                float(refined_display_col),
                float(refined_display_row),
            )
            updated = True

    if isinstance(peak_overlay_cache, dict):
        overlay_records = peak_overlay_cache.get("records")
        matched_overlay_indexes: list[int] = _collect_matching_indexes(overlay_records)
        if isinstance(overlay_records, list):
            if len(matched_overlay_indexes) == 1:
                raw_record = overlay_records[matched_overlay_indexes[0]]
                if isinstance(raw_record, dict) and _apply_updates(raw_record):
                    updated = True
        if (
            len(matched_overlay_indexes) == 1
            and np.isfinite(refined_display_col)
            and np.isfinite(refined_display_row)
        ):
            overlay_positions = peak_overlay_cache.get("positions")
            if isinstance(overlay_positions, list):
                idx = int(matched_overlay_indexes[0])
                if 0 <= idx < len(overlay_positions):
                    overlay_positions[idx] = (
                        float(refined_display_col),
                        float(refined_display_row),
                    )
                    updated = True
        if updated:
            peak_overlay_cache["click_spatial_index"] = None

    return bool(updated)


def caked_axis_to_image_index(
    value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one caked-axis coordinate in degrees to a floating image index."""

    if axis_values is None or not np.isfinite(value):
        return float("nan")
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis_arr.size <= 0:
        return float("nan")
    finite_idx = np.flatnonzero(np.isfinite(axis_arr))
    if finite_idx.size <= 0:
        return float("nan")
    if finite_idx.size == 1:
        return float(finite_idx[0])
    axis_used = axis_arr[finite_idx]
    idx_used = finite_idx.astype(float)
    if axis_used[0] > axis_used[-1]:
        axis_used = axis_used[::-1]
        idx_used = idx_used[::-1]
    return float(np.interp(float(value), axis_used, idx_used))


def caked_image_index_to_axis(
    index_value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one floating caked image index back to axis-space degrees."""

    if axis_values is None or not np.isfinite(index_value):
        return float("nan")
    axis_arr = np.asarray(axis_values, dtype=float).reshape(-1)
    if axis_arr.size <= 0:
        return float("nan")
    finite_idx = np.flatnonzero(np.isfinite(axis_arr))
    if finite_idx.size <= 0:
        return float("nan")
    if finite_idx.size == 1:
        return float(axis_arr[finite_idx[0]])
    axis_used = axis_arr[finite_idx]
    idx_used = finite_idx.astype(float)
    if axis_used[0] > axis_used[-1]:
        axis_used = axis_used[::-1]
        idx_used = idx_used[::-1]
    return float(np.interp(float(index_value), idx_used, axis_used))


def caked_display_coords_to_angles(
    col: float,
    row: float,
    *,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
    image_shape: Sequence[int] | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
) -> tuple[float, float] | None:
    """Convert one caked display point into caked 2theta/phi coordinates."""

    if radial_axis is None or azimuth_axis is None:
        return None
    try:
        radial_axis_arr = np.asarray(radial_axis, dtype=float).reshape(-1)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float).reshape(-1)
    except Exception:
        return None
    if radial_axis_arr.size <= 0 or azimuth_axis_arr.size <= 0:
        return None
    try:
        col_val = float(col)
        row_val = float(row)
    except Exception:
        return None
    if not (np.isfinite(col_val) and np.isfinite(row_val)):
        return None

    def _axis_contains(value: float, axis_values: np.ndarray) -> bool:
        finite_axis = axis_values[np.isfinite(axis_values)]
        if finite_axis.size <= 0:
            return False
        axis_min = float(np.nanmin(finite_axis))
        axis_max = float(np.nanmax(finite_axis))
        return bool(min(axis_min, axis_max) <= float(value) <= max(axis_min, axis_max))

    def _image_index_candidate(value: float, axis_values: np.ndarray, size: int) -> float:
        if (
            int(size) > 0
            and 0.0 <= float(value) <= float(size - 1)
            and not _axis_contains(
                float(value),
                axis_values,
            )
        ):
            return float(value)
        return float(caked_axis_to_image_index_fn(float(value), axis_values))

    try:
        shape = tuple(int(v) for v in (image_shape or ())[:2])
    except Exception:
        shape = ()
    height = int(shape[0]) if len(shape) >= 2 and int(shape[0]) > 0 else int(azimuth_axis_arr.size)
    width = int(shape[1]) if len(shape) >= 2 and int(shape[1]) > 0 else int(radial_axis_arr.size)

    col_index = _image_index_candidate(col_val, radial_axis_arr, width)
    row_index = _image_index_candidate(row_val, azimuth_axis_arr, height)
    if not (np.isfinite(col_index) and np.isfinite(row_index)):
        return None
    two_theta = caked_image_index_to_axis_fn(float(col_index), radial_axis_arr)
    phi = caked_image_index_to_axis_fn(float(row_index), azimuth_axis_arr)
    if not (np.isfinite(two_theta) and np.isfinite(phi)):
        return None
    return float(two_theta), float(phi)


def refine_profile_peak_index(
    profile: Sequence[float] | None,
    seed_index: float,
) -> float:
    """Return one subpixel 1D peak center focused on the top of a local profile."""

    arr = np.asarray(profile, dtype=float).reshape(-1)
    if arr.size <= 0:
        return float(seed_index)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(seed_index)
    arr = np.where(finite, arr, np.nan)
    baseline = float(np.nanpercentile(arr, 35.0))
    weights = np.clip(arr - baseline, 0.0, None)
    if not np.any(weights > 0.0):
        floor = float(np.nanmin(arr))
        weights = np.clip(arr - floor, 0.0, None)
    if not np.any(weights > 0.0):
        return float(np.nanargmax(arr))

    peak_idx = int(np.nanargmax(weights))
    half_window = 2
    lo = max(0, peak_idx - half_window)
    hi = min(arr.size, peak_idx + half_window + 1)
    local_weights = np.asarray(weights[lo:hi], dtype=float)
    local_idx = np.arange(lo, hi, dtype=float)
    if local_weights.size <= 0 or not np.any(local_weights > 0.0):
        return float(peak_idx)
    crest_mask = local_weights >= 0.5 * float(np.max(local_weights))
    if np.any(crest_mask):
        local_weights = local_weights[crest_mask]
        local_idx = local_idx[crest_mask]
    total = float(np.sum(local_weights))
    if not np.isfinite(total) or total <= 0.0:
        return float(peak_idx)
    return float(np.sum(local_weights * local_idx) / total)


def refine_caked_peak_center(
    image: np.ndarray | None,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
    two_theta_deg: float,
    phi_deg: float,
    *,
    tth_window_deg: float | None = None,
    phi_window_deg: float | None = None,
    default_tth_window_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    default_phi_window_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
) -> tuple[float, float]:
    """Refine one caked click to the crest of the local 2theta/phi ridge."""

    if image is None:
        return float(two_theta_deg), float(phi_deg)
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.size <= 0:
        return float(two_theta_deg), float(phi_deg)
    if radial_axis is None or azimuth_axis is None:
        return float(two_theta_deg), float(phi_deg)

    col_seed = caked_axis_to_image_index(float(two_theta_deg), radial_axis)
    row_seed = caked_axis_to_image_index(float(phi_deg), azimuth_axis)
    if not (np.isfinite(col_seed) and np.isfinite(row_seed)):
        return float(two_theta_deg), float(phi_deg)

    tth_window = float(default_tth_window_deg if tth_window_deg is None else tth_window_deg)
    phi_window = float(default_phi_window_deg if phi_window_deg is None else phi_window_deg)
    col_min = caked_axis_to_image_index(float(two_theta_deg) - tth_window, radial_axis)
    col_max = caked_axis_to_image_index(float(two_theta_deg) + tth_window, radial_axis)
    row_min = caked_axis_to_image_index(float(phi_deg) - phi_window, azimuth_axis)
    row_max = caked_axis_to_image_index(float(phi_deg) + phi_window, azimuth_axis)
    if not all(np.isfinite(v) for v in (col_min, col_max, row_min, row_max)):
        return float(two_theta_deg), float(phi_deg)

    c0 = max(0, int(np.floor(min(col_min, col_max))))
    c1 = min(int(img.shape[1]), int(np.ceil(max(col_min, col_max))) + 1)
    r0 = max(0, int(np.floor(min(row_min, row_max))))
    r1 = min(int(img.shape[0]), int(np.ceil(max(row_min, row_max))) + 1)
    if c0 >= c1 or r0 >= r1:
        return float(two_theta_deg), float(phi_deg)

    patch = np.asarray(img[r0:r1, c0:c1], dtype=float)
    if patch.size <= 0 or not np.isfinite(patch).any():
        return float(two_theta_deg), float(phi_deg)
    baseline = float(np.nanpercentile(patch, 35.0))
    signal = np.clip(patch - baseline, 0.0, None)
    if not np.any(signal > 0.0):
        signal = np.clip(patch - float(np.nanmin(patch)), 0.0, None)
    if not np.any(signal > 0.0):
        return float(two_theta_deg), float(phi_deg)

    col_local = float(col_seed - c0)
    row_local = float(row_seed - r0)
    row_band = max(1, min(6, int(round(0.10 * signal.shape[0]))))
    col_band = max(1, min(6, int(round(0.10 * signal.shape[1]))))
    for _ in range(2):
        row_center = int(np.clip(round(row_local), 0, max(signal.shape[0] - 1, 0)))
        rr0 = max(0, row_center - row_band)
        rr1 = min(signal.shape[0], row_center + row_band + 1)
        radial_profile = np.nansum(signal[rr0:rr1, :], axis=0)
        col_local = refine_profile_peak_index(radial_profile, col_local)

        col_center = int(np.clip(round(col_local), 0, max(signal.shape[1] - 1, 0)))
        cc0 = max(0, col_center - col_band)
        cc1 = min(signal.shape[1], col_center + col_band + 1)
        az_profile = np.nansum(signal[:, cc0:cc1], axis=1)
        row_local = refine_profile_peak_index(az_profile, row_local)

    refined_col = float(c0 + col_local)
    refined_row = float(r0 + row_local)
    return (
        float(caked_image_index_to_axis(refined_col, radial_axis)),
        float(caked_image_index_to_axis(refined_row, azimuth_axis)),
    )


def geometry_manual_candidate_source_key(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> tuple[object, ...] | None:
    """Return a stable lookup key for one manual-pick candidate or match."""

    if not isinstance(entry, dict):
        return None
    source_branch_index, _source_branch_source, _source_branch_reason = resolve_canonical_branch(
        _geometry_manual_identity_resolution_entry(entry),
        allow_legacy_peak_fallback=True,
    )
    if source_branch_index in {0, 1}:
        source_anchor = (
            _coerce_nonnegative_index(entry.get("source_reflection_index"))
            if _entry_has_explicit_trusted_reflection_identity(entry)
            else None
        )
        if source_anchor is None:
            source_anchor = _coerce_nonnegative_index(entry.get("source_table_index"))
        if source_anchor is not None:
            return (
                "source_branch",
                int(source_anchor),
                int(source_branch_index),
            )
    try:
        return (
            "source",
            int(entry.get("source_table_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        pass
    normalized_hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
    if normalized_hkl is not None:
        return ("hkl",) + tuple(int(v) for v in normalized_hkl)
    label = str(entry.get("label", "")).strip()
    if label:
        return ("label", label)
    return None


def _geometry_manual_identity_resolution_entry(
    entry: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if not (isinstance(entry, Mapping) and bool(entry.get("stale_caked_fields"))):
        return entry

    identity_entry = dict(entry)
    for key in (
        "display_col",
        "display_row",
        "background_two_theta_deg",
        "background_phi_deg",
        "caked_x",
        "caked_y",
        "raw_caked_x",
        "raw_caked_y",
        "simulated_two_theta_deg",
        "simulated_phi_deg",
        "two_theta_deg",
        "phi_deg",
    ):
        identity_entry.pop(key, None)
    return identity_entry


def _geometry_manual_entry_has_stale_caked_fields(
    entry: Mapping[str, object] | None,
) -> bool:
    return bool(isinstance(entry, Mapping) and bool(entry.get("stale_caked_fields")))


def _geometry_manual_entry_has_explicit_branch_identity(
    entry: Mapping[str, object] | None,
) -> bool:
    branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=False,
    )
    return branch_idx in {0, 1}


def _geometry_manual_source_row_key(
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


def _geometry_manual_entry_normalized_hkl(
    entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> tuple[int, int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    return normalize_hkl_key(entry.get("hkl", entry.get("label")))


def _geometry_manual_entry_label(entry: Mapping[str, object] | None) -> str:
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get("label", "") or "").strip()


def _geometry_manual_nonbranch_identity_fields_match(
    left_entry: Mapping[str, object] | None,
    right_entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> bool:
    left_hkl = _geometry_manual_entry_normalized_hkl(
        left_entry,
        normalize_hkl_key=normalize_hkl_key,
    )
    right_hkl = _geometry_manual_entry_normalized_hkl(
        right_entry,
        normalize_hkl_key=normalize_hkl_key,
    )
    if left_hkl is not None and right_hkl is not None:
        return left_hkl == right_hkl

    left_label = _geometry_manual_entry_label(left_entry)
    right_label = _geometry_manual_entry_label(right_entry)
    if left_label and right_label:
        return left_label == right_label

    return False


def _geometry_manual_lookup_bucket_entries(
    bucket: object,
) -> list[dict[str, object]]:
    if isinstance(bucket, Mapping):
        return [dict(bucket)]
    if isinstance(bucket, Sequence) and not isinstance(
        bucket,
        (str, bytes, bytearray),
    ):
        return [dict(entry) for entry in bucket if isinstance(entry, Mapping)]
    return []


def _geometry_manual_copy_lookup(
    lookup: Mapping[tuple[object, ...], object] | None,
) -> GeometryManualLookupMap:
    copied: GeometryManualLookupMap = {}
    for raw_key, raw_bucket in (lookup or {}).items():
        if not isinstance(raw_key, tuple):
            continue
        bucket_entries = _geometry_manual_lookup_bucket_entries(raw_bucket)
        if not bucket_entries:
            continue
        if len(bucket_entries) == 1:
            copied[tuple(raw_key)] = dict(bucket_entries[0])
        else:
            copied[tuple(raw_key)] = [dict(entry) for entry in bucket_entries]
    return copied


def _geometry_manual_add_lookup_entry(
    lookup: GeometryManualLookupMap,
    key: tuple[object, ...],
    entry: Mapping[str, object] | None,
) -> None:
    if not isinstance(entry, Mapping):
        return

    bucket_entries = _geometry_manual_lookup_bucket_entries(lookup.get(key))
    bucket_entries.append(dict(entry))
    if len(bucket_entries) == 1:
        lookup[key] = dict(bucket_entries[0])
    else:
        lookup[key] = [dict(bucket_entry) for bucket_entry in bucket_entries]


def _geometry_manual_flatten_lookup_entries(
    lookup: Mapping[tuple[object, ...], object] | None,
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for raw_bucket in (lookup or {}).values():
        flattened.extend(_geometry_manual_lookup_bucket_entries(raw_bucket))
    return flattened


def _geometry_manual_finite_point(
    entry: Mapping[str, object] | None,
    key_pairs: Sequence[tuple[str, str]],
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    for key_x, key_y in key_pairs:
        try:
            col = float(entry.get(key_x, np.nan))
            row = float(entry.get(key_y, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    return None


def _geometry_manual_identity_value_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return True


def _geometry_manual_caked_qr_projection_key(
    entry: Mapping[str, object] | None,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    has_source_identity = any(
        _geometry_manual_identity_value_present(entry.get(key))
        for key in (
            "source_table_index",
            "source_row_index",
            "source_reflection_index",
        )
    )
    has_branch_identity = any(
        _geometry_manual_identity_value_present(entry.get(key))
        for key in (
            "source_branch_index",
            "source_ray_id",
            "branch_id",
        )
    )
    if not (has_source_identity and has_branch_identity):
        return None
    return tuple(entry.get(field) for field in _GEOMETRY_MANUAL_CAKED_QR_ID_FIELDS)


def _geometry_manual_caked_qr_projection_source(
    entry: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(entry, Mapping):
        return None
    if _geometry_manual_caked_qr_projection_key(entry) is None:
        return None

    copied: dict[str, object] = {}
    exact_keys = {
        "label",
        "hkl",
        "q_group_key",
        "branch_id",
        "branch_source",
        "best_sample_index",
        "mosaic_weight",
        "mosaic_top_rank_key",
        "selection_reason",
        "selection_scope",
        "selected_q_group_key",
        "native_col",
        "native_row",
        "sim_col_raw",
        "sim_row_raw",
        "sim_col",
        "sim_row",
        "sim_native_x",
        "sim_native_y",
        "sim_detector_anchor_x",
        "sim_detector_anchor_y",
        "sim_detector_display_col",
        "sim_detector_display_row",
        "sim_detector_frame_provenance",
        "refined_sim_native_x",
        "refined_sim_native_y",
        "refined_sim_caked_x",
        "refined_sim_caked_y",
        "sim_visual_detector_display_px",
        "sim_visual_detector_native_px",
        "sim_visual_detector_native_source",
        "sim_visual_caked_deg",
        "sim_visual_caked_source",
        "sim_refined_caked_deg",
        "sim_visual_deg",
        "sim_visual_source",
        "sim_caked",
        "sim_refinement_source",
        "sim_refinement_status",
        "qr",
        "qz",
    }
    for raw_key, value in entry.items():
        key = str(raw_key)
        if (
            key in exact_keys
            or key.startswith("source_")
            or key.startswith("ray_")
            or key.startswith("reflection_")
        ):
            copied[key] = value
    visual_caked = _geometry_manual_tuple_point(copied, "sim_visual_deg")
    if visual_caked is None:
        visual_caked = _geometry_manual_tuple_point(copied, "sim_caked")
    if visual_caked is None:
        visual_caked = _geometry_manual_finite_point(
            copied,
            (("refined_sim_caked_x", "refined_sim_caked_y"),),
        )
    if visual_caked is not None:
        copied["sim_visual_deg"] = (float(visual_caked[0]), float(visual_caked[1]))
        copied["sim_caked"] = (float(visual_caked[0]), float(visual_caked[1]))
        copied.setdefault("sim_visual_source", "sim_visual_caked_deg")
    for alias_key in _GEOMETRY_MANUAL_QR_SIM_ALIAS_FIELDS:
        copied.pop(alias_key, None)

    if (
        _geometry_manual_finite_point(
            copied,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
            ),
        )
        is None
    ):
        detector_anchor = _geometry_manual_finite_point(
            copied,
            (("sim_detector_anchor_x", "sim_detector_anchor_y"),),
        )
        if detector_anchor is None:
            return None
        copied["native_col"] = float(detector_anchor[0])
        copied["native_row"] = float(detector_anchor[1])
        copied["sim_native_x"] = float(detector_anchor[0])
        copied["sim_native_y"] = float(detector_anchor[1])
    return copied


def _geometry_manual_caked_qr_projection_entry(
    entry: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(entry, Mapping):
        return None
    if _geometry_manual_caked_qr_projection_key(entry) is None:
        return None
    if (
        _geometry_manual_finite_point(
            entry,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
            ),
        )
        is None
    ):
        return None
    caked_point = _geometry_manual_finite_point(
        entry,
        (
            ("caked_x", "caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
    )
    if caked_point is None:
        display_frame = str(
            entry.get("display_frame", entry.get("current_view_frame", ""))
            or ""
        ).strip().lower()
        if display_frame == "caked_display":
            caked_point = _geometry_manual_finite_point(
                entry,
                (("display_col", "display_row"),),
            )
    if caked_point is None:
        return None

    caked_x = float(caked_point[0])
    caked_y = float(caked_point[1])
    caked_pair = (caked_x, caked_y)
    visual_caked = _geometry_manual_tuple_point(entry, "sim_visual_caked_deg")
    if visual_caked is None:
        visual_caked = _geometry_manual_tuple_point(entry, "sim_visual_deg")
    if visual_caked is None:
        visual_caked = _geometry_manual_tuple_point(entry, "sim_caked")
    if visual_caked is None:
        visual_pair = caked_pair
    else:
        visual_pair = (float(visual_caked[0]), float(visual_caked[1]))
    projected = dict(entry)
    projected["_caked_qr_projection_cache"] = True
    projected["display_frame"] = "caked_display"
    projected["current_view_frame"] = "caked_display"
    projected["display_col"] = caked_x
    projected["display_row"] = caked_y
    projected["caked_x"] = caked_x
    projected["caked_y"] = caked_y
    projected["two_theta_deg"] = caked_x
    projected["phi_deg"] = caked_y
    projected["simulated_two_theta_deg"] = caked_x
    projected["simulated_phi_deg"] = caked_y
    projected["refined_sim_caked_x"] = caked_x
    projected["refined_sim_caked_y"] = caked_y
    projected["sim_caked_display"] = caked_pair
    projected["sim_refined_caked_deg"] = caked_pair
    projected["sim_visual_caked_deg"] = visual_pair
    projected["sim_visual_deg"] = visual_pair
    projected["sim_caked"] = visual_pair
    projected.setdefault("sim_visual_source", "sim_visual_caked_deg")
    return projected


def _geometry_manual_build_caked_qr_projection_cache(
    rows: Sequence[dict[str, object]] | None,
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None,
    build_grouped_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ],
    filter_active_rows: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None,
) -> tuple[
    list[dict[str, object]],
    dict[tuple[object, ...], list[dict[str, object]]],
    GeometryManualLookupMap,
]:
    _ = build_simulated_lookup
    if not callable(project_peaks_to_current_view):
        return [], {}, {}

    source_rows = [
        source
        for raw_entry in rows or ()
        if (source := _geometry_manual_caked_qr_projection_source(raw_entry)) is not None
    ]
    if not source_rows:
        return [], {}, {}
    try:
        projected_rows = project_peaks_to_current_view(source_rows)
    except Exception:
        return [], {}, {}

    projected_entries = [
        entry
        for raw_entry in projected_rows or ()
        if (entry := _geometry_manual_caked_qr_projection_entry(raw_entry)) is not None
    ]
    if callable(filter_active_rows):
        try:
            active_entries = [
                dict(entry)
                for entry in (filter_active_rows(projected_entries) or ())
                if isinstance(entry, Mapping)
            ]
        except Exception:
            active_entries = []
    else:
        active_entries = [dict(entry) for entry in projected_entries]
    if not active_entries:
        return [], {}, {}

    grouped_entries = _geometry_manual_collapse_q_group_representatives(active_entries)
    if not grouped_entries:
        return [], {}, {}

    try:
        grouped_candidates = build_grouped_candidates(grouped_entries)
    except Exception:
        grouped_candidates = {}
    grouped_candidates = {
        key: [dict(entry) for entry in entries if isinstance(entry, Mapping)]
        for key, entries in (grouped_candidates or {}).items()
        if isinstance(key, tuple)
    }

    projection_lookup: GeometryManualLookupMap = {}
    for entry in active_entries:
        key = _geometry_manual_caked_qr_projection_key(entry)
        if key is not None:
            _geometry_manual_add_lookup_entry(projection_lookup, key, entry)
    return [dict(entry) for entry in active_entries], grouped_candidates, projection_lookup


def _geometry_manual_lookup_caked_qr_projection_entry(
    lookup: Mapping[tuple[object, ...], object] | None,
    entry: Mapping[str, object] | None,
) -> dict[str, object] | None:
    key = _geometry_manual_caked_qr_projection_key(entry)
    direct_candidates = (
        _geometry_manual_lookup_bucket_entries(lookup.get(key))
        if isinstance(lookup, Mapping) and key is not None
        else []
    )
    if len(direct_candidates) == 1:
        return dict(direct_candidates[0])
    if direct_candidates:
        resolved_index = _geometry_manual_resolve_identity_candidate_index(entry, direct_candidates)
        if resolved_index is not None and 0 <= int(resolved_index) < len(direct_candidates):
            return dict(direct_candidates[int(resolved_index)])

    all_candidates = _geometry_manual_flatten_lookup_entries(lookup)
    resolved_index = _geometry_manual_resolve_identity_candidate_index(entry, all_candidates)
    if resolved_index is not None and 0 <= int(resolved_index) < len(all_candidates):
        return dict(all_candidates[int(resolved_index)])
    return None


def _geometry_manual_current_sim_caked_projection_entry(
    entry: Mapping[str, object] | None,
    *,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ),
) -> dict[str, object] | None:
    cached_projection = _geometry_manual_caked_qr_projection_entry(entry)
    if cached_projection is not None and bool(cached_projection.get("_caked_qr_projection_cache")):
        return cached_projection

    source_entry = _geometry_manual_caked_qr_projection_source(entry)
    native_point = _geometry_manual_tuple_point(
        source_entry,
        "sim_visual_detector_native_px",
    )
    if native_point is None:
        native_point = _geometry_manual_finite_point(
            source_entry,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
            ),
        )
    if native_point is None or not callable(native_detector_coords_to_caked_display_coords):
        return None

    try:
        caked_point = native_detector_coords_to_caked_display_coords(
            float(native_point[0]),
            float(native_point[1]),
        )
    except Exception:
        caked_point = None
    if (
        not isinstance(caked_point, tuple)
        or len(caked_point) < 2
        or not np.isfinite(float(caked_point[0]))
        or not np.isfinite(float(caked_point[1]))
    ):
        return None

    projected_entry = dict(source_entry)
    projected_entry["display_col"] = float(caked_point[0])
    projected_entry["display_row"] = float(caked_point[1])
    projected_entry["caked_x"] = float(caked_point[0])
    projected_entry["caked_y"] = float(caked_point[1])
    projected_entry["two_theta_deg"] = float(caked_point[0])
    projected_entry["phi_deg"] = float(caked_point[1])
    projected_entry["_caked_qr_projection_cache"] = True
    projected_entry["display_frame"] = "caked_display"
    projected_entry["current_view_frame"] = "caked_display"
    return _geometry_manual_caked_qr_projection_entry(projected_entry)


def resolve_sim_detector_replay_from_caked_projection(
    saved_entry: Mapping[str, object] | None,
    current_projected_sim_entry: Mapping[str, object] | None,
    *,
    caked_angles_to_background_display_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ),
    background_display_to_native_detector_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ),
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ),
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None,
    stale_caked_tolerance_px: float = 0.5,
) -> dict[str, object] | None:
    if _geometry_manual_caked_qr_projection_source(saved_entry) is None:
        return None

    projected_entry = _geometry_manual_caked_qr_projection_entry(current_projected_sim_entry)
    current_caked_point, _current_caked_source = (
        _geometry_manual_candidate_visual_caked_sim_point(projected_entry)
    )
    if current_caked_point is None:
        current_caked_point = _geometry_manual_finite_point(
            projected_entry,
            (
                ("caked_x", "caked_y"),
                ("two_theta_deg", "phi_deg"),
            ),
        )
    if current_caked_point is None:
        return None

    def _finite_tuple_pair(
        value: object,
    ) -> tuple[float, float] | None:
        if not isinstance(value, tuple) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _closure_matches_current_projection(
        native_point: tuple[float, float] | None,
    ) -> bool:
        if native_point is None or not callable(native_detector_coords_to_caked_display_coords):
            return False
        try:
            replayed_caked_point = _finite_tuple_pair(
                native_detector_coords_to_caked_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                )
            )
        except Exception:
            replayed_caked_point = None
        return bool(
            replayed_caked_point is not None
            and np.hypot(
                float(replayed_caked_point[0]) - float(current_caked_point[0]),
                float(replayed_caked_point[1]) - float(current_caked_point[1]),
            )
            <= float(stale_caked_tolerance_px)
        )

    cached_native_anchor = _geometry_manual_finite_point(
        saved_entry,
        (("sim_detector_anchor_x", "sim_detector_anchor_y"),),
    )
    detector_native_anchor = (
        cached_native_anchor if _closure_matches_current_projection(cached_native_anchor) else None
    )
    detector_display_point = _geometry_manual_finite_point(
        saved_entry,
        (("sim_detector_display_col", "sim_detector_display_row"),),
    )
    replayed_from_reverse_lut = False

    if detector_native_anchor is None:
        if not (
            callable(caked_angles_to_background_display_coords)
            and callable(background_display_to_native_detector_coords)
        ):
            return None
        try:
            detector_display_point = _finite_tuple_pair(
                caked_angles_to_background_display_coords(
                    float(current_caked_point[0]),
                    float(current_caked_point[1]),
                )
            )
        except Exception:
            detector_display_point = None
        if detector_display_point is None:
            return None
        try:
            detector_native_anchor = _finite_tuple_pair(
                background_display_to_native_detector_coords(
                    float(detector_display_point[0]),
                    float(detector_display_point[1]),
                )
            )
        except Exception:
            detector_native_anchor = None
        if not _closure_matches_current_projection(detector_native_anchor):
            return None
        replayed_from_reverse_lut = True

    if callable(native_detector_coords_to_detector_display_coords):
        try:
            projected_detector_display = _finite_tuple_pair(
                native_detector_coords_to_detector_display_coords(
                    float(detector_native_anchor[0]),
                    float(detector_native_anchor[1]),
                )
            )
        except Exception:
            projected_detector_display = None
        if projected_detector_display is not None:
            detector_display_point = projected_detector_display
        elif not replayed_from_reverse_lut:
            detector_display_point = None
    elif not replayed_from_reverse_lut:
        detector_display_point = None

    result = {
        "sim_detector_anchor_x": float(detector_native_anchor[0]),
        "sim_detector_anchor_y": float(detector_native_anchor[1]),
        "sim_detector_frame_provenance": "sim_reverse_lut_replay_cache",
    }
    if detector_display_point is not None:
        result["sim_detector_display_col"] = float(detector_display_point[0])
        result["sim_detector_display_row"] = float(detector_display_point[1])
    return result


_GEOMETRY_MANUAL_BACKGROUND_DETECTOR_FRAMES = {
    "background",
    "background_detector",
    "background_detector_native",
    "detector_native",
    "native_detector",
}
_GEOMETRY_MANUAL_SIMULATION_NATIVE_FRAMES = {
    "simulation",
    "simulation_native",
    "sim_native",
    "native_sim",
}
_GEOMETRY_MANUAL_BACKGROUND_DISPLAY_SOURCES = {
    "background_detector",
    "native_detector",
    "native_detector_coords_to_detector_display_coords",
    "detector_fields",
    "background_detector_fields",
    "caked_inverse_projection",
}
_GEOMETRY_MANUAL_SIMULATION_DISPLAY_SOURCES = {
    "native_sim_to_display",
    "simulation_native",
    "sim_col_raw",
    "sim_reverse_lut_replay_cache",
}


def _geometry_manual_normalized_frame_token(value: object) -> str:
    return str(value or "").strip().lower()


def _geometry_manual_set_background_input_provenance(
    entry: dict[str, object],
    *,
    use_caked_space: bool,
) -> None:
    if bool(use_caked_space):
        entry["manual_background_input_origin"] = "caked"
        entry["manual_background_input_frame"] = "caked_2theta_phi"
    else:
        entry["manual_background_input_origin"] = "detector"
        entry["manual_background_input_frame"] = "detector_display"


def geometry_manual_entry_is_background_detector_frame(
    entry: Mapping[str, object] | None,
) -> bool:
    """Return whether one row explicitly belongs to the background detector frame."""

    if not isinstance(entry, Mapping):
        return False
    for key in (
        "coordinate_frame",
        "native_coordinate_frame",
        "detector_coordinate_frame",
        "display_coordinate_frame",
    ):
        frame = _geometry_manual_normalized_frame_token(entry.get(key))
        if frame in _GEOMETRY_MANUAL_BACKGROUND_DETECTOR_FRAMES:
            return True
        if frame in _GEOMETRY_MANUAL_SIMULATION_NATIVE_FRAMES:
            return False
    for key in ("detector_display_source", "projection_source", "detector_point_source"):
        source = _geometry_manual_normalized_frame_token(entry.get(key))
        if source in _GEOMETRY_MANUAL_BACKGROUND_DISPLAY_SOURCES:
            return True
        if source in _GEOMETRY_MANUAL_SIMULATION_DISPLAY_SOURCES:
            return False
    return (
        _geometry_manual_finite_point(
            entry,
            (
                ("background_detector_x", "background_detector_y"),
                ("detector_x", "detector_y"),
            ),
        )
        is not None
    )


def geometry_manual_entry_is_simulation_native_frame(
    entry: Mapping[str, object] | None,
) -> bool:
    """Return whether one row carries simulation-native detector coordinates."""

    if not isinstance(entry, Mapping):
        return False
    if geometry_manual_entry_is_background_detector_frame(entry):
        return False
    for key in (
        "coordinate_frame",
        "native_coordinate_frame",
        "detector_coordinate_frame",
    ):
        if (
            _geometry_manual_normalized_frame_token(entry.get(key))
            in _GEOMETRY_MANUAL_SIMULATION_NATIVE_FRAMES
        ):
            return True
    if (
        _geometry_manual_normalized_frame_token(entry.get("detector_display_source"))
        in _GEOMETRY_MANUAL_SIMULATION_DISPLAY_SOURCES
    ):
        return True
    has_hit_provenance = (
        entry.get("source_table_index") is not None or entry.get("source_row_index") is not None
    )
    return bool(
        entry.get("q_group_key") is not None
        and has_hit_provenance
        and _geometry_manual_finite_point(
            entry,
            (
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
            ),
        )
        is not None
    )


def _geometry_manual_entry_refined_sim_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    current_view_point = _geometry_manual_entry_explicit_current_view_display_point(entry)
    if current_view_point is not None:
        return current_view_point

    if _geometry_manual_entry_has_caked_evidence(entry):
        return None

    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_x", "refined_sim_y"),
            ("sim_col", "sim_row"),
            ("simulated_x", "simulated_y"),
        ),
    )


def _geometry_manual_entry_has_caked_evidence(
    entry: Mapping[str, object] | None,
) -> bool:
    if _geometry_manual_entry_has_stale_caked_fields(entry):
        return False
    return (
        _geometry_manual_finite_point(
            entry,
            (
                ("refined_sim_caked_x", "refined_sim_caked_y"),
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
                ("background_two_theta_deg", "background_phi_deg"),
                ("simulated_two_theta_deg", "simulated_phi_deg"),
                ("two_theta_deg", "phi_deg"),
            ),
        )
        is not None
    )


def _geometry_manual_entry_display_frame(entry: Mapping[str, object] | None) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    for key in (
        "display_frame",
        "display_coordinate_frame",
        "current_view_frame",
        "active_view_frame",
    ):
        raw_frame = entry.get(key)
        if raw_frame is None:
            continue
        frame = str(raw_frame).strip().lower()
        if frame in {
            "detector",
            "detector_display",
            "detector_display_px",
            "detector_px",
            "current_detector_display",
        }:
            return "detector"
        if frame in {
            "caked",
            "caked_display",
            "caked_2theta_phi",
            "two_theta_phi",
            "current_caked_display",
        }:
            return "caked"
    return None


def _geometry_manual_entry_display_matches_caked_point(
    entry: Mapping[str, object] | None,
    display_point: tuple[float, float] | None,
) -> bool:
    display_frame = _geometry_manual_entry_display_frame(entry)
    if display_frame == "detector":
        return False
    if display_frame == "caked":
        return True
    if display_point is None or _geometry_manual_entry_has_stale_caked_fields(entry):
        return False
    for point_keys in (
        ("refined_sim_caked_x", "refined_sim_caked_y"),
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
        ("background_two_theta_deg", "background_phi_deg"),
        ("simulated_two_theta_deg", "simulated_phi_deg"),
        ("two_theta_deg", "phi_deg"),
    ):
        caked_point = _geometry_manual_finite_point(entry, (point_keys,))
        if (
            caked_point is not None
            and abs(float(display_point[0]) - float(caked_point[0])) <= 1.0e-9
            and abs(float(display_point[1]) - float(caked_point[1])) <= 1.0e-9
        ):
            return True
    return False


def _geometry_manual_entry_caked_point(
    entry: Mapping[str, object] | None,
    *,
    allow_stale: bool = False,
) -> tuple[float, float] | None:
    if not allow_stale and _geometry_manual_entry_has_stale_caked_fields(entry):
        return None
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("caked_x", "caked_y"),
            ("background_two_theta_deg", "background_phi_deg"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("two_theta_deg", "phi_deg"),
        ),
    )


def _geometry_manual_saved_caked_background_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    return _geometry_manual_finite_point(
        entry,
        (("background_two_theta_deg", "background_phi_deg"),),
    )


def _geometry_manual_saved_background_detector_origin(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    caked_tokens = {"caked", "caked_display", "caked_2theta_phi", "two_theta_phi"}
    detector_tokens = {
        "detector",
        "detector_display",
        "detector_native",
        "native_detector",
    }
    origin = _geometry_manual_normalized_frame_token(
        entry.get("manual_background_input_origin")
    )
    frame = _geometry_manual_normalized_frame_token(
        entry.get("manual_background_input_frame")
    )
    if origin in detector_tokens:
        return True
    if origin in caked_tokens:
        return False
    if frame in caked_tokens:
        return False
    if frame in detector_tokens:
        return True
    strong_caked_background = _geometry_manual_finite_point(
        entry,
        (
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("refined_background_two_theta_deg", "refined_background_phi_deg"),
        ),
    )
    if strong_caked_background is not None:
        return False
    detector_specific = _geometry_manual_finite_point(
        entry,
        (
            ("detector_x", "detector_y"),
            ("background_detector_x", "background_detector_y"),
        ),
    )
    if detector_specific is not None:
        return True
    if _geometry_manual_saved_caked_background_display_point(entry) is not None:
        return False
    return _geometry_manual_finite_point(entry, (("x", "y"),)) is not None


def _geometry_manual_saved_background_display_point_for_view(
    entry: Mapping[str, object] | None,
    *,
    use_caked_display: bool,
    entry_display_coords: Callable[[dict[str, object] | None], tuple[float, float] | None],
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    detector_origin = bool(use_caked_display) and _geometry_manual_saved_background_detector_origin(
        entry
    )
    if detector_origin:
        detector_projected = entry_display_coords(dict(entry))
        if detector_projected is not None:
            return detector_projected
        return None
    caked_point = (
        _geometry_manual_saved_caked_background_display_point(entry)
        if bool(use_caked_display)
        else None
    )
    if caked_point is not None:
        return caked_point
    return entry_display_coords(dict(entry))


def _geometry_manual_entry_explicit_current_view_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if _geometry_manual_entry_has_stale_caked_fields(entry):
        return None
    return _geometry_manual_finite_point(
        entry,
        (("display_col", "display_row"),),
    )


def _geometry_manual_entry_detector_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    explicit_frame = _geometry_manual_entry_display_frame(entry)
    if explicit_frame == "detector":
        detector_point = _geometry_manual_finite_point(
            entry,
            (("display_col", "display_row"),),
        )
        if detector_point is not None:
            return detector_point

    caked_display_row = bool(
        isinstance(entry, Mapping)
        and (entry.get("_caked_qr_projection_cache") or explicit_frame == "caked")
    )

    def _valid_detector_point(point: tuple[float, float] | None) -> tuple[float, float] | None:
        if point is None:
            return None
        if _geometry_manual_detector_point_is_caked(entry, point):
            return None
        return point

    detector_point = _valid_detector_point(
        _geometry_manual_tuple_point(entry, "sim_refined_detector_display_px")
    )
    if detector_point is not None:
        return detector_point

    detector_point = _geometry_manual_finite_point(
        entry,
        (("display_col", "display_row"),),
    )
    if (
        detector_point is not None
        and not bool(isinstance(entry, Mapping) and entry.get("_caked_qr_projection_cache"))
        and _geometry_manual_entry_display_frame(entry) != "caked"
        and not _geometry_manual_entry_has_stale_caked_fields(entry)
        and not _geometry_manual_entry_display_matches_caked_point(entry, detector_point)
        and not _geometry_manual_detector_point_is_caked(entry, detector_point)
    ):
        return detector_point

    if caked_display_row:
        detector_point = _valid_detector_point(
            _geometry_manual_finite_point(
                entry,
                (("sim_detector_display_col", "sim_detector_display_row"),),
            )
        )
        if detector_point is not None:
            return detector_point
        return None

    detector_point = _valid_detector_point(
        _geometry_manual_finite_point(
            entry,
            (("sim_col_raw", "sim_row_raw"),),
        )
    )
    if detector_point is not None:
        return detector_point

    if not caked_display_row:
        detector_point = _valid_detector_point(
            _geometry_manual_finite_point(
                entry,
                (
                    ("x", "y"),
                    ("simulated_x", "simulated_y"),
                ),
            )
        )
        if detector_point is not None:
            return detector_point

    detector_point = _valid_detector_point(
        _geometry_manual_finite_point(
            entry,
            (("sim_col", "sim_row"),),
        )
    )
    if detector_point is None:
        return None

    branch_idx, branch_source, _branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=True,
    )
    if branch_idx in {0, 1} and branch_source in {
        "source_branch_index",
        "source_peak_index",
    }:
        return detector_point
    if _geometry_manual_entry_has_stale_caked_fields(entry):
        return None
    if _geometry_manual_entry_matching_current_view_point(entry) is not None:
        return None
    return detector_point


def _geometry_manual_entry_matching_current_view_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if _geometry_manual_entry_has_stale_caked_fields(entry):
        return None
    return _geometry_manual_finite_point(
        entry,
        (
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("background_two_theta_deg", "background_phi_deg"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("two_theta_deg", "phi_deg"),
        ),
    )


def _geometry_manual_entry_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    current_view_point = _geometry_manual_entry_explicit_current_view_display_point(entry)
    if current_view_point is not None:
        return current_view_point

    if _geometry_manual_entry_has_caked_evidence(entry):
        return None

    return _geometry_manual_entry_detector_display_point(entry)


def _geometry_manual_tuple_point(
    entry: Mapping[str, object] | None,
    key: str,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get(key)
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) < 2:  # type: ignore[arg-type]
            return None
        x_val = float(value[0])  # type: ignore[index]
        y_val = float(value[1])  # type: ignore[index]
    except Exception:
        return None
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return None
    return float(x_val), float(y_val)


def _geometry_manual_entry_current_view_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if _geometry_manual_entry_display_frame(entry) == "detector":
        detector_point = _geometry_manual_finite_point(
            entry,
            (("display_col", "display_row"),),
        )
        if detector_point is not None:
            return detector_point

    refined_point = _geometry_manual_tuple_point(entry, "sim_refined_detector_display_px")
    if refined_point is None:
        refined_point = _geometry_manual_finite_point(entry, (("refined_sim_x", "refined_sim_y"),))
    if refined_point is not None and not _geometry_manual_entry_display_matches_caked_point(
        entry,
        refined_point,
    ):
        return refined_point
    return _geometry_manual_entry_detector_display_point(entry)


def _geometry_manual_entry_active_view_point(
    entry: Mapping[str, object] | None,
    *,
    use_caked_display: bool,
) -> tuple[float, float] | None:
    if bool(use_caked_display):
        return _geometry_manual_entry_matching_current_view_point(entry)
    return _geometry_manual_entry_current_view_point(entry)


def normalize_geometry_point_frame(frame: object) -> str:
    """Return the compact frame vocabulary used by picker/provider parity."""

    raw = str(frame or "").strip().lower()
    if raw in {
        "display",
        "current_view_display",
        "caked_display",
        "detector_display",
        "fit_detector",
    }:
        return "display"
    if raw in {
        "detector_native",
        "native_detector",
        "native_detector_coords",
        "background_detector",
    }:
        return "detector_native"
    if raw in {
        "caked_2theta_phi",
        "caked",
        "two_theta_phi",
        "two_theta_deg_phi_deg",
    }:
        return "caked_2theta_phi"
    if raw in {"caked_qr_qz", "qr_qz", "q_space"}:
        return "caked_qr_qz"
    return "unknown"


def _geometry_manual_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_geometry_manual_json_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_geometry_manual_json_value(item) for item in value]
    if isinstance(value, list):
        return [_geometry_manual_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _geometry_manual_json_value(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
            if raw_value is not None
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    if float(numeric).is_integer():
        return int(numeric)
    return float(numeric)


def canonical_geometry_source_identity(
    row_or_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return a stable JSON source identity for saved manual geometry pairs."""

    if not isinstance(row_or_entry, Mapping):
        return {}
    identity: dict[str, object] = {}
    normalized_hkl = _default_normalize_hkl_key(row_or_entry.get("hkl", row_or_entry.get("label")))
    if normalized_hkl is not None:
        identity["normalized_hkl"] = [int(v) for v in normalized_hkl]

    for key in (
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "source_peak_index",
        "q_group_key",
        "source_q_group_key",
        "branch_group_key",
        "source_branch_index",
        "source_label",
        "label",
    ):
        value = row_or_entry.get(key)
        if value is None:
            continue
        json_value = _geometry_manual_json_value(value)
        if json_value is not None:
            identity[key] = json_value
    if row_or_entry.get("source_reflection_namespace") is not None:
        identity["source_reflection_namespace"] = str(
            row_or_entry.get("source_reflection_namespace")
        )
    source_reflection_is_full = row_or_entry.get("source_reflection_is_full")
    if source_reflection_is_full is not None:
        identity["source_reflection_is_full"] = bool(source_reflection_is_full)
    return identity


def _geometry_manual_saved_background_point_with_frame(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, str, str]:
    if not isinstance(entry, Mapping):
        return None, "unknown", "unknown"
    caked_point = _geometry_manual_finite_point(
        entry,
        (
            ("background_two_theta_deg", "background_phi_deg"),
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
        ),
    )
    manual_frame = str(entry.get("manual_background_input_frame", "")).strip().lower()
    manual_origin = str(entry.get("manual_background_input_origin", "")).strip().lower()
    if (
        caked_point is not None
        and (manual_frame == "caked_2theta_phi" or manual_origin == "caked")
    ):
        return caked_point, "caked_2theta_phi", "manual_picker_saved"
    point = _geometry_manual_finite_point(entry, (("x", "y"),))
    if point is not None:
        return point, "display", "manual_picker_saved"
    point = _geometry_manual_finite_point(entry, (("display_col", "display_row"),))
    if point is not None:
        return point, "display", "manual_picker_cache"
    point = _geometry_manual_finite_point(
        entry,
        (("background_detector_x", "background_detector_y"), ("detector_x", "detector_y")),
    )
    if point is not None:
        return point, "detector_native", "manual_picker_saved"
    if caked_point is not None:
        return caked_point, "caked_2theta_phi", "manual_picker_saved"
    return None, "unknown", "unknown"


def _geometry_manual_saved_simulated_point(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, str, str]:
    if not isinstance(entry, Mapping):
        return None, "unknown", "unknown"
    caked_point = _geometry_manual_finite_point(
        entry,
        (
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
    )
    manual_frame = str(entry.get("manual_background_input_frame", "")).strip().lower()
    manual_origin = str(entry.get("manual_background_input_origin", "")).strip().lower()
    if (
        caked_point is not None
        and (manual_frame == "caked_2theta_phi" or manual_origin == "caked")
    ):
        return caked_point, "caked_2theta_phi", "manual_picker_saved"
    point = _geometry_manual_finite_point(entry, (("refined_sim_x", "refined_sim_y"),))
    if point is not None:
        return point, "display", "manual_picker_saved"
    point = _geometry_manual_finite_point(entry, (("sim_col", "sim_row"),))
    if point is not None:
        return point, "display", "manual_picker_cache"
    point = _geometry_manual_finite_point(entry, (("display_col", "display_row"),))
    if point is not None:
        return point, "display", "manual_picker_cache"
    point = _geometry_manual_finite_point(
        entry,
        (("refined_sim_native_x", "refined_sim_native_y"),),
    )
    if point is not None:
        return point, "detector_native", "manual_picker_saved"
    if caked_point is not None:
        return caked_point, "caked_2theta_phi", "manual_picker_saved"
    return None, "unknown", "unknown"


def _geometry_manual_point_changed(
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
    *,
    tol: float = 1.0e-6,
) -> bool:
    if before is None and after is None:
        return False
    if before is None or after is None:
        return True
    return bool(
        abs(float(before[0]) - float(after[0])) > float(tol)
        or abs(float(before[1]) - float(after[1])) > float(tol)
    )


def build_geometry_manual_picker_truth_pairs(
    background_index: int,
    saved_entries: Sequence[Mapping[str, object]] | None,
    *,
    refresh_pair_entry: Callable[[Mapping[str, object] | None], Mapping[str, object] | None]
    | None = None,
) -> list[dict[str, object]]:
    """Build read-only manual-picker truth rows from saved manual pair entries."""

    truth_pairs: list[dict[str, object]] = []
    for pair_index, raw_entry in enumerate(saved_entries or ()):
        if not isinstance(raw_entry, Mapping):
            truth_pairs.append(
                {
                    "pair_index": int(pair_index),
                    "background_index": int(background_index),
                    "manual_pair_order_key": [int(background_index), int(pair_index)],
                    "manual_picker_truth_available": False,
                    "missing_truth_fields": ["saved_entry"],
                }
            )
            continue
        saved_entry = dict(raw_entry)
        background_point, background_frame, background_source = (
            _geometry_manual_saved_background_point_with_frame(saved_entry)
        )
        simulated_point, simulated_frame, simulated_source = _geometry_manual_saved_simulated_point(
            saved_entry
        )
        refreshed_entry: dict[str, object] | None = None
        if callable(refresh_pair_entry):
            try:
                refreshed = refresh_pair_entry(dict(saved_entry))
            except Exception:
                refreshed = None
            if isinstance(refreshed, Mapping):
                refreshed_entry = dict(refreshed)

        post_background_point = None
        post_simulated_point = None
        post_background_source = background_source
        post_simulated_source = simulated_source
        if refreshed_entry is not None:
            post_background_point, _post_bg_frame, post_background_source = (
                _geometry_manual_saved_background_point_with_frame(refreshed_entry)
            )
            post_simulated_point, _post_sim_frame, post_simulated_source = (
                _geometry_manual_saved_simulated_point(refreshed_entry)
            )

        missing_fields: list[str] = []
        if _default_normalize_hkl_key(saved_entry.get("hkl", saved_entry.get("label"))) is None:
            missing_fields.append("normalized_hkl")
        if background_point is None:
            missing_fields.append("manual_background_point")
        if simulated_point is None:
            missing_fields.append("manual_selected_simulated_point")
        source_locator_available = any(
            saved_entry.get(key) is not None
            for key in (
                "source_table_index",
                "source_reflection_index",
                "source_row_index",
                "source_peak_index",
                "source_label",
            )
        )
        if not source_locator_available:
            missing_fields.append("manual_picker_selected_source_identity")
        manual_distance = None
        if (
            background_point is not None
            and simulated_point is not None
            and normalize_geometry_point_frame(background_frame)
            == normalize_geometry_point_frame(simulated_frame)
            and normalize_geometry_point_frame(background_frame) != "unknown"
        ):
            manual_distance = float(
                np.hypot(
                    float(simulated_point[0]) - float(background_point[0]),
                    float(simulated_point[1]) - float(background_point[1]),
                )
            )

        branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
            saved_entry,
            allow_legacy_peak_fallback=True,
        )
        normalized_hkl = _default_normalize_hkl_key(
            saved_entry.get("hkl", saved_entry.get("label"))
        )
        semantic_pair_key = {
            "background_index": int(background_index),
            "q_group_key": _geometry_manual_json_value(saved_entry.get("q_group_key")),
            "branch_group_key": _geometry_manual_json_value(saved_entry.get("branch_group_key")),
            "normalized_hkl": (
                [int(v) for v in normalized_hkl] if normalized_hkl is not None else None
            ),
            "source_branch_index": int(branch_idx) if branch_idx is not None else None,
        }
        background_frame = normalize_geometry_point_frame(background_frame)
        simulated_frame = normalize_geometry_point_frame(simulated_frame)
        truth_pairs.append(
            {
                "pair_index": int(pair_index),
                "background_index": int(background_index),
                "manual_pair_order_key": [int(background_index), int(pair_index)],
                "semantic_pair_key": semantic_pair_key,
                "q_group_key": _geometry_manual_json_value(saved_entry.get("q_group_key")),
                "branch_group_key": _geometry_manual_json_value(
                    saved_entry.get("branch_group_key")
                ),
                "normalized_hkl": semantic_pair_key["normalized_hkl"],
                "source_branch_index": semantic_pair_key["source_branch_index"],
                "manual_picker_selected_source_identity_canonical": (
                    canonical_geometry_source_identity(saved_entry)
                ),
                "manual_background_point": (
                    [float(background_point[0]), float(background_point[1])]
                    if background_point is not None
                    else None
                ),
                "manual_background_frame": background_frame,
                "manual_background_point_source": background_source,
                "manual_selected_simulated_point": (
                    [float(simulated_point[0]), float(simulated_point[1])]
                    if simulated_point is not None
                    else None
                ),
                "manual_selected_simulated_frame": simulated_frame,
                "manual_simulated_point_source": simulated_source,
                "manual_selected_to_background_distance_px": manual_distance,
                "manual_picker_truth_available": not missing_fields,
                "missing_truth_fields": missing_fields,
                "manual_truth_mutated_by_refresh": bool(
                    refreshed_entry is not None
                    and (
                        _geometry_manual_point_changed(
                            background_point,
                            post_background_point,
                        )
                        or _geometry_manual_point_changed(
                            simulated_point,
                            post_simulated_point,
                        )
                    )
                ),
                "pre_refresh_manual_background_point": (
                    [float(background_point[0]), float(background_point[1])]
                    if background_point is not None
                    else None
                ),
                "post_refresh_manual_background_point": (
                    [float(post_background_point[0]), float(post_background_point[1])]
                    if post_background_point is not None
                    else None
                ),
                "pre_refresh_manual_simulated_point": (
                    [float(simulated_point[0]), float(simulated_point[1])]
                    if simulated_point is not None
                    else None
                ),
                "post_refresh_manual_simulated_point": (
                    [float(post_simulated_point[0]), float(post_simulated_point[1])]
                    if post_simulated_point is not None
                    else None
                ),
                "post_refresh_manual_background_point_source": post_background_source,
                "post_refresh_manual_simulated_point_source": post_simulated_source,
            }
        )
    return truth_pairs


def _geometry_manual_entry_native_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("sim_native_x", "sim_native_y"),
            ("native_col", "native_row"),
            ("detector_x", "detector_y"),
            ("background_detector_x", "background_detector_y"),
            ("simulated_detector_x", "simulated_detector_y"),
        ),
    )


def _geometry_manual_entry_stronger_native_hint_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("sim_native_x", "sim_native_y"),
            ("native_col", "native_row"),
            ("simulated_detector_x", "simulated_detector_y"),
        ),
    )


def _geometry_manual_entry_resolver_native_hint_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("sim_native_x", "sim_native_y"),
            ("native_col", "native_row"),
            ("simulated_detector_x", "simulated_detector_y"),
            ("detector_x", "detector_y"),
            ("background_detector_x", "background_detector_y"),
        ),
    )


def _geometry_manual_entry_saved_detector_hint_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_finite_point(
        entry,
        (
            ("detector_x", "detector_y"),
            ("background_detector_x", "background_detector_y"),
        ),
    )


def _geometry_manual_resolve_nearest_candidate_index(
    entry: Mapping[str, object] | None,
    indexed_candidates: Sequence[tuple[int, Mapping[str, object]]],
) -> int | None:
    prefer_detector_sim_for_stale_entry = _geometry_manual_entry_has_stale_caked_fields(entry)
    entry_native_hint = _geometry_manual_entry_resolver_native_hint_point(entry)
    entry_stronger_native_hint = _geometry_manual_entry_stronger_native_hint_point(entry)
    native_candidate_point_getter = _geometry_manual_entry_native_point
    if entry_stronger_native_hint is not None:
        native_candidate_point_getter = _geometry_manual_entry_resolver_native_hint_point

    def _candidate_detector_point(
        candidate: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        detector_point = _geometry_manual_entry_detector_display_point(candidate)
        if detector_point is not None:
            return detector_point
        if not prefer_detector_sim_for_stale_entry:
            return None
        return _geometry_manual_finite_point(
            candidate,
            (("sim_col", "sim_row"),),
        )

    point_getter_pairs: tuple[
        tuple[
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            bool,
        ],
        ...,
    ] = (
        (
            _geometry_manual_entry_refined_sim_point,
            _geometry_manual_entry_refined_sim_point,
            False,
        ),
        (
            _geometry_manual_entry_caked_point,
            lambda candidate: _geometry_manual_entry_caked_point(
                candidate,
                allow_stale=True,
            ),
            False,
        ),
    )
    display_point_getter_pairs: tuple[
        tuple[
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            bool,
        ],
        ...,
    ] = (
        (
            _geometry_manual_entry_explicit_current_view_display_point,
            _geometry_manual_entry_matching_current_view_point,
            False,
        ),
    )
    saved_detector_point_getter_pairs: tuple[
        tuple[
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            bool,
        ],
        ...,
    ] = (
        (
            _geometry_manual_entry_saved_detector_hint_point,
            _geometry_manual_entry_native_point,
            True,
        ),
    )
    detector_point_getter_pairs: tuple[
        tuple[
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            bool,
        ],
        ...,
    ] = (
        (
            _geometry_manual_entry_detector_display_point,
            _candidate_detector_point,
            False,
        ),
    )
    native_point_getter_pairs: tuple[
        tuple[
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            bool,
        ],
        ...,
    ] = (
        (
            _geometry_manual_entry_resolver_native_hint_point,
            native_candidate_point_getter,
            True,
        ),
    )
    if entry_native_hint is not None:
        native_priority_tail = (
            display_point_getter_pairs
            + saved_detector_point_getter_pairs
            + detector_point_getter_pairs
        )
        if entry_stronger_native_hint is None:
            native_priority_tail = (
                saved_detector_point_getter_pairs
                + display_point_getter_pairs
                + detector_point_getter_pairs
            )
        point_getter_pairs = point_getter_pairs + native_point_getter_pairs + native_priority_tail
    else:
        point_getter_pairs = (
            point_getter_pairs
            + display_point_getter_pairs
            + saved_detector_point_getter_pairs
            + detector_point_getter_pairs
            + native_point_getter_pairs
        )

    for (
        entry_point_getter,
        candidate_point_getter,
        continue_on_tie,
    ) in point_getter_pairs:
        entry_point = entry_point_getter(entry)
        if entry_point is None:
            continue

        candidate_distances: list[tuple[float, int]] = []
        for idx, candidate in indexed_candidates:
            candidate_point = candidate_point_getter(candidate)
            if candidate_point is None:
                continue
            candidate_distances.append(
                (
                    float(
                        np.hypot(
                            float(candidate_point[0]) - float(entry_point[0]),
                            float(candidate_point[1]) - float(entry_point[1]),
                        )
                    ),
                    int(idx),
                )
            )
        if not candidate_distances:
            continue

        candidate_distances.sort(key=lambda item: (float(item[0]), int(item[1])))
        best_distance, best_idx = candidate_distances[0]
        if (
            len(candidate_distances) == 1
            or abs(float(candidate_distances[1][0]) - float(best_distance)) > 1.0e-9
        ):
            return int(best_idx)
        if bool(continue_on_tie):
            continue
        return None

    return None


def _geometry_manual_disambiguate_source_candidates(
    entry: Mapping[str, object] | None,
    indexed_candidates: Sequence[tuple[int, Mapping[str, object]]],
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> int | None:
    candidate_pool = list(indexed_candidates)
    if not candidate_pool:
        return None
    if len(candidate_pool) == 1:
        return int(candidate_pool[0][0])

    normalized_hkl = _geometry_manual_entry_normalized_hkl(
        entry,
        normalize_hkl_key=normalize_hkl_key,
    )
    if normalized_hkl is not None:
        hkl_matches = [
            (idx, candidate)
            for idx, candidate in candidate_pool
            if _geometry_manual_entry_normalized_hkl(
                candidate,
                normalize_hkl_key=normalize_hkl_key,
            )
            == normalized_hkl
        ]
        if len(hkl_matches) == 1:
            return int(hkl_matches[0][0])
        if hkl_matches:
            candidate_pool = hkl_matches

    label = _geometry_manual_entry_label(entry)
    if label:
        label_matches = [
            (idx, candidate)
            for idx, candidate in candidate_pool
            if _geometry_manual_entry_label(candidate) == label
        ]
        if len(label_matches) == 1:
            return int(label_matches[0][0])
        if label_matches:
            candidate_pool = label_matches

    return _geometry_manual_resolve_nearest_candidate_index(entry, candidate_pool)


def geometry_manual_resolve_source_entry_index(
    entry: Mapping[str, object] | None,
    candidate_entries: Sequence[object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> int | None:
    """Resolve one saved/manual entry to an index within a candidate sequence."""

    resolution_entry = _geometry_manual_identity_resolution_entry(entry)
    if not isinstance(resolution_entry, Mapping):
        return None

    indexed_candidates = [
        (int(idx), candidate)
        for idx, candidate in enumerate(candidate_entries or ())
        if isinstance(candidate, Mapping)
    ]
    if not indexed_candidates:
        return None

    source_key = geometry_manual_candidate_source_key(
        dict(resolution_entry),
        normalize_hkl_key=normalize_hkl_key,
    )
    if source_key is not None:
        exact_matches = [
            (int(idx), candidate)
            for idx, candidate in indexed_candidates
            if geometry_manual_candidate_source_key(
                dict(candidate),
                normalize_hkl_key=normalize_hkl_key,
            )
            == source_key
            and (
                str(source_key[0]) != "source_branch"
                or geometry_manual_source_entries_share_identity(
                    resolution_entry,
                    candidate,
                    normalize_hkl_key=normalize_hkl_key,
                )
            )
        ]
        if len(exact_matches) == 1:
            return int(exact_matches[0][0])
        if exact_matches:
            exact_match_index = _geometry_manual_disambiguate_source_candidates(
                resolution_entry,
                exact_matches,
                normalize_hkl_key=normalize_hkl_key,
            )
            if exact_match_index is not None:
                return int(exact_match_index)

        if len(source_key) >= 1 and str(source_key[0]) == "source_branch":
            identity_matches = [
                (idx, candidate)
                for idx, candidate in indexed_candidates
                if geometry_manual_source_entries_share_identity(
                    resolution_entry,
                    candidate,
                    normalize_hkl_key=normalize_hkl_key,
                )
            ]
            if len(identity_matches) == 1:
                return int(identity_matches[0][0])
            if identity_matches:
                return _geometry_manual_disambiguate_source_candidates(
                    resolution_entry,
                    identity_matches,
                    normalize_hkl_key=normalize_hkl_key,
                )

        if len(source_key) >= 3 and str(source_key[0]) == "source_branch":
            alias_matches = [
                (idx, candidate)
                for idx, candidate in indexed_candidates
                if geometry_manual_source_key_matches_entry(
                    source_key,
                    candidate,
                    normalize_hkl_key=normalize_hkl_key,
                )
            ]
            if len(alias_matches) == 1:
                return int(alias_matches[0][0])
            if alias_matches:
                return _geometry_manual_disambiguate_source_candidates(
                    resolution_entry,
                    alias_matches,
                    normalize_hkl_key=normalize_hkl_key,
                )

            query_row_key = _geometry_manual_source_row_key(resolution_entry)
            query_branch_index, _query_branch_source, _query_branch_reason = (
                resolve_canonical_branch(
                    resolution_entry,
                    allow_legacy_peak_fallback=True,
                )
            )
            if query_row_key is not None and query_branch_index in {0, 1}:
                row_branch_matches = [
                    (idx, candidate)
                    for idx, candidate in indexed_candidates
                    if _geometry_manual_source_row_key(candidate) == query_row_key
                    and resolve_canonical_branch(
                        _geometry_manual_identity_resolution_entry(candidate),
                        allow_legacy_peak_fallback=True,
                    )[0]
                    == int(query_branch_index)
                ]
                if len(row_branch_matches) == 1:
                    return int(row_branch_matches[0][0])
                if row_branch_matches:
                    return _geometry_manual_disambiguate_source_candidates(
                        resolution_entry,
                        row_branch_matches,
                        normalize_hkl_key=normalize_hkl_key,
                    )

        if (
            len(source_key) >= 3
            and str(source_key[0]) == "source"
            and not _geometry_manual_entry_has_explicit_branch_identity(resolution_entry)
        ):
            row_key = (int(source_key[1]), int(source_key[2]))
            nonbranch_matches = [
                (idx, candidate)
                for idx, candidate in indexed_candidates
                if _geometry_manual_source_row_key(candidate) == row_key
                and not _geometry_manual_entry_has_explicit_branch_identity(candidate)
            ]
            if len(nonbranch_matches) == 1:
                return int(nonbranch_matches[0][0])
            if nonbranch_matches:
                return _geometry_manual_disambiguate_source_candidates(
                    resolution_entry,
                    nonbranch_matches,
                    normalize_hkl_key=normalize_hkl_key,
                )

    row_key = _geometry_manual_source_row_key(resolution_entry)
    if row_key is None:
        return None

    row_matches = [
        (idx, candidate)
        for idx, candidate in indexed_candidates
        if _geometry_manual_source_row_key(candidate) == row_key
    ]
    if not row_matches:
        return None
    return _geometry_manual_disambiguate_source_candidates(
        resolution_entry,
        row_matches,
        normalize_hkl_key=normalize_hkl_key,
    )


def geometry_manual_source_key_matches_entry(
    source_key: tuple[object, ...] | None,
    entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> bool:
    """Return whether one persisted/manual source key resolves to an entry."""

    if not (isinstance(source_key, tuple) and len(source_key) >= 2):
        return False
    if not isinstance(entry, Mapping):
        return False

    entry_dict = dict(entry)
    entry_key = geometry_manual_candidate_source_key(
        entry_dict,
        normalize_hkl_key=normalize_hkl_key,
    )
    key_kind = str(source_key[0])
    if key_kind != "source_branch" and entry_key == tuple(source_key):
        return True

    if len(source_key) == 2:
        if _geometry_manual_entry_has_explicit_branch_identity(entry_dict):
            return False
        try:
            return int(entry_dict.get("source_table_index")) == int(source_key[0]) and int(
                entry_dict.get("source_row_index")
            ) == int(source_key[1])
        except Exception:
            return False

    if key_kind == "source":
        if _geometry_manual_entry_has_explicit_branch_identity(entry_dict):
            return False
        try:
            return int(entry_dict.get("source_table_index")) == int(source_key[1]) and int(
                entry_dict.get("source_row_index")
            ) == int(source_key[2])
        except Exception:
            return False

    if key_kind != "source_branch":
        return False

    try:
        source_anchor = int(source_key[1])
        source_branch_index = int(source_key[2])
    except Exception:
        return False

    entry_branch_index, _entry_branch_source, _entry_branch_reason = resolve_canonical_branch(
        _geometry_manual_identity_resolution_entry(entry_dict),
        allow_legacy_peak_fallback=True,
    )
    if entry_branch_index not in {0, 1} or int(entry_branch_index) != int(source_branch_index):
        return False

    entry_anchor = (
        _coerce_nonnegative_index(entry_dict.get("source_reflection_index"))
        if _entry_has_explicit_trusted_reflection_identity(entry_dict)
        else _coerce_nonnegative_index(entry_dict.get("source_table_index"))
    )
    return entry_anchor is not None and int(entry_anchor) == int(source_anchor)


def geometry_manual_source_entries_share_identity(
    left_entry: Mapping[str, object] | None,
    right_entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> bool:
    """Return whether two saved/live entries describe the same source identity."""

    if not (isinstance(left_entry, Mapping) and isinstance(right_entry, Mapping)):
        return False

    left_key = geometry_manual_candidate_source_key(
        dict(left_entry),
        normalize_hkl_key=normalize_hkl_key,
    )
    right_key = geometry_manual_candidate_source_key(
        dict(right_entry),
        normalize_hkl_key=normalize_hkl_key,
    )
    nonbranch_identity_matches = _geometry_manual_nonbranch_identity_fields_match(
        left_entry,
        right_entry,
        normalize_hkl_key=normalize_hkl_key,
    )
    if left_key is not None and left_key == right_key:
        if str(left_key[0]) != "source_branch":
            return bool(nonbranch_identity_matches)
        if _entry_has_explicit_trusted_reflection_identity(
            left_entry
        ) and _entry_has_explicit_trusted_reflection_identity(right_entry):
            return True

    left_branch_index, _left_branch_source, _left_branch_reason = resolve_canonical_branch(
        _geometry_manual_identity_resolution_entry(left_entry),
        allow_legacy_peak_fallback=True,
    )
    right_branch_index, _right_branch_source, _right_branch_reason = resolve_canonical_branch(
        _geometry_manual_identity_resolution_entry(right_entry),
        allow_legacy_peak_fallback=True,
    )
    if left_branch_index in {0, 1} or right_branch_index in {0, 1}:
        if left_branch_index not in {0, 1} or right_branch_index not in {0, 1}:
            return False
        if int(left_branch_index) != int(right_branch_index):
            return False

        left_has_trusted_reflection = _entry_has_explicit_trusted_reflection_identity(left_entry)
        right_has_trusted_reflection = _entry_has_explicit_trusted_reflection_identity(right_entry)
        if left_has_trusted_reflection and right_has_trusted_reflection:
            left_reflection_index = _coerce_nonnegative_index(
                left_entry.get("source_reflection_index")
            )
            right_reflection_index = _coerce_nonnegative_index(
                right_entry.get("source_reflection_index")
            )
            return (
                left_reflection_index is not None
                and left_reflection_index == right_reflection_index
            )

        left_row_key = _geometry_manual_source_row_key(left_entry)
        right_row_key = _geometry_manual_source_row_key(right_entry)
        if left_row_key is None or left_row_key != right_row_key:
            return False

        return bool(nonbranch_identity_matches)

    return bool(
        left_key is not None
        and str(left_key[0]) != "source_branch"
        and nonbranch_identity_matches
        and geometry_manual_source_key_matches_entry(
            left_key,
            right_entry,
            normalize_hkl_key=normalize_hkl_key,
        )
    ) or bool(
        right_key is not None
        and str(right_key[0]) != "source_branch"
        and nonbranch_identity_matches
        and geometry_manual_source_key_matches_entry(
            right_key,
            left_entry,
            normalize_hkl_key=normalize_hkl_key,
        )
    )


def _geometry_manual_resolve_identity_candidate_index(
    entry: Mapping[str, object] | None,
    candidate_entries: Sequence[object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> int | None:
    if not isinstance(entry, Mapping):
        return None

    normalized_candidates = [
        dict(candidate) for candidate in (candidate_entries or ()) if isinstance(candidate, Mapping)
    ]
    if not normalized_candidates:
        return None

    identity_matches = [
        (idx, candidate)
        for idx, candidate in enumerate(normalized_candidates)
        if geometry_manual_source_entries_share_identity(
            entry,
            candidate,
            normalize_hkl_key=normalize_hkl_key,
        )
    ]
    if len(identity_matches) == 1:
        return int(identity_matches[0][0])
    if identity_matches:
        return _geometry_manual_disambiguate_source_candidates(
            entry,
            identity_matches,
            normalize_hkl_key=normalize_hkl_key,
        )
    return None


def _geometry_manual_tagged_candidate_requires_identity(
    pick_session: Mapping[str, object] | None,
) -> bool:
    if not isinstance(pick_session, Mapping):
        return False
    explicit_flag = pick_session.get("_tagged_candidate_requires_identity")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    return isinstance(pick_session.get("tagged_candidate"), Mapping)


def _geometry_manual_allow_tagged_key_fallback(
    _tagged_candidate_key: tuple[object, ...] | None,
    *,
    requires_identity: bool = False,
    stored_tagged_candidate: Mapping[str, object] | None,
) -> bool:
    _ = stored_tagged_candidate
    return not bool(requires_identity)


def geometry_manual_lookup_source_entry(
    simulated_lookup: Mapping[tuple[object, ...], object] | None,
    entry: Mapping[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> dict[str, object] | None:
    """Resolve one saved/manual entry against cached simulated-peak lookup data."""

    if not (isinstance(simulated_lookup, Mapping) and isinstance(entry, Mapping)):
        return None

    source_key = geometry_manual_candidate_source_key(
        dict(entry),
        normalize_hkl_key=normalize_hkl_key,
    )
    direct_candidates = (
        _geometry_manual_lookup_bucket_entries(simulated_lookup.get(source_key))
        if source_key is not None
        else []
    )
    if direct_candidates:
        if not (
            isinstance(source_key, tuple)
            and len(source_key) >= 1
            and str(source_key[0]) == "source_branch"
        ):
            resolved_index = geometry_manual_resolve_source_entry_index(
                entry,
                direct_candidates,
                normalize_hkl_key=normalize_hkl_key,
            )
            if resolved_index is not None and 0 <= int(resolved_index) < len(direct_candidates):
                return dict(direct_candidates[int(resolved_index)])
            if len(direct_candidates) == 1:
                return dict(direct_candidates[0])

        direct_identity_index = _geometry_manual_resolve_identity_candidate_index(
            entry,
            direct_candidates,
            normalize_hkl_key=normalize_hkl_key,
        )
        if direct_identity_index is not None and 0 <= int(direct_identity_index) < len(
            direct_candidates
        ):
            return dict(direct_candidates[int(direct_identity_index)])
    if (
        source_key is not None
        and isinstance(source_key, tuple)
        and len(source_key) >= 3
        and str(source_key[0]) == "source"
        and not _geometry_manual_entry_has_explicit_branch_identity(entry)
    ):
        legacy_candidates = [
            candidate
            for candidate in _geometry_manual_lookup_bucket_entries(
                simulated_lookup.get((int(source_key[1]), int(source_key[2])))
            )
            if not _geometry_manual_entry_has_explicit_branch_identity(candidate)
        ]
        if legacy_candidates:
            resolved_index = geometry_manual_resolve_source_entry_index(
                entry,
                legacy_candidates,
                normalize_hkl_key=normalize_hkl_key,
            )
            if resolved_index is not None and 0 <= int(resolved_index) < len(legacy_candidates):
                return dict(legacy_candidates[int(resolved_index)])
            if len(legacy_candidates) == 1:
                return dict(legacy_candidates[0])

    candidate_entries = _geometry_manual_flatten_lookup_entries(simulated_lookup)
    if (
        isinstance(source_key, tuple)
        and len(source_key) >= 1
        and str(source_key[0]) == "source_branch"
    ):
        identity_match_index = _geometry_manual_resolve_identity_candidate_index(
            entry,
            candidate_entries,
            normalize_hkl_key=normalize_hkl_key,
        )
        if identity_match_index is not None and 0 <= int(identity_match_index) < len(
            candidate_entries
        ):
            return dict(candidate_entries[int(identity_match_index)])

    resolved_index = geometry_manual_resolve_source_entry_index(
        entry,
        candidate_entries,
        normalize_hkl_key=normalize_hkl_key,
    )
    if resolved_index is None:
        return None
    if 0 <= int(resolved_index) < len(candidate_entries):
        return dict(candidate_entries[int(resolved_index)])
    return None


def geometry_manual_candidate_distance_to_point(
    col: float,
    row: float,
    candidate: dict[str, object] | None,
    *,
    use_caked_display: bool = False,
) -> float:
    """Return one display-space point-to-candidate distance."""

    details = geometry_manual_candidate_distance_details(
        col,
        row,
        candidate,
        use_caked_display=bool(use_caked_display),
    )
    return float(details.get("distance", float("nan")))


def geometry_manual_prioritize_candidate_entries(
    candidate_entries: Sequence[dict[str, object]] | None,
    preferred_candidate: dict[str, object] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[dict[str, object]]:
    """Return candidate entries with the preferred entry, if any, moved to the front."""

    entries = [dict(entry) for entry in candidate_entries or [] if isinstance(entry, dict)]
    if not entries:
        return entries

    preferred_index = _geometry_manual_resolve_identity_candidate_index(
        preferred_candidate,
        entries,
    )
    if preferred_index is None:
        preferred_key = candidate_source_key(preferred_candidate)
        if preferred_key is None:
            return entries
        preferred_index = next(
            (
                idx
                for idx, entry in enumerate(entries)
                if candidate_source_key(entry) == preferred_key
            ),
            None,
        )
    if preferred_index is None or not (0 <= int(preferred_index) < len(entries)):
        return entries
    preferred_entry = dict(entries[int(preferred_index)])
    return [
        preferred_entry,
        *[dict(entry) for idx, entry in enumerate(entries) if int(idx) != int(preferred_index)],
    ]


def _geometry_manual_include_tagged_q_group_candidate(
    entries: Sequence[dict[str, object]] | None,
    tagged_candidate: Mapping[str, object] | None,
    *,
    group_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    output = [dict(entry) for entry in entries or () if isinstance(entry, dict)]
    if not isinstance(tagged_candidate, Mapping):
        return output
    tagged = dict(tagged_candidate)
    if _geometry_manual_resolve_identity_candidate_index(tagged, output) is not None:
        return output
    normalized_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_key is not None:
        tagged["q_group_key"] = normalized_key
        tagged.setdefault("source_label", _geometry_manual_q_group_source_label(normalized_key))
    tagged_slot = _geometry_manual_q_group_physical_branch_slot(
        tagged,
        group_key=normalized_key,
        profile_cache=profile_cache,
    )
    if tagged_slot is None:
        return [tagged, *output]
    replaced = False
    next_entries: list[dict[str, object]] = []
    for raw_entry in output:
        entry = dict(raw_entry)
        if (
            normalized_key is not None
            and _geometry_manual_real_q_group_key(entry) != normalized_key
        ):
            next_entries.append(entry)
            continue
        entry_slot = _geometry_manual_q_group_physical_branch_slot(
            entry,
            group_key=normalized_key,
            profile_cache=profile_cache,
        )
        if not replaced and entry_slot == tagged_slot:
            next_entries.append(dict(tagged))
            replaced = True
            continue
        next_entries.append(entry)
    if not replaced:
        next_entries.insert(0, dict(tagged))
    return next_entries


def _geometry_manual_source_priority(source_label: object) -> int:
    label = normalize_bragg_qr_source_label(str(source_label) if source_label is not None else None)
    if label == "primary":
        return 0
    if label == DISORDERED_PHASE_SOURCE_LABEL:
        return 1
    if label == "secondary":
        return 2
    if label == "pbii_6h_ref":
        return 3
    return 9


def _geometry_manual_real_q_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, Mapping):
        candidate = value.get("q_group_key")
        value = candidate if candidate is not None else value.get("key")
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        plain = gui_mosaic_top.normalize_q_group_key(value)
        if plain is not None:
            return plain
    normalized = gui_controllers.normalize_geometry_q_group_key(value)
    if normalized is not None:
        source_label, m_index, gz_index = normalized
        return ("q_group", str(source_label), int(m_index), int(gz_index))
    return gui_mosaic_top.normalize_q_group_key(value)


def _geometry_manual_q_group_source_label(value: object) -> str:
    key = _geometry_manual_real_q_group_key(value)
    if isinstance(key, tuple) and len(key) >= 4 and str(key[0]) == "q_group":
        return normalize_bragg_qr_source_label(str(key[1]))
    if isinstance(value, Mapping):
        return normalize_bragg_qr_source_label(
            str(value.get("source_label")) if value.get("source_label") is not None else None
        )
    return "primary"


def _geometry_manual_entry_source_label(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
) -> str:
    key = _geometry_manual_real_q_group_key(group_key)
    if key is None:
        key = _geometry_manual_real_q_group_key(entry)
    if isinstance(key, tuple) and len(key) >= 4 and str(key[0]) == "q_group":
        return normalize_bragg_qr_source_label(str(key[1]))
    if isinstance(entry, Mapping) and entry.get("source_label") is not None:
        return normalize_bragg_qr_source_label(str(entry.get("source_label")))
    return "primary"


def _geometry_manual_source_phase_label(source_label: object) -> str:
    label = normalize_bragg_qr_source_label(str(source_label) if source_label is not None else None)
    if label == DISORDERED_PHASE_SOURCE_LABEL:
        return DISORDERED_PHASE_DISPLAY_LABEL
    if label == "primary":
        return "Primary phase"
    return str(source_label or label)


def _geometry_manual_source_structure_role(source_label: object) -> str:
    label = normalize_bragg_qr_source_label(str(source_label) if source_label is not None else None)
    if label == DISORDERED_PHASE_SOURCE_LABEL:
        return "disordered"
    return str(label)


def _geometry_manual_detector_position_source(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
) -> str:
    source_label = _geometry_manual_entry_source_label(entry, group_key=group_key)
    if source_label == DISORDERED_PHASE_SOURCE_LABEL:
        return "disordered_phase_hit_table"
    return f"{source_label}_hit_table"


def _geometry_manual_source_consistency_fields(
    *,
    selected_group_key: object = None,
    candidate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected_source = _geometry_manual_entry_source_label(None, group_key=selected_group_key)
    candidate_source = _geometry_manual_entry_source_label(candidate, group_key=selected_group_key)
    hit_table_source = candidate_source
    return {
        "selected_group_source": selected_source,
        "candidate_source": candidate_source,
        "sim_peak_source": candidate_source,
        "hit_table_source": hit_table_source,
        "detector_position_source": _geometry_manual_detector_position_source(
            candidate,
            group_key=selected_group_key,
        ),
        "assigned_hkl_source": candidate_source,
    }


def _geometry_manual_candidate_matches_selected_group_source(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
) -> bool:
    normalized_group_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_group_key is None:
        return True
    candidate_group_key = _geometry_manual_real_q_group_key(entry)
    if candidate_group_key is not None and candidate_group_key != normalized_group_key:
        return False
    return _geometry_manual_entry_source_label(
        entry, group_key=candidate_group_key
    ) == _geometry_manual_entry_source_label(None, group_key=normalized_group_key)


def _geometry_manual_filter_candidates_for_selected_group_source(
    candidates: Sequence[dict[str, object]] | None,
    *,
    group_key: object = None,
) -> list[dict[str, object]]:
    return [
        dict(entry)
        for entry in (candidates or ())
        if isinstance(entry, dict)
        and _geometry_manual_candidate_matches_selected_group_source(entry, group_key=group_key)
    ]


def _geometry_manual_available_candidate_sources(
    candidates: Sequence[dict[str, object]] | None,
) -> list[str]:
    seen: list[str] = []
    for entry in candidates or ():
        if not isinstance(entry, Mapping):
            continue
        source_label = _geometry_manual_entry_source_label(entry)
        if source_label not in seen:
            seen.append(source_label)
    return seen


def _geometry_manual_q_group_entry_is_00l(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = (
        _default_normalize_hkl_key
    ),
) -> bool:
    hkl_key = (
        normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if isinstance(entry, Mapping)
        else None
    )
    if hkl_key is not None:
        return int(hkl_key[0]) == 0 and int(hkl_key[1]) == 0 and int(hkl_key[2]) != 0
    normalized_group_key = gui_controllers.normalize_geometry_q_group_key(
        group_key if group_key is not None else entry
    )
    if isinstance(normalized_group_key, tuple) and len(normalized_group_key) >= 3:
        try:
            return int(normalized_group_key[1]) == 0 and int(normalized_group_key[2]) != 0
        except Exception:
            return False
    normalized_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_key is None and isinstance(entry, Mapping):
        normalized_key = _geometry_manual_real_q_group_key(entry)
    if isinstance(normalized_key, tuple) and len(normalized_key) >= 4:
        try:
            return int(normalized_key[2]) == 0 and int(normalized_key[3]) != 0
        except Exception:
            return False
    if isinstance(normalized_key, tuple) and len(normalized_key) >= 3:
        try:
            return int(normalized_key[1]) == 0 and int(normalized_key[2]) != 0
        except Exception:
            return False
    return False


def _geometry_manual_q_group_entry_is_000(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = (
        _default_normalize_hkl_key
    ),
) -> bool:
    hkl_key = (
        normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if isinstance(entry, Mapping)
        else None
    )
    if hkl_key is not None:
        return int(hkl_key[0]) == 0 and int(hkl_key[1]) == 0 and int(hkl_key[2]) == 0
    normalized_group_key = gui_controllers.normalize_geometry_q_group_key(
        group_key if group_key is not None else entry
    )
    if isinstance(normalized_group_key, tuple) and len(normalized_group_key) >= 3:
        try:
            return int(normalized_group_key[1]) == 0 and int(normalized_group_key[2]) == 0
        except Exception:
            return False
    normalized_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_key is None and isinstance(entry, Mapping):
        normalized_key = _geometry_manual_real_q_group_key(entry)
    if isinstance(normalized_key, tuple) and len(normalized_key) >= 4:
        try:
            return int(normalized_key[2]) == 0 and int(normalized_key[3]) == 0
        except Exception:
            return False
    if isinstance(normalized_key, tuple) and len(normalized_key) >= 3:
        try:
            return int(normalized_key[1]) == 0 and int(normalized_key[2]) == 0
        except Exception:
            return False
    return False


def _geometry_manual_branch_index_from_value(value: object) -> int | None:
    try:
        branch = int(value)
    except Exception:
        return None
    return int(branch) if branch in {0, 1} else None


def _geometry_manual_q_group_unknown_source_identity_slot(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    identity_parts: list[tuple[str, object]] = []
    strong_identity_seen = False
    for key in (
        "source_label",
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "best_sample_index",
        "source_peak_index",
    ):
        value = entry.get(key)
        if value is None:
            continue
        if key != "source_label":
            strong_identity_seen = True
        identity_parts.append((key, _geometry_manual_hashable_identity_value(value)))
    if strong_identity_seen:
        normalized_key = _geometry_manual_real_q_group_key(group_key)
        if normalized_key is None:
            normalized_key = _geometry_manual_real_q_group_key(entry)
        return (
            "source_identity",
            _geometry_manual_hashable_identity_value(normalized_key),
            tuple(identity_parts),
        )
    raw_hkl = entry.get("normalized_hkl", entry.get("hkl", entry.get("source_hkl")))
    if raw_hkl is not None:
        hkl_identity = _geometry_manual_hashable_identity_value(raw_hkl)
        return ("hkl_identity", hkl_identity)
    return None


def _geometry_manual_q_group_physical_branch_slot(
    entry: Mapping[str, object] | None,
    *,
    group_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[object, ...] | None:
    """Return the physical manual-pick branch slot for one Qr/Qz row."""

    if _geometry_manual_q_group_entry_is_00l(entry, group_key=group_key):
        return ("00l",)
    if not isinstance(entry, Mapping):
        return None

    for key in ("source_branch_index", "source_peak_index", "branch_index"):
        branch = _geometry_manual_branch_index_from_value(entry.get(key))
        if branch in {0, 1}:
            return ("branch", int(branch))

    branch_text = str(entry.get("branch_id", "") or "").strip()
    if branch_text == "+x":
        return ("branch", 0)
    if branch_text == "-x":
        return ("branch", 1)

    normalized_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_key is None:
        normalized_key = _geometry_manual_real_q_group_key(entry)
    source_identity_slot = _geometry_manual_q_group_unknown_source_identity_slot(
        entry,
        group_key=normalized_key,
    )
    if source_identity_slot is not None:
        return source_identity_slot
    branch_id, _branch_source = gui_mosaic_top.normalize_branch_id(
        entry,
        target_key=normalized_key,
        profile_cache=profile_cache,
    )
    if branch_id == gui_mosaic_top.ZERO_QR_BRANCH_ID:
        return ("00l",)
    if branch_id == "+x":
        return ("branch", 0)
    if branch_id == "-x":
        return ("branch", 1)
    if branch_id:
        return ("branch_id", str(branch_id))
    return None


def _huber_loss(r: float, delta: float = 1.0) -> float:
    try:
        residual = float(r)
        threshold = abs(float(delta))
    except Exception:
        return float("inf")
    if not np.isfinite(residual):
        return float("inf")
    if not np.isfinite(threshold) or threshold <= 0.0:
        threshold = 1.0
    abs_residual = abs(residual)
    if abs_residual <= threshold:
        return float(0.5 * residual * residual)
    return float(threshold * (abs_residual - 0.5 * threshold))


def _geometry_manual_auto_branch_pair_tolerances(
    *,
    auto_refine_search_radius_px: float | None = None,
    branch_pair_abs_tol_px: float | None = None,
    branch_pair_rel_tol: float | None = None,
    branch_pair_sigma_px: float | None = None,
    predicted_length_px: float = 0.0,
) -> dict[str, float]:
    try:
        radius = float(auto_refine_search_radius_px)
    except Exception:
        radius = 24.0
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 24.0
    scale = min(5.0 / 3.0, max(2.0 / 3.0, radius / 24.0))

    try:
        abs_tol = float(branch_pair_abs_tol_px)
    except Exception:
        abs_tol = float("nan")
    if not np.isfinite(abs_tol) or abs_tol <= 0.0:
        abs_tol = 12.0 * scale

    try:
        rel_tol = float(branch_pair_rel_tol)
    except Exception:
        rel_tol = float("nan")
    if not np.isfinite(rel_tol) or rel_tol < 0.0:
        rel_tol = 0.25 * scale

    try:
        predicted_length = float(predicted_length_px)
    except Exception:
        predicted_length = 0.0
    if not np.isfinite(predicted_length):
        predicted_length = 0.0
    gate_px = max(float(abs_tol), float(rel_tol) * max(float(predicted_length), 1.0))

    try:
        sigma_px = float(branch_pair_sigma_px)
    except Exception:
        sigma_px = float("nan")
    if not np.isfinite(sigma_px) or sigma_px <= 0.0:
        sigma_px = max(3.0, 0.25 * float(gate_px))

    return {
        "abs_tol_px": float(abs_tol),
        "rel_tol": float(rel_tol),
        "gate_px": float(gate_px),
        "sigma_px": float(sigma_px),
        "scale": float(scale),
    }


def _geometry_manual_observed_native_detector_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    tuple_point = _geometry_manual_tuple_point(entry, "geometry_detector_native_px")
    if tuple_point is not None:
        return tuple_point
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_detector_native_col", "refined_detector_native_row"),
            ("background_detector_x", "background_detector_y"),
            ("detector_x", "detector_y"),
        ),
    )


def _geometry_manual_predicted_native_detector_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_candidate_native_sim_point(entry)


def _geometry_manual_observed_caked_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_background_two_theta_deg", "refined_background_phi_deg"),
            ("background_two_theta_deg", "background_phi_deg"),
            ("caked_x", "caked_y"),
        ),
    )


def _geometry_manual_predicted_caked_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for key in (
        "sim_refined_caked_deg",
        "sim_visual_caked_deg",
        "sim_visual_deg",
        "sim_caked",
    ):
        point = _geometry_manual_tuple_point(entry, key)
        if point is not None:
            return point
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
        ),
    )


def _geometry_manual_observed_current_view_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    tuple_point = _geometry_manual_tuple_point(entry, "geometry_detector_display_px")
    if tuple_point is not None:
        return tuple_point
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_detector_display_col", "refined_detector_display_row"),
            ("x", "y"),
        ),
    )


def _geometry_manual_predicted_current_view_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for key in ("sim_refined_detector_display_px", "sim_visual_detector_display_px"):
        point = _geometry_manual_tuple_point(entry, key)
        if point is not None and not _geometry_manual_detector_point_is_caked(entry, point):
            return point
    point = _geometry_manual_finite_point(
        entry,
        (
            ("refined_sim_x", "refined_sim_y"),
            ("sim_col_raw", "sim_row_raw"),
            ("sim_col", "sim_row"),
        ),
    )
    if point is not None and not _geometry_manual_detector_point_is_caked(entry, point):
        return point
    return None


def _geometry_manual_branch_pair_same_frame_points(
    branch0_entry: Mapping[str, object] | None,
    branch1_entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> dict[str, object] | None:
    frame_getters: list[
        tuple[
            str,
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
            Callable[[Mapping[str, object] | None], tuple[float, float] | None],
        ]
    ] = [
        (
            "native_detector",
            _geometry_manual_observed_native_detector_point,
            _geometry_manual_predicted_native_detector_point,
        )
    ]
    if bool(use_caked_space):
        frame_getters.append(
            (
                "caked_display",
                _geometry_manual_observed_caked_display_point,
                _geometry_manual_predicted_caked_display_point,
            )
        )
    frame_getters.append(
        (
            "current_view_display",
            _geometry_manual_observed_current_view_display_point,
            _geometry_manual_predicted_current_view_display_point,
        )
    )

    for frame_name, observed_getter, predicted_getter in frame_getters:
        obs0 = observed_getter(branch0_entry)
        obs1 = observed_getter(branch1_entry)
        pred0 = predicted_getter(branch0_entry)
        pred1 = predicted_getter(branch1_entry)
        if obs0 is not None and obs1 is not None and pred0 is not None and pred1 is not None:
            return {
                "coordinate_frame": frame_name,
                "obs0": (float(obs0[0]), float(obs0[1])),
                "obs1": (float(obs1[0]), float(obs1[1])),
                "pred0": (float(pred0[0]), float(pred0[1])),
                "pred1": (float(pred1[0]), float(pred1[1])),
            }
    return None


def _score_qr_branch_pair_length_restraint(
    branch0_entry: Mapping[str, object] | None,
    branch1_entry: Mapping[str, object] | None = None,
    *,
    group_key: object = None,
    use_caked_space: bool,
    auto_refine_search_radius_px: float | None = None,
    branch_pair_abs_tol_px: float | None = None,
    branch_pair_rel_tol: float | None = None,
    branch_pair_sigma_px: float | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized_group_key = _geometry_manual_real_q_group_key(group_key)
    if _geometry_manual_q_group_entry_is_00l(branch0_entry, group_key=normalized_group_key):
        return {
            "applied": False,
            "accepted": True,
            "reason": "00l_branch_collapsed",
            "branch_pair_coordinate_frame": None,
        }

    if not isinstance(branch0_entry, Mapping) or not isinstance(branch1_entry, Mapping):
        return {
            "applied": True,
            "accepted": False,
            "reason": "missing_required_branch_pair",
            "branch_pair_coordinate_frame": None,
        }

    key0 = _geometry_manual_real_q_group_key(branch0_entry)
    key1 = _geometry_manual_real_q_group_key(branch1_entry)
    if normalized_group_key is None:
        normalized_group_key = key0
    if normalized_group_key is None or key0 != normalized_group_key or key1 != normalized_group_key:
        return {
            "applied": True,
            "accepted": False,
            "reason": "q_group_key_mismatch",
            "branch_pair_coordinate_frame": None,
        }

    slot0 = _geometry_manual_q_group_physical_branch_slot(
        branch0_entry,
        group_key=normalized_group_key,
        profile_cache=profile_cache,
    )
    slot1 = _geometry_manual_q_group_physical_branch_slot(
        branch1_entry,
        group_key=normalized_group_key,
        profile_cache=profile_cache,
    )
    if {tuple(slot0 or ()), tuple(slot1 or ())} != {("branch", 0), ("branch", 1)}:
        return {
            "applied": True,
            "accepted": False,
            "reason": "missing_required_branch_pair",
            "branch_pair_coordinate_frame": None,
        }

    points = _geometry_manual_branch_pair_same_frame_points(
        branch0_entry,
        branch1_entry,
        use_caked_space=bool(use_caked_space),
    )
    if points is None:
        return {
            "applied": True,
            "accepted": False,
            "reason": "branch_pair_points_unavailable",
            "branch_pair_coordinate_frame": None,
        }

    obs0 = points["obs0"]
    obs1 = points["obs1"]
    pred0 = points["pred0"]
    pred1 = points["pred1"]
    d_obs = float(np.hypot(float(obs1[0]) - float(obs0[0]), float(obs1[1]) - float(obs0[1])))
    d_pred = float(np.hypot(float(pred1[0]) - float(pred0[0]), float(pred1[1]) - float(pred0[1])))
    length_error = float(d_obs - d_pred)
    tolerances = _geometry_manual_auto_branch_pair_tolerances(
        auto_refine_search_radius_px=auto_refine_search_radius_px,
        branch_pair_abs_tol_px=branch_pair_abs_tol_px,
        branch_pair_rel_tol=branch_pair_rel_tol,
        branch_pair_sigma_px=branch_pair_sigma_px,
        predicted_length_px=d_pred,
    )
    accepted = bool(abs(length_error) <= float(tolerances["gate_px"]))
    return {
        "applied": True,
        "accepted": accepted,
        "reason": "within_tolerance" if accepted else "branch_pair_length_out_of_tolerance",
        "branch_pair_coordinate_frame": str(points["coordinate_frame"]),
        "d_obs": float(d_obs),
        "d_pred": float(d_pred),
        "length_error": float(length_error),
        "abs_tol_px": float(tolerances["abs_tol_px"]),
        "rel_tol": float(tolerances["rel_tol"]),
        "gate_px": float(tolerances["gate_px"]),
        "sigma_px": float(tolerances["sigma_px"]),
        "score": float(_huber_loss(length_error / max(float(tolerances["sigma_px"]), 1.0e-6))),
    }


def _geometry_manual_q_group_required_branch_entries(
    entries: Sequence[Mapping[str, object]] | None,
    *,
    group_key: object,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    normalized_group_key = _geometry_manual_real_q_group_key(group_key)
    branch_entries: dict[int, dict[str, object]] = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, Mapping):
            continue
        if _geometry_manual_real_q_group_key(raw_entry) != normalized_group_key:
            continue
        slot = _geometry_manual_q_group_physical_branch_slot(
            raw_entry,
            group_key=normalized_group_key,
            profile_cache=profile_cache,
        )
        if slot in {("branch", 0), ("branch", 1)} and int(slot[1]) not in branch_entries:
            branch_entries[int(slot[1])] = dict(raw_entry)
    return branch_entries.get(0), branch_entries.get(1)


def _geometry_manual_branch_pair_result_payload(
    group_key: object,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "q_group_key": _geometry_manual_json_value(_geometry_manual_real_q_group_key(group_key)),
        "branch_pair_coordinate_frame": result.get("branch_pair_coordinate_frame"),
        "branch_pair_restraint_applied": bool(result.get("applied", False)),
        "branch_pair_restraint_accepted": bool(result.get("accepted", False)),
        "branch_pair_length_observed_px": result.get("d_obs"),
        "branch_pair_length_predicted_px": result.get("d_pred"),
        "branch_pair_length_error_px": result.get("length_error"),
        "branch_pair_length_gate_px": result.get("gate_px"),
        "branch_pair_length_score": result.get("score"),
        "branch_pair_reject_reason": str(result.get("reason", "")),
    }


def _geometry_manual_select_q_group_representative(
    entries: Sequence[dict[str, object]] | None,
    *,
    group_key: object = None,
    seed_candidate: Mapping[str, object] | None = None,
    branch_id: str | None = None,
    branch_slot: tuple[object, ...] | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    normalized_key = _geometry_manual_real_q_group_key(group_key)
    if normalized_key is None:
        for raw_entry in entries or ():
            if isinstance(raw_entry, dict):
                return dict(raw_entry)
        return None
    normalized_entries = [
        {**dict(entry), "q_group_key": normalized_key}
        for entry in entries or ()
        if isinstance(entry, dict)
    ]
    selected_branch_id = branch_id
    selected_branch_slot = branch_slot
    if selected_branch_slot is None and isinstance(seed_candidate, Mapping):
        selected_branch_slot = _geometry_manual_q_group_physical_branch_slot(
            seed_candidate,
            group_key=normalized_key,
            profile_cache=profile_cache,
        )
    if selected_branch_slot is not None:
        slot_entries = [
            entry
            for entry in normalized_entries
            if _geometry_manual_q_group_physical_branch_slot(
                entry,
                group_key=normalized_key,
                profile_cache=profile_cache,
            )
            == selected_branch_slot
        ]
        selected = gui_mosaic_top.select_mosaic_top_representative(
            slot_entries,
            branch_id=None,
            target_key=normalized_key,
            profile_cache=profile_cache,
        )
        if isinstance(selected, dict):
            selected["selection_reason"] = gui_mosaic_top.SELECTION_REASON
            selected["manual_qr_physical_branch_slot"] = tuple(selected_branch_slot)
            return dict(selected)
    if selected_branch_id is None and isinstance(seed_candidate, Mapping):
        selected_branch_id, _branch_source = gui_mosaic_top.normalize_branch_id(
            seed_candidate,
            target_key=normalized_key,
            profile_cache=profile_cache,
        )
    selected = gui_mosaic_top.select_mosaic_top_representative(
        normalized_entries,
        branch_id=selected_branch_id,
        target_key=normalized_key,
        profile_cache=profile_cache,
    )
    return dict(selected) if isinstance(selected, dict) else None


def _geometry_manual_collapse_q_group_representatives(
    entries: Sequence[dict[str, object]] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[tuple[object, ...], tuple[object, ...]], list[dict[str, object]]] = {}
    ordered_keys: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
    output: list[dict[str, object]] = []
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = _geometry_manual_real_q_group_key(entry)
        if group_key is None:
            output.append(entry)
            continue
        entry["q_group_key"] = group_key
        branch_slot = _geometry_manual_q_group_physical_branch_slot(
            entry,
            group_key=group_key,
            profile_cache=profile_cache,
        )
        if branch_slot is None:
            branch_id, _branch_source = gui_mosaic_top.normalize_branch_id(
                entry,
                target_key=group_key,
                profile_cache=profile_cache,
            )
            branch_slot = ("branch_id", str(branch_id))
        bucket_key = (group_key, tuple(branch_slot))
        if bucket_key not in grouped:
            grouped[bucket_key] = []
            ordered_keys.append(bucket_key)
        grouped[bucket_key].append(entry)
    for group_key, branch_slot in ordered_keys:
        selected = _geometry_manual_select_q_group_representative(
            grouped.get((group_key, branch_slot), []),
            group_key=group_key,
            branch_slot=branch_slot,
            profile_cache=profile_cache,
        )
        if isinstance(selected, dict):
            output.append(selected)
    return output


def _geometry_manual_hashable_identity_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    str(key),
                    _geometry_manual_hashable_identity_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_geometry_manual_hashable_identity_value(item) for item in value)
    try:
        hash(value)
    except Exception:
        return repr(value)
    return value


def _geometry_manual_refresh_group_key(
    entry: Mapping[str, object] | None,
    fallback_group_key: object = None,
) -> tuple[object, ...] | None:
    group_key = _geometry_manual_real_q_group_key(entry)
    if group_key is None:
        group_key = gui_mosaic_top.normalize_q_group_key(fallback_group_key)
    return group_key


def _geometry_manual_refresh_branch_identity_keys(
    entry: Mapping[str, object] | None,
    *,
    fallback_group_key: object = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[tuple[object, ...]]:
    if not isinstance(entry, Mapping):
        return []

    keys: list[tuple[object, ...]] = []
    group_key = _geometry_manual_refresh_group_key(entry, fallback_group_key)
    normalized_hkl = _geometry_manual_entry_normalized_hkl(entry)
    branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
        _geometry_manual_identity_resolution_entry(entry),
        allow_legacy_peak_fallback=True,
    )
    source_peak_idx = _resolve_legacy_source_peak_index(entry)
    source_table_idx = _coerce_nonnegative_index(entry.get("source_table_index"))
    source_row_idx = _coerce_nonnegative_index(entry.get("source_row_index"))
    branch_id_raw = entry.get("branch_id")
    branch_id = (
        str(branch_id_raw).strip()
        if branch_id_raw is not None and str(branch_id_raw).strip()
        else None
    )

    if group_key is None:
        return list(dict.fromkeys(keys))

    group_key_part = _geometry_manual_hashable_identity_value(group_key)
    if normalized_hkl is not None and source_table_idx is not None and source_row_idx is not None:
        hkl_part = tuple(int(value) for value in normalized_hkl)
        row_part = (int(source_table_idx), int(source_row_idx))
        if branch_idx in {0, 1}:
            keys.append(
                (
                    "q_group_hkl_source_row_source_branch",
                    group_key_part,
                    hkl_part,
                    row_part,
                    int(branch_idx),
                )
            )
        if source_peak_idx is not None:
            keys.append(
                (
                    "q_group_hkl_source_row_source_peak",
                    group_key_part,
                    hkl_part,
                    row_part,
                    int(source_peak_idx),
                )
            )
        if branch_id is not None:
            keys.append(
                (
                    "q_group_hkl_source_row_branch_id",
                    group_key_part,
                    hkl_part,
                    row_part,
                    branch_id,
                )
            )

    if source_table_idx is not None and source_row_idx is not None:
        row_part = (int(source_table_idx), int(source_row_idx))
        if branch_idx in {0, 1}:
            keys.append(
                (
                    "q_group_source_row_source_branch",
                    group_key_part,
                    row_part,
                    int(branch_idx),
                )
            )
        if source_peak_idx is not None:
            keys.append(
                (
                    "q_group_source_row_source_peak",
                    group_key_part,
                    row_part,
                    int(source_peak_idx),
                )
            )
        if branch_id is not None:
            keys.append(("q_group_source_row_branch_id", group_key_part, row_part, branch_id))

    if normalized_hkl is not None:
        hkl_part = tuple(int(value) for value in normalized_hkl)
        if branch_idx in {0, 1}:
            keys.append(
                (
                    "q_group_hkl_source_branch",
                    group_key_part,
                    hkl_part,
                    int(branch_idx),
                )
            )
        if source_peak_idx is not None:
            keys.append(
                (
                    "q_group_hkl_source_peak",
                    group_key_part,
                    hkl_part,
                    int(source_peak_idx),
                )
            )
        if branch_id is not None:
            keys.append(("q_group_hkl_branch_id", group_key_part, hkl_part, branch_id))

    return list(dict.fromkeys(keys))


def _geometry_manual_entry_has_caked_visual_sim(
    entry: Mapping[str, object] | None,
) -> bool:
    visual_point, _source = _geometry_manual_candidate_visual_caked_sim_point(entry)
    return visual_point is not None


def _geometry_manual_entry_is_caked_visual_refresh_row(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if bool(entry.get("_caked_qr_projection_cache")):
        return True
    if _geometry_manual_entry_display_frame(entry) == "caked":
        return _geometry_manual_entry_has_caked_visual_sim(entry)
    return False


def _geometry_manual_caked_visual_preservation_lookup(
    entries: Sequence[dict[str, object]] | None,
    *,
    fallback_group_key: object = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[tuple[object, ...], dict[str, object]]:
    lookup: dict[tuple[object, ...], dict[str, object]] = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, Mapping):
            continue
        if not _geometry_manual_entry_has_caked_visual_sim(raw_entry):
            continue
        entry = dict(raw_entry)
        for identity_key in _geometry_manual_refresh_branch_identity_keys(
            entry,
            fallback_group_key=fallback_group_key,
            candidate_source_key=candidate_source_key,
        ):
            lookup.setdefault(identity_key, entry)
    return lookup


def _geometry_manual_restore_caked_visual_fields(
    entry: dict[str, object],
    existing_visual_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(existing_visual_entry, Mapping):
        return entry
    for key in _GEOMETRY_MANUAL_CAKED_VISUAL_PRESERVE_KEYS:
        if key in existing_visual_entry:
            entry[key] = existing_visual_entry[key]
    return entry


def _geometry_manual_preserve_caked_visual_fields_by_identity(
    entries: Sequence[dict[str, object]] | None,
    *,
    existing_entries: Sequence[dict[str, object]] | None,
    group_key: object = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[dict[str, object]]:
    refreshed_entries = [dict(entry) for entry in entries or () if isinstance(entry, dict)]
    existing_lookup = _geometry_manual_caked_visual_preservation_lookup(
        existing_entries,
        fallback_group_key=group_key,
        candidate_source_key=candidate_source_key,
    )
    if not existing_lookup:
        return refreshed_entries

    preserved_entries: list[dict[str, object]] = []
    for entry in refreshed_entries:
        preserved = dict(entry)
        for identity_key in _geometry_manual_refresh_branch_identity_keys(
            preserved,
            fallback_group_key=group_key,
            candidate_source_key=candidate_source_key,
        ):
            existing_entry = existing_lookup.get(identity_key)
            if existing_entry is not None:
                _geometry_manual_restore_caked_visual_fields(preserved, existing_entry)
                break
        preserved_entries.append(preserved)
    return preserved_entries


def refresh_geometry_manual_pick_session_candidates(
    pick_session: dict[str, object] | None,
    *,
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
    cache_signature: object = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    prioritize_candidate_entries: Callable[
        [Sequence[dict[str, object]] | None, dict[str, object] | None],
        list[dict[str, object]],
    ] = geometry_manual_prioritize_candidate_entries,
    group_target_count_fn: Callable[
        [tuple[object, ...] | None, Sequence[dict[str, object]] | None],
        int,
    ]
    | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Refresh one active manual-pick session against the latest simulated group entries."""

    if not isinstance(pick_session, dict):
        return {}
    if not geometry_manual_pick_session_active(
        pick_session,
        require_current_background=False,
    ):
        return dict(pick_session)

    group_key = pick_session.get("group_key")
    if group_key is None or not isinstance(grouped_candidates, Mapping):
        return dict(pick_session)

    live_entries_raw = grouped_candidates.get(group_key)
    if not isinstance(live_entries_raw, Sequence):
        return dict(pick_session)
    live_entries = [
        {**dict(entry), "q_group_key": group_key}
        for entry in live_entries_raw
        if isinstance(entry, dict)
    ]
    if not live_entries:
        return dict(pick_session)
    existing_group_entries = [
        dict(entry) for entry in pick_session.get("group_entries", []) if isinstance(entry, dict)
    ]
    if any(
        _geometry_manual_entry_is_caked_visual_refresh_row(entry)
        for entry in existing_group_entries
    ) and not any(
        _geometry_manual_entry_is_caked_visual_refresh_row(entry) for entry in live_entries
    ):
        refreshed_session = dict(pick_session)
        if cache_signature is not None:
            refreshed_session["cache_signature"] = cache_signature
        return refreshed_session
    live_entries = _geometry_manual_collapse_q_group_representatives(
        live_entries,
        profile_cache=profile_cache,
    )
    if not live_entries:
        return dict(pick_session)
    if group_target_count_fn is None:
        group_target_count_fn = geometry_manual_group_target_count

    preferred_candidate = None
    tagged_candidate_requires_identity = _geometry_manual_tagged_candidate_requires_identity(
        pick_session
    )
    stored_tagged_candidate = (
        dict(pick_session["tagged_candidate"])
        if isinstance(pick_session.get("tagged_candidate"), dict)
        else None
    )
    tagged_candidate_key = pick_session.get("tagged_candidate_key")
    allow_tagged_key_fallback = _geometry_manual_allow_tagged_key_fallback(
        tagged_candidate_key,
        requires_identity=tagged_candidate_requires_identity,
        stored_tagged_candidate=stored_tagged_candidate,
    )
    if stored_tagged_candidate is not None:
        preferred_index = _geometry_manual_resolve_identity_candidate_index(
            stored_tagged_candidate,
            live_entries,
        )
        if preferred_index is not None and 0 <= int(preferred_index) < len(live_entries):
            preferred_candidate = dict(live_entries[int(preferred_index)])
    if (
        allow_tagged_key_fallback
        and preferred_candidate is None
        and tagged_candidate_key is not None
    ):
        preferred_candidate = next(
            (
                dict(entry)
                for entry in live_entries
                if candidate_source_key(entry) == tagged_candidate_key
            ),
            None,
        )

    try:
        refreshed_entries = prioritize_candidate_entries(
            live_entries,
            preferred_candidate,
            candidate_source_key=candidate_source_key,
        )
    except TypeError:
        refreshed_entries = prioritize_candidate_entries(
            live_entries,
            preferred_candidate,
        )
    visual_preservation_entries = list(existing_group_entries)
    if isinstance(stored_tagged_candidate, dict):
        visual_preservation_entries.append(dict(stored_tagged_candidate))
    refreshed_entries = _geometry_manual_preserve_caked_visual_fields_by_identity(
        refreshed_entries,
        existing_entries=visual_preservation_entries,
        group_key=group_key,
        candidate_source_key=candidate_source_key,
    )
    refreshed_session = dict(pick_session)
    if (
        isinstance(pick_session.get("_tagged_candidate_requires_identity"), bool)
        or stored_tagged_candidate is not None
        or tagged_candidate_key is not None
    ):
        refreshed_session["_tagged_candidate_requires_identity"] = bool(
            tagged_candidate_requires_identity
        )
    refreshed_session["group_entries"] = list(refreshed_entries)
    refreshed_session["target_count"] = int(group_target_count_fn(group_key, refreshed_entries))
    if cache_signature is not None:
        refreshed_session["cache_signature"] = cache_signature

    tagged_candidate = None
    if stored_tagged_candidate is not None:
        tagged_index = _geometry_manual_resolve_identity_candidate_index(
            stored_tagged_candidate,
            refreshed_entries,
        )
        if tagged_index is not None and 0 <= int(tagged_index) < len(refreshed_entries):
            tagged_candidate = dict(refreshed_entries[int(tagged_index)])
    if allow_tagged_key_fallback and tagged_candidate is None and tagged_candidate_key is not None:
        tagged_candidate = next(
            (
                dict(entry)
                for entry in refreshed_entries
                if candidate_source_key(entry) == tagged_candidate_key
            ),
            None,
        )
    if tagged_candidate is not None:
        preserved_tagged = _geometry_manual_preserve_caked_visual_fields_by_identity(
            [tagged_candidate],
            existing_entries=visual_preservation_entries,
            group_key=group_key,
            candidate_source_key=candidate_source_key,
        )
        if preserved_tagged:
            tagged_candidate = dict(preserved_tagged[0])
        refreshed_session["tagged_candidate"] = tagged_candidate
    elif "tagged_candidate" in refreshed_session:
        refreshed_session.pop("tagged_candidate", None)

    return refreshed_session


def geometry_manual_tagged_candidate_from_session(
    pick_session: dict[str, object] | None,
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[str, object] | None:
    """Return the exact tagged candidate for the active session when it remains available."""

    if not isinstance(pick_session, dict):
        return None

    stored_tagged_candidate = (
        dict(pick_session["tagged_candidate"])
        if isinstance(pick_session.get("tagged_candidate"), dict)
        else None
    )
    tagged_candidate_requires_identity = _geometry_manual_tagged_candidate_requires_identity(
        pick_session
    )
    tagged_key = pick_session.get("tagged_candidate_key")
    allow_tagged_key_fallback = _geometry_manual_allow_tagged_key_fallback(
        tagged_key if isinstance(tagged_key, tuple) else None,
        requires_identity=tagged_candidate_requires_identity,
        stored_tagged_candidate=stored_tagged_candidate,
    )
    if stored_tagged_candidate is not None:
        tagged_index = _geometry_manual_resolve_identity_candidate_index(
            stored_tagged_candidate,
            candidate_entries,
        )
        if tagged_index is not None:
            normalized_candidates = [
                dict(raw_entry)
                for raw_entry in (candidate_entries or ())
                if isinstance(raw_entry, dict)
            ]
            if 0 <= int(tagged_index) < len(normalized_candidates):
                return dict(normalized_candidates[int(tagged_index)])
        if not allow_tagged_key_fallback:
            return None
    if tagged_candidate_requires_identity:
        return None

    if tagged_key is None:
        tagged_key = (
            candidate_source_key(stored_tagged_candidate)
            if stored_tagged_candidate is not None
            else None
        )
    if tagged_key is None:
        return None

    for raw_entry in candidate_entries or []:
        if not isinstance(raw_entry, dict):
            continue
        if candidate_source_key(raw_entry) == tagged_key:
            return dict(raw_entry)
    return None


def current_geometry_manual_match_config(
    fit_config: dict[str, object] | None,
) -> dict[str, object]:
    """Return the refined background-peak matcher config for manual picking."""

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}

    manual_cfg = dict(auto_match_cfg)
    search_radius = max(1.0, float(manual_cfg.get("search_radius_px", 24.0)))
    manual_cfg["console_progress"] = False
    manual_cfg["relax_on_low_matches"] = False
    manual_cfg.setdefault("context_margin_px", max(96.0, 6.0 * search_radius))
    manual_cfg.setdefault("require_candidate_ownership", True)
    manual_cfg.setdefault("k_neighbors", 12)
    manual_cfg.setdefault("max_candidate_peaks", 1200)
    return manual_cfg


def geometry_manual_pick_placed_cache_signature(
    *,
    source_snapshot_signature: object,
    background_index: int,
    background_image: object | None,
    use_caked_space: bool,
    caked_projection_signature: object = None,
    include_caked_projection_signature: bool = False,
) -> tuple[object, ...]:
    """Return stable signature for placed manual-pick rows before Q-group masks."""

    bg_token = None
    if background_image is not None and not bool(use_caked_space):
        raw_arr = np.asarray(background_image)
        try:
            bg_ptr = int(raw_arr.__array_interface__["data"][0])
        except Exception:
            bg_ptr = int(id(raw_arr))
        bg_token = (
            bg_ptr,
            tuple(int(v) for v in raw_arr.shape),
            tuple(int(v) for v in raw_arr.strides),
            str(raw_arr.dtype),
        )

    caked_signature_token = None
    if bool(use_caked_space):
        caked_signature_token = _manual_pick_cache_jsonable(caked_projection_signature)
    elif bool(include_caked_projection_signature):
        caked_signature_token = (
            "detector_caked_projection_sidecar",
            _manual_pick_cache_jsonable(caked_projection_signature),
        )

    return (
        source_snapshot_signature,
        int(background_index),
        bool(use_caked_space),
        bg_token,
        caked_signature_token,
    )


def geometry_manual_pick_cache_signature(
    *,
    placed_cache_signature: object,
    disabled_qr_sets: Sequence[object] | None,
    disabled_qz_sections: Sequence[object] | None,
) -> tuple[object, ...]:
    """Return stable signature for filtered active manual-pick rows."""

    normalized_qr_sets = sorted(
        {
            parent_key
            for raw_key in disabled_qr_sets or ()
            if (parent_key := gui_controllers.qr_set_mask_key(raw_key)) is not None
        },
        key=lambda item: (str(item[0]), int(item[1])),
    )
    normalized_qz_sections = sorted(
        {
            child_key
            for raw_key in disabled_qz_sections or ()
            if (child_key := gui_controllers.qz_section_mask_key(raw_key)) is not None
        },
        key=lambda item: (str(item[0]), int(item[1]), int(item[2])),
    )
    return (
        placed_cache_signature,
        tuple(normalized_qr_sets),
        tuple(normalized_qz_sections),
    )


def _manual_pick_cache_placed_signature(signature: object) -> tuple[object, ...] | None:
    if not isinstance(signature, tuple):
        return None
    if (
        len(signature) >= 3
        and isinstance(signature[0], tuple)
        and not isinstance(signature[2], bool)
    ):
        placed_signature = signature[0]
        return placed_signature if isinstance(placed_signature, tuple) else None
    return signature


def _manual_pick_cache_source_signature_key(signature: object) -> tuple[object, object] | None:
    placed_signature = _manual_pick_cache_placed_signature(signature)
    if not isinstance(placed_signature, tuple) or len(placed_signature) < 2:
        return None
    return placed_signature[0], placed_signature[1]


def _manual_pick_cache_mask_signature_key(signature: object) -> tuple[object, object] | None:
    if (
        not isinstance(signature, tuple)
        or len(signature) < 3
        or not isinstance(signature[0], tuple)
        or isinstance(signature[2], bool)
    ):
        return None
    return tuple(signature[1] or ()), tuple(signature[2] or ())


def _manual_pick_cache_caked_projection_token(signature: object) -> object:
    placed_signature = _manual_pick_cache_placed_signature(signature)
    if not isinstance(placed_signature, tuple) or len(placed_signature) < 5:
        return None
    token = _manual_pick_cache_jsonable(placed_signature[4])
    if (
        isinstance(token, list)
        and len(token) >= 2
        and token[0] == "detector_caked_projection_sidecar"
    ):
        return token[1]
    if (
        isinstance(token, tuple)
        and len(token) >= 2
        and token[0] == "detector_caked_projection_sidecar"
    ):
        return _manual_pick_cache_jsonable(token[1])
    return token


def _manual_pick_cache_mask_signature_matches(
    existing_signature: object,
    current_signature: object,
) -> bool:
    existing_mask = _manual_pick_cache_mask_signature_key(existing_signature)
    current_mask = _manual_pick_cache_mask_signature_key(current_signature)
    if existing_mask is None or current_mask is None:
        return True
    return bool(existing_mask == current_mask)


def _manual_pick_cache_source_and_masks_reusable(
    existing_signature: object,
    current_signature: object,
) -> bool:
    existing_source = _manual_pick_cache_source_signature_key(existing_signature)
    current_source = _manual_pick_cache_source_signature_key(current_signature)
    return bool(
        existing_source is not None
        and existing_source == current_source
        and _manual_pick_cache_mask_signature_matches(existing_signature, current_signature)
    )


def _manual_pick_cache_has_caked_projection_sidecar(
    cache_data: Mapping[str, object] | None,
) -> bool:
    if not isinstance(cache_data, Mapping):
        return False
    grouped = cache_data.get("caked_qr_projection_grouped_candidates")
    if not isinstance(grouped, Mapping) or not grouped:
        return False
    return bool(
        cache_data.get("caked_qr_projection_lookup")
        or cache_data.get("caked_qr_projection_entries")
    )


def _manual_pick_cache_caked_sidecar_signature_compatible(
    existing_signature: object,
    current_signature: object,
) -> bool:
    current_token = _manual_pick_cache_caked_projection_token(current_signature)
    if current_token is None:
        return False
    return bool(_manual_pick_cache_caked_projection_token(existing_signature) == current_token)


def _manual_pick_cache_groups_reusable(
    existing_signature: object,
    current_signature: object,
) -> bool:
    """Return whether cached candidate groups remain valid for one snapshot."""

    if not isinstance(existing_signature, tuple) or not isinstance(current_signature, tuple):
        return False
    if existing_signature == current_signature:
        return True
    return _manual_pick_cache_source_signature_key(
        existing_signature
    ) == _manual_pick_cache_source_signature_key(current_signature)


def _manual_pick_cache_background_token_compatible(
    existing_signature: object,
    current_signature: object,
) -> bool:
    """Return true when only the background array identity changed."""

    if not _manual_pick_cache_groups_reusable(existing_signature, current_signature):
        return False
    if not isinstance(existing_signature, tuple) or not isinstance(current_signature, tuple):
        return False
    if len(existing_signature) < 4 or len(current_signature) < 4:
        return False
    existing_bg_token = existing_signature[3]
    current_bg_token = current_signature[3]
    if existing_bg_token == current_bg_token:
        return True
    if not isinstance(existing_bg_token, (list, tuple)) or not isinstance(
        current_bg_token,
        (list, tuple),
    ):
        return False
    if len(existing_bg_token) < 4 or len(current_bg_token) < 4:
        return False
    return bool(tuple(existing_bg_token[1:4]) == tuple(current_bg_token[1:4]))


def _geometry_manual_cache_signature_uses_caked(signature: object) -> bool:
    if not isinstance(signature, tuple):
        return False
    if len(signature) >= 3 and isinstance(signature[2], bool):
        return bool(signature[2])
    placed_signature = signature[0] if signature else None
    return bool(
        isinstance(placed_signature, tuple)
        and len(placed_signature) >= 3
        and isinstance(placed_signature[2], bool)
        and bool(placed_signature[2])
    )


def _manual_pick_cache_normalized_hkl(
    value: object,
) -> tuple[int, int, int] | None:
    """Return one normalized HKL triplet when the value is usable."""

    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except Exception:
        return None


def _manual_pick_cache_jsonable(value: object) -> object:
    """Convert one cache metadata value into a stable JSON-safe shape."""

    if isinstance(value, tuple):
        return [_manual_pick_cache_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_manual_pick_cache_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_manual_pick_cache_jsonable(item) for item in value.tolist()]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _manual_pick_cache_finite_float(
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


def _manual_pick_cache_table_summaries(
    simulated_peaks: Sequence[dict[str, object]] | None,
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
) -> list[dict[str, object]]:
    """Summarize raw-to-grouped manual-pick cache rows per source table."""

    raw_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)
    grouped_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)

    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            table_idx = int(raw_entry.get("source_table_index"))
        except Exception:
            continue
        raw_by_table[int(table_idx)].append(dict(raw_entry))

    for entries in (grouped_candidates or {}).values():
        if not isinstance(entries, Sequence):
            continue
        for raw_entry in entries or ():
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                table_idx = int(raw_entry.get("source_table_index"))
            except Exception:
                continue
            grouped_by_table[int(table_idx)].append(dict(raw_entry))

    summaries: list[dict[str, object]] = []
    for table_idx in sorted(raw_by_table):
        raw_entries = list(raw_by_table.get(table_idx, ()))
        grouped_entries = list(grouped_by_table.get(table_idx, ()))
        representative_entries = grouped_entries or raw_entries
        representative = (
            dict(representative_entries[0])
            if representative_entries and isinstance(representative_entries[0], Mapping)
            else {}
        )
        representative_rows: list[int] = []
        for entry in grouped_entries:
            try:
                representative_rows.append(int(entry.get("source_row_index")))
            except Exception:
                continue
        representative_rows = sorted(set(representative_rows))
        dropped_nonfinite = 0
        for entry in raw_entries:
            if _geometry_manual_entry_current_view_point(entry) is None:
                dropped_nonfinite += 1
        nominal_hkl_recovery_count = sum(
            1 for entry in raw_entries if bool(entry.get("q_group_nominal_hkl", False))
        )
        row_count_before = int(len(raw_entries))
        row_count_after = int(len(grouped_entries))
        merged_group_count = max(
            0,
            row_count_before - row_count_after - int(dropped_nonfinite),
        )
        nominal_hkl = _manual_pick_cache_normalized_hkl(representative.get("hkl"))
        summaries.append(
            {
                "source_table_index": int(table_idx),
                "nominal_hkl": (list(nominal_hkl) if isinstance(nominal_hkl, tuple) else None),
                "q_group_key": _manual_pick_cache_jsonable(representative.get("q_group_key")),
                "qr": _manual_pick_cache_finite_float(representative.get("qr")),
                "qz": _manual_pick_cache_finite_float(representative.get("qz")),
                "row_count_before_grouping": int(row_count_before),
                "row_count_after_grouping": int(row_count_after),
                "dropped_nonfinite_row_count": int(dropped_nonfinite),
                "nominal_hkl_recovery_count": int(nominal_hkl_recovery_count),
                "merged_group_count": int(merged_group_count),
                "representative_row_indices_kept": [int(idx) for idx in representative_rows],
            }
        )
    return summaries


def _manual_pick_cache_metadata(
    *,
    cache_action: str,
    stale_reason: str | None,
    cache_source: str,
    cache_provenance: Sequence[str] | None,
    simulated_peaks: Sequence[dict[str, object]] | None,
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
    background_index: int,
    current_background_index: int,
    prefer_cache: bool,
) -> dict[str, object]:
    """Build the structured cache metadata attached to manual-pick cache payloads."""

    table_summaries = _manual_pick_cache_table_summaries(
        simulated_peaks,
        grouped_candidates,
    )
    return {
        "cache_action": str(cache_action),
        "reused": bool(str(cache_action).lower() == "reused"),
        "rebuilt": bool(str(cache_action).lower() == "rebuilt"),
        "stale_reason": None if stale_reason is None else str(stale_reason),
        "cache_source": str(cache_source),
        "cache_provenance": [str(step) for step in (cache_provenance or ())],
        "prefer_cache": bool(prefer_cache),
        "background_index": int(background_index),
        "current_background_index": int(current_background_index),
        "simulated_peak_count": int(
            len([entry for entry in simulated_peaks or () if isinstance(entry, Mapping)])
        ),
        "group_count": int(len(grouped_candidates or {})),
        "table_count": int(len(table_summaries)),
        "table_summaries": table_summaries,
    }


def _geometry_manual_elapsed_ms(start: float) -> float:
    """Return a finite non-negative elapsed time in milliseconds."""

    try:
        elapsed = 1000.0 * (perf_counter() - float(start))
    except Exception:
        return 0.0
    if not np.isfinite(elapsed):
        return 0.0
    return float(max(0.0, elapsed))


def _geometry_manual_cache_metadata_update(
    cache_data: Mapping[str, object] | None,
    **fields: object,
) -> dict[str, object]:
    """Return cache metadata updated with optional diagnostic fields."""

    cache = cache_data if isinstance(cache_data, Mapping) else {}
    metadata = (
        dict(cache.get("cache_metadata"))
        if isinstance(cache.get("cache_metadata"), Mapping)
        else {}
    )
    metadata.update(fields)
    return metadata


def geometry_manual_no_build_cache_sentinel(
    *,
    match_config: Mapping[str, object] | None = None,
    manual_background_reference: bool = False,
    stale_reason: str = "manual picker cache build disabled for this path",
) -> dict[str, object]:
    """Return non-empty cache data that forbids fallback cache construction."""

    metadata = {
        "cache_action": "miss",
        "cache_ready": False,
        "allow_cache_build": False,
        "stale_reason": str(stale_reason),
    }
    sentinel: dict[str, object] = {
        "cache_ready": False,
        "manual_no_build_cache": True,
        "match_config": dict(match_config or {}),
        "cache_metadata": metadata,
    }
    if bool(manual_background_reference):
        sentinel["manual_background_reference"] = True
        metadata["manual_background_reference"] = True
    return sentinel


def geometry_manual_background_reference_refine_cache(
    display_background: object | None,
    *,
    get_cache_data: Callable[..., dict[str, object]] | None = None,
    current_match_config: Callable[[], Mapping[str, object]] | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return local-only cache data for background QR reference refinement."""

    del display_background, get_cache_data
    try:
        raw_match_config = (
            current_match_config() if callable(current_match_config) else current_match_config
        )
    except Exception:
        raw_match_config = None
    return geometry_manual_no_build_cache_sentinel(
        match_config=raw_match_config if isinstance(raw_match_config, Mapping) else {},
        manual_background_reference=True,
        stale_reason="background QR reference uses local refinement without picker cache",
    )


def _geometry_manual_detector_shape(
    image: object | None,
) -> tuple[int, int]:
    try:
        arr = np.asarray(image)
    except Exception:
        return 0, 0
    if arr.ndim < 2:
        return 0, 0
    return int(arr.shape[0]), int(arr.shape[1])


def _geometry_manual_point_inside_bounds(
    point: tuple[float, float] | None,
    *,
    width: int,
    height: int,
) -> bool:
    if point is None or int(width) <= 0 or int(height) <= 0:
        return False
    return bool(0.0 <= float(point[0]) < float(width) and 0.0 <= float(point[1]) < float(height))


def _geometry_manual_detector_point_is_caked(
    entry: Mapping[str, object] | None,
    point: tuple[float, float] | None,
) -> bool:
    if not isinstance(entry, Mapping) or point is None:
        return False
    if _geometry_manual_entry_display_frame(entry) == "detector":
        return False
    for key_pair in (
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
        ("two_theta_deg", "phi_deg"),
        ("refined_sim_caked_x", "refined_sim_caked_y"),
        ("background_two_theta_deg", "background_phi_deg"),
        ("simulated_two_theta_deg", "simulated_phi_deg"),
    ):
        caked_point = _geometry_manual_finite_point(entry, (key_pair,))
        if caked_point is not None and np.allclose(
            point,
            caked_point,
            atol=1.0e-9,
            rtol=0.0,
        ):
            return True
    return False


def _geometry_manual_local_maximum_near_in_image(
    image: object | None,
    col: float,
    row: float,
    *,
    search_radius: int = 5,
) -> tuple[tuple[float, float] | None, str]:
    if image is None:
        return None, "no_sim_image"
    try:
        image_arr = np.asarray(image, dtype=float)
    except Exception:
        return None, "no_sim_image"
    if image_arr.ndim != 2 or image_arr.size <= 0:
        return None, "no_sim_image"
    try:
        col_val = float(col)
        row_val = float(row)
    except Exception:
        return None, "outside_window"
    if not (np.isfinite(col_val) and np.isfinite(row_val)):
        return None, "outside_window"
    height = int(image_arr.shape[0])
    width = int(image_arr.shape[1])
    c = int(round(col_val))
    r = int(round(row_val))
    if c < 0 or c >= width or r < 0 or r >= height:
        return None, "outside_window"
    radius = max(0, int(round(float(search_radius))))
    r0 = max(0, r - radius)
    r1 = min(height, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(width, c + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return None, "outside_window"
    window = image_arr[r0:r1, c0:c1]
    if window.size == 0 or not np.isfinite(window).any():
        return None, "no_peak_found"
    finite_window = window[np.isfinite(window)]
    if finite_window.size == 0:
        return None, "no_peak_found"
    if not (float(np.nanmax(finite_window)) > float(np.nanmin(finite_window))):
        return None, "no_peak_found"
    max_idx = int(np.nanargmax(window))
    win_r, win_c = np.unravel_index(max_idx, window.shape)
    return (float(c0 + win_c), float(r0 + win_r)), "refined"


def _geometry_manual_set_tuple_point(
    entry: dict[str, object],
    key: str,
    point: tuple[float, float] | None,
) -> None:
    if point is None:
        return
    entry[key] = (float(point[0]), float(point[1]))


def _geometry_manual_valid_tuple_point(value: object) -> tuple[float, float] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        if len(value) < 2:  # type: ignore[arg-type]
            return None
        col = float(value[0])  # type: ignore[index]
        row = float(value[1])  # type: ignore[index]
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _geometry_manual_callback_point(
    callback: Callable[[float, float], object] | None,
    point: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point is None or not callable(callback):
        return None
    try:
        return _geometry_manual_valid_tuple_point(callback(float(point[0]), float(point[1])))
    except Exception:
        return None


def _geometry_manual_set_sim_visual_caked(
    entry: dict[str, object],
    *,
    refined_caked: tuple[float, float] | None,
    nominal_caked: tuple[float, float] | None,
) -> None:
    visual_caked = refined_caked if refined_caked is not None else nominal_caked
    if visual_caked is None:
        return
    entry["sim_visual_deg"] = (float(visual_caked[0]), float(visual_caked[1]))
    entry["sim_visual_caked_deg"] = (float(visual_caked[0]), float(visual_caked[1]))
    entry["sim_caked"] = (float(visual_caked[0]), float(visual_caked[1]))
    entry["sim_visual_source"] = (
        "sim_refined_caked_deg" if refined_caked is not None else "sim_nominal_caked_deg"
    )


def _geometry_manual_detector_display_to_native(
    detector_display_to_native_coords: Callable[[float, float], object] | None,
    display_point: tuple[float, float] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_callback_point(detector_display_to_native_coords, display_point)


def _geometry_manual_native_to_caked(
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object] | None,
    native_point: tuple[float, float] | None,
) -> tuple[float, float] | None:
    return _geometry_manual_callback_point(
        native_detector_coords_to_caked_display_coords,
        native_point,
    )


def _geometry_manual_apply_sim_visual_detector_fields(
    entry: dict[str, object],
    *,
    detector_display_to_native_coords: Callable[[float, float], object] | None = None,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object] | None = None,
    display_point: tuple[float, float] | None = None,
) -> None:
    """Keep detector-origin sim visual display/native fields in detector coords."""

    if not isinstance(entry, dict):
        return
    if display_point is None:
        display_point = _geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_display_px",
        )
    if display_point is None:
        display_point = _geometry_manual_tuple_point(
            entry,
            "sim_refined_detector_display_px",
        )
    if display_point is None:
        display_point = _geometry_manual_finite_point(
            entry,
            (
                ("refined_sim_x", "refined_sim_y"),
                ("sim_col_raw", "sim_row_raw"),
                ("sim_col", "sim_row"),
                ("display_col", "display_row"),
            ),
        )
    display_point = _geometry_manual_valid_tuple_point(display_point)
    if display_point is not None:
        entry["sim_visual_detector_display_px"] = (
            float(display_point[0]),
            float(display_point[1]),
        )

    existing_native = _geometry_manual_tuple_point(
        entry,
        "sim_visual_detector_native_existing",
    )
    if existing_native is None:
        existing_native = _geometry_manual_tuple_point(
            entry,
            "sim_visual_detector_native_px",
        )
    if existing_native is None:
        existing_native = _geometry_manual_tuple_point(
            entry,
            "sim_refined_detector_native_px",
        )
    if existing_native is None:
        existing_native = _geometry_manual_finite_point(
            entry,
            (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("sim_native_x", "sim_native_y"),
                ("native_col", "native_row"),
            ),
        )
    if existing_native is not None:
        entry["sim_visual_detector_native_existing"] = (
            float(existing_native[0]),
            float(existing_native[1]),
        )

    derived_native = _geometry_manual_detector_display_to_native(
        detector_display_to_native_coords,
        display_point,
    )
    if derived_native is not None:
        entry["sim_visual_detector_native_px"] = (
            float(derived_native[0]),
            float(derived_native[1]),
        )
        entry["sim_visual_detector_native_source"] = "display_to_native_callback"
        entry["sim_refined_detector_native_px"] = (
            float(derived_native[0]),
            float(derived_native[1]),
        )
        entry["refined_sim_native_x"] = float(derived_native[0])
        entry["refined_sim_native_y"] = float(derived_native[1])
    elif existing_native is not None and "sim_visual_detector_native_px" not in entry:
        entry["sim_visual_detector_native_unavailable_reason"] = (
            "display_to_native_callback_unavailable"
        )

    visual_native = _geometry_manual_tuple_point(entry, "sim_visual_detector_native_px")
    visual_caked = _geometry_manual_native_to_caked(
        native_detector_coords_to_caked_display_coords,
        visual_native,
    )
    if visual_caked is not None:
        entry["sim_visual_caked_deg"] = (
            float(visual_caked[0]),
            float(visual_caked[1]),
        )
        entry["sim_refined_caked_deg"] = (
            float(visual_caked[0]),
            float(visual_caked[1]),
        )
        entry["sim_caked"] = (float(visual_caked[0]), float(visual_caked[1]))
        entry["sim_visual_deg"] = (float(visual_caked[0]), float(visual_caked[1]))
        entry["sim_visual_caked_source"] = "sim_visual_detector_native_px"
        entry["refined_sim_caked_x"] = float(visual_caked[0])
        entry["refined_sim_caked_y"] = float(visual_caked[1])


def _geometry_manual_apply_detector_origin_conversion_ledger(
    entry: dict[str, object],
    *,
    background_display_to_native_detector_coords: Callable[[float, float], object] | None,
    native_detector_coords_to_caked_display_coords: Callable[[float, float], object] | None,
) -> None:
    if not isinstance(entry, dict):
        return
    origin_token = _geometry_manual_normalized_frame_token(
        entry.get("manual_background_input_origin")
    )
    saved_caked_point = _geometry_manual_finite_point(
        entry,
        (
            ("background_two_theta_deg", "background_phi_deg"),
            ("caked_x", "caked_y"),
        ),
    )
    detector_origin = bool(origin_token == "detector" or saved_caked_point is None)

    raw_display = _geometry_manual_tuple_point(entry, "raw_detector_display_px")
    if raw_display is None:
        raw_display = _geometry_manual_finite_point(
            entry,
            (
                ("detector_seed_col", "detector_seed_row"),
                ("raw_x", "raw_y"),
            ),
        )
    geometry_display = _geometry_manual_tuple_point(
        entry,
        "geometry_detector_display_px",
    )
    if geometry_display is None:
        geometry_display = _geometry_manual_finite_point(
            entry,
            (
                ("refined_detector_display_col", "refined_detector_display_row"),
                ("x", "y"),
                ("display_col", "display_row"),
            ),
        )
    if raw_display is not None:
        entry["raw_detector_display_px"] = (
            float(raw_display[0]),
            float(raw_display[1]),
        )
    if geometry_display is not None:
        entry["geometry_detector_display_px"] = (
            float(geometry_display[0]),
            float(geometry_display[1]),
        )

    raw_native = _geometry_manual_tuple_point(entry, "raw_detector_native_px")
    raw_native_source = "saved_native"
    if raw_native is None:
        raw_native = _geometry_manual_detector_display_to_native(
            background_display_to_native_detector_coords,
            raw_display,
        )
        raw_native_source = "display_to_native_callback"
    geometry_native = _geometry_manual_tuple_point(entry, "geometry_detector_native_px")
    geometry_native_source = "saved_native"
    if geometry_native is None:
        geometry_native = _geometry_manual_finite_point(
            entry,
            (
                ("refined_detector_native_col", "refined_detector_native_row"),
                ("background_detector_x", "background_detector_y"),
                ("detector_native_col", "detector_native_row"),
                ("detector_x", "detector_y"),
                ("native_col", "native_row"),
            ),
        )
    if geometry_native is None:
        geometry_native = _geometry_manual_detector_display_to_native(
            background_display_to_native_detector_coords,
            geometry_display,
        )
        geometry_native_source = "display_to_native_callback"
    if raw_native is not None:
        entry["raw_detector_native_px"] = (float(raw_native[0]), float(raw_native[1]))
        entry["raw_detector_native_source"] = raw_native_source
    if geometry_native is not None:
        entry["geometry_detector_native_px"] = (
            float(geometry_native[0]),
            float(geometry_native[1]),
        )
        entry["geometry_detector_native_source"] = geometry_native_source
        entry["refined_detector_native_col"] = float(geometry_native[0])
        entry["refined_detector_native_row"] = float(geometry_native[1])

    raw_caked = _geometry_manual_native_to_caked(
        native_detector_coords_to_caked_display_coords,
        raw_native,
    )
    geometry_caked = _geometry_manual_native_to_caked(
        native_detector_coords_to_caked_display_coords,
        geometry_native,
    )
    if raw_caked is not None:
        entry["raw_caked_deg"] = (float(raw_caked[0]), float(raw_caked[1]))
        if (
            detector_origin
            or _geometry_manual_finite_point(
                entry,
                (("raw_caked_x", "raw_caked_y"),),
            )
            is None
        ):
            entry["raw_caked_x"] = float(raw_caked[0])
            entry["raw_caked_y"] = float(raw_caked[1])
            entry["raw_caked_two_theta_deg"] = float(raw_caked[0])
            entry["raw_caked_phi_deg"] = float(raw_caked[1])
    if geometry_caked is not None:
        entry["geometry_caked_deg"] = (
            float(geometry_caked[0]),
            float(geometry_caked[1]),
        )
        if detector_origin:
            entry["refined_background_two_theta_deg"] = float(geometry_caked[0])
            entry["refined_background_phi_deg"] = float(geometry_caked[1])
            entry["background_two_theta_deg"] = float(geometry_caked[0])
            entry["background_phi_deg"] = float(geometry_caked[1])
            entry["caked_x"] = float(geometry_caked[0])
            entry["caked_y"] = float(geometry_caked[1])
            entry["two_theta_deg"] = float(geometry_caked[0])
            entry["phi_deg"] = float(geometry_caked[1])


def _geometry_manual_caked_refinement_window_status(
    image: np.ndarray,
    radial_axis: Sequence[float],
    azimuth_axis: Sequence[float],
    two_theta_deg: float,
    phi_deg: float,
) -> str:
    try:
        img = np.asarray(image, dtype=float)
        radial = np.asarray(radial_axis, dtype=float)
        azimuth = np.asarray(azimuth_axis, dtype=float)
    except Exception:
        return "no_sim_image"
    if img.ndim != 2 or img.size <= 0 or radial.size <= 0 or azimuth.size <= 0:
        return "no_sim_image"
    col_seed = caked_axis_to_image_index(float(two_theta_deg), radial)
    row_seed = caked_axis_to_image_index(float(phi_deg), azimuth)
    if not (np.isfinite(col_seed) and np.isfinite(row_seed)):
        return "outside_window"
    col_min = caked_axis_to_image_index(
        float(two_theta_deg) - float(DEFAULT_CAKED_SEARCH_TTH_DEG),
        radial,
    )
    col_max = caked_axis_to_image_index(
        float(two_theta_deg) + float(DEFAULT_CAKED_SEARCH_TTH_DEG),
        radial,
    )
    row_min = caked_axis_to_image_index(
        float(phi_deg) - float(DEFAULT_CAKED_SEARCH_PHI_DEG),
        azimuth,
    )
    row_max = caked_axis_to_image_index(
        float(phi_deg) + float(DEFAULT_CAKED_SEARCH_PHI_DEG),
        azimuth,
    )
    if not all(np.isfinite(v) for v in (col_min, col_max, row_min, row_max)):
        return "outside_window"
    c0 = max(0, int(np.floor(min(col_min, col_max))))
    c1 = min(int(img.shape[1]), int(np.ceil(max(col_min, col_max))) + 1)
    r0 = max(0, int(np.floor(min(row_min, row_max))))
    r1 = min(int(img.shape[0]), int(np.ceil(max(row_min, row_max))) + 1)
    if c0 >= c1 or r0 >= r1:
        return "outside_window"
    patch = np.asarray(img[r0:r1, c0:c1], dtype=float)
    if patch.size <= 0 or not np.isfinite(patch).any():
        return "unavailable_zero_intensity_window"
    finite_patch = patch[np.isfinite(patch)]
    if finite_patch.size == 0:
        return "unavailable_zero_intensity_window"
    local_positive = np.where(np.isfinite(patch) & (patch > 0.0), patch, 0.0)
    if (
        int(np.count_nonzero(local_positive > 0.0)) <= 0
        or not (float(np.sum(local_positive, dtype=float)) > 0.0)
        or not (float(np.max(local_positive)) > 0.0)
    ):
        return "unavailable_zero_intensity_window"
    if not (float(np.nanmax(finite_patch)) > float(np.nanmin(finite_patch))):
        return "no_peak_found"
    baseline = float(np.nanpercentile(patch, 35.0))
    signal = np.clip(patch - baseline, 0.0, None)
    if not np.any(signal > 0.0):
        signal = np.clip(patch - float(np.nanmin(patch)), 0.0, None)
    if not np.any(signal > 0.0):
        return "no_peak_found"
    return "ready"


def _geometry_manual_projection_callback_name(callback: object) -> str:
    name = str(getattr(callback, "__name__", "") or "").strip()
    module = str(getattr(callback, "__module__", "") or "").strip()
    if module and name:
        return f"{module}.{name}"
    if name:
        return name
    return type(callback).__name__


def _geometry_manual_projection_callback_is_real(callback: object) -> bool:
    if not callable(callback):
        return False
    name = str(getattr(callback, "__name__", "") or "").strip().lower()
    qualname = str(getattr(callback, "__qualname__", "") or "").strip().lower()
    module = str(getattr(callback, "__module__", "") or "").strip().lower()
    callback_text = " ".join((name, qualname, module))
    if not module or name == "<lambda>" or "lambda" in qualname:
        return False
    if "test" in callback_text or "fake" in callback_text or "dummy" in callback_text:
        return False
    return bool(module.startswith("ra_sim."))


def _geometry_manual_nominal_detector_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    point = _geometry_manual_tuple_point(entry, "sim_nominal_detector_display_px")
    if point is not None:
        return point
    return _geometry_manual_entry_detector_display_point(entry)


def _geometry_manual_nominal_native_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    point = _geometry_manual_tuple_point(entry, "sim_nominal_detector_native_px")
    if point is not None:
        return point
    return _geometry_manual_entry_native_point(entry)


def _geometry_manual_nominal_caked_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    point = _geometry_manual_tuple_point(entry, "sim_nominal_caked_deg")
    if point is not None:
        return point
    return _geometry_manual_finite_point(
        entry,
        (
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
    )


def geometry_manual_refine_qr_sim_peak_detector(
    candidate: Mapping[str, object] | None,
    *,
    detector_simulation_image: object | None = None,
    search_radius_px: int = 5,
    detector_display_to_native_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
) -> dict[str, object] | None:
    if not isinstance(candidate, Mapping):
        return None
    entry = dict(candidate)
    nominal_display = _geometry_manual_nominal_detector_display_point(entry)
    nominal_native = _geometry_manual_nominal_native_point(entry)
    nominal_caked = _geometry_manual_nominal_caked_point(entry)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_detector_display_px", nominal_display)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_detector_native_px", nominal_native)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_caked_deg", nominal_caked)
    if nominal_display is None:
        entry["sim_refinement_status"] = "outside_window"
        _geometry_manual_set_sim_visual_caked(
            entry,
            refined_caked=None,
            nominal_caked=nominal_caked,
        )
        return entry

    refined_display, status = _geometry_manual_local_maximum_near_in_image(
        detector_simulation_image,
        float(nominal_display[0]),
        float(nominal_display[1]),
        search_radius=int(search_radius_px),
    )
    entry["sim_refinement_source"] = "detector_simulation_image"
    entry["sim_refinement_status"] = str(status)
    if refined_display is None:
        _geometry_manual_set_sim_visual_caked(
            entry,
            refined_caked=None,
            nominal_caked=nominal_caked,
        )
        return entry

    entry["sim_refined_detector_display_px"] = (
        float(refined_display[0]),
        float(refined_display[1]),
    )
    entry["sim_visual_detector_display_px"] = (
        float(refined_display[0]),
        float(refined_display[1]),
    )
    entry["refined_sim_x"] = float(refined_display[0])
    entry["refined_sim_y"] = float(refined_display[1])
    entry["sim_refinement_delta_detector_px"] = float(
        np.hypot(
            float(refined_display[0]) - float(nominal_display[0]),
            float(refined_display[1]) - float(nominal_display[1]),
        )
    )

    refined_native = None
    if callable(detector_display_to_native_coords):
        try:
            native_candidate = detector_display_to_native_coords(
                float(refined_display[0]),
                float(refined_display[1]),
            )
        except Exception:
            native_candidate = None
        if (
            isinstance(native_candidate, tuple)
            and len(native_candidate) >= 2
            and np.isfinite(float(native_candidate[0]))
            and np.isfinite(float(native_candidate[1]))
        ):
            refined_native = (float(native_candidate[0]), float(native_candidate[1]))
    elif nominal_native is not None and np.allclose(
        refined_display,
        nominal_display,
        atol=1.0e-9,
        rtol=0.0,
    ):
        refined_native = nominal_native
    if refined_native is not None:
        entry["sim_refined_detector_native_px"] = (
            float(refined_native[0]),
            float(refined_native[1]),
        )
        entry["refined_sim_native_x"] = float(refined_native[0])
        entry["refined_sim_native_y"] = float(refined_native[1])

    _geometry_manual_apply_sim_visual_detector_fields(
        entry,
        detector_display_to_native_coords=detector_display_to_native_coords,
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
        display_point=(float(refined_display[0]), float(refined_display[1])),
    )
    refined_native = _geometry_manual_tuple_point(entry, "sim_visual_detector_native_px")

    refined_caked = None
    if refined_native is not None and callable(native_detector_coords_to_caked_display_coords):
        callback_name = _geometry_manual_projection_callback_name(
            native_detector_coords_to_caked_display_coords
        )
        callback_real = _geometry_manual_projection_callback_is_real(
            native_detector_coords_to_caked_display_coords
        )
        entry["sim_refined_caked_projection_callback"] = callback_name
        entry["sim_refined_caked_projection_real_callback"] = bool(callback_real)
        entry["sim_refined_caked_projection_status"] = (
            "real_callback" if callback_real else "fake_or_test_callback"
        )
        try:
            caked_candidate = native_detector_coords_to_caked_display_coords(
                float(refined_native[0]),
                float(refined_native[1]),
            )
        except Exception:
            caked_candidate = None
        if (
            isinstance(caked_candidate, tuple)
            and len(caked_candidate) >= 2
            and np.isfinite(float(caked_candidate[0]))
            and np.isfinite(float(caked_candidate[1]))
        ):
            refined_caked = (float(caked_candidate[0]), float(caked_candidate[1]))
    elif refined_native is not None:
        entry["sim_refined_caked_projection_status"] = "projection_callback_unavailable"
    if refined_caked is not None:
        entry["sim_refined_caked_deg"] = (
            float(refined_caked[0]),
            float(refined_caked[1]),
        )
        entry["refined_sim_caked_x"] = float(refined_caked[0])
        entry["refined_sim_caked_y"] = float(refined_caked[1])
        if nominal_caked is not None:
            entry["sim_refinement_delta_caked_deg"] = float(
                np.hypot(
                    float(refined_caked[0]) - float(nominal_caked[0]),
                    float(refined_caked[1]) - float(nominal_caked[1]),
                )
            )
    _geometry_manual_set_sim_visual_caked(
        entry,
        refined_caked=refined_caked,
        nominal_caked=nominal_caked,
    )
    return entry


def geometry_manual_refine_qr_sim_peak_caked(
    candidate: Mapping[str, object] | None,
    *,
    caked_simulation_image: object | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_angles_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    detector_display_to_native_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
) -> dict[str, object] | None:
    if not isinstance(candidate, Mapping):
        return None
    entry = dict(candidate)
    nominal_caked = _geometry_manual_nominal_caked_point(entry)
    nominal_display = _geometry_manual_nominal_detector_display_point(entry)
    nominal_native = _geometry_manual_nominal_native_point(entry)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_caked_deg", nominal_caked)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_detector_display_px", nominal_display)
    _geometry_manual_set_tuple_point(entry, "sim_nominal_detector_native_px", nominal_native)
    entry["sim_refinement_source"] = "caked_simulation_image"
    if nominal_caked is None:
        entry["sim_refinement_status"] = "outside_window"
        return entry
    try:
        caked_image = np.asarray(caked_simulation_image, dtype=float)
        radial = np.asarray(radial_axis, dtype=float)
        azimuth = np.asarray(azimuth_axis, dtype=float)
    except Exception:
        caked_image = np.asarray([])
        radial = np.asarray([])
        azimuth = np.asarray([])
    if caked_image.ndim != 2 or caked_image.size <= 0 or radial.size <= 0 or azimuth.size <= 0:
        entry["sim_refinement_status"] = "no_sim_image"
        _geometry_manual_set_sim_visual_caked(
            entry,
            refined_caked=None,
            nominal_caked=nominal_caked,
        )
        return entry

    window_status = _geometry_manual_caked_refinement_window_status(
        caked_image,
        radial,
        azimuth,
        float(nominal_caked[0]),
        float(nominal_caked[1]),
    )
    if window_status != "ready":
        entry["sim_refinement_status"] = str(window_status)
        _geometry_manual_set_sim_visual_caked(
            entry,
            refined_caked=None,
            nominal_caked=nominal_caked,
        )
        return entry

    refined_caked = None
    try:
        refined_candidate = refine_caked_peak_center_fn(
            caked_image,
            radial,
            azimuth,
            float(nominal_caked[0]),
            float(nominal_caked[1]),
        )
    except Exception:
        refined_candidate = None
    if (
        isinstance(refined_candidate, tuple)
        and len(refined_candidate) >= 2
        and np.isfinite(float(refined_candidate[0]))
        and np.isfinite(float(refined_candidate[1]))
    ):
        refined_caked = (float(refined_candidate[0]), float(refined_candidate[1]))
    if refined_caked is None:
        entry["sim_refinement_status"] = "no_peak_found"
        _geometry_manual_set_sim_visual_caked(
            entry,
            refined_caked=None,
            nominal_caked=nominal_caked,
        )
        return entry

    entry["sim_refinement_status"] = "refined"
    entry["sim_refined_caked_deg"] = (
        float(refined_caked[0]),
        float(refined_caked[1]),
    )
    entry["sim_refined_caked_projection_status"] = "caked_simulation_image_axes"
    entry["sim_refined_caked_projection_real_callback"] = False
    entry["refined_sim_caked_x"] = float(refined_caked[0])
    entry["refined_sim_caked_y"] = float(refined_caked[1])
    entry["sim_refinement_delta_caked_deg"] = float(
        np.hypot(
            float(refined_caked[0]) - float(nominal_caked[0]),
            float(refined_caked[1]) - float(nominal_caked[1]),
        )
    )
    refined_display = None
    if callable(caked_angles_to_detector_display_coords):
        try:
            display_candidate = caked_angles_to_detector_display_coords(
                float(refined_caked[0]),
                float(refined_caked[1]),
            )
        except Exception:
            display_candidate = None
        if (
            isinstance(display_candidate, tuple)
            and len(display_candidate) >= 2
            and display_candidate[0] is not None
            and display_candidate[1] is not None
            and np.isfinite(float(display_candidate[0]))
            and np.isfinite(float(display_candidate[1]))
        ):
            refined_display = (float(display_candidate[0]), float(display_candidate[1]))
    if refined_display is not None:
        entry["sim_refined_detector_display_px"] = (
            float(refined_display[0]),
            float(refined_display[1]),
        )
        entry["sim_visual_detector_display_px"] = (
            float(refined_display[0]),
            float(refined_display[1]),
        )
        entry["refined_sim_x"] = float(refined_display[0])
        entry["refined_sim_y"] = float(refined_display[1])
        if nominal_display is not None:
            entry["sim_refinement_delta_detector_px"] = float(
                np.hypot(
                    float(refined_display[0]) - float(nominal_display[0]),
                    float(refined_display[1]) - float(nominal_display[1]),
                )
            )
        if callable(detector_display_to_native_coords):
            try:
                native_candidate = detector_display_to_native_coords(
                    float(refined_display[0]),
                    float(refined_display[1]),
                )
            except Exception:
                native_candidate = None
            if (
                isinstance(native_candidate, tuple)
                and len(native_candidate) >= 2
                and np.isfinite(float(native_candidate[0]))
                and np.isfinite(float(native_candidate[1]))
            ):
                entry["sim_refined_detector_native_px"] = (
                    float(native_candidate[0]),
                    float(native_candidate[1]),
                )
                entry["refined_sim_native_x"] = float(native_candidate[0])
                entry["refined_sim_native_y"] = float(native_candidate[1])
    _geometry_manual_set_sim_visual_caked(
        entry,
        refined_caked=refined_caked,
        nominal_caked=nominal_caked,
    )
    return entry


def geometry_manual_refine_qr_sim_peak_for_view(
    candidate: Mapping[str, object] | None,
    *,
    view_mode: str,
    detector_simulation_image: object | None = None,
    caked_simulation_image: object | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    detector_display_to_native_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    caked_angles_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
    search_radius_px: int = 5,
) -> dict[str, object] | None:
    if str(view_mode).strip().lower() == "caked":
        return geometry_manual_refine_qr_sim_peak_caked(
            candidate,
            caked_simulation_image=caked_simulation_image,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
            detector_display_to_native_coords=detector_display_to_native_coords,
            refine_caked_peak_center_fn=refine_caked_peak_center_fn,
        )
    return geometry_manual_refine_qr_sim_peak_detector(
        candidate,
        detector_simulation_image=detector_simulation_image,
        search_radius_px=search_radius_px,
        detector_display_to_native_coords=detector_display_to_native_coords,
        native_detector_coords_to_caked_display_coords=native_detector_coords_to_caked_display_coords,
    )


def _geometry_manual_generation_token(value: object) -> object:
    """Return a cheap generation/revision token when one is carried by an object."""

    generation_keys = (
        "generation",
        "generation_id",
        "revision",
        "revision_id",
        "version",
        "cache_generation",
        "cache_revision",
        "cache_version",
        "simulation_generation",
        "simulation_revision",
        "simulation_version",
        "update_counter",
        "counter",
        "sequence",
        "source_snapshot_generation",
        "source_snapshot_revision",
    )
    if isinstance(value, Mapping):
        return tuple((key, value.get(key)) for key in generation_keys if value.get(key) is not None)
    tokens: list[tuple[str, object]] = []
    for key in generation_keys:
        try:
            attr_value = getattr(value, key)
        except Exception:
            continue
        if attr_value is not None:
            tokens.append((key, attr_value))
    return tuple(tokens)


def _geometry_manual_scalar_token(value: object) -> tuple[object, object] | None:
    if value is None or isinstance(value, (bool, int, str)):
        return (type(value).__name__, value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return ("float", numeric) if np.isfinite(numeric) else ("float", repr(numeric))
    if isinstance(value, np.generic):
        return _geometry_manual_scalar_token(value.item())
    return None


def _geometry_manual_sampled_array_content_token(
    arr: np.ndarray,
    *,
    max_samples: int = 32,
) -> tuple[object, ...]:
    if arr.size <= 0:
        return ("samples", 0)
    flat = arr.reshape(-1)
    sample_count = int(min(max(1, int(max_samples)), int(flat.size)))
    if sample_count == int(flat.size):
        indices = np.arange(int(flat.size), dtype=np.int64)
    else:
        indices = np.linspace(0, int(flat.size) - 1, sample_count, dtype=np.int64)
    samples: list[tuple[int, object]] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        try:
            value = flat[idx]
        except Exception:
            continue
        scalar = _geometry_manual_scalar_token(value)
        if scalar is not None:
            samples.append((idx, scalar))
            continue
        try:
            numeric = float(value)
        except Exception:
            samples.append((idx, ("repr", repr(value))))
        else:
            samples.append(
                (idx, ("float", numeric) if np.isfinite(numeric) else ("float", repr(numeric)))
            )
    return ("samples", int(flat.size), tuple(samples))


def _geometry_manual_array_content_or_identity_token(
    arr: np.ndarray,
    source_object: object,
    generation_token: object,
    *,
    max_samples: int = 32,
) -> tuple[object, ...]:
    content_token = _geometry_manual_sampled_array_content_token(
        arr,
        max_samples=max_samples,
    )
    if arr.size > int(max_samples) and not generation_token:
        return ("identity_plus_samples", id(source_object), content_token)
    return content_token


def _geometry_manual_array_token(value: object) -> tuple[object, ...]:
    generation_token = _geometry_manual_generation_token(value)
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            tuple(value.shape),
            str(value.dtype),
            tuple(value.strides),
            generation_token,
            _geometry_manual_array_content_or_identity_token(value, value, generation_token),
        )
    try:
        arr = np.asarray(value)
    except Exception:
        scalar = _geometry_manual_scalar_token(value)
        if scalar is not None:
            return ("scalar_object", scalar, generation_token)
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            generation_token,
        )
    if arr.ndim == 0:
        scalar_value = None
        try:
            scalar_value = _geometry_manual_scalar_token(arr.item())
        except Exception:
            scalar_value = None
        return ("scalar", str(arr.dtype), scalar_value, generation_token)
    return (
        "array_like",
        tuple(arr.shape),
        str(arr.dtype),
        tuple(arr.strides),
        generation_token,
        _geometry_manual_array_content_or_identity_token(arr, value, generation_token),
    )


def _geometry_manual_callable_token(fn: object) -> tuple[object, ...]:
    if fn is None:
        return ("callable", None)
    generation_token = _geometry_manual_generation_token(fn)
    bound_self = getattr(fn, "__self__", None)
    self_token = None
    if bound_self is not None:
        self_generation = _geometry_manual_generation_token(bound_self)
        self_token = (
            type(bound_self).__module__,
            type(bound_self).__qualname__,
            self_generation,
        )
    target = getattr(fn, "__func__", fn)
    module = getattr(target, "__module__", type(target).__module__)
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))
    code = getattr(target, "__code__", None)
    code_token = None
    if code is not None:
        scalar_consts = []
        for const in tuple(getattr(code, "co_consts", ()) or ())[:16]:
            scalar = _geometry_manual_scalar_token(const)
            if scalar is not None:
                scalar_consts.append(scalar)
        code_token = (
            getattr(code, "co_filename", None),
            getattr(code, "co_name", None),
            int(getattr(code, "co_argcount", 0)),
            int(getattr(code, "co_kwonlyargcount", 0)),
            int(getattr(code, "co_posonlyargcount", 0)),
            _geometry_manual_projection_blake2b_hex(getattr(code, "co_code", b"")),
            tuple(scalar_consts),
        )
    defaults = []
    for value in tuple(getattr(target, "__defaults__", ()) or ()):
        scalar = _geometry_manual_scalar_token(value)
        defaults.append(scalar if scalar is not None else (type(value).__name__,))
    kwdefaults = []
    raw_kwdefaults = getattr(target, "__kwdefaults__", None)
    if isinstance(raw_kwdefaults, Mapping):
        for key in sorted(raw_kwdefaults):
            value = raw_kwdefaults[key]
            scalar = _geometry_manual_scalar_token(value)
            kwdefaults.append((key, scalar if scalar is not None else (type(value).__name__,)))
    closure_values = []
    for cell in tuple(getattr(target, "__closure__", ()) or ())[:8]:
        try:
            value = cell.cell_contents
        except Exception:
            continue
        scalar = _geometry_manual_scalar_token(value)
        if scalar is not None:
            closure_values.append(scalar)
            continue
        generation = _geometry_manual_generation_token(value)
        if generation:
            closure_values.append((type(value).__module__, type(value).__qualname__, generation))
    return (
        "callable",
        module,
        qualname,
        generation_token,
        self_token,
        code_token,
        tuple(defaults),
        tuple(kwdefaults),
        tuple(closure_values),
    )


def geometry_manual_qr_sim_refinement_signature(
    cache_data: Mapping[str, object] | None,
    *,
    simulation_signature: object = None,
    caked_projection_signature: object = None,
    detector_simulation_image: object | None = None,
    caked_simulation_image: object | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    detector_display_to_native_coords: object | None = None,
    native_detector_coords_to_caked_display_coords: object | None = None,
    caked_angles_to_detector_display_coords: object | None = None,
    search_radius_px: int = 5,
) -> tuple[object, ...]:
    cache = cache_data if isinstance(cache_data, Mapping) else {}
    metadata = (
        cache.get("cache_metadata") if isinstance(cache.get("cache_metadata"), Mapping) else {}
    )
    return (
        "manual_qr_sim_refinement_v2",
        cache.get("signature"),
        cache.get("placed_signature"),
        simulation_signature,
        caked_projection_signature,
        _geometry_manual_generation_token(cache),
        _geometry_manual_generation_token(metadata),
        _geometry_manual_array_token(detector_simulation_image),
        _geometry_manual_array_token(caked_simulation_image),
        geometry_manual_stable_axis_value_token(radial_axis),
        geometry_manual_stable_axis_value_token(azimuth_axis),
        _geometry_manual_callable_token(detector_display_to_native_coords),
        _geometry_manual_callable_token(native_detector_coords_to_caked_display_coords),
        _geometry_manual_callable_token(caked_angles_to_detector_display_coords),
        int(search_radius_px),
    )


def geometry_manual_qr_sim_refinement_identity(
    entry: Mapping[str, object] | None,
    *,
    view_mode: str,
) -> tuple[object, ...]:
    """Return one physical QR sim candidate identity for refinement memoization."""

    candidate = entry if isinstance(entry, Mapping) else {}
    source_identity = tuple(
        (key, candidate.get(key))
        for key in (
            "source_table_index",
            "source_row_index",
            "source_branch_index",
            "source_reflection_index",
            "source_ray_id",
            "branch_id",
        )
        if _geometry_manual_identity_value_present(candidate.get(key))
    )
    group_key = _geometry_manual_real_q_group_key(candidate)
    hkl = _default_normalize_hkl_key(candidate.get("hkl", candidate.get("label")))
    if source_identity:
        return (
            "qr_sim_refinement_identity_v1",
            str(view_mode),
            source_identity,
            group_key,
            hkl,
        )

    if str(view_mode) == "caked":
        point = _geometry_manual_finite_point(
            candidate,
            (
                ("refined_sim_caked_x", "refined_sim_caked_y"),
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
                ("two_theta_deg", "phi_deg"),
            ),
        )
    else:
        point = _geometry_manual_finite_point(
            candidate,
            (
                ("refined_sim_x", "refined_sim_y"),
                ("sim_col_raw", "sim_row_raw"),
                ("sim_col", "sim_row"),
                ("display_col", "display_row"),
                ("x", "y"),
            ),
        )
    point_token = (
        (round(float(point[0]), 6), round(float(point[1]), 6)) if point is not None else None
    )
    return (
        "qr_sim_refinement_identity_v1",
        str(view_mode),
        group_key,
        hkl,
        point_token,
    )


def geometry_manual_merge_refined_qr_sim_fields(
    raw_entry: Mapping[str, object] | None,
    refined_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge refined simulation fields into a raw cache row without dropping row identity."""

    merged = dict(raw_entry) if isinstance(raw_entry, Mapping) else {}
    if not isinstance(refined_entry, Mapping):
        return merged
    refined_keys = set(_GEOMETRY_MANUAL_QR_SIM_ALIAS_FIELDS)
    refined_keys.update(
        {
            "refined_sim_native_x",
            "refined_sim_native_y",
            "sim_refined_detector_display_px",
            "sim_refined_detector_native_px",
            "sim_refined_caked_deg",
            "sim_visual_detector_display_px",
            "sim_visual_detector_native_px",
            "sim_visual_detector_native_source",
            "sim_visual_detector_native_unavailable_reason",
            "sim_visual_caked_deg",
            "sim_visual_caked_source",
            "sim_visual_deg",
            "sim_visual_source",
            "sim_caked",
            "sim_refinement_source",
            "sim_refinement_status",
            "sim_refinement_delta_detector_px",
            "sim_refinement_delta_caked_deg",
            "detector_display_px",
            "detector_native_px",
            "display_frame",
            "current_view_frame",
            "refined",
        }
    )
    for key in refined_keys:
        if key in refined_entry:
            merged[key] = refined_entry[key]
    return merged


def geometry_manual_refine_qr_sim_candidates_in_cache(
    cache_data: Mapping[str, object] | None,
    *,
    detector_simulation_image: object | None = None,
    caked_simulation_image: object | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    detector_display_to_native_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    caked_angles_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
    search_radius_px: int = 5,
    refinement_signature: object = None,
    force_refine: bool = False,
) -> dict[str, object]:
    if not isinstance(cache_data, Mapping):
        return {}
    if (
        not force_refine
        and refinement_signature is not None
        and cache_data.get("qr_sim_refinement_signature") == refinement_signature
        and cache_data.get("qr_sim_refinement_complete") is True
    ):
        return dict(cache_data)

    refined_by_identity: dict[tuple[object, ...], dict[str, object]] = {}
    refinement_total_row_count = 0
    refinement_unique_candidate_count = 0
    refinement_reused_candidate_count = 0

    def _refine_one(raw_entry: Mapping[str, object], *, view_mode: str) -> dict[str, object]:
        nonlocal \
            refinement_total_row_count, \
            refinement_unique_candidate_count, \
            refinement_reused_candidate_count

        refinement_total_row_count += 1
        identity = geometry_manual_qr_sim_refinement_identity(raw_entry, view_mode=view_mode)
        if identity in refined_by_identity:
            refinement_reused_candidate_count += 1
            return geometry_manual_merge_refined_qr_sim_fields(
                raw_entry,
                refined_by_identity[identity],
            )
        refined_entry = geometry_manual_refine_qr_sim_peak_for_view(
            raw_entry,
            view_mode=view_mode,
            detector_simulation_image=detector_simulation_image,
            caked_simulation_image=caked_simulation_image,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            detector_display_to_native_coords=detector_display_to_native_coords,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
            refine_caked_peak_center_fn=refine_caked_peak_center_fn,
            search_radius_px=search_radius_px,
        )
        refined_dict = (
            dict(refined_entry) if isinstance(refined_entry, Mapping) else dict(raw_entry)
        )
        refined_by_identity[identity] = dict(refined_dict)
        refinement_unique_candidate_count += 1
        return geometry_manual_merge_refined_qr_sim_fields(raw_entry, refined_dict)

    def _refine_rows(rows: object, *, view_mode: str) -> list[dict[str, object]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            return []
        output: list[dict[str, object]] = []
        for raw_entry in rows:
            if not isinstance(raw_entry, Mapping):
                continue
            output.append(_refine_one(raw_entry, view_mode=view_mode))
        return output

    def _refine_grouped(
        grouped: object,
        *,
        view_mode: str,
    ) -> dict[tuple[object, ...], list[dict[str, object]]]:
        if not isinstance(grouped, Mapping):
            return {}
        output: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for raw_key, entries in grouped.items():
            key = tuple(raw_key) if isinstance(raw_key, list) else raw_key
            if not isinstance(key, tuple):
                continue
            output[key] = _refine_rows(entries, view_mode=view_mode)
        return output

    refined_cache = dict(cache_data)
    detector_keys = (
        "simulated_peaks",
        "active_simulated_peaks",
        "fresh_source_rows",
        "detector_picker_source_rows",
        "detector_picker_rows",
    )
    for key in detector_keys:
        if key in refined_cache:
            refined_cache[key] = _refine_rows(refined_cache.get(key), view_mode="detector")
    if "detector_picker_grouped_candidates" in refined_cache:
        refined_cache["detector_picker_grouped_candidates"] = _refine_grouped(
            refined_cache.get("detector_picker_grouped_candidates"),
            view_mode="detector",
        )
    if "grouped_candidates" in refined_cache:
        refined_cache["grouped_candidates"] = _refine_grouped(
            refined_cache.get("grouped_candidates"),
            view_mode="detector",
        )
    if "caked_qr_projection_entries" in refined_cache:
        refined_cache["caked_qr_projection_entries"] = _refine_rows(
            refined_cache.get("caked_qr_projection_entries"),
            view_mode="caked",
        )
    if "caked_qr_projection_grouped_candidates" in refined_cache:
        refined_cache["caked_qr_projection_grouped_candidates"] = _refine_grouped(
            refined_cache.get("caked_qr_projection_grouped_candidates"),
            view_mode="caked",
        )
    if refinement_signature is not None:
        refined_cache["qr_sim_refinement_signature"] = refinement_signature
        refined_cache["qr_sim_refinement_complete"] = True
    else:
        refined_cache.pop("qr_sim_refinement_signature", None)
        refined_cache.pop("qr_sim_refinement_complete", None)
    refined_cache["cache_metadata"] = _geometry_manual_cache_metadata_update(
        refined_cache,
        qr_refinement_unique_candidate_count=int(refinement_unique_candidate_count),
        qr_refinement_reused_candidate_count=int(refinement_reused_candidate_count),
        qr_refinement_total_row_count=int(refinement_total_row_count),
    )
    return refined_cache


def geometry_manual_rebuild_refined_qr_cache_lookups(
    cache_data: Mapping[str, object] | None,
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ],
    *,
    refinement_signature: object = None,
    force_rebuild: bool = False,
) -> dict[str, object]:
    """Rebuild Qr lookup maps after simulated rows have been refined."""

    if not isinstance(cache_data, Mapping):
        return {}
    requires_sim_lookup = "active_simulated_peaks" in cache_data or "simulated_peaks" in cache_data
    requires_projection_lookup = "caked_qr_projection_entries" in cache_data
    has_required_lookups = bool(
        (not requires_sim_lookup or "simulated_lookup" in cache_data)
        and (not requires_projection_lookup or "caked_qr_projection_lookup" in cache_data)
    )
    if (
        not force_rebuild
        and refinement_signature is not None
        and cache_data.get("qr_sim_refinement_lookup_signature") == refinement_signature
        and cache_data.get("qr_sim_refinement_lookup_complete") is True
        and has_required_lookups
    ):
        return dict(cache_data)

    def _row_dicts(rows: object) -> list[dict[str, object]]:
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            return []
        return [dict(entry) for entry in rows if isinstance(entry, Mapping)]

    rebuilt_cache = dict(cache_data)
    lookup_complete = True
    if "active_simulated_peaks" in rebuilt_cache or "simulated_peaks" in rebuilt_cache:
        if "active_simulated_peaks" in rebuilt_cache:
            lookup_source_rows = _row_dicts(rebuilt_cache.get("active_simulated_peaks"))
        else:
            lookup_source_rows = _row_dicts(rebuilt_cache.get("simulated_peaks"))
        try:
            rebuilt_lookup = build_simulated_lookup(lookup_source_rows)
        except Exception:
            rebuilt_lookup = {}
            lookup_complete = False
        rebuilt_cache["simulated_lookup"] = _geometry_manual_copy_lookup(rebuilt_lookup)

    if "caked_qr_projection_entries" in rebuilt_cache:
        try:
            projection_lookup: GeometryManualLookupMap = {}
            for entry in _row_dicts(rebuilt_cache.get("caked_qr_projection_entries")):
                key = _geometry_manual_caked_qr_projection_key(entry)
                if key is not None:
                    _geometry_manual_add_lookup_entry(projection_lookup, key, entry)
        except Exception:
            projection_lookup = {}
            lookup_complete = False
        rebuilt_cache["caked_qr_projection_lookup"] = _geometry_manual_copy_lookup(
            projection_lookup
        )

    if refinement_signature is not None and lookup_complete:
        rebuilt_cache["qr_sim_refinement_lookup_signature"] = refinement_signature
        rebuilt_cache["qr_sim_refinement_lookup_complete"] = True
    else:
        rebuilt_cache.pop("qr_sim_refinement_lookup_signature", None)
        rebuilt_cache["qr_sim_refinement_lookup_complete"] = False
    return rebuilt_cache


def geometry_manual_detector_picker_row(
    entry: Mapping[str, object] | None,
    *,
    display_width: int = 0,
    display_height: int = 0,
    native_width: int = 0,
    native_height: int = 0,
) -> dict[str, object] | None:
    """Return one detector-space Qr picker row without caked projection."""

    if not isinstance(entry, Mapping):
        return None
    group_key = _geometry_manual_real_q_group_key(entry)
    if group_key is None:
        return None
    display_point = _geometry_manual_candidate_detector_sim_point(entry)
    if display_point is None or _geometry_manual_detector_point_is_caked(entry, display_point):
        return None
    if (
        int(display_width) > 0
        and int(display_height) > 0
        and not _geometry_manual_point_inside_bounds(
            display_point,
            width=int(display_width),
            height=int(display_height),
        )
    ):
        return None

    refined_display_point = _geometry_manual_tuple_point(entry, "sim_refined_detector_display_px")
    uses_refined_display = _geometry_manual_points_match(display_point, refined_display_point)
    if uses_refined_display:
        native_point = _geometry_manual_tuple_point(entry, "sim_refined_detector_native_px")
    else:
        native_point = _geometry_manual_entry_native_point(entry)
    if native_point is not None and _geometry_manual_detector_point_is_caked(entry, native_point):
        return None
    if (
        native_point is not None
        and int(native_width) > 0
        and int(native_height) > 0
        and not _geometry_manual_point_inside_bounds(
            native_point,
            width=int(native_width),
            height=int(native_height),
        )
    ):
        return None

    row = dict(entry)
    row["q_group_key"] = group_key
    row.setdefault("source_label", _geometry_manual_q_group_source_label(group_key))
    if row.get("source_label") == DISORDERED_PHASE_SOURCE_LABEL:
        row.setdefault("phase_label", DISORDERED_PHASE_DISPLAY_LABEL)
        row.setdefault("structure_role", "disordered")
    row["detector_display_px"] = (float(display_point[0]), float(display_point[1]))
    row["display_col"] = float(display_point[0])
    row["display_row"] = float(display_point[1])
    row["sim_col"] = float(display_point[0])
    row["sim_row"] = float(display_point[1])
    row["sim_col_raw"] = float(display_point[0])
    row["sim_row_raw"] = float(display_point[1])
    row["display_frame"] = "detector_display"
    if uses_refined_display:
        row["detector_display_source"] = "sim_refined_detector_display_px"
    else:
        row.setdefault("detector_display_source", "detector_picker")
    if uses_refined_display and native_point is None:
        for stale_native_key in (
            "detector_native_px",
            "native_col",
            "native_row",
            "sim_native_x",
            "sim_native_y",
        ):
            row.pop(stale_native_key, None)
    if native_point is not None:
        row["detector_native_px"] = (float(native_point[0]), float(native_point[1]))
        row["native_col"] = float(native_point[0])
        row["native_row"] = float(native_point[1])
        row["sim_native_x"] = float(native_point[0])
        row["sim_native_y"] = float(native_point[1])
    return row


def geometry_manual_detector_picker_candidates_from_rows(
    rows: Sequence[dict[str, object]] | None,
    *,
    display_background: object | None = None,
    native_background: object | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[tuple[object, ...], list[dict[str, object]]]]:
    """Build detector-mode picker candidates from detector-space Qr rows only."""

    display_height, display_width = _geometry_manual_detector_shape(display_background)
    native_height, native_width = _geometry_manual_detector_shape(native_background)
    source_rows: list[dict[str, object]] = []
    for raw_entry in rows or ():
        candidate = geometry_manual_detector_picker_row(
            raw_entry,
            display_width=display_width,
            display_height=display_height,
            native_width=native_width,
            native_height=native_height,
        )
        if candidate is not None:
            source_rows.append(candidate)
    if not source_rows:
        return [], {}

    collapsed_rows = _geometry_manual_collapse_q_group_representatives(
        source_rows,
        profile_cache=profile_cache,
    )
    candidate_rows: list[dict[str, object]] = []
    for raw_entry in collapsed_rows:
        candidate = geometry_manual_detector_picker_row(
            raw_entry,
            display_width=display_width,
            display_height=display_height,
            native_width=native_width,
            native_height=native_height,
        )
        if candidate is not None:
            candidate_rows.append(candidate)

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for entry in candidate_rows:
        group_key = _geometry_manual_real_q_group_key(entry)
        if group_key is None:
            continue
        grouped[group_key].append(dict(entry))
    for entry_list in grouped.values():
        entry_list.sort(
            key=lambda entry: (
                (_geometry_manual_entry_detector_display_point(entry) or (float("inf"),))[0],
                (
                    _geometry_manual_entry_detector_display_point(entry)
                    or (float("inf"), float("inf"))
                )[1],
            )
        )
    candidate_rows.sort(key=_geometry_manual_flattened_picker_sort_key)
    return candidate_rows, dict(
        sorted(
            grouped.items(),
            key=lambda item: _geometry_manual_group_sort_key(item[0], item[1]),
        )
    )


def _geometry_manual_safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("inf")
    return float(numeric) if np.isfinite(numeric) else float("inf")


def _geometry_manual_flattened_picker_sort_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _geometry_manual_safe_float(
            entry.get("distance_to_click", entry.get("distance", float("inf")))
        ),
        _geometry_manual_safe_float(entry.get("qr", float("inf"))),
        _geometry_manual_safe_float(entry.get("qz", float("inf"))),
        _geometry_manual_source_priority(entry.get("source_label")),
        str(entry.get("source_label", "")),
    )


def _geometry_manual_group_sort_key(
    group_key: object,
    entries: Sequence[Mapping[str, object]] | None,
) -> tuple[object, ...]:
    first_entry = next((entry for entry in entries or () if isinstance(entry, Mapping)), {})
    source_label = _geometry_manual_q_group_source_label(group_key)
    return (
        _geometry_manual_safe_float(
            first_entry.get("distance_to_click", first_entry.get("distance", float("inf")))
            if isinstance(first_entry, Mapping)
            else float("inf")
        ),
        _geometry_manual_safe_float(first_entry.get("qr", float("inf"))),
        _geometry_manual_safe_float(first_entry.get("qz", float("inf"))),
        _geometry_manual_source_priority(source_label),
        str(source_label),
        tuple(group_key) if isinstance(group_key, tuple) else repr(group_key),
    )


def _geometry_manual_flatten_grouped_candidates(
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not isinstance(grouped_candidates, Mapping):
        return rows
    for group_key, entries in grouped_candidates.items():
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        for raw_entry in entries:
            if isinstance(raw_entry, Mapping):
                entry = dict(raw_entry)
                entry.setdefault("q_group_key", group_key)
                rows.append(entry)
    rows.sort(key=_geometry_manual_flattened_picker_sort_key)
    return rows


def _geometry_manual_detector_picker_source_label(
    entry: Mapping[str, object] | None,
) -> str:
    if not isinstance(entry, Mapping):
        return "unavailable"
    for key in ("detector_picker_source", "diagnostic_source", "cache_source"):
        raw_value = entry.get(key)
        if raw_value is not None and str(raw_value).strip():
            return str(raw_value)
    if (
        entry.get("manual_pair_order_key") is not None
        or entry.get("manual_picker_truth_available") is not None
    ):
        return "manual_saved_pair"
    raw_source_label = entry.get("source_label")
    if raw_source_label is not None and str(raw_source_label).strip():
        return f"fresh_source_rows:{raw_source_label}"
    return "fresh_source_rows"


def _geometry_manual_detector_picker_is_manual_saved_pair(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    label = _geometry_manual_detector_picker_source_label(entry).lower()
    return (
        "manual_saved_pair" in label
        or entry.get("manual_pair_order_key") is not None
        or entry.get("manual_picker_truth_available") is not None
    )


def _geometry_manual_detector_picker_source_breakdown(
    rows: Sequence[Mapping[str, object]] | None,
) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for entry in rows or ():
        label = _geometry_manual_detector_picker_source_label(entry)
        breakdown[label] = int(breakdown.get(label, 0)) + 1
    return dict(sorted(breakdown.items(), key=lambda item: item[0]))


def _geometry_manual_detector_picker_row_diagnostics(
    rows: Sequence[Mapping[str, object]] | None,
    *,
    display_width: int = 0,
    display_height: int = 0,
    native_width: int = 0,
    native_height: int = 0,
) -> dict[str, int]:
    q_group_count = 0
    detector_display_count = 0
    caked_display_count = 0
    missing_detector_display_count = 0
    outside_display_count = 0
    native_count = 0
    outside_native_count = 0
    detector_candidate_count = 0
    for raw_entry in rows or ():
        if not isinstance(raw_entry, Mapping):
            continue
        if _geometry_manual_real_q_group_key(raw_entry) is None:
            continue
        q_group_count += 1
        display_point = _geometry_manual_tuple_point(
            raw_entry,
            "sim_refined_detector_display_px",
        )
        if display_point is None:
            display_point = _geometry_manual_finite_point(
                raw_entry,
                (
                    ("display_col", "display_row"),
                    ("sim_col_raw", "sim_row_raw"),
                    ("x", "y"),
                    ("simulated_x", "simulated_y"),
                    ("sim_col", "sim_row"),
                ),
            )
        if display_point is None:
            missing_detector_display_count += 1
        elif _geometry_manual_detector_point_is_caked(raw_entry, display_point):
            caked_display_count += 1
        else:
            detector_display_count += 1
            if (
                int(display_width) > 0
                and int(display_height) > 0
                and not _geometry_manual_point_inside_bounds(
                    display_point,
                    width=int(display_width),
                    height=int(display_height),
                )
            ):
                outside_display_count += 1
        native_point = _geometry_manual_entry_native_point(raw_entry)
        if native_point is not None:
            native_count += 1
            if (
                int(native_width) > 0
                and int(native_height) > 0
                and not _geometry_manual_point_inside_bounds(
                    native_point,
                    width=int(native_width),
                    height=int(native_height),
                )
            ):
                outside_native_count += 1
        if (
            geometry_manual_detector_picker_row(
                raw_entry,
                display_width=display_width,
                display_height=display_height,
                native_width=native_width,
                native_height=native_height,
            )
            is not None
        ):
            detector_candidate_count += 1
    return {
        "q_group_source_row_count": int(q_group_count),
        "detector_display_source_row_count": int(detector_display_count),
        "caked_display_source_row_count": int(caked_display_count),
        "missing_detector_display_source_row_count": int(missing_detector_display_count),
        "outside_display_source_row_count": int(outside_display_count),
        "native_source_row_count": int(native_count),
        "outside_native_source_row_count": int(outside_native_count),
        "detector_candidate_source_row_count": int(detector_candidate_count),
    }


def _geometry_manual_detector_picker_identity(
    entry: Mapping[str, object],
) -> tuple[object, ...]:
    display_point = _geometry_manual_candidate_detector_sim_point(entry)
    refined_display_point = _geometry_manual_tuple_point(entry, "sim_refined_detector_display_px")
    if _geometry_manual_points_match(display_point, refined_display_point):
        native_point = _geometry_manual_tuple_point(entry, "sim_refined_detector_native_px")
    else:
        native_point = _geometry_manual_entry_native_point(entry)
    return (
        _geometry_manual_real_q_group_key(entry),
        _default_normalize_hkl_key(entry.get("hkl", entry.get("label"))),
        entry.get("source_table_index"),
        entry.get("source_row_index"),
        entry.get("source_branch_index"),
        entry.get("source_peak_index"),
        entry.get("branch_id"),
        (
            round(float(display_point[0]), 6),
            round(float(display_point[1]), 6),
        )
        if display_point is not None
        else None,
        (
            round(float(native_point[0]), 6),
            round(float(native_point[1]), 6),
        )
        if native_point is not None
        else None,
    )


def _geometry_manual_detector_picker_rows_from_value(
    value: object,
    *,
    source: str | None = None,
) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, object]] = []
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        if source:
            entry.setdefault("detector_picker_source", source)
        rows.append(entry)
    return rows


def _geometry_manual_detector_picker_grouped_rows_from_value(
    value: object,
    *,
    source: str,
) -> list[dict[str, object]]:
    rows = _geometry_manual_flatten_grouped_candidates(
        value if isinstance(value, Mapping) else None
    )
    for entry in rows:
        entry.setdefault("detector_picker_source", source)
    return rows


def _geometry_manual_detector_picker_cache_source_groups(
    cache_data: Mapping[str, object] | None,
) -> list[tuple[str, list[dict[str, object]]]]:
    if not isinstance(cache_data, Mapping):
        return []
    groups: list[tuple[str, list[dict[str, object]]]] = []

    def _append_rows(key: str, *, source: str | None = None) -> None:
        rows = _geometry_manual_detector_picker_rows_from_value(
            cache_data.get(key),
            source=source or key,
        )
        if rows:
            groups.append((source or key, rows))

    def _append_grouped(key: str, *, source: str | None = None) -> None:
        rows = _geometry_manual_detector_picker_grouped_rows_from_value(
            cache_data.get(key),
            source=source or key,
        )
        if rows:
            groups.append((source or key, rows))

    _append_rows("detector_picker_rows")
    _append_rows("detector_picker_source_rows")
    _append_grouped("detector_picker_grouped_candidates")

    manual_rows: list[dict[str, object]] = []
    for key in (
        "manual_saved_pair_rows",
        "manual_pair_rows",
        "manual_picker_truth_pairs",
        "manual_picker_truth_rows",
    ):
        manual_rows.extend(
            _geometry_manual_detector_picker_rows_from_value(
                cache_data.get(key),
                source="manual_saved_pair",
            )
        )
    if manual_rows:
        groups.append(("manual_saved_pair", manual_rows))

    for key in (
        "fresh_source_rows",
        "detector_picker_fresh_source_rows",
        "raw_source_rows",
        "source_rows",
    ):
        _append_rows(key)

    _append_rows("active_simulated_peaks")
    _append_rows("simulated_peaks")
    _append_grouped("grouped_candidates")
    return groups


def geometry_manual_pick_cache_has_detector_picker_sources(
    cache_data: Mapping[str, object] | None,
    *,
    display_background: object | None = None,
    native_background: object | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> bool:
    """Return true when a manual-pick cache has detector-space Qr picker rows."""

    if not isinstance(cache_data, Mapping):
        return False
    for _source_name, rows in _geometry_manual_detector_picker_cache_source_groups(cache_data):
        candidate_rows, _grouped = geometry_manual_detector_picker_candidates_from_rows(
            rows,
            display_background=display_background,
            native_background=native_background,
            profile_cache=profile_cache,
        )
        if candidate_rows:
            return True
    return False


def _geometry_manual_detector_picker_rows_build_candidates(
    rows: Sequence[dict[str, object]] | None,
) -> bool:
    candidate_rows, _grouped = geometry_manual_detector_picker_candidates_from_rows(rows)
    return bool(candidate_rows)


def _geometry_manual_detector_picker_dedupe_rows(
    rows: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for raw_entry in rows or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        identity = _geometry_manual_detector_picker_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(entry)
    return deduped


def geometry_manual_detector_picker_source_rows_from_cache(
    cache_data: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    if not isinstance(cache_data, Mapping):
        return []
    first_non_empty_rows: list[dict[str, object]] = []
    for _source_name, rows in _geometry_manual_detector_picker_cache_source_groups(cache_data):
        deduped_rows = _geometry_manual_detector_picker_dedupe_rows(rows)
        if not deduped_rows:
            continue
        if not first_non_empty_rows:
            first_non_empty_rows = [dict(entry) for entry in deduped_rows]
        if _geometry_manual_detector_picker_rows_build_candidates(deduped_rows):
            return deduped_rows
    return _geometry_manual_detector_picker_dedupe_rows(first_non_empty_rows)


def geometry_manual_detector_picker_grouped_candidates_from_cache(
    cache_data: Mapping[str, object] | None,
    *,
    display_background: object | None = None,
    native_background: object | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    rows = geometry_manual_detector_picker_source_rows_from_cache(cache_data)
    _candidate_rows, grouped = geometry_manual_detector_picker_candidates_from_rows(
        rows,
        display_background=display_background,
        native_background=native_background,
        profile_cache=profile_cache,
    )
    if not grouped and isinstance(cache_data, Mapping):
        raw_grouped = cache_data.get("grouped_candidates")
        if isinstance(raw_grouped, Mapping):
            grouped = {
                group_key: [
                    {**dict(entry), "q_group_key": group_key}
                    for entry in (entries or ())
                    if isinstance(entry, Mapping)
                ]
                for raw_key, entries in raw_grouped.items()
                if (group_key := _geometry_manual_real_q_group_key(raw_key)) is not None
            }
    return grouped


def _geometry_manual_detector_picker_empty_reason(
    cache_data: Mapping[str, object] | None,
    *,
    source_rows: Sequence[dict[str, object]],
    picker_rows: Sequence[dict[str, object]],
    display_width: int,
    display_height: int,
    native_width: int,
    native_height: int,
) -> str:
    if picker_rows:
        return ""
    if not isinstance(cache_data, Mapping):
        return "cache_data_unavailable"
    if not source_rows:
        return "no_detector_picker_source_rows"
    for _source_name, cache_rows in _geometry_manual_detector_picker_cache_source_groups(
        cache_data
    ):
        if _geometry_manual_detector_picker_rows_build_candidates(cache_rows):
            return "valid_detector_rows_available_but_not_selected"
    qr_rows = [
        entry for entry in source_rows if _geometry_manual_real_q_group_key(entry) is not None
    ]
    if not qr_rows:
        return "source_rows_have_no_q_group_key"
    display_points = [_geometry_manual_entry_detector_display_point(entry) for entry in qr_rows]
    finite_display_points = [point for point in display_points if point is not None]
    if not finite_display_points:
        caked_like_count = sum(
            1
            for entry in qr_rows
            for point in (
                _geometry_manual_finite_point(
                    entry,
                    (
                        ("display_col", "display_row"),
                        ("sim_col_raw", "sim_row_raw"),
                        ("sim_col", "sim_row"),
                        ("x", "y"),
                    ),
                ),
            )
            if point is not None and _geometry_manual_detector_point_is_caked(entry, point)
        )
        if caked_like_count:
            return "source_rows_detector_display_px_are_caked_deg"
        return "source_rows_have_no_detector_display_px"
    if int(display_width) > 0 and int(display_height) > 0:
        inside_display = [
            point
            for point in finite_display_points
            if _geometry_manual_point_inside_bounds(
                point,
                width=int(display_width),
                height=int(display_height),
            )
        ]
        if not inside_display:
            return "detector_display_px_outside_display_bounds"
    if int(native_width) > 0 and int(native_height) > 0:
        native_points = [_geometry_manual_entry_native_point(entry) for entry in qr_rows]
        finite_native_points = [point for point in native_points if point is not None]
        if finite_native_points and not any(
            _geometry_manual_point_inside_bounds(
                point,
                width=int(native_width),
                height=int(native_height),
            )
            for point in finite_native_points
        ):
            return "detector_native_px_outside_native_bounds"
    return "detector_rows_collapsed_or_filtered_to_zero"


def geometry_manual_format_detector_picker_diagnostic_block(
    trace: Mapping[str, object] | None,
) -> str:
    if not isinstance(trace, Mapping):
        return "[geometry] detector Qr picker diagnostics unavailable"
    keys = (
        "view_mode",
        "background_index",
        "simulation_ready",
        "caked_ready",
        "qr_overlay_visible",
        "manual_saved_pair_count",
        "fresh_source_row_count",
        "detector_picker_candidate_count",
        "detector_picker_candidate_count_by_source",
        "picker_candidates_ready",
        "refinement_lookup_ready",
        "cache_signature_compatible_for_picker",
        "sidecar_cache_reused_for_detector",
        "bounded_picker_build_attempted",
        "bounded_picker_build_returned_row_count",
        "manual_pick_cache_source",
        "manual_pick_cache_stale_reason",
        "source_rows_provider_attempted",
        "detector_picker_reused_after_background_refresh",
        "source_snapshot_status",
        "source_snapshot_rebuild_attempted",
        "source_snapshot_rebuild_returned_row_count",
        "detector_picker_source_rows_with_caked_coords",
        "detector_picker_source_rows_with_detector_display_px",
        "detector_picker_source_rows_missing_detector_display_px",
        "reason_candidates_are_empty",
    )
    lines = ["[geometry] detector Qr picker diagnostics"]
    for key in keys:
        lines.append(f"{key}={trace.get(key)}")
    return "\n".join(lines)


def geometry_manual_detector_picker_input_trace(
    cache_data: Mapping[str, object] | None,
    *,
    view_mode: str = "detector",
    background_index: int | None = None,
    display_background: object | None = None,
    native_background: object | None = None,
    qr_overlay_visible: object | None = None,
    grouped_candidates: Mapping[tuple[object, ...], Sequence[dict[str, object]]] | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object]:
    source_rows = geometry_manual_detector_picker_source_rows_from_cache(cache_data)
    cache_metadata = (
        cache_data.get("cache_metadata")
        if isinstance(cache_data, Mapping) and isinstance(cache_data.get("cache_metadata"), Mapping)
        else {}
    )
    source_snapshot_diagnostics = (
        cache_data.get("source_snapshot_diagnostics")
        if isinstance(cache_data, Mapping)
        and isinstance(cache_data.get("source_snapshot_diagnostics"), Mapping)
        else {}
    )
    display_height, display_width = _geometry_manual_detector_shape(display_background)
    native_height, native_width = _geometry_manual_detector_shape(native_background)
    picker_rows, resolved_grouped = geometry_manual_detector_picker_candidates_from_rows(
        source_rows,
        display_background=display_background,
        native_background=native_background,
        profile_cache=profile_cache,
    )
    active_grouped = (
        grouped_candidates if isinstance(grouped_candidates, Mapping) else resolved_grouped
    )
    overlay_grouped = (
        cache_data.get("grouped_candidates") if isinstance(cache_data, Mapping) else None
    )
    caked_grouped = (
        cache_data.get("caked_qr_projection_grouped_candidates")
        if isinstance(cache_data, Mapping)
        else None
    )
    flattened_active_grouped = _geometry_manual_flatten_grouped_candidates(active_grouped)
    active_picker_breakdown = _geometry_manual_detector_picker_source_breakdown(
        flattened_active_grouped
    )
    source_breakdown = _geometry_manual_detector_picker_source_breakdown(source_rows)
    source_row_diagnostics = _geometry_manual_detector_picker_row_diagnostics(
        source_rows,
        display_width=display_width,
        display_height=display_height,
        native_width=native_width,
        native_height=native_height,
    )
    manual_saved_pair_count = sum(
        1 for entry in source_rows if _geometry_manual_detector_picker_is_manual_saved_pair(entry)
    )
    inside_count = 0
    for entry in picker_rows:
        display_point = _geometry_manual_entry_detector_display_point(entry)
        native_point = _geometry_manual_entry_native_point(entry)
        display_inside = _geometry_manual_point_inside_bounds(
            display_point,
            width=display_width,
            height=display_height,
        )
        native_inside = True
        if native_point is not None and native_width > 0 and native_height > 0:
            native_inside = _geometry_manual_point_inside_bounds(
                native_point,
                width=native_width,
                height=native_height,
            )
        if display_inside and native_inside:
            inside_count += 1

    empty_reason = _geometry_manual_detector_picker_empty_reason(
        cache_data,
        source_rows=source_rows,
        picker_rows=flattened_active_grouped,
        display_width=display_width,
        display_height=display_height,
        native_width=native_width,
        native_height=native_height,
    )
    return {
        "view_mode": str(view_mode),
        "background_index": background_index,
        "simulation_ready": bool(source_rows),
        "caked_ready": bool(isinstance(caked_grouped, Mapping) and caked_grouped),
        "qr_overlay_visible": qr_overlay_visible,
        "manual_saved_pair_count": int(manual_saved_pair_count),
        "fresh_source_row_count": int(len(source_rows) - manual_saved_pair_count),
        "source_rows_count": int(len(source_rows)),
        "fresh_source_rows_count": int(len(source_rows) - manual_saved_pair_count),
        "projected_rows_count": int(
            len(
                [
                    entry
                    for entry in (
                        cache_data.get("active_simulated_peaks", ())
                        if isinstance(cache_data, Mapping)
                        else ()
                    )
                    if isinstance(entry, Mapping)
                ]
            )
        ),
        "qr_group_rows_count": int(
            len(
                [
                    entry
                    for entry in source_rows
                    if _geometry_manual_real_q_group_key(entry) is not None
                ]
            )
        ),
        "inside_detector_count": int(inside_count),
        "picker_candidate_count": int(len(flattened_active_grouped)),
        "detector_picker_candidate_count": int(len(flattened_active_grouped)),
        "detector_picker_candidate_count_by_source": active_picker_breakdown,
        "detector_picker_source_row_count_by_source": source_breakdown,
        "detector_picker_source_row_diagnostics": source_row_diagnostics,
        "detector_picker_source_rows_with_caked_coords": int(
            source_row_diagnostics.get("caked_display_source_row_count", 0)
        ),
        "detector_picker_source_rows_with_detector_display_px": int(
            source_row_diagnostics.get("detector_display_source_row_count", 0)
        ),
        "detector_picker_source_rows_missing_detector_display_px": int(
            source_row_diagnostics.get("missing_detector_display_source_row_count", 0)
        ),
        "manual_pick_cache_source": cache_metadata.get("cache_source"),
        "manual_pick_cache_stale_reason": cache_metadata.get("stale_reason"),
        "picker_candidates_ready": bool(cache_metadata.get("picker_candidates_ready", False)),
        "refinement_lookup_ready": bool(
            cache_metadata.get(
                "refinement_lookup_ready",
                cache_metadata.get("qr_lookup_rebuild_skipped") is False
                and cache_metadata.get("qr_lookup_rebuild_wall_ms") is not None,
            )
        ),
        "cache_signature_compatible_for_picker": bool(
            cache_metadata.get("cache_signature_compatible_for_picker", False)
        ),
        "sidecar_cache_reused_for_detector": bool(
            cache_metadata.get("sidecar_cache_reused_for_detector", False)
        ),
        "bounded_picker_build_attempted": bool(
            cache_metadata.get(
                "bounded_picker_build_attempted",
                cache_metadata.get("picker_candidates_only", False),
            )
        ),
        "bounded_picker_build_returned_row_count": cache_metadata.get(
            "bounded_picker_build_returned_row_count"
        ),
        "source_rows_provider_attempted": bool(
            cache_metadata.get("source_rows_provider_attempted", False)
        ),
        "detector_picker_reused_after_background_refresh": bool(
            cache_metadata.get("detector_picker_reused_after_background_refresh", False)
        ),
        "source_snapshot_status": cache_metadata.get(
            "source_snapshot_status",
            source_snapshot_diagnostics.get("status"),
        ),
        "source_snapshot_rebuild_attempted": cache_metadata.get(
            "source_snapshot_rebuild_attempted",
            source_snapshot_diagnostics.get("rebuild_attempted"),
        ),
        "source_snapshot_rebuild_returned_row_count": cache_metadata.get(
            "source_snapshot_rebuild_returned_row_count",
            source_snapshot_diagnostics.get("rebuild_returned_row_count"),
        ),
        "overlay_drawn_count": int(
            len(
                _geometry_manual_flatten_grouped_candidates(
                    overlay_grouped if isinstance(overlay_grouped, Mapping) else None
                )
            )
        ),
        "caked_callback_available": bool(
            isinstance(cache_data, Mapping)
            and "caked_qr_projection_grouped_candidates" in cache_data
        ),
        "reason_candidates_are_empty": empty_reason,
        "empty_candidate_reason": empty_reason,
    }


def geometry_manual_simulated_peaks_from_callback(
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]] | None,
    *,
    param_set: dict[str, object] | None = None,
    prefer_cache: bool,
) -> list[dict[str, object]]:
    """Safely call one simulated-peak provider and normalize dict rows."""

    if not callable(simulated_peaks_for_params):
        return []
    try:
        raw_entries = simulated_peaks_for_params(
            param_set,
            prefer_cache=bool(prefer_cache),
        )
    except TypeError:
        try:
            raw_entries = simulated_peaks_for_params(param_set)
        except Exception:
            return []
    except Exception:
        return []
    return [dict(entry) for entry in (raw_entries or ()) if isinstance(entry, Mapping)]


def geometry_manual_live_peak_candidates_from_records(
    peak_records: Sequence[object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    source_reflection_indices_local: Sequence[object] | None = None,
    source_row_hkl_lookup: Mapping[tuple[int, int], tuple[int, int, int]] | None = None,
    provenance_signature_matches: bool = False,
    provenance_revision_matches: bool = False,
    expected_table_count: int | None = None,
    active_signature_matches: bool | None = None,
) -> list[dict[str, object]]:
    """Normalize live HKL-pick peak records into manual-pick candidate rows."""

    if not isinstance(peak_records, Sequence) or isinstance(peak_records, (str, bytes)):
        return []
    if active_signature_matches is not None:
        provenance_signature_matches = bool(active_signature_matches)

    def _finite_point(
        source: Mapping[str, object],
        x_key: str,
        y_key: str,
    ) -> tuple[float, float] | None:
        try:
            col = float(source.get(x_key, np.nan))
            row = float(source.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _points_match(
        left: tuple[float, float] | None,
        right: tuple[float, float] | None,
        *,
        tol: float = 1.0e-9,
    ) -> bool:
        return bool(
            left is not None
            and right is not None
            and abs(float(left[0]) - float(right[0])) <= float(tol)
            and abs(float(left[1]) - float(right[1])) <= float(tol)
        )

    candidates: list[dict[str, object]] = []
    for raw_record in peak_records:
        if not isinstance(raw_record, Mapping):
            continue
        entry = dict(raw_record)
        display_point = _finite_point(entry, "display_col", "display_row")
        native_point = _finite_point(entry, "native_col", "native_row")
        if native_point is None:
            native_point = _finite_point(entry, "sim_native_x", "sim_native_y")
        caked_point_source: str | None = None
        caked_point = _finite_point(entry, "caked_x", "caked_y")
        if caked_point is not None:
            caked_point_source = "caked"
        if caked_point is None:
            caked_point = _finite_point(entry, "raw_caked_x", "raw_caked_y")
            if caked_point is not None:
                caked_point_source = "raw_caked"
        if caked_point is None:
            caked_point = _finite_point(entry, "two_theta_deg", "phi_deg")
            if caked_point is not None:
                caked_point_source = "angles"
        raw_detector_display = _finite_point(entry, "sim_col_raw", "sim_row_raw")
        legacy_sim_point = _finite_point(entry, "sim_col", "sim_row")

        if (
            raw_detector_display is None
            and native_point is None
            and caked_point is None
            and legacy_sim_point is None
            and display_point is None
        ):
            continue

        if display_point is None:
            if caked_point is not None and (
                legacy_sim_point is None or _points_match(legacy_sim_point, caked_point)
            ):
                display_point = caked_point
            else:
                display_point = legacy_sim_point or raw_detector_display or caked_point
        if display_point is not None:
            entry["display_col"] = float(display_point[0])
            entry["display_row"] = float(display_point[1])
        detector_display = raw_detector_display
        if (
            detector_display is None
            and legacy_sim_point is not None
            and (
                native_point is not None
                or caked_point is None
                or not _points_match(legacy_sim_point, caked_point)
            )
        ):
            detector_display = legacy_sim_point
        if detector_display is not None:
            entry["sim_col"] = float(detector_display[0])
            entry["sim_row"] = float(detector_display[1])
            entry["sim_col_raw"] = float(detector_display[0])
            entry["sim_row_raw"] = float(detector_display[1])
        if native_point is not None:
            entry["native_col"] = float(native_point[0])
            entry["native_row"] = float(native_point[1])
            entry["sim_native_x"] = float(native_point[0])
            entry["sim_native_y"] = float(native_point[1])
        if caked_point is not None and (caked_point_source != "angles" or native_point is None):
            entry["caked_x"] = float(caked_point[0])
            entry["caked_y"] = float(caked_point[1])
            entry.setdefault("raw_caked_x", float(caked_point[0]))
            entry.setdefault("raw_caked_y", float(caked_point[1]))
        if detector_display is None and native_point is None and caked_point is None:
            continue
        if entry.get("weight") is None:
            try:
                entry["weight"] = max(0.0, float(entry.get("intensity", 0.0)))
            except Exception:
                pass
        normalized_entry = geometry_manual_canonicalize_live_source_entry(
            entry,
            normalize_hkl_key=normalize_hkl_key,
            allow_legacy_peak_fallback=False,
            preserve_existing_trusted_identity=False,
            source_reflection_indices_local=source_reflection_indices_local,
            source_row_hkl_lookup=source_row_hkl_lookup,
            provenance_signature_matches=bool(provenance_signature_matches),
            provenance_revision_matches=bool(provenance_revision_matches),
            expected_table_count=expected_table_count,
        )
        if normalized_entry is not None:
            candidates.append(normalized_entry)
    return candidates


def build_geometry_manual_pick_cache(
    *,
    param_set: dict[str, object] | None = None,
    prefer_cache: bool = True,
    background_index: int,
    current_background_index: int,
    background_image: object | None,
    existing_cache_signature: object = None,
    existing_cache_data: dict[str, object] | None = None,
    placed_cache_signature_fn: Callable[..., tuple[object, ...]] | None = None,
    cache_signature_fn: Callable[..., tuple[object, ...]],
    source_rows_for_background: Callable[
        [int, dict[str, object] | None],
        Sequence[dict[str, object]],
    ]
    | None = None,
    listed_q_group_keys_for_picker: Callable[[], object] | object = (),
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]] | None = None,
    peak_records: Sequence[object] | None = None,
    build_grouped_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ],
    filter_active_rows: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    project_peaks_to_caked_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    project_peaks_for_background_view: Callable[
        [int, Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    current_match_config: Callable[[], dict[str, object]],
    auto_match_background_context: Callable[
        [object, dict[str, object]],
        tuple[dict[str, object], object],
    ]
    | None = None,
    build_caked_projection_sidecar: bool = False,
    picker_candidates_only: bool = False,
) -> tuple[dict[str, object], object, dict[str, object]]:
    """Build or reuse the current manual-pick simulation/background cache."""

    build_start = perf_counter()
    bg_index = int(background_index)
    current_bg_index = int(current_background_index)
    if placed_cache_signature_fn is None:
        placed_cache_signature_fn = cache_signature_fn
    if filter_active_rows is None:
        filter_active_rows = lambda rows: [
            dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)
        ]
    cache_action = "rebuilt"
    cache_source = "source_snapshot_unavailable"
    cache_provenance = ["source_snapshot_unavailable"]
    stale_reason: str | None = (
        None if prefer_cache else "prefer_cache disabled; rebuilding manual-pick cache."
    )
    placed_cache_sig = placed_cache_signature_fn(
        param_set=param_set,
        background_index=bg_index,
        background_image=background_image,
    )
    cache_sig = cache_signature_fn(
        param_set=param_set,
        background_index=bg_index,
        background_image=background_image,
    )
    reuse_requires_caked_projection = _geometry_manual_cache_signature_uses_caked(cache_sig)
    build_requires_caked_projection = bool(
        not bool(picker_candidates_only)
        and (reuse_requires_caked_projection or build_caked_projection_sidecar)
    )
    profile_cache_for_picker = (
        param_set.get("mosaic_params")
        if isinstance(param_set, Mapping) and isinstance(param_set.get("mosaic_params"), Mapping)
        else None
    )
    matching_empty_detector_cache_stale_reason: str | None = None
    if (
        prefer_cache
        and bg_index == current_bg_index
        and existing_cache_signature == cache_sig
        and isinstance(existing_cache_data, dict)
    ):
        if build_requires_caked_projection:
            if existing_cache_data.get("caked_qr_projection_grouped_candidates"):
                return existing_cache_data, existing_cache_signature, existing_cache_data
        elif geometry_manual_pick_cache_has_detector_picker_sources(
            existing_cache_data,
            display_background=background_image,
            profile_cache=profile_cache_for_picker,
        ):
            return existing_cache_data, existing_cache_signature, existing_cache_data
        else:
            matching_empty_detector_cache_stale_reason = (
                "cached manual-pick detector source rows were empty; rebuilding."
            )
            stale_reason = matching_empty_detector_cache_stale_reason

    resolved_existing_signature = existing_cache_signature
    if not isinstance(resolved_existing_signature, tuple) and isinstance(existing_cache_data, dict):
        cached_signature = existing_cache_data.get("signature")
        if isinstance(cached_signature, tuple):
            resolved_existing_signature = cached_signature
    resolved_existing_placed_signature = (
        existing_cache_data.get("placed_signature")
        if isinstance(existing_cache_data, dict)
        else None
    )
    if not isinstance(resolved_existing_placed_signature, tuple):
        resolved_existing_placed_signature = resolved_existing_signature

    def _active_mask_signature_matches(
        existing_signature: object,
        current_signature: object,
    ) -> bool:
        return _manual_pick_cache_mask_signature_matches(
            existing_signature,
            current_signature,
        )

    if (
        prefer_cache
        and bg_index == current_bg_index
        and not build_requires_caked_projection
        and isinstance(existing_cache_data, dict)
        and _manual_pick_cache_groups_reusable(
            resolved_existing_placed_signature,
            placed_cache_sig,
        )
        and _active_mask_signature_matches(resolved_existing_signature, cache_sig)
        and geometry_manual_pick_cache_has_detector_picker_sources(
            existing_cache_data,
            display_background=background_image,
            profile_cache=profile_cache_for_picker,
        )
    ):
        return existing_cache_data, existing_cache_signature, existing_cache_data

    simulated_peaks: list[dict[str, object]] = []
    active_simulated_peaks: list[dict[str, object]] = []
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] = {}
    simulated_lookup: GeometryManualLookupMap = {}
    detector_picker_source_rows: list[dict[str, object]] = []
    detector_picker_rows: list[dict[str, object]] = []
    detector_picker_grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] = {}
    detector_picker_reused_after_background_refresh = False
    detector_picker_coverage_rebuild_attempted = False
    detector_picker_coverage_missing_before: list[tuple[object, ...]] = []
    detector_picker_coverage_missing_after: list[tuple[object, ...]] = []
    source_rows_provider_calls = 0
    fresh_simulation_prefer_cache_false_attempted = False
    existing_cache_reproject_failed = False

    def _listed_picker_q_group_keys() -> list[tuple[object, ...]]:
        raw_keys = (
            listed_q_group_keys_for_picker()
            if callable(listed_q_group_keys_for_picker)
            else listed_q_group_keys_for_picker
        )
        if isinstance(raw_keys, Mapping):
            iterable = raw_keys.keys()
        elif isinstance(raw_keys, Iterable) and not isinstance(raw_keys, (str, bytes)):
            iterable = raw_keys
        else:
            iterable = ()
        normalized: list[tuple[object, ...]] = []
        seen: set[tuple[object, ...]] = set()
        for raw_key in iterable:
            candidate = raw_key
            if isinstance(raw_key, Mapping):
                candidate = raw_key.get("key", raw_key.get("q_group_key"))
            key = _geometry_manual_real_q_group_key(candidate)
            if key is None and isinstance(raw_key, Mapping):
                key = _geometry_manual_real_q_group_key(raw_key)
            if key is None or key in seen:
                continue
            normalized.append(key)
            seen.add(key)
        return normalized

    def _detector_picker_covered_q_group_keys() -> set[tuple[object, ...]]:
        covered: set[tuple[object, ...]] = set()
        for raw_key, entries in detector_picker_grouped_candidates.items():
            if not entries:
                continue
            key = _geometry_manual_real_q_group_key(raw_key)
            if key is not None:
                covered.add(key)
        return covered

    def _missing_listed_detector_picker_q_group_keys() -> list[tuple[object, ...]]:
        listed_keys = _listed_picker_q_group_keys()
        if not listed_keys:
            return []
        covered = _detector_picker_covered_q_group_keys()
        return [key for key in listed_keys if key not in covered]

    def _apply_candidate_source(
        candidate_rows: Sequence[dict[str, object]] | None,
        *,
        action: str,
        source: str,
        provenance: Sequence[str],
        reproject: bool = True,
        stale_reason_override: str | None = None,
    ) -> bool:
        nonlocal simulated_peaks, active_simulated_peaks, grouped_candidates, simulated_lookup
        nonlocal \
            detector_picker_source_rows, \
            detector_picker_rows, \
            detector_picker_grouped_candidates
        nonlocal cache_action, cache_source, cache_provenance, stale_reason

        raw_source_rows = [
            dict(entry) for entry in (candidate_rows or ()) if isinstance(entry, Mapping)
        ]
        for entry in raw_source_rows:
            entry.setdefault("detector_picker_source", str(source))
        profile_cache_local = (
            param_set.get("mosaic_params")
            if isinstance(param_set, Mapping)
            and isinstance(param_set.get("mosaic_params"), Mapping)
            else None
        )
        (
            raw_detector_picker_rows,
            raw_detector_picker_grouped_candidates,
        ) = geometry_manual_detector_picker_candidates_from_rows(
            raw_source_rows,
            display_background=background_image,
            profile_cache=profile_cache_local,
        )
        normalized_rows = [dict(entry) for entry in raw_source_rows]
        if reproject and (
            callable(project_peaks_for_background_view) or callable(project_peaks_to_current_view)
        ):
            reprojection_rows = _filter_cached_source_rows_for_reprojection(normalized_rows)
            if reprojection_rows:
                try:
                    normalized_rows = _filter_reprojected_cache_rows(
                        _reproject_cache_rows(reprojection_rows)
                    )
                except Exception:
                    normalized_rows = []
        if not normalized_rows:
            (
                fallback_detector_rows,
                fallback_detector_grouped_candidates,
            ) = geometry_manual_detector_picker_candidates_from_rows(
                raw_source_rows,
                display_background=background_image,
                profile_cache=profile_cache_local,
            )
            if not fallback_detector_rows:
                return False
            simulated_peaks = [dict(entry) for entry in raw_source_rows]
            active_simulated_peaks = []
            detector_picker_source_rows = [dict(entry) for entry in raw_source_rows]
            detector_picker_rows = [dict(entry) for entry in fallback_detector_rows]
            detector_picker_grouped_candidates = {
                key: [dict(entry) for entry in entries]
                for key, entries in fallback_detector_grouped_candidates.items()
            }
            grouped_candidates = {}
            simulated_lookup = {}
            cache_action = str(action)
            cache_source = str(source)
            cache_provenance = [str(step) for step in provenance]
            if (
                stale_reason_override is None
                and stale_reason == matching_empty_detector_cache_stale_reason
            ):
                stale_reason = matching_empty_detector_cache_stale_reason
            else:
                stale_reason = stale_reason_override
            return True
        for entry in normalized_rows:
            entry.setdefault("detector_picker_source", str(source))

        try:
            filtered_active_rows = filter_active_rows(normalized_rows)
        except Exception:
            filtered_active_rows = []
        candidate_groups = build_grouped_candidates(filtered_active_rows)

        local_active_simulated_peaks = [
            dict(entry) for entry in filtered_active_rows if isinstance(entry, Mapping)
        ]
        if raw_detector_picker_rows:
            detector_source_candidates = raw_source_rows
            local_detector_picker_rows = [dict(entry) for entry in raw_detector_picker_rows]
            local_detector_picker_grouped_candidates = {
                key: [dict(entry) for entry in entries]
                for key, entries in raw_detector_picker_grouped_candidates.items()
            }
        else:
            detector_source_candidates = local_active_simulated_peaks or raw_source_rows
            local_detector_picker_rows, local_detector_picker_grouped_candidates = (
                geometry_manual_detector_picker_candidates_from_rows(
                    detector_source_candidates,
                    display_background=background_image,
                    profile_cache=profile_cache_local,
                )
            )
        if not local_detector_picker_rows and detector_source_candidates is not raw_source_rows:
            detector_source_candidates = raw_source_rows
            local_detector_picker_rows, local_detector_picker_grouped_candidates = (
                geometry_manual_detector_picker_candidates_from_rows(
                    detector_source_candidates,
                    display_background=background_image,
                    profile_cache=profile_cache_local,
                )
            )
        local_detector_picker_source_rows = [
            dict(entry) for entry in detector_source_candidates if isinstance(entry, Mapping)
        ]
        if not reuse_requires_caked_projection and not local_detector_picker_rows:
            legacy_detector_grouped = {
                key: [
                    dict(entry)
                    for entry in entries
                    if isinstance(entry, Mapping)
                    and _geometry_manual_entry_detector_display_point(entry) is not None
                ]
                for key, entries in (candidate_groups or {}).items()
                if isinstance(key, tuple)
            }
            legacy_detector_grouped = {
                key: entries for key, entries in legacy_detector_grouped.items() if entries
            }
            if legacy_detector_grouped:
                simulated_peaks = normalized_rows
                active_simulated_peaks = local_active_simulated_peaks
                detector_picker_source_rows = local_detector_picker_source_rows
                detector_picker_rows = []
                detector_picker_grouped_candidates = {}
                grouped_candidates = legacy_detector_grouped
                simulated_lookup = build_simulated_lookup(active_simulated_peaks)
                cache_action = str(action)
                cache_source = str(source)
                cache_provenance = [str(step) for step in provenance]
                if (
                    stale_reason_override is None
                    and stale_reason == matching_empty_detector_cache_stale_reason
                ):
                    stale_reason = matching_empty_detector_cache_stale_reason
                else:
                    stale_reason = stale_reason_override
                return True
            display_height, display_width = _geometry_manual_detector_shape(background_image)
            detector_diag = _geometry_manual_detector_picker_row_diagnostics(
                local_detector_picker_source_rows,
                display_width=display_width,
                display_height=display_height,
            )
            if int(detector_diag.get("q_group_source_row_count", 0)) <= 0:
                stale_reason = f"{source} rows had no detector Qr/Qz group keys."
            elif (
                int(detector_diag.get("caked_display_source_row_count", 0)) > 0
                and int(detector_diag.get("detector_display_source_row_count", 0)) <= 0
            ):
                stale_reason = f"{source} rows were caked-only; rebuilding detector picker rows."
            elif int(detector_diag.get("missing_detector_display_source_row_count", 0)) > 0:
                stale_reason = f"{source} rows lacked detector display coordinates; rebuilding."
            else:
                stale_reason = (
                    f"{source} rows did not produce detector picker candidates; rebuilding."
                )
            return False

        simulated_peaks = normalized_rows
        active_simulated_peaks = local_active_simulated_peaks
        detector_picker_source_rows = local_detector_picker_source_rows
        detector_picker_rows = [dict(entry) for entry in local_detector_picker_rows]
        detector_picker_grouped_candidates = {
            key: [dict(entry) for entry in entries]
            for key, entries in local_detector_picker_grouped_candidates.items()
        }
        grouped_candidates = candidate_groups
        simulated_lookup = build_simulated_lookup(active_simulated_peaks)
        cache_action = str(action)
        cache_source = str(source)
        cache_provenance = [str(step) for step in provenance]
        if (
            stale_reason_override is None
            and stale_reason == matching_empty_detector_cache_stale_reason
        ):
            stale_reason = matching_empty_detector_cache_stale_reason
        else:
            stale_reason = stale_reason_override
        return True

    def _source_rows_from_background_provider(
        consumer: str = "manual_pick_cache",
    ) -> list[dict[str, object]]:
        nonlocal source_rows_provider_calls
        if not callable(source_rows_for_background):
            return []
        source_rows_provider_calls += 1
        raw_rows = source_rows_for_background(
            bg_index,
            param_set,
            consumer=consumer,
        )
        return [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]

    def _existing_detector_picker_rows_for_background_refresh(
        cache_data: Mapping[str, object] | None,
    ) -> list[dict[str, object]]:
        if not isinstance(cache_data, Mapping):
            return []
        rows = geometry_manual_detector_picker_source_rows_from_cache(cache_data)
        if not rows:
            return []
        candidate_rows, _grouped = geometry_manual_detector_picker_candidates_from_rows(
            rows,
            display_background=background_image,
            profile_cache=profile_cache_for_picker,
        )
        if not candidate_rows:
            return []
        refreshed_rows: list[dict[str, object]] = []
        for raw_entry in rows:
            entry = dict(raw_entry)
            entry.setdefault(
                "detector_picker_background_refresh_source",
                str(entry.get("detector_picker_source", "existing_cache_detector_sources")),
            )
            refreshed_rows.append(entry)
        return refreshed_rows

    def _filter_reprojected_cache_rows(
        candidate_rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        filtered_rows: list[dict[str, object]] = []
        for raw_entry in candidate_rows or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            raw_display = _geometry_manual_finite_point(
                entry,
                (("sim_col_raw", "sim_row_raw"),),
            )
            current_display = _geometry_manual_entry_active_view_point(
                entry,
                use_caked_display=bool(reuse_requires_caked_projection),
            )
            if raw_display is None or current_display is None:
                continue
            if reuse_requires_caked_projection:
                try:
                    caked_col = float(entry.get("caked_x", np.nan))
                    caked_row = float(entry.get("caked_y", np.nan))
                except Exception:
                    continue
                if not (np.isfinite(caked_col) and np.isfinite(caked_row)):
                    continue
                if (
                    abs(float(current_display[0]) - float(caked_col)) > 1.0e-9
                    or abs(float(current_display[1]) - float(caked_row)) > 1.0e-9
                ):
                    continue
            filtered_rows.append(entry)
        return filtered_rows

    def _filter_cached_source_rows_for_reprojection(
        candidate_rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        filtered_rows: list[dict[str, object]] = []
        for raw_entry in candidate_rows or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            stable_pairs = (
                ("sim_col_raw", "sim_row_raw"),
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
                ("display_col", "display_row"),
                ("x", "y"),
            )
            has_reprojectable_coords = False
            for x_key, y_key in stable_pairs:
                try:
                    raw_col = float(entry.get(x_key, np.nan))
                    raw_row = float(entry.get(y_key, np.nan))
                except Exception:
                    continue
                if np.isfinite(raw_col) and np.isfinite(raw_row):
                    has_reprojectable_coords = True
                    break
            if not has_reprojectable_coords:
                continue
            filtered_rows.append(entry)
        return filtered_rows

    def _reproject_cache_rows(
        candidate_rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        normalized_rows = [
            dict(entry) for entry in (candidate_rows or ()) if isinstance(entry, Mapping)
        ]
        if not normalized_rows:
            return []
        if callable(project_peaks_for_background_view):
            order_key = "__ra_sim_manual_cache_projection_order__"
            grouped_rows: dict[int, list[dict[str, object]]] = {}
            ordered_backgrounds: list[int] = []
            for position, raw_entry in enumerate(normalized_rows):
                entry = dict(raw_entry)
                raw_background_idx = entry.get("background_index", bg_index)
                if raw_background_idx is None:
                    raw_background_idx = bg_index
                try:
                    background_idx = int(raw_background_idx)
                except Exception:
                    background_idx = int(bg_index)
                entry.setdefault("background_index", int(background_idx))
                entry[order_key] = int(position)
                if int(background_idx) not in grouped_rows:
                    ordered_backgrounds.append(int(background_idx))
                    grouped_rows[int(background_idx)] = []
                grouped_rows[int(background_idx)].append(entry)
            projected_rows: list[dict[str, object]] = []
            for background_idx in ordered_backgrounds:
                try:
                    projected_for_background = project_peaks_for_background_view(
                        int(background_idx),
                        grouped_rows[int(background_idx)],
                    )
                except Exception:
                    projected_for_background = []
                projected_rows.extend(
                    dict(entry)
                    for entry in (projected_for_background or ())
                    if isinstance(entry, Mapping)
                )
            sorted_rows = sorted(
                projected_rows,
                key=lambda entry: int(entry.get(order_key, int(1e12))),
            )
            cleaned_rows: list[dict[str, object]] = []
            for raw_entry in sorted_rows:
                entry = dict(raw_entry)
                entry.pop(order_key, None)
                cleaned_rows.append(entry)
            return cleaned_rows
        if callable(project_peaks_to_current_view):
            return [
                dict(entry)
                for entry in (project_peaks_to_current_view(normalized_rows) or ())
                if isinstance(entry, Mapping)
            ]
        return normalized_rows

    if prefer_cache and bg_index == current_bg_index:
        cached_simulated_peaks = geometry_manual_simulated_peaks_from_callback(
            simulated_peaks_for_params,
            param_set=param_set,
            prefer_cache=True,
        )
        if _apply_candidate_source(
            cached_simulated_peaks,
            action="reused",
            source="geometry_manual_simulated_peaks_for_params(prefer_cache=True)",
            provenance=[
                "geometry_manual_simulated_peaks_for_params(prefer_cache=True)",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=None,
        ):
            pass
        else:
            stale_reason = "cached preview rows were empty."
    if prefer_cache and bg_index == current_bg_index and not simulated_peaks:
        cached_simulated_peaks = geometry_manual_live_peak_candidates_from_records(peak_records)
        if _apply_candidate_source(
            cached_simulated_peaks,
            action="reused",
            source="peak_records",
            provenance=[
                "peak_records",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=None,
        ):
            pass
        elif stale_reason is None:
            stale_reason = "live peak records were empty."
    if prefer_cache and not simulated_peaks:
        cached_simulated_peaks = _source_rows_from_background_provider()
        if _apply_candidate_source(
            cached_simulated_peaks,
            action="reused",
            source="geometry_manual_source_rows_for_background",
            provenance=[
                "geometry_manual_source_rows_for_background",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=None,
        ):
            pass
        elif stale_reason is None and callable(source_rows_for_background):
            stale_reason = "source snapshot rows were empty."
    if (
        prefer_cache
        and not simulated_peaks
        and bg_index == current_bg_index
        and not bool(picker_candidates_only)
    ):
        fresh_simulation_prefer_cache_false_attempted = True
        rebuilt_simulated_peaks = geometry_manual_simulated_peaks_from_callback(
            simulated_peaks_for_params,
            param_set=param_set,
            prefer_cache=False,
        )
        if _apply_candidate_source(
            rebuilt_simulated_peaks,
            action="rebuilt",
            source="geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
            provenance=[
                "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=stale_reason,
        ):
            pass
    if (
        prefer_cache
        and not simulated_peaks
        and isinstance(existing_cache_data, dict)
        and resolved_existing_placed_signature == placed_cache_sig
    ):
        cached_simulated_peaks = [
            dict(entry)
            for entry in (existing_cache_data.get("simulated_peaks", ()) or ())
            if isinstance(entry, dict)
        ]
        if _apply_candidate_source(
            cached_simulated_peaks,
            action="reused",
            source="existing_cache_data.simulated_peaks(mask_refresh)",
            provenance=[
                "existing_cache_data.simulated_peaks",
                "filter_active_rows",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override="reused placed candidates; rebuilt active mask cache.",
        ):
            pass
    if (
        prefer_cache
        and not simulated_peaks
        and not reuse_requires_caked_projection
        and isinstance(existing_cache_data, dict)
        and _manual_pick_cache_background_token_compatible(
            resolved_existing_placed_signature,
            placed_cache_sig,
        )
    ):
        cached_detector_picker_rows = _existing_detector_picker_rows_for_background_refresh(
            existing_cache_data
        )
        if _apply_candidate_source(
            cached_detector_picker_rows,
            action="reused",
            source="existing_cache_data.detector_picker_rows(background_refresh)",
            provenance=[
                "existing_cache_data.detector_picker_rows",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            reproject=False,
            stale_reason_override=(
                "background-only cache signature change; reused detector picker rows."
            ),
        ):
            detector_picker_reused_after_background_refresh = True
    if (
        prefer_cache
        and not simulated_peaks
        and isinstance(existing_cache_data, dict)
        and _manual_pick_cache_groups_reusable(
            resolved_existing_placed_signature,
            placed_cache_sig,
        )
    ):
        cached_simulated_peaks = [
            dict(entry)
            for entry in (existing_cache_data.get("simulated_peaks", ()) or ())
            if isinstance(entry, dict)
        ]
        cached_source_rows = _filter_cached_source_rows_for_reprojection(cached_simulated_peaks)
        if cached_source_rows and (
            callable(project_peaks_for_background_view) or callable(project_peaks_to_current_view)
        ):
            reprojected_cached_peaks = _filter_reprojected_cache_rows(
                _reproject_cache_rows(cached_source_rows)
            )
            if _apply_candidate_source(
                reprojected_cached_peaks,
                action="reused",
                source="existing_cache_data.simulated_peaks(reprojected)",
                provenance=[
                    "existing_cache_data.simulated_peaks",
                    (
                        "project_peaks_for_background_view"
                        if callable(project_peaks_for_background_view)
                        else "project_peaks_to_current_view"
                    ),
                    "build_grouped_candidates",
                    "build_simulated_lookup",
                ],
                reproject=False,
                stale_reason_override=(
                    "background-only cache signature change; reprojected cached simulated peaks."
                ),
            ):
                pass
            else:
                stale_reason = (
                    "background-only cache signature change; "
                    "cached simulated peaks could not be reprojected."
                )
                existing_cache_reproject_failed = True
        else:
            stale_reason = (
                "background-only cache signature change; "
                "cached simulated peaks could not be reprojected."
            )
            existing_cache_reproject_failed = True
    if (
        prefer_cache
        and not simulated_peaks
        and not reuse_requires_caked_projection
        and existing_cache_reproject_failed
        and callable(source_rows_for_background)
        and source_rows_provider_calls > 0
    ):
        retried_source_rows = _source_rows_from_background_provider()
        if _apply_candidate_source(
            retried_source_rows,
            action="reused",
            source="geometry_manual_source_rows_for_background",
            provenance=[
                "geometry_manual_source_rows_for_background",
                "detector_retry_after_stale_existing_cache",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=stale_reason,
        ):
            pass
    if (
        not prefer_cache
        and not simulated_peaks
        and bg_index == current_bg_index
        and not bool(picker_candidates_only)
    ):
        fresh_simulation_prefer_cache_false_attempted = True
        rebuilt_simulated_peaks = geometry_manual_simulated_peaks_from_callback(
            simulated_peaks_for_params,
            param_set=param_set,
            prefer_cache=False,
        )
        if _apply_candidate_source(
            rebuilt_simulated_peaks,
            action="rebuilt",
            source="geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
            provenance=[
                "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                "build_grouped_candidates",
                "build_simulated_lookup",
            ],
            stale_reason_override=stale_reason,
        ):
            pass
    if bg_index == current_bg_index and callable(source_rows_for_background):
        detector_picker_coverage_missing_before = _missing_listed_detector_picker_q_group_keys()
        if detector_picker_coverage_missing_before:
            detector_picker_coverage_rebuild_attempted = True
            coverage_rows = _source_rows_from_background_provider("manual_pick_cache_coverage")
            if coverage_rows:
                _coverage_detector_rows, coverage_grouped_candidates = (
                    geometry_manual_detector_picker_candidates_from_rows(
                        coverage_rows,
                        display_background=background_image,
                        profile_cache=profile_cache_for_picker,
                    )
                )
                coverage_group_keys = {
                    key
                    for key, entries in coverage_grouped_candidates.items()
                    if key is not None and entries
                }
                if not detector_picker_grouped_candidates or any(
                    key in coverage_group_keys for key in detector_picker_coverage_missing_before
                ):
                    _apply_candidate_source(
                        coverage_rows,
                        action="rebuilt",
                        source=(
                            "geometry_manual_source_rows_for_background(manual_pick_cache_coverage)"
                        ),
                        provenance=[
                            "geometry_manual_source_rows_for_background",
                            "manual_pick_cache_coverage",
                            "build_grouped_candidates",
                            "build_simulated_lookup",
                        ],
                        reproject=False,
                        stale_reason_override=stale_reason,
                    )
            detector_picker_coverage_missing_after = _missing_listed_detector_picker_q_group_keys()
    if not simulated_peaks and stale_reason is None:
        stale_reason = (
            "source snapshot rows were unavailable; no reusable cached "
            "grouped candidates were available."
        )
    match_cfg = dict(current_match_config())
    resolved_match_cfg = dict(match_cfg)
    background_context = None
    if background_image is not None and callable(auto_match_background_context):
        try:
            resolved_match_cfg, background_context = auto_match_background_context(
                background_image,
                match_cfg,
            )
        except Exception:
            resolved_match_cfg = dict(match_cfg)
            background_context = None

    caked_qr_projection_entries: list[dict[str, object]] = []
    caked_qr_projection_grouped_candidates: dict[
        tuple[object, ...],
        list[dict[str, object]],
    ] = {}
    caked_qr_projection_lookup: GeometryManualLookupMap = {}
    if build_requires_caked_projection:
        caked_projection_source_rows = list(active_simulated_peaks)
        if not caked_projection_source_rows and bool(build_caked_projection_sidecar):
            caked_projection_source_rows = list(detector_picker_source_rows or simulated_peaks)
        (
            caked_qr_projection_entries,
            caked_qr_projection_grouped_candidates,
            caked_qr_projection_lookup,
        ) = _geometry_manual_build_caked_qr_projection_cache(
            caked_projection_source_rows,
            project_peaks_to_caked_view or project_peaks_to_current_view,
            build_grouped_candidates,
            build_simulated_lookup,
            filter_active_rows,
        )

    cache_result = {
        "signature": cache_sig,
        "placed_signature": placed_cache_sig,
        "simulated_peaks": [dict(entry) for entry in simulated_peaks],
        "active_simulated_peaks": [dict(entry) for entry in active_simulated_peaks],
        "detector_picker_source_rows": [dict(entry) for entry in detector_picker_source_rows],
        "fresh_source_rows": [
            dict(entry)
            for entry in detector_picker_source_rows
            if not _geometry_manual_detector_picker_is_manual_saved_pair(entry)
        ],
        "detector_picker_rows": [dict(entry) for entry in detector_picker_rows],
        "detector_picker_grouped_candidates": {
            key: [dict(entry) for entry in entries]
            for key, entries in detector_picker_grouped_candidates.items()
        },
        "simulated_lookup": _geometry_manual_copy_lookup(simulated_lookup),
        "grouped_candidates": {
            key: [dict(entry) for entry in entries] for key, entries in grouped_candidates.items()
        },
        "caked_qr_projection_entries": [dict(entry) for entry in caked_qr_projection_entries],
        "caked_qr_projection_grouped_candidates": {
            key: [dict(entry) for entry in entries]
            for key, entries in caked_qr_projection_grouped_candidates.items()
        },
        "caked_qr_projection_lookup": _geometry_manual_copy_lookup(caked_qr_projection_lookup),
        "match_config": dict(resolved_match_cfg),
        "background_context": background_context,
        "cache_metadata": _manual_pick_cache_metadata(
            cache_action=cache_action,
            stale_reason=stale_reason,
            cache_source=cache_source,
            cache_provenance=cache_provenance,
            simulated_peaks=active_simulated_peaks,
            grouped_candidates=grouped_candidates,
            background_index=bg_index,
            current_background_index=current_bg_index,
            prefer_cache=prefer_cache,
        ),
    }
    source_rows_provider_attempted = bool(source_rows_provider_calls > 0)
    cache_metadata = dict(cache_result.get("cache_metadata") or {})
    cache_metadata["source_rows_provider_attempted"] = source_rows_provider_attempted
    cache_metadata["source_rows_provider_calls"] = int(source_rows_provider_calls)
    cache_metadata["fresh_simulation_prefer_cache_false_attempted"] = bool(
        fresh_simulation_prefer_cache_false_attempted
    )
    cache_metadata["picker_candidates_only"] = bool(picker_candidates_only)
    if bool(picker_candidates_only):
        cache_metadata["qr_refinement_skipped"] = True
        cache_metadata["qr_lookup_rebuild_skipped"] = True
    cache_metadata["detector_picker_coverage_rebuild_attempted"] = bool(
        detector_picker_coverage_rebuild_attempted
    )
    cache_metadata["active_simulated_peak_count"] = int(len(active_simulated_peaks))
    cache_metadata["detector_picker_candidate_count"] = int(len(detector_picker_rows))
    cache_metadata["caked_qr_projection_candidate_count"] = int(len(caked_qr_projection_entries))
    cache_metadata["build_pick_cache_wall_ms"] = _geometry_manual_elapsed_ms(build_start)
    listed_picker_q_group_keys = _listed_picker_q_group_keys()
    if listed_picker_q_group_keys:
        detector_picker_coverage_missing_after = _missing_listed_detector_picker_q_group_keys()
        cache_metadata["listed_q_group_key_count"] = int(len(listed_picker_q_group_keys))
        cache_metadata["detector_picker_group_count"] = int(len(detector_picker_grouped_candidates))
        cache_metadata["detector_picker_coverage_rebuild_attempted"] = bool(
            detector_picker_coverage_rebuild_attempted
        )
        if detector_picker_coverage_missing_before:
            cache_metadata["detector_picker_missing_listed_q_group_keys_before"] = [
                list(key) for key in detector_picker_coverage_missing_before
            ]
        if detector_picker_coverage_missing_after:
            cache_metadata["detector_picker_missing_listed_q_group_keys"] = [
                list(key) for key in detector_picker_coverage_missing_after
            ]
    if detector_picker_reused_after_background_refresh:
        cache_metadata["detector_picker_reused_after_background_refresh"] = True
    cache_metadata["caked_qr_projection_sidecar_requested"] = bool(build_caked_projection_sidecar)
    cache_metadata["caked_qr_projection_entry_count"] = int(len(caked_qr_projection_entries))
    cache_metadata["caked_qr_projection_group_count"] = int(
        len(caked_qr_projection_grouped_candidates)
    )
    cache_result["cache_metadata"] = cache_metadata

    if (
        prefer_cache
        and bg_index == current_bg_index
        and not reuse_requires_caked_projection
        and not detector_picker_rows
        and isinstance(existing_cache_data, dict)
    ):
        retained_source_rows = _existing_detector_picker_rows_for_background_refresh(
            existing_cache_data
        )
        if retained_source_rows:
            (
                retained_picker_rows,
                retained_grouped_candidates,
            ) = geometry_manual_detector_picker_candidates_from_rows(
                retained_source_rows,
                display_background=background_image,
                profile_cache=profile_cache_for_picker,
            )
            retained_cache = dict(existing_cache_data)
            retained_cache["signature"] = cache_sig
            retained_cache["placed_signature"] = placed_cache_sig
            retained_cache["detector_picker_source_rows"] = [
                dict(entry) for entry in retained_source_rows
            ]
            retained_cache["fresh_source_rows"] = [
                dict(entry)
                for entry in retained_source_rows
                if not _geometry_manual_detector_picker_is_manual_saved_pair(entry)
            ]
            retained_cache["detector_picker_rows"] = [dict(entry) for entry in retained_picker_rows]
            retained_cache["detector_picker_grouped_candidates"] = {
                key: [dict(entry) for entry in entries]
                for key, entries in retained_grouped_candidates.items()
            }
            retained_metadata = (
                dict(retained_cache.get("cache_metadata"))
                if isinstance(retained_cache.get("cache_metadata"), Mapping)
                else {}
            )
            retained_metadata.update(
                {
                    "cache_action": "reused",
                    "reused": True,
                    "rebuilt": False,
                    "stale_reason": (
                        "detector cache rebuild empty; retained previous detector picker rows."
                    ),
                    "cache_source": (
                        "existing_cache_data.detector_picker_rows(retained_after_empty_rebuild)"
                    ),
                    "cache_provenance": [
                        "existing_cache_data.detector_picker_rows",
                        "empty_detector_rebuild_guard",
                    ],
                    "prefer_cache": bool(prefer_cache),
                    "background_index": int(bg_index),
                    "current_background_index": int(current_bg_index),
                    "source_rows_provider_attempted": source_rows_provider_attempted,
                    "source_rows_provider_calls": int(source_rows_provider_calls),
                    "fresh_simulation_prefer_cache_false_attempted": bool(
                        fresh_simulation_prefer_cache_false_attempted
                    ),
                    "detector_picker_coverage_rebuild_attempted": bool(
                        detector_picker_coverage_rebuild_attempted
                    ),
                    "active_simulated_peak_count": int(len(active_simulated_peaks)),
                    "detector_picker_candidate_count": int(len(retained_picker_rows)),
                    "caked_qr_projection_candidate_count": int(
                        len(retained_cache.get("caked_qr_projection_entries", ()) or ())
                    ),
                    "build_pick_cache_wall_ms": _geometry_manual_elapsed_ms(build_start),
                    "detector_picker_reused_after_background_refresh": True,
                    "empty_rebuild_cache_source": str(cache_source),
                    "empty_rebuild_stale_reason": (
                        None if stale_reason is None else str(stale_reason)
                    ),
                }
            )
            retained_cache["cache_metadata"] = retained_metadata
            retained_cache["detector_picker_trace"] = geometry_manual_detector_picker_input_trace(
                retained_cache,
                background_index=bg_index,
                display_background=background_image,
                grouped_candidates=retained_grouped_candidates,
                profile_cache=profile_cache_for_picker,
            )
            return retained_cache, cache_sig, retained_cache

    cache_result["detector_picker_trace"] = geometry_manual_detector_picker_input_trace(
        cache_result,
        background_index=bg_index,
        display_background=background_image,
        grouped_candidates=detector_picker_grouped_candidates,
        profile_cache=(
            param_set.get("mosaic_params")
            if isinstance(param_set, Mapping)
            and isinstance(param_set.get("mosaic_params"), Mapping)
            else None
        ),
    )

    next_cache_signature = existing_cache_signature
    next_cache_data = existing_cache_data if isinstance(existing_cache_data, dict) else {}
    if bg_index == current_bg_index:
        next_cache_signature = cache_sig
        next_cache_data = cache_result
    return cache_result, next_cache_signature, next_cache_data


def geometry_manual_choose_group_at(
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] | None,
    col: float,
    row: float,
    *,
    window_size_px: float,
    use_caked_display: bool = False,
) -> tuple[tuple[object, ...] | None, list[dict[str, object]], float]:
    """Return the nearest clickable Qr/Qz group inside a local click window."""

    best_group_key = None
    best_group_entries: list[dict[str, object]] = []
    best_sort_key: tuple[object, ...] | None = None
    half_window = max(1.0, 0.5 * float(window_size_px))
    for group_key, candidate_entries in (grouped_candidates or {}).items():
        for candidate in candidate_entries or []:
            current_point = _geometry_manual_entry_active_view_point(
                candidate,
                use_caked_display=use_caked_display,
            )
            if current_point is None:
                continue
            if (
                abs(float(current_point[0]) - float(col)) > half_window
                or abs(float(current_point[1]) - float(row)) > half_window
            ):
                continue
            d2 = (float(current_point[0]) - float(col)) ** 2 + (
                float(current_point[1]) - float(row)
            ) ** 2
            source_label = _geometry_manual_q_group_source_label(group_key)
            candidate_sort_key = (
                float(d2),
                _geometry_manual_safe_float(candidate.get("qr", float("inf"))),
                _geometry_manual_safe_float(candidate.get("qz", float("inf"))),
                _geometry_manual_source_priority(source_label),
                str(source_label),
            )
            if best_sort_key is None or candidate_sort_key < best_sort_key:
                best_sort_key = candidate_sort_key
                best_group_key = group_key
                best_group_entries = [dict(entry) for entry in candidate_entries]

    if best_group_key is None or best_sort_key is None or not np.isfinite(float(best_sort_key[0])):
        return None, [], float("nan")
    return best_group_key, best_group_entries, float(np.sqrt(float(best_sort_key[0])))


def geometry_manual_zoom_bounds(
    col: float,
    row: float,
    image_shape: Sequence[int] | None,
    *,
    window_size_px: float = 100.0,
) -> tuple[float, float, float, float]:
    """Return clamped image-space bounds for a square manual-pick zoom window."""

    try:
        height = int(image_shape[0]) if image_shape is not None else 0
        width = int(image_shape[1]) if image_shape is not None else 0
    except Exception:
        height = 0
        width = 0
    width = max(1, width)
    height = max(1, height)
    half = max(1.0, 0.5 * float(window_size_px))

    x_min = float(col) - half
    x_max = float(col) + half
    y_min = float(row) - half
    y_max = float(row) + half

    if x_min < 0.0:
        x_max = min(float(width), x_max - x_min)
        x_min = 0.0
    if x_max > float(width):
        x_min = max(0.0, x_min - (x_max - float(width)))
        x_max = float(width)
    if y_min < 0.0:
        y_max = min(float(height), y_max - y_min)
        y_min = 0.0
    if y_max > float(height):
        y_min = max(0.0, y_min - (y_max - float(height)))
        y_max = float(height)

    return float(x_min), float(x_max), float(y_min), float(y_max)


def geometry_manual_anchor_axis_limits(
    value: float,
    span: float,
    anchor_fraction: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Return clamped axis limits that keep *value* at a fixed screen fraction."""

    try:
        span_signed = float(span)
    except Exception:
        span_signed = 0.0
    if not np.isfinite(span_signed) or abs(span_signed) <= 1.0e-12:
        value_f = float(value)
        return value_f, value_f

    try:
        frac = float(anchor_fraction)
    except Exception:
        frac = 0.5
    if not np.isfinite(frac):
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)

    try:
        bound_lo = float(min(lower_bound, upper_bound))
        bound_hi = float(max(lower_bound, upper_bound))
    except Exception:
        bound_lo = float(lower_bound)
        bound_hi = float(upper_bound)
    available_span = max(0.0, bound_hi - bound_lo)
    if available_span > 0.0:
        span_abs = min(abs(span_signed), available_span)
        span_signed = np.copysign(span_abs, span_signed)

    start = float(value) - frac * span_signed
    end = start + span_signed
    low = min(start, end)
    high = max(start, end)
    if low < bound_lo:
        shift = bound_lo - low
        start += shift
        end += shift
    if high > bound_hi:
        shift = high - bound_hi
        start -= shift
        end -= shift
    return float(start), float(end)


def geometry_manual_group_target_count(
    group_key: tuple[object, ...] | None,
    group_entries: Sequence[dict[str, object]] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> int:
    """Return how many manual background peaks a selected group should collect."""

    entries = [dict(entry) for entry in group_entries or [] if isinstance(entry, dict)]
    if not entries:
        return 0

    if isinstance(group_key, tuple) and len(group_key) >= 4:
        try:
            if int(group_key[2]) == 0:
                return 1
        except Exception:
            pass

    branch_slots: set[tuple[object, ...]] = set()
    for entry in entries:
        branch_slot = _geometry_manual_q_group_physical_branch_slot(
            entry,
            group_key=group_key,
        )
        if branch_slot is not None:
            branch_slots.add(tuple(branch_slot))
    non_00l_slots = {slot for slot in branch_slots if slot != ("00l",)}
    if non_00l_slots:
        return int(min(2, len(non_00l_slots)))
    if branch_slots == {("00l",)}:
        return 1

    for entry in entries:
        hkl = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl is None or int(hkl[0]) != 0 or int(hkl[1]) != 0:
            return int(min(2, len(entries)))
    return 1


def geometry_manual_pick_session_active(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
) -> bool:
    """Return whether a manual background-placement session is in progress."""

    if not isinstance(pick_session, dict):
        return False
    if pick_session.get("group_key") is None:
        return False
    if not isinstance(pick_session.get("group_entries"), list):
        return False
    if require_current_background:
        try:
            return int(pick_session.get("background_index")) == int(current_background_index)
        except Exception:
            return False
    return True


def geometry_manual_unassigned_group_candidates(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> list[dict[str, object]]:
    """Return manual-pick group candidates that do not yet have a BG assignment."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
    ):
        return []
    if pick_session is None:
        return []

    group_entries = pick_session.get("group_entries", [])
    pending_entries = pick_session.get("pending_entries", [])
    if not isinstance(group_entries, list) or not isinstance(pending_entries, list):
        return []
    assigned_keys = {
        candidate_source_key(entry) for entry in pending_entries if isinstance(entry, dict)
    }
    out: list[dict[str, object]] = []
    for raw_entry in group_entries:
        if not isinstance(raw_entry, dict):
            continue
        source_key = candidate_source_key(raw_entry)
        if source_key in assigned_keys:
            continue
        out.append(dict(raw_entry))
    return out


def _geometry_manual_background_branch_key(
    entry: Mapping[str, object] | None,
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    entry_dict = dict(entry)
    group_key = _geometry_manual_real_q_group_key(entry_dict)
    if group_key is not None:
        branch_slot = _geometry_manual_q_group_physical_branch_slot(
            entry_dict,
            group_key=group_key,
            profile_cache=profile_cache,
        )
        if branch_slot is not None:
            return ("q_group_physical_branch", group_key, tuple(branch_slot))
        branch_id, _branch_source = gui_mosaic_top.normalize_branch_id(
            entry_dict,
            target_key=group_key,
            profile_cache=profile_cache,
        )
        if branch_id:
            return ("q_group_branch", group_key, str(branch_id))
        branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
            _geometry_manual_identity_resolution_entry(entry_dict),
            allow_legacy_peak_fallback=True,
        )
        if branch_idx in {0, 1}:
            return ("q_group_branch_index", group_key, int(branch_idx))
    return candidate_source_key(entry_dict)


def _geometry_manual_replace_same_branch_entry(
    entries: Sequence[dict[str, object]] | None,
    new_entry: dict[str, object],
    *,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    new_key = _geometry_manual_background_branch_key(
        new_entry,
        candidate_source_key=candidate_source_key,
        profile_cache=profile_cache,
    )
    output: list[dict[str, object]] = []
    replaced = False
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        entry_key = _geometry_manual_background_branch_key(
            entry,
            candidate_source_key=candidate_source_key,
            profile_cache=profile_cache,
        )
        same_branch = bool(new_key is not None and entry_key == new_key)
        if not same_branch:
            same_branch = geometry_manual_source_entries_share_identity(entry, new_entry)
        if same_branch:
            if not replaced:
                output.append(dict(new_entry))
                replaced = True
            continue
        output.append(entry)
    if not replaced:
        output.append(dict(new_entry))
    return output


def geometry_manual_current_pending_candidate(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[str, object] | None:
    """Return one remaining simulated peak awaiting a manual background click."""

    remaining = geometry_manual_unassigned_group_candidates(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
        candidate_source_key=candidate_source_key,
    )
    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        remaining,
        candidate_source_key=candidate_source_key,
    )
    if tagged_candidate is not None:
        return tagged_candidate
    if not remaining:
        return None
    return dict(remaining[0])


def geometry_manual_nearest_candidate_to_point(
    col: float,
    row: float,
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    use_caked_display: bool = False,
) -> tuple[dict[str, object] | None, float]:
    """Return the nearest simulated candidate to one display-space point."""

    best_entry = None
    best_d2 = float("inf")
    for raw_entry in candidate_entries or []:
        if not isinstance(raw_entry, dict):
            continue
        current_point = _geometry_manual_entry_active_view_point(
            raw_entry,
            use_caked_display=use_caked_display,
        )
        if current_point is None:
            continue
        d2 = (float(current_point[0]) - float(col)) ** 2 + (
            float(current_point[1]) - float(row)
        ) ** 2
        if d2 < best_d2:
            best_d2 = float(d2)
            best_entry = dict(raw_entry)
    if best_entry is None or not np.isfinite(best_d2):
        return None, float("nan")
    return best_entry, float(np.sqrt(best_d2))


def _geometry_manual_saved_background_point(
    entry: Mapping[str, object] | None,
    *,
    use_caked_display: bool,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    if bool(use_caked_display):
        point = _geometry_manual_finite_point(
            entry,
            (
                ("refined_background_two_theta_deg", "refined_background_phi_deg"),
                ("background_two_theta_deg", "background_phi_deg"),
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
            ),
        )
        if point is not None:
            return point
        if _geometry_manual_entry_display_frame(entry) == "caked":
            return _geometry_manual_finite_point(entry, (("display_col", "display_row"),))
        return None

    point = _geometry_manual_tuple_point(entry, "geometry_detector_display_px")
    if point is not None:
        return point
    return _geometry_manual_finite_point(
        entry,
        (
            ("refined_detector_display_col", "refined_detector_display_row"),
            ("x", "y"),
        ),
    )


def _geometry_manual_saved_background_pair_at(
    entries: Sequence[dict[str, object]] | None,
    col: float,
    row: float,
    *,
    use_caked_display: bool,
    window_size: float,
) -> tuple[int | None, dict[str, object] | None, float]:
    best_index: int | None = None
    best_entry: dict[str, object] | None = None
    best_d2 = float("inf")
    try:
        hit_window = max(0.0, float(window_size))
    except Exception:
        hit_window = 0.0
    if hit_window <= 0.0:
        return None, None, float("nan")

    for index, raw_entry in enumerate(entries or ()):
        if not isinstance(raw_entry, dict):
            continue
        point = _geometry_manual_saved_background_point(
            raw_entry,
            use_caked_display=bool(use_caked_display),
        )
        if point is None:
            continue
        d2 = (float(point[0]) - float(col)) ** 2 + (float(point[1]) - float(row)) ** 2
        if d2 < best_d2:
            best_d2 = float(d2)
            best_index = int(index)
            best_entry = dict(raw_entry)

    if best_entry is None or not np.isfinite(best_d2):
        return None, None, float("nan")
    distance = float(np.sqrt(best_d2))
    if distance > float(hit_window):
        return None, None, float("nan")
    return best_index, best_entry, distance


def _geometry_manual_points_match(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    *,
    tol: float = 1.0e-9,
) -> bool:
    return bool(
        left is not None
        and right is not None
        and abs(float(left[0]) - float(right[0])) <= float(tol)
        and abs(float(left[1]) - float(right[1])) <= float(tol)
    )


def _geometry_manual_candidate_detector_sim_point(
    candidate: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    refined_point = _geometry_manual_tuple_point(candidate, "sim_refined_detector_display_px")
    if refined_point is None:
        refined_point = _geometry_manual_finite_point(
            candidate,
            (("refined_sim_x", "refined_sim_y"),),
        )
    if refined_point is not None and not _geometry_manual_detector_point_is_caked(
        candidate,
        refined_point,
    ):
        return refined_point
    detector_point = _geometry_manual_entry_detector_display_point(candidate)
    if detector_point is None:
        return None
    caked_point = _geometry_manual_finite_point(
        candidate,
        (
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("two_theta_deg", "phi_deg"),
        ),
    )
    if _geometry_manual_points_match(detector_point, caked_point):
        return None
    return detector_point


def _geometry_manual_candidate_visual_detector_sim_point(
    candidate: Mapping[str, object] | None,
) -> tuple[tuple[float, float], str] | tuple[None, str]:
    visual_detector = _geometry_manual_tuple_point(candidate, "sim_visual_detector_display_px")
    if visual_detector is not None and not _geometry_manual_detector_point_is_caked(
        candidate,
        visual_detector,
    ):
        return visual_detector, "sim_visual_detector_display_px"
    refined_point = _geometry_manual_tuple_point(candidate, "sim_refined_detector_display_px")
    if refined_point is None:
        refined_point = _geometry_manual_finite_point(
            candidate,
            (("refined_sim_x", "refined_sim_y"),),
        )
    if refined_point is not None and not _geometry_manual_detector_point_is_caked(
        candidate,
        refined_point,
    ):
        return refined_point, "sim_visual_detector_display_px"
    detector_point = _geometry_manual_entry_detector_display_point(candidate)
    if detector_point is not None:
        return detector_point, "sim_visual_detector_display_px"
    native_point = _geometry_manual_candidate_native_sim_point(candidate)
    if native_point is not None:
        return native_point, "sim_visual_detector_native_px"
    return None, "<unavailable>"


def _geometry_manual_candidate_native_sim_point(
    candidate: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    visual_native = _geometry_manual_tuple_point(candidate, "sim_visual_detector_native_px")
    if visual_native is not None:
        return visual_native
    refined_native = _geometry_manual_tuple_point(candidate, "sim_refined_detector_native_px")
    if refined_native is not None:
        return refined_native
    return _geometry_manual_finite_point(
        candidate,
        (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("native_col", "native_row"),
            ("sim_native_x", "sim_native_y"),
            ("simulated_detector_x", "simulated_detector_y"),
        ),
    )


def _geometry_manual_candidate_caked_sim_point(
    candidate: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    refined_caked = _geometry_manual_tuple_point(candidate, "sim_refined_caked_deg")
    if refined_caked is not None:
        return refined_caked
    refined_caked = _geometry_manual_finite_point(
        candidate,
        (("refined_sim_caked_x", "refined_sim_caked_y"),),
    )
    if refined_caked is not None:
        return refined_caked
    simulated_caked = _geometry_manual_finite_point(
        candidate,
        (("simulated_two_theta_deg", "simulated_phi_deg"),),
    )
    if simulated_caked is not None:
        return simulated_caked
    caked_display = _geometry_manual_tuple_point(candidate, "sim_caked_display")
    if caked_display is not None:
        return caked_display
    return _geometry_manual_finite_point(
        candidate,
        (
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("two_theta_deg", "phi_deg"),
        ),
    )


def _geometry_manual_candidate_visual_caked_sim_point(
    candidate: Mapping[str, object] | None,
) -> tuple[tuple[float, float], str] | tuple[None, str]:
    visual_caked = _geometry_manual_tuple_point(candidate, "sim_visual_caked_deg")
    if visual_caked is not None:
        return visual_caked, "sim_visual_caked_deg"
    visual_caked = _geometry_manual_tuple_point(candidate, "sim_visual_deg")
    if visual_caked is not None:
        return visual_caked, "sim_visual_caked_deg"
    visual_caked = _geometry_manual_tuple_point(candidate, "sim_caked")
    if visual_caked is not None:
        return visual_caked, "sim_visual_caked_deg"
    fallback_caked = _geometry_manual_candidate_caked_sim_point(candidate)
    if fallback_caked is not None:
        return fallback_caked, "sim_visual_caked_deg"
    return None, "<unavailable>"


def _geometry_manual_wrap_phi_delta_deg(delta_phi: float) -> float:
    return ((float(delta_phi) + 180.0) % 360.0) - 180.0


def _geometry_manual_caked_delta_deg(
    point: tuple[float, float] | None,
    sim_point: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point is None or sim_point is None:
        return None
    return (
        float(point[0]) - float(sim_point[0]),
        _geometry_manual_wrap_phi_delta_deg(float(point[1]) - float(sim_point[1])),
    )


def _geometry_manual_caked_distance_deg(
    point: tuple[float, float] | None,
    sim_point: tuple[float, float] | None,
) -> float:
    delta = _geometry_manual_caked_delta_deg(point, sim_point)
    if delta is None:
        return float("nan")
    return float(np.hypot(float(delta[0]), float(delta[1])))


def geometry_manual_candidate_distance_details(
    col: float,
    row: float,
    candidate: dict[str, object] | None,
    *,
    use_caked_display: bool = False,
) -> dict[str, object]:
    """Return user-facing and cache/current distances for one candidate."""

    if not isinstance(candidate, dict):
        return {
            "distance": float("nan"),
            "distance_source": "<unavailable>",
            "cache_current_distance": float("nan"),
            "cache_current_delta": None,
        }
    point = (float(col), float(row))
    if bool(use_caked_display):
        visual_point, visual_source = _geometry_manual_candidate_visual_caked_sim_point(candidate)
        current_point = _geometry_manual_entry_matching_current_view_point(candidate)
        visual_distance = _geometry_manual_caked_distance_deg(point, visual_point)
        cache_distance = _geometry_manual_caked_distance_deg(point, current_point)
        return {
            "distance": float(visual_distance),
            "distance_source": str(visual_source),
            "distance_units": "deg",
            "visual_caked_point": visual_point,
            "visual_point_frame": "caked_display" if visual_point is not None else None,
            "visual_point_units": "deg" if visual_point is not None else None,
            "cached_two_theta_deg": (float(visual_point[0]) if visual_point is not None else None),
            "cached_phi_deg": (float(visual_point[1]) if visual_point is not None else None),
            "cached_point_source": str(visual_source),
            "cached_projection_frame": "caked_display" if visual_point is not None else None,
            "cached_units": "deg" if visual_point is not None else None,
            "cache_current_caked_point": current_point,
            "cache_current_distance": float(cache_distance),
            "cache_current_delta": _geometry_manual_caked_delta_deg(point, current_point),
        }

    visual_point, visual_source = _geometry_manual_candidate_visual_detector_sim_point(candidate)
    visual_caked_point, _visual_caked_source = _geometry_manual_candidate_visual_caked_sim_point(
        candidate
    )
    if visual_point is None:
        distance = float("nan")
    else:
        distance = float(
            np.hypot(
                float(visual_point[0]) - float(col),
                float(visual_point[1]) - float(row),
            )
        )
    return {
        "distance": float(distance),
        "distance_source": str(visual_source),
        "distance_units": "px",
        "visual_detector_display_point": (
            visual_point if visual_source == "sim_visual_detector_display_px" else None
        ),
        "visual_detector_native_point": (
            visual_point if visual_source == "sim_visual_detector_native_px" else None
        ),
        "visual_caked_point": visual_caked_point,
        "cache_current_distance": float("nan"),
        "cache_current_delta": None,
    }


def _geometry_manual_status_float_text(value: object) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "<unavailable>"
    if not np.isfinite(numeric):
        return "<unavailable>"
    return f"{float(numeric):.6f}"


def _geometry_manual_status_point_text(point: object) -> str:
    if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
        return "<unavailable>"
    try:
        x_val = float(point[0])
        y_val = float(point[1])
    except Exception:
        return "<unavailable>"
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return "<unavailable>"
    return f"({float(x_val):.6f},{float(y_val):.6f})"


def geometry_manual_status_distance_trace(
    distance_details: Mapping[str, object] | None,
    *,
    observed_refined_caked_deg: tuple[float, float] | None = None,
    use_caked_space: bool,
) -> dict[str, object]:
    details = distance_details if isinstance(distance_details, Mapping) else {}
    try:
        distance_value = float(details.get("distance", float("nan")))
    except Exception:
        distance_value = float("nan")
    try:
        cache_distance_value = float(details.get("cache_current_distance", float("nan")))
    except Exception:
        cache_distance_value = float("nan")
    distance_source = str(details.get("distance_source", "<unavailable>"))
    units = str(details.get("distance_units", "deg" if use_caked_space else "px"))
    trace = {
        "status_distance_source": distance_source,
        "status_distance_value": float(distance_value),
        "status_distance_units": units,
        "status_sim_visual_caked_deg": details.get("visual_caked_point"),
        "status_sim_cache_current_caked_deg": details.get("cache_current_caked_point"),
        "status_observed_refined_caked_deg": observed_refined_caked_deg,
        "status_distance_to_cache_current_deg": float(cache_distance_value),
    }
    return trace


def geometry_manual_format_status_distance_trace(trace: Mapping[str, object]) -> str:
    return (
        "status_distance_source={source} "
        "status_distance_value={value} "
        "status_distance_units={units} "
        "status_sim_visual_caked_deg={visual} "
        "status_sim_cache_current_caked_deg={cache} "
        "status_observed_refined_caked_deg={observed} "
        "status_distance_to_cache_current_deg={cache_distance}"
    ).format(
        source=str(trace.get("status_distance_source", "<unavailable>")),
        value=_geometry_manual_status_float_text(trace.get("status_distance_value")),
        units=str(trace.get("status_distance_units", "<unavailable>")),
        visual=_geometry_manual_status_point_text(trace.get("status_sim_visual_caked_deg")),
        cache=_geometry_manual_status_point_text(trace.get("status_sim_cache_current_caked_deg")),
        observed=_geometry_manual_status_point_text(trace.get("status_observed_refined_caked_deg")),
        cache_distance=_geometry_manual_status_float_text(
            trace.get("status_distance_to_cache_current_deg")
        ),
    )


_LIVE_CAKED_TRACE_Q_GROUP_KEY = ("q_group", "primary", 1, 10)
_LIVE_CAKED_TRACE_HKL = (-1, 0, 10)
_LIVE_CAKED_TRACE_LAST: dict[int, tuple[float, float] | None] = {}


def _geometry_manual_env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def geometry_manual_live_caked_trace_enabled() -> bool:
    if _geometry_manual_env_flag("RA_SIM_SUPPRESS_LIVE_CAKED_TRACE"):
        return False
    return _geometry_manual_env_flag("RA_SIM_LIVE_CAKED_TRACE")


def geometry_manual_live_caked_trace_emit_unchanged() -> bool:
    return _geometry_manual_env_flag("RA_SIM_LIVE_CAKED_TRACE_ALL")


def _geometry_manual_trace_branch(entry: Mapping[str, object] | None) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("source_branch_index", "branch_index"):
        if key not in entry:
            continue
        try:
            return int(entry.get(key))
        except Exception:
            return None
    return None


def _geometry_manual_trace_hkl(entry: Mapping[str, object] | None) -> tuple[int, int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    raw_hkl = entry.get("hkl")
    if not isinstance(raw_hkl, (list, tuple, np.ndarray)) or len(raw_hkl) < 3:
        return None
    try:
        return (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
    except Exception:
        return None


def _geometry_manual_trace_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, Mapping):
        raw_key = value.get("q_group_key")
    else:
        raw_key = value
    if isinstance(raw_key, tuple):
        return raw_key
    if isinstance(raw_key, list):
        return tuple(raw_key)
    return None


def _geometry_manual_trace_target_entry(entry: Mapping[str, object] | None) -> bool:
    return bool(
        _geometry_manual_trace_group_key(entry) == _LIVE_CAKED_TRACE_Q_GROUP_KEY
        and _geometry_manual_trace_hkl(entry) == _LIVE_CAKED_TRACE_HKL
    )


def _geometry_manual_trace_branch_map(
    entries: Sequence[Mapping[str, object]] | object | None,
) -> dict[int, dict[str, object]]:
    mapped: dict[int, dict[str, object]] = {}
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return mapped
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping) or not _geometry_manual_trace_target_entry(raw_entry):
            continue
        branch = _geometry_manual_trace_branch(raw_entry)
        if branch is not None:
            mapped[int(branch)] = dict(raw_entry)
    return mapped


def _geometry_manual_trace_points_match(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    *,
    tol: float = 1.0e-3,
) -> bool:
    return bool(
        left is not None
        and right is not None
        and abs(float(left[0]) - float(right[0])) <= float(tol)
        and abs(_geometry_manual_wrap_phi_delta_deg(float(left[1]) - float(right[1]))) <= float(tol)
    )


def _geometry_manual_trace_changed(branch: int, point: tuple[float, float] | None) -> str:
    previous = _LIVE_CAKED_TRACE_LAST.get(int(branch))
    changed = "no" if _geometry_manual_trace_points_match(previous, point) else "yes"
    _LIVE_CAKED_TRACE_LAST[int(branch)] = point
    return changed


def _geometry_manual_trace_classification(
    point: tuple[float, float] | None,
    *,
    candidate_point: tuple[float, float] | None,
    pending_point: tuple[float, float] | None,
    saved_point: tuple[float, float] | None,
    cache_point: tuple[float, float] | None,
) -> str:
    if point is None:
        return "fallback"
    if _geometry_manual_trace_points_match(point, candidate_point):
        return "clicked_visual_candidate"
    if _geometry_manual_trace_points_match(point, pending_point):
        return "pending_visual_map"
    if _geometry_manual_trace_points_match(point, saved_point):
        return "saved_pair_visual"
    if _geometry_manual_trace_points_match(point, cache_point):
        return "cache_current"
    return "fallback"


def geometry_manual_trace_live_caked_visual_source_event(
    event: str,
    *,
    manual_geometry_run_id: object | None = None,
    selected_click_caked_deg: tuple[float, float] | None = None,
    selected_candidate: Mapping[str, object] | None = None,
    pending_entries: Sequence[Mapping[str, object]] | None = None,
    preview_state: Mapping[str, object] | None = None,
    placement_entry: Mapping[str, object] | None = None,
    saved_entries: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """Print live-only visual-source ledger rows for the (-1,0,10) caked picker."""

    if not geometry_manual_live_caked_trace_enabled():
        return

    resolved_run_id = (
        str(manual_geometry_run_id)
        if manual_geometry_run_id is not None
        else _geometry_manual_entry_run_id(
            selected_candidate,
            pending_entries,
            preview_state,
            placement_entry,
            saved_entries,
        )
    )
    candidate_branch = _geometry_manual_trace_branch(selected_candidate)
    candidate_point, _candidate_source = _geometry_manual_candidate_visual_caked_sim_point(
        selected_candidate
    )
    pending_map = _geometry_manual_trace_branch_map(pending_entries)
    saved_map = _geometry_manual_trace_branch_map(saved_entries)
    placement_branch = _geometry_manual_trace_branch(placement_entry)
    placement_point, _placement_source = _geometry_manual_candidate_visual_caked_sim_point(
        placement_entry
    )
    preview_branch = None
    preview_point = None
    if isinstance(preview_state, Mapping):
        preview_candidate = preview_state.get("candidate")
        preview_branch = _geometry_manual_trace_branch(
            preview_candidate if isinstance(preview_candidate, Mapping) else None
        )
        preview_point = _geometry_manual_tuple_point(
            preview_state,
            "status_sim_visual_caked_deg",
        )

    branches = sorted(
        {
            int(branch)
            for branch in (
                *(pending_map.keys()),
                *(saved_map.keys()),
                candidate_branch,
                placement_branch,
                preview_branch,
            )
            if branch is not None
        }
    )
    if not branches:
        return
    if not any(
        _geometry_manual_trace_target_entry(entry)
        for entry in (
            selected_candidate,
            placement_entry,
            *(pending_map.values()),
            *(saved_map.values()),
        )
        if isinstance(entry, Mapping)
    ):
        return

    for branch in branches:
        pending_entry = pending_map.get(int(branch))
        saved_entry = saved_map.get(int(branch))
        row_entry = (
            dict(placement_entry)
            if placement_branch == branch and isinstance(placement_entry, Mapping)
            else dict(saved_entry)
            if isinstance(saved_entry, Mapping)
            else dict(pending_entry)
            if isinstance(pending_entry, Mapping)
            else dict(selected_candidate)
            if candidate_branch == branch and isinstance(selected_candidate, Mapping)
            else {}
        )
        pending_point, _pending_source = _geometry_manual_candidate_visual_caked_sim_point(
            pending_entry
        )
        saved_point, _saved_source = _geometry_manual_candidate_visual_caked_sim_point(saved_entry)
        cache_source_entry = (
            dict(selected_candidate)
            if candidate_branch == branch and isinstance(selected_candidate, Mapping)
            else dict(pending_entry)
            if isinstance(pending_entry, Mapping)
            else {}
        )
        cache_point = _geometry_manual_entry_matching_current_view_point(cache_source_entry)
        branch_candidate_point = candidate_point if candidate_branch == branch else None
        branch_preview_point = preview_point if preview_branch == branch else None
        branch_placement_point = placement_point if placement_branch == branch else None
        actual_point = (
            branch_preview_point
            or branch_placement_point
            or saved_point
            or pending_point
            or branch_candidate_point
        )
        classification = _geometry_manual_trace_classification(
            actual_point,
            candidate_point=branch_candidate_point,
            pending_point=pending_point,
            saved_point=saved_point,
            cache_point=cache_point,
        )
        changed = _geometry_manual_trace_changed(int(branch), actual_point)
        if changed == "no" and not geometry_manual_live_caked_trace_emit_unchanged():
            continue
        print(
            "[ra-sim] live_caked_visual_source "
            + geometry_manual_cmd_provenance_text(
                run_id=resolved_run_id,
                emitter="geometry_manual_trace_live_caked_visual_source_event",
                event=event,
                branch=int(branch),
                actual_source=classification,
                expected_source="sim_visual_caked_deg",
            )
            + " "
            f"event={event} "
            f"branch={int(branch)} "
            f"q_group_key={_LIVE_CAKED_TRACE_Q_GROUP_KEY!r} "
            f"hkl={_LIVE_CAKED_TRACE_HKL!r} "
            f"source_table_index={row_entry.get('source_table_index', '<none>')} "
            f"source_row_index={row_entry.get('source_row_index', '<none>')} "
            f"source_branch_index={row_entry.get('source_branch_index', '<none>')} "
            f"branch_id={row_entry.get('branch_id', '<none>')} "
            f"selected_click_caked_deg={_geometry_manual_status_point_text(selected_click_caked_deg)} "
            f"candidate_used_for_selection_caked_deg={_geometry_manual_status_point_text(branch_candidate_point)} "
            f"pending_visual_map_caked_deg={_geometry_manual_status_point_text(pending_point)} "
            f"saved_pair_sim_visual_caked_deg={_geometry_manual_status_point_text(saved_point)} "
            f"preview_status_sim_visual_caked_deg={_geometry_manual_status_point_text(branch_preview_point)} "
            f"placement_sim_visual_caked_deg={_geometry_manual_status_point_text(branch_placement_point)} "
            f"cache_current_caked_deg={_geometry_manual_status_point_text(cache_point)} "
            f"source_classification={classification} "
            f"changed_since_previous_event={changed}"
        )


def _geometry_manual_pair_provenance_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if int(value.size) > 64:
            return repr(value)
        return tuple(_geometry_manual_pair_provenance_value(item) for item in value.tolist())
    if isinstance(value, tuple):
        return tuple(_geometry_manual_pair_provenance_value(item) for item in value)
    if isinstance(value, list):
        if len(value) > 64:
            return repr(value)
        return tuple(_geometry_manual_pair_provenance_value(item) for item in value)
    if isinstance(value, Mapping):
        return {
            str(key): _geometry_manual_pair_provenance_value(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
            if raw_value is not None
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else repr(value)
    return repr(value)


def geometry_manual_pair_entry_from_candidate(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    group_key: tuple[object, ...] | None,
    detector_col: float | None = None,
    detector_row: float | None = None,
    raw_col: float | None = None,
    raw_row: float | None = None,
    caked_col: float | None = None,
    caked_row: float | None = None,
    raw_caked_col: float | None = None,
    raw_caked_row: float | None = None,
    placement_error_px: float | None = None,
    sigma_px: float | None = None,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
) -> dict[str, object] | None:
    """Build one saved manual pair entry from a candidate + measured background point."""

    if not isinstance(candidate, dict):
        return None
    source_label = _geometry_manual_entry_source_label(candidate, group_key=group_key)
    phase_label = str(
        candidate.get("phase_label") or _geometry_manual_source_phase_label(source_label)
    )
    structure_role = str(
        candidate.get("structure_role") or _geometry_manual_source_structure_role(source_label)
    )
    entry: dict[str, object] = {
        "label": str(candidate.get("label", "")),
        "hkl": normalize_hkl_key(candidate.get("hkl", candidate.get("label"))),
        "x": float(peak_col),
        "y": float(peak_row),
        "obs_detector_x": float(peak_col),
        "obs_detector_y": float(peak_row),
        "source_table_index": candidate.get("source_table_index"),
        "source_reflection_index": candidate.get("source_reflection_index"),
        "source_reflection_namespace": candidate.get("source_reflection_namespace"),
        "source_reflection_is_full": candidate.get("source_reflection_is_full"),
        "source_row_index": candidate.get("source_row_index"),
        "source_branch_index": candidate.get("source_branch_index"),
        "source_peak_index": candidate.get("source_peak_index"),
        "source_label": source_label,
        "phase_label": phase_label,
        "structure_role": structure_role,
        "q_group_key": group_key,
    }
    provenance_keys = set(
        key for key in candidate if str(key).startswith(("source_", "ray_", "reflection_"))
    )
    provenance_keys.update(
        (
            "branch_id",
            "branch_source",
            "best_sample_index",
            "mosaic_weight",
            "mosaic_top_rank_key",
            "selection_reason",
            "selection_scope",
            "selected_q_group_key",
            "phase_label",
            "structure_role",
            "raw_detector_display_px",
            "raw_detector_native_px",
            "raw_detector_native_source",
            "geometry_detector_display_px",
            "geometry_detector_native_px",
            "geometry_detector_native_source",
            "raw_caked_deg",
            "geometry_caked_deg",
            "sim_nominal_detector_display_px",
            "sim_nominal_detector_native_px",
            "sim_nominal_caked_deg",
            "sim_visual_detector_display_px",
            "sim_visual_detector_native_px",
            "sim_visual_detector_native_existing",
            "sim_visual_detector_native_source",
            "sim_visual_caked_deg",
            "sim_visual_caked_source",
            "sim_refined_detector_display_px",
            "sim_refined_detector_native_px",
            "sim_refined_caked_deg",
            "sim_refinement_delta_detector_px",
            "sim_refinement_delta_caked_deg",
            "sim_refinement_source",
            "sim_refinement_status",
            "sim_detector_anchor_x",
            "sim_detector_anchor_y",
            "sim_detector_display_col",
            "sim_detector_display_row",
            "sim_detector_frame_provenance",
            "sim_refined_caked_projection_callback",
            "sim_refined_caked_projection_real_callback",
            "sim_refined_caked_projection_status",
            "sim_visual_deg",
            "sim_visual_source",
            "sim_caked",
            "sim_caked_display",
            "simulated_two_theta_deg",
            "simulated_phi_deg",
        )
    )
    for provenance_key in sorted(provenance_keys):
        if provenance_key in candidate:
            entry[provenance_key] = _geometry_manual_pair_provenance_value(
                candidate.get(provenance_key)
            )
    if detector_col is not None and detector_row is not None:
        entry["detector_x"] = float(detector_col)
        entry["detector_y"] = float(detector_row)
    if raw_col is not None and raw_row is not None:
        entry["raw_x"] = float(raw_col)
        entry["raw_y"] = float(raw_row)
    if caked_col is not None and caked_row is not None:
        entry["manual_background_input_frame"] = "caked_2theta_phi"
        entry["background_two_theta_deg"] = float(caked_col)
        entry["background_phi_deg"] = float(caked_row)
        entry["caked_x"] = float(caked_col)
        entry["caked_y"] = float(caked_row)
    if raw_caked_col is not None and raw_caked_row is not None:
        entry["raw_caked_x"] = float(raw_caked_col)
        entry["raw_caked_y"] = float(raw_caked_row)
        entry.setdefault("background_two_theta_deg", float(raw_caked_col))
        entry.setdefault("background_phi_deg", float(raw_caked_row))

    simulated_detector_point = _geometry_manual_candidate_detector_sim_point(candidate)
    if simulated_detector_point is not None:
        entry["refined_sim_x"] = float(simulated_detector_point[0])
        entry["refined_sim_y"] = float(simulated_detector_point[1])
        entry["sim_detector_x"] = float(simulated_detector_point[0])
        entry["sim_detector_y"] = float(simulated_detector_point[1])

    simulated_native_point = _geometry_manual_candidate_native_sim_point(candidate)
    if simulated_native_point is not None:
        entry["refined_sim_native_x"] = float(simulated_native_point[0])
        entry["refined_sim_native_y"] = float(simulated_native_point[1])

    fit_caked_point = _geometry_manual_candidate_caked_sim_point(candidate)
    if fit_caked_point is None and (
        bool(candidate.get("_caked_qr_projection_cache"))
        or str(candidate.get("display_frame", candidate.get("current_view_frame", "")) or "")
        .strip()
        .lower()
        == "caked_display"
    ):
        fit_caked_point = _geometry_manual_finite_point(
            candidate,
            (("caked_x", "caked_y"),),
        )
    visual_caked_point = _geometry_manual_tuple_point(candidate, "sim_visual_caked_deg")
    if visual_caked_point is None:
        visual_caked_point = _geometry_manual_tuple_point(candidate, "sim_visual_deg")
    if visual_caked_point is None:
        visual_caked_point = _geometry_manual_tuple_point(candidate, "sim_caked")
    if visual_caked_point is None:
        visual_caked_point = fit_caked_point
    if fit_caked_point is not None:
        fit_caked_x = float(fit_caked_point[0])
        fit_caked_y = float(fit_caked_point[1])
        fit_caked_pair = (fit_caked_x, fit_caked_y)
        entry["refined_sim_caked_x"] = fit_caked_x
        entry["refined_sim_caked_y"] = fit_caked_y
        entry["simulated_two_theta_deg"] = fit_caked_x
        entry["simulated_phi_deg"] = fit_caked_y
        entry["sim_caked_display"] = fit_caked_pair
        entry["sim_refined_caked_deg"] = fit_caked_pair
    if visual_caked_point is not None:
        visual_caked_pair = (float(visual_caked_point[0]), float(visual_caked_point[1]))
        entry["sim_visual_caked_deg"] = visual_caked_pair
        entry["sim_visual_deg"] = visual_caked_pair
        entry["sim_caked"] = visual_caked_pair
        entry.setdefault("sim_visual_source", "sim_visual_caked_deg")

    if placement_error_px is not None and np.isfinite(float(placement_error_px)):
        entry["placement_error_px"] = max(0.0, float(placement_error_px))
    if sigma_px is not None and np.isfinite(float(sigma_px)) and float(sigma_px) > 0.0:
        entry["sigma_px"] = float(sigma_px)
    _canonicalize_manual_entry_branch_fields(
        entry,
        allow_legacy_peak_fallback=False,
        preserve_legacy_peak_when_unresolved=False,
    )
    _copy_q_values_from_sources(entry, candidate)
    return entry


GEOMETRY_MANUAL_BACKGROUND_QR_REFERENCE_GROUP = "background_qr_reference"
GEOMETRY_MANUAL_BACKGROUND_QR_REFERENCE_LABEL_SOURCE = "background_2theta_phi"


def geometry_manual_background_qr_reference_label(
    two_theta_deg: object,
    phi_deg: object,
    *,
    precision: int = 4,
) -> str:
    """Return the HKL-name label for a background-only Qr reference."""

    try:
        two_theta_value = float(two_theta_deg)
        phi_value = float(phi_deg)
    except Exception:
        return "2theta=nan,phi=nan"
    digits = max(0, int(precision))
    return f"2theta={two_theta_value:.{digits}f},phi={phi_value:.{digits}f}"


def geometry_manual_entry_is_background_qr_reference(
    entry: Mapping[str, object] | None,
) -> bool:
    """Return whether a saved manual entry is a background-only Qr reference."""

    if not isinstance(entry, Mapping):
        return False
    return bool(
        entry.get("background_qr_set_reference", False)
        or entry.get("manual_background_reference", False)
    )


def geometry_manual_pick_session_is_background_qr_reference(
    pick_session: Mapping[str, object] | None,
) -> bool:
    """Return whether the active manual session places a background-only Qr point."""

    if not isinstance(pick_session, Mapping):
        return False
    if bool(pick_session.get("background_qr_set_reference_mode", False)):
        return True
    group_key = pick_session.get("group_key")
    return (
        isinstance(group_key, tuple)
        and len(group_key) >= 1
        and group_key[0] == GEOMETRY_MANUAL_BACKGROUND_QR_REFERENCE_GROUP
    )


def start_geometry_manual_background_qr_reference_session(
    *,
    current_background_index: object,
    base_entries: Sequence[Mapping[str, object]] | None = None,
    manual_run_id: object | None = None,
) -> dict[str, object]:
    """Build a one-click session for a background-only Qr reference point."""

    try:
        background_idx = int(current_background_index)
    except Exception:
        background_idx = 0
    run_id = str(manual_run_id) if manual_run_id is not None and str(manual_run_id) else ""
    if not run_id:
        run_id = geometry_manual_start_run_id()
    cleaned_base = [dict(entry) for entry in (base_entries or ()) if isinstance(entry, Mapping)]
    return {
        "background_qr_set_reference_mode": True,
        "background_index": int(background_idx),
        "group_key": (
            GEOMETRY_MANUAL_BACKGROUND_QR_REFERENCE_GROUP,
            int(background_idx),
            run_id,
        ),
        "group_entries": [],
        "pending_entries": [],
        "target_count": 1,
        "base_entries": cleaned_base,
        "q_label": "background Qr reference",
        "manual_geometry_run_id": run_id,
        "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
    }


def geometry_manual_background_qr_reference_pair_entry(
    *,
    peak_col: float,
    peak_row: float,
    raw_col: float | None = None,
    raw_row: float | None = None,
    detector_col: float | None = None,
    detector_row: float | None = None,
    caked_col: float | None = None,
    caked_row: float | None = None,
    raw_caked_col: float | None = None,
    raw_caked_row: float | None = None,
    placement_error_px: float | None = None,
    sigma_px: float | None = None,
    manual_run_id: object | None = None,
) -> dict[str, object]:
    """Build a saved manual pair for a background peak with no simulated HKL."""

    caked_label_col = caked_col if caked_col is not None else peak_col
    caked_label_row = caked_row if caked_row is not None else peak_row
    entry: dict[str, object] = {
        "label": geometry_manual_background_qr_reference_label(
            caked_label_col,
            caked_label_row,
        ),
        "x": float(peak_col),
        "y": float(peak_row),
        "background_qr_set_reference": True,
        "manual_background_reference": True,
        "geometry_fit_disabled": True,
        "background_qr_set_reference_label_source": (
            GEOMETRY_MANUAL_BACKGROUND_QR_REFERENCE_LABEL_SOURCE
        ),
        "source_label": "background",
        "selection_reason": "background_qr_set_reference",
        "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
    }
    if manual_run_id is not None and str(manual_run_id):
        entry["manual_geometry_run_id"] = str(manual_run_id)
    if raw_col is not None and raw_row is not None:
        entry["raw_x"] = float(raw_col)
        entry["raw_y"] = float(raw_row)
        entry["raw_detector_display_px"] = (float(raw_col), float(raw_row))
    entry["geometry_detector_display_px"] = (float(peak_col), float(peak_row))
    if detector_col is not None and detector_row is not None:
        entry["detector_x"] = float(detector_col)
        entry["detector_y"] = float(detector_row)
        entry["background_detector_x"] = float(detector_col)
        entry["background_detector_y"] = float(detector_row)
        entry["background_detector_input_frame"] = "native_detector"
    if caked_col is not None and caked_row is not None:
        entry["manual_background_input_frame"] = "caked_2theta_phi"
        entry["background_two_theta_deg"] = float(caked_col)
        entry["background_phi_deg"] = float(caked_row)
        entry["refined_background_two_theta_deg"] = float(caked_col)
        entry["refined_background_phi_deg"] = float(caked_row)
        entry["caked_x"] = float(caked_col)
        entry["caked_y"] = float(caked_row)
        entry["geometry_caked_deg"] = (float(caked_col), float(caked_row))
    if raw_caked_col is not None and raw_caked_row is not None:
        entry["raw_caked_x"] = float(raw_caked_col)
        entry["raw_caked_y"] = float(raw_caked_row)
        entry["raw_caked_deg"] = (float(raw_caked_col), float(raw_caked_row))
        entry["raw_caked_two_theta_deg"] = float(raw_caked_col)
        entry["raw_caked_phi_deg"] = float(raw_caked_row)
        entry.setdefault("background_two_theta_deg", float(raw_caked_col))
        entry.setdefault("background_phi_deg", float(raw_caked_row))
    if placement_error_px is not None and np.isfinite(float(placement_error_px)):
        entry["placement_error_px"] = max(0.0, float(placement_error_px))
    if sigma_px is not None and np.isfinite(float(sigma_px)) and float(sigma_px) > 0.0:
        entry["sigma_px"] = float(sigma_px)
    return entry


def _copy_q_values_from_sources(
    target: dict[str, object],
    *sources: Mapping[str, object] | None,
) -> None:
    """Copy finite ``qr``/``qz`` values from the first source that exposes them."""

    for field_name in ("qr", "qz"):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            try:
                value = float(source.get(field_name, np.nan))
            except Exception:
                continue
            if np.isfinite(value):
                target[field_name] = float(value)
                break


def geometry_manual_preview_due(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    min_interval_s: float = DEFAULT_PREVIEW_MIN_INTERVAL_S,
    min_move_px: float = DEFAULT_PREVIEW_MIN_MOVE_PX,
    perf_counter_fn: Callable[[], float] = perf_counter,
) -> bool:
    """Throttle manual-placement preview updates during mouse motion."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return False
    if pick_session is None:
        return False

    now = float(perf_counter_fn())
    last_t = float(pick_session.get("preview_last_t", 0.0))
    last_xy = pick_session.get("preview_last_xy")
    due = False
    if not (isinstance(last_xy, tuple) and len(last_xy) >= 2):
        due = True
    else:
        dx = float(col) - float(last_xy[0])
        dy = float(row) - float(last_xy[1])
        if (dx * dx + dy * dy) >= float(min_move_px * min_move_px):
            due = True
    if not due and (now - last_t) >= float(min_interval_s):
        due = True
    if not due:
        return False
    pick_session["preview_last_t"] = float(now)
    pick_session["preview_last_xy"] = (float(col), float(row))
    return True


def geometry_manual_refine_preview_point(
    source_entry: dict[str, object] | None,
    raw_col: float,
    raw_row: float,
    *,
    display_background: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    allow_cache_build: bool = True,
    use_caked_space: bool,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    match_simulated_peaks_to_peak_context: Callable[
        [Sequence[dict[str, object]], dict[str, object], dict[str, object]],
        tuple[Sequence[dict[str, object]], object],
    ]
    | None = None,
    peak_maximum_near_in_image_fn: Callable[
        [np.ndarray | None, float, float], tuple[float, float]
    ] = peak_maximum_near_in_image,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
) -> tuple[float, float]:
    """Refine one manual raw click/release position to the best background peak."""

    background_local = display_background
    if background_local is None:
        return float(raw_col), float(raw_row)

    def _preserve_sim_seed(
        seed_entry: dict[str, object],
        key: str,
        fallback: float,
    ) -> float:
        try:
            current = float(seed_entry.get(key, np.nan))
        except Exception:
            current = float("nan")
        if np.isfinite(current):
            return float(current)
        seed_entry[key] = float(fallback)
        return float(fallback)

    state = cache_data if isinstance(cache_data, dict) else {}
    allow_build = bool(allow_cache_build)
    if isinstance(state, dict):
        metadata = state.get("cache_metadata")
        if bool(state.get("manual_no_build_cache", False)):
            allow_build = False
        if isinstance(metadata, Mapping) and metadata.get("allow_cache_build") is False:
            allow_build = False
    if not state and allow_build and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = {}
        if isinstance(built_state, dict):
            state = built_state

    match_cfg = dict(state.get("match_config", {})) if isinstance(state, dict) else {}
    background_context = state.get("background_context") if isinstance(state, dict) else None
    auto_refine_radius = None
    if isinstance(state, dict) and state.get("manual_auto_refine_search_radius_px") is not None:
        try:
            raw_auto_refine_radius = float(state.get("manual_auto_refine_search_radius_px"))
        except Exception:
            raw_auto_refine_radius = float("nan")
        if np.isfinite(raw_auto_refine_radius):
            auto_refine_radius = geometry_manual_normalize_auto_refine_search_radius_px(
                raw_auto_refine_radius
            )
    if auto_refine_radius is not None:
        fallback_radius = max(3, int(round(float(auto_refine_radius))))
    else:
        fallback_radius = max(
            3,
            int(round(min(8.0, 0.33 * float(match_cfg.get("search_radius_px", 18.0))))),
        )

    refined_col = float(raw_col)
    refined_row = float(raw_row)
    used_peak_context = False
    if use_caked_space:
        radial_axis_arr = np.asarray(radial_axis, dtype=float)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float)
        raw_col_local = caked_axis_to_image_index_fn(float(raw_col), radial_axis_arr)
        raw_row_local = caked_axis_to_image_index_fn(float(raw_row), azimuth_axis_arr)
        if (
            isinstance(background_context, dict)
            and bool(background_context.get("img_valid", False))
            and callable(match_simulated_peaks_to_peak_context)
            and np.isfinite(raw_col_local)
            and np.isfinite(raw_row_local)
        ):
            seed_entry = dict(source_entry) if isinstance(source_entry, dict) else {}
            _preserve_sim_seed(seed_entry, "sim_col", float(raw_col))
            _preserve_sim_seed(seed_entry, "sim_row", float(raw_row))
            _preserve_sim_seed(seed_entry, "sim_col_global", float(raw_col))
            _preserve_sim_seed(seed_entry, "sim_row_global", float(raw_row))
            _preserve_sim_seed(seed_entry, "sim_col_local", float(raw_col_local))
            _preserve_sim_seed(seed_entry, "sim_row_local", float(raw_row_local))
            try:
                manual_matches, _manual_stats = match_simulated_peaks_to_peak_context(
                    [seed_entry],
                    background_context,
                    match_cfg,
                )
            except Exception:
                manual_matches = []
            if manual_matches:
                try:
                    refined_col = float(
                        caked_image_index_to_axis_fn(
                            float(manual_matches[0].get("x", raw_col_local)),
                            radial_axis_arr,
                        )
                    )
                    refined_row = float(
                        caked_image_index_to_axis_fn(
                            float(manual_matches[0].get("y", raw_row_local)),
                            azimuth_axis_arr,
                        )
                    )
                    used_peak_context = np.isfinite(refined_col) and np.isfinite(refined_row)
                except Exception:
                    refined_col = float(raw_col)
                    refined_row = float(raw_row)
        if not used_peak_context or not (np.isfinite(refined_col) and np.isfinite(refined_row)):
            refined_col, refined_row = refine_caked_peak_center_fn(
                np.asarray(background_local, dtype=float),
                radial_axis_arr,
                azimuth_axis_arr,
                float(raw_col),
                float(raw_row),
            )
        return float(refined_col), float(refined_row)

    if (
        isinstance(background_context, dict)
        and bool(background_context.get("img_valid", False))
        and callable(match_simulated_peaks_to_peak_context)
    ):
        seed_entry = dict(source_entry) if isinstance(source_entry, dict) else {}
        _preserve_sim_seed(seed_entry, "sim_col", float(raw_col))
        _preserve_sim_seed(seed_entry, "sim_row", float(raw_row))
        try:
            manual_matches, _manual_stats = match_simulated_peaks_to_peak_context(
                [seed_entry],
                background_context,
                match_cfg,
            )
        except Exception:
            manual_matches = []
        if manual_matches:
            try:
                refined_col = float(manual_matches[0].get("x", refined_col))
                refined_row = float(manual_matches[0].get("y", refined_row))
                used_peak_context = np.isfinite(refined_col) and np.isfinite(refined_row)
            except Exception:
                refined_col = float(raw_col)
                refined_row = float(raw_row)
    if not used_peak_context or not (np.isfinite(refined_col) and np.isfinite(refined_row)):
        refined_col, refined_row = peak_maximum_near_in_image_fn(
            background_local,
            float(raw_col),
            float(raw_row),
            search_radius=fallback_radius,
        )
    return float(refined_col), float(refined_row)


def refine_detector_pick_via_caked_background(
    candidate: dict[str, object] | None,
    raw_col: float,
    raw_row: float,
    *,
    detector_display_to_caked_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ],
    caked_angles_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ],
    caked_background: np.ndarray | None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    cache_data: dict[str, object] | None = None,
    match_simulated_peaks_to_peak_context: Callable[
        [Sequence[dict[str, object]], dict[str, object], dict[str, object]],
        tuple[Sequence[dict[str, object]], object],
    ]
    | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
) -> dict[str, float] | None:
    """Refine one detector-view click against the cached caked background."""

    if not callable(detector_display_to_caked_coords) or not callable(
        caked_angles_to_detector_display_coords
    ):
        return None

    try:
        caked_background_arr = np.asarray(caked_background, dtype=float)
        radial_axis_arr = np.asarray(radial_axis, dtype=float)
        azimuth_axis_arr = np.asarray(azimuth_axis, dtype=float)
    except Exception:
        return None
    if (
        caked_background_arr.ndim != 2
        or caked_background_arr.size == 0
        or radial_axis_arr.size == 0
        or azimuth_axis_arr.size == 0
    ):
        return None

    raw_caked = detector_display_to_caked_coords(float(raw_col), float(raw_row))
    if not (
        isinstance(raw_caked, tuple)
        and len(raw_caked) >= 2
        and np.isfinite(float(raw_caked[0]))
        and np.isfinite(float(raw_caked[1]))
    ):
        return None

    caked_candidate = None
    if isinstance(candidate, dict):
        try:
            sim_caked_col = float(candidate.get("caked_x", candidate.get("two_theta_deg", np.nan)))
            sim_caked_row = float(candidate.get("caked_y", candidate.get("phi_deg", np.nan)))
        except Exception:
            sim_caked_col = float("nan")
            sim_caked_row = float("nan")
        if np.isfinite(sim_caked_col) and np.isfinite(sim_caked_row):
            caked_candidate = dict(candidate)
            caked_candidate["sim_col"] = float(sim_caked_col)
            caked_candidate["sim_row"] = float(sim_caked_row)
            caked_candidate["sim_col_global"] = float(sim_caked_col)
            caked_candidate["sim_row_global"] = float(sim_caked_row)
            caked_candidate["sim_col_local"] = float(
                caked_axis_to_image_index_fn(float(sim_caked_col), radial_axis_arr)
            )
            caked_candidate["sim_row_local"] = float(
                caked_axis_to_image_index_fn(float(sim_caked_row), azimuth_axis_arr)
            )

    refined_caked_col, refined_caked_row = geometry_manual_refine_preview_point(
        caked_candidate,
        float(raw_caked[0]),
        float(raw_caked[1]),
        display_background=caked_background_arr,
        cache_data=cache_data,
        use_caked_space=True,
        radial_axis=radial_axis_arr,
        azimuth_axis=azimuth_axis_arr,
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
        caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
        refine_caked_peak_center_fn=refine_caked_peak_center_fn,
    )
    if not (np.isfinite(refined_caked_col) and np.isfinite(refined_caked_row)):
        return None

    refined_display = caked_angles_to_detector_display_coords(
        float(refined_caked_col),
        float(refined_caked_row),
    )
    if not (
        isinstance(refined_display, tuple)
        and len(refined_display) >= 2
        and refined_display[0] is not None
        and refined_display[1] is not None
        and np.isfinite(float(refined_display[0]))
        and np.isfinite(float(refined_display[1]))
    ):
        return None

    return {
        "raw_caked_col": float(raw_caked[0]),
        "raw_caked_row": float(raw_caked[1]),
        "refined_caked_col": float(refined_caked_col),
        "refined_caked_row": float(refined_caked_row),
        "refined_display_col": float(refined_display[0]),
        "refined_display_row": float(refined_display[1]),
    }


def resolve_background_pick_to_caked_angles(
    candidate: dict[str, object] | None,
    seed_col: float,
    seed_row: float,
    *,
    active_view: str,
    display_background: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
    refine_detector_pick_fn: Callable[..., tuple[float, float]] | None,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    refine_caked_peak_center_fn: Callable[
        [np.ndarray | None, Sequence[float] | None, Sequence[float] | None, float, float],
        tuple[float, float],
    ] = refine_caked_peak_center,
) -> dict[str, float] | None:
    """Resolve one detector/caked background seed to caked truth plus detector cache."""

    if not callable(background_display_to_native_detector_coords):
        return None
    try:
        seed_col_val = float(seed_col)
        seed_row_val = float(seed_row)
    except Exception:
        return None
    if not (np.isfinite(seed_col_val) and np.isfinite(seed_row_val)):
        return None

    view_token = str(active_view).strip().lower()
    if view_token not in {"detector", "caked"}:
        return None

    def _finite_tuple_pair(value: object) -> tuple[float, float] | None:
        if isinstance(value, (str, bytes)):
            return None
        try:
            if len(value) < 2:  # type: ignore[arg-type]
                return None
            col_val = float(value[0])
            row_val = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col_val) and np.isfinite(row_val)):
            return None
        return float(col_val), float(row_val)

    try:
        image_shape = tuple(int(v) for v in np.asarray(display_background).shape[:2])
    except Exception:
        image_shape = ()

    result: dict[str, float] = {}
    raw_caked_angles: tuple[float, float] | None = None
    if view_token == "caked":
        if not (
            callable(caked_angles_to_background_display_coords)
            and callable(refine_caked_peak_center_fn)
        ):
            return None
        raw_caked_angles = caked_display_coords_to_angles(
            seed_col_val,
            seed_row_val,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            image_shape=image_shape,
            caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
            caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
        )
        if raw_caked_angles is None:
            return None
        try:
            refined_angles = _finite_tuple_pair(
                refine_caked_peak_center_fn(
                    np.asarray(display_background, dtype=float),
                    radial_axis,
                    azimuth_axis,
                    float(raw_caked_angles[0]),
                    float(raw_caked_angles[1]),
                )
            )
        except Exception:
            refined_angles = None
        if refined_angles is None:
            refined_angles = (float(raw_caked_angles[0]), float(raw_caked_angles[1]))
        detector_anchor_display = _finite_tuple_pair(
            caked_angles_to_background_display_coords(
                float(refined_angles[0]),
                float(refined_angles[1]),
            )
        )
        if detector_anchor_display is None:
            return None
        detector_anchor_native = _finite_tuple_pair(
            background_display_to_native_detector_coords(
                float(detector_anchor_display[0]),
                float(detector_anchor_display[1]),
            )
        )
        if detector_anchor_native is None:
            return None
        result["raw_caked_display_col"] = float(seed_col_val)
        result["raw_caked_display_row"] = float(seed_row_val)
        result["raw_caked_two_theta_deg"] = float(raw_caked_angles[0])
        result["raw_caked_phi_deg"] = float(raw_caked_angles[1])
        result["refined_detector_display_col"] = float(detector_anchor_display[0])
        result["refined_detector_display_row"] = float(detector_anchor_display[1])
        result["refined_detector_native_col"] = float(detector_anchor_native[0])
        result["refined_detector_native_row"] = float(detector_anchor_native[1])
        result["background_detector_x"] = float(detector_anchor_native[0])
        result["background_detector_y"] = float(detector_anchor_native[1])
        result["refined_background_two_theta_deg"] = float(refined_angles[0])
        result["refined_background_phi_deg"] = float(refined_angles[1])
        return result
    else:
        if not (
            callable(refine_detector_pick_fn)
            and callable(native_detector_coords_to_caked_display_coords)
        ):
            return None
        detector_seed = (float(seed_col_val), float(seed_row_val))
        seed_native = _finite_tuple_pair(
            background_display_to_native_detector_coords(
                float(detector_seed[0]),
                float(detector_seed[1]),
            )
        )
        if seed_native is not None:
            raw_caked_angles = _finite_tuple_pair(
                native_detector_coords_to_caked_display_coords(
                    float(seed_native[0]),
                    float(seed_native[1]),
                )
            )

    result["detector_seed_col"] = float(detector_seed[0])
    result["detector_seed_row"] = float(detector_seed[1])
    if raw_caked_angles is not None:
        result["raw_caked_two_theta_deg"] = float(raw_caked_angles[0])
        result["raw_caked_phi_deg"] = float(raw_caked_angles[1])

    try:
        refined_detector = refine_detector_pick_fn(
            dict(candidate) if isinstance(candidate, dict) else None,
            float(detector_seed[0]),
            float(detector_seed[1]),
            display_background=display_background,
            cache_data=cache_data,
            force_detector_space=True,
        )
    except TypeError:
        refined_detector = refine_detector_pick_fn(
            dict(candidate) if isinstance(candidate, dict) else None,
            float(detector_seed[0]),
            float(detector_seed[1]),
            display_background=display_background,
            cache_data=cache_data,
        )
    refined_detector_point = _finite_tuple_pair(refined_detector)
    if refined_detector_point is None:
        return None

    refined_native = _finite_tuple_pair(
        background_display_to_native_detector_coords(
            float(refined_detector_point[0]),
            float(refined_detector_point[1]),
        )
    )
    if refined_native is None:
        return None

    refined_angles = _finite_tuple_pair(
        native_detector_coords_to_caked_display_coords(
            float(refined_native[0]),
            float(refined_native[1]),
        )
    )
    if refined_angles is None:
        return None

    result["refined_detector_display_col"] = float(refined_detector_point[0])
    result["refined_detector_display_row"] = float(refined_detector_point[1])
    result["refined_detector_native_col"] = float(refined_native[0])
    result["refined_detector_native_row"] = float(refined_native[1])
    result["background_detector_x"] = float(refined_native[0])
    result["background_detector_y"] = float(refined_native[1])
    result["refined_background_two_theta_deg"] = float(refined_angles[0])
    result["refined_background_phi_deg"] = float(refined_angles[1])
    return result


def restore_geometry_manual_pick_view(
    pick_session: dict[str, object] | None,
    *,
    axis: Any,
    canvas: Any = None,
    redraw: bool = True,
) -> bool:
    """Restore the pre-zoom axis view for manual background placement."""

    if not isinstance(pick_session, dict):
        return False
    if not bool(pick_session.get("zoom_active", False)):
        return False
    saved_xlim = pick_session.get("saved_xlim")
    saved_ylim = pick_session.get("saved_ylim")
    if isinstance(saved_xlim, tuple) and len(saved_xlim) == 2:
        axis.set_xlim(float(saved_xlim[0]), float(saved_xlim[1]))
    if isinstance(saved_ylim, tuple) and len(saved_ylim) == 2:
        axis.set_ylim(float(saved_ylim[0]), float(saved_ylim[1]))
    pick_session["zoom_active"] = False
    pick_session["zoom_center"] = None
    pick_session["saved_xlim"] = None
    pick_session["saved_ylim"] = None
    if redraw and canvas is not None:
        canvas.draw_idle()
    return True


def apply_geometry_manual_pick_zoom(
    pick_session: dict[str, object] | None,
    col: float,
    row: float,
    *,
    display_background: np.ndarray | None,
    axis: Any,
    canvas: Any = None,
    use_caked_space: bool,
    last_caked_extent: Sequence[float] | None = None,
    caked_zoom_tth_deg: float,
    caked_zoom_phi_deg: float,
    pick_zoom_window_px: float,
    anchor_fraction_x: float = 0.5,
    anchor_fraction_y: float = 0.5,
    anchor_axis_limits_fn: Callable[
        [float, float, float, float, float], tuple[float, float]
    ] = geometry_manual_anchor_axis_limits,
) -> bool:
    """Zoom to a fixed local window while the user is placing manual points."""

    if not geometry_manual_pick_session_active(pick_session, require_current_background=False):
        return False
    if pick_session is None or display_background is None:
        return False

    try:
        current_xlim = tuple(float(v) for v in axis.get_xlim())
    except Exception:
        current_xlim = (0.0, 1.0)
    try:
        current_ylim = tuple(float(v) for v in axis.get_ylim())
    except Exception:
        current_ylim = (0.0, 1.0)
    x_sign = 1.0 if len(current_xlim) < 2 or current_xlim[1] >= current_xlim[0] else -1.0
    y_sign = 1.0 if len(current_ylim) < 2 or current_ylim[1] >= current_ylim[0] else -1.0

    if use_caked_space:
        if last_caked_extent is not None and len(last_caked_extent) >= 4:
            x_lo = float(last_caked_extent[0])
            x_hi = float(last_caked_extent[1])
            y_lo = float(last_caked_extent[2])
            y_hi = float(last_caked_extent[3])
        else:
            x_lo, x_hi = sorted(current_xlim)
            y_lo, y_hi = sorted(current_ylim)
        x_span = x_sign * min(abs(float(caked_zoom_tth_deg)), abs(float(x_hi) - float(x_lo)))
        y_span = y_sign * min(abs(float(caked_zoom_phi_deg)), abs(float(y_hi) - float(y_lo)))
        x_min, x_max = anchor_axis_limits_fn(
            float(col),
            float(x_span),
            float(anchor_fraction_x),
            float(x_lo),
            float(x_hi),
        )
        y_min, y_max = anchor_axis_limits_fn(
            float(row),
            float(y_span),
            float(anchor_fraction_y),
            float(y_lo),
            float(y_hi),
        )
        if x_min == x_max or y_min == y_max:
            return False
        if not bool(pick_session.get("zoom_active", False)):
            pick_session["saved_xlim"] = tuple(float(v) for v in axis.get_xlim())
            pick_session["saved_ylim"] = tuple(float(v) for v in axis.get_ylim())
        pick_session["zoom_active"] = True
        pick_session["zoom_center"] = (float(col), float(row))
        axis.set_xlim(float(x_min), float(x_max))
        axis.set_ylim(float(y_min), float(y_max))
        if canvas is not None:
            canvas.draw_idle()
        return True

    background_shape = np.asarray(display_background).shape
    height = max(1.0, float(background_shape[0]))
    width = max(1.0, float(background_shape[1]))
    x_span = x_sign * min(float(pick_zoom_window_px), width)
    y_span = y_sign * min(float(pick_zoom_window_px), height)
    x_min, x_max = anchor_axis_limits_fn(
        float(col),
        float(x_span),
        float(anchor_fraction_x),
        0.0,
        float(width),
    )
    y_min, y_max = anchor_axis_limits_fn(
        float(row),
        float(y_span),
        float(anchor_fraction_y),
        0.0,
        float(height),
    )
    if not bool(pick_session.get("zoom_active", False)):
        pick_session["saved_xlim"] = tuple(float(v) for v in axis.get_xlim())
        pick_session["saved_ylim"] = tuple(float(v) for v in axis.get_ylim())
    pick_session["zoom_active"] = True
    pick_session["zoom_center"] = (float(col), float(row))
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    if canvas is not None:
        canvas.draw_idle()
    return True


def geometry_manual_pick_preview_state(
    raw_col: float,
    raw_row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    force: bool = False,
    remaining_candidates: Sequence[dict[str, object]] | None,
    display_background: np.ndarray | None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    refine_preview_point: Callable[..., tuple[float, float]],
    preview_due: Callable[[float, float], bool] | None = None,
    nearest_candidate_to_point: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    position_error_px: Callable[
        [float, float, float, float], float
    ] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    use_caked_space: bool,
    caked_angles_to_background_display_coords: Callable[
        [float, float], tuple[float | None, float | None]
    ]
    | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    profile_cache: Mapping[str, object] | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
) -> dict[str, object] | None:
    """Return preview state for one manual placement cursor position."""

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return None
    if pick_session is None:
        return None
    if not force and callable(preview_due) and not preview_due(float(raw_col), float(raw_row)):
        return None
    if display_background is None:
        return None

    state = cache_data if isinstance(cache_data, dict) else None
    if state is None and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = None
        if isinstance(built_state, dict):
            state = built_state

    selected_group_key = pick_session.get("group_key") if isinstance(pick_session, dict) else None
    source_candidates = _geometry_manual_filter_candidates_for_selected_group_source(
        remaining_candidates,
        group_key=selected_group_key,
    )
    if remaining_candidates and not source_candidates:
        selected_source = _geometry_manual_entry_source_label(
            None,
            group_key=selected_group_key,
        )
        available_sources = _geometry_manual_available_candidate_sources(remaining_candidates)
        message = (
            "Manual pick preview source mismatch: "
            "source_mismatch_error=true "
            f"selected_group_source={selected_source} "
            f"available_sources={available_sources}"
        )
        return {
            "manual_geometry_run_id": _geometry_manual_session_run_id(pick_session),
            "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
            "raw_col": float(raw_col),
            "raw_row": float(raw_row),
            "candidate": None,
            "sim_dist": float("nan"),
            "sim_dist_source": "source_mismatch",
            "preview_distance_source": "source_mismatch",
            "message": message,
            "source_mismatch_error": True,
            "selected_group_source": selected_source,
            "available_sources": available_sources,
        }
    active_remaining_candidates = (
        source_candidates if remaining_candidates else remaining_candidates
    )

    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        active_remaining_candidates,
    )
    seed_candidate: dict[str, object] | None = None
    if isinstance(tagged_candidate, dict):
        seed_candidate = dict(tagged_candidate)
        candidate_relation = "tagged sim"
    else:
        if nearest_candidate_to_point is geometry_manual_nearest_candidate_to_point:
            seed_candidate, _seed_dist = nearest_candidate_to_point(
                float(raw_col),
                float(raw_row),
                active_remaining_candidates,
                use_caked_display=use_caked_space,
            )
        else:
            seed_candidate, _seed_dist = nearest_candidate_to_point(
                float(raw_col),
                float(raw_row),
                active_remaining_candidates,
            )
        candidate_relation = "nearest sim"
    candidate = dict(seed_candidate) if isinstance(seed_candidate, dict) else None
    if isinstance(seed_candidate, dict):
        selected_candidate = _geometry_manual_select_q_group_representative(
            active_remaining_candidates,
            group_key=selected_group_key,
            seed_candidate=seed_candidate,
            profile_cache=profile_cache,
        )
        if isinstance(selected_candidate, dict):
            candidate = selected_candidate
    refined_col, refined_row = refine_preview_point(
        candidate,
        float(raw_col),
        float(raw_row),
        display_background=display_background,
        cache_data=state,
    )
    distance_details = geometry_manual_candidate_distance_details(
        float(refined_col),
        float(refined_row),
        candidate,
        use_caked_display=use_caked_space,
    )
    status_trace = geometry_manual_status_distance_trace(
        distance_details,
        observed_refined_caked_deg=(
            (float(refined_col), float(refined_row)) if use_caked_space else None
        ),
        use_caked_space=bool(use_caked_space),
    )
    manual_run_id = _geometry_manual_session_run_id(pick_session)
    candidate_branch = _geometry_manual_trace_branch(candidate)
    sim_dist = float(distance_details.get("distance", float("nan")))
    sim_dist_source = str(distance_details.get("distance_source", "<unavailable>"))
    cache_current_dist = float(distance_details.get("cache_current_distance", float("nan")))
    cache_current_delta = distance_details.get("cache_current_delta")
    delta = float(
        position_error_px(
            float(raw_col),
            float(raw_row),
            float(refined_col),
            float(refined_row),
        )
    )
    if use_caked_space and callable(caked_angles_to_background_display_coords):
        raw_display = caked_angles_to_background_display_coords(float(raw_col), float(raw_row))
        refined_display = caked_angles_to_background_display_coords(
            float(refined_col),
            float(refined_row),
        )
        if (
            raw_display[0] is not None
            and raw_display[1] is not None
            and refined_display[0] is not None
            and refined_display[1] is not None
        ):
            delta = float(
                position_error_px(
                    float(raw_display[0]),
                    float(raw_display[1]),
                    float(refined_display[0]),
                    float(refined_display[1]),
                )
            )
    sigma_px = position_sigma_px(delta)
    match_confidence = geometry_manual_preview_match_confidence(
        candidate,
        float(refined_col),
        float(refined_row),
        cache_data=state,
        use_caked_space=use_caked_space,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
    )
    if np.isfinite(match_confidence):
        preview_color = geometry_manual_preview_confidence_color(float(match_confidence))
        quality_label = geometry_manual_preview_confidence_quality_label(float(match_confidence))
    else:
        preview_color = geometry_manual_preview_color(float(sigma_px))
        quality_label = geometry_manual_preview_quality_label(float(sigma_px))
    candidate_label = str(candidate.get("label", "")) if isinstance(candidate, dict) else ""
    q_label = str(
        pick_session.get(
            "q_label",
            pick_session.get("group_key", "selected Qr/Qz set"),
        )
    )
    message = (
        f"Manual pick preview for {q_label}: "
        f"release=({float(raw_col):.1f},{float(raw_row):.1f}) -> "
        f"refined=({float(refined_col):.1f},{float(refined_row):.1f}), "
        f"placement error={delta:.2f}px, sigma={float(sigma_px):.2f}px, "
        + (
            f"confidence={float(match_confidence):.2f}, quality={quality_label}"
            if np.isfinite(match_confidence)
            else f"quality={quality_label}"
        )
    )
    if candidate_label:
        message += f" -> {candidate_relation} [{candidate_label}]"
        if np.isfinite(sim_dist):
            message += f" ({float(sim_dist):.2f}{' deg' if use_caked_space else 'px'})"
    message += (
        " "
        + geometry_manual_cmd_provenance_text(
            run_id=manual_run_id,
            emitter="geometry_manual_pick_preview_state",
            event="manual_pick_preview",
            branch=candidate_branch if candidate_branch is not None else "<none>",
            actual_source=sim_dist_source,
            expected_source=_geometry_manual_expected_distance_source(use_caked_space),
            **_geometry_manual_source_consistency_fields(
                selected_group_key=selected_group_key,
                candidate=candidate,
            ),
        )
        + " "
        + geometry_manual_format_status_distance_trace(status_trace)
    )
    preview_result = {
        "manual_geometry_run_id": manual_run_id,
        "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
        "raw_col": float(raw_col),
        "raw_row": float(raw_row),
        "refined_col": float(refined_col),
        "refined_row": float(refined_row),
        "candidate": candidate,
        "sim_dist": float(sim_dist),
        "sim_dist_source": sim_dist_source,
        "preview_distance_source": sim_dist_source,
        **status_trace,
        "preview_distance_to_cache_current_sim": float(cache_current_dist),
        "geometry_minus_sim_cache_current_delta_deg": cache_current_delta,
        "delta": float(delta),
        "sigma_px": float(sigma_px),
        "match_confidence": float(match_confidence),
        "preview_color": str(preview_color),
        "quality_label": str(quality_label),
        "message": message,
    }
    if use_caked_space and geometry_manual_live_caked_trace_enabled():
        geometry_manual_trace_live_caked_visual_source_event(
            "manual_pick_preview",
            manual_geometry_run_id=manual_run_id,
            selected_click_caked_deg=(float(raw_col), float(raw_row)),
            selected_candidate=candidate,
            pending_entries=pick_session.get("group_entries", []),
            preview_state=preview_result,
            saved_entries=pick_session.get("pending_entries", []),
        )
    return preview_result


def geometry_manual_session_initial_pairs_display(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    require_current_background: bool = True,
    use_caked_display: bool = False,
    refresh_entry_geometry: Callable[
        [Mapping[str, object] | None],
        dict[str, object] | None,
    ]
    | None = None,
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
) -> list[dict[str, object]]:
    """Return overlay-ready display entries for the in-progress manual pick session.

    Group entries are already the selected simulated branch candidates. Their
    detector/caked display coordinates must stay in the same coordinate frame as
    the visible simulation points. Background-entry refresh is only a fallback for
    missing simulated display coordinates because it uses the background-display
    rotation.
    """

    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=require_current_background,
    ):
        return []
    if pick_session is None:
        return []

    group_entries = pick_session.get("group_entries", [])
    pending_entries = pick_session.get("pending_entries", [])
    if not isinstance(group_entries, list) or not isinstance(pending_entries, list):
        return []

    pending_lookup: GeometryManualLookupMap = {}
    for raw_entry in pending_entries:
        if not isinstance(raw_entry, dict):
            continue
        source_key = candidate_source_key(raw_entry)
        if source_key is not None:
            _geometry_manual_add_lookup_entry(
                pending_lookup,
                source_key,
                raw_entry,
            )

    def _current_view_sim_entry(raw_entry: dict[str, object]) -> dict[str, object]:
        source_entry = dict(raw_entry)
        if bool(use_caked_display) and bool(source_entry.get("_caked_qr_projection_cache")):
            return source_entry
        if callable(project_peaks_to_current_view):
            projection_source = (
                _geometry_manual_caked_qr_projection_source(source_entry)
                if bool(use_caked_display)
                else source_entry
            )
            if bool(use_caked_display) and projection_source is None:
                return {"_caked_qr_projection_unresolved": True}
            try:
                projected_entries = project_peaks_to_current_view([dict(projection_source)])
            except Exception:
                projected_entries = []
            projected_entry = next(
                (
                    dict(candidate)
                    for candidate in (projected_entries or ())
                    if isinstance(candidate, Mapping)
                ),
                None,
            )
            if isinstance(projected_entry, dict):
                if bool(use_caked_display):
                    projected_entry["_caked_qr_projection_cache"] = True
                return projected_entry
        if bool(use_caked_display):
            return {"_caked_qr_projection_unresolved": True}
        return source_entry

    def _selected_sim_display_point(
        display_entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        if bool(use_caked_display):
            if not isinstance(display_entry, Mapping) or bool(
                display_entry.get("_caked_qr_projection_unresolved")
            ):
                return None
            if _geometry_manual_entry_has_stale_caked_fields(display_entry):
                return None
            live_caked_point = _geometry_manual_finite_point(
                display_entry,
                (
                    ("caked_x", "caked_y"),
                    ("raw_caked_x", "raw_caked_y"),
                    ("two_theta_deg", "phi_deg"),
                ),
            )
            if live_caked_point is not None:
                return live_caked_point
            if (
                isinstance(display_entry, Mapping)
                and _geometry_manual_normalized_frame_token(display_entry.get("display_frame"))
                == "caked_display"
            ):
                live_display_point = _geometry_manual_finite_point(
                    display_entry,
                    (("display_col", "display_row"),),
                )
                if live_display_point is not None:
                    return live_display_point
            return _geometry_manual_entry_caked_point(display_entry)
        return _geometry_manual_entry_current_view_point(display_entry)

    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, raw_entry in enumerate(group_entries):
        if not isinstance(raw_entry, dict):
            continue
        display_entry = _current_view_sim_entry(raw_entry)
        entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": raw_entry.get("hkl", raw_entry.get("label")),
        }
        raw_group_key = raw_entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            entry["q_group_key"] = tuple(raw_group_key)
        sim_display = _selected_sim_display_point(display_entry)
        caked_projection_unresolved = bool(
            use_caked_display
            and isinstance(display_entry, Mapping)
            and display_entry.get("_caked_qr_projection_unresolved")
        )
        if (
            sim_display is None
            and not caked_projection_unresolved
            and callable(refresh_entry_geometry)
        ):
            try:
                refreshed_entry = refresh_entry_geometry(raw_entry)
            except Exception:
                refreshed_entry = None
            if isinstance(refreshed_entry, Mapping):
                sim_display = _selected_sim_display_point(refreshed_entry)
        if sim_display is not None:
            entry["sim_display"] = (float(sim_display[0]), float(sim_display[1]))
        elif bool(use_caked_display):
            entry["sim_display_unresolved"] = True

        source_key = candidate_source_key(raw_entry)
        pending_candidates = (
            _geometry_manual_lookup_bucket_entries(pending_lookup.get(source_key))
            if source_key is not None
            else []
        )
        measured_entry = None
        pending_index = _geometry_manual_resolve_identity_candidate_index(
            raw_entry,
            pending_candidates,
        )
        if pending_index is not None and 0 <= int(pending_index) < len(pending_candidates):
            measured_entry = dict(pending_candidates[int(pending_index)])
        elif pending_candidates:
            resolved_index = geometry_manual_resolve_source_entry_index(
                raw_entry,
                pending_candidates,
            )
            if resolved_index is not None and 0 <= int(resolved_index) < len(pending_candidates):
                measured_entry = dict(pending_candidates[int(resolved_index)])
        _copy_q_values_from_sources(entry, display_entry, measured_entry)
        if isinstance(measured_entry, dict):
            bg_coords = _geometry_manual_saved_background_display_point_for_view(
                measured_entry,
                use_caked_display=bool(use_caked_display),
                entry_display_coords=entry_display_coords,
            )
            if bg_coords is not None:
                entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        initial_pairs_display.append(entry)
    return initial_pairs_display


def geometry_manual_background_reference_initial_display_entry(
    entry: Mapping[str, object] | None,
    *,
    use_caked_display: bool,
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
) -> dict[str, object]:
    """Build a redraw entry for one saved background-only QR reference."""

    saved = dict(entry) if isinstance(entry, Mapping) else {}
    initial_entry: dict[str, object] = {
        "background_qr_set_reference": True,
        "manual_background_reference": True,
    }
    label = saved.get("label")
    if label is not None:
        initial_entry["label"] = str(label)
    for key in (
        "background_two_theta_deg",
        "background_phi_deg",
        "refined_background_two_theta_deg",
        "refined_background_phi_deg",
        "raw_caked_x",
        "raw_caked_y",
        "raw_caked_two_theta_deg",
        "raw_caked_phi_deg",
        "x",
        "y",
        "raw_x",
        "raw_y",
    ):
        if key in saved:
            initial_entry[key] = saved[key]
    bg_coords = _geometry_manual_saved_background_display_point_for_view(
        saved,
        use_caked_display=bool(use_caked_display),
        entry_display_coords=entry_display_coords,
    )
    if bg_coords is None:
        bg_coords = _geometry_manual_finite_point(
            saved,
            (
                ("x", "y"),
                ("raw_x", "raw_y"),
                ("background_two_theta_deg", "background_phi_deg"),
                ("raw_caked_x", "raw_caked_y"),
            ),
        )
    if bg_coords is not None:
        initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
    else:
        initial_entry["background_reference_display_unresolved"] = True
    return initial_entry


def build_geometry_manual_initial_pairs_display(
    background_index: int,
    *,
    param_set: dict[str, object] | None = None,
    current_background_index: object = None,
    prefer_cache: bool = False,
    reuse_only: bool = False,
    use_caked_display: bool | None = None,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    current_geometry_fit_params: Callable[[], dict[str, object]] | None = None,
    get_cache_data: Callable[..., dict[str, object]],
    source_rows_for_background: Callable[
        [int, dict[str, object] | None],
        Sequence[dict[str, object]],
    ]
    | None = None,
    listed_q_group_keys_for_picker: Callable[[], object] | object = (),
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]] | None = None,
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ],
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
    filter_active_rows: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build overlay-ready manual geometry pairs for one background image."""

    saved_entries = [dict(entry) for entry in pairs_for_index(int(background_index))]
    if not saved_entries:
        return [], []

    if isinstance(param_set, dict):
        params_local = dict(param_set)
    elif callable(current_geometry_fit_params):
        params_local = dict(current_geometry_fit_params())
    else:
        params_local = {}

    def _background_reference_display_payload(
        entry: Mapping[str, object],
        pair_idx: int,
    ) -> dict[str, object]:
        initial_entry = geometry_manual_background_reference_initial_display_entry(
            entry,
            use_caked_display=bool(use_caked_display),
            entry_display_coords=entry_display_coords,
        )
        initial_entry["overlay_match_index"] = int(pair_idx)
        return initial_entry

    if all(geometry_manual_entry_is_background_qr_reference(entry) for entry in saved_entries):
        measured_display: list[dict[str, object]] = []
        initial_pairs_display: list[dict[str, object]] = []
        for pair_idx, entry in enumerate(saved_entries):
            measured_entry = dict(entry)
            measured_entry["overlay_match_index"] = int(pair_idx)
            measured_display.append(measured_entry)
            initial_pairs_display.append(_background_reference_display_payload(entry, pair_idx))
        return measured_display, initial_pairs_display

    def _initial_overlay_sim_display_point(
        saved_entry: Mapping[str, object] | None,
        resolved_sim_entry: Mapping[str, object] | None,
        projected_sim_entry: Mapping[str, object] | None,
        *,
        prefer_caked_detector_replay: bool = False,
    ) -> tuple[float, float] | None:
        def _saved_caked_sim_point(
            candidate: Mapping[str, object] | None,
        ) -> tuple[float, float] | None:
            return _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_caked_x", "refined_sim_caked_y"),
                    ("sim_caked_x", "sim_caked_y"),
                    ("sim_two_theta_deg", "sim_phi_deg"),
                    ("simulated_two_theta_deg", "simulated_phi_deg"),
                ),
            )

        def _live_caked_point(
            candidate: Mapping[str, object] | None,
        ) -> tuple[float, float] | None:
            caked_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("caked_x", "caked_y"),
                    ("two_theta_deg", "phi_deg"),
                ),
            )
            if caked_point is not None:
                return caked_point
            display_frame = (
                str(
                    candidate.get("display_frame", candidate.get("current_view_frame", ""))
                    if isinstance(candidate, Mapping)
                    else ""
                )
                .strip()
                .lower()
            )
            if display_frame == "caked_display":
                caked_point = _geometry_manual_finite_point(
                    candidate,
                    (("display_col", "display_row"),),
                )
            if caked_point is not None:
                return caked_point
            return _geometry_manual_finite_point(
                candidate,
                (("refined_sim_caked_x", "refined_sim_caked_y"),),
            )

        def _normalize_sim_identity_value(value: object) -> object:
            if isinstance(value, list):
                return tuple(value)
            return value

        def _candidate_matches_saved_sim_source(
            candidate: Mapping[str, object] | None,
        ) -> bool:
            if not isinstance(saved_entry, Mapping) or not isinstance(candidate, Mapping):
                return False

            saw_strong_match = False
            for field in (
                "source_table_index",
                "source_row_index",
                "source_reflection_index",
                "source_branch_index",
                "source_ray_id",
                "branch_id",
                "q_group_key",
            ):
                saved_value = saved_entry.get(field)
                candidate_value = candidate.get(field)
                if not (
                    _geometry_manual_identity_value_present(saved_value)
                    and _geometry_manual_identity_value_present(candidate_value)
                ):
                    continue
                if _normalize_sim_identity_value(saved_value) != _normalize_sim_identity_value(
                    candidate_value,
                ):
                    return False
                if field != "q_group_key":
                    saw_strong_match = True

            if saw_strong_match:
                return True

            saved_key = geometry_manual_candidate_source_key(dict(saved_entry))
            candidate_key = geometry_manual_candidate_source_key(dict(candidate))
            return bool(saved_key is not None and saved_key == candidate_key)

        def _source_matched_live_caked_point(
            candidate: Mapping[str, object] | None,
        ) -> tuple[float, float] | None:
            if not _candidate_matches_saved_sim_source(candidate):
                return None
            return _live_caked_point(candidate)

        def _live_detector_display_point(
            candidate: Mapping[str, object] | None,
        ) -> tuple[float, float] | None:
            detector_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_x", "refined_sim_y"),
                    ("sim_col_raw", "sim_row_raw"),
                ),
            )
            if detector_point is not None:
                return detector_point

            legacy_sim_point = _geometry_manual_finite_point(
                candidate,
                (("sim_col", "sim_row"),),
            )
            caked_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_caked_x", "refined_sim_caked_y"),
                    ("caked_x", "caked_y"),
                    ("raw_caked_x", "raw_caked_y"),
                    ("two_theta_deg", "phi_deg"),
                ),
            )
            if legacy_sim_point is not None and not _geometry_manual_points_match(
                legacy_sim_point,
                caked_point,
            ):
                return legacy_sim_point

            detector_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("x", "y"),
                    ("simulated_x", "simulated_y"),
                ),
            )
            if detector_point is not None and not _geometry_manual_points_match(
                detector_point,
                caked_point,
            ):
                return detector_point

            if not _geometry_manual_entry_has_caked_evidence(candidate):
                return _geometry_manual_finite_point(
                    candidate,
                    (("display_col", "display_row"),),
                )
            return None

        def _saved_native_detector_point(
            candidate: Mapping[str, object] | None,
        ) -> tuple[float, float] | None:
            return _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_native_x", "refined_sim_native_y"),
                    ("native_col", "native_row"),
                    ("sim_native_x", "sim_native_y"),
                    ("detector_x", "detector_y"),
                    ("background_detector_x", "background_detector_y"),
                    ("simulated_detector_x", "simulated_detector_y"),
                ),
            )

        def _saved_native_overlay_detector_point(
            candidate: Mapping[str, object] | None,
            *,
            allow_detector_aliases: bool = True,
        ) -> tuple[float, float] | None:
            detector_point = _geometry_manual_finite_point(
                candidate,
                (("refined_sim_x", "refined_sim_y"),),
            )
            if detector_point is not None:
                return detector_point
            if allow_detector_aliases:
                return _live_detector_display_point(candidate)
            if not isinstance(candidate, Mapping):
                return None
            overlay_candidate = dict(candidate)
            overlay_candidate.pop("x", None)
            overlay_candidate.pop("y", None)
            overlay_candidate.pop("simulated_x", None)
            overlay_candidate.pop("simulated_y", None)
            return _live_detector_display_point(overlay_candidate)

        def _has_explicit_detector_display_alias(
            candidate: Mapping[str, object] | None,
        ) -> bool:
            if not isinstance(candidate, Mapping):
                return False
            detector_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_x", "refined_sim_y"),
                    ("sim_col_raw", "sim_row_raw"),
                ),
            )
            if detector_point is not None:
                return True
            caked_point = _geometry_manual_finite_point(
                candidate,
                (
                    ("refined_sim_caked_x", "refined_sim_caked_y"),
                    ("caked_x", "caked_y"),
                    ("raw_caked_x", "raw_caked_y"),
                    ("two_theta_deg", "phi_deg"),
                ),
            )
            for x_key, y_key in (
                ("sim_col", "sim_row"),
                ("x", "y"),
                ("simulated_x", "simulated_y"),
            ):
                detector_point = _geometry_manual_finite_point(
                    candidate,
                    ((x_key, y_key),),
                )
                if detector_point is not None and not _geometry_manual_points_match(
                    detector_point,
                    caked_point,
                ):
                    return True
            return False

        saved_refined_detector_point = _geometry_manual_finite_point(
            saved_entry,
            (("refined_sim_x", "refined_sim_y"),),
        )
        saved_native_detector_point = _geometry_manual_finite_point(
            saved_entry,
            (("refined_sim_native_x", "refined_sim_native_y"),),
        )

        if bool(use_caked_display):
            projected_caked_point = _source_matched_live_caked_point(projected_sim_entry)
            if projected_caked_point is not None:
                return projected_caked_point
            resolved_caked_point = _source_matched_live_caked_point(resolved_sim_entry)
            if resolved_caked_point is not None:
                return resolved_caked_point
            saved_caked_point = _saved_caked_sim_point(saved_entry)
            if saved_caked_point is not None:
                return saved_caked_point
            return None

        saved_overlay_detector_point = _saved_native_overlay_detector_point(
            saved_entry,
            allow_detector_aliases=False,
        )

        if prefer_caked_detector_replay:
            projected_detector_point = _live_detector_display_point(projected_sim_entry)
            if projected_detector_point is not None:
                return projected_detector_point
            resolved_detector_point = _live_detector_display_point(resolved_sim_entry)
            if resolved_detector_point is not None:
                return resolved_detector_point
            return None

        if saved_native_detector_point is not None:
            projected_saved_native_point = _saved_native_overlay_detector_point(projected_sim_entry)
            resolved_saved_native_point = _saved_native_overlay_detector_point(resolved_sim_entry)
            if projected_saved_native_point is not None and not _geometry_manual_points_match(
                projected_saved_native_point,
                resolved_saved_native_point,
            ):
                return projected_saved_native_point
            if (
                saved_refined_detector_point is None
                and resolved_saved_native_point is not None
                and (
                    _has_explicit_detector_display_alias(projected_sim_entry)
                    or _has_explicit_detector_display_alias(resolved_sim_entry)
                )
            ):
                return saved_native_detector_point
            if projected_saved_native_point is not None:
                return projected_saved_native_point
            if resolved_saved_native_point is not None:
                return resolved_saved_native_point
            return None

        if saved_refined_detector_point is not None:
            projected_detector_point = _live_detector_display_point(projected_sim_entry)
            if projected_detector_point is not None:
                return projected_detector_point

        projected_detector_point = _live_detector_display_point(projected_sim_entry)
        if projected_detector_point is not None:
            return projected_detector_point

        resolved_detector_point = _live_detector_display_point(resolved_sim_entry)
        if resolved_detector_point is not None:
            return resolved_detector_point

        if (
            not _geometry_manual_entry_has_explicit_branch_identity(saved_entry)
            and _geometry_manual_entry_normalized_hkl(saved_entry) is None
            and not _geometry_manual_entry_label(saved_entry)
            and (
                _geometry_manual_entry_has_caked_evidence(saved_entry)
                or _geometry_manual_entry_has_stale_caked_fields(saved_entry)
            )
            and (
                _geometry_manual_entry_has_explicit_branch_identity(resolved_sim_entry)
                or _geometry_manual_entry_has_explicit_branch_identity(projected_sim_entry)
            )
        ):
            return None

        return saved_overlay_detector_point

    raw_simulated_lookup: GeometryManualLookupMap = {}
    raw_caked_qr_projection_lookup: GeometryManualLookupMap = {}
    caked_qr_projection_lookup: GeometryManualLookupMap = {}
    rebuilt_caked_qr_projection_lookup: GeometryManualLookupMap | None = None

    def _lookup_entry_was_stale_unrefined(
        raw_lookup_entry: Mapping[str, object] | None,
        resolved_entry: Mapping[str, object] | None,
    ) -> bool:
        if not isinstance(raw_lookup_entry, Mapping) or not isinstance(
            resolved_entry,
            Mapping,
        ):
            return False
        raw_status = str(raw_lookup_entry.get("sim_refinement_status", "") or "").strip().lower()
        resolved_status = str(resolved_entry.get("sim_refinement_status", "") or "").strip().lower()
        return raw_status != "refined" and resolved_status == "refined"

    def _call_source_rows_for_projection_rebuild() -> list[dict[str, object]]:
        if not callable(source_rows_for_background):
            return []
        try:
            raw_rows = source_rows_for_background(
                int(background_index),
                params_local,
                consumer="initial_pairs_caked_projection",
            )
        except TypeError:
            try:
                raw_rows = source_rows_for_background(int(background_index), params_local)
            except Exception:
                raw_rows = []
        except Exception:
            raw_rows = []
        return [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]

    def _rebuild_caked_qr_projection_lookup() -> GeometryManualLookupMap:
        if bool(reuse_only):
            return {}
        source_rows = _call_source_rows_for_projection_rebuild()
        if not source_rows:
            source_rows = geometry_manual_simulated_peaks_from_callback(
                simulated_peaks_for_params,
                param_set=params_local,
                prefer_cache=bool(
                    prefer_cache and int(background_index) == int(current_background_index)
                ),
            )
        _entries, _grouped, lookup = _geometry_manual_build_caked_qr_projection_cache(
            source_rows,
            project_peaks_to_current_view,
            lambda _rows: {},
            build_simulated_lookup,
            filter_active_rows,
        )
        return _geometry_manual_copy_lookup(lookup)

    def _caked_qr_projection_entry_for_saved(
        saved_entry: Mapping[str, object],
    ) -> dict[str, object] | None:
        nonlocal rebuilt_caked_qr_projection_lookup
        projected = _geometry_manual_lookup_caked_qr_projection_entry(
            caked_qr_projection_lookup,
            saved_entry,
        )
        if projected is not None:
            return projected
        if bool(reuse_only):
            return None
        if rebuilt_caked_qr_projection_lookup is None:
            rebuilt_caked_qr_projection_lookup = _rebuild_caked_qr_projection_lookup()
        return _geometry_manual_lookup_caked_qr_projection_entry(
            rebuilt_caked_qr_projection_lookup,
            saved_entry,
        )

    def _copy_sim_identity_fields(
        target: dict[str, object],
        *sources: Mapping[str, object] | None,
    ) -> None:
        for field in (*_GEOMETRY_MANUAL_CAKED_QR_ID_FIELDS, "q_group_key"):
            if field in target:
                continue
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                value = source.get(field)
                if _geometry_manual_identity_value_present(value):
                    target[field] = value
                    break

    if prefer_cache and int(background_index) == int(current_background_index):
        cache_data = get_cache_data(
            param_set=params_local,
            prefer_cache=True,
            reuse_only=bool(reuse_only),
            background_index=background_index,
        )
        if not isinstance(cache_data, Mapping):
            cache_data = {}
        raw_simulated_lookup = _geometry_manual_copy_lookup(cache_data.get("simulated_lookup", {}))
        raw_caked_qr_projection_lookup = _geometry_manual_copy_lookup(
            cache_data.get("caked_qr_projection_lookup", {})
        )
        if not bool(reuse_only):
            cache_data = geometry_manual_rebuild_refined_qr_cache_lookups(
                cache_data,
                build_simulated_lookup,
            )
        simulated_lookup = _geometry_manual_copy_lookup(cache_data.get("simulated_lookup", {}))
        caked_qr_projection_lookup = _geometry_manual_copy_lookup(
            cache_data.get("caked_qr_projection_lookup", {})
        )
    else:
        simulated_lookup: GeometryManualLookupMap = {}
    if not simulated_lookup:
        if bool(reuse_only):
            source_rows = []
        else:
            source_rows = (
                [
                    dict(entry)
                    for entry in source_rows_for_background(
                        int(background_index),
                        params_local,
                        consumer="initial_pairs_display",
                    )
                    if isinstance(entry, Mapping)
                ]
                if callable(source_rows_for_background)
                else []
            )
            if not source_rows:
                source_rows = geometry_manual_simulated_peaks_from_callback(
                    simulated_peaks_for_params,
                    param_set=params_local,
                    prefer_cache=bool(
                        prefer_cache and int(background_index) == int(current_background_index)
                    ),
                )
        if source_rows:
            normalized_source_rows = [
                dict(entry) for entry in source_rows if isinstance(entry, Mapping)
            ]
            if callable(project_peaks_to_current_view):
                try:
                    projected_source_rows = project_peaks_to_current_view(normalized_source_rows)
                except Exception:
                    projected_source_rows = normalized_source_rows
                normalized_source_rows = [
                    dict(entry)
                    for entry in (projected_source_rows or ())
                    if isinstance(entry, Mapping)
                ]
            if callable(filter_active_rows):
                try:
                    active_source_rows = filter_active_rows(normalized_source_rows)
                except Exception:
                    active_source_rows = []
            else:
                active_source_rows = normalized_source_rows
            simulated_lookup = build_simulated_lookup(active_source_rows)

    projected_detector_source_lookup: GeometryManualLookupMap | None = None
    needs_projected_detector_source_lookup = (
        not bool(use_caked_display)
        and not bool(reuse_only)
        and callable(source_rows_for_background)
        and callable(project_peaks_to_current_view)
        and any(
            _geometry_manual_finite_point(
                entry,
                (
                    ("refined_sim_native_x", "refined_sim_native_y"),
                    ("native_col", "native_row"),
                    ("sim_native_x", "sim_native_y"),
                    ("detector_x", "detector_y"),
                    ("background_detector_x", "background_detector_y"),
                    ("simulated_detector_x", "simulated_detector_y"),
                ),
            )
            is not None
            for entry in saved_entries
            if not geometry_manual_entry_is_background_qr_reference(entry)
        )
    )

    def _projected_detector_source_lookup() -> GeometryManualLookupMap:
        nonlocal projected_detector_source_lookup
        if projected_detector_source_lookup is not None:
            return projected_detector_source_lookup
        source_rows: list[dict[str, object]] = []
        if callable(source_rows_for_background):
            try:
                raw_rows = source_rows_for_background(
                    int(background_index),
                    params_local,
                    consumer="initial_pairs_detector_projection",
                )
            except TypeError:
                try:
                    raw_rows = source_rows_for_background(int(background_index), params_local)
                except Exception:
                    raw_rows = []
            except Exception:
                raw_rows = []
            source_rows = [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]
        if source_rows and callable(project_peaks_to_current_view):
            try:
                projected_rows = project_peaks_to_current_view(source_rows)
            except Exception:
                projected_rows = source_rows
            source_rows = [
                dict(entry) for entry in (projected_rows or ()) if isinstance(entry, Mapping)
            ]
        if source_rows and callable(filter_active_rows):
            try:
                source_rows = filter_active_rows(source_rows)
            except Exception:
                source_rows = []
        projected_detector_source_lookup = (
            build_simulated_lookup(source_rows) if source_rows else {}
        )
        return projected_detector_source_lookup

    flattened_lookup_entries = _geometry_manual_flatten_lookup_entries(simulated_lookup)
    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, entry in enumerate(saved_entries):
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        measured_display.append(measured_entry)

        if geometry_manual_entry_is_background_qr_reference(entry):
            initial_pairs_display.append(_background_reference_display_payload(entry, pair_idx))
            continue

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": entry.get("hkl", entry.get("label")),
        }
        raw_group_key = entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            initial_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            initial_entry["q_group_key"] = tuple(raw_group_key)
        bg_coords = _geometry_manual_saved_background_display_point_for_view(
            entry,
            use_caked_display=bool(use_caked_display),
            entry_display_coords=entry_display_coords,
        )
        if bg_coords is not None:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        raw_caked_projection_entry = _geometry_manual_lookup_caked_qr_projection_entry(
            raw_caked_qr_projection_lookup,
            entry,
        )
        caked_projection_entry = (
            _caked_qr_projection_entry_for_saved(entry)
            if bool(use_caked_display)
            else _geometry_manual_lookup_caked_qr_projection_entry(
                caked_qr_projection_lookup,
                entry,
            )
        )
        raw_sim_source_entry: dict[str, object] | None = None
        if bool(use_caked_display):
            sim_source_entry = (
                dict(caked_projection_entry)
                if isinstance(caked_projection_entry, Mapping)
                else geometry_manual_lookup_source_entry(simulated_lookup, entry)
            )
        else:
            raw_sim_source_entry = geometry_manual_lookup_source_entry(
                raw_simulated_lookup,
                entry,
            )
            projected_detector_source_entry = (
                geometry_manual_lookup_source_entry(
                    _projected_detector_source_lookup(),
                    entry,
                )
                if needs_projected_detector_source_lookup
                else None
            )
            sim_source_entry = (
                projected_detector_source_entry
                if isinstance(projected_detector_source_entry, Mapping)
                else geometry_manual_lookup_source_entry(simulated_lookup, entry)
            )
        detector_replay_from_caked_projection = bool(
            not bool(use_caked_display)
            and isinstance(caked_projection_entry, Mapping)
            and _geometry_manual_caked_qr_projection_source(entry) is not None
        )
        if detector_replay_from_caked_projection:
            if isinstance(projected_detector_source_entry, Mapping):
                sim_source_entry = dict(projected_detector_source_entry)
            elif isinstance(raw_sim_source_entry, Mapping):
                sim_source_entry = dict(raw_sim_source_entry)
            elif isinstance(caked_projection_entry, Mapping):
                sim_source_entry = dict(caked_projection_entry)
            else:
                sim_source_entry = _geometry_manual_caked_qr_projection_source(entry)
        if (
            not bool(use_caked_display)
            and not detector_replay_from_caked_projection
            and bg_coords is not None
            and not _geometry_manual_entry_has_explicit_branch_identity(entry)
            and _geometry_manual_entry_normalized_hkl(entry) is None
            and not _geometry_manual_entry_label(entry)
            and (
                _geometry_manual_entry_has_caked_evidence(entry)
                or _geometry_manual_entry_has_stale_caked_fields(entry)
            )
        ):
            source_row_key = _geometry_manual_source_row_key(entry)
            if source_row_key is not None:
                detector_candidates = [
                    dict(candidate)
                    for candidate in flattened_lookup_entries
                    if _geometry_manual_source_row_key(candidate) == source_row_key
                    and _geometry_manual_entry_has_explicit_branch_identity(candidate)
                ]
                ranked_detector_candidates: list[tuple[float, int, dict[str, object]]] = []
                for candidate_index, candidate in enumerate(detector_candidates):
                    detector_point = _geometry_manual_entry_detector_display_point(candidate)
                    if detector_point is None:
                        continue
                    ranked_detector_candidates.append(
                        (
                            float(
                                np.hypot(
                                    float(detector_point[0]) - float(bg_coords[0]),
                                    float(detector_point[1]) - float(bg_coords[1]),
                                )
                            ),
                            int(candidate_index),
                            dict(candidate),
                        )
                    )
                ranked_detector_candidates.sort(key=lambda item: (float(item[0]), int(item[1])))
                if ranked_detector_candidates and (
                    len(ranked_detector_candidates) == 1
                    or abs(
                        float(ranked_detector_candidates[1][0])
                        - float(ranked_detector_candidates[0][0])
                    )
                    > 1.0e-9
                ):
                    sim_source_entry = dict(ranked_detector_candidates[0][2])
        overlay_sim_source_entry = (
            dict(sim_source_entry) if isinstance(sim_source_entry, Mapping) else None
        )
        sim_entry = dict(sim_source_entry) if isinstance(sim_source_entry, Mapping) else None
        caked_projection_resolved = bool(
            use_caked_display and isinstance(caked_projection_entry, Mapping)
        )
        caked_sim_source_available = bool(
            use_caked_display and isinstance(sim_source_entry, Mapping)
        )
        if _lookup_entry_was_stale_unrefined(
            raw_caked_projection_entry,
            caked_projection_entry,
        ):
            initial_entry["caked_qr_projection_lookup_stale_unrefined_rebuilt"] = True
        if not bool(use_caked_display) and _lookup_entry_was_stale_unrefined(
            raw_sim_source_entry,
            sim_source_entry,
        ):
            initial_entry["simulated_lookup_stale_unrefined_rebuilt"] = True
        if bool(use_caked_display) and not caked_sim_source_available:
            initial_entry["sim_display_unresolved"] = True
            _copy_q_values_from_sources(initial_entry, entry)
            initial_pairs_display.append(initial_entry)
            continue
        if not caked_projection_resolved and not detector_replay_from_caked_projection:
            sim_entry = geometry_manual_apply_refined_simulated_override(
                entry,
                sim_entry,
                prefer_caked_display=use_caked_display,
            )
        if detector_replay_from_caked_projection and not callable(project_peaks_to_current_view):
            initial_entry["sim_display_unresolved"] = True
            _copy_q_values_from_sources(initial_entry, entry)
            initial_pairs_display.append(initial_entry)
            continue
        _copy_q_values_from_sources(initial_entry, entry, sim_source_entry, sim_entry)
        if (
            isinstance(sim_entry, dict)
            and callable(project_peaks_to_current_view)
            and not caked_projection_resolved
        ):
            try:
                projected_sim_entries = project_peaks_to_current_view([sim_entry])
            except Exception:
                projected_sim_entries = [sim_entry]
            projected_sim_entry = next(
                (
                    dict(candidate)
                    for candidate in (projected_sim_entries or ())
                    if isinstance(candidate, Mapping)
                ),
                None,
            )
            if isinstance(projected_sim_entry, dict):
                sim_entry = projected_sim_entry
        if detector_replay_from_caked_projection and not isinstance(sim_entry, dict):
            initial_entry["sim_display_unresolved"] = True
            _copy_q_values_from_sources(initial_entry, entry, sim_source_entry)
            initial_pairs_display.append(initial_entry)
            continue
        if isinstance(sim_entry, dict):
            sim_display = _initial_overlay_sim_display_point(
                entry,
                overlay_sim_source_entry,
                sim_entry,
                prefer_caked_detector_replay=detector_replay_from_caked_projection,
            )
            if sim_display is not None:
                initial_entry["sim_display"] = (
                    float(sim_display[0]),
                    float(sim_display[1]),
                )
            _copy_q_values_from_sources(initial_entry, sim_entry, entry)
            _copy_sim_identity_fields(initial_entry, sim_entry, sim_source_entry, entry)
        initial_pairs_display.append(initial_entry)

    return measured_display, initial_pairs_display


def make_runtime_geometry_manual_cache_callbacks(
    *,
    fit_config: Mapping[str, object] | None,
    last_simulation_signature: Callable[[], object] | object,
    current_background_index: Callable[[], object] | object,
    current_background_image: Callable[[], object | None] | object | None,
    use_caked_space: Callable[[], object] | object,
    replace_cache_state: Callable[[object, dict[str, object]], None],
    current_geometry_fit_params: Callable[[], dict[str, object]] | None,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    source_rows_for_background: Callable[
        [int, dict[str, object] | None],
        Sequence[dict[str, object]],
    ]
    | None = None,
    listed_q_group_keys_for_picker: Callable[[], object] | object = (),
    simulated_peaks_for_params: Callable[..., Sequence[dict[str, object]]] | None = None,
    build_grouped_candidates: Callable[
        [Sequence[dict[str, object]] | None],
        dict[tuple[object, ...], list[dict[str, object]]],
    ],
    build_simulated_lookup: Callable[
        [Sequence[dict[str, object]] | None],
        GeometryManualLookupMap,
    ],
    project_peaks_to_current_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    project_peaks_to_caked_view: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    project_peaks_for_background_view: Callable[
        [int, Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    entry_display_coords: Callable[
        [dict[str, object] | None],
        tuple[float, float] | None,
    ],
    disabled_qr_sets: Callable[[], object] | object = (),
    disabled_qz_sections: Callable[[], object] | object = (),
    filter_active_rows: Callable[
        [Sequence[dict[str, object]] | None],
        list[dict[str, object]],
    ]
    | None = None,
    stored_max_positions_local: Callable[[], object] | object = (),
    stored_peak_table_lattice: Callable[[], object] | object = (),
    peak_records: Callable[[], object] | object = (),
    current_cache_signature: Callable[[], object] | object = None,
    current_cache_data: Callable[[], dict[str, object] | None] | dict[str, object] | None = None,
    caked_projection_signature: Callable[[], object] | object = None,
    source_snapshot_signature_for_background: (
        Callable[[int, dict[str, object] | None], object] | None
    ) = None,
    source_snapshot_diagnostics: Callable[[], Mapping[str, object] | None] | None = None,
    detector_simulation_image: Callable[[], object] | object | None = None,
    caked_simulation_image: Callable[[], object] | object | None = None,
    radial_axis: Callable[[], object] | object | None = None,
    azimuth_axis: Callable[[], object] | object | None = None,
    detector_display_to_native_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    caked_angles_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    auto_match_background_context: Callable[
        [object, dict[str, object]],
        tuple[dict[str, object], object],
    ]
    | None = None,
) -> GeometryManualRuntimeCacheCallbacks:
    """Build live manual-geometry cache/display callbacks around shared helpers."""

    def _background_index() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _background_image() -> object | None:
        return _resolve_runtime_value(current_background_image)

    def _manual_pick_uses_caked_space() -> bool:
        return bool(_resolve_runtime_value(use_caked_space))

    default_project_peaks_to_caked_view = project_peaks_to_caked_view

    def _current_match_config() -> dict[str, object]:
        return current_geometry_manual_match_config(fit_config)

    def _filter_active_rows(
        candidate_rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        normalized_rows = [
            dict(entry) for entry in (candidate_rows or ()) if isinstance(entry, Mapping)
        ]
        if not callable(filter_active_rows):
            return normalized_rows
        try:
            filtered_rows = filter_active_rows(normalized_rows)
        except Exception:
            filtered_rows = []
        return [dict(entry) for entry in (filtered_rows or ()) if isinstance(entry, Mapping)]

    def _placed_pick_cache_signature(
        *,
        param_set: dict[str, object] | None = None,
        background_index: int | None = None,
        background_image: object | None = None,
    ) -> tuple[object, ...]:
        resolved_background_index = (
            _background_index() if background_index is None else int(background_index)
        )
        if callable(source_snapshot_signature_for_background):
            source_snapshot_signature = source_snapshot_signature_for_background(
                int(resolved_background_index),
                param_set,
            )
        else:
            source_snapshot_signature = _resolve_runtime_value(last_simulation_signature)
        return geometry_manual_pick_placed_cache_signature(
            source_snapshot_signature=source_snapshot_signature,
            background_index=resolved_background_index,
            background_image=(
                _background_image() if background_image is None else background_image
            ),
            use_caked_space=_manual_pick_uses_caked_space(),
            caked_projection_signature=_resolve_runtime_value(caked_projection_signature),
        )

    def _pick_cache_signature(
        *,
        param_set: dict[str, object] | None = None,
        background_index: int | None = None,
        background_image: object | None = None,
    ) -> tuple[object, ...]:
        return geometry_manual_pick_cache_signature(
            placed_cache_signature=_placed_pick_cache_signature(
                param_set=param_set,
                background_index=background_index,
                background_image=background_image,
            ),
            disabled_qr_sets=_resolve_runtime_value(disabled_qr_sets),
            disabled_qz_sections=_resolve_runtime_value(disabled_qz_sections),
        )

    def _attach_source_snapshot_diagnostics(
        cache_data: dict[str, object],
    ) -> dict[str, object]:
        if not callable(source_snapshot_diagnostics) or not isinstance(cache_data, dict):
            return cache_data
        try:
            diagnostics = source_snapshot_diagnostics()
        except Exception:
            diagnostics = None
        if not isinstance(diagnostics, Mapping):
            return cache_data
        diagnostic_payload = dict(diagnostics)
        metadata = (
            dict(cache_data.get("cache_metadata"))
            if isinstance(cache_data.get("cache_metadata"), Mapping)
            else {}
        )
        if metadata.get(
            "cache_source"
        ) != "geometry_manual_source_rows_for_background" and not bool(
            metadata.get("source_rows_provider_attempted", False)
        ):
            return cache_data
        status = diagnostic_payload.get("status")
        metadata.update(
            {
                "source_snapshot_cache_family": diagnostic_payload.get("cache_family"),
                "source_snapshot_action": diagnostic_payload.get("action"),
                "source_snapshot_status": status,
                "source_snapshot_consumer": diagnostic_payload.get("consumer"),
                "source_snapshot_created_from": diagnostic_payload.get("created_from"),
                "source_snapshot_signature_match": diagnostic_payload.get("signature_match"),
                "source_snapshot_row_count": diagnostic_payload.get(
                    "final_returned_row_count",
                    diagnostic_payload.get(
                        "projected_peak_count",
                        diagnostic_payload.get("raw_peak_count", 0),
                    ),
                ),
                "source_snapshot_raw_peak_count": diagnostic_payload.get("raw_peak_count"),
                "source_snapshot_projected_peak_count": diagnostic_payload.get(
                    "projected_peak_count"
                ),
                "source_snapshot_requested_signature_summary": diagnostic_payload.get(
                    "requested_signature_summary"
                ),
                "source_snapshot_stored_signature_summary": diagnostic_payload.get(
                    "stored_signature_summary"
                ),
                "source_snapshot_rebuild_attempted": diagnostic_payload.get("rebuild_attempted"),
                "source_snapshot_rebuild_returned_row_count": diagnostic_payload.get(
                    "rebuild_returned_row_count"
                ),
            }
        )
        cache_data["cache_metadata"] = metadata
        cache_data["source_snapshot_diagnostics"] = diagnostic_payload
        if isinstance(cache_data.get("detector_picker_trace"), Mapping):
            trace = dict(cache_data["detector_picker_trace"])
            trace.update(
                {
                    "manual_pick_cache_source": metadata.get("cache_source"),
                    "manual_pick_cache_stale_reason": metadata.get("stale_reason"),
                    "source_snapshot_status": status,
                    "source_snapshot_rebuild_attempted": metadata.get(
                        "source_snapshot_rebuild_attempted"
                    ),
                    "source_snapshot_rebuild_returned_row_count": metadata.get(
                        "source_snapshot_rebuild_returned_row_count"
                    ),
                }
            )
            cache_data["detector_picker_trace"] = trace
        return cache_data

    def _get_pick_cache(
        *,
        param_set: dict[str, object] | None = None,
        prefer_cache: bool = True,
        reuse_only: bool = False,
        background_index: int | None = None,
        background_image: object | None = None,
        build_caked_projection_sidecar: bool = False,
        picker_candidates_only: bool = False,
        caked_projection_signature_override: object = None,
        project_peaks_to_caked_view: Callable[
            [Sequence[dict[str, object]] | None],
            list[dict[str, object]],
        ]
        | None = None,
    ) -> dict[str, object]:
        get_start = perf_counter()
        bg_index = _background_index() if background_index is None else int(background_index)
        current_bg_index = _background_index()
        background_local = _background_image() if background_image is None else background_image
        existing_cache_signature_value = _resolve_runtime_value(current_cache_signature)
        existing_cache_data_value = _resolve_runtime_value(current_cache_data)
        peak_records_value = _resolve_runtime_value(peak_records)
        last_simulation_signature_value = _resolve_runtime_value(last_simulation_signature)
        caked_projection_signature_value = (
            caked_projection_signature_override
            if caked_projection_signature_override is not None
            else _resolve_runtime_value(caked_projection_signature)
        )
        detector_simulation_image_value = _resolve_runtime_value(detector_simulation_image)
        caked_simulation_image_value = _resolve_runtime_value(caked_simulation_image)
        radial_axis_value = _resolve_runtime_value(radial_axis)
        azimuth_axis_value = _resolve_runtime_value(azimuth_axis)

        def _request_caked_projection_signature() -> object:
            return caked_projection_signature_value

        def _placed_pick_cache_signature_for_request(
            *,
            param_set: dict[str, object] | None = None,
            background_index: int | None = None,
            background_image: object | None = None,
        ) -> tuple[object, ...]:
            resolved_background_index = (
                _background_index() if background_index is None else int(background_index)
            )
            if callable(source_snapshot_signature_for_background):
                source_snapshot_signature = source_snapshot_signature_for_background(
                    int(resolved_background_index),
                    param_set,
                )
            else:
                source_snapshot_signature = last_simulation_signature_value
            return geometry_manual_pick_placed_cache_signature(
                source_snapshot_signature=source_snapshot_signature,
                background_index=resolved_background_index,
                background_image=(
                    _background_image() if background_image is None else background_image
                ),
                use_caked_space=_manual_pick_uses_caked_space(),
                caked_projection_signature=_request_caked_projection_signature(),
                include_caked_projection_signature=bool(build_caked_projection_sidecar),
            )

        def _pick_cache_signature_for_request(
            *,
            param_set: dict[str, object] | None = None,
            background_index: int | None = None,
            background_image: object | None = None,
        ) -> tuple[object, ...]:
            return geometry_manual_pick_cache_signature(
                placed_cache_signature=_placed_pick_cache_signature_for_request(
                    param_set=param_set,
                    background_index=background_index,
                    background_image=background_image,
                ),
                disabled_qr_sets=_resolve_runtime_value(disabled_qr_sets),
                disabled_qz_sections=_resolve_runtime_value(disabled_qz_sections),
            )

        def _reuse_only_miss(stale_reason: str) -> dict[str, object]:
            miss = geometry_manual_no_build_cache_sentinel(
                match_config=_current_match_config(),
                stale_reason=stale_reason,
            )
            miss["cache_metadata"] = _geometry_manual_cache_metadata_update(
                miss,
                cache_action="miss",
                stale_reason=stale_reason,
                reuse_only=True,
                get_pick_cache_total_wall_ms=_geometry_manual_elapsed_ms(get_start),
                get_pick_cache_build_wall_ms=0.0,
                qr_sim_refinement_wall_ms=0.0,
                qr_lookup_rebuild_wall_ms=0.0,
                qr_refinement_skipped=True,
                qr_lookup_rebuild_skipped=True,
            )
            return miss

        if bool(reuse_only):
            requested_signature = _pick_cache_signature_for_request(
                param_set=param_set,
                background_index=bg_index,
                background_image=background_local,
            )
            existing_signature = existing_cache_signature_value
            if (
                not isinstance(existing_signature, tuple)
                and isinstance(existing_cache_data_value, Mapping)
                and isinstance(existing_cache_data_value.get("signature"), tuple)
            ):
                existing_signature = existing_cache_data_value.get("signature")
            if (
                bg_index != current_bg_index
                or not isinstance(existing_cache_data_value, Mapping)
            ):
                return _reuse_only_miss("manual pick cache not warm for click")
            existing_cache = dict(existing_cache_data_value)
            if bool(build_caked_projection_sidecar):
                signature_ready = bool(
                    existing_signature == requested_signature
                    or (
                        _manual_pick_cache_source_and_masks_reusable(
                            existing_signature,
                            requested_signature,
                        )
                        and _manual_pick_cache_caked_sidecar_signature_compatible(
                            existing_cache.get("placed_signature", existing_signature),
                            requested_signature,
                        )
                    )
                )
                grouped_ready = _manual_pick_cache_has_caked_projection_sidecar(existing_cache)
            else:
                grouped_ready = geometry_manual_pick_cache_has_detector_picker_sources(
                    existing_cache,
                    display_background=background_local,
                    profile_cache=(
                        param_set.get("mosaic_params")
                        if isinstance(param_set, Mapping)
                        and isinstance(param_set.get("mosaic_params"), Mapping)
                        else None
                    ),
                )
                relaxed_signature_ready = bool(
                    grouped_ready
                    and (
                        _manual_pick_cache_source_and_masks_reusable(
                            existing_signature,
                            requested_signature,
                        )
                        or _manual_pick_cache_source_and_masks_reusable(
                            existing_cache.get("placed_signature", existing_signature),
                            requested_signature,
                        )
                    )
                )
                signature_ready = bool(
                    existing_signature == requested_signature or relaxed_signature_ready
                )
            if not signature_ready:
                return _reuse_only_miss("manual pick cache not warm for click")
            if not grouped_ready:
                return _reuse_only_miss("manual pick cache not warm for click")
            sidecar_cache_reused_for_detector = bool(
                not build_caked_projection_sidecar
                and existing_signature != requested_signature
                and _manual_pick_cache_caked_projection_token(existing_signature) is not None
            )
            refinement_signature = geometry_manual_qr_sim_refinement_signature(
                existing_cache,
                simulation_signature=last_simulation_signature_value,
                caked_projection_signature=caked_projection_signature_value,
                detector_simulation_image=detector_simulation_image_value,
                caked_simulation_image=caked_simulation_image_value,
                radial_axis=radial_axis_value,
                azimuth_axis=azimuth_axis_value,
                detector_display_to_native_coords=detector_display_to_native_coords,
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
            )
            refinement_ready = bool(
                existing_cache.get("qr_sim_refinement_signature") == refinement_signature
                and existing_cache.get("qr_sim_refinement_complete") is True
                and existing_cache.get("qr_sim_refinement_lookup_signature") == refinement_signature
                and existing_cache.get("qr_sim_refinement_lookup_complete") is True
            )
            if not refinement_ready:
                if not bool(build_caked_projection_sidecar):
                    existing_cache["cache_ready"] = False
                    existing_cache["picker_candidates_ready"] = True
                    existing_cache["cache_metadata"] = _geometry_manual_cache_metadata_update(
                        existing_cache,
                        cache_ready=False,
                        picker_candidates_ready=True,
                        refinement_lookup_ready=False,
                        cache_signature_compatible_for_picker=True,
                        sidecar_cache_reused_for_detector=sidecar_cache_reused_for_detector,
                        cache_action="reused",
                        reuse_only=True,
                        stale_reason=(
                            "manual pick detector candidates ready; refinement lookup not warm"
                        ),
                        get_pick_cache_total_wall_ms=_geometry_manual_elapsed_ms(get_start),
                        get_pick_cache_build_wall_ms=0.0,
                        qr_sim_refinement_wall_ms=0.0,
                        qr_lookup_rebuild_wall_ms=0.0,
                        qr_refinement_skipped=True,
                        qr_lookup_rebuild_skipped=True,
                    )
                    return _attach_source_snapshot_diagnostics(existing_cache)
                return _reuse_only_miss("manual pick cache not warm for click")
            existing_cache["cache_ready"] = True
            existing_cache["cache_metadata"] = _geometry_manual_cache_metadata_update(
                existing_cache,
                cache_ready=True,
                picker_candidates_ready=True,
                refinement_lookup_ready=True,
                cache_signature_compatible_for_picker=True,
                sidecar_cache_reused_for_detector=sidecar_cache_reused_for_detector,
                cache_action="reused",
                reuse_only=True,
                stale_reason=None,
                get_pick_cache_total_wall_ms=_geometry_manual_elapsed_ms(get_start),
                get_pick_cache_build_wall_ms=0.0,
                qr_sim_refinement_wall_ms=0.0,
                qr_lookup_rebuild_wall_ms=0.0,
                qr_refinement_skipped=True,
                qr_lookup_rebuild_skipped=True,
            )
            return existing_cache

        build_start = perf_counter()
        cache_data, next_signature, next_cache_data = build_geometry_manual_pick_cache(
            param_set=param_set,
            prefer_cache=prefer_cache,
            background_index=bg_index,
            current_background_index=current_bg_index,
            background_image=background_local,
            existing_cache_signature=existing_cache_signature_value,
            existing_cache_data=existing_cache_data_value,
            placed_cache_signature_fn=_placed_pick_cache_signature_for_request,
            cache_signature_fn=_pick_cache_signature_for_request,
            source_rows_for_background=source_rows_for_background,
            listed_q_group_keys_for_picker=listed_q_group_keys_for_picker,
            simulated_peaks_for_params=simulated_peaks_for_params,
            peak_records=peak_records_value,
            build_grouped_candidates=build_grouped_candidates,
            build_simulated_lookup=build_simulated_lookup,
            filter_active_rows=_filter_active_rows,
            project_peaks_to_current_view=project_peaks_to_current_view,
            project_peaks_to_caked_view=(
                project_peaks_to_caked_view or default_project_peaks_to_caked_view
            ),
            project_peaks_for_background_view=project_peaks_for_background_view,
            current_match_config=_current_match_config,
            auto_match_background_context=auto_match_background_context,
            build_caked_projection_sidecar=(
                False if bool(picker_candidates_only) else bool(build_caked_projection_sidecar)
            ),
            picker_candidates_only=bool(picker_candidates_only),
        )
        get_build_wall_ms = _geometry_manual_elapsed_ms(build_start)
        same_next_cache = isinstance(next_cache_data, dict) and next_cache_data is cache_data

        def _mark_picker_candidates_only(raw_cache: Mapping[str, object] | None) -> dict[str, object]:
            if not isinstance(raw_cache, Mapping):
                return {}
            picker_cache = dict(raw_cache)
            picker_ready = geometry_manual_pick_cache_has_detector_picker_sources(
                picker_cache,
                display_background=background_local,
                profile_cache=(
                    param_set.get("mosaic_params")
                    if isinstance(param_set, Mapping)
                    and isinstance(param_set.get("mosaic_params"), Mapping)
                    else None
                ),
            )
            picker_cache["cache_ready"] = False
            picker_cache["picker_candidates_ready"] = bool(picker_ready)
            picker_cache["cache_metadata"] = _geometry_manual_cache_metadata_update(
                picker_cache,
                cache_ready=False,
                picker_candidates_ready=bool(picker_ready),
                bounded_picker_build_attempted=True,
                bounded_picker_build_returned_row_count=int(
                    len(picker_cache.get("detector_picker_rows", ()) or ())
                ),
                cache_action=picker_cache.get("cache_metadata", {}).get("cache_action")
                if isinstance(picker_cache.get("cache_metadata"), Mapping)
                else "rebuilt",
                reuse_only=False,
                picker_candidates_only=True,
                stale_reason=(
                    None
                    if picker_ready
                    else "manual pick detector picker-only recovery found no candidates"
                ),
                get_pick_cache_total_wall_ms=_geometry_manual_elapsed_ms(get_start),
                get_pick_cache_build_wall_ms=get_build_wall_ms,
                qr_sim_refinement_wall_ms=0.0,
                qr_lookup_rebuild_wall_ms=0.0,
                qr_refinement_skipped=True,
                qr_lookup_rebuild_skipped=True,
            )
            return _attach_source_snapshot_diagnostics(picker_cache)

        if bool(picker_candidates_only):
            cache_data = _mark_picker_candidates_only(cache_data)
            if isinstance(next_cache_data, dict):
                next_cache_data = cache_data if same_next_cache else _mark_picker_candidates_only(
                    next_cache_data
                )
            replace_cache_state(
                next_signature,
                dict(next_cache_data) if isinstance(next_cache_data, dict) else {},
            )
            return cache_data

        def _refine_cache_once(raw_cache: Mapping[str, object] | None) -> dict[str, object]:
            if not isinstance(raw_cache, Mapping):
                return {}
            refinement_signature = geometry_manual_qr_sim_refinement_signature(
                raw_cache,
                simulation_signature=last_simulation_signature_value,
                caked_projection_signature=caked_projection_signature_value,
                detector_simulation_image=detector_simulation_image_value,
                caked_simulation_image=caked_simulation_image_value,
                radial_axis=radial_axis_value,
                azimuth_axis=azimuth_axis_value,
                detector_display_to_native_coords=detector_display_to_native_coords,
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
            )
            refinement_skipped = bool(
                raw_cache.get("qr_sim_refinement_signature") == refinement_signature
                and raw_cache.get("qr_sim_refinement_complete") is True
            )
            refine_start = perf_counter()
            refined = geometry_manual_refine_qr_sim_candidates_in_cache(
                raw_cache,
                detector_simulation_image=detector_simulation_image_value,
                caked_simulation_image=caked_simulation_image_value,
                radial_axis=radial_axis_value,
                azimuth_axis=azimuth_axis_value,
                detector_display_to_native_coords=detector_display_to_native_coords,
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
                refinement_signature=refinement_signature,
            )
            refinement_wall_ms = _geometry_manual_elapsed_ms(refine_start)
            lookup_skipped = bool(
                isinstance(refined, Mapping)
                and refined.get("qr_sim_refinement_lookup_signature") == refinement_signature
                and refined.get("qr_sim_refinement_lookup_complete") is True
            )
            lookup_start = perf_counter()
            refined = geometry_manual_rebuild_refined_qr_cache_lookups(
                refined,
                build_simulated_lookup,
                refinement_signature=refinement_signature,
            )
            lookup_wall_ms = _geometry_manual_elapsed_ms(lookup_start)
            refined["cache_metadata"] = _geometry_manual_cache_metadata_update(
                refined,
                get_pick_cache_build_wall_ms=get_build_wall_ms,
                qr_sim_refinement_wall_ms=refinement_wall_ms,
                qr_lookup_rebuild_wall_ms=lookup_wall_ms,
                qr_refinement_skipped=bool(refinement_skipped),
                qr_lookup_rebuild_skipped=bool(lookup_skipped),
            )
            return _attach_source_snapshot_diagnostics(refined)

        cache_data = _refine_cache_once(cache_data)
        if isinstance(next_cache_data, dict):
            if same_next_cache:
                next_cache_data = cache_data
            else:
                next_cache_data = _refine_cache_once(next_cache_data)
        replace_cache_state(
            next_signature,
            dict(next_cache_data) if isinstance(next_cache_data, dict) else {},
        )
        cache_data["cache_metadata"] = _geometry_manual_cache_metadata_update(
            cache_data,
            get_pick_cache_total_wall_ms=_geometry_manual_elapsed_ms(get_start),
            get_pick_cache_build_wall_ms=get_build_wall_ms,
        )
        return cache_data

    def _build_initial_pairs_display(
        background_index: int,
        *,
        param_set: dict[str, object] | None = None,
        prefer_cache: bool = False,
        reuse_only: bool = False,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return build_geometry_manual_initial_pairs_display(
            background_index,
            param_set=param_set,
            current_background_index=_background_index(),
            prefer_cache=prefer_cache,
            reuse_only=bool(reuse_only),
            use_caked_display=_manual_pick_uses_caked_space(),
            pairs_for_index=pairs_for_index,
            current_geometry_fit_params=current_geometry_fit_params,
            get_cache_data=_get_pick_cache,
            source_rows_for_background=source_rows_for_background,
            simulated_peaks_for_params=simulated_peaks_for_params,
            build_simulated_lookup=build_simulated_lookup,
            project_peaks_to_current_view=project_peaks_to_current_view,
            entry_display_coords=entry_display_coords,
            filter_active_rows=_filter_active_rows,
        )

    def _prewarm_pick_cache(
        *,
        param_set: dict[str, object] | None = None,
        background_index: int | None = None,
        background_image: object | None = None,
        build_caked_projection_sidecar: bool | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        background_local = _background_image() if background_image is None else background_image
        if background_local is None:
            return {}
        sidecar = (
            bool(_manual_pick_uses_caked_space())
            if build_caked_projection_sidecar is None
            else bool(build_caked_projection_sidecar)
        )
        return _get_pick_cache(
            param_set=param_set,
            prefer_cache=True,
            background_index=background_index,
            background_image=background_local,
            build_caked_projection_sidecar=sidecar,
            **kwargs,
        )

    return GeometryManualRuntimeCacheCallbacks(
        current_match_config=_current_match_config,
        pick_cache_signature=_pick_cache_signature,
        get_pick_cache=_get_pick_cache,
        prewarm_pick_cache=_prewarm_pick_cache,
        build_initial_pairs_display=_build_initial_pairs_display,
    )


def make_runtime_geometry_manual_projection_callbacks(
    *,
    caked_view_enabled: Callable[[], object] | object,
    last_caked_background_image_unscaled: Callable[[], object] | object,
    last_caked_radial_values: Callable[[], object] | object,
    last_caked_azimuth_values: Callable[[], object] | object,
    current_background_display: Callable[[], object] | object,
    current_background_native: Callable[[], object] | object,
    ai: Callable[[], object] | object = None,
    center: Callable[[], Sequence[float] | None] | Sequence[float] | None = None,
    detector_distance: Callable[[], object] | object = 0.0,
    pixel_size: Callable[[], object] | object = 0.0,
    caked_transform_bundle: Callable[[], object] | object = None,
    wrap_phi_range: Callable[[object], object] = lambda value: value,
    rotate_point_for_display: Callable[
        [float, float, tuple[int, ...], int], tuple[float, float]
    ] = _default_rotate_point,
    display_rotate_k: int = 0,
    current_background_index: Callable[[], object] | object = None,
    caked_projection_payload: Callable[[], object] | object = None,
    last_caked_raw_azimuth_values: Callable[[], object] | object = None,
    last_caked_raw_to_gui_row_permutation: Callable[[], object] | object = None,
    current_geometry_fit_params: Callable[[], dict[str, object]] | None = None,
    build_live_preview_simulated_peaks_from_cache: (
        Callable[[], Sequence[dict[str, object]]] | None
    ) = None,
    ensure_peak_overlay_data: Callable[..., object] | None = None,
    miller: Callable[[], object] | object = None,
    intensities: Callable[[], object] | object = None,
    image_size: Callable[[], object] | object = 0,
    display_to_native_sim_coords: Callable[..., tuple[float, float]] | None = None,
    native_sim_to_display_coords: Callable[
        [float, float, tuple[int, ...]],
        tuple[float, float],
    ]
    | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]] = lambda _ai: (None, None),
    detector_pixel_to_scattering_angles: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    backend_detector_coords_to_native_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    native_detector_coords_to_bundle_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    simulation_native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
    bundle_detector_coords_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    scattering_angles_to_detector_pixel: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    filter_simulated_peaks: Callable[..., tuple[Sequence[dict[str, object]], object, object]]
    | None = None,
    collapse_simulated_peaks: Callable[..., tuple[Sequence[dict[str, object]], object]]
    | None = None,
    merge_radius_px: float = 6.0,
    profile_cache: Callable[[], Mapping[str, object] | None] | Mapping[str, object] | None = None,
) -> GeometryManualRuntimeProjectionCallbacks:
    """Build live manual-geometry projection callbacks around shared helpers."""

    def _pick_uses_caked_space() -> bool:
        if not bool(_resolve_runtime_value(caked_view_enabled)):
            return False
        return bool(_caked_grid_available() or _caked_projection_context_available())

    def _caked_grid_available() -> bool:
        background_image = _resolve_runtime_value(last_caked_background_image_unscaled)
        return (
            isinstance(background_image, np.ndarray)
            and background_image.ndim == 2
            and background_image.size > 0
            and np.asarray(
                _resolve_runtime_value(last_caked_radial_values),
                dtype=float,
            ).size
            > 1
            and np.asarray(
                _resolve_runtime_value(last_caked_azimuth_values),
                dtype=float,
            ).size
            > 1
        )

    def _caked_projection_context_available() -> bool:
        if callable(simulation_native_detector_coords_to_caked_display_coords):
            return True
        if isinstance(_resolve_runtime_value(caked_transform_bundle), CakeTransformBundle):
            return True
        ai_value = _resolve_runtime_value(ai)
        return isinstance(
            getattr(ai_value, "_live_caked_transform_bundle", None),
            CakeTransformBundle,
        )

    def _current_background_image() -> object | None:
        if _pick_uses_caked_space():
            return _resolve_runtime_value(last_caked_background_image_unscaled)
        return _resolve_runtime_value(current_background_display)

    def _detector_center() -> Sequence[float] | None:
        value = _resolve_runtime_value(center)
        if value is None:
            return None
        try:
            return [float(value[0]), float(value[1])]
        except Exception:
            return None

    def _native_to_caked_display_coords(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        return native_detector_coords_to_caked_display_coords(
            col,
            row,
            ai=_resolve_runtime_value(ai),
            get_detector_angular_maps=get_detector_angular_maps,
            detector_pixel_to_scattering_angles=detector_pixel_to_scattering_angles,
            center=_detector_center(),
            detector_distance=float(_resolve_runtime_value(detector_distance) or 0.0),
            pixel_size=float(_resolve_runtime_value(pixel_size) or 0.0),
            transform_bundle=_resolve_runtime_value(caked_transform_bundle),
            native_detector_coords_to_bundle_detector_coords=(
                native_detector_coords_to_bundle_detector_coords
            ),
            wrap_phi_range=wrap_phi_range,
            caked_radial_values=_resolve_runtime_value(last_caked_radial_values),
            caked_azimuth_values=_resolve_runtime_value(last_caked_azimuth_values),
        )

    def _finite_projection_point(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _simulation_native_to_caked_display_coords(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        if callable(simulation_native_detector_coords_to_caked_display_coords):
            try:
                projected = simulation_native_detector_coords_to_caked_display_coords(
                    float(col),
                    float(row),
                )
            except Exception:
                projected = None
            projected_tuple = _finite_projection_point(projected)
            if projected_tuple is not None:
                return float(projected_tuple[0]), float(projected_tuple[1])
        return _native_to_caked_display_coords(float(col), float(row))

    def _caked_angles_to_background_display(
        two_theta_deg: float,
        phi_deg: float,
    ) -> tuple[float | None, float | None]:
        return caked_angles_to_background_display_coords(
            two_theta_deg,
            phi_deg,
            ai=_resolve_runtime_value(ai),
            native_background=_resolve_runtime_value(current_background_native),
            caked_radial_values=_resolve_runtime_value(last_caked_radial_values),
            caked_azimuth_values=_resolve_runtime_value(last_caked_azimuth_values),
            get_detector_angular_maps=get_detector_angular_maps,
            scattering_angles_to_detector_pixel=scattering_angles_to_detector_pixel,
            center=_detector_center(),
            detector_distance=float(_resolve_runtime_value(detector_distance) or 0.0),
            pixel_size=float(_resolve_runtime_value(pixel_size) or 0.0),
            transform_bundle=_resolve_runtime_value(caked_transform_bundle),
            backend_detector_coords_to_native_detector_coords=(
                backend_detector_coords_to_native_detector_coords
            ),
            bundle_detector_coords_to_background_display_coords=(
                bundle_detector_coords_to_background_display_coords
            ),
            rotate_point_for_display=rotate_point_for_display,
            display_rotate_k=int(display_rotate_k),
        )

    def _background_display_to_caked_display(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        native_point = _background_display_to_native_detector_coords(float(col), float(row))
        if native_point is None:
            return None
        return _native_to_caked_display_coords(
            float(native_point[0]),
            float(native_point[1]),
        )

    def _background_display_to_native_detector_coords(
        col: float,
        row: float,
    ) -> tuple[float, float] | None:
        native_background = _resolve_runtime_value(current_background_native)
        if native_background is None:
            return None
        try:
            shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
            native_col, native_row = _default_display_point_to_native_for_rotation(
                float(col),
                float(row),
                shape,
                int(display_rotate_k),
                rotate_point_for_display_fn=rotate_point_for_display,
            )
        except Exception:
            return None
        if not (np.isfinite(native_col) and np.isfinite(native_row)):
            return None
        return float(native_col), float(native_row)

    def _refresh_entry_geometry(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        native_background = _resolve_runtime_value(current_background_native)
        try:
            background_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
        except Exception:
            background_shape = ()
        current_projected_sim_entry = _geometry_manual_current_sim_caked_projection_entry(
            entry,
            native_detector_coords_to_caked_display_coords=(
                _simulation_native_to_caked_display_coords
            ),
        )
        return refresh_geometry_manual_pair_entry(
            entry,
            background_display_shape=background_shape,
            background_display_to_native_detector_coords=(
                _background_display_to_native_detector_coords
            ),
            caked_angles_to_background_display_coords=(_caked_angles_to_background_display),
            native_detector_coords_to_caked_display_coords=(_native_to_caked_display_coords),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            rotate_point_for_display=rotate_point_for_display,
            display_rotate_k=int(display_rotate_k),
            allow_legacy_peak_fallback=False,
            current_projected_sim_entry=current_projected_sim_entry,
        )

    def _entry_display_coords(
        entry: dict[str, object] | None,
    ) -> tuple[float, float] | None:
        detector_origin = _geometry_manual_saved_background_detector_origin(entry)
        refreshed_entry = _refresh_entry_geometry(entry)
        if not isinstance(refreshed_entry, dict):
            return None
        use_caked = _pick_uses_caked_space()
        if use_caked and detector_origin:
            for detector_source in (entry, refreshed_entry):
                if not isinstance(detector_source, Mapping):
                    continue
                native_point = None
                for x_key, y_key in (
                    ("detector_x", "detector_y"),
                    ("background_detector_x", "background_detector_y"),
                    ("detector_native_x", "detector_native_y"),
                    ("refined_detector_native_col", "refined_detector_native_row"),
                ):
                    native_point = _finite_projection_point(
                        (detector_source.get(x_key), detector_source.get(y_key)),
                    )
                    if native_point is not None:
                        break
                if native_point is None:
                    for tuple_key in ("geometry_detector_native_px", "raw_detector_native_px"):
                        native_point = _finite_projection_point(detector_source.get(tuple_key))
                        if native_point is not None:
                            break
                if native_point is not None:
                    converted = _native_to_caked_display_coords(
                        float(native_point[0]),
                        float(native_point[1]),
                    )
                    if converted is not None:
                        return float(converted[0]), float(converted[1])
            for detector_source in (entry, refreshed_entry):
                if not isinstance(detector_source, Mapping):
                    continue
                display_point = None
                for tuple_key in ("geometry_detector_display_px", "raw_detector_display_px"):
                    display_point = _finite_projection_point(detector_source.get(tuple_key))
                    if display_point is not None:
                        break
                if display_point is None:
                    display_point = _finite_projection_point(
                        (detector_source.get("x"), detector_source.get("y")),
                    )
                if display_point is None:
                    continue
                converted = _background_display_to_caked_display(
                    float(display_point[0]),
                    float(display_point[1]),
                )
                if converted is not None:
                    return float(converted[0]), float(converted[1])
        key_x = "caked_x" if use_caked else "x"
        key_y = "caked_y" if use_caked else "y"
        try:
            col = float(refreshed_entry.get(key_x, np.nan))
            row = float(refreshed_entry.get(key_y, np.nan))
        except Exception:
            col = float("nan")
            row = float("nan")
        if use_caked and not (np.isfinite(col) and np.isfinite(row)):
            try:
                detector_col = float(refreshed_entry.get("detector_x", np.nan))
                detector_row = float(refreshed_entry.get("detector_y", np.nan))
            except Exception:
                detector_col = float("nan")
                detector_row = float("nan")
            if np.isfinite(detector_col) and np.isfinite(detector_row):
                converted = _native_to_caked_display_coords(
                    float(detector_col),
                    float(detector_row),
                )
                if converted is not None:
                    col = float(converted[0])
                    row = float(converted[1])
        if use_caked and not (np.isfinite(col) and np.isfinite(row)):
            try:
                raw_col = float(refreshed_entry.get("x", np.nan))
                raw_row = float(refreshed_entry.get("y", np.nan))
            except Exception:
                raw_col = float("nan")
                raw_row = float("nan")
            converted = _background_display_to_caked_display(raw_col, raw_row)
            if converted is not None:
                col = float(converted[0])
                row = float(converted[1])
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _project_peaks_to_view(
        simulated_peaks: Sequence[dict[str, object]] | None,
        *,
        force_caked: bool | None = None,
    ) -> list[dict[str, object]]:
        def _entry_point(
            source: Mapping[str, object],
            x_key: str,
            y_key: str,
        ) -> tuple[float, float] | None:
            try:
                col = float(source.get(x_key, np.nan))
                row = float(source.get(y_key, np.nan))
            except Exception:
                return None
            if not (np.isfinite(col) and np.isfinite(row)):
                return None
            return float(col), float(row)

        def _points_match(
            left: tuple[float, float] | None,
            right: tuple[float, float] | None,
            *,
            tol: float = 1.0e-9,
        ) -> bool:
            return bool(
                left is not None
                and right is not None
                and abs(float(left[0]) - float(right[0])) <= float(tol)
                and abs(float(left[1]) - float(right[1])) <= float(tol)
            )

        projected: list[dict[str, object]] = []
        use_caked = _pick_uses_caked_space() if force_caked is None else bool(force_caked)
        radial_axis = (
            np.asarray(_resolve_runtime_value(last_caked_radial_values), dtype=float)
            if use_caked
            else np.array([])
        )
        azimuth_axis = (
            np.asarray(_resolve_runtime_value(last_caked_azimuth_values), dtype=float)
            if use_caked
            else np.array([])
        )
        try:
            sim_shape = (
                int(_resolve_runtime_value(image_size)),
                int(_resolve_runtime_value(image_size)),
            )
        except Exception:
            sim_shape = (0, 0)
        try:
            native_background = _resolve_runtime_value(current_background_native)
            background_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
        except Exception:
            background_shape = ()

        for raw_entry in simulated_peaks or []:
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            refined_detector_display = _entry_point(
                entry,
                "refined_sim_x",
                "refined_sim_y",
            )
            refined_native_point = _entry_point(
                entry,
                "refined_sim_native_x",
                "refined_sim_native_y",
            )
            refined_caked_point = _entry_point(
                entry,
                "refined_sim_caked_x",
                "refined_sim_caked_y",
            )

            native_point = refined_native_point
            native_point_source = "refined_native" if native_point is not None else None
            if native_point is None:
                native_point = _entry_point(entry, "native_col", "native_row")
                if native_point is not None:
                    native_point_source = "native"
            if native_point is None:
                native_point = _entry_point(entry, "sim_native_x", "sim_native_y")
                if native_point is not None:
                    native_point_source = "sim_native"

            caked_point_source: str | None = None
            caked_point = refined_caked_point
            if caked_point is not None:
                caked_point_source = "refined_caked"
            if caked_point is None:
                caked_point = _entry_point(entry, "caked_x", "caked_y")
                if caked_point is not None:
                    caked_point_source = "caked"
            if caked_point is None:
                caked_point = _entry_point(entry, "raw_caked_x", "raw_caked_y")
                if caked_point is not None:
                    caked_point_source = "raw_caked"
            if caked_point is None:
                caked_point = _entry_point(entry, "two_theta_deg", "phi_deg")
                if caked_point is not None:
                    caked_point_source = "angles"

            display_detector_candidate = _entry_point(
                entry,
                "display_col",
                "display_row",
            )
            legacy_sim_point = _entry_point(entry, "sim_col", "sim_row")
            legacy_xy_point = _entry_point(entry, "x", "y")
            raw_sim_detector_point = _entry_point(entry, "sim_col_raw", "sim_row_raw")
            raw_detector_display = raw_sim_detector_point
            raw_detector_display_source = (
                "sim_col_raw" if raw_sim_detector_point is not None else None
            )
            if refined_detector_display is not None and (
                native_point_source == "refined_native"
                or (not use_caked and native_point_source is None)
            ):
                raw_detector_display = refined_detector_display
                raw_detector_display_source = "refined_sim_x"
            elif (
                native_point is None
                and refined_native_point is None
                and refined_caked_point is not None
            ):
                raw_detector_display = None
            if (
                use_caked
                and native_point is None
                and raw_detector_display is None
                and caked_point_source != "angles"
            ):
                continue
            # display_col/display_row are active-view coordinates. After a GUI
            # state restore they may be detector display pixels, caked axes, or
            # stale overlay coordinates. They are therefore not stable detector
            # provenance and must not be converted back into native detector
            # coordinates. Only explicit detector-space fields, native fields,
            # or caked angle fields are allowed to seed reprojection.
            frozen_display_point = None
            background_detector_frame = geometry_manual_entry_is_background_detector_frame(entry)
            simulation_native_frame = geometry_manual_entry_is_simulation_native_frame(entry)
            detector_display_source_token = _geometry_manual_normalized_frame_token(
                entry.get("detector_display_source")
            )
            legacy_matches_display = bool(
                legacy_sim_point is not None
                and display_detector_candidate is not None
                and _points_match(legacy_sim_point, display_detector_candidate)
            )
            if (
                raw_detector_display is None
                and native_point is None
                and caked_point is None
                and display_detector_candidate is not None
                and not legacy_matches_display
            ):
                frozen_display_point = (
                    float(display_detector_candidate[0]),
                    float(display_detector_candidate[1]),
                )
            for stale_key in (
                "caked_x",
                "caked_y",
                "raw_caked_x",
                "raw_caked_y",
                "sim_col_local",
                "sim_row_local",
                "sim_col_global",
                "sim_row_global",
            ):
                entry.pop(stale_key, None)

            detector_display_shape = (
                background_shape
                if len(background_shape) >= 2 and min(background_shape) > 0
                else sim_shape
            )
            detector_replay_from_caked_projection = False
            sim_detector_replay = None
            trusted_sim_detector_display = False
            can_replay_caked_projection = bool(
                _caked_projection_context_available()
                and (
                    callable(native_detector_coords_to_detector_display_coords)
                    or _caked_grid_available()
                )
            )
            if not use_caked and can_replay_caked_projection:
                replay_source_entry = _geometry_manual_caked_qr_projection_source(entry)
                current_projected_caked_entry = _geometry_manual_current_sim_caked_projection_entry(
                    entry,
                    native_detector_coords_to_caked_display_coords=(
                        _simulation_native_to_caked_display_coords
                    ),
                )
                if current_projected_caked_entry is not None and replay_source_entry is not None:
                    detector_replay_from_caked_projection = True
                    caked_projection_point = _entry_point(
                        current_projected_caked_entry,
                        "caked_x",
                        "caked_y",
                    )
                    if caked_projection_point is None:
                        caked_projection_point = _entry_point(
                            current_projected_caked_entry,
                            "two_theta_deg",
                            "phi_deg",
                        )
                    if caked_projection_point is not None:
                        caked_point = (
                            float(caked_projection_point[0]),
                            float(caked_projection_point[1]),
                        )
                        caked_point_source = "current_projection"
                    pre_replay_native_point = native_point
                    pre_replay_native_point_source = native_point_source
                    native_point = None
                    native_point_source = None
                    raw_detector_display = None
                    raw_detector_display_source = None
                    refined_detector_display = None
                    sim_detector_replay = resolve_sim_detector_replay_from_caked_projection(
                        entry,
                        current_projected_caked_entry,
                        caked_angles_to_background_display_coords=(
                            _caked_angles_to_background_display
                        ),
                        background_display_to_native_detector_coords=(
                            _background_display_to_native_detector_coords
                        ),
                        native_detector_coords_to_caked_display_coords=(
                            _simulation_native_to_caked_display_coords
                        ),
                        native_detector_coords_to_detector_display_coords=(
                            native_detector_coords_to_detector_display_coords
                        ),
                    )
                    if sim_detector_replay is not None:
                        native_point = _entry_point(
                            sim_detector_replay,
                            "sim_detector_anchor_x",
                            "sim_detector_anchor_y",
                        )
                        if native_point is not None:
                            native_point_source = "sim_detector_anchor"
                        raw_detector_display = _entry_point(
                            sim_detector_replay,
                            "sim_detector_display_col",
                            "sim_detector_display_row",
                        )
                        if raw_detector_display is not None:
                            trusted_sim_detector_display = True
                        replay_provenance = sim_detector_replay.get("sim_detector_frame_provenance")
                        if replay_provenance is not None:
                            raw_detector_display_source = str(replay_provenance)
                    else:
                        native_point = pre_replay_native_point
                        native_point_source = pre_replay_native_point_source
            if native_point is not None:
                projected_native = None
                preserve_raw_detector_display = bool(
                    raw_detector_display is not None
                    and (
                        detector_replay_from_caked_projection
                        or (
                            not background_detector_frame
                            and (
                                detector_display_source_token
                                in _GEOMETRY_MANUAL_BACKGROUND_DISPLAY_SOURCES
                                or (
                                    native_point_source == "refined_native"
                                    and (
                                        raw_detector_display_source == "refined_sim_x"
                                        or detector_display_source_token
                                        in _GEOMETRY_MANUAL_SIMULATION_DISPLAY_SOURCES
                                        or (
                                            simulation_native_frame
                                            and callable(native_sim_to_display_coords)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                if not use_caked and preserve_raw_detector_display:
                    projected_native = raw_detector_display
                if (
                    projected_native is None
                    and not use_caked
                    and not detector_replay_from_caked_projection
                    and simulation_native_frame
                    and callable(native_sim_to_display_coords)
                ):
                    try:
                        projected_native = native_sim_to_display_coords(
                            float(native_point[0]),
                            float(native_point[1]),
                            sim_shape,
                        )
                        raw_detector_display_source = "native_sim_to_display"
                    except Exception:
                        projected_native = None
                if (
                    projected_native is None
                    and callable(native_detector_coords_to_detector_display_coords)
                    and (
                        detector_replay_from_caked_projection
                        or (
                            background_detector_frame
                            or use_caked
                            or not simulation_native_frame
                            or not callable(native_sim_to_display_coords)
                        )
                    )
                ):
                    try:
                        projected_native = native_detector_coords_to_detector_display_coords(
                            float(native_point[0]),
                            float(native_point[1]),
                        )
                        raw_detector_display_source = (
                            "native_detector_coords_to_detector_display_coords"
                        )
                    except Exception:
                        projected_native = None
                    projected_native = _finite_projection_point(projected_native)
                    if detector_replay_from_caked_projection and projected_native is not None:
                        trusted_sim_detector_display = True
                if (
                    projected_native is None
                    and not detector_replay_from_caked_projection
                    and callable(native_sim_to_display_coords)
                ):
                    try:
                        projected_native = native_sim_to_display_coords(
                            float(native_point[0]),
                            float(native_point[1]),
                            sim_shape,
                        )
                        raw_detector_display_source = "native_sim_to_display"
                    except Exception:
                        projected_native = None
                if (
                    projected_native is None
                    and len(detector_display_shape) >= 2
                    and min(detector_display_shape) > 0
                ):
                    try:
                        rotate_k = (
                            -int(display_rotate_k)
                            if detector_replay_from_caked_projection
                            else int(display_rotate_k)
                        )
                        projected_native = rotate_point_for_display(
                            float(native_point[0]),
                            float(native_point[1]),
                            detector_display_shape,
                            rotate_k,
                        )
                    except Exception:
                        projected_native = None
                projected_native = _finite_projection_point(projected_native)
                if projected_native is not None:
                    raw_detector_display = (
                        float(projected_native[0]),
                        float(projected_native[1]),
                    )

            if native_point is None and raw_detector_display is not None:
                use_background_display_inverse = bool(
                    refined_detector_display is not None
                    and abs(float(raw_detector_display[0]) - float(refined_detector_display[0]))
                    <= 1.0e-9
                    and abs(float(raw_detector_display[1]) - float(refined_detector_display[1]))
                    <= 1.0e-9
                    and callable(_background_display_to_native_detector_coords)
                )
                if use_background_display_inverse:
                    try:
                        derived_native = _background_display_to_native_detector_coords(
                            float(raw_detector_display[0]),
                            float(raw_detector_display[1]),
                        )
                    except Exception:
                        derived_native = None
                elif raw_detector_display_source == "sim_col_raw" and callable(
                    display_to_native_sim_coords
                ):
                    try:
                        derived_native = display_to_native_sim_coords(
                            float(raw_detector_display[0]),
                            float(raw_detector_display[1]),
                            sim_shape,
                        )
                    except Exception:
                        derived_native = None
                else:
                    derived_native = None
                if (
                    isinstance(derived_native, tuple)
                    and len(derived_native) >= 2
                    and np.isfinite(float(derived_native[0]))
                    and np.isfinite(float(derived_native[1]))
                ):
                    native_point = (
                        float(derived_native[0]),
                        float(derived_native[1]),
                    )
                    if not use_caked and raw_detector_display_source == "sim_col_raw":
                        projected_detector = None
                        if callable(native_detector_coords_to_detector_display_coords):
                            try:
                                projected_detector = (
                                    native_detector_coords_to_detector_display_coords(
                                        float(native_point[0]),
                                        float(native_point[1]),
                                    )
                                )
                            except Exception:
                                projected_detector = None
                            projected_detector = _finite_projection_point(projected_detector)
                        if (
                            projected_detector is None
                            and len(detector_display_shape) >= 2
                            and min(detector_display_shape) > 0
                        ):
                            try:
                                projected_detector = rotate_point_for_display(
                                    float(native_point[0]),
                                    float(native_point[1]),
                                    detector_display_shape,
                                    int(display_rotate_k),
                                )
                            except Exception:
                                projected_detector = None
                        projected_detector = _finite_projection_point(projected_detector)
                        if projected_detector is not None:
                            raw_detector_display = (
                                float(projected_detector[0]),
                                float(projected_detector[1]),
                            )
                            raw_detector_display_source = (
                                "native_detector_coords_to_detector_display_coords"
                            )

            if (
                raw_detector_display is None
                and caked_point is not None
                and not detector_replay_from_caked_projection
                and callable(caked_angles_to_background_display_coords)
            ):
                try:
                    projected_detector = caked_angles_to_background_display_coords(
                        float(caked_point[0]),
                        float(caked_point[1]),
                    )
                except Exception:
                    projected_detector = None
                if (
                    isinstance(projected_detector, tuple)
                    and len(projected_detector) >= 2
                    and projected_detector[0] is not None
                    and projected_detector[1] is not None
                    and np.isfinite(float(projected_detector[0]))
                    and np.isfinite(float(projected_detector[1]))
                ):
                    raw_detector_display = (
                        float(projected_detector[0]),
                        float(projected_detector[1]),
                    )
                    if native_point is None and callable(
                        _background_display_to_native_detector_coords
                    ):
                        derived_native = _background_display_to_native_detector_coords(
                            float(raw_detector_display[0]),
                            float(raw_detector_display[1]),
                        )
                        if (
                            isinstance(derived_native, tuple)
                            and len(derived_native) >= 2
                            and np.isfinite(float(derived_native[0]))
                            and np.isfinite(float(derived_native[1]))
                        ):
                            native_point = (
                                float(derived_native[0]),
                                float(derived_native[1]),
                            )

            if native_point is not None:
                entry["native_col"] = float(native_point[0])
                entry["native_row"] = float(native_point[1])
                entry["sim_native_x"] = float(native_point[0])
                entry["sim_native_y"] = float(native_point[1])
                if simulation_native_frame and not background_detector_frame:
                    entry.setdefault("coordinate_frame", "simulation_native")

            if native_point is not None:
                caked_point = None
                caked_coords = _simulation_native_to_caked_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                )
                if caked_coords is not None:
                    caked_point = (float(caked_coords[0]), float(caked_coords[1]))

            if raw_detector_display is not None:
                if detector_replay_from_caked_projection:
                    entry.pop("refined_sim_x", None)
                    entry.pop("refined_sim_y", None)
                entry["sim_col_raw"] = float(raw_detector_display[0])
                entry["sim_row_raw"] = float(raw_detector_display[1])
                entry["sim_col"] = float(raw_detector_display[0])
                entry["sim_row"] = float(raw_detector_display[1])
                if raw_detector_display_source is not None and (
                    not use_caked
                    or raw_detector_display_source
                    in {
                        "native_sim_to_display",
                        "native_detector_coords_to_detector_display_coords",
                    }
                ):
                    entry["detector_display_source"] = str(raw_detector_display_source)
            else:
                for stale_key in (
                    "sim_col_raw",
                    "sim_row_raw",
                    "sim_col",
                    "sim_row",
                ):
                    entry.pop(stale_key, None)
            if native_point is None:
                for stale_key in (
                    "native_col",
                    "native_row",
                    "sim_native_x",
                    "sim_native_y",
                ):
                    entry.pop(stale_key, None)
            if (
                native_point is None
                and not use_caked
                and refined_detector_display is None
                and (
                    legacy_sim_point is not None
                    or (
                        raw_sim_detector_point is not None
                        and raw_detector_display_source != "sim_col_raw"
                    )
                )
            ):
                continue
            if detector_replay_from_caked_projection or use_caked:
                if native_point is not None:
                    entry["sim_detector_anchor_x"] = float(native_point[0])
                    entry["sim_detector_anchor_y"] = float(native_point[1])
                    replay_provenance = None
                    if isinstance(sim_detector_replay, Mapping):
                        replay_provenance = sim_detector_replay.get("sim_detector_frame_provenance")
                    if replay_provenance is None and raw_detector_display_source is not None:
                        replay_provenance = raw_detector_display_source
                    if replay_provenance is None and native_point_source is not None:
                        replay_provenance = native_point_source
                    if replay_provenance is not None:
                        entry["sim_detector_frame_provenance"] = str(replay_provenance)
                    else:
                        entry.pop("sim_detector_frame_provenance", None)
                else:
                    for stale_key in (
                        "sim_detector_anchor_x",
                        "sim_detector_anchor_y",
                        "sim_detector_frame_provenance",
                    ):
                        entry.pop(stale_key, None)
                if raw_detector_display is not None and (
                    not detector_replay_from_caked_projection or trusted_sim_detector_display
                ):
                    entry["sim_detector_display_col"] = float(raw_detector_display[0])
                    entry["sim_detector_display_row"] = float(raw_detector_display[1])
                else:
                    entry.pop("sim_detector_display_col", None)
                    entry.pop("sim_detector_display_row", None)
            else:
                for stale_key in (
                    "sim_detector_anchor_x",
                    "sim_detector_anchor_y",
                    "sim_detector_display_col",
                    "sim_detector_display_row",
                    "sim_detector_frame_provenance",
                ):
                    entry.pop(stale_key, None)
            if caked_point is not None:
                entry["caked_x"] = float(caked_point[0])
                entry["caked_y"] = float(caked_point[1])
                entry["raw_caked_x"] = float(caked_point[0])
                entry["raw_caked_y"] = float(caked_point[1])
                entry["two_theta_deg"] = float(caked_point[0])
                entry["phi_deg"] = float(caked_point[1])

            if use_caked:
                active_caked_point = caked_point or frozen_display_point
                if active_caked_point is not None:
                    entry["display_col"] = float(active_caked_point[0])
                    entry["display_row"] = float(active_caked_point[1])
                    entry["display_frame"] = "caked_display"
                    entry["current_view_frame"] = "caked_display"
                    if caked_point is None:
                        entry["caked_x"] = float(active_caked_point[0])
                        entry["caked_y"] = float(active_caked_point[1])
                    entry["sim_col_global"] = float(active_caked_point[0])
                    entry["sim_row_global"] = float(active_caked_point[1])
                    entry["sim_col_local"] = float(
                        caked_axis_to_image_index(float(active_caked_point[0]), radial_axis)
                    )
                    entry["sim_row_local"] = float(
                        caked_axis_to_image_index(float(active_caked_point[1]), azimuth_axis)
                    )
                    projected.append(entry)
                continue

            if frozen_display_point is not None:
                entry["display_col"] = float(frozen_display_point[0])
                entry["display_row"] = float(frozen_display_point[1])
                entry["display_frame"] = "detector_display"
                projected.append(entry)
                continue

            if raw_detector_display is None:
                continue
            entry["sim_col"] = float(raw_detector_display[0])
            entry["sim_row"] = float(raw_detector_display[1])
            entry["display_col"] = float(raw_detector_display[0])
            entry["display_row"] = float(raw_detector_display[1])
            entry["display_frame"] = "detector_display"
            projected.append(entry)
        return projected

    def _project_peaks_to_current_view(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        return _project_peaks_to_view(simulated_peaks, force_caked=None)

    def _project_peaks_to_caked_view(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        return _project_peaks_to_view(simulated_peaks, force_caked=True)

    last_simulation_diagnostics_state: dict[str, object] = {}

    def _copy_simulation_diag_value(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): _copy_simulation_diag_value(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return _copy_simulation_diag_value(value.tolist())
        if isinstance(value, (list, tuple)):
            return [_copy_simulation_diag_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _copy_simulation_diagnostics(
        diagnostics: Mapping[str, object] | None,
    ) -> dict[str, object]:
        copied = _copy_simulation_diag_value(diagnostics if diagnostics is not None else {})
        return copied if isinstance(copied, dict) else {}

    def _set_last_simulation_diagnostics(
        diagnostics: Mapping[str, object] | None = None,
        **updates: object,
    ) -> dict[str, object]:
        snapshot = _copy_simulation_diagnostics(diagnostics)
        for key, value in updates.items():
            snapshot[str(key)] = _copy_simulation_diag_value(value)
        last_simulation_diagnostics_state.clear()
        last_simulation_diagnostics_state.update(snapshot)
        return _copy_simulation_diagnostics(last_simulation_diagnostics_state)

    def _last_simulation_diagnostics() -> dict[str, object]:
        return _copy_simulation_diagnostics(last_simulation_diagnostics_state)

    def _shape_list(value: object) -> list[int]:
        try:
            return [int(v) for v in np.asarray(value).shape]
        except Exception:
            return []

    def _array_count(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(np.asarray(value).size)
        except Exception:
            try:
                return int(len(value))  # type: ignore[arg-type]
            except Exception:
                return None

    def _mapping_entry_list(
        value: Sequence[dict[str, object]] | object | None,
    ) -> list[dict[str, object]]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            iterable = value.tolist()
        else:
            iterable = value
        try:
            return [dict(entry) for entry in iterable if isinstance(entry, Mapping)]
        except Exception:
            return []

    def _visible_simulated_peak_entries(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        filtered_entries = _mapping_entry_list(simulated_peaks)
        if callable(filter_simulated_peaks):
            filtered_result = filter_simulated_peaks(filtered_entries)
            if isinstance(filtered_result, tuple) and filtered_result:
                filtered_entries = _mapping_entry_list(filtered_result[0])
        collapsed_entries = list(filtered_entries)
        if callable(collapse_simulated_peaks):
            try:
                collapsed_result = collapse_simulated_peaks(
                    filtered_entries,
                    merge_radius_px=float(merge_radius_px),
                    one_per_q_group=False,
                )
            except TypeError:
                collapsed_result = collapse_simulated_peaks(
                    filtered_entries,
                    merge_radius_px=float(merge_radius_px),
                )
            if isinstance(collapsed_result, tuple) and collapsed_result:
                collapsed_entries = _mapping_entry_list(collapsed_result[0])
        collapsed_representatives = _geometry_manual_collapse_q_group_representatives(
            _mapping_entry_list(collapsed_entries),
            profile_cache=_resolve_runtime_value(profile_cache),
        )
        return collapsed_representatives or _mapping_entry_list(collapsed_entries)

    def _missing_required_param_keys(
        params_local: Mapping[str, object],
    ) -> list[str]:
        required_keys = (
            "a",
            "c",
            "lambda",
            "corto_detector",
            "gamma",
            "Gamma",
            "chi",
            "zs",
            "zb",
            "n2",
            "debye_x",
            "debye_y",
            "center",
            "theta_initial",
        )
        missing: list[str] = []
        for key in required_keys:
            value = params_local.get(key)
            if key == "center":
                if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
                    missing.append(key)
                    continue
                try:
                    coords_ok = np.isfinite(float(value[0])) and np.isfinite(float(value[1]))
                except Exception:
                    coords_ok = False
                if not coords_ok:
                    missing.append(key)
                continue
            if value is None:
                missing.append(key)
                continue
            try:
                if not np.isfinite(float(value)):
                    missing.append(key)
            except Exception:
                continue
        return missing

    def _mosaic_array_sizes(
        params_local: Mapping[str, object],
    ) -> tuple[dict[str, int | None], list[str]]:
        mosaic = params_local.get("mosaic_params")
        mosaic_params = dict(mosaic) if isinstance(mosaic, Mapping) else {}
        array_sizes = {
            "beam_x_array": _array_count(mosaic_params.get("beam_x_array")),
            "beam_y_array": _array_count(mosaic_params.get("beam_y_array")),
            "theta_array": _array_count(mosaic_params.get("theta_array")),
            "phi_array": _array_count(mosaic_params.get("phi_array")),
            "wavelength_array": _array_count(mosaic_params.get("wavelength_array")),
            "wavelength_i_array": _array_count(mosaic_params.get("wavelength_i_array")),
            "sample_weights": _array_count(mosaic_params.get("sample_weights")),
        }
        missing = [
            key
            for key in ("beam_x_array", "beam_y_array", "theta_array", "phi_array")
            if not int(array_sizes.get(key) or 0)
        ]
        if not max(
            int(array_sizes.get("wavelength_array") or 0),
            int(array_sizes.get("wavelength_i_array") or 0),
        ):
            missing.append("wavelength_array")
        return array_sizes, missing

    def _simulated_peaks_for_params(
        param_set: dict[str, object] | None = None,
        *,
        prefer_cache: bool = True,
    ) -> list[dict[str, object]]:
        def _cached_preview_peaks() -> list[dict[str, object]]:
            if not callable(build_live_preview_simulated_peaks_from_cache):
                return []
            try:
                return [
                    dict(entry)
                    for entry in (build_live_preview_simulated_peaks_from_cache() or ())
                    if isinstance(entry, dict)
                ]
            except Exception:
                return []

        if prefer_cache and callable(build_live_preview_simulated_peaks_from_cache):
            cached_peaks = _cached_preview_peaks()
            if not cached_peaks and callable(ensure_peak_overlay_data):
                try:
                    ensure_peak_overlay_data(force=False)
                except TypeError:
                    ensure_peak_overlay_data()
                except Exception:
                    pass
                cached_peaks = _cached_preview_peaks()
            if cached_peaks:
                visible_cached_peaks = _visible_simulated_peak_entries(cached_peaks)
                projected_cached_peaks = _project_peaks_to_current_view(visible_cached_peaks)
                _set_last_simulation_diagnostics(
                    source="cache",
                    requested_prefer_cache=bool(prefer_cache),
                    status="cache_hit",
                    raw_peak_count=int(len(cached_peaks)),
                    visible_peak_count=int(len(visible_cached_peaks)),
                    projected_peak_count=int(len(projected_cached_peaks)),
                )
                return projected_cached_peaks
        _set_last_simulation_diagnostics(
            source="fresh",
            requested_prefer_cache=bool(prefer_cache),
            status="simulated_peak_rebuild_disabled",
        )
        return []

    def _pick_candidates(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> dict[tuple[object, ...], list[dict[str, object]]]:
        def _group_entries(
            candidate_entries: Sequence[dict[str, object]] | None,
        ) -> dict[tuple[object, ...], list[dict[str, object]]]:
            grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
            for raw_entry in candidate_entries or []:
                if not isinstance(raw_entry, dict):
                    continue
                group_key = raw_entry.get("q_group_key")
                if not isinstance(group_key, tuple):
                    continue
                grouped[group_key].append(dict(raw_entry))
            for entry_list in grouped.values():
                entry_list.sort(
                    key=lambda entry: (
                        (
                            _geometry_manual_entry_active_view_point(
                                entry,
                                use_caked_display=_pick_uses_caked_space(),
                            )
                            or (float("inf"), float("inf"))
                        )[0],
                        (
                            _geometry_manual_entry_active_view_point(
                                entry,
                                use_caked_display=_pick_uses_caked_space(),
                            )
                            or (float("inf"), float("inf"))
                        )[1],
                    ),
                )
            return dict(grouped)

        return _group_entries(_visible_simulated_peak_entries(simulated_peaks))

    def _simulated_lookup(
        simulated_peaks: Sequence[dict[str, object]] | None,
    ) -> GeometryManualLookupMap:
        lookup: GeometryManualLookupMap = {}
        for raw_entry in simulated_peaks or []:
            if not isinstance(raw_entry, dict):
                continue
            key = geometry_manual_candidate_source_key(raw_entry)
            if key is None:
                continue
            _geometry_manual_add_lookup_entry(
                lookup,
                key,
                raw_entry,
            )
        return lookup

    def _axis_signature(values: object) -> tuple[object, ...] | None:
        return _geometry_manual_projection_axis_token(values)

    def _permutation_signature(values: object) -> tuple[object, ...] | None:
        return _geometry_manual_projection_permutation_token(values)

    def _detector_shape_signature(values: object) -> tuple[int, int] | None:
        return _geometry_manual_detector_shape_token(values)

    def _background_index_signature() -> int | None:
        try:
            return int(_resolve_runtime_value(current_background_index))
        except Exception:
            return None

    def _projection_payload_mapping() -> Mapping[str, object] | None:
        payload = _resolve_runtime_value(caked_projection_payload)
        return payload if isinstance(payload, Mapping) else None

    def _bundle_signature(bundle: object) -> tuple[object, ...] | None:
        return _geometry_manual_projection_bundle_token(bundle)

    def _caked_projection_signature() -> object:
        if not (_caked_grid_available() or _caked_projection_context_available()):
            return None
        payload = _projection_payload_mapping()
        verified_projection_token = None
        if (
            isinstance(payload, Mapping)
            and str(payload.get("projection_token_schema") or "").strip()
            == _GEOMETRY_MANUAL_TRUSTED_PROJECTION_TOKEN_SCHEMA
            and str(payload.get("projection_content_token_source") or "").strip()
            == _GEOMETRY_MANUAL_TRUSTED_PROJECTION_TOKEN_SOURCE
        ):
            verified_projection_token = payload.get("projection_content_token_v3")
        bundle = (
            payload.get("transform_bundle")
            if isinstance(payload, Mapping)
            else _resolve_runtime_value(caked_transform_bundle)
        )
        detector_shape = payload.get("detector_shape") if isinstance(payload, Mapping) else None
        if isinstance(bundle, CakeTransformBundle):
            detector_shape = detector_shape or bundle.detector_shape
        radial_axis = (
            payload.get("radial_axis")
            if isinstance(payload, Mapping)
            else _resolve_runtime_value(last_caked_radial_values)
        )
        azimuth_axis = (
            payload.get("azimuth_axis")
            if isinstance(payload, Mapping)
            else _resolve_runtime_value(last_caked_azimuth_values)
        )
        raw_azimuth_axis = (
            payload.get("raw_azimuth_axis")
            if isinstance(payload, Mapping)
            else (
                _resolve_runtime_value(last_caked_raw_azimuth_values)
                if last_caked_raw_azimuth_values is not None
                else getattr(bundle, "raw_azimuth_deg", None)
            )
        )
        raw_to_gui = (
            payload.get("raw_to_gui_row_permutation")
            if isinstance(payload, Mapping)
            else _resolve_runtime_value(last_caked_raw_to_gui_row_permutation)
        )
        if isinstance(bundle, CakeTransformBundle) and verified_projection_token is None:
            return None
        try:
            display_rotate_token = int(display_rotate_k)
        except Exception:
            display_rotate_token = None
        try:
            image_size_token = int(_resolve_runtime_value(image_size))
        except Exception:
            image_size_token = None
        return (
            "caked_qr_projection_v3",
            _background_index_signature(),
            _detector_shape_signature(detector_shape),
            _axis_signature(radial_axis),
            _axis_signature(azimuth_axis),
            _axis_signature(raw_azimuth_axis),
            _permutation_signature(raw_to_gui),
            ("projection_content_token_v3", verified_projection_token),
            ("display_transform_v3", display_rotate_token, image_size_token),
        )

    return GeometryManualRuntimeProjectionCallbacks(
        pick_uses_caked_space=_pick_uses_caked_space,
        current_background_image=_current_background_image,
        entry_display_coords=_entry_display_coords,
        refresh_entry_geometry=_refresh_entry_geometry,
        caked_angles_to_background_display_coords=(_caked_angles_to_background_display),
        background_display_to_native_detector_coords=(
            _background_display_to_native_detector_coords
        ),
        native_detector_coords_to_caked_display_coords=(_native_to_caked_display_coords),
        project_peaks_to_current_view=_project_peaks_to_current_view,
        project_peaks_to_caked_view=_project_peaks_to_caked_view,
        simulated_peaks_for_params=_simulated_peaks_for_params,
        last_simulation_diagnostics=_last_simulation_diagnostics,
        pick_candidates=_pick_candidates,
        simulated_lookup=_simulated_lookup,
        caked_projection_signature=_caked_projection_signature,
    )


def render_current_geometry_manual_pairs(
    *,
    background_visible: bool,
    current_background_index: int,
    current_background_image: object | None,
    pick_session: dict[str, object] | None,
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ],
    session_initial_pairs_display: Callable[..., Sequence[dict[str, object]]],
    clear_geometry_pick_artists: Callable[..., None],
    draw_initial_geometry_pairs_overlay: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_background_file_status_text_fn: Callable[[], None],
    pair_group_count: Callable[[int], int],
    set_status_text: Callable[[str], None] | None = None,
    update_status: bool = False,
    reuse_only: bool = False,
) -> bool:
    """Redraw the saved manual geometry-pair overlay for the current background."""

    if not background_visible or current_background_image is None:
        clear_geometry_pick_artists()
        return False

    build_accepts_reuse_only = _geometry_manual_callable_accepts_keyword(
        build_initial_pairs_display,
        "reuse_only",
    )
    if build_accepts_reuse_only is False:
        measured_display, initial_pairs_display = build_initial_pairs_display(
            int(current_background_index),
            prefer_cache=True,
        )
    else:
        try:
            measured_display, initial_pairs_display = build_initial_pairs_display(
                int(current_background_index),
                prefer_cache=True,
                reuse_only=bool(reuse_only),
            )
        except TypeError as exc:
            if not _geometry_manual_type_error_is_keyword_arity_mismatch(exc, "reuse_only"):
                raise
            measured_display, initial_pairs_display = build_initial_pairs_display(
                int(current_background_index),
                prefer_cache=True,
            )
    session_accepts_reuse_only = _geometry_manual_callable_accepts_keyword(
        session_initial_pairs_display,
        "reuse_only",
    )
    if session_accepts_reuse_only is False:
        pending_pairs_display = list(session_initial_pairs_display())
    else:
        try:
            pending_pairs_display = list(
                session_initial_pairs_display(reuse_only=bool(reuse_only))
            )
        except TypeError as exc:
            if not _geometry_manual_type_error_is_keyword_arity_mismatch(exc, "reuse_only"):
                raise
            pending_pairs_display = list(session_initial_pairs_display())
    combined_pairs_display = list(initial_pairs_display) + list(pending_pairs_display)

    if not measured_display and not combined_pairs_display:
        clear_geometry_pick_artists()
        if update_status and callable(set_status_text):
            set_status_text("No saved manual geometry pairs for the current background image.")
        return False

    draw_initial_geometry_pairs_overlay(
        combined_pairs_display,
        max_display_markers=max(1, len(combined_pairs_display)),
    )
    update_button_label_fn()
    set_background_file_status_text_fn()

    if update_status and callable(set_status_text):
        if geometry_manual_pick_session_active(
            pick_session,
            current_background_index=current_background_index,
        ):
            pending_entries = (
                pick_session.get("pending_entries", []) if isinstance(pick_session, dict) else []
            )
            target_count = (
                pick_session.get("target_count") if isinstance(pick_session, dict) else None
            )
            q_label = str(
                pick_session.get(
                    "q_label",
                    pick_session.get("group_key", "selected Qr/Qz set"),
                )
                if isinstance(pick_session, dict)
                else "selected Qr/Qz set"
            )
            next_index = len(pending_entries) + 1 if isinstance(pending_entries, list) else 1
            try:
                total_count = int(target_count)
            except Exception:
                total_count = len(
                    pick_session.get("group_entries", []) if isinstance(pick_session, dict) else []
                )
            remaining_candidates = geometry_manual_unassigned_group_candidates(
                pick_session,
                current_background_index=current_background_index,
            )
            tagged_candidate = geometry_manual_tagged_candidate_from_session(
                pick_session,
                remaining_candidates,
            )
            set_status_text(
                f"Click background peak {next_index} of {max(1, total_count)} for {q_label}. "
                + (
                    "It will attach to the tagged central-beam seed."
                    if tagged_candidate is not None
                    else "It will attach to the nearest remaining simulated peak."
                )
            )
        else:
            set_status_text(
                "Current background has "
                f"{len(initial_pairs_display)} saved manual points across "
                f"{pair_group_count(int(current_background_index))} Qr/Qz groups."
            )
    return True


def make_runtime_geometry_manual_callbacks(
    *,
    background_visible: Callable[[], object] | object,
    current_background_index: Callable[[], object] | object,
    current_background_image: Callable[[], object | None] | object | None,
    pick_session: Callable[[], dict[str, object] | None] | dict[str, object] | None,
    build_initial_pairs_display: Callable[
        ...,
        tuple[list[dict[str, object]], list[dict[str, object]]],
    ],
    session_initial_pairs_display: Callable[..., Sequence[dict[str, object]]],
    clear_geometry_pick_artists: Callable[..., None],
    draw_initial_geometry_pairs_overlay: Callable[..., None],
    update_button_label: Callable[[], None],
    set_background_file_status_text: Callable[[], None],
    pair_group_count: Callable[[int], int],
    set_status_text: Callable[[str], None] | None,
    get_cache_data: Callable[..., dict[str, object]],
    set_pairs_for_index: Callable[
        [int, Sequence[dict[str, object]] | None],
        Sequence[dict[str, object]],
    ],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    set_pick_session: Callable[[dict[str, object]], None],
    restore_view: Callable[..., None],
    clear_preview_artists: Callable[..., None],
    push_undo_state: Callable[[], None] | None = None,
    listed_q_group_entries: Callable[[], Sequence[dict[str, object]]]
    | Sequence[dict[str, object]] = (),
    format_q_group_line: Callable[[dict[str, object]], str] | None = None,
    use_caked_space: Callable[[], object] | object = False,
    pick_search_window_px: float,
    caked_search_tth_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    caked_search_phi_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
    set_suppress_drag_press_once: Callable[[bool], None] | None = None,
    sync_peak_selection_state: Callable[[], None] | None = None,
    refine_preview_point: Callable[..., tuple[float, float]] | None = None,
    remaining_candidates: Callable[[], Sequence[dict[str, object]]] | None = None,
    preview_due: Callable[[float, float], bool] | None = None,
    nearest_candidate_to_point: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    find_peak_record_for_click: Callable[
        [float, float, float],
        tuple[int, dict[str, object] | None, float, bool],
    ]
    | None = None,
    position_error_px: Callable[
        [float, float, float, float],
        float,
    ] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    last_caked_radial_values: Callable[[], object] | object = (),
    last_caked_azimuth_values: Callable[[], object] | object = (),
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    refine_saved_pair_entry: Callable[
        [dict[str, object], dict[str, object] | None],
        dict[str, object] | None,
    ]
    | None = None,
    show_preview: Callable[..., None] | None = None,
    refresh_pick_session: Callable[..., dict[str, object] | None] | None = None,
    profile_cache: Callable[[], Mapping[str, object] | None] | Mapping[str, object] | None = None,
) -> GeometryManualRuntimeCallbacks:
    """Build live manual-geometry callbacks around the shared helper surface."""

    print_geometry_manual_live_code_path_stamp()
    restore_view_callback = restore_view

    def _background_index() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _background_image() -> object | None:
        return _resolve_runtime_value(current_background_image)

    def _refresh_pick_session_signature_capabilities() -> tuple[bool, bool] | None:
        if not callable(refresh_pick_session):
            return (False, False)
        try:
            signature = inspect.signature(refresh_pick_session)
        except (TypeError, ValueError):
            return None
        parameters = signature.parameters
        accepts_reuse_only = "reuse_only" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        accepts_session = any(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            )
            for parameter in parameters.values()
        )
        return (accepts_session, accepts_reuse_only)

    def _refresh_pick_session_type_error_is_arity_mismatch(exc: TypeError) -> bool:
        message = str(exc)
        if "reuse_only" in message and (
            "unexpected keyword" in message
            or "got an unexpected" in message
            or "invalid keyword" in message
        ):
            return True
        return bool(
            "positional argument" in message
            or "required positional" in message
            or "takes" in message and "argument" in message
        )

    def _call_refresh_pick_session(
        session: dict[str, object],
        *,
        reuse_only: bool,
    ) -> dict[str, object] | None:
        if not callable(refresh_pick_session):
            return session
        capabilities = _refresh_pick_session_signature_capabilities()
        if capabilities is not None:
            accepts_session, accepts_reuse_only = capabilities
            if accepts_session and accepts_reuse_only:
                return refresh_pick_session(session, reuse_only=bool(reuse_only))
            if accepts_session:
                return refresh_pick_session(session)
            if accepts_reuse_only:
                return refresh_pick_session(reuse_only=bool(reuse_only))
            return refresh_pick_session()
        try:
            return refresh_pick_session(session, reuse_only=bool(reuse_only))
        except TypeError as exc:
            if not _refresh_pick_session_type_error_is_arity_mismatch(exc):
                raise
            try:
                return refresh_pick_session(session)
            except TypeError as one_arg_exc:
                if not _refresh_pick_session_type_error_is_arity_mismatch(one_arg_exc):
                    raise
                return refresh_pick_session()

    def _pick_session(*, click_path: bool = False) -> dict[str, object] | None:
        session = _resolve_runtime_value(pick_session)
        if not isinstance(session, dict):
            return None
        if callable(refresh_pick_session):
            refreshed = _call_refresh_pick_session(
                session,
                reuse_only=bool(click_path),
            )
            if isinstance(refreshed, dict):
                session = refreshed
        return session

    def _use_caked_space() -> bool:
        return bool(_resolve_runtime_value(use_caked_space))

    def _profile_cache() -> Mapping[str, object] | None:
        cache = _resolve_runtime_value(profile_cache)
        return cache if isinstance(cache, Mapping) else None

    def _render_current_pairs(
        *,
        update_status: bool = False,
        click_path: bool = False,
    ) -> bool:
        return render_current_geometry_manual_pairs(
            background_visible=bool(_resolve_runtime_value(background_visible)),
            current_background_index=_background_index(),
            current_background_image=_background_image(),
            pick_session=_pick_session(click_path=bool(click_path)),
            build_initial_pairs_display=build_initial_pairs_display,
            session_initial_pairs_display=session_initial_pairs_display,
            clear_geometry_pick_artists=clear_geometry_pick_artists,
            draw_initial_geometry_pairs_overlay=draw_initial_geometry_pairs_overlay,
            update_button_label_fn=update_button_label,
            set_background_file_status_text_fn=set_background_file_status_text,
            pair_group_count=pair_group_count,
            set_status_text=set_status_text,
            update_status=update_status,
            reuse_only=bool(click_path),
        )

    def _toggle_selection_at(
        col: float,
        row: float,
        *,
        saved_pair_move_only: bool = False,
    ) -> bool:
        print_geometry_manual_live_code_path_stamp()
        handled, _next_session, suppress_drag = geometry_manual_toggle_selection_at(
            float(col),
            float(row),
            pick_session=_pick_session(click_path=True),
            current_background_index=_background_index(),
            display_background=_background_image(),
            get_cache_data=get_cache_data,
            pairs_for_index=pairs_for_index,
            set_pairs_for_index_fn=set_pairs_for_index,
            set_pick_session_fn=set_pick_session,
            restore_view_fn=restore_view,
            clear_preview_artists_fn=clear_preview_artists,
            render_current_pairs_fn=(
                lambda **kwargs: _render_current_pairs(click_path=True, **kwargs)
            ),
            update_button_label_fn=update_button_label,
            set_status_text=set_status_text,
            push_undo_state_fn=push_undo_state,
            listed_q_group_entries=listed_q_group_entries,
            format_q_group_line=format_q_group_line,
            use_caked_space=_use_caked_space(),
            pick_search_window_px=float(pick_search_window_px),
            caked_search_tth_deg=float(caked_search_tth_deg),
            caked_search_phi_deg=float(caked_search_phi_deg),
            find_peak_record_for_click_fn=find_peak_record_for_click,
            profile_cache=_profile_cache(),
            saved_pair_move_only=bool(saved_pair_move_only),
        )
        if callable(set_suppress_drag_press_once):
            set_suppress_drag_press_once(bool(suppress_drag))
        if callable(sync_peak_selection_state):
            sync_peak_selection_state()
        return bool(handled)

    def _place_selection_at(col: float, row: float) -> bool:
        handled, _next_session = geometry_manual_place_selection_at(
            float(col),
            float(row),
            pick_session=_pick_session(click_path=True),
            current_background_index=_background_index(),
            display_background=_background_image(),
            get_cache_data=get_cache_data,
            refine_preview_point=refine_preview_point,
            set_pairs_for_index_fn=set_pairs_for_index,
            set_pick_session_fn=set_pick_session,
            clear_preview_artists_fn=clear_preview_artists,
            restore_view_fn=restore_view,
            render_current_pairs_fn=(
                lambda **kwargs: _render_current_pairs(click_path=True, **kwargs)
            ),
            update_button_label_fn=update_button_label,
            set_status_text=set_status_text,
            push_undo_state_fn=push_undo_state,
            use_caked_space=_use_caked_space(),
            caked_angles_to_background_display_coords=(caked_angles_to_background_display_coords),
            background_display_to_native_detector_coords=(
                background_display_to_native_detector_coords
            ),
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            radial_axis=np.asarray(
                _resolve_runtime_value(last_caked_radial_values),
                dtype=float,
            ),
            azimuth_axis=np.asarray(
                _resolve_runtime_value(last_caked_azimuth_values),
                dtype=float,
            ),
            caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
            caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
            nearest_candidate_to_point_fn=nearest_candidate_to_point,
            position_error_px=position_error_px,
            position_sigma_px=position_sigma_px,
            refine_saved_pair_entry_fn=refine_saved_pair_entry,
            profile_cache=_profile_cache(),
        )
        return bool(handled)

    def _update_pick_preview(raw_col: float, raw_row: float, *, force: bool = False) -> None:
        preview_state = geometry_manual_pick_preview_state(
            float(raw_col),
            float(raw_row),
            pick_session=_pick_session(),
            current_background_index=_background_index(),
            force=bool(force),
            remaining_candidates=(
                list(remaining_candidates()) if callable(remaining_candidates) else []
            ),
            display_background=_background_image(),
            build_cache_data=(get_cache_data if callable(get_cache_data) else None),
            refine_preview_point=refine_preview_point,
            preview_due=preview_due,
            nearest_candidate_to_point=nearest_candidate_to_point,
            position_error_px=position_error_px,
            position_sigma_px=position_sigma_px,
            use_caked_space=_use_caked_space(),
            caked_angles_to_background_display_coords=(caked_angles_to_background_display_coords),
            radial_axis=np.asarray(
                _resolve_runtime_value(last_caked_radial_values),
                dtype=float,
            ),
            azimuth_axis=np.asarray(
                _resolve_runtime_value(last_caked_azimuth_values),
                dtype=float,
            ),
            profile_cache=_profile_cache(),
            caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
            caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
        )
        if preview_state is None:
            return
        if callable(show_preview):
            show_preview(
                float(preview_state["raw_col"]),
                float(preview_state["raw_row"]),
                float(preview_state["refined_col"]),
                float(preview_state["refined_row"]),
                delta_px=float(preview_state["delta"]),
                sigma_px=float(preview_state["sigma_px"]),
                preview_color=str(preview_state["preview_color"]),
            )
        if callable(set_status_text):
            set_status_text(str(preview_state["message"]))

    def _cancel_pick_session(
        *,
        restore_view: bool = True,
        redraw: bool = True,
        message: str | None = None,
    ) -> None:
        set_pick_session(
            cancel_geometry_manual_pick_session(
                _pick_session(),
                current_background_index=_background_index(),
                restore_view_fn=restore_view_callback,
                clear_preview_artists_fn=clear_preview_artists,
                render_current_pairs_fn=_render_current_pairs,
                update_button_label_fn=update_button_label,
                set_status_text=set_status_text,
                restore_view=restore_view,
                redraw=redraw,
                message=message,
                use_caked_space=_use_caked_space(),
            )
        )

    return GeometryManualRuntimeCallbacks(
        render_current_pairs=_render_current_pairs,
        toggle_selection_at=_toggle_selection_at,
        place_selection_at=_place_selection_at,
        update_pick_preview=_update_pick_preview,
        cancel_pick_session=_cancel_pick_session,
    )


def geometry_manual_pick_button_label(
    *,
    armed: bool,
    current_background_index: object,
    pick_session: dict[str, object] | None,
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_group_count: Callable[[int], int],
    base_label: str = "Pick Qr Sets on Image",
) -> str:
    """Return the manual-geometry button label for the current GUI state."""

    label = str(base_label)
    if armed:
        label += " (Armed)"
    try:
        pair_count = len(pairs_for_index(int(current_background_index)))
        group_count = pair_group_count(int(current_background_index))
    except Exception:
        pair_count = 0
        group_count = 0
    if group_count > 0 or pair_count > 0:
        label += f" [{group_count} groups/{pair_count} pts]"
    if geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        pending_entries = (
            pick_session.get("pending_entries", []) if isinstance(pick_session, dict) else []
        )
        target_count = pick_session.get("target_count") if isinstance(pick_session, dict) else None
        if isinstance(pending_entries, list):
            try:
                total_count = int(target_count)
            except Exception:
                total_count = len(
                    pick_session.get("group_entries", []) if isinstance(pick_session, dict) else []
                )
            label += f" <placing {len(pending_entries)}/{max(0, total_count)}>"
    return label


def geometry_manual_toggle_selection_at(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: int,
    display_background: object | None,
    get_cache_data: Callable[..., dict[str, object]],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    set_pairs_for_index_fn: Callable[
        [int, Sequence[dict[str, object]] | None], Sequence[dict[str, object]]
    ],
    set_pick_session_fn: Callable[[dict[str, object]], None],
    restore_view_fn: Callable[..., None],
    clear_preview_artists_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    push_undo_state_fn: Callable[[], None] | None = None,
    listed_q_group_entries: Callable[[], Sequence[dict[str, object]]]
    | Sequence[dict[str, object]] = (),
    format_q_group_line: Callable[[dict[str, object]], str] | None = None,
    choose_group_at_fn: Callable[
        ..., tuple[tuple[object, ...] | None, list[dict[str, object]], float]
    ] = geometry_manual_choose_group_at,
    nearest_candidate_to_point_fn: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    find_peak_record_for_click_fn: Callable[
        [float, float, float],
        tuple[int, dict[str, object] | None, float, bool],
    ]
    | None = None,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    group_target_count_fn: Callable[
        [tuple[object, ...] | None, Sequence[dict[str, object]] | None],
        int,
    ] = geometry_manual_group_target_count,
    use_caked_space: bool,
    pick_search_window_px: float,
    caked_search_tth_deg: float = DEFAULT_CAKED_SEARCH_TTH_DEG,
    caked_search_phi_deg: float = DEFAULT_CAKED_SEARCH_PHI_DEG,
    profile_cache: Mapping[str, object] | None = None,
    trace_picker_input_fn: Callable[[dict[str, object]], None] | None = None,
    saved_pair_move_only: bool = False,
) -> tuple[bool, dict[str, object], bool]:
    """Select one manual Qr/Qz group and arm background-point placement."""

    current_session = dict(pick_session) if isinstance(pick_session, dict) else {}
    if display_background is None:
        if callable(set_status_text):
            set_status_text("No background image is loaded for manual geometry picking.")
        return False, current_session, False

    cache_data = get_cache_data(
        background_image=display_background,
        reuse_only=True,
        build_caked_projection_sidecar=bool(use_caked_space),
    )
    picker_trace: dict[str, object] | None = None
    existing_entries = [dict(entry) for entry in pairs_for_index(int(current_background_index))]
    caked_projection_fallback_note = ""
    picker_uses_caked_space = bool(use_caked_space)
    if picker_uses_caked_space:
        caked_grouped_candidates = (
            cache_data.get("caked_qr_projection_grouped_candidates")
            if isinstance(cache_data, Mapping)
            else None
        )
        if isinstance(caked_grouped_candidates, Mapping) and caked_grouped_candidates:
            grouped_candidates = dict(caked_grouped_candidates)
        else:
            cache_metadata = (
                cache_data.get("cache_metadata")
                if isinstance(cache_data, Mapping)
                and isinstance(cache_data.get("cache_metadata"), Mapping)
                else {}
            )
            caked_authoritative = bool(
                isinstance(caked_grouped_candidates, Mapping)
                and not caked_grouped_candidates
                and (
                    cache_metadata.get("caked_qr_projection_authoritative") is True
                    or cache_metadata.get("exact_caked_projection_authoritative") is True
                )
                and (
                    cache_metadata.get("caked_qr_projection_suppress_detector_fallback")
                    is True
                    or cache_metadata.get("exact_caked_projection_suppress_detector_fallback")
                    is True
                    or cache_metadata.get("suppress_detector_fallback") is True
                )
            )
            detector_grouped_candidates = (
                {}
                if caked_authoritative
                else geometry_manual_detector_picker_grouped_candidates_from_cache(
                    cache_data,
                    display_background=display_background,
                    profile_cache=profile_cache,
                )
            )
            if detector_grouped_candidates:
                caked_projection_fallback_note = (
                    "Exact caked Qr projection unavailable; using detector-space Qr picking. "
                )
                grouped_candidates = detector_grouped_candidates
                picker_trace = geometry_manual_detector_picker_input_trace(
                    cache_data,
                    view_mode="detector",
                    background_index=int(current_background_index),
                    display_background=display_background,
                    grouped_candidates=grouped_candidates,
                    profile_cache=profile_cache,
                )
                if callable(trace_picker_input_fn):
                    trace_picker_input_fn(dict(picker_trace))
                picker_uses_caked_space = False
            elif (
                isinstance(cache_data, Mapping)
                and "caked_qr_projection_grouped_candidates" in cache_data
            ):
                grouped_candidates = {}
            else:
                caked_projection_fallback_note = (
                    "Exact caked Qr projection unavailable; using detector-space Qr picking. "
                )
                grouped_candidates = detector_grouped_candidates
                picker_trace = geometry_manual_detector_picker_input_trace(
                    cache_data,
                    view_mode="detector",
                    background_index=int(current_background_index),
                    display_background=display_background,
                    grouped_candidates=grouped_candidates,
                    profile_cache=profile_cache,
                )
                if callable(trace_picker_input_fn):
                    trace_picker_input_fn(dict(picker_trace))
    else:
        grouped_candidates = geometry_manual_detector_picker_grouped_candidates_from_cache(
            cache_data,
            display_background=display_background,
            profile_cache=profile_cache,
        )
        picker_trace = geometry_manual_detector_picker_input_trace(
            cache_data,
            view_mode="detector",
            background_index=int(current_background_index),
            display_background=display_background,
            grouped_candidates=grouped_candidates,
            profile_cache=profile_cache,
        )
        if callable(trace_picker_input_fn):
            trace_picker_input_fn(dict(picker_trace))
    use_caked_space = bool(picker_uses_caked_space)
    group_window = (
        float(pick_search_window_px)
        if not use_caked_space
        else float(max(caked_search_phi_deg, 2.0 * caked_search_tth_deg))
    )
    move_window = (
        float(group_window)
        if bool(use_caked_space)
        else float(max(4.0, min(float(group_window), 8.0)))
    )
    move_index, move_entry, move_dist = _geometry_manual_saved_background_pair_at(
        existing_entries,
        float(col),
        float(row),
        use_caked_display=bool(use_caked_space),
        window_size=float(move_window),
    )
    if bool(saved_pair_move_only) and move_index is not None and isinstance(move_entry, dict):
        move_group_key = _geometry_manual_real_q_group_key(move_entry)
        if move_group_key is not None:
            q_entries = (
                listed_q_group_entries()
                if callable(listed_q_group_entries)
                else listed_q_group_entries
            )
            q_entry = next(
                (
                    entry
                    for entry in q_entries
                    if isinstance(entry, dict) and entry.get("key") == move_group_key
                ),
                None,
            )
            q_label = (
                format_q_group_line(q_entry)
                if isinstance(q_entry, dict) and callable(format_q_group_line)
                else f"group={move_group_key}"
            )
            manual_run_id = geometry_manual_start_run_id()
            tagged_candidate_key = candidate_source_key(move_entry)
            next_session = {
                "manual_geometry_run_id": manual_run_id,
                "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
                "background_index": int(current_background_index),
                "group_key": move_group_key,
                "group_entries": [dict(move_entry)],
                "pending_entries": [],
                "target_count": 1,
                "base_entries": [
                    dict(entry)
                    for idx, entry in enumerate(existing_entries)
                    if int(idx) != int(move_index)
                ],
                "cache_signature": (
                    cache_data.get("signature") if isinstance(cache_data, dict) else None
                ),
                "q_label": q_label,
                "zoom_active": False,
                "zoom_center": None,
                "saved_xlim": None,
                "saved_ylim": None,
                "preview_last_t": 0.0,
                "preview_last_xy": None,
                "moving_saved_pair": True,
                "moving_saved_pair_index": int(move_index),
            }
            if tagged_candidate_key is not None:
                next_session["tagged_candidate_key"] = tagged_candidate_key
            next_session["tagged_candidate"] = dict(move_entry)
            next_session["_tagged_candidate_requires_identity"] = True
            set_pick_session_fn(next_session)
            render_current_pairs_fn(update_status=False)
            update_button_label_fn()
            if callable(set_status_text):
                label = str(move_entry.get("label", "")).strip()
                label_text = f" [{label}]" if label else ""
                set_status_text(
                    f"Move saved Qr/Qz background point{label_text} for {q_label} "
                    f"on background {int(current_background_index) + 1}. "
                    f"Click the new local peak; it will refine locally before saving. "
                    f"Hit distance={float(move_dist):.2f}{' deg' if use_caked_space else 'px'}."
                )
            return True, next_session, not bool(saved_pair_move_only)

    if bool(saved_pair_move_only):
        return False, current_session, False

    cache_ready = not (isinstance(cache_data, Mapping) and cache_data.get("cache_ready") is False)
    if (
        not grouped_candidates
        and not bool(use_caked_space)
        and not bool(saved_pair_move_only)
        and not cache_ready
        and not geometry_manual_pick_session_active(
            current_session,
            current_background_index=current_background_index,
            require_current_background=True,
        )
    ):
        cache_data = get_cache_data(
            background_image=display_background,
            prefer_cache=True,
            reuse_only=False,
            build_caked_projection_sidecar=False,
            picker_candidates_only=True,
        )
        grouped_candidates = geometry_manual_detector_picker_grouped_candidates_from_cache(
            cache_data,
            display_background=display_background,
            profile_cache=profile_cache,
        )
        picker_trace = geometry_manual_detector_picker_input_trace(
            cache_data,
            view_mode="detector",
            background_index=int(current_background_index),
            display_background=display_background,
            grouped_candidates=grouped_candidates,
            profile_cache=profile_cache,
        )
        if callable(trace_picker_input_fn):
            trace_picker_input_fn(dict(picker_trace))
        cache_ready = not (
            isinstance(cache_data, Mapping) and cache_data.get("cache_ready") is False
        )
    if not grouped_candidates:
        if not bool(use_caked_space) and picker_trace is not None:
            print(geometry_manual_format_detector_picker_diagnostic_block(picker_trace))
        if callable(set_status_text):
            if not cache_ready:
                empty_reason = (
                    str(picker_trace.get("reason_candidates_are_empty", ""))
                    if isinstance(picker_trace, Mapping)
                    else ""
                )
                if empty_reason == "no_detector_picker_source_rows":
                    set_status_text(
                        "Manual Qr picker has no detector source rows. "
                        "Update simulation or refresh the Qr/Qz list."
                    )
                else:
                    set_status_text(
                        "Manual Qr picker cache is warming; click again after the current "
                        "simulation update settles."
                    )
            else:
                set_status_text(
                    caked_projection_fallback_note
                    + "No simulated Qr/Qz groups are available to pick. Run a simulation update first."
                )
        return False, current_session, False

    best_group_key = None
    best_group_entries: list[dict[str, object]] = []
    best_dist = float("nan")
    shared_tagged_candidate = None
    shared_seed_dist = float("nan")

    if not use_caked_space and callable(find_peak_record_for_click_fn):
        try:
            (
                _shared_peak_index,
                shared_peak_record,
                shared_peak_dist,
                shared_within_window,
            ) = find_peak_record_for_click_fn(
                float(col),
                float(row),
                max(0.0, 0.5 * float(group_window)),
            )
        except Exception:
            shared_peak_record = None
            shared_peak_dist = float("nan")
            shared_within_window = False
        if shared_within_window and isinstance(shared_peak_record, Mapping):
            raw_group_key = shared_peak_record.get("q_group_key")
            if isinstance(raw_group_key, list):
                shared_group_key = tuple(raw_group_key)
            elif isinstance(raw_group_key, tuple):
                shared_group_key = raw_group_key
            else:
                shared_group_key = None
            shared_group_entries_raw = (
                grouped_candidates.get(shared_group_key)
                if isinstance(shared_group_key, tuple)
                else None
            )
            if isinstance(shared_group_entries_raw, Sequence):
                shared_group_entries = [
                    dict(entry) for entry in shared_group_entries_raw if isinstance(entry, dict)
                ]
                if shared_group_entries:
                    best_group_key = shared_group_key
                    best_group_entries = list(shared_group_entries)
                    best_dist = float(shared_peak_dist)
                    matched_index = _geometry_manual_resolve_identity_candidate_index(
                        shared_peak_record,
                        shared_group_entries,
                    )
                    if matched_index is not None and 0 <= int(matched_index) < len(
                        shared_group_entries
                    ):
                        shared_tagged_candidate = dict(shared_group_entries[int(matched_index)])
                        shared_seed_dist = float(shared_peak_dist)

    nearest_group_key, nearest_group_entries, nearest_dist = choose_group_at_fn(
        grouped_candidates,
        float(col),
        float(row),
        window_size_px=float(group_window),
        **(
            {"use_caked_display": bool(use_caked_space)}
            if choose_group_at_fn is geometry_manual_choose_group_at
            else {}
        ),
    )
    if nearest_group_key is not None:
        if best_group_key is not None and nearest_group_key != best_group_key:
            shared_tagged_candidate = None
            shared_seed_dist = float("nan")
        best_group_key = nearest_group_key
        best_group_entries = list(nearest_group_entries)
        best_dist = float(nearest_dist)
    if best_group_key is None:
        if callable(set_status_text):
            set_status_text(
                caked_projection_fallback_note
                + (
                    "No Qr/Qz set found within a "
                    f"{group_window:.1f}x"
                    f"{group_window:.1f}{' deg' if use_caked_space else 'px'} "
                    "window around the clicked position."
                )
            )
        return False, current_session, False

    raw_best_group_entries = [
        {**dict(entry), "q_group_key": best_group_key}
        for entry in best_group_entries
        if isinstance(entry, dict)
    ]
    best_group_entries = _geometry_manual_collapse_q_group_representatives(
        raw_best_group_entries,
        profile_cache=profile_cache,
    )
    if not best_group_entries:
        if callable(set_status_text):
            set_status_text(
                "No simulated Qr/Qz branches remain available for manual geometry picking."
            )
        return False, current_session, False

    existing_group_keys = {
        _geometry_manual_real_q_group_key(entry)
        for entry in existing_entries
        if isinstance(entry, Mapping)
    }
    if best_group_key in existing_group_keys:
        restore_view_fn(redraw=False)
        clear_preview_artists_fn(redraw=False)
        set_pick_session_fn({})
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text(
                "That Qr/Qz set is already selected on this background. "
                "Use Click Remove Placed Peaks to delete it, or drag a saved peak to move it."
            )
        return True, {}, False

    q_entries = (
        listed_q_group_entries() if callable(listed_q_group_entries) else listed_q_group_entries
    )
    q_entry = next(
        (
            entry
            for entry in q_entries
            if isinstance(entry, dict) and entry.get("key") == best_group_key
        ),
        None,
    )
    q_label = (
        format_q_group_line(q_entry)
        if isinstance(q_entry, dict) and callable(format_q_group_line)
        else f"group={best_group_key}"
    )
    target_count = group_target_count_fn(
        best_group_key,
        best_group_entries,
    )
    seed_candidate: dict[str, object] | None = None
    seed_dist = float("nan")
    if isinstance(shared_tagged_candidate, dict):
        seed_candidate = dict(shared_tagged_candidate)
        seed_dist = float(shared_seed_dist)
    elif nearest_candidate_to_point_fn is geometry_manual_nearest_candidate_to_point:
        seed_candidate, seed_dist = nearest_candidate_to_point_fn(
            float(col),
            float(row),
            raw_best_group_entries,
            use_caked_display=use_caked_space,
        )
    else:
        seed_candidate, seed_dist = nearest_candidate_to_point_fn(
            float(col),
            float(row),
            raw_best_group_entries,
        )
    tagged_candidate = dict(seed_candidate) if isinstance(seed_candidate, dict) else None
    if isinstance(seed_candidate, dict):
        selected_candidate = _geometry_manual_select_q_group_representative(
            best_group_entries,
            group_key=best_group_key,
            seed_candidate=seed_candidate,
            profile_cache=profile_cache,
        )
        if isinstance(selected_candidate, dict) and not bool(use_caked_space):
            tagged_candidate = selected_candidate
    if bool(use_caked_space) and isinstance(tagged_candidate, dict):
        best_group_entries = _geometry_manual_include_tagged_q_group_candidate(
            best_group_entries,
            tagged_candidate,
            group_key=best_group_key,
            profile_cache=profile_cache,
        )
    tagged_dist = (
        geometry_manual_candidate_distance_to_point(
            float(col),
            float(row),
            tagged_candidate,
            use_caked_display=use_caked_space,
        )
        if isinstance(tagged_candidate, dict)
        else float("nan")
    )
    if not np.isfinite(float(tagged_dist)) and np.isfinite(float(seed_dist)):
        tagged_dist = float(seed_dist)
    tagged_group_entries = geometry_manual_prioritize_candidate_entries(
        best_group_entries,
        tagged_candidate,
        candidate_source_key=candidate_source_key,
    )
    tagged_candidate_key = candidate_source_key(tagged_candidate)
    manual_run_id = geometry_manual_start_run_id()
    tagged_label = (
        str(tagged_candidate.get("label", "")) if isinstance(tagged_candidate, dict) else ""
    )
    next_session = {
        "manual_geometry_run_id": manual_run_id,
        "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
        "background_index": int(current_background_index),
        "group_key": best_group_key,
        "group_entries": [dict(entry) for entry in tagged_group_entries],
        "pending_entries": [],
        "target_count": int(target_count),
        "base_entries": [
            entry for entry in existing_entries if entry.get("q_group_key") != best_group_key
        ],
        "cache_signature": cache_data.get("signature") if isinstance(cache_data, dict) else None,
        "q_label": q_label,
        "zoom_active": False,
        "zoom_center": None,
        "saved_xlim": None,
        "saved_ylim": None,
        "preview_last_t": 0.0,
        "preview_last_xy": None,
    }
    if tagged_candidate_key is not None:
        next_session["tagged_candidate_key"] = tagged_candidate_key
    if isinstance(tagged_candidate, dict):
        next_session["tagged_candidate"] = dict(tagged_candidate)
        next_session["_tagged_candidate_requires_identity"] = True
    set_pick_session_fn(next_session)
    if use_caked_space and geometry_manual_live_caked_trace_enabled():
        geometry_manual_trace_live_caked_visual_source_event(
            "caked_qr_simulation_selection",
            manual_geometry_run_id=manual_run_id,
            selected_click_caked_deg=(float(col), float(row)),
            selected_candidate=tagged_candidate,
            pending_entries=tagged_group_entries,
        )
        geometry_manual_trace_live_caked_visual_source_event(
            "pending_visual_map_built",
            manual_geometry_run_id=manual_run_id,
            selected_click_caked_deg=(float(col), float(row)),
            selected_candidate=tagged_candidate,
            pending_entries=tagged_group_entries,
        )
    render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    if callable(set_status_text):
        seed_dist = (
            float(shared_seed_dist)
            if np.isfinite(float(shared_seed_dist))
            else tagged_dist
            if np.isfinite(tagged_dist)
            else best_dist
        )
        set_status_text(
            caked_projection_fallback_note
            + f"Selected {q_label} (nearest Bragg seed {seed_dist:.1f}{' deg' if use_caked_space else 'px'}). "
            + geometry_manual_cmd_provenance_text(
                run_id=manual_run_id,
                emitter="geometry_manual_toggle_selection_at",
                event="manual_geometry_select_group",
                branch=(
                    _geometry_manual_trace_branch(tagged_candidate)
                    if isinstance(tagged_candidate, Mapping)
                    else "<none>"
                ),
                actual_source="tagged_seed" if tagged_label else "nearest_seed",
                expected_source=(
                    "sim_visual_caked_deg"
                    if bool(use_caked_space)
                    else "sim_visual_detector_display_px"
                ),
                **_geometry_manual_source_consistency_fields(
                    selected_group_key=best_group_key,
                    candidate=tagged_candidate if isinstance(tagged_candidate, Mapping) else None,
                ),
            )
            + " "
            + (f"Tagged seed [{tagged_label}]. " if tagged_label else "")
            + f"Click background peak 1 of {max(1, int(target_count))}; "
            + (
                "it will attach to that tagged simulated peak."
                if tagged_label
                else "it will be assigned to the nearest simulated peak."
            )
        )
    return True, next_session, True


def geometry_manual_place_selection_at(
    col: float,
    row: float,
    *,
    pick_session: dict[str, object] | None,
    current_background_index: object,
    display_background: object | None,
    get_cache_data: Callable[..., dict[str, object]],
    refine_preview_point: Callable[..., tuple[float, float]],
    set_pairs_for_index_fn: Callable[
        [int, Sequence[dict[str, object]] | None], Sequence[dict[str, object]]
    ],
    set_pick_session_fn: Callable[[dict[str, object]], None],
    clear_preview_artists_fn: Callable[..., None],
    restore_view_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    push_undo_state_fn: Callable[[], None] | None = None,
    use_caked_space: bool,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    resolve_background_pick_fn: Callable[..., dict[str, float] | None]
    | None = resolve_background_pick_to_caked_angles,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    nearest_candidate_to_point_fn: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    pair_entry_from_candidate_fn: Callable[
        ..., dict[str, object] | None
    ] = geometry_manual_pair_entry_from_candidate,
    position_error_px: Callable[
        [float, float, float, float], float
    ] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    refine_saved_pair_entry_fn: Callable[
        [dict[str, object], dict[str, object] | None],
        dict[str, object] | None,
    ]
    | None = None,
    profile_cache: Mapping[str, object] | None = None,
    current_match_config: Callable[[], Mapping[str, object]] | Mapping[str, object] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Record the next manual background point for the active Qr/Qz pick session."""

    current_session = dict(pick_session) if isinstance(pick_session, dict) else {}
    if not geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
    ):
        return False, current_session
    if display_background is None:
        if callable(set_status_text):
            set_status_text("No background image is loaded for manual geometry picking.")
        return False, current_session
    manual_run_id = _geometry_manual_session_run_id(pick_session)

    if geometry_manual_pick_session_is_background_qr_reference(current_session):
        cache_data = geometry_manual_background_reference_refine_cache(
            display_background,
            get_cache_data=get_cache_data,
            current_match_config=current_match_config,
        )
        resolved_background_pick = None
        if callable(resolve_background_pick_fn):
            try:
                resolved_background_pick = resolve_background_pick_fn(
                    None,
                    float(col),
                    float(row),
                    active_view="caked" if use_caked_space else "detector",
                    display_background=display_background,
                    cache_data=cache_data,
                    refine_detector_pick_fn=refine_preview_point,
                    caked_angles_to_background_display_coords=(
                        caked_angles_to_background_display_coords
                    ),
                    background_display_to_native_detector_coords=(
                        background_display_to_native_detector_coords
                    ),
                    native_detector_coords_to_caked_display_coords=(
                        native_detector_coords_to_caked_display_coords
                    ),
                    radial_axis=radial_axis,
                    azimuth_axis=azimuth_axis,
                    caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
                    caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
                )
            except Exception:
                resolved_background_pick = None

        def _finite_pair(value: object) -> tuple[float, float] | None:
            if isinstance(value, (str, bytes)):
                return None
            try:
                if len(value) < 2:  # type: ignore[arg-type]
                    return None
                x_value = float(value[0])  # type: ignore[index]
                y_value = float(value[1])  # type: ignore[index]
            except Exception:
                return None
            if not (np.isfinite(x_value) and np.isfinite(y_value)):
                return None
            return float(x_value), float(y_value)

        peak_display: tuple[float, float] | None = None
        raw_display: tuple[float, float] | None = None
        detector_native: tuple[float, float] | None = None
        caked_point: tuple[float, float] | None = None
        raw_caked_point: tuple[float, float] | None = None
        if isinstance(resolved_background_pick, dict):
            peak_display = (
                float(resolved_background_pick["refined_detector_display_col"]),
                float(resolved_background_pick["refined_detector_display_row"]),
            )
            detector_native = (
                float(resolved_background_pick["refined_detector_native_col"]),
                float(resolved_background_pick["refined_detector_native_row"]),
            )
            caked_point = (
                float(resolved_background_pick["refined_background_two_theta_deg"]),
                float(resolved_background_pick["refined_background_phi_deg"]),
            )
            raw_caked_point = (
                float(
                    resolved_background_pick.get(
                        "raw_caked_two_theta_deg",
                        resolved_background_pick["refined_background_two_theta_deg"],
                    )
                ),
                float(
                    resolved_background_pick.get(
                        "raw_caked_phi_deg",
                        resolved_background_pick["refined_background_phi_deg"],
                    )
                ),
            )
            if "detector_seed_col" in resolved_background_pick:
                raw_display = (
                    float(resolved_background_pick["detector_seed_col"]),
                    float(resolved_background_pick["detector_seed_row"]),
                )
            elif callable(caked_angles_to_background_display_coords):
                raw_display = _finite_pair(
                    caked_angles_to_background_display_coords(
                        float(raw_caked_point[0]),
                        float(raw_caked_point[1]),
                    )
                )
            if raw_display is None:
                raw_display = peak_display
        elif use_caked_space:
            refined_caked_col, refined_caked_row = refine_preview_point(
                None,
                float(col),
                float(row),
                display_background=display_background,
                cache_data=cache_data,
                allow_cache_build=False,
            )
            if np.isfinite(float(refined_caked_col)) and np.isfinite(float(refined_caked_row)):
                caked_point = (float(refined_caked_col), float(refined_caked_row))
                raw_caked_point = (float(col), float(row))
                if callable(caked_angles_to_background_display_coords):
                    peak_display = _finite_pair(
                        caked_angles_to_background_display_coords(
                            float(caked_point[0]),
                            float(caked_point[1]),
                        )
                    )
                    raw_display = _finite_pair(
                        caked_angles_to_background_display_coords(float(col), float(row))
                    )
                if peak_display is not None and callable(
                    background_display_to_native_detector_coords
                ):
                    detector_native = _finite_pair(
                        background_display_to_native_detector_coords(
                            float(peak_display[0]),
                            float(peak_display[1]),
                        )
                    )
        else:
            refined_col, refined_row = refine_preview_point(
                None,
                float(col),
                float(row),
                display_background=display_background,
                cache_data=cache_data,
                allow_cache_build=False,
            )
            if np.isfinite(float(refined_col)) and np.isfinite(float(refined_row)):
                peak_display = (float(refined_col), float(refined_row))
                raw_display = (float(col), float(row))
                if callable(background_display_to_native_detector_coords):
                    detector_native = _finite_pair(
                        background_display_to_native_detector_coords(
                            float(peak_display[0]),
                            float(peak_display[1]),
                        )
                    )
                    raw_detector_native = _finite_pair(
                        background_display_to_native_detector_coords(float(col), float(row))
                    )
                    if raw_detector_native is not None and callable(
                        native_detector_coords_to_caked_display_coords
                    ):
                        raw_caked_point = _finite_pair(
                            native_detector_coords_to_caked_display_coords(
                                float(raw_detector_native[0]),
                                float(raw_detector_native[1]),
                            )
                        )
                if detector_native is not None and callable(
                    native_detector_coords_to_caked_display_coords
                ):
                    caked_point = _finite_pair(
                        native_detector_coords_to_caked_display_coords(
                            float(detector_native[0]),
                            float(detector_native[1]),
                        )
                    )

        if peak_display is None or caked_point is None:
            if callable(set_status_text):
                set_status_text(
                    "Background Qr reference picking could not resolve that background peak to 2theta/phi."
                )
            return False, current_session
        if raw_display is None:
            raw_display = peak_display
        if raw_caked_point is None:
            raw_caked_point = caked_point
        placement_error_px_value = position_error_px(
            float(raw_display[0]),
            float(raw_display[1]),
            float(peak_display[0]),
            float(peak_display[1]),
        )
        sigma_px_value = position_sigma_px(float(placement_error_px_value))
        pair_entry = geometry_manual_background_qr_reference_pair_entry(
            peak_col=float(peak_display[0]),
            peak_row=float(peak_display[1]),
            raw_col=float(raw_display[0]),
            raw_row=float(raw_display[1]),
            detector_col=(float(detector_native[0]) if detector_native is not None else None),
            detector_row=(float(detector_native[1]) if detector_native is not None else None),
            caked_col=float(caked_point[0]),
            caked_row=float(caked_point[1]),
            raw_caked_col=float(raw_caked_point[0]),
            raw_caked_row=float(raw_caked_point[1]),
            placement_error_px=float(placement_error_px_value),
            sigma_px=float(sigma_px_value),
            manual_run_id=manual_run_id,
        )
        _geometry_manual_set_background_input_provenance(
            pair_entry,
            use_caked_space=bool(use_caked_space),
        )
        if callable(push_undo_state_fn):
            push_undo_state_fn()
        base_entries = current_session.get("base_entries", [])
        updated_entries = [dict(entry) for entry in (base_entries or ()) if isinstance(entry, dict)]
        updated_entries.append(pair_entry)
        clear_preview_artists_fn(redraw=False)
        if use_caked_space:
            restore_view_fn(redraw=False)
        set_pick_session_fn({})
        set_pairs_for_index_fn(int(current_background_index), updated_entries)
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text(
                "Saved background Qr reference "
                f"{pair_entry['label']} on background {int(current_background_index) + 1}; "
                f"local peak fit moved {float(placement_error_px_value):.2f}px."
            )
        return True, {}

    remaining_candidates = geometry_manual_unassigned_group_candidates(
        pick_session,
        current_background_index=current_background_index,
        candidate_source_key=candidate_source_key,
    )
    selected_group_key = pick_session.get("group_key") if isinstance(pick_session, dict) else None
    raw_remaining_candidates = [dict(entry) for entry in remaining_candidates or ()]
    remaining_candidates = _geometry_manual_filter_candidates_for_selected_group_source(
        remaining_candidates,
        group_key=selected_group_key,
    )
    if raw_remaining_candidates and not remaining_candidates:
        selected_source = _geometry_manual_entry_source_label(
            None,
            group_key=selected_group_key,
        )
        available_sources = _geometry_manual_available_candidate_sources(raw_remaining_candidates)
        if callable(set_status_text):
            set_status_text(
                "Manual geometry picking source mismatch: "
                "source_mismatch_error=true "
                f"selected_group_source={selected_source} "
                f"available_sources={available_sources}"
            )
        return False, current_session
    if not remaining_candidates:
        if use_caked_space:
            restore_view_fn(redraw=False)
        clear_preview_artists_fn(redraw=False)
        set_pick_session_fn({})
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            set_status_text("Manual geometry picking had no remaining simulated peaks to place.")
        return False, {}

    cache_data = get_cache_data(
        background_image=display_background,
        reuse_only=True,
        build_caked_projection_sidecar=bool(use_caked_space),
    )
    if isinstance(cache_data, Mapping) and cache_data.get("cache_ready") is False:
        cache_data = geometry_manual_no_build_cache_sentinel(
            match_config=(
                cache_data.get("match_config", {})
                if isinstance(cache_data.get("match_config"), Mapping)
                else {}
            ),
            stale_reason="manual placement uses local refinement without warmed picker cache",
        )
    tagged_candidate = geometry_manual_tagged_candidate_from_session(
        pick_session,
        remaining_candidates,
        candidate_source_key=candidate_source_key,
    )
    seed_candidate = tagged_candidate
    if seed_candidate is None:
        if nearest_candidate_to_point_fn is geometry_manual_nearest_candidate_to_point:
            seed_candidate, _seed_dist = nearest_candidate_to_point_fn(
                float(col),
                float(row),
                remaining_candidates,
                use_caked_display=use_caked_space,
            )
        else:
            seed_candidate, _seed_dist = nearest_candidate_to_point_fn(
                float(col),
                float(row),
                remaining_candidates,
            )
    candidate = dict(seed_candidate) if isinstance(seed_candidate, dict) else None
    if isinstance(seed_candidate, dict):
        selected_candidate = _geometry_manual_select_q_group_representative(
            remaining_candidates,
            group_key=selected_group_key,
            seed_candidate=seed_candidate,
            profile_cache=profile_cache,
        )
        if isinstance(selected_candidate, dict):
            candidate = selected_candidate
    if candidate is None:
        if callable(set_status_text):
            set_status_text(
                "Manual geometry picking could not find an unassigned simulated peak for that background point."
            )
        return False, current_session
    resolved_background_pick = None
    if callable(resolve_background_pick_fn):
        try:
            resolved_background_pick = resolve_background_pick_fn(
                candidate,
                float(col),
                float(row),
                active_view="caked" if use_caked_space else "detector",
                display_background=display_background,
                cache_data=cache_data,
                refine_detector_pick_fn=refine_preview_point,
                caked_angles_to_background_display_coords=(
                    caked_angles_to_background_display_coords
                ),
                background_display_to_native_detector_coords=(
                    background_display_to_native_detector_coords
                ),
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                radial_axis=radial_axis,
                azimuth_axis=azimuth_axis,
                caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
                caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
            )
        except Exception:
            resolved_background_pick = None

    pair_kwargs: dict[str, object]
    detector_col = None
    detector_row = None
    detector_to_caked_unavailable = False
    detector_to_caked_refreshed = False
    placement_error_px_value: float
    if isinstance(resolved_background_pick, dict):
        if use_caked_space:
            refined_branch_candidate = None
            if nearest_candidate_to_point_fn is geometry_manual_nearest_candidate_to_point:
                refined_branch_candidate, _refined_branch_dist = nearest_candidate_to_point_fn(
                    float(resolved_background_pick["refined_background_two_theta_deg"]),
                    float(resolved_background_pick["refined_background_phi_deg"]),
                    remaining_candidates,
                    use_caked_display=True,
                )
            else:
                refined_branch_candidate, _refined_branch_dist = nearest_candidate_to_point_fn(
                    float(resolved_background_pick["refined_background_two_theta_deg"]),
                    float(resolved_background_pick["refined_background_phi_deg"]),
                    remaining_candidates,
                )
            if isinstance(refined_branch_candidate, dict):
                selected_candidate = _geometry_manual_select_q_group_representative(
                    remaining_candidates,
                    group_key=selected_group_key,
                    seed_candidate=refined_branch_candidate,
                    profile_cache=profile_cache,
                )
                candidate = (
                    dict(selected_candidate)
                    if isinstance(selected_candidate, dict)
                    else dict(refined_branch_candidate)
                )
        peak_col = (
            float(resolved_background_pick["refined_background_two_theta_deg"])
            if use_caked_space
            else float(resolved_background_pick["refined_detector_display_col"])
        )
        peak_row = (
            float(resolved_background_pick["refined_background_phi_deg"])
            if use_caked_space
            else float(resolved_background_pick["refined_detector_display_row"])
        )
        candidate_distance_details = geometry_manual_candidate_distance_details(
            float(peak_col),
            float(peak_row),
            candidate,
            use_caked_display=use_caked_space,
        )
        candidate_dist = float(candidate_distance_details.get("distance", float("nan")))
        if use_caked_space:
            placement_error_px_value = position_error_px(
                float(
                    resolved_background_pick.get(
                        "raw_caked_two_theta_deg",
                        resolved_background_pick["refined_background_two_theta_deg"],
                    )
                ),
                float(
                    resolved_background_pick.get(
                        "raw_caked_phi_deg",
                        resolved_background_pick["refined_background_phi_deg"],
                    )
                ),
                float(resolved_background_pick["refined_background_two_theta_deg"]),
                float(resolved_background_pick["refined_background_phi_deg"]),
            )
        else:
            placement_error_px_value = position_error_px(
                float(resolved_background_pick["detector_seed_col"]),
                float(resolved_background_pick["detector_seed_row"]),
                float(resolved_background_pick["refined_detector_display_col"]),
                float(resolved_background_pick["refined_detector_display_row"]),
            )
        pair_kwargs = {
            "peak_col": float(resolved_background_pick["refined_detector_display_col"]),
            "peak_row": float(resolved_background_pick["refined_detector_display_row"]),
            "raw_col": float(
                resolved_background_pick.get(
                    "detector_seed_col",
                    resolved_background_pick["refined_detector_display_col"],
                )
            ),
            "raw_row": float(
                resolved_background_pick.get(
                    "detector_seed_row",
                    resolved_background_pick["refined_detector_display_row"],
                )
            ),
            "caked_col": float(resolved_background_pick["refined_background_two_theta_deg"]),
            "caked_row": float(resolved_background_pick["refined_background_phi_deg"]),
            "raw_caked_col": float(
                resolved_background_pick.get(
                    "raw_caked_two_theta_deg",
                    resolved_background_pick["refined_background_two_theta_deg"],
                )
            ),
            "raw_caked_row": float(
                resolved_background_pick.get(
                    "raw_caked_phi_deg",
                    resolved_background_pick["refined_background_phi_deg"],
                )
            ),
        }
        detector_col = float(resolved_background_pick["refined_detector_native_col"])
        detector_row = float(resolved_background_pick["refined_detector_native_row"])
    else:
        if (
            use_caked_space
            and callable(caked_angles_to_background_display_coords)
            and callable(background_display_to_native_detector_coords)
            and callable(native_detector_coords_to_caked_display_coords)
        ):
            if callable(set_status_text):
                set_status_text(
                    "Manual geometry picking could not resolve the selected caked background point through the detector oracle."
                )
            return False, current_session

        peak_col, peak_row = refine_preview_point(
            candidate,
            float(col),
            float(row),
            display_background=display_background,
            cache_data=cache_data,
            allow_cache_build=False,
        )
        candidate_distance_details = geometry_manual_candidate_distance_details(
            float(peak_col),
            float(peak_row),
            candidate,
            use_caked_display=use_caked_space,
        )
        candidate_dist = float(candidate_distance_details.get("distance", float("nan")))

        pair_kwargs = {
            "peak_col": float(peak_col),
            "peak_row": float(peak_row),
            "raw_col": float(col),
            "raw_row": float(row),
        }
        placement_error_px_value = position_error_px(
            float(col),
            float(row),
            float(peak_col),
            float(peak_row),
        )
        if use_caked_space:
            raw_display = (
                caked_angles_to_background_display_coords(float(col), float(row))
                if callable(caked_angles_to_background_display_coords)
                else (None, None)
            )
            peak_display = (
                caked_angles_to_background_display_coords(float(peak_col), float(peak_row))
                if callable(caked_angles_to_background_display_coords)
                else (None, None)
            )
            if (
                raw_display[0] is None
                or raw_display[1] is None
                or peak_display[0] is None
                or peak_display[1] is None
            ):
                if callable(set_status_text):
                    set_status_text(
                        "Manual geometry picking could not back-project the selected caked peak onto the detector."
                    )
                return False, current_session
            placement_error_px_value = position_error_px(
                float(raw_display[0]),
                float(raw_display[1]),
                float(peak_display[0]),
                float(peak_display[1]),
            )
            pair_kwargs = {
                "peak_col": float(peak_display[0]),
                "peak_row": float(peak_display[1]),
                "raw_col": float(raw_display[0]),
                "raw_row": float(raw_display[1]),
                "caked_col": float(peak_col),
                "caked_row": float(peak_row),
                "raw_caked_col": float(col),
                "raw_caked_row": float(row),
            }

    def _finite_pair(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None
        try:
            x_value = float(value[0])
            y_value = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            return None
        return float(x_value), float(y_value)

    raw_detector_display = (
        float(pair_kwargs["raw_col"]),
        float(pair_kwargs["raw_row"]),
    )
    geometry_detector_display = (
        float(pair_kwargs["peak_col"]),
        float(pair_kwargs["peak_row"]),
    )
    raw_detector_native = _geometry_manual_detector_display_to_native(
        background_display_to_native_detector_coords,
        raw_detector_display,
    )
    geometry_detector_native = None
    if detector_col is not None and detector_row is not None:
        geometry_detector_native = (float(detector_col), float(detector_row))
    else:
        geometry_detector_native = _geometry_manual_detector_display_to_native(
            background_display_to_native_detector_coords,
            geometry_detector_display,
        )
        if geometry_detector_native is not None:
            detector_col = float(geometry_detector_native[0])
            detector_row = float(geometry_detector_native[1])

    if not bool(use_caked_space) and "caked_col" not in pair_kwargs:
        if callable(background_display_to_native_detector_coords) and callable(
            native_detector_coords_to_caked_display_coords
        ):
            geometry_caked = (
                _finite_pair(
                    native_detector_coords_to_caked_display_coords(
                        float(geometry_detector_native[0]),
                        float(geometry_detector_native[1]),
                    )
                )
                if geometry_detector_native is not None
                else None
            )
            raw_caked = (
                _finite_pair(
                    native_detector_coords_to_caked_display_coords(
                        float(raw_detector_native[0]),
                        float(raw_detector_native[1]),
                    )
                )
                if raw_detector_native is not None
                else None
            )
            if geometry_caked is not None:
                pair_kwargs["caked_col"] = float(geometry_caked[0])
                pair_kwargs["caked_row"] = float(geometry_caked[1])
                pair_kwargs["raw_caked_col"] = float(
                    raw_caked[0] if raw_caked is not None else geometry_caked[0]
                )
                pair_kwargs["raw_caked_row"] = float(
                    raw_caked[1] if raw_caked is not None else geometry_caked[1]
                )
                detector_to_caked_refreshed = True
            else:
                detector_to_caked_unavailable = True
        else:
            detector_to_caked_unavailable = True

    status_observed_caked = (
        (float(pair_kwargs["caked_col"]), float(pair_kwargs["caked_row"]))
        if "caked_col" in pair_kwargs and "caked_row" in pair_kwargs
        else None
    )
    status_trace = geometry_manual_status_distance_trace(
        candidate_distance_details,
        observed_refined_caked_deg=status_observed_caked,
        use_caked_space=bool(use_caked_space),
    )
    candidate_branch = _geometry_manual_trace_branch(candidate)
    assignment_source = str(status_trace.get("status_distance_source", "<unavailable>"))
    sigma_px_value = position_sigma_px(float(placement_error_px_value))
    if detector_col is None or detector_row is None:
        detector_anchor = (
            background_display_to_native_detector_coords(
                float(pair_kwargs["peak_col"]),
                float(pair_kwargs["peak_row"]),
            )
            if callable(background_display_to_native_detector_coords)
            else None
        )
        if (
            isinstance(detector_anchor, tuple)
            and len(detector_anchor) >= 2
            and np.isfinite(float(detector_anchor[0]))
            and np.isfinite(float(detector_anchor[1]))
        ):
            detector_col = float(detector_anchor[0])
            detector_row = float(detector_anchor[1])
        elif np.isfinite(float(pair_kwargs["peak_col"])) and np.isfinite(
            float(pair_kwargs["peak_row"])
        ):
            detector_col = float(pair_kwargs["peak_col"])
            detector_row = float(pair_kwargs["peak_row"])
    pair_entry = pair_entry_from_candidate_fn(
        candidate,
        float(pair_kwargs["peak_col"]),
        float(pair_kwargs["peak_row"]),
        group_key=selected_group_key,
        detector_col=detector_col,
        detector_row=detector_row,
        raw_col=float(pair_kwargs["raw_col"]),
        raw_row=float(pair_kwargs["raw_row"]),
        caked_col=(float(pair_kwargs["caked_col"]) if "caked_col" in pair_kwargs else None),
        caked_row=(float(pair_kwargs["caked_row"]) if "caked_row" in pair_kwargs else None),
        raw_caked_col=(
            float(pair_kwargs["raw_caked_col"]) if "raw_caked_col" in pair_kwargs else None
        ),
        raw_caked_row=(
            float(pair_kwargs["raw_caked_row"]) if "raw_caked_row" in pair_kwargs else None
        ),
        placement_error_px=float(placement_error_px_value),
        sigma_px=float(sigma_px_value),
    )
    if pair_entry is None:
        if callable(set_status_text):
            set_status_text("Failed to build the manual geometry pair entry.")
        return False, current_session
    pair_entry["manual_geometry_run_id"] = manual_run_id
    pair_entry["manual_trace_version"] = MANUAL_GEOMETRY_TRACE_VERSION
    _geometry_manual_set_background_input_provenance(
        pair_entry,
        use_caked_space=bool(use_caked_space),
    )
    if (
        bool(use_caked_space)
        and detector_col is not None
        and detector_row is not None
        and (
            isinstance(resolved_background_pick, dict)
            or callable(background_display_to_native_detector_coords)
        )
    ):
        pair_entry["background_detector_x"] = float(detector_col)
        pair_entry["background_detector_y"] = float(detector_row)
        pair_entry["background_detector_input_frame"] = "native_detector"
        pair_entry["background_detector_frame_provenance"] = "caked_inverse_projection_refresh"
    pair_entry["detector_display_x"] = float(geometry_detector_display[0])
    pair_entry["detector_display_y"] = float(geometry_detector_display[1])
    pair_entry["raw_detector_display_px"] = raw_detector_display
    pair_entry["geometry_detector_display_px"] = geometry_detector_display
    if raw_detector_native is not None:
        pair_entry["raw_detector_native_px"] = (
            float(raw_detector_native[0]),
            float(raw_detector_native[1]),
        )
        pair_entry["raw_detector_native_source"] = "display_to_native_callback"
    if detector_col is not None and detector_row is not None:
        pair_entry["detector_native_x"] = float(detector_col)
        pair_entry["detector_native_y"] = float(detector_row)
        pair_entry["geometry_detector_native_px"] = (
            float(detector_col),
            float(detector_row),
        )
        pair_entry["geometry_detector_native_source"] = "saved_native"
    if detector_to_caked_refreshed and detector_col is not None and detector_row is not None:
        pair_entry["background_detector_x"] = float(detector_col)
        pair_entry["background_detector_y"] = float(detector_row)
        pair_entry["background_detector_input_frame"] = "native_detector"
        pair_entry["background_detector_frame_provenance"] = "detector_to_caked_refresh"
    if detector_to_caked_unavailable:
        pair_entry["detector_to_caked_unavailable"] = True
    sim_visual_display, _sim_visual_display_source = (
        _geometry_manual_candidate_visual_detector_sim_point(candidate)
    )
    if not use_caked_space:
        _geometry_manual_apply_sim_visual_detector_fields(
            pair_entry,
            detector_display_to_native_coords=background_display_to_native_detector_coords,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            display_point=sim_visual_display,
        )
    _geometry_manual_apply_detector_origin_conversion_ledger(
        pair_entry,
        background_display_to_native_detector_coords=(background_display_to_native_detector_coords),
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
    )

    def _apply_resolved_background_fields(entry: dict[str, object]) -> None:
        if not isinstance(resolved_background_pick, dict):
            return
        optional_fields = (
            "raw_caked_display_col",
            "raw_caked_display_row",
            "raw_caked_two_theta_deg",
            "raw_caked_phi_deg",
        )
        for field_name in optional_fields:
            if field_name in resolved_background_pick:
                entry[field_name] = float(resolved_background_pick[field_name])
        for field_name in ("detector_seed_col", "detector_seed_row"):
            if field_name in resolved_background_pick:
                entry[field_name] = float(resolved_background_pick[field_name])
        entry["refined_detector_display_col"] = float(
            resolved_background_pick["refined_detector_display_col"]
        )
        entry["refined_detector_display_row"] = float(
            resolved_background_pick["refined_detector_display_row"]
        )
        entry["refined_detector_native_col"] = float(
            resolved_background_pick["refined_detector_native_col"]
        )
        entry["refined_detector_native_row"] = float(
            resolved_background_pick["refined_detector_native_row"]
        )
        entry["refined_background_two_theta_deg"] = float(
            resolved_background_pick["refined_background_two_theta_deg"]
        )
        entry["refined_background_phi_deg"] = float(
            resolved_background_pick["refined_background_phi_deg"]
        )
        entry["background_detector_x"] = float(resolved_background_pick["background_detector_x"])
        entry["background_detector_y"] = float(resolved_background_pick["background_detector_y"])
        entry["background_detector_frame_provenance"] = "reverse_lut_replay_cache"
        entry["background_two_theta_deg"] = float(
            resolved_background_pick["refined_background_two_theta_deg"]
        )
        entry["background_phi_deg"] = float(resolved_background_pick["refined_background_phi_deg"])
        entry["caked_x"] = float(resolved_background_pick["refined_background_two_theta_deg"])
        entry["caked_y"] = float(resolved_background_pick["refined_background_phi_deg"])

    _apply_resolved_background_fields(pair_entry)
    if callable(refine_saved_pair_entry_fn):
        try:
            refined_pair_entry = refine_saved_pair_entry_fn(
                dict(pair_entry),
                dict(candidate) if isinstance(candidate, dict) else None,
            )
        except Exception:
            refined_pair_entry = pair_entry
        if isinstance(refined_pair_entry, dict):
            pair_entry = dict(refined_pair_entry)
            pair_entry["manual_geometry_run_id"] = manual_run_id
            pair_entry["manual_trace_version"] = MANUAL_GEOMETRY_TRACE_VERSION
            _apply_resolved_background_fields(pair_entry)
            _geometry_manual_set_background_input_provenance(
                pair_entry,
                use_caked_space=bool(use_caked_space),
            )
            pair_entry.setdefault("raw_detector_display_px", raw_detector_display)
            pair_entry.setdefault("geometry_detector_display_px", geometry_detector_display)
            if raw_detector_native is not None:
                pair_entry["raw_detector_native_px"] = (
                    float(raw_detector_native[0]),
                    float(raw_detector_native[1]),
                )
                pair_entry["raw_detector_native_source"] = "display_to_native_callback"
            if detector_col is not None and detector_row is not None:
                pair_entry["geometry_detector_native_px"] = (
                    float(detector_col),
                    float(detector_row),
                )
                pair_entry["geometry_detector_native_source"] = "saved_native"
                pair_entry["detector_native_x"] = float(detector_col)
                pair_entry["detector_native_y"] = float(detector_row)
            pair_entry["detector_display_x"] = float(geometry_detector_display[0])
            pair_entry["detector_display_y"] = float(geometry_detector_display[1])
            if (
                detector_to_caked_refreshed
                and detector_col is not None
                and detector_row is not None
            ):
                pair_entry["background_detector_x"] = float(detector_col)
                pair_entry["background_detector_y"] = float(detector_row)
                pair_entry["background_detector_input_frame"] = "native_detector"
                pair_entry["background_detector_frame_provenance"] = "detector_to_caked_refresh"
                if "caked_col" in pair_kwargs and "caked_row" in pair_kwargs:
                    pair_entry["background_two_theta_deg"] = float(pair_kwargs["caked_col"])
                    pair_entry["background_phi_deg"] = float(pair_kwargs["caked_row"])
                    pair_entry["caked_x"] = float(pair_kwargs["caked_col"])
                    pair_entry["caked_y"] = float(pair_kwargs["caked_row"])
                if "raw_caked_col" in pair_kwargs and "raw_caked_row" in pair_kwargs:
                    pair_entry["raw_caked_x"] = float(pair_kwargs["raw_caked_col"])
                    pair_entry["raw_caked_y"] = float(pair_kwargs["raw_caked_row"])
            if detector_to_caked_unavailable:
                pair_entry["detector_to_caked_unavailable"] = True
            if not use_caked_space:
                _geometry_manual_apply_sim_visual_detector_fields(
                    pair_entry,
                    detector_display_to_native_coords=background_display_to_native_detector_coords,
                    native_detector_coords_to_caked_display_coords=(
                        native_detector_coords_to_caked_display_coords
                    ),
                    display_point=sim_visual_display,
                )
            _geometry_manual_apply_detector_origin_conversion_ledger(
                pair_entry,
                background_display_to_native_detector_coords=(
                    background_display_to_native_detector_coords
                ),
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
            )

    if use_caked_space:
        for stale_sim_caked_key in (
            "refined_sim_caked_x",
            "refined_sim_caked_y",
            "sim_refined_caked_deg",
            "sim_caked_display",
            "simulated_two_theta_deg",
            "simulated_phi_deg",
        ):
            pair_entry.pop(stale_sim_caked_key, None)
        if pair_entry.get("sim_visual_caked_deg") is not None:
            pair_entry["sim_visual_source"] = "sim_visual_caked_deg"

    if isinstance(candidate_distance_details, Mapping):
        if np.isfinite(float(candidate_dist)):
            pair_entry["assignment_distance_to_sim"] = float(candidate_dist)
        pair_entry["assignment_distance_source"] = str(
            candidate_distance_details.get("distance_source", "<unavailable>")
        )
        for status_key, status_value in status_trace.items():
            if isinstance(status_value, float) and not np.isfinite(float(status_value)):
                continue
            pair_entry[status_key] = status_value
        try:
            cache_distance = float(candidate_distance_details.get("cache_current_distance", np.nan))
        except Exception:
            cache_distance = float("nan")
        if np.isfinite(cache_distance):
            pair_entry["assignment_distance_to_cache_current_sim"] = float(cache_distance)
        cache_delta = candidate_distance_details.get("cache_current_delta")
        if (
            isinstance(cache_delta, tuple)
            and len(cache_delta) >= 2
            and np.isfinite(float(cache_delta[0]))
            and np.isfinite(float(cache_delta[1]))
        ):
            pair_entry["geometry_minus_sim_cache_current_delta_deg"] = (
                float(cache_delta[0]),
                float(cache_delta[1]),
            )

    if callable(push_undo_state_fn):
        push_undo_state_fn()

    next_session = dict(current_session)
    pending_entries = next_session.get("pending_entries", [])
    if not isinstance(pending_entries, list):
        pending_entries = []
    pending_entries = [dict(entry) for entry in pending_entries if isinstance(entry, dict)]
    pending_entries = _geometry_manual_replace_same_branch_entry(
        pending_entries,
        pair_entry,
        candidate_source_key=candidate_source_key,
        profile_cache=profile_cache,
    )
    next_session["pending_entries"] = pending_entries
    if use_caked_space and geometry_manual_live_caked_trace_enabled():
        placement_branch = _geometry_manual_trace_branch(pair_entry)
        geometry_manual_trace_live_caked_visual_source_event(
            (
                f"placement_branch_{int(placement_branch)}"
                if placement_branch is not None
                else "placement_branch_unknown"
            ),
            manual_geometry_run_id=manual_run_id,
            selected_click_caked_deg=(float(col), float(row)),
            selected_candidate=candidate,
            pending_entries=next_session.get("group_entries", []),
            placement_entry=pair_entry,
            saved_entries=pending_entries,
        )

    try:
        total_count = int(next_session.get("target_count", 0))
    except Exception:
        total_count = 0
    if total_count <= 0:
        group_entries = next_session.get("group_entries", [])
        total_count = (
            len(group_entries) if isinstance(group_entries, list) else len(pending_entries)
        )
    placed_count = len(pending_entries)
    q_label = str(
        next_session.get(
            "q_label",
            next_session.get("group_key", "selected Qr/Qz set"),
        )
    )
    candidate_label = str(candidate.get("label", "")) if isinstance(candidate, dict) else ""

    clear_preview_artists_fn(redraw=False)
    if use_caked_space:
        restore_view_fn(redraw=False)
    next_session["zoom_active"] = False
    next_session["zoom_center"] = None
    next_session["saved_xlim"] = None
    next_session["saved_ylim"] = None

    if placed_count >= total_count:
        base_entries = next_session.get("base_entries", [])
        updated_entries = list(base_entries) if isinstance(base_entries, list) else []
        for pending_entry in pending_entries:
            if not isinstance(pending_entry, dict):
                continue
            updated_entries = _geometry_manual_replace_same_branch_entry(
                updated_entries,
                dict(pending_entry),
                candidate_source_key=candidate_source_key,
                profile_cache=profile_cache,
            )
        if use_caked_space and geometry_manual_live_caked_trace_enabled():
            geometry_manual_trace_live_caked_visual_source_event(
                "completed_group_comparison",
                manual_geometry_run_id=manual_run_id,
                selected_click_caked_deg=(float(col), float(row)),
                selected_candidate=candidate,
                pending_entries=next_session.get("group_entries", []),
                placement_entry=pair_entry,
                saved_entries=updated_entries,
            )
        set_pick_session_fn({})
        set_pairs_for_index_fn(int(current_background_index), updated_entries)
        render_current_pairs_fn(update_status=False)
        update_button_label_fn()
        if callable(set_status_text):
            status_verb = "Moved" if bool(next_session.get("moving_saved_pair")) else "Saved"
            set_status_text(
                f"{status_verb} {placed_count} manual background points for {q_label} "
                + geometry_manual_cmd_provenance_text(
                    run_id=manual_run_id,
                    emitter="geometry_manual_place_selection_at",
                    event=(
                        "moved_saved_point"
                        if bool(next_session.get("moving_saved_pair"))
                        else "saved_group"
                    ),
                    branch=candidate_branch if candidate_branch is not None else "<none>",
                    actual_source=assignment_source,
                    expected_source=_geometry_manual_expected_distance_source(use_caked_space),
                    **_geometry_manual_source_consistency_fields(
                        selected_group_key=selected_group_key,
                        candidate=candidate,
                    ),
                    detector_to_caked_unavailable=(
                        "true" if detector_to_caked_unavailable else None
                    ),
                )
                + " "
                f"on background {int(current_background_index) + 1}. "
                f"Last placement error={float(placement_error_px_value):.2f}px, sigma={float(sigma_px_value):.2f}px. "
                + geometry_manual_format_status_distance_trace(status_trace)
            )
        return True, {}

    set_pick_session_fn(next_session)
    render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    next_index = placed_count + 1
    if callable(set_status_text):
        set_status_text(
            f"Placed peak {placed_count} of {total_count} for {q_label}. "
            + geometry_manual_cmd_provenance_text(
                run_id=manual_run_id,
                emitter="geometry_manual_place_selection_at",
                event="manual_place_selection",
                branch=candidate_branch if candidate_branch is not None else "<none>",
                actual_source=assignment_source,
                expected_source=_geometry_manual_expected_distance_source(use_caked_space),
                **_geometry_manual_source_consistency_fields(
                    selected_group_key=selected_group_key,
                    candidate=candidate,
                ),
                detector_to_caked_unavailable=("true" if detector_to_caked_unavailable else None),
            )
            + " "
            + (
                f"Assigned to {candidate_label}"
                + (
                    f" ({float(candidate_dist):.2f}{' deg' if use_caked_space else 'px'} from sim)."
                    if np.isfinite(candidate_dist)
                    else "."
                )
                + " "
                + geometry_manual_format_status_distance_trace(status_trace)
                if candidate_label
                else ""
            )
            + " "
            + f"Placement error={float(placement_error_px_value):.2f}px, sigma={float(sigma_px_value):.2f}px. "
            + f"Click background peak {next_index} of {total_count}; it will be assigned to the nearest remaining simulated peak."
        )
    return True, next_session


def _geometry_manual_auto_refined_seed_point(
    entry: Mapping[str, object] | None,
    *,
    use_caked_space: bool,
) -> tuple[float, float] | None:
    if bool(use_caked_space):
        refined_caked = _geometry_manual_tuple_point(entry, "sim_refined_caked_deg")
        if refined_caked is None:
            refined_caked = _geometry_manual_finite_point(
                entry,
                (("refined_sim_caked_x", "refined_sim_caked_y"),),
            )
        if refined_caked is not None:
            return refined_caked
        visual_caked, _source = _geometry_manual_candidate_visual_caked_sim_point(entry)
        if visual_caked is not None:
            return visual_caked
        return _geometry_manual_entry_active_view_point(
            entry,
            use_caked_display=True,
        )

    refined_detector = _geometry_manual_tuple_point(
        entry,
        "sim_refined_detector_display_px",
    )
    if refined_detector is None:
        refined_detector = _geometry_manual_finite_point(
            entry,
            (("refined_sim_x", "refined_sim_y"),),
        )
    if refined_detector is not None and not _geometry_manual_detector_point_is_caked(
        entry,
        refined_detector,
    ):
        return refined_detector
    return _geometry_manual_entry_active_view_point(
        entry,
        use_caked_display=False,
    )


def geometry_manual_auto_add_q_group_peaks(
    *,
    current_background_index: object,
    display_background: object | None,
    get_cache_data: Callable[..., dict[str, object]],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    set_pairs_for_index_fn: Callable[
        [int, Sequence[dict[str, object]] | None], Sequence[dict[str, object]]
    ],
    set_pick_session_fn: Callable[[dict[str, object]], None],
    clear_preview_artists_fn: Callable[..., None],
    restore_view_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    refine_preview_point: Callable[..., tuple[float, float]],
    selected_q_group_entries: Sequence[Mapping[str, object]] | None = None,
    selected_q_group_keys: Sequence[object] | None = None,
    format_q_group_line: Callable[[dict[str, object]], str] | None = None,
    set_status_text: Callable[[str], None] | None = None,
    push_undo_state_fn: Callable[[], None] | None = None,
    use_caked_space: bool,
    caked_angles_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    background_display_to_native_detector_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float],
        tuple[float, float] | None,
    ]
    | None = None,
    radial_axis: Sequence[float] | None = None,
    azimuth_axis: Sequence[float] | None = None,
    caked_axis_to_image_index_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_axis_to_image_index,
    caked_image_index_to_axis_fn: Callable[
        [float, Sequence[float] | None], float
    ] = caked_image_index_to_axis,
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
    nearest_candidate_to_point_fn: Callable[
        [float, float, Sequence[dict[str, object]] | None],
        tuple[dict[str, object] | None, float],
    ] = geometry_manual_nearest_candidate_to_point,
    position_error_px: Callable[
        [float, float, float, float], float
    ] = geometry_manual_position_error_px,
    position_sigma_px: Callable[[object], float] = geometry_manual_position_sigma_px,
    refine_saved_pair_entry_fn: Callable[
        [dict[str, object], dict[str, object] | None],
        dict[str, object] | None,
    ]
    | None = None,
    refine_sim_candidate_fn: Callable[..., dict[str, object] | None] | None = None,
    auto_refine_search_radius_px: float | None = None,
    profile_cache: Mapping[str, object] | None = None,
    branch_pair_abs_tol_px: float | None = None,
    branch_pair_rel_tol: float | None = None,
    branch_pair_sigma_px: float | None = None,
    branch_pair_reject: bool = True,
    final_refine_parallel_workers: int | str | None = None,
) -> dict[str, object]:
    """Auto-save manual Qr/Qz placements by refining under simulated group spots."""

    try:
        background_idx = int(current_background_index)
    except Exception:
        background_idx = 0

    if display_background is None:
        if callable(set_status_text):
            set_status_text("No background image is loaded for automatic Qr/Qz placement.")
        return {
            "placed_count": 0,
            "completed_group_count": 0,
            "attempted_group_count": 0,
            "skipped_group_count": 0,
            "branch_pair_restraint_results": [],
        }
    if not callable(refine_preview_point):
        if callable(set_status_text):
            set_status_text("Automatic Qr/Qz placement unavailable: no peak refiner is bound.")
        return {
            "placed_count": 0,
            "completed_group_count": 0,
            "attempted_group_count": 0,
            "skipped_group_count": 0,
            "branch_pair_restraint_results": [],
        }

    try:
        cache_data = get_cache_data(background_image=display_background)
    except Exception:
        cache_data = {}
    if not isinstance(cache_data, Mapping):
        cache_data = {}
    cache_data = geometry_manual_cache_with_auto_refine_search_radius(
        cache_data,
        search_radius_px=auto_refine_search_radius_px,
    )
    normalized_auto_refine_radius = (
        geometry_manual_normalize_auto_refine_search_radius_px(auto_refine_search_radius_px)
        if auto_refine_search_radius_px is not None
        else None
    )

    if bool(use_caked_space):
        raw_grouped = cache_data.get("caked_qr_projection_grouped_candidates")
        grouped_candidates = (
            {
                gui_controllers.normalize_geometry_q_group_key(group_key): [
                    {**dict(entry), "q_group_key": group_key}
                    for entry in (entries or ())
                    if isinstance(entry, Mapping)
                ]
                for group_key, entries in raw_grouped.items()
                if gui_controllers.normalize_geometry_q_group_key(group_key) is not None
            }
            if isinstance(raw_grouped, Mapping)
            else {}
        )
    else:
        grouped_candidates = geometry_manual_detector_picker_grouped_candidates_from_cache(
            cache_data,
            display_background=display_background,
            profile_cache=profile_cache,
        )

    normalized_grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for raw_key, raw_entries in grouped_candidates.items():
        group_key = gui_controllers.normalize_geometry_q_group_key(raw_key)
        if group_key is None:
            continue
        normalized_entries: list[dict[str, object]] = []
        for raw_entry in raw_entries or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            entry["q_group_key"] = group_key
            if _geometry_manual_q_group_entry_is_000(entry, group_key=group_key):
                continue
            normalized_entries.append(entry)
        if normalized_entries:
            normalized_grouped[group_key] = normalized_entries

    selected_entries = [
        dict(entry) for entry in (selected_q_group_entries or ()) if isinstance(entry, Mapping)
    ]
    label_by_key: dict[tuple[object, ...], str] = {}
    ordered_keys: list[tuple[object, ...]] = []

    def _append_group_key(value: object) -> None:
        group_key = gui_controllers.normalize_geometry_q_group_key(value)
        if group_key is None:
            return
        if group_key not in ordered_keys:
            ordered_keys.append(group_key)

    for raw_entry in selected_entries:
        key = raw_entry.get("key", raw_entry.get("q_group_key"))
        group_key = gui_controllers.normalize_geometry_q_group_key(key)
        if group_key is None:
            continue
        if callable(format_q_group_line):
            try:
                label_by_key[group_key] = str(format_q_group_line(dict(raw_entry)))
            except Exception:
                label_by_key[group_key] = f"group={group_key}"
        _append_group_key(group_key)
    for raw_key in selected_q_group_keys or ():
        _append_group_key(raw_key)
    if not ordered_keys:
        ordered_keys = list(normalized_grouped.keys())

    ordered_keys = [key for key in ordered_keys if key in normalized_grouped]
    if not ordered_keys:
        if callable(set_status_text):
            set_status_text(
                "No active Qr/Qz groups are available for automatic placement. "
                "Run a simulation update or update the listed Qr sets first."
            )
        return {
            "placed_count": 0,
            "completed_group_count": 0,
            "attempted_group_count": 0,
            "skipped_group_count": 0,
            "branch_pair_restraint_results": [],
        }

    selected_key_set = set(ordered_keys)

    def _entry_group_key(entry: Mapping[str, object] | None) -> tuple[object, ...] | None:
        if not isinstance(entry, Mapping):
            return None
        return gui_controllers.normalize_geometry_q_group_key(entry.get("q_group_key"))

    initial_entries = [dict(entry) for entry in pairs_for_index(background_idx)]
    working_entries = [
        dict(entry) for entry in initial_entries if _entry_group_key(entry) not in selected_key_set
    ]
    session_state: dict[str, object] = {}

    def _store_session(session: dict[str, object]) -> None:
        nonlocal session_state
        session_state = dict(session or {})

    def _store_pairs(
        _index: int,
        entries: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        nonlocal working_entries
        working_entries = [dict(entry) for entry in (entries or ()) if isinstance(entry, Mapping)]
        return list(working_entries)

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    attempted_groups = 0
    completed_groups = 0
    skipped_groups = 0
    placed_count = 0
    skipped_points = 0
    final_refined_geometry_peaks = 0
    completed_keys: list[tuple[object, ...]] = []
    branch_pair_results: list[dict[str, object]] = []
    final_refine_parallel_worker_count = 1

    def _auto_final_refine_worker_count(item_count: int) -> int:
        if int(item_count) <= 1:
            return 1
        value = final_refine_parallel_workers
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"", "auto"}:
                value = None
            elif text in {"off", "false", "none", "disabled", "serial", "sequential"}:
                return 1
        if value is None:
            try:
                workers = int(default_reserved_cpu_worker_count())
            except Exception:
                workers = 1
        else:
            try:
                workers = int(value)
            except Exception:
                workers = 1
        return int(max(1, min(int(workers), int(item_count), MAX_AUTO_FINAL_REFINE_WORKERS)))

    def _refine_sim_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
        refined_candidate = dict(candidate)
        if callable(refine_sim_candidate_fn):
            try:
                candidate_update = refine_sim_candidate_fn(
                    dict(refined_candidate),
                    use_caked_space=bool(use_caked_space),
                    cache_data=dict(cache_data),
                    auto_refine_search_radius_px=normalized_auto_refine_radius,
                )
            except Exception:
                candidate_update = None
            if isinstance(candidate_update, Mapping):
                refined_candidate.update(dict(candidate_update))
        return refined_candidate

    def _final_geometry_refinement_source_entry(
        saved_entry: Mapping[str, object],
        source_entries: Sequence[Mapping[str, object]],
    ) -> dict[str, object] | None:
        saved_source_key = candidate_source_key(dict(saved_entry))
        for raw_source_entry in source_entries:
            if not isinstance(raw_source_entry, Mapping):
                continue
            try:
                source_key = candidate_source_key(dict(raw_source_entry))
            except Exception:
                source_key = None
            if saved_source_key is not None and source_key == saved_source_key:
                return dict(raw_source_entry)

        saved_branch = _geometry_manual_q_group_physical_branch_slot(
            saved_entry,
            profile_cache=profile_cache,
        )
        if saved_branch in {0, 1}:
            for raw_source_entry in source_entries:
                if not isinstance(raw_source_entry, Mapping):
                    continue
                source_branch = _geometry_manual_q_group_physical_branch_slot(
                    raw_source_entry,
                    profile_cache=profile_cache,
                )
                if source_branch == saved_branch:
                    return dict(raw_source_entry)
        return None

    def _merge_final_geometry_refinement(
        base_entry: Mapping[str, object],
        refined_entry: Mapping[str, object],
    ) -> dict[str, object]:
        merged = dict(base_entry)
        for key, value in refined_entry.items():
            key_text = str(key)
            if (
                key_text.startswith("refined_sim_")
                or key_text.startswith("sim_refined_")
                or key_text.startswith("sim_refinement_")
            ):
                merged[key_text] = value
        return merged

    def _final_refine_group_geometry_entries(
        group_key: tuple[object, ...],
        saved_entries: Sequence[Mapping[str, object]],
        source_entries: Sequence[Mapping[str, object]],
    ) -> tuple[list[dict[str, object]], int]:
        nonlocal final_refine_parallel_worker_count
        if not callable(refine_saved_pair_entry_fn):
            return [dict(entry) for entry in saved_entries], 0

        saved_items = [dict(entry) for entry in saved_entries if isinstance(entry, Mapping)]

        def _refine_one_saved_entry(
            raw_saved_entry: Mapping[str, object],
        ) -> tuple[dict[str, object], int]:
            if not isinstance(raw_saved_entry, Mapping):
                return {}, 0
            saved_entry = dict(raw_saved_entry)
            source_entry = _final_geometry_refinement_source_entry(
                saved_entry,
                source_entries,
            )
            try:
                refined_entry = refine_saved_pair_entry_fn(
                    dict(saved_entry),
                    dict(source_entry) if isinstance(source_entry, dict) else None,
                )
            except Exception:
                refined_entry = None
            if isinstance(refined_entry, Mapping):
                saved_entry = _merge_final_geometry_refinement(saved_entry, refined_entry)
                return saved_entry, 1
            return saved_entry, 0

        worker_count = _auto_final_refine_worker_count(len(saved_items))
        actual_worker_count = int(worker_count)
        if worker_count > 1:
            try:
                with make_detached_thread_pool_executor(
                    max_workers=int(worker_count),
                    thread_name_prefix="ra-sim-auto-final-refine",
                ) as executor:
                    refined_results = list(executor.map(_refine_one_saved_entry, saved_items))
            except Exception:
                actual_worker_count = 1
                refined_results = [_refine_one_saved_entry(entry) for entry in saved_items]
        else:
            refined_results = [_refine_one_saved_entry(entry) for entry in saved_items]
        final_refine_parallel_worker_count = max(
            int(final_refine_parallel_worker_count),
            int(actual_worker_count),
        )

        refined_entries = [entry for entry, _count in refined_results if entry]
        refined_count = sum(int(count) for _entry, count in refined_results)

        nonlocal working_entries
        working_entries = [
            entry for entry in working_entries if _entry_group_key(entry) != group_key
        ]
        working_entries.extend(dict(entry) for entry in refined_entries)
        return refined_entries, int(refined_count)

    for group_key in ordered_keys:
        raw_group_entries = normalized_grouped.get(group_key, [])
        group_entries = _geometry_manual_collapse_q_group_representatives(
            raw_group_entries,
            profile_cache=profile_cache,
        )
        group_entries = [_refine_sim_candidate(entry) for entry in group_entries]
        group_is_00l = any(
            _geometry_manual_q_group_entry_is_00l(entry, group_key=group_key)
            for entry in group_entries
        )
        if not group_is_00l:
            branch0_entry, branch1_entry = _geometry_manual_q_group_required_branch_entries(
                group_entries,
                group_key=group_key,
                profile_cache=profile_cache,
            )
            if branch0_entry is None or branch1_entry is None:
                branch_pair_results.append(
                    _geometry_manual_branch_pair_result_payload(
                        group_key,
                        {
                            "applied": True,
                            "accepted": False,
                            "reason": "missing_required_branch_pair",
                            "branch_pair_coordinate_frame": None,
                        },
                    )
                )
                attempted_groups += 1
                skipped_groups += 1
                continue
        target_count = geometry_manual_group_target_count(group_key, group_entries)
        if not group_entries or target_count <= 0:
            skipped_groups += 1
            continue

        attempted_groups += 1
        q_label = label_by_key.get(group_key, f"group={group_key}")
        manual_run_id = geometry_manual_start_run_id()
        session_state = {
            "manual_geometry_run_id": manual_run_id,
            "manual_trace_version": MANUAL_GEOMETRY_TRACE_VERSION,
            "background_index": int(background_idx),
            "group_key": group_key,
            "group_entries": [dict(entry) for entry in group_entries],
            "pending_entries": [],
            "target_count": int(target_count),
            "base_entries": [dict(entry) for entry in working_entries],
            "cache_signature": cache_data.get("signature"),
            "q_label": q_label,
            "zoom_active": False,
            "zoom_center": None,
            "saved_xlim": None,
            "saved_ylim": None,
            "preview_last_t": 0.0,
            "preview_last_xy": None,
        }

        for raw_candidate in group_entries:
            if not geometry_manual_pick_session_active(
                session_state,
                current_background_index=background_idx,
            ):
                break
            candidate = dict(raw_candidate)
            seed_point = _geometry_manual_auto_refined_seed_point(
                candidate,
                use_caked_space=bool(use_caked_space),
            )
            if seed_point is None:
                skipped_points += 1
                continue

            session_state["tagged_candidate_key"] = candidate_source_key(candidate)
            session_state["tagged_candidate"] = dict(candidate)
            handled, next_session = geometry_manual_place_selection_at(
                float(seed_point[0]),
                float(seed_point[1]),
                pick_session=session_state,
                current_background_index=background_idx,
                display_background=display_background,
                get_cache_data=lambda **_kwargs: dict(cache_data),
                refine_preview_point=refine_preview_point,
                set_pairs_for_index_fn=_store_pairs,
                set_pick_session_fn=_store_session,
                clear_preview_artists_fn=_noop,
                restore_view_fn=_noop,
                render_current_pairs_fn=_noop,
                update_button_label_fn=_noop,
                set_status_text=None,
                push_undo_state_fn=None,
                use_caked_space=bool(use_caked_space),
                caked_angles_to_background_display_coords=(
                    caked_angles_to_background_display_coords
                ),
                background_display_to_native_detector_coords=(
                    background_display_to_native_detector_coords
                ),
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                radial_axis=radial_axis,
                azimuth_axis=azimuth_axis,
                caked_axis_to_image_index_fn=caked_axis_to_image_index_fn,
                caked_image_index_to_axis_fn=caked_image_index_to_axis_fn,
                candidate_source_key=candidate_source_key,
                nearest_candidate_to_point_fn=nearest_candidate_to_point_fn,
                position_error_px=position_error_px,
                position_sigma_px=position_sigma_px,
                refine_saved_pair_entry_fn=refine_saved_pair_entry_fn,
                profile_cache=profile_cache,
            )
            if not handled:
                skipped_points += 1
                continue
            session_state = dict(next_session) if isinstance(next_session, dict) else {}

        saved_for_group = [
            entry for entry in working_entries if _entry_group_key(entry) == group_key
        ]
        group_complete = bool(
            saved_for_group
            and not geometry_manual_pick_session_active(
                session_state,
                current_background_index=background_idx,
            )
        )
        if group_complete:
            if group_is_00l:
                branch_pair_results.append(
                    _geometry_manual_branch_pair_result_payload(
                        group_key,
                        {
                            "applied": False,
                            "accepted": True,
                            "reason": "00l_branch_collapsed",
                            "branch_pair_coordinate_frame": None,
                        },
                    )
                )
            else:
                branch0_entry, branch1_entry = _geometry_manual_q_group_required_branch_entries(
                    saved_for_group,
                    group_key=group_key,
                    profile_cache=profile_cache,
                )
                branch_pair_result = _score_qr_branch_pair_length_restraint(
                    branch0_entry,
                    branch1_entry,
                    group_key=group_key,
                    use_caked_space=bool(use_caked_space),
                    auto_refine_search_radius_px=normalized_auto_refine_radius,
                    branch_pair_abs_tol_px=branch_pair_abs_tol_px,
                    branch_pair_rel_tol=branch_pair_rel_tol,
                    branch_pair_sigma_px=branch_pair_sigma_px,
                    profile_cache=profile_cache,
                )
                branch_pair_results.append(
                    _geometry_manual_branch_pair_result_payload(group_key, branch_pair_result)
                )
                if bool(branch_pair_reject) and not bool(branch_pair_result.get("accepted", False)):
                    working_entries = [
                        entry for entry in working_entries if _entry_group_key(entry) != group_key
                    ]
                    group_complete = False

        if group_complete:
            saved_for_group, refined_count = _final_refine_group_geometry_entries(
                group_key,
                saved_for_group,
                group_entries,
            )
            final_refined_geometry_peaks += int(refined_count)
            completed_groups += 1
            placed_count += len(saved_for_group)
            completed_keys.append(group_key)
        else:
            working_entries = [
                entry for entry in working_entries if _entry_group_key(entry) != group_key
            ]
            skipped_groups += 1
            session_state = {}

    if placed_count <= 0:
        set_pick_session_fn({})
        if callable(set_status_text):
            set_status_text(
                "Automatic Qr/Qz placement did not save any peaks from the current background."
            )
        return {
            "placed_count": 0,
            "completed_group_count": int(completed_groups),
            "attempted_group_count": int(attempted_groups),
            "skipped_group_count": int(skipped_groups),
            "skipped_point_count": int(skipped_points),
            "completed_q_group_keys": [],
            "branch_pair_restraint_results": list(branch_pair_results),
            "final_refined_geometry_peak_count": int(final_refined_geometry_peaks),
            "final_refine_parallel_worker_count": int(final_refine_parallel_worker_count),
        }

    if callable(push_undo_state_fn):
        push_undo_state_fn()
    set_pick_session_fn({})
    set_pairs_for_index_fn(background_idx, working_entries)
    clear_preview_artists_fn(redraw=False)
    if bool(use_caked_space):
        restore_view_fn(redraw=False)
    render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    if callable(set_status_text):
        set_status_text(
            f"Auto-added {placed_count} manual Qr/Qz peak placements across "
            f"{completed_groups}/{attempted_groups} groups on background "
            f"{background_idx + 1}"
            + (f" ({skipped_groups} groups skipped)." if skipped_groups else ".")
        )
    return {
        "placed_count": int(placed_count),
        "completed_group_count": int(completed_groups),
        "attempted_group_count": int(attempted_groups),
        "skipped_group_count": int(skipped_groups),
        "skipped_point_count": int(skipped_points),
        "completed_q_group_keys": [tuple(key) for key in completed_keys],
        "branch_pair_restraint_results": list(branch_pair_results),
        "final_refined_geometry_peak_count": int(final_refined_geometry_peaks),
        "final_refine_parallel_worker_count": int(final_refine_parallel_worker_count),
    }


def cancel_geometry_manual_pick_session(
    pick_session: dict[str, object] | None,
    *,
    current_background_index: object = None,
    restore_view_fn: Callable[..., None],
    clear_preview_artists_fn: Callable[..., None],
    render_current_pairs_fn: Callable[..., None],
    update_button_label_fn: Callable[[], None],
    set_status_text: Callable[[str], None] | None = None,
    restore_view: bool = True,
    redraw: bool = True,
    message: str | None = None,
    use_caked_space: bool = True,
) -> dict[str, object]:
    """Discard any in-progress manual Qr/Qz placement state."""

    had_session = geometry_manual_pick_session_active(
        pick_session,
        current_background_index=current_background_index,
        require_current_background=False,
    )
    if restore_view and use_caked_space:
        restore_view_fn(redraw=False)
    clear_preview_artists_fn(redraw=False)
    if had_session and redraw:
        render_current_pairs_fn(update_status=False)
    update_button_label_fn()
    if message and callable(set_status_text):
        set_status_text(message)
    return {}


def match_geometry_manual_group_to_background(
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    background_image: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
    build_cache_data: Callable[[], dict[str, object]] | None = None,
    auto_match_background_context: Callable[
        [np.ndarray, dict[str, object]],
        tuple[dict[str, object], dict[str, object]],
    ]
    | None = None,
    match_simulated_peaks_to_peak_context: Callable[
        [Sequence[dict[str, object]], dict[str, object], dict[str, object]],
        tuple[Sequence[dict[str, object]], object],
    ],
    candidate_source_key: Callable[
        [dict[str, object] | None],
        tuple[object, ...] | None,
    ] = geometry_manual_candidate_source_key,
) -> dict[tuple[object, ...], tuple[float, float]]:
    """Return refined measured peak centers for one clicked symmetric Qr/Qz group."""

    entries = [dict(entry) for entry in candidate_entries or [] if isinstance(entry, dict)]
    if not entries:
        return {}

    background_local = background_image
    if background_local is None:
        return {}

    state = cache_data if isinstance(cache_data, dict) else {}
    if not state and callable(build_cache_data):
        try:
            built_state = build_cache_data()
        except Exception:
            built_state = {}
        if isinstance(built_state, dict):
            state = built_state

    match_cfg = dict(state.get("match_config", {})) if isinstance(state, dict) else {}
    background_context = state.get("background_context") if isinstance(state, dict) else None
    if not isinstance(background_context, dict) or not bool(
        background_context.get("img_valid", False)
    ):
        if not callable(auto_match_background_context):
            return {}
        try:
            match_cfg, background_context = auto_match_background_context(
                background_local,
                match_cfg,
            )
        except Exception:
            return {}

    try:
        matches, _stats = match_simulated_peaks_to_peak_context(
            entries,
            background_context,
            match_cfg,
        )
    except Exception:
        return {}

    matched_lookup: dict[tuple[object, ...], tuple[float, float]] = {}
    for match_entry in matches:
        source_key = candidate_source_key(match_entry)
        if source_key is None:
            continue
        try:
            match_col = float(match_entry.get("x", np.nan))
            match_row = float(match_entry.get("y", np.nan))
        except Exception:
            continue
        if np.isfinite(match_col) and np.isfinite(match_row):
            matched_lookup[source_key] = (float(match_col), float(match_row))
    return matched_lookup


def ensure_geometry_fit_caked_view(
    *,
    show_caked_2d_var: Any,
    pick_uses_caked_space: Callable[[], bool],
    toggle_caked_2d: Callable[[], None],
    do_update: Callable[[], None],
    schedule_update: Callable[[], None],
    root: Any,
    update_pending: object | None,
    integration_update_pending: object | None,
    update_running: bool = False,
    force_refresh: bool = False,
) -> tuple[object | None, object | None]:
    """Switch geometry fitting/import into the 2D caked integration view now."""

    needs_refresh = bool(force_refresh)
    if not bool(show_caked_2d_var.get()):
        show_caked_2d_var.set(True)
        toggle_caked_2d()
        needs_refresh = True
    elif not pick_uses_caked_space():
        needs_refresh = True

    if not needs_refresh:
        return update_pending, integration_update_pending

    gui_controllers.clear_tk_after_token(root, integration_update_pending)
    integration_update_pending = None
    gui_controllers.clear_tk_after_token(root, update_pending)
    update_pending = None

    if bool(update_running):
        schedule_update()
        return update_pending, integration_update_pending

    do_update()
    return update_pending, integration_update_pending


def caked_angles_to_background_display_coords(
    two_theta_deg: float,
    phi_deg: float,
    *,
    ai: object = None,
    native_background: np.ndarray | None = None,
    caked_radial_values: Sequence[float] | None = None,
    caked_azimuth_values: Sequence[float] | None = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]],
    scattering_angles_to_detector_pixel: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ]
    | None,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    transform_bundle: CakeTransformBundle | None = None,
    backend_detector_coords_to_native_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    bundle_detector_coords_to_background_display_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
    rotate_point_for_display: Callable[
        [float, float, tuple[int, ...], int], tuple[float, float]
    ] = _default_rotate_point,
    display_rotate_k: int = 0,
) -> tuple[float | None, float | None]:
    """Back-project one caked-space point to the displayed detector background.

    Legacy angular-map and analytic args remain for call-site compatibility,
    but live inversion now uses only the active exact-cake LUT transform. If
    a future caller inverts caked intensities or a full caked image, remember
    that the caked data may already be solid-angle corrected. Restore the
    detector solid-angle weighting before undoing backend orientation so the
    reconstructed detector intensities remain in detector-count space.
    """

    if native_background is None or not (np.isfinite(two_theta_deg) and np.isfinite(phi_deg)):
        return None, None

    native_shape = tuple(int(v) for v in native_background.shape[:2])
    del (
        get_detector_angular_maps,
        scattering_angles_to_detector_pixel,
        center,
        detector_distance,
        pixel_size,
    )

    def _finite_tuple_pair(
        value: object,
    ) -> tuple[float, float] | None:
        if not isinstance(value, tuple) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _display_point_from_detector_coords(
        col: float,
        row: float,
    ) -> tuple[float | None, float | None]:
        if callable(bundle_detector_coords_to_background_display_coords):
            try:
                display_point = bundle_detector_coords_to_background_display_coords(
                    float(col),
                    float(row),
                )
            except Exception:
                display_point = None
            display_tuple = _finite_tuple_pair(display_point)
            if display_tuple is not None:
                return float(display_tuple[0]), float(display_tuple[1])
        native_col = float(col)
        native_row = float(row)
        if callable(backend_detector_coords_to_native_detector_coords):
            try:
                native_point = backend_detector_coords_to_native_detector_coords(
                    float(col),
                    float(row),
                )
            except Exception:
                native_point = None
            if (
                isinstance(native_point, tuple)
                and len(native_point) >= 2
                and native_point[0] is not None
                and native_point[1] is not None
                and np.isfinite(float(native_point[0]))
                and np.isfinite(float(native_point[1]))
            ):
                native_col = float(native_point[0])
                native_row = float(native_point[1])
        return rotate_point_for_display(
            float(native_col),
            float(native_row),
            native_shape,
            display_rotate_k,
        )

    try:
        native_point = _caked_point_to_detector_pixel(
            ai if ai is not None else None,
            native_shape,
            caked_radial_values,
            caked_azimuth_values,
            float(two_theta_deg),
            float(phi_deg),
            transform_bundle=(
                transform_bundle if isinstance(transform_bundle, CakeTransformBundle) else None
            ),
        )
    except Exception:
        native_point = (None, None)
    if (
        isinstance(native_point, tuple)
        and len(native_point) >= 2
        and native_point[0] is not None
        and native_point[1] is not None
        and np.isfinite(float(native_point[0]))
        and np.isfinite(float(native_point[1]))
    ):
        return _display_point_from_detector_coords(
            float(native_point[0]),
            float(native_point[1]),
        )
    return None, None


def native_detector_coords_to_caked_display_coords(
    col: float,
    row: float,
    *,
    ai: object = None,
    get_detector_angular_maps: Callable[[object], tuple[object, object]],
    detector_pixel_to_scattering_angles: Callable[
        [float, float, Sequence[float] | None, float, float],
        tuple[float | None, float | None],
    ],
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
    wrap_phi_range: Callable[[object], object],
    transform_bundle: CakeTransformBundle | None = None,
    caked_radial_values: Sequence[float] | None = None,
    caked_azimuth_values: Sequence[float] | None = None,
    native_detector_coords_to_bundle_detector_coords: Callable[
        [float, float],
        tuple[float | None, float | None],
    ]
    | None = None,
) -> tuple[float, float] | None:
    """Project one native detector point into continuous caked display coords.

    Legacy angular-map and analytic parameters remain in the signature for
    call-site compatibility, but live projection now uses only the active exact
    cake transform bundle. No bundle means no projection.
    """

    del (
        get_detector_angular_maps,
        detector_pixel_to_scattering_angles,
        center,
        detector_distance,
        pixel_size,
        wrap_phi_range,
        caked_radial_values,
        caked_azimuth_values,
    )

    try:
        col_val = float(col)
        row_val = float(row)
    except Exception:
        return None
    if not (np.isfinite(col_val) and np.isfinite(row_val)):
        return None

    if callable(native_detector_coords_to_bundle_detector_coords):
        bundle_point = native_detector_coords_to_bundle_detector_coords(
            float(col_val),
            float(row_val),
        )
        if not isinstance(bundle_point, tuple) or len(bundle_point) < 2:
            return None
        try:
            col_val = float(bundle_point[0])
            row_val = float(bundle_point[1])
        except Exception:
            return None
        if not (np.isfinite(col_val) and np.isfinite(row_val)):
            return None

    live_bundle = (
        transform_bundle
        if isinstance(transform_bundle, CakeTransformBundle)
        else getattr(ai, "_live_caked_transform_bundle", None)
    )
    bundle_two_theta, bundle_phi = _detector_pixel_to_caked_bin(
        live_bundle,
        col_val,
        row_val,
    )
    if bundle_two_theta is None or bundle_phi is None:
        return None
    return float(bundle_two_theta), float(bundle_phi)


def should_collect_hit_tables_for_update(
    *,
    background_visible: bool,
    current_background_index: object,
    skip_preview_once: bool = False,
    manual_pick_armed: bool = False,
    hkl_pick_armed: bool,
    selected_hkl_target: object,
    selected_peak_record: object,
    geometry_q_group_refresh_requested: bool,
    live_geometry_preview_enabled: Callable[[], bool],
    current_manual_pick_background_image: Callable[[], object],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    geometry_manual_pick_session_active: Callable[[], bool],
) -> bool:
    """Return whether the next redraw needs per-hit detector tables."""

    if bool(skip_preview_once):
        return False

    manual_geometry_overlay_requested = False
    if background_visible and current_manual_pick_background_image() is not None:
        try:
            has_manual_geometry_overlay = bool(
                geometry_manual_pairs_for_index(int(current_background_index))
                or geometry_manual_pick_session_active()
            )
        except Exception:
            has_manual_geometry_overlay = False
        manual_geometry_overlay_requested = bool(manual_pick_armed or has_manual_geometry_overlay)

    return bool(
        hkl_pick_armed
        or selected_hkl_target is not None
        or selected_peak_record is not None
        or live_geometry_preview_enabled()
        or geometry_q_group_refresh_requested
        or manual_geometry_overlay_requested
    )


def normalize_bragg_qr_source_label(source_label: str | None) -> str:
    """Normalize the serialized Qr-source label used in manual-geometry data."""

    label = str(source_label or "primary").strip().lower()
    if label == "secondary":
        return "secondary"
    if label in gui_controllers.BRAGG_QR_AUX_SOURCE_LABELS:
        return label
    return "primary"


def q_group_key_component(value: float) -> int | float:
    """Normalize a floating Q-group component into a stable hashable value."""

    if np.isfinite(value) and abs(value - round(value)) <= 1e-6:
        return int(round(value))
    return round(float(value), 6)


def integer_gz_index(value: object, *, tol: float = 1e-6) -> int | None:
    """Return the integer Gz/L index when the value is close enough to an integer."""

    try:
        raw = float(value)
    except Exception:
        return None
    if not np.isfinite(raw):
        return None
    rounded = int(round(raw))
    if abs(raw - rounded) > float(tol):
        return None
    return rounded


def geometry_q_group_key_to_jsonable(group_key: object) -> list[object] | None:
    """Convert one stable Qr/Qz group key into a JSON-safe list."""

    if not isinstance(group_key, tuple) or len(group_key) < 4:
        return None
    try:
        prefix = str(group_key[0])
        source_label = normalize_bragg_qr_source_label(str(group_key[1]))
        m_component = q_group_key_component(float(group_key[2]))
        gz_index = int(group_key[3])
    except Exception:
        return None
    return [prefix, source_label, m_component, gz_index]


def geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""

    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        prefix = str(value[0])
        source_label = normalize_bragg_qr_source_label(str(value[1]))
        m_component = q_group_key_component(float(value[2]))
        gz_index = integer_gz_index(value[3])
    except Exception:
        return None
    if prefix != "q_group" or gz_index is None:
        return None
    return (prefix, source_label, m_component, int(gz_index))


def geometry_manual_pair_entry_to_jsonable(
    entry: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Convert one saved manual pair entry into a JSON-safe dictionary."""

    normalized = normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=normalize_hkl_key,
        sigma_floor_px=sigma_floor_px,
    )
    if normalized is None:
        return None

    row: dict[str, object] = {
        "x": float(normalized["x"]),
        "y": float(normalized["y"]),
        "label": str(normalized.get("label", "")),
    }

    for key in (
        "raw_x",
        "raw_y",
        "detector_x",
        "detector_y",
        "background_detector_x",
        "background_detector_y",
        "background_two_theta_deg",
        "background_phi_deg",
        "caked_x",
        "caked_y",
        "raw_caked_x",
        "raw_caked_y",
        "placement_error_px",
        "sigma_px",
        "refined_sim_x",
        "refined_sim_y",
        "refined_sim_native_x",
        "refined_sim_native_y",
        "refined_sim_caked_x",
        "refined_sim_caked_y",
        "sim_detector_x",
        "sim_detector_y",
        "obs_detector_x",
        "obs_detector_y",
    ):
        value = normalized.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            row[key] = float(numeric)

    hkl_key = normalized.get("hkl")
    if isinstance(hkl_key, tuple) and len(hkl_key) >= 3:
        try:
            row["hkl"] = [int(hkl_key[0]), int(hkl_key[1]), int(hkl_key[2])]
        except Exception:
            pass

    serialized_group_key = geometry_q_group_key_to_jsonable(normalized.get("q_group_key"))
    if serialized_group_key is not None:
        row["q_group_key"] = serialized_group_key

    for key in (
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "source_branch_index",
        "source_peak_index",
    ):
        if key in normalized:
            try:
                row[key] = int(normalized[key])
            except Exception:
                continue

    if normalized.get("source_reflection_namespace") is not None:
        row["source_reflection_namespace"] = str(normalized.get("source_reflection_namespace"))
    if "source_reflection_is_full" in normalized:
        row["source_reflection_is_full"] = bool(normalized.get("source_reflection_is_full", False))

    if normalized.get("source_label") is not None:
        row["source_label"] = str(normalized.get("source_label"))
    for text_key in ("phase_label", "structure_role"):
        if normalized.get(text_key) is not None:
            row[text_key] = str(normalized.get(text_key))
    for bool_key in (
        "background_qr_set_reference",
        "manual_background_reference",
        "geometry_fit_disabled",
    ):
        if bool_key in normalized:
            row[bool_key] = bool(normalized.get(bool_key, False))
    if normalized.get("background_qr_set_reference_label_source") is not None:
        row["background_qr_set_reference_label_source"] = str(
            normalized.get("background_qr_set_reference_label_source")
        )

    for key in ("branch_id", "branch_source", "selection_reason"):
        if normalized.get(key) is not None:
            row[key] = str(normalized.get(key))
    if normalized.get("best_sample_index") is not None:
        try:
            row["best_sample_index"] = int(normalized.get("best_sample_index"))
        except Exception:
            pass
    if normalized.get("mosaic_weight") is not None:
        try:
            mosaic_weight = float(normalized.get("mosaic_weight"))
        except Exception:
            mosaic_weight = float("nan")
        if np.isfinite(mosaic_weight):
            row["mosaic_weight"] = float(mosaic_weight)
    rank_key = normalized.get("mosaic_top_rank_key")
    if isinstance(rank_key, (tuple, list)):
        serialized_rank: list[object] = []
        for item in rank_key:
            if isinstance(item, np.generic):
                item = item.item()
            if isinstance(item, int):
                serialized_rank.append(int(item))
            elif isinstance(item, float) and np.isfinite(float(item)):
                serialized_rank.append(float(item))
            elif isinstance(item, str):
                serialized_rank.append(str(item))
        if serialized_rank:
            row["mosaic_top_rank_key"] = serialized_rank

    return row


def geometry_manual_pair_entry_from_jsonable(
    row: dict[str, object] | None,
    *,
    normalize_hkl_key: Callable[[object], tuple[int, int, int] | None] = _default_normalize_hkl_key,
    sigma_floor_px: float = DEFAULT_POSITION_SIGMA_FLOOR_PX,
) -> dict[str, object] | None:
    """Rebuild one saved manual pair entry from imported JSON data."""

    if not isinstance(row, dict):
        return None

    entry = dict(row)
    raw_hkl = row.get("hkl")
    if isinstance(raw_hkl, (list, tuple)) and len(raw_hkl) >= 3:
        try:
            entry["hkl"] = (
                int(raw_hkl[0]),
                int(raw_hkl[1]),
                int(raw_hkl[2]),
            )
        except Exception:
            pass

    restored_group_key = geometry_q_group_key_from_jsonable(row.get("q_group_key"))
    if restored_group_key is not None:
        entry["q_group_key"] = restored_group_key

    normalized = normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=normalize_hkl_key,
        sigma_floor_px=sigma_floor_px,
    )
    if normalized is None:
        return None

    _canonicalize_manual_entry_branch_fields(
        normalized,
        allow_legacy_peak_fallback=True,
        preserve_legacy_peak_when_unresolved=True,
        retry_legacy_peak_on_deadband=True,
    )
    return normalized


def normalized_background_path_for_compare(raw_path: object) -> str | None:
    """Return a normalized path string suitable for background matching."""

    try:
        candidate = Path(str(raw_path)).expanduser()
    except Exception:
        return None
    if not str(candidate):
        return None
    return os.path.normcase(os.path.normpath(str(candidate)))


def geometry_manual_pairs_export_rows(
    *,
    pairs_by_background: Mapping[object, Sequence[dict[str, object]]] | None,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_entry_to_jsonable: Callable[
        [dict[str, object] | None],
        dict[str, object] | None,
    ] = geometry_manual_pair_entry_to_jsonable,
) -> list[dict[str, object]]:
    """Return saved manual placements as portable per-background rows."""

    background_indices: set[int] = set()
    raw_keys = pairs_by_background.keys() if isinstance(pairs_by_background, Mapping) else ()
    for raw_idx in raw_keys:
        try:
            background_indices.add(int(raw_idx))
        except Exception:
            continue

    rows: list[dict[str, object]] = []
    for background_idx in sorted(background_indices):
        entries = [
            serialized
            for serialized in (
                pair_entry_to_jsonable(entry) for entry in pairs_for_index(int(background_idx))
            )
            if serialized is not None
        ]
        if not entries:
            continue

        if 0 <= int(background_idx) < len(osc_files):
            background_path = str(Path(str(osc_files[background_idx])).expanduser())
            background_name = Path(str(osc_files[background_idx])).name
        else:
            background_path = None
            background_name = f"background_{int(background_idx) + 1}"

        rows.append(
            {
                "background_index": int(background_idx),
                "background_path": background_path,
                "background_name": background_name,
                "entries": entries,
            }
        )
    return rows


def collect_geometry_manual_pairs_snapshot(
    *,
    osc_files: Sequence[object],
    current_background_index: object,
    manual_pair_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Return a portable snapshot of saved manual geometry placements."""

    try:
        background_index = int(current_background_index)
    except Exception:
        background_index = 0

    return {
        "background_files": [str(Path(str(path)).expanduser()) for path in osc_files],
        "current_background_index": int(background_index),
        "manual_pairs": list(manual_pair_rows),
    }


def apply_geometry_manual_pairs_rows(
    rows: Sequence[object] | None,
    *,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    pair_entry_from_jsonable: Callable[
        [dict[str, object] | None],
        dict[str, object] | None,
    ] = geometry_manual_pair_entry_from_jsonable,
    replace_pairs_by_background: Callable[[dict[int, list[dict[str, object]]]], object],
    clear_preview_artists: Callable[..., None],
    cancel_pick_session: Callable[..., None],
    invalidate_pick_cache: Callable[[], None],
    clear_manual_undo_stack: Callable[[], None],
    clear_geometry_fit_undo_stack: Callable[[], None],
    render_current_pairs: Callable[..., None],
    update_button_label: Callable[[], None],
    refresh_status: Callable[[], None],
    backfill_imported_pairs_for_background: Callable[
        [int, Sequence[dict[str, object]]],
        Sequence[dict[str, object]],
    ]
    | None = None,
    replace_existing: bool = True,
) -> tuple[int, int, list[str]]:
    """Import saved manual geometry pairs onto the currently loaded backgrounds."""

    exact_path_lookup: dict[str, int] = {}
    name_lookup: defaultdict[str, list[int]] = defaultdict(list)
    for idx, raw_path in enumerate(osc_files):
        normalized_path = normalized_background_path_for_compare(raw_path)
        if normalized_path is not None:
            exact_path_lookup[normalized_path] = int(idx)
        name_lookup[Path(str(raw_path)).name].append(int(idx))

    imported_map: dict[int, list[dict[str, object]]]
    if replace_existing:
        imported_map = {}
    else:
        imported_map = {
            int(idx): list(pairs_for_index(int(idx)))
            for idx in range(len(osc_files))
            if pairs_for_index(int(idx))
        }

    warnings: list[str] = []
    matched_backgrounds: set[int] = set()
    pair_count = 0
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue

        target_index = None
        normalized_path = normalized_background_path_for_compare(raw_row.get("background_path"))
        if normalized_path is not None:
            target_index = exact_path_lookup.get(normalized_path)
        if target_index is None:
            background_name = raw_row.get("background_name")
            if background_name is not None:
                matches = name_lookup.get(Path(str(background_name)).name, [])
                if len(matches) == 1:
                    target_index = int(matches[0])
        if target_index is None:
            try:
                fallback_index = int(raw_row.get("background_index"))
            except Exception:
                fallback_index = None
            if fallback_index is not None and 0 <= fallback_index < len(osc_files):
                target_index = int(fallback_index)

        if target_index is None:
            warnings.append(
                f"Skipped placements for '{raw_row.get('background_name', 'unknown background')}'."
            )
            continue

        imported_entries = [
            restored
            for restored in (
                pair_entry_from_jsonable(entry) for entry in raw_row.get("entries", [])
            )
            if restored is not None
        ]
        if imported_entries and callable(backfill_imported_pairs_for_background):
            try:
                backfilled_entries = backfill_imported_pairs_for_background(
                    int(target_index),
                    [dict(entry) for entry in imported_entries],
                )
            except Exception as exc:
                warnings.append(
                    "Imported placements for "
                    f"'{raw_row.get('background_name', 'unknown background')}' "
                    f"without detector-to-caked backfill: {exc}"
                )
            else:
                imported_entries = [
                    dict(entry) for entry in backfilled_entries or () if isinstance(entry, Mapping)
                ]
        imported_map[int(target_index)] = imported_entries
        if imported_entries:
            matched_backgrounds.add(int(target_index))
            pair_count += len(imported_entries)

    replace_pairs_by_background(
        {int(idx): list(entries) for idx, entries in imported_map.items() if entries}
    )
    clear_preview_artists(redraw=False)
    cancel_pick_session(restore_view=True, redraw=False)
    invalidate_pick_cache()
    clear_manual_undo_stack()
    clear_geometry_fit_undo_stack()
    render_current_pairs(update_status=False)
    update_button_label()
    refresh_status()
    return int(len(matched_backgrounds)), int(pair_count), warnings


def apply_geometry_manual_pairs_snapshot(
    snapshot: Mapping[str, object] | dict[str, object],
    *,
    allow_background_reload: bool = True,
    osc_files: Sequence[object],
    load_background_files: Callable[[Sequence[str], int], None] | None = None,
    apply_pairs_rows: Callable[..., tuple[int, int, list[str]]],
    schedule_update: Callable[[], None] | None = None,
) -> str:
    """Restore saved manual geometry placements from a snapshot dictionary."""

    warnings: list[str] = []

    if allow_background_reload:
        raw_background_paths = snapshot.get("background_files", [])
        background_paths: list[str] = []
        if isinstance(raw_background_paths, list):
            for raw_path in raw_background_paths:
                if raw_path is None:
                    continue
                background_paths.append(str(Path(str(raw_path)).expanduser()))

        if background_paths:
            saved_paths_norm = [
                path_norm
                for path_norm in (
                    normalized_background_path_for_compare(path) for path in background_paths
                )
                if path_norm is not None
            ]
            current_paths_norm = [
                path_norm
                for path_norm in (
                    normalized_background_path_for_compare(path) for path in osc_files
                )
                if path_norm is not None
            ]
            if saved_paths_norm != current_paths_norm:
                missing_paths = [path for path in background_paths if not Path(path).is_file()]
                if not missing_paths:
                    try:
                        if callable(load_background_files):
                            load_background_files(
                                background_paths,
                                int(snapshot.get("current_background_index", 0)),
                            )
                    except Exception as exc:
                        warnings.append(f"background reload: {exc}")
                else:
                    warnings.append(
                        "saved background files are missing; placements were mapped onto the "
                        "currently loaded backgrounds where possible"
                    )

    imported_backgrounds, imported_pairs, import_warnings = apply_pairs_rows(
        snapshot.get("manual_pairs", []),
        replace_existing=True,
    )
    warnings.extend(import_warnings)
    if callable(schedule_update):
        schedule_update()

    message = f"Imported {imported_pairs} manual placement(s) across {imported_backgrounds} background(s)."
    if warnings:
        message += " Warnings: " + "; ".join(warnings[:4])
        if len(warnings) > 4:
            message += f"; +{len(warnings) - 4} more"
    return message


def export_geometry_manual_pairs(
    *,
    osc_files: Sequence[object],
    pairs_for_index: Callable[[int], Sequence[dict[str, object]]],
    collect_snapshot: Callable[[], Mapping[str, object] | dict[str, object]],
    initial_dir: str | Path | None,
    asksaveasfilename: Callable[..., object],
    save_file: Callable[..., None],
    set_status_text: Callable[[str], None] | None = None,
    stamp_factory: Callable[[], str] | None = None,
    entrypoint: str = DEFAULT_GUI_ENTRYPOINT,
) -> str | None:
    """Run the manual-placement export dialog workflow."""

    if not any(pairs_for_index(idx) for idx in range(len(osc_files))):
        if callable(set_status_text):
            set_status_text("No saved manual placements are available to export.")
        return None

    initial_dir_value = (
        str(Path(initial_dir).expanduser()) if initial_dir is not None else str(Path.cwd())
    )
    stamp = str(stamp_factory()).strip() if callable(stamp_factory) else ""
    initial_file = "ra_sim_geometry_placements.json"
    if stamp:
        initial_file = f"ra_sim_geometry_placements_{stamp}.json"

    file_path = asksaveasfilename(
        title="Export Geometry Placements",
        initialdir=initial_dir_value,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=initial_file,
    )
    if not file_path:
        if callable(set_status_text):
            set_status_text("Geometry placement export canceled.")
        return None

    try:
        save_file(
            file_path,
            collect_snapshot(),
            metadata={"entrypoint": str(entrypoint)},
        )
    except Exception as exc:
        if callable(set_status_text):
            set_status_text(f"Failed to export geometry placements: {exc}")
        return None

    if callable(set_status_text):
        set_status_text(f"Saved manual geometry placements to {file_path}")
    return str(file_path)


def import_geometry_manual_pairs(
    *,
    initial_dir: str | Path | None,
    askopenfilename: Callable[..., object],
    load_file: Callable[[str | Path], Mapping[str, object] | dict[str, object]],
    apply_snapshot: Callable[..., str],
    ensure_geometry_fit_caked_view: Callable[[], None] | None = None,
    set_status_text: Callable[[str], None] | None = None,
) -> str | None:
    """Run the manual-placement import dialog workflow."""

    initial_dir_value = (
        str(Path(initial_dir).expanduser()) if initial_dir is not None else str(Path.cwd())
    )
    file_path = askopenfilename(
        title="Import Geometry Placements",
        initialdir=initial_dir_value,
        filetypes=[("RA-SIM geometry placements", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        if callable(set_status_text):
            set_status_text("Geometry placement import canceled.")
        return None

    try:
        payload = load_file(file_path)
        message = apply_snapshot(
            payload.get("state", {}),
            allow_background_reload=True,
        )
    except Exception as exc:
        if callable(set_status_text):
            set_status_text(f"Failed to import geometry placements: {exc}")
        return None

    if callable(ensure_geometry_fit_caked_view):
        try:
            ensure_geometry_fit_caked_view()
        except Exception as exc:
            message += (
                f" Warning: imported placements but could not switch to 2D caked view ({exc})."
            )

    if callable(set_status_text):
        set_status_text(message)
    return message


_geometry_manual_toggle_selection_at_base = geometry_manual_toggle_selection_at


def _geometry_manual_click_remove_runtime_arg(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    name: str,
) -> object | None:
    if name in kwargs:
        return kwargs.get(name)
    for candidate in args:
        if name == "manual_state" and hasattr(candidate, "pairs_by_background"):
            return candidate
        if name == "projection_callbacks" and hasattr(candidate, "entry_display_coords"):
            return candidate
    return None


def _geometry_manual_click_remove_current_background_index(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    pairs_by_background: Mapping[object, object],
) -> int:
    for key in ("current_background_index", "background_index"):
        raw_value = kwargs.get(key)
        raw_value = _resolve_runtime_value(raw_value)
        try:
            return int(raw_value)
        except Exception:
            pass
    keys = list(pairs_by_background.keys())
    if len(keys) == 1:
        try:
            return int(keys[0])
        except Exception:
            pass
    return 0


def _geometry_manual_click_remove_point(
    entry: Mapping[str, object],
    keys: Sequence[tuple[str, str]],
) -> tuple[float, float] | None:
    for x_key, y_key in keys:
        try:
            x_val = float(entry.get(x_key, np.nan))
            y_val = float(entry.get(y_key, np.nan))
        except Exception:
            continue
        if np.isfinite(x_val) and np.isfinite(y_val):
            return float(x_val), float(y_val)
    return None


def _geometry_manual_click_remove_entry_points(
    entry: Mapping[str, object],
    projection_callbacks: object | None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    entry_display_coords = getattr(projection_callbacks, "entry_display_coords", None)
    if callable(entry_display_coords):
        try:
            display_point = entry_display_coords(dict(entry))
        except Exception:
            display_point = None
        if isinstance(display_point, (tuple, list)) and len(display_point) >= 2:
            try:
                x_val = float(display_point[0])
                y_val = float(display_point[1])
            except Exception:
                x_val = y_val = float("nan")
            if np.isfinite(x_val) and np.isfinite(y_val):
                points.append((float(x_val), float(y_val)))

    for keys in (
        (
            ("refined_sim_x", "refined_sim_y"),
            ("sim_col", "sim_row"),
            ("sim_col_local", "sim_row_local"),
            ("display_col", "display_row"),
            ("x", "y"),
        ),
        (
            ("refined_sim_caked_x", "refined_sim_caked_y"),
            ("simulated_two_theta_deg", "simulated_phi_deg"),
            ("two_theta_deg", "phi_deg"),
            ("caked_x", "caked_y"),
            ("raw_caked_x", "raw_caked_y"),
            ("background_two_theta_deg", "background_phi_deg"),
        ),
    ):
        point = _geometry_manual_click_remove_point(entry, keys)
        if point is not None and point not in points:
            points.append(point)
    return points


def _geometry_manual_click_remove_hit_q_group_key(
    entries: Sequence[Mapping[str, object]],
    col: float,
    row: float,
    projection_callbacks: object | None,
    *,
    hit_radius: float = 12.0,
) -> tuple[object, ...] | None:
    best_key: tuple[object, ...] | None = None
    best_distance = float("inf")
    for entry in entries:
        raw_key = entry.get("q_group_key")
        if isinstance(raw_key, list):
            q_group_key = tuple(raw_key)
        elif isinstance(raw_key, tuple):
            q_group_key = raw_key
        else:
            continue
        for point_x, point_y in _geometry_manual_click_remove_entry_points(
            entry,
            projection_callbacks,
        ):
            distance = float(np.hypot(float(point_x) - float(col), float(point_y) - float(row)))
            if distance < best_distance:
                best_distance = distance
                best_key = q_group_key
    if best_key is not None and best_distance <= float(hit_radius):
        return best_key
    return None


def geometry_manual_remove_q_group_at(
    col: float,
    row: float,
    *args: object,
    **kwargs: object,
) -> bool:
    manual_state = _geometry_manual_click_remove_runtime_arg(
        args,
        kwargs,
        "manual_state",
    )
    if manual_state is None:
        return False
    pairs_by_background = getattr(manual_state, "pairs_by_background", None)
    if not isinstance(pairs_by_background, dict):
        return False
    has_explicit_background = any(
        key in kwargs for key in ("current_background_index", "background_index")
    )
    background_index = _geometry_manual_click_remove_current_background_index(
        args,
        kwargs,
        pairs_by_background,
    )
    entries = list(pairs_by_background.get(int(background_index), []))
    projection_callbacks = _geometry_manual_click_remove_runtime_arg(
        args,
        kwargs,
        "projection_callbacks",
    )
    q_group_key = _geometry_manual_click_remove_hit_q_group_key(
        entries,
        float(col),
        float(row),
        projection_callbacks,
    )
    if q_group_key is None and not has_explicit_background:
        for raw_background_index, raw_entries in pairs_by_background.items():
            try:
                candidate_background_index = int(raw_background_index)
            except Exception:
                continue
            candidate_entries = list(raw_entries or [])
            candidate_key = _geometry_manual_click_remove_hit_q_group_key(
                candidate_entries,
                float(col),
                float(row),
                projection_callbacks,
            )
            if candidate_key is None:
                continue
            background_index = candidate_background_index
            entries = candidate_entries
            q_group_key = candidate_key
            break
    if q_group_key is None:
        return False

    kept_entries = []
    for entry in entries:
        raw_key = entry.get("q_group_key")
        entry_key = tuple(raw_key) if isinstance(raw_key, (list, tuple)) else None
        if entry_key != tuple(q_group_key):
            kept_entries.append(dict(entry))
    removed_count = len(entries) - len(kept_entries)
    if removed_count <= 0:
        return False
    if kept_entries:
        pairs_by_background[int(background_index)] = kept_entries
    else:
        pairs_by_background.pop(int(background_index), None)

    for callback_name in (
        "cancel_pick_session",
        "invalidate_pick_cache",
        "clear_geometry_fit_undo_stack",
        "render_current_pairs",
        "refresh_status",
    ):
        callback = kwargs.get(callback_name)
        if not callable(callback):
            continue
        try:
            callback()
        except TypeError:
            try:
                callback(update_status=False)
            except Exception:
                pass
        except Exception:
            pass
    set_status_text = kwargs.get("set_status_text")
    if callable(set_status_text):
        try:
            set_status_text(
                f"Removed {removed_count} placed Qr-set peak(s) from current background."
            )
        except Exception:
            pass
    return True


def geometry_manual_toggle_selection_at(
    *args: object,
    remove_q_group_only: bool = False,
    **kwargs: object,
) -> object:
    if bool(remove_q_group_only):
        try:
            col = float(args[0])
            row = float(args[1])
        except Exception:
            return False
        return geometry_manual_remove_q_group_at(col, row, *args[2:], **kwargs)
    return _geometry_manual_toggle_selection_at_base(*args, **kwargs)
