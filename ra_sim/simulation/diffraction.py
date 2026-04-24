"""Core diffraction routines used by the simulator."""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ra_sim.config import get_dir
from numba import get_num_threads, get_thread_id, njit, prange
from math import sin, cos, sqrt, pi, exp, acos
from ra_sim.simulation.intersection_cache_schema import (
    CACHE_COL_BEST_SAMPLE_INDEX,
    CACHE_COL_SOURCE_ROW_INDEX,
    CACHE_COL_SOURCE_TABLE_INDEX,
    CURRENT_DETECTOR_CACHE_WIDTH,
    HIT_ROW_COL_BEST_SAMPLE_INDEX,
    HIT_ROW_COL_SOURCE_ROW_INDEX,
    HIT_ROW_COL_SOURCE_TABLE_INDEX,
    HIT_ROW_WITH_PROVENANCE_WIDTH,
    cache_table_to_hit_table,
    coerce_float64_table,
)
from ra_sim.simulation.mosaic_profiles import cluster_beam_profiles
from ra_sim.debug_controls import (
    current_startup_debug_log_path,
    intersection_cache_logging_enabled,
    retain_optional_cache,
    resolve_intersection_cache_log_root,
)
from ra_sim.utils.calculations import (
    IndexofRefraction,
    SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG,
    complex_sqrt,
    fresnel_transmission,
)
from numba import types
from numba.typed import List  #  only List lives here


# Optical transport modes
# Canonical names:
#   - FRESNEL_CTR_DAMPING: fast Fresnel-weighted CTR damping path
#   - COMPLEX_K_DWBA_SLAB: precise complex-k slab optics path
FRESNEL_CTR_DAMPING = 0
COMPLEX_K_DWBA_SLAB = 1

# Backward-compatible aliases used throughout the existing codebase.
OPTICS_MODE_FAST = FRESNEL_CTR_DAMPING
OPTICS_MODE_EXACT = COMPLEX_K_DWBA_SLAB

# solve_q hot-loop constants
DEFAULT_SOLVE_Q_STEPS = 1000
MIN_SOLVE_Q_STEPS = 32
MAX_SOLVE_Q_STEPS = 8192
_DEFAULT_SOLVE_Q_DTHETA = (2.0 * np.pi) / DEFAULT_SOLVE_Q_STEPS
_DEFAULT_SOLVE_Q_COS = np.cos(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_DEFAULT_SOLVE_Q_SIN = np.sin(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_DEFAULT_SOLVE_Q_COS.setflags(write=False)
_DEFAULT_SOLVE_Q_SIN.setflags(write=False)
_CANONICAL_DEFAULT_SOLVE_Q_DTHETA = float(_DEFAULT_SOLVE_Q_DTHETA)
_CANONICAL_DEFAULT_SOLVE_Q_COS = np.array(_DEFAULT_SOLVE_Q_COS, copy=True)
_CANONICAL_DEFAULT_SOLVE_Q_SIN = np.array(_DEFAULT_SOLVE_Q_SIN, copy=True)
_CANONICAL_DEFAULT_SOLVE_Q_COS.setflags(write=False)
_CANONICAL_DEFAULT_SOLVE_Q_SIN.setflags(write=False)
DEFAULT_SOLVE_Q_BASE_INTERVALS = 48
MIN_SOLVE_Q_BASE_INTERVALS = 8
DEFAULT_SOLVE_Q_REL_TOL = 5.0e-4
MIN_SOLVE_Q_REL_TOL = 1.0e-6
MAX_SOLVE_Q_REL_TOL = 5.0e-2
SOLVE_Q_MODE_UNIFORM = 0
SOLVE_Q_MODE_ADAPTIVE = 1
DEFAULT_SOLVE_Q_MODE = SOLVE_Q_MODE_UNIFORM
_INTENSITY_CUTOFF = float(np.exp(-100.0))
_Q_RING_SAMPLE_MIN_MASS = 1.0e-5
_SOLVE_Q_ABS_ERR_TOL = 1.0e-20
_LOCAL_ARC_MAX_ROOTS = 4
_LOCAL_ARC_MAX_WINDOWS = 8
_LOCAL_ARC_MIN_SEARCH_STEPS = 64
_LOCAL_ARC_MAX_SEARCH_STEPS = 256
_LOCAL_ARC_MIN_STEPS_PER_WINDOW = 8
_LOCAL_ARC_GAUSS_SIGMAS = 10.0
_LOCAL_ARC_LORENTZ_GAMMAS = 24.0
_LOCAL_ARC_MIN_DTHETA = 5.0e-4
_LOCAL_ARC_FULL_CIRCLE_THETA_WINDOW = 0.75 * np.pi
_LOCAL_ARC_ROOT_TOL = 1.0e-10
_LOCAL_ARC_BOUNDARY_TOL = 1.0e-7
# Keep thread-local image buffers bounded to avoid runaway allocations.
_THREAD_LOCAL_IMAGE_MAX_BYTES = 768 * 1024 * 1024
_THREAD_LOCAL_MAX_IMAGE_SIZE = 1536
_THREAD_LOCAL_MERGE_WORK_FACTOR = 64.0
_LOCAL_PIXEL_CACHE_MIN_CAPACITY = 1024
_LOCAL_PIXEL_CACHE_MAX_CAPACITY = 32768
_LOCAL_PIXEL_CACHE_SCALE = 32
_LOCAL_PIXEL_CACHE_LOAD_NUM = 1
_LOCAL_PIXEL_CACHE_LOAD_DEN = 2
_FAST_OPTICS_LUT_SIZE = 96
_FAST_OPTICS_LUT_COLS = 4
_FAST_OPTICS_COL_TF2 = 0
_FAST_OPTICS_COL_IM_KZ = 1
_FAST_OPTICS_COL_L_OUT = 2
_FAST_OPTICS_COL_OUT_ANGLE = 3
_FAST_OPTICS_MAX_ANGLE = 0.5 * np.pi

# Per-sample precompute table columns (reflection-invariant terms).
_SAMPLE_COL_VALID = 0
_SAMPLE_COL_I_PLANE_X = 1
_SAMPLE_COL_I_PLANE_Y = 2
_SAMPLE_COL_I_PLANE_Z = 3
_SAMPLE_COL_KX_SCAT = 4
_SAMPLE_COL_KY_SCAT = 5
_SAMPLE_COL_RE_KZ = 6
_SAMPLE_COL_IM_KZ = 7
_SAMPLE_COL_K_SCAT = 8
_SAMPLE_COL_K0 = 9
_SAMPLE_COL_TI2 = 10
_SAMPLE_COL_L_IN = 11
_SAMPLE_COL_N2_REAL = 12
_SAMPLE_COL_SOLVE_Q_REP = 13
_SAMPLE_COL_SOLVE_Q_NEXT = 14
_SAMPLE_COLS = 15

EXIT_PROJECTION_INTERNAL = 0
EXIT_PROJECTION_REFRACTED = 1

_PROJECTION_DEBUG_COUNTER_TOTAL = 0
_PROJECTION_DEBUG_COUNTER_INVALID_DET = 1
_PROJECTION_DEBUG_COUNTER_PROP_ATT_REJECT = 2
_PROJECTION_DEBUG_COUNTER_PROP_FAC_REJECT = 3
_PROJECTION_DEBUG_COUNTER_NO_VALID_SAMPLE = 4
_PROJECTION_DEBUG_COUNTER_LARGE_EXIT_REMAP = 5
_PROJECTION_DEBUG_COUNTER_NEAR_CRITICAL_BAND = 6
_PROJECTION_DEBUG_COUNTER_COLS = 7

_PROJECTION_DEBUG_REASON_UNKNOWN = 0
_PROJECTION_DEBUG_REASON_NO_VALID_SAMPLE = 1
_PROJECTION_DEBUG_REASON_INVALID_DET = 2
_PROJECTION_DEBUG_REASON_PROP_ATT = 3
_PROJECTION_DEBUG_REASON_PROP_FAC = 4
_PROJECTION_DEBUG_REASON_PROJECTION_NORM = 5
_PROJECTION_DEBUG_REASON_INVALID_DET_COORD = 6
_PROJECTION_DEBUG_REASON_OFF_DETECTOR = 7
_PROJECTION_DEBUG_REJECT_RECORD_COLS = 10

_PROJECTION_DEBUG_LARGE_EXIT_REMAP_RAD = 0.05 * (pi / 180.0)
_PROJECTION_DEBUG_NEAR_CRITICAL_EPS_RAD = 0.01 * (pi / 180.0)


def _readonly_array_view(array):
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


def get_default_solve_q_trig_kwargs():
    """Return reusable default-step trig tables as runtime kwargs."""

    return {
        "default_solve_q_dtheta": float(_CANONICAL_DEFAULT_SOLVE_Q_DTHETA),
        "default_solve_q_cos": _readonly_array_view(_CANONICAL_DEFAULT_SOLVE_Q_COS),
        "default_solve_q_sin": _readonly_array_view(_CANONICAL_DEFAULT_SOLVE_Q_SIN),
    }


def get_process_peaks_runtime_kwargs(*, numba_thread_count=None):
    """Return cache-friendly runtime kwargs for ``process_peaks_parallel``."""

    kwargs = get_default_solve_q_trig_kwargs()
    thread_count = numba_thread_count
    if thread_count is None:
        try:
            thread_count = int(get_num_threads())
        except Exception:
            thread_count = 1
    if int(thread_count) < 1:
        thread_count = 1
    kwargs["numba_thread_count"] = int(thread_count)
    return kwargs


def _resolve_process_peaks_runtime_values(
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
    numba_thread_count,
):
    runtime_kwargs = get_process_peaks_runtime_kwargs(numba_thread_count=numba_thread_count)
    if (
        default_solve_q_cos is None
        and default_solve_q_sin is None
        and float(default_solve_q_dtheta) <= 0.0
    ):
        default_solve_q_dtheta = runtime_kwargs["default_solve_q_dtheta"]
        default_solve_q_cos = runtime_kwargs["default_solve_q_cos"]
        default_solve_q_sin = runtime_kwargs["default_solve_q_sin"]
    return (
        float(default_solve_q_dtheta),
        default_solve_q_cos,
        default_solve_q_sin,
        int(runtime_kwargs["numba_thread_count"]),
    )


def _uses_canonical_default_solve_q_trig_tables(
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
):
    if default_solve_q_cos is None or default_solve_q_sin is None:
        return False
    cos_arr = np.asarray(default_solve_q_cos, dtype=np.float64)
    sin_arr = np.asarray(default_solve_q_sin, dtype=np.float64)
    if cos_arr.shape != _CANONICAL_DEFAULT_SOLVE_Q_COS.shape:
        return False
    if sin_arr.shape != _CANONICAL_DEFAULT_SOLVE_Q_SIN.shape:
        return False
    dtheta = float(default_solve_q_dtheta)
    if dtheta <= 0.0:
        dtheta = float(_CANONICAL_DEFAULT_SOLVE_Q_DTHETA)
    return bool(
        np.isclose(
            dtheta,
            float(_CANONICAL_DEFAULT_SOLVE_Q_DTHETA),
            rtol=0.0,
            atol=1e-15,
        )
        and np.array_equal(cos_arr, _CANONICAL_DEFAULT_SOLVE_Q_COS)
        and np.array_equal(sin_arr, _CANONICAL_DEFAULT_SOLVE_Q_SIN)
    )


_PROCESS_PEAKS_PARALLEL_PARAM_NAMES = (
    "miller",
    "intensities",
    "image_size",
    "av",
    "cv",
    "lambda_",
    "image",
    "Distance_CoR_to_Detector",
    "gamma_deg",
    "Gamma_deg",
    "chi_deg",
    "psi_deg",
    "psi_z_deg",
    "zs",
    "zb",
    "n2",
    "beam_x_array",
    "beam_y_array",
    "theta_array",
    "phi_array",
    "sigma_pv_deg",
    "gamma_pv_deg",
    "eta_pv",
    "wavelength_array",
    "debye_x",
    "debye_y",
    "center",
    "theta_initial_deg",
    "cor_angle_deg",
    "unit_x",
    "n_detector",
    "save_flag",
    "record_status",
    "thickness",
    "optics_mode",
    "solve_q_steps",
    "solve_q_rel_tol",
    "solve_q_mode",
    "sample_weights",
    "best_sample_indices_out",
    "collect_hit_tables",
    "pixel_size_m",
    "sample_width_m",
    "sample_length_m",
    "n2_sample_array_override",
    "accumulate_image",
    "sample_qr_ring_once",
    "exit_projection_mode",
    "projection_debug_counters",
    "projection_debug_reject_counts",
    "projection_debug_reject_records",
    "projection_debug_row_hit_counts",
    "projection_debug_row_tthp_sums",
    "projection_debug_row_tth_sums",
    "default_solve_q_dtheta",
    "default_solve_q_cos",
    "default_solve_q_sin",
    "numba_thread_count",
    "events_per_beam_phase",
)

_PROCESS_PEAKS_PARALLEL_DEFAULTS = {
    "record_status": False,
    "thickness": 50e-9,
    "optics_mode": OPTICS_MODE_FAST,
    "solve_q_steps": DEFAULT_SOLVE_Q_STEPS,
    "solve_q_rel_tol": DEFAULT_SOLVE_Q_REL_TOL,
    "solve_q_mode": DEFAULT_SOLVE_Q_MODE,
    "sample_weights": None,
    "best_sample_indices_out": None,
    "collect_hit_tables": True,
    "pixel_size_m": 100e-6,
    "sample_width_m": 0.0,
    "sample_length_m": 0.0,
    "n2_sample_array_override": None,
    "accumulate_image": True,
    "sample_qr_ring_once": True,
    "exit_projection_mode": EXIT_PROJECTION_INTERNAL,
    "projection_debug_counters": None,
    "projection_debug_reject_counts": None,
    "projection_debug_reject_records": None,
    "projection_debug_row_hit_counts": None,
    "projection_debug_row_tthp_sums": None,
    "projection_debug_row_tth_sums": None,
    "default_solve_q_dtheta": -1.0,
    "default_solve_q_cos": None,
    "default_solve_q_sin": None,
    "numba_thread_count": None,
    "events_per_beam_phase": 50,
}


def normalize_events_per_beam_phase_backend(
    value,
    default=50,
    minimum=1,
    maximum=1000,
):
    """Normalize weighted-event count for backend execution."""

    try:
        count = int(value)
    except Exception:
        try:
            count = int(default)
        except Exception:
            count = 50
    lower = max(int(minimum), 1)
    upper = max(lower, int(maximum))
    if count < lower:
        return lower
    if count > upper:
        return upper
    return int(count)


@njit(fastmath=True, cache=True)
def _projection_debug_log_ray(
    projection_debug_counters,
    twotheta_t_prime,
    twotheta_t,
    n2_real,
):
    projection_debug_counters[_PROJECTION_DEBUG_COUNTER_TOTAL] += 1
    if abs(twotheta_t - twotheta_t_prime) > _PROJECTION_DEBUG_LARGE_EXIT_REMAP_RAD:
        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_LARGE_EXIT_REMAP] += 1
    alpha_c = np.arccos(_clamp(n2_real, -1.0, 1.0))
    if abs(twotheta_t) < (alpha_c + _PROJECTION_DEBUG_NEAR_CRITICAL_EPS_RAD):
        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_NEAR_CRITICAL_BAND] += 1


@njit(fastmath=True, cache=True)
def _projection_debug_record_reject(
    projection_debug_reject_count,
    projection_debug_reject_records,
    H,
    K,
    L,
    twotheta_t_prime,
    twotheta_t,
    kr,
    k0,
    n2_real,
    reason_code,
):
    reject_idx = int(projection_debug_reject_count[0])
    if reject_idx < projection_debug_reject_records.shape[0]:
        projection_debug_reject_records[reject_idx, 0] = H
        projection_debug_reject_records[reject_idx, 1] = K
        projection_debug_reject_records[reject_idx, 2] = L
        projection_debug_reject_records[reject_idx, 3] = reason_code
        projection_debug_reject_records[reject_idx, 4] = twotheta_t_prime
        projection_debug_reject_records[reject_idx, 5] = twotheta_t
        projection_debug_reject_records[reject_idx, 6] = kr
        projection_debug_reject_records[reject_idx, 7] = k0
        projection_debug_reject_records[reject_idx, 8] = n2_real
        projection_debug_reject_records[reject_idx, 9] = reason_code
    projection_debug_reject_count[0] = reject_idx + 1


def _should_log_intersection_cache() -> bool:
    return intersection_cache_logging_enabled()


def _resolve_intersection_cache_log_root() -> Path:
    return resolve_intersection_cache_log_root()


def _write_intersection_cache_log(
    cache: list[np.ndarray],
    *,
    av: object,
    cv: object,
    beam_x_center: float,
    beam_y_center: float,
    theta_center: float,
    phi_center: float,
    wavelength_center: float,
    source_label: str | None = None,
    cache_metadata: dict[str, object] | None = None,
) -> None:
    if not _should_log_intersection_cache():
        return
    if cache is None:
        cache = []

    log_root = _resolve_intersection_cache_log_root() / "intersection_cache"
    log_dir = log_root / (f"intersection_cache_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    row_counts = []
    cache_tables = []
    table_arrays: list[np.ndarray] = []
    table_files: list[str] = []
    for idx, table in enumerate(cache):
        table_arr = coerce_float64_table(
            table,
            empty_width=CURRENT_DETECTOR_CACHE_WIDTH,
        )
        table_name = f"table_{int(idx):04d}.npy"
        row_counts.append(int(table_arr.shape[0]))
        table_arrays.append(np.asarray(table_arr, dtype=np.float64).copy())
        table_files.append(table_name)
        cache_tables.append(
            {
                "index": int(idx),
                "row_count": int(table_arr.shape[0]),
                "table_file": table_name,
            }
        )

    metadata = {
        "created_at": datetime.now().isoformat(),
        "created_at_unix": float(time.time()),
        "source": source_label or "build_intersection_cache",
        "reused": False,
        "rebuilt": True,
        "cache_action": "rebuilt",
        "stale_reason": None,
        "cache_source": source_label or "build_intersection_cache",
        "cache_provenance": {
            "grouping": "nominal_bragg_family",
            "collapse": "detector_branch_top_mosaic_rows",
            "beam_context": "per_hit_sample_index_preferred",
        },
        "av": float(av) if isinstance(av, (int, float, np.floating)) else None,
        "cv": float(cv) if isinstance(cv, (int, float, np.floating)) else None,
        "beam_x_center": beam_x_center,
        "beam_y_center": beam_y_center,
        "theta_center": theta_center,
        "phi_center": phi_center,
        "wavelength_center": wavelength_center,
        "table_count": int(len(cache)),
        "rows_per_table": row_counts,
        "total_rows": int(sum(int(v) for v in row_counts)),
        "cache_tables": cache_tables,
        "table_files": table_files,
        "log_root": str(log_root),
        "log_dir": str(log_dir),
        "meta_path": str(log_dir / "meta.json"),
    }
    if isinstance(cache_metadata, dict) and cache_metadata:
        metadata.update(cache_metadata)
    metadata.setdefault(
        "signature_digest",
        hashlib.blake2b(
            json.dumps(
                {
                    "av": metadata.get("av"),
                    "cv": metadata.get("cv"),
                    "theta_center": metadata.get("theta_center"),
                    "phi_center": metadata.get("phi_center"),
                    "wavelength_center": metadata.get("wavelength_center"),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest(),
    )

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        for table_name, table_arr in zip(table_files, table_arrays):
            np.save(log_dir / table_name, table_arr, allow_pickle=False)
        with (log_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[intersection_cache] wrote cache to: {log_dir}")
        startup_log_path = current_startup_debug_log_path()
        if startup_log_path is not None:
            summary = {
                "log_dir": str(log_dir),
                "meta_path": str(log_dir / "meta.json"),
                "source": metadata.get("source"),
                "table_count": metadata.get("table_count"),
                "cache_tables": metadata.get("cache_tables", []),
            }
            needs_separator = startup_log_path.exists() and startup_log_path.stat().st_size > 0
            startup_log_path.parent.mkdir(parents=True, exist_ok=True)
            with startup_log_path.open("a", encoding="utf-8") as handle:
                if needs_separator:
                    handle.write("\n")
                handle.write("Intersection cache updated:\n")
                json.dump(summary, handle, indent=2, sort_keys=True)
                handle.write("\n")
    except Exception:
        return


_EMPTY_PROCESS_PEAKS_SAFE_STATS = {
    "used_safe_cache": False,
    "used_python_runner": False,
    "source_templates_built": 0,
    "source_templates_reused": 0,
    "rays_reused": 0,
}

_PHASE_SPACE_CACHE = {}
_SOURCE_TEMPLATE_CACHE = {}
_Q_VECTOR_CACHE = {}
_LAST_PROCESS_PEAKS_SAFE_STATS = dict(_EMPTY_PROCESS_PEAKS_SAFE_STATS)
_LAST_INTERSECTION_CACHE = []
_LAST_INTERSECTION_CACHE_VIEWS = {
    "sampled_event_rows": [],
    "branch_representative_rows": [],
}
_LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES = None
_EMPTY_PROCESS_PEAKS_WEIGHTED_EVENT_STATS = {
    "n_solve_q_calls": 0,
    "n_project_candidate_calls": 0,
    "n_valid_candidates": 0,
    "n_selected_events": 0,
    "pass2_mass_mismatch_count": 0,
    "pass2_mass_mismatch_max_abs": 0.0,
    "tail_fill_events": 0,
    "time_precompute": 0.0,
    "time_solve_q": 0.0,
    "time_project": 0.0,
    "time_select": 0.0,
    "time_emit_cache": 0.0,
    "pass1_total_mass": 0.0,
    "pass2_total_mass": 0.0,
}
_LAST_PROCESS_PEAKS_WEIGHTED_EVENT_STATS = dict(_EMPTY_PROCESS_PEAKS_WEIGHTED_EVENT_STATS)


def _retain_diffraction_safe_cache() -> bool:
    return retain_optional_cache("diffraction_safe", feature_needed=True)


def _retain_last_intersection_cache() -> bool:
    return retain_optional_cache(
        "diffraction_last_intersection",
        feature_needed=False,
    )


def _set_last_process_peaks_weighted_event_stats(**updates):
    global _LAST_PROCESS_PEAKS_WEIGHTED_EVENT_STATS
    stats = dict(_EMPTY_PROCESS_PEAKS_WEIGHTED_EVENT_STATS)
    stats.update(updates)
    _LAST_PROCESS_PEAKS_WEIGHTED_EVENT_STATS = stats


def get_last_process_peaks_weighted_event_stats():
    return dict(_LAST_PROCESS_PEAKS_WEIGHTED_EVENT_STATS)


# =============================================================================
# 1) FINITE-STACK INTERFERENCE FOR N LAYERS
# =============================================================================


@njit(cache=True)
def attenuation(N, Qz, c):
    """
    Compute the coherent interference (Kiessig-like) from a stack of N layers,
    each of thickness 'c'. Qz is a complex wavevector transfer along z:
        Qz = Qz.real + i * Qz.imag

    Physically:
      - Qz.imag encodes absorption or evanescent decay,
      - Qz.real encodes the wave's momentum transfer in z.

    The formula is:
        |S|^2 = [1 - 2*exp(-2*N*A)*cos(2*N*B) + exp(-4*N*A)]
                 / [1 - 2*exp(-2*A)*cos(2*B) + exp(-4*A)]
      where A = Qz.imag * c / 2, B = Qz.real * c / 2.

    Returns the intensity factor for the finite stack, capturing thickness fringes.
    """
    A = (Qz.imag * c) / 2.0
    B = (Qz.real * c) / 2.0
    num = 1.0 - 2.0 * np.exp(-2.0 * N * A) * np.cos(2.0 * N * B) + np.exp(-4.0 * N * A)
    den = 1.0 - 2.0 * np.exp(-2.0 * A) * np.cos(2.0 * B) + np.exp(-4.0 * A)
    if abs(den) < 1e-12:
        # If near zero in the denominator, we take the limit ~ N^2
        return N**2
    return num / den


@njit(parallel=True, cache=True)
def attenuation_array(N, Qz_array, c):
    """
    Vectorized version of attenuation(N, Qz, c) for an entire array Qz_array.
    Runs in parallel using prange for efficiency.

    Parameters
    ----------
    N : int
        Number of layers in the stack.
    Qz_array : ndarray of complex128
        Complex wavevector transfers for which to compute the finite-stack
        interference factor.
    c : float
        Single-layer thickness (Å).

    Returns
    -------
    out : ndarray of float64
        The computed intensities from the finite stack for each Qz in Qz_array.
    """
    n = Qz_array.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = attenuation(N, Qz_array[i], c)
    return out


# =============================================================================
# 2) PARALLEL CUSTOM MESHGRID
# =============================================================================


@njit(parallel=True, cache=True)
def custom_meshgrid(qx_vals, qy_vals, qz_vals):
    """
    Parallel replacement for np.meshgrid in 3D. Builds Qx, Qy, Qz arrays of
    shape (len(qx_vals), len(qy_vals), len(qz_vals)).

    Physically:
      - We are building a reciprocal-space grid around a nominal G_vec or region.
      - Each point (Qx, Qy, Qz) is one point in the 3D mosaic distribution.

    Returns
    -------
    Qx, Qy, Qz : 3D arrays
        The coordinate grids for Qx, Qy, Qz.
    """
    nx, ny, nz = len(qx_vals), len(qy_vals), len(qz_vals)
    Qx = np.empty((nx, ny, nz), dtype=qx_vals.dtype)
    Qy = np.empty((nx, ny, nz), dtype=qy_vals.dtype)
    Qz = np.empty((nx, ny, nz), dtype=qz_vals.dtype)

    for i in prange(nx):
        qx = qx_vals[i]
        Qx[i, :, :] = qx
        Qz[i, :, :] = qz_vals
        for j in range(ny):
            Qy[i, j, :] = qy_vals[j]
    return Qx, Qy, Qz


# =============================================================================
# 3) PARALLEL SAMPLE_FROM_PDF
# =============================================================================


@njit(parallel=True, cache=True)
def sample_from_pdf(Qx_grid, Qy_grid, Qz_grid, pdf_3d, n_samples):
    """
    Sample from a 3D probability density function (pdf_3d). This is a parallel
    version that uses prange. Each iteration draws a random number and picks a
    location from the cumulative distribution.

    Physically:
      - pdf_3d is the mosaic orientation distribution in reciprocal space.
      - We are generating random Qx, Qy, Qz from that distribution, each representing
        a possible microcrystallite orientation or partial scattering vector.

    Parameters
    ----------
    Qx_grid, Qy_grid, Qz_grid : 3D arrays
        The coordinate grids from custom_meshgrid.
    pdf_3d : 3D array
        Probability density at each (Qx, Qy, Qz).
    n_samples : int
        Number of random samples to draw.

    Returns
    -------
    (out_Qx, out_Qy, out_Qz) : tuple of 1D arrays
        The random draws from pdf_3d in flattened form.
    """
    Nx, Ny, Nz = Qx_grid.shape
    pdf_flat = pdf_3d.ravel()

    sum_pdf = np.sum(pdf_flat)
    if sum_pdf < 1e-14:
        # No distribution means everything is near zero => return empty arrays
        return (np.zeros(0), np.zeros(0), np.zeros(0))

    # Normalize
    pdf_flat /= sum_pdf

    # Build a cumulative distribution
    cdf_flat = np.cumsum(pdf_flat)

    out_Qx = np.empty(n_samples, dtype=np.float64)
    out_Qy = np.empty(n_samples, dtype=np.float64)
    out_Qz = np.empty(n_samples, dtype=np.float64)

    # Parallel sampling loop
    for i in prange(n_samples):
        r = np.random.rand()  # random uniform in [0,1)
        idx = np.searchsorted(cdf_flat, r)
        if idx == pdf_flat.size:
            idx = pdf_flat.size - 1

        iz = idx % Nz
        iy = (idx // Nz) % Ny
        ix = (idx // (Ny * Nz)) % Nx

        out_Qx[i] = Qx_grid[ix, iy, iz]
        out_Qy[i] = Qy_grid[ix, iy, iz]
        out_Qz[i] = Qz_grid[ix, iy, iz]

    return (out_Qx, out_Qy, out_Qz)


@njit(cache=True)
def wrap_to_pi(x):
    while x <= -pi:
        x += 2.0 * pi
    while x > pi:
        x -= 2.0 * pi
    return x


@njit(parallel=True, cache=True)
def compute_intensity_array(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv):
    """
    Compute the mosaic surface density sigma(theta) on the Bragg sphere for
    each (Qx, Qy, Qz). Uses a pseudo-Voigt in the grazing-angle offset.

    Parameters
    ----------
    Qx, Qy, Qz : array-like
        Coordinates of Q vectors.
    G_vec : length-3 array
        The reciprocal-space vector for the reflection.
    sigma : float
        Gaussian width (rad).
    gamma_pv : float
        Lorentzian half-width at half-maximum (rad).
    eta_pv : float
        Mixing parameter (0=Gaussian, 1=Lorentzian).

    Returns
    -------
    intensities : array-like
        Surface density sigma(theta), same shape as Qx.
    """
    # Unpack G and compute magnitudes
    Gx, Gy, Gz = G_vec[0], G_vec[1], G_vec[2]
    G_mag = np.sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
    if G_mag < 1e-14:
        return np.zeros_like(Qx)

    Qr = np.sqrt(Qx * Qx + Qy * Qy)

    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    # Amplitude factors for normalized 1D profiles
    A_gauss = 1.0 / (sigma_eff * np.sqrt(2.0 * np.pi))
    A_lor = 1.0 / (np.pi * gamma_eff)

    # Reference grazing angle for the reflection
    Gr = np.sqrt(Gx * Gx + Gy * Gy)
    theta0 = np.arctan2(Gz, Gr)

    denom_base = 2.0 * np.pi * G_mag * G_mag

    intensities = np.empty_like(Qx)
    Qz_flat = Qz.ravel()
    Qr_flat = Qr.ravel()
    out_flat = intensities.ravel()

    for i in prange(out_flat.size):
        theta = np.arctan2(Qz_flat[i], Qr_flat[i])
        dtheta = wrap_to_pi(theta - theta0)

        gauss_val = A_gauss * np.exp(-0.5 * (dtheta / sigma_eff) ** 2)
        lor_val = A_lor / (1.0 + (dtheta / gamma_eff) ** 2)
        omega = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

        # Keep a geometry normalization that is stable for pseudo-Voigt tails.
        # The previous 1/cos(theta) factor caused pole amplification when eta>0
        # (Lorentzian component), which collapsed Bragg-sphere color scales.
        out_flat[i] = omega / denom_base

    return intensities


@njit(cache=True)
def Generate_PDF_Grid(
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    Qrange=0.1,  # default value; will be overridden if dynamic_Qrange is True
    n_grid=51,  # grid resolution
    n_samples=10000,  # number of samples
    dynamic_Qrange=True,  # new flag to control dynamic Qrange selection
    multiplier=3.0,  # factor to extend the half-width (adjust as needed)
):
    # Compute effective FWHM in angle (radians)
    # For a Gaussian, FWHM = 2.35482*sigma
    # For a Lorentzian, FWHM = 2*gamma_pv
    fwhm_gauss = 2.35482 * sigma
    fwhm_lorentz = 2.0 * gamma_pv
    effective_fwhm = (1 - eta_pv) * fwhm_gauss + eta_pv * fwhm_lorentz

    # Use half of the effective FWHM as the angular half-width
    effective_angle_half_width = effective_fwhm / 2.0

    # Convert the angular half-width into Q-space using |G_vec|
    G_mag = sqrt(G_vec[0] ** 2 + G_vec[1] ** 2 + G_vec[2] ** 2)
    effective_Q_half_width = G_mag * effective_angle_half_width

    # If dynamic Qrange is enabled, override the Qrange value
    if dynamic_Qrange:
        Qrange = multiplier * effective_Q_half_width

    # Build grid in reciprocal space centered around G_vec
    Gx, Gy, Gz = G_vec
    qx_vals = np.linspace(Gx - Qrange, Gx + Qrange, n_grid)
    qy_vals = np.linspace(Gy - Qrange, Gy + Qrange, n_grid)
    qz_vals = np.linspace(Gz - Qrange, Gz + Qrange, n_grid)

    # 1) Build the meshgrid in parallel
    Qx_grid, Qy_grid, Qz_grid = custom_meshgrid(qx_vals, qy_vals, qz_vals)

    # 2) Evaluate mosaic distribution => pdf_3d
    pdf_3d = compute_intensity_array(Qx_grid, Qy_grid, Qz_grid, G_vec, sigma, gamma_pv, eta_pv)

    # 3) Sample from this 3D pdf
    Qx_s, Qy_s, Qz_s = sample_from_pdf(Qx_grid, Qy_grid, Qz_grid, pdf_3d, n_samples)

    Q_grid = np.stack((Qx_s, Qy_s, Qz_s), axis=1)
    return Q_grid


# =============================================================================
# 5) VECTORIZED INCOHERENT AVERAGING
# =============================================================================


@njit(cache=True)
def incoherent_averaging(Q_grid, N, c, thickness, re_k_z, im_k_z, k_in_crystal, k_mag, n2, bt):
    """
    For a mosaic-sampled Q_grid, compute the average finite-stack interference
    by building Qz_complex for each sample and calling attenuation.

    Physical steps:
      1) For each mosaic point (Qx, Qy, Qz), offset by the incident wave in
         crystal (k_in_crystal) to get the scattered wavevector in the crystal.
      2) Convert to angles, build final Qz_complex with real_k_tz_f, im_k_tz_f.
      3) Summation of attenuation(...) across all mosaic samples (incoherent sum).

    Parameters
    ----------
    Q_grid : ndarray (n_samples, 3)
        Random mosaic points in Q. Q_grid[i] = (Qx_i, Qy_i, Qz_i).
    N : int
        Number of layers in the finite stack.
    c : float
        Single-layer thickness.
    thickness : float
        Full film thickness (not always used explicitly if N*c is thickness).
    re_k_z, im_k_z : float
        Real and imaginary parts of the incident wavevector's z-component.
    k_in_crystal : array_like (3,)
        The in-plane scattering wavevector for the crystal domain.
    k_mag : float
        Magnitude of the wavevector in vacuum 2π/λ.
    n2 : complex
        Refractive index factor used to compute scattering angles.
    bt : float
        A leftover factor from the absorption imaginary part: (Im(n2))^2 * k_mag^2.

    Returns
    -------
    average_int : float
        The average (incoherent) intensity factor from the mosaic sum.
    """
    Qx_s = Q_grid[:, 0]
    Qy_s = Q_grid[:, 1]
    Qz_s = Q_grid[:, 2]

    # Add the crystal wavevector offset
    kx = Qx_s + k_in_crystal[0]
    ky = Qy_s + k_in_crystal[1]
    kz = Qz_s + re_k_z

    # Magnitude in-plane => twotheta
    kr = np.sqrt(kx * kx + ky * ky)
    twotheta = np.arctan2(kz, kr)

    af = k_mag**2 * (np.real(n2) * np.real(n2)) * np.sin(twotheta) ** 2

    real_k_tz_f = np.sqrt((np.sqrt(af * af + bt * bt) + af) / 2.0)
    im_k_tz_f = np.sqrt((np.sqrt(af * af + bt * bt) - af) / 2.0)

    n_samples = Qx_s.size
    Qz_complex_arr = np.empty(n_samples, dtype=np.complex128)

    # Build array of Qz for each sample
    for i in range(n_samples):
        Qz_complex_arr[i] = (real_k_tz_f[i] - re_k_z) - 1j * (im_k_z + im_k_tz_f[i])

    # Evaluate finite-stack interference for all samples
    att_vals = attenuation_array(N, Qz_complex_arr, c)
    total_int = np.sum(att_vals)

    # Return the average (incoherent) intensity
    return total_int / n_samples


# =============================================================================
# 3) INTERSECT_LINE_PLANE, BATCH
# =============================================================================


@njit(cache=True)
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a single ray (start=P0, direction=k_vec) with a plane
    defined by (P_plane, n_plane). Returns the intersection point (ix, iy, iz)
    and a boolean if valid.

    Physical meaning:
      - Used to find where the scattered beam intersects e.g. the sample plane
        or a detector plane in real space.
    """
    denom = k_vec[0] * n_plane[0] + k_vec[1] * n_plane[1] + k_vec[2] * n_plane[2]
    if abs(denom) < 1e-14:
        # The ray is parallel to the plane. If the starting point already lies
        # on the plane (within a tolerance) we treat it as the intersection
        # point so that grazing rays are not discarded.
        dist = (
            (P0[0] - P_plane[0]) * n_plane[0]
            + (P0[1] - P_plane[1]) * n_plane[1]
            + (P0[2] - P_plane[2]) * n_plane[2]
        )
        if abs(dist) < 1e-6:
            return (P0[0], P0[1], P0[2], True)
        return (np.nan, np.nan, np.nan, False)
    num = (
        (P_plane[0] - P0[0]) * n_plane[0]
        + (P_plane[1] - P0[1]) * n_plane[1]
        + (P_plane[2] - P0[2]) * n_plane[2]
    )
    t = num / denom
    # Numerical precision can yield tiny negative values for *t* when the ray
    # should intersect exactly on the plane.  Allow a small tolerance so these
    # near-zero cases are not discarded which previously produced missing bands
    # on the detector when the beam was almost parallel to the sample plane.
    if t < -1e-9:
        return (np.nan, np.nan, np.nan, False)
    if t < 0.0:
        t = 0.0
    ix = P0[0] + t * k_vec[0]
    iy = P0[1] + t * k_vec[1]
    iz = P0[2] + t * k_vec[2]
    return (ix, iy, iz, True)


@njit(cache=True)
def intersect_infinite_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect an infinite line (start=P0, direction=k_vec) with a plane.
    Unlike ``intersect_line_plane`` this does not reject negative t values.
    """
    denom = k_vec[0] * n_plane[0] + k_vec[1] * n_plane[1] + k_vec[2] * n_plane[2]
    if abs(denom) < 1e-14:
        # Parallel direction: project start point orthogonally onto the plane
        # instead of declaring "no hit", avoiding artificial horizon cutoffs.
        dist = (
            (P0[0] - P_plane[0]) * n_plane[0]
            + (P0[1] - P_plane[1]) * n_plane[1]
            + (P0[2] - P_plane[2]) * n_plane[2]
        )
        n_sq = n_plane[0] * n_plane[0] + n_plane[1] * n_plane[1] + n_plane[2] * n_plane[2]
        if n_sq < 1e-20:
            return (np.nan, np.nan, np.nan, False)
        scale = dist / n_sq
        ix = P0[0] - scale * n_plane[0]
        iy = P0[1] - scale * n_plane[1]
        iz = P0[2] - scale * n_plane[2]
        return (ix, iy, iz, True)

    num = (
        (P_plane[0] - P0[0]) * n_plane[0]
        + (P_plane[1] - P0[1]) * n_plane[1]
        + (P_plane[2] - P0[2]) * n_plane[2]
    )
    t = num / denom
    ix = P0[0] + t * k_vec[0]
    iy = P0[1] + t * k_vec[1]
    iz = P0[2] + t * k_vec[2]
    return (ix, iy, iz, True)


@njit(cache=True)
def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    """
    Batch version: intersect multiple directions with a plane.

    directions : shape (N,3)
        Each row is a vector to test.

    Returns
    -------
    intersects : shape (N,3)
        Intersection points (ix, iy, iz).
    valid : boolean array of length N
        True if the intersection is valid (t>=0, not parallel).
    """
    Ndir = directions.shape[0]
    intersects = np.full((Ndir, 3), np.nan, dtype=np.float64)
    valid = np.zeros(Ndir, dtype=np.bool_)

    dist = (
        (start_pt[0] - plane_pt[0]) * plane_n[0]
        + (start_pt[1] - plane_pt[1]) * plane_n[1]
        + (start_pt[2] - plane_pt[2]) * plane_n[2]
    )

    for i in range(Ndir):
        kx = directions[i, 0]
        ky = directions[i, 1]
        kz = directions[i, 2]
        dot_dn = kx * plane_n[0] + ky * plane_n[1] + kz * plane_n[2]
        if abs(dot_dn) < 1e-14:
            if abs(dist) < 1e-6:
                intersects[i, 0] = start_pt[0]
                intersects[i, 1] = start_pt[1]
                intersects[i, 2] = start_pt[2]
                valid[i] = True
            continue
        num = (
            (plane_pt[0] - start_pt[0]) * plane_n[0]
            + (plane_pt[1] - start_pt[1]) * plane_n[1]
            + (plane_pt[2] - start_pt[2]) * plane_n[2]
        )
        t = num / dot_dn
        if t < -1e-9:
            continue
        if t < 0.0:
            t = 0.0
        ix = start_pt[0] + t * kx
        iy = start_pt[1] + t * ky
        iz = start_pt[2] + t * kz
        intersects[i, 0] = ix
        intersects[i, 1] = iy
        intersects[i, 2] = iz
        valid[i] = True
    return intersects, valid


# ---------- NEW JIT-SAFE HELPERS ----------
@njit(cache=True)
def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _kz_branch_decay(arg):
    """Return sqrt(arg) with the physically decaying branch (Im(kz) >= 0)."""
    kz = complex_sqrt(arg)
    if kz.imag < 0.0:
        kz = -kz
    elif abs(kz.imag) < 1e-15 and kz.real < 0.0:
        kz = -kz
    return kz


@njit(cache=True)
def _fresnel_t_exact(kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Exact Fresnel transmission amplitude in kz/epsilon form."""
    if s_polarization:
        den = kz_i + kz_j
        if abs(den) < 1e-30:
            return 0j
        return (2.0 * kz_i) / den

    den = eps_j * kz_i + eps_i * kz_j
    if abs(den) < 1e-30:
        return 0j
    return (2.0 * eps_j * kz_i) / den


@njit(cache=True)
def _fresnel_power_t_exact(t_amp, kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Convert exact transmission amplitude to power transmission."""
    abs_t2 = t_amp.real * t_amp.real + t_amp.imag * t_amp.imag
    if abs_t2 <= 0.0:
        return 0.0

    if s_polarization:
        denom = kz_i.real
        if abs(denom) < 1e-30:
            return 0.0
        ratio = kz_j.real / denom
    else:
        den = kz_i / eps_i
        if abs(den) < 1e-30:
            return 0.0
        ratio = ((kz_j / eps_j) / den).real

    out = ratio * abs_t2
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    # For passive interfaces the transmitted power fraction should stay bounded.
    if out > 1.0:
        return 1.0
    return out


@njit(cache=True)
def _sanitize_transmission_power(power):
    """Clamp transmission-like power factors to a stable physical range."""
    if not np.isfinite(power) or power <= 0.0:
        return 0.0
    if power > 1.0:
        return 1.0
    return power


@njit(cache=True)
def _thickness_to_angstrom(depth):
    """Interpret sub-mm values as meters and convert to angstrom."""
    if depth <= 0.0:
        return 0.0
    # Existing defaults pass thickness in meters (e.g., 50e-9).
    if depth < 1e-3:
        return depth * 1.0e10
    # Larger values are assumed to already be in angstrom.
    return depth


@njit(cache=True)
def _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_layers):
    """
    Finite discrete CTR:
      |sum_{l=0}^{N-1} exp(i (qz + iV) c l)|^2
    with V >= 0 and c in angstrom.
    """
    if n_layers <= 0 or c_ang <= 0.0:
        return 0.0

    v_eff = np.maximum(v_sum, 0.0)
    phase = qz * c_ang
    rho = np.exp(-v_eff * c_ang)
    rho_n = rho**n_layers

    num = 1.0 - 2.0 * rho_n * np.cos(n_layers * phase) + rho_n * rho_n
    den = 1.0 - 2.0 * rho * np.cos(phase) + rho * rho
    if den < 1e-30:
        # Bragg-pole limit of the finite geometric series.
        return float(n_layers * n_layers)

    out = num / den
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _ctr_discrete_infinite_intensity(qz, v_sum, c_ang):
    """
    Infinite discrete CTR:
      |sum_{l=0}^{inf} exp(i (qz + iV) c l)|^2
      = 1 / |1 - exp(i (qz + iV) c)|^2
    where V > 0 for convergence.
    """
    if c_ang <= 0.0:
        return 1.0

    # Keep convergence explicit even if v_sum is numerically zero.
    v_eff = np.maximum(v_sum, 1e-12 / np.maximum(c_ang, 1e-30))
    phase = qz * c_ang
    rho = np.exp(-v_eff * c_ang)

    den = 1.0 - 2.0 * rho * np.cos(phase) + rho * rho
    if den < 1e-30:
        den = 1e-30

    out = 1.0 / den
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _ctr_effective_layers_from_absorption(v_sum, c_ang):
    """
    Semi-infinite fallback depth in layer units.

    For n_layers<=0, use an absorption-limited effective finite depth to avoid
    the undamped infinite-crystal pole in I(q,0), which otherwise drives the
    attenuation ratio to zero near q=0.
    """
    if c_ang <= 0.0:
        return 1

    vc = np.maximum(v_sum, 0.0) * c_ang
    if vc <= 1e-6:
        n_eff = 1000000
    else:
        n_eff = int(np.ceil(1.0 / vc))

    if n_eff < 1:
        return 1
    if n_eff > 1000000:
        return 1000000
    return n_eff


@njit(cache=True)
def _ctr_attenuation_factor(qz, v_in, v_out, c_ang, n_layers):
    """
    CTR-only absorption correction for fast mode.

    Use the same discrete CTR model with and without damping at the same qz:

      finite:    f(qz) = I_N(qz, V)   / I_N(qz, 0)
      semi-infinite fallback (n_layers<=0):
                 f(qz) = I_Neff(qz, V) / I_Neff(qz, 0)
                 with Neff set by absorption depth

    where V = v_in + v_out >= 0.
    """
    if c_ang <= 0.0:
        return 1.0

    v_sum = np.maximum(v_in + v_out, 0.0)
    if n_layers > 0:
        i_v = _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_layers)
        i_0 = _ctr_discrete_finite_intensity(qz, 0.0, c_ang, n_layers)
    else:
        # Avoid pole-driven collapse at qz~0 for an undamped infinite
        # denominator by using an absorption-limited effective finite depth.
        n_eff = _ctr_effective_layers_from_absorption(v_sum, c_ang)
        i_v = _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_eff)
        i_0 = _ctr_discrete_finite_intensity(qz, 0.0, c_ang, n_eff)

    if i_0 < 1e-30:
        i_0 = 1e-30

    out = i_v / i_0
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def transmit_angle_grazing(theta_i_plane, n2):
    """
    Grazing angle form of Snell: cos(theta_t) = cos(theta_i) / Re(n2)
    Inputs/outputs are grazing angles with respect to the surface plane.
    """
    c = np.cos(theta_i_plane) / np.maximum(np.real(n2), 1e-12)
    c = _clamp(c, -1.0, 1.0)
    # preserve sign of the incident grazing angle
    return np.arccos(c) * np.sign(theta_i_plane)


@njit(cache=True)
def ktz_components(k0, n2, theta_t_plane):
    """
    Decompose K_tz for a complex refractive index.
    Uses: a = Re(n^2) k0^2 sin^2(theta_t), b = Im(n^2) k0^2
    Returns positive magnitudes (sign handled at call site).
    """
    n2_sq = n2 * n2
    a = np.real(n2_sq) * (k0 * k0) * (np.sin(theta_t_plane) ** 2)
    b = np.abs(np.imag(n2_sq)) * (k0 * k0)  # magnitude for stability
    root = np.sqrt(a * a + b * b)
    re_kz = np.sqrt(0.5 * (root + a))
    im_kz = np.sqrt(0.5 * (root - a))
    return re_kz, im_kz


@njit(cache=True)
def safe_path_length(thickness_m, theta_plane):
    """
    Path length through a slab of thickness_m for a ray with grazing angle
    theta_plane (with respect to the plane). For semi-infinite thickness_m<=0,
    returns 0 to signal use of penetration depth.
    """
    if thickness_m <= 0.0:
        return 0.0
    s = np.abs(np.sin(theta_plane))
    if s < 1e-12:
        s = 1e-12
    return thickness_m / s


@njit(cache=True)
def _choose_local_pixel_cache_capacity(n_samp):
    desired = n_samp * _LOCAL_PIXEL_CACHE_SCALE
    if desired < _LOCAL_PIXEL_CACHE_MIN_CAPACITY:
        desired = _LOCAL_PIXEL_CACHE_MIN_CAPACITY
    if desired > _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        desired = _LOCAL_PIXEL_CACHE_MAX_CAPACITY

    capacity = _LOCAL_PIXEL_CACHE_MIN_CAPACITY
    while capacity < desired and capacity < _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        capacity *= 2
    if capacity > _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        capacity = _LOCAL_PIXEL_CACHE_MAX_CAPACITY
    return capacity


@njit(cache=True)
def _clear_local_pixel_cache(cache_keys, cache_values):
    for i in range(cache_keys.shape[0]):
        cache_keys[i] = -1
        cache_values[i] = 0.0


@njit(cache=True)
def _flush_local_pixel_cache(image, image_size, cache_keys, cache_values):
    for i in range(cache_keys.shape[0]):
        flat_idx = cache_keys[i]
        if flat_idx < 0:
            continue
        row = flat_idx // image_size
        col = flat_idx - row * image_size
        image[row, col] += cache_values[i]
        cache_keys[i] = -1
        cache_values[i] = 0.0
    return 0


@njit(parallel=True, cache=True)
def _merge_thread_local_images(image, image_partials):
    for r in prange(image.shape[0]):
        for c in range(image.shape[1]):
            total = image[r, c]
            for tid in range(image_partials.shape[0]):
                total += image_partials[tid, r, c]
            image[r, c] = total


@njit(cache=True)
def _copy_scaled_hit_table(src_hits, scale, H, K, L):
    n_src_hits = src_hits.shape[0]
    src_width = src_hits.shape[1]
    pixel_hits = np.empty((n_src_hits, src_width), dtype=np.float64)
    if n_src_hits <= 0:
        return pixel_hits
    for row_idx in range(n_src_hits):
        for col_idx in range(src_width):
            pixel_hits[row_idx, col_idx] = src_hits[row_idx, col_idx]
        pixel_hits[row_idx, 0] *= scale
        pixel_hits[row_idx, 4] = H
        pixel_hits[row_idx, 5] = K
        pixel_hits[row_idx, 6] = L
    return pixel_hits


@njit(cache=True)
def _copy_miss_table(src_miss):
    n_src_miss = src_miss.shape[0]
    missed_arr = np.empty((n_src_miss, 3), dtype=np.float64)
    if n_src_miss <= 0:
        return missed_arr
    missed_arr[:, :] = src_miss
    return missed_arr


@njit(cache=True)
def _copy_scaled_q_rows(q_data, dst_idx, src_idx, qn, scale):
    if qn <= 0:
        return
    q_data[dst_idx, :qn, :] = q_data[src_idx, :qn, :]
    q_data[dst_idx, :qn, 3] *= scale


@njit(cache=True)
def _insert_local_pixel_cache(cache_keys, cache_values, flat_idx, value):
    capacity = cache_keys.shape[0]
    mask = capacity - 1
    slot = flat_idx & mask
    for _ in range(capacity):
        key = cache_keys[slot]
        if key == -1:
            cache_keys[slot] = flat_idx
            cache_values[slot] = value
            return True, 1
        if key == flat_idx:
            cache_values[slot] += value
            return True, 0
        slot = (slot + 1) & mask
    return False, 0


@njit(cache=True)
def _accumulate_bilinear_cached(
    image_size,
    row_f,
    col_f,
    value,
    cache_keys,
    cache_values,
    entry_count,
    flush_limit,
):
    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)
    contrib_count = 0

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            contrib_count += 1

    if contrib_count == 0:
        return False, False, entry_count
    if entry_count + contrib_count > flush_limit:
        return True, True, entry_count

    new_count = entry_count
    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            ok, added = _insert_local_pixel_cache(
                cache_keys,
                cache_values,
                rr * image_size + cc,
                value * w_row * w_col,
            )
            if not ok:
                return True, True, entry_count
            new_count += added
    return True, False, new_count


@njit(cache=True)
def _build_fast_optics_lut_row(lut_row, k0, n2_samp, n2_real, thickness):
    lut_size = lut_row.shape[0]
    if lut_size <= 0:
        return
    denom = float(max(lut_size - 1, 1))
    for i in range(lut_size):
        u = float(i) / denom
        theta = _FAST_OPTICS_MAX_ANGLE * u * u
        Tf_s = fresnel_transmission(theta, n2_samp, True, False)
        Tf_p = fresnel_transmission(theta, n2_samp, False, False)
        Tf2 = 0.5 * (
            (np.real(Tf_s) * np.real(Tf_s) + np.imag(Tf_s) * np.imag(Tf_s))
            + (np.real(Tf_p) * np.real(Tf_p) + np.imag(Tf_p) * np.imag(Tf_p))
        )
        Tf2 = _sanitize_transmission_power(Tf2)

        _, im_k_z_f = ktz_components(k0, n2_samp, theta)
        if thickness > 0.0:
            L_out = thickness
        else:
            L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1e-30)

        out_angle = acos(_clamp(cos(theta) * n2_real, -1.0, 1.0))

        lut_row[i, _FAST_OPTICS_COL_TF2] = Tf2
        lut_row[i, _FAST_OPTICS_COL_IM_KZ] = im_k_z_f
        lut_row[i, _FAST_OPTICS_COL_L_OUT] = L_out
        lut_row[i, _FAST_OPTICS_COL_OUT_ANGLE] = out_angle


@njit(cache=True)
def _lookup_fast_optics_lut_row(lut_row, theta):
    lut_size = lut_row.shape[0]
    if lut_size <= 1:
        return (
            lut_row[0, _FAST_OPTICS_COL_TF2],
            lut_row[0, _FAST_OPTICS_COL_IM_KZ],
            lut_row[0, _FAST_OPTICS_COL_L_OUT],
            lut_row[0, _FAST_OPTICS_COL_OUT_ANGLE],
        )

    theta_eff = theta
    if theta_eff < 0.0:
        theta_eff = 0.0
    elif theta_eff > _FAST_OPTICS_MAX_ANGLE:
        theta_eff = _FAST_OPTICS_MAX_ANGLE

    if theta_eff <= 0.0:
        idx0 = 0
        frac = 0.0
    else:
        u = sqrt(theta_eff / _FAST_OPTICS_MAX_ANGLE) * float(lut_size - 1)
        idx0 = int(u)
        if idx0 >= lut_size - 1:
            idx0 = lut_size - 2
            frac = 1.0
        else:
            frac = u - float(idx0)
    idx1 = idx0 + 1

    tf2 = (
        lut_row[idx0, _FAST_OPTICS_COL_TF2] * (1.0 - frac)
        + lut_row[idx1, _FAST_OPTICS_COL_TF2] * frac
    )
    im_k_z_f = (
        lut_row[idx0, _FAST_OPTICS_COL_IM_KZ] * (1.0 - frac)
        + lut_row[idx1, _FAST_OPTICS_COL_IM_KZ] * frac
    )
    L_out = (
        lut_row[idx0, _FAST_OPTICS_COL_L_OUT] * (1.0 - frac)
        + lut_row[idx1, _FAST_OPTICS_COL_L_OUT] * frac
    )
    out_angle = (
        lut_row[idx0, _FAST_OPTICS_COL_OUT_ANGLE] * (1.0 - frac)
        + lut_row[idx1, _FAST_OPTICS_COL_OUT_ANGLE] * frac
    )
    return tf2, im_k_z_f, L_out, out_angle


# =============================================================================
# 4) solve_q
# =============================================================================


@njit(fastmath=True, cache=True)
def _mosaic_density_scalar(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv):
    Gx = G_vec[0]
    Gy = G_vec[1]
    Gz = G_vec[2]
    G_mag = sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
    if G_mag < 1e-14:
        return 0.0

    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    Qr = sqrt(Qx * Qx + Qy * Qy)
    Gr = sqrt(Gx * Gx + Gy * Gy)
    theta0 = np.arctan2(Gz, Gr)
    theta = np.arctan2(Qz, Qr)
    dtheta = wrap_to_pi(theta - theta0)

    A_gauss = 1.0 / (sigma_eff * sqrt(2.0 * pi))
    A_lor = 1.0 / (pi * gamma_eff)
    gauss_val = A_gauss * exp(-0.5 * (dtheta / sigma_eff) * (dtheta / sigma_eff))
    lor_val = A_lor / (1.0 + (dtheta / gamma_eff) * (dtheta / gamma_eff))
    omega = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

    denom_base = 2.0 * pi * G_mag * G_mag
    return omega / denom_base


@njit(fastmath=True, cache=True)
def _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z):
    cphi = cos(phi)
    sphi = sin(phi)
    Qx = Ox + circle_r * (cphi * e1x + sphi * e2x)
    Qy = Oy + circle_r * (cphi * e1y + sphi * e2y)
    Qz = Oz + circle_r * (cphi * e1z + sphi * e2z)
    return Qx, Qy, Qz


@njit(fastmath=True, cache=True)
def _circle_density(
    phi,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
):
    Qx, Qy, Qz = _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
    return _mosaic_density_scalar(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv)


@njit(fastmath=True, cache=True)
def _interval_mass_error(phi_a, phi_b, f_a, f_m, f_b, circle_r):
    dphi = phi_b - phi_a
    mass = circle_r * dphi * (f_a + 4.0 * f_m + f_b) / 6.0
    trap = circle_r * dphi * (f_a + f_b) * 0.5
    err = abs(mass - trap)
    return mass, err


@njit(fastmath=True, cache=True)
def _circle_theta_offset(
    phi,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
):
    Qx, Qy, Qz = _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
    Qr = sqrt(Qx * Qx + Qy * Qy)
    theta = np.arctan2(Qz, Qr)
    return wrap_to_pi(theta - theta0)


@njit(fastmath=True, cache=True)
def _phi_periodic_distance(phi_a, phi_b):
    delta = abs(phi_a - phi_b)
    two_pi = 2.0 * pi
    while delta >= two_pi:
        delta -= two_pi
    if delta > pi:
        delta = two_pi - delta
    return delta


@njit(fastmath=True, cache=True)
def _store_local_arc_root(roots, root_count, phi_root, min_separation):
    if not np.isfinite(phi_root):
        return root_count
    for i in range(root_count):
        if _phi_periodic_distance(phi_root, roots[i]) <= min_separation:
            return root_count
    if root_count < roots.shape[0]:
        roots[root_count] = phi_root
        return root_count + 1
    return root_count


@njit(fastmath=True, cache=True)
def _refine_theta_root(
    phi_a,
    phi_b,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
):
    fa = _circle_theta_offset(phi_a, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)
    fb = _circle_theta_offset(phi_b, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)

    if abs(fa) <= _LOCAL_ARC_ROOT_TOL:
        return phi_a, True
    if abs(fb) <= _LOCAL_ARC_ROOT_TOL:
        return phi_b, True
    if fa * fb > 0.0:
        return 0.5 * (phi_a + phi_b), False

    left = phi_a
    right = phi_b
    for _ in range(48):
        mid = 0.5 * (left + right)
        fm = _circle_theta_offset(mid, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)
        if abs(fm) <= _LOCAL_ARC_ROOT_TOL:
            return mid, True
        if fa * fm <= 0.0:
            right = mid
            fb = fm
        else:
            left = mid
            fa = fm
    return 0.5 * (left + right), True


@njit(fastmath=True, cache=True)
def _refine_theta_boundary(
    phi_inside,
    phi_outside,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
    theta_limit,
):
    left = phi_inside
    right = phi_outside
    f_left = (
        abs(_circle_theta_offset(left, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0))
        - theta_limit
    )
    f_right = (
        abs(_circle_theta_offset(right, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0))
        - theta_limit
    )

    if f_left > 0.0:
        return left
    if f_right < 0.0:
        return right

    for _ in range(48):
        mid = 0.5 * (left + right)
        f_mid = (
            abs(
                _circle_theta_offset(
                    mid, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
                )
            )
            - theta_limit
        )
        if abs(f_mid) <= _LOCAL_ARC_BOUNDARY_TOL:
            return mid
        if f_mid <= 0.0:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


@njit(fastmath=True, cache=True)
def _local_arc_theta_window(sigma, gamma_pv, eta_pv):
    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    gauss_window = 0.0
    if (1.0 - eta_pv) > 1e-8:
        gauss_window = _LOCAL_ARC_GAUSS_SIGMAS * sigma_eff

    lor_window = 0.0
    if eta_pv > 1e-8:
        lor_window = _LOCAL_ARC_LORENTZ_GAMMAS * gamma_eff

    theta_window = max(gauss_window, lor_window, _LOCAL_ARC_MIN_DTHETA)
    if theta_window > pi:
        theta_window = pi
    return theta_window


@njit(fastmath=True, cache=True)
def _append_local_arc_window(starts, ends, count, start, end):
    two_pi = 2.0 * pi
    span = end - start
    if span >= two_pi - 1.0e-9:
        starts[0] = 0.0
        ends[0] = two_pi
        return 1, True

    while start < 0.0:
        start += two_pi
        end += two_pi
    while start >= two_pi:
        start -= two_pi
        end -= two_pi

    if end <= two_pi:
        if count >= starts.shape[0]:
            return 1, True
        starts[count] = start
        ends[count] = end
        return count + 1, False

    if count + 1 >= starts.shape[0]:
        return 1, True
    starts[count] = start
    ends[count] = two_pi
    starts[count + 1] = 0.0
    ends[count + 1] = end - two_pi
    return count + 2, False


@njit(fastmath=True, cache=True)
def _build_local_arc_windows(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
):
    starts = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    ends = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    two_pi = 2.0 * pi

    theta_window = _local_arc_theta_window(sigma, gamma_pv, eta_pv)
    if theta_window >= _LOCAL_ARC_FULL_CIRCLE_THETA_WINDOW:
        starts[0] = 0.0
        ends[0] = two_pi
        return starts, ends, 1, True

    Gr = sqrt(G_vec[0] * G_vec[0] + G_vec[1] * G_vec[1])
    theta0 = np.arctan2(G_vec[2], Gr)

    search_steps = int(n_steps // 2)
    if search_steps < _LOCAL_ARC_MIN_SEARCH_STEPS:
        search_steps = _LOCAL_ARC_MIN_SEARCH_STEPS
    elif search_steps > _LOCAL_ARC_MAX_SEARCH_STEPS:
        search_steps = _LOCAL_ARC_MAX_SEARCH_STEPS
    dphi = two_pi / float(search_steps)

    roots = np.empty(_LOCAL_ARC_MAX_ROOTS, dtype=np.float64)
    root_count = 0
    best_abs_0 = np.inf
    best_abs_1 = np.inf
    best_phi_0 = 0.0
    best_phi_1 = 0.0

    prev_phi = 0.0
    prev_val = _circle_theta_offset(
        prev_phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
    )
    prev_abs = abs(prev_val)
    best_abs_0 = prev_abs
    best_phi_0 = prev_phi

    for i in range(1, search_steps + 1):
        phi_val = float(i) * dphi
        cur_val = _circle_theta_offset(
            phi_val, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
        )
        cur_abs = abs(cur_val)

        if cur_abs < best_abs_0:
            if _phi_periodic_distance(phi_val, best_phi_0) > (2.0 * dphi):
                best_abs_1 = best_abs_0
                best_phi_1 = best_phi_0
            best_abs_0 = cur_abs
            best_phi_0 = phi_val
        elif cur_abs < best_abs_1 and _phi_periodic_distance(phi_val, best_phi_0) > (2.0 * dphi):
            best_abs_1 = cur_abs
            best_phi_1 = phi_val

        if prev_abs <= _LOCAL_ARC_ROOT_TOL:
            root_count = _store_local_arc_root(roots, root_count, prev_phi, 2.0 * dphi)
        elif cur_abs <= _LOCAL_ARC_ROOT_TOL or (prev_val * cur_val < 0.0):
            phi_root, ok_root = _refine_theta_root(
                prev_phi,
                phi_val,
                Ox,
                Oy,
                Oz,
                circle_r,
                e1x,
                e1y,
                e1z,
                e2x,
                e2y,
                e2z,
                theta0,
            )
            if ok_root:
                root_count = _store_local_arc_root(roots, root_count, phi_root, 2.0 * dphi)

        prev_phi = phi_val
        prev_val = cur_val
        prev_abs = cur_abs

    if root_count == 0:
        if best_abs_0 <= theta_window:
            root_count = _store_local_arc_root(roots, root_count, best_phi_0, 2.0 * dphi)
        if best_abs_1 <= theta_window:
            root_count = _store_local_arc_root(roots, root_count, best_phi_1, 2.0 * dphi)

    if root_count <= 0:
        starts[0] = 0.0
        ends[0] = two_pi
        return starts, ends, 1, True

    window_count = 0
    for i_root in range(root_count):
        phi_root = roots[i_root]
        left_inside = phi_root
        left_outside = phi_root - dphi
        left_found = False
        full_circle = False
        for _ in range(search_steps):
            abs_val = abs(
                _circle_theta_offset(
                    left_outside,
                    Ox,
                    Oy,
                    Oz,
                    circle_r,
                    e1x,
                    e1y,
                    e1z,
                    e2x,
                    e2y,
                    e2z,
                    theta0,
                )
            )
            if abs_val >= theta_window:
                left_found = True
                break
            left_inside = left_outside
            left_outside -= dphi
        if not left_found:
            full_circle = True

        right_inside = phi_root
        right_outside = phi_root + dphi
        right_found = False
        if not full_circle:
            for _ in range(search_steps):
                abs_val = abs(
                    _circle_theta_offset(
                        right_outside,
                        Ox,
                        Oy,
                        Oz,
                        circle_r,
                        e1x,
                        e1y,
                        e1z,
                        e2x,
                        e2y,
                        e2z,
                        theta0,
                    )
                )
                if abs_val >= theta_window:
                    right_found = True
                    break
                right_inside = right_outside
                right_outside += dphi
        if not right_found:
            full_circle = True

        if full_circle:
            starts[0] = 0.0
            ends[0] = two_pi
            return starts, ends, 1, True

        left_bound = _refine_theta_boundary(
            left_inside,
            left_outside,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            theta0,
            theta_window,
        )
        right_bound = _refine_theta_boundary(
            right_inside,
            right_outside,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            theta0,
            theta_window,
        )

        window_count, full_circle = _append_local_arc_window(
            starts, ends, window_count, left_bound, right_bound
        )
        if full_circle:
            starts[0] = 0.0
            ends[0] = two_pi
            return starts, ends, 1, True
    if window_count <= 1:
        return starts, ends, window_count, False

    for i in range(1, window_count):
        start_val = starts[i]
        end_val = ends[i]
        j = i - 1
        while j >= 0 and starts[j] > start_val:
            starts[j + 1] = starts[j]
            ends[j + 1] = ends[j]
            j -= 1
        starts[j + 1] = start_val
        ends[j + 1] = end_val

    merged_starts = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    merged_ends = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    merged_count = 0
    for i in range(window_count):
        if merged_count == 0:
            merged_starts[0] = starts[i]
            merged_ends[0] = ends[i]
            merged_count = 1
            continue
        if starts[i] <= merged_ends[merged_count - 1] + 1.0e-9:
            if ends[i] > merged_ends[merged_count - 1]:
                merged_ends[merged_count - 1] = ends[i]
        else:
            merged_starts[merged_count] = starts[i]
            merged_ends[merged_count] = ends[i]
            merged_count += 1

    for i in range(merged_count):
        starts[i] = merged_starts[i]
        ends[i] = merged_ends[i]
    return starts, ends, merged_count, False


@njit(fastmath=True, cache=True)
def _solve_q_adaptive_domain(
    phi_start,
    phi_stop,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    max_intervals,
    base_intervals,
    rel_err_tol,
):
    if max_intervals <= 0 or phi_stop <= phi_start:
        return np.zeros((0, 4), dtype=np.float64)

    n_base = base_intervals
    if n_base < MIN_SOLVE_Q_BASE_INTERVALS:
        n_base = MIN_SOLVE_Q_BASE_INTERVALS
    if n_base > max_intervals:
        n_base = max_intervals
    if n_base < 1:
        n_base = 1

    phi_a = np.empty(max_intervals, dtype=np.float64)
    phi_b = np.empty(max_intervals, dtype=np.float64)
    f_a = np.empty(max_intervals, dtype=np.float64)
    f_m = np.empty(max_intervals, dtype=np.float64)
    f_b = np.empty(max_intervals, dtype=np.float64)
    mass_arr = np.empty(max_intervals, dtype=np.float64)
    err_arr = np.empty(max_intervals, dtype=np.float64)

    n_intervals = n_base
    total_mass = 0.0
    total_err = 0.0
    dphi0 = (phi_stop - phi_start) / n_base
    for i in range(n_base):
        a = phi_start + i * dphi0
        b = phi_start + (i + 1) * dphi0
        m = 0.5 * (a + b)

        fa = _circle_density(
            a, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fm = _circle_density(
            m, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fb = _circle_density(
            b, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )

        mass_i, err_i = _interval_mass_error(a, b, fa, fm, fb, circle_r)
        phi_a[i] = a
        phi_b[i] = b
        f_a[i] = fa
        f_m[i] = fm
        f_b[i] = fb
        mass_arr[i] = mass_i
        err_arr[i] = err_i
        total_mass += mass_i
        total_err += err_i

    err_tol = _SOLVE_Q_ABS_ERR_TOL + rel_err_tol * abs(total_mass)

    while n_intervals < max_intervals and total_err > err_tol:
        split_idx = 0
        max_err = err_arr[0]
        for i in range(1, n_intervals):
            if err_arr[i] > max_err:
                max_err = err_arr[i]
                split_idx = i
        if max_err <= 0.0:
            break

        a = phi_a[split_idx]
        b = phi_b[split_idx]
        m = 0.5 * (a + b)
        q1 = 0.5 * (a + m)
        q3 = 0.5 * (m + b)

        fa = f_a[split_idx]
        fm = f_m[split_idx]
        fb = f_b[split_idx]
        fq1 = _circle_density(
            q1, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fq3 = _circle_density(
            q3, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )

        old_mass = mass_arr[split_idx]
        old_err = err_arr[split_idx]

        left_mass, left_err = _interval_mass_error(a, m, fa, fq1, fm, circle_r)
        right_mass, right_err = _interval_mass_error(m, b, fm, fq3, fb, circle_r)

        phi_a[split_idx] = a
        phi_b[split_idx] = m
        f_a[split_idx] = fa
        f_m[split_idx] = fq1
        f_b[split_idx] = fm
        mass_arr[split_idx] = left_mass
        err_arr[split_idx] = left_err

        phi_a[n_intervals] = m
        phi_b[n_intervals] = b
        f_a[n_intervals] = fm
        f_m[n_intervals] = fq3
        f_b[n_intervals] = fb
        mass_arr[n_intervals] = right_mass
        err_arr[n_intervals] = right_err

        total_mass += left_mass + right_mass - old_mass
        total_err += left_err + right_err - old_err
        n_intervals += 1
        err_tol = _SOLVE_Q_ABS_ERR_TOL + rel_err_tol * abs(total_mass)

    n_valid = 0
    for i in range(n_intervals):
        if mass_arr[i] > _INTENSITY_CUTOFF:
            n_valid += 1

    out = np.zeros((n_valid, 4), dtype=np.float64)
    out_idx = 0
    for i in range(n_intervals):
        mass_i = mass_arr[i]
        if mass_i <= _INTENSITY_CUTOFF:
            continue
        phi_m = 0.5 * (phi_a[i] + phi_b[i])
        Qx, Qy, Qz = _circle_point(phi_m, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
        out[out_idx, 0] = Qx
        out[out_idx, 1] = Qy
        out[out_idx, 2] = Qz
        out[out_idx, 3] = mass_i
        out_idx += 1

    return out


@njit(fastmath=True, cache=True)
def _solve_q_uniform_full_circle(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
):
    if n_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    if (
        n_steps == DEFAULT_SOLVE_Q_STEPS
        and default_solve_q_cos is not None
        and default_solve_q_sin is not None
        and default_solve_q_cos.shape[0] == n_steps
        and default_solve_q_sin.shape[0] == n_steps
    ):
        dtheta = float(default_solve_q_dtheta)
        if dtheta <= 0.0:
            dtheta = 2.0 * np.pi / n_steps
        cth = default_solve_q_cos
        sth = default_solve_q_sin
    else:
        dtheta = 2.0 * np.pi / n_steps
        theta_arr = dtheta * np.arange(n_steps)
        cth = np.cos(theta_arr)
        sth = np.sin(theta_arr)

    Qx_arr = Ox + circle_r * (cth * e1x + sth * e2x)
    Qy_arr = Oy + circle_r * (cth * e1y + sth * e2y)
    Qz_arr = Oz + circle_r * (cth * e1z + sth * e2z)

    sigma_arr = compute_intensity_array(Qx_arr, Qy_arr, Qz_arr, G_vec, sigma, gamma_pv, eta_pv)
    ds = circle_r * dtheta
    all_int = sigma_arr * ds

    valid_idx = np.nonzero(all_int > _INTENSITY_CUTOFF)[0]
    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]

    return out


@njit(fastmath=True, cache=True)
def _solve_q_uniform(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
):
    if n_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    starts, ends, window_count, use_full_circle = _build_local_arc_windows(
        Ox,
        Oy,
        Oz,
        circle_r,
        e1x,
        e1y,
        e1z,
        e2x,
        e2y,
        e2z,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
        n_steps,
    )
    if use_full_circle or window_count <= 0:
        return _solve_q_uniform_full_circle(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            n_steps,
            default_solve_q_dtheta,
            default_solve_q_cos,
            default_solve_q_sin,
        )

    target_dphi = (2.0 * pi) / float(n_steps)
    window_steps = np.empty(window_count, dtype=np.int64)
    total_samples = 0
    for i_win in range(window_count):
        span = ends[i_win] - starts[i_win]
        n_window = int(np.ceil(span / max(target_dphi, 1.0e-12)))
        if n_window < _LOCAL_ARC_MIN_STEPS_PER_WINDOW:
            n_window = _LOCAL_ARC_MIN_STEPS_PER_WINDOW
        window_steps[i_win] = n_window
        total_samples += n_window

    Qx_arr = np.empty(total_samples, dtype=np.float64)
    Qy_arr = np.empty(total_samples, dtype=np.float64)
    Qz_arr = np.empty(total_samples, dtype=np.float64)
    ds_arr = np.empty(total_samples, dtype=np.float64)
    offset = 0
    for i_win in range(window_count):
        n_window = int(window_steps[i_win])
        span = ends[i_win] - starts[i_win]
        dphi_local = span / float(n_window)
        ds = circle_r * dphi_local
        step_idx = np.arange(n_window, dtype=np.float64)
        phi_mid = starts[i_win] + (step_idx + 0.5) * dphi_local
        cos_phi = np.cos(phi_mid)
        sin_phi = np.sin(phi_mid)
        next_offset = offset + n_window

        Qx_arr[offset:next_offset] = Ox + circle_r * (cos_phi * e1x + sin_phi * e2x)
        Qy_arr[offset:next_offset] = Oy + circle_r * (cos_phi * e1y + sin_phi * e2y)
        Qz_arr[offset:next_offset] = Oz + circle_r * (cos_phi * e1z + sin_phi * e2z)
        ds_arr[offset:next_offset] = ds
        offset = next_offset

    sigma_arr = compute_intensity_array(Qx_arr, Qy_arr, Qz_arr, G_vec, sigma, gamma_pv, eta_pv)
    all_int = sigma_arr * ds_arr
    valid_idx = np.nonzero(all_int > _INTENSITY_CUTOFF)[0]
    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]
    return out


@njit(fastmath=True, cache=True)
def _solve_q_adaptive(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    max_intervals,
    base_intervals,
    rel_err_tol,
):
    starts, ends, window_count, use_full_circle = _build_local_arc_windows(
        Ox,
        Oy,
        Oz,
        circle_r,
        e1x,
        e1y,
        e1z,
        e2x,
        e2y,
        e2z,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
        max_intervals,
    )
    if use_full_circle or window_count <= 0:
        return _solve_q_adaptive_domain(
            0.0,
            2.0 * pi,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            max_intervals,
            base_intervals,
            rel_err_tol,
        )

    total_span = 0.0
    for i_win in range(window_count):
        total_span += ends[i_win] - starts[i_win]
    if total_span <= 0.0:
        return np.zeros((0, 4), dtype=np.float64)

    chunks = List.empty_list(types.float64[:, ::1])
    total_rows = 0
    for i_win in range(window_count):
        span = ends[i_win] - starts[i_win]
        frac = span / total_span
        max_intervals_i = int(round(float(max_intervals) * frac))
        if max_intervals_i < MIN_SOLVE_Q_BASE_INTERVALS:
            max_intervals_i = MIN_SOLVE_Q_BASE_INTERVALS
        base_intervals_i = int(round(float(base_intervals) * frac))
        if base_intervals_i < MIN_SOLVE_Q_BASE_INTERVALS:
            base_intervals_i = MIN_SOLVE_Q_BASE_INTERVALS
        if max_intervals_i < base_intervals_i:
            max_intervals_i = base_intervals_i

        chunk = _solve_q_adaptive_domain(
            starts[i_win],
            ends[i_win],
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            max_intervals_i,
            base_intervals_i,
            rel_err_tol,
        )
        chunks.append(chunk)
        total_rows += chunk.shape[0]
    out = np.zeros((total_rows, 4), dtype=np.float64)
    out_idx = 0
    for i_chunk in range(len(chunks)):
        chunk = chunks[i_chunk]
        for i_row in range(chunk.shape[0]):
            out[out_idx, 0] = chunk[i_row, 0]
            out[out_idx, 1] = chunk[i_row, 1]
            out[out_idx, 2] = chunk[i_row, 2]
            out[out_idx, 3] = chunk[i_row, 3]
            out_idx += 1
    return out


@njit(fastmath=True, cache=True)
def solve_q(
    k_in_crystal,
    k_scat,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    H,
    K,
    L,
    N_steps=DEFAULT_SOLVE_Q_STEPS,
    base_intervals=DEFAULT_SOLVE_Q_BASE_INTERVALS,
    rel_err_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
):
    """
    Build a 'circle' in reciprocal space for the reflection G_vec, i.e. the
    set of Q that satisfies |Q|=|G| or an intersection with Ewald sphere, then
    filter by mosaic surface density compute_intensity_array.

    Physically:
      - In uniform mode, sample the full circle at fixed angular steps.
      - In adaptive mode, refine intervals deterministically where the
        pseudo-Voigt profile varies most.
      - Adaptive mode uses Simpson-weighted interval masses to preserve long
        Lorentzian tails without stochastic noise.

    Returns
    -------
    out : ndarray of shape (M,4)
        For the valid points, columns = (Qx, Qy, Qz, mosaic_intensity).
    status : int
        0 for success or a negative code indicating the failure reason.
    """
    status = 0
    if N_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64), status
    if base_intervals <= 0:
        return np.zeros((0, 4), dtype=np.float64), status
    if rel_err_tol < 0.0:
        rel_err_tol = 0.0

    G_sq = G_vec[0] * G_vec[0] + G_vec[1] * G_vec[1] + G_vec[2] * G_vec[2]
    if G_sq < 1e-14:
        status = -1
        return np.zeros((0, 4), dtype=np.float64), status

    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    rA = k_scat
    A_sq = Ax * Ax + Ay * Ay + Az * Az
    if A_sq < 1e-14:
        status = -2
        return np.zeros((0, 4), dtype=np.float64), status
    A_len = sqrt(A_sq)

    c = (G_sq + A_sq - rA * rA) / (2.0 * A_len)
    # Compute circle parameters
    circle_r_sq = G_sq - c * c
    if circle_r_sq < 0.0:
        status = -3
        return np.zeros((0, 4), dtype=np.float64), status
    circle_r = np.sqrt(circle_r_sq)

    Ax_hat = Ax / A_len
    Ay_hat = Ay / A_len
    Az_hat = Az / A_len

    Ox = c * Ax_hat
    Oy = c * Ay_hat
    Oz = c * Az_hat

    # Build two orthonormal vectors (e1, e2) in the plane perpendicular to Ax_hat.
    ax, ay, az = 1.0, 0.0, 0.0
    dot_aA = ax * Ax_hat + ay * Ay_hat + az * Az_hat
    if abs(dot_aA) > 0.9999:
        ax, ay, az = 0.0, 1.0, 0.0
        dot_aA = ax * Ax_hat + ay * Ay_hat + az * Az_hat
    aox = ax - dot_aA * Ax_hat
    aoy = ay - dot_aA * Ay_hat
    aoz = az - dot_aA * Az_hat
    ao_len = np.sqrt(aox * aox + aoy * aoy + aoz * aoz)
    if ao_len < 1e-14:
        status = -4
        return np.zeros((0, 4), dtype=np.float64), status
    e1x = aox / ao_len
    e1y = aoy / ao_len
    e1z = aoz / ao_len

    e2x = Az_hat * e1y - Ay_hat * e1z
    e2y = Ax_hat * e1z - Az_hat * e1x
    e2z = Ay_hat * e1x - Ax_hat * e1y
    e2_len = np.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    if e2_len < 1e-14:
        status = -5
        return np.zeros((0, 4), dtype=np.float64), status
    e2x /= e2_len
    e2y /= e2_len
    e2z /= e2_len

    mode_i = int(solve_q_mode)
    if mode_i == SOLVE_Q_MODE_UNIFORM:
        out = _solve_q_uniform(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            int(N_steps),
            default_solve_q_dtheta,
            default_solve_q_cos,
            default_solve_q_sin,
        )
    else:
        out = _solve_q_adaptive(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            int(N_steps),
            int(base_intervals),
            float(rel_err_tol),
        )

    return out, status


def _set_last_process_peaks_safe_stats(**updates):
    global _LAST_PROCESS_PEAKS_SAFE_STATS
    stats = dict(_EMPTY_PROCESS_PEAKS_SAFE_STATS)
    stats.update(updates)
    _LAST_PROCESS_PEAKS_SAFE_STATS = stats


def get_last_process_peaks_safe_stats():
    return dict(_LAST_PROCESS_PEAKS_SAFE_STATS)


def _bind_process_peaks_parallel_call(args, kwargs):
    if len(args) > len(_PROCESS_PEAKS_PARALLEL_PARAM_NAMES):
        return None

    bound = {}
    for idx, value in enumerate(args):
        name = _PROCESS_PEAKS_PARALLEL_PARAM_NAMES[idx]
        if name in kwargs:
            return None
        bound[name] = value

    for name in _PROCESS_PEAKS_PARALLEL_PARAM_NAMES[len(args) :]:
        if name in kwargs:
            bound[name] = kwargs[name]
        elif name in _PROCESS_PEAKS_PARALLEL_DEFAULTS:
            bound[name] = _PROCESS_PEAKS_PARALLEL_DEFAULTS[name]
        else:
            return None

    extras = {key: value for key, value in kwargs.items() if key not in bound}
    return bound, extras


def _normalize_bound_process_peaks_parallel_call(args, kwargs):
    bound_result = _bind_process_peaks_parallel_call(args, kwargs)
    if bound_result is None:
        return args, dict(kwargs)
    bound, extra_kwargs = bound_result
    bound["events_per_beam_phase"] = normalize_events_per_beam_phase_backend(
        bound.get("events_per_beam_phase", 50)
    )
    positional_count = len(args)
    normalized_args = tuple(
        bound[name] for name in _PROCESS_PEAKS_PARALLEL_PARAM_NAMES[:positional_count]
    )
    normalized_kwargs = dict(extra_kwargs)
    for name in _PROCESS_PEAKS_PARALLEL_PARAM_NAMES[positional_count:]:
        normalized_kwargs[name] = bound[name]
    return normalized_args, normalized_kwargs


def _get_phase_entry_n_samp(phase_entry, fallback_n_samp):
    if isinstance(phase_entry, dict):
        n_samp = phase_entry.get("n_samp", fallback_n_samp)
    else:
        n_samp = fallback_n_samp
    try:
        n_samp_i = int(n_samp)
    except (TypeError, ValueError):
        n_samp_i = int(fallback_n_samp)
    if n_samp_i < 0:
        n_samp_i = 0
    return n_samp_i


def _build_phase_space_entry(params):
    beam_x_array = np.asarray(params.get("beam_x_array", np.zeros(0, dtype=np.float64)))
    return {
        "n_samp": int(beam_x_array.size),
        "theta_initial_deg": float(params.get("theta_initial_deg", 0.0)),
    }


def _build_source_unit_template(params, phase_entry, H, K, L):
    theta_initial_deg = float(params.get("theta_initial_deg", 0.0))
    lambda_angstrom = float(params.get("lambda_", 1.0))
    if not np.isfinite(lambda_angstrom) or lambda_angstrom <= 0.0:
        lambda_angstrom = 1.0
    k_scat = (2.0 * np.pi) / lambda_angstrom
    theta_initial_rad = theta_initial_deg * (pi / 180.0)
    k_in_crystal = np.array(
        [
            0.0,
            k_scat * cos(theta_initial_rad),
            -k_scat * sin(theta_initial_rad),
        ],
        dtype=np.float64,
    )
    G_vec = np.array([float(H), float(K), float(L)], dtype=np.float64)
    solve_q_steps = int(params.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS))
    solve_q_rel_tol = float(params.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL))
    solve_q_mode = int(params.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE))
    default_solve_q_dtheta = float(params.get("default_solve_q_dtheta", -1.0))
    default_solve_q_cos = params.get("default_solve_q_cos")
    if default_solve_q_cos is not None:
        default_solve_q_cos = np.asarray(default_solve_q_cos, dtype=np.float64)
    default_solve_q_sin = params.get("default_solve_q_sin")
    if default_solve_q_sin is not None:
        default_solve_q_sin = np.asarray(default_solve_q_sin, dtype=np.float64)
    sigma = float(params.get("sigma_pv_deg", 0.0)) * (pi / 180.0)
    gamma_pv = float(params.get("gamma_pv_deg", 0.0)) * (pi / 180.0)
    eta_pv = float(params.get("eta_pv", 0.0))
    canonical_default_trig = _uses_canonical_default_solve_q_trig_tables(
        default_solve_q_dtheta,
        default_solve_q_cos,
        default_solve_q_sin,
    )
    q_cache_key = None
    if canonical_default_trig:
        q_cache_key = (
            round(theta_initial_deg, 12),
            round(lambda_angstrom, 12),
            round(float(H), 12),
            round(float(K), 12),
            round(float(L), 12),
            solve_q_steps,
            round(solve_q_rel_tol, 12),
            solve_q_mode,
            "canonical_default_trig",
        )
    q_cache_hits = 0
    retain_safe_cache = _retain_diffraction_safe_cache()
    if not retain_safe_cache:
        _Q_VECTOR_CACHE.clear()
    q_result = None
    if retain_safe_cache and q_cache_key is not None:
        q_result = _Q_VECTOR_CACHE.get(q_cache_key)
    if q_result is None:
        q_result = solve_q(
            k_in_crystal,
            k_scat,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            float(H),
            float(K),
            float(L),
            N_steps=solve_q_steps,
            rel_err_tol=solve_q_rel_tol,
            solve_q_mode=solve_q_mode,
            default_solve_q_dtheta=default_solve_q_dtheta,
            default_solve_q_cos=default_solve_q_cos,
            default_solve_q_sin=default_solve_q_sin,
        )
        if retain_safe_cache and q_cache_key is not None:
            _Q_VECTOR_CACHE[q_cache_key] = q_result
    else:
        q_cache_hits = 1

    q_points, status = q_result
    n_samp = _get_phase_entry_n_samp(
        phase_entry,
        np.asarray(params.get("beam_x_array", np.zeros(0, dtype=np.float64))).size,
    )
    return {
        "flat_indices": np.empty(0, dtype=np.int64),
        "flat_values": np.empty(0, dtype=np.float64),
        "hit_template": np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64),
        "miss_template": np.empty((0, 3), dtype=np.float64),
        "status_template": np.full(n_samp, int(status), dtype=np.int64),
        "best_sample_idx": -1,
        "q_cache_hits": int(q_cache_hits),
        "q_count": int(np.asarray(q_points).shape[0]),
    }


_DEFAULT_BUILD_PHASE_SPACE_ENTRY = _build_phase_space_entry
_DEFAULT_BUILD_SOURCE_UNIT_TEMPLATE = _build_source_unit_template
_DEFAULT_SOLVE_Q = solve_q


def _safe_cache_hooks_active(enable_safe_cache):
    if enable_safe_cache is not None:
        return bool(enable_safe_cache)
    return (
        _build_phase_space_entry is not _DEFAULT_BUILD_PHASE_SPACE_ENTRY
        or _build_source_unit_template is not _DEFAULT_BUILD_SOURCE_UNIT_TEMPLATE
        or solve_q is not _DEFAULT_SOLVE_Q
    )


def _maybe_run_process_peaks_safe_cache(args, kwargs, enable_safe_cache):
    return None


@njit(fastmath=True, cache=True)
def _circle_frame_components(k_in_crystal, k_scat, g_vec):
    """Return the Bragg-circle frame used to classify solutions around G."""

    g_sq = g_vec[0] * g_vec[0] + g_vec[1] * g_vec[1] + g_vec[2] * g_vec[2]
    if g_sq < 1e-14:
        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    ax = -k_in_crystal[0]
    ay = -k_in_crystal[1]
    az = -k_in_crystal[2]
    a_sq = ax * ax + ay * ay + az * az
    if a_sq < 1e-14:
        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    a_len = sqrt(a_sq)

    c = (g_sq + a_sq - k_scat * k_scat) / (2.0 * a_len)
    circle_r_sq = g_sq - c * c
    if circle_r_sq < 0.0:
        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    ahx = ax / a_len
    ahy = ay / a_len
    ahz = az / a_len

    ox = c * ahx
    oy = c * ahy
    oz = c * ahz

    anchor_x = 1.0
    anchor_y = 0.0
    anchor_z = 0.0
    dot_a = anchor_x * ahx + anchor_y * ahy + anchor_z * ahz
    if abs(dot_a) > 0.9999:
        anchor_x = 0.0
        anchor_y = 1.0
        anchor_z = 0.0
        dot_a = anchor_x * ahx + anchor_y * ahy + anchor_z * ahz

    e1x = anchor_x - dot_a * ahx
    e1y = anchor_y - dot_a * ahy
    e1z = anchor_z - dot_a * ahz
    e1_len = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    if e1_len < 1e-14:
        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    e1x /= e1_len
    e1y /= e1_len
    e1z /= e1_len

    e2x = ahz * e1y - ahy * e1z
    e2y = ahx * e1z - ahz * e1x
    e2z = ahy * e1x - ahx * e1y
    e2_len = sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    if e2_len < 1e-14:
        return (
            False,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    e2x /= e2_len
    e2y /= e2_len
    e2z /= e2_len

    return True, ox, oy, oz, e1x, e1y, e1z, e2x, e2y, e2z


@njit(fastmath=True, cache=True)
def _select_g_peak_solution_indices(all_q, k_in_crystal, k_scat, g_vec):
    """Keep the strongest mosaic-intensity solution(s) on either side of G."""

    count = all_q.shape[0]
    out = np.full(2, -1, dtype=np.int64)
    if count <= 0:
        return out, 0

    best_idx = 0
    best_iq = all_q[0, 3]
    for idx in range(1, count):
        iq = all_q[idx, 3]
        if iq > best_iq:
            best_iq = iq
            best_idx = idx

    gr_sq = g_vec[0] * g_vec[0] + g_vec[1] * g_vec[1]
    if gr_sq <= 1e-14 or count == 1:
        out[0] = best_idx
        return out, 1

    (
        ok,
        ox,
        oy,
        oz,
        e1x,
        e1y,
        e1z,
        e2x,
        e2y,
        e2z,
    ) = _circle_frame_components(k_in_crystal, k_scat, g_vec)
    if not ok:
        out[0] = best_idx
        return out, 1

    grx = g_vec[0] - ox
    gry = g_vec[1] - oy
    grz = g_vec[2] - oz
    g_a1 = grx * e1x + gry * e1y + grz * e1z
    g_a2 = grx * e2x + gry * e2y + grz * e2z
    g_angle = np.arctan2(g_a2, g_a1)

    best_neg_idx = -1
    best_pos_idx = -1
    best_neg_iq = -1.0
    best_pos_iq = -1.0
    best_neg_abs = np.inf
    best_pos_abs = np.inf

    for idx in range(count):
        iq = all_q[idx, 3]
        if not np.isfinite(iq) or iq <= 0.0:
            continue
        rx = all_q[idx, 0] - ox
        ry = all_q[idx, 1] - oy
        rz = all_q[idx, 2] - oz
        a1 = rx * e1x + ry * e1y + rz * e1z
        a2 = rx * e2x + ry * e2y + rz * e2z
        delta = wrap_to_pi(np.arctan2(a2, a1) - g_angle)
        abs_delta = abs(delta)

        if delta < 0.0:
            if iq > best_neg_iq or (abs(iq - best_neg_iq) <= 1e-18 and abs_delta < best_neg_abs):
                best_neg_iq = iq
                best_neg_abs = abs_delta
                best_neg_idx = idx
        else:
            if iq > best_pos_iq or (abs(iq - best_pos_iq) <= 1e-18 and abs_delta < best_pos_abs):
                best_pos_iq = iq
                best_pos_abs = abs_delta
                best_pos_idx = idx

    n_keep = 0
    if best_neg_idx >= 0:
        out[n_keep] = best_neg_idx
        n_keep += 1
    if best_pos_idx >= 0 and best_pos_idx != best_neg_idx:
        out[n_keep] = best_pos_idx
        n_keep += 1
    if n_keep <= 0:
        out[0] = best_idx
        return out, 1
    return out, n_keep


@njit(fastmath=True, cache=True)
def _build_sample_rotation(
    theta_initial_deg,
    cor_angle_deg,
    psi_z_deg,
    R_z_R_y,
    R_ZY_n,
    P0,
):
    """Build reflection-invariant sample frame for the current geometry."""
    rad_theta_i = theta_initial_deg * (pi / 180.0)
    cor_axis_rad = cor_angle_deg * (pi / 180.0)
    cor_axis_yaw_rad = psi_z_deg * (pi / 180.0)

    # Pitch the CoR axis in its local x-z plane, then yaw that axis about
    # laboratory z by psi_z.
    ax = cos(cor_axis_rad)
    ay = 0.0
    az = sin(cor_axis_rad)
    c_axis_yaw = cos(cor_axis_yaw_rad)
    s_axis_yaw = sin(cor_axis_yaw_rad)
    ax_yawed = c_axis_yaw * ax + s_axis_yaw * ay
    ay_yawed = -s_axis_yaw * ax + c_axis_yaw * ay
    ax = ax_yawed
    ay = ay_yawed
    axis_norm = sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12:
        axis_norm = 1.0
    ax /= axis_norm
    ay /= axis_norm
    az /= axis_norm

    ct = cos(rad_theta_i)
    st = sin(rad_theta_i)
    one_ct = 1.0 - ct
    R_cor = np.array(
        [
            [
                ct + ax * ax * one_ct,
                ax * ay * one_ct - az * st,
                ax * az * one_ct + ay * st,
            ],
            [
                ay * ax * one_ct + az * st,
                ct + ay * ay * one_ct,
                ay * az * one_ct - ax * st,
            ],
            [
                az * ax * one_ct - ay * st,
                az * ay * one_ct + ax * st,
                ct + az * az * one_ct,
            ],
        ]
    )
    R_sample = R_cor @ R_z_R_y

    n_surf = R_cor @ R_ZY_n
    n_surf /= sqrt(n_surf[0] * n_surf[0] + n_surf[1] * n_surf[1] + n_surf[2] * n_surf[2])

    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0
    return R_sample, n_surf, P0_rot


@njit(fastmath=True, cache=True)
def _precompute_sample_terms(
    wavelength_array,
    n2,
    n2_array,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    zb,
    thickness,
    sample_width_m,
    sample_length_m,
    optics_mode,
    theta_initial_deg,
    cor_angle_deg,
    psi_z_deg,
    R_z_R_y,
    R_ZY_n,
    P0,
):
    """Precompute sample- and beam-dependent terms shared by all reflections."""
    n_samp = beam_x_array.size
    sample_terms = np.zeros((n_samp, _SAMPLE_COLS), dtype=np.float64)
    sample_terms[:, _SAMPLE_COL_SOLVE_Q_REP] = -1.0
    sample_terms[:, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
    n2_samp_out = np.empty(n_samp, dtype=np.complex128)
    eps2_out = np.empty(n_samp, dtype=np.complex128)

    R_sample, n_surf, P0_rot = _build_sample_rotation(
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    )

    best_idx = 0
    if n_samp > 0:
        best_angle = theta_array[0] * theta_array[0] + phi_array[0] * phi_array[0]
        best_beam = beam_x_array[0] * beam_x_array[0] + beam_y_array[0] * beam_y_array[0]
        for ii in range(1, n_samp):
            metric = theta_array[ii] * theta_array[ii] + phi_array[ii] * phi_array[ii]
            beam_metric = beam_x_array[ii] * beam_x_array[ii] + beam_y_array[ii] * beam_y_array[ii]
            if metric < best_angle:
                best_angle = metric
                best_beam = beam_metric
                best_idx = ii
            elif abs(metric - best_angle) <= 1e-18 and beam_metric < best_beam:
                best_beam = beam_metric
                best_idx = ii

    # Build local incidence basis around the sample normal.
    sample_axis_x = np.array(
        [R_sample[0, 0], R_sample[1, 0], R_sample[2, 0]],
        dtype=np.float64,
    )
    sample_axis_y = np.array(
        [R_sample[0, 1], R_sample[1, 1], R_sample[2, 1]],
        dtype=np.float64,
    )
    half_width = 0.5 * sample_width_m if sample_width_m > 0.0 else 0.0
    half_length = 0.5 * sample_length_m if sample_length_m > 0.0 else 0.0
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = sqrt(e1_temp[0] * e1_temp[0] + e1_temp[1] * e1_temp[1] + e1_temp[2] * e1_temp[2])
    if e1_norm < 1e-12:
        alt_refs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for ar in alt_refs:
            cross_tmp = np.cross(n_surf, ar)
            cross_norm_tmp = sqrt(
                cross_tmp[0] * cross_tmp[0]
                + cross_tmp[1] * cross_tmp[1]
                + cross_tmp[2] * cross_tmp[2]
            )
            if cross_norm_tmp > 1e-12:
                e1_temp = cross_tmp / cross_norm_tmp
                break
    else:
        e1_temp /= e1_norm
    e2_temp = np.cross(n_surf, e1_temp)

    use_exact_optics = optics_mode == OPTICS_MODE_EXACT
    eps1 = 1.0 + 0.0j

    for i_samp in range(n_samp):
        lam_samp = wavelength_array[i_samp]
        k0 = 2.0 * pi / lam_samp

        n2_samp = n2
        if i_samp < n2_array.size:
            n2_samp = n2_array[i_samp]
        eps2 = n2_samp * n2_samp
        n2_sq_real = np.real(eps2)

        n2_samp_out[i_samp] = n2_samp
        eps2_out[i_samp] = eps2
        sample_terms[i_samp, _SAMPLE_COL_K0] = k0
        sample_terms[i_samp, _SAMPLE_COL_N2_REAL] = np.real(n2_samp)

        dtheta = theta_array[i_samp]
        dphi = phi_array[i_samp]
        k_in_x = cos(dtheta) * sin(dphi)
        k_in_y = cos(dtheta) * cos(dphi)
        k_in_z = sin(dtheta)

        beam_start = np.array(
            [beam_x_array[i_samp], -20e-3, beam_y_array[i_samp] - zb],
            dtype=np.float64,
        )
        k_in = np.array([k_in_x, k_in_y, k_in_z], dtype=np.float64)

        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            continue

        rel_x = ix - P0_rot[0]
        rel_y = iy - P0_rot[1]
        rel_z = iz - P0_rot[2]
        if half_width > 0.0:
            x_local = rel_x * sample_axis_x[0] + rel_y * sample_axis_x[1] + rel_z * sample_axis_x[2]
            if np.abs(x_local) > half_width:
                continue
        if half_length > 0.0:
            y_local = rel_x * sample_axis_y[0] + rel_y * sample_axis_y[1] + rel_z * sample_axis_y[2]
            if np.abs(y_local) > half_length:
                continue

        sample_terms[i_samp, _SAMPLE_COL_VALID] = 1.0
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_X] = ix
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_Y] = iy
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_Z] = iz

        kn_dot = k_in_x * n_surf[0] + k_in_y * n_surf[1] + k_in_z * n_surf[2]
        if kn_dot > 1.0:
            kn_dot = 1.0
        elif kn_dot < -1.0:
            kn_dot = -1.0
        th_i_prime = (pi / 2.0) - acos(kn_dot)

        proj_incident_x = k_in_x - kn_dot * n_surf[0]
        proj_incident_y = k_in_y - kn_dot * n_surf[1]
        proj_incident_z = k_in_z - kn_dot * n_surf[2]
        pln = sqrt(
            proj_incident_x * proj_incident_x
            + proj_incident_y * proj_incident_y
            + proj_incident_z * proj_incident_z
        )
        if pln > 1e-12:
            proj_incident_x /= pln
            proj_incident_y /= pln
            proj_incident_z /= pln
        else:
            proj_incident_x = 0.0
            proj_incident_y = 0.0
            proj_incident_z = 0.0

        p1 = (
            proj_incident_x * e1_temp[0]
            + proj_incident_y * e1_temp[1]
            + proj_incident_z * e1_temp[2]
        )
        p2 = (
            proj_incident_x * e2_temp[0]
            + proj_incident_y * e2_temp[1]
            + proj_incident_z * e2_temp[2]
        )
        phi_i_prime = (pi / 2.0) - np.arctan2(p2, p1)

        if use_exact_optics:
            k0_sq = k0 * k0
            k_par_i = k0 * np.abs(np.cos(th_i_prime))
            k_par_i_sq = k_par_i * k_par_i

            kz1_i = _kz_branch_decay((k0_sq - k_par_i_sq) + 0.0j)
            kz2_i = _kz_branch_decay((eps2 * k0_sq) - k_par_i_sq)

            k_x_scat = k_par_i * np.sin(phi_i_prime)
            k_y_scat = k_par_i * np.cos(phi_i_prime)
            re_k_z = -np.abs(kz2_i.real)
            im_k_z = np.abs(kz2_i.imag)
            k_scat = np.sqrt(np.maximum(k_par_i_sq + kz2_i.real * kz2_i.real, 0.0))

            Ti_s = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, True)
            Ti_p = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, False)
            Ti2 = 0.5 * (
                _fresnel_power_t_exact(Ti_s, kz1_i, kz2_i, eps1, eps2, True)
                + _fresnel_power_t_exact(Ti_p, kz1_i, kz2_i, eps1, eps2, False)
            )
            Ti2 = _sanitize_transmission_power(Ti2)

            if thickness > 0.0:
                L_in = thickness
            else:
                L_in = 1.0 / np.maximum(2.0 * im_k_z, 1e-30)
        else:
            th_t = transmit_angle_grazing(th_i_prime, n2_samp)
            k_scat = k0 * np.sqrt(np.maximum(n2_sq_real, 0.0))
            k_x_scat = k_scat * np.cos(th_t) * np.sin(phi_i_prime)
            k_y_scat = k_scat * np.cos(th_t) * np.cos(phi_i_prime)

            re_k_z, im_k_z = ktz_components(k0, n2_samp, th_t)
            re_k_z = -re_k_z

            Ti_s = fresnel_transmission(th_i_prime, n2_samp, True, True)
            Ti_p = fresnel_transmission(th_i_prime, n2_samp, False, True)
            Ti2 = 0.5 * (
                (np.real(Ti_s) * np.real(Ti_s) + np.imag(Ti_s) * np.imag(Ti_s))
                + (np.real(Ti_p) * np.real(Ti_p) + np.imag(Ti_p) * np.imag(Ti_p))
            )
            Ti2 = _sanitize_transmission_power(Ti2)

            if thickness > 0.0:
                L_in = thickness
            else:
                L_in = 1.0 / np.maximum(2.0 * im_k_z, 1e-30)

        sample_terms[i_samp, _SAMPLE_COL_KX_SCAT] = k_x_scat
        sample_terms[i_samp, _SAMPLE_COL_KY_SCAT] = k_y_scat
        sample_terms[i_samp, _SAMPLE_COL_RE_KZ] = re_k_z
        sample_terms[i_samp, _SAMPLE_COL_IM_KZ] = im_k_z
        sample_terms[i_samp, _SAMPLE_COL_K_SCAT] = k_scat
        sample_terms[i_samp, _SAMPLE_COL_TI2] = Ti2
        sample_terms[i_samp, _SAMPLE_COL_L_IN] = L_in

    _annotate_solve_q_sample_reuse(sample_terms)
    return R_sample, sample_terms, n2_samp_out, eps2_out, best_idx


@njit(fastmath=True, cache=True)
def _solve_q_reuse_terms_match(sample_terms, idx_a, idx_b):
    cols = (
        _SAMPLE_COL_KX_SCAT,
        _SAMPLE_COL_KY_SCAT,
        _SAMPLE_COL_RE_KZ,
        _SAMPLE_COL_K_SCAT,
    )
    for col in cols:
        aval = sample_terms[idx_a, col]
        bval = sample_terms[idx_b, col]
        scale = np.abs(aval)
        if np.abs(bval) > scale:
            scale = np.abs(bval)
        if np.abs(aval - bval) > (1.0e-12 * (1.0 + scale)):
            return False
    return True


@njit(fastmath=True, cache=True)
def _annotate_solve_q_sample_reuse(sample_terms):
    """Link samples whose `solve_q` inputs are numerically identical."""

    n_samp = sample_terms.shape[0]
    if n_samp <= 1:
        return

    group_reps = np.empty(n_samp, dtype=np.int64)
    group_tails = np.empty(n_samp, dtype=np.int64)
    group_count = 0

    for i_samp in range(n_samp):
        if sample_terms[i_samp, _SAMPLE_COL_VALID] <= 0.5:
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = -1.0
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
            continue

        matched_group = -1
        for i_grp in range(group_count):
            rep_idx = group_reps[i_grp]
            if _solve_q_reuse_terms_match(sample_terms, i_samp, rep_idx):
                matched_group = i_grp
                break

        if matched_group < 0:
            group_reps[group_count] = i_samp
            group_tails[group_count] = i_samp
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = float(i_samp)
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
            group_count += 1
            continue

        rep_idx = group_reps[matched_group]
        tail_idx = group_tails[matched_group]
        sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = float(rep_idx)
        sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
        sample_terms[tail_idx, _SAMPLE_COL_SOLVE_Q_NEXT] = float(i_samp)
        group_tails[matched_group] = i_samp


@njit(fastmath=True, cache=True)
def _accumulate_bilinear_hit(image, image_size, row_f, col_f, value):
    """Deposit ``value`` into the four neighboring pixels around a float hit."""

    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)
    deposited = False

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            image[rr, cc] += value * w_row * w_col
            deposited = True

    return deposited


@njit(fastmath=True, cache=True)
def _candidate_has_detector_support(image_size, row_f, col_f):
    """Return whether a floating detector hit reaches at least one image pixel."""

    if not np.isfinite(row_f) or not np.isfinite(col_f):
        return False

    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col > 0.0:
                return True
    return False


@njit(cache=True)
def _event_sample_unit_interval(sample_idx: int, event_idx: int, stream: int = 0) -> float:
    x = (
        0.5
        + (sample_idx + 1) * 0.6180339887498949
        + (event_idx + 1) * 0.41421356237309503
        + (stream + 1) * 0.7320508075688772
    )
    return x - np.floor(x)


def _weighted_event_targets(total_mass: float, event_count: int, sample_idx: int) -> np.ndarray:
    total_mass_f = float(total_mass)
    event_count_i = int(event_count)
    if not np.isfinite(total_mass_f) or total_mass_f <= 0.0 or event_count_i <= 0:
        return np.empty((0,), dtype=np.float64)
    targets = np.empty(event_count_i, dtype=np.float64)
    for event_idx in range(event_count_i):
        targets[event_idx] = (
            _event_sample_unit_interval(int(sample_idx), int(event_idx), stream=0) * total_mass_f
        )
    targets.sort()
    return targets


def _select_weighted_event_indices_from_targets(
    masses: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    mass_arr = np.asarray(masses, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(targets, dtype=np.float64).reshape(-1)
    if mass_arr.size <= 0 or target_arr.size <= 0:
        return np.empty((0,), dtype=np.int64)

    selected = np.empty(target_arr.shape[0], dtype=np.int64)
    cumulative = 0.0
    target_idx = 0
    for mass_idx, mass in enumerate(mass_arr):
        if not np.isfinite(mass) or mass <= 0.0:
            continue
        cumulative += float(mass)
        while target_idx < target_arr.shape[0] and cumulative > float(target_arr[target_idx]):
            selected[target_idx] = int(mass_idx)
            target_idx += 1
    if target_idx <= 0:
        return np.empty((0,), dtype=np.int64)
    if target_idx < target_arr.shape[0]:
        selected[target_idx:] = selected[target_idx - 1]
    return selected


def _weighted_event_deposit(total_mass: float, event_count: int) -> float:
    total_mass_f = float(total_mass)
    event_count_i = int(event_count)
    if not np.isfinite(total_mass_f) or total_mass_f <= 0.0 or event_count_i <= 0:
        return 0.0
    return total_mass_f / float(event_count_i)


def _candidate_branch_id(H: float, K: float, phi_f: float, qx: float) -> int:
    if abs(float(H)) < 1.0e-12 and abs(float(K)) < 1.0e-12:
        return 0
    phi_deg = float(phi_f) * (180.0 / pi)
    if phi_deg > SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG:
        return 1
    if phi_deg < -SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG:
        return 0
    return 1 if float(qx) > 0.0 else 0


def _candidate_top_mosaic_distance(
    sample_idx: int,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    wavelength_array,
) -> float:
    metric = 0.0
    if beam_x_array is not None and sample_idx < np.asarray(beam_x_array).shape[0]:
        beam_x_val = float(np.asarray(beam_x_array)[sample_idx])
        if np.isfinite(beam_x_val):
            metric += beam_x_val * beam_x_val
    if beam_y_array is not None and sample_idx < np.asarray(beam_y_array).shape[0]:
        beam_y_val = float(np.asarray(beam_y_array)[sample_idx])
        if np.isfinite(beam_y_val):
            metric += beam_y_val * beam_y_val
    if theta_array is not None and sample_idx < np.asarray(theta_array).shape[0]:
        theta_val = float(np.asarray(theta_array)[sample_idx])
        if np.isfinite(theta_val):
            metric += theta_val * theta_val
    if phi_array is not None and sample_idx < np.asarray(phi_array).shape[0]:
        phi_val = float(np.asarray(phi_array)[sample_idx])
        if np.isfinite(phi_val):
            metric += phi_val * phi_val
    if wavelength_array is not None:
        wavelength_arr = np.asarray(wavelength_array, dtype=np.float64).reshape(-1)
        if sample_idx < wavelength_arr.shape[0]:
            wavelength_val = float(wavelength_arr[sample_idx])
            if np.isfinite(wavelength_val):
                finite_wavelengths = wavelength_arr[np.isfinite(wavelength_arr)]
                if finite_wavelengths.size > 0:
                    metric += abs(wavelength_val - float(np.mean(finite_wavelengths)))
    return float(metric)


def _candidate_valid_mass(value: float) -> bool:
    return bool(np.isfinite(value) and float(value) > 0.0)


def _lookup_fast_optics(k0, n2_samp, n2_real, thickness, theta):
    Tf_s = fresnel_transmission(theta, n2_samp, True, False)
    Tf_p = fresnel_transmission(theta, n2_samp, False, False)
    Tf2 = 0.5 * (
        (np.real(Tf_s) * np.real(Tf_s) + np.imag(Tf_s) * np.imag(Tf_s))
        + (np.real(Tf_p) * np.real(Tf_p) + np.imag(Tf_p) * np.imag(Tf_p))
    )
    Tf2 = _sanitize_transmission_power(Tf2)
    _, im_k_z_f = ktz_components(k0, n2_samp, theta)
    if thickness > 0.0:
        L_out = thickness
    else:
        L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1.0e-30)
    out_angle = acos(_clamp(cos(theta) * n2_real, -1.0, 1.0))
    return float(Tf2), float(im_k_z_f), float(L_out), float(out_angle)


def _project_weighted_candidate(
    *,
    H,
    K,
    L,
    Qx,
    Qy,
    Qz,
    I_Q,
    reflection_intensity,
    sample_weight,
    debye_x_sq,
    debye_y_sq,
    center,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    I_plane,
    k_x_scat,
    k_y_scat,
    re_k_z,
    im_k_z,
    k_scat,
    k0,
    Ti2,
    L_in,
    n2_real,
    n2_samp,
    eps2,
    thickness,
    optics_mode,
    pixel_size_m,
    image_size,
    exit_projection_mode,
    projection_debug_enabled=False,
    projection_debug_counters=None,
    projection_debug_reject_count=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
):
    pixel_size_eff = float(pixel_size_m)
    if (not np.isfinite(pixel_size_eff)) or pixel_size_eff <= 0.0:
        pixel_size_eff = 100e-6
    pixel_scale = 1.0 / pixel_size_eff
    use_exact_optics = int(optics_mode) == OPTICS_MODE_EXACT
    eps3 = 1.0 + 0.0j

    k_tx_prime = float(Qx) + float(k_x_scat)
    k_ty_prime = float(Qy) + float(k_y_scat)
    k_tz_prime = float(Qz) + float(re_k_z)
    kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
    if kr < 1e-12:
        twotheta_t_prime = 0.0
    else:
        twotheta_t_prime = np.arctan(k_tz_prime / kr)

    th_t_out = np.abs(twotheta_t_prime)
    if use_exact_optics:
        k0_sq = float(k0) * float(k0)
        k_par_f = kr
        k_par_f_sq = k_par_f * k_par_f
        kz2_f = _kz_branch_decay((eps2 * k0_sq) - k_par_f_sq)
        kz3_f = _kz_branch_decay((k0_sq - k_par_f_sq) + 0.0j)
        Tf_s = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, True)
        Tf_p = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, False)
        Tf2 = 0.5 * (
            _fresnel_power_t_exact(Tf_s, kz2_f, kz3_f, eps2, eps3, True)
            + _fresnel_power_t_exact(Tf_p, kz2_f, kz3_f, eps2, eps3, False)
        )
        Tf2 = _sanitize_transmission_power(Tf2)
        im_k_z_f = np.abs(kz2_f.imag)
        if thickness > 0.0:
            L_out = thickness
        else:
            L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1.0e-30)
        cos_out = _clamp(kr / np.maximum(float(k0), 1.0e-30), -1.0, 1.0)
        twotheta_t = np.arccos(cos_out) * np.sign(twotheta_t_prime)
        k_out_mag = float(k0)
    else:
        Tf2, im_k_z_f, L_out, twotheta_t_abs = _lookup_fast_optics(
            float(k0),
            n2_samp,
            float(n2_real),
            float(thickness),
            float(th_t_out),
        )
        twotheta_t = twotheta_t_abs * np.sign(twotheta_t_prime)
        k_out_mag = float(k_scat)

    if projection_debug_enabled and projection_debug_counters is not None:
        _projection_debug_log_ray(
            projection_debug_counters,
            twotheta_t_prime,
            twotheta_t,
            n2_real,
        )

    prop_att = np.exp(-2.0 * float(im_k_z) * float(L_in)) * np.exp(
        -2.0 * float(im_k_z_f) * float(L_out)
    )
    if not np.isfinite(prop_att) or prop_att <= 0.0:
        if (
            projection_debug_enabled
            and projection_debug_counters is not None
            and projection_debug_reject_count is not None
            and projection_debug_reject_records is not None
        ):
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_PROP_ATT_REJECT] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                np.nan,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_PROP_ATT,
            )
        return False, np.nan, np.nan, np.nan, 0.0

    prop_fac = float(Ti2) * float(Tf2) * float(prop_att)
    if not np.isfinite(prop_fac) or prop_fac <= 0.0:
        if (
            projection_debug_enabled
            and projection_debug_counters is not None
            and projection_debug_reject_count is not None
            and projection_debug_reject_records is not None
        ):
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_PROP_FAC_REJECT] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                np.nan,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_PROP_FAC,
            )
        return False, np.nan, np.nan, np.nan, 0.0

    phi_f = np.arctan2(k_tx_prime, k_ty_prime)
    if int(exit_projection_mode) == EXIT_PROJECTION_INTERNAL:
        k_norm = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime + k_tz_prime * k_tz_prime)
        if k_norm > 1.0e-30:
            kf = np.array(
                [k_tx_prime / k_norm, k_ty_prime / k_norm, k_tz_prime / k_norm],
                dtype=np.float64,
            )
        elif k_out_mag > 1.0e-30:
            kf = np.array(
                [
                    k_out_mag * np.cos(twotheta_t) * np.sin(phi_f),
                    k_out_mag * np.cos(twotheta_t) * np.cos(phi_f),
                    k_out_mag * np.sin(twotheta_t),
                ],
                dtype=np.float64,
            )
        else:
            return False, np.nan, np.nan, np.nan, 0.0
    else:
        kf = np.array(
            [
                k_out_mag * np.cos(twotheta_t) * np.sin(phi_f),
                k_out_mag * np.cos(twotheta_t) * np.cos(phi_f),
                k_out_mag * np.sin(twotheta_t),
            ],
            dtype=np.float64,
        )

    kf_prime = np.asarray(R_sample, dtype=np.float64) @ kf
    dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
    if not valid_det:
        return False, np.nan, np.nan, np.nan, 0.0

    plane_to_det = np.array(
        [dx - Detector_Pos[0], dy - Detector_Pos[1], dz - Detector_Pos[2]],
        dtype=np.float64,
    )
    x_det = float(np.dot(plane_to_det, e1_det))
    y_det = float(np.dot(plane_to_det, e2_det))
    if not np.isfinite(x_det) or not np.isfinite(y_det):
        return False, np.nan, np.nan, np.nan, 0.0

    row_f = float(center[0]) - y_det * pixel_scale
    col_f = float(center[1]) + x_det * pixel_scale
    if not np.isfinite(row_f) or not np.isfinite(col_f):
        return False, np.nan, np.nan, np.nan, 0.0
    if not _candidate_has_detector_support(int(image_size), float(row_f), float(col_f)):
        return False, row_f, col_f, np.nan, 0.0

    mass = (
        float(reflection_intensity)
        * float(sample_weight)
        * float(I_Q)
        * float(prop_fac)
        * exp(-float(Qz) * float(Qz) * float(debye_x_sq))
        * exp(-(float(Qx) * float(Qx) + float(Qy) * float(Qy)) * float(debye_y_sq))
    )
    if not _candidate_valid_mass(mass):
        return False, row_f, col_f, phi_f, 0.0

    if projection_debug_enabled and projection_debug_row_hit_counts is not None:
        row_idx = int(np.rint(row_f))
        if 0 <= row_idx < projection_debug_row_hit_counts.shape[0]:
            projection_debug_row_hit_counts[row_idx] += 1
            projection_debug_row_tthp_sums[row_idx] += twotheta_t_prime
            projection_debug_row_tth_sums[row_idx] += twotheta_t

    return True, row_f, col_f, phi_f, float(mass)


_DEFAULT_SOLVE_Q = solve_q
_DEFAULT_PRECOMPUTE_SAMPLE_TERMS = _precompute_sample_terms
_DEFAULT_PROJECT_WEIGHTED_CANDIDATE = _project_weighted_candidate


@njit(fastmath=True, cache=True)
def _candidate_branch_id_fast(H, K, phi_f, qx):
    if abs(H) < 1.0e-12 and abs(K) < 1.0e-12:
        return 0
    phi_deg = phi_f * (180.0 / pi)
    if phi_deg > SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG:
        return 1
    if phi_deg < -SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG:
        return 0
    return 1 if qx > 0.0 else 0


@njit(fastmath=True, cache=True)
def _project_weighted_candidate_fast(
    Qx,
    Qy,
    Qz,
    I_Q,
    reflection_intensity,
    sample_weight,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    I_plane_x,
    I_plane_y,
    I_plane_z,
    k_x_scat,
    k_y_scat,
    re_k_z,
    im_k_z,
    k_scat,
    k0,
    Ti2,
    L_in,
    n2_real,
    n2_samp,
    eps2,
    thickness,
    optics_mode,
    pixel_scale,
    image_size,
    exit_projection_mode,
    fast_optics_lut_ready,
    fast_optics_lut_row,
):
    use_exact_optics = optics_mode == OPTICS_MODE_EXACT
    eps3 = 1.0 + 0.0j

    k_tx_prime = Qx + k_x_scat
    k_ty_prime = Qy + k_y_scat
    k_tz_prime = Qz + re_k_z
    kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
    if kr < 1e-12:
        twotheta_t_prime = 0.0
    else:
        twotheta_t_prime = np.arctan(k_tz_prime / kr)

    th_t_out = np.abs(twotheta_t_prime)
    if use_exact_optics:
        k0_sq = k0 * k0
        k_par_f = kr
        k_par_f_sq = k_par_f * k_par_f
        kz2_f = _kz_branch_decay((eps2 * k0_sq) - k_par_f_sq)
        kz3_f = _kz_branch_decay((k0_sq - k_par_f_sq) + 0.0j)
        Tf_s = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, True)
        Tf_p = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, False)
        Tf2 = 0.5 * (
            _fresnel_power_t_exact(Tf_s, kz2_f, kz3_f, eps2, eps3, True)
            + _fresnel_power_t_exact(Tf_p, kz2_f, kz3_f, eps2, eps3, False)
        )
        Tf2 = _sanitize_transmission_power(Tf2)
        im_k_z_f = np.abs(kz2_f.imag)
        if thickness > 0.0:
            L_out = thickness
        else:
            L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1.0e-30)
        cos_out = _clamp(kr / np.maximum(k0, 1.0e-30), -1.0, 1.0)
        twotheta_t = np.arccos(cos_out) * np.sign(twotheta_t_prime)
        k_out_mag = k0
    else:
        if fast_optics_lut_ready:
            Tf2, im_k_z_f, L_out, twotheta_t_abs = _lookup_fast_optics_lut_row(
                fast_optics_lut_row,
                th_t_out,
            )
        else:
            return False, np.nan, np.nan, np.nan, 0.0
        twotheta_t = twotheta_t_abs * np.sign(twotheta_t_prime)
        k_out_mag = k_scat

    prop_att = np.exp(-2.0 * im_k_z * L_in) * np.exp(-2.0 * im_k_z_f * L_out)
    if (not np.isfinite(prop_att)) or prop_att <= 0.0:
        return False, np.nan, np.nan, np.nan, 0.0

    prop_fac = Ti2 * Tf2 * prop_att
    if (not np.isfinite(prop_fac)) or prop_fac <= 0.0:
        return False, np.nan, np.nan, np.nan, 0.0

    phi_f = np.arctan2(k_tx_prime, k_ty_prime)
    if exit_projection_mode == EXIT_PROJECTION_INTERNAL:
        k_norm = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime + k_tz_prime * k_tz_prime)
        if k_norm > 1.0e-30:
            kf_x = k_tx_prime / k_norm
            kf_y = k_ty_prime / k_norm
            kf_z = k_tz_prime / k_norm
        elif k_out_mag > 1.0e-30:
            kf_x = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
            kf_y = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
            kf_z = k_out_mag * np.sin(twotheta_t)
        else:
            return False, np.nan, np.nan, np.nan, 0.0
    else:
        kf_x = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
        kf_y = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
        kf_z = k_out_mag * np.sin(twotheta_t)

    kf_prime_x = R_sample[0, 0] * kf_x + R_sample[0, 1] * kf_y + R_sample[0, 2] * kf_z
    kf_prime_y = R_sample[1, 0] * kf_x + R_sample[1, 1] * kf_y + R_sample[1, 2] * kf_z
    kf_prime_z = R_sample[2, 0] * kf_x + R_sample[2, 1] * kf_y + R_sample[2, 2] * kf_z

    denom = (
        kf_prime_x * n_det_rot[0]
        + kf_prime_y * n_det_rot[1]
        + kf_prime_z * n_det_rot[2]
    )
    if abs(denom) < 1e-14:
        dist = (
            (I_plane_x - Detector_Pos[0]) * n_det_rot[0]
            + (I_plane_y - Detector_Pos[1]) * n_det_rot[1]
            + (I_plane_z - Detector_Pos[2]) * n_det_rot[2]
        )
        if abs(dist) < 1e-6:
            dx = I_plane_x
            dy = I_plane_y
            dz = I_plane_z
        else:
            return False, np.nan, np.nan, np.nan, 0.0
    else:
        num = (
            (Detector_Pos[0] - I_plane_x) * n_det_rot[0]
            + (Detector_Pos[1] - I_plane_y) * n_det_rot[1]
            + (Detector_Pos[2] - I_plane_z) * n_det_rot[2]
        )
        t = num / denom
        if t < -1e-9:
            return False, np.nan, np.nan, np.nan, 0.0
        if t < 0.0:
            t = 0.0
        dx = I_plane_x + t * kf_prime_x
        dy = I_plane_y + t * kf_prime_y
        dz = I_plane_z + t * kf_prime_z

    plane_to_det_x = dx - Detector_Pos[0]
    plane_to_det_y = dy - Detector_Pos[1]
    plane_to_det_z = dz - Detector_Pos[2]
    x_det = (
        plane_to_det_x * e1_det[0]
        + plane_to_det_y * e1_det[1]
        + plane_to_det_z * e1_det[2]
    )
    y_det = (
        plane_to_det_x * e2_det[0]
        + plane_to_det_y * e2_det[1]
        + plane_to_det_z * e2_det[2]
    )
    if (not np.isfinite(x_det)) or (not np.isfinite(y_det)):
        return False, np.nan, np.nan, np.nan, 0.0

    row_f = center_row - y_det * pixel_scale
    col_f = center_col + x_det * pixel_scale
    if (not np.isfinite(row_f)) or (not np.isfinite(col_f)):
        return False, np.nan, np.nan, np.nan, 0.0
    if not _candidate_has_detector_support(image_size, row_f, col_f):
        return False, row_f, col_f, np.nan, 0.0

    mass = (
        reflection_intensity
        * sample_weight
        * I_Q
        * prop_fac
        * exp(-Qz * Qz * debye_x_sq)
        * exp(-(Qx * Qx + Qy * Qy) * debye_y_sq)
    )
    if (not np.isfinite(mass)) or mass <= 0.0:
        return False, row_f, col_f, phi_f, 0.0
    return True, row_f, col_f, phi_f, mass


@njit(fastmath=True, cache=True)
def _weighted_event_update_representative(
    rep_slot,
    sample_weight,
    sample_top_distance,
    mass,
    sample_idx,
    peak_idx,
    q_idx,
    row_f,
    col_f,
    phi_f,
    H,
    K,
    L,
    representative_valid,
    representative_neg_sample_weight,
    representative_top_distance,
    representative_neg_mass,
    representative_sample_idx,
    representative_peak_idx,
    representative_q_idx,
    representative_rows,
):
    if rep_slot < 0 or rep_slot >= representative_valid.shape[0]:
        return

    effective_sample_weight = 1.0
    if np.isfinite(sample_weight) and sample_weight > 0.0:
        effective_sample_weight = sample_weight
    neg_sample_weight = -effective_sample_weight

    better = False
    if representative_valid[rep_slot] == 0:
        better = True
    else:
        current_neg_sample_weight = representative_neg_sample_weight[rep_slot]
        current_top_distance = representative_top_distance[rep_slot]
        current_neg_mass = representative_neg_mass[rep_slot]
        current_sample_idx = representative_sample_idx[rep_slot]
        current_peak_idx = representative_peak_idx[rep_slot]
        current_q_idx = representative_q_idx[rep_slot]
        neg_mass = -mass
        if neg_sample_weight < current_neg_sample_weight:
            better = True
        elif neg_sample_weight == current_neg_sample_weight:
            if sample_top_distance < current_top_distance:
                better = True
            elif sample_top_distance == current_top_distance:
                if neg_mass < current_neg_mass:
                    better = True
                elif neg_mass == current_neg_mass:
                    if sample_idx < current_sample_idx:
                        better = True
                    elif sample_idx == current_sample_idx:
                        if peak_idx < current_peak_idx:
                            better = True
                        elif peak_idx == current_peak_idx and q_idx < current_q_idx:
                            better = True

    if not better:
        return

    representative_valid[rep_slot] = 1
    representative_neg_sample_weight[rep_slot] = neg_sample_weight
    representative_top_distance[rep_slot] = sample_top_distance
    representative_neg_mass[rep_slot] = -mass
    representative_sample_idx[rep_slot] = sample_idx
    representative_peak_idx[rep_slot] = peak_idx
    representative_q_idx[rep_slot] = q_idx
    representative_rows[rep_slot, 0] = mass
    representative_rows[rep_slot, 1] = col_f
    representative_rows[rep_slot, 2] = row_f
    representative_rows[rep_slot, 3] = phi_f
    representative_rows[rep_slot, 4] = H
    representative_rows[rep_slot, 5] = K
    representative_rows[rep_slot, 6] = L
    representative_rows[rep_slot, 7] = float(peak_idx)
    representative_rows[rep_slot, 8] = float(q_idx)
    representative_rows[rep_slot, 9] = float(sample_idx)


@njit(fastmath=True, cache=True)
def _weighted_event_pass1_for_qset(
    all_q,
    peak_idx,
    sample_idx,
    H,
    K,
    L,
    reflection_intensity,
    sample_weight,
    sample_top_distance,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    sample_n2_array,
    sample_eps2_array,
    thickness,
    optics_mode,
    pixel_scale,
    image_size,
    exit_projection_mode,
    fast_optics_lut_ready,
    fast_optics_lut_row,
    save_flag,
    q_data,
    q_count,
    representative_slot_by_peak_branch,
    representative_valid,
    representative_neg_sample_weight,
    representative_top_distance,
    representative_neg_mass,
    representative_sample_idx,
    representative_peak_idx,
    representative_q_idx,
    representative_rows,
):
    total_mass = 0.0
    valid_count = 0
    project_calls = 0

    I_plane_x = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_X]
    I_plane_y = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Y]
    I_plane_z = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Z]
    k_x_scat = sample_terms[sample_idx, _SAMPLE_COL_KX_SCAT]
    k_y_scat = sample_terms[sample_idx, _SAMPLE_COL_KY_SCAT]
    re_k_z = sample_terms[sample_idx, _SAMPLE_COL_RE_KZ]
    im_k_z = sample_terms[sample_idx, _SAMPLE_COL_IM_KZ]
    k_scat = sample_terms[sample_idx, _SAMPLE_COL_K_SCAT]
    k0 = sample_terms[sample_idx, _SAMPLE_COL_K0]
    Ti2 = sample_terms[sample_idx, _SAMPLE_COL_TI2]
    L_in = sample_terms[sample_idx, _SAMPLE_COL_L_IN]
    n2_real = sample_terms[sample_idx, _SAMPLE_COL_N2_REAL]
    n2_samp = sample_n2_array[sample_idx]
    eps2 = sample_eps2_array[sample_idx]

    for q_idx in range(all_q.shape[0]):
        Qx = all_q[q_idx, 0]
        Qy = all_q[q_idx, 1]
        Qz = all_q[q_idx, 2]
        I_Q = all_q[q_idx, 3]
        project_calls += 1
        candidate_valid, row_f, col_f, phi_f, mass = _project_weighted_candidate_fast(
            Qx,
            Qy,
            Qz,
            I_Q,
            reflection_intensity,
            sample_weight,
            debye_x_sq,
            debye_y_sq,
            center_row,
            center_col,
            R_sample,
            n_det_rot,
            Detector_Pos,
            e1_det,
            e2_det,
            I_plane_x,
            I_plane_y,
            I_plane_z,
            k_x_scat,
            k_y_scat,
            re_k_z,
            im_k_z,
            k_scat,
            k0,
            Ti2,
            L_in,
            n2_real,
            n2_samp,
            eps2,
            thickness,
            optics_mode,
            pixel_scale,
            image_size,
            exit_projection_mode,
            fast_optics_lut_ready,
            fast_optics_lut_row,
        )
        if save_flag == 1 and np.isfinite(mass) and mass > 0.0 and q_count[peak_idx] < q_data.shape[1]:
            q_store_idx = q_count[peak_idx]
            q_data[peak_idx, q_store_idx, 0] = Qx
            q_data[peak_idx, q_store_idx, 1] = Qy
            q_data[peak_idx, q_store_idx, 2] = Qz
            q_data[peak_idx, q_store_idx, 3] = mass
            q_data[peak_idx, q_store_idx, 4] = float(sample_idx)
            q_count[peak_idx] += 1
        if not candidate_valid:
            continue
        total_mass += mass
        valid_count += 1
        branch_id = _candidate_branch_id_fast(H, K, phi_f, Qx)
        rep_slot = representative_slot_by_peak_branch[peak_idx, branch_id]
        if rep_slot >= 0:
            _weighted_event_update_representative(
                rep_slot,
                sample_weight,
                sample_top_distance,
                mass,
                sample_idx,
                peak_idx,
                q_idx,
                row_f,
                col_f,
                phi_f,
                H,
                K,
                L,
                representative_valid,
                representative_neg_sample_weight,
                representative_top_distance,
                representative_neg_mass,
                representative_sample_idx,
                representative_peak_idx,
                representative_q_idx,
                representative_rows,
            )

    return total_mass, valid_count, project_calls


@njit(fastmath=True, cache=True)
def _weighted_event_pass2_for_qset(
    all_q,
    peak_idx,
    sample_idx,
    H,
    K,
    L,
    reflection_intensity,
    sample_weight,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    sample_n2_array,
    sample_eps2_array,
    thickness,
    optics_mode,
    pixel_scale,
    image_size,
    exit_projection_mode,
    fast_optics_lut_ready,
    fast_optics_lut_row,
    targets,
    target_idx,
    cumulative_mass,
    deposit,
    collect_tables,
    flat_event_rows,
    flat_event_peak_indices,
    flat_event_count,
    event_counts,
    accumulate_image,
    image,
    cache_keys,
    cache_values,
    cache_entry_count,
    cache_flush_limit,
):
    project_calls = 0
    pass2_total_mass = 0.0
    selected_events = 0
    have_last_valid = False
    last_row_f = np.nan
    last_col_f = np.nan
    last_phi_f = np.nan
    last_peak_idx = -1
    last_H = np.nan
    last_K = np.nan
    last_L = np.nan

    I_plane_x = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_X]
    I_plane_y = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Y]
    I_plane_z = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Z]
    k_x_scat = sample_terms[sample_idx, _SAMPLE_COL_KX_SCAT]
    k_y_scat = sample_terms[sample_idx, _SAMPLE_COL_KY_SCAT]
    re_k_z = sample_terms[sample_idx, _SAMPLE_COL_RE_KZ]
    im_k_z = sample_terms[sample_idx, _SAMPLE_COL_IM_KZ]
    k_scat = sample_terms[sample_idx, _SAMPLE_COL_K_SCAT]
    k0 = sample_terms[sample_idx, _SAMPLE_COL_K0]
    Ti2 = sample_terms[sample_idx, _SAMPLE_COL_TI2]
    L_in = sample_terms[sample_idx, _SAMPLE_COL_L_IN]
    n2_real = sample_terms[sample_idx, _SAMPLE_COL_N2_REAL]
    n2_samp = sample_n2_array[sample_idx]
    eps2 = sample_eps2_array[sample_idx]

    for q_idx in range(all_q.shape[0]):
        Qx = all_q[q_idx, 0]
        Qy = all_q[q_idx, 1]
        Qz = all_q[q_idx, 2]
        I_Q = all_q[q_idx, 3]
        project_calls += 1
        candidate_valid, row_f, col_f, phi_f, mass = _project_weighted_candidate_fast(
            Qx,
            Qy,
            Qz,
            I_Q,
            reflection_intensity,
            sample_weight,
            debye_x_sq,
            debye_y_sq,
            center_row,
            center_col,
            R_sample,
            n_det_rot,
            Detector_Pos,
            e1_det,
            e2_det,
            I_plane_x,
            I_plane_y,
            I_plane_z,
            k_x_scat,
            k_y_scat,
            re_k_z,
            im_k_z,
            k_scat,
            k0,
            Ti2,
            L_in,
            n2_real,
            n2_samp,
            eps2,
            thickness,
            optics_mode,
            pixel_scale,
            image_size,
            exit_projection_mode,
            fast_optics_lut_ready,
            fast_optics_lut_row,
        )
        if not candidate_valid:
            continue

        cumulative_mass += mass
        pass2_total_mass += mass
        have_last_valid = True
        last_row_f = row_f
        last_col_f = col_f
        last_phi_f = phi_f
        last_peak_idx = peak_idx
        last_H = H
        last_K = K
        last_L = L

        hit_count = 0
        while target_idx < targets.shape[0] and cumulative_mass > targets[target_idx]:
            hit_count += 1
            target_idx += 1

        if hit_count <= 0:
            continue

        selected_events += hit_count
        event_counts[peak_idx, sample_idx] += hit_count

        if accumulate_image:
            deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                image_size,
                row_f,
                col_f,
                float(hit_count) * deposit,
                cache_keys,
                cache_values,
                cache_entry_count,
                cache_flush_limit,
            )
            if needs_flush:
                cache_entry_count = _flush_local_pixel_cache(
                    image,
                    image_size,
                    cache_keys,
                    cache_values,
                )
                deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                    image_size,
                    row_f,
                    col_f,
                    float(hit_count) * deposit,
                    cache_keys,
                    cache_values,
                    cache_entry_count,
                    cache_flush_limit,
                )
                if needs_flush:
                    deposited = _accumulate_bilinear_hit(
                        image,
                        image_size,
                        row_f,
                        col_f,
                        float(hit_count) * deposit,
                    )
                    cache_entry_count = 0
            if not deposited:
                continue

        if collect_tables:
            for event_idx in range(hit_count):
                flat_event_peak_indices[flat_event_count] = peak_idx
                flat_event_rows[flat_event_count, 0] = deposit
                flat_event_rows[flat_event_count, 1] = col_f
                flat_event_rows[flat_event_count, 2] = row_f
                flat_event_rows[flat_event_count, 3] = phi_f
                flat_event_rows[flat_event_count, 4] = H
                flat_event_rows[flat_event_count, 5] = K
                flat_event_rows[flat_event_count, 6] = L
                flat_event_rows[flat_event_count, 7] = np.nan
                flat_event_rows[flat_event_count, 8] = np.nan
                flat_event_rows[flat_event_count, 9] = float(sample_idx)
                flat_event_count += 1

    return (
        target_idx,
        cumulative_mass,
        flat_event_count,
        cache_entry_count,
        project_calls,
        pass2_total_mass,
        selected_events,
        have_last_valid,
        last_row_f,
        last_col_f,
        last_phi_f,
        last_peak_idx,
        last_H,
        last_K,
        last_L,
    )


@njit(fastmath=True, cache=True)
def _choose_nominal_sample_index(sample_terms, preferred_idx):
    n_samp = sample_terms.shape[0]
    if 0 <= preferred_idx < n_samp:
        if sample_terms[preferred_idx, _SAMPLE_COL_VALID] > 0.5:
            return preferred_idx
    for i_samp in range(n_samp):
        if sample_terms[i_samp, _SAMPLE_COL_VALID] > 0.5:
            return i_samp
    return -1


@njit(fastmath=True, cache=True)
def _nominal_reflection_visible(
    H,
    K,
    L,
    G_vec,
    image_size,
    center,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    best_idx,
    sigma_rad,
    gamma_pv,
    optics_mode,
    exit_projection_mode,
    projection_debug_enabled,
    projection_debug_counters,
    projection_debug_reject_count,
    projection_debug_reject_records,
):
    nominal_idx = _choose_nominal_sample_index(sample_terms, best_idx)
    if nominal_idx < 0:
        if projection_debug_enabled:
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_NO_VALID_SAMPLE] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                _PROJECTION_DEBUG_REASON_NO_VALID_SAMPLE,
            )
        return False, -1, True

    I_plane = np.empty(3, dtype=np.float64)
    I_plane[0] = sample_terms[nominal_idx, _SAMPLE_COL_I_PLANE_X]
    I_plane[1] = sample_terms[nominal_idx, _SAMPLE_COL_I_PLANE_Y]
    I_plane[2] = sample_terms[nominal_idx, _SAMPLE_COL_I_PLANE_Z]

    k_x_scat = sample_terms[nominal_idx, _SAMPLE_COL_KX_SCAT]
    k_y_scat = sample_terms[nominal_idx, _SAMPLE_COL_KY_SCAT]
    re_k_z = sample_terms[nominal_idx, _SAMPLE_COL_RE_KZ]
    k_scat = sample_terms[nominal_idx, _SAMPLE_COL_K_SCAT]
    k0 = sample_terms[nominal_idx, _SAMPLE_COL_K0]
    n2_real = sample_terms[nominal_idx, _SAMPLE_COL_N2_REAL]

    k_tx_prime = G_vec[0] + k_x_scat
    k_ty_prime = G_vec[1] + k_y_scat
    k_tz_prime = G_vec[2] + re_k_z
    kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
    if kr < 1e-12:
        twotheta_t_prime = 0.0
    else:
        twotheta_t_prime = np.arctan(k_tz_prime / kr)

    if optics_mode == OPTICS_MODE_EXACT:
        cos_out = _clamp(kr / np.maximum(k0, 1.0e-30), -1.0, 1.0)
        twotheta_t = np.arccos(cos_out) * np.sign(twotheta_t_prime)
        k_out_mag = k0
    else:
        twotheta_t = np.arccos(
            _clamp(
                np.cos(twotheta_t_prime) * n2_real,
                -1.0,
                1.0,
            )
        ) * np.sign(twotheta_t_prime)
        k_out_mag = k_scat

    if projection_debug_enabled:
        _projection_debug_log_ray(
            projection_debug_counters,
            twotheta_t_prime,
            twotheta_t,
            n2_real,
        )

    kf = np.empty(3, dtype=np.float64)
    kf_prime = np.empty(3, dtype=np.float64)
    phi_f = np.arctan2(k_tx_prime, k_ty_prime)
    if exit_projection_mode == EXIT_PROJECTION_INTERNAL:
        k_norm = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime + k_tz_prime * k_tz_prime)
        if k_norm > 1.0e-30:
            kf[0] = k_tx_prime / k_norm
            kf[1] = k_ty_prime / k_norm
            kf[2] = k_tz_prime / k_norm
        elif k_out_mag > 1.0e-30:
            kf[0] = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
            kf[1] = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
            kf[2] = k_out_mag * np.sin(twotheta_t)
        else:
            if projection_debug_enabled:
                projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                _projection_debug_record_reject(
                    projection_debug_reject_count,
                    projection_debug_reject_records,
                    H,
                    K,
                    L,
                    twotheta_t_prime,
                    twotheta_t,
                    kr,
                    k0,
                    n2_real,
                    _PROJECTION_DEBUG_REASON_PROJECTION_NORM,
                )
            return False, nominal_idx, False
    else:
        kf[0] = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
        kf[1] = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
        kf[2] = k_out_mag * np.sin(twotheta_t)
    kf_prime[0] = R_sample[0, 0] * kf[0] + R_sample[0, 1] * kf[1] + R_sample[0, 2] * kf[2]
    kf_prime[1] = R_sample[1, 0] * kf[0] + R_sample[1, 1] * kf[1] + R_sample[1, 2] * kf[2]
    kf_prime[2] = R_sample[2, 0] * kf[0] + R_sample[2, 1] * kf[1] + R_sample[2, 2] * kf[2]

    pixel_scale = 1.0 / 100e-6
    sigma_pad = _LOCAL_ARC_GAUSS_SIGMAS * max(sigma_rad, 1.0e-12)
    gamma_pad = _LOCAL_ARC_LORENTZ_GAMMAS * max(gamma_pv, 1.0e-12)
    angle_pad = max(sigma_pad, gamma_pad, 4.0 * _LOCAL_ARC_MIN_DTHETA)

    min_tth = np.inf
    max_tth = -np.inf
    for i_row in range(2):
        row = 0.0 if i_row == 0 else float(image_size - 1)
        for i_col in range(2):
            col = 0.0 if i_col == 0 else float(image_size - 1)
            x_det = (col - center[1]) / pixel_scale
            y_det = (center[0] - row) / pixel_scale
            ray_x = Detector_Pos[0] + x_det * e1_det[0] + y_det * e2_det[0] - I_plane[0]
            ray_y = Detector_Pos[1] + x_det * e1_det[1] + y_det * e2_det[1] - I_plane[1]
            ray_z = Detector_Pos[2] + x_det * e1_det[2] + y_det * e2_det[2] - I_plane[2]
            ray_r = sqrt(ray_x * ray_x + ray_y * ray_y)
            tth_corner = np.arctan2(ray_z, ray_r)
            if tth_corner < min_tth:
                min_tth = tth_corner
            if tth_corner > max_tth:
                max_tth = tth_corner
        # Include the detector center line in the same pass.
        center_col = center[1]
        x_det = (center_col - center[1]) / pixel_scale
        y_det = (center[0] - row) / pixel_scale
        ray_x = Detector_Pos[0] + x_det * e1_det[0] + y_det * e2_det[0] - I_plane[0]
        ray_y = Detector_Pos[1] + x_det * e1_det[1] + y_det * e2_det[1] - I_plane[1]
        ray_z = Detector_Pos[2] + x_det * e1_det[2] + y_det * e2_det[2] - I_plane[2]
        ray_r = sqrt(ray_x * ray_x + ray_y * ray_y)
        tth_edge = np.arctan2(ray_z, ray_r)
        if tth_edge < min_tth:
            min_tth = tth_edge
        if tth_edge > max_tth:
            max_tth = tth_edge

    kf_prime_r = sqrt(kf_prime[0] * kf_prime[0] + kf_prime[1] * kf_prime[1])
    nominal_tth = np.arctan2(kf_prime[2], kf_prime_r)
    if nominal_tth < (min_tth - angle_pad) or nominal_tth > (max_tth + angle_pad):
        return False, nominal_idx, False

    dx, dy, dz, valid_det = intersect_infinite_line_plane(
        I_plane, kf_prime, Detector_Pos, n_det_rot
    )
    if not valid_det:
        if projection_debug_enabled:
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                twotheta_t,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_INVALID_DET,
            )
        return True, nominal_idx, False

    plane_to_det_x = dx - Detector_Pos[0]
    plane_to_det_y = dy - Detector_Pos[1]
    plane_to_det_z = dz - Detector_Pos[2]
    x_det = plane_to_det_x * e1_det[0] + plane_to_det_y * e1_det[1] + plane_to_det_z * e1_det[2]
    y_det = plane_to_det_x * e2_det[0] + plane_to_det_y * e2_det[1] + plane_to_det_z * e2_det[2]
    row_f = center[0] - y_det * pixel_scale
    col_f = center[1] + x_det * pixel_scale
    if not np.isfinite(row_f) or not np.isfinite(col_f):
        if projection_debug_enabled:
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                twotheta_t,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_INVALID_DET_COORD,
            )
        return True, nominal_idx, False

    det_dist = sqrt(
        Detector_Pos[0] * Detector_Pos[0]
        + Detector_Pos[1] * Detector_Pos[1]
        + Detector_Pos[2] * Detector_Pos[2]
    )
    pixel_pad = det_dist * np.tan(min(angle_pad, 0.45 * pi)) * pixel_scale
    if pixel_pad < 24.0:
        pixel_pad = 24.0

    if row_f < -pixel_pad or row_f > (float(image_size - 1) + pixel_pad):
        if projection_debug_enabled:
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                twotheta_t,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_OFF_DETECTOR,
            )
        return False, nominal_idx, False
    if col_f < -pixel_pad or col_f > (float(image_size - 1) + pixel_pad):
        if projection_debug_enabled:
            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
            _projection_debug_record_reject(
                projection_debug_reject_count,
                projection_debug_reject_records,
                H,
                K,
                L,
                twotheta_t_prime,
                twotheta_t,
                kr,
                k0,
                n2_real,
                _PROJECTION_DEBUG_REASON_OFF_DETECTOR,
            )
        return False, nominal_idx, False
    return True, nominal_idx, False


@njit(fastmath=True, cache=True)
def _ring_sample_unit_interval(peak_idx, sample_idx):
    """Return a deterministic quasi-random ``u`` in ``[0, 1)`` for ring sampling."""

    u = (
        0.5
        + (float(peak_idx) + 1.0) * 0.6180339887498949
        + (float(sample_idx) + 1.0) * 0.41421356237309503
    )
    return u - np.floor(u)


@njit(fastmath=True, cache=True)
def _sample_q_ring_solution(all_q, peak_idx, sample_idx):
    """Sample one ``solve_q`` row using the ring mass as the PDF."""

    total_mass = 0.0
    last_valid_idx = -1
    for idx in range(all_q.shape[0]):
        mass_i = all_q[idx, 3]
        if not np.isfinite(mass_i) or mass_i < _Q_RING_SAMPLE_MIN_MASS:
            continue
        total_mass += mass_i
        last_valid_idx = idx

    if total_mass <= 0.0 or last_valid_idx < 0:
        return -1, 0.0

    target_mass = _ring_sample_unit_interval(peak_idx, sample_idx) * total_mass
    running_mass = 0.0
    for idx in range(all_q.shape[0]):
        mass_i = all_q[idx, 3]
        if not np.isfinite(mass_i) or mass_i < _Q_RING_SAMPLE_MIN_MASS:
            continue
        running_mass += mass_i
        if running_mass >= target_mass:
            return idx, total_mass

    return last_valid_idx, total_mass


@njit(fastmath=True, cache=True)
def _calculate_phi_from_precomputed(
    H,
    K,
    L,
    av,
    cv,
    image,
    image_size,
    reflection_intensity,
    sigma_rad,
    gamma_pv,
    eta_pv,
    debye_x,
    debye_y,
    center,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    n2_samp_array,
    eps2_array,
    best_idx,
    save_flag,
    q_data,
    q_count,
    i_peaks_index,
    record_status=False,
    thickness=0.0,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    pixel_size_m=100e-6,
    sample_qr_ring_once=True,
    sample_weights=None,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_enabled=False,
    projection_debug_counters=None,
    projection_debug_reject_count=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
):
    """Reflection core using precomputed sample-geometry terms."""
    gz0 = 2.0 * pi * (L / cv)
    gr0 = 4.0 * pi / av * sqrt((H * H + H * K + K * K) / 3.0)
    G_vec = np.array([0.0, gr0, gz0], dtype=np.float64)

    n_samp = sample_terms.shape[0]
    debye_x_sq = debye_x * debye_x
    debye_y_sq = debye_y * debye_y
    pixel_size_eff = float(pixel_size_m)
    if (not np.isfinite(pixel_size_eff)) or pixel_size_eff <= 0.0:
        pixel_size_eff = 100e-6
    pixel_scale = 1.0 / pixel_size_eff
    save_flag_eff = int(save_flag)
    capture_aux = True
    accumulate_image = True
    if save_flag_eff >= 4:
        save_flag_eff -= 4
        accumulate_image = False
    if save_flag_eff >= 2:
        save_flag_eff -= 2
        capture_aux = False
    if capture_aux:
        max_hits = max(n_samp * 2, 16)
        pixel_hits = np.empty((max_hits, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        missed_kf = np.empty((max_hits, 3), dtype=np.float64)
    else:
        max_hits = 0
        pixel_hits = np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        missed_kf = np.empty((0, 3), dtype=np.float64)
    n_hits = 0
    n_missed = 0

    best_candidate = np.empty(HIT_ROW_WITH_PROVENANCE_WIDTH, dtype=np.float64)
    best_candidate_val = -1.0
    have_candidate = False
    recorded_nominal_hit = False
    if record_status:
        statuses = np.zeros(n_samp, dtype=np.int64)

    k_in_crystal = np.empty(3, dtype=np.float64)
    I_plane = np.empty(3, dtype=np.float64)
    kf = np.empty(3, dtype=np.float64)
    kf_prime = np.empty(3, dtype=np.float64)
    plane_to_det = np.empty(3, dtype=np.float64)
    if accumulate_image:
        cache_capacity = _choose_local_pixel_cache_capacity(n_samp)
        cache_keys = np.empty(cache_capacity, dtype=np.int64)
        cache_values = np.empty(cache_capacity, dtype=np.float64)
        _clear_local_pixel_cache(cache_keys, cache_values)
        cache_entry_count = 0
        cache_flush_limit = (
            cache_capacity * _LOCAL_PIXEL_CACHE_LOAD_NUM
        ) // _LOCAL_PIXEL_CACHE_LOAD_DEN
        if cache_flush_limit < 4:
            cache_flush_limit = 4
    else:
        cache_keys = np.empty(1, dtype=np.int64)
        cache_values = np.empty(1, dtype=np.float64)
        cache_entry_count = 0
        cache_flush_limit = 0

    use_exact_optics = optics_mode == OPTICS_MODE_EXACT
    eps3 = 1.0 + 0.0j
    if use_exact_optics:
        fast_optics_ready = np.zeros(1, dtype=np.uint8)
        fast_optics_lut = np.zeros((1, 1, _FAST_OPTICS_LUT_COLS), dtype=np.float64)
    else:
        fast_optics_ready = np.zeros(n_samp, dtype=np.uint8)
        fast_optics_lut = np.empty(
            (n_samp, _FAST_OPTICS_LUT_SIZE, _FAST_OPTICS_LUT_COLS),
            dtype=np.float64,
        )

    nominal_visible, nominal_sample_idx, no_valid_samples = _nominal_reflection_visible(
        H,
        K,
        L,
        G_vec,
        image_size,
        center,
        R_sample,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        sample_terms,
        best_idx,
        sigma_rad,
        gamma_pv,
        optics_mode,
        exit_projection_mode,
        projection_debug_enabled,
        projection_debug_counters,
        projection_debug_reject_count,
        projection_debug_reject_records,
    )
    if no_valid_samples:
        if record_status:
            statuses[:] = -10
            return pixel_hits[:n_hits], statuses, missed_kf[:n_missed], -1
        return (
            pixel_hits[:n_hits],
            np.empty(0, dtype=np.int64),
            missed_kf[:n_missed],
            -1,
        )
    if not nominal_visible:
        if record_status:
            statuses[:] = -11
            return pixel_hits[:n_hits], statuses, missed_kf[:n_missed], nominal_sample_idx
        return (
            pixel_hits[:n_hits],
            np.empty(0, dtype=np.int64),
            missed_kf[:n_missed],
            nominal_sample_idx,
        )

    record_sample_idx = best_idx
    loop_start = 0
    loop_stop = n_samp

    can_reuse_q_solutions = False
    if loop_start == 0 and loop_stop == n_samp and n_samp > 1:
        valid_sample_count = 0
        unique_q_group_count = 0
        for i_samp in range(n_samp):
            if sample_terms[i_samp, _SAMPLE_COL_VALID] <= 0.5:
                continue
            valid_sample_count += 1
            if int(sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP]) == i_samp:
                unique_q_group_count += 1
        can_reuse_q_solutions = (
            unique_q_group_count > 0 and unique_q_group_count < valid_sample_count
        )

    best_candidate_sample_idx = -1
    for i_samp in range(loop_start, loop_stop):
        if sample_terms[i_samp, _SAMPLE_COL_VALID] <= 0.5:
            if record_status:
                statuses[i_samp] = -10
            continue

        group_rep_idx = i_samp
        if can_reuse_q_solutions:
            group_rep_idx = int(sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP])
            if group_rep_idx < 0:
                group_rep_idx = i_samp
            if group_rep_idx != i_samp:
                continue

        solve_k_x_scat = sample_terms[group_rep_idx, _SAMPLE_COL_KX_SCAT]
        solve_k_y_scat = sample_terms[group_rep_idx, _SAMPLE_COL_KY_SCAT]
        solve_re_k_z = sample_terms[group_rep_idx, _SAMPLE_COL_RE_KZ]
        solve_k_scat = sample_terms[group_rep_idx, _SAMPLE_COL_K_SCAT]

        k_in_crystal[0] = solve_k_x_scat
        k_in_crystal[1] = solve_k_y_scat
        k_in_crystal[2] = solve_re_k_z

        All_Q, stat = solve_q(
            k_in_crystal,
            solve_k_scat,
            G_vec,
            sigma_rad,
            gamma_pv,
            eta_pv,
            H,
            K,
            L,
            solve_q_steps,
            DEFAULT_SOLVE_Q_BASE_INTERVALS,
            solve_q_rel_tol,
            solve_q_mode,
            default_solve_q_dtheta,
            default_solve_q_cos,
            default_solve_q_sin,
        )

        selected_sol_idx = np.full(2, -1, dtype=np.int64)
        selected_sol_count = 0
        if not sample_qr_ring_once:
            record_group_sample = (
                capture_aux
                and 0 <= record_sample_idx < n_samp
                and (
                    (not can_reuse_q_solutions and i_samp == record_sample_idx)
                    or (
                        can_reuse_q_solutions
                        and int(sample_terms[record_sample_idx, _SAMPLE_COL_SOLVE_Q_REP])
                        == group_rep_idx
                    )
                )
            )
            if record_group_sample:
                selected_sol_idx, selected_sol_count = _select_g_peak_solution_indices(
                    All_Q,
                    k_in_crystal,
                    solve_k_scat,
                    G_vec,
                )
        chain_idx = i_samp
        while chain_idx >= 0:
            chain_has_valid_q = False
            sample_weight = 1.0
            if sample_weights is not None:
                sample_weight = sample_weights[chain_idx]
                if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                    if record_status:
                        statuses[chain_idx] = -12
                    if not can_reuse_q_solutions:
                        break
                    chain_idx = int(sample_terms[chain_idx, _SAMPLE_COL_SOLVE_Q_NEXT])
                    continue

            sampled_sol_idx = -1
            sampled_ring_mass = 1.0
            if sample_qr_ring_once:
                sampled_sol_idx, sampled_ring_mass = _sample_q_ring_solution(
                    All_Q,
                    i_peaks_index,
                    chain_idx,
                )
                if sampled_sol_idx < 0 or sampled_ring_mass <= 0.0:
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_NO_VALID_SAMPLE] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            _PROJECTION_DEBUG_REASON_NO_VALID_SAMPLE,
                        )
                    if not can_reuse_q_solutions:
                        break
                    chain_idx = int(sample_terms[chain_idx, _SAMPLE_COL_SOLVE_Q_NEXT])
                    continue

            I_plane[0] = sample_terms[chain_idx, _SAMPLE_COL_I_PLANE_X]
            I_plane[1] = sample_terms[chain_idx, _SAMPLE_COL_I_PLANE_Y]
            I_plane[2] = sample_terms[chain_idx, _SAMPLE_COL_I_PLANE_Z]
            k_x_scat = sample_terms[chain_idx, _SAMPLE_COL_KX_SCAT]
            k_y_scat = sample_terms[chain_idx, _SAMPLE_COL_KY_SCAT]
            re_k_z = sample_terms[chain_idx, _SAMPLE_COL_RE_KZ]
            im_k_z = sample_terms[chain_idx, _SAMPLE_COL_IM_KZ]
            k_scat = sample_terms[chain_idx, _SAMPLE_COL_K_SCAT]
            k0 = sample_terms[chain_idx, _SAMPLE_COL_K0]
            Ti2 = sample_terms[chain_idx, _SAMPLE_COL_TI2]
            L_in = sample_terms[chain_idx, _SAMPLE_COL_L_IN]
            n2_real = sample_terms[chain_idx, _SAMPLE_COL_N2_REAL]
            n2_samp = n2_samp_array[chain_idx]
            eps2 = eps2_array[chain_idx]

            if record_status:
                statuses[chain_idx] = stat

            if (not use_exact_optics) and All_Q.shape[0] > 0 and fast_optics_ready[chain_idx] == 0:
                _build_fast_optics_lut_row(
                    fast_optics_lut[chain_idx],
                    k0,
                    n2_samp,
                    n2_real,
                    thickness,
                )
                fast_optics_ready[chain_idx] = 1

            for i_sol in range(All_Q.shape[0]):
                if sample_qr_ring_once and i_sol != sampled_sol_idx:
                    continue
                Qx = All_Q[i_sol, 0]
                Qy = All_Q[i_sol, 1]
                Qz = All_Q[i_sol, 2]
                I_Q = All_Q[i_sol, 3]
                if I_Q < _Q_RING_SAMPLE_MIN_MASS:
                    continue
                chain_has_valid_q = True

                record_this_solution = False
                if sample_qr_ring_once:
                    record_this_solution = capture_aux
                elif capture_aux and chain_idx == record_sample_idx:
                    for keep_idx in range(selected_sol_count):
                        if i_sol == selected_sol_idx[keep_idx]:
                            record_this_solution = True
                            break

                k_tx_prime = Qx + k_x_scat
                k_ty_prime = Qy + k_y_scat
                k_tz_prime = Qz + re_k_z

                kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
                if kr < 1e-12:
                    twotheta_t_prime = 0.0
                else:
                    twotheta_t_prime = np.arctan(k_tz_prime / kr)

                th_t_out = np.abs(twotheta_t_prime)
                if use_exact_optics:
                    k0_sq = k0 * k0
                    k_par_f = kr
                    k_par_f_sq = k_par_f * k_par_f

                    kz2_f = _kz_branch_decay((eps2 * k0_sq) - k_par_f_sq)
                    kz3_f = _kz_branch_decay((k0_sq - k_par_f_sq) + 0.0j)

                    Tf_s = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, True)
                    Tf_p = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, False)
                    Tf2 = 0.5 * (
                        _fresnel_power_t_exact(Tf_s, kz2_f, kz3_f, eps2, eps3, True)
                        + _fresnel_power_t_exact(Tf_p, kz2_f, kz3_f, eps2, eps3, False)
                    )
                    Tf2 = _sanitize_transmission_power(Tf2)

                    im_k_z_f = np.abs(kz2_f.imag)
                    if thickness > 0.0:
                        L_out = thickness
                    else:
                        L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1e-30)
                else:
                    Tf2, im_k_z_f, L_out, twotheta_t_abs = _lookup_fast_optics_lut_row(
                        fast_optics_lut[chain_idx],
                        th_t_out,
                    )

                if use_exact_optics:
                    cos_out = _clamp(kr / np.maximum(k0, 1e-30), -1.0, 1.0)
                    twotheta_t = np.arccos(cos_out) * np.sign(twotheta_t_prime)
                    k_out_mag = k0
                else:
                    twotheta_t = twotheta_t_abs * np.sign(twotheta_t_prime)
                    k_out_mag = k_scat

                if projection_debug_enabled:
                    _projection_debug_log_ray(
                        projection_debug_counters,
                        twotheta_t_prime,
                        twotheta_t,
                        n2_real,
                    )

                prop_att = np.exp(-2.0 * im_k_z * L_in) * np.exp(-2.0 * im_k_z_f * L_out)
                if not np.isfinite(prop_att) or prop_att <= 0.0:
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_PROP_ATT_REJECT] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            np.nan,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_PROP_ATT,
                        )
                    continue
                prop_fac = Ti2 * Tf2 * prop_att
                if not np.isfinite(prop_fac) or prop_fac <= 0.0:
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_PROP_FAC_REJECT] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            np.nan,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_PROP_FAC,
                        )
                    continue

                phi_f = np.arctan2(k_tx_prime, k_ty_prime)
                if exit_projection_mode == EXIT_PROJECTION_INTERNAL:
                    k_norm = sqrt(
                        k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime + k_tz_prime * k_tz_prime
                    )
                    if k_norm > 1.0e-30:
                        kf[0] = k_tx_prime / k_norm
                        kf[1] = k_ty_prime / k_norm
                        kf[2] = k_tz_prime / k_norm
                    elif k_out_mag > 1.0e-30:
                        kf[0] = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
                        kf[1] = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
                        kf[2] = k_out_mag * np.sin(twotheta_t)
                    else:
                        if projection_debug_enabled:
                            projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                            _projection_debug_record_reject(
                                projection_debug_reject_count,
                                projection_debug_reject_records,
                                H,
                                K,
                                L,
                                twotheta_t_prime,
                                twotheta_t,
                                kr,
                                k0,
                                n2_real,
                                _PROJECTION_DEBUG_REASON_PROJECTION_NORM,
                            )
                        continue
                else:
                    kf[0] = k_out_mag * np.cos(twotheta_t) * np.sin(phi_f)
                    kf[1] = k_out_mag * np.cos(twotheta_t) * np.cos(phi_f)
                    kf[2] = k_out_mag * np.sin(twotheta_t)

                kf_prime[0] = (
                    R_sample[0, 0] * kf[0] + R_sample[0, 1] * kf[1] + R_sample[0, 2] * kf[2]
                )
                kf_prime[1] = (
                    R_sample[1, 0] * kf[0] + R_sample[1, 1] * kf[1] + R_sample[1, 2] * kf[2]
                )
                kf_prime[2] = (
                    R_sample[2, 0] * kf[0] + R_sample[2, 1] * kf[1] + R_sample[2, 2] * kf[2]
                )

                dx, dy, dz, valid_det = intersect_line_plane(
                    I_plane, kf_prime, Detector_Pos, n_det_rot
                )
                if not valid_det:
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            twotheta_t,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_INVALID_DET,
                        )
                    if capture_aux and n_missed < max_hits:
                        missed_kf[n_missed, 0] = kf_prime[0]
                        missed_kf[n_missed, 1] = kf_prime[1]
                        missed_kf[n_missed, 2] = kf_prime[2]
                        n_missed += 1
                    continue

                plane_to_det[0] = dx - Detector_Pos[0]
                plane_to_det[1] = dy - Detector_Pos[1]
                plane_to_det[2] = dz - Detector_Pos[2]
                x_det = (
                    plane_to_det[0] * e1_det[0]
                    + plane_to_det[1] * e1_det[1]
                    + plane_to_det[2] * e1_det[2]
                )
                y_det = (
                    plane_to_det[0] * e2_det[0]
                    + plane_to_det[1] * e2_det[1]
                    + plane_to_det[2] * e2_det[2]
                )
                if not np.isfinite(x_det) or not np.isfinite(y_det):
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            twotheta_t,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_INVALID_DET_COORD,
                        )
                    continue

                row_f = center[0] - y_det * pixel_scale
                col_f = center[1] + x_det * pixel_scale
                if not np.isfinite(row_f) or not np.isfinite(col_f):
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            twotheta_t,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_INVALID_DET_COORD,
                        )
                    continue

                val = (
                    reflection_intensity
                    * sample_weight
                    * (sampled_ring_mass if sample_qr_ring_once else I_Q)
                    * prop_fac
                    * exp(-Qz * Qz * debye_x_sq)
                    * exp(-(Qx * Qx + Qy * Qy) * debye_y_sq)
                )
                if not np.isfinite(val) or val <= 0.0:
                    continue

                deposited = True
                if accumulate_image:
                    deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                        image_size,
                        row_f,
                        col_f,
                        val,
                        cache_keys,
                        cache_values,
                        cache_entry_count,
                        cache_flush_limit,
                    )
                    if needs_flush:
                        cache_entry_count = _flush_local_pixel_cache(
                            image,
                            image_size,
                            cache_keys,
                            cache_values,
                        )
                        deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                            image_size,
                            row_f,
                            col_f,
                            val,
                            cache_keys,
                            cache_values,
                            cache_entry_count,
                            cache_flush_limit,
                        )
                        if needs_flush:
                            deposited = _accumulate_bilinear_hit(
                                image,
                                image_size,
                                row_f,
                                col_f,
                                val,
                            )
                if not deposited:
                    if projection_debug_enabled:
                        projection_debug_counters[_PROJECTION_DEBUG_COUNTER_INVALID_DET] += 1
                        _projection_debug_record_reject(
                            projection_debug_reject_count,
                            projection_debug_reject_records,
                            H,
                            K,
                            L,
                            twotheta_t_prime,
                            twotheta_t,
                            kr,
                            k0,
                            n2_real,
                            _PROJECTION_DEBUG_REASON_OFF_DETECTOR,
                        )
                    continue
                if projection_debug_enabled:
                    row_idx = int(np.rint(row_f))
                    if 0 <= row_idx < projection_debug_row_hit_counts.shape[0]:
                        projection_debug_row_hit_counts[row_idx] += 1
                        projection_debug_row_tthp_sums[row_idx] += twotheta_t_prime
                        projection_debug_row_tth_sums[row_idx] += twotheta_t

                if record_this_solution and val > best_candidate_val:
                    best_candidate_val = val
                    if capture_aux:
                        best_candidate[0] = val
                        best_candidate[1] = col_f
                        best_candidate[2] = row_f
                        best_candidate[3] = phi_f
                        best_candidate[4] = H
                        best_candidate[5] = K
                        best_candidate[6] = L
                        best_candidate[7] = np.nan
                        best_candidate[8] = np.nan
                        best_candidate[9] = float(chain_idx)
                    have_candidate = True
                    best_candidate_sample_idx = chain_idx

                if record_this_solution:
                    if n_hits < max_hits:
                        pixel_hits[n_hits, 0] = val
                        pixel_hits[n_hits, 1] = col_f
                        pixel_hits[n_hits, 2] = row_f
                        pixel_hits[n_hits, 3] = phi_f
                        pixel_hits[n_hits, 4] = H
                        pixel_hits[n_hits, 5] = K
                        pixel_hits[n_hits, 6] = L
                        pixel_hits[n_hits, 7] = np.nan
                        pixel_hits[n_hits, 8] = np.nan
                        pixel_hits[n_hits, 9] = float(chain_idx)
                        n_hits += 1
                    recorded_nominal_hit = True

                if save_flag_eff == 1 and q_count[i_peaks_index] < q_data.shape[1]:
                    idx = q_count[i_peaks_index]
                    q_data[i_peaks_index, idx, 0] = Qx
                    q_data[i_peaks_index, idx, 1] = Qy
                    q_data[i_peaks_index, idx, 2] = Qz
                    q_data[i_peaks_index, idx, 3] = val
                    q_count[i_peaks_index] += 1

            if (not chain_has_valid_q) and projection_debug_enabled:
                projection_debug_counters[_PROJECTION_DEBUG_COUNTER_NO_VALID_SAMPLE] += 1
                _projection_debug_record_reject(
                    projection_debug_reject_count,
                    projection_debug_reject_records,
                    H,
                    K,
                    L,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    n2_real,
                    _PROJECTION_DEBUG_REASON_NO_VALID_SAMPLE,
                )
            if not can_reuse_q_solutions:
                break
            chain_idx = int(sample_terms[chain_idx, _SAMPLE_COL_SOLVE_Q_NEXT])

    if accumulate_image and cache_entry_count > 0:
        cache_entry_count = _flush_local_pixel_cache(
            image,
            image_size,
            cache_keys,
            cache_values,
        )

    if capture_aux:
        add_candidate = False
        if have_candidate:
            add_candidate = not recorded_nominal_hit
        if add_candidate and n_hits < max_hits:
            pixel_hits[n_hits, :] = best_candidate
            n_hits += 1

    best_sample_idx = best_idx
    if not recorded_nominal_hit and best_candidate_sample_idx >= 0:
        best_sample_idx = best_candidate_sample_idx

    if record_status:
        return pixel_hits[:n_hits], statuses, missed_kf[:n_missed], best_sample_idx
    return (
        pixel_hits[:n_hits],
        np.empty(0, dtype=np.int64),
        missed_kf[:n_missed],
        best_sample_idx,
    )


# =============================================================================
# 5) CALCULATE_PHI
# =============================================================================


@njit(fastmath=True, cache=True)
def calculate_phi(
    H,
    K,
    L,
    av,
    cv,
    wavelength_array,
    image,
    image_size,
    gamma_rad,
    Gamma_rad,
    chi_rad,
    psi_rad,
    zs,
    zb,
    n2,
    n2_array,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    reflection_intensity,
    sigma_rad,
    gamma_pv,
    eta_pv,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    R_x_detector,
    R_z_detector,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    R_z_R_y,
    R_ZY_n,
    P0,
    unit_x,
    save_flag,
    q_data,
    q_count,
    i_peaks_index,
    record_status=False,
    thickness=0.0,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    sample_qr_ring_once=True,
):
    """
    For a single reflection (H,K,L), build a mosaic Q_grid around G_vec.
    Then for each sample beam (beam_x_array, etc.), compute the intersection
    with the sample plane, wavevectors in the crystal, do "incoherent_averaging"
    + "solve_q", and deposit intensities into the final 2D 'image'.

    Physical meaning:
      - This function is the "core" for each reflection, merging mosaic
        sampling (Q_grid) with geometry (plane intersection, wavevectors)
        and the final detection step.

    Returns
    -------
    pixel_hits : ndarray
        Record of peak intensities and pixel positions for the best beam sample.
    statuses : ndarray
        ``solve_q`` status codes when ``record_status`` is True.
    missed_kf : ndarray
        Outgoing wavevectors that failed to intersect the detector plane.
    """
    R_sample, sample_terms, n2_samp_array, eps2_array, best_idx = _precompute_sample_terms(
        wavelength_array,
        n2,
        n2_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
        0.0,
        R_z_R_y,
        R_ZY_n,
        P0,
    )
    projection_debug_counters = np.zeros(_PROJECTION_DEBUG_COUNTER_COLS, dtype=np.int64)
    projection_debug_reject_count = np.zeros(1, dtype=np.int64)
    projection_debug_reject_records = np.full(
        (1, _PROJECTION_DEBUG_REJECT_RECORD_COLS),
        np.nan,
        dtype=np.float64,
    )
    projection_debug_row_hit_counts = np.zeros(image_size, dtype=np.int64)
    projection_debug_row_tthp_sums = np.zeros(image_size, dtype=np.float64)
    projection_debug_row_tth_sums = np.zeros(image_size, dtype=np.float64)
    return _calculate_phi_from_precomputed(
        H,
        K,
        L,
        av,
        cv,
        image,
        image_size,
        reflection_intensity,
        sigma_rad,
        gamma_pv,
        eta_pv,
        debye_x,
        debye_y,
        center,
        R_sample,
        n_det_rot,
        Detector_Pos,
        e1_det,
        e2_det,
        sample_terms,
        n2_samp_array,
        eps2_array,
        best_idx,
        save_flag,
        q_data,
        q_count,
        i_peaks_index,
        record_status,
        thickness,
        optics_mode,
        solve_q_steps,
        solve_q_rel_tol,
        solve_q_mode,
        pixel_size_m,
        sample_qr_ring_once,
        None,
        EXIT_PROJECTION_INTERNAL,
        False,
        projection_debug_counters,
        projection_debug_reject_count,
        projection_debug_reject_records,
        projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums,
    )


# =============================================================================
# 6) PROCESS_PEAKS_PARALLEL
# =============================================================================


def _weighted_event_fast_path_available(
    *,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
):
    if _project_weighted_candidate is not _DEFAULT_PROJECT_WEIGHTED_CANDIDATE:
        return False
    debug_args = (
        projection_debug_counters,
        projection_debug_reject_counts,
        projection_debug_reject_records,
        projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums,
    )
    return all(arg is None for arg in debug_args)


def _build_weighted_event_representative_slot_map(miller, av):
    miller_arr = np.asarray(miller, dtype=np.float64)
    if miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
        return np.empty((0, 2), dtype=np.int64), []

    num_peaks = int(miller_arr.shape[0])
    representative_slot_by_peak_branch = np.full((num_peaks, 2), -1, dtype=np.int64)
    representative_slot_keys: list[tuple] = []
    slot_index_by_key: dict[tuple, int] = {}
    qr_scale = np.nan
    av_val = float(av)
    if np.isfinite(av_val) and abs(av_val) > 1.0e-12:
        qr_scale = 4.0 * np.pi / av_val

    for peak_idx in range(num_peaks):
        H = float(miller_arr[peak_idx, 0])
        K = float(miller_arr[peak_idx, 1])
        L = float(miller_arr[peak_idx, 2])
        l_key = int(round(L))
        if abs(H) < 1.0e-12 and abs(K) < 1.0e-12:
            key = ("specular", l_key, 0)
            slot_idx = slot_index_by_key.get(key)
            if slot_idx is None:
                slot_idx = len(representative_slot_keys)
                representative_slot_keys.append(key)
                slot_index_by_key[key] = slot_idx
            representative_slot_by_peak_branch[peak_idx, 0] = int(slot_idx)
            representative_slot_by_peak_branch[peak_idx, 1] = int(slot_idx)
            continue

        qr = qr_scale * sqrt(max((H * H + H * K + K * K) / 3.0, 0.0))
        if not np.isfinite(qr):
            qr = 0.0
        qr_key = round(float(qr), 8)
        for branch_id in range(2):
            key = ("non_specular", qr_key, l_key, int(branch_id))
            slot_idx = slot_index_by_key.get(key)
            if slot_idx is None:
                slot_idx = len(representative_slot_keys)
                representative_slot_keys.append(key)
                slot_index_by_key[key] = slot_idx
            representative_slot_by_peak_branch[peak_idx, branch_id] = int(slot_idx)

    return representative_slot_by_peak_branch, representative_slot_keys


def _build_weighted_event_hit_tables(
    flat_event_rows,
    flat_event_peak_indices,
    flat_event_count,
    num_peaks,
):
    if int(num_peaks) <= 0:
        return []
    counts = np.zeros(int(num_peaks), dtype=np.int64)
    for row_idx in range(int(flat_event_count)):
        peak_idx = int(flat_event_peak_indices[row_idx])
        if 0 <= peak_idx < counts.shape[0]:
            counts[peak_idx] += 1
    hit_tables = [
        np.empty((int(counts[peak_idx]), HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        if counts[peak_idx] > 0
        else np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        for peak_idx in range(int(num_peaks))
    ]
    offsets = np.zeros(int(num_peaks), dtype=np.int64)
    for row_idx in range(int(flat_event_count)):
        peak_idx = int(flat_event_peak_indices[row_idx])
        if peak_idx < 0 or peak_idx >= int(num_peaks):
            continue
        out_idx = int(offsets[peak_idx])
        hit_tables[peak_idx][out_idx, :] = flat_event_rows[row_idx, :]
        offsets[peak_idx] += 1
    return hit_tables


def _build_weighted_event_representative_hit_tables(
    representative_valid,
    representative_rows,
    representative_slot_keys,
):
    tables = []
    for slot_idx, _slot_key in enumerate(representative_slot_keys):
        if slot_idx >= int(representative_valid.shape[0]):
            break
        if int(representative_valid[slot_idx]) <= 0:
            continue
        tables.append(
            np.asarray(
                representative_rows[slot_idx : slot_idx + 1, :],
                dtype=np.float64,
            ).copy()
        )
    return tables


def _process_peaks_parallel_weighted_events_fast(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=50e-9,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    sample_qr_ring_once=True,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
    numba_thread_count=None,
    events_per_beam_phase=50,
):
    del lambda_
    del sample_qr_ring_once
    del numba_thread_count

    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    image = np.asarray(image, dtype=np.float64)
    beam_x_array = np.asarray(beam_x_array, dtype=np.float64).reshape(-1)
    beam_y_array = np.asarray(beam_y_array, dtype=np.float64).reshape(-1)
    theta_array = np.asarray(theta_array, dtype=np.float64).reshape(-1)
    phi_array = np.asarray(phi_array, dtype=np.float64).reshape(-1)
    wavelength_array = np.asarray(wavelength_array, dtype=np.float64).reshape(-1)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    unit_x = np.asarray(unit_x, dtype=np.float64).reshape(-1)
    n_detector = np.asarray(n_detector, dtype=np.float64).reshape(-1)
    events_per_beam_phase = normalize_events_per_beam_phase_backend(events_per_beam_phase)
    _set_last_process_peaks_weighted_event_stats()

    gamma_rad = float(gamma_deg) * (pi / 180.0)
    Gamma_rad = float(Gamma_deg) * (pi / 180.0)
    chi_rad = float(chi_deg) * (pi / 180.0)
    psi_rad = float(psi_deg) * (pi / 180.0)
    sigma_rad = float(sigma_pv_deg) * (pi / 180.0)
    gamma_rad_m = float(gamma_pv_deg) * (pi / 180.0)
    solve_q_steps_i = int(np.clip(int(solve_q_steps), MIN_SOLVE_Q_STEPS, MAX_SOLVE_Q_STEPS))
    solve_q_rel_tol_i = float(
        np.clip(float(solve_q_rel_tol), MIN_SOLVE_Q_REL_TOL, MAX_SOLVE_Q_REL_TOL)
    )
    solve_q_mode_i = (
        SOLVE_Q_MODE_UNIFORM
        if int(solve_q_mode) == SOLVE_Q_MODE_UNIFORM
        else SOLVE_Q_MODE_ADAPTIVE
    )

    cg = cos(gamma_rad)
    sg = sin(gamma_rad)
    cG = cos(Gamma_rad)
    sG = sin(Gamma_rad)
    R_x_det = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]], dtype=np.float64)
    R_z_det = np.array([[cG, sG, 0.0], [-sG, cG, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len = sqrt(float(np.dot(n_det_rot, n_det_rot)))
    if nd_len > 1.0e-30:
        n_det_rot = n_det_rot / nd_len

    Detector_Pos = np.array([0.0, float(Distance_CoR_to_Detector), 0.0], dtype=np.float64)

    dot_e1 = float(np.dot(unit_x, n_det_rot))
    e1_det = unit_x - dot_e1 * n_det_rot
    e1_len = sqrt(float(np.dot(e1_det, e1_det)))
    if e1_len < 1.0e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det = e1_det / e1_len

    e2_det = -np.cross(n_det_rot, e1_det)
    e2_len = sqrt(float(np.dot(e2_det, e2_det)))
    if e2_len < 1.0e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det = e2_det / e2_len

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    R_y = np.array([[c_chi, 0.0, s_chi], [0.0, 1.0, 0.0], [-s_chi, 0.0, c_chi]], dtype=np.float64)
    c_psi = cos(psi_rad)
    s_psi = sin(psi_rad)
    R_z = np.array([[c_psi, s_psi, 0.0], [-s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_z_R_y = R_z @ R_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n = R_z_R_y @ n1
    nzy_len = sqrt(float(np.dot(R_ZY_n, R_ZY_n)))
    if nzy_len > 1.0e-30:
        R_ZY_n = R_ZY_n / nzy_len
    P0 = np.array([0.0, 0.0, -float(zs)], dtype=np.float64)

    num_peaks = int(miller.shape[0])
    n_samp = int(beam_x_array.size)
    collect_tables = bool(collect_hit_tables)
    accumulate_image_flag = bool(accumulate_image)
    all_status = np.zeros((num_peaks, n_samp), dtype=np.int64)
    event_counts = np.zeros((num_peaks, n_samp), dtype=np.int64)

    if save_flag == 1:
        max_solutions = 2000000
        q_data = np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count = np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data = np.zeros((1, 1, 5), dtype=np.float64)
        q_count = np.zeros(1, dtype=np.int64)

    if collect_tables:
        flat_capacity = max(n_samp * int(events_per_beam_phase), 0)
        flat_event_rows = np.empty(
            (flat_capacity, HIT_ROW_WITH_PROVENANCE_WIDTH),
            dtype=np.float64,
        )
        flat_event_peak_indices = np.empty(flat_capacity, dtype=np.int64)
        miss_tables = [np.empty((0, 3), dtype=np.float64) for _ in range(num_peaks)]
    else:
        flat_event_rows = np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        flat_event_peak_indices = np.empty(0, dtype=np.int64)
        miss_tables = []
    flat_event_count = 0

    representative_slot_by_peak_branch, representative_slot_keys = (
        _build_weighted_event_representative_slot_map(miller, av)
    )
    n_rep_slots = int(len(representative_slot_keys))
    representative_valid = np.zeros(n_rep_slots, dtype=np.uint8)
    representative_neg_sample_weight = np.full(n_rep_slots, np.inf, dtype=np.float64)
    representative_top_distance = np.full(n_rep_slots, np.inf, dtype=np.float64)
    representative_neg_mass = np.full(n_rep_slots, np.inf, dtype=np.float64)
    representative_sample_idx = np.full(n_rep_slots, -1, dtype=np.int64)
    representative_peak_idx = np.full(n_rep_slots, -1, dtype=np.int64)
    representative_q_idx = np.full(n_rep_slots, -1, dtype=np.int64)
    representative_rows = np.empty((n_rep_slots, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
    representative_rows.fill(np.nan)

    sample_weight_array = sample_weights
    if sample_weight_array is not None:
        sample_weight_array = np.asarray(sample_weight_array, dtype=np.float64).reshape(-1)
        if sample_weight_array.shape[0] != n_samp:
            sample_weight_array = None

    n2_sample_array = np.empty(n_samp, dtype=np.complex128)
    n2_override_array = np.empty(0, dtype=np.complex128)
    has_n2_override = n2_sample_array_override is not None
    if has_n2_override:
        n2_override_array = np.asarray(n2_sample_array_override, dtype=np.complex128).reshape(-1)
    if has_n2_override and n2_override_array.size == n_samp:
        n2_sample_array[:] = n2_override_array
    elif wavelength_array.size == n_samp:
        for i_samp in range(n_samp):
            lam_angstrom = wavelength_array[i_samp]
            if np.isfinite(lam_angstrom) and lam_angstrom > 0.0:
                n2_sample_array[i_samp] = IndexofRefraction(lam_angstrom * 1.0e-10)
            else:
                n2_sample_array[i_samp] = n2
    else:
        for i_samp in range(n_samp):
            n2_sample_array[i_samp] = n2

    time_precompute = 0.0
    time_solve_q = 0.0
    time_project = 0.0
    time_select = 0.0
    time_emit_cache = 0.0
    n_solve_q_calls = 0
    n_project_candidate_calls = 0
    n_valid_candidates = 0
    n_selected_events = 0
    pass1_total_mass = 0.0
    pass2_total_mass = 0.0
    pass2_mass_mismatch_count = 0
    pass2_mass_mismatch_max_abs = 0.0
    tail_fill_events = 0

    precompute_start = time.perf_counter()
    (
        R_sample_precomputed,
        sample_terms,
        sample_n2_array,
        sample_eps2_array,
        _best_idx_precomputed,
    ) = _precompute_sample_terms(
        wavelength_array,
        n2,
        n2_sample_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    )
    finite_wavelengths = wavelength_array[np.isfinite(wavelength_array)]
    mean_wavelength = float(np.mean(finite_wavelengths)) if finite_wavelengths.size > 0 else np.nan
    sample_top_distances = np.zeros(n_samp, dtype=np.float64)
    if beam_x_array.size == n_samp:
        sample_top_distances += np.where(np.isfinite(beam_x_array), beam_x_array * beam_x_array, 0.0)
    if beam_y_array.size == n_samp:
        sample_top_distances += np.where(np.isfinite(beam_y_array), beam_y_array * beam_y_array, 0.0)
    if theta_array.size == n_samp:
        sample_top_distances += np.where(np.isfinite(theta_array), theta_array * theta_array, 0.0)
    if phi_array.size == n_samp:
        sample_top_distances += np.where(np.isfinite(phi_array), phi_array * phi_array, 0.0)
    if np.isfinite(mean_wavelength) and wavelength_array.size == n_samp:
        sample_top_distances += np.where(
            np.isfinite(wavelength_array),
            np.abs(wavelength_array - mean_wavelength),
            0.0,
        )
    use_exact_optics = int(optics_mode) == OPTICS_MODE_EXACT
    fast_optics_ready = np.zeros(n_samp, dtype=np.uint8)
    fast_optics_lut = np.zeros(
        (max(n_samp, 1), _FAST_OPTICS_LUT_SIZE, _FAST_OPTICS_LUT_COLS),
        dtype=np.float64,
    )
    if not use_exact_optics:
        for i_samp in range(n_samp):
            if sample_terms[i_samp, _SAMPLE_COL_VALID] <= 0.5:
                continue
            _build_fast_optics_lut_row(
                fast_optics_lut[i_samp],
                sample_terms[i_samp, _SAMPLE_COL_K0],
                sample_n2_array[i_samp],
                sample_terms[i_samp, _SAMPLE_COL_N2_REAL],
                thickness,
            )
            fast_optics_ready[i_samp] = 1
    time_precompute += time.perf_counter() - precompute_start

    debye_x_sq = float(debye_x) * float(debye_x)
    debye_y_sq = float(debye_y) * float(debye_y)
    pixel_size_eff = float(pixel_size_m)
    if (not np.isfinite(pixel_size_eff)) or pixel_size_eff <= 0.0:
        pixel_size_eff = 100e-6
    pixel_scale = 1.0 / pixel_size_eff
    center_row = float(center[0]) if center.size > 0 else 0.0
    center_col = float(center[1]) if center.size > 1 else center_row

    peak_h = miller[:, 0].astype(np.float64, copy=False) if num_peaks > 0 else np.empty(0, dtype=np.float64)
    peak_k = miller[:, 1].astype(np.float64, copy=False) if num_peaks > 0 else np.empty(0, dtype=np.float64)
    peak_l = miller[:, 2].astype(np.float64, copy=False) if num_peaks > 0 else np.empty(0, dtype=np.float64)
    peak_reflection_intensity = np.zeros(num_peaks, dtype=np.float64)
    peak_gr0 = np.zeros(num_peaks, dtype=np.float64)
    peak_gz0 = np.zeros(num_peaks, dtype=np.float64)
    peak_valid = np.zeros(num_peaks, dtype=np.uint8)
    for peak_idx in range(num_peaks):
        reflection_intensity = float(intensities[peak_idx]) if peak_idx < intensities.shape[0] else 0.0
        peak_reflection_intensity[peak_idx] = reflection_intensity
        H = float(peak_h[peak_idx])
        K = float(peak_k[peak_idx])
        L = float(peak_l[peak_idx])
        if np.isfinite(reflection_intensity) and reflection_intensity > 0.0 and L >= 0.0:
            peak_valid[peak_idx] = 1
            peak_gz0[peak_idx] = 2.0 * pi * (L / float(cv))
            peak_gr0[peak_idx] = 4.0 * pi / float(av) * sqrt(
                max((H * H + H * K + K * K) / 3.0, 0.0)
            )

    if accumulate_image_flag:
        cache_capacity = int(_choose_local_pixel_cache_capacity(n_samp))
        cache_keys = np.empty(cache_capacity, dtype=np.int64)
        cache_values = np.empty(cache_capacity, dtype=np.float64)
        cache_flush_limit = (cache_capacity * _LOCAL_PIXEL_CACHE_LOAD_NUM) // _LOCAL_PIXEL_CACHE_LOAD_DEN
        if cache_flush_limit < 4:
            cache_flush_limit = 4
    else:
        cache_keys = np.empty(1, dtype=np.int64)
        cache_values = np.empty(1, dtype=np.float64)
        cache_flush_limit = 0

    # Current-call cache is safe with (peak_idx, rep_idx) because all solve_q
    # settings and geometry inputs are fixed for this function invocation.
    all_q_cache = {}

    for sample_idx in range(n_samp):
        if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
            all_status[:, sample_idx] = -10
            continue

        sample_weight = 1.0
        if sample_weight_array is not None:
            sample_weight = float(sample_weight_array[sample_idx])
            if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                all_status[:, sample_idx] = -12
                continue

        sample_total_mass = 0.0
        if accumulate_image_flag:
            _clear_local_pixel_cache(cache_keys, cache_values)
        cache_entry_count = 0

        for peak_idx in range(num_peaks):
            if int(peak_valid[peak_idx]) == 0:
                continue

            H = float(peak_h[peak_idx])
            K = float(peak_k[peak_idx])
            L = float(peak_l[peak_idx])
            reflection_intensity = float(peak_reflection_intensity[peak_idx])
            rep_idx = int(sample_terms[sample_idx, _SAMPLE_COL_SOLVE_Q_REP])
            if rep_idx < 0:
                rep_idx = sample_idx
            cache_key = (int(peak_idx), int(rep_idx))
            cached = all_q_cache.get(cache_key)
            if cached is None:
                k_in_crystal = np.array(
                    [
                        sample_terms[rep_idx, _SAMPLE_COL_KX_SCAT],
                        sample_terms[rep_idx, _SAMPLE_COL_KY_SCAT],
                        sample_terms[rep_idx, _SAMPLE_COL_RE_KZ],
                    ],
                    dtype=np.float64,
                )
                G_vec = np.array([0.0, peak_gr0[peak_idx], peak_gz0[peak_idx]], dtype=np.float64)
                solve_start = time.perf_counter()
                all_q_raw, stat = solve_q(
                    k_in_crystal,
                    float(sample_terms[rep_idx, _SAMPLE_COL_K_SCAT]),
                    G_vec,
                    sigma_rad,
                    gamma_rad_m,
                    eta_pv,
                    H,
                    K,
                    L,
                    solve_q_steps_i,
                    DEFAULT_SOLVE_Q_BASE_INTERVALS,
                    solve_q_rel_tol_i,
                    solve_q_mode_i,
                    default_solve_q_dtheta,
                    default_solve_q_cos,
                    default_solve_q_sin,
                )
                time_solve_q += time.perf_counter() - solve_start
                all_q = np.asarray(all_q_raw, dtype=np.float64).reshape(-1, 4)
                cached = (all_q, int(stat))
                all_q_cache[cache_key] = cached
                n_solve_q_calls += 1
            else:
                all_q, stat = cached
            all_status[peak_idx, sample_idx] = int(stat)

            project_start = time.perf_counter()
            peak_mass, valid_count, project_calls = _weighted_event_pass1_for_qset(
                all_q,
                peak_idx,
                sample_idx,
                H,
                K,
                L,
                reflection_intensity,
                sample_weight,
                float(sample_top_distances[sample_idx]),
                debye_x_sq,
                debye_y_sq,
                center_row,
                center_col,
                R_sample_precomputed,
                n_det_rot,
                Detector_Pos,
                e1_det,
                e2_det,
                sample_terms,
                sample_n2_array,
                sample_eps2_array,
                thickness,
                optics_mode,
                pixel_scale,
                int(image_size),
                exit_projection_mode,
                int(fast_optics_ready[sample_idx]),
                fast_optics_lut[sample_idx],
                int(save_flag),
                q_data,
                q_count,
                representative_slot_by_peak_branch,
                representative_valid,
                representative_neg_sample_weight,
                representative_top_distance,
                representative_neg_mass,
                representative_sample_idx,
                representative_peak_idx,
                representative_q_idx,
                representative_rows,
            )
            time_project += time.perf_counter() - project_start
            sample_total_mass += float(peak_mass)
            pass1_total_mass += float(peak_mass)
            n_valid_candidates += int(valid_count)
            n_project_candidate_calls += int(project_calls)

        if not np.isfinite(sample_total_mass) or sample_total_mass <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                flush_start = time.perf_counter()
                _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
                time_emit_cache += time.perf_counter() - flush_start
            continue

        select_start = time.perf_counter()
        targets = _weighted_event_targets(sample_total_mass, events_per_beam_phase, sample_idx)
        deposit = _weighted_event_deposit(sample_total_mass, events_per_beam_phase)
        time_select += time.perf_counter() - select_start
        if targets.size <= 0 or not np.isfinite(deposit) or deposit <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                flush_start = time.perf_counter()
                _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
                time_emit_cache += time.perf_counter() - flush_start
            continue

        target_idx = 0
        cumulative_mass = 0.0
        phase_have_last_valid = False
        phase_last_row_f = np.nan
        phase_last_col_f = np.nan
        phase_last_phi_f = np.nan
        phase_last_peak_idx = -1
        phase_last_H = np.nan
        phase_last_K = np.nan
        phase_last_L = np.nan
        for peak_idx in range(num_peaks):
            if int(peak_valid[peak_idx]) == 0:
                continue

            H = float(peak_h[peak_idx])
            K = float(peak_k[peak_idx])
            L = float(peak_l[peak_idx])
            reflection_intensity = float(peak_reflection_intensity[peak_idx])
            rep_idx = int(sample_terms[sample_idx, _SAMPLE_COL_SOLVE_Q_REP])
            if rep_idx < 0:
                rep_idx = sample_idx
            all_q, _stat = all_q_cache[(int(peak_idx), int(rep_idx))]

            project_start = time.perf_counter()
            (
                target_idx,
                cumulative_mass,
                flat_event_count,
                cache_entry_count,
                project_calls,
                peak_pass2_mass,
                selected_events,
                qset_have_last_valid,
                qset_last_row_f,
                qset_last_col_f,
                qset_last_phi_f,
                qset_last_peak_idx,
                qset_last_H,
                qset_last_K,
                qset_last_L,
            ) = _weighted_event_pass2_for_qset(
                all_q,
                peak_idx,
                sample_idx,
                H,
                K,
                L,
                reflection_intensity,
                sample_weight,
                debye_x_sq,
                debye_y_sq,
                center_row,
                center_col,
                R_sample_precomputed,
                n_det_rot,
                Detector_Pos,
                e1_det,
                e2_det,
                sample_terms,
                sample_n2_array,
                sample_eps2_array,
                thickness,
                optics_mode,
                pixel_scale,
                int(image_size),
                exit_projection_mode,
                int(fast_optics_ready[sample_idx]),
                fast_optics_lut[sample_idx],
                targets,
                int(target_idx),
                float(cumulative_mass),
                float(deposit),
                collect_tables,
                flat_event_rows,
                flat_event_peak_indices,
                int(flat_event_count),
                event_counts,
                accumulate_image_flag,
                image,
                cache_keys,
                cache_values,
                int(cache_entry_count),
                int(cache_flush_limit),
            )
            time_project += time.perf_counter() - project_start
            pass2_total_mass += float(peak_pass2_mass)
            n_project_candidate_calls += int(project_calls)
            n_selected_events += int(selected_events)
            if bool(qset_have_last_valid):
                phase_have_last_valid = True
                phase_last_row_f = float(qset_last_row_f)
                phase_last_col_f = float(qset_last_col_f)
                phase_last_phi_f = float(qset_last_phi_f)
                phase_last_peak_idx = int(qset_last_peak_idx)
                phase_last_H = float(qset_last_H)
                phase_last_K = float(qset_last_K)
                phase_last_L = float(qset_last_L)

        sample_pass2_mass_delta = abs(float(cumulative_mass) - float(sample_total_mass))
        sample_pass2_mass_tol = max(
            1.0e-10,
            1.0e-9 * max(1.0, abs(float(sample_total_mass))),
        )
        if (not np.isfinite(sample_pass2_mass_delta)) or sample_pass2_mass_delta > sample_pass2_mass_tol:
            pass2_mass_mismatch_count += 1
            if np.isfinite(sample_pass2_mass_delta):
                pass2_mass_mismatch_max_abs = max(
                    float(pass2_mass_mismatch_max_abs),
                    float(sample_pass2_mass_delta),
                )
            else:
                pass2_mass_mismatch_max_abs = float("inf")
        elif target_idx < targets.shape[0] and phase_have_last_valid:
            hit_count = int(targets.shape[0] - target_idx)
            n_selected_events += int(hit_count)
            tail_fill_events += int(hit_count)
            if 0 <= phase_last_peak_idx < event_counts.shape[0]:
                event_counts[phase_last_peak_idx, sample_idx] += int(hit_count)
            target_idx = targets.shape[0]

            deposited = True
            if accumulate_image_flag:
                deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                    int(image_size),
                    float(phase_last_row_f),
                    float(phase_last_col_f),
                    float(hit_count) * float(deposit),
                    cache_keys,
                    cache_values,
                    int(cache_entry_count),
                    int(cache_flush_limit),
                )
                if needs_flush:
                    cache_entry_count = _flush_local_pixel_cache(
                        image,
                        int(image_size),
                        cache_keys,
                        cache_values,
                    )
                    deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                        int(image_size),
                        float(phase_last_row_f),
                        float(phase_last_col_f),
                        float(hit_count) * float(deposit),
                        cache_keys,
                        cache_values,
                        int(cache_entry_count),
                        int(cache_flush_limit),
                    )
                    if needs_flush:
                        deposited = _accumulate_bilinear_hit(
                            image,
                            int(image_size),
                            float(phase_last_row_f),
                            float(phase_last_col_f),
                            float(hit_count) * float(deposit),
                        )
                        cache_entry_count = 0

            if collect_tables and deposited:
                for _event_idx in range(hit_count):
                    flat_event_peak_indices[flat_event_count] = phase_last_peak_idx
                    flat_event_rows[flat_event_count, 0] = float(deposit)
                    flat_event_rows[flat_event_count, 1] = float(phase_last_col_f)
                    flat_event_rows[flat_event_count, 2] = float(phase_last_row_f)
                    flat_event_rows[flat_event_count, 3] = float(phase_last_phi_f)
                    flat_event_rows[flat_event_count, 4] = float(phase_last_H)
                    flat_event_rows[flat_event_count, 5] = float(phase_last_K)
                    flat_event_rows[flat_event_count, 6] = float(phase_last_L)
                    flat_event_rows[flat_event_count, 7] = np.nan
                    flat_event_rows[flat_event_count, 8] = np.nan
                    flat_event_rows[flat_event_count, 9] = float(sample_idx)
                    flat_event_count += 1

        if accumulate_image_flag and cache_entry_count > 0:
            flush_start = time.perf_counter()
            _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
            time_emit_cache += time.perf_counter() - flush_start

    emit_start = time.perf_counter()
    if best_sample_indices_out is not None:
        best_sample_indices_out[:] = -1
        for peak_idx in range(num_peaks):
            best_count = -1
            best_sample = -1
            for sample_idx in range(n_samp):
                count = int(event_counts[peak_idx, sample_idx])
                if count > best_count:
                    best_count = count
                    best_sample = sample_idx
            if best_count > 0 and peak_idx < best_sample_indices_out.shape[0]:
                best_sample_indices_out[peak_idx] = int(best_sample)
            elif peak_idx < best_sample_indices_out.shape[0]:
                best_sample_indices_out[peak_idx] = -1

    if collect_tables:
        hit_tables = _build_weighted_event_hit_tables(
            flat_event_rows,
            flat_event_peak_indices,
            flat_event_count,
            num_peaks,
        )
    else:
        hit_tables = []
    representative_hit_tables = _build_weighted_event_representative_hit_tables(
        representative_valid,
        representative_rows,
        representative_slot_keys,
    )
    time_emit_cache += time.perf_counter() - emit_start

    _set_last_process_peaks_weighted_event_stats(
        n_solve_q_calls=int(n_solve_q_calls),
        n_project_candidate_calls=int(n_project_candidate_calls),
        n_valid_candidates=int(n_valid_candidates),
        n_selected_events=int(n_selected_events),
        pass2_mass_mismatch_count=int(pass2_mass_mismatch_count),
        pass2_mass_mismatch_max_abs=float(pass2_mass_mismatch_max_abs),
        tail_fill_events=int(tail_fill_events),
        time_precompute=float(time_precompute),
        time_solve_q=float(time_solve_q),
        time_project=float(time_project),
        time_select=float(time_select),
        time_emit_cache=float(time_emit_cache),
        pass1_total_mass=float(pass1_total_mass),
        pass2_total_mass=float(pass2_total_mass),
    )
    return (
        image,
        hit_tables,
        q_data,
        q_count,
        all_status,
        miss_tables,
        representative_hit_tables,
    )


def _process_peaks_parallel_impl(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=50e-9,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    sample_qr_ring_once=True,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
    numba_thread_count=1,
    events_per_beam_phase=50,
):
    """Run weighted-event sampling through compiled helpers when hooks are default."""

    events_per_beam_phase = normalize_events_per_beam_phase_backend(events_per_beam_phase)
    if not _weighted_event_fast_path_available(
        projection_debug_counters=projection_debug_counters,
        projection_debug_reject_counts=projection_debug_reject_counts,
        projection_debug_reject_records=projection_debug_reject_records,
        projection_debug_row_hit_counts=projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums=projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums=projection_debug_row_tth_sums,
    ):
        _set_last_process_peaks_weighted_event_stats()
        result = _process_peaks_parallel_weighted_events_python(
            miller,
            intensities,
            image_size,
            av,
            cv,
            lambda_,
            image,
            Distance_CoR_to_Detector,
            gamma_deg,
            Gamma_deg,
            chi_deg,
            psi_deg,
            psi_z_deg,
            zs,
            zb,
            n2,
            beam_x_array,
            beam_y_array,
            theta_array,
            phi_array,
            sigma_pv_deg,
            gamma_pv_deg,
            eta_pv,
            wavelength_array,
            debye_x,
            debye_y,
            center,
            theta_initial_deg,
            cor_angle_deg,
            unit_x,
            n_detector,
            save_flag,
            record_status=record_status,
            thickness=thickness,
            optics_mode=optics_mode,
            solve_q_steps=solve_q_steps,
            solve_q_rel_tol=solve_q_rel_tol,
            solve_q_mode=solve_q_mode,
            sample_weights=sample_weights,
            best_sample_indices_out=best_sample_indices_out,
            collect_hit_tables=collect_hit_tables,
            pixel_size_m=pixel_size_m,
            sample_width_m=sample_width_m,
            sample_length_m=sample_length_m,
            n2_sample_array_override=n2_sample_array_override,
            accumulate_image=accumulate_image,
            sample_qr_ring_once=sample_qr_ring_once,
            exit_projection_mode=exit_projection_mode,
            projection_debug_counters=projection_debug_counters,
            projection_debug_reject_counts=projection_debug_reject_counts,
            projection_debug_reject_records=projection_debug_reject_records,
            projection_debug_row_hit_counts=projection_debug_row_hit_counts,
            projection_debug_row_tthp_sums=projection_debug_row_tthp_sums,
            projection_debug_row_tth_sums=projection_debug_row_tth_sums,
            default_solve_q_dtheta=default_solve_q_dtheta,
            default_solve_q_cos=default_solve_q_cos,
            default_solve_q_sin=default_solve_q_sin,
            numba_thread_count=numba_thread_count,
            events_per_beam_phase=events_per_beam_phase,
        )
        _set_last_process_peaks_representative_hit_tables(result[6])
        return result[:6]
    result = _process_peaks_parallel_weighted_events_fast(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        record_status=record_status,
        thickness=thickness,
        optics_mode=optics_mode,
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
        sample_weights=sample_weights,
        best_sample_indices_out=best_sample_indices_out,
        collect_hit_tables=collect_hit_tables,
        pixel_size_m=pixel_size_m,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
        n2_sample_array_override=n2_sample_array_override,
        accumulate_image=accumulate_image,
        sample_qr_ring_once=sample_qr_ring_once,
        exit_projection_mode=exit_projection_mode,
        projection_debug_counters=projection_debug_counters,
        projection_debug_reject_counts=projection_debug_reject_counts,
        projection_debug_reject_records=projection_debug_reject_records,
        projection_debug_row_hit_counts=projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums=projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums=projection_debug_row_tth_sums,
        default_solve_q_dtheta=default_solve_q_dtheta,
        default_solve_q_cos=default_solve_q_cos,
        default_solve_q_sin=default_solve_q_sin,
        numba_thread_count=numba_thread_count,
        events_per_beam_phase=events_per_beam_phase,
    )
    _set_last_process_peaks_representative_hit_tables(result[6])
    return result[:6]


def _process_peaks_parallel_weighted_events_python(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=50e-9,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    sample_qr_ring_once=True,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
    numba_thread_count=None,
    events_per_beam_phase=50,
):
    del sample_qr_ring_once
    del numba_thread_count

    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    image = np.asarray(image, dtype=np.float64)
    beam_x_array = np.asarray(beam_x_array, dtype=np.float64).reshape(-1)
    beam_y_array = np.asarray(beam_y_array, dtype=np.float64).reshape(-1)
    theta_array = np.asarray(theta_array, dtype=np.float64).reshape(-1)
    phi_array = np.asarray(phi_array, dtype=np.float64).reshape(-1)
    wavelength_array = np.asarray(wavelength_array, dtype=np.float64).reshape(-1)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    unit_x = np.asarray(unit_x, dtype=np.float64).reshape(-1)
    n_detector = np.asarray(n_detector, dtype=np.float64).reshape(-1)
    events_per_beam_phase = normalize_events_per_beam_phase_backend(events_per_beam_phase)

    gamma_rad = float(gamma_deg) * (pi / 180.0)
    Gamma_rad = float(Gamma_deg) * (pi / 180.0)
    chi_rad = float(chi_deg) * (pi / 180.0)
    psi_rad = float(psi_deg) * (pi / 180.0)
    sigma_rad = float(sigma_pv_deg) * (pi / 180.0)
    gamma_rad_m = float(gamma_pv_deg) * (pi / 180.0)
    solve_q_steps_i = int(np.clip(int(solve_q_steps), MIN_SOLVE_Q_STEPS, MAX_SOLVE_Q_STEPS))
    solve_q_rel_tol_i = float(np.clip(float(solve_q_rel_tol), MIN_SOLVE_Q_REL_TOL, MAX_SOLVE_Q_REL_TOL))
    solve_q_mode_i = SOLVE_Q_MODE_UNIFORM if int(solve_q_mode) == SOLVE_Q_MODE_UNIFORM else SOLVE_Q_MODE_ADAPTIVE

    cg = cos(gamma_rad)
    sg = sin(gamma_rad)
    cG = cos(Gamma_rad)
    sG = sin(Gamma_rad)
    R_x_det = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    R_z_det = np.array([[cG, sG, 0.0], [-sG, cG, 0.0], [0.0, 0.0, 1.0]])
    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len = sqrt(float(np.dot(n_det_rot, n_det_rot)))
    if nd_len > 1.0e-30:
        n_det_rot = n_det_rot / nd_len

    Detector_Pos = np.array([0.0, float(Distance_CoR_to_Detector), 0.0], dtype=np.float64)

    dot_e1 = float(np.dot(unit_x, n_det_rot))
    e1_det = unit_x - dot_e1 * n_det_rot
    e1_len = sqrt(float(np.dot(e1_det, e1_det)))
    if e1_len < 1.0e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det = e1_det / e1_len

    e2_det = -np.cross(n_det_rot, e1_det)
    e2_len = sqrt(float(np.dot(e2_det, e2_det)))
    if e2_len < 1.0e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det = e2_det / e2_len

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    R_y = np.array([[c_chi, 0.0, s_chi], [0.0, 1.0, 0.0], [-s_chi, 0.0, c_chi]])
    c_psi = cos(psi_rad)
    s_psi = sin(psi_rad)
    R_z = np.array([[c_psi, s_psi, 0.0], [-s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]])
    R_z_R_y = R_z @ R_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n = R_z_R_y @ n1
    nzy_len = sqrt(float(np.dot(R_ZY_n, R_ZY_n)))
    if nzy_len > 1.0e-30:
        R_ZY_n = R_ZY_n / nzy_len
    P0 = np.array([0.0, 0.0, -float(zs)], dtype=np.float64)

    num_peaks = int(miller.shape[0])
    n_samp = int(beam_x_array.size)
    collect_tables = bool(collect_hit_tables)
    accumulate_image_flag = bool(accumulate_image)
    all_status = np.zeros((num_peaks, n_samp), dtype=np.int64)
    event_counts = np.zeros((num_peaks, n_samp), dtype=np.int64)

    if save_flag == 1:
        max_solutions = 2000000
        q_data = np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count = np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data = np.zeros((1, 1, 5), dtype=np.float64)
        q_count = np.zeros(1, dtype=np.int64)

    if collect_tables:
        hit_rows_by_peak = [[] for _ in range(num_peaks)]
        miss_tables = [np.empty((0, 3), dtype=np.float64) for _ in range(num_peaks)]
    else:
        hit_rows_by_peak = None
        miss_tables = []
    representative_rows_by_peak: list[dict[int, tuple[tuple[float, float, int, int, int], np.ndarray]]] = [
        {} for _ in range(num_peaks)
    ]

    sample_weight_array = sample_weights
    if sample_weight_array is not None:
        sample_weight_array = np.asarray(sample_weight_array, dtype=np.float64).reshape(-1)
        if sample_weight_array.shape[0] != n_samp:
            sample_weight_array = None

    n2_sample_array = np.empty(n_samp, dtype=np.complex128)
    n2_override_array = np.empty(0, dtype=np.complex128)
    has_n2_override = n2_sample_array_override is not None
    if has_n2_override:
        n2_override_array = np.asarray(n2_sample_array_override, dtype=np.complex128).reshape(-1)
    if has_n2_override and n2_override_array.size == n_samp:
        n2_sample_array[:] = n2_override_array
    elif wavelength_array.size == n_samp:
        for i_samp in range(n_samp):
            lam_angstrom = wavelength_array[i_samp]
            if np.isfinite(lam_angstrom) and lam_angstrom > 0.0:
                n2_sample_array[i_samp] = IndexofRefraction(lam_angstrom * 1.0e-10)
            else:
                n2_sample_array[i_samp] = n2
    else:
        for i_samp in range(n_samp):
            n2_sample_array[i_samp] = n2

    (
        R_sample_precomputed,
        sample_terms,
        sample_n2_array,
        sample_eps2_array,
        _best_idx_precomputed,
    ) = _precompute_sample_terms(
        wavelength_array,
        n2,
        n2_sample_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        optics_mode,
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    )

    debye_x_sq = float(debye_x) * float(debye_x)
    debye_y_sq = float(debye_y) * float(debye_y)

    for sample_idx in range(n_samp):
        if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
            all_status[:, sample_idx] = -10
            continue

        sample_weight = 1.0
        if sample_weight_array is not None:
            sample_weight = float(sample_weight_array[sample_idx])
            if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                all_status[:, sample_idx] = -12
                continue

        sample_candidates: list[tuple[int, float, float, float, float, float, float, float]] = []
        sample_masses: list[float] = []
        sample_total_mass = 0.0

        for i_pk in range(num_peaks):
            reflection_intensity = float(intensities[i_pk]) if i_pk < intensities.shape[0] else 0.0
            if not np.isfinite(reflection_intensity) or reflection_intensity <= 0.0:
                continue

            H = float(miller[i_pk, 0])
            K = float(miller[i_pk, 1])
            L = float(miller[i_pk, 2])
            if L < 0.0:
                continue
            gz0 = 2.0 * pi * (L / float(cv))
            gr0 = 4.0 * pi / float(av) * sqrt(max((H * H + H * K + K * K) / 3.0, 0.0))
            G_vec = np.array([0.0, gr0, gz0], dtype=np.float64)

            k_in_crystal = np.array(
                [
                    sample_terms[sample_idx, _SAMPLE_COL_KX_SCAT],
                    sample_terms[sample_idx, _SAMPLE_COL_KY_SCAT],
                    sample_terms[sample_idx, _SAMPLE_COL_RE_KZ],
                ],
                dtype=np.float64,
            )
            solve_k_scat = float(sample_terms[sample_idx, _SAMPLE_COL_K_SCAT])
            all_q, stat = solve_q(
                k_in_crystal,
                solve_k_scat,
                G_vec,
                sigma_rad,
                gamma_rad_m,
                eta_pv,
                H,
                K,
                L,
                solve_q_steps_i,
                DEFAULT_SOLVE_Q_BASE_INTERVALS,
                solve_q_rel_tol_i,
                solve_q_mode_i,
                default_solve_q_dtheta,
                default_solve_q_cos,
                default_solve_q_sin,
            )
            all_status[i_pk, sample_idx] = int(stat)

            peak_projection_debug_counters = None
            peak_projection_debug_reject_count = None
            peak_projection_debug_reject_records = None
            peak_projection_debug_row_hit_counts = None
            peak_projection_debug_row_tthp_sums = None
            peak_projection_debug_row_tth_sums = None
            peak_projection_debug_enabled = (
                projection_debug_counters is not None
                and projection_debug_reject_counts is not None
                and projection_debug_reject_records is not None
                and projection_debug_row_hit_counts is not None
                and projection_debug_row_tthp_sums is not None
                and projection_debug_row_tth_sums is not None
            )
            if peak_projection_debug_enabled:
                peak_projection_debug_counters = projection_debug_counters[i_pk]
                peak_projection_debug_reject_count = projection_debug_reject_counts[i_pk : i_pk + 1]
                peak_projection_debug_reject_records = projection_debug_reject_records[i_pk]
                peak_projection_debug_row_hit_counts = projection_debug_row_hit_counts[0]
                peak_projection_debug_row_tthp_sums = projection_debug_row_tthp_sums[0]
                peak_projection_debug_row_tth_sums = projection_debug_row_tth_sums[0]

            for q_idx in range(int(np.asarray(all_q).shape[0])):
                Qx = float(all_q[q_idx, 0])
                Qy = float(all_q[q_idx, 1])
                Qz = float(all_q[q_idx, 2])
                I_Q = float(all_q[q_idx, 3])
                candidate_valid, row_f, col_f, phi_f, mass = _project_weighted_candidate(
                    H=H,
                    K=K,
                    L=L,
                    Qx=Qx,
                    Qy=Qy,
                    Qz=Qz,
                    I_Q=I_Q,
                    reflection_intensity=reflection_intensity,
                    sample_weight=sample_weight,
                    debye_x_sq=debye_x_sq,
                    debye_y_sq=debye_y_sq,
                    center=center,
                    R_sample=R_sample_precomputed,
                    n_det_rot=n_det_rot,
                    Detector_Pos=Detector_Pos,
                    e1_det=e1_det,
                    e2_det=e2_det,
                    I_plane=np.array(
                        [
                            sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_X],
                            sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Y],
                            sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Z],
                        ],
                        dtype=np.float64,
                    ),
                    k_x_scat=sample_terms[sample_idx, _SAMPLE_COL_KX_SCAT],
                    k_y_scat=sample_terms[sample_idx, _SAMPLE_COL_KY_SCAT],
                    re_k_z=sample_terms[sample_idx, _SAMPLE_COL_RE_KZ],
                    im_k_z=sample_terms[sample_idx, _SAMPLE_COL_IM_KZ],
                    k_scat=sample_terms[sample_idx, _SAMPLE_COL_K_SCAT],
                    k0=sample_terms[sample_idx, _SAMPLE_COL_K0],
                    Ti2=sample_terms[sample_idx, _SAMPLE_COL_TI2],
                    L_in=sample_terms[sample_idx, _SAMPLE_COL_L_IN],
                    n2_real=sample_terms[sample_idx, _SAMPLE_COL_N2_REAL],
                    n2_samp=sample_n2_array[sample_idx],
                    eps2=sample_eps2_array[sample_idx],
                    thickness=thickness,
                    optics_mode=optics_mode,
                    pixel_size_m=pixel_size_m,
                    image_size=image_size,
                    exit_projection_mode=exit_projection_mode,
                    projection_debug_enabled=peak_projection_debug_enabled,
                    projection_debug_counters=peak_projection_debug_counters,
                    projection_debug_reject_count=peak_projection_debug_reject_count,
                    projection_debug_reject_records=peak_projection_debug_reject_records,
                    projection_debug_row_hit_counts=peak_projection_debug_row_hit_counts,
                    projection_debug_row_tthp_sums=peak_projection_debug_row_tthp_sums,
                    projection_debug_row_tth_sums=peak_projection_debug_row_tth_sums,
                )
                if save_flag == 1 and _candidate_valid_mass(mass) and q_count[i_pk] < q_data.shape[1]:
                    idx = int(q_count[i_pk])
                    q_data[i_pk, idx, 0] = Qx
                    q_data[i_pk, idx, 1] = Qy
                    q_data[i_pk, idx, 2] = Qz
                    q_data[i_pk, idx, 3] = float(mass)
                    q_data[i_pk, idx, 4] = float(sample_idx)
                    q_count[i_pk] += 1
                candidate_valid = bool(
                    candidate_valid
                    and _candidate_has_detector_support(int(image_size), float(row_f), float(col_f))
                )
                if not candidate_valid or not _candidate_valid_mass(mass):
                    continue

                branch_id = _candidate_branch_id(H, K, phi_f, Qx)
                representative_key = (
                    float(_candidate_top_mosaic_distance(
                        sample_idx,
                        beam_x_array,
                        beam_y_array,
                        theta_array,
                        phi_array,
                        wavelength_array,
                    )),
                    float(-mass),
                    int(sample_idx),
                    int(i_pk),
                    int(q_idx),
                )
                representative_row = np.array(
                    [
                        float(mass),
                        float(col_f),
                        float(row_f),
                        float(phi_f),
                        H,
                        K,
                        L,
                        np.nan,
                        np.nan,
                        float(sample_idx),
                    ],
                    dtype=np.float64,
                )
                current_rep = representative_rows_by_peak[i_pk].get(branch_id)
                if current_rep is None or representative_key < current_rep[0]:
                    representative_rows_by_peak[i_pk][branch_id] = (
                        representative_key,
                        representative_row,
                    )

                sample_candidates.append(
                    (
                        int(i_pk),
                        float(row_f),
                        float(col_f),
                        float(phi_f),
                        H,
                        K,
                        L,
                        float(mass),
                    )
                )
                sample_masses.append(float(mass))
                sample_total_mass += float(mass)

        if not np.isfinite(sample_total_mass) or sample_total_mass <= 0.0:
            continue

        targets = _weighted_event_targets(sample_total_mass, events_per_beam_phase, sample_idx)
        selected_indices = _select_weighted_event_indices_from_targets(
            np.asarray(sample_masses, dtype=np.float64),
            targets,
        )
        deposit = _weighted_event_deposit(sample_total_mass, events_per_beam_phase)
        if not _candidate_valid_mass(deposit):
            continue

        for selected_idx in np.asarray(selected_indices, dtype=np.int64):
            i_pk, row_f, col_f, phi_f, H, K, L, _mass = sample_candidates[int(selected_idx)]
            if accumulate_image_flag:
                deposited = _accumulate_bilinear_hit(
                    image,
                    int(image_size),
                    float(row_f),
                    float(col_f),
                    float(deposit),
                )
                if not deposited:
                    continue
            if collect_tables:
                hit_rows_by_peak[int(i_pk)].append(
                    [
                        float(deposit),
                        float(col_f),
                        float(row_f),
                        float(phi_f),
                        float(H),
                        float(K),
                        float(L),
                        np.nan,
                        np.nan,
                        float(sample_idx),
                    ]
                )
            event_counts[int(i_pk), sample_idx] += 1

    if best_sample_indices_out is not None:
        best_sample_indices_out[:] = -1
        for i_pk in range(num_peaks):
            best_count = -1
            best_sample = -1
            for sample_idx in range(n_samp):
                count = int(event_counts[i_pk, sample_idx])
                if count > best_count:
                    best_count = count
                    best_sample = sample_idx
            if best_count > 0 and i_pk < best_sample_indices_out.shape[0]:
                best_sample_indices_out[i_pk] = int(best_sample)
            elif i_pk < best_sample_indices_out.shape[0]:
                best_sample_indices_out[i_pk] = -1

    if collect_tables:
        hit_tables = [
            np.asarray(rows, dtype=np.float64).reshape(-1, HIT_ROW_WITH_PROVENANCE_WIDTH)
            if len(rows) > 0
            else np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
            for rows in hit_rows_by_peak
        ]
    else:
        hit_tables = []

    representative_hit_tables = []
    for i_pk in range(num_peaks):
        branch_rows = representative_rows_by_peak[i_pk]
        if not branch_rows:
            representative_hit_tables.append(
                np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
            )
            continue
        ordered_rows = [
            np.asarray(branch_rows[branch_id][1], dtype=np.float64)
            for branch_id in sorted(branch_rows.keys())
        ]
        representative_hit_tables.append(
            np.asarray(ordered_rows, dtype=np.float64).reshape(-1, HIT_ROW_WITH_PROVENANCE_WIDTH)
        )

    return image, hit_tables, q_data, q_count, all_status, miss_tables, representative_hit_tables


def process_peaks_parallel(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=50e-9,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    sample_qr_ring_once=True,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    default_solve_q_dtheta=-1.0,
    default_solve_q_cos=None,
    default_solve_q_sin=None,
    numba_thread_count=None,
    events_per_beam_phase=50,
):
    """Inject runtime defaults before entering cached compiled kernel."""

    (
        default_solve_q_dtheta,
        default_solve_q_cos,
        default_solve_q_sin,
        numba_thread_count,
    ) = _resolve_process_peaks_runtime_values(
        default_solve_q_dtheta,
        default_solve_q_cos,
        default_solve_q_sin,
        numba_thread_count,
    )
    events_per_beam_phase = normalize_events_per_beam_phase_backend(events_per_beam_phase)
    (
        image_out,
        hit_tables,
        q_data,
        q_count,
        all_status,
        miss_tables,
    ) = _process_peaks_parallel_impl(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        record_status,
        thickness,
        optics_mode,
        solve_q_steps,
        solve_q_rel_tol,
        solve_q_mode,
        sample_weights,
        best_sample_indices_out,
        collect_hit_tables,
        pixel_size_m,
        sample_width_m,
        sample_length_m,
        n2_sample_array_override,
        accumulate_image,
        sample_qr_ring_once,
        exit_projection_mode,
        projection_debug_counters,
        projection_debug_reject_counts,
        projection_debug_reject_records,
        projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums,
        default_solve_q_dtheta=default_solve_q_dtheta,
        default_solve_q_cos=default_solve_q_cos,
        default_solve_q_sin=default_solve_q_sin,
        numba_thread_count=numba_thread_count,
        events_per_beam_phase=events_per_beam_phase,
    )
    return image_out, hit_tables, q_data, q_count, all_status, miss_tables


def _process_peaks_parallel_py_func(*args, **kwargs):
    args, kwargs = _normalize_bound_process_peaks_parallel_call(args, kwargs)
    bound_result = _bind_process_peaks_parallel_call(args, kwargs)
    if bound_result is None:
        _set_last_process_peaks_weighted_event_stats()
        result = _process_peaks_parallel_weighted_events_python(*args, **kwargs)
        _set_last_process_peaks_representative_hit_tables(result[6])
        return result[:6]
    bound, extra_kwargs = bound_result
    (
        bound["default_solve_q_dtheta"],
        bound["default_solve_q_cos"],
        bound["default_solve_q_sin"],
        bound["numba_thread_count"],
    ) = _resolve_process_peaks_runtime_values(
        bound["default_solve_q_dtheta"],
        bound["default_solve_q_cos"],
        bound["default_solve_q_sin"],
        bound["numba_thread_count"],
    )
    call_kwargs = dict(extra_kwargs)
    for name in _PROCESS_PEAKS_PARALLEL_PARAM_NAMES[len(args) :]:
        call_kwargs[name] = bound[name]
    _set_last_process_peaks_weighted_event_stats()
    result = _process_peaks_parallel_weighted_events_python(*args, **call_kwargs)
    _set_last_process_peaks_representative_hit_tables(result[6])
    return result[:6]


process_peaks_parallel.py_func = _process_peaks_parallel_py_func


_PROCESS_PEAKS_RUNTIME_KWARG_NAMES = (
    "default_solve_q_dtheta",
    "default_solve_q_cos",
    "default_solve_q_sin",
    "numba_thread_count",
)


def _get_unexpected_keyword_name(exc: TypeError):
    message = str(exc)
    marker = "unexpected keyword argument "
    marker_index = message.find(marker)
    if marker_index < 0:
        return None
    rest = message[marker_index + len(marker) :].lstrip()
    if not rest or rest[0] not in ("'", '"'):
        return None
    quote = rest[0]
    end_index = rest.find(quote, 1)
    if end_index < 0:
        return None
    return rest[1:end_index]


def _is_unexpected_keyword_error(exc: TypeError, keyword: str) -> bool:
    return _get_unexpected_keyword_name(exc) == keyword


def _prepare_clustered_process_peaks_call(args, kwargs):
    return args, dict(kwargs), None


def _finalize_clustered_process_peaks_result(result, cluster_meta):
    if cluster_meta is None:
        return result

    image, hit_tables, q_data, q_count, all_status, miss_tables = result
    raw_to_cluster = cluster_meta["raw_to_cluster"]
    cluster_to_rep = cluster_meta["cluster_to_rep"]
    if (
        isinstance(all_status, np.ndarray)
        and all_status.ndim == 2
        and all_status.shape[1] == cluster_meta["cluster_count"]
    ):
        all_status = np.asarray(all_status[:, raw_to_cluster], dtype=np.int64)

    best_sample_indices_out = cluster_meta["best_sample_indices_out"]
    cluster_best_indices_out = cluster_meta["cluster_best_indices_out"]
    if best_sample_indices_out is not None and cluster_best_indices_out is not None:
        best_sample_indices_out[:] = -1
        valid = (np.asarray(cluster_best_indices_out) >= 0) & (
            np.asarray(cluster_best_indices_out) < cluster_to_rep.shape[0]
        )
        best_sample_indices_out[valid] = cluster_to_rep[np.asarray(cluster_best_indices_out)[valid]]

    return image, hit_tables, q_data, q_count, all_status, miss_tables


def process_peaks_parallel_safe(*args, **kwargs):
    """Run ``process_peaks_parallel`` with Python fallback if JIT execution fails."""

    safe_stats_out = kwargs.pop("_safe_stats_out", None)
    prefer_python_runner = bool(kwargs.pop("prefer_python_runner", False))
    enable_safe_cache = kwargs.pop("enable_safe_cache", None)
    sample_qr_ring_once = bool(kwargs.pop("sample_qr_ring_once", True))
    kwargs["sample_qr_ring_once"] = sample_qr_ring_once
    for name, value in get_process_peaks_runtime_kwargs().items():
        kwargs.setdefault(name, value)
    args, kwargs = _normalize_bound_process_peaks_parallel_call(args, kwargs)
    _set_last_process_peaks_safe_stats()
    _set_last_process_peaks_representative_hit_tables(None)
    if isinstance(safe_stats_out, dict):
        safe_stats_out.clear()
    (
        av,
        cv,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        wavelength_array,
        best_sample_indices_out,
    ) = _extract_process_peaks_context(args, kwargs)
    _set_last_intersection_cache([])
    safe_cache_result = _maybe_run_process_peaks_safe_cache(
        args,
        kwargs,
        enable_safe_cache,
    )
    if safe_cache_result is not None:
        if isinstance(safe_stats_out, dict):
            safe_stats_out["used_python_runner"] = False
        _set_last_intersection_cache_from_hit_tables(
            safe_cache_result[1],
            av,
            cv,
            beam_x_array=beam_x_array,
            beam_y_array=beam_y_array,
            theta_array=theta_array,
            phi_array=phi_array,
            wavelength_array=wavelength_array,
            best_sample_indices_out=best_sample_indices_out,
            representative_hit_tables=get_last_process_peaks_representative_hit_tables(),
        )
        return safe_cache_result
    clustered_args, clustered_kwargs, cluster_meta = _prepare_clustered_process_peaks_call(
        args,
        kwargs,
    )
    call_variants = []
    if cluster_meta is not None:
        call_variants.append((clustered_args, clustered_kwargs, cluster_meta))
    call_variants.append((args, dict(kwargs), None))

    py_runner = getattr(process_peaks_parallel, "py_func", None)
    runners = []
    if prefer_python_runner and callable(py_runner):
        runners.append(py_runner)
    runners.append(process_peaks_parallel)
    if (not prefer_python_runner) and callable(py_runner):
        runners.append(py_runner)

    last_exc = None
    for runner in runners:
        for call_args, base_call_kwargs, call_meta in call_variants:
            call_kwargs = dict(base_call_kwargs)
            while True:
                try:
                    result = runner(*call_args, **call_kwargs)
                    used_python_runner = bool(callable(py_runner) and runner is py_runner)
                    _set_last_process_peaks_safe_stats(
                        used_python_runner=used_python_runner,
                    )
                    if isinstance(safe_stats_out, dict):
                        safe_stats_out["used_python_runner"] = used_python_runner
                    _set_last_intersection_cache_from_hit_tables(
                        result[1],
                        av,
                        cv,
                        beam_x_array=beam_x_array,
                        beam_y_array=beam_y_array,
                        theta_array=theta_array,
                        phi_array=phi_array,
                        wavelength_array=wavelength_array,
                        best_sample_indices_out=best_sample_indices_out,
                        representative_hit_tables=get_last_process_peaks_representative_hit_tables(),
                    )
                    return _finalize_clustered_process_peaks_result(result, call_meta)
                except TypeError as exc:
                    last_exc = exc
                    unexpected_keyword = _get_unexpected_keyword_name(exc)
                    if (
                        unexpected_keyword in _PROCESS_PEAKS_RUNTIME_KWARG_NAMES
                        and unexpected_keyword in call_kwargs
                    ):
                        call_kwargs = dict(call_kwargs)
                        call_kwargs.pop(unexpected_keyword, None)
                        continue
                    if call_meta is not None and unexpected_keyword == "sample_weights":
                        break
                    break
                except Exception as exc:
                    last_exc = exc
                    break

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("process_peaks_parallel_safe could not execute any runner")


def _quadratic_peak_from_samples(
    cols,
    rows,
    intensities,
    *,
    anchor_col,
    anchor_row,
    support_radius_px,
):
    """Estimate a local maximum by fitting a quadratic surface in log-intensity."""

    cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
    rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
    intensities_arr = np.asarray(intensities, dtype=np.float64).reshape(-1)
    finite_mask = (
        np.isfinite(cols_arr)
        & np.isfinite(rows_arr)
        & np.isfinite(intensities_arr)
        & (intensities_arr > 0.0)
    )
    if np.count_nonzero(finite_mask) < 6:
        return None

    cols_arr = cols_arr[finite_mask]
    rows_arr = rows_arr[finite_mask]
    intensities_arr = intensities_arr[finite_mask]
    x = cols_arr - float(anchor_col)
    y = rows_arr - float(anchor_row)

    radius = max(float(support_radius_px), 0.5)
    local_mask = (x * x + y * y) <= radius * radius + 1.0e-12
    if np.count_nonzero(local_mask) >= 6:
        x = x[local_mask]
        y = y[local_mask]
        intensities_arr = intensities_arr[local_mask]

    if x.size < 6:
        return None

    max_intensity = float(np.max(intensities_arr))
    if not np.isfinite(max_intensity) or max_intensity <= 0.0:
        return None

    z = np.log(np.clip(intensities_arr / max_intensity, 1.0e-12, None))
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
    weights = np.sqrt(np.clip(intensities_arr / max_intensity, 1.0e-6, None))

    try:
        coeffs, *_ = np.linalg.lstsq(design * weights[:, None], z * weights, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if coeffs.shape[0] != 6 or not np.all(np.isfinite(coeffs)):
        return None

    a, b, c, d, e, _ = coeffs
    hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=np.float64)
    det = float(np.linalg.det(hessian))
    if not np.isfinite(det) or det <= 1.0e-12 or not (a < -1.0e-12 and b < -1.0e-12):
        return None

    try:
        offset = -np.linalg.solve(hessian, np.array([d, e], dtype=np.float64))
    except np.linalg.LinAlgError:
        return None

    if offset.shape[0] != 2 or not np.all(np.isfinite(offset)):
        return None

    offset_col = float(offset[0])
    offset_row = float(offset[1])
    if np.hypot(offset_col, offset_row) > max(radius, 1.5):
        return None

    x_margin = max(0.35, 0.15 * radius)
    y_margin = max(0.35, 0.15 * radius)
    if offset_col < float(np.min(x)) - x_margin or offset_col > float(np.max(x)) + x_margin:
        return None
    if offset_row < float(np.min(y)) - y_margin or offset_row > float(np.max(y)) + y_margin:
        return None

    return float(anchor_col) + offset_col, float(anchor_row) + offset_row


def _refine_cluster_peak(hits_arr, *, merge_radius_px=1.5):
    """Return a peak-focused subpixel center for one merged hit cluster."""

    hits = np.asarray(hits_arr, dtype=np.float64)
    if hits.ndim != 2 or hits.shape[1] < 3 or hits.shape[0] == 0:
        return float("nan"), float("nan")

    intensities = hits[:, 0]
    cols = hits[:, 1]
    rows = hits[:, 2]
    valid_mask = (
        np.isfinite(intensities) & np.isfinite(cols) & np.isfinite(rows) & (intensities > 0.0)
    )
    if not np.any(valid_mask):
        return float("nan"), float("nan")

    intensities = intensities[valid_mask]
    cols = cols[valid_mask]
    rows = rows[valid_mask]
    anchor_idx = int(np.argmax(intensities))
    anchor_col = float(cols[anchor_idx])
    anchor_row = float(rows[anchor_idx])

    dist_sq = (cols - anchor_col) ** 2 + (rows - anchor_row) ** 2
    support_radius = max(0.75, 0.75 * float(merge_radius_px))
    local_mask = dist_sq <= support_radius * support_radius + 1.0e-12
    if np.count_nonzero(local_mask) < min(3, cols.size):
        order = np.argsort(dist_sq)
        local_mask = np.zeros_like(dist_sq, dtype=bool)
        local_mask[order[: min(max(cols.size, 1), 4)]] = True

    local_cols = cols[local_mask]
    local_rows = rows[local_mask]
    local_intensities = intensities[local_mask]
    local_dist_sq = dist_sq[local_mask]

    sigma_sq = max((0.45 * max(float(merge_radius_px), 1.0)) ** 2, 1.0e-6)
    local_weights = np.power(local_intensities, 1.5) * np.exp(-0.5 * local_dist_sq / sigma_sq)
    total_weight = float(np.sum(local_weights))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        local_weights = np.clip(local_intensities, 0.0, None)
        total_weight = float(np.sum(local_weights))

    if total_weight <= 0.0 or not np.isfinite(total_weight):
        center_col = float(anchor_col)
        center_row = float(anchor_row)
    else:
        center_col = float(np.sum(local_weights * local_cols) / total_weight)
        center_row = float(np.sum(local_weights * local_rows) / total_weight)

    refined = _quadratic_peak_from_samples(
        local_cols,
        local_rows,
        local_intensities,
        anchor_col=center_col,
        anchor_row=center_row,
        support_radius_px=support_radius,
    )
    if refined is not None:
        return refined
    return center_col, center_row


def _cluster_hit_positions(hits_arr, *, merge_radius_px=1.5):
    """Merge nearby hit-table rows into subpixel centroids."""

    merge_radius_sq = float(merge_radius_px) * float(merge_radius_px)
    clusters = []

    for hit in hits_arr[np.argsort(hits_arr[:, 0])[::-1]]:
        intensity = float(hit[0])
        col = float(hit[1])
        row = float(hit[2])
        if not (
            np.isfinite(intensity) and np.isfinite(col) and np.isfinite(row) and intensity > 0.0
        ):
            continue

        best_cluster_idx = None
        best_dist_sq = float("inf")
        for idx, cluster in enumerate(clusters):
            center_col = cluster["weighted_col_sum"] / cluster["total_intensity"]
            center_row = cluster["weighted_row_sum"] / cluster["total_intensity"]
            dist_sq = (col - center_col) ** 2 + (row - center_row) ** 2
            if dist_sq <= merge_radius_sq and dist_sq < best_dist_sq:
                best_cluster_idx = idx
                best_dist_sq = dist_sq

        if best_cluster_idx is None:
            clusters.append(
                {
                    "total_intensity": intensity,
                    "peak_intensity": intensity,
                    "weighted_col_sum": intensity * col,
                    "weighted_row_sum": intensity * row,
                    "hits": [(intensity, col, row)],
                }
            )
            continue

        cluster = clusters[best_cluster_idx]
        cluster["total_intensity"] += intensity
        cluster["weighted_col_sum"] += intensity * col
        cluster["weighted_row_sum"] += intensity * row
        cluster["hits"].append((intensity, col, row))
        if intensity > cluster["peak_intensity"]:
            cluster["peak_intensity"] = intensity

    clusters.sort(
        key=lambda cluster: (
            float(cluster["total_intensity"]),
            float(cluster["peak_intensity"]),
        ),
        reverse=True,
    )

    out = []
    for cluster in clusters:
        total_intensity = float(cluster["total_intensity"])
        if total_intensity <= 0.0:
            continue
        refined_col, refined_row = _refine_cluster_peak(
            np.asarray(cluster["hits"], dtype=np.float64),
            merge_radius_px=merge_radius_px,
        )
        if not (np.isfinite(refined_col) and np.isfinite(refined_row)):
            refined_col = float(cluster["weighted_col_sum"]) / total_intensity
            refined_row = float(cluster["weighted_row_sum"]) / total_intensity
        out.append(
            (
                total_intensity,
                float(refined_col),
                float(refined_row),
            )
        )
    return out


def hit_tables_to_max_positions(hit_tables):
    """Extract up to two subpixel peak centers per reflection from ``hit_tables``.

    ``process_peaks_parallel`` returns a list of pixel-hit tables whose first
    seven columns are ``[intensity, col, row, phi, H, K, L]``. Current kernel
    rows may also carry ``[source_table_index, source_row_index,
    best_sample_index]`` so the intersection cache can recover the detector ray
    closest to the mosaic top. The ``col``/``row`` coordinates are stored in
    floating detector-pixel units. Older callers
    expect a ``max_positions`` array shaped ``(N, 6)`` containing the two
    strongest candidate peak centers per reflection:
    ``(I0, x0, y0, I1, x1, y1)``.  Nearby hit-table rows are merged into
    intensity-weighted centroids so small parameter changes remain visible to
    the optimizer.
    """

    num_peaks = len(hit_tables)
    max_positions = np.zeros((num_peaks, 6), dtype=np.float64)

    for i, hits in enumerate(hit_tables):
        hits_arr = np.asarray(hits)
        if hits_arr.size == 0:
            continue

        clustered_hits = _cluster_hit_positions(hits_arr)
        if not clustered_hits:
            continue

        primary = clustered_hits[0]
        max_positions[i, 0:3] = primary

        if len(clustered_hits) > 1:
            secondary = clustered_hits[1]
            max_positions[i, 3:6] = secondary

    return max_positions


def intersection_cache_to_hit_tables(intersection_cache):
    """Convert intersection-cache tables into legacy hit-table rows.

    Supported cache layouts are detector ``14``/``17`` columns and caked
    ``16``/``19`` columns. Downstream geometry and selection paths expect base
    hit rows shaped ``[intensity, col, row, phi, H, K, L]`` and append
    ``[source_table_index, source_row_index, best_sample_index]`` only when the
    cache layout truly carries provenance. Input table count stays unchanged.
    """

    hit_tables = []
    if intersection_cache is None:
        return hit_tables

    for table in intersection_cache:
        hit_tables.append(cache_table_to_hit_table(table))

    return hit_tables


def _copy_intersection_cache(cache):
    """Return one detached copy of the mosaic-ring intersection cache."""

    return [
        coerce_float64_table(
            table,
            empty_width=CURRENT_DETECTOR_CACHE_WIDTH,
        )
        for table in cache
    ]


def _set_last_intersection_cache(cache):
    """Store the latest detector intersection cache."""

    global _LAST_INTERSECTION_CACHE
    global _LAST_INTERSECTION_CACHE_VIEWS
    if not _retain_last_intersection_cache():
        _LAST_INTERSECTION_CACHE = []
        _LAST_INTERSECTION_CACHE_VIEWS = {
            "sampled_event_rows": [],
            "branch_representative_rows": [],
        }
        return
    if cache is None:
        _LAST_INTERSECTION_CACHE = []
        _LAST_INTERSECTION_CACHE_VIEWS = {
            "sampled_event_rows": [],
            "branch_representative_rows": [],
        }
        return
    _LAST_INTERSECTION_CACHE = _copy_intersection_cache(cache)
    _LAST_INTERSECTION_CACHE_VIEWS = {
        "sampled_event_rows": [],
        "branch_representative_rows": _copy_intersection_cache(cache),
    }


def _copy_hit_table_sequence(hit_tables):
    copied = []
    for table in hit_tables or ():
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim != 2:
            copied.append(np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64))
            continue
        copied.append(np.array(arr, copy=True))
    return copied


def _set_last_process_peaks_representative_hit_tables(hit_tables):
    global _LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES
    if hit_tables is None:
        _LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES = None
        return
    _LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES = _copy_hit_table_sequence(hit_tables)


def get_last_process_peaks_representative_hit_tables():
    if _LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES is None:
        return None
    return _copy_hit_table_sequence(_LAST_PROCESS_PEAKS_REPRESENTATIVE_HIT_TABLES)


def get_last_intersection_cache_views():
    return {
        "sampled_event_rows": _copy_intersection_cache(
            _LAST_INTERSECTION_CACHE_VIEWS.get("sampled_event_rows", [])
        ),
        "branch_representative_rows": _copy_intersection_cache(
            _LAST_INTERSECTION_CACHE_VIEWS.get("branch_representative_rows", [])
        ),
    }


def _set_last_intersection_cache_from_hit_tables(
    hit_tables,
    av,
    cv,
    *,
    beam_x_array=None,
    beam_y_array=None,
    theta_array=None,
    phi_array=None,
    wavelength_array=None,
    best_sample_indices_out=None,
    representative_hit_tables=None,
):
    """Build and store the optional last-intersection cache when retained."""

    if not _retain_last_intersection_cache():
        return
    sampled_event_rows = build_intersection_cache(
        hit_tables,
        av,
        cv,
        beam_x_array=beam_x_array,
        beam_y_array=beam_y_array,
        theta_array=theta_array,
        phi_array=phi_array,
        wavelength_array=wavelength_array,
        best_sample_indices_out=best_sample_indices_out,
    )
    representative_rows = build_branch_representative_intersection_cache(
        representative_hit_tables if representative_hit_tables is not None else hit_tables,
        av,
        cv,
        beam_x_array=beam_x_array,
        beam_y_array=beam_y_array,
        theta_array=theta_array,
        phi_array=phi_array,
        wavelength_array=wavelength_array,
        best_sample_indices_out=best_sample_indices_out,
    )
    _set_last_intersection_cache(representative_rows)
    global _LAST_INTERSECTION_CACHE_VIEWS
    _LAST_INTERSECTION_CACHE_VIEWS = {
        "sampled_event_rows": _copy_intersection_cache(sampled_event_rows),
        "branch_representative_rows": _copy_intersection_cache(representative_rows),
    }


def get_last_intersection_cache():
    """Return the last fitter-facing branch representative detector cache.

    Fresh builds store the current 17-column detector layout:
    ``[Qr, Qz, detector_col, detector_row, intensity, phi, H, K, L,
    beam_x_offset, beam_z_offset, divergence_x_offset, divergence_z_offset,
    wavelength_offset, source_table_index, source_row_index, best_sample_index]``.
    Replay and conversion helpers remain backward-compatible with legacy
    14-column detector caches and caked layouts with widths 16 and 19.
    Sampled weighted-event rows are preserved separately under
    ``get_last_intersection_cache_views()["sampled_event_rows"]``.
    """

    return _copy_intersection_cache(_LAST_INTERSECTION_CACHE)


def load_logged_intersection_cache(
    log_dir: str | Path,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Load one persisted intersection-cache log directory from disk."""

    resolved_dir = Path(log_dir).expanduser()
    metadata: dict[str, object] = {}
    try:
        with (resolved_dir / "meta.json").open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            metadata = dict(loaded)
    except Exception:
        metadata = {}

    raw_table_files = metadata.get("table_files")
    if isinstance(raw_table_files, list) and raw_table_files:
        table_paths = [resolved_dir / str(name) for name in raw_table_files]
    else:
        table_paths = sorted(resolved_dir.glob("table_*.npy"))

    cache_tables: list[np.ndarray] = []
    for table_path in table_paths:
        try:
            table_arr = np.asarray(np.load(table_path, allow_pickle=False), dtype=np.float64)
        except Exception:
            continue
        if table_arr.ndim == 1:
            table_arr = table_arr.reshape(1, -1)
        if table_arr.ndim != 2:
            continue
        cache_tables.append(np.asarray(table_arr, dtype=np.float64).copy())

    if cache_tables:
        metadata.setdefault("log_dir", str(resolved_dir))
        metadata.setdefault("table_count", int(len(cache_tables)))
        metadata.setdefault(
            "rows_per_table",
            [int(np.asarray(table, dtype=np.float64).shape[0]) for table in cache_tables],
        )
        metadata.setdefault(
            "total_rows",
            int(sum(int(np.asarray(table, dtype=np.float64).shape[0]) for table in cache_tables)),
        )
    return cache_tables, metadata


def load_logged_intersection_cache_metadata(
    log_dir: str | Path,
) -> dict[str, object]:
    """Load only the metadata for one persisted intersection-cache directory."""

    resolved_dir = Path(log_dir).expanduser()
    try:
        with (resolved_dir / "meta.json").open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    metadata = dict(loaded)
    metadata.setdefault("log_dir", str(resolved_dir))
    return metadata


def _recent_intersection_cache_log_dirs_by_name(
    root: str | Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    """Return newest timestamp-named cache dirs without stat-sorting the log tree."""

    import heapq

    resolved_root = Path(root).expanduser()
    max_dirs = max(1, int(limit)) if limit is not None else None
    candidates: list[tuple[str, str]] = []
    try:
        with os.scandir(resolved_root) as entries:
            for entry in entries:
                name = str(entry.name)
                if not name.startswith("intersection_cache_"):
                    continue
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                item = (name, entry.path)
                if max_dirs is None:
                    candidates.append(item)
                elif len(candidates) < max_dirs:
                    heapq.heappush(candidates, item)
                else:
                    heapq.heappushpop(candidates, item)
    except Exception:
        return []
    return [Path(path) for _name, path in sorted(candidates, reverse=True)]


def load_most_recent_logged_intersection_cache(
    log_root: str | Path | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Load the newest persisted intersection-cache log that still has tables."""

    if log_root is None:
        root = _resolve_intersection_cache_log_root() / "intersection_cache"
    else:
        root = Path(log_root).expanduser()

    for log_dir in _recent_intersection_cache_log_dirs_by_name(root):
        cache_tables, metadata = load_logged_intersection_cache(log_dir)
        if cache_tables:
            metadata = dict(metadata)
            metadata.setdefault("log_dir", str(log_dir))
            return cache_tables, metadata
    return [], {}


def load_most_recent_logged_intersection_cache_metadata(
    log_root: str | Path | None = None,
) -> dict[str, object]:
    """Load only metadata for the newest persisted intersection-cache log."""

    if log_root is None:
        root = _resolve_intersection_cache_log_root() / "intersection_cache"
    else:
        root = Path(log_root).expanduser()

    for log_dir in _recent_intersection_cache_log_dirs_by_name(root):
        metadata = load_logged_intersection_cache_metadata(log_dir)
        if metadata:
            metadata.setdefault("log_dir", str(log_dir))
            return metadata
    return {}


def _intersection_cache_target_row_count(table: np.ndarray) -> int:
    """Return how many representative detector rows one cache table should keep."""

    if table.ndim != 2 or table.shape[0] <= 0 or table.shape[1] < 8:
        return 0
    try:
        h_val = float(table[0, 6])
        k_val = float(table[0, 7])
    except Exception:
        return 1
    if np.isfinite(h_val) and np.isfinite(k_val) and abs(h_val) < 1.0e-12 and abs(k_val) < 1.0e-12:
        return 1
    return 2


def _intersection_cache_representative_local_index(coords: np.ndarray) -> int:
    """Return the local row index nearest the unweighted mean detector position."""

    if coords.ndim != 2 or coords.shape[0] <= 0:
        return 0
    center = np.mean(coords, axis=0, dtype=np.float64)
    deltas = coords - center
    dist_sq = np.sum(deltas * deltas, axis=1)
    return int(np.argmin(dist_sq))


def _intersection_cache_row_sample_index(table: np.ndarray, row_idx: int) -> int | None:
    """Return the beam-sample index carried by one cache row when present."""

    if table.ndim != 2 or table.shape[1] <= CACHE_COL_BEST_SAMPLE_INDEX:
        return None
    value = table[int(row_idx), CACHE_COL_BEST_SAMPLE_INDEX]
    if not np.isfinite(value):
        return None
    sample_idx = int(np.rint(float(value)))
    if sample_idx < 0:
        return None
    return sample_idx


def _array_value(arr: np.ndarray | None, index: int) -> float:
    if arr is None or index < 0 or index >= arr.shape[0]:
        return float("nan")
    value = float(arr[index])
    return value if np.isfinite(value) else float("nan")


def _intersection_cache_top_mosaic_key(
    table: np.ndarray,
    row_idx: int,
    detector_center: np.ndarray,
    beam_x_array: np.ndarray | None = None,
    beam_y_array: np.ndarray | None = None,
    theta_array: np.ndarray | None = None,
    phi_array: np.ndarray | None = None,
    wavelength_array: np.ndarray | None = None,
) -> tuple[tuple[float, float, float, float, float, int], bool]:
    """Return a sortable key for the row nearest the top of the mosaic."""

    row_idx = int(row_idx)
    angular_metric = float("inf")
    beam_metric = float("inf")
    wavelength_metric = float("inf")
    has_top_metric = False

    sample_idx = _intersection_cache_row_sample_index(table, row_idx)
    if sample_idx is not None:
        theta_val = _array_value(theta_array, sample_idx)
        phi_val = _array_value(phi_array, sample_idx)
        if np.isfinite(theta_val) and np.isfinite(phi_val):
            angular_metric = theta_val * theta_val + phi_val * phi_val
            has_top_metric = True

        beam_x_val = _array_value(beam_x_array, sample_idx)
        beam_y_val = _array_value(beam_y_array, sample_idx)
        if np.isfinite(beam_x_val) and np.isfinite(beam_y_val):
            beam_metric = beam_x_val * beam_x_val + beam_y_val * beam_y_val

        wavelength_val = _array_value(wavelength_array, sample_idx)
        if np.isfinite(wavelength_val):
            if wavelength_array is not None and wavelength_array.size > 0:
                finite_wavelengths = wavelength_array[np.isfinite(wavelength_array)]
                if finite_wavelengths.size > 0:
                    wavelength_metric = abs(wavelength_val - float(np.mean(finite_wavelengths)))

    if not has_top_metric and table.shape[1] > 12:
        theta_offset = float(table[row_idx, 11])
        phi_offset = float(table[row_idx, 12])
        if np.isfinite(theta_offset) and np.isfinite(phi_offset):
            angular_metric = theta_offset * theta_offset + phi_offset * phi_offset
            has_top_metric = True
        if table.shape[1] > 10:
            beam_x_offset = float(table[row_idx, 9])
            beam_y_offset = float(table[row_idx, 10])
            if np.isfinite(beam_x_offset) and np.isfinite(beam_y_offset):
                beam_metric = beam_x_offset * beam_x_offset + beam_y_offset * beam_y_offset
        if table.shape[1] > 13:
            wavelength_offset = float(table[row_idx, 13])
            if np.isfinite(wavelength_offset):
                wavelength_metric = abs(wavelength_offset)

    col = float(table[row_idx, 2]) if table.shape[1] > 2 else float("nan")
    row = float(table[row_idx, 3]) if table.shape[1] > 3 else float("nan")
    if np.isfinite(col) and np.isfinite(row) and detector_center.shape[0] >= 2:
        detector_dist = (col - float(detector_center[0])) * (col - float(detector_center[0])) + (
            row - float(detector_center[1])
        ) * (row - float(detector_center[1]))
    else:
        detector_dist = float("inf")
    intensity = float(table[row_idx, 4]) if table.shape[1] > 4 else float("nan")
    intensity_key = -intensity if np.isfinite(intensity) else float("inf")
    return (
        (
            float(angular_metric),
            float(beam_metric),
            float(wavelength_metric),
            float(detector_dist),
            float(intensity_key),
            int(row_idx),
        ),
        bool(has_top_metric),
    )


def _intersection_cache_representative_row_index(
    table: np.ndarray,
    candidate_indices: np.ndarray,
    beam_x_array: np.ndarray | None = None,
    beam_y_array: np.ndarray | None = None,
    theta_array: np.ndarray | None = None,
    phi_array: np.ndarray | None = None,
    wavelength_array: np.ndarray | None = None,
) -> int:
    """Return the representative row, preferring the ray nearest mosaic center."""

    candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    if candidates.size <= 0:
        return 0

    coords = np.asarray(table[candidates, 2:4], dtype=np.float64)
    detector_center = np.mean(coords, axis=0, dtype=np.float64)
    best_idx: int | None = None
    best_key: tuple[float, float, float, float, float, int] | None = None
    for idx in candidates:
        key, has_top_metric = _intersection_cache_top_mosaic_key(
            table,
            int(idx),
            detector_center,
            beam_x_array=beam_x_array,
            beam_y_array=beam_y_array,
            theta_array=theta_array,
            phi_array=phi_array,
            wavelength_array=wavelength_array,
        )
        if not has_top_metric:
            continue
        if best_key is None or key < best_key:
            best_key = key
            best_idx = int(idx)
    if best_idx is not None:
        return best_idx

    rep_local_idx = _intersection_cache_representative_local_index(coords)
    return int(candidates[rep_local_idx])


def _intersection_cache_two_row_clusters(coords: np.ndarray) -> list[np.ndarray]:
    """Split detector coordinates into two spatial groups for non-specular peaks."""

    if coords.ndim != 2 or coords.shape[0] <= 1:
        return [np.arange(coords.shape[0], dtype=np.int64)]

    centered = coords - np.mean(coords, axis=0, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = np.asarray(vh[0], dtype=np.float64)
    except Exception:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    if axis.shape[0] < 2 or not np.all(np.isfinite(axis)):
        axis = np.array([1.0, 0.0], dtype=np.float64)

    projections = centered @ axis[:2]
    left = np.flatnonzero(projections < 0.0)
    right = np.flatnonzero(projections >= 0.0)
    if left.size > 0 and right.size > 0:
        return [
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
        ]

    ordered = np.argsort(projections, kind="mergesort")
    split_idx = max(1, coords.shape[0] // 2)
    if split_idx >= ordered.size:
        split_idx = ordered.size - 1
    if split_idx <= 0:
        return [np.asarray(ordered, dtype=np.int64)]
    return [
        np.asarray(ordered[:split_idx], dtype=np.int64),
        np.asarray(ordered[split_idx:], dtype=np.int64),
    ]


def _collapse_intersection_cache_table(cache_table: np.ndarray) -> np.ndarray:
    """Keep one central row per expected detector branch inside one Qr/Qz table."""

    table = np.asarray(cache_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] <= 0:
        return table

    selected_indices = _intersection_cache_selected_row_indices(table)
    if not selected_indices:
        return np.empty((0, table.shape[1]), dtype=np.float64)
    return table[np.asarray(selected_indices, dtype=np.int64), :].copy()


def _intersection_cache_selected_row_indices(
    cache_table: np.ndarray,
    *,
    beam_x_array: np.ndarray | None = None,
    beam_y_array: np.ndarray | None = None,
    theta_array: np.ndarray | None = None,
    phi_array: np.ndarray | None = None,
    wavelength_array: np.ndarray | None = None,
) -> list[int]:
    """Return the representative combined-table row indices kept by cache collapse."""

    table = np.asarray(cache_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] <= 0:
        return []
    if table.shape[0] == 1:
        return [0]

    target_count = _intersection_cache_target_row_count(table)
    if target_count <= 0:
        return []
    if table.shape[0] <= target_count:
        return list(range(int(table.shape[0])))

    coords = np.asarray(table[:, 2:4], dtype=np.float64)
    finite_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    valid_indices = np.flatnonzero(finite_mask)
    if valid_indices.size <= 0:
        return [int(idx) for idx in range(min(target_count, int(table.shape[0])))]

    valid_coords = coords[valid_indices, :]
    if target_count == 1 or valid_indices.size == 1:
        return [
            _intersection_cache_representative_row_index(
                table,
                valid_indices,
                beam_x_array=beam_x_array,
                beam_y_array=beam_y_array,
                theta_array=theta_array,
                phi_array=phi_array,
                wavelength_array=wavelength_array,
            )
        ]

    selected_indices: list[int] = []
    for cluster_local_indices in _intersection_cache_two_row_clusters(valid_coords):
        if cluster_local_indices.size <= 0:
            continue
        cluster_indices = valid_indices[cluster_local_indices]
        selected_indices.append(
            _intersection_cache_representative_row_index(
                table,
                cluster_indices,
                beam_x_array=beam_x_array,
                beam_y_array=beam_y_array,
                theta_array=theta_array,
                phi_array=phi_array,
                wavelength_array=wavelength_array,
            )
        )

    if not selected_indices:
        return [
            _intersection_cache_representative_row_index(
                table,
                valid_indices,
                beam_x_array=beam_x_array,
                beam_y_array=beam_y_array,
                theta_array=theta_array,
                phi_array=phi_array,
                wavelength_array=wavelength_array,
            )
        ]

    if len(selected_indices) < target_count:
        remaining = [idx for idx in valid_indices.tolist() if int(idx) not in selected_indices]
        while remaining and len(selected_indices) < target_count:
            rep_idx = _intersection_cache_representative_row_index(
                table,
                np.asarray(remaining, dtype=np.int64),
                beam_x_array=beam_x_array,
                beam_y_array=beam_y_array,
                theta_array=theta_array,
                phi_array=phi_array,
                wavelength_array=wavelength_array,
            )
            selected_indices.append(int(rep_idx))
            remaining = [idx for idx in remaining if int(idx) != int(rep_idx)]

    ordered_indices = sorted(
        {int(idx) for idx in selected_indices},
        key=lambda idx: (
            float(table[idx, 2]) if np.isfinite(table[idx, 2]) else float("inf"),
            float(table[idx, 3]) if np.isfinite(table[idx, 3]) else float("inf"),
            int(idx),
        ),
    )
    return ordered_indices[:target_count]


def _intersection_cache_nominal_integer(values: np.ndarray) -> int | None:
    """Return the nearest integer Bragg index represented by one cache column."""

    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return None
    return int(np.rint(float(np.mean(finite))))


def _intersection_cache_group_key(cache_table: np.ndarray) -> tuple[object, ...] | None:
    """Return the nominal Bragg-family key for one cache table."""

    table = np.asarray(cache_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] <= 0 or table.shape[1] < 9:
        return None

    l_key = _intersection_cache_nominal_integer(table[:, 8])
    if l_key is None:
        return None

    if _intersection_cache_target_row_count(table) <= 1:
        return ("specular", l_key)

    qr_vals = np.asarray(table[:, 0], dtype=np.float64).reshape(-1)
    finite_qr = qr_vals[np.isfinite(qr_vals)]
    if finite_qr.size > 0:
        return ("non_specular", round(float(np.mean(finite_qr)), 8), l_key)

    h_key = _intersection_cache_nominal_integer(table[:, 6])
    k_key = _intersection_cache_nominal_integer(table[:, 7])
    return ("non_specular_hkl", h_key, k_key, l_key)


def _expand_intersection_cache_group(cache_tables: list[np.ndarray]) -> list[np.ndarray]:
    """Reduce one nominal Bragg family to final one-row cache tables."""

    expanded_tables, _metadata = _expand_intersection_cache_group_with_metadata(cache_tables)
    return expanded_tables


def _intersection_cache_nominal_hkl_triplet(table: np.ndarray) -> tuple[int, int, int] | None:
    """Return the nominal integer HKL triplet represented by one cache table."""

    h_key = _intersection_cache_nominal_integer(table[:, 6])
    k_key = _intersection_cache_nominal_integer(table[:, 7])
    l_key = _intersection_cache_nominal_integer(table[:, 8])
    if h_key is None or k_key is None or l_key is None:
        return None
    return int(h_key), int(k_key), int(l_key)


def _intersection_cache_summary_q_value(values: np.ndarray) -> float | None:
    """Return one representative finite cache Q value when present."""

    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return None
    return float(np.mean(finite))


def _expand_intersection_cache_group_with_metadata(
    cache_tables: list[np.ndarray],
    *,
    beam_x_array: np.ndarray | None = None,
    beam_y_array: np.ndarray | None = None,
    theta_array: np.ndarray | None = None,
    phi_array: np.ndarray | None = None,
    wavelength_array: np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Reduce one nominal Bragg family and return cache-table provenance metadata."""

    valid_tables = []
    for cache_table in cache_tables:
        table = np.asarray(cache_table, dtype=np.float64)
        if table.ndim != 2 or table.shape[0] <= 0:
            continue
        valid_tables.append(table)
    if not valid_tables:
        return [], {}

    combined_table = (
        np.vstack(valid_tables)
        if len(valid_tables) > 1
        else np.asarray(valid_tables[0], dtype=np.float64)
    )
    selected_indices = _intersection_cache_selected_row_indices(
        combined_table,
        beam_x_array=beam_x_array,
        beam_y_array=beam_y_array,
        theta_array=theta_array,
        phi_array=phi_array,
        wavelength_array=wavelength_array,
    )
    collapsed_table = (
        combined_table[np.asarray(selected_indices, dtype=np.int64), :].copy()
        if selected_indices
        else np.empty((0, combined_table.shape[1]), dtype=np.float64)
    )
    if collapsed_table.ndim != 2 or collapsed_table.shape[0] <= 0:
        return [], {}

    group_key = _intersection_cache_group_key(combined_table)
    group_kind = str(group_key[0]) if isinstance(group_key, tuple) and group_key else "unknown"
    final_tables = [
        np.asarray(collapsed_table[row_idx : row_idx + 1, :], dtype=np.float64).copy()
        for row_idx in range(collapsed_table.shape[0])
    ]
    selected_index_set = {int(idx) for idx in selected_indices}
    coords = np.asarray(combined_table[:, 2:4], dtype=np.float64)
    finite_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    nonfinite_indices = [
        int(idx) for idx in np.flatnonzero(~finite_mask) if int(idx) not in selected_index_set
    ]
    row_count_before = int(combined_table.shape[0])
    row_count_after = int(len(selected_indices))
    merged_group_count = max(0, int(len(valid_tables)) - 1)
    nominal_hkl_recovery_count = int(row_count_before) if group_kind == "non_specular_hkl" else 0
    table_summaries = []
    nominal_hkl = _intersection_cache_nominal_hkl_triplet(combined_table)
    group_qr = _intersection_cache_summary_q_value(combined_table[:, 0])
    group_qz = _intersection_cache_summary_q_value(combined_table[:, 1])
    for summary_index, row_idx in enumerate(selected_indices):
        row_table = combined_table[int(row_idx) : int(row_idx) + 1, :]
        row_hkl = _intersection_cache_nominal_hkl_triplet(row_table) or nominal_hkl
        table_summaries.append(
            {
                "table_index": int(summary_index),
                "group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
                "q_group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
                "nominal_hkl": list(row_hkl) if isinstance(row_hkl, tuple) else None,
                "qr": _intersection_cache_summary_q_value(row_table[:, 0]),
                "qz": _intersection_cache_summary_q_value(row_table[:, 1]),
                "group_qr": group_qr,
                "group_qz": group_qz,
                "row_count_before_grouping": int(row_count_before),
                "row_count_after_grouping": 1,
                "dropped_nonfinite_row_count": int(len(nonfinite_indices)),
                "nominal_hkl_recovery_count": int(nominal_hkl_recovery_count),
                "merged_group_count": int(merged_group_count),
                "representative_row_indices_kept": [int(row_idx)],
            }
        )
    metadata = {
        "group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
        "q_group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
        "group_kind": group_kind,
        "nominal_hkl": list(nominal_hkl) if isinstance(nominal_hkl, tuple) else None,
        "qr": group_qr,
        "qz": group_qz,
        "row_count_before_grouping": int(row_count_before),
        "row_count_after_grouping": int(row_count_after),
        "dropped_nonfinite_row_count": int(len(nonfinite_indices)),
        "nominal_hkl_recovery_count": int(nominal_hkl_recovery_count),
        "merged_group_count": int(merged_group_count),
        "representative_row_indices_kept": [int(idx) for idx in selected_indices],
        "table_summaries": table_summaries,
    }
    return final_tables, metadata


def _sample_context_index_valid(
    sample_idx: int,
    beam_x_array: np.ndarray | None,
    beam_y_array: np.ndarray | None,
    theta_array: np.ndarray | None,
    phi_array: np.ndarray | None,
    wavelength_array: np.ndarray | None,
) -> bool:
    """Return whether all per-sample context arrays contain ``sample_idx``."""

    if sample_idx < 0:
        return False
    arrays = (beam_x_array, beam_y_array, theta_array, phi_array, wavelength_array)
    for arr in arrays:
        if arr is None or sample_idx >= arr.shape[0]:
            return False
    return True


def _hit_table_row_sample_indices(hits_arr: np.ndarray) -> np.ndarray:
    """Return per-hit beam sample indices carried by provenance-width hit tables."""

    n_rows = int(hits_arr.shape[0]) if hits_arr.ndim == 2 else 0
    row_sample_indices = np.full(n_rows, -1, dtype=np.int64)
    if hits_arr.ndim != 2 or hits_arr.shape[1] <= HIT_ROW_COL_BEST_SAMPLE_INDEX:
        return row_sample_indices
    raw_values = np.asarray(hits_arr[:, HIT_ROW_COL_BEST_SAMPLE_INDEX], dtype=np.float64)
    finite_mask = np.isfinite(raw_values)
    row_sample_indices[finite_mask] = np.rint(raw_values[finite_mask]).astype(np.int64)
    row_sample_indices[row_sample_indices < 0] = -1
    return row_sample_indices


def build_intersection_cache(
    hit_tables,
    av,
    cv,
    beam_x_array=None,
    beam_y_array=None,
    theta_array=None,
    phi_array=None,
    wavelength_array=None,
    best_sample_indices_out=None,
):
    """Convert hit tables into a per-Bragg-peak detector intersection cache.

    Current detector-cache rows use 17 columns:
    ``[Qr, Qz, detector_col, detector_row, intensity, phi, H, K, L,
    beam_x_offset, beam_z_offset, divergence_x_offset, divergence_z_offset,
    wavelength_offset, source_table_index, source_row_index, best_sample_index]``.
    Replay helpers also accept legacy 14-column detector caches plus legacy/current
    caked layouts with widths 16 and 19, but this builder remains the authoritative
    producer for the 17-column detector schema. Weighted-event hit tables keep
    all sampled rows, including duplicates, so this builder preserves all rows
    after nominal family grouping and only normalizes them into one-row cache
    tables.
    """

    if hit_tables is None:
        return []

    beam_x_arr = None if beam_x_array is None else np.asarray(beam_x_array, dtype=np.float64)
    beam_y_arr = None if beam_y_array is None else np.asarray(beam_y_array, dtype=np.float64)
    theta_arr = None if theta_array is None else np.asarray(theta_array, dtype=np.float64)
    phi_arr = None if phi_array is None else np.asarray(phi_array, dtype=np.float64)
    wavelength_arr = (
        None if wavelength_array is None else np.asarray(wavelength_array, dtype=np.float64)
    )
    best_sample_arr = (
        None
        if best_sample_indices_out is None
        else np.asarray(best_sample_indices_out, dtype=np.int64).reshape(-1)
    )

    av_val = float(av)
    cv_val = float(cv)
    qr_scale = np.nan
    qz_scale = np.nan
    if np.isfinite(av_val) and abs(av_val) > 1.0e-12:
        qr_scale = 4.0 * np.pi / av_val
    if np.isfinite(cv_val) and abs(cv_val) > 1.0e-12:
        qz_scale = 2.0 * np.pi / cv_val

    grouped_cache_tables: dict[tuple[object, ...], list[np.ndarray]] = {}
    cache_group_summaries: list[dict[str, object]] = []
    cache_table_summaries: list[dict[str, object]] = []
    dropped_missing_best_sample_tables: list[int] = []
    beam_x_center = float("nan")
    beam_y_center = float("nan")
    theta_center = float("nan")
    phi_center = float("nan")
    wavelength_center = float("nan")
    if beam_x_arr is not None and beam_x_arr.size > 0:
        beam_x_center = float(np.mean(beam_x_arr))
    if beam_y_arr is not None and beam_y_arr.size > 0:
        beam_y_center = float(np.mean(beam_y_arr))
    if theta_arr is not None and theta_arr.size > 0:
        theta_center = float(np.mean(theta_arr))
    if phi_arr is not None and phi_arr.size > 0:
        phi_center = float(np.mean(phi_arr))
    if wavelength_arr is not None and wavelength_arr.size > 0:
        wavelength_center = float(np.mean(wavelength_arr))
    has_beam_context_arrays = (
        beam_x_arr is not None
        and beam_y_arr is not None
        and theta_arr is not None
        and phi_arr is not None
        and wavelength_arr is not None
    )

    for table_idx, hits in enumerate(hit_tables):
        hits_arr = np.asarray(hits, dtype=np.float64)
        if hits_arr.ndim != 2 or hits_arr.shape[1] < 7 or hits_arr.shape[0] == 0:
            continue

        h_vals = hits_arr[:, 4]
        k_vals = hits_arr[:, 5]
        l_vals = hits_arr[:, 6]
        qr_vals = qr_scale * np.sqrt(
            np.clip((h_vals * h_vals + h_vals * k_vals + k_vals * k_vals) / 3.0, 0.0, None)
        )
        qz_vals = qz_scale * l_vals

        sample_idx = -1
        if best_sample_arr is not None and table_idx < best_sample_arr.shape[0]:
            sample_idx = int(best_sample_arr[table_idx])
        row_sample_indices = _hit_table_row_sample_indices(hits_arr)

        if has_beam_context_arrays:
            has_valid_sample_idx = _sample_context_index_valid(
                sample_idx,
                beam_x_arr,
                beam_y_arr,
                theta_arr,
                phi_arr,
                wavelength_arr,
            )
            has_valid_row_sample_idx = any(
                _sample_context_index_valid(
                    int(row_sample_idx),
                    beam_x_arr,
                    beam_y_arr,
                    theta_arr,
                    phi_arr,
                    wavelength_arr,
                )
                for row_sample_idx in row_sample_indices
            )
            if not (has_valid_sample_idx or has_valid_row_sample_idx):
                dropped_missing_best_sample_tables.append(int(table_idx))
                continue

        n_rows = hits_arr.shape[0]
        cache_table = np.empty((n_rows, CURRENT_DETECTOR_CACHE_WIDTH), dtype=np.float64)
        cache_table[:, 0] = qr_vals
        cache_table[:, 1] = qz_vals
        cache_table[:, 2] = hits_arr[:, 1]
        cache_table[:, 3] = hits_arr[:, 2]
        cache_table[:, 4] = hits_arr[:, 0]
        cache_table[:, 5] = hits_arr[:, 3]
        cache_table[:, 6:9] = hits_arr[:, 4:7]
        cache_table[:, 9:14] = np.nan
        cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX] = np.nan
        cache_table[:, CACHE_COL_SOURCE_TABLE_INDEX] = np.nan
        cache_table[:, CACHE_COL_SOURCE_ROW_INDEX] = np.nan
        if hits_arr.shape[1] > HIT_ROW_COL_SOURCE_TABLE_INDEX:
            raw_source_table = np.asarray(hits_arr[:, HIT_ROW_COL_SOURCE_TABLE_INDEX], dtype=np.float64)
            finite_source_table = np.isfinite(raw_source_table)
            cache_table[finite_source_table, CACHE_COL_SOURCE_TABLE_INDEX] = raw_source_table[
                finite_source_table
            ]
        if hits_arr.shape[1] > HIT_ROW_COL_SOURCE_ROW_INDEX:
            raw_source_row = np.asarray(hits_arr[:, HIT_ROW_COL_SOURCE_ROW_INDEX], dtype=np.float64)
            finite_source_row = np.isfinite(raw_source_row)
            cache_table[finite_source_row, CACHE_COL_SOURCE_ROW_INDEX] = raw_source_row[
                finite_source_row
            ]
        if has_beam_context_arrays:
            for row_idx in range(n_rows):
                row_sample_idx = int(row_sample_indices[row_idx])
                sample_for_row = row_sample_idx if row_sample_idx >= 0 else sample_idx
                if row_sample_idx >= 0:
                    cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX] = float(row_sample_idx)
                if _sample_context_index_valid(
                    sample_for_row,
                    beam_x_arr,
                    beam_y_arr,
                    theta_arr,
                    phi_arr,
                    wavelength_arr,
                ):
                    cache_table[row_idx, 9] = beam_x_arr[sample_for_row] - beam_x_center
                    cache_table[row_idx, 10] = beam_y_arr[sample_for_row] - beam_y_center
                    cache_table[row_idx, 11] = theta_arr[sample_for_row] - theta_center
                    cache_table[row_idx, 12] = phi_arr[sample_for_row] - phi_center
                    cache_table[row_idx, 13] = wavelength_arr[sample_for_row] - wavelength_center
                    if not np.isfinite(cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX]):
                        cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX] = float(sample_for_row)
        else:
            valid_row_samples = row_sample_indices >= 0
            missing_best_sample = ~np.isfinite(cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX])
            write_row_samples = valid_row_samples & missing_best_sample
            cache_table[write_row_samples, CACHE_COL_BEST_SAMPLE_INDEX] = row_sample_indices[
                write_row_samples
            ].astype(np.float64)
            if sample_idx >= 0:
                fallback_best_sample = ~valid_row_samples & ~np.isfinite(
                    cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX]
                )
                cache_table[fallback_best_sample, CACHE_COL_BEST_SAMPLE_INDEX] = float(sample_idx)
        missing_source_table = ~np.isfinite(cache_table[:, CACHE_COL_SOURCE_TABLE_INDEX])
        cache_table[missing_source_table, CACHE_COL_SOURCE_TABLE_INDEX] = float(table_idx)
        missing_source_row = ~np.isfinite(cache_table[:, CACHE_COL_SOURCE_ROW_INDEX])
        cache_table[missing_source_row, CACHE_COL_SOURCE_ROW_INDEX] = np.arange(
            n_rows,
            dtype=np.float64,
        )[missing_source_row]

        group_key = _intersection_cache_group_key(cache_table)
        if group_key is None:
            group_key = ("ungrouped", int(table_idx))
        grouped_cache_tables.setdefault(group_key, []).append(cache_table)

    cache = []
    cache_table_summaries = []
    for group_tables in grouped_cache_tables.values():
        valid_group_tables = []
        for cache_table in group_tables:
            table = np.asarray(cache_table, dtype=np.float64)
            if table.ndim != 2 or table.shape[0] <= 0:
                continue
            valid_group_tables.append(table)
        if not valid_group_tables:
            continue
        combined_table = (
            np.vstack(valid_group_tables)
            if len(valid_group_tables) > 1
            else np.asarray(valid_group_tables[0], dtype=np.float64)
        )
        expanded_tables = [
            np.asarray(combined_table[row_idx : row_idx + 1, :], dtype=np.float64).copy()
            for row_idx in range(combined_table.shape[0])
        ]
        group_key = _intersection_cache_group_key(combined_table)
        group_metadata = {
            "group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
            "q_group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
            "row_count_before_grouping": int(combined_table.shape[0]),
            "row_count_after_grouping": int(combined_table.shape[0]),
            "representative_row_indices_kept": list(range(int(combined_table.shape[0]))),
            "table_summaries": [
                {
                    "table_index": int(row_idx),
                    "group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
                    "q_group_key": list(group_key) if isinstance(group_key, tuple) else group_key,
                    "row_count_before_grouping": int(combined_table.shape[0]),
                    "row_count_after_grouping": 1,
                    "representative_row_indices_kept": [int(row_idx)],
                }
                for row_idx in range(combined_table.shape[0])
            ],
        }
        cache.extend(expanded_tables)
        if isinstance(group_metadata, dict) and group_metadata:
            cache_group_summaries.append(group_metadata)
            for item in group_metadata.get("table_summaries", []) or []:
                if isinstance(item, dict):
                    cache_table_summaries.append(dict(item))

    if _should_log_intersection_cache():
        _write_intersection_cache_log(
            cache,
            av=av,
            cv=cv,
            beam_x_center=beam_x_center,
            beam_y_center=beam_y_center,
            theta_center=theta_center,
            phi_center=phi_center,
            wavelength_center=wavelength_center,
            cache_metadata={
                "dropped_missing_best_sample_table_count": int(
                    len(dropped_missing_best_sample_tables)
                ),
                "dropped_missing_best_sample_tables": dropped_missing_best_sample_tables,
                "group_summary_count": int(len(cache_group_summaries)),
                "group_summaries": cache_group_summaries,
                "table_summaries": cache_table_summaries,
            },
        )

    return cache


def build_branch_representative_intersection_cache(
    hit_tables,
    av,
    cv,
    beam_x_array=None,
    beam_y_array=None,
    theta_array=None,
    phi_array=None,
    wavelength_array=None,
    best_sample_indices_out=None,
    group_by_qr_set=False,
):
    """Build representative cache rows without recollapsing preselected inputs."""

    del group_by_qr_set
    return build_intersection_cache(
        hit_tables,
        av,
        cv,
        beam_x_array=beam_x_array,
        beam_y_array=beam_y_array,
        theta_array=theta_array,
        phi_array=phi_array,
        wavelength_array=wavelength_array,
        best_sample_indices_out=best_sample_indices_out,
    )


def _extract_process_peaks_context(args, kwargs):
    """Return lattice and sample-context arrays from one process-peaks call."""

    if len(args) >= 5:
        av = args[3]
        cv = args[4]
    else:
        av = kwargs.get("av", np.nan)
        cv = kwargs.get("cv", np.nan)

    if len(args) >= 20:
        beam_x_array = args[16]
        beam_y_array = args[17]
        theta_array = args[18]
        phi_array = args[19]
    else:
        beam_x_array = kwargs.get("beam_x_array")
        beam_y_array = kwargs.get("beam_y_array")
        theta_array = kwargs.get("theta_array")
        phi_array = kwargs.get("phi_array")

    wavelength_array = kwargs.get("wavelength_array", None)
    if len(args) >= 24:
        wavelength_array = args[23]

    best_sample_indices_out = kwargs.get("best_sample_indices_out", None)
    return (
        av,
        cv,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        wavelength_array,
        best_sample_indices_out,
    )


def process_qr_rods_parallel(
    qr_dict,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=0.0,
    optics_mode=OPTICS_MODE_FAST,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    sample_qr_ring_once=True,
    enable_safe_cache=None,
    prefer_python_runner=False,
    _safe_stats_out=None,
    exit_projection_mode=EXIT_PROJECTION_INTERNAL,
    projection_debug_counters=None,
    projection_debug_reject_counts=None,
    projection_debug_reject_records=None,
    projection_debug_row_hit_counts=None,
    projection_debug_row_tthp_sums=None,
    projection_debug_row_tth_sums=None,
    events_per_beam_phase=50,
):
    """Wrapper to process Hendricks–Teller rods instead of individual reflections.

    The Hendricks–Teller preprocessing groups symmetry-related in-plane peaks
    into ``Qr`` rods and records how many peaks contributed to each rod in the
    ``deg`` field.  ``qr_dict_to_arrays`` already returns rod intensities as the
    total summed intensity over the grouped HK pairs, so we forward that array
    unchanged to avoid double-counting.  The degeneracy array is returned so
    downstream code can still track how many symmetry-equivalent HK pairs
    contributed to each rod.
    """
    from ra_sim.utils.stacking_fault import qr_dict_to_arrays

    miller, intensities, degeneracy, _ = qr_dict_to_arrays(qr_dict)

    result = process_peaks_parallel_safe(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        record_status,
        thickness,
        optics_mode,
        solve_q_steps,
        solve_q_rel_tol,
        solve_q_mode,
        best_sample_indices_out=best_sample_indices_out,
        collect_hit_tables=collect_hit_tables,
        pixel_size_m=pixel_size_m,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
        n2_sample_array_override=n2_sample_array_override,
        accumulate_image=accumulate_image,
        sample_qr_ring_once=sample_qr_ring_once,
        enable_safe_cache=enable_safe_cache,
        prefer_python_runner=prefer_python_runner,
        _safe_stats_out=_safe_stats_out,
        exit_projection_mode=exit_projection_mode,
        projection_debug_counters=projection_debug_counters,
        projection_debug_reject_counts=projection_debug_reject_counts,
        projection_debug_reject_records=projection_debug_reject_records,
        projection_debug_row_hit_counts=projection_debug_row_hit_counts,
        projection_debug_row_tthp_sums=projection_debug_row_tthp_sums,
        projection_debug_row_tth_sums=projection_debug_row_tth_sums,
        events_per_beam_phase=events_per_beam_phase,
    )

    return (*result, degeneracy)


def process_qr_rods_parallel_safe(*args, **kwargs):
    """Run ``process_qr_rods_parallel`` with Python fallback if needed."""

    kwargs = dict(kwargs)
    kwargs["events_per_beam_phase"] = normalize_events_per_beam_phase_backend(
        kwargs.get("events_per_beam_phase", 50)
    )
    prefer_python_runner = bool(kwargs.get("prefer_python_runner", False))
    py_runner = getattr(process_qr_rods_parallel, "py_func", None)
    runners = []
    if prefer_python_runner and callable(py_runner):
        runners.append(py_runner)
    runners.append(process_qr_rods_parallel)
    if (not prefer_python_runner) and callable(py_runner):
        runners.append(py_runner)

    last_exc = None
    for runner in runners:
        try:
            return runner(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("process_qr_rods_parallel_safe could not resolve a runner")


def debug_detector_paths(
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    theta_initial_deg,
    cor_angle_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zb,
    zs,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    n_detector=np.array([0.0, 1.0, 0.0]),
    unit_x=np.array([1.0, 0.0, 0.0]),
):
    """Trace specular reflection paths for debugging.

    For each beam sample, this returns whether the incoming ray intersects the
    sample plane and whether its specular reflection would intersect the
    detector plane.  The function is purely for debugging geometry issues and is
    not JIT compiled.

    Parameters
    ----------
    beam_x_array, beam_y_array, theta_array, phi_array : array-like
        Sampled beam parameters as used in :func:`process_peaks_parallel`.
    theta_initial_deg : float
        Sample tilt around the CoR axis.
    cor_angle_deg : float
        Angle of the CoR axis relative to the +x axis.
    chi_deg, psi_deg : float
        Additional sample rotations around y and z.
    psi_z_deg : float
        Yaw of the CoR/goniometer axis about laboratory z.
    zb, zs : float
        Beam and sample offsets used in the main simulation.
    Distance_CoR_to_Detector, gamma_deg, Gamma_deg : float
        Detector geometry parameters.

    Returns
    -------
    out : ndarray of shape (N,4)
        Columns contain (theta, phi, hit_sample, hit_detector).
    """
    gamma_rad = np.radians(gamma_deg)
    Gamma_rad = np.radians(Gamma_deg)
    chi_rad = np.radians(chi_deg)
    psi_rad = np.radians(psi_deg)
    psi_z_rad = np.radians(psi_z_deg)
    rad_theta_i = np.radians(theta_initial_deg)
    cor_axis_rad = np.radians(cor_angle_deg)

    cg = np.cos(gamma_rad)
    sg = np.sin(gamma_rad)
    cG = np.cos(Gamma_rad)
    sG = np.sin(Gamma_rad)
    R_x_det = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    R_z_det = np.array([[cG, sG, 0.0], [-sG, cG, 0.0], [0.0, 0.0, 1.0]])
    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    n_det_rot /= np.linalg.norm(n_det_rot)
    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])

    c_chi = np.cos(chi_rad)
    s_chi = np.sin(chi_rad)
    R_y = np.array([[c_chi, 0.0, s_chi], [0.0, 1.0, 0.0], [-s_chi, 0.0, c_chi]])
    c_psi = np.cos(psi_rad)
    s_psi = np.sin(psi_rad)
    R_z = np.array([[c_psi, s_psi, 0.0], [-s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]])
    R_z_R_y = R_z @ R_y

    # Construct the pitched CoR axis in x–z and rotate with Rodrigues' formula;
    # see docs/cor_rotation_math.md for the math details.
    ax = np.cos(cor_axis_rad)
    ay = 0.0
    az = np.sin(cor_axis_rad)
    c_axis_yaw = np.cos(psi_z_rad)
    s_axis_yaw = np.sin(psi_z_rad)
    ax_yawed = c_axis_yaw * ax + s_axis_yaw * ay
    ay_yawed = -s_axis_yaw * ax + c_axis_yaw * ay
    ax = ax_yawed
    ay = ay_yawed
    axis_norm = np.sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12:
        axis_norm = 1.0
    ax /= axis_norm
    ay /= axis_norm
    az /= axis_norm

    ct = np.cos(rad_theta_i)
    st = np.sin(rad_theta_i)
    one_ct = 1.0 - ct
    R_cor = np.array(
        [
            [ct + ax * ax * one_ct, ax * ay * one_ct - az * st, ax * az * one_ct + ay * st],
            [ay * ax * one_ct + az * st, ct + ay * ay * one_ct, ay * az * one_ct - ax * st],
            [az * ax * one_ct - ay * st, az * ay * one_ct + ax * st, ct + az * az * one_ct],
        ]
    )
    R_sample = R_cor @ R_z_R_y

    n_surf = R_cor @ (R_z_R_y @ np.array([0.0, 0.0, 1.0]))
    n_surf /= np.linalg.norm(n_surf)

    P0 = np.array([0.0, 0.0, -zs])
    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0

    N = len(theta_array)
    out = np.zeros((N, 4), dtype=np.float64)

    for i in range(N):
        dtheta = theta_array[i]
        dphi = phi_array[i]
        bx = beam_x_array[i]
        by = beam_y_array[i]

        k_in = np.array(
            [np.cos(dtheta) * np.sin(dphi), np.cos(dtheta) * np.cos(dphi), np.sin(dtheta)]
        )

        beam_start = np.array([bx, -20e-3, -zb + by])
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)

        hit_sample = 1.0 if valid_int else 0.0
        if not valid_int:
            out[i] = [dtheta, dphi, 0.0, 0.0]
            continue

        I_plane = np.array([ix, iy, iz])
        dot_kn = k_in[0] * n_surf[0] + k_in[1] * n_surf[1] + k_in[2] * n_surf[2]
        k_spec = k_in - 2.0 * dot_kn * n_surf

        dx, dy, dz, valid_det = intersect_line_plane(I_plane, k_spec, Detector_Pos, n_det_rot)
        hit_det = 1.0 if valid_det else 0.0

        out[i] = [dtheta, dphi, hit_sample, hit_det]

    return out
