# Generated from all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.ipynb.
# Edit this script for the headless diagnostic workflow; do not rely on notebook-only fixes.
from __future__ import annotations

# Parameters. Leave blank to use environment variables and repository defaults.
GUI_STATE_PATH = ""
OUTPUT_DIR = ""
FIGURE_OUTPUT_DIR = ""
RUN_NAME = ""
SAMPLE_NAME_OVERRIDE = ""

# 32-core workstation defaults. Fit parallelism is now a global queue across all backgrounds.
FIT_WORKERS_OVERRIDE = "28"
NUMBA_WORKERS_OVERRIDE = "24"
FIT_BACKEND_OVERRIDE = ""
PROCESS_NUMBA_THREADS_OVERRIDE = ""
CAKE_WORKERS_OVERRIDE = "24"
PROFILE_FIT_WORKERS_OVERRIDE = "16"
ROD_PROFILE_FIT_WORKERS_OVERRIDE = "16"
USE_GPU_OVERRIDE = "1"
QR_ROD_PEAK_EDIT_MODE_OVERRIDE = ""
DETECTOR_LABEL_SETTINGS_PATH_OVERRIDE = ""
RESET_PRE_EDITOR_CACHE_OVERRIDE = ""
PBI2_DISABLE_BACKGROUND_SUBTRACTION_OVERRIDE = ""

# Faster development output. Use 300 / 1 for final vector figures.
FIGURE_DPI_OVERRIDE = "200"
SAVE_VECTOR_FIGURES_OVERRIDE = "0"
ROD_PROFILE_TILT_OVERRIDE = ""


import json
import hashlib
import math
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

# Keep BLAS/OpenMP libraries from oversubscribing when Python-level workers are used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd
from IPython.display import Image, display
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares, nnls

try:
    from numba import njit, prange, set_num_threads, get_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # Numba is optional; the notebook still runs without JIT acceleration.
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[override]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range  # type: ignore[assignment]

    def set_num_threads(_n: int) -> None:  # type: ignore[override]
        return None

    def get_num_threads() -> int:  # type: ignore[override]
        return 1


from ra_sim.fitting.rod_profiles import caked_field_to_gui_phi, qz_profile_from_caked_mask
from ra_sim.gui.background import apply_background_backend_orientation
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import qr_cylinder_overlay as gui_qr_cylinder_overlay
from ra_sim.io.osc_reader import read_osc
from ra_sim.utils.calculations import IndexofRefraction
from ra_sim.simulation.exact_cake_portable import (
    FastAzimuthalIntegrator,
    build_cake_transform_bundle,
    caked_point_to_detector_pixel,
    detector_pixel_angular_maps,
    detector_two_theta_max_deg,
    prepare_gui_phi_display,
    raw_phi_to_gui_phi,
)

DEFAULT_STATE_PATH = Path.home() / ".local" / "share" / "ra_sim" / "PbI2.json"


def _setting_text(local_name: str, env_name: str, default: object = "") -> str:
    local_value = globals().get(local_name, "")
    if local_value is not None and str(local_value).strip():
        return str(local_value).strip()
    return str(os.environ.get(env_name, default)).strip()


def _safe_run_name(value: object) -> str:
    raw = str(value).strip().replace("\\", "/")
    text = Path(raw).stem if raw else "state"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or "state"


QR_ROD_PROFILE_CACHE_SCHEMA = "ra_sim.qr_rod_profile_cache.v1"
QR_ROD_FINAL_FIT_CACHE_SIGNATURE = "joint_qz_labeled_marker_fit_specular_theta_i0_l8_v8"
PRE_EDITOR_CACHE_SCHEMA = "ra_sim.background_pre_editor_cache.v1"
PRE_EDITOR_CACHE_SIGNATURE = "pre_qr_rod_marker_editor_inputs_v1"
PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE = "background_peak_fit_results_v1"
PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE = "profile_fit_cache_v1"
PRE_EDITOR_QR_ROD_STAGE_SIGNATURE = "qr_rod_pre_marker_profiles_specular_theta_i0_l8_v4"


def _cache_normalize_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(float(value), 12)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, np.generic):
        return _cache_normalize_value(value.item())
    if isinstance(value, np.ndarray):
        return [_cache_normalize_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _cache_normalize_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if isinstance(value, set):
            items = sorted(items, key=lambda item: str(item))
        return [_cache_normalize_value(item) for item in items]
    return str(value)


def pre_editor_cache_path(output_dir: Path | str, state_path: Path | str) -> Path:
    return (
        Path(output_dir).expanduser()
        / f"{_safe_run_name(Path(str(state_path)).stem)}_pre_qr_rod_marker_editor_cache.pkl"
    )


def pre_editor_cache_key(
    state_path: Path | str, *, input_signature: dict[str, object] | None = None
) -> dict[str, object]:
    normalized_inputs = _cache_normalize_value(input_signature or {})
    data = json.dumps(normalized_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "state_name": Path(state_path).expanduser().name,
        "signature": PRE_EDITOR_CACHE_SIGNATURE,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "inputs": normalized_inputs,
    }


def load_pre_editor_cache(
    cache_path: Path | str, cache_key: dict[str, object]
) -> dict[str, object] | None:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        print(f"ignored unreadable pre-editor diagnostic cache={path}: {exc}")
        return None
    if not isinstance(envelope, dict):
        print(f"ignored incompatible pre-editor diagnostic cache={path}: expected mapping")
        return None
    if envelope.get("schema") != PRE_EDITOR_CACHE_SCHEMA:
        print(
            f"ignored incompatible pre-editor diagnostic cache={path}: schema={envelope.get('schema')!r}"
        )
        return None
    if envelope.get("cache_key") != cache_key:
        print(f"ignored stale pre-editor diagnostic cache={path}: input signature changed")
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        print(f"ignored incompatible pre-editor diagnostic cache={path}: payload missing")
        return None
    return payload


def write_pre_editor_cache(
    cache_path: Path | str,
    cache_key: dict[str, object],
    payload: dict[str, object],
) -> None:
    path = Path(cache_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": PRE_EDITOR_CACHE_SCHEMA,
        "cache_key": dict(cache_key),
        "payload": dict(payload),
    }
    with path.open("wb") as handle:
        pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)


def reset_pre_editor_cache(cache_path: Path | str) -> bool:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return False
    path.unlink()
    return True


def pre_editor_cache_get_stage(
    cache_payload: dict[str, object] | None, stage_name: str, stage_signature: str
) -> dict[str, object] | None:
    if not isinstance(cache_payload, dict):
        return None
    stages = cache_payload.get("stages")
    if not isinstance(stages, dict):
        return None
    stage = stages.get(stage_name)
    if not isinstance(stage, dict) or stage.get("stage_signature") != stage_signature:
        return None
    payload = stage.get("payload")
    return payload if isinstance(payload, dict) else None


def pre_editor_cache_with_stage(
    cache_payload: dict[str, object] | None,
    stage_name: str,
    stage_signature: str,
    stage_payload: dict[str, object],
) -> dict[str, object]:
    next_payload = dict(cache_payload or {})
    stages = dict(next_payload.get("stages") or {})
    stages[str(stage_name)] = {
        "stage_signature": str(stage_signature),
        "payload": dict(stage_payload),
    }
    next_payload["stages"] = stages
    return next_payload


def background_peak_fit_stage_is_valid(
    stage_payload: dict[str, object] | None, *, expected_fit_count: int
) -> bool:
    if not isinstance(stage_payload, dict):
        return False
    try:
        if int(stage_payload.get("fit_job_count", -1)) != int(expected_fit_count):
            return False
    except Exception:
        return False
    fit_results_by_bg = stage_payload.get("fit_results_by_bg")
    if not isinstance(fit_results_by_bg, dict) or not isinstance(
        stage_payload.get("fit_failures_by_bg"), dict
    ):
        return False
    try:
        result_slots = sum(len(list(items)) for items in fit_results_by_bg.values())
    except Exception:
        return False
    return int(result_slots) == int(expected_fit_count)


def profile_fit_stage_is_valid(
    stage_payload: dict[str, object] | None, *, expected_profile_count: int
) -> bool:
    if not isinstance(stage_payload, dict):
        return False
    records = stage_payload.get("profile_fit_records")
    try:
        if int(stage_payload.get("profile_target_count", -1)) != int(expected_profile_count):
            return False
    except Exception:
        return False
    return (
        isinstance(records, list)
        and len(records) == int(expected_profile_count)
        and all(isinstance(record, dict) for record in records)
    )


def qr_rod_pre_editor_stage_is_valid(stage_payload: dict[str, object] | None) -> bool:
    if not isinstance(stage_payload, dict):
        return False
    required = {
        "rod_profile_table",
        "marker_table",
        "region_overlays",
        "rod_entries",
        "rod_qspace_calibration",
        "rod_profile_max_two_theta_deg",
    }
    if not required.issubset(stage_payload):
        return False
    rod_profile_table = pd.DataFrame(stage_payload["rod_profile_table"])
    marker_table = pd.DataFrame(stage_payload["marker_table"])
    return (
        not rod_profile_table.empty
        and "qz_center" in rod_profile_table
        and "background_density" in rod_profile_table
        and "qz_marker" in marker_table
    )


def pre_editor_fit_job_signature(
    fit_jobs: list[tuple[int, int, dict[str, object]]],
) -> list[dict[str, object]]:
    rows = []
    for bg_idx, local_idx, entry in fit_jobs:
        rows.append(
            {
                "background_index": int(bg_idx),
                "local_index": int(local_idx),
                "label": str(entry.get("_label", entry.get("label", ""))),
                "branch": str(entry.get("_branch", entry.get("branch", ""))),
                "theta_seed_deg": _cache_normalize_value(entry.get("_theta_seed_deg")),
                "phi_seed_deg": _cache_normalize_value(entry.get("_phi_seed_deg")),
                "q_group_key": _cache_normalize_value(entry.get("q_group_key")),
                "branch_id": str(entry.get("branch_id", "")),
                "source_branch_index": _cache_normalize_value(entry.get("source_branch_index")),
            }
        )
    return rows


def qr_rod_profile_cache_path(output_dir: Path | str, state_path: Path | str) -> Path:
    return (
        Path(output_dir).expanduser()
        / f"{_safe_run_name(Path(str(state_path)).stem)}_qr_rod_profile_cache.pkl"
    )


def qr_rod_profile_cache_key(state_path: Path | str) -> dict[str, object]:
    return {"state_name": Path(state_path).expanduser().name}


def load_qr_rod_profile_cache(
    cache_path: Path | str, state_path: Path | str
) -> dict[str, object] | None:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        print(f"ignored unreadable Qr-rod profile cache={path}: {exc}")
        return None
    if not isinstance(envelope, dict):
        print(f"ignored incompatible Qr-rod profile cache={path}: expected mapping")
        return None
    if envelope.get("schema") != QR_ROD_PROFILE_CACHE_SCHEMA:
        print(
            f"ignored incompatible Qr-rod profile cache={path}: schema={envelope.get('schema')!r}"
        )
        return None
    if envelope.get("state_cache_key") != qr_rod_profile_cache_key(state_path):
        print(f"ignored stale Qr-rod profile cache={path}: state filename changed")
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        print(f"ignored incompatible Qr-rod profile cache={path}: payload missing")
        return None
    return payload


def write_qr_rod_profile_cache(
    cache_path: Path | str,
    state_path: Path | str,
    payload: dict[str, object],
) -> None:
    path = Path(cache_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": QR_ROD_PROFILE_CACHE_SCHEMA,
        "state_cache_key": qr_rod_profile_cache_key(state_path),
        "payload": dict(payload),
    }
    with path.open("wb") as handle:
        pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)


def reset_qr_rod_profile_cache(cache_path: Path | str) -> bool:
    path = Path(cache_path).expanduser()
    if not path.exists():
        return False
    path.unlink()
    return True


def clean_marker_title(value: object, *, max_chars: int = 80) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return ""
    except Exception:
        pass
    text = " ".join(str(value).split())
    if text.lower() == "nan":
        return ""
    limit = max(int(max_chars), 0)
    return text[:limit] if limit else text


def hk_display_label(m_value: object) -> str:
    return rf"$HK = {int(m_value)}$"


def l_tick_label(value: float) -> str:
    if np.isfinite(value):
        return f"{int(round(float(value)))}"
    return ""


def marker_row_title(marker_row: dict[str, object] | pd.Series, m_value: int) -> str:
    marker_title = clean_marker_title(marker_row.get("marker_title", ""))
    if marker_title:
        return marker_title
    try:
        display_l = float(marker_row.get("display_l", np.nan))
    except Exception:
        display_l = np.nan
    if not np.isfinite(display_l):
        try:
            display_l = float(marker_row.get("fit_l", marker_row.get("l", np.nan)))
        except Exception:
            display_l = np.nan
    if np.isfinite(display_l):
        return f"L={l_tick_label(float(display_l))}"
    return hk_display_label(int(m_value))


def qz_l_linear_coeff_from_marker_rows(marker_rows: object) -> tuple[float, float]:
    table = pd.DataFrame(marker_rows).copy()
    if table.empty or "qz_marker" not in table:
        return 1.0, 0.0
    l_col = "fit_l" if "fit_l" in table else "l" if "l" in table else "display_l"
    if l_col not in table:
        return 1.0, 0.0
    ref_qz = pd.to_numeric(table["qz_marker"], errors="coerce").to_numpy(dtype=np.float64)
    ref_l = pd.to_numeric(table[l_col], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(ref_qz) & np.isfinite(ref_l)
    ref_qz = ref_qz[finite]
    ref_l = ref_l[finite]
    if ref_qz.size >= 2 and float(np.nanmax(ref_qz)) > float(np.nanmin(ref_qz)):
        slope, intercept = np.polyfit(ref_qz, ref_l, 1)
    elif ref_qz.size == 1 and abs(float(ref_qz[0])) > 1.0e-12:
        slope = float(ref_l[0]) / float(ref_qz[0])
        intercept = 0.0
    else:
        slope, intercept = 1.0, 0.0
    if not (np.isfinite(slope) and np.isfinite(intercept)) or abs(float(slope)) <= 1.0e-12:
        return 1.0, 0.0
    return float(slope), float(intercept)


def qz_bounds_for_l_window(
    qz_values: object,
    marker_rows: object,
    *,
    l_min: float = 0.0,
    l_max: float = 8.0,
    positive_qz_min: float = 0.0,
) -> tuple[float, float] | None:
    qz = np.asarray(qz_values, dtype=np.float64)
    qz = qz[np.isfinite(qz) & (qz > float(positive_qz_min))]
    if qz.size < 2:
        return None
    marker_table = pd.DataFrame(marker_rows).copy()
    if marker_table.empty or "qz_marker" not in marker_table:
        return None
    l_col = "fit_l" if "fit_l" in marker_table else "l" if "l" in marker_table else "display_l"
    if l_col not in marker_table:
        return None
    ref_qz = pd.to_numeric(marker_table["qz_marker"], errors="coerce").to_numpy(dtype=np.float64)
    ref_l = pd.to_numeric(marker_table[l_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.count_nonzero(np.isfinite(ref_qz) & np.isfinite(ref_l)) < 1:
        return None
    slope, intercept = qz_l_linear_coeff_from_marker_rows(marker_table)
    if not (np.isfinite(slope) and np.isfinite(intercept)) or abs(float(slope)) <= 1.0e-12:
        return None
    l_lo, l_hi = sorted((float(l_min), float(l_max)))
    qz_l_lo = (l_lo - float(intercept)) / float(slope)
    qz_l_hi = (l_hi - float(intercept)) / float(slope)
    if not (np.isfinite(qz_l_lo) and np.isfinite(qz_l_hi)):
        return None
    qz_window_lo, qz_window_hi = sorted((float(qz_l_lo), float(qz_l_hi)))
    qz_lo = max(float(np.nanmin(qz)), float(qz_window_lo), float(positive_qz_min))
    qz_hi = min(float(np.nanmax(qz)), float(qz_window_hi))
    if not (np.isfinite(qz_lo) and np.isfinite(qz_hi)) or qz_hi <= qz_lo:
        return None
    return qz_lo, qz_hi


def drawable_rod_profile_keys(
    rod_profile_table: pd.DataFrame,
    marker_table: pd.DataFrame,
    *,
    min_points: int = 2,
) -> set[tuple[int, str]]:
    table = pd.DataFrame(rod_profile_table).copy()
    if table.empty or not {"m", "branch", "qz_center"}.issubset(table.columns):
        return set()
    y_columns = [
        column
        for column in (
            "background_density",
            "joint_peak_density",
            "joint_fit_density",
            "fit_density",
        )
        if column in table
    ]
    if not y_columns:
        return set()
    table["_m_key"] = pd.to_numeric(table["m"], errors="coerce")
    table["_branch_key"] = table["branch"].astype(str)
    table = table[np.isfinite(np.asarray(table["_m_key"], dtype=np.float64))].copy()
    if table.empty:
        return set()
    markers = pd.DataFrame(marker_table).copy()
    drawable: set[tuple[int, str]] = set()
    for (m_key, branch_key), sub in table.groupby(["_m_key", "_branch_key"], sort=False):
        m_value = int(m_key)
        branch_value = str(branch_key)
        qz = pd.to_numeric(sub["qz_center"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(qz)
        if "pixel_count" in sub:
            pixel_count = pd.to_numeric(sub["pixel_count"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite &= np.isfinite(pixel_count) & (pixel_count > 0.0)
        finite_y = np.zeros(qz.shape, dtype=bool)
        for column in y_columns:
            values = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=np.float64)
            finite_y |= np.isfinite(values)
        finite &= finite_y
        if np.count_nonzero(finite) < int(min_points):
            continue
        l_values = qz.copy()
        if not markers.empty and {"m", "branch", "qz_marker"}.issubset(markers.columns):
            marker_m = pd.to_numeric(markers["m"], errors="coerce").to_numpy(dtype=np.float64)
            marker_branch = markers["branch"].astype(str).to_numpy(dtype=object)
            marker_sub = markers[
                (marker_m == float(m_value)) & (marker_branch == branch_value)
            ].copy()
            if not marker_sub.empty and {"fit_l", "l", "display_l"} & set(marker_sub.columns):
                slope, intercept = qz_l_linear_coeff_from_marker_rows(marker_sub)
                l_values = float(slope) * qz + float(intercept)
        finite &= np.isfinite(l_values) & (l_values > 0.0)
        if np.count_nonzero(finite) >= int(min_points):
            drawable.add((m_value, branch_value))
    return drawable


def qr_rod_peak_edit_cache_key(
    path_value: object = None,
    *,
    marker_table: pd.DataFrame | None = None,
    mode: object = None,
    lattice_signature: object = None,
    q_group_signature: object = None,
    rod_reference_policy: object = None,
    rod_profile_policy: object = None,
) -> dict[str, object]:
    reference_signature = {}
    if lattice_signature is not None:
        reference_signature["active_lattice"] = _cache_normalize_value(lattice_signature)
    if q_group_signature is not None:
        reference_signature["q_group_rows"] = _cache_normalize_value(q_group_signature)
    if rod_reference_policy is not None:
        reference_signature["rod_reference_policy"] = _cache_normalize_value(
            rod_reference_policy
        )
    if rod_profile_policy is not None:
        reference_signature["rod_profile_policy"] = _cache_normalize_value(rod_profile_policy)

    def with_reference_signature(payload: dict[str, object]) -> dict[str, object]:
        if reference_signature:
            payload = dict(payload)
            payload["rod_reference_signature"] = reference_signature
        return payload

    if marker_table is not None:
        table = pd.DataFrame(marker_table).copy()
        columns = [
            col
            for col in (
                "m",
                "branch",
                "qz_marker",
                "fit_l",
                "display_l",
                "marker_title",
                "l",
                "hkl",
                "label",
            )
            if col in table
        ]
        if columns:
            normalized = table[columns].copy()
            for col in ("m", "qz_marker", "fit_l", "display_l", "l"):
                if col in normalized:
                    normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
            if {"m", "branch", "qz_marker"}.issubset(normalized.columns):
                normalized = normalized.sort_values(
                    ["m", "branch", "qz_marker"], kind="mergesort"
                ).reset_index(drop=True)
            records = json.loads(
                normalized.to_json(orient="records", double_precision=12, default_handler=str)
            )
        else:
            records = []
        data = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return with_reference_signature(
            {
                "mode": str(mode or "marker_table"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "rows": int(len(table)),
                "fit_signature": QR_ROD_FINAL_FIT_CACHE_SIGNATURE,
            }
        )
    text = "" if path_value is None else str(path_value).strip()
    if not text:
        return with_reference_signature(
            {"mode": "last_cached", "fit_signature": QR_ROD_FINAL_FIT_CACHE_SIGNATURE}
        )
    path = Path(text).expanduser()
    if not path.exists():
        return with_reference_signature(
            {
                "mode": "imported_edits",
                "exists": False,
                "path_name": path.name,
                "sha256": None,
                "size": None,
                "fit_signature": QR_ROD_FINAL_FIT_CACHE_SIGNATURE,
            }
        )
    data = path.read_bytes()
    return with_reference_signature(
        {
            "mode": "imported_edits",
            "exists": True,
            "path_name": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "fit_signature": QR_ROD_FINAL_FIT_CACHE_SIGNATURE,
        }
    )


def qr_rod_profile_cache_has_final_fit(
    payload: dict[str, object],
    peak_edit_cache_key: dict[str, object],
) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "final_rod_profile_table",
        "final_marker_table",
        "final_rod_component_table",
        "final_peak_edit_cache_key",
    }
    if not required.issubset(payload):
        return False
    rod_profile_table = pd.DataFrame(payload["final_rod_profile_table"])
    if rod_profile_table.empty or "qz_center" not in rod_profile_table:
        return False
    if not {"joint_fit_density", "fit_density"} & set(rod_profile_table.columns):
        return False
    marker_table = pd.DataFrame(payload["final_marker_table"])
    if marker_table.empty or not {"qz_marker", "fit_l", "display_l"}.issubset(marker_table.columns):
        return False
    return payload.get("final_peak_edit_cache_key") == peak_edit_cache_key


def qr_rod_peak_edit_runtime_mode(
    requested: object = "auto",
    *,
    backend_name: object = None,
    env: dict[str, object] | None = None,
) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode in {"0", "false", "off", "none", "no", "skip"}:
        return "skip"
    if mode in {"1", "true", "on", "yes", "interactive", "popup"}:
        return "popup"
    env_map = os.environ if env is None else env
    for key in ("CI", "RA_SIM_HEADLESS", "RA_SIM_QR_ROD_PEAK_EDIT_HEADLESS"):
        if str(env_map.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}:
            return "skip"
    backend = str(backend_name if backend_name is not None else mpl.get_backend()).lower()
    if any(token in backend for token in ("tk", "qt", "wx", "gtk", "macosx")):
        return "popup"
    return "skip"


def replace_qr_rod_marker_group_qz(
    marker_table: pd.DataFrame,
    *,
    m_value: int,
    branch_value: str,
    qz_values: object,
) -> pd.DataFrame:
    table = pd.DataFrame(marker_table).copy()
    if table.empty:
        table = pd.DataFrame(columns=["m", "branch", "qz_marker"])
    qz = np.asarray(qz_values, dtype=np.float64).reshape(-1)
    qz = np.unique(np.sort(qz[np.isfinite(qz)]))
    m_series = pd.to_numeric(
        table["m"] if "m" in table else pd.Series(np.nan, index=table.index),
        errors="coerce",
    )
    branch_series = (
        table["branch"].astype(str)
        if "branch" in table
        else pd.Series("", index=table.index, dtype=object)
    )
    group_mask = (np.asarray(m_series, dtype=np.float64) == int(m_value)) & (
        branch_series.astype(str) == str(branch_value)
    )
    other = table.loc[~group_mask].copy()
    group = table.loc[group_mask].copy()
    if not group.empty and "qz_marker" in group:
        group = group.sort_values("qz_marker", kind="mergesort")
    ref_qz = (
        pd.to_numeric(group["qz_marker"], errors="coerce").to_numpy(dtype=np.float64)
        if "qz_marker" in group
        else np.asarray([], dtype=np.float64)
    )
    l_col = "fit_l" if "fit_l" in group else "l" if "l" in group else ""
    ref_l = (
        pd.to_numeric(group[l_col], errors="coerce").to_numpy(dtype=np.float64)
        if l_col
        else np.asarray([], dtype=np.float64)
    )
    finite_refs = np.isfinite(ref_qz) & np.isfinite(ref_l)
    ref_qz = ref_qz[finite_refs]
    ref_l = ref_l[finite_refs]
    slope = np.nan
    intercept = np.nan
    if ref_qz.size >= 2 and float(np.nanmax(ref_qz)) > float(np.nanmin(ref_qz)):
        slope, intercept = np.polyfit(ref_qz, ref_l, 1)
    elif ref_qz.size == 1 and abs(float(ref_qz[0])) > 1.0e-12:
        slope = float(ref_l[0]) / float(ref_qz[0])
        intercept = 0.0

    template_rows = group.to_dict("records")
    edited_rows: list[dict[str, object]] = []
    for index, qz_value in enumerate(qz):
        row = dict(template_rows[min(index, len(template_rows) - 1)] if template_rows else {})
        row["m"] = int(m_value)
        row["hk"] = int(m_value)
        row["branch"] = str(branch_value)
        row["qz_marker"] = float(qz_value)
        row["projected_qz_marker"] = float(qz_value)
        if np.isfinite(slope) and np.isfinite(intercept):
            l_value = float(slope * float(qz_value) + intercept)
            row["fit_l"] = l_value
            row["display_l"] = l_value
            row["l"] = l_value
        row["marker_source"] = "manual_edit"
        edited_rows.append(row)

    edited_group = pd.DataFrame(edited_rows)
    if edited_group.empty:
        return other.reset_index(drop=True)
    columns = list(dict.fromkeys([*table.columns.tolist(), *edited_group.columns.tolist()]))
    return pd.concat(
        [other.reindex(columns=columns), edited_group.reindex(columns=columns)],
        ignore_index=True,
        sort=False,
    )


def write_qr_rod_peak_edits(path_value: Path | str, marker_table: pd.DataFrame) -> Path:
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(marker_table).copy()
    records = json.loads(table.to_json(orient="records", double_precision=12, default_handler=str))
    payload = {"schema": "ra_sim.qr_rod_peak_edits.v1", "markers": records}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_qr_rod_peak_edits(path_value: Path | str) -> pd.DataFrame:
    path = Path(path_value).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("markers", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Qr-rod peak edits must be a JSON list or contain a markers list")
    table = pd.DataFrame(records)
    if table.empty:
        return pd.DataFrame(columns=["m", "branch", "qz_marker", "marker_title"])
    required = {"m", "branch", "qz_marker"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Qr-rod peak edits missing columns: {sorted(missing)}")
    for col in ("m", "hk", "qz_marker", "projected_qz_marker", "fit_l", "display_l", "l"):
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    table["m"] = np.asarray(table["m"], dtype=int)
    table["branch"] = table["branch"].astype(str)
    if "marker_title" in table:
        table["marker_title"] = [clean_marker_title(value) for value in table["marker_title"]]
    table = table[np.isfinite(np.asarray(table["qz_marker"], dtype=np.float64))].copy()
    return table.sort_values(["m", "branch", "qz_marker"], kind="mergesort").reset_index(drop=True)


def marker_table_with_specular_l_markers(
    marker_table: pd.DataFrame,
    specular_l_marker_table: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame(marker_table).copy()
    specular = pd.DataFrame(specular_l_marker_table).copy()
    if specular.empty:
        return base.reset_index(drop=True)
    columns = list(dict.fromkeys([*base.columns.tolist(), *specular.columns.tolist()]))
    duplicate_key = None
    if {"m", "branch", "hkl"}.issubset(columns):
        duplicate_key = ["m", "branch", "hkl"]
    elif {"m", "branch", "qz_marker"}.issubset(columns):
        duplicate_key = ["m", "branch", "qz_marker"]
    base = base.reindex(columns=columns)
    specular = specular.reindex(columns=columns)
    if duplicate_key is not None:
        seen = set()
        if not base.empty:
            seen.update(tuple(row[col] for col in duplicate_key) for row in base.to_dict("records"))
        keep_rows: list[dict[str, object]] = []
        for row in specular.to_dict("records"):
            key = tuple(row[col] for col in duplicate_key)
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(row)
        specular = pd.DataFrame(keep_rows, columns=columns)
    return pd.concat([base, specular], ignore_index=True, sort=False).reset_index(drop=True)


def specular_export_marker_table_from_final_markers(
    marker_table: pd.DataFrame,
    specular_l_marker_table: pd.DataFrame,
    *,
    qz_map: object,
    region_mask: object,
    theta_axis: object,
    phi_axis: object,
) -> pd.DataFrame:
    final_markers = pd.DataFrame(marker_table).copy()
    fallback = pd.DataFrame(specular_l_marker_table).copy()
    if final_markers.empty or not {"m", "branch", "qz_marker"}.issubset(final_markers.columns):
        return fallback.reset_index(drop=True)
    m_values = pd.to_numeric(final_markers["m"], errors="coerce").to_numpy(dtype=np.float64)
    branches = final_markers["branch"].astype(str)
    specular = final_markers.loc[(m_values == 0.0) & (branches == "qz")].copy()
    if specular.empty:
        return fallback.reset_index(drop=True)
    specular["qz_marker"] = pd.to_numeric(specular["qz_marker"], errors="coerce")
    specular = specular[np.isfinite(np.asarray(specular["qz_marker"], dtype=np.float64))].copy()
    if specular.empty:
        return fallback.reset_index(drop=True)

    fallback_by_hkl: dict[str, dict[str, object]] = {}
    fallback_by_l: dict[int, dict[str, object]] = {}

    def finite_number(value: object) -> float:
        try:
            number = float(value)
        except Exception:
            return np.nan
        return number if np.isfinite(number) else np.nan

    def marker_l_value(row: dict[str, object]) -> float:
        for column in ("fit_l", "l", "display_l"):
            number = finite_number(row.get(column, np.nan))
            if np.isfinite(number):
                return float(number)
        return np.nan

    def has_value(value: object) -> bool:
        if value is None:
            return False
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)):
                return not bool(missing)
        except Exception:
            pass
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return False
        except Exception:
            pass
        return True

    for record in fallback.to_dict("records"):
        hkl = str(record.get("hkl", "")).strip()
        if hkl:
            fallback_by_hkl.setdefault(hkl, record)
        l_value = marker_l_value(record)
        if np.isfinite(l_value):
            fallback_by_l.setdefault(int(round(float(l_value))), record)

    qz_values = np.asarray(qz_map, dtype=np.float64)
    valid = np.isfinite(qz_values)
    mask_values = np.asarray(region_mask, dtype=bool)
    if mask_values.shape == qz_values.shape:
        valid &= mask_values
    theta_values = np.asarray(theta_axis, dtype=np.float64)
    phi_values = np.asarray(phi_axis, dtype=np.float64)

    def axis_value(values: np.ndarray, row: int, col: int, *, prefer_col: bool) -> float:
        if values.ndim == 1:
            index = col if prefer_col else row
            if 0 <= index < values.shape[0]:
                return finite_number(values[index])
        elif values.shape == qz_values.shape:
            return finite_number(values[row, col])
        return np.nan

    def nearest_angles(qz_value: float) -> tuple[float, float] | None:
        if qz_values.ndim != 2 or qz_values.size < 1 or not np.any(valid):
            return None
        distance = np.abs(qz_values - float(qz_value))
        distance = np.where(valid, distance, np.inf)
        if not np.any(np.isfinite(distance)):
            return None
        row, col = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
        theta = axis_value(theta_values, int(row), int(col), prefer_col=True)
        phi = axis_value(phi_values, int(row), int(col), prefer_col=False)
        if not (np.isfinite(theta) and np.isfinite(phi)):
            return None
        return float(theta), float(phi)

    records: list[dict[str, object]] = []
    for edit_row in specular.sort_values("qz_marker", kind="mergesort").to_dict("records"):
        hkl = str(edit_row.get("hkl", "")).strip()
        l_value = marker_l_value(edit_row)
        base_row = fallback_by_hkl.get(hkl)
        if base_row is None and np.isfinite(l_value):
            base_row = fallback_by_l.get(int(round(float(l_value))))
        row = dict(base_row or {})
        for key, value in edit_row.items():
            if has_value(value):
                row[key] = value
        row["m"] = 0
        row["hk"] = 0
        row["branch"] = "qz"
        qz_value = finite_number(row.get("qz_marker", np.nan))
        if not np.isfinite(qz_value):
            continue
        row["qz_marker"] = float(qz_value)
        row["projected_qz_marker"] = float(qz_value)
        l_value = marker_l_value(row)
        if np.isfinite(l_value):
            row["fit_l"] = float(l_value)
            row["display_l"] = float(l_value)
            row["l"] = float(l_value)
            if not str(row.get("hkl", "")).strip():
                row["hkl"] = f"0,0,{int(round(float(l_value)))}"
        angles = nearest_angles(float(qz_value))
        if angles is not None:
            row["refined_two_theta_deg"] = angles[0]
            row["refined_phi_deg"] = angles[1]
        records.append(row)
    return pd.DataFrame(records).reset_index(drop=True)


def snap_qr_rod_markers_to_profile_peaks(
    qz_markers: object,
    qz_center: object,
    background_density: object,
    *,
    window_fraction: float = 0.035,
) -> np.ndarray:
    markers = np.asarray(qz_markers, dtype=np.float64).reshape(-1)
    x = np.asarray(qz_center, dtype=np.float64).reshape(-1)
    y = np.asarray(background_density, dtype=np.float64).reshape(-1)
    snapped = markers.copy()
    finite_markers = np.isfinite(markers)
    finite_profile = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite_markers) < 1 or np.count_nonzero(finite_profile) < 3:
        return snapped
    x_valid = x[finite_profile]
    y_valid = y[finite_profile]
    x_span = float(np.nanmax(x_valid) - np.nanmin(x_valid))
    if not np.isfinite(x_span) or x_span <= 0.0:
        return snapped
    half_window = max(x_span * max(float(window_fraction), 0.0), 1.0e-6)
    for index, marker in enumerate(markers):
        if not np.isfinite(marker):
            continue
        local = np.abs(x_valid - float(marker)) <= half_window
        if not np.any(local):
            continue
        snapped[index] = float(x_valid[local][int(np.nanargmax(y_valid[local]))])
    return snapped


def show_qr_rod_peak_marker_popup(
    marker_table: pd.DataFrame,
    rod_profile_table: pd.DataFrame,
    *,
    backend_name: object = None,
    edit_path: object = None,
    required_marker_table: pd.DataFrame | None = None,
    region_state: dict[str, object] | None = None,
    profile_update_callback: object | None = None,
) -> tuple[pd.DataFrame, bool]:
    backend = str(backend_name if backend_name is not None else mpl.get_backend())
    if qr_rod_peak_edit_runtime_mode("auto", backend_name=backend, env={}) != "popup":
        raise RuntimeError(f"Matplotlib backend {backend!r} is not interactive")
    try:
        from matplotlib.widgets import Button, Slider, TextBox
    except Exception as exc:
        raise RuntimeError(f"Matplotlib editor widgets are unavailable: {exc}") from exc

    edited = pd.DataFrame(marker_table).copy()
    if required_marker_table is not None:
        edited = marker_table_with_specular_l_markers(edited, required_marker_table)
    original = edited.copy()
    profiles = pd.DataFrame(rod_profile_table).copy()
    if edited.empty or profiles.empty or not {"m", "branch", "qz_center"}.issubset(profiles):
        return edited, False

    groups: list[tuple[int, str]] = []
    for row in profiles[["m", "branch"]].drop_duplicates().to_dict("records"):
        try:
            group = (int(row["m"]), str(row["branch"]))
        except Exception:
            continue
        if group not in groups:
            groups.append(group)
    if not groups:
        return edited, False

    region_controls_enabled = isinstance(region_state, dict)
    region_control_state = region_state if isinstance(region_state, dict) else {}
    if region_controls_enabled:
        current_delta_qr = max(1.0e-9, as_float(region_control_state.get("delta_qr"), 1.0e-3))
        current_l_min = as_float(region_control_state.get("l_min"), 0.0)
        current_l_max = as_float(
            region_control_state.get("l_max"), max(current_l_min + 1.0, 8.0)
        )
        if not np.isfinite(current_l_min):
            current_l_min = 0.0
        if not np.isfinite(current_l_max) or current_l_max <= current_l_min:
            current_l_max = max(current_l_min + 1.0, 8.0)
        region_control_state["delta_qr"] = float(current_delta_qr)
        region_control_state["l_min"] = float(current_l_min)
        region_control_state["l_max"] = float(current_l_max)
        region_control_state["rod_profile_table"] = profiles.copy()

    cols = min(3, max(1, len(groups)))
    rows = int(math.ceil(len(groups) / cols))
    control_extra_height = 1.15 if region_controls_enabled else 0.6
    fig, axes = plt.subplots(
        rows,
        cols,
        squeeze=False,
        figsize=(4.2 * cols, 2.7 * rows + control_extra_height),
        num="Qr rod peak marker editor",
    )
    try:
        fig.canvas.manager.set_window_title("Qr rod peak marker editor")
    except Exception:
        pass
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.26 if region_controls_enabled else 0.18,
        hspace=0.55,
        wspace=0.30,
    )

    flat_axes = list(axes.ravel())
    axes_by_group = dict(zip(groups, flat_axes))
    group_by_axes = {ax: group for group, ax in axes_by_group.items()}
    for ax in flat_axes[len(groups) :]:
        ax.set_axis_off()

    result = {"accepted": True}
    selected: dict[str, object] = {"group": None, "index": None, "dragging": False}
    title_box_state: dict[str, object] = {"box": None, "syncing": False}
    edit_file_state: dict[str, object] = {
        "path": "" if edit_path is None else str(edit_path).strip()
    }

    def current_l_bounds() -> tuple[float, float] | None:
        if not region_controls_enabled:
            return None
        l_min_value = as_float(region_control_state.get("l_min"), 0.0)
        l_max_value = as_float(region_control_state.get("l_max"), l_min_value + 1.0)
        if not np.isfinite(l_min_value):
            l_min_value = 0.0
        if not np.isfinite(l_max_value) or l_max_value <= l_min_value:
            l_max_value = l_min_value + 1.0
        return tuple(sorted((float(l_min_value), float(l_max_value))))

    def refresh_region_profile_table() -> None:
        nonlocal profiles
        if not region_controls_enabled or not callable(profile_update_callback):
            return
        l_bounds = current_l_bounds()
        if l_bounds is None:
            return
        delta_qr_value = max(1.0e-9, as_float(region_control_state.get("delta_qr"), 1.0e-3))
        l_min_value, l_max_value = l_bounds
        try:
            updated = profile_update_callback(delta_qr_value, l_min_value, l_max_value)
        except Exception as exc:
            region_control_state["profile_update_error"] = str(exc)
            return
        updated_table = pd.DataFrame(updated).copy()
        if updated_table.empty or not {"m", "branch", "qz_center"}.issubset(updated_table):
            return
        profiles = updated_table
        region_control_state["rod_profile_table"] = profiles.copy()
        region_control_state.pop("profile_update_error", None)

    def group_marker_rows(m_value: int, branch_value: str) -> pd.DataFrame:
        if edited.empty or not {"m", "branch", "qz_marker"}.issubset(edited):
            return pd.DataFrame()
        mask = (np.asarray(edited["m"], dtype=int) == int(m_value)) & (
            edited["branch"].astype(str) == str(branch_value)
        )
        sub = edited.loc[mask].copy()
        if sub.empty:
            return sub
        sub["_source_index"] = sub.index
        sub["qz_marker"] = pd.to_numeric(sub["qz_marker"], errors="coerce")
        sub = sub[np.isfinite(np.asarray(sub["qz_marker"], dtype=np.float64))].copy()
        return sub.sort_values("qz_marker", kind="mergesort").reset_index(drop=True)

    def group_markers(m_value: int, branch_value: str) -> np.ndarray:
        rows = group_marker_rows(m_value, branch_value)
        if rows.empty:
            return np.asarray([], dtype=np.float64)
        values = np.asarray(rows["qz_marker"], dtype=np.float64)
        return np.sort(values[np.isfinite(values)])

    def group_qz_l_coeff(m_value: int, branch_value: str) -> tuple[float, float]:
        return qz_l_linear_coeff_from_marker_rows(group_marker_rows(m_value, branch_value))

    def qz_to_editor_l(m_value: int, branch_value: str, qz_values: object) -> np.ndarray:
        slope, intercept = group_qz_l_coeff(m_value, branch_value)
        qz = np.asarray(qz_values, dtype=np.float64)
        return float(slope) * qz + float(intercept)

    def editor_l_to_qz(m_value: int, branch_value: str, l_values: object) -> np.ndarray:
        slope, intercept = group_qz_l_coeff(m_value, branch_value)
        l = np.asarray(l_values, dtype=np.float64)
        return (l - float(intercept)) / float(slope)

    def selected_marker_row() -> tuple[int, pd.Series] | None:
        group = selected.get("group")
        index = selected.get("index")
        if group is None or index is None:
            return None
        m_value, branch_value = group
        rows = group_marker_rows(int(m_value), str(branch_value))
        if rows.empty or int(index) < 0 or int(index) >= len(rows):
            return None
        row = rows.iloc[int(index)]
        return int(row["_source_index"]), row

    def sync_title_box() -> None:
        box = title_box_state.get("box")
        if box is None:
            return
        selected_row = selected_marker_row()
        title = ""
        if selected_row is not None:
            _source_index, row = selected_row
            group = selected.get("group")
            if group is not None:
                title = marker_row_title(row, int(group[0]))
        title_box_state["syncing"] = True
        try:
            box.set_val(title)
        finally:
            title_box_state["syncing"] = False

    def set_selected_marker_title(text: object, *, redraw_figure: bool = True) -> None:
        if bool(title_box_state.get("syncing", False)):
            return
        selected_row = selected_marker_row()
        if selected_row is None:
            return
        source_index, _row = selected_row
        nonlocal edited
        if "marker_title" not in edited:
            edited["marker_title"] = ""
        title = clean_marker_title(text)
        if clean_marker_title(edited.loc[source_index, "marker_title"]) == title:
            return
        edited.loc[source_index, "marker_title"] = title
        if redraw_figure:
            redraw()

    def flush_title_box() -> None:
        box = title_box_state.get("box")
        if box is None:
            return
        set_selected_marker_title(getattr(box, "text", ""), redraw_figure=False)

    def default_peak_edit_path() -> Path:
        text = str(edit_file_state.get("path", "")).strip()
        if text:
            return Path(text).expanduser()
        base_dir = Path(globals().get("OUT_DIR", Path.cwd())).expanduser()
        stem = _safe_run_name(str(globals().get("ROD_PROFILE_STEM", "qr_rod_peak_markers")))
        return base_dir / f"{stem}_peak_edits.json"

    def choose_peak_edit_path(action: str) -> Path | None:
        try:
            from tkinter import filedialog
        except Exception as exc:
            print(f"Qr-rod peak marker editor {action} dialog unavailable: {exc}")
            return None
        initial_path = default_peak_edit_path()
        options = {
            "title": f"{action.title()} Qr-rod peak edits",
            "filetypes": [
                ("RA-SIM Qr-rod peak edits", "*.json"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
            "initialdir": str(initial_path.parent),
        }
        if action == "import":
            selected_path = filedialog.askopenfilename(**options)
        else:
            selected_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=initial_path.name,
                **options,
            )
        if not selected_path:
            return None
        path = Path(selected_path).expanduser()
        edit_file_state["path"] = str(path)
        return path

    def profile_xy(m_value: int, branch_value: str) -> tuple[np.ndarray, np.ndarray]:
        sub = profiles[
            (np.asarray(profiles["m"], dtype=int) == int(m_value))
            & (profiles["branch"].astype(str) == str(branch_value))
        ].copy()
        if sub.empty or "background_density" not in sub:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        x = np.asarray(sub["qz_center"], dtype=np.float64)
        y = np.asarray(sub["background_density"], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 1:
            return x, y
        order = np.argsort(x)
        return x[order], y[order]

    def marker_y_values(markers: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if markers.size == 0:
            return np.asarray([], dtype=np.float64)
        if x.size < 2:
            return np.zeros(markers.shape, dtype=np.float64)
        return np.interp(markers, x, y, left=np.nan, right=np.nan)

    def set_group_markers(m_value: int, branch_value: str, values: object) -> None:
        nonlocal edited
        edited = replace_qr_rod_marker_group_qz(
            edited,
            m_value=int(m_value),
            branch_value=str(branch_value),
            qz_values=values,
        )

    def redraw() -> None:
        selected_group = selected.get("group")
        selected_index = selected.get("index")
        l_bounds = current_l_bounds()
        for group, ax in axes_by_group.items():
            m_value, branch_value = group
            ax.clear()
            x_qz, y = profile_xy(m_value, branch_value)
            x_l = qz_to_editor_l(m_value, branch_value, x_qz)
            y_plot = positive_log_plot_values(y)
            finite_profile = np.isfinite(x_l) & np.isfinite(y)
            if l_bounds is not None:
                l_lo, l_hi = l_bounds
                finite_profile = finite_profile & (x_l >= float(l_lo)) & (x_l <= float(l_hi))
            if np.any(finite_profile):
                order = np.argsort(x_l[finite_profile])
                ax.plot(
                    x_l[finite_profile][order],
                    y_plot[finite_profile][order],
                    color="0.25",
                    linewidth=0.9,
                )
            markers = group_markers(m_value, branch_value)
            marker_l = qz_to_editor_l(m_value, branch_value, markers)
            marker_rows = group_marker_rows(m_value, branch_value)
            y_markers = marker_y_values(markers, x_qz, y)
            y_markers_plot = positive_log_plot_values(y_markers)
            finite = np.isfinite(marker_l) & np.isfinite(y_markers)
            if l_bounds is not None:
                l_lo, l_hi = l_bounds
                finite = finite & (marker_l >= float(l_lo)) & (marker_l <= float(l_hi))
            if np.any(finite):
                colors = ["white"] * int(np.count_nonzero(finite))
                if selected_group == group and selected_index is not None:
                    finite_indices = np.flatnonzero(finite)
                    try:
                        color_index = list(finite_indices).index(int(selected_index))
                        colors[color_index] = "#d95f02"
                    except ValueError:
                        pass
                ax.scatter(
                    marker_l[finite],
                    y_markers_plot[finite],
                    s=24.0,
                    marker="o",
                    facecolors=colors,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=5,
                )
                visible_rows = marker_rows.loc[finite].reset_index(drop=True)
                for marker_x, marker_y, marker_row in zip(
                    marker_l[finite], y_markers[finite], visible_rows.to_dict("records")
                ):
                    ax.annotate(
                        marker_row_title(marker_row, int(m_value)),
                        xy=(float(marker_x), float(marker_y)),
                        xytext=(0.0, 7.0),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="black",
                        annotation_clip=True,
                        zorder=6,
                    )
            title_color = "#d95f02" if selected_group == group else "black"
            ax.set_title(f"HK={m_value} {branch_value}", fontsize=9, color=title_color)
            ax.set_xlabel("L", fontsize=8)
            ax.set_ylabel("Intensity (log)", fontsize=8)
            apply_positive_log_y_axis(ax, y, y_markers)
            if l_bounds is not None:
                ax.set_xlim(*l_bounds)
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax.tick_params(labelsize=7)
        fig.canvas.draw_idle()

    def refresh_region_controls() -> None:
        refresh_region_profile_table()
        redraw()

    def set_region_delta_qr(value: object) -> None:
        if not region_controls_enabled:
            return
        region_control_state["delta_qr"] = max(1.0e-9, as_float(value, 1.0e-3))
        refresh_region_controls()

    def set_region_l_min(text: object) -> None:
        if not region_controls_enabled:
            return
        value = as_float(text, region_control_state.get("l_min", 0.0))
        if np.isfinite(value):
            region_control_state["l_min"] = float(value)
        refresh_region_controls()

    def set_region_l_max(text: object) -> None:
        if not region_controls_enabled:
            return
        value = as_float(text, region_control_state.get("l_max", 8.0))
        if np.isfinite(value):
            region_control_state["l_max"] = float(value)
        refresh_region_controls()

    def select_nearest(group: tuple[int, str], x_value: float) -> bool:
        markers = group_markers(group[0], group[1])
        if markers.size == 0 or not np.isfinite(x_value):
            return False
        marker_l = qz_to_editor_l(group[0], group[1], markers)
        finite = np.isfinite(marker_l)
        if not np.any(finite):
            return False
        axis = axes_by_group[group]
        x_lo, x_hi = axis.get_xlim()
        tolerance = max(abs(float(x_hi) - float(x_lo)) * 0.025, 1.0e-6)
        nearest_index = int(
            np.nanargmin(np.where(finite, np.abs(marker_l - float(x_value)), np.inf))
        )
        if abs(float(marker_l[nearest_index]) - float(x_value)) > tolerance:
            return False
        selected["group"] = group
        selected["index"] = nearest_index
        sync_title_box()
        return True

    def add_marker(group: tuple[int, str], x_value: float) -> None:
        markers = group_markers(group[0], group[1]).tolist()
        qz_value = float(editor_l_to_qz(group[0], group[1], [float(x_value)])[0])
        markers.append(qz_value)
        set_group_markers(group[0], group[1], markers)
        select_nearest(group, float(x_value))
        redraw()

    def move_selected(x_value: float) -> None:
        group = selected.get("group")
        index = selected.get("index")
        if group is None or index is None or not np.isfinite(x_value):
            return
        m_value, branch_value = group
        markers = group_markers(m_value, branch_value).tolist()
        if int(index) < 0 or int(index) >= len(markers):
            return
        markers[int(index)] = float(editor_l_to_qz(m_value, branch_value, [float(x_value)])[0])
        set_group_markers(m_value, branch_value, markers)
        select_nearest((int(m_value), str(branch_value)), float(x_value))
        redraw()

    def delete_selected() -> None:
        group = selected.get("group")
        index = selected.get("index")
        if group is None or index is None:
            return
        m_value, branch_value = group
        markers = group_markers(m_value, branch_value).tolist()
        if int(index) < 0 or int(index) >= len(markers):
            return
        del markers[int(index)]
        set_group_markers(m_value, branch_value, markers)
        selected["index"] = None
        sync_title_box()
        redraw()

    def snap_selected_group() -> None:
        group = selected.get("group")
        if group is None:
            return
        flush_title_box()
        m_value, branch_value = group
        markers = group_markers(m_value, branch_value)
        if markers.size < 1:
            return
        x, y = profile_xy(m_value, branch_value)
        snapped = snap_qr_rod_markers_to_profile_peaks(markers, x, y)
        if np.array_equal(snapped, markers):
            return
        selected_marker = None
        index = selected.get("index")
        if index is not None and int(index) >= 0 and int(index) < markers.size:
            selected_marker = float(snapped[int(index)])
        set_group_markers(m_value, branch_value, snapped)
        if selected_marker is not None:
            selected_l = float(qz_to_editor_l(m_value, branch_value, [selected_marker])[0])
            select_nearest((int(m_value), str(branch_value)), selected_l)
        redraw()

    def import_peak_edits(_event) -> None:
        nonlocal edited
        flush_title_box()
        import_path = choose_peak_edit_path("import")
        if import_path is None:
            return
        try:
            imported = load_qr_rod_peak_edits(import_path)
            if required_marker_table is not None:
                imported = marker_table_with_specular_l_markers(imported, required_marker_table)
            edited = imported.copy()
            selected["group"] = None
            selected["index"] = None
            sync_title_box()
            redraw()
            print(f"imported Qr-rod peak edits={import_path}")
        except Exception as exc:
            print(f"failed importing Qr-rod peak edits={import_path}: {exc}")

    def export_peak_edits(_event) -> None:
        flush_title_box()
        export_path = choose_peak_edit_path("export")
        if export_path is None:
            return
        try:
            saved_path = write_qr_rod_peak_edits(export_path, edited)
            edit_file_state["path"] = str(saved_path)
            print(f"exported Qr-rod peak edits={saved_path}")
        except Exception as exc:
            print(f"failed exporting Qr-rod peak edits={export_path}: {exc}")

    def on_press(event) -> None:
        if event.inaxes not in group_by_axes or event.xdata is None:
            return
        flush_title_box()
        group = group_by_axes[event.inaxes]
        selected["group"] = group
        selected["index"] = None
        sync_title_box()
        if getattr(event, "dblclick", False):
            add_marker(group, float(event.xdata))
            return
        if select_nearest(group, float(event.xdata)):
            selected["dragging"] = True
        redraw()

    def on_motion(event) -> None:
        if not selected.get("dragging") or event.xdata is None:
            return
        if event.inaxes not in group_by_axes or group_by_axes[event.inaxes] != selected.get(
            "group"
        ):
            return
        move_selected(float(event.xdata))

    def on_release(_event) -> None:
        selected["dragging"] = False

    def on_key(event) -> None:
        box = title_box_state.get("box")
        if box is not None and getattr(event, "inaxes", None) is getattr(box, "ax", None):
            return
        key = str(getattr(event, "key", "")).lower()
        if key in {"delete", "backspace"}:
            delete_selected()
        elif key in {"s"}:
            snap_selected_group()
        elif key in {"enter", "return"}:
            flush_title_box()
            result["accepted"] = True
            plt.close(fig)
        elif key in {"escape"}:
            result["accepted"] = False
            plt.close(fig)

    def accept(_event) -> None:
        flush_title_box()
        result["accepted"] = True
        plt.close(fig)

    def cancel(_event) -> None:
        result["accepted"] = False
        plt.close(fig)

    region_widgets: list[object] = []
    if region_controls_enabled:
        delta_qr_ax = fig.add_axes([0.08, 0.105, 0.30, 0.045])
        l_min_ax = fig.add_axes([0.46, 0.105, 0.10, 0.045])
        l_max_ax = fig.add_axes([0.62, 0.105, 0.10, 0.045])
        delta_qr_value = max(1.0e-9, as_float(region_control_state.get("delta_qr"), 1.0e-3))
        delta_qr_slider = Slider(delta_qr_ax, "Delta Qr (+/- A^-1)", 1.0e-9, max(
            delta_qr_value * 4.0, delta_qr_value + 1.0e-6
        ), valinit=delta_qr_value)
        l_min_initial = f"{as_float(region_control_state.get('l_min'), 0.0):.6g}"
        l_max_initial = f"{as_float(region_control_state.get('l_max'), 8.0):.6g}"
        l_min_box = TextBox(l_min_ax, "L Min", initial=l_min_initial)
        l_max_box = TextBox(l_max_ax, "L Max", initial=l_max_initial)
        delta_qr_slider.on_changed(set_region_delta_qr)
        l_min_box.on_submit(set_region_l_min)
        l_max_box.on_submit(set_region_l_max)
        region_widgets.extend([delta_qr_slider, l_min_box, l_max_box])

    button_axes = [
        fig.add_axes([0.46, 0.035, 0.085, 0.05]),
        fig.add_axes([0.55, 0.035, 0.085, 0.05]),
        fig.add_axes([0.64, 0.035, 0.085, 0.05]),
        fig.add_axes([0.73, 0.035, 0.085, 0.05]),
        fig.add_axes([0.82, 0.035, 0.085, 0.05]),
    ]
    title_box = TextBox(fig.add_axes([0.08, 0.035, 0.34, 0.05]), "Label", textalignment="left")
    title_box.on_submit(set_selected_marker_title)
    title_box_state["box"] = title_box
    buttons = [
        Button(button_axes[0], "Snap"),
        Button(button_axes[1], "Import"),
        Button(button_axes[2], "Export"),
        Button(button_axes[3], "Cancel"),
        Button(button_axes[4], "Accept"),
    ]
    buttons[0].on_clicked(lambda event: snap_selected_group())
    buttons[1].on_clicked(import_peak_edits)
    buttons[2].on_clicked(export_peak_edits)
    buttons[3].on_clicked(cancel)
    buttons[4].on_clicked(accept)
    fig._ra_sim_qr_rod_peak_edit_widgets = [title_box, *region_widgets, *buttons]
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show(block=True)
    if not result["accepted"]:
        return original, False
    return edited.reset_index(drop=True), True


def edit_qr_rod_region_editor(
    marker_table: pd.DataFrame,
    rod_profile_table: pd.DataFrame,
    *,
    mode: object = "auto",
    edit_path: object = None,
    detector_label_entries: list[dict[str, object]] | None = None,
    delta_qr: object = None,
    l_min: object = None,
    l_max: object = None,
    profile_update_callback: object | None = None,
    required_marker_table: pd.DataFrame | None = None,
    backend_name: object = None,
    env: dict[str, object] | None = None,
) -> dict[str, object]:
    table = pd.DataFrame(marker_table).copy()
    fallback_profile_table = pd.DataFrame(rod_profile_table).copy()
    clean_label_entries = [
        dict(entry) for entry in (detector_label_entries or []) if isinstance(entry, dict)
    ]
    source_mode = "last_cached"
    path_text = "" if edit_path is None else str(edit_path).strip()
    if path_text:
        try:
            table = load_qr_rod_peak_edits(path_text)
            if required_marker_table is not None:
                table = marker_table_with_specular_l_markers(table, required_marker_table)
            source_mode = "imported_edits"
            print(f"loaded Qr-rod peak edits={Path(path_text).expanduser()}")
        except Exception as exc:
            print(f"ignored Qr-rod peak edits={Path(path_text).expanduser()}: {exc}")

    current_delta_qr = max(1.0e-9, as_float(delta_qr, qr_rod_delta_qr))
    current_l_min = as_float(l_min, 0.0)
    current_l_max = as_float(l_max, SPECULAR_QR_ROD_L_MAX)
    runtime_mode = qr_rod_peak_edit_runtime_mode(mode, backend_name=backend_name, env=env)

    def fallback_result(source: str) -> dict[str, object]:
        return {
            "marker_table": table,
            "rod_profile_table": fallback_profile_table.copy(),
            "detector_label_entries": [dict(entry) for entry in clean_label_entries],
            "delta_qr": float(current_delta_qr),
            "l_min": float(current_l_min),
            "l_max": float(current_l_max),
            "accepted": False,
            "source": source,
        }

    if runtime_mode != "popup":
        print(f"Qr-rod region editor: mode={runtime_mode} source={source_mode}")
        return fallback_result(source_mode)

    if not np.isfinite(current_l_min):
        current_l_min = 0.0
    if not np.isfinite(current_l_max) or current_l_max <= current_l_min:
        current_l_max = max(current_l_min + 1.0, SPECULAR_QR_ROD_L_MAX)
    region_control_state = {
        "delta_qr": float(current_delta_qr),
        "l_min": float(current_l_min),
        "l_max": float(current_l_max),
    }
    try:
        edited_markers, accepted = show_qr_rod_peak_marker_popup(
            table,
            rod_profile_table,
            backend_name=backend_name,
            edit_path=path_text,
            required_marker_table=required_marker_table,
            region_state=region_control_state,
            profile_update_callback=profile_update_callback,
        )
    except Exception as exc:
        if str(mode or "auto").strip().lower() == "popup":
            raise RuntimeError(f"Qr-rod region editor popup unavailable: {exc}") from exc
        print(f"skipped Qr-rod region editor popup: {exc}")
        return fallback_result(source_mode)

    result = {
        "marker_table": edited_markers if accepted else table,
        "rod_profile_table": pd.DataFrame(
            region_control_state.get("rod_profile_table", fallback_profile_table)
        ).copy(),
        "detector_label_entries": [dict(entry) for entry in clean_label_entries],
        "delta_qr": float(region_control_state.get("delta_qr", current_delta_qr)),
        "l_min": float(region_control_state.get("l_min", current_l_min)),
        "l_max": float(region_control_state.get("l_max", current_l_max)),
        "accepted": bool(accepted),
        "source": "popup" if accepted else source_mode,
    }
    if bool(result.get("accepted", False)) and path_text:
        try:
            saved_path = write_qr_rod_peak_edits(path_text, pd.DataFrame(result["marker_table"]))
            print(f"saved Qr-rod peak edits={saved_path}")
        except Exception as exc:
            print(f"failed saving Qr-rod peak edits={Path(path_text).expanduser()}: {exc}")
    print(
        "Qr-rod region editor: "
        f"mode=popup source={result.get('source', source_mode)} "
        f"accepted={bool(result.get('accepted', False))}"
    )
    return result


def apply_unified_qr_rod_region_editor_labels(
    label_entries: list[dict[str, object]],
    editor_result: object,
) -> list[dict[str, object]]:
    if not isinstance(editor_result, dict):
        return [dict(entry) for entry in label_entries if isinstance(entry, dict)]
    edited_entries = editor_result.get("detector_label_entries")
    if not isinstance(edited_entries, list) or not edited_entries:
        return [dict(entry) for entry in label_entries if isinstance(entry, dict)]
    return [dict(entry) for entry in edited_entries if isinstance(entry, dict)]


def qr_rod_profile_cache_with_final_fit(
    payload: dict[str, object] | None,
    rod_profile_table: pd.DataFrame,
    marker_table: pd.DataFrame,
    rod_component_table: pd.DataFrame,
    peak_edit_cache_key: dict[str, object],
) -> dict[str, object]:
    next_payload = dict(payload or {})
    next_payload.update(
        {
            "final_rod_profile_table": rod_profile_table.copy(),
            "final_marker_table": marker_table.copy(),
            "final_rod_component_table": rod_component_table.copy(),
            "final_peak_edit_cache_key": dict(peak_edit_cache_key),
        }
    )
    return next_payload


def _sample_name_from_path(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    stem = Path(str(value)).stem.strip()
    return stem or None


def _sample_label_from_name(name: object) -> str:
    text = str(name).strip() or "sample"

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}$_{match.group(2)}$"

    return re.sub(r"([A-Z][a-z]?)(\d+)", repl, text)


def _positive_int_setting(local_name: str, env_name: str, default: int) -> int:
    text = _setting_text(local_name, env_name, "")
    try:
        value = int(text)
    except Exception:
        return int(default)
    return int(value) if value > 0 else int(default)


def _truthy_setting(local_name: str, env_name: str, default: bool = False) -> bool:
    text = _setting_text(local_name, env_name, "1" if default else "0").strip().lower()
    return text in {"1", "true", "yes", "y", "on", "gpu", "cuda", "cupy"}


def should_reenter_guarded_process_runner(
    requested: object,
    *,
    workers: int,
    platform_name: str | None = None,
    process_guard_enabled: bool = False,
) -> bool:
    if platform_name is None:
        platform_name = os.name
    backend = str(requested).strip().lower() or "process"
    return (
        platform_name == "nt"
        and backend in {"auto", "process"}
        and int(workers) > 1
        and not bool(process_guard_enabled)
    )


def guarded_process_runner_command(
    *,
    python_executable: object,
    runner_path: object,
    diagnostic_path: object,
    state_path: object,
    run_name: object,
    fit_workers: int | None = None,
    numba_threads: int | None = None,
    process_numba_threads: int | None = None,
) -> list[str]:
    command = [
        str(python_executable),
        str(runner_path),
        "--notebook",
        str(diagnostic_path),
        "--run-name",
        str(run_name),
        "--fit-backend",
        "process",
    ]
    for flag, value in (
        ("--fit-workers", fit_workers),
        ("--numba-threads", numba_threads),
        ("--process-numba-threads", process_numba_threads),
    ):
        if value is not None:
            command.extend([flag, str(int(value))])
    command.append(str(state_path))
    return command


def normalize_fit_backend(
    requested: object,
    *,
    workers: int,
    platform_name: str | None = None,
    process_guard_enabled: bool = False,
) -> str:
    if platform_name is None:
        platform_name = os.name
    backend = str(requested).strip().lower() or "process"
    if backend not in {"auto", "process", "thread", "serial"}:
        backend = "process"
    if int(workers) <= 1:
        return "serial"
    if platform_name == "nt" and backend in {"auto", "process"} and not process_guard_enabled:
        return "thread"
    if backend == "auto":
        return "process"
    return backend


STATE_PATH = Path(
    _setting_text("GUI_STATE_PATH", "RA_SIM_ALL_BACKGROUND_STATE", DEFAULT_STATE_PATH)
).expanduser()
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnostics.background_peak_fit_worker import (
    configure_peak_fit_worker,
    fit_peak_from_job as fit_peak_from_process_job,
    make_peak_fit_job,
    peak_fit_settings_from_values,
    save_peak_fit_background_arrays,
)

STATE_RUN_NAME = _safe_run_name(
    _setting_text("RUN_NAME", "RA_SIM_ALL_BACKGROUND_RUN_NAME", STATE_PATH.stem)
)
OUT_DIR = Path(
    _setting_text(
        "OUTPUT_DIR",
        "RA_SIM_ALL_BACKGROUND_OUT_DIR",
        ROOT / "artifacts" / "background_peak_fits" / f"{STATE_RUN_NAME}_state",
    )
).expanduser()
if not OUT_DIR.is_absolute():
    OUT_DIR = ROOT / OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_OUT_DIR = Path(
    _setting_text(
        "FIGURE_OUTPUT_DIR",
        "RA_SIM_ALL_BACKGROUND_FIGURE_OUT_DIR",
        r"C:\Users\Kenpo\OneDrive\Documents\GitHub\PhD Work\2D-Manuscript-Draft\figures\results_ordered",
    )
).expanduser()
if not FIGURE_OUT_DIR.is_absolute():
    FIGURE_OUT_DIR = ROOT / FIGURE_OUT_DIR
FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_NAME_OVERRIDE_TEXT = _setting_text(
    "SAMPLE_NAME_OVERRIDE", "RA_SIM_ALL_BACKGROUND_SAMPLE_NAME", ""
)
SAMPLE_NAME = SAMPLE_NAME_OVERRIDE_TEXT or _sample_name_from_path(STATE_PATH) or STATE_RUN_NAME
SAMPLE_LABEL = _sample_label_from_name(SAMPLE_NAME)
SAMPLE_STEM = _safe_run_name(SAMPLE_NAME).lower()


def refresh_figure_stems() -> None:
    """Refresh output file stems after SAMPLE_NAME is detected from the state/CIF."""
    global SAMPLE_STEM
    global FIGURE7A_STEM, FIGURE7C_STEM, ROD_PROFILE_STEM, ROD_PROFILE_REGION_STEM
    global USED_PEAKS_STEM, ROI_EXAMPLES_STEM, BACKGROUND_VS_FIT_STEM_PREFIX

    SAMPLE_STEM = _safe_run_name(SAMPLE_NAME).lower()
    FIGURE7A_STEM = f"figure7a_{SAMPLE_STEM}_detector_peak_fits"
    FIGURE7C_STEM = f"figure7c_{SAMPLE_STEM}_line_profile_fits"
    ROD_PROFILE_STEM = f"figure7_{SAMPLE_STEM}_qr_rod_qz_profiles"
    ROD_PROFILE_REGION_STEM = None  # Filled after the rod-profile incident angle is selected.
    USED_PEAKS_STEM = f"figure7_{SAMPLE_STEM}_used_peaks"
    ROI_EXAMPLES_STEM = f"figure7_{SAMPLE_STEM}_specular_roi_examples"
    BACKGROUND_VS_FIT_STEM_PREFIX = f"figure7_{SAMPLE_STEM}"


FALLBACK_TILT_BY_BACKGROUND = {0: 5.0, 1: 10.0, 2: 15.0}
EXCLUDED_PEAKS_BY_TILT = {
    (10, "0,0,9"),
    (15, "0,0,12"),
    (15, "-1,0,5"),
    (5, "-1,0,7"),
}
MANUSCRIPT_DPI = _positive_int_setting("FIGURE_DPI_OVERRIDE", "BACKGROUND_FIGURE_DPI", 600)
SAVE_VECTOR_FIGURES = _truthy_setting(
    "SAVE_VECTOR_FIGURES_OVERRIDE", "BACKGROUND_SAVE_VECTOR_FIGURES", True
)

PIXEL_SIZE_M = 1.0e-4
WAVELENGTH_M = 1.54e-10
NPT_RADIAL = 1200
NPT_AZIMUTH = 720
THETA_HALF_WINDOW_DEG = 1.8
PHI_HALF_WINDOW_DEG = 6.0
CENTER_THETA_BOUND_DEG = 0.55
CENTER_PHI_BOUND_DEG = 1.6
DETECTOR_RENDER_SIGMA_RADIUS = 8.0
DETECTOR_RENDER_CHUNK_ROWS = 256
GAUSSIAN_TAIL_DISTANCE_WEIGHT = 1.25
GAUSSIAN_CORE_SIGNAL_DOWNSCALE = 0.06
GAUSSIAN_TAIL_OVERPREDICTION_START = 0.55
GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT = 1.75
PEAK_FIT_BACKGROUND_QUANTILE = 45.0
PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS = 1.15
PEAK_FIT_BACKGROUND_IRLS_STEPS = 6
PEAK_FIT_BACKGROUND_HUBER_K = 1.5
PEAK_FIT_MULTISTART_WIDTH_FACTORS = (1.0, 1.8, 2.8)
PEAK_FIT_MAX_NFEV = 1400
ROD_PROFILE_QZ_BINS = 96
SPECULAR_QR_ROD_L_MAX = 8.0
ALLOW_GENERATED_ROD_REFERENCES = _truthy_setting(
    "ALLOW_GENERATED_ROD_REFERENCES_OVERRIDE", "RA_SIM_ALLOW_GENERATED_ROD_REFERENCES", False
)
DETECTOR_ROTATION_MIN_ANCHORS = 4
DETECTOR_ROTATION_MIN_M_GROUPS = 2
refresh_figure_stems()
ROD_PROFILE_MAX_TWO_THETA_DEG = 70.3
ROD_QZ_FORWARD_FIT_CENTER_THETA_BOUND_DEG = 0.45
ROD_QZ_FORWARD_FIT_CENTER_PHI_BOUND_DEG = 1.35
ROD_QZ_FORWARD_FIT_SIGMA_SCALE_BOUNDS = (0.35, 2.25)
ROD_QZ_FORWARD_FIT_ANGLE_BOUND_RAD = math.radians(35.0)
ROD_QZ_FORWARD_FIT_SHAPE_POWER_BOUNDS = (1.6, 6.0)
ROD_QZ_FORWARD_FIT_INITIAL_SHAPE_POWER = 2.6
ROD_QZ_FORWARD_FIT_SUPPORT_SIGMA_RADIUS = 8.0
ROD_QZ_FORWARD_FIT_VALLEY_PARTITION_ENABLED = True
ROD_QZ_FORWARD_FIT_COMPACT_EDGE_BINS = 0.75
ROD_QZ_FORWARD_FIT_MIN_OUTER_SUPPORT_BINS = 3.0
ROD_QZ_FORWARD_FIT_VALLEY_OVERPREDICTION_WEIGHT = 8.0
ROD_QZ_FORWARD_FIT_MAX_NFEV = 900
ROD_QZ_NONLINEAR_REFINEMENT_ENABLED = True
ROD_QZ_NONLINEAR_MAX_COMPONENTS = 14
ROD_QZ_NONLINEAR_CENTER_BOUND_BINS = 4.0
ROD_QZ_NONLINEAR_TAIL_POWER_BOUNDS = (0.55, 12.0)
ROD_QZ_NONLINEAR_MAX_NFEV = 1200
ROD_QZ_NONLINEAR_LOG_RESIDUAL_WEIGHT = 1.0
ROD_QZ_NONLINEAR_LOG_FLOOR_FRACTION = 0.05
CAKED_FIGURE_INTENSITY_MODE = "density"
# Keep background fitting in detector-count density units. Solid-angle correction
# converts to intensity per steradian and amplifies flat detector offsets as sec(2theta)^3.
BACKGROUND_SOLID_ANGLE_CORRECTION = False

# Add one shared linear baseline to fitted peak curves so the fit model is
# peak_sum + y=m*x+b. This is a fit/plot term, not a background subtraction term.
# Peak-subtracted products still subtract only fitted peak density.
PEAK_FIT_SHARED_LINEAR_BASELINE_ENABLED = True
LOCAL_PEAK_LINEAR_BASELINE_AXIS = "two_theta"
ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED = True

PEAK_FIT_WORKER_SETTINGS = peak_fit_settings_from_values(
    THETA_HALF_WINDOW_DEG=THETA_HALF_WINDOW_DEG,
    PHI_HALF_WINDOW_DEG=PHI_HALF_WINDOW_DEG,
    CENTER_THETA_BOUND_DEG=CENTER_THETA_BOUND_DEG,
    CENTER_PHI_BOUND_DEG=CENTER_PHI_BOUND_DEG,
    GAUSSIAN_TAIL_DISTANCE_WEIGHT=GAUSSIAN_TAIL_DISTANCE_WEIGHT,
    GAUSSIAN_CORE_SIGNAL_DOWNSCALE=GAUSSIAN_CORE_SIGNAL_DOWNSCALE,
    GAUSSIAN_TAIL_OVERPREDICTION_START=GAUSSIAN_TAIL_OVERPREDICTION_START,
    GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT=GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT,
    PEAK_FIT_BACKGROUND_QUANTILE=PEAK_FIT_BACKGROUND_QUANTILE,
    PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS=PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS,
    PEAK_FIT_BACKGROUND_IRLS_STEPS=PEAK_FIT_BACKGROUND_IRLS_STEPS,
    PEAK_FIT_BACKGROUND_HUBER_K=PEAK_FIT_BACKGROUND_HUBER_K,
    PEAK_FIT_MULTISTART_WIDTH_FACTORS=PEAK_FIT_MULTISTART_WIDTH_FACTORS,
    PEAK_FIT_MAX_NFEV=PEAK_FIT_MAX_NFEV,
)

CPU_COUNT = max(1, os.cpu_count() or 1)
FIT_WORKERS = max(
    1,
    min(
        _positive_int_setting("FIT_WORKERS_OVERRIDE", "BACKGROUND_FIT_WORKERS", CPU_COUNT),
        CPU_COUNT,
    ),
)
PROFILE_FIT_WORKERS = max(
    1,
    min(
        _positive_int_setting(
            "PROFILE_FIT_WORKERS_OVERRIDE", "BACKGROUND_PROFILE_FIT_WORKERS", FIT_WORKERS
        ),
        CPU_COUNT,
    ),
)
ROD_PROFILE_FIT_WORKERS = max(
    1,
    min(
        _positive_int_setting(
            "ROD_PROFILE_FIT_WORKERS_OVERRIDE", "BACKGROUND_ROD_PROFILE_FIT_WORKERS", FIT_WORKERS
        ),
        CPU_COUNT,
    ),
)
CAKE_WORKERS = max(
    1,
    min(
        _positive_int_setting("CAKE_WORKERS_OVERRIDE", "BACKGROUND_CAKE_WORKERS", CPU_COUNT),
        CPU_COUNT,
    ),
)
NUMBA_WORKERS = max(
    1,
    min(_positive_int_setting("NUMBA_WORKERS_OVERRIDE", "NUMBA_NUM_THREADS", CPU_COUNT), CPU_COUNT),
)
REQUESTED_FIT_BACKEND = (
    _setting_text("FIT_BACKEND_OVERRIDE", "BACKGROUND_FIT_BACKEND", "process").strip().lower()
    or "process"
)
PROCESS_GUARD_ENABLED = _truthy_setting(
    "PROCESS_GUARD_OVERRIDE", "RA_SIM_ALL_BACKGROUND_PROCESS_GUARD", False
)
PROCESS_NUMBA_THREADS = max(
    1,
    min(
        _positive_int_setting(
            "PROCESS_NUMBA_THREADS_OVERRIDE", "BACKGROUND_PROCESS_NUMBA_THREADS", 1
        ),
        CPU_COUNT,
    ),
)
if should_reenter_guarded_process_runner(
    REQUESTED_FIT_BACKEND,
    workers=FIT_WORKERS,
    process_guard_enabled=PROCESS_GUARD_ENABLED,
):
    guarded_command = guarded_process_runner_command(
        python_executable=sys.executable,
        runner_path=Path(__file__).with_name("run_all_background_peak_fits.py").resolve(),
        diagnostic_path=Path(__file__).resolve(),
        state_path=STATE_PATH,
        run_name=STATE_RUN_NAME,
        fit_workers=FIT_WORKERS,
        numba_threads=NUMBA_WORKERS,
        process_numba_threads=PROCESS_NUMBA_THREADS,
    )
    guarded_env = os.environ.copy()
    guarded_env["RA_SIM_ALL_BACKGROUND_PROCESS_GUARD"] = "1"
    print(
        "direct Windows process backend: launching guarded runner "
        f"for {FIT_WORKERS} process workers",
        flush=True,
    )
    guarded_result = subprocess.run(
        guarded_command,
        cwd=str(ROOT),
        env=guarded_env,
        check=False,
    )
    raise SystemExit(int(guarded_result.returncode))
FIT_BACKEND = normalize_fit_backend(
    REQUESTED_FIT_BACKEND,
    workers=FIT_WORKERS,
    process_guard_enabled=PROCESS_GUARD_ENABLED,
)
if REQUESTED_FIT_BACKEND not in {"auto", "process", "thread", "serial"}:
    print(f"unsupported FIT_BACKEND={REQUESTED_FIT_BACKEND!r}; using {FIT_BACKEND!r}")
if FIT_BACKEND != REQUESTED_FIT_BACKEND:
    if os.name == "nt" and REQUESTED_FIT_BACKEND in {"auto", "process"}:
        print(
            "process fit backend is disabled for direct top-level diagnostic scripts on Windows; "
            "using thread backend. Use run_all_background_peak_fits.py --notebook "
            "all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py "
            "--fit-backend process for guarded process workers."
        )
    else:
        print(f"fit backend normalized from {REQUESTED_FIT_BACKEND!r} to {FIT_BACKEND!r}")
EXACT_CAKE_ENGINE = (
    _setting_text("EXACT_CAKE_ENGINE_OVERRIDE", "BACKGROUND_EXACT_CAKE_ENGINE", "numba")
    .strip()
    .lower()
    or "numba"
)
if EXACT_CAKE_ENGINE not in {"auto", "python", "numba"}:
    print(f"unsupported EXACT_CAKE_ENGINE={EXACT_CAKE_ENGINE!r}; using 'numba'")
    EXACT_CAKE_ENGINE = "numba"
if NUMBA_AVAILABLE:
    try:
        set_num_threads(NUMBA_WORKERS)
    except Exception:
        pass

try:
    import cupy as cp  # type: ignore[import-not-found]

    try:
        CUPY_AVAILABLE = bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        CUPY_AVAILABLE = False
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False
GPU_ACCELERATION_REQUESTED = _truthy_setting("USE_GPU_OVERRIDE", "BACKGROUND_USE_GPU", False)
GPU_ACCELERATION_ENABLED = bool(GPU_ACCELERATION_REQUESTED and CUPY_AVAILABLE)
if GPU_ACCELERATION_REQUESTED and not GPU_ACCELERATION_ENABLED:
    print("GPU acceleration requested but CuPy/CUDA is unavailable; using CPU paths")
print(
    "parallel settings: "
    f"fit_workers={FIT_WORKERS} profile_fit_workers={PROFILE_FIT_WORKERS} "
    f"rod_fit_workers={ROD_PROFILE_FIT_WORKERS} numba_threads={NUMBA_WORKERS} "
    f"fit_backend={FIT_BACKEND} process_numba_threads={PROCESS_NUMBA_THREADS} "
    f"process_guard={PROCESS_GUARD_ENABLED} "
    f"cake_engine={EXACT_CAKE_ENGINE} cake_workers={CAKE_WORKERS} "
    f"gpu={GPU_ACCELERATION_ENABLED}"
)
print(
    "fit backend: background peak fits use the selected safe backend; process pool is used only when requested and supported; CuPy is used only for detector Qr/Qz rod-profile accumulation"
)

# No diffuse/radial/lower-envelope background is subtracted in this notebook.
# The only subtraction product generated is measured detector-count density minus fitted peak density.
PEAK_SUBTRACTION_ONLY = True


# PRL-style figure defaults. Matplotlib consumes inches; widths are defined
# from centimeter values to keep the intended journal geometry explicit.
def cm_to_in(value_cm: float) -> float:
    return float(value_cm) / 2.54


PRL_SINGLE_COLUMN_WIDTH_IN = cm_to_in(8.6)
PRL_DOUBLE_COLUMN_WIDTH_IN = cm_to_in(17.8)
JOURNAL_FULL_WIDTH_IN = PRL_DOUBLE_COLUMN_WIDTH_IN
JOURNAL_ATLAS_WIDTH_IN = PRL_DOUBLE_COLUMN_WIDTH_IN
JOURNAL_LINE_WIDTH_PT = 0.75
PRL_SHOW_IN_PANEL_TITLES = False

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
JOURNAL_DATA_COLOR = "0.16"
JOURNAL_FIT_COLOR = OKABE_ITO["vermillion"]
JOURNAL_CONTOUR_COLOR = OKABE_ITO["blue"]
JOURNAL_CENTER_COLOR = OKABE_ITO["orange"]
JOURNAL_GRID_COLOR = "0.88"
JOURNAL_INTENSITY_CMAP = "magma"
JOURNAL_MODEL_CMAP = "cividis"
JOURNAL_DETECTOR_CMAP = "magma"
JOURNAL_REGION_COLORS = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["purple"],
    OKABE_ITO["sky"],
]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 7.6,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.6,
        "figure.titlesize": 8.5,
        "axes.linewidth": 0.65,
        "axes.titlepad": 2.0,
        "axes.unicode_minus": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "legend.frameon": False,
        "legend.handlelength": 1.55,
        "legend.borderpad": 0.28,
        "legend.labelspacing": 0.25,
        "legend.columnspacing": 0.9,
        "image.interpolation": "nearest",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": MANUSCRIPT_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
print(f"state={STATE_PATH}")
print(f"run_name={STATE_RUN_NAME}")
print(f"out={OUT_DIR}")
print(f"figures={FIGURE_OUT_DIR}")
print(
    f"cpu_count={CPU_COUNT} fit_workers={FIT_WORKERS} numba={NUMBA_AVAILABLE} numba_threads={get_num_threads()}"
)


def as_float(value: object, fallback: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    return out if np.isfinite(out) else float(fallback)


def positive_float_or_nan(value: object) -> float:
    out = as_float(value)
    return out if np.isfinite(out) and out > 0.0 else float("nan")


def active_lattice_constants_from_cif_path(cif_path: object) -> dict[str, object]:
    text = str(cif_path or "").strip()
    if not text:
        raise ValueError("primary CIF path is empty")
    path = Path(text).expanduser()
    if not path.exists():
        raise ValueError(f"primary CIF path does not exist: {path}")
    from ra_sim.cli import _parse_cif_cell_a_c

    a_value, c_value = _parse_cif_cell_a_c(str(path))
    a_value = positive_float_or_nan(a_value)
    c_value = positive_float_or_nan(c_value)
    if not (np.isfinite(a_value) and np.isfinite(c_value)):
        raise ValueError(f"primary CIF has invalid lattice constants: {path}")
    return {
        "a": float(a_value),
        "c": float(c_value),
        "source": "primary_cif_path",
        "primary_cif_path": str(path),
    }


def active_lattice_constants_from_state(state_payload: dict[str, object]) -> dict[str, object]:
    variables = state_payload.get("variables", {})
    variables = variables if isinstance(variables, dict) else {}
    files_payload = state_payload.get("files", {})
    files_payload = files_payload if isinstance(files_payload, dict) else {}
    primary_cif = str(files_payload.get("primary_cif_path", "") or "").strip()

    a_value = positive_float_or_nan(variables.get("a_var"))
    c_value = positive_float_or_nan(variables.get("c_var"))
    if np.isfinite(a_value) and np.isfinite(c_value):
        return {
            "a": float(a_value),
            "c": float(c_value),
            "source": "state.variables",
            "primary_cif_path": primary_cif,
        }

    if primary_cif:
        try:
            return active_lattice_constants_from_cif_path(primary_cif)
        except Exception as exc:
            raise ValueError(
                "active lattice constants unavailable: expected positive state variables "
                "a_var/c_var or CIF _cell_length_a/_cell_length_c"
            ) from exc

    raise ValueError(
        "active lattice constants unavailable: expected positive state variables "
        "a_var/c_var or CIF _cell_length_a/_cell_length_c"
    )


def file_cache_signature(path_value: object) -> dict[str, object]:
    text = str(path_value or "").strip()
    if not text:
        return {"path_name": "", "exists": False, "sha256": None, "size": None}
    path = Path(text).expanduser()
    if not path.exists():
        return {"path_name": path.name, "exists": False, "sha256": None, "size": None}
    data = path.read_bytes()
    return {
        "path_name": path.name,
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def active_lattice_cache_signature(state_payload: dict[str, object]) -> dict[str, object]:
    lattice = active_lattice_constants_from_state(state_payload)
    return {
        "a": _cache_normalize_value(lattice.get("a")),
        "c": _cache_normalize_value(lattice.get("c")),
        "source": str(lattice.get("source", "")),
        "primary_cif": file_cache_signature(lattice.get("primary_cif_path")),
    }


def q_group_rows_cache_signature(state_payload: dict[str, object]) -> list[dict[str, object]]:
    geometry = state_payload.get("geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    rows: list[dict[str, object]] = []
    for row in geometry.get("q_group_rows", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "key": _cache_normalize_value(row.get("key", row.get("q_group_key"))),
                "source_label": str(row.get("source_label", "")),
                "structure_role": str(row.get("structure_role", "")),
                "included": bool(row.get("included", True)),
                "gz_index": _cache_normalize_value(row.get("gz_index")),
                "qr": _cache_normalize_value(row.get("qr")),
                "qz": _cache_normalize_value(row.get("qz")),
            }
        )
    return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, default=str))


def angle_sort_value(value: object) -> float:
    out = as_float(value)
    return out if np.isfinite(out) else float("inf")


def angle_key(value: object) -> float | int:
    out = as_float(value)
    if not np.isfinite(out):
        raise ValueError(f"invalid angle value {value!r}")
    rounded = round(out)
    return int(rounded) if abs(out - rounded) < 1.0e-9 else round(float(out), 6)


def format_angle_value(value: object) -> str:
    out = as_float(value)
    if not np.isfinite(out):
        return str(value)
    rounded = round(out)
    if abs(out - rounded) < 1.0e-9:
        return str(int(rounded))
    return f"{out:g}"


def angle_stem(value: object) -> str:
    text = format_angle_value(value).replace("+", "").replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "angle"


def _parse_angle_number(text: str) -> float:
    return float(str(text).replace("p", ".").replace("P", "."))


def _unique_angle_values(values: list[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        if not np.isfinite(value):
            continue
        if not any(abs(value - existing) < 1.0e-9 for existing in unique):
            unique.append(float(value))
    return unique


def parse_incident_angle_from_osc_name(path: object) -> float:
    stem = Path(str(path)).stem
    marked = [
        _parse_angle_number(match.group(1))
        for match in re.finditer(
            r"(?i)(?:^|[_\-\s])([+-]?\d+(?:\.\d+)?)[dD](?=$|[_\-\s])",
            stem,
        )
    ]
    marked_unique = _unique_angle_values(marked)
    if len(marked_unique) == 1:
        return marked_unique[0]
    if len(marked_unique) > 1:
        raise ValueError(
            f"ambiguous incident angle markers in {Path(str(path)).name!r}: {marked_unique}"
        )
    raise ValueError(
        f"could not parse incident angle from OSC file name {Path(str(path)).name!r}; expected token like 4d or 4.5d"
    )


def build_background_tilt_map(background_paths: list[str]) -> dict[int, float]:
    tilt_by_background: dict[int, float] = {}
    fallback_messages = []
    for idx, path_text in enumerate(background_paths):
        try:
            tilt_by_background[idx] = float(parse_incident_angle_from_osc_name(path_text))
        except Exception as exc:
            if idx not in FALLBACK_TILT_BY_BACKGROUND:
                raise
            fallback = float(FALLBACK_TILT_BY_BACKGROUND[idx])
            tilt_by_background[idx] = fallback
            fallback_messages.append(
                f"{Path(path_text).name}: {exc}; fallback={format_angle_value(fallback)} deg"
            )
    if fallback_messages:
        print("incident-angle parse fallbacks:")
        for message in fallback_messages:
            print("  " + message)
    return tilt_by_background


def caked_image_for_intensity_mode(
    image: object,
    *,
    caked_sum_signal: object | None = None,
    caked_sum_normalization: object | None = None,
    caked_count: object | None = None,
    intensity_mode: str = "density",
) -> np.ndarray:
    base = np.asarray(image, dtype=np.float64)
    if base.ndim != 2:
        raise RuntimeError("caked image must be 2D")
    mode = "raw_sum" if str(intensity_mode) == "raw_sum" else "density"
    signal = None if caked_sum_signal is None else np.asarray(caked_sum_signal, dtype=np.float64)
    normalization = (
        None
        if caked_sum_normalization is None
        else np.asarray(caked_sum_normalization, dtype=np.float64)
    )
    if (signal is None) != (normalization is None):
        raise RuntimeError("caked sum_signal and sum_normalization fields must be paired")
    if signal is not None and signal.shape != base.shape:
        raise RuntimeError("caked sum_signal shape mismatch")
    if normalization is not None and normalization.shape != base.shape:
        raise RuntimeError("caked sum_normalization shape mismatch")

    if mode == "raw_sum":
        if signal is not None:
            return signal.copy()
        if caked_count is not None:
            count = np.asarray(caked_count, dtype=np.float64)
            if count.shape != base.shape:
                raise RuntimeError("caked count shape mismatch")
            return base * count
        return base.copy()

    if signal is None or normalization is None:
        return base.copy()
    density = np.full(base.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(signal) & np.isfinite(normalization) & (normalization > 0.0)
    density[valid] = signal[valid] / normalization[valid]
    return density


@njit(fastmath=True, nogil=True)
def _wrapped_delta_deg_scalar_numba(value: float, center: float) -> float:
    return ((value - center + 180.0) % 360.0) - 180.0


def wrapped_delta_deg(values: object, center: float) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - float(center) + 180.0) % 360.0) - 180.0


GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493
FIT_MODEL_NAME = "rotated_gaussian_core_lorentzian_tail_shared_center"


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_value_numba(
    params: np.ndarray, theta: float, phi: float, peak_only: bool
) -> float:
    amp = params[0]
    theta0 = params[1]
    phi0 = params[2]
    sigma_g_u = max(params[3] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    sigma_g_v = max(params[4] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    angle = params[5]
    dt = theta - theta0
    dp = _wrapped_delta_deg_scalar_numba(phi, phi0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u = cos_a * dt + sin_a * dp
    v = -sin_a * dt + cos_a * dp
    gaussian_r2 = (u / sigma_g_u) * (u / sigma_g_u) + (v / sigma_g_v) * (v / sigma_g_v)
    gaussian_core = math.exp(-0.5 * gaussian_r2)
    gamma_l_u = max(0.5 * params[9], 1.0e-12)
    gamma_l_v = max(0.5 * params[10], 1.0e-12)
    lorentzian_tail = 1.0 / (
        1.0 + (u / gamma_l_u) * (u / gamma_l_u) + (v / gamma_l_v) * (v / gamma_l_v)
    )
    eta_tail = params[11]
    if eta_tail < 0.0:
        eta_tail = 0.0
    elif eta_tail > 0.95:
        eta_tail = 0.95
    peak = amp * ((1.0 - eta_tail) * gaussian_core + eta_tail * lorentzian_tail)
    if peak_only:
        return peak
    return params[6] + params[7] * dt + params[8] * dp + peak


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_value_from_matrix_numba(
    params_matrix: np.ndarray, peak_idx: int, theta: float, phi: float, peak_only: bool
) -> float:
    amp = params_matrix[peak_idx, 0]
    theta0 = params_matrix[peak_idx, 1]
    phi0 = params_matrix[peak_idx, 2]
    sigma_g_u = max(params_matrix[peak_idx, 3] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    sigma_g_v = max(params_matrix[peak_idx, 4] / GAUSSIAN_FWHM_TO_SIGMA, 1.0e-12)
    angle = params_matrix[peak_idx, 5]
    dt = theta - theta0
    dp = _wrapped_delta_deg_scalar_numba(phi, phi0)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    u = cos_a * dt + sin_a * dp
    v = -sin_a * dt + cos_a * dp
    gaussian_r2 = (u / sigma_g_u) * (u / sigma_g_u) + (v / sigma_g_v) * (v / sigma_g_v)
    gaussian_core = math.exp(-0.5 * gaussian_r2)
    gamma_l_u = max(0.5 * params_matrix[peak_idx, 9], 1.0e-12)
    gamma_l_v = max(0.5 * params_matrix[peak_idx, 10], 1.0e-12)
    lorentzian_tail = 1.0 / (
        1.0 + (u / gamma_l_u) * (u / gamma_l_u) + (v / gamma_l_v) * (v / gamma_l_v)
    )
    eta_tail = params_matrix[peak_idx, 11]
    if eta_tail < 0.0:
        eta_tail = 0.0
    elif eta_tail > 0.95:
        eta_tail = 0.95
    peak = amp * ((1.0 - eta_tail) * gaussian_core + eta_tail * lorentzian_tail)
    if peak_only:
        return peak
    return (
        params_matrix[peak_idx, 6]
        + params_matrix[peak_idx, 7] * dt
        + params_matrix[peak_idx, 8] * dp
        + peak
    )


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_plane_points_numba(
    params: np.ndarray, theta_values: np.ndarray, phi_values: np.ndarray
) -> np.ndarray:
    n = theta_values.size
    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        out[idx] = _rotated_gaussian_value_numba(params, theta_values[idx], phi_values[idx], False)
    return out


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_peak_points_numba(
    params: np.ndarray, theta_values: np.ndarray, phi_values: np.ndarray
) -> np.ndarray:
    n = theta_values.size
    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        out[idx] = _rotated_gaussian_value_numba(params, theta_values[idx], phi_values[idx], True)
    return out


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_residual_points_numba(
    params: np.ndarray,
    theta_values: np.ndarray,
    phi_values: np.ndarray,
    y_values: np.ndarray,
    tail_weight: np.ndarray,
    denominator: np.ndarray,
    tail_overprediction_weight: np.ndarray,
) -> np.ndarray:
    n = theta_values.size
    out = np.empty(n, dtype=np.float64)
    for idx in range(n):
        model = _rotated_gaussian_value_numba(params, theta_values[idx], phi_values[idx], False)
        residual = model - y_values[idx]
        if residual > 0.0:
            residual *= tail_overprediction_weight[idx]
        out[idx] = tail_weight[idx] * residual / denominator[idx]
    return out


@njit(fastmath=True, nogil=True)
def _rotated_gaussian_grid_numba(
    params: np.ndarray, theta_axis: np.ndarray, phi_axis: np.ndarray, peak_only: bool
) -> np.ndarray:
    n_phi = phi_axis.size
    n_theta = theta_axis.size
    out = np.empty((n_phi, n_theta), dtype=np.float64)
    for row in range(n_phi):
        phi = phi_axis[row]
        for col in range(n_theta):
            out[row, col] = _rotated_gaussian_value_numba(params, theta_axis[col], phi, peak_only)
    return out


@njit(parallel=True, nogil=True)
def _render_detector_peak_model_full_sum_numba(
    theta_map: np.ndarray,
    phi_map: np.ndarray,
    params_matrix: np.ndarray,
) -> np.ndarray:
    height, width = theta_map.shape
    n_peaks = params_matrix.shape[0]
    out = np.zeros((height, width), dtype=np.float32)
    for row in prange(height):
        for col in range(width):
            theta = theta_map[row, col]
            phi = phi_map[row, col]
            if not (np.isfinite(theta) and np.isfinite(phi)):
                continue
            acc = 0.0
            for peak_idx in range(n_peaks):
                acc += _rotated_gaussian_value_from_matrix_numba(
                    params_matrix, peak_idx, theta, phi, True
                )
            out[row, col] = np.float32(acc)
    return out


@njit(parallel=True, nogil=True)
def _render_detector_peak_model_numba(
    theta_map: np.ndarray,
    phi_map: np.ndarray,
    params_matrix: np.ndarray,
    theta_radii: np.ndarray,
    phi_radii: np.ndarray,
) -> np.ndarray:
    height, width = theta_map.shape
    n_peaks = params_matrix.shape[0]
    out = np.zeros((height, width), dtype=np.float32)
    for row in prange(height):
        for col in range(width):
            theta = theta_map[row, col]
            phi = phi_map[row, col]
            if not (np.isfinite(theta) and np.isfinite(phi)):
                continue
            acc = 0.0
            for peak_idx in range(n_peaks):
                theta0 = params_matrix[peak_idx, 1]
                phi0 = params_matrix[peak_idx, 2]
                if (
                    abs(theta - theta0) <= theta_radii[peak_idx]
                    and abs(_wrapped_delta_deg_scalar_numba(phi, phi0)) <= phi_radii[peak_idx]
                ):
                    acc += _rotated_gaussian_value_from_matrix_numba(
                        params_matrix, peak_idx, theta, phi, True
                    )
            out[row, col] = np.float32(acc)
    return out


@njit(fastmath=True, nogil=True)
def _gaussian_lorentzian_profile_points_numba(
    params: np.ndarray, x_values: np.ndarray
) -> np.ndarray:
    amp = params[0]
    center = params[1]
    sigma_g = max(params[2], 1.0e-12)
    gamma_l = max(params[3], 1.0e-12)
    gaussian_fraction = params[4]
    if gaussian_fraction < 0.0:
        gaussian_fraction = 0.0
    elif gaussian_fraction > 1.0:
        gaussian_fraction = 1.0
    baseline = params[5]
    slope = params[6]
    out = np.empty(x_values.size, dtype=np.float64)
    for idx in range(x_values.size):
        dx = x_values[idx] - center
        gaussian = math.exp(-0.5 * (dx / sigma_g) * (dx / sigma_g))
        lorentzian = 1.0 / (1.0 + (dx / gamma_l) * (dx / gamma_l))
        out[idx] = (
            baseline
            + slope * dx
            + amp * (gaussian_fraction * gaussian + (1.0 - gaussian_fraction) * lorentzian)
        )
    return out


@njit(fastmath=True, nogil=True)
def _gaussian_lorentzian_residual_numba(
    params: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    tail_weight: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    model = _gaussian_lorentzian_profile_points_numba(params, x_values)
    out = np.empty(x_values.size, dtype=np.float64)
    for idx in range(x_values.size):
        out[idx] = tail_weight[idx] * (model[idx] - y_values[idx]) / denominator[idx]
    return out


def rotated_gaussian_plane(
    params: np.ndarray, theta_grid: np.ndarray, phi_grid: np.ndarray
) -> np.ndarray:
    theta_arr = np.asarray(theta_grid, dtype=np.float64)
    phi_arr = np.asarray(phi_grid, dtype=np.float64)
    if theta_arr.shape == phi_arr.shape:
        out = _rotated_gaussian_plane_points_numba(
            np.asarray(params, dtype=np.float64),
            np.ascontiguousarray(theta_arr.ravel()),
            np.ascontiguousarray(phi_arr.ravel()),
        )
        return out.reshape(theta_arr.shape)
    theta_b, phi_b = np.broadcast_arrays(theta_arr, phi_arr)
    out = _rotated_gaussian_plane_points_numba(
        np.asarray(params, dtype=np.float64),
        np.ascontiguousarray(theta_b.ravel()),
        np.ascontiguousarray(phi_b.ravel()),
    )
    return out.reshape(theta_b.shape)


def rotated_gaussian_peak_only(
    params: np.ndarray, theta_grid: np.ndarray, phi_grid: np.ndarray
) -> np.ndarray:
    theta_arr = np.asarray(theta_grid, dtype=np.float64)
    phi_arr = np.asarray(phi_grid, dtype=np.float64)
    if theta_arr.shape == phi_arr.shape:
        out = _rotated_gaussian_peak_points_numba(
            np.asarray(params, dtype=np.float64),
            np.ascontiguousarray(theta_arr.ravel()),
            np.ascontiguousarray(phi_arr.ravel()),
        )
        return out.reshape(theta_arr.shape)
    theta_b, phi_b = np.broadcast_arrays(theta_arr, phi_arr)
    out = _rotated_gaussian_peak_points_numba(
        np.asarray(params, dtype=np.float64),
        np.ascontiguousarray(theta_b.ravel()),
        np.ascontiguousarray(phi_b.ravel()),
    )
    return out.reshape(theta_b.shape)


def gaussian_plane(params: np.ndarray, theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    return rotated_gaussian_plane(params, theta_grid, phi_grid)


def gaussian_peak_only(
    params: np.ndarray, theta_grid: np.ndarray, phi_grid: np.ndarray
) -> np.ndarray:
    return rotated_gaussian_peak_only(params, theta_grid, phi_grid)


def render_detector_peak_model(
    theta_map: np.ndarray, phi_map: np.ndarray, fit_results: list[dict[str, object]]
) -> np.ndarray:
    if not fit_results:
        return np.zeros(np.asarray(theta_map).shape, dtype=np.float32)
    params_matrix = np.ascontiguousarray(
        [np.asarray(item["params"], dtype=np.float64) for item in fit_results], dtype=np.float64
    )
    theta_radii = np.empty(params_matrix.shape[0], dtype=np.float64)
    phi_radii = np.empty(params_matrix.shape[0], dtype=np.float64)
    for idx, p in enumerate(params_matrix):
        sigma_g_u = float(p[3]) / GAUSSIAN_FWHM_TO_SIGMA
        sigma_g_v = float(p[4]) / GAUSSIAN_FWHM_TO_SIGMA
        theta_radii[idx] = max(
            THETA_HALF_WINDOW_DEG, DETECTOR_RENDER_SIGMA_RADIUS * sigma_g_u, 3.0 * float(p[9])
        )
        phi_radii[idx] = max(
            PHI_HALF_WINDOW_DEG, DETECTOR_RENDER_SIGMA_RADIUS * sigma_g_v, 3.0 * float(p[10])
        )
    return _render_detector_peak_model_numba(
        np.ascontiguousarray(theta_map, dtype=np.float64),
        np.ascontiguousarray(phi_map, dtype=np.float64),
        params_matrix,
        theta_radii,
        phi_radii,
    )


def render_detector_peak_model_full_sum(
    theta_map: np.ndarray, phi_map: np.ndarray, fit_results: list[dict[str, object]]
) -> np.ndarray:
    if not fit_results:
        return np.zeros(np.asarray(theta_map).shape, dtype=np.float32)
    params_matrix = np.ascontiguousarray(
        [np.asarray(item["params"], dtype=np.float64) for item in fit_results], dtype=np.float64
    )
    return _render_detector_peak_model_full_sum_numba(
        np.ascontiguousarray(theta_map, dtype=np.float64),
        np.ascontiguousarray(phi_map, dtype=np.float64),
        params_matrix,
    )


def warm_numba_kernels() -> None:
    if not NUMBA_AVAILABLE:
        return
    try:
        p = np.array(
            [10.0, 1.0, 0.0, 0.4, 1.2, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.25], dtype=np.float64
        )
        theta = np.linspace(0.8, 1.2, 8, dtype=np.float64)
        phi = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        y = np.ones(8, dtype=np.float64)
        _rotated_gaussian_grid_numba(p, theta, phi, False)
        _rotated_gaussian_residual_points_numba(p, theta, phi, y, y, y, y)
        _render_detector_peak_model_full_sum_numba(
            theta.reshape(2, 4), phi.reshape(2, 4), p.reshape(1, -1)
        )
        _render_detector_peak_model_numba(
            theta.reshape(2, 4),
            phi.reshape(2, 4),
            p.reshape(1, -1),
            np.array([1.0]),
            np.array([1.0]),
        )
        gp = np.array([1.0, 0.0, 0.4, 0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        _gaussian_lorentzian_residual_numba(gp, phi, y, y, y)
    except Exception as exc:
        print(f"numba warm-up skipped: {exc}")


warm_numba_kernels()


def robust_display_limits(
    image: np.ndarray, low: float = 1.0, high: float = 99.7
) -> tuple[float, float]:
    values = np.ma.asarray(image, dtype=np.float64).compressed()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def hk0_00l_region_crop_bounds(
    image_shape: tuple[int, int],
    beam_center: tuple[float, float],
    region_mask: object,
    *,
    lateral_half_width_px: int = 48,
) -> tuple[slice, slice] | None:
    """Return detector row/column slices spanning the visible 00L mask top to the beam center."""
    if len(image_shape) < 2:
        return None
    height = int(image_shape[0])
    width = int(image_shape[1])
    if height <= 0 or width <= 0:
        return None
    try:
        beam_x, beam_y = float(beam_center[0]), float(beam_center[1])
    except Exception:
        return None
    if not all(np.isfinite(value) for value in (beam_x, beam_y)):
        return None
    if region_mask is None:
        return None
    mask = np.asarray(region_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape != (height, width):
        return None
    valid_rows = np.flatnonzero(np.any(mask, axis=1))
    if valid_rows.size < 1:
        return None
    valid_rows = valid_rows[valid_rows <= int(math.floor(beam_y))]
    if valid_rows.size < 1:
        return None
    lateral = max(int(lateral_half_width_px), 1)
    row_start = max(0, int(np.min(valid_rows)))
    row_stop = min(height, int(math.ceil(beam_y)) + 1)
    col_start = max(0, int(math.floor(beam_x - lateral)))
    col_stop = min(width, int(math.ceil(beam_x + lateral)) + 1)
    if row_start >= row_stop or col_start >= col_stop:
        return None
    return slice(row_start, row_stop), slice(col_start, col_stop)


def save_hk0_00l_region_crop(
    detector_image: np.ndarray,
    output_path: Path | str,
    *,
    horizontal_output_path: Path | str | None = None,
    beam_center: tuple[float, float],
    region_mask: object,
    lateral_half_width_px: int = 48,
) -> tuple[Path, Path | None] | None:
    """Save the 00L detector region as colored log-scale vertical and horizontal PNGs."""
    image = np.ma.asarray(detector_image, dtype=np.float64)
    bounds = hk0_00l_region_crop_bounds(
        tuple(image.shape),
        beam_center,
        region_mask,
        lateral_half_width_px=lateral_half_width_px,
    )
    if bounds is None:
        return None
    row_slice, col_slice = bounds
    crop = np.asarray(image[row_slice, col_slice].filled(np.nan), dtype=np.float64)
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return None
    display_crop = detector_intensity_display(crop)
    norm = detector_log_norm([display_crop])
    rgba_crop = detector_display_cmap()(norm(display_crop))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, rgba_crop, origin="upper")
    horizontal_path = None
    if horizontal_output_path is not None:
        horizontal_path = Path(horizontal_output_path)
        horizontal_path.parent.mkdir(parents=True, exist_ok=True)
        horizontal_rgba_crop = np.rot90(rgba_crop, k=1)
        plt.imsave(horizontal_path, horizontal_rgba_crop, origin="upper")
    return path, horizontal_path


def label_from_entry(entry: dict[str, object]) -> str:
    hkl = entry.get("hkl")
    if isinstance(hkl, (list, tuple, np.ndarray)) and len(hkl) >= 3:
        return f"{int(hkl[0])},{int(hkl[1])},{int(hkl[2])}"
    return str(entry.get("label", "unknown"))


def branch_from_phi(phi: float) -> str:
    return "+" if float(phi) >= 0.0 else "-"


def compact_json_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, separators=(",", ":"))
    except TypeError:
        return str(value)


def detector_xy_from_entry(entry: dict[str, object]) -> tuple[float, float] | None:
    for col_key, row_key in (
        ("background_detector_col", "background_detector_row"),
        ("detector_col", "detector_row"),
        ("x", "y"),
        ("fit_detector_col", "fit_detector_row"),
    ):
        col = as_float(entry.get(col_key))
        row = as_float(entry.get(row_key))
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    return None


def nearest_detector_angles_from_entry(
    entry: dict[str, object], theta_map: np.ndarray, phi_map: np.ndarray
) -> tuple[float, float] | None:
    theta = as_float(entry.get("background_two_theta_deg"))
    phi = as_float(entry.get("background_phi_deg"))
    if np.isfinite(theta) and np.isfinite(phi):
        return float(theta), float(phi)
    xy = detector_xy_from_entry(entry)
    if xy is None:
        return None
    col, row = xy
    row_idx = int(np.clip(np.rint(row), 0, theta_map.shape[0] - 1))
    col_idx = int(np.clip(np.rint(col), 0, theta_map.shape[1] - 1))
    theta = float(theta_map[row_idx, col_idx])
    phi = float(phi_map[row_idx, col_idx])
    if not np.isfinite(theta) or not np.isfinite(phi):
        return None
    return theta, phi


def branch_sort_key(label: str) -> tuple[int, int, int, str]:
    try:
        parts = [int(part.strip()) for part in label.split(",")]
        if len(parts) >= 3:
            return abs(parts[0]) + abs(parts[1]), abs(parts[2]), parts[2], label
    except Exception:
        pass
    return 999, 999, 999, label


def background_peak_entries_from_manual_pairs(
    state_payload: dict[str, object],
    *,
    background_files: list[str],
    background_tilt_deg: dict[int, float],
    sample_name: str,
    excluded_peaks_by_tilt_normalized: set[tuple[object, str]],
) -> dict[int, list[dict[str, object]]]:
    entries_by_bg: dict[int, list[dict[str, object]]] = {
        idx: [] for idx in range(len(background_files))
    }
    geometry = state_payload.get("geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    for group in geometry.get("manual_pairs", []) or []:
        if not isinstance(group, dict):
            continue
        bg_idx = int(group.get("background_index", -1))
        if bg_idx not in entries_by_bg:
            continue
        for entry in group.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            theta = as_float(entry.get("background_two_theta_deg"))
            phi = as_float(entry.get("background_phi_deg"))
            has_caked_seed = np.isfinite(theta) and np.isfinite(phi)
            has_detector_seed = detector_xy_from_entry(entry) is not None
            if not (has_caked_seed or has_detector_seed):
                continue
            e = dict(entry)
            e["_background_index"] = bg_idx
            e["_background_name"] = Path(background_files[bg_idx]).name
            e["_sample_name"] = sample_name
            e["_tilt_deg"] = float(background_tilt_deg[bg_idx])
            e["_display_label"] = (
                f"{sample_name} {format_angle_value(background_tilt_deg[bg_idx])} deg"
            )
            e["_label"] = label_from_entry(e)
            e["_branch"] = branch_from_phi(phi) if np.isfinite(phi) else None
            if (angle_key(e["_tilt_deg"]), e["_label"]) in excluded_peaks_by_tilt_normalized:
                continue
            entries_by_bg[bg_idx].append(e)
    return entries_by_bg


def background_peak_entries_from_peak_records(
    state_payload: dict[str, object],
    *,
    background_files: list[str],
    background_tilt_deg: dict[int, float],
    sample_name: str,
    excluded_peaks_by_tilt_normalized: set[tuple[object, str]],
) -> dict[int, list[dict[str, object]]]:
    entries_by_bg: dict[int, list[dict[str, object]]] = {
        idx: [] for idx in range(len(background_files))
    }
    geometry = state_payload.get("geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    caked_seed_keys = (
        ("background_two_theta_deg", "background_phi_deg"),
        ("two_theta_deg", "phi_deg"),
        ("caked_x", "caked_y"),
    )
    for record in geometry.get("peak_records", []) or []:
        if not isinstance(record, dict):
            continue
        raw_bg_idx = record.get("background_index")
        has_background_index = raw_bg_idx is not None and str(raw_bg_idx).strip() != ""
        bg_idx = None
        if has_background_index:
            try:
                bg_idx = int(raw_bg_idx)
            except Exception:
                continue
            if bg_idx not in entries_by_bg:
                continue

        caked_seed: tuple[float, float] | None = None
        for theta_key, phi_key in caked_seed_keys:
            theta = as_float(record.get(theta_key))
            phi = as_float(record.get(phi_key))
            if np.isfinite(theta) and np.isfinite(phi):
                caked_seed = (float(theta), float(phi))
                break

        base = dict(record)
        if caked_seed is not None:
            theta, phi = caked_seed
            base["background_two_theta_deg"] = float(theta)
            base["background_phi_deg"] = float(phi)
            base.setdefault("caked_x", float(theta))
            base.setdefault("caked_y", float(phi))
            target_backgrounds = [int(bg_idx)] if bg_idx is not None else list(entries_by_bg)
        else:
            xy = detector_xy_from_entry(base)
            if xy is None:
                col = as_float(record.get("native_col"))
                row = as_float(record.get("native_row"))
                if np.isfinite(col) and np.isfinite(row):
                    xy = (float(col), float(row))
            if xy is None or bg_idx is None:
                continue
            col, row = xy
            base.setdefault("detector_col", float(col))
            base.setdefault("detector_row", float(row))
            target_backgrounds = [int(bg_idx)]

        base.setdefault("selection_reason", "peak_records_fallback")
        for target_bg_idx in target_backgrounds:
            e = dict(base)
            e["_background_index"] = int(target_bg_idx)
            e["_background_name"] = Path(background_files[int(target_bg_idx)]).name
            e["_sample_name"] = sample_name
            e["_tilt_deg"] = float(background_tilt_deg[int(target_bg_idx)])
            e["_display_label"] = (
                f"{sample_name} {format_angle_value(background_tilt_deg[int(target_bg_idx)])} deg"
            )
            e["_label"] = label_from_entry(e)
            phi = as_float(e.get("background_phi_deg"))
            e["_branch"] = branch_from_phi(phi) if np.isfinite(phi) else None
            if (angle_key(e["_tilt_deg"]), e["_label"]) in excluded_peaks_by_tilt_normalized:
                continue
            entries_by_bg[int(target_bg_idx)].append(e)
    return entries_by_bg


def background_peak_entries_from_state(
    state_payload: dict[str, object],
    *,
    background_files: list[str],
    background_tilt_deg: dict[int, float],
    sample_name: str,
    excluded_peaks_by_tilt_normalized: set[tuple[object, str]],
) -> tuple[dict[int, list[dict[str, object]]], str]:
    manual_entries = background_peak_entries_from_manual_pairs(
        state_payload,
        background_files=background_files,
        background_tilt_deg=background_tilt_deg,
        sample_name=sample_name,
        excluded_peaks_by_tilt_normalized=excluded_peaks_by_tilt_normalized,
    )
    if sum(len(entries) for entries in manual_entries.values()) > 0:
        return manual_entries, "manual_pairs"
    peak_record_entries = background_peak_entries_from_peak_records(
        state_payload,
        background_files=background_files,
        background_tilt_deg=background_tilt_deg,
        sample_name=sample_name,
        excluded_peaks_by_tilt_normalized=excluded_peaks_by_tilt_normalized,
    )
    if sum(len(entries) for entries in peak_record_entries.values()) > 0:
        return peak_record_entries, "peak_records_fallback"
    return peak_record_entries, "none"


state_doc = json.loads(STATE_PATH.read_text(encoding="utf-8"))
state = state_doc.get("state", state_doc)
files = state.get("files", {})
variables = state.get("variables", {}) if isinstance(state.get("variables"), dict) else {}
flags = state.get("flags", {}) if isinstance(state.get("flags"), dict) else {}
background_files = [str(path) for path in files.get("background_files", [])]
if not background_files:
    raise ValueError("all.json has no background files")
BACKGROUND_TILT_DEG = build_background_tilt_map(background_files)
EXCLUDED_PEAKS_BY_TILT_NORMALIZED = {
    (angle_key(tilt), label) for tilt, label in EXCLUDED_PEAKS_BY_TILT
}

center_row_px = as_float(variables.get("center_x_var"), 0.0)
center_col_px = as_float(variables.get("center_y_var"), 0.0)
distance_m = as_float(variables.get("corto_detector_var"), 0.075)
backend_rotation_k = int(flags.get("background_backend_rotation_k", 3) or 0)
backend_flip_x = bool(flags.get("background_backend_flip_x", False))
backend_flip_y = bool(flags.get("background_backend_flip_y", False))
primary_cif_path = files.get("primary_cif_path")
detected_sample_name = (
    SAMPLE_NAME_OVERRIDE_TEXT or _sample_name_from_path(primary_cif_path) or STATE_RUN_NAME
)
if detected_sample_name:
    SAMPLE_NAME = str(detected_sample_name)
    SAMPLE_LABEL = _sample_label_from_name(SAMPLE_NAME)
refresh_figure_stems()
ACTIVE_LATTICE = active_lattice_constants_from_state(state)
ACTIVE_LATTICE_A = float(ACTIVE_LATTICE["a"])
ACTIVE_LATTICE_C = float(ACTIVE_LATTICE["c"])
ACTIVE_LATTICE_CACHE_SIGNATURE = active_lattice_cache_signature(state)
Q_GROUP_ROWS_CACHE_SIGNATURE = q_group_rows_cache_signature(state)
ROD_REFERENCE_POLICY_SIGNATURE = {
    "allow_generated": bool(ALLOW_GENERATED_ROD_REFERENCES),
    "detector_rotation_min_anchors": int(DETECTOR_ROTATION_MIN_ANCHORS),
    "detector_rotation_min_m_groups": int(DETECTOR_ROTATION_MIN_M_GROUPS),
}
try:
    n2_value = IndexofRefraction(WAVELENGTH_M)
except Exception:
    n2_value = complex(1.0, 0.0)

qr_rod_delta_qr_source = as_float(
    (state.get("analysis_range", {}) if isinstance(state.get("analysis_range"), dict) else {}).get(
        "delta_qr"
    ),
    as_float(variables.get("delta_qr_var"), 0.25),
)
QR_ROD_DELTA_QR_SCALE = 0.85
qr_rod_delta_qr = max(1.0e-9, float(qr_rod_delta_qr_source) * float(QR_ROD_DELTA_QR_SCALE))
IS_PBI2_SAMPLE_STEM = SAMPLE_STEM.startswith("pbi2")
PBI2_DISABLE_BACKGROUND_SUBTRACTION = IS_PBI2_SAMPLE_STEM and _truthy_setting(
    "PBI2_DISABLE_BACKGROUND_SUBTRACTION_OVERRIDE",
    "RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION",
    False,
)
QR_ROD_TRANSVERSE_BACKGROUND_ENABLED = (
    not bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION)
    and (
        IS_PBI2_SAMPLE_STEM
        or _truthy_setting(
            "QR_ROD_TRANSVERSE_BACKGROUND_OVERRIDE",
            "RA_SIM_QR_ROD_TRANSVERSE_BACKGROUND",
            False,
        )
    )
)
QR_ROD_BG_SIDE_BAND_INNER_SCALE = 1.30
QR_ROD_BG_SIDE_BAND_OUTER_SCALE = 2.80
QR_ROD_BG_MIN_SIDE_PIXELS = 8
QR_ROD_BG_PERCENTILE = 50.0
PBI2_PLOT_PEAK_TO_DATA_CANCEL_RATIO = 4.0
PBI2_PLOT_BASELINE_TO_PEAK_CANCEL_RATIO = 0.50
PBI2_ROD_PROFILE_L_AXIS_MAX = 3.0
ROD_PROFILE_BACKGROUND_POLICY_SIGNATURE = {
    "pbi2_disable_background_subtraction": bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION),
    "qr_rod_transverse_background": bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED),
    "qr_rod_bg_side_band_inner_scale": float(QR_ROD_BG_SIDE_BAND_INNER_SCALE),
    "qr_rod_bg_side_band_outer_scale": float(QR_ROD_BG_SIDE_BAND_OUTER_SCALE),
    "qr_rod_bg_min_side_pixels": int(QR_ROD_BG_MIN_SIDE_PIXELS),
    "qr_rod_bg_percentile": float(QR_ROD_BG_PERCENTILE),
}
rod_phi_samples = max(361, int(as_float(variables.get("rod_points_per_gz_var"), 721)))
psi_deg = as_float(variables.get("psi_var"), 0.0)


entries_by_bg, background_peak_entry_source = background_peak_entries_from_state(
    state,
    background_files=background_files,
    background_tilt_deg=BACKGROUND_TILT_DEG,
    sample_name=SAMPLE_NAME,
    excluded_peaks_by_tilt_normalized=EXCLUDED_PEAKS_BY_TILT_NORMALIZED,
)

NO_BACKGROUND_PEAK_ENTRIES_MESSAGE = (
    "fit validation failed: no background peak entries found in geometry.manual_pairs or usable "
    "geometry.peak_records; save background peak matches or peak records before running this notebook"
)
expected_fit_count = sum(len(v) for v in entries_by_bg.values())
print(
    f"backgrounds={len(background_files)} points={expected_fit_count} "
    f"entry_source={background_peak_entry_source}"
)
print(
    f"active lattice: a={ACTIVE_LATTICE_A:.6g} A c={ACTIVE_LATTICE_C:.6g} A "
    f"source={ACTIVE_LATTICE.get('source', '')}"
)
print(
    "excluded peaks:",
    ", ".join(
        f"{format_angle_value(tilt)} deg ({label})"
        for tilt, label in sorted(
            EXCLUDED_PEAKS_BY_TILT, key=lambda item: angle_sort_value(item[0])
        )
    ),
)
print(
    f"orientation flip_x={backend_flip_x} flip_y={backend_flip_y} rotation_k={backend_rotation_k} (3/-1 = 90 deg CW)"
)
for idx, path in enumerate(background_files):
    print(
        f"{SAMPLE_NAME} {format_angle_value(BACKGROUND_TILT_DEG[idx])} deg: {Path(path).name} points={len(entries_by_bg[idx])}"
    )
if expected_fit_count == 0:
    raise RuntimeError(NO_BACKGROUND_PEAK_ENTRIES_MESSAGE)


def robust_peak_background_plane(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    roi: np.ndarray,
    finite: np.ndarray,
    *,
    theta_ref: float,
    phi_ref: float,
) -> np.ndarray:
    """Robust local 2D baseline plane from pixels outside the peak core."""

    if np.count_nonzero(finite) < 3:
        fill = float(np.nanmedian(roi[finite])) if np.any(finite) else 0.0
        return np.asarray([fill, 0.0, 0.0], dtype=np.float64)

    theta_delta_grid = np.asarray(theta_grid, dtype=np.float64) - float(theta_ref)
    phi_delta_grid = wrapped_delta_deg(phi_grid, float(phi_ref))
    center_theta_bound = float(CENTER_THETA_BOUND_DEG)
    center_phi_bound = float(CENTER_PHI_BOUND_DEG)
    core_radius = np.sqrt(
        (theta_delta_grid / max(center_theta_bound, 1.0e-9)) ** 2
        + (phi_delta_grid / max(center_phi_bound, 1.0e-9)) ** 2
    )
    off_peak = finite & (core_radius >= float(PEAK_FIT_BACKGROUND_EXCLUSION_RADIUS))
    minimum = max(12, int(0.20 * np.count_nonzero(finite)))
    if np.count_nonzero(off_peak) < minimum:
        cutoff = float(np.nanpercentile(roi[finite], float(PEAK_FIT_BACKGROUND_QUANTILE)))
        off_peak = finite & (roi <= cutoff)
    if np.count_nonzero(off_peak) < 3:
        off_peak = finite

    theta_fit = np.asarray(theta_delta_grid[off_peak], dtype=np.float64)
    phi_fit = np.asarray(phi_delta_grid[off_peak], dtype=np.float64)
    y_fit = np.asarray(roi[off_peak], dtype=np.float64)
    design = np.column_stack([np.ones_like(theta_fit), theta_fit, phi_fit])

    try:
        coef, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    except Exception:
        coef = np.asarray([float(np.nanmedian(y_fit)), 0.0, 0.0], dtype=np.float64)

    coef = np.asarray(coef, dtype=np.float64)
    for _ in range(int(PEAK_FIT_BACKGROUND_IRLS_STEPS)):
        residual = y_fit - design @ coef[:3]
        sigma = 1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        threshold = float(PEAK_FIT_BACKGROUND_HUBER_K) * sigma
        weights = np.ones_like(residual, dtype=np.float64)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.maximum(np.abs(residual[large]), 1.0e-12)
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(design * root_w[:, None], y_fit * root_w, rcond=None)
        except Exception:
            break
        coef = np.asarray(coef, dtype=np.float64)

    if coef.size < 3 or not np.all(np.isfinite(coef[:3])):
        coef = np.asarray([float(np.nanmedian(y_fit)), 0.0, 0.0], dtype=np.float64)
    return np.asarray([float(coef[0]), float(coef[1]), float(coef[2])], dtype=np.float64)


def local_plane_from_coefficients(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    *,
    theta_ref: float,
    phi_ref: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=np.float64)
    intercept = float(coeff[0]) if coeff.size > 0 and np.isfinite(coeff[0]) else 0.0
    theta_slope = float(coeff[1]) if coeff.size > 1 and np.isfinite(coeff[1]) else 0.0
    phi_slope = float(coeff[2]) if coeff.size > 2 and np.isfinite(coeff[2]) else 0.0
    return (
        intercept
        + theta_slope * (np.asarray(theta_grid, dtype=np.float64) - float(theta_ref))
        + phi_slope * wrapped_delta_deg(phi_grid, float(phi_ref))
    )


def deduplicate_peak_starts(
    candidates: list[tuple[float, float]],
    *,
    theta_step: float,
    phi_step: float,
) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    theta_tol = max(abs(theta_step), 1.0e-6)
    phi_tol = max(abs(phi_step), 1.0e-6)
    for theta0, phi0 in candidates:
        if not np.isfinite(theta0) or not np.isfinite(phi0):
            continue
        duplicate = False
        for theta_prev, phi_prev in kept:
            phi_distance = abs(float(wrapped_delta_deg(float(phi0), float(phi_prev))))
            if abs(float(theta0) - float(theta_prev)) <= theta_tol and phi_distance <= phi_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append((float(theta0), float(phi0)))
    return kept


def _fit_boundary_warnings(
    params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    optimizer_success: bool,
) -> list[str]:
    warnings: list[str] = []
    if params.size < 12:
        warnings.append("unexpected_parameter_count")
        return warnings
    if not optimizer_success:
        warnings.append("optimizer_not_converged")
    if float(params[11]) >= float(upper[11]) - 0.02:
        warnings.append("eta_tail_near_upper_bound")
    for index, name in ((9, "lorentzian_fwhm_u"), (10, "lorentzian_fwhm_v")):
        span = max(float(upper[index]) - float(lower[index]), 1.0e-12)
        if float(params[index]) >= float(upper[index]) - 0.02 * span:
            warnings.append(f"{name}_near_upper_bound")
    for index, name in ((3, "gaussian_fwhm_u"), (4, "gaussian_fwhm_v")):
        span = max(float(upper[index]) - float(lower[index]), 1.0e-12)
        if float(params[index]) <= float(lower[index]) + 0.02 * span:
            warnings.append(f"{name}_near_lower_bound")
    theta_span = max(float(upper[1]) - float(lower[1]), 1.0e-12)
    phi_span = max(float(upper[2]) - float(lower[2]), 1.0e-12)
    if (
        float(params[1]) <= float(lower[1]) + 0.02 * theta_span
        or float(params[1]) >= float(upper[1]) - 0.02 * theta_span
        or float(params[2]) <= float(lower[2]) + 0.02 * phi_span
        or float(params[2]) >= float(upper[2]) - 0.02 * phi_span
    ):
        warnings.append("center_near_allowed_bound_edge")
    return warnings


def _fit_boundary_penalty(
    params: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, optimizer_success: bool
) -> float:
    warnings = _fit_boundary_warnings(params, lower, upper, optimizer_success=optimizer_success)
    if not warnings:
        return 0.0
    return 0.015 * float(len(warnings))


def fit_one_peak(
    entry: dict[str, object],
    caked_image: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    ai,
    transform_bundle,
    detector_shape,
) -> dict[str, object]:
    """Fit one caked peak ROI and return the notebook-compatible payload."""

    theta_half_window = float(THETA_HALF_WINDOW_DEG)
    phi_half_window = float(PHI_HALF_WINDOW_DEG)
    center_theta_bound = float(CENTER_THETA_BOUND_DEG)
    center_phi_bound = float(CENTER_PHI_BOUND_DEG)

    theta_seed = as_float(
        entry.get("_theta_seed_deg"), as_float(entry.get("background_two_theta_deg"))
    )
    phi_seed = as_float(entry.get("_phi_seed_deg"), as_float(entry.get("background_phi_deg")))
    if not np.isfinite(theta_seed) or not np.isfinite(phi_seed):
        raise ValueError("missing peak seed angles")

    theta_mask = np.abs(theta_axis - theta_seed) <= theta_half_window
    phi_mask = np.abs(wrapped_delta_deg(phi_axis, phi_seed)) <= phi_half_window
    theta_idx = np.flatnonzero(theta_mask)
    phi_idx = np.flatnonzero(phi_mask)
    if theta_idx.size < 6 or phi_idx.size < 6:
        raise ValueError("ROI too small")

    roi = np.ascontiguousarray(caked_image[np.ix_(phi_idx, theta_idx)], dtype=np.float64)
    theta_vals = np.ascontiguousarray(theta_axis[theta_idx], dtype=np.float64)
    phi_vals = np.ascontiguousarray(phi_axis[phi_idx], dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    finite = np.isfinite(roi)
    if np.count_nonzero(finite) < 20:
        raise ValueError("too few finite ROI pixels")

    theta_step = float(np.nanmedian(np.diff(theta_axis))) if theta_axis.size > 1 else 0.05
    phi_step = float(np.nanmedian(np.diff(phi_axis))) if phi_axis.size > 1 else 0.5
    background_coeff = robust_peak_background_plane(
        theta_grid,
        phi_grid,
        roi,
        finite,
        theta_ref=float(theta_seed),
        phi_ref=float(phi_seed),
    )
    background_plane_seed = local_plane_from_coefficients(
        theta_grid,
        phi_grid,
        theta_ref=float(theta_seed),
        phi_ref=float(phi_seed),
        coefficients=background_coeff,
    )
    net_roi = roi - background_plane_seed

    center_candidate = (
        finite
        & (np.abs(theta_grid - theta_seed) <= center_theta_bound)
        & (np.abs(wrapped_delta_deg(phi_grid, phi_seed)) <= center_phi_bound)
    )
    candidate_centers: list[tuple[float, float]] = []
    if np.any(center_candidate):
        peak_flat_index = int(np.nanargmax(np.where(center_candidate, net_roi, -np.inf)))
        peak_row, peak_col = np.unravel_index(peak_flat_index, roi.shape)
        candidate_centers.append((float(theta_vals[peak_col]), float(phi_vals[peak_row])))

        positive_core = (
            center_candidate
            & np.isfinite(net_roi)
            & (net_roi > np.nanpercentile(net_roi[center_candidate], 60.0))
        )
        if np.count_nonzero(positive_core) >= 4:
            weights = np.clip(net_roi[positive_core], 0.0, None)
            if np.sum(weights) > 0.0:
                theta_centroid = float(np.average(theta_grid[positive_core], weights=weights))
                phi_centroid = float(
                    phi_seed
                    + np.average(
                        wrapped_delta_deg(phi_grid[positive_core], phi_seed), weights=weights
                    )
                )
                candidate_centers.append((theta_centroid, phi_centroid))
    else:
        seed_distance = np.abs(theta_grid - theta_seed) + np.abs(
            wrapped_delta_deg(phi_grid, phi_seed)
        )
        nearest_flat_index = int(np.nanargmin(np.where(finite, seed_distance, np.inf)))
        peak_row, peak_col = np.unravel_index(nearest_flat_index, roi.shape)
        candidate_centers.append((float(theta_vals[peak_col]), float(phi_vals[peak_row])))
    candidate_centers.append((float(theta_seed), float(phi_seed)))
    candidate_centers = deduplicate_peak_starts(
        candidate_centers,
        theta_step=theta_step,
        phi_step=phi_step,
    )

    y = np.ascontiguousarray(roi[finite], dtype=np.float64)
    theta_fit = np.ascontiguousarray(theta_grid[finite], dtype=np.float64)
    phi_fit = np.ascontiguousarray(phi_grid[finite], dtype=np.float64)
    robust_scale = max(float(np.nanpercentile(np.abs(y - np.nanmedian(y)), 75.0)), 1.0)
    intensity_span = max(float(np.nanpercentile(y, 99.5) - np.nanpercentile(y, 5.0)), 1.0)
    theta_step_abs = max(abs(theta_step), 1.0e-12)
    phi_step_abs = max(abs(phi_step), 1.0e-12)
    fwhm_g_u_min = max(1.2 * theta_step_abs, 1.0e-6)
    fwhm_g_v_min = max(1.2 * phi_step_abs, 1.0e-6)
    fwhm_l_u_min = fwhm_g_u_min
    fwhm_l_v_min = fwhm_g_v_min
    fwhm_g_u_max = max(2.0 * theta_half_window, 4.0 * theta_step_abs)
    fwhm_g_v_max = max(2.0 * phi_half_window, 4.0 * phi_step_abs)
    fwhm_l_u_max = max(4.0 * theta_half_window, fwhm_g_u_max)
    fwhm_l_v_max = max(4.0 * phi_half_window, fwhm_g_v_max)
    theta_slope_bound = max(8.0 * intensity_span / max(theta_half_window, 1.0e-6), 1.0e-6)
    phi_slope_bound = max(8.0 * intensity_span / max(phi_half_window, 1.0e-6), 1.0e-6)

    lower = np.array(
        [
            0.0,
            theta_seed - center_theta_bound,
            phi_seed - center_phi_bound,
            fwhm_g_u_min,
            fwhm_g_v_min,
            -math.pi / 2.0,
            -np.inf,
            -theta_slope_bound,
            -phi_slope_bound,
            fwhm_l_u_min,
            fwhm_l_v_min,
            0.0,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            max(intensity_span * 20.0, 1.0),
            theta_seed + center_theta_bound,
            phi_seed + center_phi_bound,
            fwhm_g_u_max,
            fwhm_g_v_max,
            math.pi / 2.0,
            np.inf,
            theta_slope_bound,
            phi_slope_bound,
            fwhm_l_u_max,
            fwhm_l_v_max,
            0.95,
        ],
        dtype=np.float64,
    )

    fit_attempts: list[tuple[float, object, np.ndarray]] = []
    for theta0, phi0 in candidate_centers:
        baseline_at_center = float(
            background_coeff[0]
            + background_coeff[1] * (float(theta0) - float(theta_seed))
            + background_coeff[2] * float(wrapped_delta_deg(float(phi0), float(phi_seed)))
        )
        peak_distance = np.abs(theta_grid - float(theta0)) + np.abs(
            wrapped_delta_deg(phi_grid, float(phi0))
        )
        nearest_flat_index = int(np.nanargmin(np.where(finite, peak_distance, np.inf)))
        peak_row, peak_col = np.unravel_index(nearest_flat_index, roi.shape)
        local_net = float(roi[peak_row, peak_col] - baseline_at_center)
        if np.any(center_candidate):
            local_net = max(
                local_net, float(np.nanmax(np.where(center_candidate, net_roi, -np.inf)))
            )
        amp0 = max(local_net, 1.0)

        tail_distance = np.sqrt(
            ((theta_fit - float(theta0)) / max(center_theta_bound, 1.0e-6)) ** 2
            + (wrapped_delta_deg(phi_fit, float(phi0)) / max(center_phi_bound, 1.0e-6)) ** 2
        )
        tail_weight = np.ascontiguousarray(
            1.0 + float(GAUSSIAN_TAIL_DISTANCE_WEIGHT) * np.clip(tail_distance, 0.0, 2.0),
            dtype=np.float64,
        )
        tail_excess = np.clip(
            (tail_distance - float(GAUSSIAN_TAIL_OVERPREDICTION_START))
            / max(2.0 - float(GAUSSIAN_TAIL_OVERPREDICTION_START), 1.0e-6),
            0.0,
            1.0,
        )
        tail_overprediction_weight = np.ascontiguousarray(
            1.0 + float(GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT) * tail_excess,
            dtype=np.float64,
        )
        signal = np.clip(y - baseline_at_center, 0.0, None)
        tail_scale = np.ascontiguousarray(
            robust_scale + float(GAUSSIAN_CORE_SIGNAL_DOWNSCALE) * signal,
            dtype=np.float64,
        )

        fwhm_g_u0 = max(3.0 * theta_step_abs, 0.08)
        fwhm_g_v0 = max(3.0 * phi_step_abs, 0.35)
        angle_seeds = (0.0, 0.25, -0.25)
        eta_tail_seeds = (0.0, 0.15, 0.35, 0.60)
        lorentzian_width_multipliers = (1.5, 2.5, 4.0, 7.0)

        def residual(params: np.ndarray) -> np.ndarray:
            return _rotated_gaussian_residual_points_numba(
                np.asarray(params, dtype=np.float64),
                theta_fit,
                phi_fit,
                y,
                tail_weight,
                tail_scale,
                tail_overprediction_weight,
            )

        for eta_tail0 in eta_tail_seeds:
            for lorentzian_width_multiplier in lorentzian_width_multipliers:
                for angle0 in angle_seeds:
                    x0 = np.array(
                        [
                            amp0,
                            float(theta0),
                            float(phi0),
                            fwhm_g_u0,
                            fwhm_g_v0,
                            angle0,
                            baseline_at_center,
                            float(background_coeff[1]),
                            float(background_coeff[2]),
                            max(float(lorentzian_width_multiplier) * fwhm_g_u0, 0.20),
                            max(float(lorentzian_width_multiplier) * fwhm_g_v0, 0.80),
                            eta_tail0,
                        ],
                        dtype=np.float64,
                    )
                    dynamic_upper = upper.copy()
                    dynamic_upper[0] = max(dynamic_upper[0], amp0 * 80.0, amp0 + 1.0)
                    x0 = np.minimum(np.maximum(x0, lower + 1.0e-9), dynamic_upper - 1.0e-9)

                    try:
                        result = least_squares(
                            residual,
                            x0,
                            bounds=(lower, dynamic_upper),
                            loss="soft_l1",
                            f_scale=2.0,
                            max_nfev=int(PEAK_FIT_MAX_NFEV),
                            x_scale=[
                                max(amp0, 1.0),
                                0.2,
                                1.0,
                                0.2,
                                1.0,
                                1.0,
                                max(abs(baseline_at_center), 1.0),
                                max(theta_slope_bound * 0.25, 1.0),
                                max(phi_slope_bound * 0.25, 1.0),
                                0.5,
                                2.0,
                                0.25,
                            ],
                        )
                    except Exception:
                        continue

                    params_candidate = np.asarray(result.x, dtype=np.float64)
                    weighted_score = float(np.nanmean(residual(params_candidate) ** 2))
                    model_candidate = _rotated_gaussian_plane_points_numba(
                        params_candidate,
                        theta_fit,
                        phi_fit,
                    )
                    raw_residual = model_candidate - y
                    raw_score = float(np.nanmean((raw_residual / max(robust_scale, 1.0e-12)) ** 2))
                    score = (
                        0.65 * weighted_score
                        + 0.35 * raw_score
                        + _fit_boundary_penalty(
                            params_candidate,
                            lower,
                            dynamic_upper,
                            optimizer_success=bool(result.success),
                        )
                    )
                    if np.isfinite(score):
                        fit_attempts.append((score, result, params_candidate))

    if not fit_attempts:
        raise RuntimeError("all peak fit starts failed")
    fit_attempts.sort(key=lambda item: item[0])
    best_score, result, params = fit_attempts[0]
    fit_parameter_warnings = _fit_boundary_warnings(
        params,
        lower,
        upper,
        optimizer_success=bool(result.success),
    )

    full_fit = _rotated_gaussian_grid_numba(params, theta_vals, phi_vals, False)
    peak_fit = _rotated_gaussian_grid_numba(params, theta_vals, phi_vals, True)
    peak_subtracted_image = roi - peak_fit
    optimization_residual_image = roi - full_fit
    finite_resid = optimization_residual_image[finite]
    fit_col, fit_row = caked_point_to_detector_pixel(
        ai,
        detector_shape,
        theta_axis,
        phi_axis,
        float(params[1]),
        float(params[2]),
        transform_bundle=transform_bundle,
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    return {
        "entry": dict(entry),
        "background_index": int(entry["_background_index"]),
        "background_name": str(entry["_background_name"]),
        "label": str(entry["_label"]),
        "branch": str(entry["_branch"]),
        "theta_seed_deg": float(theta_seed),
        "phi_seed_deg": float(phi_seed),
        "params": params,
        "theta_idx": theta_idx,
        "phi_idx": phi_idx,
        "theta_vals": theta_vals,
        "phi_vals": phi_vals,
        "roi": roi,
        "fit": peak_fit,
        "fit_with_nuisance_background": full_fit,
        "peak_fit": peak_fit,
        "peak_subtracted_roi": peak_subtracted_image,
        "residual": peak_subtracted_image,
        "optimization_residual": optimization_residual_image,
        "success": True,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "rmse": float(np.sqrt(np.nanmean(finite_resid**2))),
        "fit_detector_col": None if fit_col is None else float(fit_col),
        "fit_detector_row": None if fit_row is None else float(fit_row),
        "fit_model": FIT_MODEL_NAME,
        "fit_score": float(best_score),
        "fit_start_count": int(len(fit_attempts)),
        "fit_parameter_warnings": fit_parameter_warnings,
        "fit_has_parameter_warning": bool(fit_parameter_warnings),
        "background_plane_coefficients": np.asarray(background_coeff, dtype=np.float64),
        "baseline_equation": "density = baseline + theta_slope*(two_theta_deg - fit_two_theta_deg) + phi_slope*wrapped_delta_deg(phi_deg, fit_phi_deg)",
    }


background_results = []
all_fit_results = []
all_table_rows = []
FIT_TABLE_COLUMNS = [
    "sample_name",
    "tilt_deg",
    "background_index",
    "background_name",
    "label",
    "branch",
    "q_group_key",
    "branch_id",
    "source_branch_index",
    "selection_reason",
    "seed_two_theta_deg",
    "seed_phi_deg",
    "fit_two_theta_deg",
    "fit_phi_deg",
    "fit_sigma_two_theta_deg",
    "fit_sigma_phi_deg",
    "fit_angle_deg",
    "fit_amplitude",
    "fit_baseline",
    "fit_baseline_slope_two_theta",
    "fit_baseline_slope_phi",
    "fit_model",
    "fit_fwhm_gaussian_u_deg",
    "fit_fwhm_gaussian_v_deg",
    "fit_fwhm_lorentzian_u_deg",
    "fit_fwhm_lorentzian_v_deg",
    "fit_lorentzian_fraction",
    "fit_rotation_angle_rad",
    "fit_rotation_angle_deg",
    "fit_theta_slope",
    "fit_phi_slope",
    "fit_fwhm_lorentzian_to_gaussian_u",
    "fit_fwhm_lorentzian_to_gaussian_v",
    "fit_parameter_warnings",
    "fit_has_parameter_warning",
    "optimizer_success",
    "fit_message",
    "fit_detector_col",
    "fit_detector_row",
    "recorded_detector_col",
    "recorded_detector_row",
    "rmse",
    "success",
]

# The expensive background-results cell is now staged:
# 1. Read/cake each background image and prepare fit seeds.
# 2. Fit every peak from every background in one global worker pool.
# 3. Render and save each background's peak-only subtraction products.
# This keeps the fitting parallelism outside the detector-to-cake setup bottleneck.
background_preps: list[dict[str, object]] = []
background_prep_by_idx: dict[int, dict[str, object]] = {}
prep_total_start = time.perf_counter()

for bg_idx, bg_path_text in enumerate(background_files):
    bg_start = time.perf_counter()
    bg_path = Path(bg_path_text).expanduser()
    tilt_deg = float(BACKGROUND_TILT_DEG[bg_idx])
    display_label = f"{SAMPLE_NAME} {format_angle_value(tilt_deg)} deg"
    native = np.asarray(read_osc(bg_path), dtype=np.float64)
    detector_image = apply_background_backend_orientation(
        native, flip_x=backend_flip_x, flip_y=backend_flip_y, rotation_k=backend_rotation_k
    )
    detector_image = np.asarray(detector_image, dtype=np.float64)

    ai = FastAzimuthalIntegrator(
        dist=float(distance_m),
        poni1=float(center_row_px) * PIXEL_SIZE_M,
        poni2=float(center_col_px) * PIXEL_SIZE_M,
        pixel1=PIXEL_SIZE_M,
        pixel2=PIXEL_SIZE_M,
        wavelength=WAVELENGTH_M,
    )
    tth_max = detector_two_theta_max_deg(detector_image.shape, ai.geometry)
    try:
        ai.warm_geometry_cache(
            detector_image.shape,
            npt_rad=NPT_RADIAL,
            npt_azim=NPT_AZIMUTH,
            engine=EXACT_CAKE_ENGINE,
            workers=CAKE_WORKERS,
        )
    except Exception as exc:
        print(f"cake geometry warm-up skipped for {display_label}: {exc}")
    theta_map, raw_phi_map = detector_pixel_angular_maps(detector_image.shape, ai.geometry)
    phi_map = raw_phi_to_gui_phi(raw_phi_map)
    raw_detector_image = detector_image.copy()
    # No diffuse/radial detector background is subtracted before caking or fitting.
    # Peak-subtracted products are generated only after fitted peak models are built.

    caked_result = ai.integrate2d(
        detector_image,
        npt_rad=NPT_RADIAL,
        npt_azim=NPT_AZIMUTH,
        correctSolidAngle=BACKGROUND_SOLID_ANGLE_CORRECTION,
        method="lut",
        unit="2th_deg",
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    caked_integrated_intensity, theta_axis, phi_axis = prepare_gui_phi_display(caked_result)
    raw_sum_signal = getattr(caked_result, "sum_signal", None)
    raw_sum_normalization = getattr(caked_result, "sum_normalization", None)
    raw_count = getattr(caked_result, "count", None)
    if (raw_sum_signal is None) != (raw_sum_normalization is None):
        raise RuntimeError("pyFAI caked sum_signal and sum_normalization fields must be paired")
    if raw_sum_signal is None:
        caked_sum_signal = None
        caked_sum_normalization = None
    else:
        caked_sum_signal, _, _ = caked_field_to_gui_phi(
            raw_sum_signal,
            caked_result.azimuthal_deg,
            caked_result.radial_deg,
        )
        caked_sum_normalization, _, _ = caked_field_to_gui_phi(
            raw_sum_normalization,
            caked_result.azimuthal_deg,
            caked_result.radial_deg,
        )
    if raw_count is None:
        caked_count = None
    else:
        caked_count, _, _ = caked_field_to_gui_phi(
            raw_count,
            caked_result.azimuthal_deg,
            caked_result.radial_deg,
        )
    caked_density_image = caked_image_for_intensity_mode(
        caked_integrated_intensity,
        caked_sum_signal=caked_sum_signal,
        caked_sum_normalization=caked_sum_normalization,
        caked_count=caked_count,
        intensity_mode="density",
    )
    caked_raw_sum_image = caked_image_for_intensity_mode(
        caked_integrated_intensity,
        caked_sum_signal=caked_sum_signal,
        caked_sum_normalization=caked_sum_normalization,
        caked_count=caked_count,
        intensity_mode="raw_sum",
    )
    caked_display_image = caked_image_for_intensity_mode(
        caked_integrated_intensity,
        caked_sum_signal=caked_sum_signal,
        caked_sum_normalization=caked_sum_normalization,
        caked_count=caked_count,
        intensity_mode=CAKED_FIGURE_INTENSITY_MODE,
    )
    caked_image = caked_density_image
    transform_bundle = build_cake_transform_bundle(
        ai,
        detector_image.shape,
        caked_result.radial_deg,
        caked_result.azimuthal_deg,
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    caked_projection_context = {
        "detector_shape": tuple(detector_image.shape),
        "radial_axis": theta_axis,
        "azimuth_axis": phi_axis,
        "raw_azimuth_axis": caked_result.azimuthal_deg,
        "transform_bundle": transform_bundle,
    }
    qr_overlay_config = gui_qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=True,
        image_size=int(detector_image.shape[0]),
        display_rotate_k=0,
        center_col=float(center_col_px),
        center_row=float(center_row_px),
        distance_cor_to_detector=float(distance_m),
        gamma_deg=as_float(variables.get("gamma_var"), 0.0),
        Gamma_deg=as_float(variables.get("Gamma_var"), 0.0),
        chi_deg=as_float(variables.get("chi_var"), 0.0),
        psi_deg=psi_deg,
        psi_z_deg=as_float(variables.get("psi_z_var"), 0.0),
        zs=as_float(variables.get("zs_var"), 0.0),
        zb=as_float(variables.get("zb_var"), 0.0),
        theta_initial_deg=float(tilt_deg),
        cor_angle_deg=as_float(variables.get("cor_angle_var"), 0.0),
        pixel_size_m=PIXEL_SIZE_M,
        wavelength=WAVELENGTH_M * 1.0e10,
        n2=n2_value,
        phi_samples=rod_phi_samples,
    )

    prepared_entries = []
    seed_failures = []
    for entry in entries_by_bg[bg_idx]:
        seeded = nearest_detector_angles_from_entry(entry, theta_map, phi_map)
        if seeded is None:
            seed_failures.append({"entry": entry, "error": "missing detector/caked seed"})
            continue
        theta_seed, phi_seed = seeded
        prepared = dict(entry)
        prepared["_theta_seed_deg"] = float(theta_seed)
        prepared["_phi_seed_deg"] = float(phi_seed)
        prepared["_branch"] = branch_from_phi(phi_seed)
        prepared_entries.append(prepared)

    prep = {
        "sample_name": SAMPLE_NAME,
        "tilt_deg": tilt_deg,
        "display_label": display_label,
        "background_index": bg_idx,
        "background_name": Path(bg_path).name,
        "detector_image": detector_image,
        "raw_detector_image": raw_detector_image,
        "caked_image": caked_image,
        "caked_integrated_intensity": caked_integrated_intensity,
        "caked_density_image": caked_density_image,
        "caked_raw_sum_image": caked_raw_sum_image,
        "caked_display_image": caked_display_image,
        "caked_intensity_mode": CAKED_FIGURE_INTENSITY_MODE,
        "caked_sum_signal": caked_sum_signal,
        "caked_sum_normalization": caked_sum_normalization,
        "caked_count": caked_count,
        "theta_axis": theta_axis,
        "phi_axis": phi_axis,
        "theta_map": theta_map,
        "phi_map": phi_map,
        "ai": ai,
        "transform_bundle": transform_bundle,
        "detector_shape": tuple(detector_image.shape),
        "prepared_entries": prepared_entries,
        "seed_failures": seed_failures,
        "tth_max": tth_max,
        "caked_projection_context": caked_projection_context,
        "qr_overlay_config": qr_overlay_config,
        "raw_azimuth_axis": caked_result.azimuthal_deg,
        "prep_elapsed_s": time.perf_counter() - bg_start,
    }
    background_preps.append(prep)
    background_prep_by_idx[int(bg_idx)] = prep
    print(
        f"{display_label}: prepared={len(prepared_entries)} seed_fail={len(seed_failures)} "
        f"tth_max={tth_max:.3f} prep_elapsed={prep['prep_elapsed_s']:.2f}s "
        f"solid_angle_correction={bool(BACKGROUND_SOLID_ANGLE_CORRECTION)}"
    )

prep_elapsed = time.perf_counter() - prep_total_start
fit_jobs: list[tuple[int, int, dict[str, object]]] = []
fit_results_by_bg: dict[int, list[dict[str, object] | None]] = {}
fit_failures_by_bg: dict[int, list[dict[str, object]]] = {}
for prep in background_preps:
    bg_idx = int(prep["background_index"])
    prepared_entries = list(prep["prepared_entries"])
    fit_results_by_bg[bg_idx] = [None] * len(prepared_entries)
    fit_failures_by_bg[bg_idx] = list(prep["seed_failures"])
    for local_idx, entry in enumerate(prepared_entries):
        fit_jobs.append((bg_idx, local_idx, entry))

if not fit_jobs:
    raise RuntimeError(NO_BACKGROUND_PEAK_ENTRIES_MESSAGE)

PRE_EDITOR_CACHE_PATH = pre_editor_cache_path(OUT_DIR, STATE_PATH)
PRE_EDITOR_CACHE_INPUT_SIGNATURE = {
    "state_filename": Path(STATE_PATH).name,
    "background_files": [Path(path).name for path in background_files],
    "background_tilt_deg": {
        int(idx): float(BACKGROUND_TILT_DEG[idx]) for idx in sorted(BACKGROUND_TILT_DEG)
    },
    "backend_orientation": {
        "rotation_k": int(backend_rotation_k),
        "flip_x": bool(backend_flip_x),
        "flip_y": bool(backend_flip_y),
    },
    "geometry": {
        "center_row_px": float(center_row_px),
        "center_col_px": float(center_col_px),
        "distance_m": float(distance_m),
        "pixel_size_m": float(PIXEL_SIZE_M),
        "wavelength_m": float(WAVELENGTH_M),
    },
    "active_lattice": ACTIVE_LATTICE_CACHE_SIGNATURE,
    "q_group_rows": Q_GROUP_ROWS_CACHE_SIGNATURE,
    "rod_reference_policy": ROD_REFERENCE_POLICY_SIGNATURE,
    "fit_settings": {
        "fit_model": FIT_MODEL_NAME,
        "theta_half_window_deg": float(THETA_HALF_WINDOW_DEG),
        "phi_half_window_deg": float(PHI_HALF_WINDOW_DEG),
        "center_theta_bound_deg": float(CENTER_THETA_BOUND_DEG),
        "center_phi_bound_deg": float(CENTER_PHI_BOUND_DEG),
        "peak_fit_max_nfev": int(PEAK_FIT_MAX_NFEV),
        "peak_fit_shared_linear_baseline": bool(PEAK_FIT_SHARED_LINEAR_BASELINE_ENABLED),
        "background_solid_angle_correction": bool(BACKGROUND_SOLID_ANGLE_CORRECTION),
        "caked_intensity_mode": str(CAKED_FIGURE_INTENSITY_MODE),
        "exact_cake_engine": str(EXACT_CAKE_ENGINE),
        "rod_profile_qz_bins": int(ROD_PROFILE_QZ_BINS),
        "rod_profile_max_two_theta_deg": float(ROD_PROFILE_MAX_TWO_THETA_DEG),
        "rod_profile_delta_qr": float(qr_rod_delta_qr),
        "rod_phi_samples": int(rod_phi_samples),
        "rod_qz_shared_linear_baseline": bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED),
        "rod_qz_nonlinear_refinement": bool(ROD_QZ_NONLINEAR_REFINEMENT_ENABLED),
        "pbi2_disable_background_subtraction": bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION),
        "qr_rod_transverse_background": bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED),
        "qr_rod_bg_side_band_inner_scale": float(QR_ROD_BG_SIDE_BAND_INNER_SCALE),
        "qr_rod_bg_side_band_outer_scale": float(QR_ROD_BG_SIDE_BAND_OUTER_SCALE),
        "qr_rod_bg_min_side_pixels": int(QR_ROD_BG_MIN_SIDE_PIXELS),
        "qr_rod_bg_percentile": float(QR_ROD_BG_PERCENTILE),
    },
    "fit_jobs": pre_editor_fit_job_signature(fit_jobs),
}
PRE_EDITOR_CACHE_KEY = pre_editor_cache_key(
    STATE_PATH, input_signature=PRE_EDITOR_CACHE_INPUT_SIGNATURE
)
if _truthy_setting("RESET_PRE_EDITOR_CACHE_OVERRIDE", "RA_SIM_RESET_PRE_EDITOR_CACHE", False):
    removed_pre_editor_cache = reset_pre_editor_cache(PRE_EDITOR_CACHE_PATH)
    print(
        f"reset pre-editor diagnostic cache={PRE_EDITOR_CACHE_PATH} removed={removed_pre_editor_cache}"
    )
pre_editor_cache = load_pre_editor_cache(PRE_EDITOR_CACHE_PATH, PRE_EDITOR_CACHE_KEY)


def _reset_fit_containers() -> None:
    fit_results_by_bg.clear()
    fit_failures_by_bg.clear()
    for prep in background_preps:
        bg_idx = int(prep["background_index"])
        prepared_entries = list(prep["prepared_entries"])
        fit_results_by_bg[bg_idx] = [None] * len(prepared_entries)
        fit_failures_by_bg[bg_idx] = list(prep["seed_failures"])


def _fit_global_peak_job(
    job: tuple[int, int, dict[str, object]],
) -> tuple[int, int, dict[str, object], float, int]:
    bg_idx, local_idx, entry = job
    start = time.perf_counter()
    prep = background_prep_by_idx[int(bg_idx)]
    item = fit_one_peak(
        entry,
        np.asarray(prep["caked_image"], dtype=np.float64),
        np.asarray(prep["theta_axis"], dtype=np.float64),
        np.asarray(prep["phi_axis"], dtype=np.float64),
        prep["ai"],
        prep["transform_bundle"],
        tuple(prep["detector_shape"]),
    )
    return int(bg_idx), int(local_idx), item, float(time.perf_counter() - start), int(os.getpid())


def _run_local_peak_jobs(*, force_serial: bool = False) -> tuple[str, dict[int, int], list[float]]:
    task_elapsed: list[float] = []
    pid_counts: dict[int, int] = {}
    if global_fit_workers > 1 and not force_serial:
        with ThreadPoolExecutor(max_workers=global_fit_workers) as executor:
            future_to_job = {executor.submit(_fit_global_peak_job, job): job for job in fit_jobs}
            for future in as_completed(future_to_job):
                bg_idx, local_idx, entry = future_to_job[future]
                try:
                    out_bg_idx, out_local_idx, item, elapsed_s, pid = future.result()
                    fit_results_by_bg[out_bg_idx][out_local_idx] = item
                    task_elapsed.append(float(elapsed_s))
                    pid_counts[int(pid)] = pid_counts.get(int(pid), 0) + 1
                except Exception as exc:
                    fit_failures_by_bg[int(bg_idx)].append({"entry": entry, "error": repr(exc)})
        return "thread_pool", pid_counts, task_elapsed

    for job in fit_jobs:
        bg_idx, local_idx, entry = job
        try:
            out_bg_idx, out_local_idx, item, elapsed_s, pid = _fit_global_peak_job(job)
            fit_results_by_bg[out_bg_idx][out_local_idx] = item
            task_elapsed.append(float(elapsed_s))
            pid_counts[int(pid)] = pid_counts.get(int(pid), 0) + 1
        except Exception as exc:
            fit_failures_by_bg[int(bg_idx)].append({"entry": entry, "error": repr(exc)})
    return "serial", pid_counts, task_elapsed


def _run_process_peak_jobs() -> tuple[str, dict[int, int], list[float]]:
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_name] = "1"
    old_numba_num_threads_env = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = str(PROCESS_NUMBA_THREADS)
    process_jobs = []
    task_elapsed: list[float] = []
    pid_counts: dict[int, int] = {}
    try:
        with TemporaryDirectory(prefix="background_peak_fit_", dir=str(OUT_DIR)) as tmp_dir:
            array_paths_by_bg: dict[int, dict[str, str]] = {}
            for prep in background_preps:
                bg_idx = int(prep["background_index"])
                array_paths_by_bg[bg_idx] = save_peak_fit_background_arrays(
                    tmp_dir,
                    background_index=bg_idx,
                    caked_image=np.asarray(prep["caked_image"], dtype=np.float64),
                    theta_axis=np.asarray(prep["theta_axis"], dtype=np.float64),
                    phi_axis=np.asarray(prep["phi_axis"], dtype=np.float64),
                )
            for bg_idx, local_idx, entry in fit_jobs:
                process_jobs.append(
                    make_peak_fit_job(
                        background_index=int(bg_idx),
                        local_index=int(local_idx),
                        entry=entry,
                        **array_paths_by_bg[int(bg_idx)],
                    )
                )
            with ProcessPoolExecutor(
                max_workers=global_fit_workers,
                initializer=configure_peak_fit_worker,
                initargs=(PEAK_FIT_WORKER_SETTINGS, PROCESS_NUMBA_THREADS),
            ) as executor:
                future_to_job = {
                    executor.submit(fit_peak_from_process_job, job): job for job in process_jobs
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    entry = dict(job["entry"])
                    bg_idx = int(job["background_index"])
                    local_idx = int(job["local_index"])
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as exc:
                        fit_failures_by_bg[bg_idx].append({"entry": entry, "error": repr(exc)})
                        continue
                    out_bg_idx = int(result["background_index"])
                    out_local_idx = int(result["local_index"])
                    fit_results_by_bg[out_bg_idx][out_local_idx] = result["item"]
                    task_elapsed.append(float(result.get("elapsed_s", np.nan)))
                    pid = int(result.get("pid", -1))
                    pid_counts[pid] = pid_counts.get(pid, 0) + 1
            return "process_pool", pid_counts, task_elapsed
    finally:
        if old_numba_num_threads_env is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = old_numba_num_threads_env


fit_start = time.perf_counter()
global_fit_workers = min(int(FIT_WORKERS), max(1, len(fit_jobs)))
requested_fit_backend = str(FIT_BACKEND).strip().lower() or "process"
if requested_fit_backend == "auto":
    requested_fit_backend = "process" if global_fit_workers > 1 else "serial"
if global_fit_workers <= 1:
    requested_fit_backend = "serial"

background_peak_fit_stage = pre_editor_cache_get_stage(
    pre_editor_cache, "background_peak_fits", PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE
)
background_peak_fit_cache_hit = background_peak_fit_stage_is_valid(
    background_peak_fit_stage, expected_fit_count=len(fit_jobs)
)
if background_peak_fit_cache_hit:
    cached_fit_results_by_bg = {
        int(bg_idx): list(items)
        for bg_idx, items in dict(background_peak_fit_stage["fit_results_by_bg"]).items()
    }
    cached_fit_failures_by_bg = {
        int(bg_idx): list(items)
        for bg_idx, items in dict(background_peak_fit_stage["fit_failures_by_bg"]).items()
    }
    for prep in background_preps:
        bg_idx = int(prep["background_index"])
        expected_count = len(list(prep["prepared_entries"]))
        if (
            bg_idx not in cached_fit_results_by_bg
            or len(cached_fit_results_by_bg[bg_idx]) != expected_count
        ):
            background_peak_fit_cache_hit = False
            print(
                f"ignored incomplete pre-editor background peak-fit cache={PRE_EDITOR_CACHE_PATH}: background {bg_idx} expected {expected_count} results"
            )
            break
    if background_peak_fit_cache_hit:
        fit_results_by_bg = cached_fit_results_by_bg
        fit_failures_by_bg = cached_fit_failures_by_bg
        active_fit_backend = "pre_editor_cache"
        fit_pid_counts = {}
        fit_task_elapsed = []
        print(f"reused pre-editor background peak-fit cache={PRE_EDITOR_CACHE_PATH}")
if not background_peak_fit_cache_hit:
    try:
        if requested_fit_backend == "process":
            active_fit_backend, fit_pid_counts, fit_task_elapsed = _run_process_peak_jobs()
        elif requested_fit_backend == "thread":
            active_fit_backend, fit_pid_counts, fit_task_elapsed = _run_local_peak_jobs(
                force_serial=False
            )
        else:
            active_fit_backend, fit_pid_counts, fit_task_elapsed = _run_local_peak_jobs(
                force_serial=True
            )
    except (BrokenProcessPool, OSError, RuntimeError) as exc:
        if requested_fit_backend != "process":
            raise
        print(f"process peak-fit pool failed; falling back to thread pool: {exc!r}")
        _reset_fit_containers()
        active_fit_backend, fit_pid_counts, fit_task_elapsed = _run_local_peak_jobs(
            force_serial=False
        )
    pre_editor_cache = pre_editor_cache_with_stage(
        pre_editor_cache,
        "background_peak_fits",
        PRE_EDITOR_BACKGROUND_FIT_STAGE_SIGNATURE,
        {
            "fit_job_count": int(len(fit_jobs)),
            "fit_results_by_bg": fit_results_by_bg,
            "fit_failures_by_bg": fit_failures_by_bg,
        },
    )
    write_pre_editor_cache(PRE_EDITOR_CACHE_PATH, PRE_EDITOR_CACHE_KEY, pre_editor_cache)
    print(f"saved pre-editor background peak-fit cache={PRE_EDITOR_CACHE_PATH}")

fit_elapsed = time.perf_counter() - fit_start
finite_task_elapsed = np.asarray(
    [value for value in fit_task_elapsed if np.isfinite(value)], dtype=np.float64
)
if finite_task_elapsed.size:
    task_stats = f"task_avg={float(np.nanmean(finite_task_elapsed)):.3f}s task_p95={float(np.nanpercentile(finite_task_elapsed, 95.0)):.3f}s"
else:
    task_stats = "task_avg=nan task_p95=nan"
print(
    f"global peak fitting: jobs={len(fit_jobs)} workers={global_fit_workers} "
    f"elapsed={fit_elapsed:.2f}s prep_elapsed={prep_elapsed:.2f}s backend={active_fit_backend} "
    f"pids={len(fit_pid_counts)} {task_stats}"
)


def _attach_fit_detector_projection(item: dict[str, object], prep: dict[str, object]) -> None:
    if item.get("fit_detector_col") is not None and item.get("fit_detector_row") is not None:
        return
    params = np.asarray(item["params"], dtype=np.float64)
    fit_col, fit_row = caked_point_to_detector_pixel(
        prep["ai"],
        tuple(prep["detector_shape"]),
        np.asarray(prep["theta_axis"], dtype=np.float64),
        np.asarray(prep["phi_axis"], dtype=np.float64),
        float(params[1]),
        float(params[2]),
        transform_bundle=prep["transform_bundle"],
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    item["fit_detector_col"] = None if fit_col is None else float(fit_col)
    item["fit_detector_row"] = None if fit_row is None else float(fit_row)


post_start = time.perf_counter()
for prep in background_preps:
    bg_idx = int(prep["background_index"])
    tilt_deg = float(prep["tilt_deg"])
    display_label = str(prep["display_label"])
    prepared_entries = list(prep["prepared_entries"])
    fit_results_by_index = fit_results_by_bg[bg_idx]
    failures = fit_failures_by_bg[bg_idx]

    fit_results = []
    for idx, item in enumerate(fit_results_by_index):
        if item is None:
            continue
        entry = prepared_entries[idx]
        _attach_fit_detector_projection(item, prep)
        fit_results.append(item)
        all_fit_results.append(item)
        p = np.asarray(item["params"], dtype=np.float64)
        fwhm_g_u = float(p[3]) if p.size > 3 else float("nan")
        fwhm_g_v = float(p[4]) if p.size > 4 else float("nan")
        fwhm_l_u = float(p[9]) if p.size > 9 else float("nan")
        fwhm_l_v = float(p[10]) if p.size > 10 else float("nan")
        eta_tail = float(p[11]) if p.size > 11 else float("nan")
        sigma_g_u = fwhm_g_u / GAUSSIAN_FWHM_TO_SIGMA if np.isfinite(fwhm_g_u) else float("nan")
        sigma_g_v = fwhm_g_v / GAUSSIAN_FWHM_TO_SIGMA if np.isfinite(fwhm_g_v) else float("nan")
        ratio_lg_u = (
            fwhm_l_u / fwhm_g_u
            if np.isfinite(fwhm_l_u) and np.isfinite(fwhm_g_u) and fwhm_g_u > 0.0
            else float("nan")
        )
        ratio_lg_v = (
            fwhm_l_v / fwhm_g_v
            if np.isfinite(fwhm_l_v) and np.isfinite(fwhm_g_v) and fwhm_g_v > 0.0
            else float("nan")
        )
        parameter_warnings = list(item.get("fit_parameter_warnings", []) or [])
        recorded_xy = detector_xy_from_entry(entry)
        all_table_rows.append(
            {
                "sample_name": SAMPLE_NAME,
                "tilt_deg": tilt_deg,
                "background_index": bg_idx,
                "background_name": str(prep["background_name"]),
                "label": item["label"],
                "branch": item["branch"],
                "q_group_key": compact_json_text(entry.get("q_group_key")),
                "branch_id": str(entry.get("branch_id", "")),
                "source_branch_index": str(entry.get("source_branch_index", "")),
                "selection_reason": str(entry.get("selection_reason", "")),
                "seed_two_theta_deg": item["theta_seed_deg"],
                "seed_phi_deg": item["phi_seed_deg"],
                "fit_two_theta_deg": float(p[1]),
                "fit_phi_deg": float(p[2]),
                "fit_sigma_two_theta_deg": sigma_g_u,
                "fit_sigma_phi_deg": sigma_g_v,
                "fit_angle_deg": float(np.rad2deg(p[5])),
                "fit_amplitude": float(p[0]),
                "fit_baseline": float(p[6]),
                "fit_baseline_slope_two_theta": float(p[7]),
                "fit_baseline_slope_phi": float(p[8]),
                "fit_model": str(item.get("fit_model", FIT_MODEL_NAME)),
                "fit_fwhm_gaussian_u_deg": fwhm_g_u,
                "fit_fwhm_gaussian_v_deg": fwhm_g_v,
                "fit_fwhm_lorentzian_u_deg": fwhm_l_u,
                "fit_fwhm_lorentzian_v_deg": fwhm_l_v,
                "fit_lorentzian_fraction": eta_tail,
                "fit_rotation_angle_rad": float(p[5]),
                "fit_rotation_angle_deg": float(np.rad2deg(p[5])),
                "fit_theta_slope": float(p[7]),
                "fit_phi_slope": float(p[8]),
                "fit_fwhm_lorentzian_to_gaussian_u": ratio_lg_u,
                "fit_fwhm_lorentzian_to_gaussian_v": ratio_lg_v,
                "fit_parameter_warnings": ";".join(str(value) for value in parameter_warnings),
                "fit_has_parameter_warning": bool(
                    item.get("fit_has_parameter_warning", bool(parameter_warnings))
                ),
                "optimizer_success": bool(item.get("optimizer_success", False)),
                "fit_message": str(item.get("message", "")),
                "fit_detector_col": item["fit_detector_col"],
                "fit_detector_row": item["fit_detector_row"],
                "recorded_detector_col": None if recorded_xy is None else float(recorded_xy[0]),
                "recorded_detector_row": None if recorded_xy is None else float(recorded_xy[1]),
                "rmse": item["rmse"],
                "success": item["success"],
            }
        )

    caked_density_image = np.asarray(prep["caked_density_image"], dtype=np.float64)
    detector_image = np.asarray(prep["detector_image"], dtype=np.float64)
    theta_map = np.asarray(prep["theta_map"], dtype=np.float64)
    phi_map = np.asarray(prep["phi_map"], dtype=np.float64)
    caked_peak_model = np.zeros_like(caked_density_image, dtype=np.float64)
    for item in fit_results:
        caked_peak_model[np.ix_(item["phi_idx"], item["theta_idx"])] += item["peak_fit"]

    detector_peak_model = render_detector_peak_model(theta_map, phi_map, fit_results)
    caked_peak_subtracted_image = caked_density_image - caked_peak_model
    detector_peak_subtracted_image = detector_image - detector_peak_model

    np.save(OUT_DIR / f"background_{bg_idx:02d}_caked_peak_fit_model.npy", caked_peak_model)
    np.save(OUT_DIR / f"background_{bg_idx:02d}_detector_peak_fit_model.npy", detector_peak_model)
    np.save(
        OUT_DIR / f"background_{bg_idx:02d}_caked_peak_subtracted_density.npy",
        caked_peak_subtracted_image,
    )
    np.save(
        OUT_DIR / f"background_{bg_idx:02d}_detector_peak_subtracted_density.npy",
        detector_peak_subtracted_image,
    )
    background_results.append(
        {
            "sample_name": SAMPLE_NAME,
            "tilt_deg": tilt_deg,
            "display_label": display_label,
            "background_index": bg_idx,
            "background_name": str(prep["background_name"]),
            "detector_image": detector_image,
            "raw_detector_image": prep["raw_detector_image"],
            "caked_image": caked_density_image,
            "caked_integrated_intensity": prep["caked_integrated_intensity"],
            "caked_density_image": caked_density_image,
            "caked_raw_sum_image": prep["caked_raw_sum_image"],
            "caked_display_image": prep["caked_display_image"],
            "caked_intensity_mode": prep["caked_intensity_mode"],
            "caked_sum_signal": prep["caked_sum_signal"],
            "caked_sum_normalization": prep["caked_sum_normalization"],
            "caked_count": prep["caked_count"],
            "theta_axis": prep["theta_axis"],
            "phi_axis": prep["phi_axis"],
            "theta_map": prep["theta_map"],
            "phi_map": prep["phi_map"],
            "ai": prep["ai"],
            "caked_peak_model": caked_peak_model,
            "detector_peak_model": detector_peak_model,
            "caked_peak_subtracted_image": caked_peak_subtracted_image,
            "detector_peak_subtracted_image": detector_peak_subtracted_image,
            "fit_results": fit_results,
            "failures": failures,
            "tth_max": prep["tth_max"],
            "caked_projection_context": prep["caked_projection_context"],
            "qr_overlay_config": prep["qr_overlay_config"],
            "raw_azimuth_axis": prep["raw_azimuth_axis"],
            "transform_bundle": prep["transform_bundle"],
            "prep_elapsed_s": float(prep["prep_elapsed_s"]),
            "global_fit_elapsed_s": float(fit_elapsed),
        }
    )
    print(
        f"{display_label}: fit={len(fit_results)} fail={len(failures)} "
        f"tth_max={float(prep['tth_max']):.3f} background_subtraction=none "
        f"peak_subtraction_only={bool(PEAK_SUBTRACTION_ONLY)} "
        f"solid_angle_correction={bool(BACKGROUND_SOLID_ANGLE_CORRECTION)} "
        f"prep_elapsed={float(prep['prep_elapsed_s']):.2f}s global_fit_workers={global_fit_workers} fit_backend={active_fit_backend}"
    )

post_elapsed = time.perf_counter() - post_start
print(f"background result assembly: elapsed={post_elapsed:.2f}s")

fit_table = pd.DataFrame(all_table_rows, columns=FIT_TABLE_COLUMNS)
fit_table_path = OUT_DIR / "all_background_peak_fit_table.csv"
fit_table.to_csv(fit_table_path, index=False)
print(f"table={fit_table_path}")
fit_table.head()


def hkl_text(label: str) -> str:
    """Human-readable compressed Miller-index label as (HK,L)."""
    text = str(label).strip()
    try:
        h_value, k_value, l_value = [int(part.strip()) for part in text.split(",")]
    except ValueError:
        return f"({text.replace('-', '−')})"
    hk_value = h_value * h_value + h_value * k_value + k_value * k_value
    return f"({hk_value},{str(l_value).replace('-', '−')})"


def tilt_text(value: int | float) -> str:
    return f"{format_angle_value(value)}°"


def tilt_math(value: int | float) -> str:
    return rf"$\theta_i={format_angle_value(value)}^\circ$"


PHI_DISPLAY_MIN_DEG = -90.0
PHI_DISPLAY_MAX_DEG = 90.0
POSITIVE_QZ_MIN = 0.0


def positive_qz_mask(qz_map: object) -> np.ndarray:
    qz = np.asarray(qz_map, dtype=np.float64)
    return np.isfinite(qz) & (qz > POSITIVE_QZ_MIN)


def positive_qz_bounds(qz_lo: object, qz_hi: object) -> tuple[float, float] | None:
    lo = float(qz_lo)
    hi = float(qz_hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    lo, hi = sorted((lo, hi))
    if hi <= POSITIVE_QZ_MIN:
        return None
    return max(lo, POSITIVE_QZ_MIN), hi


def peak_item_has_positive_l(item: dict[str, object]) -> bool:
    try:
        parts = [int(part.strip()) for part in str(item.get("label", "")).split(",")]
    except Exception:
        return False
    return len(parts) == 3 and int(parts[2]) > 0


def positive_l_fit_items(items: object) -> list[dict[str, object]]:
    return [
        item for item in (items or []) if isinstance(item, dict) and peak_item_has_positive_l(item)
    ]


def phi_display_mask(phi_values: object) -> np.ndarray:
    phi = np.asarray(phi_values, dtype=np.float64)
    return np.isfinite(phi) & (phi >= PHI_DISPLAY_MIN_DEG) & (phi <= PHI_DISPLAY_MAX_DEG)


def caked_phi_display_slice(
    image: object, phi_axis: object, theta_axis: object
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    values = np.asarray(image, dtype=np.float64)
    phi = np.asarray(phi_axis, dtype=np.float64)
    theta = np.asarray(theta_axis, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] != phi.size:
        raise ValueError("caked image first axis must match phi_axis")
    mask = phi_display_mask(phi)
    if not np.any(mask):
        raise RuntimeError("no caked rows within -90 <= phi <= 90 deg")
    phi_visible = phi[mask]
    extent = [float(theta[0]), float(theta[-1]), float(phi_visible[0]), float(phi_visible[-1])]
    return values[mask, ...], phi_visible, extent


def detector_phi_display_mask(phi_map: object) -> np.ndarray:
    return phi_display_mask(phi_map)


def detector_theta_map_for_display(bg: dict[str, object]) -> np.ndarray:
    detector_shape = tuple(np.asarray(bg["detector_image"]).shape)
    cached_theta = bg.get("theta_map")
    if cached_theta is not None:
        theta = np.asarray(cached_theta, dtype=np.float64)
        if theta.shape == detector_shape:
            return theta
    ai_value = bg.get("ai")
    if ai_value is None:
        ai_value = FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
    geometry = getattr(ai_value, "geometry", ai_value)
    theta, _raw_phi_map = detector_pixel_angular_maps(detector_shape, geometry)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != detector_shape:
        raise ValueError("computed detector theta map shape must match detector image")
    bg["theta_map"] = theta
    return theta


def detector_phi_map_for_display(bg: dict[str, object]) -> np.ndarray:
    detector_shape = tuple(np.asarray(bg["detector_image"]).shape)
    cached_phi = bg.get("phi_map")
    if cached_phi is not None:
        phi = np.asarray(cached_phi, dtype=np.float64)
        if phi.shape == detector_shape:
            return phi

    config = bg.get("qr_overlay_config")
    helper = getattr(gui_qr_cylinder_overlay, "detector_gui_phi_map_for_projection", None)
    if helper is not None and config is not None:
        try:
            phi = helper(config=config, detector_shape=detector_shape)
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                if phi.shape == detector_shape:
                    bg["phi_map"] = phi
                    return phi
        except Exception:
            pass

    ai_value = bg.get("ai")
    if ai_value is None:
        ai_value = FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
    geometry = getattr(ai_value, "geometry", ai_value)
    _theta_map, raw_phi_map = detector_pixel_angular_maps(detector_shape, geometry)
    phi = np.asarray(raw_phi_to_gui_phi(raw_phi_map), dtype=np.float64)
    if phi.shape != detector_shape:
        raise ValueError("computed detector phi map shape must match detector image")
    bg["phi_map"] = phi
    return phi


def detector_qz_map_for_display(bg: dict[str, object]) -> np.ndarray | None:
    detector_shape = tuple(np.asarray(bg["detector_image"]).shape)
    cached_qz = bg.get("_detector_qz_map_display")
    if cached_qz is not None:
        qz = np.asarray(cached_qz, dtype=np.float64)
        if qz.shape == detector_shape:
            return qz
    config = bg.get("qr_overlay_config")
    helper = getattr(gui_qr_cylinder_overlay, "detector_qr_qz_maps_for_projection", None)
    if helper is None or config is None:
        return None
    try:
        maps = helper(config=config, detector_shape=detector_shape)
    except Exception:
        return None
    if not isinstance(maps, tuple) or len(maps) < 2:
        return None
    qz = np.asarray(maps[1], dtype=np.float64)
    if qz.shape != detector_shape:
        return None
    bg["_detector_qz_map_display"] = qz
    return qz


def masked_detector_display(image: object, phi_map: object) -> np.ma.MaskedArray:
    values = np.asarray(image, dtype=np.float64)
    mask = detector_phi_display_mask(phi_map)
    if values.shape != mask.shape:
        raise ValueError("detector image shape must match detector phi map")
    return np.ma.masked_where(~mask, values)


def positive_l_detector_peak_model_for_display(
    bg: dict[str, object], *, theta_map: np.ndarray | None = None, phi_map: np.ndarray | None = None
) -> np.ndarray:
    shape = tuple(np.asarray(bg["detector_image"]).shape)
    cached_model = bg.get("_positive_l_detector_peak_model_full_sum_display_v2")
    if cached_model is not None:
        cached = np.asarray(cached_model, dtype=np.float64)
        if cached.shape == shape:
            return cached
    items = positive_l_fit_items(bg.get("fit_results", []))
    if not items:
        model = np.zeros(shape, dtype=np.float32)
        bg["_positive_l_detector_peak_model_full_sum_display_v2"] = model
        return model
    theta = (
        detector_theta_map_for_display(bg)
        if theta_map is None
        else np.asarray(theta_map, dtype=np.float64)
    )
    phi = (
        detector_phi_map_for_display(bg)
        if phi_map is None
        else np.asarray(phi_map, dtype=np.float64)
    )
    model = render_detector_peak_model_full_sum(theta, phi, items)
    qz = detector_qz_map_for_display(bg)
    if qz is not None and qz.shape == model.shape:
        model = np.where(positive_qz_mask(qz), model, 0.0)
    bg["_positive_l_detector_peak_model_full_sum_display_v2"] = model
    return model


def positive_l_caked_peak_model_for_display(bg: dict[str, object]) -> np.ndarray:
    shape = tuple(np.asarray(bg.get("caked_peak_model", bg["caked_image"])).shape)
    cached_model = bg.get("_positive_l_caked_peak_model_display")
    if cached_model is not None:
        cached = np.asarray(cached_model, dtype=np.float64)
        if cached.shape == shape:
            return cached
    model = np.zeros(shape, dtype=np.float64)
    for item in positive_l_fit_items(bg.get("fit_results", [])):
        try:
            phi_idx = np.asarray(item["phi_idx"], dtype=int)
            theta_idx = np.asarray(item["theta_idx"], dtype=int)
            peak_fit = np.asarray(item["peak_fit"], dtype=np.float64)
        except Exception:
            continue
        if peak_fit.shape == (phi_idx.size, theta_idx.size):
            model[np.ix_(phi_idx, theta_idx)] += peak_fit
    bg["_positive_l_caked_peak_model_display"] = model
    return model


def detector_display_bbox(mask: object) -> tuple[tuple[float, float], tuple[float, float]]:
    display_mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(display_mask)
    if rows.size == 0:
        raise RuntimeError("no detector pixels within -90 <= phi <= 90 deg")
    return (float(np.nanmin(cols)) - 0.5, float(np.nanmax(cols)) + 0.5), (
        float(np.nanmax(rows)) + 0.5,
        float(np.nanmin(rows)) - 0.5,
    )


def detector_bottom_left_axis_tick_labels(
    x_ticks: object,
    y_ticks: object,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[list[str], list[str]]:
    x_values = np.asarray(x_ticks, dtype=np.float64)
    y_values = np.asarray(y_ticks, dtype=np.float64)
    x_left = float(min(xlim))
    y_bottom = float(max(ylim))
    x_labels = [f"{max(0.0, float(value) - x_left):.0f}" for value in x_values]
    y_labels = [f"{max(0.0, y_bottom - float(value)):.0f}" for value in y_values]
    return x_labels, y_labels


def detector_display_union_bbox(
    masks: list[np.ndarray],
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid_masks = [np.asarray(mask, dtype=bool) for mask in masks if np.asarray(mask).size]
    if not valid_masks:
        raise RuntimeError("no detector display masks available")
    union_mask = np.logical_or.reduce(valid_masks)
    return detector_display_bbox(union_mask)


def detector_point_in_display_mask(x: object, y: object, mask: object) -> bool:
    if x is None or y is None:
        return False
    display_mask = np.asarray(mask, dtype=bool)
    x_value = float(x)
    y_value = float(y)
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return False
    col = int(round(x_value))
    row = int(round(y_value))
    return (
        0 <= row < display_mask.shape[0]
        and 0 <= col < display_mask.shape[1]
        and bool(display_mask[row, col])
    )


def filter_detector_points_to_display(points: object, mask: object) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.atleast_2d(pts)
    keep = [detector_point_in_display_mask(x, y, mask) for x, y in pts[:, :2]]
    return pts[np.asarray(keep, dtype=bool), :2]


def item_center_in_phi_display(item: dict[str, object]) -> bool:
    params = np.asarray(item.get("params", []), dtype=np.float64)
    return params.size >= 3 and bool(phi_display_mask([float(params[2])])[0])


def fit_model_vmax(model: np.ndarray, pct: float = 99.7) -> float:
    positive = np.asarray(model, dtype=np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(positive, pct))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


DETECTOR_MODEL_DISPLAY_RELATIVE_FLOOR = 1.0e-4


def detector_display_cmap(name: str = JOURNAL_DETECTOR_CMAP) -> mpl.colors.Colormap:
    cmap = mpl.colormaps[name].copy()
    cmap.set_bad("#050505")
    cmap.set_under("#050505")
    return cmap


def detector_intensity_display(image: object) -> np.ma.MaskedArray:
    values = np.ma.asarray(image, dtype=np.float64)
    return np.ma.masked_where(~np.isfinite(values) | (values <= 0.0), values)


def detector_model_intensity_display(
    model: object, *, relative_floor: float = DETECTOR_MODEL_DISPLAY_RELATIVE_FLOOR
) -> np.ma.MaskedArray:
    values = np.asarray(model, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return np.ma.masked_where(np.ones(values.shape, dtype=bool), values)
    floor = max(
        float(np.nanmax(positive)) * float(relative_floor), float(np.nanpercentile(positive, 1.0))
    )
    return np.ma.masked_where(~np.isfinite(values) | (values <= floor), values)


def detector_log_norm(images: list[object], low: float = 0.8, high: float = 99.85) -> LogNorm:
    chunks = []
    for image in images:
        values = detector_intensity_display(image).compressed()
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            chunks.append(values)
    values = np.concatenate(chunks) if chunks else np.array([1.0], dtype=np.float64)
    vmin, vmax = np.nanpercentile(values, [low, high])
    if not np.isfinite(vmin) or vmin <= 0.0:
        vmin = float(np.nanmin(values[values > 0.0]))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(float(vmin) * 10.0, float(np.nanmax(values)))
    if vmax <= vmin:
        vmax = float(vmin) * 10.0
    return LogNorm(vmin=float(vmin), vmax=float(vmax), clip=True)


def fit_log_image(model: np.ndarray) -> np.ndarray:
    return np.ma.log(1.0 + np.ma.clip(np.ma.asarray(model, dtype=np.float64), 0.0, None))


def shared_limits(images: list[np.ndarray], high: float = 99.7) -> tuple[float, float]:
    chunks = []
    for image in images:
        values = np.ma.asarray(image, dtype=np.float64).compressed()
        values = values[np.isfinite(values)]
        if values.size:
            chunks.append(values.ravel())
    values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(values, 0.5))
    vmax = float(np.nanpercentile(values, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def detector_contour_levels(model: np.ndarray) -> np.ndarray:
    values = np.ma.asarray(model, dtype=np.float64).compressed()
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return np.array([], dtype=np.float64)
    levels = np.nanpercentile(values, [98.5, 99.35, 99.75])
    levels = np.unique(levels[np.isfinite(levels) & (levels > 0.0)])
    return levels


def overlay_fit_centers(
    ax,
    items: list[dict[str, object]],
    *,
    color: str = JOURNAL_CENTER_COLOR,
    size: float = 8.0,
    display_mask: np.ndarray | None = None,
) -> None:
    # Peak-center markers intentionally hidden in all diagnostic figures.
    return


def add_panel_label(
    ax,
    text: str,
    *,
    loc: str = "upper left",
    outside: bool = False,
    fontsize: float = 8.0,
    color: str | None = None,
) -> None:
    """Add a compact panel or row label without changing data limits."""
    inside_positions = {
        "upper left": (0.025, 0.975, "left", "top"),
        "upper right": (0.975, 0.975, "right", "top"),
        "lower left": (0.025, 0.025, "left", "bottom"),
    }
    outside_positions = {
        "upper left": (-0.085, 1.035, "left", "bottom"),
        "upper right": (1.010, 1.035, "right", "bottom"),
        "lower left": (-0.085, -0.105, "left", "top"),
    }
    positions = outside_positions if outside else inside_positions
    x, y, ha, va = positions.get(loc, positions["upper left"])
    if outside:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color="black" if color is None else color,
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="white" if color is None else color,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.56,
        },
        zorder=20,
    )


def add_panel_labels(axes, labels: tuple[str, ...] | None = None, *, outside: bool = True) -> None:
    return
    flat_axes = np.asarray(axes, dtype=object).ravel()
    if labels is None:
        labels = tuple(f"({chr(97 + idx)})" for idx in range(len(flat_axes)))
    for ax, label in zip(flat_axes, labels):
        if hasattr(ax, "axison") and ax.axison:
            add_panel_label(ax, label, outside=outside, fontsize=8.5)


def maybe_suptitle(fig, text: str, **kwargs) -> None:
    if False and PRL_SHOW_IN_PANEL_TITLES:
        fig.suptitle(text, **kwargs)


def finish_axes(ax, *, grid: bool = False) -> None:
    ax.tick_params(
        which="both", direction="in", top=True, right=True, length=2.8, width=0.60, pad=2.0
    )
    if grid:
        ax.grid(True, color=JOURNAL_GRID_COLOR, linewidth=0.45, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def save_manuscript_figure(fig, stem: str) -> tuple[Path, Path]:
    """Save a high-DPI PNG, with optional vector PDF/SVG for faster development runs."""
    png_path = FIGURE_OUT_DIR / f"{stem}.png"
    pdf_path = FIGURE_OUT_DIR / f"{stem}.pdf"
    svg_path = FIGURE_OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=MANUSCRIPT_DPI, bbox_inches="tight", pad_inches=0.018)
    if bool(SAVE_VECTOR_FIGURES):
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.018)
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.018)
    else:
        pdf_path = png_path
    return png_path, pdf_path


def fit_lookup_by_key() -> dict[tuple[int, str, str], dict[str, object]]:
    lookup = {}
    for item in all_fit_results:
        key = (int(item["background_index"]), str(item["label"]), str(item["branch"]))
        lookup[key] = item
    return lookup


def get_fit_item(bg_idx: int, label: str, branch: str) -> dict[str, object]:
    lookup = fit_lookup_by_key()
    key = (int(bg_idx), str(label), str(branch))
    if key not in lookup:
        raise KeyError(f"missing fit item {key}")
    return lookup[key]


fit_count = int(len(fit_table))
if expected_fit_count == 0:
    raise RuntimeError(NO_BACKGROUND_PEAK_ENTRIES_MESSAGE)
success_count = int(fit_table["success"].astype(bool).sum())
optimizer_success_count = (
    int(fit_table.get("optimizer_success", fit_table["success"]).astype(bool).sum())
    if not fit_table.empty
    else 0
)
failure_count = sum(len(bg["failures"]) for bg in background_results)
if fit_count != expected_fit_count or failure_count != 0:
    print(
        f"fit rows={fit_count} successes={success_count} optimizer_successes={optimizer_success_count} failures={failure_count}"
    )
    for bg in background_results:
        if bg["failures"]:
            print(bg["display_label"], bg["failures"])
    raise RuntimeError("fit validation failed")
if success_count != expected_fit_count:
    print(
        f"fit rows={fit_count}; accepted={success_count}; optimizer_successes={optimizer_success_count}; "
        f"failures={failure_count}; expected={expected_fit_count}"
    )
else:
    print(
        f"fit rows={fit_count}; successes={success_count}; optimizer_successes={optimizer_success_count}; "
        f"failures={failure_count}; expected={expected_fit_count}"
    )
print(
    f"labels: {SAMPLE_NAME}; tilts={', '.join(tilt_text(BACKGROUND_TILT_DEG[i]) for i in sorted(BACKGROUND_TILT_DEG))}"
)
print(
    f"orientation: apply_background_backend_orientation rotation_k={backend_rotation_k} (90 deg CW for this state)"
)


# Complementary record of which peaks were used after exclusions.
def write_used_peaks_note() -> tuple[Path, Path]:
    rows = []
    for tilt in sorted(fit_table["tilt_deg"].unique(), key=angle_sort_value):
        sub = fit_table[fit_table["tilt_deg"] == tilt].copy()
        sub = sub.sort_values(["fit_two_theta_deg", "fit_phi_deg", "label", "branch"])
        for peak_number, (_, row) in enumerate(sub.iterrows(), start=1):
            rows.append(
                {
                    "sample_name": SAMPLE_NAME,
                    "tilt_deg": float(tilt),
                    "peak_number": peak_number,
                    "hkl": str(row["label"]),
                    "branch": str(row["branch"]),
                    "q_group_key": str(row.get("q_group_key", "")),
                    "branch_id": str(row.get("branch_id", "")),
                    "source_branch_index": str(row.get("source_branch_index", "")),
                    "fit_two_theta_deg": float(row["fit_two_theta_deg"]),
                    "fit_phi_deg": float(row["fit_phi_deg"]),
                    "fit_detector_col": float(row["fit_detector_col"]),
                    "fit_detector_row": float(row["fit_detector_row"]),
                }
            )
    used = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"{USED_PEAKS_STEM}.csv"
    md_path = OUT_DIR / f"{USED_PEAKS_STEM}.md"
    used.to_csv(csv_path, index=False)

    lines = [
        f"# Figure 7 {SAMPLE_LABEL} used peaks",
        "",
        f"Source state: `{STATE_PATH}`",
        "",
        f"The peaks below are the final filtered set used for the {SAMPLE_NAME} fitted-background figures and line-profile analysis.",
        "Per-tilt figures label peaks directly with compact `(HK,L)` text.",
        "",
        "## Excluded peaks",
        "",
    ]
    for tilt, label in sorted(EXCLUDED_PEAKS_BY_TILT, key=lambda item: angle_sort_value(item[0])):
        lines.append(f"- {tilt_text(tilt)}: {hkl_text(label)}")
    lines.extend(["", "## Used peaks", ""])
    for tilt in sorted(used["tilt_deg"].unique(), key=angle_sort_value):
        sub = used[used["tilt_deg"] == tilt]
        lines.extend(
            [
                f"### {tilt_text(tilt)}",
                "",
                r"| # | (HK,L) | branch | source branch | fitted $2\theta$ (°) | fitted $\phi$ (°) | detector column (pixel) | detector row (pixel) |",
                "|---:|:---|:---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in sub.iterrows():
            lines.append(
                f"| {int(row['peak_number'])} | {hkl_text(row['hkl'])} | {row['branch']} | {row['source_branch_index']} | "
                f"{row['fit_two_theta_deg']:.4f} | {row['fit_phi_deg']:.4f} | "
                f"{row['fit_detector_col']:.1f} | {row['fit_detector_row']:.1f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


used_peaks_md, used_peaks_csv = write_used_peaks_note()
print(f"saved={used_peaks_md}")
print(f"saved={used_peaks_csv}")

# Figure 7a: detector-space measured images with fitted-peak contours and detector-space peak models.
ordered_backgrounds = sorted(background_results, key=lambda bg: angle_sort_value(bg["tilt_deg"]))
figure7a_detector_theta_maps = [detector_theta_map_for_display(bg) for bg in ordered_backgrounds]
figure7a_detector_phi_maps = [detector_phi_map_for_display(bg) for bg in ordered_backgrounds]
figure7a_detector_masks = [
    detector_phi_display_mask(phi_map) for phi_map in figure7a_detector_phi_maps
]
figure7a_detector_xlim, figure7a_detector_ylim = detector_display_union_bbox(
    figure7a_detector_masks
)
measured_display = [
    detector_intensity_display(masked_detector_display(bg["detector_image"], phi_map))
    for bg, phi_map in zip(ordered_backgrounds, figure7a_detector_phi_maps)
]
fit_display = [
    detector_model_intensity_display(
        masked_detector_display(
            positive_l_detector_peak_model_for_display(bg, theta_map=theta_map, phi_map=phi_map),
            phi_map,
        )
    )
    for bg, theta_map, phi_map in zip(
        ordered_backgrounds, figure7a_detector_theta_maps, figure7a_detector_phi_maps
    )
]
detector_figure_norm = detector_log_norm(measured_display + fit_display, high=99.85)
detector_figure_cmap = detector_display_cmap()

fig, axes = plt.subplots(
    len(ordered_backgrounds),
    2,
    figsize=(JOURNAL_FULL_WIDTH_IN, max(2.25 * len(ordered_backgrounds), 4.8)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.asarray(axes, dtype=object).reshape(len(ordered_backgrounds), 2)
for row, bg in enumerate(ordered_backgrounds):
    detector_image = bg["detector_image"]
    detector_model = positive_l_detector_peak_model_for_display(
        bg, theta_map=figure7a_detector_theta_maps[row], phi_map=figure7a_detector_phi_maps[row]
    )
    detector_phi_map = figure7a_detector_phi_maps[row]
    detector_display_mask = figure7a_detector_masks[row]
    detector_image_display = detector_intensity_display(
        masked_detector_display(detector_image, detector_phi_map)
    )
    detector_model_display = detector_model_intensity_display(
        masked_detector_display(detector_model, detector_phi_map)
    )
    fit_items = positive_l_fit_items(bg["fit_results"])
    fit_center_mask = detector_display_mask & ~np.ma.getmaskarray(detector_model_display)
    row_label = tilt_math(bg["tilt_deg"])

    ax = axes[row, 0]
    detector_figure_image = ax.imshow(
        detector_image_display, origin="upper", cmap=detector_figure_cmap, norm=detector_figure_norm
    )
    ax.set_xlim(*figure7a_detector_xlim)
    ax.set_ylim(*figure7a_detector_ylim)
    levels = detector_contour_levels(detector_model_display)
    if levels.size:
        ax.contour(
            detector_model_display,
            levels=levels,
            colors=JOURNAL_CONTOUR_COLOR,
            linewidths=0.50,
            origin="upper",
        )
    overlay_fit_centers(
        ax, fit_items, color=JOURNAL_CENTER_COLOR, size=7.0, display_mask=fit_center_mask
    )
    add_panel_label(ax, row_label)
    if row == 0:
        ax.set_title("Measured detector image with fitted contours")
    ax.set_ylabel("Detector row (pixel)")

    ax = axes[row, 1]
    ax.imshow(
        detector_model_display, origin="upper", cmap=detector_figure_cmap, norm=detector_figure_norm
    )
    ax.set_xlim(*figure7a_detector_xlim)
    ax.set_ylim(*figure7a_detector_ylim)
    overlay_fit_centers(ax, fit_items, color="white", size=6.0, display_mask=fit_center_mask)
    add_panel_label(ax, row_label)
    if row == 0:
        ax.set_title("Fitted detector-space peak model")

for ax in axes[-1, :]:
    ax.set_xlabel("Detector column (pixel)")
for ax in axes.ravel():
    finish_axes(ax)
detector_cbar = fig.colorbar(
    detector_figure_image, ax=axes.ravel().tolist(), fraction=0.020, pad=0.010, shrink=0.86
)
detector_cbar.set_label("Intensity (log scale)", fontsize=6.5)
detector_cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.45)

maybe_suptitle(fig, f"{SAMPLE_LABEL} detector-space peak fits", y=1.01)
fig7a_png, fig7a_pdf = save_manuscript_figure(fig, FIGURE7A_STEM)
plt.close(fig)
print(f"saved={fig7a_png}")
print(f"saved={fig7a_pdf}")
display(Image(filename=str(fig7a_png)))


# Individual background-vs-fit figures for each sample tilt in caked and detector views.
def caked_log_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    offset = float(np.nanpercentile(values, 1.0))
    return np.log1p(np.clip(values - offset, 0.0, None))


def peak_id_items(items: list[dict[str, object]]) -> list[tuple[int, dict[str, object]]]:
    ordered = sorted(
        items,
        key=lambda item: (
            float(item["params"][1]),
            float(item["params"][2]),
            str(item["label"]),
            str(item["branch"]),
        ),
    )
    return list(enumerate(ordered, start=1))


def peak_id_label(item: dict[str, object]) -> str:
    return hkl_text(str(item["label"]))


def annotate_peak_id(
    ax,
    peak_id: int | str,
    x: float,
    y: float,
    *,
    marker_color: str = JOURNAL_CONTOUR_COLOR,
    label_dx: float,
    label_dy: float,
) -> None:
    ax.annotate(
        str(peak_id),
        xy=(x, y),
        xytext=(x + label_dx, y + label_dy),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="black",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "0.15",
            "linewidth": 0.35,
            "alpha": 0.94,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": marker_color,
            "linewidth": 0.70,
            "shrinkA": 1.0,
            "shrinkB": 4.0,
        },
        zorder=8,
    )


def overlay_caked_peak_ids(
    ax,
    numbered_items: list[tuple[int, dict[str, object]]],
    *,
    marker_color: str = JOURNAL_CONTOUR_COLOR,
    text_color: str = "black",
) -> None:
    label_dx = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.012
    label_dy = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
    for _peak_id, item in numbered_items:
        x = float(item["params"][1])
        y = float(item["params"][2])
        if not bool(phi_display_mask([y])[0]):
            continue
        annotate_peak_id(
            ax,
            peak_id_label(item),
            x,
            y,
            marker_color=marker_color,
            label_dx=label_dx,
            label_dy=label_dy,
        )


def overlay_detector_peak_ids(
    ax,
    numbered_items: list[tuple[int, dict[str, object]]],
    *,
    detector_model: np.ndarray,
    marker_color: str = JOURNAL_CONTOUR_COLOR,
    text_color: str = "black",
    display_mask: np.ndarray | None = None,
) -> None:
    label_dx = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.018
    label_dy = -abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.018
    for _peak_id, item in numbered_items:
        x = item.get("fit_detector_col")
        y = item.get("fit_detector_row")
        if x is None or y is None:
            continue
        if display_mask is not None and not detector_point_in_display_mask(x, y, display_mask):
            continue
        annotate_peak_id(
            ax,
            peak_id_label(item),
            float(x),
            float(y),
            marker_color=marker_color,
            label_dx=label_dx,
            label_dy=label_dy,
        )


def add_peak_key(fig, numbered_items: list[tuple[int, dict[str, object]]]) -> None:
    lines = [f"{peak_id}: {peak_id_label(item)}" for peak_id, item in numbered_items]
    half = int(math.ceil(len(lines) / 2.0))
    key_text = "\n".join(lines[:half]) + "\n\n" + "\n".join(lines[half:])
    fig.text(
        1.015,
        0.50,
        key_text,
        ha="left",
        va="center",
        fontsize=6.0,
        family="serif",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.65",
            "linewidth": 0.6,
        },
    )


def save_background_fit_comparison(bg: dict[str, object]) -> tuple[Path, Path]:
    tilt = float(bg["tilt_deg"])
    fit_items = positive_l_fit_items(bg["fit_results"])
    numbered_items = peak_id_items([item for item in fit_items if item_center_in_phi_display(item)])
    theta_axis = bg["theta_axis"]
    phi_axis = bg["phi_axis"]
    caked_image = bg.get("caked_display_image", bg.get("caked_density_image", bg["caked_image"]))
    caked_mode_label = (
        "raw accumulated intensity"
        if bg.get("caked_intensity_mode") == "raw_sum"
        else "intensity density"
    )
    caked_model = positive_l_caked_peak_model_for_display(bg)
    detector_image = bg["detector_image"]
    detector_model = positive_l_detector_peak_model_for_display(bg)

    caked_image_display, _visible_phi_axis, caked_extent = caked_phi_display_slice(
        caked_image, phi_axis, theta_axis
    )
    caked_model_display, _visible_phi_axis, _caked_fit_extent = caked_phi_display_slice(
        caked_model, phi_axis, theta_axis
    )
    caked_bg = caked_log_image(caked_image_display)
    caked_fit = fit_log_image(caked_model_display)
    caked_bg_vmin, caked_bg_vmax = robust_display_limits(caked_bg, low=1.0, high=99.65)
    caked_fit_vmin, caked_fit_vmax = robust_display_limits(caked_fit, low=0.0, high=99.7)
    detector_phi_map = detector_phi_map_for_display(bg)
    detector_display_mask = detector_phi_display_mask(detector_phi_map)
    detector_xlim, detector_ylim = detector_display_bbox(detector_display_mask)
    detector_bg = detector_intensity_display(
        masked_detector_display(detector_image, detector_phi_map)
    )
    detector_model_display = detector_model_intensity_display(
        masked_detector_display(detector_model, detector_phi_map)
    )
    detector_fit_center_mask = detector_display_mask & ~np.ma.getmaskarray(detector_model_display)
    detector_norm = detector_log_norm([detector_bg, detector_model_display], high=99.85)
    detector_cmap = detector_display_cmap()

    fig, axes = plt.subplots(2, 2, figsize=(JOURNAL_FULL_WIDTH_IN, 6.9), constrained_layout=True)
    maybe_suptitle(
        fig,
        rf"{SAMPLE_LABEL}, $\theta_i={format_angle_value(tilt)}^\circ$: measured background and fitted peak model",
        y=1.01,
    )

    ax = axes[0, 0]
    ax.imshow(
        caked_bg,
        extent=caked_extent,
        origin="lower",
        aspect="auto",
        cmap=JOURNAL_INTENSITY_CMAP,
        vmin=caked_bg_vmin,
        vmax=caked_bg_vmax,
    )
    overlay_caked_peak_ids(ax, numbered_items, marker_color=JOURNAL_CONTOUR_COLOR)
    ax.set_title(f"Measured caked image ({caked_mode_label})")
    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel(r"$\phi$ (°)")

    ax = axes[0, 1]
    ax.imshow(
        caked_fit,
        extent=caked_extent,
        origin="lower",
        aspect="auto",
        cmap=JOURNAL_MODEL_CMAP,
        vmin=max(0.0, caked_fit_vmin),
        vmax=caked_fit_vmax,
    )
    overlay_caked_peak_ids(ax, numbered_items, marker_color=JOURNAL_CONTOUR_COLOR)
    ax.set_title("Fitted caked-space peak model")
    ax.set_xlabel(r"$2\theta$ (°)")
    ax.set_ylabel(r"$\phi$ (°)")

    ax = axes[1, 0]
    detector_comparison_image = ax.imshow(
        detector_bg, origin="upper", cmap=detector_cmap, norm=detector_norm
    )
    ax.set_xlim(*detector_xlim)
    ax.set_ylim(*detector_ylim)
    overlay_detector_peak_ids(
        ax,
        numbered_items,
        detector_model=detector_model,
        marker_color=JOURNAL_CONTOUR_COLOR,
        display_mask=detector_fit_center_mask,
    )
    ax.set_title("Measured detector image")
    ax.set_xlabel("Detector column (pixel)")
    ax.set_ylabel("Detector row (pixel)")

    ax = axes[1, 1]
    ax.imshow(detector_model_display, origin="upper", cmap=detector_cmap, norm=detector_norm)
    ax.set_xlim(*detector_xlim)
    ax.set_ylim(*detector_ylim)
    overlay_detector_peak_ids(
        ax,
        numbered_items,
        detector_model=detector_model_display,
        marker_color=JOURNAL_CONTOUR_COLOR,
        display_mask=detector_fit_center_mask,
    )
    ax.set_title("Fitted detector-space peak model")
    ax.set_xlabel("Detector column (pixel)")
    ax.set_ylabel("Detector row (pixel)")

    detector_cbar = fig.colorbar(
        detector_comparison_image, ax=axes[1, :].tolist(), fraction=0.032, pad=0.012, shrink=0.88
    )
    detector_cbar.set_label("Intensity (log scale)", fontsize=6.5)
    detector_cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.45)

    for ax in axes.ravel():
        finish_axes(ax)
    stem = f"{BACKGROUND_VS_FIT_STEM_PREFIX}_{angle_stem(tilt)}deg_background_vs_fit"
    png_path, pdf_path = save_manuscript_figure(fig, stem)
    plt.close(fig)
    return png_path, pdf_path


background_fit_comparison_paths = []
for bg in ordered_backgrounds:
    png_path, pdf_path = save_background_fit_comparison(bg)
    background_fit_comparison_paths.append((png_path, pdf_path))
    print(f"saved={png_path}")
    print(f"saved={pdf_path}")
    display(Image(filename=str(png_path)))

# Specular ROI example output intentionally disabled for this diagnostic figure.

# Figure 7c: local line-profile atlas comparing measured background to fitted 1D profile models.
FIGURE7C_PHI_PROFILE_FIT_STEM = f"{FIGURE7C_STEM}_phi_lorentzian_gaussian_fits"
FIGURE7C_PROFILE_QUALITY_STEM = f"{FIGURE7C_STEM}_profile_fit_quality"
PHI_GAUSSIAN_FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
PROFILE_PARAM_COUNT = 8
PROFILE_AXES = ("theta", "phi")
PROFILE_AXIS_CONFIG = {
    "theta": {
        "center_bound": CENTER_THETA_BOUND_DEG,
        "half_window": THETA_HALF_WINDOW_DEG,
        "min_width_floor": 0.020,
        "max_width": max(0.55, THETA_HALF_WINDOW_DEG * 0.95),
        "quality_nrmse_max": 0.22,
        "quality_r2_min": 0.65,
        "quality_tight_nrmse": 0.10,
    },
    "phi": {
        "center_bound": CENTER_PHI_BOUND_DEG,
        "half_window": PHI_HALF_WINDOW_DEG,
        "min_width_floor": 0.045,
        "max_width": max(2.8, PHI_HALF_WINDOW_DEG * 0.85),
        "quality_nrmse_max": 0.24,
        "quality_r2_min": 0.55,
        "quality_tight_nrmse": 0.09,
    },
}


def profile_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=axis)
    sums = np.nansum(np.where(finite, arr, 0.0), axis=axis)
    out = np.full(np.asarray(sums).shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def raw_averaged_line_profile(
    item: dict[str, object], axis: str, band_half_width: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_vals = np.asarray(item["theta_vals"], dtype=np.float64)
    phi_vals = np.asarray(item["phi_vals"], dtype=np.float64)
    measured = np.asarray(item["roi"], dtype=np.float64)
    fitted = np.asarray(item.get("peak_fit", item["fit"]), dtype=np.float64)
    theta0 = float(item["params"][1])
    phi0 = float(item["params"][2])

    phi_row = int(np.nanargmin(np.abs(wrapped_delta_deg(phi_vals, phi0))))
    theta_col = int(np.nanargmin(np.abs(theta_vals - theta0)))
    row_slice = slice(
        max(0, phi_row - band_half_width), min(measured.shape[0], phi_row + band_half_width + 1)
    )
    col_slice = slice(
        max(0, theta_col - band_half_width), min(measured.shape[1], theta_col + band_half_width + 1)
    )

    if axis == "theta":
        x = theta_vals - theta0
        measured_line = profile_nanmean(measured[row_slice, :], axis=0)
        fitted_line = profile_nanmean(fitted[row_slice, :], axis=0)
    elif axis == "phi":
        x = wrapped_delta_deg(phi_vals, phi0)
        measured_line = profile_nanmean(measured[:, col_slice], axis=1)
        fitted_line = profile_nanmean(fitted[:, col_slice], axis=1)
    else:
        raise ValueError(f"unknown profile axis {axis!r}")
    return x, measured_line, fitted_line


def normalize_line_pair(
    item: dict[str, object], measured_line: np.ndarray, fitted_line: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(measured_line) | np.isfinite(fitted_line)
    if not np.any(finite):
        return measured_line, fitted_line
    measured_abs = np.abs(np.asarray(measured_line, dtype=np.float64)[np.isfinite(measured_line)])
    fitted_abs = np.abs(np.asarray(fitted_line, dtype=np.float64)[np.isfinite(fitted_line)])
    candidate_scales = []
    if measured_abs.size:
        candidate_scales.append(float(np.nanpercentile(measured_abs, 99.0)))
    if fitted_abs.size:
        candidate_scales.append(float(np.nanpercentile(fitted_abs, 99.0)))
    candidate_scales.append(abs(float(item["params"][0])))
    scale = max([value for value in candidate_scales if np.isfinite(value)] + [1.0])
    return np.asarray(measured_line, dtype=np.float64) / scale, np.asarray(
        fitted_line, dtype=np.float64
    ) / scale


def averaged_line_profile(
    item: dict[str, object], axis: str, band_half_width: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, measured_line, fitted_line = raw_averaged_line_profile(
        item, axis=axis, band_half_width=band_half_width
    )
    measured_norm, fitted_norm = normalize_line_pair(item, measured_line, fitted_line)
    return x, measured_norm, fitted_norm


def gaussian_lorentzian_profile(params: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    if p.size < PROFILE_PARAM_COUNT:
        padded = np.zeros(PROFILE_PARAM_COUNT, dtype=np.float64)
        padded[: p.size] = p
        p = padded
    amp = p[0]
    center = p[1]
    sigma_g = max(float(p[2]), 1.0e-12)
    gamma_l = max(float(p[3]), 1.0e-12)
    gaussian_fraction = float(np.clip(p[4], 0.0, 1.0))
    baseline = p[5]
    slope = p[6]
    curvature = p[7]
    dx = x - center
    gaussian = np.exp(-0.5 * (dx / sigma_g) ** 2)
    lorentzian = 1.0 / (1.0 + (dx / gamma_l) ** 2)
    return (
        baseline
        + slope * dx
        + curvature * dx**2
        + amp * (gaussian_fraction * gaussian + (1.0 - gaussian_fraction) * lorentzian)
    )


def gaussian_lorentzian_phi_profile(params: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    return gaussian_lorentzian_profile(params, x_values)


def least_squares_parameter_stderr(result, n_points: int) -> np.ndarray:
    params = np.asarray(result.x, dtype=np.float64)
    stderr = np.full(params.shape, np.nan, dtype=np.float64)
    jac = np.asarray(result.jac, dtype=np.float64)
    if jac.ndim != 2 or jac.size == 0:
        return stderr
    dof = max(int(n_points) - int(params.size), 1)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * (2.0 * float(result.cost) / float(dof))
    except np.linalg.LinAlgError:
        return stderr
    diag = np.diag(cov)
    valid = np.isfinite(diag) & (diag >= 0.0)
    stderr[valid] = np.sqrt(diag[valid])
    return stderr


def hkl_tuple_from_label(label: str) -> tuple[int, int, int] | None:
    try:
        parts = [
            int(part.strip()) for part in str(label).replace("(", "").replace(")", "").split(",")
        ]
    except Exception:
        return None
    return tuple(parts[:3]) if len(parts) >= 3 else None


def is_hk_zero_peak(item: dict[str, object]) -> bool:
    hkl = hkl_tuple_from_label(str(item.get("label", "")))
    return bool(hkl is not None and hkl[0] == 0 and hkl[1] == 0)


def profile_width_guess(x_values: np.ndarray, signal: np.ndarray, axis: str) -> float:
    config = PROFILE_AXIS_CONFIG[axis]
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(finite) >= 3:
        x_fit = x[finite]
        y_fit = y[finite]
        peak = float(np.nanmax(y_fit))
        if np.isfinite(peak) and peak > 0.0:
            half_max = y_fit >= 0.5 * peak
            if np.count_nonzero(half_max) >= 2:
                fwhm = float(np.nanmax(x_fit[half_max]) - np.nanmin(x_fit[half_max]))
                if np.isfinite(fwhm) and fwhm > 0.0:
                    return float(
                        np.clip(0.5 * fwhm, config["min_width_floor"], config["max_width"])
                    )
        weights = y_fit
        center = float(np.sum(x_fit * weights) / np.sum(weights))
        variance = float(np.sum(weights * (x_fit - center) ** 2) / np.sum(weights))
        if np.isfinite(variance) and variance > 0.0:
            return float(
                np.clip(math.sqrt(variance), config["min_width_floor"], config["max_width"])
            )
    return float(
        np.clip(0.25 * config["center_bound"], config["min_width_floor"], config["max_width"])
    )


def _profile_empty_payload(axis: str, message: str, measured_raw: np.ndarray) -> dict[str, object]:
    return {
        "axis": axis,
        "success": False,
        "message": str(message),
        "cost": float("nan"),
        "params": np.full(PROFILE_PARAM_COUNT, np.nan, dtype=np.float64),
        "stderr": np.full(PROFILE_PARAM_COUNT, np.nan, dtype=np.float64),
        "model": np.full_like(np.asarray(measured_raw, dtype=np.float64), np.nan),
        "quality": {},
    }


def fit_gaussian_lorentzian_profile(
    axis: str, x_values: np.ndarray, measured_line: np.ndarray
) -> dict[str, object]:
    config = PROFILE_AXIS_CONFIG[axis]
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(measured_line, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 8:
        raise ValueError(f"too few finite {axis}-profile points")
    x_fit = np.ascontiguousarray(x[finite], dtype=np.float64)
    y_fit = np.ascontiguousarray(y[finite], dtype=np.float64)
    baseline0 = float(np.nanpercentile(y_fit, 8.0))
    high0 = float(np.nanpercentile(y_fit, 98.0))
    span0 = max(high0 - baseline0, float(np.nanmax(y_fit) - np.nanmin(y_fit)), 1.0)
    signal = np.clip(y_fit - baseline0, 0.0, None)
    peak_idx = int(np.nanargmax(signal if np.any(signal > 0.0) else y_fit))
    center0 = float(np.clip(x_fit[peak_idx], -config["center_bound"], config["center_bound"]))
    amp0 = max(float(y_fit[peak_idx] - baseline0), 0.20 * span0, 1.0)
    width0 = profile_width_guess(x_fit, signal, axis)
    if x_fit.size > 1:
        unique_x = np.sort(np.unique(x_fit))
        step = (
            float(np.nanmedian(np.abs(np.diff(unique_x))))
            if unique_x.size > 1
            else config["min_width_floor"]
        )
    else:
        step = config["min_width_floor"]
    min_width = max(float(step) * 0.40, float(config["min_width_floor"]))
    max_width = float(config["max_width"])
    slope_bound = 4.0 * span0 / max(float(config["half_window"]), 1.0e-6)
    curvature_bound = 4.0 * span0 / max(float(config["half_window"]) ** 2, 1.0e-6)
    x0 = np.array([amp0, center0, width0, width0, 0.55, baseline0, 0.0, 0.0], dtype=np.float64)
    lower = np.array(
        [
            0.0,
            -config["center_bound"],
            min_width,
            min_width,
            0.0,
            baseline0 - 3.0 * span0,
            -slope_bound,
            -curvature_bound,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            max(amp0 * 80.0, baseline0 + 8.0 * span0, amp0 + 1.0),
            config["center_bound"],
            max_width,
            max_width,
            1.0,
            high0 + 3.0 * span0,
            slope_bound,
            curvature_bound,
        ],
        dtype=np.float64,
    )
    x0 = np.minimum(np.maximum(x0, lower + 1.0e-9), upper - 1.0e-9)
    noise = max(
        float(np.nanmedian(np.abs(y_fit - np.nanmedian(y_fit)))) / 0.6745, 0.04 * span0, 1.0
    )
    denominator = np.ascontiguousarray(
        noise + 0.02 * np.clip(y_fit - baseline0, 0.0, None), dtype=np.float64
    )
    edge_weight = 1.0 + 0.15 * np.clip(
        np.abs(x_fit - center0) / max(float(config["half_window"]), 1.0e-6), 0.0, 1.5
    )
    edge_weight = np.ascontiguousarray(edge_weight, dtype=np.float64)

    def residual(params: np.ndarray) -> np.ndarray:
        model = gaussian_lorentzian_profile(params, x_fit)
        return edge_weight * (model - y_fit) / denominator

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.25,
        max_nfev=1600,
        x_scale=[
            max(amp0, 1.0),
            max(0.25 * config["center_bound"], min_width),
            width0,
            width0,
            0.35,
            max(abs(baseline0), span0, 1.0),
            max(slope_bound * 0.25, 1.0),
            max(curvature_bound * 0.25, 1.0),
        ],
    )
    params = np.asarray(result.x, dtype=np.float64)
    stderr = least_squares_parameter_stderr(result, int(np.count_nonzero(finite)))
    model = gaussian_lorentzian_profile(params, x)
    return {
        "axis": axis,
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "params": params,
        "stderr": stderr,
        "model": model,
        "quality": {},
    }


def fit_phi_gaussian_lorentzian_profile(
    x_values: np.ndarray, measured_line: np.ndarray
) -> dict[str, object]:
    return fit_gaussian_lorentzian_profile("phi", x_values, measured_line)


def profile_fit_quality(
    axis: str, x_values: np.ndarray, measured_line: np.ndarray, fit_payload: dict[str, object]
) -> dict[str, object]:
    config = PROFILE_AXIS_CONFIG[axis]
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(measured_line, dtype=np.float64)
    model = np.asarray(fit_payload.get("model", np.full_like(y, np.nan)), dtype=np.float64)
    params = np.asarray(
        fit_payload.get("params", np.full(PROFILE_PARAM_COUNT, np.nan)), dtype=np.float64
    )
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(model)
    n_points = int(np.count_nonzero(finite))
    if n_points < 8 or params.size < 5:
        return {"good": False, "reason": "too_few_profile_points", "n_points": n_points}
    y_fit = y[finite]
    model_fit = model[finite]
    residual = model_fit - y_fit
    baseline = float(np.nanpercentile(y_fit, 8.0))
    high = float(np.nanpercentile(y_fit, 98.0))
    span = max(
        high - baseline,
        float(np.nanmax(y_fit) - np.nanmin(y_fit)),
        abs(float(params[0])) if np.isfinite(params[0]) else 0.0,
        1.0,
    )
    rmse = float(np.sqrt(np.nanmean(residual**2)))
    nrmse = rmse / span
    mae = float(np.nanmedian(np.abs(residual)) / span)
    ss_res = float(np.nansum(residual**2))
    ss_tot = float(np.nansum((y_fit - float(np.nanmean(y_fit))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    model_peak_idx = int(np.nanargmax(model_fit)) if np.any(np.isfinite(model_fit)) else -1
    edge_margin = max(2, int(math.ceil(0.06 * n_points)))
    peak_not_on_edge = edge_margin <= model_peak_idx < (n_points - edge_margin)
    center_offset = abs(float(params[1])) if np.isfinite(params[1]) else float("inf")
    widths = np.asarray(params[2:4], dtype=np.float64)
    finite_widths = np.all(np.isfinite(widths)) and np.all(widths > 0.0)
    center_ok = center_offset <= 0.98 * float(config["center_bound"])
    shape_ok = bool(
        finite_widths
        and float(params[0]) > 0.0
        and 0.0 <= float(params[4]) <= 1.0
        and peak_not_on_edge
        and center_ok
    )
    fit_ok = bool(fit_payload.get("success", False))
    score_ok = bool(
        nrmse <= float(config["quality_nrmse_max"])
        and (r2 >= float(config["quality_r2_min"]) or nrmse <= float(config["quality_tight_nrmse"]))
    )
    good = bool(fit_ok and shape_ok and score_ok)
    reasons = []
    if not fit_ok:
        reasons.append("optimizer_failed")
    if not finite_widths:
        reasons.append("bad_width")
    if not center_ok:
        reasons.append("center_at_bound")
    if not peak_not_on_edge:
        reasons.append("peak_on_edge")
    if not score_ok:
        reasons.append("poor_match")
    return {
        "good": good,
        "reason": "ok" if good else ";".join(reasons or ["failed_quality_gate"]),
        "n_points": n_points,
        "rmse": rmse,
        "nrmse": float(nrmse),
        "mae": mae,
        "r2": r2,
        "span": float(span),
        "center_offset_abs_deg": float(center_offset),
        "peak_index": int(model_peak_idx),
        "edge_margin": int(edge_margin),
    }


def phi_profile_fit_report_row(
    bg: dict[str, object], item: dict[str, object], fit_payload: dict[str, object]
) -> dict[str, object]:
    label = str(item["label"])
    hkl = hkl_tuple_from_label(label) or (None, None, None)
    params = np.asarray(
        fit_payload.get("params", np.full(PROFILE_PARAM_COUNT, np.nan)), dtype=np.float64
    )
    stderr = np.asarray(
        fit_payload.get("stderr", np.full(PROFILE_PARAM_COUNT, np.nan)), dtype=np.float64
    )
    gaussian_fraction = float(params[4]) if params.size > 4 else float("nan")
    gaussian_fraction_err = float(stderr[4]) if stderr.size > 4 else float("nan")
    quality = dict(fit_payload.get("quality", {}))
    entry = dict(item.get("entry", {}))
    return {
        "sample_name": SAMPLE_NAME,
        "tilt_deg": float(bg["tilt_deg"]),
        "background_index": int(bg["background_index"]),
        "peak_label": label,
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        "branch": str(item["branch"]),
        "q_group_key": entry.get("q_group_key"),
        "q_group_m": entry.get("q_group_m"),
        "q_group_qr": entry.get("q_group_qr"),
        "success": bool(fit_payload.get("success", False)),
        "accepted_in_figure": bool(quality.get("good", False)),
        "quality_reason": str(quality.get("reason", "")),
        "profile_nrmse": float(quality.get("nrmse", float("nan"))),
        "profile_r2": float(quality.get("r2", float("nan"))),
        "message": str(fit_payload.get("message", "")),
        "phi_center_offset_deg": float(params[1]) if params.size > 1 else float("nan"),
        "phi_center_offset_deg_stderr": float(stderr[1]) if stderr.size > 1 else float("nan"),
        "gaussian_height_percent": 100.0 * gaussian_fraction,
        "gaussian_height_percent_stderr": 100.0 * gaussian_fraction_err,
        "lorentzian_height_percent": 100.0 * (1.0 - gaussian_fraction),
        "lorentzian_height_percent_stderr": 100.0 * gaussian_fraction_err,
        "gaussian_fwhm_phi_deg": PHI_GAUSSIAN_FWHM_FACTOR * float(params[2])
        if params.size > 2
        else float("nan"),
        "gaussian_fwhm_phi_deg_stderr": PHI_GAUSSIAN_FWHM_FACTOR * float(stderr[2])
        if stderr.size > 2
        else float("nan"),
        "lorentzian_fwhm_phi_deg": 2.0 * float(params[3]) if params.size > 3 else float("nan"),
        "lorentzian_fwhm_phi_deg_stderr": 2.0 * float(stderr[3])
        if stderr.size > 3
        else float("nan"),
        "fit_cost": float(fit_payload.get("cost", float("nan"))),
    }


ordered_backgrounds = sorted(background_results, key=lambda bg: angle_sort_value(bg["tilt_deg"]))
columns = [
    ("+", "theta", r"$+$ branch, $2\theta$ profiles"),
    ("+", "phi", r"$+$ branch, $\phi$ profiles"),
    ("-", "theta", r"$-$ branch, $2\theta$ profiles"),
    ("-", "phi", r"$-$ branch, $\phi$ profiles"),
]
axis_limits = {"theta": (-1.15, 1.15), "phi": (-3.9, 3.9)}
xlabels = {"theta": r"$\Delta 2\theta$ (°)", "phi": r"$\Delta\phi$ (°)"}
phi_profile_fit_rows = []
profile_fit_quality_rows = []
profile_fit_failures = []
PROFILE_USE_INDEPENDENT_1D_BACKGROUND_FITS = False


def fit_shared_linear_baseline_to_profile_residual(
    x_values: np.ndarray, measured_line: np.ndarray, peak_line: np.ndarray
) -> dict[str, object]:
    x = np.asarray(x_values, dtype=np.float64)
    measured = np.asarray(measured_line, dtype=np.float64)
    peak = np.asarray(peak_line, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(measured) & np.isfinite(peak)
    if np.count_nonzero(finite) < 3:
        zeros = np.zeros(x.shape, dtype=np.float64)
        return {"density": zeros, "intercept": 0.0, "slope": 0.0, "x_ref": 0.0, "success": False}

    residual = measured - peak
    x_ref = float(np.nanmedian(x[finite]))
    dx = x - x_ref
    design = np.column_stack([np.ones(np.count_nonzero(finite), dtype=np.float64), dx[finite]])
    y = residual[finite]
    try:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    except Exception:
        coef = np.asarray([float(np.nanmedian(y)), 0.0], dtype=np.float64)

    coef = np.asarray(coef, dtype=np.float64)
    for _ in range(6):
        resid = y - design @ coef
        mad = float(np.nanmedian(np.abs(resid - np.nanmedian(resid)))) if resid.size else 0.0
        sigma = max(1.4826 * mad, float(np.nanstd(resid)) if resid.size else 0.0, 1.0e-12)
        # Downweight positive residuals because they are usually unmodelled peak tails.
        weights = 1.0 / (1.0 + np.clip(resid / sigma, 0.0, None) ** 2)
        weights = np.clip(weights, 1.0e-3, 1.0)
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
        except Exception:
            break
        coef = np.asarray(coef, dtype=np.float64)

    if coef.size < 2 or not np.all(np.isfinite(coef[:2])):
        coef = np.asarray([float(np.nanmedian(y)), 0.0], dtype=np.float64)
    baseline = float(coef[0]) + float(coef[1]) * dx
    baseline = np.where(np.isfinite(baseline), baseline, 0.0)
    return {
        "density": baseline,
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "x_ref": x_ref,
        "success": True,
    }


def peak_only_profile_payload(
    axis: str, x_values: np.ndarray, measured_line: np.ndarray, peak_line: np.ndarray
) -> dict[str, object]:
    x = np.asarray(x_values, dtype=np.float64)
    peak_model = np.asarray(peak_line, dtype=np.float64)
    baseline_payload = fit_shared_linear_baseline_to_profile_residual(x, measured_line, peak_model)
    baseline = np.asarray(
        baseline_payload.get("density", np.zeros(x.shape, dtype=np.float64)), dtype=np.float64
    )
    model = peak_model + baseline
    finite_model = np.isfinite(x) & np.isfinite(model)
    params = np.full(PROFILE_PARAM_COUNT, np.nan, dtype=np.float64)
    if np.any(finite_model):
        peak_index = int(np.nanargmax(np.where(finite_model, peak_model, -np.inf)))
        params[0] = max(float(peak_model[peak_index]), 0.0)
        params[1] = float(x[peak_index])
        width = profile_width_guess(x, np.clip(peak_model, 0.0, None), axis)
        params[2] = width
        params[3] = width
        params[4] = 1.0
        x_ref = float(baseline_payload.get("x_ref", 0.0))
        intercept = float(baseline_payload.get("intercept", 0.0))
        slope = float(baseline_payload.get("slope", 0.0))
        # gaussian_lorentzian_profile represents the line as baseline + slope*(x-center).
        params[5] = intercept + slope * (float(params[1]) - x_ref)
        params[6] = slope
        params[7] = 0.0
    return {
        "axis": axis,
        "success": bool(np.any(finite_model)),
        "message": "projected_2d_peak_model_plus_shared_linear_baseline_no_background_subtraction",
        "cost": float("nan"),
        "params": params,
        "stderr": np.full(PROFILE_PARAM_COUNT, np.nan, dtype=np.float64),
        "model": model,
        "peak_model": peak_model,
        "linear_baseline": baseline,
        "linear_baseline_intercept": float(baseline_payload.get("intercept", np.nan)),
        "linear_baseline_slope": float(baseline_payload.get("slope", np.nan)),
        "linear_baseline_x_ref": float(baseline_payload.get("x_ref", np.nan)),
        "quality": {},
    }


def _fit_profile_cache_item(
    bg: dict[str, object], item: dict[str, object]
) -> tuple[int, dict[str, object]]:
    axis_payloads = {}
    failures = []
    for axis in PROFILE_AXES:
        x, measured_raw, local_fit_raw = raw_averaged_line_profile(
            item, axis=axis, band_half_width=1
        )
        if bool(PROFILE_USE_INDEPENDENT_1D_BACKGROUND_FITS):
            try:
                payload = fit_gaussian_lorentzian_profile(axis, x, measured_raw)
            except Exception as exc:
                payload = _profile_empty_payload(axis, str(exc), measured_raw)
                failures.append(
                    (float(bg["tilt_deg"]), str(item["label"]), str(item["branch"]), axis, str(exc))
                )
        else:
            payload = peak_only_profile_payload(axis, x, measured_raw, local_fit_raw)
        quality = profile_fit_quality(axis, x, measured_raw, payload)
        payload["quality"] = quality
        axis_payloads[axis] = {
            "x": x,
            "measured_raw": measured_raw,
            "local_fit_raw": local_fit_raw,
            "payload": payload,
            "quality": quality,
        }
    accepted = bool(all(axis_payloads[axis]["quality"].get("good", False) for axis in PROFILE_AXES))
    return id(item), {"axes": axis_payloads, "accepted": accepted, "failures": failures}


profile_fit_cache: dict[int, dict[str, object]] = {}
profile_targets = [(bg, item) for bg in ordered_backgrounds for item in bg["fit_results"]]
profile_worker_count = min(PROFILE_FIT_WORKERS, max(1, len(profile_targets)))
profile_fit_stage = pre_editor_cache_get_stage(
    pre_editor_cache, "profile_fits", PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE
)
profile_fit_cache_hit = profile_fit_stage_is_valid(
    profile_fit_stage, expected_profile_count=len(profile_targets)
)
if profile_fit_cache_hit:
    profile_fit_records = list(profile_fit_stage["profile_fit_records"])
    for (_bg, item), cached in zip(profile_targets, profile_fit_records, strict=False):
        if isinstance(cached, dict):
            profile_fit_cache[id(item)] = cached
            profile_fit_failures.extend(list(cached.get("failures", []) or []))
    print(f"reused pre-editor profile-fit cache={PRE_EDITOR_CACHE_PATH}")
else:
    if profile_worker_count > 1:
        with ThreadPoolExecutor(max_workers=profile_worker_count) as executor:
            future_to_item = {
                executor.submit(_fit_profile_cache_item, bg, item): (bg, item)
                for bg, item in profile_targets
            }
            for future in as_completed(future_to_item):
                bg, item = future_to_item[future]
                try:
                    item_id, cached = future.result()
                except Exception as exc:
                    axis_payloads = {}
                    for axis in PROFILE_AXES:
                        empty = np.array([], dtype=np.float64)
                        payload = _profile_empty_payload(axis, str(exc), empty)
                        quality = profile_fit_quality(axis, empty, empty, payload)
                        payload["quality"] = quality
                        axis_payloads[axis] = {
                            "x": empty,
                            "measured_raw": empty,
                            "local_fit_raw": empty,
                            "payload": payload,
                            "quality": quality,
                        }
                    cached = {
                        "axes": axis_payloads,
                        "accepted": False,
                        "failures": [
                            (
                                float(bg["tilt_deg"]),
                                str(item["label"]),
                                str(item["branch"]),
                                "both",
                                str(exc),
                            )
                        ],
                    }
                    item_id = id(item)
                profile_fit_cache[item_id] = cached
                profile_fit_failures.extend(cached["failures"])
    else:
        for bg, item in profile_targets:
            item_id, cached = _fit_profile_cache_item(bg, item)
            profile_fit_cache[item_id] = cached
            profile_fit_failures.extend(cached["failures"])
    pre_editor_cache = pre_editor_cache_with_stage(
        pre_editor_cache,
        "profile_fits",
        PRE_EDITOR_PROFILE_FIT_STAGE_SIGNATURE,
        {
            "profile_target_count": int(len(profile_targets)),
            "profile_fit_records": [profile_fit_cache[id(item)] for _bg, item in profile_targets],
        },
    )
    write_pre_editor_cache(PRE_EDITOR_CACHE_PATH, PRE_EDITOR_CACHE_KEY, pre_editor_cache)
    print(f"saved pre-editor profile-fit cache={PRE_EDITOR_CACHE_PATH}")

for bg, item in profile_targets:
    cached = profile_fit_cache[id(item)]
    axis_reasons = []
    for axis in PROFILE_AXES:
        quality = dict(cached["axes"][axis]["quality"])
        payload = cached["axes"][axis]["payload"]
        if not bool(quality.get("good", False)):
            axis_reasons.append(f"{axis}:{quality.get('reason', 'failed_quality_gate')}")
        profile_fit_quality_rows.append(
            {
                "sample_name": SAMPLE_NAME,
                "tilt_deg": float(bg["tilt_deg"]),
                "background_index": int(bg["background_index"]),
                "peak_label": str(item["label"]),
                "branch": str(item["branch"]),
                "axis": axis,
                "accepted_in_figure": bool(cached["accepted"]),
                "axis_good": bool(quality.get("good", False)),
                "reason": str(quality.get("reason", "")),
                "n_points": int(quality.get("n_points", 0)),
                "nrmse": float(quality.get("nrmse", float("nan"))),
                "r2": float(quality.get("r2", float("nan"))),
                "rmse": float(quality.get("rmse", float("nan"))),
                "mae": float(quality.get("mae", float("nan"))),
                "span": float(quality.get("span", float("nan"))),
                "center_offset_abs_deg": float(quality.get("center_offset_abs_deg", float("nan"))),
                "optimizer_success": bool(payload.get("success", False)),
                "optimizer_message": str(payload.get("message", "")),
                "item_rejection_reason": "ok"
                if bool(cached["accepted"])
                else ";".join(axis_reasons),
            }
        )
    if bool(cached["accepted"]):
        phi_payload = cached["axes"]["phi"]["payload"]
        if not is_hk_zero_peak(item):
            phi_profile_fit_rows.append(phi_profile_fit_report_row(bg, item, phi_payload))

accepted_profile_count = int(
    sum(1 for cached in profile_fit_cache.values() if bool(cached["accepted"]))
)
rejected_profile_count = int(len(profile_fit_cache) - accepted_profile_count)
print(
    f"profile fits cached={len(profile_fit_cache)} accepted={accepted_profile_count} rejected={rejected_profile_count} workers={profile_worker_count}"
)

max_branch_count = max(
    (
        sum(
            1
            for item in bg["fit_results"]
            if str(item["branch"]) == branch
            and bool(profile_fit_cache[id(item)]["accepted"])
            and item_center_in_phi_display(item)
        )
        for bg in ordered_backgrounds
        for branch in ("+", "-")
    ),
    default=1,
)
profile_row_height = max(3.0, 1.1 + 0.22 * max_branch_count)

fig, axes = plt.subplots(
    len(ordered_backgrounds),
    len(columns),
    figsize=(JOURNAL_ATLAS_WIDTH_IN, max(profile_row_height * len(ordered_backgrounds), 9.2)),
    constrained_layout=True,
)
axes = np.asarray(axes, dtype=object).reshape(len(ordered_backgrounds), len(columns))
for row, bg in enumerate(ordered_backgrounds):
    for col, (branch, axis, title) in enumerate(columns):
        ax = axes[row, col]
        items = [
            item
            for item in bg["fit_results"]
            if str(item["branch"]) == branch
            and bool(profile_fit_cache[id(item)]["accepted"])
            and item_center_in_phi_display(item)
        ]
        items = sorted(
            items, key=lambda item: (float(item["params"][1]), branch_sort_key(str(item["label"])))
        )
        offsets = np.arange(len(items), dtype=np.float64) * 1.18

        for offset, item in zip(offsets, items):
            cached_axis = profile_fit_cache[id(item)]["axes"][axis]
            x = np.asarray(cached_axis["x"], dtype=np.float64)
            measured_raw = np.asarray(cached_axis["measured_raw"], dtype=np.float64)
            payload = cached_axis["payload"]
            profile_fit_raw = np.asarray(
                payload.get("model", cached_axis["local_fit_raw"]), dtype=np.float64
            )
            measured_line, fitted_line = normalize_line_pair(item, measured_raw, profile_fit_raw)
            mask = np.isfinite(x) & np.isfinite(measured_line) & np.isfinite(fitted_line)
            if axis == "phi":
                mask &= phi_display_mask(float(item["params"][2]) + x)
            if not np.any(mask):
                continue
            ax.plot(
                x[mask],
                measured_line[mask] + offset,
                color=JOURNAL_DATA_COLOR,
                linewidth=0.72,
                alpha=0.90,
            )
            ax.plot(
                x[mask],
                fitted_line[mask] + offset,
                color=JOURNAL_FIT_COLOR,
                linewidth=0.88,
                alpha=0.96,
            )

        ax.set_xlim(*axis_limits[axis])
        ax.set_ylim(-0.35, (offsets[-1] + 1.05) if offsets.size else 1.0)
        ax.grid(True, axis="x", color=JOURNAL_GRID_COLOR, linewidth=0.45)
        ax.axvline(0.0, color="0.72", linewidth=0.45)
        if row == 0:
            ax.set_title(title)
        if row == len(ordered_backgrounds) - 1:
            ax.set_xlabel(xlabels[axis])
        if col in (0, 2):
            ax.set_yticks(offsets)
            ax.set_yticklabels([hkl_text(str(item["label"])) for item in items], fontsize=6.0)
        else:
            ax.set_yticks(offsets)
            ax.set_yticklabels([])
        if col == 0:
            ax.text(
                -0.34,
                0.5,
                tilt_math(bg["tilt_deg"]),
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.tick_params(axis="x", labelsize=6.5, length=2.0, width=0.45, direction="in", top=True)
        ax.tick_params(axis="y", length=0.0, width=0.0, pad=1.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)

legend_handles = [
    axes[0, 0].plot([], [], color=JOURNAL_DATA_COLOR, linewidth=0.9, label="Data")[0],
    axes[0, 0].plot([], [], color=JOURNAL_FIT_COLOR, linewidth=1.0, label="Simulation")[0],
]
fig.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.52, 1.015),
)
maybe_suptitle(fig, f"{SAMPLE_LABEL} accepted local line-profile fits", y=1.04)

phi_profile_fit_table = pd.DataFrame(phi_profile_fit_rows)
profile_fit_quality_table = pd.DataFrame(profile_fit_quality_rows)
phi_profile_fit_csv = OUT_DIR / f"{FIGURE7C_PHI_PROFILE_FIT_STEM}.csv"
phi_profile_fit_summary_csv = OUT_DIR / f"{FIGURE7C_PHI_PROFILE_FIT_STEM}_summary.csv"
phi_profile_fit_note = OUT_DIR / f"{FIGURE7C_PHI_PROFILE_FIT_STEM}.md"
profile_fit_quality_csv = OUT_DIR / f"{FIGURE7C_PROFILE_QUALITY_STEM}.csv"
phi_profile_fit_table.to_csv(phi_profile_fit_csv, index=False)
profile_fit_quality_table.to_csv(profile_fit_quality_csv, index=False)

summary_rows = []
metric_columns = [
    "gaussian_height_percent",
    "lorentzian_height_percent",
    "gaussian_fwhm_phi_deg",
    "lorentzian_fwhm_phi_deg",
]
if not phi_profile_fit_table.empty:
    good_phi_fits = phi_profile_fit_table[
        phi_profile_fit_table["success"].astype(bool)
        & phi_profile_fit_table["accepted_in_figure"].astype(bool)
    ].copy()
    for (peak_label, branch), sub in good_phi_fits.groupby(["peak_label", "branch"], sort=True):
        row_summary = {
            "sample_name": SAMPLE_NAME,
            "peak_label": peak_label,
            "branch": branch,
            "h": sub["h"].iloc[0],
            "k": sub["k"].iloc[0],
            "l": sub["l"].iloc[0],
            "fit_count": int(len(sub)),
        }
        for metric in metric_columns:
            values = np.asarray(sub[metric], dtype=np.float64)
            values = values[np.isfinite(values)]
            row_summary[f"{metric}_mean"] = (
                float(np.nanmean(values)) if values.size else float("nan")
            )
            row_summary[f"{metric}_observed_sem"] = (
                float(np.nanstd(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else float("nan")
            )
            stderr_col = f"{metric}_stderr"
            if stderr_col in sub:
                stderr_values = np.asarray(sub[stderr_col], dtype=np.float64)
                stderr_values = stderr_values[np.isfinite(stderr_values)]
                row_summary[f"{metric}_fit_stderr_on_mean"] = (
                    float(math.sqrt(np.nansum(stderr_values**2)) / max(stderr_values.size, 1))
                    if stderr_values.size
                    else float("nan")
                )
        summary_rows.append(row_summary)
phi_profile_fit_summary = pd.DataFrame(summary_rows)
phi_profile_fit_summary.to_csv(phi_profile_fit_summary_csv, index=False)

phi_fit_note_lines = [
    f"# {SAMPLE_NAME} phi-profile Gaussian/Lorentzian fits",
    "",
    "Only non-(0,L) rod peaks accepted in Figure 7c are reported.",
    "Figure 7c displays the projected 2D fitted peak model without subtracting a percentile/local baseline.",
    "Independent 1D pseudo-Voigt background/profile fits are disabled by default in this notebook.",
    "Gaussian/Lorentzian percentage columns are retained for schema compatibility and are not used for the plotted peak-only model.",
    "The summary CSV groups accepted peak-only profiles by peak label and branch; observed SEM is across all used tilts/instances when available.",
    "Rows are hidden from Figure 7c unless both profile-axis fits pass the quality gates.",
    "",
    f"Detailed CSV: `{phi_profile_fit_csv.name}`",
    f"Summary CSV: `{phi_profile_fit_summary_csv.name}`",
    f"Quality CSV: `{profile_fit_quality_csv.name}`",
    f"Accepted profile rows: {accepted_profile_count}",
    f"Rejected profile rows: {rejected_profile_count}",
    f"Reported phi rows: {len(phi_profile_fit_table)}",
    f"Successful reported phi rows: {int(phi_profile_fit_table['success'].sum()) if not phi_profile_fit_table.empty else 0}",
]
phi_profile_fit_note.write_text("\n".join(phi_fit_note_lines) + "\n", encoding="utf-8")
print(f"saved={phi_profile_fit_csv}")
print(f"saved={phi_profile_fit_summary_csv}")
print(f"saved={profile_fit_quality_csv}")
print(f"saved={phi_profile_fit_note}")
if profile_fit_failures:
    print("profile fit failures:")
    for failure in profile_fit_failures:
        print(
            f"  tilt={failure[0]} label={failure[1]} branch={failure[2]} axis={failure[3]} reason={failure[4]}"
        )

fig7c_png, fig7c_pdf = save_manuscript_figure(fig, FIGURE7C_STEM)
plt.close(fig)
print(f"saved={fig7c_png}")
print(f"saved={fig7c_pdf}")
display(Image(filename=str(fig7c_png)))

# $Q_r$-rod profiles from detector-space $Q_r$/$Q_z$ integration for one selected background.
import importlib
from collections import Counter
from ra_sim.simulation import exact_qspace_portable
from ra_sim.simulation.exact_cake import integrate_detector_to_cake_lut

try:
    from ra_sim.simulation import intersection_analysis as _intersection_analysis

    _intersection_analysis = importlib.reload(_intersection_analysis)
except Exception:
    _intersection_analysis = None
IntersectionGeometry = getattr(_intersection_analysis, "IntersectionGeometry", None)
detector_points_to_sample_qr_qz = getattr(
    _intersection_analysis, "detector_points_to_sample_qr_qz", None
)
project_qr_cylinder_to_detector = getattr(
    _intersection_analysis, "project_qr_cylinder_to_detector", None
)

ordered_backgrounds = sorted(background_results, key=lambda bg: angle_sort_value(bg["tilt_deg"]))


def select_rod_profile_tilt_deg(backgrounds: list[dict[str, object]]) -> float:
    available = [float(bg["tilt_deg"]) for bg in backgrounds]
    if not available:
        raise RuntimeError("no backgrounds available for Qr rod profiles")
    override_text = _setting_text("ROD_PROFILE_TILT_OVERRIDE", "RA_SIM_ROD_PROFILE_TILT_DEG", "")
    if override_text:
        requested = as_float(override_text)
        if not np.isfinite(requested):
            raise ValueError(f"invalid ROD_PROFILE_TILT_OVERRIDE={override_text!r}")
        for value in available:
            if abs(value - requested) < 1.0e-6:
                return float(value)
        raise ValueError(
            f"requested rod-profile incident angle {format_angle_value(requested)} deg is not among parsed backgrounds: {', '.join(format_angle_value(v) for v in available)}"
        )
    return float(min(available))


ROD_PROFILE_TILT_DEG = select_rod_profile_tilt_deg(ordered_backgrounds)
ROD_PROFILE_TILT_LABEL = format_angle_value(ROD_PROFILE_TILT_DEG)
ROD_PROFILE_TILT_STEM = angle_stem(ROD_PROFILE_TILT_DEG)
ROD_PROFILE_MARKER_STEM = f"{ROD_PROFILE_STEM}_peak_markers_{ROD_PROFILE_TILT_STEM}deg"
ROD_PROFILE_REGION_STEM = (
    f"{ROD_PROFILE_STEM}_detector_selected_q_regions_{ROD_PROFILE_TILT_STEM}deg"
)


def parse_hkl_label(label: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in str(label).split(",")]
    if len(parts) != 3:
        raise ValueError(f"invalid H,K,L label {label!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def rod_m_from_hk(h: int, k: int) -> int:
    return int(h * h + h * k + k * k)


def active_lattice_qr_value_for_m(m_idx: object, lattice_a: object) -> float:
    m_value = as_float(m_idx)
    a_value = as_float(lattice_a)
    if not (np.isfinite(m_value) and np.isfinite(a_value)) or m_value < 0.0 or a_value <= 0.0:
        return float("nan")
    return float((2.0 * np.pi / a_value) * np.sqrt((4.0 / 3.0) * m_value))


def active_lattice_qz_value_for_l(l_idx: object, lattice_c: object) -> float:
    l_value = as_float(l_idx)
    c_value = as_float(lattice_c)
    if not (np.isfinite(l_value) and np.isfinite(c_value)) or c_value <= 0.0:
        return float("nan")
    return float((2.0 * np.pi / c_value) * l_value)


def derived_primary_rod_entry_for_m(m_value: object, lattice_a: object) -> dict[str, object] | None:
    try:
        m_int = int(m_value)
    except Exception:
        return None
    if m_int <= 0:
        return None
    qr_value = active_lattice_qr_value_for_m(m_int, lattice_a)
    if not np.isfinite(qr_value):
        return None
    return {
        "key": ("generated_qr", "primary", int(m_int), "active_lattice"),
        "source": "primary",
        "source_label": "primary",
        "m": int(m_int),
        "qr": float(qr_value),
        "qr_source": "active_lattice",
        "generated": True,
    }


def profile_rod_entries_from_q_group_rows(
    q_group_rows: object,
    *,
    candidate_m_values: object = None,
    lattice_a: object = None,
    allow_generated: bool = False,
) -> list[dict[str, object]]:
    q_group_by_m = {}
    for row in q_group_rows or []:
        if not isinstance(row, dict) or row.get("included", True) is False:
            continue
        key = row.get("key")
        try:
            m_value = int(key[2]) if isinstance(key, (list, tuple)) and len(key) >= 3 else None
            qr_value = float(row["qr"])
            qz_value = float(row.get("qz", np.nan))
        except Exception:
            continue
        if m_value is not None and int(m_value) > 0 and m_value not in q_group_by_m:
            q_group_key = tuple(key) if isinstance(key, (list, tuple)) else key
            source = (
                str(key[1])
                if isinstance(key, (list, tuple)) and len(key) >= 2
                else str(row.get("source", "primary"))
            )
            source_label = str(row.get("source_label", source))
            q_group_by_m[m_value] = {
                "key": q_group_key,
                "qr": qr_value,
                "qz": qz_value,
                "source": source,
                "source_label": source_label,
            }

    rods = {}
    for m_value, q_group in q_group_by_m.items():
        rods.setdefault(
            int(m_value),
            {
                "key": ("q_group", q_group["source"], int(m_value), "rod"),
                "q_group_key": q_group["key"],
                "source": q_group["source"],
                "source_label": q_group["source_label"],
                "m": int(m_value),
                "qr": float(q_group["qr"]),
                "qz": float(q_group["qz"]),
                "qr_source": "saved_q_group_rows",
                "generated": False,
            },
        )

    if bool(allow_generated):
        for raw_m in candidate_m_values or []:
            try:
                m_value = int(raw_m)
            except Exception:
                continue
            if m_value in rods:
                continue
            derived = derived_primary_rod_entry_for_m(m_value, lattice_a)
            if derived is not None:
                rods[m_value] = derived
    return [rods[m] for m in sorted(rods)]


def detector_complete_branch_rod_entries(
    rod_entries: object,
    region_overlays: object,
    *,
    required_branches: tuple[str, ...] = ("-", "+"),
) -> list[dict[str, object]]:
    rods = [dict(row) for row in rod_entries or [] if isinstance(row, dict)]
    overlay_branches: dict[tuple[str, int], set[str]] = {}
    for item in region_overlays or []:
        if not isinstance(item, dict):
            continue
        try:
            m_value = int(item["m"])
            qz_min = float(item.get("qz_min", np.nan))
            qz_max = float(item.get("qz_max", np.nan))
        except Exception:
            continue
        if not (np.isfinite(qz_min) and np.isfinite(qz_max) and qz_max > qz_min):
            continue
        source = str(item.get("source", ""))
        branch = str(item.get("branch", ""))
        overlay_branches.setdefault((source, m_value), set()).add(branch)

    required = {str(branch) for branch in required_branches}
    complete: list[dict[str, object]] = []
    for rod in rods:
        try:
            m_value = int(rod["m"])
        except Exception:
            continue
        source = str(rod.get("source", ""))
        if required.issubset(overlay_branches.get((source, m_value), set())):
            complete.append(rod)
    return complete


def detector_overlay_rod_entries(
    rod_entries: object,
    *,
    allow_generated: bool = False,
    region_overlays: object = None,
) -> list[dict[str, object]]:
    rods = [dict(row) for row in rod_entries or [] if isinstance(row, dict)]
    if bool(allow_generated):
        source_filtered = rods
    else:
        source_filtered = [row for row in rods if not bool(row.get("generated", False))]
    if region_overlays is None:
        return source_filtered
    return detector_complete_branch_rod_entries(source_filtered, region_overlays)


def rod_reference_source_summary(
    rod_entries: object,
    *,
    candidate_m_values: object = None,
    allow_generated: bool = False,
) -> dict[str, object]:
    rods = [dict(row) for row in rod_entries or [] if isinstance(row, dict)]
    available_m: set[int] = set()
    saved = 0
    generated = 0
    for row in rods:
        if bool(row.get("generated", False)):
            generated += 1
        else:
            saved += 1
        try:
            available_m.add(int(row["m"]))
        except Exception:
            continue
    candidate_m: set[int] = set()
    for raw_m in candidate_m_values or []:
        try:
            m_value = int(raw_m)
        except Exception:
            continue
        if m_value > 0:
            candidate_m.add(m_value)
    skipped_generated = len(candidate_m - available_m)
    return {
        "saved": int(saved),
        "generated": int(generated),
        "skipped_generated": int(skipped_generated),
        "allow_generated": bool(allow_generated),
    }


def fit_result_m_values_for_rod_entries(backgrounds: object, *, tilt_deg: object) -> list[int]:
    tilt_value = as_float(tilt_deg)
    values: set[int] = set()
    for bg in backgrounds or []:
        if not isinstance(bg, dict):
            continue
        bg_tilt = as_float(bg.get("tilt_deg"))
        if np.isfinite(tilt_value) and np.isfinite(bg_tilt) and abs(bg_tilt - tilt_value) > 1.0e-6:
            continue
        for item in bg.get("fit_results", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                h, k, _l_value = parse_hkl_label(str(item["label"]))
            except Exception:
                continue
            m_value = rod_m_from_hk(h, k)
            if int(m_value) > 0:
                values.add(int(m_value))
    return sorted(values)


def build_profile_rod_entries(tilt_deg: float) -> list[dict[str, object]]:
    candidate_m_values = fit_result_m_values_for_rod_entries(
        globals().get("ordered_backgrounds", []), tilt_deg=tilt_deg
    )
    geometry = state.get("geometry", {}) if isinstance(state.get("geometry"), dict) else {}
    return profile_rod_entries_from_q_group_rows(
        geometry.get("q_group_rows", []),
        candidate_m_values=candidate_m_values,
        lattice_a=ACTIVE_LATTICE_A,
        allow_generated=ALLOW_GENERATED_ROD_REFERENCES,
    )


def projected_qz_at_phi_endpoint(samples: dict[str, object], target_phi: float) -> float | None:
    phi_values = np.asarray(samples.get("phi"), dtype=np.float64)
    qz_values = np.asarray(samples.get("qz"), dtype=np.float64)
    two_theta_values = np.asarray(samples.get("two_theta"), dtype=np.float64)
    if two_theta_values.shape != phi_values.shape:
        two_theta_values = np.full(phi_values.shape, np.nan, dtype=np.float64)
    finite = (
        np.isfinite(phi_values)
        & np.isfinite(qz_values)
        & np.isfinite(two_theta_values)
        & (two_theta_values <= float(ROD_PROFILE_MAX_TWO_THETA_DEG))
    )
    if not np.any(finite):
        return None
    local_index = int(np.nanargmin(np.abs(phi_values[finite] - float(target_phi))))
    absolute_index = np.flatnonzero(finite)[local_index]
    return float(qz_values[absolute_index])


def qz_bounds_for_phi_window(
    samples: dict[str, object], phi_min: float, phi_max: float
) -> tuple[float, float] | None:
    phi_values = np.asarray(samples.get("phi"), dtype=np.float64)
    qz_values = np.asarray(samples.get("qz"), dtype=np.float64)
    two_theta_values = np.asarray(samples.get("two_theta"), dtype=np.float64)
    if two_theta_values.shape != phi_values.shape:
        two_theta_values = np.full(phi_values.shape, np.nan, dtype=np.float64)
    selected = (
        np.isfinite(phi_values)
        & np.isfinite(qz_values)
        & np.isfinite(two_theta_values)
        & (two_theta_values <= float(ROD_PROFILE_MAX_TWO_THETA_DEG))
        & (phi_values >= float(phi_min))
        & (phi_values <= float(phi_max))
    )
    if np.count_nonzero(selected) < 2:
        return None
    branch_edge_phi = float(phi_min) if float(phi_max) <= 0.0 else float(phi_max)
    branch_edge_qz = projected_qz_at_phi_endpoint(samples, branch_edge_phi)
    selected_qz = qz_values[selected]
    if branch_edge_qz is not None and np.isfinite(branch_edge_qz):
        selected_qz = np.concatenate([selected_qz, np.asarray([branch_edge_qz], dtype=np.float64)])
    lo = float(np.nanmin(selected_qz))
    hi = float(np.nanmax(selected_qz))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def filter_projected_samples_by_branch_sign(
    samples: dict[str, object], branch_sign: int | None
) -> dict[str, object]:
    if branch_sign is None or "branch_sign" not in samples:
        return samples
    sign_values = np.asarray(samples.get("branch_sign"), dtype=np.int16)
    selected = sign_values == int(branch_sign)
    filtered = dict(samples)
    for key in ("two_theta", "phi", "qz", "branch_sign"):
        values = np.asarray(samples.get(key))
        if values.shape == selected.shape:
            filtered[key] = values[selected]
    filtered["selected_branch_sign"] = int(branch_sign)
    return filtered


def _state_peak_record_marker_item(record: dict[str, object]) -> dict[str, object] | None:
    try:
        label = str(record.get("label"))
        h, k, _l = parse_hkl_label(label)
        theta0 = float(record.get("two_theta_deg", record.get("caked_x")))
        phi0 = float(record.get("phi_deg", record.get("caked_y")))
    except Exception:
        return None
    if not (np.isfinite(theta0) and np.isfinite(phi0)):
        return None
    return {
        "label": label,
        "branch": branch_from_phi(phi0),
        "params": np.asarray(
            [float(record.get("intensity", 1.0)), theta0, phi0, 0.35, 1.2, 0.0], dtype=np.float64
        ),
        "_qz_marker": as_float(record.get("qz"), np.nan),
        "_marker_source": "state_peak_record",
        "_rod_m": rod_m_from_hk(h, k),
    }


def candidate_marker_items_for_background(bg: dict[str, object]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in bg.get("fit_results", []) or []:
        key = (str(item.get("label")), str(item.get("branch")))
        items.append(item)
        seen.add(key)

    bg_index = int(bg.get("background_index", -1))
    for record in state.get("geometry", {}).get("peak_records", []) or []:
        if not isinstance(record, dict):
            continue
        try:
            if int(record.get("background_index", -999999)) != bg_index:
                continue
        except Exception:
            continue
        item = _state_peak_record_marker_item(record)
        if item is None:
            continue
        key = (str(item.get("label")), str(item.get("branch")))
        if key in seen:
            continue
        items.append(item)
        seen.add(key)
    return items


def fit_items_for_rod_branch(
    bg: dict[str, object], rod: dict[str, object], branch_name: str
) -> list[dict[str, object]]:
    items = []
    rejected_rod_keys = active_rejected_rod_keys()
    for item in candidate_marker_items_for_background(bg):
        row = dict(item)
        row["_required_branch"] = str(branch_name)
        if accept_marker_for_plotted_rod(row, rod, rejected_rod_keys):
            items.append(item)
    return items


def rod_profile_two_theta_limit_for_background(
    bg: dict[str, object], configured_limit: float
) -> float:
    candidates = [float(configured_limit)]
    for item in candidate_marker_items_for_background(bg):
        try:
            theta0 = float(np.asarray(item["params"], dtype=np.float64)[1])
        except Exception:
            continue
        if np.isfinite(theta0):
            candidates.append(theta0 + 1.0)
    theta_axis = np.asarray(bg.get("theta_axis", []), dtype=np.float64)
    theta_axis = theta_axis[np.isfinite(theta_axis)]
    axis_max = float(np.nanmax(theta_axis)) if theta_axis.size else np.inf
    limit = max(candidates) if candidates else float(configured_limit)
    if np.isfinite(axis_max):
        limit = min(limit, axis_max)
    return max(float(configured_limit), float(limit))


def select_projected_samples_for_rod_branch(
    bg: dict[str, object],
    rod: dict[str, object],
    samples: dict[str, object],
    *,
    branch_name: str,
    phi_min: float,
    phi_max: float,
) -> tuple[dict[str, object], int | None]:
    sign_values = np.asarray(samples.get("branch_sign", []), dtype=np.int16).reshape(-1)
    if sign_values.size == 0:
        return samples, None
    signs = [int(value) for value in sorted(set(sign_values.tolist())) if int(value) != 0]
    if not signs:
        return samples, None

    tth = np.asarray(samples.get("two_theta"), dtype=np.float64)
    phi = np.asarray(samples.get("phi"), dtype=np.float64)
    qz = np.asarray(samples.get("qz"), dtype=np.float64)
    fit_items = fit_items_for_rod_branch(bg, rod, branch_name)
    scored: list[tuple[float, int]] = []
    for sign in signs:
        selected = (
            (sign_values == sign)
            & np.isfinite(tth)
            & np.isfinite(phi)
            & np.isfinite(qz)
            & (phi >= float(phi_min))
            & (phi <= float(phi_max))
        )
        if not np.any(selected):
            continue
        if fit_items:
            distances = []
            for item in fit_items:
                theta0 = float(item["params"][1])
                phi0 = float(item["params"][2])
                distance = ((tth[selected] - theta0) / 0.6) ** 2 + (
                    wrapped_delta_deg(phi[selected], phi0) / 3.0
                ) ** 2
                distances.append(float(np.nanmin(distance)))
            score = float(np.nanmedian(distances))
        else:
            score = -float(np.count_nonzero(selected))
        scored.append((score, sign))
    if not scored:
        return samples, None
    _score, branch_sign = min(scored, key=lambda item: item[0])
    return filter_projected_samples_by_branch_sign(samples, branch_sign), int(branch_sign)


def config_float_attr(config: object, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(config, name, default))
    except Exception:
        return float(default)


def notebook_detector_qr_qz_maps(
    config: object, detector_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    helper = getattr(gui_qr_cylinder_overlay, "detector_qr_qz_maps_for_projection", None)
    if helper is None:
        try:
            reloaded_overlay = importlib.reload(gui_qr_cylinder_overlay)
            globals()["gui_qr_cylinder_overlay"] = reloaded_overlay
            helper = getattr(reloaded_overlay, "detector_qr_qz_maps_for_projection", None)
        except Exception:
            helper = None
    if helper is not None:
        return helper(config=config, detector_shape=detector_shape)

    try:
        height, width = int(detector_shape[0]), int(detector_shape[1])
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    if IntersectionGeometry is None or detector_points_to_sample_qr_qz is None:
        return None
    cols, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    geometry = IntersectionGeometry(
        image_size=int(config.image_size),
        center_col=float(config.center_col),
        center_row=float(config.center_row),
        distance_cor_to_detector=float(config.distance_cor_to_detector),
        gamma_deg=float(config.gamma_deg),
        Gamma_deg=float(config.Gamma_deg),
        chi_deg=float(config.chi_deg),
        psi_deg=float(config.psi_deg),
        psi_z_deg=float(config.psi_z_deg),
        zs=float(config.zs),
        zb=float(config.zb),
        theta_initial_deg=float(config.theta_initial_deg),
        cor_angle_deg=float(config.cor_angle_deg),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        pixel_size_m=float(config.pixel_size_m),
    )
    try:
        return detector_points_to_sample_qr_qz(
            detector_col=cols,
            detector_row=rows,
            geometry=geometry,
            wavelength=float(config.wavelength),
            n2=config.n2,
            beam_x=config_float_attr(config, "beam_x", 0.0),
            beam_y=config_float_attr(config, "beam_y", 0.0),
            dtheta=config_float_attr(config, "dtheta", 0.0),
            dphi=config_float_attr(config, "dphi", 0.0),
        )
    except Exception:
        return None


def caked_qz_map_for_background(
    bg: dict[str, object],
    q_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    config = bg["qr_overlay_config"]
    if q_maps is None:
        q_maps = notebook_detector_qr_qz_maps(
            config=config,
            detector_shape=tuple(bg["detector_image"].shape),
        )
    if q_maps is None:
        qspace_geometry = exact_qspace_portable.PortableQSpaceGeometry(
            pixel_size_m=PIXEL_SIZE_M,
            distance_m=float(getattr(config, "distance_cor_to_detector", distance_m)),
            center_row_px=float(center_row_px),
            center_col_px=float(center_col_px),
            wavelength_m=WAVELENGTH_M,
            gamma_deg=float(config.gamma_deg),
            Gamma_deg=float(config.Gamma_deg),
            chi_deg=float(config.chi_deg),
            psi_deg=float(config.psi_deg),
            psi_z_deg=float(config.psi_z_deg),
            theta_initial_deg=float(config.theta_initial_deg),
            cor_angle_deg=float(config.cor_angle_deg),
            zs=float(config.zs),
            zb=float(config.zb),
        )
        _qx_detector, qz_detector_corners, _qy_detector = (
            exact_qspace_portable._shared_detector_maps_for_shape(  # noqa: SLF001
                tuple(bg["detector_image"].shape),
                qspace_geometry,
            )
        )
        qz_detector_corners = np.asarray(qz_detector_corners, dtype=np.float64)
        qz_detector_center = -0.25 * (
            qz_detector_corners[:-1, :-1]
            + qz_detector_corners[1:, :-1]
            + qz_detector_corners[:-1, 1:]
            + qz_detector_corners[1:, 1:]
        )
    else:
        _qr_detector, qz_detector, valid_q = q_maps
        qz_detector_center = np.asarray(qz_detector, dtype=np.float64).copy()
        qz_detector_center[~np.asarray(valid_q, dtype=bool)] = np.nan
    if qz_detector_center.shape != np.asarray(bg["detector_image"]).shape:
        raise RuntimeError("detector Qz map shape mismatch")
    qz_cake_raw = integrate_detector_to_cake_lut(
        np.asarray(qz_detector_center, dtype=np.float32),
        np.asarray(bg["transform_bundle"].radial_deg, dtype=np.float64),
        np.asarray(bg["transform_bundle"].raw_azimuth_deg, dtype=np.float64),
        bg["transform_bundle"].lut,
    )
    qz_caked, qz_theta_axis, qz_phi_axis = prepare_gui_phi_display(qz_cake_raw)
    if qz_caked.shape != np.asarray(bg["caked_image"]).shape:
        raise RuntimeError("caked Qz map shape mismatch")
    if not np.allclose(qz_theta_axis, np.asarray(bg["theta_axis"], dtype=np.float64)):
        raise RuntimeError("caked Qz theta axis mismatch")
    if not np.allclose(qz_phi_axis, np.asarray(bg["phi_axis"], dtype=np.float64)):
        raise RuntimeError("caked Qz phi axis mismatch")
    return np.asarray(qz_caked, dtype=np.float64)


@njit(parallel=True, nogil=True)
def _profile_accumulate_uniform_numba(
    image: np.ndarray,
    model: np.ndarray,
    mask: np.ndarray,
    qz_map: np.ndarray,
    qz_lo: float,
    qz_hi: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    row_counts = np.zeros((height, n_bins), dtype=np.int64)
    row_background = np.zeros((height, n_bins), dtype=np.float64)
    row_fit = np.zeros((height, n_bins), dtype=np.float64)
    if n_bins <= 0 or qz_hi <= qz_lo:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    scale = float(n_bins) / (qz_hi - qz_lo)
    for row in prange(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            qz = qz_map[row, col]
            background_value = image[row, col]
            fit_value = model[row, col]
            if not (np.isfinite(qz) and np.isfinite(background_value) and np.isfinite(fit_value)):
                continue
            if qz < qz_lo or qz >= qz_hi:
                continue
            bin_index = int((qz - qz_lo) * scale)
            if 0 <= bin_index < n_bins:
                row_counts[row, bin_index] += 1
                row_background[row, bin_index] += background_value
                row_fit[row, bin_index] += fit_value
    counts = np.zeros(n_bins, dtype=np.int64)
    background_sum = np.zeros(n_bins, dtype=np.float64)
    fit_sum = np.zeros(n_bins, dtype=np.float64)
    for row in range(height):
        for bin_index in range(n_bins):
            counts[bin_index] += row_counts[row, bin_index]
            background_sum[bin_index] += row_background[row, bin_index]
            fit_sum[bin_index] += row_fit[row, bin_index]
    return counts, background_sum, fit_sum


@njit(parallel=True, nogil=True)
def _detector_qr_qz_profile_accumulate_uniform_numba(
    image: np.ndarray,
    model: np.ndarray,
    qr_map: np.ndarray,
    qz_map: np.ndarray,
    valid_q: np.ndarray,
    phi_map: np.ndarray,
    theta_map: np.ndarray,
    solid_angle: np.ndarray,
    use_solid_angle: bool,
    use_theta_map: bool,
    qz_lo: float,
    qz_hi: float,
    n_bins: int,
    qr_lo: float,
    qr_hi: float,
    phi_min: float,
    phi_max: float,
    phi_wrap: bool,
    theta_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    row_counts = np.zeros((height, n_bins), dtype=np.int64)
    row_background = np.zeros((height, n_bins), dtype=np.float64)
    row_fit = np.zeros((height, n_bins), dtype=np.float64)
    row_theta_sum = np.zeros((height, n_bins), dtype=np.float64)
    row_theta_count = np.zeros((height, n_bins), dtype=np.int64)
    row_theta_min = np.empty((height, n_bins), dtype=np.float64)
    row_theta_max = np.empty((height, n_bins), dtype=np.float64)
    for row in prange(height):
        for bin_index in range(n_bins):
            row_theta_min[row, bin_index] = np.inf
            row_theta_max[row, bin_index] = -np.inf

    if n_bins <= 0 or qz_hi <= qz_lo:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f

    scale = float(n_bins) / (qz_hi - qz_lo)
    for row in prange(height):
        for col in range(width):
            if not valid_q[row, col]:
                continue
            qr = qr_map[row, col]
            qz = qz_map[row, col]
            phi = phi_map[row, col]
            if not (np.isfinite(qr) and np.isfinite(qz) and np.isfinite(phi)):
                continue
            if qz <= POSITIVE_QZ_MIN:
                continue
            if qr < qr_lo or qr > qr_hi:
                continue
            if phi_wrap:
                if not (phi >= phi_min or phi <= phi_max):
                    continue
            else:
                if not (phi >= phi_min and phi <= phi_max):
                    continue
            if qz < qz_lo or qz > qz_hi:
                continue
            if use_theta_map:
                theta = theta_map[row, col]
                if not (np.isfinite(theta) and theta <= theta_limit):
                    continue
            else:
                theta = np.nan
            if use_solid_angle:
                sa = solid_angle[row, col]
                if not (np.isfinite(sa) and sa > 0.0):
                    continue
                background_value = image[row, col] / sa if np.isfinite(image[row, col]) else 0.0
                # Preserve the notebook's existing detector-peak model convention.
                fit_value = model[row, col] if np.isfinite(model[row, col]) else 0.0
            else:
                background_value = image[row, col] if np.isfinite(image[row, col]) else 0.0
                fit_value = model[row, col] if np.isfinite(model[row, col]) else 0.0
            bin_index = int((qz - qz_lo) * scale)
            if bin_index >= n_bins:
                bin_index = n_bins - 1
            if bin_index < 0:
                continue
            row_counts[row, bin_index] += 1
            row_background[row, bin_index] += background_value
            row_fit[row, bin_index] += fit_value
            if use_theta_map:
                row_theta_sum[row, bin_index] += theta
                row_theta_count[row, bin_index] += 1
                if theta < row_theta_min[row, bin_index]:
                    row_theta_min[row, bin_index] = theta
                if theta > row_theta_max[row, bin_index]:
                    row_theta_max[row, bin_index] = theta

    counts = np.zeros(n_bins, dtype=np.int64)
    background_sum = np.zeros(n_bins, dtype=np.float64)
    fit_sum = np.zeros(n_bins, dtype=np.float64)
    theta_min = np.full(n_bins, np.nan, dtype=np.float64)
    theta_max = np.full(n_bins, np.nan, dtype=np.float64)
    theta_mean = np.full(n_bins, np.nan, dtype=np.float64)
    for bin_index in range(n_bins):
        theta_sum_acc = 0.0
        theta_count_acc = 0
        theta_min_acc = np.inf
        theta_max_acc = -np.inf
        for row in range(height):
            counts[bin_index] += row_counts[row, bin_index]
            background_sum[bin_index] += row_background[row, bin_index]
            fit_sum[bin_index] += row_fit[row, bin_index]
            theta_sum_acc += row_theta_sum[row, bin_index]
            theta_count_acc += row_theta_count[row, bin_index]
            if row_theta_min[row, bin_index] < theta_min_acc:
                theta_min_acc = row_theta_min[row, bin_index]
            if row_theta_max[row, bin_index] > theta_max_acc:
                theta_max_acc = row_theta_max[row, bin_index]
        if theta_count_acc > 0:
            theta_min[bin_index] = theta_min_acc
            theta_max[bin_index] = theta_max_acc
            theta_mean[bin_index] = theta_sum_acc / float(theta_count_acc)
    return counts, background_sum, fit_sum, theta_min, theta_max, theta_mean


def _detector_qr_qz_profile_accumulate_uniform_cupy(
    image: np.ndarray,
    model: np.ndarray,
    qr_map: np.ndarray,
    qz_map: np.ndarray,
    valid_q: np.ndarray,
    phi_map: np.ndarray,
    theta_map: np.ndarray | None,
    *,
    qz_lo: float,
    qz_hi: float,
    n_bins: int,
    qr_lo: float,
    qr_hi: float,
    phi_min: float,
    phi_max: float,
    phi_wrap: bool,
    theta_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not bool(GPU_ACCELERATION_ENABLED) or cp is None:
        return None
    try:
        c_image = cp.asarray(image, dtype=cp.float64)
        c_model = cp.asarray(model, dtype=cp.float64)
        c_qr = cp.asarray(qr_map, dtype=cp.float64)
        c_qz = cp.asarray(qz_map, dtype=cp.float64)
        c_valid = cp.asarray(valid_q, dtype=cp.bool_)
        c_phi = cp.asarray(phi_map, dtype=cp.float64)
        base = c_valid & cp.isfinite(c_qr) & cp.isfinite(c_qz) & cp.isfinite(c_phi)
        base &= c_qz > float(POSITIVE_QZ_MIN)
        base &= (c_qr >= float(qr_lo)) & (c_qr <= float(qr_hi))
        if bool(phi_wrap):
            base &= (c_phi >= float(phi_min)) | (c_phi <= float(phi_max))
        else:
            base &= (c_phi >= float(phi_min)) & (c_phi <= float(phi_max))
        if theta_map is not None:
            c_theta = cp.asarray(theta_map, dtype=cp.float64)
            base &= cp.isfinite(c_theta) & (c_theta <= float(theta_limit))
        else:
            c_theta = None
        selected_qz = c_qz[base]
        if int(selected_qz.size) == 0:
            zeros_i = np.zeros(int(n_bins), dtype=np.int64)
            zeros_f = np.zeros(int(n_bins), dtype=np.float64)
            nans = np.full(int(n_bins), np.nan, dtype=np.float64)
            return zeros_i, zeros_f, zeros_f.copy(), nans, nans.copy(), nans.copy()
        scale = float(n_bins) / (float(qz_hi) - float(qz_lo))
        selected = (selected_qz >= float(qz_lo)) & (selected_qz <= float(qz_hi))
        selected_qz = selected_qz[selected]
        if int(selected_qz.size) == 0:
            zeros_i = np.zeros(int(n_bins), dtype=np.int64)
            zeros_f = np.zeros(int(n_bins), dtype=np.float64)
            nans = np.full(int(n_bins), np.nan, dtype=np.float64)
            return zeros_i, zeros_f, zeros_f.copy(), nans, nans.copy(), nans.copy()
        bin_index = cp.floor((selected_qz - float(qz_lo)) * scale).astype(cp.int64)
        bin_index = cp.clip(bin_index, 0, int(n_bins) - 1)
        background_values = cp.where(
            cp.isfinite(c_image[base][selected]), c_image[base][selected], 0.0
        )
        fit_values = cp.where(cp.isfinite(c_model[base][selected]), c_model[base][selected], 0.0)
        counts = cp.bincount(bin_index, minlength=int(n_bins))[: int(n_bins)]
        background_sum = cp.bincount(bin_index, weights=background_values, minlength=int(n_bins))[
            : int(n_bins)
        ]
        fit_sum = cp.bincount(bin_index, weights=fit_values, minlength=int(n_bins))[: int(n_bins)]
        theta_min = cp.full(int(n_bins), cp.nan, dtype=cp.float64)
        theta_max = cp.full(int(n_bins), cp.nan, dtype=cp.float64)
        theta_mean = cp.full(int(n_bins), cp.nan, dtype=cp.float64)
        if c_theta is not None:
            theta_values = c_theta[base][selected]
            finite_theta = cp.isfinite(theta_values)
            if bool(cp.any(finite_theta).item()):
                finite_bins = bin_index[finite_theta]
                finite_values = theta_values[finite_theta]
                theta_sum = cp.bincount(finite_bins, weights=finite_values, minlength=int(n_bins))[
                    : int(n_bins)
                ]
                theta_count = cp.bincount(finite_bins, minlength=int(n_bins))[: int(n_bins)]
                theta_mean = cp.where(theta_count > 0, theta_sum / theta_count, cp.nan)
                tmin = cp.full(int(n_bins), cp.inf, dtype=cp.float64)
                tmax = cp.full(int(n_bins), -cp.inf, dtype=cp.float64)
                cp.minimum.at(tmin, finite_bins, finite_values)
                cp.maximum.at(tmax, finite_bins, finite_values)
                theta_min = cp.where(cp.isfinite(tmin), tmin, cp.nan)
                theta_max = cp.where(cp.isfinite(tmax), tmax, cp.nan)
        return (
            cp.asnumpy(counts).astype(np.int64, copy=False),
            cp.asnumpy(background_sum).astype(np.float64, copy=False),
            cp.asnumpy(fit_sum).astype(np.float64, copy=False),
            cp.asnumpy(theta_min).astype(np.float64, copy=False),
            cp.asnumpy(theta_max).astype(np.float64, copy=False),
            cp.asnumpy(theta_mean).astype(np.float64, copy=False),
        )
    except Exception as exc:
        print(f"GPU detector Qr/Qz profile accumulation fell back to CPU: {exc}")
        return None


def _detector_profile_dataframe_from_arrays(
    bg: dict[str, object],
    rod: dict[str, object],
    *,
    branch_name: str,
    branch_label: str,
    edges: np.ndarray,
    qr0: float,
    delta_qr: float,
    solid_angle_corrected: bool,
    counts: np.ndarray,
    background_sums: np.ndarray,
    fit_sums: np.ndarray,
    two_theta_min: np.ndarray,
    two_theta_max: np.ndarray,
    two_theta_mean: np.ndarray,
    theta_initial_deg_used_for_q: float | None = None,
    qz_map_source: str = "detector_qspace_per_pixel",
    qr_integration_source: str = "detector_qr_band_per_qz_bin",
) -> pd.DataFrame:
    counts = np.asarray(counts, dtype=np.int64)
    background_sums = np.asarray(background_sums, dtype=np.float64)
    fit_sums = np.asarray(fit_sums, dtype=np.float64)
    theta_initial_for_q = (
        float(bg["qr_overlay_config"].theta_initial_deg)
        if theta_initial_deg_used_for_q is None
        else float(theta_initial_deg_used_for_q)
    )
    nonempty = counts > 0
    background_mean = np.full(counts.shape, np.nan, dtype=np.float64)
    fit_mean = np.full(counts.shape, np.nan, dtype=np.float64)
    background_mean[nonempty] = background_sums[nonempty] / counts[nonempty].astype(np.float64)
    fit_mean[nonempty] = fit_sums[nonempty] / counts[nonempty].astype(np.float64)
    sideband_density = np.full(counts.shape, np.nan, dtype=np.float64)
    sideband_counts = np.zeros(counts.shape, dtype=np.int64)
    return pd.DataFrame(
        {
            "sample_name": [SAMPLE_NAME] * int(counts.size),
            "tilt_deg": [float(bg["tilt_deg"])] * int(counts.size),
            "theta_initial_deg_used_for_q": [theta_initial_for_q] * int(counts.size),
            "qz_map_source": [str(qz_map_source)] * int(counts.size),
            "qr_integration_source": [str(qr_integration_source)] * int(counts.size),
            "solid_angle_corrected": [bool(solid_angle_corrected)] * int(counts.size),
            "m": [int(rod["m"])] * int(counts.size),
            "qr": [float(qr0)] * int(counts.size),
            "qr_original": [float(rod.get("qr_original", qr0))] * int(counts.size),
            "qr_fit_count": [int(rod.get("qr_fit_count", 0))] * int(counts.size),
            "qr_fit_sample_count": [int(rod.get("qr_fit_sample_count", 0))] * int(counts.size),
            "qr_fit_method": [str(rod.get("qr_fit_method", "original"))] * int(counts.size),
            "branch": [str(branch_name)] * int(counts.size),
            "phi_window": [str(branch_label)] * int(counts.size),
            "delta_qr": [float(delta_qr)] * int(counts.size),
            "qz_bin": np.arange(1, int(counts.size) + 1, dtype=int),
            "qz_min": edges[:-1],
            "qz_max": edges[1:],
            "qz_center": 0.5 * (edges[:-1] + edges[1:]),
            "pixel_count": counts,
            "acceptance_sum": counts.astype(np.float64),
            "acceptance_source": ["pixel_count"] * int(counts.size),
            "background_sum_raw": np.where(nonempty, background_sums, np.nan),
            "background_sum": np.where(nonempty, background_sums, np.nan),
            "fit_sum": np.where(nonempty, fit_sums, np.nan),
            "background_mean_raw": background_mean,
            "background_mean": background_mean,
            "fit_mean": fit_mean,
            "background_weighted_sum_raw": np.where(nonempty, background_sums, np.nan),
            "background_weighted_sum": np.where(nonempty, background_sums, np.nan),
            "fit_weighted_sum": np.where(nonempty, fit_sums, np.nan),
            "background_density_raw": background_mean,
            "qr_sideband_background_density": sideband_density,
            "qr_sideband_pixel_count": sideband_counts,
            "qr_sideband_inner_delta_qr": [float("nan")] * int(counts.size),
            "qr_sideband_outer_delta_qr": [float("nan")] * int(counts.size),
            "qr_sideband_percentile": [float("nan")] * int(counts.size),
            "qr_transverse_background_subtracted": [False] * int(counts.size),
            "background_density": background_mean,
            "fit_density": fit_mean,
            "two_theta_min": np.asarray(two_theta_min, dtype=np.float64),
            "two_theta_max": np.asarray(two_theta_max, dtype=np.float64),
            "two_theta_mean": np.asarray(two_theta_mean, dtype=np.float64),
        }
    )


@njit(nogil=True)
def _is_local_peak_top_numba(values: np.ndarray, row: int, col: int) -> bool:
    if row <= 0 or col <= 0 or row >= values.shape[0] - 1 or col >= values.shape[1] - 1:
        return False
    center = values[row, col]
    if not np.isfinite(center):
        return False
    for rr in range(row - 1, row + 2):
        for cc in range(col - 1, col + 2):
            if rr == row and cc == col:
                continue
            neighbor = values[rr, cc]
            if np.isfinite(neighbor) and center <= neighbor:
                return False
    return True


@njit(nogil=True)
def _find_marker_peak_in_window_numba(
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    model: np.ndarray,
    image: np.ndarray,
    branch_mask: np.ndarray,
    qz_map: np.ndarray,
    theta0: float,
    phi0: float,
    theta_half_width: float,
    phi_half_width: float,
) -> tuple[int, int]:
    best_model = -np.inf
    best_model_row = -1
    best_model_col = -1
    best_image = -np.inf
    best_image_row = -1
    best_image_col = -1
    any_local = False
    for row in range(phi_axis.size):
        phi = phi_axis[row]
        if abs(_wrapped_delta_deg_scalar_numba(phi, phi0)) > phi_half_width:
            continue
        for col in range(theta_axis.size):
            theta = theta_axis[col]
            if abs(theta - theta0) > theta_half_width:
                continue
            if not branch_mask[row, col]:
                continue
            if not np.isfinite(qz_map[row, col]) or qz_map[row, col] <= POSITIVE_QZ_MIN:
                continue
            any_local = True
            model_value = model[row, col]
            if (
                np.isfinite(model_value)
                and model_value > 0.0
                and model_value > best_model
                and _is_local_peak_top_numba(model, row, col)
            ):
                best_model = model_value
                best_model_row = row
                best_model_col = col
            image_value = image[row, col]
            if (
                np.isfinite(image_value)
                and image_value > best_image
                and _is_local_peak_top_numba(image, row, col)
            ):
                best_image = image_value
                best_image_row = row
                best_image_col = col
    if best_model_row >= 0:
        return best_model_row, best_model_col
    if any_local and best_image_row >= 0:
        return best_image_row, best_image_col
    return -1, -1


@njit(nogil=True)
def _refine_marker_to_local_peak_numba(
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    model: np.ndarray,
    image: np.ndarray,
    branch_mask: np.ndarray,
    qz_map: np.ndarray,
    theta0: float,
    phi0: float,
    theta_half_width: float,
    phi_half_width: float,
    fallback_theta_half_width: float,
    fallback_phi_half_width: float,
) -> tuple[int, float, float, float]:
    row, col = _find_marker_peak_in_window_numba(
        theta_axis,
        phi_axis,
        model,
        image,
        branch_mask,
        qz_map,
        theta0,
        phi0,
        theta_half_width,
        phi_half_width,
    )
    if row < 0:
        row, col = _find_marker_peak_in_window_numba(
            theta_axis,
            phi_axis,
            model,
            image,
            branch_mask,
            qz_map,
            theta0,
            phi0,
            fallback_theta_half_width,
            fallback_phi_half_width,
        )
    if row < 0:
        return 0, np.nan, np.nan, np.nan
    return 1, qz_map[row, col], theta_axis[col], phi_axis[row]


def profile_from_full_mask(
    bg: dict[str, object],
    rod: dict[str, object],
    *,
    branch_name: str,
    branch_label: str,
    qz_edges: np.ndarray,
    mask: np.ndarray,
    qz_map: np.ndarray,
) -> pd.DataFrame:
    image = np.asarray(bg["caked_image"], dtype=np.float64)
    theta_axis = np.asarray(bg["theta_axis"], dtype=np.float64).reshape(-1)
    if image.ndim != 2 or theta_axis.size != image.shape[1]:
        raise RuntimeError("caked image/theta axis shape mismatch")
    theta_map = np.broadcast_to(theta_axis[None, :], image.shape)
    qz_values = np.asarray(qz_map, dtype=np.float64)
    profile_mask = np.asarray(mask, dtype=bool) & positive_qz_mask(qz_values)
    payload = qz_profile_from_caked_mask(
        image=image,
        model=positive_l_caked_peak_model_for_display(bg),
        qz_map=qz_values,
        qz_edges=np.asarray(qz_edges, dtype=np.float64),
        mask=profile_mask,
        signal_sum=bg.get("caked_sum_signal"),
        normalization_sum=bg.get("caked_sum_normalization"),
        theta_map=theta_map,
    )
    rows = pd.DataFrame(payload)
    metadata = {
        "sample_name": SAMPLE_NAME,
        "tilt_deg": float(bg["tilt_deg"]),
        "theta_initial_deg_used_for_q": float(bg["qr_overlay_config"].theta_initial_deg),
        "qz_map_source": "detector_qspace_reprojected_to_caked",
        "qr_integration_source": "caked_selected_qr_mask",
        "solid_angle_corrected": bool(BACKGROUND_SOLID_ANGLE_CORRECTION),
        "m": int(rod["m"]),
        "qr": float(rod["qr"]),
        "qr_original": float(rod.get("qr_original", rod["qr"])),
        "qr_fit_count": int(rod.get("qr_fit_count", 0)),
        "qr_fit_sample_count": int(rod.get("qr_fit_sample_count", 0)),
        "qr_fit_method": str(rod.get("qr_fit_method", "original")),
        "branch": str(branch_name),
        "phi_window": str(branch_label),
        "delta_qr": float(qr_rod_delta_qr),
    }
    for key, value in reversed(list(metadata.items())):
        rows.insert(0, key, value)
    return rows


def detector_gui_phi_map_for_background(bg: dict[str, object]) -> np.ndarray | None:
    config = bg["qr_overlay_config"]
    detector_shape = tuple(bg["detector_image"].shape)
    helper = getattr(gui_qr_cylinder_overlay, "detector_gui_phi_map_for_projection", None)
    if helper is not None:
        try:
            phi = helper(config=config, detector_shape=detector_shape)
            if phi is not None:
                return np.asarray(phi, dtype=np.float64)
        except Exception:
            pass
    try:
        ai = FastAzimuthalIntegrator(
            dist=float(getattr(config, "distance_cor_to_detector", distance_m)),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
        _theta_map, raw_phi_map = detector_pixel_angular_maps(detector_shape, ai.geometry)
        return np.asarray(raw_phi_to_gui_phi(raw_phi_map), dtype=np.float64)
    except Exception:
        return None


def detector_solid_angle_for_background(bg: dict[str, object]) -> np.ndarray | None:
    config = bg["qr_overlay_config"]
    detector_shape = tuple(bg["detector_image"].shape)
    helper = getattr(gui_qr_cylinder_overlay, "detector_solid_angle_for_projection", None)
    if helper is not None:
        try:
            solid_angle = helper(config=config, detector_shape=detector_shape)
            if solid_angle is not None:
                return np.asarray(solid_angle, dtype=np.float64)
        except Exception:
            pass
    try:
        ai = FastAzimuthalIntegrator(
            dist=float(getattr(config, "distance_cor_to_detector", distance_m)),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
        if hasattr(ai, "_solid_angle_for_shape"):
            return np.asarray(ai._solid_angle_for_shape(detector_shape), dtype=np.float64)  # noqa: SLF001
    except Exception:
        pass
    return None


def detector_two_theta_map_for_background(bg: dict[str, object]) -> np.ndarray | None:
    detector_shape = tuple(bg["detector_image"].shape)
    try:
        config = bg["qr_overlay_config"]
        ai = FastAzimuthalIntegrator(
            dist=float(getattr(config, "distance_cor_to_detector", distance_m)),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
        theta_map, _raw_phi_map = detector_pixel_angular_maps(detector_shape, ai.geometry)
        return np.asarray(theta_map, dtype=np.float64)
    except Exception:
        return None


def detector_qspace_config_with_theta_initial(config: object, theta_initial_deg: float) -> object:
    return replace(config, theta_initial_deg=float(theta_initial_deg))


def _binned_nanpercentile(
    qz_values: object,
    density_values: object,
    edges: object,
    *,
    percentile: float,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    qz = np.asarray(qz_values, dtype=np.float64).reshape(-1)
    values = np.asarray(density_values, dtype=np.float64).reshape(-1)
    edges_arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    n_bins = max(int(edges_arr.size) - 1, 0)
    out = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    if n_bins <= 0 or qz.size != values.size:
        return out, counts
    finite = np.isfinite(qz) & np.isfinite(values)
    if not np.any(finite):
        return out, counts
    qz_f = qz[finite]
    values_f = values[finite]
    bin_index = np.searchsorted(edges_arr, qz_f, side="right") - 1
    if edges_arr.size:
        last_edge = np.isclose(qz_f, edges_arr[-1], rtol=1.0e-12, atol=1.0e-12)
        bin_index[last_edge] = n_bins - 1
    valid = (bin_index >= 0) & (bin_index < n_bins)
    bin_index = bin_index[valid]
    values_f = values_f[valid]
    if bin_index.size == 0:
        return out, counts
    order = np.argsort(bin_index, kind="mergesort")
    bin_index = bin_index[order]
    values_f = values_f[order]
    starts = np.r_[0, np.flatnonzero(np.diff(bin_index)) + 1]
    stops = np.r_[starts[1:], bin_index.size]
    for start, stop in zip(starts, stops):
        idx = int(bin_index[start])
        group = values_f[start:stop]
        counts[idx] = int(group.size)
        if group.size >= int(min_count):
            out[idx] = float(np.nanpercentile(group, float(percentile)))
    return out, counts


def _sideband_qr_mask(
    qr_map: np.ndarray,
    *,
    qr0: float,
    delta_qr: float,
    inner_scale: float = QR_ROD_BG_SIDE_BAND_INNER_SCALE,
    outer_scale: float = QR_ROD_BG_SIDE_BAND_OUTER_SCALE,
) -> tuple[np.ndarray, float, float]:
    inner = max(float(delta_qr) * float(inner_scale), float(delta_qr) + 1.0e-12)
    outer = max(float(delta_qr) * float(outer_scale), inner + max(float(delta_qr), 1.0e-12))
    left_lo = max(0.0, float(qr0) - outer)
    left_hi = max(0.0, float(qr0) - inner)
    right_lo = float(qr0) + inner
    right_hi = float(qr0) + outer
    mask = np.zeros(qr_map.shape, dtype=bool)
    if left_hi > left_lo:
        mask |= (qr_map >= left_lo) & (qr_map <= left_hi)
    mask |= (qr_map >= right_lo) & (qr_map <= right_hi)
    return mask, inner, outer


def profile_from_detector_qr_qz(
    bg: dict[str, object],
    rod: dict[str, object],
    *,
    branch_name: str,
    branch_label: str,
    qz_edges: np.ndarray,
    phi_min: float,
    phi_max: float,
    detector_q_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
    detector_phi_map: np.ndarray,
    detector_solid_angle: np.ndarray | None,
    detector_two_theta_map: np.ndarray | None,
    theta_initial_deg_used_for_q: float | None = None,
    delta_qr_override: object = None,
) -> pd.DataFrame:
    image = np.asarray(bg["detector_image"], dtype=np.float64)
    model = np.asarray(positive_l_detector_peak_model_for_display(bg), dtype=np.float64)
    qr_map = np.asarray(detector_q_maps[0], dtype=np.float64)
    qz_map = np.asarray(detector_q_maps[1], dtype=np.float64)
    valid_q = np.asarray(detector_q_maps[2], dtype=bool)
    phi_map = np.asarray(detector_phi_map, dtype=np.float64)
    edges = np.asarray(qz_edges, dtype=np.float64).reshape(-1)
    if (
        image.ndim != 2
        or model.shape != image.shape
        or qr_map.shape != image.shape
        or qz_map.shape != image.shape
        or valid_q.shape != image.shape
        or phi_map.shape != image.shape
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
    ):
        return pd.DataFrame()

    if detector_solid_angle is None:
        signal = image
        finite_solid = np.ones(image.shape, dtype=bool)
        solid_angle_corrected = False
    else:
        solid_angle = np.asarray(detector_solid_angle, dtype=np.float64)
        if solid_angle.shape != image.shape:
            return pd.DataFrame()
        finite_solid = np.isfinite(solid_angle) & (solid_angle > 0.0)
        signal = np.full(image.shape, np.nan, dtype=np.float64)
        signal[finite_solid] = image[finite_solid] / solid_angle[finite_solid]
        solid_angle_corrected = True

    theta_map = None
    if detector_two_theta_map is None:
        below_theta_limit = np.ones(image.shape, dtype=bool)
    else:
        theta_map = np.asarray(detector_two_theta_map, dtype=np.float64)
        if theta_map.shape != image.shape:
            return pd.DataFrame()
        below_theta_limit = np.isfinite(theta_map) & (
            theta_map <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)
        )

    qr0 = float(rod["qr"])
    delta_qr = max(1.0e-9, as_float(delta_qr_override, qr_rod_delta_qr))
    qr_lo = max(0.0, qr0 - delta_qr)
    qr_hi = qr0 + delta_qr
    theta_initial_for_q = (
        float(bg["qr_overlay_config"].theta_initial_deg)
        if theta_initial_deg_used_for_q is None
        else float(theta_initial_deg_used_for_q)
    )

    edge_steps = np.diff(edges)
    edges_uniform = bool(
        edge_steps.size and np.allclose(edge_steps, edge_steps[0], rtol=1.0e-8, atol=1.0e-12)
    )
    fast_payload = None
    if edges_uniform:
        qz_lo_fast = float(edges[0])
        qz_hi_fast = float(edges[-1])
        n_bins_fast = int(edges.size - 1)
        phi_wrap = bool(float(phi_min) > float(phi_max))
        if (
            not bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED)
            and bool(GPU_ACCELERATION_ENABLED)
            and detector_solid_angle is None
        ):
            fast_payload = _detector_qr_qz_profile_accumulate_uniform_cupy(
                image,
                model,
                qr_map,
                qz_map,
                valid_q,
                phi_map,
                theta_map,
                qz_lo=qz_lo_fast,
                qz_hi=qz_hi_fast,
                n_bins=n_bins_fast,
                qr_lo=float(qr_lo),
                qr_hi=float(qr_hi),
                phi_min=float(phi_min),
                phi_max=float(phi_max),
                phi_wrap=phi_wrap,
                theta_limit=float(ROD_PROFILE_MAX_TWO_THETA_DEG),
            )
        if (
            fast_payload is None
            and not bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED)
            and bool(NUMBA_AVAILABLE)
        ):
            solid_angle_array = (
                np.zeros(image.shape, dtype=np.float64)
                if detector_solid_angle is None
                else np.asarray(detector_solid_angle, dtype=np.float64)
            )
            theta_array = (
                np.zeros(image.shape, dtype=np.float64)
                if theta_map is None
                else np.asarray(theta_map, dtype=np.float64)
            )
            try:
                fast_payload = _detector_qr_qz_profile_accumulate_uniform_numba(
                    np.ascontiguousarray(image, dtype=np.float64),
                    np.ascontiguousarray(model, dtype=np.float64),
                    np.ascontiguousarray(qr_map, dtype=np.float64),
                    np.ascontiguousarray(qz_map, dtype=np.float64),
                    np.ascontiguousarray(valid_q, dtype=np.bool_),
                    np.ascontiguousarray(phi_map, dtype=np.float64),
                    np.ascontiguousarray(theta_array, dtype=np.float64),
                    np.ascontiguousarray(solid_angle_array, dtype=np.float64),
                    bool(detector_solid_angle is not None),
                    bool(theta_map is not None),
                    qz_lo_fast,
                    qz_hi_fast,
                    n_bins_fast,
                    float(qr_lo),
                    float(qr_hi),
                    float(phi_min),
                    float(phi_max),
                    phi_wrap,
                    float(ROD_PROFILE_MAX_TWO_THETA_DEG),
                )
            except Exception as exc:
                print(
                    f"CPU detector Qr/Qz profile fast path fell back to vectorized histogram: {exc}"
                )
                fast_payload = None
        if fast_payload is not None:
            (
                counts_fast,
                background_sums_fast,
                fit_sums_fast,
                theta_min_fast,
                theta_max_fast,
                theta_mean_fast,
            ) = fast_payload
            if int(np.nansum(counts_fast)) <= 0:
                return pd.DataFrame()
            return _detector_profile_dataframe_from_arrays(
                bg,
                rod,
                branch_name=branch_name,
                branch_label=branch_label,
                edges=edges,
                qr0=qr0,
                delta_qr=delta_qr,
                solid_angle_corrected=solid_angle_corrected,
                counts=counts_fast,
                background_sums=background_sums_fast,
                fit_sums=fit_sums_fast,
                two_theta_min=theta_min_fast,
                two_theta_max=theta_max_fast,
                two_theta_mean=theta_mean_fast,
                theta_initial_deg_used_for_q=theta_initial_for_q,
                qr_integration_source="detector_qr_band_per_qz_bin_fast",
            )

    if float(phi_min) <= float(phi_max):
        phi_selected = (phi_map >= float(phi_min)) & (phi_map <= float(phi_max))
    else:
        phi_selected = (phi_map >= float(phi_min)) | (phi_map <= float(phi_max))
    base_common = (
        valid_q
        & finite_solid
        & below_theta_limit
        & phi_selected
        & np.isfinite(qr_map)
        & np.isfinite(qz_map)
        & (qz_map > POSITIVE_QZ_MIN)
    )
    base = base_common & (qr_map >= qr_lo) & (qr_map <= qr_hi)
    selected_qz = qz_map[base]
    if selected_qz.size == 0:
        return pd.DataFrame()
    background_values = np.where(np.isfinite(signal[base]), signal[base], 0.0)
    fit_values = np.where(np.isfinite(model[base]), model[base], 0.0)
    counts, _ = np.histogram(selected_qz, bins=edges)
    background_sums, _ = np.histogram(selected_qz, bins=edges, weights=background_values)
    fit_sums, _ = np.histogram(selected_qz, bins=edges, weights=fit_values)
    counts = np.asarray(counts, dtype=np.int64)
    background_sums = np.asarray(background_sums, dtype=np.float64)
    fit_sums = np.asarray(fit_sums, dtype=np.float64)
    background_mean_raw = np.full(counts.shape, np.nan, dtype=np.float64)
    fit_mean = np.full(counts.shape, np.nan, dtype=np.float64)
    nonempty = counts > 0
    background_mean_raw[nonempty] = background_sums[nonempty] / counts[nonempty].astype(np.float64)
    fit_mean[nonempty] = fit_sums[nonempty] / counts[nonempty].astype(np.float64)
    background_mean = background_mean_raw.copy()

    sideband_density = np.full(counts.shape, np.nan, dtype=np.float64)
    sideband_counts = np.zeros(counts.shape, dtype=np.int64)
    sideband_inner = float("nan")
    sideband_outer = float("nan")
    transverse_bg_used = False
    if bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED):
        side_qr_mask, sideband_inner, sideband_outer = _sideband_qr_mask(
            qr_map, qr0=qr0, delta_qr=delta_qr
        )
        sideband_mask = base_common & side_qr_mask
        side_qz = qz_map[sideband_mask]
        side_values = np.where(np.isfinite(signal[sideband_mask]), signal[sideband_mask], np.nan)
        sideband_density, sideband_counts = _binned_nanpercentile(
            side_qz,
            side_values,
            edges,
            percentile=float(QR_ROD_BG_PERCENTILE),
            min_count=int(QR_ROD_BG_MIN_SIDE_PIXELS),
        )
        subtract = nonempty & (sideband_counts >= int(QR_ROD_BG_MIN_SIDE_PIXELS)) & np.isfinite(
            sideband_density
        )
        if np.any(subtract):
            background_mean[subtract] = background_mean_raw[subtract] - sideband_density[subtract]
            transverse_bg_used = True

    two_theta_min = np.full(counts.shape, np.nan, dtype=np.float64)
    two_theta_max = np.full(counts.shape, np.nan, dtype=np.float64)
    two_theta_mean = np.full(counts.shape, np.nan, dtype=np.float64)
    if theta_map is not None:
        theta_values = theta_map[base]
        for bin_index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if bin_index == counts.size - 1:
                in_bin = (selected_qz >= lo) & (selected_qz <= hi)
            else:
                in_bin = (selected_qz >= lo) & (selected_qz < hi)
            finite_theta = theta_values[in_bin & np.isfinite(theta_values)]
            if finite_theta.size:
                two_theta_min[bin_index] = float(np.nanmin(finite_theta))
                two_theta_max[bin_index] = float(np.nanmax(finite_theta))
                two_theta_mean[bin_index] = float(np.nanmean(finite_theta))

    return pd.DataFrame(
        {
            "sample_name": [SAMPLE_NAME] * int(counts.size),
            "tilt_deg": [float(bg["tilt_deg"])] * int(counts.size),
            "theta_initial_deg_used_for_q": [theta_initial_for_q] * int(counts.size),
            "qz_map_source": ["detector_qspace_per_pixel"] * int(counts.size),
            "qr_integration_source": [
                "detector_qr_band_per_qz_bin_with_qr_sideband_background"
                if transverse_bg_used
                else "detector_qr_band_per_qz_bin"
            ]
            * int(counts.size),
            "solid_angle_corrected": [solid_angle_corrected] * int(counts.size),
            "m": [int(rod["m"])] * int(counts.size),
            "qr": [qr0] * int(counts.size),
            "qr_original": [float(rod.get("qr_original", qr0))] * int(counts.size),
            "qr_fit_count": [int(rod.get("qr_fit_count", 0))] * int(counts.size),
            "qr_fit_sample_count": [int(rod.get("qr_fit_sample_count", 0))] * int(counts.size),
            "qr_fit_method": [str(rod.get("qr_fit_method", "original"))] * int(counts.size),
            "branch": [str(branch_name)] * int(counts.size),
            "phi_window": [str(branch_label)] * int(counts.size),
            "delta_qr": [delta_qr] * int(counts.size),
            "qz_bin": np.arange(1, int(counts.size) + 1, dtype=int),
            "qz_min": edges[:-1],
            "qz_max": edges[1:],
            "qz_center": 0.5 * (edges[:-1] + edges[1:]),
            "pixel_count": counts,
            "acceptance_sum": counts.astype(np.float64),
            "acceptance_source": ["pixel_count"] * int(counts.size),
            "background_sum_raw": np.where(nonempty, background_sums, np.nan),
            "background_sum": np.where(nonempty, background_sums, np.nan),
            "fit_sum": np.where(nonempty, fit_sums, np.nan),
            "background_mean_raw": background_mean_raw,
            "background_mean": background_mean,
            "fit_mean": fit_mean,
            "background_weighted_sum_raw": np.where(nonempty, background_sums, np.nan),
            "background_weighted_sum": np.where(nonempty, background_sums, np.nan),
            "fit_weighted_sum": np.where(nonempty, fit_sums, np.nan),
            "background_density_raw": background_mean_raw,
            "qr_sideband_background_density": sideband_density,
            "qr_sideband_pixel_count": sideband_counts,
            "qr_sideband_inner_delta_qr": [sideband_inner] * int(counts.size),
            "qr_sideband_outer_delta_qr": [sideband_outer] * int(counts.size),
            "qr_sideband_percentile": [float(QR_ROD_BG_PERCENTILE)] * int(counts.size),
            "qr_transverse_background_subtracted": [bool(transverse_bg_used)] * int(counts.size),
            "background_density": background_mean,
            "fit_density": fit_mean,
            "two_theta_min": two_theta_min,
            "two_theta_max": two_theta_max,
            "two_theta_mean": two_theta_mean,
        }
    )


# Rod-profile baselines are added to fitted peak curves as a shared y=m*x+b term.
# They are not used as a background-subtraction product.
def normalized_profile_payload(
    background_density: pd.Series,
    fit_density: pd.Series,
    baseline_density: pd.Series | np.ndarray | None = None,
    peak_density: pd.Series | np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    measured = np.asarray(background_density, dtype=np.float64)
    fitted = np.asarray(fit_density, dtype=np.float64)
    baseline = (
        np.zeros(measured.shape, dtype=np.float64)
        if baseline_density is None
        else np.asarray(baseline_density, dtype=np.float64)
    )
    peak = fitted - baseline if peak_density is None else np.asarray(peak_density, dtype=np.float64)
    positive_values = np.concatenate(
        [
            measured[np.isfinite(measured) & (measured > 0.0)],
            fitted[np.isfinite(fitted) & (fitted > 0.0)],
            np.abs(baseline[np.isfinite(baseline)]),
            peak[np.isfinite(peak) & (peak > 0.0)],
        ]
    )
    scale = float(np.nanpercentile(positive_values, 99.0)) if positive_values.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return {
        "measured": measured / scale,
        "fitted": fitted / scale,
        "baseline": baseline / scale,
        "peak": peak / scale,
        "scale": float(scale),
    }


def normalized_profile_pair(
    background_density: pd.Series, fit_density: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    payload = normalized_profile_payload(background_density, fit_density)
    return np.asarray(payload["measured"], dtype=np.float64), np.asarray(
        payload["fitted"], dtype=np.float64
    )


def normalized_data_simulation_payload(
    background_density: object,
    peak_density: object,
    baseline_density: object | None = None,
    *,
    subtract_baseline_from_data: bool = True,
) -> dict[str, np.ndarray | float]:
    measured = np.asarray(background_density, dtype=np.float64)
    baseline = (
        np.zeros(measured.shape, dtype=np.float64)
        if baseline_density is None
        else np.asarray(baseline_density, dtype=np.float64)
    )
    peak = np.asarray(peak_density, dtype=np.float64)
    if bool(subtract_baseline_from_data):
        data = measured - baseline
        simulation = peak
    else:
        data = measured
        simulation = peak + baseline
    finite_values = np.concatenate(
        [
            np.abs(data[np.isfinite(data)]),
            np.abs(simulation[np.isfinite(simulation)]),
        ]
    )
    scale = float(np.nanpercentile(finite_values, 99.0)) if finite_values.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return {"data": data / scale, "simulation": simulation / scale, "scale": float(scale)}


def _finite_abs_percentile(values: object, percentile: float = 90.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return 0.0
    value = float(np.nanpercentile(finite, float(percentile)))
    return value if np.isfinite(value) else 0.0


def rod_profile_marker_l_mapping_is_valid(
    marker_source: pd.DataFrame | None,
    *,
    m_value: int,
    branch_value: str,
    min_points: int = 2,
) -> bool:
    markers = pd.DataFrame() if marker_source is None else pd.DataFrame(marker_source).copy()
    if markers.empty or not {"m", "branch", "qz_marker"}.issubset(markers.columns):
        return False
    l_column = "fit_l" if "fit_l" in markers else "l" if "l" in markers else "display_l"
    if l_column not in markers:
        return False
    marker_m = pd.to_numeric(markers["m"], errors="coerce").to_numpy(dtype=np.float64)
    marker_branch = markers["branch"].astype(str).to_numpy(dtype=object)
    sub = markers[(marker_m == float(int(m_value))) & (marker_branch == str(branch_value))].copy()
    if sub.empty:
        return False
    qz = pd.to_numeric(sub["qz_marker"], errors="coerce").to_numpy(dtype=np.float64)
    l_values = pd.to_numeric(sub[l_column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(qz) & np.isfinite(l_values)
    qz = qz[finite]
    l_values = l_values[finite]
    if qz.size < 1:
        return False
    order = np.argsort(qz)
    qz = qz[order]
    l_values = l_values[order]
    qz_unique = []
    l_unique = []
    for qz_value in np.unique(qz):
        same_qz = np.isclose(qz, float(qz_value), rtol=1.0e-9, atol=1.0e-9)
        same_l = l_values[same_qz]
        if same_l.size == 0:
            continue
        if float(np.nanmax(same_l) - np.nanmin(same_l)) > 1.0e-6:
            return False
        qz_unique.append(float(qz_value))
        l_unique.append(float(np.nanmedian(same_l)))
    qz = np.asarray(qz_unique, dtype=np.float64)
    l_values = np.asarray(l_unique, dtype=np.float64)
    if qz.size == 1:
        return bool(qz[0] > 0.0 and l_values[0] > 0.0)
    if qz.size < int(min_points) or np.nanmax(qz) <= np.nanmin(qz):
        return False
    if np.any(np.diff(l_values) < -1.0e-6) or np.nanmax(l_values) <= np.nanmin(l_values):
        return False
    slope, _intercept = np.polyfit(qz, l_values, 1)
    return bool(np.isfinite(slope) and float(slope) > 0.0)


def rod_profile_plot_model_decision(
    sample_stem: str,
    m_value: int,
    branch_value: str,
    profile_rows: pd.DataFrame,
    marker_source: pd.DataFrame | None,
    *,
    transverse_background_enabled: bool,
    background_subtraction_disabled: bool = False,
    peak_to_data_cancel_ratio: float = 4.0,
    baseline_to_peak_cancel_ratio: float = 0.50,
) -> dict[str, object]:
    table = pd.DataFrame(profile_rows).copy()
    is_pbi2_nonzero = str(sample_stem).lower().startswith("pbi2") and int(m_value) != 0
    default_density_column = (
        "joint_peak_density"
        if "joint_peak_density" in table.columns
        else "fit_density"
        if "fit_density" in table.columns
        else None
    )
    default_baseline_column = (
        None
        if bool(transverse_background_enabled)
        else "joint_linear_baseline_density"
        if "joint_linear_baseline_density" in table.columns
        else None
    )
    if is_pbi2_nonzero and bool(background_subtraction_disabled):
        model_column = default_density_column
        for candidate_column in ("joint_fit_density", "fit_density"):
            if candidate_column in table.columns:
                model_column = candidate_column
                break
        reason = (
            "pbi2_no_background_subtraction_debug"
            if model_column
            else "missing_model_density"
        )
        return {
            "plot_model": bool(model_column),
            "data_column": "background_density",
            "density_column": model_column,
            "baseline_column": None,
            "subtract_baseline_from_data": False,
            "label": "Fit" if model_column else None,
            "reason": reason,
            "metrics": {"background_subtraction_disabled": True},
        }
    if not (is_pbi2_nonzero and bool(transverse_background_enabled)):
        return {
            "plot_model": bool(default_density_column),
            "data_column": "background_density",
            "density_column": default_density_column,
            "baseline_column": default_baseline_column,
            "subtract_baseline_from_data": True,
            "label": "Simulation" if default_density_column else None,
            "reason": "default_peak_model" if default_density_column else "missing_model_density",
            "metrics": {},
        }

    l_mapping_valid = rod_profile_marker_l_mapping_is_valid(
        marker_source, m_value=int(m_value), branch_value=str(branch_value)
    )
    model_column = (
        "joint_fit_density"
        if "joint_fit_density" in table.columns
        else "fit_density"
        if "fit_density" in table.columns
        else None
    )
    data_column = (
        "background_density_raw"
        if "background_density_raw" in table.columns
        else "background_density"
    )
    additive_background_column = (
        "qr_sideband_background_density"
        if "qr_sideband_background_density" in table.columns
        else None
    )
    if model_column is None:
        return {
            "plot_model": False,
            "data_column": data_column,
            "density_column": None,
            "baseline_column": additive_background_column,
            "subtract_baseline_from_data": False,
            "label": None,
            "reason": "missing_model_density",
            "metrics": {"valid_l_mapping": bool(l_mapping_valid)},
        }

    data_scale = _finite_abs_percentile(table.get("background_density", []), 90.0)
    raw_data_scale = _finite_abs_percentile(table.get(data_column, []), 90.0)
    peak_scale = _finite_abs_percentile(table.get("joint_peak_density", []), 90.0)
    baseline_scale = _finite_abs_percentile(table.get("joint_linear_baseline_density", []), 90.0)
    sideband_scale = _finite_abs_percentile(table.get(additive_background_column, []), 90.0)
    denominator = max(data_scale, 1.0e-12)
    peak_to_data = peak_scale / denominator
    baseline_to_peak = baseline_scale / max(peak_scale, 1.0e-12)
    baseline_cancellation_suspected = bool(
        peak_to_data >= float(peak_to_data_cancel_ratio)
        and baseline_to_peak >= float(baseline_to_peak_cancel_ratio)
    )
    metrics = {
        "valid_l_mapping": bool(l_mapping_valid),
        "data_abs_p90": float(data_scale),
        "raw_data_abs_p90": float(raw_data_scale),
        "peak_abs_p90": float(peak_scale),
        "baseline_abs_p90": float(baseline_scale),
        "sideband_background_abs_p90": float(sideband_scale),
        "peak_to_data_ratio": float(peak_to_data),
        "baseline_to_peak_ratio": float(baseline_to_peak),
        "baseline_cancellation_suspected": baseline_cancellation_suspected,
    }

    return {
        "plot_model": True,
        "data_column": data_column,
        "density_column": model_column,
        "baseline_column": additive_background_column,
        "subtract_baseline_from_data": False,
        "label": "Fit",
        "reason": "pbi2_raw_with_background_fit",
        "metrics": metrics,
    }


def rod_profile_normalized_payload_for_plot_decision(
    profile_rows: pd.DataFrame,
    plot_decision: dict[str, object],
) -> tuple[bool, dict[str, np.ndarray | float]]:
    table = pd.DataFrame(profile_rows)
    model_column = str(plot_decision.get("density_column") or "")
    baseline_column = str(plot_decision.get("baseline_column") or "")
    data_column = str(plot_decision.get("data_column") or "background_density")
    plot_model = bool(plot_decision.get("plot_model")) and model_column in table
    model_density = (
        table[model_column] if plot_model else np.zeros(table.shape[0], dtype=np.float64)
    )
    baseline_density = table[baseline_column] if baseline_column in table else None
    data_density = table[data_column] if data_column in table else table["background_density"]
    return plot_model, normalized_data_simulation_payload(
        data_density,
        model_density,
        baseline_density,
        subtract_baseline_from_data=bool(plot_decision.get("subtract_baseline_from_data", True)),
    )


def positive_log_plot_values(values: object):
    arr = np.asarray(values, dtype=np.float64)
    return np.ma.masked_where(~np.isfinite(arr) | (arr <= 0.0), arr)


def apply_positive_log_y_axis(ax: object, *series: object) -> None:
    positive_values = []
    for values in series:
        arr = np.ma.asarray(values, dtype=np.float64).compressed()
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size:
            positive_values.append(arr)
    if not positive_values:
        ax.set_yscale("log")
        ax.set_ylim(1.0e-4, 1.0)
        return
    y = np.concatenate(positive_values)
    ymin = max(float(np.nanmin(y)) * 0.65, 1.0e-4)
    ymax = float(np.nanmax(y)) * 1.25
    if not np.isfinite(ymin) or ymin <= 0.0:
        ymin = 1.0e-4
    if not np.isfinite(ymax) or ymax <= ymin:
        ymax = ymin * 10.0
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)


def sample_uses_pbi2_rod_plot_policy(sample_stem: object) -> bool:
    return str(sample_stem).lower().startswith("pbi2")


def rod_profile_l_axis_limits_for_sample(
    sample_stem: object,
    limits: tuple[float, float],
    *,
    pbi2_l_max: float = 3.0,
) -> tuple[float, float]:
    lower, upper = (float(limits[0]), float(limits[1]))
    if not sample_uses_pbi2_rod_plot_policy(sample_stem):
        return lower, upper
    capped_upper = float(pbi2_l_max)
    if not np.isfinite(capped_upper):
        return lower, upper
    return lower, max(lower + 1.0e-6, capped_upper)


try:
    from scipy.optimize import nnls as _rod_qz_nnls
except Exception:
    _rod_qz_nnls = None

# Tail-aware 1D fit for rod profiles. Pearson-VII/Moffat basis keeps physical tails without the slow detector-forward solve.
ROD_QZ_TAIL_POWER_GRID = np.asarray([0.75, 1.0, 1.35, 2.0, 3.5, 7.0], dtype=np.float64)
ROD_QZ_TAIL_WIDTH_SCALE_GRID = np.asarray([0.55, 0.80, 1.15, 1.70, 2.50], dtype=np.float64)
ROD_QZ_TAIL_CENTER_SEARCH_BINS = 5.0
ROD_QZ_TAIL_MIN_HALFWIDTH_BINS = 0.55
ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION = 0.70
ROD_QZ_TAIL_MAX_AUTO_PEAKS = 10
ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE = 1.0e-5


def gaussian_sum_qz_model(
    x_values: object, amplitudes: object, centers: object, sigmas: object
) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    amps = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    ctrs = np.asarray(centers, dtype=np.float64).reshape(-1)
    widths = np.maximum(np.asarray(sigmas, dtype=np.float64).reshape(-1), 1.0e-12)
    model = np.zeros(x.shape, dtype=np.float64)
    for amp, center, sigma in zip(amps, ctrs, widths, strict=False):
        model += float(amp) * np.exp(-0.5 * ((x - float(center)) / float(sigma)) ** 2)
    return model


def _unique_sorted_markers(values: object, *, min_separation: float) -> np.ndarray:
    markers = np.asarray(values, dtype=np.float64).reshape(-1)
    markers = np.sort(markers[np.isfinite(markers)])
    if markers.size <= 1:
        return markers
    kept = [float(markers[0])]
    for value in markers[1:]:
        if abs(float(value) - kept[-1]) >= float(min_separation):
            kept.append(float(value))
    return np.asarray(kept, dtype=np.float64)


def _qz_grid_step(x_values: object) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    finite = np.sort(np.unique(x[np.isfinite(x)]))
    if finite.size < 2:
        return 1.0e-3
    dx = float(np.nanmedian(np.diff(finite)))
    return max(dx, 1.0e-9)


def _nanfilled_profile(values: object) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y)
    if not np.any(finite):
        return np.zeros(y.shape, dtype=np.float64)
    fill = float(np.nanmedian(y[finite]))
    return np.where(finite, y, fill).astype(np.float64, copy=False)


def _center_search_profile(values: object, *, window: int = 5) -> np.ndarray:
    del window
    return _nanfilled_profile(values)


def _nearest_marker_spacing(markers: np.ndarray, fallback: float) -> np.ndarray:
    markers = np.asarray(markers, dtype=np.float64).reshape(-1)
    out = np.full(markers.shape, max(float(fallback), 1.0e-9), dtype=np.float64)
    if markers.size <= 1:
        return out
    for idx, marker in enumerate(markers):
        distances = np.abs(markers - float(marker))
        distances = distances[distances > 0.0]
        if distances.size:
            out[idx] = max(float(np.nanmin(distances)), 1.0e-9)
    return out


def _refine_centers_to_local_maxima(
    x: np.ndarray, y: np.ndarray, markers: np.ndarray, dx: float
) -> np.ndarray:
    y_center = _center_search_profile(np.clip(y, 0.0, None), window=5)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), dx)
    spacing = _nearest_marker_spacing(markers, x_span)
    centers = markers.copy()
    for idx, marker in enumerate(markers):
        half_window = max(
            2.0 * dx, min(ROD_QZ_TAIL_CENTER_SEARCH_BINS * dx, 0.35 * float(spacing[idx]))
        )
        local = np.isfinite(x) & np.isfinite(y_center) & (np.abs(x - float(marker)) <= half_window)
        if np.any(local):
            local_indices = np.flatnonzero(local)
            best = local_indices[int(np.nanargmax(y_center[local]))]
            centers[idx] = float(x[best])
    return centers


def _fallback_markers_from_profile(x: np.ndarray, target: np.ndarray, dx: float) -> np.ndarray:
    y = _center_search_profile(np.clip(target, 0.0, None), window=5)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 5:
        return np.empty(0, dtype=np.float64)
    positive = y[finite & (y > 0.0)]
    if positive.size == 0:
        return np.empty(0, dtype=np.float64)
    threshold = max(
        float(np.nanpercentile(positive, 65.0)), 0.08 * float(np.nanpercentile(positive, 98.0))
    )
    candidates = []
    for idx in range(1, y.size - 1):
        if not (
            finite[idx] and y[idx] >= threshold and y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]
        ):
            continue
        candidates.append((float(y[idx]), float(x[idx])))
    if not candidates:
        best = int(np.nanargmax(np.where(finite, y, -np.inf)))
        return np.asarray([float(x[best])], dtype=np.float64)
    candidates.sort(reverse=True)
    kept: list[float] = []
    min_sep = max(3.0 * dx, 1.0e-9)
    for _height, center in candidates:
        if all(abs(center - prev) >= min_sep for prev in kept):
            kept.append(float(center))
        if len(kept) >= int(ROD_QZ_TAIL_MAX_AUTO_PEAKS):
            break
    return np.asarray(sorted(kept), dtype=np.float64)


def _estimate_peak_hwhm(
    x: np.ndarray, y: np.ndarray, *, center: float, spacing: float, dx: float
) -> float:
    y_center = _center_search_profile(np.clip(y, 0.0, None), window=5)
    finite = np.isfinite(x) & np.isfinite(y_center)
    if not np.any(finite):
        return max(1.5 * dx, 1.0e-9)
    x_span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), dx)
    center_index = int(np.nanargmin(np.where(finite, np.abs(x - float(center)), np.inf)))
    height = max(float(y_center[center_index]), 1.0e-12)
    half_height = 0.5 * height
    search_radius = max(5.0 * dx, min(0.60 * max(float(spacing), dx), 0.45 * x_span))
    local = finite & (np.abs(x - float(center)) <= search_radius)
    local_indices = np.flatnonzero(local)
    left_candidates = local_indices[
        (local_indices < center_index) & (y_center[local_indices] <= half_height)
    ]
    right_candidates = local_indices[
        (local_indices > center_index) & (y_center[local_indices] <= half_height)
    ]
    left_distance = (
        abs(float(center) - float(x[left_candidates[-1]]))
        if left_candidates.size
        else search_radius
    )
    right_distance = (
        abs(float(x[right_candidates[0]]) - float(center))
        if right_candidates.size
        else search_radius
    )
    hwhm = 0.5 * max(left_distance + right_distance, dx)
    min_hwhm = float(ROD_QZ_TAIL_MIN_HALFWIDTH_BINS) * dx
    max_hwhm = max(
        min_hwhm,
        min(float(ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION) * max(float(spacing), dx), 0.45 * x_span),
    )
    return float(np.clip(hwhm, min_hwhm, max_hwhm))


def _pearson_vii_profile(
    x_values: object, *, center: float, hwhm: float, tail_power: float
) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    hwhm = max(float(hwhm), 1.0e-12)
    tail_power = max(float(tail_power), 0.25)
    alpha_denominator = max(2.0 ** (1.0 / tail_power) - 1.0, 1.0e-12)
    alpha = hwhm / np.sqrt(alpha_denominator)
    profile = np.power(1.0 + ((x - float(center)) / alpha) ** 2, -tail_power)
    return np.where(np.isfinite(profile), profile, 0.0)


def _tail_aware_basis(
    x: np.ndarray,
    target: np.ndarray,
    centers: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x_span = max(float(np.nanmax(x[np.isfinite(x)]) - np.nanmin(x[np.isfinite(x)])), dx)
    spacing = _nearest_marker_spacing(centers, x_span)
    columns = []
    metadata: list[dict[str, object]] = []
    for peak_index, center in enumerate(centers):
        base_hwhm = _estimate_peak_hwhm(
            x, target, center=float(center), spacing=float(spacing[peak_index]), dx=dx
        )
        min_hwhm = float(ROD_QZ_TAIL_MIN_HALFWIDTH_BINS) * dx
        max_hwhm = max(
            min_hwhm,
            float(ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION) * max(float(spacing[peak_index]), dx),
        )
        for width_scale in ROD_QZ_TAIL_WIDTH_SCALE_GRID:
            hwhm = float(np.clip(base_hwhm * float(width_scale), min_hwhm, max_hwhm))
            for tail_power in ROD_QZ_TAIL_POWER_GRID:
                basis = _pearson_vii_profile(
                    x, center=float(center), hwhm=hwhm, tail_power=float(tail_power)
                )
                if np.nanmax(basis) <= 0.0:
                    continue
                columns.append(basis)
                metadata.append(
                    {
                        "peak_index": int(peak_index),
                        "center": float(center),
                        "hwhm": float(hwhm),
                        "tail_power": float(tail_power),
                        "width_scale": float(width_scale),
                    }
                )
    if not columns:
        return np.zeros((x.size, 0), dtype=np.float64), []
    return np.column_stack(columns).astype(np.float64, copy=False), metadata


def _shared_linear_qz_baseline(
    x_values: object,
    y_values: object,
    *,
    finite_mask: np.ndarray | None = None,
    peak_centers: object | None = None,
    dx: float | None = None,
) -> dict[str, object]:
    """Robust y=b+m*(Qz-Qz_ref) baseline shared by all peaks in one rod/branch profile."""
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite_mask is not None:
        finite &= np.asarray(finite_mask, dtype=bool).reshape(-1)
    if np.count_nonzero(finite) < 3:
        baseline = np.zeros(x.shape, dtype=np.float64)
        return {"density": baseline, "intercept": 0.0, "slope": 0.0, "x_ref": 0.0, "success": False}

    qz_ref = float(np.nanmedian(x[finite]))
    dx_values = x - qz_ref
    fit_mask = finite.copy()
    centers = np.asarray([] if peak_centers is None else peak_centers, dtype=np.float64).reshape(-1)
    centers = centers[np.isfinite(centers)]
    grid_step = max(
        float(dx) if dx is not None and np.isfinite(dx) else _qz_grid_step(x[finite]), 1.0e-9
    )
    if centers.size:
        x_span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), grid_step)
        spacing = _nearest_marker_spacing(centers, x_span)
        for center, local_spacing in zip(centers, spacing, strict=False):
            exclusion = max(2.0 * grid_step, min(0.30 * float(local_spacing), 7.0 * grid_step))
            fit_mask &= np.abs(x - float(center)) > exclusion

    if np.count_nonzero(fit_mask) < max(4, int(0.25 * np.count_nonzero(finite))):
        cutoff = float(np.nanpercentile(y[finite], 55.0))
        fit_mask = finite & (y <= cutoff)
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = finite

    design = np.column_stack(
        [np.ones(np.count_nonzero(fit_mask), dtype=np.float64), dx_values[fit_mask]]
    )
    response = y[fit_mask]
    try:
        coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    except Exception:
        coef = np.asarray([float(np.nanmedian(response)), 0.0], dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)

    full_design = np.column_stack(
        [np.ones(np.count_nonzero(finite), dtype=np.float64), dx_values[finite]]
    )
    full_response = y[finite]
    for _ in range(8):
        residual = full_response - full_design @ coef
        finite_resid = residual[np.isfinite(residual)]
        if finite_resid.size < 3:
            break
        mad = float(np.nanmedian(np.abs(finite_resid - np.nanmedian(finite_resid))))
        sigma = max(1.4826 * mad, float(np.nanstd(finite_resid)), 1.0e-12)
        positive = np.clip(residual / sigma, 0.0, None)
        negative = np.clip(-residual / (3.0 * sigma), 0.0, None)
        weights = 1.0 / (1.0 + positive**2 + 0.20 * negative**2)
        # Keep lower points influential and suppress high positive peak tails.
        low_cutoff = float(np.nanpercentile(full_response, 65.0))
        weights[full_response > low_cutoff] *= 0.45
        weights = np.clip(weights, 1.0e-3, 1.0)
        root_w = np.sqrt(weights)
        try:
            coef, *_ = np.linalg.lstsq(
                full_design * root_w[:, None], full_response * root_w, rcond=None
            )
        except Exception:
            break
        coef = np.asarray(coef, dtype=np.float64)

    if coef.size < 2 or not np.all(np.isfinite(coef[:2])):
        coef = np.asarray([float(np.nanmedian(y[finite])), 0.0], dtype=np.float64)
    baseline = float(coef[0]) + float(coef[1]) * dx_values
    baseline = np.where(np.isfinite(baseline), baseline, 0.0)
    return {
        "density": baseline,
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "x_ref": qz_ref,
        "success": True,
    }


def _weighted_nonnegative_amplitudes(
    basis: np.ndarray, target: np.ndarray, finite: np.ndarray
) -> np.ndarray:
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    positive = target[finite & (target > 0.0)]
    scale = float(np.nanpercentile(positive, 95.0)) if positive.size else 1.0
    scale = max(scale, 1.0e-12)
    relative = np.clip(np.clip(target, 0.0, None) / scale, 0.015, 1.0)
    weight = 1.0 / np.sqrt(relative)
    design = basis[finite, :] * weight[finite, None]
    response = np.clip(target[finite], 0.0, None) * weight[finite]
    usable_cols = (
        np.nanmax(np.abs(design), axis=0) > 0.0
        if design.size
        else np.zeros(basis.shape[1], dtype=bool)
    )
    amps = np.zeros(basis.shape[1], dtype=np.float64)
    if not np.any(usable_cols):
        return amps
    if _rod_qz_nnls is not None:
        solved, _ = _rod_qz_nnls(design[:, usable_cols], response)
    else:
        solved, *_ = np.linalg.lstsq(design[:, usable_cols], response, rcond=None)
        solved = np.clip(solved, 0.0, None)
    amps[usable_cols] = np.asarray(solved, dtype=np.float64)
    return amps


def _aggregate_tail_components(
    basis: np.ndarray,
    amplitudes: np.ndarray,
    basis_metadata: list[dict[str, object]],
    centers: np.ndarray,
    markers: np.ndarray,
    inverse: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    finite: np.ndarray,
    dx: float,
    measured: np.ndarray | None = None,
) -> list[dict[str, object]]:
    components = []
    if basis.size == 0 or amplitudes.size == 0:
        return components
    model = basis @ amplitudes
    model_peak = max(float(np.nanmax(model)) if model.size else 0.0, 1.0e-12)
    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    finite_values = np.asarray(finite, dtype=bool).reshape(-1)
    measured_values = (
        target_values if measured is None else np.asarray(measured, dtype=np.float64).reshape(-1)
    )
    if (
        x_values.shape != target_values.shape
        or x_values.shape != finite_values.shape
        or x_values.shape != model.shape
        or x_values.shape != measured_values.shape
    ):
        return components
    dx_value = max(float(dx), 1.0e-9) if np.isfinite(dx) else 1.0e-9
    finite_target = np.clip(target_values[finite_values], 0.0, None)
    target_peak = max(float(np.nanmax(finite_target)) if finite_target.size else 0.0, 1.0e-12)
    x_span = (
        max(
            float(np.nanmax(x_values[finite_values]) - np.nanmin(x_values[finite_values])), dx_value
        )
        if np.any(finite_values)
        else dx_value
    )
    spacing = _nearest_marker_spacing(centers, x_span)
    for peak_index, center in enumerate(centers):
        cols = [
            idx
            for idx, meta in enumerate(basis_metadata)
            if int(meta["peak_index"]) == int(peak_index)
        ]
        if not cols:
            continue
        component = basis[:, cols] @ amplitudes[cols]
        component_peak = float(np.nanmax(component)) if component.size else 0.0
        active = [
            basis_metadata[idx] | {"amplitude": float(amplitudes[idx])}
            for idx in cols
            if float(amplitudes[idx]) > 0.0
        ]
        fallback_hwhm = np.nan
        fallback_tail_power = np.nan
        marker_supported = False
        if component_peak < float(ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE) * model_peak:
            marker = float(markers[peak_index]) if peak_index < markers.size else float(center)
            support_center = marker if np.isfinite(marker) else float(center)
            half_window = max(
                2.0 * dx_value,
                min(0.35 * float(spacing[peak_index]), 6.0 * dx_value),
            )
            local = (
                finite_values
                & np.isfinite(target_values)
                & (np.abs(x_values - support_center) <= half_window)
            )
            if not np.any(local):
                continue
            local_indices = np.flatnonzero(local)
            best = int(local_indices[int(np.nanargmax(target_values[local]))])
            local_peak = float(target_values[best])
            local_floor = float(np.nanmedian(target_values[local]))
            local_prominence = local_peak - local_floor
            fallback_peak = max(local_peak, local_prominence, 0.0)
            fallback_shape = target_values
            is_local_maximum = (
                best != int(local_indices[0])
                and best != int(local_indices[-1])
                and best > 0
                and best < target_values.size - 1
                and local_peak >= float(target_values[best - 1])
                and local_peak >= float(target_values[best + 1])
            )
            marker_supported = (
                is_local_maximum
                and local_prominence > max(1.0e-4 * target_peak, 1.0e-12)
                and fallback_peak > 0.0
            )
            if not marker_supported and measured is not None:
                measured_local = (
                    finite_values
                    & np.isfinite(measured_values)
                    & (np.abs(x_values - support_center) <= half_window)
                )
                if np.any(measured_local):
                    measured_indices = np.flatnonzero(measured_local)
                    measured_best = int(
                        measured_indices[int(np.nanargmax(measured_values[measured_local]))]
                    )
                    measured_peak = float(measured_values[measured_best])
                    measured_floor = float(np.nanmedian(measured_values[measured_local]))
                    measured_prominence = measured_peak - measured_floor
                    measured_is_local_maximum = (
                        measured_best != int(measured_indices[0])
                        and measured_best != int(measured_indices[-1])
                        and measured_best > 0
                        and measured_best < measured_values.size - 1
                        and measured_peak >= float(measured_values[measured_best - 1])
                        and measured_peak >= float(measured_values[measured_best + 1])
                    )
                    marker_supported = measured_is_local_maximum and measured_prominence > max(
                        1.0e-4 * target_peak, 1.0e-12
                    )
                    if marker_supported:
                        best = measured_best
                        fallback_peak = max(measured_prominence, 0.0)
                        fallback_shape = measured_values - measured_floor
            if not marker_supported:
                continue
            center = float(x_values[best])
            fallback_hwhm = _estimate_peak_hwhm(
                x_values,
                fallback_shape,
                center=float(center),
                spacing=float(spacing[peak_index]),
                dx=dx_value,
            )
            fallback_tail_power = 2.0
            component = fallback_peak * _pearson_vii_profile(
                x_values,
                center=float(center),
                hwhm=float(fallback_hwhm),
                tail_power=float(fallback_tail_power),
            )
            component_peak = float(np.nanmax(component)) if component.size else 0.0
        else:
            marker = float(markers[peak_index]) if peak_index < markers.size else float(center)
        if active:
            best = max(active, key=lambda item: float(item["amplitude"]))
            hwhm = float(best["hwhm"])
            tail_power = float(best["tail_power"])
        else:
            hwhm = float(fallback_hwhm) if np.isfinite(fallback_hwhm) else np.nan
            tail_power = float(fallback_tail_power) if np.isfinite(fallback_tail_power) else np.nan
        components.append(
            {
                "component_id": int(peak_index + 1),
                "marker": marker,
                "center": float(center),
                "hwhm": hwhm,
                "tail_power": tail_power,
                "marker_supported": marker_supported,
                "density": np.asarray(component[inverse], dtype=np.float64).tolist(),
            }
        )
    return components


def _pearson_vii_component_sum(x_values: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    model = np.zeros(x.shape, dtype=np.float64)
    if p.size % 4 != 0:
        return model
    for idx in range(p.size // 4):
        amp = max(float(p[4 * idx + 0]), 0.0)
        center = float(p[4 * idx + 1])
        hwhm = max(float(p[4 * idx + 2]), 1.0e-12)
        tail_power = max(float(p[4 * idx + 3]), 0.25)
        model += amp * _pearson_vii_profile(x, center=center, hwhm=hwhm, tail_power=tail_power)
    return np.where(np.isfinite(model), np.clip(model, 0.0, None), 0.0)


def _component_density_to_sorted(
    component: dict[str, object], order: np.ndarray, n: int
) -> np.ndarray:
    density = np.asarray(component.get("density", []), dtype=np.float64).reshape(-1)
    if density.size != n:
        return np.zeros(n, dtype=np.float64)
    return density[order]


def _refine_pearson_vii_components(
    x: np.ndarray,
    target: np.ndarray,
    finite: np.ndarray,
    components: list[dict[str, object]],
    order: np.ndarray,
    inverse: np.ndarray,
    dx: float,
    *,
    measured: np.ndarray | None = None,
    baseline_initial: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if not bool(ROD_QZ_NONLINEAR_REFINEMENT_ENABLED) or not components:
        return None

    target = np.asarray(target, dtype=np.float64)
    fit_observed = target if measured is None else np.asarray(measured, dtype=np.float64)
    include_baseline = bool(
        baseline_initial is not None and bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED)
    )
    finite_target = np.clip(target[finite], 0.0, None)
    target_peak = float(np.nanmax(finite_target)) if finite_target.size else 0.0
    if not np.isfinite(target_peak) or target_peak <= 0.0:
        return None

    initial_components = []
    for component in components:
        density_sorted = _component_density_to_sorted(component, order, x.size)
        amp0 = float(np.nanmax(density_sorted)) if density_sorted.size else 0.0
        marker_supported = bool(component.get("marker_supported", False))
        min_component_amp = max(1.0e-12, float(ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE) * target_peak)
        if not np.isfinite(amp0) or (amp0 <= min_component_amp and not marker_supported):
            continue
        center0 = float(component.get("center", np.nan))
        hwhm0 = float(component.get("hwhm", np.nan))
        tail0 = float(component.get("tail_power", np.nan))
        marker0 = float(component.get("marker", center0))
        if not np.isfinite(center0):
            center0 = marker0
        if not np.isfinite(hwhm0) or hwhm0 <= 0.0:
            hwhm0 = max(2.0 * dx, 1.0e-9)
        if not np.isfinite(tail0) or tail0 <= 0.0:
            tail0 = 2.0
        initial_components.append(
            {
                "component_id": int(component.get("component_id", len(initial_components) + 1)),
                "marker": marker0,
                "marker_supported": marker_supported,
                "amp": amp0,
                "center": center0,
                "hwhm": hwhm0,
                "tail_power": tail0,
            }
        )

    if len(initial_components) == 0 or len(initial_components) > int(
        ROD_QZ_NONLINEAR_MAX_COMPONENTS
    ):
        return None

    centers = np.asarray([item["center"] for item in initial_components], dtype=np.float64)
    x_span = max(float(np.nanmax(x[finite]) - np.nanmin(x[finite])), dx)
    spacing = _nearest_marker_spacing(centers, x_span)
    p0 = []
    lower = []
    upper = []
    min_tail, max_tail = ROD_QZ_NONLINEAR_TAIL_POWER_BOUNDS
    max_amp = max(10.0 * target_peak, 1.0)
    for idx, item in enumerate(initial_components):
        center0 = float(item["center"])
        center_radius = max(
            float(ROD_QZ_NONLINEAR_CENTER_BOUND_BINS) * dx,
            min(0.35 * float(spacing[idx]), 0.25 * x_span),
        )
        min_hwhm = max(float(ROD_QZ_TAIL_MIN_HALFWIDTH_BINS) * dx, 1.0e-9)
        max_hwhm = max(
            min_hwhm * 1.01,
            min(
                float(ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION) * max(float(spacing[idx]), dx),
                0.50 * x_span,
            ),
        )
        hwhm0 = float(np.clip(item["hwhm"], min_hwhm, max_hwhm))
        p0.extend(
            [
                float(np.clip(item["amp"], 1.0e-12, max_amp)),
                center0,
                hwhm0,
                float(np.clip(item["tail_power"], min_tail, max_tail)),
            ]
        )
        lower.extend([0.0, center0 - center_radius, min_hwhm, float(min_tail)])
        upper.extend([max_amp, center0 + center_radius, max_hwhm, float(max_tail)])

    baseline_x_ref = float(np.nanmedian(x[finite]))
    baseline_intercept0 = 0.0
    baseline_slope0 = 0.0
    if include_baseline:
        baseline_x_ref = float(baseline_initial.get("x_ref", baseline_x_ref))
        baseline_intercept0 = float(baseline_initial.get("intercept", 0.0))
        baseline_slope0 = float(baseline_initial.get("slope", 0.0))
        observed = fit_observed[finite]
        observed_span = max(
            float(np.nanpercentile(observed, 97.5) - np.nanpercentile(observed, 2.5))
            if observed.size
            else 0.0,
            target_peak,
            1.0,
        )
        slope_bound = max(
            8.0 * observed_span / max(x_span, dx),
            abs(baseline_slope0) * 5.0 + observed_span / max(x_span, dx),
            1.0e-12,
        )
        p0.extend([baseline_intercept0, baseline_slope0])
        lower.extend([baseline_intercept0 - 8.0 * observed_span, -slope_bound])
        upper.extend([baseline_intercept0 + 8.0 * observed_span, slope_bound])

    p0 = np.asarray(p0, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    p0 = np.minimum(np.maximum(p0, lower + 1.0e-10), upper - 1.0e-10)

    positive = target[finite & (target > 0.0)]
    scale = float(np.nanpercentile(positive, 95.0)) if positive.size else max(target_peak, 1.0)
    if include_baseline:
        observed_abs = np.abs(fit_observed[finite])
        if observed_abs.size:
            scale = max(scale, float(np.nanpercentile(observed_abs, 95.0)))
    scale = max(scale, 1.0e-12)
    relative = np.clip(np.clip(target, 0.0, None) / scale, 0.02, 1.0)
    weight = 1.0 / np.sqrt(relative)

    peak_param_count = 4 * len(initial_components)

    def _split_model(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        peak_params = np.asarray(params[:peak_param_count], dtype=np.float64)
        peak_model = _pearson_vii_component_sum(x, peak_params)
        if include_baseline:
            intercept = float(params[peak_param_count])
            slope = float(params[peak_param_count + 1])
            baseline_model = intercept + slope * (x - baseline_x_ref)
        else:
            baseline_model = np.zeros(x.shape, dtype=np.float64)
        total_model = peak_model + baseline_model
        return peak_model, baseline_model, total_model

    response = fit_observed if include_baseline else np.clip(target, 0.0, None)
    log_mask = finite & np.isfinite(response) & (response > 0.0)
    log_response_positive = response[log_mask]
    log_floor = (
        max(
            float(np.nanpercentile(log_response_positive, 5.0))
            * float(ROD_QZ_NONLINEAR_LOG_FLOOR_FRACTION),
            1.0e-12,
        )
        if log_response_positive.size
        else 1.0e-12
    )
    log_response = np.log10(np.clip(log_response_positive, 0.0, None) + log_floor)
    log_residual_weight = max(float(ROD_QZ_NONLINEAR_LOG_RESIDUAL_WEIGHT), 0.0)

    def residual(params: np.ndarray) -> np.ndarray:
        _peak_model, _baseline_model, total_model = _split_model(
            np.asarray(params, dtype=np.float64)
        )
        linear_residual = weight[finite] * (total_model[finite] - response[finite]) / scale
        if log_residual_weight <= 0.0 or log_response.size == 0:
            return linear_residual
        log_model = np.log10(np.clip(total_model[log_mask], 0.0, None) + log_floor)
        return np.concatenate([linear_residual, log_residual_weight * (log_model - log_response)])

    initial_peak_model, initial_baseline_model, initial_total_model = _split_model(p0)
    initial_cost = float(np.nanmean((initial_total_model[finite] - response[finite]) ** 2))
    try:
        result = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=int(ROD_QZ_NONLINEAR_MAX_NFEV),
            x_scale=np.maximum(np.abs(p0), 1.0),
        )
    except Exception:
        return None

    params = np.asarray(result.x, dtype=np.float64)
    peak_model, baseline_model, total_model = _split_model(params)
    refined_cost = float(np.nanmean((total_model[finite] - response[finite]) ** 2))
    if not np.isfinite(refined_cost) or (
        np.isfinite(initial_cost) and refined_cost > 1.15 * initial_cost
    ):
        return None

    refined_components = []
    model_peak = max(float(np.nanmax(peak_model)) if peak_model.size else 0.0, 1.0e-12)
    for idx, item in enumerate(initial_components):
        amp = max(float(params[4 * idx + 0]), 0.0)
        center = float(params[4 * idx + 1])
        hwhm = max(float(params[4 * idx + 2]), 1.0e-12)
        tail_power = max(float(params[4 * idx + 3]), 0.25)
        density_sorted = amp * _pearson_vii_profile(
            x, center=center, hwhm=hwhm, tail_power=tail_power
        )
        if float(np.nanmax(density_sorted)) < float(
            ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE
        ) * model_peak and not bool(item.get("marker_supported", False)):
            continue
        refined_components.append(
            {
                "component_id": int(item["component_id"]),
                "marker": float(item["marker"]),
                "center": center,
                "hwhm": hwhm,
                "tail_power": tail_power,
                "amplitude": amp,
                "density": np.asarray(density_sorted[inverse], dtype=np.float64).tolist(),
            }
        )

    if include_baseline:
        baseline_intercept = float(params[peak_param_count])
        baseline_slope = float(params[peak_param_count + 1])
    else:
        baseline_intercept = 0.0
        baseline_slope = 0.0

    return {
        "model_density": np.asarray(total_model[inverse], dtype=np.float64),
        "peak_density": np.asarray(peak_model[inverse], dtype=np.float64),
        "baseline_density": np.asarray(baseline_model[inverse], dtype=np.float64),
        "components": refined_components,
        "cost": refined_cost,
        "params": params,
        "peak_params": params[:peak_param_count],
        "baseline_intercept": baseline_intercept,
        "baseline_slope": baseline_slope,
        "baseline_x_ref": baseline_x_ref,
        "success": bool(result.success),
        "message": str(result.message),
    }


def _empty_joint_qz_payload(x_shape: tuple[int, ...], message: str) -> dict[str, object]:
    nan_model = np.full(x_shape, np.nan, dtype=np.float64)
    return {
        "success": False,
        "message": str(message),
        "model_density": nan_model,
        "peak_density": nan_model.copy(),
        "baseline_density": np.zeros(x_shape, dtype=np.float64),
        "components": [],
        "linear_baseline_intercept": np.nan,
        "linear_baseline_slope": np.nan,
        "linear_baseline_x_ref": np.nan,
    }


def fit_joint_qz_peak_sum(
    qz_center: object, background_density: object, qz_markers: object
) -> dict[str, object]:
    x_in = np.asarray(qz_center, dtype=np.float64).reshape(-1)
    measured_in = np.asarray(background_density, dtype=np.float64).reshape(-1)
    if x_in.size != measured_in.size or x_in.size < 6:
        return _empty_joint_qz_payload(x_in.shape, "bad_profile_shape")

    order = np.argsort(x_in)
    inverse = np.empty(order.size, dtype=np.int64)
    inverse[order] = np.arange(order.size)
    x = x_in[order]
    measured = measured_in[order]

    finite_measured = np.isfinite(x) & np.isfinite(measured)
    if np.count_nonzero(finite_measured) < 6:
        return _empty_joint_qz_payload(x_in.shape, "too_few_profile_points")

    dx = _qz_grid_step(x[finite_measured])
    x_min = float(np.nanmin(x[finite_measured]))
    x_max = float(np.nanmax(x[finite_measured]))
    markers = _unique_sorted_markers(qz_markers, min_separation=0.5 * dx)
    marker_source = "projected_markers"
    markers = markers[(markers >= x_min) & (markers <= x_max)]

    baseline_payload = (
        _shared_linear_qz_baseline(
            x,
            measured,
            finite_mask=finite_measured,
            peak_centers=markers,
            dx=dx,
        )
        if bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED)
        else {
            "density": np.zeros(x.shape, dtype=np.float64),
            "intercept": 0.0,
            "slope": 0.0,
            "x_ref": float(np.nanmedian(x[finite_measured])),
            "success": True,
        }
    )
    baseline_sorted = np.asarray(
        baseline_payload.get("density", np.zeros(x.shape, dtype=np.float64)), dtype=np.float64
    )
    target = measured - baseline_sorted
    finite = finite_measured & np.isfinite(target)
    if np.count_nonzero(finite) < 6:
        return _empty_joint_qz_payload(x_in.shape, "too_few_baseline_corrected_profile_points")

    if markers.size < 1:
        markers = _fallback_markers_from_profile(x, target, dx)
        marker_source = "auto_profile_maxima"
    if markers.size < 1:
        return _empty_joint_qz_payload(x_in.shape, "no_qz_markers")

    centers = _refine_centers_to_local_maxima(x, target, markers, dx)
    basis, basis_metadata = _tail_aware_basis(x, target, centers, dx)
    amps = _weighted_nonnegative_amplitudes(basis, target, finite)
    peak_model_sorted = basis @ amps if amps.size else np.zeros(x.shape, dtype=np.float64)
    peak_model_sorted = np.where(
        np.isfinite(peak_model_sorted), np.clip(peak_model_sorted, 0.0, None), 0.0
    )
    total_model_sorted = peak_model_sorted + baseline_sorted
    components = _aggregate_tail_components(
        basis,
        amps,
        basis_metadata,
        centers,
        markers,
        inverse,
        x,
        target,
        finite,
        dx,
        measured=measured,
    )
    dictionary_cost = float(np.nanmean((total_model_sorted[finite] - measured[finite]) ** 2))
    refined = _refine_pearson_vii_components(
        x,
        target,
        finite,
        components,
        order,
        inverse,
        dx,
        measured=measured,
        baseline_initial=baseline_payload if bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED) else None,
    )

    if refined is not None:
        family = (
            "nonlinear_refined_pearson_vii_peak_sum_plus_shared_linear_baseline"
            if bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED)
            else "nonlinear_refined_pearson_vii_peak_sum"
        )
        return {
            "success": True,
            "message": f"nonlinear_refined_pearson_vii_from_peak_markers_plus_shared_linear_baseline; marker_source={marker_source}; optimizer={refined.get('message', '')}",
            "model_density": np.asarray(refined["model_density"], dtype=np.float64),
            "peak_density": np.asarray(
                refined.get("peak_density", refined["model_density"]), dtype=np.float64
            ),
            "baseline_density": np.asarray(
                refined.get("baseline_density", np.zeros(x_in.shape, dtype=np.float64)),
                dtype=np.float64,
            ),
            "peak_count": int(len(refined.get("components", []))),
            "params": {
                "dictionary_amplitudes": amps.tolist(),
                "markers": markers.tolist(),
                "centers": centers.tolist(),
                "basis_metadata": basis_metadata,
                "tail_power_grid": np.asarray(ROD_QZ_TAIL_POWER_GRID, dtype=np.float64).tolist(),
                "width_scale_grid": np.asarray(
                    ROD_QZ_TAIL_WIDTH_SCALE_GRID, dtype=np.float64
                ).tolist(),
                "nonlinear_params": np.asarray(
                    refined.get("params", []), dtype=np.float64
                ).tolist(),
                "nonlinear_peak_params": np.asarray(
                    refined.get("peak_params", []), dtype=np.float64
                ).tolist(),
                "linear_baseline": {
                    "intercept": float(refined.get("baseline_intercept", np.nan)),
                    "slope": float(refined.get("baseline_slope", np.nan)),
                    "x_ref": float(refined.get("baseline_x_ref", np.nan)),
                    "equation": "density = intercept + slope*(qz_center - x_ref)",
                },
            },
            "components": refined.get("components", []),
            "cost": float(refined.get("cost", dictionary_cost)),
            "model_family": family,
            "shape_power": np.nan,
            "linear_baseline_intercept": float(refined.get("baseline_intercept", np.nan)),
            "linear_baseline_slope": float(refined.get("baseline_slope", np.nan)),
            "linear_baseline_x_ref": float(refined.get("baseline_x_ref", np.nan)),
        }

    baseline_unsorted = baseline_sorted[inverse]
    peak_unsorted = peak_model_sorted[inverse]
    total_unsorted = peak_unsorted + baseline_unsorted
    family = (
        "fast_tail_aware_pearson_vii_dictionary_nnls_plus_shared_linear_baseline"
        if bool(ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED)
        else "fast_tail_aware_pearson_vii_dictionary_nnls"
    )
    return {
        "success": True,
        "message": f"fast_tail_aware_pearson_vii_nnls_plus_shared_linear_baseline; marker_source={marker_source}",
        "model_density": total_unsorted,
        "peak_density": peak_unsorted,
        "baseline_density": baseline_unsorted,
        "peak_count": int(markers.size),
        "params": {
            "amplitudes": amps.tolist(),
            "markers": markers.tolist(),
            "centers": centers.tolist(),
            "basis_metadata": basis_metadata,
            "tail_power_grid": np.asarray(ROD_QZ_TAIL_POWER_GRID, dtype=np.float64).tolist(),
            "width_scale_grid": np.asarray(ROD_QZ_TAIL_WIDTH_SCALE_GRID, dtype=np.float64).tolist(),
            "linear_baseline": {
                "intercept": float(baseline_payload.get("intercept", np.nan)),
                "slope": float(baseline_payload.get("slope", np.nan)),
                "x_ref": float(baseline_payload.get("x_ref", np.nan)),
                "equation": "density = intercept + slope*(qz_center - x_ref)",
            },
        },
        "components": components,
        "cost": dictionary_cost,
        "model_family": family,
        "shape_power": np.nan,
        "linear_baseline_intercept": float(baseline_payload.get("intercept", np.nan)),
        "linear_baseline_slope": float(baseline_payload.get("slope", np.nan)),
        "linear_baseline_x_ref": float(baseline_payload.get("x_ref", np.nan)),
    }


def marker_qz_values_for_profile(
    marker_table: pd.DataFrame, *, m_value: int, branch_value: str
) -> np.ndarray:
    if marker_table.empty or "qz_marker" not in marker_table:
        return np.array([], dtype=np.float64)
    sub = marker_table[
        (np.asarray(marker_table["m"], dtype=int) == int(m_value))
        & (marker_table["branch"].astype(str) == str(branch_value))
    ]
    return np.asarray(sub["qz_marker"], dtype=np.float64)


def add_joint_qz_fit_columns(
    rod_profile_table: pd.DataFrame,
    marker_table: pd.DataFrame,
    *,
    bg: dict[str, object] | None = None,
    detector_q_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    detector_phi_map: np.ndarray | None = None,
    detector_solid_angle: np.ndarray | None = None,
    detector_two_theta_map: np.ndarray | None = None,
) -> pd.DataFrame:
    del bg, detector_q_maps, detector_phi_map, detector_solid_angle, detector_two_theta_map

    table = rod_profile_table.copy()
    fallback_fit = np.asarray(table.get("fit_density", np.nan), dtype=np.float64)
    table["joint_peak_density"] = fallback_fit
    table["joint_linear_baseline_density"] = np.zeros(table.shape[0], dtype=np.float64)
    table["joint_fit_density"] = fallback_fit
    table["joint_fit_success"] = False
    table["joint_fit_peak_count"] = 0
    table["joint_fit_message"] = "fallback_detector_peak_model"
    table["joint_fit_model_family"] = "detector_peak_model"
    table["joint_fit_cost"] = np.nan
    table["joint_fit_shape_power"] = np.nan
    table["joint_linear_baseline_intercept"] = np.nan
    table["joint_linear_baseline_slope"] = np.nan
    table["joint_linear_baseline_x_ref"] = np.nan
    table["joint_linear_baseline_equation"] = "density = intercept + slope*(qz_center - x_ref)"
    component_rows: list[dict[str, object]] = []
    component_columns = [
        "sample_name",
        "tilt_deg",
        "theta_initial_deg_used_for_q",
        "m",
        "qr",
        "branch",
        "phi_window",
        "component_id",
        "marker",
        "center",
        "hwhm",
        "tail_power",
        "qz_center",
        "component_density",
    ]
    if table.empty:
        table.attrs["joint_fit_component_table"] = pd.DataFrame(columns=component_columns)
        return table

    grouped_profiles = [
        (m_value, branch_value, sub.copy())
        for (m_value, branch_value), sub in table.groupby(["m", "branch"], sort=False)
    ]

    def _fit_joint_qz_group(
        group: tuple[object, object, pd.DataFrame],
    ) -> tuple[object, object, pd.DataFrame, dict[str, object]]:
        m_value, branch_value, sub = group
        qz_markers = marker_qz_values_for_profile(
            marker_table, m_value=int(m_value), branch_value=str(branch_value)
        )
        fit_payload = fit_joint_qz_peak_sum(sub["qz_center"], sub["background_density"], qz_markers)
        return m_value, branch_value, sub, fit_payload

    if len(grouped_profiles) > 1 and int(ROD_PROFILE_FIT_WORKERS) > 1:
        group_results = []
        with ThreadPoolExecutor(
            max_workers=min(int(ROD_PROFILE_FIT_WORKERS), len(grouped_profiles))
        ) as executor:
            futures = [executor.submit(_fit_joint_qz_group, group) for group in grouped_profiles]
            for future in as_completed(futures):
                group_results.append(future.result())
        group_results.sort(
            key=lambda payload: (
                branch_sort_key(str(payload[1]))
                if isinstance(payload[1], str)
                else (999, 999, 999, str(payload[1])),
                int(payload[0]) if str(payload[0]).lstrip("-").isdigit() else 0,
            )
        )
    else:
        group_results = [_fit_joint_qz_group(group) for group in grouped_profiles]

    for m_value, branch_value, sub, fit_payload in group_results:
        if not bool(fit_payload.get("success")):
            table.loc[sub.index, "joint_fit_message"] = str(
                fit_payload.get("message", "joint_fit_failed")
            )
            table.loc[sub.index, "joint_fit_model_family"] = str(
                fit_payload.get("model_family", "tail_aware_fit_failed")
            )
            continue

        peak_density = np.asarray(
            fit_payload.get("peak_density", fit_payload["model_density"]), dtype=np.float64
        )
        baseline_density = np.asarray(
            fit_payload.get("baseline_density", np.zeros(peak_density.shape, dtype=np.float64)),
            dtype=np.float64,
        )
        model_density = np.asarray(
            fit_payload.get("model_density", peak_density + baseline_density), dtype=np.float64
        )
        table.loc[sub.index, "joint_peak_density"] = peak_density
        table.loc[sub.index, "joint_linear_baseline_density"] = baseline_density
        table.loc[sub.index, "joint_fit_density"] = model_density
        table.loc[sub.index, "joint_fit_success"] = True
        table.loc[sub.index, "joint_fit_peak_count"] = int(fit_payload.get("peak_count", 0))
        table.loc[sub.index, "joint_fit_message"] = str(fit_payload.get("message", ""))
        table.loc[sub.index, "joint_fit_model_family"] = str(fit_payload.get("model_family", ""))
        table.loc[sub.index, "joint_fit_cost"] = float(fit_payload.get("cost", np.nan))
        table.loc[sub.index, "joint_fit_shape_power"] = float(
            fit_payload.get("shape_power", np.nan)
        )
        table.loc[sub.index, "joint_linear_baseline_intercept"] = float(
            fit_payload.get("linear_baseline_intercept", np.nan)
        )
        table.loc[sub.index, "joint_linear_baseline_slope"] = float(
            fit_payload.get("linear_baseline_slope", np.nan)
        )
        table.loc[sub.index, "joint_linear_baseline_x_ref"] = float(
            fit_payload.get("linear_baseline_x_ref", np.nan)
        )

        qz_values = np.asarray(sub["qz_center"], dtype=np.float64)
        first = sub.iloc[0]
        for component in fit_payload.get("components", []) or []:
            density = np.asarray(component.get("density", []), dtype=np.float64)
            if density.shape != qz_values.shape:
                continue
            for qz_value, density_value in zip(qz_values, density, strict=False):
                component_rows.append(
                    {
                        "sample_name": first.get("sample_name", SAMPLE_NAME),
                        "tilt_deg": float(first.get("tilt_deg", np.nan)),
                        "theta_initial_deg_used_for_q": float(
                            first.get("theta_initial_deg_used_for_q", np.nan)
                        ),
                        "m": int(m_value),
                        "qr": float(first.get("qr", np.nan)),
                        "branch": str(branch_value),
                        "phi_window": str(first.get("phi_window", "")),
                        "component_id": int(component.get("component_id", 0)),
                        "marker": float(component.get("marker", np.nan)),
                        "center": float(component.get("center", np.nan)),
                        "hwhm": float(component.get("hwhm", np.nan)),
                        "tail_power": float(component.get("tail_power", np.nan)),
                        "qz_center": float(qz_value),
                        "component_density": float(density_value),
                    }
                )

    table.attrs["joint_fit_component_table"] = pd.DataFrame(
        component_rows, columns=component_columns
    )
    return table


def nearest_projected_qz_for_fit(
    item: dict[str, object], samples: dict[str, object], phi_min: float, phi_max: float
) -> tuple[float, float, float] | None:
    tth = np.asarray(samples.get("two_theta"), dtype=np.float64)
    phi = np.asarray(samples.get("phi"), dtype=np.float64)
    qz = np.asarray(samples.get("qz"), dtype=np.float64)
    theta0 = float(item["params"][1])
    phi0 = float(item["params"][2])
    selected = (
        np.isfinite(tth)
        & np.isfinite(phi)
        & np.isfinite(qz)
        & (qz > POSITIVE_QZ_MIN)
        & (tth <= float(ROD_PROFILE_MAX_TWO_THETA_DEG))
        & (phi >= float(phi_min))
        & (phi <= float(phi_max))
    )
    if not np.any(selected):
        return None
    distance = ((tth[selected] - theta0) / 0.5) ** 2 + (
        wrapped_delta_deg(phi[selected], phi0) / 2.0
    ) ** 2
    local_idx = int(np.nanargmin(distance))
    absolute_idx = np.flatnonzero(selected)[local_idx]
    return float(qz[absolute_idx]), float(tth[absolute_idx]), float(phi[absolute_idx])


def refine_marker_to_local_peak(
    bg: dict[str, object], item: dict[str, object], branch_mask: np.ndarray, qz_map: np.ndarray
) -> tuple[float, float, float] | None:
    theta_axis = np.ascontiguousarray(bg["theta_axis"], dtype=np.float64)
    phi_axis = np.ascontiguousarray(bg["phi_axis"], dtype=np.float64)
    model = np.ascontiguousarray(bg["caked_peak_model"], dtype=np.float64)
    image = np.ascontiguousarray(bg["caked_image"], dtype=np.float64)
    params = np.asarray(item["params"], dtype=np.float64)
    theta0 = float(params[1])
    phi0 = float(params[2])
    theta_half_width = max(0.35, 3.0 * float(abs(params[3])))
    phi_half_width = max(1.2, 3.0 * float(abs(params[4])))
    success, qz_value, theta_value, phi_value = _refine_marker_to_local_peak_numba(
        theta_axis,
        phi_axis,
        model,
        image,
        np.ascontiguousarray(branch_mask, dtype=np.bool_),
        np.ascontiguousarray(qz_map, dtype=np.float64),
        theta0,
        phi0,
        theta_half_width,
        phi_half_width,
        THETA_HALF_WINDOW_DEG,
        PHI_HALF_WINDOW_DEG,
    )
    if not success:
        return None
    return float(qz_value), float(theta_value), float(phi_value)


def fallback_qz_marker_for_item(
    item: dict[str, object], phi_min: float, phi_max: float
) -> tuple[float, float, float] | None:
    try:
        params = np.asarray(item["params"], dtype=np.float64)
        theta0 = float(params[1])
        phi0 = float(params[2])
        qz0 = as_float(item.get("_qz_marker"), np.nan)
    except Exception:
        return None
    if not (np.isfinite(qz0) and np.isfinite(theta0) and np.isfinite(phi0)):
        return None
    if qz0 <= POSITIVE_QZ_MIN:
        return None
    if theta0 > float(ROD_PROFILE_MAX_TWO_THETA_DEG):
        return None
    if float(phi_min) <= float(phi_max):
        if not (float(phi_min) <= phi0 <= float(phi_max)):
            return None
    elif not (phi0 >= float(phi_min) or phi0 <= float(phi_max)):
        return None
    return qz0, theta0, phi0


def marker_rows_for_rod_branch(
    bg: dict[str, object],
    rod: dict[str, object],
    samples: dict[str, object],
    *,
    branch_name: str,
    branch_label: str,
    phi_min: float,
    phi_max: float,
    branch_mask: np.ndarray,
    qz_map: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    rejected_rod_keys = active_rejected_rod_keys()
    for item in candidate_marker_items_for_background(bg):
        row = dict(item)
        row["_required_branch"] = str(branch_name)
        if not accept_marker_for_plotted_rod(row, rod, rejected_rod_keys):
            continue
        try:
            _h, _k, l_value = parse_hkl_label(str(item["label"]))
        except Exception:
            continue
        nearest = nearest_projected_qz_for_fit(item, samples, phi_min, phi_max)
        if nearest is None:
            nearest = fallback_qz_marker_for_item(item, phi_min, phi_max)
        if nearest is None:
            continue
        qz_value, projected_tth, projected_phi = nearest
        refined = refine_marker_to_local_peak(bg, item, branch_mask, qz_map)
        if refined is None:
            refined_qz, refined_tth, refined_phi = qz_value, projected_tth, projected_phi
            marker_source = (
                str(item.get("_marker_source", "projected_nearest"))
                if nearest is not None
                else "projected_nearest"
            )
        else:
            refined_qz, refined_tth, refined_phi = refined
            marker_source = "local_peak_model_max"
        if (
            not np.isfinite(refined_tth)
            or not np.isfinite(refined_qz)
            or float(refined_tth) > float(ROD_PROFILE_MAX_TWO_THETA_DEG)
            or float(refined_qz) <= POSITIVE_QZ_MIN
        ):
            continue
        details = marker_acceptance_details_for_plotted_rod(row, rod, rejected_rod_keys)
        target_payload = details.get("target", {}) if bool(details.get("accepted", False)) else {}
        rows.append(
            {
                "sample_name": SAMPLE_NAME,
                "tilt_deg": float(bg["tilt_deg"]),
                "theta_initial_deg_used_for_q": float(bg["qr_overlay_config"].theta_initial_deg),
                "target_source_label": str(
                    target_payload.get("source_label", rod.get("source", ""))
                )
                if isinstance(target_payload, dict)
                else str(rod.get("source", "")),
                "m": int(rod["m"]),
                "qr": float(rod["qr"]),
                "qr_original": float(rod.get("qr_original", rod["qr"])),
                "qr_fit_count": int(rod.get("qr_fit_count", 0)),
                "qr_fit_sample_count": int(rod.get("qr_fit_sample_count", 0)),
                "qr_fit_method": str(rod.get("qr_fit_method", "original")),
                "branch": str(branch_name),
                "phi_window": str(branch_label),
                "hkl": str(item["label"]),
                "l": int(l_value),
                "fit_l": int(l_value),
                "display_l": float(l_value),
                "fit_two_theta_deg": float(item["params"][1]),
                "fit_phi_deg": float(item["params"][2]),
                "projected_two_theta_deg": projected_tth,
                "projected_phi_deg": projected_phi,
                "projected_qz_marker": qz_value,
                "refined_two_theta_deg": refined_tth,
                "refined_phi_deg": refined_phi,
                "qz_marker": refined_qz,
                "marker_source": marker_source,
            }
        )
    return sorted(rows, key=lambda row: row["qz_marker"])


def build_rod_qspace_config_from_base(
    base_config: object,
    *,
    distance_value: float,
    theta_initial_value: float,
    gamma_value: float | None = None,
    Gamma_value: float | None = None,
):
    gamma_out = (
        float(getattr(base_config, "gamma_deg", 0.0)) if gamma_value is None else float(gamma_value)
    )
    Gamma_out = (
        float(getattr(base_config, "Gamma_deg", 0.0)) if Gamma_value is None else float(Gamma_value)
    )
    return gui_qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=bool(getattr(base_config, "render_in_caked_space", True)),
        image_size=int(getattr(base_config, "image_size", 1)),
        display_rotate_k=int(getattr(base_config, "display_rotate_k", 0)),
        center_col=float(getattr(base_config, "center_col", center_col_px)),
        center_row=float(getattr(base_config, "center_row", center_row_px)),
        distance_cor_to_detector=float(distance_value),
        gamma_deg=gamma_out,
        Gamma_deg=Gamma_out,
        chi_deg=float(getattr(base_config, "chi_deg", 0.0)),
        psi_deg=float(getattr(base_config, "psi_deg", psi_deg)),
        psi_z_deg=float(getattr(base_config, "psi_z_deg", 0.0)),
        zs=float(getattr(base_config, "zs", 0.0)),
        zb=float(getattr(base_config, "zb", 0.0)),
        theta_initial_deg=float(theta_initial_value),
        cor_angle_deg=float(getattr(base_config, "cor_angle_deg", 0.0)),
        pixel_size_m=float(getattr(base_config, "pixel_size_m", PIXEL_SIZE_M)),
        wavelength=float(getattr(base_config, "wavelength", WAVELENGTH_M * 1.0e10)),
        n2=getattr(base_config, "n2", n2_value),
        phi_samples=int(getattr(base_config, "phi_samples", rod_phi_samples)),
        two_theta_limits=tuple(getattr(base_config, "two_theta_limits", (0.0, 90.0))),
    )


def intersection_geometry_from_qspace_config(config: object):
    if IntersectionGeometry is None:
        return None
    return IntersectionGeometry(
        image_size=int(config.image_size),
        center_col=float(config.center_col),
        center_row=float(config.center_row),
        distance_cor_to_detector=float(config.distance_cor_to_detector),
        gamma_deg=float(config.gamma_deg),
        Gamma_deg=float(config.Gamma_deg),
        chi_deg=float(config.chi_deg),
        psi_deg=float(config.psi_deg),
        psi_z_deg=float(config.psi_z_deg),
        zs=float(config.zs),
        zb=float(config.zb),
        theta_initial_deg=float(config.theta_initial_deg),
        cor_angle_deg=float(config.cor_angle_deg),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        pixel_size_m=float(config.pixel_size_m),
    )


def detector_qr_qz_for_config_point(
    config: object, col: float, row: float
) -> tuple[float, float, bool]:
    if detector_points_to_sample_qr_qz is None:
        return float("nan"), float("nan"), False
    geometry = intersection_geometry_from_qspace_config(config)
    if geometry is None:
        return float("nan"), float("nan"), False
    qr_arr, qz_arr, valid_arr = detector_points_to_sample_qr_qz(
        detector_col=np.asarray([float(col)], dtype=np.float64),
        detector_row=np.asarray([float(row)], dtype=np.float64),
        geometry=geometry,
        wavelength=float(config.wavelength),
        n2=config.n2,
        beam_x=config_float_attr(config, "beam_x", 0.0),
        beam_y=config_float_attr(config, "beam_y", 0.0),
        dtheta=config_float_attr(config, "dtheta", 0.0),
        dphi=config_float_attr(config, "dphi", 0.0),
    )
    qr = float(np.ravel(qr_arr)[0]) if np.size(qr_arr) else float("nan")
    qz = float(np.ravel(qz_arr)[0]) if np.size(qz_arr) else float("nan")
    valid = bool(np.ravel(valid_arr)[0]) if np.size(valid_arr) else False
    return qr, qz, valid


def normalized_q_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, dict):
        value = value.get("q_group_key", value.get("key"))
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception:
            return None
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def saved_q_target_lookup() -> tuple[
    dict[tuple[object, ...], dict[str, object]], dict[tuple[str, int, int], list[dict[str, object]]]
]:
    by_key: dict[tuple[object, ...], dict[str, object]] = {}
    by_identity: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for row in state.get("geometry", {}).get("q_group_rows", []) or []:
        if not isinstance(row, dict) or row.get("included", True) is False:
            continue
        key = normalized_q_group_key(row.get("key", row.get("q_group_key")))
        if key is None or len(key) < 4:
            continue
        try:
            m_value = int(key[2])
            gz_index = int(key[3])
            qr_value = float(row["qr"])
            qz_value = float(row["qz"])
        except Exception:
            continue
        if not (np.isfinite(qr_value) and np.isfinite(qz_value)):
            continue
        source_label = str(row.get("source_label", key[1] if len(key) > 1 else ""))
        target = {
            "q_group_key": key,
            "source_label": source_label,
            "m": int(m_value),
            "gz_index": int(gz_index),
            "qr": float(qr_value),
            "qz": float(qz_value),
        }
        by_key[key] = target
        by_identity.setdefault((source_label, int(m_value), int(gz_index)), []).append(target)
    return by_key, by_identity


def detector_rotation_target_for_item(
    item: dict[str, object],
    target_by_key: dict[tuple[object, ...], dict[str, object]],
    target_by_identity: dict[tuple[str, int, int], list[dict[str, object]]],
) -> tuple[dict[str, object] | None, str]:
    entry = item.get("entry") if isinstance(item.get("entry"), dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    label = str(item.get("label", entry.get("label", entry.get("_label", ""))))
    try:
        h, k, l_value = parse_hkl_label(label)
    except Exception:
        return None, "invalid_hkl"
    m_value = rod_m_from_hk(h, k)
    if "target_source_label" in item and "target_qr" in item and "target_qz" in item:
        try:
            target = {
                "q_group_key": normalized_q_group_key(
                    item.get("target_q_group_key", item.get("q_group_key"))
                ),
                "source_label": str(item["target_source_label"]),
                "m": int(item.get("m", m_value)),
                "gz_index": int(item.get("l", l_value)),
                "qr": float(item["target_qr"]),
                "qz": float(item["target_qz"]),
            }
        except Exception:
            return None, "invalid_precomputed_target"
        if int(target["m"]) != int(m_value) or int(target["gz_index"]) != int(l_value):
            return None, "precomputed_target_hkl_mismatch"
        return target, str(item.get("target_match_method", "precomputed_target"))
    q_group_key = normalized_q_group_key(entry.get("q_group_key", item.get("q_group_key")))
    if q_group_key is not None and q_group_key in target_by_key:
        target = target_by_key[q_group_key]
        if int(target["m"]) != int(m_value) or int(target["gz_index"]) != int(l_value):
            return None, "q_group_key_hkl_mismatch"
        return target, "q_group_key"
    source_label = str(
        entry.get(
            "source_label",
            q_group_key[1] if q_group_key is not None and len(q_group_key) > 1 else "primary",
        )
    )
    matches = target_by_identity.get((source_label, int(m_value), int(l_value)), [])
    if len(matches) == 1:
        return matches[0], "unique_source_hkl"
    return None, "missing_or_ambiguous_target"


MIXED_QR_SPREAD_TOL = 1.0e-4
ACTIVE_REJECTED_ROD_KEYS: set[tuple[object, ...]] = set()


def rejected_rod_key_set(keys: object) -> set[tuple[object, ...]]:
    rejected: set[tuple[object, ...]] = set()
    if keys is None:
        return rejected
    values = keys if isinstance(keys, (list, tuple, set)) else [keys]
    for raw_key in values:
        key = (
            raw_key.get("key", raw_key.get("rejection_key"))
            if isinstance(raw_key, dict)
            else raw_key
        )
        try:
            parts = tuple(key)  # type: ignore[arg-type]
        except Exception:
            continue
        if len(parts) < 2:
            continue
        try:
            source = str(parts[0])
            m_value = int(parts[1])
        except Exception:
            continue
        if len(parts) >= 3 and str(parts[2]) not in {"", "None", "nan"}:
            rejected.add((source, m_value, str(parts[2])))
        else:
            rejected.add((source, m_value))
    return rejected


def active_rejected_rod_keys() -> set[tuple[object, ...]]:
    return set(ACTIVE_REJECTED_ROD_KEYS)


def rod_identity_key(rod_entry: dict[str, object]) -> tuple[str, int]:
    return (str(rod_entry.get("source", "")), int(rod_entry["m"]))


def point_identity_key(row: dict[str, object]) -> tuple[str, int]:
    return (str(row.get("target_source_label", row.get("source", ""))), int(row.get("m", -999999)))


def rod_rejected_for_plot(rod_entry: dict[str, object], rejected_rod_keys: object) -> bool:
    rejected = rejected_rod_key_set(rejected_rod_keys)
    source, m_value = rod_identity_key(rod_entry)
    return (source, m_value) in rejected or any(
        len(key) >= 3 and str(key[0]) == source and int(key[1]) == int(m_value) for key in rejected
    )


def marker_item_branch(row: dict[str, object]) -> str:
    entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    return str(row.get("branch", entry.get("_branch", "")))


def marker_target_for_acceptance(row: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    target_by_key, target_by_identity = saved_q_target_lookup()
    return detector_rotation_target_for_item(row, target_by_key, target_by_identity)


def marker_acceptance_details_for_plotted_rod(
    row: dict[str, object],
    rod_entry: dict[str, object],
    rejected_rod_keys: object,
) -> dict[str, object]:
    entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    label = str(row.get("label", entry.get("label", entry.get("_label", ""))))
    try:
        h, k, l_value = parse_hkl_label(label)
    except Exception:
        return {"accepted": False, "reason": "invalid_hkl", "label": label}
    if int(l_value) <= 0:
        return {"accepted": False, "reason": "non_positive_l", "label": label}
    m_value = rod_m_from_hk(h, k)
    if int(m_value) <= 0:
        return {
            "accepted": False,
            "reason": "specular_not_used_for_qr_rod_fit",
            "label": label,
            "m": int(m_value),
        }
    if int(m_value) != int(rod_entry["m"]):
        return {"accepted": False, "reason": "rod_m_mismatch", "label": label, "m": int(m_value)}
    branch_value = marker_item_branch(row)
    required_branch = row.get("_required_branch")
    if required_branch is not None and str(branch_value) != str(required_branch):
        return {
            "accepted": False,
            "reason": "branch_mismatch_for_plotted_rod",
            "label": label,
            "m": int(m_value),
            "branch": branch_value,
        }
    target, match_method = marker_target_for_acceptance(row)
    if target is None:
        return {
            "accepted": False,
            "reason": match_method,
            "label": label,
            "m": int(m_value),
            "branch": branch_value,
        }
    target_source = str(target.get("source_label", ""))
    plotted_source = str(rod_entry.get("source", ""))
    if plotted_source and target_source and target_source != plotted_source:
        return {
            "accepted": False,
            "reason": "source_mismatch_for_plotted_rod",
            "label": label,
            "m": int(m_value),
            "branch": branch_value,
            "target_source_label": target_source,
            "plotted_source_label": plotted_source,
        }
    rejected = rejected_rod_key_set(rejected_rod_keys)
    rejection_key = (target_source, int(m_value))
    branch_rejection_key = (target_source, int(m_value), str(branch_value))
    if (
        rejection_key in rejected
        or branch_rejection_key in rejected
        or rod_rejected_for_plot({"source": target_source, "m": int(m_value)}, rejected)
    ):
        return {
            "accepted": False,
            "reason": "mixed_qr_targets_for_plotted_rod",
            "label": label,
            "m": int(m_value),
            "branch": branch_value,
            "target_source_label": target_source,
            "rejection_key": branch_rejection_key
            if branch_rejection_key in rejected
            else rejection_key,
        }
    return {
        "accepted": True,
        "reason": "accepted",
        "label": label,
        "h": int(h),
        "k": int(k),
        "l": int(l_value),
        "m": int(m_value),
        "branch": branch_value,
        "target": target,
        "target_match_method": match_method,
        "target_source_label": target_source,
        "rejection_key": rejection_key,
    }


def accept_marker_for_plotted_rod(
    row: dict[str, object], rod_entry: dict[str, object], rejected_rod_keys: object
) -> bool:
    return bool(
        marker_acceptance_details_for_plotted_rod(row, rod_entry, rejected_rod_keys).get(
            "accepted", False
        )
    )


def rod_entry_for_target(
    rods: list[dict[str, object]], target: dict[str, object]
) -> dict[str, object] | None:
    target_key = (str(target.get("source_label", "")), int(target.get("m", -999999)))
    for rod in rods:
        if rod_identity_key(rod) == target_key:
            return rod
    return None


def detector_rotation_qr_spread_table(
    fit_points: list[dict[str, object]],
    *,
    label: str,
    group_columns: tuple[str, ...] = ("target_source_label", "m"),
) -> pd.DataFrame:
    if not fit_points:
        return pd.DataFrame()
    fit_debug_targets = pd.DataFrame(fit_points)
    if fit_debug_targets.empty or "target_qr" not in fit_debug_targets:
        return pd.DataFrame()
    available_columns = [column for column in group_columns if column in fit_debug_targets.columns]
    if not available_columns:
        return pd.DataFrame()
    qr_spread = (
        fit_debug_targets.groupby(available_columns, dropna=False)["target_qr"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    qr_spread["spread"] = qr_spread["max"] - qr_spread["min"]
    qr_spread["selection"] = str(label)
    return qr_spread.sort_values("spread", ascending=False)


def mixed_qr_rejection_keys_after_source_filter(
    fit_points: list[dict[str, object]],
) -> tuple[set[tuple[object, ...]], list[dict[str, object]]]:
    spread_table = detector_rotation_qr_spread_table(
        fit_points, label="source_filtered", group_columns=("target_source_label", "m")
    )
    rejected: set[tuple[object, ...]] = set()
    reasons: list[dict[str, object]] = []
    if spread_table.empty:
        return rejected, reasons
    for row in spread_table.to_dict("records"):
        spread = float(row.get("spread", np.nan))
        if not np.isfinite(spread) or spread <= MIXED_QR_SPREAD_TOL:
            continue
        source = str(row.get("target_source_label", ""))
        m_value = int(row.get("m", -999999))
        group_points = [
            point
            for point in fit_points
            if str(point.get("target_source_label", "")) == source
            and int(point.get("m", -999999)) == m_value
        ]
        branch_table = detector_rotation_qr_spread_table(
            group_points, label="branch_check", group_columns=("target_source_label", "m", "branch")
        )
        branch_specific = bool(
            not branch_table.empty
            and int(branch_table.shape[0]) > 1
            and np.all(np.asarray(branch_table["spread"], dtype=np.float64) <= MIXED_QR_SPREAD_TOL)
        )
        if branch_specific:
            for branch_row in branch_table.to_dict("records"):
                branch = str(branch_row.get("branch", ""))
                key = (source, m_value, branch)
                rejected.add(key)
                reasons.append(
                    {
                        "key": list(key),
                        "reason": "mixed_qr_targets_for_plotted_rod",
                        "target_source_label": source,
                        "m": m_value,
                        "branch": branch,
                        "count": int(branch_row.get("count", 0)),
                        "min": float(branch_row.get("min", np.nan)),
                        "max": float(branch_row.get("max", np.nan)),
                        "spread": spread,
                        "branch_spread": float(branch_row.get("spread", np.nan)),
                    }
                )
        else:
            key = (source, m_value)
            rejected.add(key)
            reasons.append(
                {
                    "key": list(key),
                    "reason": "mixed_qr_targets_for_plotted_rod",
                    "target_source_label": source,
                    "m": m_value,
                    "count": int(row.get("count", 0)),
                    "min": float(row.get("min", np.nan)),
                    "max": float(row.get("max", np.nan)),
                    "spread": spread,
                }
            )
    return rejected, reasons


def collect_detector_rotation_fit_points(
    bg: dict[str, object],
    rods: list[dict[str, object]],
    rejected_rod_keys: object | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rejected_rod_keys = rejected_rod_key_set(rejected_rod_keys)
    points: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for item in bg.get("fit_results", []) or []:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry") if isinstance(item.get("entry"), dict) else {}
        entry = entry if isinstance(entry, dict) else {}
        label = str(item.get("label", entry.get("label", entry.get("_label", ""))))
        try:
            h, k, l_value = parse_hkl_label(label)
        except Exception:
            skipped.append({"label": label, "reason": "invalid_hkl"})
            continue
        if int(l_value) <= 0:
            skipped.append({"label": label, "reason": "non_positive_l"})
            continue
        m_value = rod_m_from_hk(h, k)
        if int(m_value) <= 0:
            skipped.append({"label": label, "reason": "specular_not_used_for_qr_rod_fit"})
            continue
        col = as_float(item.get("fit_detector_col"))
        row = as_float(item.get("fit_detector_row"))
        if not (np.isfinite(col) and np.isfinite(row)):
            skipped.append({"label": label, "reason": "missing_fit_detector_xy"})
            continue
        target, match_method = marker_target_for_acceptance(item)
        if target is None:
            skipped.append({"label": label, "reason": match_method})
            continue
        rod = rod_entry_for_target(rods, target)
        if rod is None:
            skipped.append(
                {
                    "label": label,
                    "m": int(m_value),
                    "reason": "source_mismatch_for_plotted_rod",
                    "target_source_label": str(target.get("source_label", "")),
                }
            )
            continue
        if not accept_marker_for_plotted_rod(item, rod, rejected_rod_keys):
            details = marker_acceptance_details_for_plotted_rod(item, rod, rejected_rod_keys)
            skipped.append(
                {
                    "label": label,
                    "m": int(m_value),
                    "reason": str(details.get("reason", "marker_rejected_for_plotted_rod")),
                }
            )
            continue
        details = marker_acceptance_details_for_plotted_rod(item, rod, rejected_rod_keys)
        target = details["target"]
        points.append(
            {
                "label": label,
                "h": int(h),
                "k": int(k),
                "l": int(l_value),
                "m": int(m_value),
                "branch": str(item.get("branch", entry.get("_branch", ""))),
                "branch_id": str(entry.get("branch_id", "")),
                "source_branch_index": entry.get("source_branch_index", None),
                "q_group_key": normalized_q_group_key(
                    entry.get("q_group_key", item.get("q_group_key"))
                ),
                "target_match_method": str(details.get("target_match_method", match_method)),
                "target_q_group_key": target["q_group_key"],
                "target_source_label": target["source_label"],
                "target_qr": float(target["qr"]),
                "target_qz": float(target["qz"]),
                "fit_detector_col": float(col),
                "fit_detector_row": float(row),
            }
        )
    return points, skipped


def detector_rotation_anchor_summary(
    fit_points: object, active_mask: object | None = None
) -> dict[str, int]:
    points = [dict(point) for point in fit_points or [] if isinstance(point, dict)]
    if active_mask is None:
        selected = points
    else:
        mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        selected = [point for idx, point in enumerate(points) if idx < mask.size and bool(mask[idx])]
    m_groups: set[tuple[str, int]] = set()
    for point in selected:
        try:
            m_groups.add((str(point.get("target_source_label", "")), int(point["m"])))
        except Exception:
            continue
    return {
        "anchor_count": int(len(selected)),
        "anchor_m_group_count": int(len(m_groups)),
    }


def detector_rotation_fit_has_anchor_coverage(
    fit_points: object,
    active_mask: object | None = None,
    *,
    min_anchors: int = 4,
    min_m_groups: int = 2,
) -> bool:
    summary = detector_rotation_anchor_summary(fit_points, active_mask)
    return bool(
        int(summary["anchor_count"]) >= int(min_anchors)
        and int(summary["anchor_m_group_count"]) >= int(min_m_groups)
    )


def annotate_rods_with_detector_rotation_fit(
    rods: list[dict[str, object]],
    fit_points: list[dict[str, object]],
    active_mask: np.ndarray | None,
) -> list[dict[str, object]]:
    active = (
        np.asarray(active_mask, dtype=bool)
        if active_mask is not None
        else np.zeros(len(fit_points), dtype=bool)
    )
    fitted_rods: list[dict[str, object]] = []
    for rod in rods:
        rod_key = rod_identity_key(rod)
        sample_indices = [
            idx for idx, point in enumerate(fit_points) if point_identity_key(point) == rod_key
        ]
        fit_count = int(sum(1 for idx in sample_indices if idx < active.size and bool(active[idx])))
        fitted = dict(rod)
        fitted["qr_original"] = float(rod["qr"])
        fitted["qr"] = float(rod["qr"])
        fitted["qr_fit_count"] = int(fit_count)
        fitted["qr_fit_sample_count"] = int(len(sample_indices))
        fitted["qr_fit_method"] = (
            "saved_q_group_rows_detector_rotation_fit"
            if sample_indices
            else "saved_q_group_rows_no_fit_points"
        )
        fitted_rods.append(fitted)
    return fitted_rods


def fit_rod_qspace_calibration(
    bg: dict[str, object], rods: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    original_config = bg.get("qr_overlay_config")
    if (
        original_config is None
        or detector_points_to_sample_qr_qz is None
        or IntersectionGeometry is None
    ):
        return annotate_rods_with_detector_rotation_fit(rods, [], None), {
            "success": False,
            "message": "IntersectionGeometry inverse helper unavailable",
            "anchor_count": 0,
            "active_anchor_count": 0,
            "distance_original_m": float(distance_m),
            "distance_fitted_m": float(distance_m),
            "theta_original_deg": float(ROD_PROFILE_TILT_DEG),
            "theta_fitted_deg": float(ROD_PROFILE_TILT_DEG),
            "gamma_original_deg": 0.0,
            "gamma_fitted_deg": 0.0,
            "Gamma_original_deg": 0.0,
            "Gamma_fitted_deg": 0.0,
            "rms_before_qspace": float("nan"),
            "rms_after_qspace": float("nan"),
            "rms_before_qr": float("nan"),
            "rms_after_qr": float("nan"),
            "rms_before_qz": float("nan"),
            "rms_after_qz": float("nan"),
        }

    fit_points, skipped_points = collect_detector_rotation_fit_points(
        bg, rods, rejected_rod_keys=set()
    )
    qr_spread_all = detector_rotation_qr_spread_table(
        fit_points, label="source_filtered", group_columns=("target_source_label", "m")
    )
    if not qr_spread_all.empty:
        display(qr_spread_all)
    rejected_rod_keys, rejected_rod_reasons = mixed_qr_rejection_keys_after_source_filter(
        fit_points
    )
    if rejected_rod_reasons:
        display(pd.DataFrame(rejected_rod_reasons))
    if rejected_rod_keys:
        retained_points: list[dict[str, object]] = []
        for point in fit_points:
            rod = rod_entry_for_target(
                rods,
                {
                    "source_label": point.get("target_source_label", ""),
                    "m": point.get("m", -999999),
                },
            )
            if rod is None or not accept_marker_for_plotted_rod(point, rod, rejected_rod_keys):
                skipped_points.append(
                    {
                        "label": point.get("label", ""),
                        "m": int(point.get("m", -999999)),
                        "branch": str(point.get("branch", "")),
                        "reason": "mixed_qr_targets_for_plotted_rod",
                        "target_source_label": str(point.get("target_source_label", "")),
                    }
                )
                continue
            retained_points.append(point)
        fit_points = retained_points
    qr_spread_filtered = detector_rotation_qr_spread_table(
        fit_points, label="source_filtered_non_rejected", group_columns=("target_source_label", "m")
    )
    if not qr_spread_filtered.empty:
        display(qr_spread_filtered)
    mixed_qr_targets_after_source_filter = bool(rejected_rod_keys)
    original_distance = float(getattr(original_config, "distance_cor_to_detector", distance_m))
    original_theta = float(getattr(original_config, "theta_initial_deg", ROD_PROFILE_TILT_DEG))
    original_gamma = float(getattr(original_config, "gamma_deg", 0.0))
    original_Gamma = float(getattr(original_config, "Gamma_deg", 0.0))
    original_qr_by_m = {int(rod["m"]): float(rod["qr"]) for rod in rods}
    anchor_summary = detector_rotation_anchor_summary(fit_points)

    base_payload = {
        "anchor_count": int(len(fit_points)),
        "anchor_m_group_count": int(anchor_summary["anchor_m_group_count"]),
        "skipped_anchor_count": int(len(skipped_points)),
        "distance_original_m": original_distance,
        "distance_fitted_m": original_distance,
        "theta_original_deg": original_theta,
        "theta_fitted_deg": original_theta,
        "theta_original_by_bg": {int(bg.get("background_index", -1)): original_theta},
        "theta_fitted_by_bg": {int(bg.get("background_index", -1)): original_theta},
        "gamma_original_deg": original_gamma,
        "gamma_fitted_deg": original_gamma,
        "Gamma_original_deg": original_Gamma,
        "Gamma_fitted_deg": original_Gamma,
        "qr_original_by_m": original_qr_by_m,
        "qr_fitted_by_m": dict(original_qr_by_m),
        "rms_before_qspace": float("nan"),
        "rms_after_qspace": float("nan"),
        "rms_before_qr": float("nan"),
        "rms_after_qr": float("nan"),
        "rms_before_qz": float("nan"),
        "rms_after_qz": float("nan"),
        "median_abs_dqr_before": float("nan"),
        "median_abs_dqr_after": float("nan"),
        "p90_abs_dqr_before": float("nan"),
        "p90_abs_dqr_after": float("nan"),
        "mixed_qr_targets_after_source_filter": mixed_qr_targets_after_source_filter,
        "rejected_rod_keys": [
            list(key)
            for key in sorted(rejected_rod_keys, key=lambda item: tuple(str(part) for part in item))
        ],
        "rejected_rod_reasons": [dict(row) for row in rejected_rod_reasons],
        "rotation_bound_hit": False,
    }
    if not fit_points:
        payload = dict(base_payload)
        payload.update(
            {
                "success": False,
                "message": "no non-specular detector points matched plotted rod sources",
                "active_anchor_count": 0,
            }
        )
        return annotate_rods_with_detector_rotation_fit(rods, fit_points, None), payload
    if not detector_rotation_fit_has_anchor_coverage(
        fit_points,
        min_anchors=DETECTOR_ROTATION_MIN_ANCHORS,
        min_m_groups=DETECTOR_ROTATION_MIN_M_GROUPS,
    ):
        payload = dict(base_payload)
        payload.update(
            {
                "success": False,
                "message": "insufficient detector-rotation anchor coverage",
                "active_anchor_count": int(anchor_summary["anchor_count"]),
                "active_anchor_m_group_count": int(anchor_summary["anchor_m_group_count"]),
                "fit_points": [dict(point) for point in fit_points],
                "skipped_fit_points": [dict(point) for point in skipped_points],
            }
        )
        fitted_rods = annotate_rods_with_detector_rotation_fit(rods, fit_points, None)
        for rod in fitted_rods:
            if int(rod.get("qr_fit_sample_count", 0)) > 0:
                rod["qr_fit_method"] = "saved_q_group_rows_insufficient_detector_rotation_anchors"
        print(
            "skipped detector-rotation Qr overlay fit: insufficient detector-rotation anchor coverage "
            f"points={int(anchor_summary['anchor_count'])} "
            f"m_groups={int(anchor_summary['anchor_m_group_count'])}"
        )
        return fitted_rods, payload

    rod_counts = Counter(point_identity_key(point) for point in fit_points)
    for point in fit_points:
        point["fit_weight"] = 1.0 / np.sqrt(max(1, int(rod_counts[point_identity_key(point)])))
    point_weight = np.asarray(
        [point.get("fit_weight", 1.0) for point in fit_points], dtype=np.float64
    )
    fit_col = np.asarray([point["fit_detector_col"] for point in fit_points], dtype=np.float64)
    fit_row = np.asarray([point["fit_detector_row"] for point in fit_points], dtype=np.float64)
    target_qr = np.asarray([point["target_qr"] for point in fit_points], dtype=np.float64)
    target_qz = np.asarray([point["target_qz"] for point in fit_points], dtype=np.float64)

    FIT_QZ_WEIGHT = 0.0
    FIT_QR_SIGMA = 0.005
    FIT_QZ_SIGMA = 0.02
    invalid_q_penalty = 25.0
    rot_prior_sigma_deg = 1.0

    def config_from_params(params: np.ndarray):
        theta_i_deg, detector_distance_m, gamma_deg, Gamma_deg = map(float, params)
        return build_rod_qspace_config_from_base(
            original_config,
            distance_value=detector_distance_m,
            theta_initial_value=theta_i_deg,
            gamma_value=gamma_deg,
            Gamma_value=Gamma_deg,
        )

    def qspace_delta_payload(
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = config_from_params(params)
        geometry = intersection_geometry_from_qspace_config(config)
        if geometry is None:
            nan = np.full(fit_col.shape, np.nan, dtype=np.float64)
            return nan, nan.copy(), np.zeros(fit_col.shape, dtype=bool), nan.copy(), nan.copy()
        try:
            qr_obs, qz_obs, valid = detector_points_to_sample_qr_qz(
                detector_col=fit_col,
                detector_row=fit_row,
                geometry=geometry,
                wavelength=float(config.wavelength),
                n2=config.n2,
                beam_x=config_float_attr(config, "beam_x", 0.0),
                beam_y=config_float_attr(config, "beam_y", 0.0),
                dtheta=config_float_attr(config, "dtheta", 0.0),
                dphi=config_float_attr(config, "dphi", 0.0),
            )
        except Exception:
            nan = np.full(fit_col.shape, np.nan, dtype=np.float64)
            return nan, nan.copy(), np.zeros(fit_col.shape, dtype=bool), nan.copy(), nan.copy()
        qr_obs = np.asarray(qr_obs, dtype=np.float64)
        qz_obs = np.asarray(qz_obs, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        good = valid & np.isfinite(qr_obs) & np.isfinite(target_qr)
        return qr_obs, qz_obs, good, qr_obs - target_qr, qz_obs - target_qz

    def qspace_rms_components(params: np.ndarray) -> dict[str, object]:
        _qr_obs, _qz_obs, good, dqr, dqz = qspace_delta_payload(params)
        if not np.any(good):
            return {
                "rms_qspace": float("nan"),
                "rms_qr": float("nan"),
                "rms_qz": float("nan"),
                "rms_objective": float("nan"),
                "median_abs_dqr": float("nan"),
                "p90_abs_dqr": float("nan"),
                "active_count": 0,
                "good": good,
            }
        dqr_good = np.asarray(dqr[good], dtype=np.float64)
        dqz_good = np.asarray(dqz[good], dtype=np.float64)
        qspace_residual = np.square(dqr_good) + np.square(dqz_good)
        finite_qspace = qspace_residual[np.isfinite(qspace_residual)]
        finite_dqz = dqz_good[np.isfinite(dqz_good)]
        objective_parts = [point_weight[good] * dqr_good / FIT_QR_SIGMA]
        if FIT_QZ_WEIGHT > 0.0:
            qz_objective_good = good & np.isfinite(dqz)
            if np.any(qz_objective_good):
                objective_parts.append(
                    point_weight[qz_objective_good]
                    * FIT_QZ_WEIGHT
                    * dqz[qz_objective_good]
                    / FIT_QZ_SIGMA
                )
        objective = (
            np.concatenate(objective_parts) if objective_parts else np.asarray([], dtype=np.float64)
        )
        abs_dqr = np.abs(dqr_good)
        return {
            "rms_qspace": float(np.sqrt(np.nanmean(finite_qspace)))
            if finite_qspace.size
            else float("nan"),
            "rms_qr": float(np.sqrt(np.nanmean(np.square(dqr_good)))),
            "rms_qz": float(np.sqrt(np.nanmean(np.square(finite_dqz))))
            if finite_dqz.size
            else float("nan"),
            "rms_objective": float(np.sqrt(np.nanmean(np.square(objective))))
            if objective.size
            else float("nan"),
            "median_abs_dqr": float(np.nanmedian(abs_dqr)),
            "p90_abs_dqr": float(np.nanpercentile(abs_dqr, 90.0)),
            "active_count": int(np.count_nonzero(good)),
            "good": good,
        }

    def residual(params: np.ndarray) -> np.ndarray:
        _qr_obs, _qz_obs, good, dqr, dqz = qspace_delta_payload(params)
        parts: list[np.ndarray] = []
        if np.any(good):
            parts.append(point_weight[good] * dqr[good] / FIT_QR_SIGMA)
            if FIT_QZ_WEIGHT > 0.0:
                qz_objective_good = good & np.isfinite(dqz)
                if np.any(qz_objective_good):
                    parts.append(
                        point_weight[qz_objective_good]
                        * FIT_QZ_WEIGHT
                        * dqz[qz_objective_good]
                        / FIT_QZ_SIGMA
                    )
        bad_count = int(good.size - np.count_nonzero(good))
        if bad_count:
            parts.append(np.full(bad_count, invalid_q_penalty, dtype=np.float64))
        gamma_deg = float(params[2])
        Gamma_deg = float(params[3])
        parts.append(
            np.asarray(
                [
                    (gamma_deg - original_gamma) / rot_prior_sigma_deg,
                    (Gamma_deg - original_Gamma) / rot_prior_sigma_deg,
                ],
                dtype=np.float64,
            )
        )
        return np.concatenate(parts)

    x0 = np.asarray(
        [original_theta, original_distance, original_gamma, original_Gamma], dtype=np.float64
    )
    lower = np.asarray(
        [
            original_theta - 2.0,
            max(0.90 * original_distance, 1.0e-9),
            original_gamma - 3.0,
            original_Gamma - 3.0,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            original_theta + 2.0,
            1.10 * original_distance,
            original_gamma + 3.0,
            original_Gamma + 3.0,
        ],
        dtype=np.float64,
    )
    before = qspace_rms_components(x0)
    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=400,
        x_scale="jac",
    )
    final_params = np.asarray(result.x if np.all(np.isfinite(result.x)) else x0, dtype=np.float64)
    after = qspace_rms_components(final_params)
    before_count = int(before["active_count"])
    after_count = int(after["active_count"])
    after_good = np.asarray(after["good"], dtype=bool)
    active_anchor_summary = detector_rotation_anchor_summary(fit_points, after_good)
    if not detector_rotation_fit_has_anchor_coverage(
        fit_points,
        after_good,
        min_anchors=DETECTOR_ROTATION_MIN_ANCHORS,
        min_m_groups=DETECTOR_ROTATION_MIN_M_GROUPS,
    ):
        payload = dict(base_payload)
        payload.update(
            {
                "success": False,
                "message": "insufficient detector-rotation active anchor coverage",
                "active_anchor_count": int(active_anchor_summary["anchor_count"]),
                "active_anchor_m_group_count": int(active_anchor_summary["anchor_m_group_count"]),
                "active_anchor_count_before": int(before_count),
                "fit_points": [dict(point) for point in fit_points],
                "skipped_fit_points": [dict(point) for point in skipped_points],
                "rms_before_qr": float(before["rms_qr"]),
                "rms_after_qr": float(after["rms_qr"]),
                "rms_before_qz": float(before["rms_qz"]),
                "rms_after_qz": float(after["rms_qz"]),
            }
        )
        fitted_rods = annotate_rods_with_detector_rotation_fit(rods, fit_points, after_good)
        for rod in fitted_rods:
            if int(rod.get("qr_fit_sample_count", 0)) > 0:
                rod["qr_fit_method"] = "saved_q_group_rows_insufficient_detector_rotation_anchors"
        print(
            "skipped detector-rotation Qr overlay fit: insufficient detector-rotation active anchor coverage "
            f"points={int(active_anchor_summary['anchor_count'])} "
            f"m_groups={int(active_anchor_summary['anchor_m_group_count'])}"
        )
        return fitted_rods, payload
    theta_fit, distance_fit, gamma_fit, Gamma_fit = map(float, final_params)
    rotation_bound_hit = bool(
        abs(gamma_fit - float(lower[2])) < 1.0e-6
        or abs(gamma_fit - float(upper[2])) < 1.0e-6
        or abs(Gamma_fit - float(lower[3])) < 1.0e-6
        or abs(Gamma_fit - float(upper[3])) < 1.0e-6
    )
    fitted_rods = annotate_rods_with_detector_rotation_fit(rods, fit_points, after_good)
    result_message = str(result.message)
    if mixed_qr_targets_after_source_filter:
        result_message = f"{result_message}; mixed_qr_targets_after_source_filter; mixed_qr_targets_for_plotted_rod"
    payload = dict(base_payload)
    payload.update(
        {
            "success": bool(result.success),
            "message": result_message,
            "active_anchor_count": int(after_count),
            "active_anchor_m_group_count": int(active_anchor_summary["anchor_m_group_count"]),
            "active_anchor_count_before": int(before_count),
            "distance_fitted_m": distance_fit,
            "theta_fitted_deg": theta_fit,
            "theta_fitted_by_bg": {int(bg.get("background_index", -1)): theta_fit},
            "gamma_fitted_deg": gamma_fit,
            "Gamma_fitted_deg": Gamma_fit,
            "rms_before_qspace": float(before["rms_qspace"]),
            "rms_after_qspace": float(after["rms_qspace"]),
            "rms_before_qspace_scaled": float(before["rms_objective"]),
            "rms_after_qspace_scaled": float(after["rms_objective"]),
            "rms_before_qr": float(before["rms_qr"]),
            "rms_after_qr": float(after["rms_qr"]),
            "rms_before_qz": float(before["rms_qz"]),
            "rms_after_qz": float(after["rms_qz"]),
            "median_abs_dqr_before": float(before["median_abs_dqr"]),
            "median_abs_dqr_after": float(after["median_abs_dqr"]),
            "p90_abs_dqr_before": float(before["p90_abs_dqr"]),
            "p90_abs_dqr_after": float(after["p90_abs_dqr"]),
            "fit_qr_sigma": float(FIT_QR_SIGMA),
            "fit_qz_sigma": float(FIT_QZ_SIGMA),
            "fit_qz_weight": float(FIT_QZ_WEIGHT),
            "fit_points": [dict(point) for point in fit_points],
            "skipped_fit_points": [dict(point) for point in skipped_points],
            "rejected_rod_keys": [
                list(key)
                for key in sorted(
                    rejected_rod_keys, key=lambda item: tuple(str(part) for part in item)
                )
            ],
            "rejected_rod_reasons": [dict(row) for row in rejected_rod_reasons],
            "cost": float(result.cost),
            "rotation_bound_hit": rotation_bound_hit,
        }
    )
    print(
        "detector-rotation Qr overlay fit: "
        f"points={len(fit_points)} active={after_count} skipped={len(skipped_points)} "
        f"rms_qr={float(before['rms_qr']):.5g}->{float(after['rms_qr']):.5g} "
        f"p90_abs_dqr={float(before['p90_abs_dqr']):.5g}->{float(after['p90_abs_dqr']):.5g} "
        f"rms_qz_diag={float(before['rms_qz']):.5g}->{float(after['rms_qz']):.5g} "
        f"theta={original_theta:.6g}->{theta_fit:.6g} "
        f"distance={original_distance:.8g}->{distance_fit:.8g} "
        f"gamma={original_gamma:.6g}->{gamma_fit:.6g} "
        f"Gamma={original_Gamma:.6g}->{Gamma_fit:.6g}"
    )
    if rotation_bound_hit:
        print(
            "detector-rotation Q fit warning: gamma/Gamma hit a bound; check sign convention before widening bounds"
        )
    return fitted_rods, payload


def apply_rod_qspace_calibration(
    backgrounds: list[dict[str, object]], calibration: dict[str, object]
) -> None:
    distance_value = float(calibration.get("distance_fitted_m", distance_m))
    theta_value = float(calibration.get("theta_fitted_deg", ROD_PROFILE_TILT_DEG))
    gamma_value = float(
        calibration.get("gamma_fitted_deg", calibration.get("gamma_original_deg", 0.0))
    )
    Gamma_value = float(
        calibration.get("Gamma_fitted_deg", calibration.get("Gamma_original_deg", 0.0))
    )
    for bg in backgrounds:
        original_config = bg.get("qr_overlay_config")
        if original_config is None:
            continue
        bg.setdefault("qr_overlay_config_saved_state", original_config)
        bg["qr_overlay_config"] = build_rod_qspace_config_from_base(
            original_config,
            distance_value=distance_value,
            theta_initial_value=theta_value,
            gamma_value=gamma_value,
            Gamma_value=Gamma_value,
        )
        bg["rod_qspace_calibration"] = dict(calibration)


matching_profile_bgs = [
    bg
    for bg in ordered_backgrounds
    if abs(float(bg["tilt_deg"]) - float(ROD_PROFILE_TILT_DEG)) < 1.0e-6
]
if not matching_profile_bgs:
    raise RuntimeError(
        f"no background found for Qr rod incident angle {ROD_PROFILE_TILT_LABEL} deg"
    )
profile_bg = matching_profile_bgs[0]
ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG = float(ROD_PROFILE_MAX_TWO_THETA_DEG)
qr_rod_pre_editor_stage = pre_editor_cache_get_stage(
    pre_editor_cache, "qr_rod_pre_editor", PRE_EDITOR_QR_ROD_STAGE_SIGNATURE
)
qr_rod_pre_editor_cache_hit = qr_rod_pre_editor_stage_is_valid(qr_rod_pre_editor_stage)
rod_candidate_m_values = fit_result_m_values_for_rod_entries(
    ordered_backgrounds, tilt_deg=ROD_PROFILE_TILT_DEG
)
if qr_rod_pre_editor_cache_hit:
    ROD_PROFILE_MAX_TWO_THETA_DEG = float(qr_rod_pre_editor_stage["rod_profile_max_two_theta_deg"])
    rod_entries = [dict(row) for row in qr_rod_pre_editor_stage["rod_entries"]]
    rod_qspace_calibration = dict(qr_rod_pre_editor_stage["rod_qspace_calibration"])
    ACTIVE_REJECTED_ROD_KEYS = rejected_rod_key_set(
        qr_rod_pre_editor_stage.get(
            "active_rejected_rod_keys", rod_qspace_calibration.get("rejected_rod_keys", [])
        )
    )
    apply_rod_qspace_calibration([profile_bg], rod_qspace_calibration)
    print(f"reused pre-editor Qr-rod profile cache={PRE_EDITOR_CACHE_PATH}")
else:
    ROD_PROFILE_MAX_TWO_THETA_DEG = rod_profile_two_theta_limit_for_background(
        profile_bg, ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG
    )
    if ROD_PROFILE_MAX_TWO_THETA_DEG > ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG + 1.0e-9:
        print(
            f"expanded Qr rod 2theta limit from {ROD_PROFILE_CONFIGURED_MAX_TWO_THETA_DEG:.1f} to {ROD_PROFILE_MAX_TWO_THETA_DEG:.1f} deg to include selected-state peaks"
        )
    rod_entries = build_profile_rod_entries(ROD_PROFILE_TILT_DEG)
    if not rod_entries:
        raise RuntimeError(
            f"no non-specular Qr rods were available for {ROD_PROFILE_TILT_LABEL} deg branch plotting"
        )
    rod_entries, rod_qspace_calibration = fit_rod_qspace_calibration(profile_bg, rod_entries)
    ACTIVE_REJECTED_ROD_KEYS = rejected_rod_key_set(
        rod_qspace_calibration.get("rejected_rod_keys", [])
    )
    if ACTIVE_REJECTED_ROD_KEYS:
        rejected_for_plot = [
            rod for rod in rod_entries if rod_rejected_for_plot(rod, ACTIVE_REJECTED_ROD_KEYS)
        ]
        if rejected_for_plot:
            print(
                "rejected mixed-Qr plotted rods: "
                + ", ".join(f"{rod_identity_key(rod)}" for rod in rejected_for_plot)
            )
        rod_entries = [
            rod for rod in rod_entries if not rod_rejected_for_plot(rod, ACTIVE_REJECTED_ROD_KEYS)
        ]
        if not rod_entries:
            raise RuntimeError("all non-specular Qr rods rejected by mixed target Qr spread")
    apply_rod_qspace_calibration([profile_bg], rod_qspace_calibration)

rod_reference_summary = rod_reference_source_summary(
    rod_entries,
    candidate_m_values=rod_candidate_m_values,
    allow_generated=ALLOW_GENERATED_ROD_REFERENCES,
)
print(
    f"rod references: saved={int(rod_reference_summary['saved'])} "
    f"generated={int(rod_reference_summary['generated'])} "
    f"skipped_generated={int(rod_reference_summary['skipped_generated'])} "
    f"allow_generated={bool(rod_reference_summary['allow_generated'])}"
)


def display_detector_rotation_fit_debug(
    bg: dict[str, object],
    calibration: dict[str, object],
    rod_line_lookup: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> None:
    fit_points = list(calibration.get("fit_points", []) or [])
    if not fit_points:
        return
    fit_debug = pd.DataFrame(fit_points).copy()
    config = bg.get("qr_overlay_config")
    geometry = intersection_geometry_from_qspace_config(config) if config is not None else None
    if geometry is None or detector_points_to_sample_qr_qz is None:
        return
    try:
        qr_obs, qz_obs, valid = detector_points_to_sample_qr_qz(
            detector_col=np.asarray(fit_debug["fit_detector_col"], dtype=np.float64),
            detector_row=np.asarray(fit_debug["fit_detector_row"], dtype=np.float64),
            geometry=geometry,
            wavelength=float(config.wavelength),
            n2=config.n2,
            beam_x=config_float_attr(config, "beam_x", 0.0),
            beam_y=config_float_attr(config, "beam_y", 0.0),
            dtheta=config_float_attr(config, "dtheta", 0.0),
            dphi=config_float_attr(config, "dphi", 0.0),
        )
    except Exception as exc:
        print(f"detector-rotation fit debug unavailable: {exc}")
        return
    fit_debug["qr_obs"] = np.asarray(qr_obs, dtype=np.float64)
    fit_debug["qz_obs"] = np.asarray(qz_obs, dtype=np.float64)
    fit_debug["dqr"] = fit_debug["qr_obs"] - np.asarray(fit_debug["target_qr"], dtype=np.float64)
    fit_debug["dqz"] = fit_debug["qz_obs"] - np.asarray(fit_debug["target_qz"], dtype=np.float64)
    fit_debug["q_valid"] = np.asarray(valid, dtype=bool)
    if rod_line_lookup is not None:
        curve_distances = []
        for row_payload in fit_debug.to_dict("records"):
            key = point_identity_key(row_payload)
            curve_distances.append(
                point_to_projected_polyline_distance_px(
                    float(row_payload.get("fit_detector_col", np.nan)),
                    float(row_payload.get("fit_detector_row", np.nan)),
                    rod_line_lookup.get(key, []),
                )
            )
        fit_debug["curve_distance_px"] = np.asarray(curve_distances, dtype=np.float64)
    debug_columns = [
        "label",
        "m",
        "branch",
        "target_match_method",
        "fit_detector_col",
        "fit_detector_row",
        "curve_distance_px",
        "target_qr",
        "qr_obs",
        "dqr",
        "target_qz",
        "qz_obs",
        "dqz",
        "q_valid",
    ]
    display_table = fit_debug[
        [column for column in debug_columns if column in fit_debug.columns]
    ].copy()
    display_table["_sort_abs_dqr"] = np.abs(np.asarray(fit_debug["dqr"], dtype=np.float64))
    if "curve_distance_px" in fit_debug:
        display_table["_sort_curve_distance_px"] = np.asarray(
            fit_debug["curve_distance_px"], dtype=np.float64
        )
        sort_columns = ["_sort_abs_dqr", "_sort_curve_distance_px"]
    else:
        sort_columns = ["_sort_abs_dqr"]
    display(
        display_table.sort_values(sort_columns, ascending=False).drop(
            columns=[column for column in sort_columns if column in display_table.columns]
        )
    )


profile_detector_q_maps = notebook_detector_qr_qz_maps(
    config=profile_bg["qr_overlay_config"],
    detector_shape=tuple(profile_bg["detector_image"].shape),
)
if profile_detector_q_maps is None:
    raise RuntimeError("detector Qr/Qz maps are required for detector-space Qr rod profiles")
specular_detector_theta_initial_deg = 3.0 * float(ROD_PROFILE_TILT_DEG)
specular_detector_q_config = detector_qspace_config_with_theta_initial(
    profile_bg["qr_overlay_config"],
    specular_detector_theta_initial_deg,
)
specular_detector_q_maps = notebook_detector_qr_qz_maps(
    config=specular_detector_q_config,
    detector_shape=tuple(profile_bg["detector_image"].shape),
)
if specular_detector_q_maps is None:
    raise RuntimeError("detector Qr/Qz maps are required for m=0 detector-space Qr profiles")
profile_detector_phi_map = detector_gui_phi_map_for_background(profile_bg)
if profile_detector_phi_map is None:
    raise RuntimeError("detector phi map is required for detector-space Qr rod profiles")
profile_detector_solid_angle = (
    detector_solid_angle_for_background(profile_bg)
    if bool(BACKGROUND_SOLID_ANGLE_CORRECTION)
    else None
)
profile_detector_two_theta_map = detector_two_theta_map_for_background(profile_bg)
if profile_detector_two_theta_map is None:
    raise RuntimeError("detector 2theta map is required for Qr rod profile cutoff")
profile_qz_map = caked_qz_map_for_background(profile_bg, q_maps=profile_detector_q_maps)
theta_region_axis = np.asarray(profile_bg["theta_axis"], dtype=np.float64)
phi_region_axis = np.asarray(profile_bg["phi_axis"], dtype=np.float64)
theta_region_within_profile_limit = np.isfinite(theta_region_axis) & (
    theta_region_axis <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)
)


def bilinear_sample_detector_map(map_values: object, col: float, row: float) -> float:
    arr = np.asarray(map_values, dtype=np.float64)
    if arr.ndim != 2 or not (np.isfinite(col) and np.isfinite(row)):
        return float("nan")
    height, width = arr.shape
    if height < 1 or width < 1 or col < 0.0 or row < 0.0 or col > width - 1 or row > height - 1:
        return float("nan")
    c0 = int(np.floor(col))
    r0 = int(np.floor(row))
    c1 = min(c0 + 1, width - 1)
    r1 = min(r0 + 1, height - 1)
    dc = float(col - c0)
    dr = float(row - r0)
    values = np.asarray([arr[r0, c0], arr[r0, c1], arr[r1, c0], arr[r1, c1]], dtype=np.float64)
    weights = np.asarray(
        [(1.0 - dc) * (1.0 - dr), dc * (1.0 - dr), (1.0 - dc) * dr, dc * dr], dtype=np.float64
    )
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("nan")
    return float(np.sum(values[finite] * weights[finite]) / np.sum(weights[finite]))


def detector_xy_for_profile_peak(
    bg: dict[str, object], two_theta_deg: float, phi_deg: float
) -> tuple[float, float] | None:
    ai_value = bg.get("ai")
    if ai_value is None:
        ai_value = FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
    col, row = caked_point_to_detector_pixel(
        ai_value,
        tuple(bg["detector_image"].shape),
        np.asarray(bg["theta_axis"], dtype=np.float64),
        np.asarray(bg["phi_axis"], dtype=np.float64),
        float(two_theta_deg),
        float(phi_deg),
        transform_bundle=bg["transform_bundle"],
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    if col is None or row is None or not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def qr_fit_samples_for_rod(
    bg: dict[str, object],
    rod: dict[str, object],
    *,
    qr_map: np.ndarray,
    qz_map: np.ndarray,
    detector_phi_map: np.ndarray,
    detector_two_theta_map: np.ndarray,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    rejected_rod_keys = active_rejected_rod_keys()
    for item in candidate_marker_items_for_background(bg):
        if not accept_marker_for_plotted_rod(item, rod, rejected_rod_keys):
            continue
        try:
            params = np.asarray(item["params"], dtype=np.float64)
            theta0 = float(params[1])
            phi0 = float(params[2])
        except Exception:
            continue
        if not (np.isfinite(theta0) and np.isfinite(phi0)):
            continue
        if theta0 > float(ROD_PROFILE_MAX_TWO_THETA_DEG) or not (
            PHI_DISPLAY_MIN_DEG <= phi0 <= PHI_DISPLAY_MAX_DEG
        ):
            continue
        xy = detector_xy_for_profile_peak(bg, theta0, phi0)
        if xy is None:
            continue
        col, row = xy
        qr_value = bilinear_sample_detector_map(qr_map, col, row)
        qz_value = bilinear_sample_detector_map(qz_map, col, row)
        detector_phi = bilinear_sample_detector_map(detector_phi_map, col, row)
        detector_two_theta = bilinear_sample_detector_map(detector_two_theta_map, col, row)
        if not (np.isfinite(qr_value) and np.isfinite(qz_value) and qz_value > POSITIVE_QZ_MIN):
            continue
        if np.isfinite(detector_phi) and not (
            PHI_DISPLAY_MIN_DEG <= detector_phi <= PHI_DISPLAY_MAX_DEG
        ):
            continue
        if np.isfinite(detector_two_theta) and detector_two_theta > float(
            ROD_PROFILE_MAX_TWO_THETA_DEG
        ):
            continue
        samples.append(
            {
                "qr": float(qr_value),
                "qz": float(qz_value),
                "hkl": str(item["label"]),
                "branch": str(item.get("branch", branch_from_phi(phi0))),
                "detector_col": float(col),
                "detector_row": float(row),
            }
        )
    return samples


def fitted_qr_center_payload(
    original_qr: float, samples: list[dict[str, object]], delta_qr: float
) -> dict[str, object]:
    qrs = np.asarray(
        [float(sample["qr"]) for sample in samples if np.isfinite(float(sample.get("qr", np.nan)))],
        dtype=np.float64,
    )
    qrs = np.sort(qrs[np.isfinite(qrs) & (qrs > 0.0)])
    if qrs.size == 0:
        return {
            "qr": float(original_qr),
            "qr_fit_count": 0,
            "qr_fit_sample_count": int(len(samples)),
            "qr_fit_method": "original_no_valid_peak_qr_samples",
        }
    half_width = max(float(delta_qr), 1.0e-9)
    full_width = 2.0 * half_width
    best_center = float(qrs[0])
    best_count = 1
    best_key = (-1, float("inf"), float("inf"))
    for left in range(qrs.size):
        right = left
        while right + 1 < qrs.size and qrs[right + 1] - qrs[left] <= full_width:
            right += 1
        cluster = qrs[left : right + 1]
        center = float(np.nanmedian(cluster))
        final_cluster = qrs[np.abs(qrs - center) <= half_width]
        spread = (
            float(np.nanmax(final_cluster) - np.nanmin(final_cluster))
            if final_cluster.size
            else float("inf")
        )
        key = (-int(final_cluster.size), spread, abs(center - float(original_qr)))
        if key < best_key:
            best_key = key
            best_center = center
            best_count = int(final_cluster.size)
    return {
        "qr": best_center,
        "qr_fit_count": int(best_count),
        "qr_fit_sample_count": int(qrs.size),
        "qr_fit_method": "median_peak_qr_cluster",
    }


def fit_qr_centers_for_rod_entries(
    bg: dict[str, object],
    rods: list[dict[str, object]],
    *,
    detector_q_maps: tuple[np.ndarray, np.ndarray, np.ndarray],
    detector_phi_map: np.ndarray,
    detector_two_theta_map: np.ndarray,
) -> list[dict[str, object]]:
    qr_map, qz_map, _valid_q = detector_q_maps
    fitted_rods: list[dict[str, object]] = []
    for rod in rods:
        original_qr = float(rod["qr"])
        samples = qr_fit_samples_for_rod(
            bg,
            rod,
            qr_map=np.asarray(qr_map, dtype=np.float64),
            qz_map=np.asarray(qz_map, dtype=np.float64),
            detector_phi_map=np.asarray(detector_phi_map, dtype=np.float64),
            detector_two_theta_map=np.asarray(detector_two_theta_map, dtype=np.float64),
        )
        payload = fitted_qr_center_payload(original_qr, samples, qr_rod_delta_qr)
        fitted = dict(rod)
        fitted["qr_original"] = original_qr
        fitted["qr"] = float(payload["qr"])
        fitted["qr_fit_count"] = int(payload["qr_fit_count"])
        fitted["qr_fit_sample_count"] = int(payload["qr_fit_sample_count"])
        fitted["qr_fit_method"] = str(payload["qr_fit_method"])
        fitted_rods.append(fitted)
    return fitted_rods


if not qr_rod_pre_editor_cache_hit:
    if bool(rod_qspace_calibration.get("success", False)):
        rod_entries = fit_qr_centers_for_rod_entries(
            profile_bg,
            rod_entries,
            detector_q_maps=profile_detector_q_maps,
            detector_phi_map=profile_detector_phi_map,
            detector_two_theta_map=profile_detector_two_theta_map,
        )
        print(
            "Qr rod center adjustment before integration: "
            + ", ".join(
                f"HK={int(rod['m'])}: {float(rod.get('qr_original', rod['qr'])):.5g}->{float(rod['qr']):.5g} "
                f"({int(rod.get('qr_fit_count', 0))}/{int(rod.get('qr_fit_sample_count', 0))})"
                for rod in rod_entries
            )
        )
    else:
        print(
            "Qr rod center adjustment before integration: skipped; using saved Qr centers "
            f"because {rod_qspace_calibration.get('message', 'detector-rotation fit was not accepted')}"
        )
    rod_qspace_calibration["qr_fitted_by_m"] = {
        int(rod["m"]): float(rod["qr"]) for rod in rod_entries
    }
    rod_qspace_calibration["qr_original_by_m"] = {
        int(rod["m"]): float(rod.get("qr_original", rod["qr"])) for rod in rod_entries
    }
    rod_qspace_calibration["qr_fit_method_by_m"] = {
        int(rod["m"]): str(rod.get("qr_fit_method", "original")) for rod in rod_entries
    }


def specular_window_for_background(bg: dict[str, object]) -> tuple[float, float, float, float]:
    theta_values = []
    phi_values = []
    for item in candidate_marker_items_for_background(bg):
        try:
            h, k, _l = parse_hkl_label(str(item.get("label")))
            params = np.asarray(item["params"], dtype=np.float64)
            theta0 = float(params[1])
            phi0 = float(params[2])
        except Exception:
            continue
        if (
            h == 0
            and k == 0
            and int(_l) > 0
            and np.isfinite(theta0)
            and np.isfinite(phi0)
            and theta0 <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)
        ):
            theta_values.append(theta0)
            phi_values.append(phi0)
    theta_axis_finite = theta_region_axis[np.isfinite(theta_region_axis)]
    phi_axis_finite = phi_region_axis[np.isfinite(phi_region_axis)]
    theta_axis_min = float(np.nanmin(theta_axis_finite)) if theta_axis_finite.size else 0.0
    theta_axis_max = (
        float(np.nanmax(theta_axis_finite))
        if theta_axis_finite.size
        else float(ROD_PROFILE_MAX_TWO_THETA_DEG)
    )
    phi_axis_min = float(np.nanmin(phi_axis_finite)) if phi_axis_finite.size else -90.0
    phi_axis_max = float(np.nanmax(phi_axis_finite)) if phi_axis_finite.size else 90.0
    if theta_values and phi_values:
        theta_min = max(theta_axis_min, min(theta_values) - 2.0)
        theta_max = min(
            theta_axis_max, max(theta_values) + 2.0, float(ROD_PROFILE_MAX_TWO_THETA_DEG)
        )
        phi_min = max(phi_axis_min, min(phi_values) - 8.0)
        phi_max = min(phi_axis_max, max(phi_values) + 8.0)
    else:
        theta_min = max(theta_axis_min, 5.0)
        theta_max = min(theta_axis_max, 25.0, float(ROD_PROFILE_MAX_TWO_THETA_DEG))
        phi_min = max(phi_axis_min, -10.0)
        phi_max = min(phi_axis_max, 10.0)
    return float(theta_min), float(theta_max), float(phi_min), float(phi_max)


specular_theta_min_deg, specular_theta_max_deg, specular_phi_min_deg, specular_phi_max_deg = (
    specular_window_for_background(profile_bg)
)
specular_region_mask = (
    (theta_region_axis[None, :] >= float(specular_theta_min_deg))
    & (theta_region_axis[None, :] <= float(specular_theta_max_deg))
    & (phi_region_axis[:, None] > float(specular_phi_min_deg))
    & (phi_region_axis[:, None] < float(specular_phi_max_deg))
    & positive_qz_mask(profile_qz_map)
)
specular_detector_qr_map = np.asarray(specular_detector_q_maps[0], dtype=np.float64)
specular_detector_qz_map = np.asarray(specular_detector_q_maps[1], dtype=np.float64)
specular_detector_valid_q = np.asarray(specular_detector_q_maps[2], dtype=bool)
if float(specular_phi_min_deg) <= float(specular_phi_max_deg):
    specular_detector_phi_mask = (
        np.asarray(profile_detector_phi_map, dtype=np.float64) >= float(specular_phi_min_deg)
    ) & (np.asarray(profile_detector_phi_map, dtype=np.float64) <= float(specular_phi_max_deg))
else:
    specular_detector_phi_mask = (
        np.asarray(profile_detector_phi_map, dtype=np.float64) >= float(specular_phi_min_deg)
    ) | (np.asarray(profile_detector_phi_map, dtype=np.float64) <= float(specular_phi_max_deg))
specular_detector_region_mask = (
    specular_detector_valid_q
    & specular_detector_phi_mask
    & np.isfinite(specular_detector_qr_map)
    & np.isfinite(specular_detector_qz_map)
    & np.isfinite(profile_detector_two_theta_map)
    & (
        np.asarray(profile_detector_two_theta_map, dtype=np.float64)
        <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)
    )
    & (specular_detector_qz_map > POSITIVE_QZ_MIN)
    & (specular_detector_qr_map >= 0.0)
    & (specular_detector_qr_map <= float(qr_rod_delta_qr))
)


def specular_l_marker_rows_with_lattice_fallback(
    marker_rows: object, *, lattice_c: object, l_max: object
) -> pd.DataFrame:
    table = pd.DataFrame(marker_rows).copy()
    records = table.to_dict("records") if not table.empty else []
    existing_l: set[int] = set()
    for row in records:
        l_value = as_float(row.get("fit_l", row.get("l", row.get("display_l"))))
        if np.isfinite(l_value) and int(round(l_value)) > 0:
            existing_l.add(int(round(l_value)))
    max_l = int(np.floor(as_float(l_max, 0.0)))
    for l_value in range(1, max_l + 1):
        if l_value in existing_l:
            continue
        qz_value = active_lattice_qz_value_for_l(l_value, lattice_c)
        if not np.isfinite(qz_value) or qz_value <= POSITIVE_QZ_MIN:
            continue
        records.append(
            {
                "m": 0,
                "hk": 0,
                "branch": "qz",
                "hkl": f"0,0,{int(l_value)}",
                "l": int(l_value),
                "fit_l": int(l_value),
                "display_l": float(l_value),
                "qz_marker": float(qz_value),
                "projected_qz_marker": float(qz_value),
                "marker_source": "active_lattice",
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    sort_l = pd.to_numeric(
        out["fit_l"] if "fit_l" in out else out.get("l", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out = out.assign(_sort_l=sort_l)
    return out.sort_values(["_sort_l", "qz_marker"], kind="mergesort").drop(
        columns=["_sort_l"]
    ).reset_index(drop=True)


def specular_l_marker_rows_for_background(
    bg: dict[str, object], qz_map: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    theta_axis = np.asarray(bg["theta_axis"], dtype=np.float64)
    phi_axis = np.asarray(bg["phi_axis"], dtype=np.float64)
    qz_values = np.asarray(qz_map, dtype=np.float64)
    for item in candidate_marker_items_for_background(bg):
        try:
            h, k, l_value = parse_hkl_label(str(item["label"]))
            theta0 = float(item["params"][1])
            phi0 = float(item["params"][2])
        except Exception:
            continue
        if int(l_value) <= 0:
            continue
        if h != 0 or k != 0 or not (np.isfinite(theta0) and np.isfinite(phi0)):
            continue
        if theta0 > float(ROD_PROFILE_MAX_TWO_THETA_DEG):
            continue
        row = int(np.nanargmin(np.abs(wrapped_delta_deg(phi_axis, phi0))))
        col = int(np.nanargmin(np.abs(theta_axis - theta0)))
        qz_value = float(qz_values[row, col])
        if not np.isfinite(qz_value) or qz_value <= POSITIVE_QZ_MIN:
            continue
        rows.append(
            {
                "m": 0,
                "hk": 0,
                "branch": "qz",
                "hkl": str(item["label"]),
                "l": int(l_value),
                "fit_l": int(l_value),
                "display_l": float(l_value),
                "qz_marker": qz_value,
                "refined_two_theta_deg": theta0,
                "refined_phi_deg": phi0,
            }
        )
    return specular_l_marker_rows_with_lattice_fallback(
        rows, lattice_c=ACTIVE_LATTICE_C, l_max=SPECULAR_QR_ROD_L_MAX
    )


phi_windows = [
    ("-", -90.0, 0.0, "-"),
    ("+", 0.0, 90.0, "+"),
]
profile_rows = []
marker_rows = []
region_overlays = []
profile_failures = []
if qr_rod_pre_editor_cache_hit:
    rod_profile_table_cached = pd.DataFrame(qr_rod_pre_editor_stage["rod_profile_table"]).copy()
    marker_table_cached = pd.DataFrame(qr_rod_pre_editor_stage["marker_table"]).copy()
    profile_rows = [rod_profile_table_cached]
    marker_rows = marker_table_cached.to_dict("records")
    region_overlays = list(qr_rod_pre_editor_stage.get("region_overlays", []) or [])
    profile_failures = list(qr_rod_pre_editor_stage.get("profile_failures", []) or [])
    specular_l_marker_table = pd.DataFrame(
        qr_rod_pre_editor_stage.get("specular_l_marker_table", pd.DataFrame())
    ).copy()
else:
    specular_l_marker_table = specular_l_marker_rows_for_background(profile_bg, profile_qz_map)
specular_rod_entry = {
    "key": ("q_group", "primary", 0, "qz_rod"),
    "source": "primary",
    "m": 0,
    "qr": 0.0,
}
specular_all_qz_values = specular_detector_qz_map[
    specular_detector_region_mask & np.isfinite(specular_detector_qz_map)
]
specular_qz_bounds = qz_bounds_for_l_window(
    specular_all_qz_values,
    specular_l_marker_table,
    l_min=0.0,
    l_max=SPECULAR_QR_ROD_L_MAX,
    positive_qz_min=POSITIVE_QZ_MIN,
)
if specular_qz_bounds is None:
    specular_qz_values = np.asarray([], dtype=np.float64)
else:
    specular_qz_lo, specular_qz_hi = specular_qz_bounds
    specular_detector_region_mask = specular_detector_region_mask & (
        (specular_detector_qz_map >= float(specular_qz_lo))
        & (specular_detector_qz_map <= float(specular_qz_hi))
    )
    specular_qz_values = specular_detector_qz_map[
        specular_detector_region_mask & np.isfinite(specular_detector_qz_map)
    ]
if not qr_rod_pre_editor_cache_hit:
    if specular_qz_values.size >= 2:
        specular_edges = np.linspace(
            float(np.nanmin(specular_qz_values)),
            float(np.nanmax(specular_qz_values)),
            ROD_PROFILE_QZ_BINS + 1,
            dtype=np.float64,
        )
        profile_rows.append(
            profile_from_detector_qr_qz(
                profile_bg,
                specular_rod_entry,
                branch_name="qz",
                branch_label="m = 0",
                qz_edges=specular_edges,
                phi_min=float(specular_phi_min_deg),
                phi_max=float(specular_phi_max_deg),
                detector_q_maps=specular_detector_q_maps,
                detector_phi_map=profile_detector_phi_map,
                detector_solid_angle=profile_detector_solid_angle,
                detector_two_theta_map=profile_detector_two_theta_map,
                theta_initial_deg_used_for_q=specular_detector_theta_initial_deg,
            )
        )
    else:
        profile_failures.append(
            (
                ROD_PROFILE_TILT_DEG,
                0.0,
                "(0,L)",
                f"no detector Qr/Qz values in dynamic specular window for L <= {SPECULAR_QR_ROD_L_MAX:g}",
            )
        )


for rod in [] if qr_rod_pre_editor_cache_hit else rod_entries:
    samples = gui_qr_cylinder_overlay.project_selected_qr_rod_caked_samples(
        selected_entry=rod,
        config=profile_bg["qr_overlay_config"],
        projection_context=profile_bg["caked_projection_context"],
    )
    if samples is None:
        profile_failures.append(
            (ROD_PROFILE_TILT_DEG, float(rod["qr"]), "projected samples unavailable")
        )
        continue
    for branch_name, phi_min, phi_max, branch_label in phi_windows:
        branch_samples, branch_sign = select_projected_samples_for_rod_branch(
            profile_bg,
            rod,
            samples,
            branch_name=branch_name,
            phi_min=phi_min,
            phi_max=phi_max,
        )
        qz_bounds = qz_bounds_for_phi_window(branch_samples, phi_min, phi_max)
        if qz_bounds is None:
            profile_failures.append(
                (ROD_PROFILE_TILT_DEG, float(rod["qr"]), branch_label, "no projected qz span")
            )
            continue
        qz_bounds = positive_qz_bounds(*qz_bounds)
        if qz_bounds is None:
            profile_failures.append(
                (ROD_PROFILE_TILT_DEG, float(rod["qr"]), branch_label, "no positive qz span")
            )
            continue
        qz_lo, qz_hi = qz_bounds
        mask_payload = gui_qr_cylinder_overlay.build_selected_qr_rod_qz_caked_mask(
            selected_entry=rod,
            config=profile_bg["qr_overlay_config"],
            projection_context=profile_bg["caked_projection_context"],
            radial_axis=np.asarray(profile_bg["theta_axis"], dtype=np.float64),
            azimuth_axis=np.asarray(profile_bg["phi_axis"], dtype=np.float64),
            delta_qr=qr_rod_delta_qr,
            qz_min=qz_lo,
            qz_max=qz_hi,
            phi_min=phi_min,
            phi_max=phi_max,
        )
        if not isinstance(mask_payload, dict):
            profile_failures.append(
                (ROD_PROFILE_TILT_DEG, float(rod["qr"]), branch_label, "mask unavailable")
            )
            continue
        branch_mask = np.asarray(mask_payload.get("mask"), dtype=bool)
        if branch_mask.shape != np.asarray(profile_bg["caked_image"]).shape:
            profile_failures.append(
                (ROD_PROFILE_TILT_DEG, float(rod["qr"]), branch_label, "mask shape mismatch")
            )
            continue
        branch_mask = (
            branch_mask
            & theta_region_within_profile_limit[None, :]
            & positive_qz_mask(profile_qz_map)
        )
        if not np.any(branch_mask):
            profile_failures.append(
                (ROD_PROFILE_TILT_DEG, float(rod["qr"]), branch_label, "empty mask")
            )
            continue
        qz_edge_lo, qz_edge_hi = sorted((float(qz_lo), float(qz_hi)))
        edges = np.linspace(qz_edge_lo, qz_edge_hi, ROD_PROFILE_QZ_BINS + 1, dtype=np.float64)
        detector_profile = profile_from_detector_qr_qz(
            profile_bg,
            rod,
            branch_name=branch_name,
            branch_label=branch_label,
            qz_edges=edges,
            phi_min=phi_min,
            phi_max=phi_max,
            detector_q_maps=profile_detector_q_maps,
            detector_phi_map=profile_detector_phi_map,
            detector_solid_angle=profile_detector_solid_angle,
            detector_two_theta_map=profile_detector_two_theta_map,
        )
        if detector_profile.empty or int(np.nansum(detector_profile["pixel_count"])) <= 0:
            profile_failures.append(
                (
                    ROD_PROFILE_TILT_DEG,
                    float(rod["qr"]),
                    branch_label,
                    "empty detector Qr/Qz profile",
                )
            )
            continue
        region_overlays.append(
            {
                "source": str(rod.get("source", "")),
                "m": int(rod["m"]),
                "qr": float(rod["qr"]),
                "qr_original": float(rod.get("qr_original", rod["qr"])),
                "qr_fit_count": int(rod.get("qr_fit_count", 0)),
                "qr_fit_sample_count": int(rod.get("qr_fit_sample_count", 0)),
                "qr_fit_method": str(rod.get("qr_fit_method", "original")),
                "branch": str(branch_name),
                "branch_sign": None if branch_sign is None else int(branch_sign),
                "branch_label": str(branch_label),
                "qz_min": qz_edge_lo,
                "qz_max": qz_edge_hi,
                "mask": branch_mask.copy(),
            }
        )
        profile_rows.append(detector_profile)
        marker_rows.extend(
            marker_rows_for_rod_branch(
                profile_bg,
                rod,
                branch_samples,
                branch_name=branch_name,
                branch_label=branch_label,
                phi_min=phi_min,
                phi_max=phi_max,
                branch_mask=branch_mask,
                qz_map=profile_qz_map,
            )
        )

if not profile_rows:
    raise RuntimeError(f"no {ROD_PROFILE_TILT_LABEL} deg Qr rod profiles were generated")

marker_table = pd.DataFrame(marker_rows)
if not marker_table.empty and "qz_marker" in marker_table and "l" in marker_table:
    marker_table = marker_table[
        (np.asarray(marker_table["qz_marker"], dtype=np.float64) > POSITIVE_QZ_MIN)
        & (np.asarray(marker_table["l"], dtype=np.float64) > 0.0)
    ].copy()
rod_profile_table = pd.concat(profile_rows, ignore_index=True)
rod_profile_table = rod_profile_table[
    np.asarray(rod_profile_table["qz_center"], dtype=np.float64) > POSITIVE_QZ_MIN
].copy()
if rod_profile_table.empty:
    raise RuntimeError(
        f"no positive-Qz {ROD_PROFILE_TILT_LABEL} deg Qr rod profiles were generated"
    )
marker_table = marker_table_with_specular_l_markers(marker_table, specular_l_marker_table)
if not qr_rod_pre_editor_cache_hit:
    pre_editor_cache = pre_editor_cache_with_stage(
        pre_editor_cache,
        "qr_rod_pre_editor",
        PRE_EDITOR_QR_ROD_STAGE_SIGNATURE,
        {
            "rod_profile_table": rod_profile_table,
            "marker_table": marker_table,
            "region_overlays": region_overlays,
            "profile_failures": profile_failures,
            "specular_l_marker_table": specular_l_marker_table,
            "rod_entries": [dict(rod) for rod in rod_entries],
            "rod_qspace_calibration": dict(rod_qspace_calibration),
            "active_rejected_rod_keys": [
                list(key)
                for key in sorted(
                    ACTIVE_REJECTED_ROD_KEYS, key=lambda item: tuple(str(part) for part in item)
                )
            ],
            "rod_profile_max_two_theta_deg": float(ROD_PROFILE_MAX_TWO_THETA_DEG),
        },
    )
    write_pre_editor_cache(PRE_EDITOR_CACHE_PATH, PRE_EDITOR_CACHE_KEY, pre_editor_cache)
    print(f"saved pre-editor Qr-rod profile cache={PRE_EDITOR_CACHE_PATH}")


def qz_edges_from_profile_group(profile_group: pd.DataFrame) -> np.ndarray | None:
    group = pd.DataFrame(profile_group).copy()
    if group.empty:
        return None
    if {"qz_min", "qz_max"}.issubset(group.columns):
        group = group.sort_values("qz_min", kind="mergesort")
        qz_min = pd.to_numeric(group["qz_min"], errors="coerce").to_numpy(dtype=np.float64)
        qz_max = pd.to_numeric(group["qz_max"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(qz_min) & np.isfinite(qz_max) & (qz_max > qz_min)
        if np.count_nonzero(finite) >= 1:
            qz_min = qz_min[finite]
            qz_max = qz_max[finite]
            edges = np.concatenate([qz_min[:1], qz_max])
            if edges.size >= 2 and np.all(np.isfinite(edges)) and np.all(np.diff(edges) > 0.0):
                return edges.astype(np.float64, copy=False)
    if "qz_center" not in group:
        return None
    centers = pd.to_numeric(group["qz_center"], errors="coerce").to_numpy(dtype=np.float64)
    centers = np.unique(np.sort(centers[np.isfinite(centers)]))
    if centers.size < 2:
        return None
    mids = 0.5 * (centers[:-1] + centers[1:])
    first_width = max(float(mids[0] - centers[0]), 1.0e-9)
    last_width = max(float(centers[-1] - mids[-1]), 1.0e-9)
    return np.concatenate(
        [
            np.asarray([max(POSITIVE_QZ_MIN, float(centers[0] - first_width))]),
            mids,
            np.asarray([float(centers[-1] + last_width)]),
        ]
    ).astype(np.float64, copy=False)


def rod_profile_table_for_l_window(
    profile_table: pd.DataFrame,
    marker_source: pd.DataFrame,
    l_min: object,
    l_max: object,
) -> pd.DataFrame:
    table = pd.DataFrame(profile_table).copy().reset_index(drop=True)
    if table.empty or "qz_center" not in table or not {"m", "branch"}.issubset(table.columns):
        return table
    l_lo = as_float(l_min, np.nan)
    l_hi = as_float(l_max, np.nan)
    if not (np.isfinite(l_lo) and np.isfinite(l_hi)):
        return table
    l_lo, l_hi = sorted((float(l_lo), float(l_hi)))
    if l_hi <= l_lo:
        return table
    keep = np.zeros(len(table), dtype=bool)
    for (m_value, branch_value), sub in table.groupby(["m", "branch"], sort=False):
        l_values = qz_values_to_l_axis(
            sub["qz_center"],
            m_value=int(m_value),
            branch_value=str(branch_value),
            marker_source=marker_source,
        )
        keep[np.asarray(sub.index, dtype=int)] = (
            np.isfinite(l_values) & (l_values >= l_lo) & (l_values <= l_hi)
        )
    return table.loc[keep].reset_index(drop=True)


def rod_profile_l_window_from_table(
    profile_table: pd.DataFrame,
    marker_source: pd.DataFrame,
    *,
    fallback: tuple[float, float] = (0.0, 8.0),
) -> tuple[float, float]:
    table = pd.DataFrame(profile_table).copy()
    if table.empty or "qz_center" not in table or not {"m", "branch"}.issubset(table.columns):
        return tuple(float(value) for value in fallback)
    values: list[np.ndarray] = []
    for (m_value, branch_value), sub in table.groupby(["m", "branch"], sort=False):
        values.append(
            qz_values_to_l_axis(
                sub["qz_center"],
                m_value=int(m_value),
                branch_value=str(branch_value),
                marker_source=marker_source,
            )
        )
    if not values:
        return tuple(float(value) for value in fallback)
    finite = np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in values])
    finite = finite[np.isfinite(finite)]
    if finite.size < 1:
        return tuple(float(value) for value in fallback)
    l_min = float(np.nanmin(finite))
    l_max = float(np.nanmax(finite))
    if not (np.isfinite(l_min) and np.isfinite(l_max)) or l_max <= l_min:
        return tuple(float(value) for value in fallback)
    return l_min, l_max


qr_rod_editor_base_profiles = rod_profile_table.copy()
qr_rod_editor_profile_cache: dict[tuple[float, float, float], pd.DataFrame] = {}


def recompute_qr_rod_region_profiles(
    delta_qr_value: object,
    l_min_value: object,
    l_max_value: object,
) -> pd.DataFrame:
    delta_qr = max(1.0e-9, as_float(delta_qr_value, qr_rod_delta_qr))
    l_min_number = as_float(l_min_value, 0.0)
    l_max_number = as_float(l_max_value, np.nan)
    cache_key = (
        round(float(delta_qr), 12),
        round(float(l_min_number), 9) if np.isfinite(l_min_number) else float("nan"),
        round(float(l_max_number), 9) if np.isfinite(l_max_number) else float("nan"),
    )
    cached = qr_rod_editor_profile_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    branch_lookup = {
        str(branch_name): (float(phi_min), float(phi_max), str(branch_label))
        for branch_name, phi_min, phi_max, branch_label in phi_windows
    }
    rod_lookup = {int(rod["m"]): dict(rod) for rod in rod_entries}
    rebuilt_rows: list[pd.DataFrame] = []
    for (m_value, branch_value), sub in qr_rod_editor_base_profiles.groupby(
        ["m", "branch"], sort=False
    ):
        edges = qz_edges_from_profile_group(sub)
        if edges is None:
            continue
        m_int = int(m_value)
        branch_text = str(branch_value)
        if m_int == 0 and branch_text == "qz":
            rebuilt = profile_from_detector_qr_qz(
                profile_bg,
                specular_rod_entry,
                branch_name="qz",
                branch_label="m = 0",
                qz_edges=edges,
                phi_min=float(specular_phi_min_deg),
                phi_max=float(specular_phi_max_deg),
                detector_q_maps=specular_detector_q_maps,
                detector_phi_map=profile_detector_phi_map,
                detector_solid_angle=profile_detector_solid_angle,
                detector_two_theta_map=profile_detector_two_theta_map,
                theta_initial_deg_used_for_q=specular_detector_theta_initial_deg,
                delta_qr_override=delta_qr_value,
            )
        elif m_int in rod_lookup and branch_text in branch_lookup:
            phi_min, phi_max, branch_label = branch_lookup[branch_text]
            rebuilt = profile_from_detector_qr_qz(
                profile_bg,
                rod_lookup[m_int],
                branch_name=branch_text,
                branch_label=branch_label,
                qz_edges=edges,
                phi_min=phi_min,
                phi_max=phi_max,
                detector_q_maps=profile_detector_q_maps,
                detector_phi_map=profile_detector_phi_map,
                detector_solid_angle=profile_detector_solid_angle,
                detector_two_theta_map=profile_detector_two_theta_map,
                delta_qr_override=delta_qr_value,
            )
        else:
            continue
        if not rebuilt.empty:
            rebuilt_rows.append(rebuilt)

    if rebuilt_rows:
        refreshed = pd.concat(rebuilt_rows, ignore_index=True)
        refreshed = refreshed[
            np.asarray(refreshed["qz_center"], dtype=np.float64) > POSITIVE_QZ_MIN
        ].copy()
    else:
        refreshed = qr_rod_editor_base_profiles.copy()
    refreshed = rod_profile_table_for_l_window(
        refreshed, marker_table, l_min_number, l_max_number
    )
    if refreshed.empty:
        refreshed = qr_rod_editor_base_profiles.copy()
    qr_rod_editor_profile_cache[cache_key] = refreshed.copy()
    return refreshed


QR_ROD_PROFILE_CACHE_PATH = qr_rod_profile_cache_path(OUT_DIR, STATE_PATH)
if _truthy_setting(
    "RESET_QR_ROD_PROFILE_CACHE_OVERRIDE", "RA_SIM_RESET_QR_ROD_PROFILE_CACHE", False
):
    removed_qr_rod_cache = reset_qr_rod_profile_cache(QR_ROD_PROFILE_CACHE_PATH)
    print(f"reset Qr-rod profile cache={QR_ROD_PROFILE_CACHE_PATH} removed={removed_qr_rod_cache}")
qr_rod_profile_cache = load_qr_rod_profile_cache(QR_ROD_PROFILE_CACHE_PATH, STATE_PATH)
qr_rod_peak_edits_path = _setting_text(
    "QR_ROD_PEAK_EDITS_PATH_OVERRIDE", "RA_SIM_QR_ROD_PEAK_EDITS", ""
)
qr_rod_peak_edit_mode = _setting_text(
    "QR_ROD_PEAK_EDIT_MODE_OVERRIDE", "RA_SIM_QR_ROD_PEAK_EDIT_MODE", "auto"
)
qr_rod_editor_initial_l_min, qr_rod_editor_initial_l_max = rod_profile_l_window_from_table(
    rod_profile_table,
    marker_table,
    fallback=(0.0, SPECULAR_QR_ROD_L_MAX),
)
qr_rod_region_editor_result = edit_qr_rod_region_editor(
    marker_table,
    rod_profile_table,
    mode=qr_rod_peak_edit_mode,
    edit_path=qr_rod_peak_edits_path or None,
    detector_label_entries=[],
    delta_qr=qr_rod_delta_qr,
    l_min=qr_rod_editor_initial_l_min,
    l_max=qr_rod_editor_initial_l_max,
    profile_update_callback=recompute_qr_rod_region_profiles,
    required_marker_table=specular_l_marker_table,
    backend_name=mpl.get_backend(),
    env=os.environ,
)
marker_table = pd.DataFrame(qr_rod_region_editor_result["marker_table"]).copy()
qr_rod_peak_edit_source = str(qr_rod_region_editor_result.get("source", "last_cached"))
qr_rod_delta_qr = max(
    1.0e-9, as_float(qr_rod_region_editor_result.get("delta_qr"), qr_rod_delta_qr)
)
qr_rod_editor_l_min = as_float(
    qr_rod_region_editor_result.get("l_min"), qr_rod_editor_initial_l_min
)
qr_rod_editor_l_max = as_float(
    qr_rod_region_editor_result.get("l_max"), qr_rod_editor_initial_l_max
)
rod_profile_table = pd.DataFrame(
    qr_rod_region_editor_result.get("rod_profile_table", rod_profile_table)
).copy()
rod_profile_table = rod_profile_table_for_l_window(
    rod_profile_table,
    marker_table,
    qr_rod_editor_l_min,
    qr_rod_editor_l_max,
)
qr_rod_peak_edit_key = qr_rod_peak_edit_cache_key(
    None if qr_rod_peak_edit_source == "last_cached" else qr_rod_peak_edits_path or None,
    marker_table=marker_table if qr_rod_peak_edit_source != "last_cached" else None,
    mode=qr_rod_peak_edit_source,
    lattice_signature=ACTIVE_LATTICE_CACHE_SIGNATURE,
    q_group_signature=Q_GROUP_ROWS_CACHE_SIGNATURE,
    rod_reference_policy=ROD_REFERENCE_POLICY_SIGNATURE,
    rod_profile_policy={
        **ROD_PROFILE_BACKGROUND_POLICY_SIGNATURE,
        "editor_delta_qr": float(qr_rod_delta_qr),
        "editor_l_min": float(qr_rod_editor_l_min),
        "editor_l_max": float(qr_rod_editor_l_max),
    },
)
if qr_rod_profile_cache_has_final_fit(qr_rod_profile_cache or {}, qr_rod_peak_edit_key):
    rod_profile_table = pd.DataFrame(qr_rod_profile_cache["final_rod_profile_table"]).copy()
    marker_table = pd.DataFrame(qr_rod_profile_cache["final_marker_table"]).copy()
    rod_component_table = pd.DataFrame(qr_rod_profile_cache["final_rod_component_table"]).copy()
    rod_profile_table.attrs["joint_fit_component_table"] = rod_component_table
    print(f"reused final Qr-rod fit cache={QR_ROD_PROFILE_CACHE_PATH}")
else:
    rod_profile_table = add_joint_qz_fit_columns(
        rod_profile_table,
        marker_table,
        bg=profile_bg,
        detector_q_maps=profile_detector_q_maps,
        detector_phi_map=profile_detector_phi_map,
        detector_solid_angle=profile_detector_solid_angle,
        detector_two_theta_map=profile_detector_two_theta_map,
    )
    rod_component_table = rod_profile_table.attrs.get("joint_fit_component_table", pd.DataFrame())
    qr_rod_profile_cache = qr_rod_profile_cache_with_final_fit(
        qr_rod_profile_cache,
        rod_profile_table,
        marker_table,
        rod_component_table,
        qr_rod_peak_edit_key,
    )
    write_qr_rod_profile_cache(QR_ROD_PROFILE_CACHE_PATH, STATE_PATH, qr_rod_profile_cache)
    print(f"saved final Qr-rod fit cache={QR_ROD_PROFILE_CACHE_PATH}")
marker_table = marker_table_with_specular_l_markers(marker_table, specular_l_marker_table)
specular_l_marker_table = specular_export_marker_table_from_final_markers(
    marker_table,
    specular_l_marker_table,
    qz_map=profile_qz_map,
    region_mask=specular_region_mask,
    theta_axis=theta_region_axis,
    phi_axis=phi_region_axis,
)
if not marker_table.empty and {"m", "branch"}.issubset(marker_table.columns):
    marker_m = pd.to_numeric(marker_table["m"], errors="coerce").to_numpy(dtype=np.float64)
    marker_branch = marker_table["branch"].astype(str).to_numpy(dtype=object)
    marker_table = marker_table.loc[~((marker_m == 0.0) & (marker_branch == "qz"))].copy()
marker_table = marker_table_with_specular_l_markers(marker_table, specular_l_marker_table)
print(
    f"rod-profile joint fits: groups={int(rod_profile_table.groupby(['m', 'branch']).ngroups) if not rod_profile_table.empty else 0} workers={ROD_PROFILE_FIT_WORKERS} gpu={GPU_ACCELERATION_ENABLED}"
)
if "joint_peak_density" in rod_profile_table:
    rod_profile_table["peak_subtracted_density"] = np.asarray(
        rod_profile_table["background_density"], dtype=np.float64
    ) - np.asarray(rod_profile_table["joint_peak_density"], dtype=np.float64)
    rod_profile_table["fit_residual_density"] = np.asarray(
        rod_profile_table["background_density"], dtype=np.float64
    ) - np.asarray(rod_profile_table["joint_fit_density"], dtype=np.float64)
elif "joint_fit_density" in rod_profile_table:
    rod_profile_table["peak_subtracted_density"] = np.asarray(
        rod_profile_table["background_density"], dtype=np.float64
    ) - np.asarray(rod_profile_table["joint_fit_density"], dtype=np.float64)
elif "fit_density" in rod_profile_table:
    rod_profile_table["peak_subtracted_density"] = np.asarray(
        rod_profile_table["background_density"], dtype=np.float64
    ) - np.asarray(rod_profile_table["fit_density"], dtype=np.float64)
rod_profile_csv = OUT_DIR / f"{ROD_PROFILE_STEM}.csv"
rod_profile_table.to_csv(rod_profile_csv, index=False)
marker_csv = OUT_DIR / f"{ROD_PROFILE_MARKER_STEM}.csv"
marker_table.to_csv(marker_csv, index=False)
component_csv = OUT_DIR / f"{ROD_PROFILE_STEM}_tail_components.csv"
rod_component_table.to_csv(component_csv, index=False)
print(f"saved={rod_profile_csv}")
print(f"saved={marker_csv}")
print(f"saved={component_csv}")

plot_marker_table = marker_table.copy()
if not plot_marker_table.empty and "m" in plot_marker_table:
    plot_marker_table["hk"] = np.asarray(plot_marker_table["m"], dtype=int)


def l_reference_rows(
    *, m_value: int, branch_value: str, marker_source: pd.DataFrame | None = None
) -> pd.DataFrame:
    source = plot_marker_table if marker_source is None else marker_source
    if source is None or source.empty or "qz_marker" not in source or "l" not in source:
        return pd.DataFrame()
    sub = source[
        (np.asarray(source["m"], dtype=int) == int(m_value))
        & (source["branch"].astype(str) == str(branch_value))
    ].copy()
    if sub.empty:
        return sub
    qz = np.asarray(sub["qz_marker"], dtype=np.float64)
    fit_l_column = "fit_l" if "fit_l" in sub else "l"
    l_values = np.asarray(sub[fit_l_column], dtype=np.float64)
    finite = np.isfinite(qz) & np.isfinite(l_values)
    sub = sub.loc[finite].copy()
    if sub.empty:
        return sub
    if "fit_l" not in sub:
        sub["fit_l"] = np.asarray(sub["l"], dtype=np.float64)
    if "display_l" not in sub:
        sub["display_l"] = np.asarray(sub["fit_l"], dtype=np.float64)
    sub = sub.sort_values("qz_marker")
    return sub.drop_duplicates(subset=["qz_marker"], keep="first")


def qz_values_to_l_axis(
    qz_values: object, *, m_value: int, branch_value: str, marker_source: pd.DataFrame | None = None
) -> np.ndarray:
    qz = np.asarray(qz_values, dtype=np.float64)
    refs = l_reference_rows(
        m_value=int(m_value), branch_value=str(branch_value), marker_source=marker_source
    )
    if refs.empty:
        return qz.copy()
    ref_qz = np.asarray(refs["qz_marker"], dtype=np.float64)
    ref_l = np.asarray(refs["fit_l"] if "fit_l" in refs else refs["l"], dtype=np.float64)
    if ref_qz.size >= 2 and np.nanmax(ref_qz) > np.nanmin(ref_qz):
        slope, intercept = np.polyfit(ref_qz, ref_l, 1)
        return slope * qz + intercept
    if ref_qz.size == 1 and np.isfinite(ref_qz[0]) and abs(float(ref_qz[0])) > 1.0e-12:
        return qz * (float(ref_l[0]) / float(ref_qz[0]))
    return qz.copy()


def qz_to_l_linear_coeff(*, m_value: int, branch_value: str) -> tuple[float, float]:
    refs = l_reference_rows(m_value=int(m_value), branch_value=str(branch_value))
    if refs.empty:
        return 1.0, 0.0
    ref_qz = np.asarray(refs["qz_marker"], dtype=np.float64)
    ref_l = np.asarray(refs["fit_l"] if "fit_l" in refs else refs["l"], dtype=np.float64)
    finite = np.isfinite(ref_qz) & np.isfinite(ref_l)
    ref_qz = ref_qz[finite]
    ref_l = ref_l[finite]
    if ref_qz.size >= 2 and np.nanmax(ref_qz) > np.nanmin(ref_qz):
        slope, intercept = np.polyfit(ref_qz, ref_l, 1)
        return float(slope), float(intercept)
    if ref_qz.size == 1 and abs(float(ref_qz[0])) > 1.0e-12:
        return float(ref_l[0] / ref_qz[0]), 0.0
    return 1.0, 0.0


def positive_l_qz_bounds(
    *, m_value: int, branch_value: str, qz_lo: object, qz_hi: object
) -> tuple[float, float] | None:
    bounds = positive_qz_bounds(qz_lo, qz_hi)
    if bounds is None:
        return None
    lo, hi = bounds
    slope, intercept = qz_to_l_linear_coeff(m_value=int(m_value), branch_value=str(branch_value))
    if abs(float(slope)) <= 1.0e-12:
        return (lo, hi) if float(intercept) > 0.0 else None
    root = -float(intercept) / float(slope)
    if float(slope) > 0.0:
        lo = max(lo, np.nextafter(root, np.inf))
    else:
        hi = min(hi, np.nextafter(root, -np.inf))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None
    return float(lo), float(hi)


def positive_l_rows(table: pd.DataFrame, *, m_value: int, branch_value: str) -> pd.DataFrame:
    if table.empty or "qz_center" not in table:
        return table.copy()
    l_values = qz_values_to_l_axis(
        table["qz_center"], m_value=int(m_value), branch_value=str(branch_value)
    )
    return table[np.asarray(l_values, dtype=np.float64) > 0.0].copy()


def rod_profile_panel_label(m_value: int, branch_value: str) -> str:
    if int(m_value) == 0:
        return "$m = 0$"
    return f"$m = {int(m_value)}\\ {str(branch_value).strip()}$"


def shared_rod_profile_l_axis_limits(
    profile_table: pd.DataFrame,
    marker_source: pd.DataFrame,
    plot_entries: list[dict[str, object]],
    branch_windows: object,
    *,
    min_l: float = 2.0,
    fallback_span: float = 1.0,
) -> tuple[float, float]:
    l_values: list[np.ndarray] = []
    table = pd.DataFrame(profile_table).copy()
    markers = pd.DataFrame(marker_source).copy()
    for rod in plot_entries:
        m_value = int(rod["m"])
        branches = ("qz",) if m_value == 0 else tuple(str(item[0]) for item in branch_windows)
        for branch_value in branches:
            refs = l_reference_rows(
                m_value=m_value, branch_value=branch_value, marker_source=markers
            )
            if not refs.empty:
                l_values.append(np.asarray(refs["fit_l"], dtype=np.float64))
            if table.empty or "qz_center" not in table:
                continue
            sub = table[
                (np.asarray(table["m"], dtype=int) == m_value)
                & (table["branch"].astype(str) == str(branch_value))
            ]
            if sub.empty:
                continue
            l_values.append(
                qz_values_to_l_axis(
                    sub["qz_center"],
                    m_value=m_value,
                    branch_value=branch_value,
                    marker_source=markers,
                )
            )
    if not l_values:
        return float(min_l), float(min_l + max(float(fallback_span), 1.0e-6))
    finite = np.concatenate(
        [np.asarray(values, dtype=np.float64).reshape(-1) for values in l_values]
    )
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return float(min_l), float(min_l + max(float(fallback_span), 1.0e-6))
    upper = max(float(np.nanmax(finite)), float(min_l) + max(float(fallback_span), 1.0e-6))
    return float(min_l), float(upper)


def shared_nonzero_rod_profile_y_axis_limits(
    profile_table: pd.DataFrame,
    marker_source: pd.DataFrame,
    plot_entries: list[dict[str, object]],
    branch_windows: object,
    *,
    fallback_limits: tuple[float, float] = (-1.0, 1.0),
    margin_fraction: float = 0.08,
) -> tuple[float, float]:
    table = pd.DataFrame(profile_table).copy()
    markers = pd.DataFrame(marker_source).copy()
    y_values: list[np.ndarray] = []
    if table.empty:
        return tuple(float(value) for value in fallback_limits)
    for rod in plot_entries:
        m_value = int(rod["m"])
        if m_value == 0:
            continue
        for branch_value in tuple(str(item[0]) for item in branch_windows):
            sub = table[
                (np.asarray(table["m"], dtype=int) == m_value)
                & (table["branch"].astype(str) == str(branch_value))
            ].copy()
            if "pixel_count" in sub:
                sub = sub[np.asarray(sub["pixel_count"], dtype=np.float64) > 0.0].copy()
            if sub.empty:
                continue
            if "qz_center" in sub:
                l_values = qz_values_to_l_axis(
                    sub["qz_center"],
                    m_value=m_value,
                    branch_value=branch_value,
                    marker_source=markers,
                )
                sub = sub[np.asarray(l_values, dtype=np.float64) > 0.0].copy()
            if sub.empty or "background_density" not in sub:
                continue
            plot_decision = rod_profile_plot_model_decision(
                str(globals().get("SAMPLE_STEM", "")),
                m_value,
                branch_value,
                sub,
                markers,
                transverse_background_enabled=bool(
                    globals().get("QR_ROD_TRANSVERSE_BACKGROUND_ENABLED", False)
                ),
                background_subtraction_disabled=bool(
                    globals().get("PBI2_DISABLE_BACKGROUND_SUBTRACTION", False)
                ),
            )
            plot_model, norm_payload = rod_profile_normalized_payload_for_plot_decision(
                sub, plot_decision
            )
            y_values.append(np.asarray(norm_payload["data"], dtype=np.float64))
            if plot_model:
                y_values.append(np.asarray(norm_payload["simulation"], dtype=np.float64))
    if not y_values:
        return tuple(float(value) for value in fallback_limits)
    finite = np.concatenate([np.asarray([0.0], dtype=np.float64), *y_values])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return tuple(float(value) for value in fallback_limits)
    y_min = float(np.nanmin(finite))
    y_max = float(np.nanmax(finite))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(y_max), 1.0)
    pad = max(span * float(margin_fraction), 1.0e-6)
    return float(y_min - pad), float(y_max + pad)


support_diagnostic_stem = f"{ROD_PROFILE_STEM}_support_diagnostics"
fig, support_axes = plt.subplots(
    2, 2, figsize=(JOURNAL_FULL_WIDTH_IN, 4.8), constrained_layout=True
)
for ax, metric, ylabel in zip(
    support_axes.ravel(),
    ("pixel_count", "acceptance_sum", "background_sum", "background_density"),
    ("Pixel count", "Acceptance sum", "Raw background sum", "Background density"),
):
    for (m_value, branch_value), sub in rod_profile_table.groupby(["m", "branch"]):
        sub = sub[np.asarray(sub["pixel_count"], dtype=np.float64) > 0].copy()
        if sub.empty or metric not in sub:
            continue
        x = qz_values_to_l_axis(
            sub["qz_center"], m_value=int(m_value), branch_value=str(branch_value)
        )
        y = np.asarray(sub[metric], dtype=np.float64)
        order = np.argsort(x)
        ax.plot(
            x[order], y[order], linewidth=0.7, alpha=0.65, label=f"m={int(m_value)} {branch_value}"
        )
    ax.set_ylabel(ylabel)
    ax.grid(True, color=JOURNAL_GRID_COLOR, linewidth=0.45)
    finish_axes(ax)
support_axes[0, 0].legend(loc="best", fontsize=5.2, frameon=False, ncol=2)
maybe_suptitle(fig, "Rod-profile support diagnostics over L", y=1.02)
support_png, support_pdf = save_manuscript_figure(fig, support_diagnostic_stem)
plt.close(fig)
print(f"saved={support_png}")
print(f"saved={support_pdf}")

rod_profile_note = OUT_DIR / f"{ROD_PROFILE_STEM}.md"
rod_note_lines = [
    f"# {SAMPLE_NAME} {ROD_PROFILE_TILT_LABEL}° HK-rod L profiles",
    "",
    f"Source state: `{STATE_PATH}`",
    (
        f"Active lattice: `a={ACTIVE_LATTICE_A:.6g} A`, `c={ACTIVE_LATTICE_C:.6g} A`, "
        f"source=`{ACTIVE_LATTICE.get('source', '')}`."
    ),
    (
        "Rod reference policy: "
        f"`allow_generated={bool(rod_reference_summary['allow_generated'])}`, "
        f"saved=`{int(rod_reference_summary['saved'])}`, "
        f"generated=`{int(rod_reference_summary['generated'])}`, "
        f"skipped_generated=`{int(rod_reference_summary['skipped_generated'])}`."
    ),
    f"Tilt used for Q conversion: `{ROD_PROFILE_TILT_LABEL}°`",
    f"Source Delta $Q_r$: `{qr_rod_delta_qr_source:.4g}` Å^-1",
    f"Active Delta $Q_r$: `{qr_rod_delta_qr:.4g}` Å^-1 (`{QR_ROD_DELTA_QR_SCALE:.2f}` x source)",
    f"Detector-rotation calibration is Qr-driven for centerline overlay alignment: success=`{bool(rod_qspace_calibration.get('success', False))}`, non-specular fitted peaks=`{int(rod_qspace_calibration.get('anchor_count', 0))}`, active peaks=`{int(rod_qspace_calibration.get('active_anchor_count', 0))}`.",
    f"Detector distance for rod Q-space: `{float(rod_qspace_calibration.get('distance_original_m', distance_m)):.8g}` m -> `{float(rod_qspace_calibration.get('distance_fitted_m', distance_m)):.8g}` m.",
    f"Qr RMS: `{float(rod_qspace_calibration.get('rms_before_qr', np.nan)):.5g}` -> `{float(rod_qspace_calibration.get('rms_after_qr', np.nan)):.5g}` Å^-1; Qz diagnostic RMS: `{float(rod_qspace_calibration.get('rms_before_qz', np.nan)):.5g}` -> `{float(rod_qspace_calibration.get('rms_after_qz', np.nan)):.5g}` Å^-1.",
    f"Displayed/integrated detector support is limited to `2theta <= {ROD_PROFILE_MAX_TWO_THETA_DEG:.1f}°` and `Qz > 0`.",
    "Phi signs: `-` and `+`.",
    "",
    f"The figure uses only the {ROD_PROFILE_TILT_LABEL}° background. Non-specular traces are integrated directly in detector Qr/Qz space.",
    f"Nonzero rods use detector Q maps at `theta_i = {float(ROD_PROFILE_TILT_DEG):.6g}°`.",
    f"Specular `m = 0` Qr integration uses detector Q maps at `theta_i0 = 3*theta_i = {float(specular_detector_theta_initial_deg):.6g}°` and is limited to `L <= {SPECULAR_QR_ROD_L_MAX:g}`.",
    "Before integration, each non-specular HK rod center is adjusted from fitted detector peak Qr samples; every Qz bin then uses the adjusted Qr0 +/- delta_Qr, branch, positive Qz, and the 2theta display limit before summing intensity.",
    "The detector-region figure is a detector-space Qr overlay diagnostic: the background is linear detector intensity with robust percentile clipping, translucent ribbons show the active Delta Qr support with dashed boundary strokes, solid curves are projected fitted-geometry rod centerlines, and solid-white m labels start from the default geometry before manual adjustment; the intensity scale is saved as a separate file.",
    "The plotted traces are acceptance-normalized detector-count densities unless BACKGROUND_SOLID_ANGLE_CORRECTION is enabled. Raw summed columns are retained for audit only.",
    f"Solid-angle correction enabled: `{bool(BACKGROUND_SOLID_ANGLE_CORRECTION)}`.",
    f"Qr-sideband transverse background subtraction enabled: `{bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED)}`. When enabled, `background_density_raw` is the central rod-band density, `qr_sideband_background_density` is the same-Qz off-rod estimate, and `background_density` is their difference.",
    f"PbI2 no-background debug mode: `{bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION)}`. When enabled, PbI2 Qr-rod transverse sideband subtraction is forced off and raw `background_density` is plotted against full `joint_fit_density`.",
    "When caked sum fields are available, density uses sum_signal / sum_normalization. Otherwise it falls back to acceptance weights, then pixel_count.",
    "For nonzero HK rods outside the PbI2 Qr-sideband plot policy, plotted `Data` is `background_density - joint_linear_baseline_density` unless Qr-sideband subtraction is enabled; with sideband subtraction, plotted `Data` is sideband-corrected `background_density`.",
    "For PbI2 nonzero rods with Qr-sideband subtraction, plotted `Data` is raw central `background_density_raw` and the dashed `Fit` is `joint_fit_density + qr_sideband_background_density`. Marker/L mapping and Qz-baseline cancellation checks are reported as diagnostics instead of suppressing available overlays.",
    f"PbI2 Qr-rod profile plots use log-scaled intensity on all panels and cap the displayed L axis at `{PBI2_ROD_PROFILE_L_AXIS_MAX:g}`.",
    "For non-PbI2 nonzero rods, the plotted dashed `Simulation` remains peak-only `joint_peak_density`.",
    "For m=0 only, plotted `Data` is raw `background_density` and plotted `Simulation` is `joint_peak_density + joint_linear_baseline_density`.",
    "The CSV includes `joint_peak_density`, `joint_linear_baseline_density`, and `joint_fit_density = joint_peak_density + joint_linear_baseline_density`.",
    "The CSV includes `peak_subtracted_density = background_density - joint_peak_density`; this fitted-peak removal is the only subtraction product. `fit_residual_density = background_density - joint_fit_density` is reported only as a fit diagnostic.",
    "The detector-space fit model remains in the CSV as `fit_density`. Individual component distributions are saved to the component CSV but not drawn.",
    "Subplot labels show `m = H^2 + H*K + K^2`; marker tick labels show compressed (HK,L) values for the fitted points.",
    "Nonzero-m masks still use detector-space Qr/Qz internally and extend to the projected sign endpoint.",
    "The detector-region figure labels the specular rod as `m = 0`; the displayed support is the same detector Qr/Qz region used for profile extraction.",
    "",
    "## Rods",
    "",
    "| HK | source | generated | saved Qr | active fit peaks | fit samples | method | marker count |",
    "|---:|:---|:---:|---:|---:|---:|:---|---:|",
]
for rod in rod_entries:
    marker_count = int((marker_table["m"] == int(rod["m"])).sum()) if not marker_table.empty else 0
    rod_note_lines.append(
        f"| {int(rod['m'])} | {rod.get('qr_source', 'saved_q_group_rows')} | {bool(rod.get('generated', False))} | {float(rod['qr']):.5g} | "
        f"{int(rod.get('qr_fit_count', 0))} | {int(rod.get('qr_fit_sample_count', 0))} | {rod.get('qr_fit_method', 'saved_q_group_rows')} | {marker_count} |"
    )
rod_note_lines.extend(
    [
        "",
        "Specular `(0,L)` rod uses `Qr = 0` and the dynamic specular strip.",
        "",
        "## Fitted detector geometry",
        "",
        "| parameter | original | fitted |",
        "|:---|---:|---:|",
    ]
)
rod_note_lines.append(
    f"| theta_i_deg | {float(rod_qspace_calibration.get('theta_original_deg', ROD_PROFILE_TILT_DEG)):.6g} | {float(rod_qspace_calibration.get('theta_fitted_deg', ROD_PROFILE_TILT_DEG)):.6g} |"
)
rod_note_lines.append(
    f"| detector_distance_m | {float(rod_qspace_calibration.get('distance_original_m', distance_m)):.8g} | {float(rod_qspace_calibration.get('distance_fitted_m', distance_m)):.8g} |"
)
rod_note_lines.append(
    f"| gamma_deg | {float(rod_qspace_calibration.get('gamma_original_deg', 0.0)):.6g} | {float(rod_qspace_calibration.get('gamma_fitted_deg', 0.0)):.6g} |"
)
rod_note_lines.append(
    f"| Gamma_deg | {float(rod_qspace_calibration.get('Gamma_original_deg', 0.0)):.6g} | {float(rod_qspace_calibration.get('Gamma_fitted_deg', 0.0)):.6g} |"
)
rod_note_lines.append(
    f"| rotation_bound_hit | {bool(rod_qspace_calibration.get('rotation_bound_hit', False))} | {bool(rod_qspace_calibration.get('rotation_bound_hit', False))} |"
)
if profile_failures:
    print("profile notes:")
    for item in profile_failures:
        print(" -", item)


def plot_tail_component_distributions(
    ax: object,
    component_table: pd.DataFrame,
    *,
    m_value: int,
    branch_value: str,
    scale: float,
) -> None:
    if component_table.empty or not np.isfinite(scale) or float(scale) <= 0.0:
        return
    sub_components = component_table[
        (np.asarray(component_table["m"], dtype=int) == int(m_value))
        & (component_table["branch"].astype(str) == str(branch_value))
    ].copy()
    if sub_components.empty:
        return
    for _component_id, component in sub_components.groupby("component_id", sort=True):
        x_component = qz_values_to_l_axis(
            component["qz_center"], m_value=int(m_value), branch_value=str(branch_value)
        )
        y_component = np.asarray(component["component_density"], dtype=np.float64) / float(scale)
        finite_component = np.isfinite(x_component) & np.isfinite(y_component)
        if np.count_nonzero(finite_component) < 2:
            continue
        order_component = np.argsort(x_component[finite_component])
        ax.plot(
            x_component[finite_component][order_component],
            y_component[finite_component][order_component],
            color="0.45",
            linewidth=0.55,
            alpha=0.58,
            zorder=2,
        )


drawable_profile_keys = drawable_rod_profile_keys(rod_profile_table, plot_marker_table)
detector_plot_rod_entries = detector_complete_branch_rod_entries(rod_entries, region_overlays)
plot_rod_entries = [
    rod
    for rod in detector_plot_rod_entries
    if any((int(rod["m"]), branch_name) in drawable_profile_keys for branch_name, *_ in phi_windows)
]
detector_plot_rod_keys = {rod_identity_key(rod) for rod in detector_plot_rod_entries}
skipped_incomplete_detector_hk = [
    int(rod["m"]) for rod in rod_entries if rod_identity_key(rod) not in detector_plot_rod_keys
]
skipped_empty_plot_hk = [
    int(rod["m"])
    for rod in detector_plot_rod_entries
    if not any(
        (int(rod["m"]), branch_name) in drawable_profile_keys for branch_name, *_ in phi_windows
    )
]
if (0, "qz") in drawable_profile_keys:
    plot_rod_entries.append(specular_rod_entry)
if skipped_incomplete_detector_hk:
    print(
        "skipped incomplete detector-support Qr-rod final figure rows: "
        + ", ".join(f"HK={value}" for value in skipped_incomplete_detector_hk)
    )
if skipped_empty_plot_hk:
    print(
        "skipped empty Qr-rod final figure rows: "
        + ", ".join(f"HK={value}" for value in skipped_empty_plot_hk)
    )
if not plot_rod_entries:
    raise RuntimeError("no drawable Qr-rod profile rows are available for the final figure")
nonzero_plot_rod_entries = [rod for rod in plot_rod_entries if int(rod["m"]) != 0]
plot_model_decision_by_key: dict[tuple[int, str], dict[str, object]] = {}
plot_model_decision_rows: list[dict[str, object]] = []
for rod in nonzero_plot_rod_entries:
    m_value = int(rod["m"])
    for branch_name, *_ in phi_windows:
        branch_value = str(branch_name)
        sub = rod_profile_table[
            (rod_profile_table["m"] == m_value)
            & (rod_profile_table["branch"] == branch_value)
            & (rod_profile_table["pixel_count"] > 0)
        ].copy()
        sub = positive_l_rows(sub, m_value=m_value, branch_value=branch_value)
        if sub.empty:
            continue
        plot_decision = rod_profile_plot_model_decision(
            SAMPLE_STEM,
            m_value,
            branch_value,
            sub,
            plot_marker_table,
            transverse_background_enabled=bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED),
            background_subtraction_disabled=bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION),
            peak_to_data_cancel_ratio=float(PBI2_PLOT_PEAK_TO_DATA_CANCEL_RATIO),
            baseline_to_peak_cancel_ratio=float(PBI2_PLOT_BASELINE_TO_PEAK_CANCEL_RATIO),
        )
        plot_model_decision_by_key[(m_value, branch_value)] = plot_decision
        metrics = dict(plot_decision.get("metrics", {}))
        plot_model_decision_rows.append(
            {
                "m": m_value,
                "branch": branch_value,
                "plot_model": bool(plot_decision.get("plot_model", False)),
                "label": str(plot_decision.get("label") or ""),
                "data_column": str(plot_decision.get("data_column") or ""),
                "density_column": str(plot_decision.get("density_column") or ""),
                "baseline_column": str(plot_decision.get("baseline_column") or ""),
                "reason": str(plot_decision.get("reason", "")),
                "valid_l_mapping": metrics.get("valid_l_mapping", ""),
                "baseline_cancellation_suspected": metrics.get(
                    "baseline_cancellation_suspected", ""
                ),
                "peak_to_data_ratio": float(metrics.get("peak_to_data_ratio", np.nan)),
                "baseline_to_peak_ratio": float(metrics.get("baseline_to_peak_ratio", np.nan)),
            }
        )
if plot_model_decision_rows:
    rod_note_lines.extend(
        [
            "",
            "## Plot model decisions",
            "",
            "| HK | branch | plotted model | data source | fit source | added background | reason | valid L map | Qz-baseline cancellation | peak/data p90 | baseline/peak p90 |",
            "|---:|:---:|:---:|:---|:---|:---|:---|:---:|:---:|---:|---:|",
        ]
    )
    for row in plot_model_decision_rows:
        peak_to_data = float(row["peak_to_data_ratio"])
        baseline_to_peak = float(row["baseline_to_peak_ratio"])
        rod_note_lines.append(
            f"| {int(row['m'])} | {row['branch']} | {row['label'] if row['plot_model'] else 'omitted'} | "
            f"{row['data_column'] or '-'} | {row['density_column'] or '-'} | "
            f"{row['baseline_column'] or '-'} | {row['reason']} | "
            f"{row['valid_l_mapping']} | {row['baseline_cancellation_suspected']} | "
            f"{peak_to_data:.4g} | {baseline_to_peak:.4g} |"
        )
rod_profile_note.write_text("\n".join(rod_note_lines) + "\n", encoding="utf-8")
print(f"saved={rod_profile_note}")
rod_profile_l_axis_limits = shared_rod_profile_l_axis_limits(
    rod_profile_table,
    plot_marker_table,
    nonzero_plot_rod_entries,
    phi_windows,
)
rod_profile_l_axis_limits = rod_profile_l_axis_limits_for_sample(
    SAMPLE_STEM,
    rod_profile_l_axis_limits,
    pbi2_l_max=float(PBI2_ROD_PROFILE_L_AXIS_MAX),
)
rod_profile_nonzero_y_axis_limits = shared_nonzero_rod_profile_y_axis_limits(
    rod_profile_table,
    plot_marker_table,
    nonzero_plot_rod_entries,
    phi_windows,
)
rod_profile_hk0_l_axis_limits = rod_profile_l_axis_limits_for_sample(
    SAMPLE_STEM,
    (0.0, SPECULAR_QR_ROD_L_MAX),
    pbi2_l_max=float(PBI2_ROD_PROFILE_L_AXIS_MAX),
)
last_nonzero_plot_row = max(
    (row for row, rod in enumerate(plot_rod_entries) if int(rod["m"]) != 0),
    default=-1,
)
has_hk0_profile_row = any(int(rod["m"]) == 0 for rod in plot_rod_entries)
fig = plt.figure(
    figsize=(JOURNAL_FULL_WIDTH_IN, max(2.05 * len(plot_rod_entries), 5.0)),
    constrained_layout=False,
)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.075, top=0.955)
if nonzero_plot_rod_entries and has_hk0_profile_row:
    profile_grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[float(len(nonzero_plot_rod_entries)), 1.0],
        hspace=0.18,
    )
    nonzero_profile_grid = profile_grid[0].subgridspec(
        len(nonzero_plot_rod_entries), len(phi_windows), wspace=0.0, hspace=0.0
    )
    hk0_profile_grid = profile_grid[1].subgridspec(1, len(phi_windows))
elif nonzero_plot_rod_entries:
    profile_grid = fig.add_gridspec(1, 1)
    nonzero_profile_grid = profile_grid[0].subgridspec(
        len(nonzero_plot_rod_entries), len(phi_windows), wspace=0.0, hspace=0.0
    )
    hk0_profile_grid = None
else:
    profile_grid = fig.add_gridspec(1, len(phi_windows))
    nonzero_profile_grid = None
    hk0_profile_grid = profile_grid
axes = np.empty((len(plot_rod_entries), len(phi_windows)), dtype=object)
pbi2_rod_profile_figure = sample_uses_pbi2_rod_plot_policy(SAMPLE_STEM)
nonzero_profile_row = 0
for row, rod in enumerate(plot_rod_entries):
    if int(rod["m"]) == 0:
        ax = fig.add_subplot(hk0_profile_grid[0, :])
        axes[row, 0] = ax
        axes[row, 1] = ax
        ax.set_xlim(*rod_profile_hk0_l_axis_limits)
        sub = rod_profile_table[
            (rod_profile_table["m"] == 0)
            & (rod_profile_table["branch"] == "qz")
            & (rod_profile_table["pixel_count"] > 0)
        ].copy()
        sub = positive_l_rows(sub, m_value=0, branch_value="qz")
        hk0_positive_y = np.asarray([], dtype=np.float64)
        if not sub.empty:
            norm_payload = normalized_data_simulation_payload(
                sub["background_density"],
                sub.get("joint_peak_density", sub["fit_density"]),
                sub.get("joint_linear_baseline_density", None),
                subtract_baseline_from_data=False,
            )
            data_norm = np.asarray(norm_payload["data"], dtype=np.float64)
            simulation_norm = np.asarray(norm_payload["simulation"], dtype=np.float64)
            x = qz_values_to_l_axis(sub["qz_center"], m_value=0, branch_value="qz")
            order = np.argsort(x)
            x = x[order]
            data_norm = data_norm[order]
            simulation_norm = simulation_norm[order]
            hk0_visible = (x >= rod_profile_hk0_l_axis_limits[0]) & (
                x <= rod_profile_hk0_l_axis_limits[1]
            )
            data_plot = (
                positive_log_plot_values(data_norm) if pbi2_rod_profile_figure else data_norm
            )
            simulation_plot = (
                positive_log_plot_values(simulation_norm)
                if pbi2_rod_profile_figure
                else simulation_norm
            )
            ax.plot(
                x,
                data_plot,
                color=JOURNAL_DATA_COLOR,
                linewidth=1.0,
                alpha=0.92,
                label="Data",
                zorder=4,
            )
            ax.plot(
                x,
                simulation_plot,
                color=JOURNAL_FIT_COLOR,
                linewidth=1.0,
                alpha=0.96,
                linestyle="--",
                label="Simulation",
                zorder=5,
            )
            hk0_y_values = np.concatenate([data_norm[hk0_visible], simulation_norm[hk0_visible]])
            hk0_positive_y = hk0_y_values[np.isfinite(hk0_y_values) & (hk0_y_values > 0.0)]
        ax.grid(True, color=JOURNAL_GRID_COLOR, linewidth=0.45)
        ax.set_yscale("log")
        if hk0_positive_y.size:
            hk0_y_min = max(float(np.nanmin(hk0_positive_y)) * 0.80, 1.0e-6)
            hk0_y_max = max(float(np.nanmax(hk0_positive_y)) * 1.18, hk0_y_min * 10.0)
            ax.set_ylim(hk0_y_min, hk0_y_max)
        ax.text(
            0.5,
            0.985,
            rod_profile_panel_label(0, "qz"),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.4,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=8.0,
        )
        ax.set_xlabel(r"$L$")
        ax.set_ylabel("Intensity (a.u.)")
        ax.tick_params(length=2.0, width=0.45, labelsize=6.1, direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        continue
    for col, (branch_name, _phi_min, _phi_max, _phi_sign_label) in enumerate(phi_windows):
        ax = fig.add_subplot(nonzero_profile_grid[nonzero_profile_row, col])
        axes[row, col] = ax
        ax.set_xlim(*rod_profile_l_axis_limits)
        if not pbi2_rod_profile_figure:
            ax.set_ylim(*rod_profile_nonzero_y_axis_limits)
        nonzero_log_y_series: list[np.ndarray] = []
        sub = rod_profile_table[
            (rod_profile_table["m"] == int(rod["m"]))
            & (rod_profile_table["branch"] == branch_name)
            & (rod_profile_table["pixel_count"] > 0)
        ].copy()
        sub = positive_l_rows(sub, m_value=int(rod["m"]), branch_value=branch_name)
        if not sub.empty:
            plot_decision = plot_model_decision_by_key.get(
                (int(rod["m"]), str(branch_name)),
                rod_profile_plot_model_decision(
                    SAMPLE_STEM,
                    int(rod["m"]),
                    str(branch_name),
                    sub,
                    plot_marker_table,
                    transverse_background_enabled=bool(QR_ROD_TRANSVERSE_BACKGROUND_ENABLED),
                    background_subtraction_disabled=bool(PBI2_DISABLE_BACKGROUND_SUBTRACTION),
                    peak_to_data_cancel_ratio=float(PBI2_PLOT_PEAK_TO_DATA_CANCEL_RATIO),
                    baseline_to_peak_cancel_ratio=float(PBI2_PLOT_BASELINE_TO_PEAK_CANCEL_RATIO),
                ),
            )
            plot_model, norm_payload = rod_profile_normalized_payload_for_plot_decision(
                sub, plot_decision
            )
            data_norm = np.asarray(norm_payload["data"], dtype=np.float64)
            simulation_norm = np.asarray(norm_payload["simulation"], dtype=np.float64)
            x = qz_values_to_l_axis(
                sub["qz_center"], m_value=int(rod["m"]), branch_value=branch_name
            )
            order = np.argsort(x)
            x = x[order]
            data_norm = data_norm[order]
            simulation_norm = simulation_norm[order]
            visible = (x >= rod_profile_l_axis_limits[0]) & (x <= rod_profile_l_axis_limits[1])
            data_plot = (
                positive_log_plot_values(data_norm) if pbi2_rod_profile_figure else data_norm
            )
            if pbi2_rod_profile_figure:
                nonzero_log_y_series.append(data_norm[visible])
            ax.plot(
                x,
                data_plot,
                color=JOURNAL_DATA_COLOR,
                linewidth=1.0,
                alpha=0.92,
                label="Data",
                zorder=4,
            )
            if plot_model:
                simulation_plot = (
                    positive_log_plot_values(simulation_norm)
                    if pbi2_rod_profile_figure
                    else simulation_norm
                )
                if pbi2_rod_profile_figure:
                    nonzero_log_y_series.append(simulation_norm[visible])
                ax.plot(
                    x,
                    simulation_plot,
                    color=JOURNAL_FIT_COLOR,
                    linewidth=1.0,
                    alpha=0.96,
                    linestyle="--",
                    label=str(plot_decision.get("label", "Fit")),
                    zorder=5,
                )
        if pbi2_rod_profile_figure:
            apply_positive_log_y_axis(ax, *nonzero_log_y_series)
        else:
            ax.axhline(0.0, color="0.80", linewidth=0.45)
        ax.grid(True, color=JOURNAL_GRID_COLOR, linewidth=0.45)
        ax.text(
            0.5,
            0.985,
            rod_profile_panel_label(int(rod["m"]), branch_name),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.4,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            zorder=8.0,
        )
        ax.tick_params(
            length=2.0,
            width=0.45,
            labelsize=6.1,
            direction="in",
            top=True,
            right=True,
            labelleft=col == 0,
            labelbottom=row == last_nonzero_plot_row,
        )
        if col == 0:
            ax.set_ylabel("Intensity (a.u.)")
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    nonzero_profile_row += 1

plot_model_labels = sorted(
    {
        str(decision.get("label"))
        for decision in plot_model_decision_by_key.values()
        if bool(decision.get("plot_model")) and str(decision.get("label") or "")
    }
)
if not plot_model_labels and has_hk0_profile_row:
    plot_model_labels = ["Simulation"]
legend_handles = [
    mpl.lines.Line2D([], [], color=JOURNAL_DATA_COLOR, linewidth=1.0, label="Data"),
    *[
        mpl.lines.Line2D(
            [], [], color=JOURNAL_FIT_COLOR, linewidth=1.0, linestyle="--", label=label
        )
        for label in plot_model_labels
    ],
]
legend_ax = axes[0, -1]
legend_ax.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=6.1,
    frameon=False,
    handlelength=1.45,
    borderaxespad=0.25,
)
maybe_suptitle(fig, rf"{SAMPLE_LABEL}: $Q_r$ rod $L$ profiles", y=1.02)
rod_profile_png, rod_profile_pdf = save_manuscript_figure(fig, ROD_PROFILE_STEM)
plt.close(fig)
print(f"saved={rod_profile_png}")
print(f"saved={rod_profile_pdf}")
display(Image(filename=str(rod_profile_png)))

detector_region_display_mask = detector_phi_display_mask(profile_detector_phi_map)
qr_map, qz_map, valid_q_map = profile_detector_q_maps
detector_region_positive_qz_mask = positive_qz_mask(qz_map)
detector_region_shape = tuple(np.asarray(profile_bg["detector_image"]).shape)
detector_height, detector_width = int(detector_region_shape[0]), int(detector_region_shape[1])
region_colors = JOURNAL_REGION_COLORS
detector_region_shape_mask = (
    np.isfinite(profile_detector_two_theta_map)
    & (
        np.asarray(profile_detector_two_theta_map, dtype=np.float64)
        <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)
    )
    & detector_region_display_mask
    & detector_region_positive_qz_mask
)
detector_region_values = np.asarray(profile_bg["detector_image"], dtype=np.float64)
detector_region_bg = np.ma.masked_where(
    ~detector_region_shape_mask | ~np.isfinite(detector_region_values), detector_region_values
)
detector_region_xlim, detector_region_ylim = detector_display_bbox(detector_region_shape_mask)
detector_region_roi = np.asarray(detector_region_bg.compressed(), dtype=np.float64)
detector_region_roi = detector_region_roi[np.isfinite(detector_region_roi)]
if detector_region_roi.size:
    detector_region_vmin, detector_region_vmax = np.nanpercentile(detector_region_roi, [1.0, 99.7])
    if not np.isfinite(detector_region_vmin):
        detector_region_vmin = float(np.nanmin(detector_region_roi))
    if not np.isfinite(detector_region_vmax) or detector_region_vmax <= detector_region_vmin:
        detector_region_vmax = float(np.nanmax(detector_region_roi))
    if not np.isfinite(detector_region_vmax) or detector_region_vmax <= detector_region_vmin:
        detector_region_vmax = detector_region_vmin + 1.0
else:
    detector_region_vmin, detector_region_vmax = 0.0, 1.0
detector_region_cmap = mpl.colormaps[JOURNAL_DETECTOR_CMAP].copy()
detector_region_cmap.set_bad("#050505")
detector_region_centerline_lw = 1.1
detector_region_band_alpha = 0.11
detector_region_boundary_alpha = 0.72
detector_region_boundary_lw = 1.45
detector_region_boundary_linestyle = "dashed"
detector_region_specular_centerline_lw = detector_region_centerline_lw
detector_region_specular_band_alpha = 0.42
detector_region_specular_boundary_alpha = 1.0
detector_region_specular_boundary_expand_px = 2
detector_region_label_fontsize = 8.6


def save_detector_region_intensity_scale(
    cmap: object,
    *,
    vmin: float,
    vmax: float,
    stem: str,
) -> tuple[Path, Path]:
    scale_fig, scale_ax = plt.subplots(figsize=(0.55, 2.35), constrained_layout=True)
    norm = mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    scale_cbar = scale_fig.colorbar(mappable, cax=scale_ax, orientation="vertical")
    scale_cbar.set_label("Intensity", fontsize=6.5)
    scale_cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.45)
    paths = save_manuscript_figure(scale_fig, stem)
    plt.close(scale_fig)
    return paths


def detector_rod_label(m_value: int, branch_suffix: str) -> str:
    return f"m = {int(m_value)} {str(branch_suffix).strip()}"


def detector_specular_label() -> str:
    return "m = 0"


def detector_rod_label_id(m_value: int, branch_suffix: str) -> str:
    return f"m={int(m_value)}:{str(branch_suffix).strip()}"


def detector_specular_label_id() -> str:
    return "m=0"


def detector_label_id(entry: dict[str, object]) -> str:
    existing = str(entry.get("label_id", "")).strip()
    if existing:
        return existing
    text = str(entry.get("text", "")).strip()
    match = re.fullmatch(r"m\s*=\s*(-?\d+)\s*([+-])?", text)
    if match:
        suffix = "" if match.group(2) is None else f":{match.group(2)}"
        return f"m={int(match.group(1))}{suffix}"
    return re.sub(r"\s+", "", text) or "label"


def detector_label_fontsize(entry: dict[str, object]) -> float:
    default = float(globals().get("detector_region_label_fontsize", 11.0))
    try:
        value = float(entry.get("fontsize", default))
    except Exception:
        value = default
    if not np.isfinite(value):
        value = default
    return float(np.clip(value, 4.0, 32.0))


def detector_label_xy(entry: dict[str, object]) -> np.ndarray | None:
    if "label_xy" not in entry:
        return None
    pos = np.asarray(entry.get("label_xy"), dtype=np.float64).reshape(-1)
    if pos.size < 2 or not np.all(np.isfinite(pos[:2])):
        return None
    return pos[:2].copy()


def detector_label_settings_payload(label_entries: list[dict[str, object]]) -> dict[str, object]:
    labels: list[dict[str, object]] = []
    for entry in label_entries:
        payload: dict[str, object] = {
            "label_id": detector_label_id(entry),
            "text": str(entry.get("text", "")),
            "fontsize": detector_label_fontsize(entry),
        }
        pos = detector_label_xy(entry)
        if pos is not None:
            payload["label_xy"] = [float(pos[0]), float(pos[1])]
        labels.append(payload)
    return {"schema": "ra_sim.detector_label_settings.v1", "labels": labels}


def apply_detector_label_settings(
    label_entries: list[dict[str, object]], payload: object
) -> list[dict[str, object]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "ra_sim.detector_label_settings.v1"
    ):
        return [dict(entry) for entry in label_entries]
    lookup: dict[str, dict[str, object]] = {}
    for item in payload.get("labels", []):
        if isinstance(item, dict):
            label_id = str(item.get("label_id", "")).strip()
            if label_id:
                lookup[label_id] = item
    updated: list[dict[str, object]] = []
    for entry in label_entries:
        new_entry = dict(entry)
        settings = lookup.get(detector_label_id(new_entry))
        if settings is not None:
            pos = detector_label_xy(settings)
            if pos is not None:
                new_entry["label_xy"] = pos
            if "fontsize" in settings:
                new_entry["fontsize"] = detector_label_fontsize(settings)
        updated.append(new_entry)
    return updated


def load_detector_label_settings(path: object) -> dict[str, object]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def save_detector_label_settings(path: object, label_entries: list[dict[str, object]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(detector_label_settings_payload(label_entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def draw_detector_region_label_artists(
    ax: object, label_entries: list[dict[str, object]]
) -> list[object]:
    artists: list[object] = []
    for index, entry in enumerate(label_entries):
        if entry.get("label_xy", None) is None:
            continue
        pos = detector_label_xy(entry)
        if pos is None:
            continue
        artist = ax.text(
            float(pos[0]),
            float(pos[1]),
            str(entry.get("text", "")),
            color="white",
            fontsize=detector_label_fontsize(entry),
            fontweight="semibold",
            ha="center",
            va="center",
            zorder=9.0,
            clip_on=True,
        )
        artist._ra_sim_label_entry_index = int(index)
        artists.append(artist)
    return artists


def detector_label_edit_runtime_mode(
    mode: object = "auto", *, backend_name: object = None, env: dict[str, str] | None = None
) -> str:
    return qr_rod_peak_edit_runtime_mode(mode, backend_name=backend_name, env=env)


def detector_xy_from_caked_angles(
    bg: dict[str, object], two_theta_deg: float, phi_deg: float
) -> tuple[float, float] | None:
    ai_value = bg.get("ai")
    if ai_value is None:
        ai_value = FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center_row_px) * PIXEL_SIZE_M,
            poni2=float(center_col_px) * PIXEL_SIZE_M,
            pixel1=PIXEL_SIZE_M,
            pixel2=PIXEL_SIZE_M,
            wavelength=WAVELENGTH_M,
        )
    col, row = caked_point_to_detector_pixel(
        ai_value,
        tuple(bg["detector_image"].shape),
        np.asarray(bg["theta_axis"], dtype=np.float64),
        np.asarray(bg["phi_axis"], dtype=np.float64),
        float(two_theta_deg),
        float(phi_deg),
        transform_bundle=bg["transform_bundle"],
        engine=EXACT_CAKE_ENGINE,
        workers=CAKE_WORKERS,
    )
    if col is None or row is None or not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def projected_qr_detector_trace_payloads(
    rod: dict[str, object], *, config: object
) -> list[dict[str, object]]:
    if project_qr_cylinder_to_detector is None:
        return []
    geometry = intersection_geometry_from_qspace_config(config)
    if geometry is None:
        return []
    try:
        traces = project_qr_cylinder_to_detector(
            qr_value=float(rod["qr"]),
            geometry=geometry,
            wavelength=float(config.wavelength),
            n2=config.n2,
            beam_x=config_float_attr(config, "beam_x", 0.0),
            beam_y=config_float_attr(config, "beam_y", 0.0),
            dtheta=config_float_attr(config, "dtheta", 0.0),
            dphi=config_float_attr(config, "dphi", 0.0),
            phi_samples=max(2401, int(getattr(config, "phi_samples", rod_phi_samples))),
        )
    except Exception:
        return []
    payloads: list[dict[str, object]] = []
    for trace in traces:
        col = np.asarray(trace.detector_col, dtype=np.float64).copy()
        row = np.asarray(trace.detector_row, dtype=np.float64).copy()
        qz = np.asarray(getattr(trace, "qz", np.full(col.shape, np.nan)), dtype=np.float64).copy()
        valid = np.asarray(trace.valid_mask, dtype=bool) & np.isfinite(col) & np.isfinite(row)
        if np.count_nonzero(valid) < 2:
            continue
        col[~valid] = np.nan
        row[~valid] = np.nan
        qz[~valid] = np.nan
        payloads.append(
            {
                "x_line": col,
                "y_line": row,
                "qz_line": qz,
                "branch_sign": int(getattr(trace, "branch_sign", 0)),
            }
        )
    return payloads


def projected_qr_detector_lines(
    rod: dict[str, object], *, config: object
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (
            np.asarray(payload["x_line"], dtype=np.float64),
            np.asarray(payload["y_line"], dtype=np.float64),
        )
        for payload in projected_qr_detector_trace_payloads(rod, config=config)
    ]


def clipped_detector_trace_to_qz_bounds(
    x_line: object,
    y_line: object,
    qz_line: object,
    qz_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_line, dtype=np.float64).copy()
    y = np.asarray(y_line, dtype=np.float64).copy()
    qz = np.asarray(qz_line, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size != qz.size:
        return (
            np.full(np.asarray(x_line, dtype=np.float64).shape, np.nan, dtype=np.float64),
            np.full(np.asarray(y_line, dtype=np.float64).shape, np.nan, dtype=np.float64),
        )
    qz_min, qz_max = sorted((float(qz_bounds[0]), float(qz_bounds[1])))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(qz) & (qz >= qz_min) & (qz <= qz_max)
    x[~keep] = np.nan
    y[~keep] = np.nan
    return x, y


def detector_qz_values_for_polyline(
    x_line: object,
    y_line: object,
    detector_qz_map: object,
) -> np.ndarray:
    x = np.asarray(x_line, dtype=np.float64).reshape(-1)
    y = np.asarray(y_line, dtype=np.float64).reshape(-1)
    qz_map_values = np.asarray(detector_qz_map, dtype=np.float64)
    qz = np.full(x.shape, np.nan, dtype=np.float64)
    if x.shape != y.shape or qz_map_values.ndim != 2:
        return qz
    finite = np.isfinite(x) & np.isfinite(y)
    cols = np.zeros(x.shape, dtype=np.int64)
    rows = np.zeros(y.shape, dtype=np.int64)
    cols[finite] = np.rint(x[finite]).astype(np.int64, copy=False)
    rows[finite] = np.rint(y[finite]).astype(np.int64, copy=False)
    valid = (
        finite
        & (rows >= 0)
        & (rows < qz_map_values.shape[0])
        & (cols >= 0)
        & (cols < qz_map_values.shape[1])
    )
    if np.any(valid):
        qz[valid] = qz_map_values[rows[valid], cols[valid]]
    return qz


def detector_point_in_region(x: object, y: object) -> bool:
    x_value = as_float(x)
    y_value = as_float(y)
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return False
    row_idx = int(np.rint(y_value))
    col_idx = int(np.rint(x_value))
    if (
        row_idx < 0
        or row_idx >= detector_region_shape[0]
        or col_idx < 0
        or col_idx >= detector_region_shape[1]
    ):
        return False
    return bool(detector_region_shape_mask[row_idx, col_idx])


def point_to_projected_polyline_distance_px(
    x_value: float,
    y_value: float,
    lines: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if not (np.isfinite(x_value) and np.isfinite(y_value)):
        return float("nan")
    point = np.asarray([float(x_value), float(y_value)], dtype=np.float64)
    best = float("inf")
    for x_line, y_line in lines:
        x = np.asarray(x_line, dtype=np.float64)
        y = np.asarray(y_line, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        finite_indices = np.flatnonzero(finite)
        if finite_indices.size < 2:
            continue
        breaks = np.flatnonzero(np.diff(finite_indices) > 1) + 1
        for run in np.split(finite_indices, breaks):
            if run.size < 2:
                continue
            pts = np.column_stack([x[run], y[run]])
            a = pts[:-1]
            b = pts[1:]
            segment = b - a
            length2 = np.sum(segment * segment, axis=1)
            good = np.isfinite(length2) & (length2 > 1.0e-12)
            if not np.any(good):
                continue
            a_good = a[good]
            segment_good = segment[good]
            length2_good = length2[good]
            t = np.sum((point[None, :] - a_good) * segment_good, axis=1) / length2_good
            t = np.clip(t, 0.0, 1.0)
            closest = a_good + t[:, None] * segment_good
            distances = np.hypot(closest[:, 0] - point[0], closest[:, 1] - point[1])
            finite_distances = distances[np.isfinite(distances)]
            if finite_distances.size:
                best = min(best, float(np.nanmin(finite_distances)))
    return best if np.isfinite(best) else float("nan")


def specular_detector_lines_from_markers(
    table: pd.DataFrame,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if table.empty or "refined_two_theta_deg" not in table or "refined_phi_deg" not in table:
        return []
    sort_key = "qz_marker" if "qz_marker" in table else "refined_two_theta_deg"
    points: list[tuple[float, float]] = []
    for _idx, marker in table.sort_values(sort_key).iterrows():
        xy = detector_xy_from_caked_angles(
            profile_bg,
            float(marker["refined_two_theta_deg"]),
            float(marker["refined_phi_deg"]),
        )
        if xy is not None and detector_point_in_region(xy[0], xy[1]):
            points.append((float(xy[0]), float(xy[1])))
    if len(points) < 2:
        return []
    pts = np.asarray(points, dtype=np.float64)
    return [(pts[:, 0], pts[:, 1])]


def specular_detector_centerline_fallback(
    sample_count: int = 320,
) -> list[tuple[np.ndarray, np.ndarray]]:
    phi_candidates: list[float] = []
    if not specular_l_marker_table.empty and "refined_phi_deg" in specular_l_marker_table:
        phi_values = np.asarray(specular_l_marker_table["refined_phi_deg"], dtype=np.float64)
        phi_candidates.extend([float(value) for value in phi_values[np.isfinite(phi_values)]])
    phi_center = (
        float(np.nanmedian(phi_candidates))
        if phi_candidates
        else 0.5 * (float(specular_phi_min_deg) + float(specular_phi_max_deg))
    )
    theta_values = np.linspace(
        float(specular_theta_min_deg),
        float(specular_theta_max_deg),
        int(sample_count),
        dtype=np.float64,
    )
    points: list[tuple[float, float]] = []
    for theta_value in theta_values:
        xy = detector_xy_from_caked_angles(profile_bg, float(theta_value), phi_center)
        if xy is not None and detector_point_in_region(xy[0], xy[1]):
            points.append((float(xy[0]), float(xy[1])))
        elif points:
            points.append((np.nan, np.nan))
    finite_count = sum(
        1 for x_value, y_value in points if np.isfinite(x_value) and np.isfinite(y_value)
    )
    if finite_count < 2:
        return []
    pts = np.asarray(points, dtype=np.float64)
    return [(pts[:, 0], pts[:, 1])]


def expanded_detector_mask(mask: np.ndarray, radius_px: int = 0) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    radius = max(int(radius_px), 0)
    if radius == 0 or mask_array.ndim != 2 or not np.any(mask_array):
        return mask_array.copy()
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return np.asarray(binary_dilation(mask_array, structure=structure), dtype=bool)


def draw_detector_mask_layer(mask: np.ndarray, color: str, *, alpha: float, zorder: float) -> None:
    mask = np.asarray(mask, dtype=bool) & detector_region_shape_mask
    if mask.shape != detector_region_shape or not np.any(mask):
        return
    layer = np.ma.masked_where(~mask, np.ones(detector_region_shape, dtype=np.float64))
    ax.imshow(
        layer,
        origin="upper",
        aspect="equal",
        cmap=ListedColormap([color]),
        alpha=float(alpha),
        interpolation="nearest",
        rasterized=True,
        zorder=zorder,
    )


def draw_detector_boundary_contour(
    mask: np.ndarray,
    color: str,
    *,
    alpha: float,
    zorder: float,
) -> None:
    boundary = np.asarray(mask, dtype=bool) & detector_region_shape_mask
    if boundary.shape != detector_region_shape or not np.any(boundary):
        return
    if not np.any((~boundary) & detector_region_shape_mask):
        return
    values = np.ma.masked_where(~detector_region_shape_mask, np.asarray(boundary, dtype=np.float64))
    ax.contour(
        values,
        levels=[0.5],
        colors=[color],
        linewidths=detector_region_boundary_lw,
        linestyles=detector_region_boundary_linestyle,
        alpha=float(alpha),
        antialiased=True,
        zorder=zorder,
    )


def draw_detector_delta_q_region(
    visual: dict[str, object] | None,
    color: str,
    *,
    fill_alpha: float = detector_region_band_alpha,
    boundary_alpha: float = detector_region_boundary_alpha,
    boundary_expand_px: int = 0,
) -> None:
    if not isinstance(visual, dict):
        return
    fill = np.asarray(visual.get("band_fill_mask"), dtype=bool)
    if fill.shape == detector_region_shape:
        draw_detector_mask_layer(fill, color, alpha=float(fill_alpha), zorder=3.0)
    boundary_inner = np.asarray(visual.get("band_boundary_inner_visible"), dtype=bool)
    boundary_outer = np.asarray(visual.get("band_boundary_outer_visible"), dtype=bool)
    if (
        boundary_inner.shape == detector_region_shape
        and boundary_outer.shape == detector_region_shape
    ):
        boundary_mask = expanded_detector_mask(boundary_inner | boundary_outer, boundary_expand_px)
        draw_detector_boundary_contour(
            boundary_mask,
            color,
            alpha=float(boundary_alpha),
            zorder=4.1,
        )


def detector_mask_centerline_from_visual(
    visual: dict[str, object] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(visual, dict):
        return []
    mask = np.asarray(visual.get("band_fill_mask"), dtype=bool)
    shape_mask = np.asarray(
        globals().get("detector_region_shape_mask", np.ones(mask.shape, dtype=bool)), dtype=bool
    )
    if mask.ndim != 2 or shape_mask.shape != mask.shape:
        return []
    mask = mask & shape_mask
    if not np.any(mask):
        return []
    rows: list[float] = []
    cols: list[float] = []
    for row in np.flatnonzero(np.any(mask, axis=1)):
        row_cols = np.flatnonzero(mask[int(row), :])
        if row_cols.size:
            rows.append(float(row))
            cols.append(float(np.nanmedian(row_cols.astype(np.float64))))
    if len(rows) < 2:
        return []
    return [(np.asarray(cols, dtype=np.float64), np.asarray(rows, dtype=np.float64))]


def visible_polyline_anchor(
    x_line: np.ndarray, y_line: np.ndarray, *, fraction: float = 0.50
) -> tuple[float, float] | None:
    x_values = np.asarray(x_line, dtype=np.float64)
    y_values = np.asarray(y_line, dtype=np.float64)
    x_lo, x_hi = sorted((float(detector_region_xlim[0]), float(detector_region_xlim[1])))
    y_lo, y_hi = sorted((float(detector_region_ylim[0]), float(detector_region_ylim[1])))
    visible = np.asarray(
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values >= x_lo)
        & (x_values <= x_hi)
        & (y_values >= y_lo)
        & (y_values <= y_hi),
        dtype=bool,
    )
    if np.count_nonzero(visible) < 2:
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if np.count_nonzero(finite) < 2:
            return None
        x_values = x_values[finite]
        y_values = y_values[finite]
    else:
        visible_indices = np.flatnonzero(visible)
        breaks = np.where(np.diff(visible_indices) > 1)[0] + 1
        runs = np.split(visible_indices, breaks)
        best_run = max(
            (run for run in runs if run.size >= 2),
            key=lambda run: float(
                np.nansum(np.hypot(np.diff(x_values[run]), np.diff(y_values[run])))
            ),
            default=visible_indices,
        )
        x_values = x_values[best_run]
        y_values = y_values[best_run]
    step = np.hypot(np.diff(x_values), np.diff(y_values))
    if step.size < 1 or not np.any(np.isfinite(step)):
        index = int(np.clip(round(float(fraction) * (x_values.size - 1)), 0, x_values.size - 1))
        return float(x_values[index]), float(y_values[index])
    distance = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))])
    target = float(np.clip(fraction, 0.05, 0.95)) * float(distance[-1])
    index = int(np.nanargmin(np.abs(distance - target)))
    return float(x_values[index]), float(y_values[index])


def clamp_detector_label_position(pos: np.ndarray, *, padding_px: float = 30.0) -> np.ndarray:
    x_lo, x_hi = sorted((float(detector_region_xlim[0]), float(detector_region_xlim[1])))
    y_lo, y_hi = sorted((float(detector_region_ylim[0]), float(detector_region_ylim[1])))
    if x_hi - x_lo > 2.0 * padding_px:
        pos[0] = np.clip(pos[0], x_lo + padding_px, x_hi - padding_px)
    else:
        pos[0] = 0.5 * (x_lo + x_hi)
    if y_hi - y_lo > 2.0 * padding_px:
        pos[1] = np.clip(pos[1], y_lo + padding_px, y_hi - padding_px)
    else:
        pos[1] = 0.5 * (y_lo + y_hi)
    return pos


def low_l_rod_anchor(
    x_line: np.ndarray, y_line: np.ndarray, qz_line: np.ndarray
) -> tuple[float, float] | None:
    x = np.asarray(x_line, dtype=np.float64)
    y = np.asarray(y_line, dtype=np.float64)
    qz = np.asarray(qz_line, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(qz)
    if not np.any(finite):
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return None
        idx = int(np.flatnonzero(finite)[np.argmax(y[finite])])
        return float(x[idx]), float(y[idx])
    finite_indices = np.flatnonzero(finite)
    nonnegative = finite & (qz >= 0.0)
    if np.any(nonnegative):
        candidates = np.flatnonzero(nonnegative)
        idx = int(candidates[np.nanargmin(qz[candidates])])
    else:
        idx = int(finite_indices[np.nanargmin(np.abs(qz[finite_indices]))])
    return float(x[idx]), float(y[idx])


def place_low_l_rod_label(
    ax,
    x_line: np.ndarray,
    y_line: np.ndarray,
    qz_line: np.ndarray,
    text: str,
    color: str,
    *,
    offset_px: float = 16.0,
) -> np.ndarray | None:
    pts = np.column_stack(
        [np.asarray(x_line, dtype=np.float64), np.asarray(y_line, dtype=np.float64)]
    )
    qz = np.asarray(qz_line, dtype=np.float64)
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if pts.shape[0] != qz.shape[0]:
        qz = np.full(pts.shape[0], np.nan, dtype=np.float64)
    anchor = low_l_rod_anchor(pts[:, 0], pts[:, 1], qz)
    if anchor is None or np.count_nonzero(finite) < 2:
        return None
    x_anchor, y_anchor = anchor
    base_pos = clamp_detector_label_position(
        np.array([float(x_anchor), float(y_anchor) + float(offset_px)], dtype=np.float64)
    )
    return base_pos.copy()


def place_rod_label(
    ax,
    x_line: np.ndarray,
    y_line: np.ndarray,
    text: str,
    color: str,
    *,
    beam_center: tuple[float, float],
    offset_px: float = 14.0,
    flip_normal: bool = False,
) -> np.ndarray | None:
    pts = np.column_stack(
        [np.asarray(x_line, dtype=np.float64), np.asarray(y_line, dtype=np.float64)]
    )
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite]
    if pts.shape[0] < 2:
        return None
    anchor = visible_polyline_anchor(pts[:, 0], pts[:, 1])
    if anchor is None:
        return None
    x_anchor, y_anchor = anchor
    d2 = np.square(pts[:, 0] - float(x_anchor)) + np.square(pts[:, 1] - float(y_anchor))
    j = int(np.argmin(d2))
    j0 = max(j - 1, 0)
    j1 = min(j + 1, pts.shape[0] - 1)
    tangent = pts[j1] - pts[j0]
    tangent = tangent / (np.linalg.norm(tangent) + 1.0e-12)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    radial = np.array(
        [float(x_anchor) - float(beam_center[0]), float(y_anchor) - float(beam_center[1])],
        dtype=np.float64,
    )
    if np.dot(normal, radial) < 0.0:
        normal = -normal
    if bool(flip_normal):
        normal = -normal
    base_pos = clamp_detector_label_position(
        np.array([float(x_anchor), float(y_anchor)], dtype=np.float64) + float(offset_px) * normal
    )
    return base_pos.copy()


def show_detector_region_label_position_popup(
    fig: object,
    ax: object,
    label_entries: list[dict[str, object]],
    *,
    settings_path: object = None,
) -> tuple[list[dict[str, object]], bool]:
    try:
        from matplotlib.widgets import Button, TextBox
    except Exception as exc:
        raise RuntimeError(f"Matplotlib detector label editor widgets are unavailable: {exc}") from exc

    edited = [dict(entry) for entry in label_entries]
    original = [dict(entry) for entry in label_entries]
    drawable_indices = [
        index for index, entry in enumerate(edited) if detector_label_xy(entry) is not None
    ]
    if not drawable_indices:
        return edited, True

    drawable_index_set = set(drawable_indices)
    settings_text = "" if settings_path is None else str(settings_path).strip()
    selected: dict[str, object] = {
        "index": int(drawable_indices[0]),
        "accepted": True,
        "dragging": False,
        "drag_offset_x": 0.0,
        "drag_offset_y": 0.0,
    }
    title_box_state: dict[str, object] = {"box": None, "syncing": False}
    label_artists: list[object] = []
    control_axes: list[object] = []
    key_nudges = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
    }

    def active_index() -> int:
        return int(selected["index"])

    def active_position() -> np.ndarray | None:
        return detector_label_xy(edited[active_index()])

    def remove_label_artists() -> None:
        while label_artists:
            artist = label_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def sync_controls() -> None:
        box = title_box_state.get("box")
        if box is None:
            return
        title_box_state["syncing"] = True
        try:
            box.set_val(str(edited[active_index()].get("text", "")))
        finally:
            title_box_state["syncing"] = False

    def redraw_labels() -> None:
        remove_label_artists()
        label_artists.extend(draw_detector_region_label_artists(ax, edited))
        active = active_index()
        for artist in label_artists:
            try:
                artist.set_picker(True)
            except Exception:
                pass
            if int(getattr(artist, "_ra_sim_label_entry_index", -1)) == active:
                try:
                    artist.set_bbox(
                        {
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "none",
                            "edgecolor": "#00d5ff",
                            "linewidth": 0.8,
                        }
                    )
                except Exception:
                    pass
        sync_controls()
        fig.canvas.draw_idle()

    def set_label_position(index: int, x_value: object, y_value: object) -> None:
        try:
            x_new = float(x_value)
            y_new = float(y_value)
        except Exception:
            return
        if not (np.isfinite(x_new) and np.isfinite(y_new)):
            return
        edited[int(index)]["label_xy"] = clamp_detector_label_position(
            np.asarray([x_new, y_new], dtype=np.float64)
        )
        redraw_labels()

    def set_selected_label_text(text: object, *, redraw_figure: bool = True) -> None:
        if bool(title_box_state.get("syncing", False)):
            return
        edited[active_index()]["text"] = str(text)
        if redraw_figure:
            redraw_labels()

    def flush_label_box() -> None:
        box = title_box_state.get("box")
        if box is not None:
            set_selected_label_text(getattr(box, "text", ""), redraw_figure=False)

    def nudge_label(dx: float, dy: float) -> None:
        pos = active_position()
        if pos is None:
            return
        set_label_position(active_index(), float(pos[0]) + float(dx), float(pos[1]) + float(dy))

    def step_font(delta: float) -> None:
        edited[active_index()]["fontsize"] = float(
            np.clip(detector_label_fontsize(edited[active_index()]) + float(delta), 4.0, 32.0)
        )
        redraw_labels()

    def label_index_from_event(event) -> int | None:
        if event.inaxes is not ax:
            return None
        for artist in reversed(label_artists):
            try:
                contains, _details = artist.contains(event)
            except Exception:
                contains = False
            if contains:
                try:
                    index = int(getattr(artist, "_ra_sim_label_entry_index"))
                except Exception:
                    continue
                if index in drawable_index_set:
                    return index
        if event.xdata is None or event.ydata is None:
            return None
        event_xy = np.asarray([float(event.x), float(event.y)], dtype=np.float64)
        nearest_index: int | None = None
        nearest_distance = np.inf
        for index in drawable_indices:
            pos = detector_label_xy(edited[int(index)])
            if pos is None:
                continue
            display_xy = np.asarray(ax.transData.transform((float(pos[0]), float(pos[1]))))
            distance = float(np.linalg.norm(display_xy - event_xy))
            if distance < nearest_distance:
                nearest_index = int(index)
                nearest_distance = distance
        if nearest_index is not None and nearest_distance <= 18.0:
            return nearest_index
        return None

    def import_label_settings(_event=None) -> None:
        nonlocal edited
        flush_label_box()
        if not settings_text:
            print("detector label editor: no label settings path configured")
            return
        try:
            edited = apply_detector_label_settings(
                edited, load_detector_label_settings(settings_text)
            )
            if active_index() not in drawable_index_set:
                selected["index"] = int(drawable_indices[0])
            redraw_labels()
            print(f"loaded detector label settings={Path(settings_text).expanduser()}")
        except Exception as exc:
            print(f"ignored detector label settings={Path(settings_text).expanduser()}: {exc}")

    def export_label_settings(_event=None) -> None:
        flush_label_box()
        if not settings_text:
            print("detector label editor: no label settings path configured")
            return
        try:
            saved_path = save_detector_label_settings(settings_text, edited)
            print(f"saved detector label settings={saved_path}")
        except Exception as exc:
            print(f"failed detector label settings export={Path(settings_text).expanduser()}: {exc}")

    def finish_editor(accepted: bool) -> None:
        flush_label_box()
        selected["accepted"] = bool(accepted)
        try:
            fig.canvas.stop_event_loop()
        except Exception:
            pass

    def accept(_event=None) -> None:
        finish_editor(True)

    def cancel(_event=None) -> None:
        finish_editor(False)

    def on_press(event) -> None:
        if event.inaxes is not ax:
            return
        index = label_index_from_event(event)
        if index is None or event.xdata is None or event.ydata is None:
            return
        selected["index"] = int(index)
        pos = detector_label_xy(edited[int(index)])
        if pos is not None:
            selected["drag_offset_x"] = float(pos[0]) - float(event.xdata)
            selected["drag_offset_y"] = float(pos[1]) - float(event.ydata)
        selected["dragging"] = True
        redraw_labels()

    def on_motion(event) -> None:
        if not bool(selected.get("dragging", False)):
            return
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        dragging_index = selected.get("index")
        if dragging_index is None:
            return
        set_label_position(
            int(dragging_index),
            float(event.xdata) + float(selected.get("drag_offset_x", 0.0)),
            float(event.ydata) + float(selected.get("drag_offset_y", 0.0)),
        )

    def on_release(_event) -> None:
        selected["dragging"] = False
        selected["drag_offset_x"] = 0.0
        selected["drag_offset_y"] = 0.0

    def on_key(event) -> None:
        box = title_box_state.get("box")
        if box is not None and getattr(event, "inaxes", None) is getattr(box, "ax", None):
            return
        key = str(getattr(event, "key", "")).lower()
        if key in key_nudges:
            nudge_label(*key_nudges[key])
        elif key in {"enter", "return"}:
            finish_editor(True)
        elif key == "escape":
            finish_editor(False)

    def cleanup_editor_artifacts() -> None:
        remove_label_artists()
        for control_ax in control_axes:
            try:
                control_ax.remove()
            except Exception:
                pass
        try:
            delattr(fig, "_ra_sim_detector_label_edit_widgets")
        except Exception:
            pass
        fig.canvas.draw_idle()

    label_box_ax = fig.add_axes([0.08, 0.018, 0.28, 0.045])
    font_down_ax = fig.add_axes([0.39, 0.018, 0.075, 0.045])
    font_up_ax = fig.add_axes([0.47, 0.018, 0.075, 0.045])
    import_ax = fig.add_axes([0.57, 0.018, 0.075, 0.045])
    export_ax = fig.add_axes([0.65, 0.018, 0.075, 0.045])
    cancel_ax = fig.add_axes([0.75, 0.018, 0.075, 0.045])
    accept_ax = fig.add_axes([0.83, 0.018, 0.075, 0.045])
    control_axes.extend(
        [label_box_ax, font_down_ax, font_up_ax, import_ax, export_ax, cancel_ax, accept_ax]
    )

    title_box = TextBox(label_box_ax, "Label", textalignment="left")
    title_box.on_submit(set_selected_label_text)
    title_box_state["box"] = title_box
    font_down_button = Button(font_down_ax, "Font -")
    font_up_button = Button(font_up_ax, "Font +")
    import_button = Button(import_ax, "Import")
    export_button = Button(export_ax, "Export")
    cancel_button = Button(cancel_ax, "Cancel")
    accept_button = Button(accept_ax, "Accept")
    font_down_button.on_clicked(lambda event: step_font(-1.0))
    font_up_button.on_clicked(lambda event: step_font(1.0))
    import_button.on_clicked(import_label_settings)
    export_button.on_clicked(export_label_settings)
    cancel_button.on_clicked(cancel)
    accept_button.on_clicked(accept)
    fig._ra_sim_detector_label_edit_widgets = [
        title_box,
        font_down_button,
        font_up_button,
        import_button,
        export_button,
        cancel_button,
        accept_button,
    ]
    connection_ids = [
        fig.canvas.mpl_connect("button_press_event", on_press),
        fig.canvas.mpl_connect("motion_notify_event", on_motion),
        fig.canvas.mpl_connect("button_release_event", on_release),
        fig.canvas.mpl_connect("key_press_event", on_key),
        fig.canvas.mpl_connect("close_event", cancel),
    ]

    try:
        redraw_labels()
        fig.canvas.start_event_loop(timeout=-1)
    finally:
        for connection_id in connection_ids:
            try:
                fig.canvas.mpl_disconnect(connection_id)
            except Exception:
                pass
        cleanup_editor_artifacts()

    if not bool(selected["accepted"]):
        return original, False
    return edited, True


def edit_detector_region_label_positions(
    fig: object,
    ax: object,
    label_entries: list[dict[str, object]],
    *,
    mode: object = "auto",
    settings_path: object = None,
    backend_name: object = None,
    env: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    entries = [dict(entry) for entry in label_entries]
    path_text = "" if settings_path is None else str(settings_path).strip()
    runtime_mode = detector_label_edit_runtime_mode(mode, backend_name=backend_name, env=env)
    if runtime_mode != "popup":
        print(f"detector label editor: mode={runtime_mode}")
        return entries
    try:
        edited, accepted = show_detector_region_label_position_popup(
            fig, ax, entries, settings_path=path_text
        )
    except Exception as exc:
        print(f"skipped detector label editor: {exc}")
        return entries
    if not accepted:
        print("detector label editor: canceled")
        return entries
    return edited


detector_region_display_width = max(detector_region_xlim[1] - detector_region_xlim[0], 1.0)
detector_region_display_height = max(detector_region_ylim[0] - detector_region_ylim[1], 1.0)
fig_height = min(
    6.8,
    max(
        4.8, JOURNAL_FULL_WIDTH_IN * detector_region_display_height / detector_region_display_width
    ),
)
fig, ax = plt.subplots(figsize=(JOURNAL_FULL_WIDTH_IN, fig_height), constrained_layout=False)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.985)
ax.imshow(
    detector_region_bg,
    origin="upper",
    aspect="equal",
    cmap=detector_region_cmap,
    vmin=float(detector_region_vmin),
    vmax=float(detector_region_vmax),
    rasterized=True,
)

detector_label_settings_path = _setting_text(
    "DETECTOR_LABEL_SETTINGS_PATH_OVERRIDE",
    "RA_SIM_DETECTOR_LABEL_SETTINGS",
    FIGURE_OUT_DIR / f"{ROD_PROFILE_REGION_STEM}_detector_label_settings.json",
)
rod_label_entries: list[dict[str, object]] = []
detector_label_ids_added: set[str] = set()


def append_detector_rod_label_entry(
    *,
    m_value: int,
    branch_suffix: str,
    x_line: object,
    y_line: object,
    qz_line: object | None = None,
    color: object,
) -> None:
    label_id = detector_rod_label_id(m_value, branch_suffix)
    if label_id in detector_label_ids_added:
        return
    x_values = np.asarray(x_line, dtype=np.float64)
    y_values = np.asarray(y_line, dtype=np.float64)
    if x_values.shape != y_values.shape:
        return
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return
    if qz_line is None:
        qz_values = detector_qz_values_for_polyline(x_values, y_values, qz_map)
    else:
        qz_values = np.asarray(qz_line, dtype=np.float64)
        if qz_values.shape != x_values.shape:
            qz_values = detector_qz_values_for_polyline(x_values, y_values, qz_map)
    rod_label_entries.append(
        {
            "x_line": x_values,
            "y_line": y_values,
            "qz_line": qz_values,
            "label_id": label_id,
            "text": detector_rod_label(m_value, branch_suffix),
            "fontsize": detector_region_label_fontsize,
            "color": color,
            "label_mode": "low_l_base",
        }
    )
    detector_label_ids_added.add(label_id)


detector_overlay_rods = detector_overlay_rod_entries(
    rod_entries,
    allow_generated=ALLOW_GENERATED_ROD_REFERENCES,
    region_overlays=region_overlays,
)
final_detector_rod_trace_payloads_by_key: dict[tuple[str, int], list[dict[str, object]]] = {
    rod_identity_key(rod): projected_qr_detector_trace_payloads(
        rod, config=profile_bg["qr_overlay_config"]
    )
    for rod in detector_overlay_rods
    if not rod_rejected_for_plot(rod, ACTIVE_REJECTED_ROD_KEYS)
}
final_detector_rod_lines_by_key: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {
    key: [
        (
            np.asarray(payload["x_line"], dtype=np.float64),
            np.asarray(payload["y_line"], dtype=np.float64),
        )
        for payload in payloads
    ]
    for key, payloads in final_detector_rod_trace_payloads_by_key.items()
}
rod_color_by_key = {
    rod_identity_key(rod): region_colors[index % len(region_colors)]
    for index, rod in enumerate(detector_overlay_rods)
}
for index, rod in enumerate(detector_overlay_rods):
    color = region_colors[index % len(region_colors)]
    m_value = int(rod["m"])
    rod_key = rod_identity_key(rod)
    detector_overlay_qz_bounds_by_key: dict[tuple[str, int, str], tuple[float, float]] = {}
    for branch_name, phi_min, phi_max in (("-", -90.0, 0.0), ("+", 0.0, 90.0)):
        overlays = [
            item
            for item in region_overlays
            if int(item["m"]) == m_value
            and str(item.get("source", rod.get("source", ""))) == str(rod.get("source", ""))
            and str(item["branch"]) == branch_name
        ]
        for item in overlays:
            qz_bounds = positive_l_qz_bounds(
                m_value=m_value,
                branch_value=branch_name,
                qz_lo=item["qz_min"],
                qz_hi=item["qz_max"],
            )
            if qz_bounds is None:
                continue
            qz_min, qz_max = qz_bounds
            detector_overlay_qz_bounds_by_key[
                (str(rod.get("source", "")), m_value, branch_name)
            ] = (float(qz_min), float(qz_max))
            visual = gui_qr_cylinder_overlay.build_detector_selected_qr_rod_band_visual_payload(
                qr_map=qr_map,
                qz_map=qz_map,
                valid_q=valid_q_map,
                qr_center=float(item["qr"]),
                delta_qr=float(qr_rod_delta_qr),
                qz_min=float(qz_min),
                qz_max=float(qz_max),
                detector_phi_deg=profile_detector_phi_map,
                phi_min=float(phi_min),
                phi_max=float(phi_max),
                shape_mask=detector_region_shape_mask,
            )
            draw_detector_delta_q_region(visual, color)
            detector_visual_label_lines = detector_mask_centerline_from_visual(visual)
            for visual_col, visual_row in detector_visual_label_lines:
                append_detector_rod_label_entry(
                    m_value=m_value,
                    branch_suffix=branch_name,
                    x_line=visual_col,
                    y_line=visual_row,
                    qz_line=None,
                    color=color,
                )
    rod_payloads = final_detector_rod_trace_payloads_by_key.get(rod_key, [])
    for payload in rod_payloads:
        branch_suffix = "+" if int(payload.get("branch_sign", 0)) > 0 else "-"
        raw_projected_col = np.asarray(payload["x_line"], dtype=np.float64)
        raw_projected_row = np.asarray(payload["y_line"], dtype=np.float64)
        raw_projected_qz = np.asarray(
            payload.get("qz_line", np.full(raw_projected_col.shape, np.nan)),
            dtype=np.float64,
        )
        if raw_projected_qz.shape != raw_projected_col.shape:
            raw_projected_qz = np.full(raw_projected_col.shape, np.nan, dtype=np.float64)
        trace_qz_bounds = detector_overlay_qz_bounds_by_key.get(
            (str(rod.get("source", "")), m_value, branch_suffix)
        )
        if trace_qz_bounds is None:
            projected_col = raw_projected_col.copy()
            projected_row = raw_projected_row.copy()
        else:
            projected_col, projected_row = clipped_detector_trace_to_qz_bounds(
                raw_projected_col,
                raw_projected_row,
                raw_projected_qz,
                trace_qz_bounds,
            )
        if np.count_nonzero(np.isfinite(projected_col) & np.isfinite(projected_row)) < 2:
            continue
        finite_projected = np.isfinite(projected_col) & np.isfinite(projected_row)
        projected_qz_line = np.where(finite_projected, raw_projected_qz, np.nan)
        should_draw_centerline = trace_qz_bounds is not None
        if should_draw_centerline:
            ax.plot(
                projected_col,
                projected_row,
                color=color,
                linewidth=detector_region_centerline_lw,
                alpha=1.0,
                zorder=5.0,
            )
        append_detector_rod_label_entry(
            m_value=m_value,
            branch_suffix=branch_suffix,
            x_line=projected_col,
            y_line=projected_row,
            qz_line=projected_qz_line,
            color=color,
        )

specular_color = OKABE_ITO["sky"]
specular_detector_qz_values = np.asarray(specular_qz_values, dtype=np.float64)
specular_detector_qz_values = specular_detector_qz_values[
    np.isfinite(specular_detector_qz_values) & (specular_detector_qz_values > POSITIVE_QZ_MIN)
]
specular_delta_q_visual = None
if specular_detector_qz_values.size >= 2:
    specular_delta_q_visual = (
        gui_qr_cylinder_overlay.build_detector_selected_qr_rod_band_visual_payload(
            qr_map=specular_detector_qr_map,
            qz_map=specular_detector_qz_map,
            valid_q=specular_detector_valid_q,
            qr_center=0.0,
            delta_qr=float(qr_rod_delta_qr),
            qz_min=float(np.nanmin(specular_detector_qz_values)),
            qz_max=float(np.nanmax(specular_detector_qz_values)),
            detector_phi_deg=profile_detector_phi_map,
            phi_min=float(specular_phi_min_deg),
            phi_max=float(specular_phi_max_deg),
            shape_mask=detector_region_shape_mask,
        )
    )
    draw_detector_delta_q_region(
        specular_delta_q_visual,
        specular_color,
        fill_alpha=detector_region_specular_band_alpha,
        boundary_alpha=detector_region_specular_boundary_alpha,
        boundary_expand_px=detector_region_specular_boundary_expand_px,
    )
specular_lines = specular_detector_lines_from_markers(specular_l_marker_table)
if not specular_lines:
    specular_lines = detector_mask_centerline_from_visual(specular_delta_q_visual)
if not specular_lines:
    specular_lines = specular_detector_centerline_fallback()
for projected_col, projected_row in specular_lines:
    ax.plot(
        projected_col,
        projected_row,
        color=specular_color,
        linewidth=detector_region_specular_centerline_lw,
        alpha=1.0,
        zorder=5.2,
    )
    rod_label_entries.append(
        {
            "x_line": projected_col,
            "y_line": projected_row,
            "label_id": detector_specular_label_id(),
            "qz_line": detector_qz_values_for_polyline(
                projected_col, projected_row, specular_detector_qz_map
            ),
            "text": detector_specular_label(),
            "fontsize": detector_region_label_fontsize,
            "color": specular_color,
            "label_mode": "low_l_base",
            "flip_normal": False,
        }
    )

display_detector_rotation_fit_debug(
    profile_bg, rod_qspace_calibration, final_detector_rod_lines_by_key
)

beam_center = (
    float(getattr(profile_bg["qr_overlay_config"], "center_col", center_col_px)),
    float(getattr(profile_bg["qr_overlay_config"], "center_row", center_row_px)),
)
hk0_00l_region_mask = None
if isinstance(specular_delta_q_visual, dict):
    candidate_mask = np.asarray(specular_delta_q_visual.get("band_fill_mask"), dtype=bool)
    if candidate_mask.shape == detector_region_shape:
        shape_mask = np.asarray(detector_region_shape_mask, dtype=bool)
        if shape_mask.shape == candidate_mask.shape:
            candidate_mask = candidate_mask & shape_mask
        hk0_00l_region_mask = candidate_mask
hk0_00l_region_path = FIGURE_OUT_DIR / "00L_region.png"
hk0_00l_region_horizontal_path = FIGURE_OUT_DIR / "00L_region_horizontal.png"
if hk0_00l_region_mask is None or not np.any(hk0_00l_region_mask):
    print("skipped 00L_region.png: specular 00L region mask was unavailable")
else:
    hk0_00l_region_saved = save_hk0_00l_region_crop(
        np.asarray(
            profile_bg.get("raw_detector_image", profile_bg["detector_image"]), dtype=np.float64
        ),
        hk0_00l_region_path,
        horizontal_output_path=hk0_00l_region_horizontal_path,
        beam_center=beam_center,
        region_mask=hk0_00l_region_mask,
    )
    if hk0_00l_region_saved is None:
        print("skipped 00L_region.png: crop bounds were empty or invalid")
    else:
        for saved_path in hk0_00l_region_saved:
            if saved_path is not None:
                print(f"saved={saved_path}")
for label_entry in rod_label_entries:
    if str(label_entry.get("label_mode", "")) == "low_l_base":
        label_xy = place_low_l_rod_label(
            ax,
            np.asarray(label_entry["x_line"], dtype=np.float64),
            np.asarray(label_entry["y_line"], dtype=np.float64),
            np.asarray(label_entry.get("qz_line", []), dtype=np.float64),
            str(label_entry["text"]),
            str(label_entry["color"]),
        )
    else:
        label_xy = place_rod_label(
            ax,
            np.asarray(label_entry["x_line"], dtype=np.float64),
            np.asarray(label_entry["y_line"], dtype=np.float64),
            str(label_entry["text"]),
            str(label_entry["color"]),
            beam_center=beam_center,
            flip_normal=bool(label_entry.get("flip_normal", False)),
        )
    if label_xy is not None:
        label_entry["label_xy"] = label_xy.copy()

ax.set_xlabel("Detector x pixel (bottom-left origin)")
ax.set_ylabel("Detector y pixel (bottom-left origin)")
finish_axes(ax)
ax.set_xlim(*detector_region_xlim)
ax.set_ylim(*detector_region_ylim)
x_ticks = np.linspace(detector_region_xlim[0], detector_region_xlim[1], 5)
y_ticks = np.linspace(detector_region_ylim[1], detector_region_ylim[0], 5)
x_tick_labels, y_tick_labels = detector_bottom_left_axis_tick_labels(
    x_ticks,
    y_ticks,
    xlim=detector_region_xlim,
    ylim=detector_region_ylim,
)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

rod_label_entries = apply_unified_qr_rod_region_editor_labels(
    rod_label_entries,
    qr_rod_region_editor_result,
)
draw_detector_region_label_artists(ax, rod_label_entries)

detector_region_png, detector_region_pdf = save_manuscript_figure(fig, ROD_PROFILE_REGION_STEM)
plt.close(fig)
detector_region_scale_png, detector_region_scale_pdf = save_detector_region_intensity_scale(
    detector_region_cmap,
    vmin=float(detector_region_vmin),
    vmax=float(detector_region_vmax),
    stem=f"{ROD_PROFILE_REGION_STEM}_intensity_scale",
)
print(f"saved={detector_region_png}")
print(f"saved={detector_region_pdf}")
print(f"saved={FIGURE_OUT_DIR / f'{ROD_PROFILE_REGION_STEM}.svg'}")
print(f"saved={detector_region_scale_png}")
print(f"saved={detector_region_scale_pdf}")
display(Image(filename=str(detector_region_png)))
