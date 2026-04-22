"""Run bounded new4 geometry-fit optimizer probes.

This script is intentionally narrow: it validates that the optimizer can
consume the already-verified new4 manual point-provider pairs. It does not
change saved GUI state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim import debug_controls
from ra_sim.fitting import optimization as opt
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.io.data_loading import load_gui_state_file
from scripts.debug import validate_geometry_preflight_rebind as preflight_probe


REQUIRED_PROVIDER_COUNTS = {
    "manual_picker_pair_count": 7,
    "point_provider_pair_count": 7,
    "dataset_provider_mismatch_count": 0,
    "fallback_pair_count": 0,
}
EXPECTED_PROVIDER_PAIR_COUNT = 7
TIMING_METADATA_KEYS = {
    "started_at_iso",
    "finished_at_iso",
    "elapsed_s",
    "elapsed_seconds",
    "stage_elapsed_s",
    "run_id",
    "run_dir",
    "rung_id",
    "rung_index",
    "report_path",
}

STRICT_POINT_SUMMARY_KEYS = (
    "measured_count",
    "fixed_source_resolved_count",
    "fixed_source_reflection_count",
    "matched_fixed_pair_count",
    "missing_fixed_pair_count",
    "fallback_entry_count",
    "fallback_hkl_count",
    "subset_fallback_hkl_count",
)

REQUEST_HANDOFF_FIELDS = (
    "provider_pair_count",
    "dataset_pair_count",
    "optimizer_request_pair_count",
    "fixed_source_pair_count",
    "fallback_row_count",
    "fixed_source_resolution_fallback_count",
    "missing_fixed_source_count",
    "branch_mismatch_count",
    "provider_to_optimizer_identity_match",
    "provider_to_optimizer_point_match",
)

NO_MATCH_REJECTION = "No matched peak pairs were available for the fitted solution."

ONE_PARAM_FIXED_SOURCE_COUNTS = {
    "fixed_source_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_row_count": 0,
    "fixed_source_resolution_fallback_count": 0,
    "missing_fixed_source_count": 0,
    "fixed_source_resolved_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_entry_count": 0,
    "matched_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "missing_pair_count": 0,
    "branch_mismatch_count": 0,
}

RUNG2_FIXED_SOURCE_COUNTS = {
    "provider_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "dataset_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "optimizer_request_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    **ONE_PARAM_FIXED_SOURCE_COUNTS,
}

PROVIDER_MATCH_BOOLS = (
    "provider_to_optimizer_identity_match",
    "provider_to_optimizer_point_match",
)

LIVE_HEARTBEAT_COUNTERS = {
    "fixed_source_resolved_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_entry_count": 0,
    "matched_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "missing_pair_count": 0,
    "branch_mismatch_count": 0,
}

ONE_PARAM_A_VARIANTS = (
    ("a_nfev5_t120", 5, 120.0),
    ("a_nfev10_t120", 10, 120.0),
    ("a_nfev20_t300", 20, 300.0),
)

ONE_PARAM_ORDER = [
    "center_x",
    "center_y",
    "gamma",
    "Gamma",
    "chi",
    "cor_angle",
    "theta_initial",
    "corto_detector",
    "zs",
    "zb",
    "a",
    "c",
    "psi_z",
]

CENTER_PARAMS = ["center_x", "center_y"]
FEATURE_RUNS = [
    "dynamic_reanchor",
    "discrete_modes",
    "seed_multistart",
    "full_beam_polish",
    "identifiability_features",
]

RUNG4_DEFAULT_PAIRS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("a_c", ("a", "c")),
    ("chi_cor_angle", ("chi", "cor_angle")),
    ("theta_initial_cor_angle", ("theta_initial", "cor_angle")),
    ("corto_detector_theta_initial", ("corto_detector", "theta_initial")),
    ("zs_zb", ("zs", "zb")),
)
RUNG4_PSI_EXTENSION_PAIRS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("c_psi_z", ("c", "psi_z")),
    ("a_psi_z", ("a", "psi_z")),
)
RUNG5_BLOCKS: tuple[tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]], ...] = (
    (
        "corto_detector_theta_initial_cor_angle",
        ("corto_detector", "theta_initial", "cor_angle"),
        (("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
    ),
    (
        "chi_cor_angle_theta_initial",
        ("chi", "cor_angle", "theta_initial"),
        (("chi", "cor_angle"), ("theta_initial", "cor_angle")),
    ),
    (
        "corto_detector_theta_initial_zs_zb",
        ("corto_detector", "theta_initial", "zs", "zb"),
        (("corto_detector", "theta_initial"), ("zs", "zb")),
    ),
    (
        "a_c_psi_z",
        ("a", "c", "psi_z"),
        (("a", "c"),),
    ),
)
CAKED_REPROJECTION_REQUIRED_PARAMS = {"theta_initial", "theta_offset", "corto_detector"}
PAIR_RMS_TOLERANCE_PX = 0.25
PAIR_MAX_ERROR_TOLERANCE_PX = 1.0


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _perf_counter() -> float:
    return time.perf_counter()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite_seconds(value: object) -> float | None:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


def _timing_sum(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return None
    total = 0.0
    seen = False
    for raw in value.values():
        seconds = _finite_seconds(raw)
        if seconds is None:
            continue
        total += seconds
        seen = True
    return total if seen else None


class _TimingWindow:
    def __init__(
        self,
        *,
        rung_id: str | None,
        rung_name: str | None,
        started_perf: float,
        started_at: datetime,
    ) -> None:
        self.rung_id = rung_id
        self.rung_name = rung_name
        self.started_perf = started_perf
        self.started_at = started_at


class _TimingCollector:
    def __init__(
        self,
        *,
        run_dir: Path,
        expected_rung_ids: Sequence[str] = (),
        started_perf: float | None = None,
        started_at: datetime | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.expected_rung_ids = tuple(str(rung_id) for rung_id in expected_rung_ids)
        self.started_perf = _perf_counter() if started_perf is None else float(started_perf)
        self.started_at = _utc_now() if started_at is None else started_at
        self.records: dict[str, dict[str, object]] = {}

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def _relative_parts(self, path: Path) -> tuple[str, ...] | None:
        try:
            return path.expanduser().resolve().relative_to(self.run_dir).parts
        except ValueError:
            return None

    def _rung_id_for_path(self, path: Path) -> str | None:
        parts = self._relative_parts(path)
        if not parts:
            return None
        name = parts[-1]
        if name.endswith(".heartbeat.json") or "heartbeat" in name:
            return None
        if "timing_summary" in name or name == "rung_timing_summary.json":
            return None
        if len(parts) == 2 and parts == (
            "rung_03a_a_diagnosis",
            "variant_summary.json",
        ):
            return "3A"
        if len(parts) == 2 and parts == (
            "rung_03b_caked_point_reprojection",
            "rung_03b_caked_point_reprojection.json",
        ):
            return "3B"
        if len(parts) != 1:
            return None
        if name == "ladder_summary.json":
            return "summary"
        if "provider_guard_after" in name:
            return None
        if name.startswith("rung_00_"):
            return "0"
        if name.startswith("rung_01_"):
            return "1"
        if name.startswith("rung_02_"):
            return "2"
        if name.startswith("rung_03_"):
            return "3"
        if name.startswith("rung_04_"):
            return "4"
        if name.startswith("rung_05_"):
            return "5"
        return None

    def decorate_payload(
        self,
        path: Path,
        payload: Mapping[str, object],
        window: _TimingWindow | None,
    ) -> dict[str, object]:
        resolved = path.expanduser().resolve()
        rung_id = self._rung_id_for_path(resolved)
        if rung_id is None:
            return dict(payload)

        finished_at = _utc_now()
        finished_perf = _perf_counter()
        existing_elapsed = _finite_seconds(
            payload.get("elapsed_s", payload.get("elapsed_seconds"))
        )
        if rung_id == "summary":
            elapsed_s = max(0.0, finished_perf - float(self.started_perf))
            started_at = self.started_at
        elif window is not None:
            elapsed_s = max(0.0, finished_perf - float(window.started_perf))
            started_at = window.started_at
        elif existing_elapsed is not None:
            elapsed_s = max(0.0, existing_elapsed)
            started_at = finished_at - timedelta(seconds=elapsed_s)
        else:
            elapsed_s = 0.0
            started_at = finished_at

        decorated = dict(payload)
        if rung_id != "summary":
            decorated["rung_id"] = rung_id
            decorated["rung_index"] = rung_id
        decorated.setdefault("rung_name", window.rung_name if window else resolved.stem)
        decorated["run_id"] = self.run_id
        decorated["run_dir"] = str(self.run_dir)
        decorated["report_path"] = str(resolved)
        decorated["started_at_iso"] = _iso_z(started_at)
        decorated["finished_at_iso"] = _iso_z(finished_at)
        decorated["elapsed_s"] = float(elapsed_s)
        decorated["elapsed_seconds"] = float(elapsed_s)
        stage_elapsed = _timing_sum(decorated.get("stage_timing_s"))
        if stage_elapsed is None:
            stage_elapsed = _timing_sum(decorated.get("phase_timing_s"))
        if stage_elapsed is not None:
            decorated["stage_elapsed_s"] = float(stage_elapsed)

        if rung_id != "summary":
            self.records[str(resolved)] = {
                "rung_id": rung_id,
                "rung_index": rung_id,
                "rung_name": str(decorated.get("rung_name", resolved.stem)),
                "status": str(decorated.get("status", "")),
                "elapsed_s": float(elapsed_s),
                "report_path": str(resolved),
            }
        return decorated

    def summary(self) -> dict[str, object]:
        finished_at = _utc_now()
        total_elapsed_s = max(0.0, _perf_counter() - float(self.started_perf))
        timings = sorted(
            self.records.values(),
            key=lambda item: (
                str(item.get("rung_id", "")),
                str(item.get("report_path", "")),
            ),
        )
        slowest = max(
            timings,
            key=lambda item: float(item.get("elapsed_s", 0.0) or 0.0),
            default=None,
        )
        present = {str(item.get("rung_id", "")) for item in timings}
        missing = [
            rung_id for rung_id in self.expected_rung_ids if rung_id not in present
        ]
        threshold_max = _finite_seconds(os.environ.get("RA_SIM_NEW4_LADDER_TIMING_MAX_S"))
        exceeded: list[dict[str, object]] = []
        threshold_status = "not_configured"
        if threshold_max is not None:
            exceeded = [
                dict(item)
                for item in timings
                if float(item.get("elapsed_s", 0.0) or 0.0) > threshold_max
            ]
            threshold_status = "exceeded" if exceeded else "ok"
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at_iso": _iso_z(self.started_at),
            "finished_at_iso": _iso_z(finished_at),
            "total_elapsed_s": float(total_elapsed_s),
            "timing_collection_mode": "current_run",
            "completed_rung_count": len(timings),
            "missing_expected_rungs": missing,
            "rung_timings": timings,
            "slowest_rung": (
                str(slowest.get("rung_name", "")) if slowest is not None else None
            ),
            "slowest_rung_elapsed_s": (
                float(slowest.get("elapsed_s", 0.0)) if slowest is not None else None
            ),
            "timing_threshold_status": threshold_status,
            "timing_threshold_max_s": threshold_max,
            "timing_threshold_exceeded_rungs": exceeded,
        }


_ACTIVE_TIMING_COLLECTOR: ContextVar[_TimingCollector | None] = ContextVar(
    "_ACTIVE_TIMING_COLLECTOR",
    default=None,
)
_ACTIVE_TIMING_WINDOW: ContextVar[_TimingWindow | None] = ContextVar(
    "_ACTIVE_TIMING_WINDOW",
    default=None,
)
_SUPPRESS_TIMING_COLLECTION: ContextVar[bool] = ContextVar(
    "_SUPPRESS_TIMING_COLLECTION",
    default=False,
)


@contextmanager
def _timed_report_window(rung_id: str | None, rung_name: str | None = None):
    token = _ACTIVE_TIMING_WINDOW.set(
        _TimingWindow(
            rung_id=str(rung_id) if rung_id is not None else None,
            rung_name=str(rung_name) if rung_name is not None else None,
            started_perf=_perf_counter(),
            started_at=_utc_now(),
        )
    )
    try:
        yield
    finally:
        _ACTIVE_TIMING_WINDOW.reset(token)


@contextmanager
def _suppress_timing_collection():
    token = _SUPPRESS_TIMING_COLLECTION.set(True)
    try:
        yield
    finally:
        _SUPPRESS_TIMING_COLLECTION.reset(token)


def _format_timing_table(summary: Mapping[str, object]) -> str:
    rows = ["Rung | Status | elapsed_s | report_path"]
    for item in summary.get("rung_timings", []) or []:
        if not isinstance(item, Mapping):
            continue
        elapsed = _finite_seconds(item.get("elapsed_s"))
        rows.append(
            " | ".join(
                (
                    str(item.get("rung_id", item.get("rung_index", ""))),
                    str(item.get("status", "")),
                    f"{elapsed:.6f}" if elapsed is not None else "",
                    str(item.get("report_path", "")),
                )
            )
        )
    return "\n".join(rows)


def _expected_rung_ids_for_run(
    max_rung: str,
    *,
    one_param_summary: Path | None = None,
    pair_summary: Path | None = None,
    caked_point_reprojection_report: Path | None = None,
) -> tuple[str, ...]:
    name = str(max_rung).strip().lower()
    if name == "sensitivity":
        return ("0", "1", "2")
    if name == "one-param":
        return ("0", "1", "2", "3")
    if name in {"pair", "pairs"}:
        return ("0", "1", "2", "4")
    if name in {"block", "blocks"}:
        expected = ["0", "1", "2", "5"]
        if pair_summary is None:
            expected[3:3] = ["3", "4"]
            if caked_point_reprojection_report is None:
                expected.insert(4, "3B")
        return tuple(expected)
    if name in {"center", "full"}:
        return ("0", "1", "2", "3", "4", "5")
    return ("0", "1", "2")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    output_payload = dict(payload)
    collector = _ACTIVE_TIMING_COLLECTOR.get()
    if collector is not None and not _SUPPRESS_TIMING_COLLECTION.get():
        output_payload = collector.decorate_payload(
            Path(path),
            output_payload,
            _ACTIVE_TIMING_WINDOW.get(),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(output_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_jsonable(dict(payload)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provider_report_evidence_hash(payload: Mapping[str, object]) -> str:
    stable = {
        str(key): value
        for key, value in dict(payload).items()
        if str(key) not in TIMING_METADATA_KEYS
    }
    return _payload_sha256(stable)


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@contextmanager
def _solver_debug_logging_scope(enabled: bool):
    """Disable expensive diagnostic file logging around normal warm solver probes."""

    if bool(enabled):
        yield
        return

    previous_disable = os.environ.get("RA_SIM_DISABLE_LOGGING")
    previous_disable_all = os.environ.get("RA_SIM_DISABLE_ALL_LOGGING")
    previous_intersection = os.environ.get("RA_SIM_LOG_INTERSECTION_CACHE")
    os.environ["RA_SIM_DISABLE_LOGGING"] = "1"
    os.environ["RA_SIM_DISABLE_ALL_LOGGING"] = "1"
    os.environ["RA_SIM_LOG_INTERSECTION_CACHE"] = "0"
    try:
        with debug_controls.temporary_startup_debug_override("disable_all"):
            yield
    finally:
        for key, value in (
            ("RA_SIM_DISABLE_LOGGING", previous_disable),
            ("RA_SIM_DISABLE_ALL_LOGGING", previous_disable_all),
            ("RA_SIM_LOG_INTERSECTION_CACHE", previous_intersection),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _same_resolved_path(left: object, right: Path) -> bool:
    if not left:
        return False
    try:
        return Path(str(left)).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return False


def _state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _rung_path(run_dir: Path, number: int, name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    return run_dir / f"rung_{int(number):02d}_{safe_name}.json"


def _provider_guard_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if bool(report.get("ok", False)) is not True:
        failures.append("ok_not_true")
    if str(report.get("classification", "")) != "point_provider_parity_ok":
        failures.append("classification_not_point_provider_parity_ok")
    gate = report.get("point_provider_parity_gate")
    if not isinstance(gate, Mapping) or bool(gate.get("ok", False)) is not True:
        failures.append("point_provider_parity_gate_not_ok")
    for key, expected in REQUIRED_PROVIDER_COUNTS.items():
        try:
            actual = int(report.get(key, -999999))
        except Exception:
            actual = -999999
        if actual != int(expected):
            failures.append(f"{key}_{actual}_expected_{expected}")
    if bool(report.get("optimizer_called", True)):
        failures.append("optimizer_called")
    return failures


def run_provider_guard(
    *,
    state_path: Path,
    background_index: int,
    output_path: Path,
) -> dict[str, object]:
    started = _perf_counter()
    report = preflight_probe._run_point_provider_report_only(
        Path(state_path),
        background_index=int(background_index),
    )
    failures = _provider_guard_failures(report)
    payload = {
        **dict(report),
        "rung": 0,
        "rung_name": "provider_guard",
        "status": "pass" if not failures else "fail",
        "provider_guard_ok": not failures,
        "provider_guard_failures": failures,
        "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
    }
    _write_json(output_path, payload)
    return payload


def _run_provider_guard_report(
    *,
    state_path: Path,
    background_index: int,
    rung: int,
    rung_name: str,
) -> dict[str, object]:
    started = _perf_counter()
    report = preflight_probe._run_point_provider_report_only(
        Path(state_path),
        background_index=int(background_index),
    )
    failures = _provider_guard_failures(report)
    return {
        **dict(report),
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "pass" if not failures else "fail",
        "provider_guard_ok": not failures,
        "provider_guard_failures": failures,
        "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
    }


def _load_saved_state_for_background(state_path: Path, background_index: int) -> dict[str, object]:
    payload = load_gui_state_file(Path(state_path))
    saved_state = copy.deepcopy(dict(payload["state"]))
    files_state = saved_state.setdefault("files", {})
    if isinstance(files_state, dict):
        files_state["current_background_index"] = int(background_index)
    variables = saved_state.setdefault("variables", {})
    if isinstance(variables, dict):
        variables["geometry_fit_background_selection_var"] = "current"
    return saved_state


def _capture_solver_context(state_path: Path, background_index: int) -> dict[str, object]:
    saved_state = _load_saved_state_for_background(state_path, background_index)
    captured = preflight_probe._capture_execution_setup(
        saved_state=saved_state,
        state_path=Path(state_path),
    )
    prepare_result = captured.get("prepare_result")
    prepared_run = getattr(prepare_result, "prepared_run", None)
    if prepared_run is None:
        error_text = str(getattr(prepare_result, "error_text", "") or "")
        raise RuntimeError(error_text or "Geometry fit preflight failed.")
    execute_kwargs = (
        dict(captured.get("execute_kwargs", {}))
        if isinstance(captured.get("execute_kwargs"), Mapping)
        else {}
    )
    setup = execute_kwargs.get("setup")
    postprocess_config = getattr(setup, "postprocess_config", None)
    solver_inputs = getattr(postprocess_config, "solver_inputs", None)
    if solver_inputs is None:
        raise RuntimeError("Captured headless setup did not include solver inputs.")
    prepare_kwargs = (
        dict(captured.get("prepare_kwargs", {}))
        if isinstance(captured.get("prepare_kwargs"), Mapping)
        else {}
    )
    return {
        "state_path": str(state_path),
        "background_index": int(background_index),
        "prepared_run": prepared_run,
        "solver_inputs": solver_inputs,
        "saved_var_names": [
            str(name)
            for name in execute_kwargs.get("var_names", prepare_kwargs.get("var_names", ())) or ()
        ],
    }


def _coerce_active_names(context: Mapping[str, object], names: Sequence[object]) -> list[str]:
    prepared_run = context["prepared_run"]
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {})
        if prepared_run is not None
        else {}
    )
    active: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        if name == "theta_offset" and "theta_offset" not in fit_params:
            name = "theta_initial"
        if name not in fit_params:
            continue
        if name not in active:
            active.append(name)
    return active


def _lean_runtime_config(
    runtime_cfg: Mapping[str, object] | None,
    *,
    active_names: Sequence[str],
    max_nfev: int,
    feature: str | None = None,
) -> dict[str, object]:
    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    cfg["candidate_param_names"] = [str(name) for name in active_names]

    solver_raw = cfg.get("solver", cfg.get("optimizer", {}))
    solver = dict(solver_raw) if isinstance(solver_raw, Mapping) else {}
    solver["manual_point_fit_mode"] = True
    solver.pop("dynamic_point_geometry_fit", None)
    solver["max_nfev"] = int(max_nfev)
    solver["loss"] = "linear"
    solver["f_scale_px"] = 1.0
    solver["weighted_matching"] = False
    solver["use_measurement_uncertainty"] = False
    solver["anisotropic_measurement_uncertainty"] = False
    solver["workers"] = solver.get("workers", "auto")
    solver["parallel_mode"] = "off"
    solver["worker_numba_threads"] = 0
    cfg["solver"] = solver
    cfg["optimizer"] = solver

    seed = dict(cfg.get("seed_search", {}) if isinstance(cfg.get("seed_search"), Mapping) else {})
    seed["prescore_top_k"] = 1
    seed["n_global"] = 0
    seed["n_jitter"] = 0
    seed["min_seed_separation_u"] = 2.0
    cfg["seed_search"] = seed

    cfg["use_numba"] = bool(cfg.get("use_numba", False))
    cfg["allow_unsafe_runtime"] = False
    cfg.pop("full_beam_polish", None)
    cfg.pop("ridge_refinement", None)
    cfg.pop("image_refinement", None)

    discrete = dict(
        cfg.get("discrete_modes", {}) if isinstance(cfg.get("discrete_modes"), Mapping) else {}
    )
    discrete["enabled"] = False
    cfg["discrete_modes"] = discrete

    ident = dict(
        cfg.get("identifiability", {}) if isinstance(cfg.get("identifiability"), Mapping) else {}
    )
    ident["enabled"] = True
    ident.pop("auto_freeze", None)
    ident.pop("selective_thaw", None)
    ident.pop("adaptive_regularization", None)
    cfg["identifiability"] = ident

    feature_name = str(feature or "").strip().lower()
    if feature_name == "discrete_modes":
        cfg["discrete_modes"] = {"enabled": True}
    elif feature_name == "seed_multistart":
        seed["prescore_top_k"] = 4
        seed["n_global"] = 4
        seed["n_jitter"] = 2
        seed["min_seed_separation_u"] = 0.5
    elif feature_name == "full_beam_polish":
        cfg["full_beam_polish"] = {"enabled": True, "max_nfev": max(5, int(max_nfev))}
    elif feature_name == "identifiability_features":
        ident["auto_freeze"] = True
        ident["selective_thaw"] = {"enabled": True}
        ident["adaptive_regularization"] = {"enabled": True}
    elif feature_name == "dynamic_reanchor":
        cfg["dynamic_reanchor_probe_requested"] = True
    return cfg


def _prepared_run_for_active_names(
    context: Mapping[str, object],
    active_names: Sequence[str],
    *,
    max_nfev: int,
    feature: str | None = None,
):
    prepared_run = context["prepared_run"]
    runtime_cfg = _lean_runtime_config(
        getattr(prepared_run, "geometry_runtime_cfg", {}),
        active_names=active_names,
        max_nfev=int(max_nfev),
        feature=feature,
    )
    return preflight_probe.replace(
        prepared_run,
        geometry_runtime_cfg=runtime_cfg,
    )


def build_solver_request(
    context: Mapping[str, object],
    active_names: Sequence[object],
    *,
    max_nfev: int = 20,
    feature: str | None = None,
) -> gui_geometry_fit.GeometryFitSolverRequest:
    names = _coerce_active_names(context, active_names)
    prepared_run = _prepared_run_for_active_names(
        context,
        names,
        max_nfev=int(max_nfev),
        feature=feature,
    )
    request = gui_geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=names,
        solver_inputs=context["solver_inputs"],
    )
    if request.candidate_param_names != names:
        request = gui_geometry_fit.GeometryFitSolverRequest(
            miller=request.miller,
            intensities=request.intensities,
            image_size=request.image_size,
            params=request.params,
            measured_peaks=request.measured_peaks,
            var_names=list(names),
            candidate_param_names=list(names),
            dataset_specs=request.dataset_specs,
            refinement_config={
                **dict(request.refinement_config),
                "candidate_param_names": list(names),
            },
            runtime_safety_note=request.runtime_safety_note,
        )
    return request


def _rows_from_request(request: gui_geometry_fit.GeometryFitSolverRequest) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in request.dataset_specs or ():
        if not isinstance(spec, Mapping):
            continue
        for entry in spec.get("measured_peaks", ()) or ():
            if isinstance(entry, Mapping):
                rows.append(dict(entry))
    if rows:
        return rows
    for entry in request.measured_peaks or ():
        if isinstance(entry, Mapping):
            rows.append(dict(entry))
    return rows


def _request_handoff_summary(
    request: gui_geometry_fit.GeometryFitSolverRequest,
) -> dict[str, object]:
    rows = _rows_from_request(request)
    cfg = request.refinement_config if isinstance(request.refinement_config, Mapping) else {}
    bridge_summary = (
        dict(cfg.get("optimizer_request_handoff_summary", {}))
        if isinstance(cfg.get("optimizer_request_handoff_summary", {}), Mapping)
        else {}
    )
    fallback_rows: list[dict[str, object]] = []
    trusted_rows = 0
    source_identity_rows = 0
    for index, row in enumerate(rows):
        fit_kind = str(row.get("fit_source_resolution_kind", "") or "").strip().lower()
        resolution_kind = str(row.get("resolution_kind", "") or "").strip().lower()
        fallback_used = bool(row.get("rebinding_fallback_used", False))
        request_fallback = bool(row.get("optimizer_request_fallback_row", False))
        is_fallback = bool(
            request_fallback
            or fallback_used
            or "fallback" in fit_kind
            or resolution_kind == "hkl_fallback"
        )
        if is_fallback:
            fallback_rows.append(
                {
                    "row_index": int(index),
                    "pair_id": row.get("pair_id"),
                    "hkl": row.get("hkl"),
                    "fit_source_resolution_kind": row.get("fit_source_resolution_kind"),
                    "resolution_kind": row.get("resolution_kind"),
                    "rebinding_fallback_used": fallback_used,
                    "optimizer_request_fallback_reason": row.get(
                        "optimizer_request_fallback_reason"
                    ),
                }
            )
        if bool(row.get("optimizer_request_has_fixed_source", False)) or (
            row.get("source_reflection_namespace") in {"full", "full_reflection", "miller"}
            or bool(row.get("source_reflection_is_full", False))
        ):
            trusted_rows += 1
        if any(
            row.get(key) is not None
            for key in (
                "source_reflection_index",
                "source_table_index",
                "source_row_index",
                "source_branch_index",
                "source_peak_index",
            )
        ):
            source_identity_rows += 1
    computed = {
        "handoff_row_count": int(len(rows)),
        "handoff_source_identity_row_count": int(source_identity_rows),
        "handoff_trusted_full_source_row_count": int(trusted_rows),
        "provider_pair_count": int(bridge_summary.get("provider_pair_count", 0) or 0),
        "dataset_pair_count": int(bridge_summary.get("dataset_pair_count", 0) or 0),
        "optimizer_request_pair_count": int(len(rows)),
        "fixed_source_pair_count": int(trusted_rows),
        "fallback_row_count": int(len(fallback_rows)),
        "fixed_source_resolution_fallback_count": int(len(fallback_rows)),
        "missing_fixed_source_count": max(0, int(len(rows)) - int(trusted_rows)),
        "provider_to_optimizer_identity_match": bool(
            bridge_summary.get("provider_to_optimizer_identity_match", False)
        ),
        "provider_to_optimizer_point_match": bool(
            bridge_summary.get("provider_to_optimizer_point_match", False)
        ),
        "provider_row_fallbacks": fallback_rows,
    }
    computed.update(bridge_summary)
    computed["provider_row_fallback_count"] = int(computed.get("fallback_row_count", 0) or 0)
    computed["provider_row_fallbacks"] = fallback_rows
    for key in REQUEST_HANDOFF_FIELDS:
        computed.setdefault(key, None if key.startswith("provider_to_") else 0)
    return computed


def _metric_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _summary_rms(summary: Mapping[str, object] | None) -> float:
    if not isinstance(summary, Mapping):
        return float("nan")
    return _metric_float(summary.get("unweighted_peak_rms_px", summary.get("rms_px", np.nan)))


def _summary_max(summary: Mapping[str, object] | None) -> float:
    if not isinstance(summary, Mapping):
        return float("nan")
    return _metric_float(summary.get("unweighted_peak_max_px", summary.get("max_error_px", np.nan)))


def _point_match_summary(result: object) -> dict[str, object]:
    summary = getattr(result, "point_match_summary", None)
    return dict(summary) if isinstance(summary, Mapping) else {}


def _summary_int(summary: Mapping[str, object], key: str) -> int:
    try:
        return int(summary.get(key, 0) or 0)
    except Exception:
        return 0


def _append_unique(target: list[str], items: Sequence[object]) -> None:
    for item in items:
        text = str(item)
        if text and text not in target:
            target.append(text)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item)
        if text and text not in result:
            result.append(text)
    return result


def _as_mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _bounds_for_request(request: gui_geometry_fit.GeometryFitSolverRequest) -> dict[str, object]:
    cfg = request.refinement_config if isinstance(request.refinement_config, Mapping) else {}
    bounds = cfg.get("bounds", {}) if isinstance(cfg, Mapping) else {}
    return dict(bounds) if isinstance(bounds, Mapping) else {}


def _param_report(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    result: object,
) -> list[dict[str, object]]:
    final_x = np.asarray(getattr(result, "x", []), dtype=float).reshape(-1)
    bounds = _bounds_for_request(request)
    params: list[dict[str, object]] = []
    for idx, name in enumerate(request.var_names):
        start = _metric_float(request.params.get(str(name), np.nan))
        final = float(final_x[idx]) if idx < final_x.size and np.isfinite(final_x[idx]) else float("nan")
        bound = bounds.get(str(name))
        lower = upper = float("nan")
        in_bounds = True
        if isinstance(bound, Sequence) and not isinstance(bound, (str, bytes)) and len(bound) >= 2:
            lower = _metric_float(bound[0])
            upper = _metric_float(bound[1])
            if math.isfinite(lower) and math.isfinite(upper) and math.isfinite(final):
                in_bounds = lower - 1.0e-9 <= final <= upper + 1.0e-9
        params.append(
            {
                "name": str(name),
                "start": start,
                "final": final,
                "delta": final - start if math.isfinite(start) and math.isfinite(final) else float("nan"),
                "lower": lower,
                "upper": upper,
                "within_bounds": bool(in_bounds),
            }
        )
    return params


def _rejection_reasons(result: object, rms: float) -> list[str]:
    try:
        return gui_geometry_fit.build_geometry_fit_rejection_reason_lines(result, rms=float(rms))
    except Exception as exc:
        return [f"rejection_reason_build_failed: {exc}"]


def _apply_single_param_fields(report: dict[str, object]) -> None:
    names = _as_str_list(report.get("var_names"))
    if not names:
        names = _as_str_list(report.get("active_params"))
    candidates = _as_str_list(report.get("candidate_param_names"))
    if len(names) == 1:
        report["param_name"] = str(report.get("param_name") or names[0])
        report["var_names"] = [names[0]]
        if not candidates:
            report["candidate_param_names"] = [names[0]]
    elif names:
        report["var_names"] = list(names)
    if len(candidates) == 1:
        report["candidate_param_names"] = [candidates[0]]

    param_name = str(report.get("param_name", ""))
    deltas = report.get("parameter_deltas", [])
    entry: Mapping[str, object] | None = None
    if isinstance(deltas, Sequence) and not isinstance(deltas, (str, bytes)):
        for raw_entry in deltas:
            if not isinstance(raw_entry, Mapping):
                continue
            if str(raw_entry.get("name", "")) == param_name or entry is None:
                entry = raw_entry
            if str(raw_entry.get("name", "")) == param_name:
                break
    if entry is None:
        return

    before = _metric_float(entry.get("start", np.nan))
    after = _metric_float(entry.get("final", np.nan))
    delta = _metric_float(entry.get("delta", np.nan))
    lower = _metric_float(entry.get("lower", np.nan))
    upper = _metric_float(entry.get("upper", np.nan))
    report["parameter_before"] = before
    report["parameter_after"] = after
    report["parameter_delta"] = delta
    report["parameter_bounds"] = {"lower": lower, "upper": upper}
    report["parameter_within_bounds"] = bool(entry.get("within_bounds", True))


def _residual_norm_from_value(value: object) -> float:
    numeric = _metric_float(value)
    if math.isfinite(numeric):
        return float(numeric)
    return float("nan")


def _residual_norm_from_cost(value: object) -> float:
    cost = _metric_float(value)
    if math.isfinite(cost) and cost >= 0.0:
        return float(math.sqrt(2.0 * cost))
    return float("nan")


def _finite_residual_norm(value: object) -> tuple[float, bool]:
    residual_arr = np.asarray(value, dtype=float).reshape(-1)
    if residual_arr.size == 0 or not bool(np.all(np.isfinite(residual_arr))):
        return float("nan"), False
    return float(np.linalg.norm(residual_arr)), True


def _current_bounds_for_param(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    param_name: str,
) -> dict[str, object] | None:
    bound = _bounds_for_request(request).get(str(param_name))
    if not isinstance(bound, Sequence) or isinstance(bound, (str, bytes)) or len(bound) < 2:
        return None
    return {
        "lower": _metric_float(bound[0]),
        "upper": _metric_float(bound[1]),
    }


def _live_fixed_source_counter_failures(summary: Mapping[str, object] | None) -> list[str]:
    if not isinstance(summary, Mapping) or not summary:
        return ["missing_point_match_summary"]
    failures: list[str] = []
    fixed_value = None
    for key in (
        "fixed_source_pair_count",
        "fixed_source_resolved_count",
        "matched_fixed_pair_count",
        "fixed_source_reflection_count",
    ):
        if key in summary:
            fixed_value = summary.get(key)
            break
    fixed_actual = _safe_int(fixed_value, default=-999999)
    if fixed_actual != EXPECTED_PROVIDER_PAIR_COUNT:
        failures.append(
            "fixed_source_resolved_count_"
            f"{fixed_actual}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}"
        )
    for key, expected in LIVE_HEARTBEAT_COUNTERS.items():
        if key == "fixed_source_resolved_count":
            continue
        actual = _safe_int(summary.get(key, -999999), default=-999999)
        if actual != expected:
            failures.append(f"{key}_{actual}_expected_{expected}")
    for key in ("fallback_hkl_count", "subset_fallback_hkl_count", "missing_fixed_pair_count"):
        if key not in summary:
            continue
        actual = _safe_int(summary.get(key, -999999), default=-999999)
        if actual != 0:
            failures.append(f"{key}_{actual}_expected_0")
    return failures


def _heartbeat_fixed_source_clean(summary: Mapping[str, object] | None) -> tuple[bool, list[str]]:
    failures = _live_fixed_source_counter_failures(summary)
    return not failures, failures


def _finite_timeout_progress(report: Mapping[str, object]) -> bool:
    last_nfev = _safe_int(report.get("last_nfev", report.get("nfev")), default=0)
    last_residual = _metric_float(report.get("last_residual_norm", np.nan))
    last_rms = _metric_float(report.get("last_rms_px", np.nan))
    last_max = _metric_float(report.get("last_max_error_px", np.nan))
    return bool(
        last_nfev > 0
        and math.isfinite(last_residual)
        and math.isfinite(last_rms)
        and math.isfinite(last_max)
    )


def _diagnosis_classification(report: Mapping[str, object]) -> str | None:
    status = str(report.get("status", ""))
    heartbeat_dirty = bool(report.get("fixed_source_counters_dirty_seen", False))
    heartbeat_failures = (
        _as_str_list(report.get("fixed_source_counter_failures_seen"))
        + _as_str_list(report.get("fixed_source_counter_failures_at_last_heartbeat"))
    )
    if status == "timeout":
        if heartbeat_dirty or any(
            failure != "missing_point_match_summary" for failure in heartbeat_failures
        ):
            return "fixed_source_or_pair_integrity_lost"
        if (
            _finite_timeout_progress(report)
            and bool(report.get("fixed_source_counters_clean_at_last_heartbeat", False))
            and report.get("child_process_killed_cleanly") is True
        ):
            return "slow_needs_separate_strategy"
        return "hang_solver_pathology"
    if heartbeat_dirty or heartbeat_failures:
        return "fixed_source_or_pair_integrity_lost"
    if _one_param_integrity_failures(report):
        return "fixed_source_or_pair_integrity_lost"
    residual_finite = bool(report.get("residuals_finite", False))
    last_residual = _metric_float(report.get("last_residual_norm", report.get("residual_norm", np.nan)))
    after_rms = _metric_float(report.get("after_rms_px", report.get("last_rms_px", np.nan)))
    after_max = _metric_float(
        report.get("after_max_error_px", report.get("last_max_error_px", np.nan))
    )
    if (
        status in {"ok", "pass"}
        and residual_finite
        and math.isfinite(last_residual)
        and math.isfinite(after_rms)
        and math.isfinite(after_max)
    ):
        return "usable"
    return None


def _apply_one_param_diagnostic_aliases(report: dict[str, object]) -> None:
    trace = _as_mapping_list(report.get("residual_eval_trace"))
    last_record = (
        dict(report.get("last_residual_eval"))
        if isinstance(report.get("last_residual_eval"), Mapping)
        else (dict(trace[-1]) if trace else {})
    )
    point_summary = (
        dict(report.get("last_point_match_summary"))
        if isinstance(report.get("last_point_match_summary"), Mapping)
        else (
            dict(last_record.get("point_match_summary"))
            if isinstance(last_record.get("point_match_summary"), Mapping)
            else (
                dict(report.get("point_match_summary"))
                if isinstance(report.get("point_match_summary"), Mapping)
                else {}
            )
        )
    )
    if trace:
        report["residual_eval_trace"] = trace
        report["last_residual_eval"] = last_record
    report.setdefault("heartbeat_count", int(len(trace)))
    report.setdefault("last_heartbeat_elapsed_s", last_record.get("elapsed_s"))
    report.setdefault(
        "last_nfev",
        last_record.get("nfev", report.get("nfev")),
    )
    residual_norm = _residual_norm_from_value(
        last_record.get("residual_norm", report.get("residual_norm", np.nan))
    )
    if not math.isfinite(residual_norm):
        residual_norm = _residual_norm_from_cost(
            last_record.get("cost", report.get("cost", report.get("last_cost", np.nan)))
        )
    report.setdefault("last_residual_norm", residual_norm)
    report.setdefault("last_cost", last_record.get("cost", report.get("cost")))
    report.setdefault(
        "last_rms_px",
        last_record.get("rms_px", report.get("after_rms_px", report.get("current_rms_px"))),
    )
    report.setdefault(
        "last_max_error_px",
        last_record.get("max_error_px", report.get("after_max_error_px")),
    )
    report.setdefault(
        "last_parameter_value",
        last_record.get("parameter_value", report.get("parameter_after")),
    )
    report.setdefault(
        "current_bounds",
        last_record.get("bounds", report.get("parameter_bounds")),
    )
    report.setdefault("last_point_match_summary", point_summary if point_summary else None)
    if "fixed_source_counters_clean_at_last_heartbeat" not in report:
        if isinstance(last_record.get("fixed_source_counters_clean"), bool):
            report["fixed_source_counters_clean_at_last_heartbeat"] = bool(
                last_record.get("fixed_source_counters_clean")
            )
            report["fixed_source_counter_failures_at_last_heartbeat"] = _as_str_list(
                last_record.get("fixed_source_counter_failures")
            )
        elif point_summary:
            clean, failures = _heartbeat_fixed_source_clean(point_summary)
            report["fixed_source_counters_clean_at_last_heartbeat"] = bool(clean)
            report["fixed_source_counter_failures_at_last_heartbeat"] = failures
        elif str(report.get("status", "")) != "timeout":
            failures = _one_param_integrity_failures(report)
            report["fixed_source_counters_clean_at_last_heartbeat"] = not failures
            report["fixed_source_counter_failures_at_last_heartbeat"] = failures
        else:
            report["fixed_source_counters_clean_at_last_heartbeat"] = False
            report["fixed_source_counter_failures_at_last_heartbeat"] = []
    report.setdefault("fixed_source_counters_dirty_seen", False)
    report.setdefault("fixed_source_counter_failures_seen", [])
    report.setdefault("child_process_killed_cleanly", None)
    report.setdefault("dirty_timeout_abort", False)
    report["diagnosis_classification"] = _diagnosis_classification(report)


def _result_report(
    *,
    request: gui_geometry_fit.GeometryFitSolverRequest,
    result: object,
    rung: int,
    rung_name: str,
    started_at: float,
    before_summary: Mapping[str, object] | None = None,
    status: str = "ok",
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    point_summary = _point_match_summary(result)
    residual_norm, residuals_finite = _finite_residual_norm(getattr(result, "fun", []))
    after_rms = _summary_rms(point_summary)
    if not math.isfinite(after_rms):
        after_rms = _metric_float(getattr(result, "rms_px", np.nan))
    after_max = _summary_max(point_summary)
    rejection_reasons = _rejection_reasons(result, after_rms)
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": str(status),
        "active_params": [str(name) for name in request.var_names],
        "var_names": [str(name) for name in request.var_names],
        "candidate_param_names": (
            [str(name) for name in request.candidate_param_names]
            if request.candidate_param_names is not None
            else None
        ),
        "solver_config": dict(
            request.refinement_config.get("solver", {})
            if isinstance(request.refinement_config, Mapping)
            and isinstance(request.refinement_config.get("solver"), Mapping)
            else {}
        ),
        "before_rms_px": _summary_rms(before_summary),
        "before_max_error_px": _summary_max(before_summary),
        "after_rms_px": float(after_rms),
        "after_max_error_px": float(after_max),
        "residual_norm": residual_norm,
        "matched_pair_count": int(point_summary.get("matched_pair_count", 0) or 0),
        "missing_pair_count": int(point_summary.get("missing_pair_count", 0) or 0),
        "branch_mismatch_count": int(point_summary.get("branch_mismatch_count", 0) or 0),
        "nfev": getattr(result, "nfev", None),
        "elapsed_seconds": float(max(0.0, _perf_counter() - started_at)),
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "") or ""),
        "rejection_reasons": rejection_reasons,
        "rejection_reason": " ".join(str(reason) for reason in rejection_reasons),
        "point_match_summary": point_summary,
        "stage_timing_s": dict(
            getattr(result, "geometry_fit_stage_timings", {})
            if isinstance(getattr(result, "geometry_fit_stage_timings", {}), Mapping)
            else {}
        ),
        "parameter_deltas": _param_report(request, result),
    }
    report["elapsed_s"] = report["elapsed_seconds"]
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = _summary_int(point_summary, key)
    report.update(_request_handoff_summary(request))
    report["branch_mismatch_count"] = int(
        point_summary.get("branch_mismatch_count", 0) or 0
    )
    if isinstance(extra, Mapping):
        report.update(dict(extra))
    report["residuals_finite"] = bool(
        residuals_finite
        and math.isfinite(_metric_float(report.get("before_rms_px", np.nan)))
        and math.isfinite(_metric_float(report.get("after_rms_px", np.nan)))
        and math.isfinite(_metric_float(report.get("before_max_error_px", np.nan)))
        and math.isfinite(_metric_float(report.get("after_max_error_px", np.nan)))
    )
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    fallback_failures = _strict_no_fallback_failures(report)
    if fallback_failures:
        report["fallback_guard_failures"] = fallback_failures
    report["pass"] = bool(_rung_passed(report))
    if report["status"] == "ok" and not bool(report["pass"]):
        report["status"] = "fail"
    return report


def _strict_no_fallback_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        if item not in seen:
            seen.add(item)
            failures.append(item)

    def _report_int(key: str) -> int:
        try:
            return int(report.get(key, 0) or 0)
        except Exception:
            return 0

    for key in (
        "fallback_row_count",
        "fixed_source_resolution_fallback_count",
        "missing_fixed_source_count",
        "provider_row_fallback_count",
        "fallback_entry_count",
        "fallback_hkl_count",
        "subset_fallback_hkl_count",
        "missing_fixed_pair_count",
    ):
        value = _report_int(key)
        if value != 0:
            _add(f"{key}_{value}_expected_0")
    for key in (
        "measured_count",
        "fixed_source_resolved_count",
        "fixed_source_reflection_count",
        "matched_fixed_pair_count",
    ):
        value = _report_int(key)
        if value and value != EXPECTED_PROVIDER_PAIR_COUNT:
            _add(f"{key}_{value}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}")
    if _report_int("measured_count") == EXPECTED_PROVIDER_PAIR_COUNT:
        for key in (
            "fixed_source_resolved_count",
            "fixed_source_reflection_count",
            "matched_fixed_pair_count",
        ):
            value = _report_int(key)
            if value != EXPECTED_PROVIDER_PAIR_COUNT:
                _add(f"{key}_{value}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}")
    optimizer_pair_count = _report_int("optimizer_request_pair_count")
    fixed_source_pair_count = _report_int("fixed_source_pair_count")
    if optimizer_pair_count:
        if fixed_source_pair_count != optimizer_pair_count:
            _add(
                "fixed_source_pair_count_"
                f"{fixed_source_pair_count}_expected_{optimizer_pair_count}"
            )
        for key in (
            "provider_to_optimizer_identity_match",
            "provider_to_optimizer_point_match",
        ):
            if report.get(key) is False:
                _add(f"{key}_false")
    return failures


def _one_param_integrity_failures(report: Mapping[str, object]) -> list[str]:
    return _fixed_source_contract_failures(
        report,
        expected_counts=ONE_PARAM_FIXED_SOURCE_COUNTS,
        required_bool_keys=PROVIDER_MATCH_BOOLS,
    )


def _fixed_source_contract_failures(
    report: Mapping[str, object],
    *,
    expected_counts: Mapping[str, int],
    required_bool_keys: Sequence[str],
    prefix: str = "",
    require_present: bool = True,
) -> list[str]:
    failures: list[str] = []
    for key, expected in expected_counts.items():
        if not require_present and key not in report:
            continue
        raw_value = report.get(key, -999999)
        actual = -999999 if isinstance(raw_value, bool) else _safe_int(raw_value, default=-999999)
        if actual != expected:
            failures.append(f"{prefix}{key}_{actual}_expected_{expected}")
    for key in required_bool_keys:
        if not require_present and key not in report:
            continue
        actual = report.get(key)
        if actual is not True:
            failures.append(f"{prefix}{key}_{actual}_expected_True")
    return failures


def _one_param_metric_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    residual_norm = _metric_float(
        report.get("last_residual_norm", report.get("residual_norm", np.nan))
    )
    before_rms = _metric_float(report.get("before_rms_px", np.nan))
    after_rms = _metric_float(report.get("after_rms_px", np.nan))
    before_max = _metric_float(report.get("before_max_error_px", np.nan))
    after_max = _metric_float(report.get("after_max_error_px", np.nan))
    if (
        not bool(report.get("residuals_finite", False))
        or not math.isfinite(residual_norm)
        or not math.isfinite(before_rms)
        or not math.isfinite(after_rms)
    ):
        failures.append("non_finite_residual")
    if not math.isfinite(before_max) or not math.isfinite(after_max):
        failures.append("non_finite_max_error")
    if math.isfinite(before_rms) and math.isfinite(after_rms) and after_rms > before_rms + 0.25:
        failures.append("rms_guard_failed")
    if math.isfinite(before_max) and math.isfinite(after_max) and after_max > before_max + 1.0:
        failures.append("max_error_guard_failed")
    if NO_MATCH_REJECTION in str(report.get("rejection_reason", "")):
        failures.append("no_matched_peak_rejection")
    if bool(report.get("parameter_within_bounds", True)) is not True:
        failures.append("parameter_out_of_bounds")
    return failures


def _rung_passed(report: Mapping[str, object]) -> bool:
    if str(report.get("status", "")) not in {"ok", "pass"}:
        return False
    if not math.isfinite(_metric_float(report.get("after_rms_px", np.nan))):
        return False
    if int(report.get("rung", 0) or 0) == 3:
        if _one_param_integrity_failures(report):
            return False
        if _one_param_metric_failures(report):
            return False
        if bool(report.get("least_squares_called", False)) is not True:
            return False
        if not (
            bool(report.get("optimizer_solve_called", False))
            or bool(report.get("real_solve_called", False))
        ):
            return False
        if bool(report.get("state_hash_unchanged", True)) is not True:
            return False
        names = _as_str_list(report.get("var_names"))
        candidates = _as_str_list(report.get("candidate_param_names"))
        if len(names) != 1 or candidates != names:
            return False
        return True
    if _strict_no_fallback_failures(report):
        return False
    if int(report.get("matched_pair_count", 0) or 0) != 7:
        return False
    if int(report.get("missing_pair_count", 0) or 0) != 0:
        return False
    if int(report.get("branch_mismatch_count", 0) or 0) != 0:
        return False
    rejection_reason = str(report.get("rejection_reason", ""))
    if "No matched peak pairs were available" in rejection_reason:
        return False
    before_rms = _metric_float(report.get("before_rms_px", np.nan))
    after_rms = _metric_float(report.get("after_rms_px", np.nan))
    if math.isfinite(before_rms) and math.isfinite(after_rms) and after_rms > before_rms + 0.25:
        return False
    before_max = _metric_float(report.get("before_max_error_px", np.nan))
    after_max = _metric_float(report.get("after_max_error_px", np.nan))
    if math.isfinite(before_max) and math.isfinite(after_max) and after_max > before_max + 1.0:
        return False
    for entry in report.get("parameter_deltas", []) or []:
        if isinstance(entry, Mapping) and not bool(entry.get("within_bounds", True)):
            return False
    if bool(report.get("least_squares_called", False)):
        return False
    if bool(report.get("optimizer_solve_called", False)):
        return False
    if "objective_dry_run_residual_finite" in report and not bool(
        report.get("objective_dry_run_residual_finite", False)
    ):
        return False
    return True


class _ProbeLeastSquares:
    def __init__(self, *, mode: str) -> None:
        self.mode = str(mode)
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _residual_norm(residual: np.ndarray) -> float:
        finite = residual[np.isfinite(residual)]
        if not finite.size:
            return float("nan")
        return float(np.linalg.norm(finite))

    @staticmethod
    def _step_for_value(value: float) -> float:
        if not math.isfinite(value):
            return 1.0e-6
        return float(max(abs(float(value)) * 1.0e-4, 1.0e-6))

    @staticmethod
    def _bounds(kwargs: Mapping[str, object], size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        lower, upper = kwargs.get("bounds", (None, None))
        lower_arr = np.asarray(lower, dtype=float).reshape(-1) if lower is not None else None
        upper_arr = np.asarray(upper, dtype=float).reshape(-1) if upper is not None else None
        if lower_arr is not None and lower_arr.size != size:
            lower_arr = None
        if upper_arr is not None and upper_arr.size != size:
            upper_arr = None
        return lower_arr, upper_arr

    def _safe_eval(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        vector: np.ndarray,
        *,
        label: str,
        moved: bool = True,
    ) -> tuple[dict[str, object], np.ndarray | None]:
        try:
            residual = np.asarray(fun(np.array(vector, dtype=float)), dtype=float)
        except Exception as exc:
            return (
                {
                    "label": str(label),
                    "x": np.asarray(vector, dtype=float).tolist(),
                    "moved": bool(moved),
                    "raised": True,
                    "error_text": str(exc),
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                },
                None,
            )
        return (
            {
                "label": str(label),
                "x": np.asarray(vector, dtype=float).tolist(),
                "moved": bool(moved),
                "raised": False,
                "error_text": "",
                "residual_size": int(residual.size),
                "residual_norm": self._residual_norm(residual),
                "finite": bool(np.all(np.isfinite(residual))),
            },
            residual,
        )

    def __call__(self, fun: Callable[[np.ndarray], np.ndarray], x0, *args, **kwargs):
        del args
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if self.mode == "sensitivity":
            base_eval, residual0 = self._safe_eval(fun, x0_arr, label="base")
            if residual0 is None:
                residual0 = np.array([float("nan")], dtype=float)
        else:
            residual0 = np.asarray(fun(np.array(x0_arr, dtype=float)), dtype=float)
            base_eval = {
                "label": "base",
                "x": x0_arr.tolist(),
                "moved": True,
                "raised": False,
                "error_text": "",
                "residual_size": int(residual0.size),
                "residual_norm": self._residual_norm(residual0),
                "finite": bool(np.all(np.isfinite(residual0))),
            }
        record: dict[str, object] = {
            "x0": x0_arr.tolist(),
            "residual_size": int(base_eval.get("residual_size", 0) or 0),
            "residual_norm": _metric_float(base_eval.get("residual_norm", np.nan)),
            "finite": bool(base_eval.get("finite", False)),
        }
        if self.mode == "sensitivity" and x0_arr.size:
            lower_arr, upper_arr = self._bounds(kwargs, x0_arr.size)
            base_value = float(x0_arr[0])
            requested_step = self._step_for_value(base_value)

            def _trial(direction: float) -> tuple[np.ndarray, float, bool]:
                target = np.array(x0_arr, dtype=float)
                requested = float(direction) * requested_step
                target[0] = target[0] + requested
                clipped = False
                if lower_arr is not None:
                    clipped = clipped or bool(np.any(target < lower_arr))
                    target = np.maximum(target, lower_arr)
                if upper_arr is not None:
                    clipped = clipped or bool(np.any(target > upper_arr))
                    target = np.minimum(target, upper_arr)
                applied = float(target[0] - x0_arr[0])
                if not math.isclose(applied, requested, rel_tol=0.0, abs_tol=1.0e-15):
                    clipped = True
                return target, applied, clipped

            plus_x, plus_applied, plus_clipped = _trial(1.0)
            minus_x, minus_applied, minus_clipped = _trial(-1.0)

            evals = [dict(base_eval)]
            nfev = 1
            if abs(plus_applied) > 0.0:
                plus_eval, residual_plus = self._safe_eval(fun, plus_x, label="plus")
                if residual_plus is not None and residual_plus.shape == residual0.shape:
                    delta = residual_plus - residual0
                    plus_eval["delta_norm"] = self._residual_norm(delta)
                else:
                    plus_eval["delta_norm"] = float("nan")
                nfev += 1
            else:
                plus_eval = {
                    "label": "plus",
                    "x": plus_x.tolist(),
                    "moved": False,
                    "raised": False,
                    "error_text": "",
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                    "delta_norm": float("nan"),
                }
            plus_eval["step_applied"] = float(plus_applied)
            plus_eval["clipped"] = bool(plus_clipped)
            evals.append(plus_eval)

            if abs(minus_applied) > 0.0:
                minus_eval, residual_minus = self._safe_eval(fun, minus_x, label="minus")
                if residual_minus is not None and residual_minus.shape == residual0.shape:
                    delta = residual_minus - residual0
                    minus_eval["delta_norm"] = self._residual_norm(delta)
                else:
                    minus_eval["delta_norm"] = float("nan")
                nfev += 1
            else:
                minus_eval = {
                    "label": "minus",
                    "x": minus_x.tolist(),
                    "moved": False,
                    "raised": False,
                    "error_text": "",
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                    "delta_norm": float("nan"),
                }
            minus_eval["step_applied"] = float(minus_applied)
            minus_eval["clipped"] = bool(minus_clipped)
            evals.append(minus_eval)

            record.update(
                {
                    "requested_step": float(requested_step),
                    "base_value": float(base_value),
                    "evals": evals,
                    "plus_step_applied": float(plus_applied),
                    "minus_step_applied": float(minus_applied),
                    "plus_clipped": bool(plus_clipped),
                    "minus_clipped": bool(minus_clipped),
                    "delta_step": float(plus_applied),
                    "delta_norm": _metric_float(plus_eval.get("delta_norm", np.nan)),
                    "delta_finite": bool(plus_eval.get("finite", False)),
                }
            )
        else:
            nfev = 1
        self.records.append(record)
        return opt.OptimizeResult(
            x=x0_arr,
            fun=residual0,
            success=True,
            status=1,
            message=f"{self.mode} probe",
            nfev=nfev,
            active_mask=np.zeros(x0_arr.shape, dtype=int),
            optimality=float("nan"),
        )


def _run_with_probe_least_squares(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    *,
    mode: str,
) -> tuple[object, list[dict[str, object]]]:
    probe = _ProbeLeastSquares(mode=mode)
    live_payloads: list[dict[str, object]] = []

    def _live_update(payload: Mapping[str, object]) -> None:
        live_payloads.append(dict(payload))

    original = opt.least_squares
    opt.least_squares = probe
    try:
        result = gui_geometry_fit.solve_geometry_fit_request(
            request,
            solve_fit=opt.fit_geometry_parameters,
            live_update_callback=_live_update,
        )
    finally:
        opt.least_squares = original
    records = list(probe.records)
    payload_index = 0
    for record in records:
        evals = record.get("evals")
        if isinstance(evals, list):
            for entry in evals:
                if not isinstance(entry, dict) or not bool(entry.get("moved", True)):
                    continue
                if payload_index >= len(live_payloads):
                    continue
                entry["live_update_payload"] = dict(live_payloads[payload_index])
                point_summary = live_payloads[payload_index].get("point_match_summary")
                if isinstance(point_summary, Mapping):
                    entry["point_match_summary"] = dict(point_summary)
                payload_index += 1
            continue
        if payload_index < len(live_payloads):
            record["live_update_payload"] = dict(live_payloads[payload_index])
            point_summary = live_payloads[payload_index].get("point_match_summary")
            if isinstance(point_summary, Mapping):
                record["point_match_summary"] = dict(point_summary)
            payload_index += 1
    return result, records


def _request_only_report(
    *,
    request: gui_geometry_fit.GeometryFitSolverRequest,
    rung: int,
    rung_name: str,
    started_at: float,
    status: str,
    failure_reason: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": str(status),
        "pass": False,
        "failure_reason": str(failure_reason),
        "active_params": [str(name) for name in request.var_names],
        "var_names": [str(name) for name in request.var_names],
        "candidate_param_names": (
            [str(name) for name in request.candidate_param_names]
            if request.candidate_param_names is not None
            else None
        ),
        "solver_config": dict(
            request.refinement_config.get("solver", {})
            if isinstance(request.refinement_config, Mapping)
            and isinstance(request.refinement_config.get("solver"), Mapping)
            else {}
        ),
        "before_rms_px": float("nan"),
        "before_max_error_px": float("nan"),
        "after_rms_px": float("nan"),
        "after_max_error_px": float("nan"),
        "matched_pair_count": 0,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "nfev": None,
        "elapsed_seconds": float(max(0.0, _perf_counter() - started_at)),
        "elapsed_s": float(max(0.0, _perf_counter() - started_at)),
        "solver_success": False,
        "solver_message": "",
        "rejection_reasons": [],
        "rejection_reason": "",
        "point_match_summary": {},
        "stage_timing_s": {},
        "parameter_deltas": [],
        "optimizer_called": False,
        "objective_eval_called": False,
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "objective_dry_run_residual_finite": False,
    }
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = 0
    report.update(_request_handoff_summary(request))
    if isinstance(extra, Mapping):
        report.update(dict(extra))
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    return report


def run_objective_dry_run(
    context: Mapping[str, object],
    *,
    output_path: Path,
    max_nfev: int,
) -> dict[str, object]:
    started = _perf_counter()
    request = build_solver_request(
        context,
        ["center_x"],
        max_nfev=max_nfev,
    )
    request_summary = _request_handoff_summary(request)
    request_fallback_failures = _strict_no_fallback_failures(request_summary)
    if request_fallback_failures:
        report = _request_only_report(
            request=request,
            rung=1,
            rung_name="objective_dry_run",
            started_at=started,
            status="failed",
            failure_reason="optimizer_request_used_fallback_rows",
            extra={
                "fallback_guard_failures": request_fallback_failures,
                "least_squares_probe_records": [],
                "objective_eval_called": False,
                "least_squares_called": False,
                "optimizer_solve_called": False,
                "objective_dry_run_residual_finite": False,
            },
        )
        _write_json(output_path, report)
        return report
    result, records = _run_with_probe_least_squares(request, mode="dry_run")
    residual_finite = any(
        bool(record.get("finite", False))
        and math.isfinite(_metric_float(record.get("residual_norm", np.nan)))
        for record in records
        if isinstance(record, Mapping)
    )
    report = _result_report(
        request=request,
        result=result,
        rung=1,
        rung_name="objective_dry_run",
        started_at=started,
        status="ok",
        extra={
            "least_squares_probe_records": records,
            "objective_eval_called": bool(records),
            "least_squares_called": False,
            "optimizer_solve_called": False,
            "objective_dry_run_residual_finite": bool(residual_finite),
        },
    )
    fallback_failures = _strict_no_fallback_failures(report)
    if fallback_failures:
        report["status"] = "failed"
        report["failure_reason"] = "optimizer_request_used_fallback_rows"
        report["fallback_guard_failures"] = fallback_failures
    if int(report.get("matched_pair_count", 0) or 0) != 7:
        report["status"] = "failed"
        report.setdefault("failure_reason", "pair_count_not_7")
    if int(report.get("missing_pair_count", 0) or 0) != 0:
        report["status"] = "failed"
        report.setdefault("failure_reason", "missing_pairs_present")
    if int(report.get("branch_mismatch_count", 0) or 0) != 0:
        report["status"] = "failed"
        report.setdefault("failure_reason", "branch_mismatch_present")
    if not math.isfinite(_metric_float(report.get("after_rms_px", np.nan))):
        report["status"] = "failed"
        report.setdefault("failure_reason", "non_finite_initial_residual")
    report["pass"] = str(report.get("status")) == "ok"
    _write_json(output_path, report)
    return report


def _active_theta_name(context: Mapping[str, object]) -> str | None:
    saved_var_names = [str(name) for name in context.get("saved_var_names", []) or []]
    prepared_run = context.get("prepared_run")
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {})
        if prepared_run is not None
        else {}
    )
    if "theta_offset" in saved_var_names and "theta_offset" in fit_params:
        return "theta_offset"
    if "theta_initial" in saved_var_names and "theta_initial" in fit_params:
        return "theta_initial"
    if "theta_offset" in fit_params and "theta_initial" not in fit_params:
        return "theta_offset"
    if "theta_initial" in fit_params:
        return "theta_initial"
    return None


def _candidate_order(context: Mapping[str, object]) -> list[str]:
    theta_name = _active_theta_name(context)
    ordered: list[str] = []
    for name in ONE_PARAM_ORDER:
        if name == "theta_initial" and theta_name is None:
            continue
        actual = theta_name if name == "theta_initial" else name
        if actual not in ordered:
            ordered.append(actual)
    return _coerce_active_names(context, ordered)


def _safe_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        numeric = float(value)
    except Exception:
        return int(default)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return int(default)
    return int(numeric)


def _rung1_green_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if str(report.get("status", "")) != "ok" or bool(report.get("pass", False)) is not True:
        failures.append("rung_1_status_not_ok")
    expected_counts = {
        "provider_pair_count": 7,
        "dataset_pair_count": 7,
        "optimizer_request_pair_count": 7,
        "fixed_source_pair_count": 7,
        "fallback_row_count": 0,
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
    }
    for key, expected in expected_counts.items():
        actual = _safe_int(report.get(key, -999999), default=-999999)
        if actual != expected:
            failures.append(f"{key}_{actual}_expected_{expected}")
    for key in (
        "provider_to_optimizer_identity_match",
        "provider_to_optimizer_point_match",
    ):
        actual = report.get(key)
        if actual is not True:
            failures.append(f"{key}_{actual}_expected_True")
    if bool(report.get("objective_dry_run_residual_finite", False)) is not True:
        failures.append("objective_dry_run_residual_not_finite")
    if bool(report.get("least_squares_called", True)):
        failures.append("least_squares_called")
    if bool(report.get("optimizer_solve_called", True)):
        failures.append("optimizer_solve_called")
    return failures


def _eval_int(summary: Mapping[str, object], keys: Sequence[str], default: int = -1) -> int:
    for key in keys:
        if key not in summary:
            continue
        return _safe_int(summary.get(key, default), default=default)
    return int(default)


def _sensitivity_eval_summary(
    eval_record: Mapping[str, object] | None,
    *,
    request_summary: Mapping[str, object],
) -> dict[str, object]:
    del request_summary
    record = dict(eval_record or {})
    point_summary = (
        dict(record.get("point_match_summary", {}))
        if isinstance(record.get("point_match_summary", {}), Mapping)
        else {}
    )
    moved = bool(record.get("moved", True))
    if not moved:
        point_summary = {}
        counter_source = "not_evaluated"
    elif point_summary:
        counter_source = "point_match_summary"
    else:
        counter_source = "missing"
    summary = {
        "label": str(record.get("label", "")),
        "moved": moved,
        "raised": bool(record.get("raised", False)),
        "error_text": str(record.get("error_text", "") or ""),
        "residual_norm": _metric_float(record.get("residual_norm", np.nan)),
        "finite": bool(record.get("finite", False)),
        "delta_norm": _metric_float(record.get("delta_norm", np.nan)),
        "step_applied": _metric_float(record.get("step_applied", 0.0)),
        "clipped": bool(record.get("clipped", False)),
        "fixed_source_pair_count": _eval_int(
            point_summary,
            (
                "fixed_source_resolved_count",
                "matched_fixed_pair_count",
                "fixed_source_reflection_count",
            ),
            default=-1,
        ),
        "fallback_entry_count": _eval_int(
            point_summary,
            ("fallback_entry_count",),
            default=-1,
        ),
        "matched_pair_count": _eval_int(
            point_summary,
            ("matched_pair_count",),
            default=-1,
        ),
        "missing_pair_count": _eval_int(
            point_summary,
            ("missing_pair_count",),
            default=-1,
        ),
        "branch_mismatch_count": _eval_int(
            point_summary,
            ("branch_mismatch_count",),
            default=-1,
        ),
        "counter_source": counter_source,
    }
    failures: list[str] = []
    if moved and not point_summary:
        failures.append("missing_point_match_summary")
    dirty_counter = False
    if point_summary:
        for key, expected in (
            ("fixed_source_pair_count", 7),
            ("fallback_entry_count", 0),
            ("matched_pair_count", 7),
            ("missing_pair_count", 0),
            ("branch_mismatch_count", 0),
        ):
            actual = _safe_int(summary.get(key, -999999), default=-999999)
            if actual != expected:
                failures.append(f"{key}_{actual}_expected_{expected}")
            if actual < 0:
                dirty_counter = True
    if dirty_counter and counter_source == "point_match_summary":
        counter_source = "missing"
        summary["counter_source"] = counter_source
    summary["fixed_source_clean"] = not failures
    summary["fixed_source_failures"] = failures
    return summary


def _eval_by_label(record: Mapping[str, object], label: str) -> Mapping[str, object] | None:
    evals = record.get("evals")
    if not isinstance(evals, Sequence) or isinstance(evals, (str, bytes)):
        if label == "base":
            return record
        return None
    for entry in evals:
        if isinstance(entry, Mapping) and str(entry.get("label", "")) == label:
            return entry
    return None


def _sensitivity_status(
    *,
    base_eval: Mapping[str, object],
    plus_eval: Mapping[str, object],
    minus_eval: Mapping[str, object],
    threshold: float,
) -> tuple[str, list[str], float]:
    reasons: list[str] = []
    if bool(base_eval.get("raised", False)):
        return "unsafe", ["base_eval_raised"], float("nan")
    if not bool(base_eval.get("fixed_source_clean", False)):
        return "unsafe", list(base_eval.get("fixed_source_failures", []) or []), float("nan")
    moved = [
        eval_report
        for eval_report in (plus_eval, minus_eval)
        if bool(eval_report.get("moved", False))
    ]
    if not moved:
        return "unsafe", ["no_valid_movement"], float("nan")
    for eval_report in moved:
        label = str(eval_report.get("label", "direction"))
        if bool(eval_report.get("raised", False)):
            reasons.append(f"{label}_eval_raised")
        if not bool(eval_report.get("fixed_source_clean", False)):
            reasons.extend(
                f"{label}_{reason}"
                for reason in eval_report.get("fixed_source_failures", []) or []
            )
    if reasons:
        return "unsafe", reasons, float("nan")
    if any(not bool(eval_report.get("finite", False)) for eval_report in moved):
        return "non_finite", ["non_finite_residual"], float("nan")
    delta_values = [
        _metric_float(eval_report.get("delta_norm", np.nan))
        for eval_report in moved
        if math.isfinite(_metric_float(eval_report.get("delta_norm", np.nan)))
    ]
    if not delta_values:
        return "non_finite", ["non_finite_delta"], float("nan")
    sensitivity_values: list[float] = []
    for eval_report in moved:
        delta_norm = _metric_float(eval_report.get("delta_norm", np.nan))
        step = abs(_metric_float(eval_report.get("step_applied", np.nan)))
        if math.isfinite(delta_norm) and math.isfinite(step) and step > 0.0:
            sensitivity_values.append(float(delta_norm / step))
    sensitivity_norm = max(sensitivity_values) if sensitivity_values else float("nan")
    if any(delta > threshold for delta in delta_values):
        return "active", [], sensitivity_norm
    return "near_zero", [], sensitivity_norm


def run_sensitivity_scan(
    context: Mapping[str, object],
    *,
    output_path: Path,
    max_nfev: int,
    rung_1_report: Mapping[str, object] | None = None,
    state_path: Path | None = None,
    state_hash_before: str | None = None,
) -> dict[str, object]:
    started = _perf_counter()
    initial_state_hash = (
        str(state_hash_before)
        if state_hash_before is not None
        else (_state_sha256(Path(state_path)) if state_path is not None else None)
    )
    entries: list[dict[str, object]] = []
    rung_1 = dict(rung_1_report or {})
    rung1_failures = _rung1_green_failures(rung_1)
    if rung1_failures:
        state_hash_after = (
            _state_sha256(Path(state_path)) if state_path is not None else initial_state_hash
        )
        state_hash_unchanged = (
            bool(initial_state_hash == state_hash_after)
            if initial_state_hash is not None and state_hash_after is not None
            else True
        )
        payload = {
            "rung": 2,
            "rung_name": "sensitivity_scan",
            "status": "aborted",
            "pass": False,
            "reason": "rung_1_not_green",
            "rung_1_status": str(rung_1.get("status", "")),
            "rung_1_failures": list(rung1_failures),
            "provider_pair_count": _safe_int(rung_1.get("provider_pair_count"), default=0),
            "dataset_pair_count": _safe_int(rung_1.get("dataset_pair_count"), default=0),
            "optimizer_request_pair_count": _safe_int(
                rung_1.get("optimizer_request_pair_count"),
                default=0,
            ),
            "fixed_source_pair_count": _safe_int(
                rung_1.get("fixed_source_pair_count"),
                default=0,
            ),
            "fallback_row_count": _safe_int(rung_1.get("fallback_row_count"), default=0),
            "fixed_source_resolution_fallback_count": _safe_int(
                rung_1.get("fixed_source_resolution_fallback_count"),
                default=0,
            ),
            "missing_fixed_source_count": _safe_int(
                rung_1.get("missing_fixed_source_count"),
                default=0,
            ),
            "provider_to_optimizer_identity_match": (
                rung_1.get("provider_to_optimizer_identity_match") is True
            ),
            "provider_to_optimizer_point_match": (
                rung_1.get("provider_to_optimizer_point_match") is True
            ),
            "fallback_entry_count": _safe_int(
                rung_1.get("fallback_entry_count"),
                default=0,
            ),
            "residual_probe_called": False,
            "least_squares_called": False,
            "optimizer_solve_called": False,
            "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
            "params": [],
            "parameters": [],
            "active_param_count": 0,
            "near_zero_param_count": 0,
            "non_finite_param_count": 0,
            "unsafe_param_count": 0,
            "active_params": [],
            "near_zero_params": [],
            "non_finite_params": [],
            "unsafe_params": [],
            "active_parameters": [],
            "near_zero_parameters": [],
            "unsafe_parameters": [],
            "state_sha256_before": initial_state_hash,
            "state_sha256_after": state_hash_after,
            "state_hash_unchanged": bool(state_hash_unchanged),
        }
        _write_json(output_path, payload)
        return payload
    residual_probe_called = False
    for name in _candidate_order(context):
        param_started = _perf_counter()
        try:
            request = build_solver_request(context, [name], max_nfev=max_nfev)
            request_summary = _request_handoff_summary(request)
            request_failures = _strict_no_fallback_failures(request_summary)
            if request_failures:
                entries.append(
                    {
                        "param_name": str(name),
                        "name": str(name),
                        "status": "unsafe",
                        "unsafe_reasons": list(request_failures),
                        "elapsed_seconds": float(max(0.0, _perf_counter() - param_started)),
                    }
                )
                continue
            residual_probe_called = True
            result, records = _run_with_probe_least_squares(request, mode="sensitivity")
            probe_record = records[-1] if records else {}
            base_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "base"),
                request_summary=request_summary,
            )
            plus_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "plus"),
                request_summary=request_summary,
            )
            minus_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "minus"),
                request_summary=request_summary,
            )
            residual_norm_base = _metric_float(base_eval.get("residual_norm", np.nan))
            threshold = max(
                1.0e-7,
                1.0e-7 * abs(residual_norm_base)
                if math.isfinite(residual_norm_base)
                else 0.0,
            )
            status, unsafe_reasons, sensitivity_norm = _sensitivity_status(
                base_eval=base_eval,
                plus_eval=plus_eval,
                minus_eval=minus_eval,
                threshold=float(threshold),
            )
            plus_step = _metric_float(plus_eval.get("step_applied", 0.0))
            minus_step = _metric_float(minus_eval.get("step_applied", 0.0))
            entries.append(
                {
                    "param_name": str(name),
                    "name": str(name),
                    "status": str(status),
                    "classification": str(status),
                    "base_value": _metric_float(probe_record.get("base_value", np.nan)),
                    "step_size": _metric_float(probe_record.get("requested_step", np.nan)),
                    "plus_step_applied": plus_step,
                    "minus_step_applied": minus_step,
                    "plus_clipped": bool(plus_eval.get("clipped", False)),
                    "minus_clipped": bool(minus_eval.get("clipped", False)),
                    "residual_norm_base": residual_norm_base,
                    "residual_norm_plus": _metric_float(plus_eval.get("residual_norm", np.nan)),
                    "residual_norm_minus": _metric_float(minus_eval.get("residual_norm", np.nan)),
                    "delta_norm_plus": _metric_float(plus_eval.get("delta_norm", np.nan)),
                    "delta_norm_minus": _metric_float(minus_eval.get("delta_norm", np.nan)),
                    "sensitivity_norm": sensitivity_norm,
                    "finite_plus": bool(plus_eval.get("finite", False)),
                    "finite_minus": bool(minus_eval.get("finite", False)),
                    "unsafe_reasons": list(unsafe_reasons),
                    "base_eval": base_eval,
                    "plus_eval": plus_eval,
                    "minus_eval": minus_eval,
                    "provider_pair_count": _safe_int(
                        request_summary.get("provider_pair_count"),
                        default=-1,
                    ),
                    "dataset_pair_count": _safe_int(
                        request_summary.get("dataset_pair_count"),
                        default=-1,
                    ),
                    "optimizer_request_pair_count": _safe_int(
                        request_summary.get("optimizer_request_pair_count"),
                        default=-1,
                    ),
                    "fixed_source_pair_count": int(base_eval.get("fixed_source_pair_count", -1)),
                    "fallback_row_count": _safe_int(
                        request_summary.get("fallback_row_count"),
                        default=-1,
                    ),
                    "fixed_source_resolution_fallback_count": _safe_int(
                        request_summary.get("fixed_source_resolution_fallback_count"),
                        default=-1,
                    ),
                    "missing_fixed_source_count": _safe_int(
                        request_summary.get("missing_fixed_source_count"),
                        default=-1,
                    ),
                    "fixed_source_resolved_count": int(
                        base_eval.get("fixed_source_pair_count", -1)
                    ),
                    "fallback_entry_count": int(base_eval.get("fallback_entry_count", -1)),
                    "matched_pair_count": int(base_eval.get("matched_pair_count", -1)),
                    "missing_pair_count": int(base_eval.get("missing_pair_count", -1)),
                    "branch_mismatch_count": int(base_eval.get("branch_mismatch_count", -1)),
                    "provider_to_optimizer_identity_match": (
                        request_summary.get("provider_to_optimizer_identity_match") is True
                    ),
                    "provider_to_optimizer_point_match": (
                        request_summary.get("provider_to_optimizer_point_match") is True
                    ),
                    "probe_records": records,
                    "elapsed_seconds": float(max(0.0, _perf_counter() - param_started)),
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "param_name": str(name),
                    "name": str(name),
                    "status": "unsafe",
                    "classification": "unsafe",
                    "error_text": str(exc),
                    "unsafe_reasons": ["residual_eval_raised"],
                    "elapsed_seconds": float(max(0.0, _perf_counter() - param_started)),
                }
            )
    state_hash_after = _state_sha256(Path(state_path)) if state_path is not None else initial_state_hash
    state_hash_unchanged = (
        bool(initial_state_hash == state_hash_after)
        if initial_state_hash is not None and state_hash_after is not None
        else True
    )
    active_params = [str(entry["param_name"]) for entry in entries if entry.get("status") == "active"]
    near_zero_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "near_zero"
    ]
    non_finite_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "non_finite"
    ]
    unsafe_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "unsafe"
    ]
    status = "ok" if active_params and state_hash_unchanged else "fail"
    payload = {
        "rung": 2,
        "rung_name": "sensitivity_scan",
        "status": status,
        "rung_1_status": str(rung_1.get("status", "")),
        "provider_pair_count": _safe_int(rung_1.get("provider_pair_count"), default=0),
        "dataset_pair_count": _safe_int(rung_1.get("dataset_pair_count"), default=0),
        "optimizer_request_pair_count": _safe_int(
            rung_1.get("optimizer_request_pair_count"),
            default=0,
        ),
        "fixed_source_pair_count": _safe_int(
            rung_1.get("fixed_source_pair_count"),
            default=0,
        ),
        "fallback_row_count": _safe_int(rung_1.get("fallback_row_count"), default=0),
        "fixed_source_resolution_fallback_count": _safe_int(
            rung_1.get("fixed_source_resolution_fallback_count"),
            default=0,
        ),
        "missing_fixed_source_count": _safe_int(
            rung_1.get("missing_fixed_source_count"),
            default=0,
        ),
        "fixed_source_resolved_count": _safe_int(
            rung_1.get("fixed_source_resolved_count", rung_1.get("fixed_source_pair_count")),
            default=0,
        ),
        "fallback_entry_count": _safe_int(rung_1.get("fallback_entry_count"), default=0),
        "matched_pair_count": _safe_int(rung_1.get("matched_pair_count"), default=0),
        "missing_pair_count": _safe_int(rung_1.get("missing_pair_count"), default=0),
        "branch_mismatch_count": _safe_int(rung_1.get("branch_mismatch_count"), default=0),
        "provider_to_optimizer_identity_match": (
            rung_1.get("provider_to_optimizer_identity_match") is True
        ),
        "provider_to_optimizer_point_match": (
            rung_1.get("provider_to_optimizer_point_match") is True
        ),
        "residual_probe_called": bool(residual_probe_called),
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
        "params": entries,
        "parameters": entries,
        "active_param_count": int(len(active_params)),
        "near_zero_param_count": int(len(near_zero_params)),
        "non_finite_param_count": int(len(non_finite_params)),
        "unsafe_param_count": int(len(unsafe_params)),
        "active_params": active_params,
        "near_zero_params": near_zero_params,
        "non_finite_params": non_finite_params,
        "unsafe_params": unsafe_params,
        "active_parameters": active_params,
        "near_zero_parameters": near_zero_params,
        "unsafe_parameters": unsafe_params,
        "state_sha256_before": initial_state_hash,
        "state_sha256_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_unchanged),
    }
    payload["pass"] = str(payload["status"]) == "ok"
    _write_json(output_path, payload)
    return payload


def _heartbeat_write(path: Path, payload: Mapping[str, object]) -> None:
    if not path:
        return
    merged = _read_json(path)
    merged.update(dict(payload))
    merged["heartbeat_updated_at"] = time.time()
    _write_json(path, merged)


def _worker_solve_once(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    heartbeat_path: Path,
    max_nfev: int,
    rung: int,
    rung_name: str,
    feature: str | None = None,
    context: Mapping[str, object] | None = None,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    started = _perf_counter()
    phase_timing_s: dict[str, float] = {}
    context_reused = context is not None

    def _record_phase(name: str, phase_started: float) -> None:
        phase_timing_s[name] = float(max(0.0, _perf_counter() - phase_started))

    state_hash_before = _state_sha256(Path(state_path))
    _heartbeat_write(
        heartbeat_path,
        {
            "status": "starting",
            "rung": int(rung),
            "rung_name": str(rung_name),
            "active_params": [str(name) for name in active_names],
            "state_sha256_before": state_hash_before,
            "solver_context_reused": bool(context_reused),
            "diagnostic_logging": bool(diagnostic_logging),
        },
    )
    phase_started = _perf_counter()
    if context is None:
        with _solver_debug_logging_scope(diagnostic_logging):
            context = _capture_solver_context(Path(state_path), int(background_index))
    _record_phase("capture_solver_context_s", phase_started)

    phase_started = _perf_counter()
    with _solver_debug_logging_scope(diagnostic_logging):
        request = build_solver_request(
            context,
            active_names,
            max_nfev=int(max_nfev),
            feature=feature,
        )
    _record_phase("build_solver_request_s", phase_started)
    phase_started = _perf_counter()
    request_summary = _request_handoff_summary(request)
    request_summary["requested_var_names"] = [str(name) for name in request.var_names]
    request_summary["requested_candidate_param_names"] = (
        [str(name) for name in request.candidate_param_names]
        if request.candidate_param_names is not None
        else []
    )
    _heartbeat_write(heartbeat_path, request_summary)
    request_fallback_failures = _strict_no_fallback_failures(request_summary)
    _record_phase("request_handoff_guard_s", phase_started)
    if request_fallback_failures:
        report = _request_only_report(
            request=request,
            rung=int(rung),
            rung_name=str(rung_name),
            started_at=started,
            status="failed",
            failure_reason="optimizer_request_used_fallback_rows",
            extra={
                "feature": feature,
                "fallback_guard_failures": request_fallback_failures,
                "phase_timing_s": dict(phase_timing_s),
                "solver_context_reused": bool(context_reused),
                "diagnostic_logging": bool(diagnostic_logging),
            },
        )
        _write_json(output_path, report)
        _heartbeat_write(heartbeat_path, {"status": str(report.get("status", "")), "done": True})
        return report
    phase_started = _perf_counter()
    with _solver_debug_logging_scope(diagnostic_logging):
        before_result, _records = _run_with_probe_least_squares(request, mode="dry_run")
    _record_phase("dry_run_s", phase_started)
    before_summary = _point_match_summary(before_result)
    param_name = str(active_names[0]) if len(active_names) == 1 else ""
    current_bounds = _current_bounds_for_param(request, param_name) if param_name else None
    heartbeat_count = 0
    residual_eval_trace: list[dict[str, object]] = []
    fixed_source_counters_dirty_seen = False
    fixed_source_counter_failures_seen: list[str] = []

    def _status_callback(text: str) -> None:
        payload: dict[str, object] = {
            "status": "running",
            "last_optimizer_status": str(text),
        }
        if "eval=" in str(text):
            try:
                payload["nfev"] = int(str(text).split("eval=", 1)[1].split()[0])
            except Exception:
                pass
        _heartbeat_write(heartbeat_path, payload)

    def _live_update(payload: Mapping[str, object]) -> None:
        nonlocal heartbeat_count, fixed_source_counters_dirty_seen
        point_summary = (
            dict(payload.get("point_match_summary", {}))
            if isinstance(payload.get("point_match_summary", {}), Mapping)
            else {}
        )
        heartbeat_count += 1
        eval_count = _safe_int(
            payload.get("evaluation_count", payload.get("nfev", heartbeat_count)),
            default=heartbeat_count,
        )
        cost = _metric_float(payload.get("current_cost", payload.get("cost", np.nan)))
        residual_norm = _residual_norm_from_value(payload.get("residual_norm", np.nan))
        if not math.isfinite(residual_norm):
            residual_norm = _residual_norm_from_cost(cost)
        rms_px = _summary_rms(point_summary)
        if not math.isfinite(rms_px):
            rms_px = _metric_float(payload.get("weighted_rms_px", np.nan))
        max_error_px = _summary_max(point_summary)
        params = payload.get("params", {})
        parameter_value = None
        if param_name and isinstance(params, Mapping) and param_name in params:
            parameter_value = params.get(param_name)
        elif isinstance(payload.get("x_trial"), Sequence) and not isinstance(
            payload.get("x_trial"),
            (str, bytes),
        ):
            try:
                parameter_value = list(payload.get("x_trial", []))[0]
            except Exception:
                parameter_value = None
        clean, failures = _heartbeat_fixed_source_clean(point_summary)
        if not clean:
            fixed_source_counters_dirty_seen = True
            _append_unique(fixed_source_counter_failures_seen, failures)
        trace_entry = {
            "eval_count": int(eval_count),
            "nfev": int(eval_count),
            "elapsed_s": float(max(0.0, _perf_counter() - started)),
            "residual_norm": residual_norm,
            "cost": cost,
            "rms_px": rms_px,
            "max_error_px": max_error_px,
            "parameter_name": param_name or None,
            "parameter_value": parameter_value,
            "bounds": current_bounds,
            "point_match_summary": point_summary,
            "fixed_source_counters_clean": bool(clean),
            "fixed_source_counter_failures": failures,
        }
        if heartbeat_count == 1:
            phase_timing_s["first_residual_elapsed_s"] = float(trace_entry["elapsed_s"])
        residual_eval_trace.append(trace_entry)
        _heartbeat_write(
            heartbeat_path,
            {
                "status": "running",
                "last_point_match_summary": point_summary,
                "current_rms_px": rms_px,
                "current_max_error_px": max_error_px,
                "last_residual_eval": trace_entry,
                "residual_eval_trace": residual_eval_trace,
                "heartbeat_count": int(heartbeat_count),
                "last_heartbeat_elapsed_s": trace_entry["elapsed_s"],
                "last_nfev": int(eval_count),
                "nfev": int(eval_count),
                "last_residual_norm": residual_norm,
                "last_cost": cost,
                "last_rms_px": rms_px,
                "last_max_error_px": max_error_px,
                "last_parameter_value": parameter_value,
                "current_bounds": current_bounds,
                "fixed_source_counters_clean_at_last_heartbeat": bool(clean),
                "fixed_source_counter_failures_at_last_heartbeat": failures,
                "fixed_source_counters_dirty_seen": bool(fixed_source_counters_dirty_seen),
                "fixed_source_counter_failures_seen": list(
                    fixed_source_counter_failures_seen
                ),
                "phase_timing_s": dict(phase_timing_s),
                "first_residual_elapsed_s": phase_timing_s.get(
                    "first_residual_elapsed_s"
                ),
                "solver_context_reused": bool(context_reused),
                "diagnostic_logging": bool(diagnostic_logging),
            },
        )

    least_squares_called = False
    optimizer_solve_called = False
    effective_var_names_seen_by_solver: list[str] = []
    effective_candidate_param_names_seen_by_solver: list[str] = []
    try:
        original_least_squares = opt.least_squares

        def _counted_least_squares(*args, **kwargs):
            nonlocal least_squares_called
            least_squares_called = True
            _heartbeat_write(heartbeat_path, {"least_squares_called": True})
            return original_least_squares(*args, **kwargs)

        def _counted_solve_fit(*args, **kwargs):
            nonlocal optimizer_solve_called
            nonlocal effective_var_names_seen_by_solver
            nonlocal effective_candidate_param_names_seen_by_solver
            optimizer_solve_called = True
            try:
                effective_var_names_seen_by_solver = [
                    str(name) for name in (args[5] if len(args) > 5 else [])
                ]
            except Exception:
                effective_var_names_seen_by_solver = []
            try:
                effective_candidate_param_names_seen_by_solver = [
                    str(name)
                    for name in kwargs.get("candidate_param_names", []) or []
                ]
            except Exception:
                effective_candidate_param_names_seen_by_solver = []
            _heartbeat_write(
                heartbeat_path,
                {
                    "optimizer_solve_called": True,
                    "effective_var_names_seen_by_solver": list(
                        effective_var_names_seen_by_solver
                    ),
                    "effective_candidate_param_names_seen_by_solver": list(
                        effective_candidate_param_names_seen_by_solver
                    ),
                },
            )
            return opt.fit_geometry_parameters(*args, **kwargs)

        opt.least_squares = _counted_least_squares
        try:
            phase_started = _perf_counter()
            with _solver_debug_logging_scope(diagnostic_logging):
                result = gui_geometry_fit.solve_geometry_fit_request(
                    request,
                    solve_fit=_counted_solve_fit,
                    status_callback=_status_callback,
                    live_update_callback=_live_update,
                )
            _record_phase("solve_geometry_fit_request_s", phase_started)
        finally:
            opt.least_squares = original_least_squares
        state_hash_after = _state_sha256(Path(state_path))
        report = _result_report(
            request=request,
            result=result,
            rung=int(rung),
            rung_name=str(rung_name),
            started_at=started,
            before_summary=before_summary,
            status="ok",
            extra={
                "feature": feature,
                "least_squares_called": bool(least_squares_called),
                "optimizer_solve_called": bool(optimizer_solve_called),
                "real_solve_called": bool(optimizer_solve_called),
                "effective_var_names_seen_by_solver": list(
                    effective_var_names_seen_by_solver
                ),
                "effective_candidate_param_names_seen_by_solver": list(
                    effective_candidate_param_names_seen_by_solver
                ),
                "state_sha256_before": state_hash_before,
                "state_sha256_after": state_hash_after,
                "state_hash_unchanged": state_hash_before == state_hash_after,
                "heartbeat_count": int(heartbeat_count),
                "residual_eval_trace": list(residual_eval_trace),
                "fixed_source_counters_dirty_seen": bool(
                    fixed_source_counters_dirty_seen
                ),
                "fixed_source_counter_failures_seen": list(
                    fixed_source_counter_failures_seen
                ),
                "dirty_timeout_abort": False,
                "child_process_killed_cleanly": None,
                "phase_timing_s": dict(phase_timing_s),
                "first_residual_elapsed_s": phase_timing_s.get(
                    "first_residual_elapsed_s"
                ),
                "solver_context_reused": bool(context_reused),
                "diagnostic_logging": bool(diagnostic_logging),
            },
        )
    except Exception as exc:
        state_hash_after = _state_sha256(Path(state_path))
        report = {
            "rung": int(rung),
            "rung_name": str(rung_name),
            "status": "error",
            "pass": False,
            "active_params": [str(name) for name in active_names],
            "var_names": [str(name) for name in active_names],
            "candidate_param_names": [str(name) for name in active_names],
            "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
            "elapsed_s": float(max(0.0, _perf_counter() - started)),
            "error_text": str(exc),
            "feature": feature,
            "least_squares_called": bool(least_squares_called),
            "optimizer_solve_called": bool(optimizer_solve_called),
            "real_solve_called": bool(optimizer_solve_called),
            "effective_var_names_seen_by_solver": list(
                effective_var_names_seen_by_solver
            ),
            "effective_candidate_param_names_seen_by_solver": list(
                effective_candidate_param_names_seen_by_solver
            ),
            "state_sha256_before": state_hash_before,
            "state_sha256_after": state_hash_after,
            "state_hash_unchanged": state_hash_before == state_hash_after,
            "heartbeat_count": int(heartbeat_count),
            "residual_eval_trace": list(residual_eval_trace),
            "fixed_source_counters_dirty_seen": bool(fixed_source_counters_dirty_seen),
            "fixed_source_counter_failures_seen": list(
                fixed_source_counter_failures_seen
            ),
            "dirty_timeout_abort": False,
            "child_process_killed_cleanly": None,
            "phase_timing_s": dict(phase_timing_s),
            "first_residual_elapsed_s": phase_timing_s.get(
                "first_residual_elapsed_s"
            ),
            "solver_context_reused": bool(context_reused),
            "diagnostic_logging": bool(diagnostic_logging),
        }
        report.update(request_summary)
        _apply_single_param_fields(report)
        _apply_one_param_diagnostic_aliases(report)
    report["phase_timing_s"] = dict(phase_timing_s)
    report["first_residual_elapsed_s"] = phase_timing_s.get("first_residual_elapsed_s")
    report["solver_context_reused"] = bool(context_reused)
    report["diagnostic_logging"] = bool(diagnostic_logging)
    phase_started = _perf_counter()
    _write_json(output_path, report)
    _record_phase("report_write_s", phase_started)
    report["phase_timing_s"] = dict(phase_timing_s)
    _write_json(output_path, report)
    _heartbeat_write(heartbeat_path, {"status": str(report.get("status", "")), "done": True})
    return report


def _solver_worker_command(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    heartbeat_path: Path,
    max_nfev: int,
    rung: int,
    rung_name: str,
    feature: str | None = None,
    diagnostic_logging: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-solve",
        "--state",
        str(state_path),
        "--background-index",
        str(int(background_index)),
        "--output-json",
        str(output_path),
        "--heartbeat-json",
        str(heartbeat_path),
        "--active-params-json",
        json.dumps([str(name) for name in active_names]),
        "--max-nfev",
        str(int(max_nfev)),
        "--rung-number",
        str(int(rung)),
        "--rung-name",
        str(rung_name),
    ]
    if feature:
        cmd.extend(["--feature", str(feature)])
    if bool(diagnostic_logging):
        cmd.append("--diagnostic-logging")
    return cmd


def _timeout_report(
    *,
    rung: int,
    rung_name: str,
    active_names: Sequence[str],
    started_at: float,
    heartbeat_path: Path,
    timeout_seconds: float,
    state_path: Path | None = None,
    state_hash_before: str | None = None,
    dirty_timeout_abort: bool = False,
    child_process_killed_cleanly: bool | None = None,
    feature: str | None = None,
) -> dict[str, object]:
    heartbeat = _read_json(heartbeat_path)
    hash_before = str(state_hash_before or heartbeat.get("state_sha256_before") or "")
    hash_after = _state_sha256(Path(state_path)) if state_path is not None else ""
    point_summary = (
        dict(heartbeat.get("last_point_match_summary", {}))
        if isinstance(heartbeat.get("last_point_match_summary", {}), Mapping)
        else {}
    )
    trace = _as_mapping_list(heartbeat.get("residual_eval_trace"))
    last_record = (
        dict(heartbeat.get("last_residual_eval"))
        if isinstance(heartbeat.get("last_residual_eval"), Mapping)
        else (dict(trace[-1]) if trace else {})
    )

    def _partial_value(key: str, *aliases: str) -> object:
        for candidate in (key, *aliases):
            if candidate in last_record and last_record.get(candidate) is not None:
                return last_record.get(candidate)
            if candidate in heartbeat and heartbeat.get(candidate) is not None:
                return heartbeat.get(candidate)
            if candidate in point_summary and point_summary.get(candidate) is not None:
                return point_summary.get(candidate)
        return None

    after_rms = _partial_value("after_rms_px", "rms_px", "current_rms_px")
    before_rms = _partial_value("before_rms_px")
    before_max = _partial_value("before_max_error_px")
    after_max = _partial_value("after_max_error_px", "max_error_px", "current_max_error_px")
    last_residual_norm = _residual_norm_from_value(
        _partial_value("last_residual_norm", "residual_norm")
    )
    if not math.isfinite(last_residual_norm):
        last_residual_norm = _residual_norm_from_cost(_partial_value("last_cost", "cost"))
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "timeout",
        "pass": False,
        "failure_reason": "timeout",
        "active_params": [str(name) for name in active_names],
        "var_names": [str(name) for name in active_names],
        "candidate_param_names": [str(name) for name in active_names],
        "before_rms_px": before_rms,
        "after_rms_px": after_rms,
        "before_max_error_px": before_max,
        "after_max_error_px": after_max,
        "parameter_before": _partial_value("parameter_before"),
        "parameter_after": _partial_value("parameter_after"),
        "parameter_delta": _partial_value("parameter_delta"),
        "parameter_bounds": _partial_value("parameter_bounds"),
        "residuals_finite": bool(
            math.isfinite(_metric_float(before_rms))
            and math.isfinite(_metric_float(after_rms))
        ),
        "elapsed_seconds": float(max(0.0, _perf_counter() - started_at)),
        "elapsed_s": float(max(0.0, _perf_counter() - started_at)),
        "timeout_seconds": float(timeout_seconds),
        "timeout_s": float(timeout_seconds),
        "last_heartbeat_elapsed_s": _partial_value("last_heartbeat_elapsed_s", "elapsed_s"),
        "heartbeat_count": _safe_int(heartbeat.get("heartbeat_count", len(trace)), default=0),
        "last_emitted_optimizer_status": heartbeat.get("last_optimizer_status"),
        "nfev": _partial_value("nfev", "last_nfev", "eval_count"),
        "last_nfev": _partial_value("last_nfev", "nfev", "eval_count"),
        "last_residual_norm": last_residual_norm,
        "last_cost": _partial_value("last_cost", "cost"),
        "last_rms_px": after_rms,
        "last_max_error_px": after_max,
        "last_parameter_value": _partial_value("last_parameter_value", "parameter_value"),
        "current_bounds": _partial_value("current_bounds", "bounds"),
        "current_rms_px": heartbeat.get("current_rms_px"),
        "last_point_match_summary": point_summary if point_summary else None,
        "point_match_summary": point_summary,
        "last_residual_eval": last_record if last_record else None,
        "residual_eval_trace": trace,
        "least_squares_called": bool(heartbeat.get("least_squares_called", False)),
        "optimizer_solve_called": bool(heartbeat.get("optimizer_solve_called", False)),
        "real_solve_called": bool(
            heartbeat.get("real_solve_called", heartbeat.get("optimizer_solve_called", False))
        ),
        "effective_var_names_seen_by_solver": _as_str_list(
            heartbeat.get("effective_var_names_seen_by_solver")
        ),
        "effective_candidate_param_names_seen_by_solver": _as_str_list(
            heartbeat.get("effective_candidate_param_names_seen_by_solver")
        ),
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "child_process_killed_cleanly": child_process_killed_cleanly,
        "fixed_source_counters_clean_at_last_heartbeat": bool(
            heartbeat.get("fixed_source_counters_clean_at_last_heartbeat", False)
        ),
        "fixed_source_counter_failures_at_last_heartbeat": _as_str_list(
            heartbeat.get("fixed_source_counter_failures_at_last_heartbeat")
        ),
        "fixed_source_counters_dirty_seen": bool(
            heartbeat.get("fixed_source_counters_dirty_seen", False)
        ),
        "fixed_source_counter_failures_seen": _as_str_list(
            heartbeat.get("fixed_source_counter_failures_seen")
        ),
        "state_sha256_before": hash_before or None,
        "state_sha256_after": hash_after or None,
        "state_hash_unchanged": bool(hash_before and hash_after and hash_before == hash_after),
        "feature": feature,
        "phase_timing_s": (
            dict(heartbeat.get("phase_timing_s", {}))
            if isinstance(heartbeat.get("phase_timing_s", {}), Mapping)
            else {}
        ),
        "first_residual_elapsed_s": heartbeat.get("first_residual_elapsed_s"),
        "solver_context_reused": bool(heartbeat.get("solver_context_reused", False)),
        "diagnostic_logging": bool(heartbeat.get("diagnostic_logging", False)),
    }
    counter_aliases = {
        "fixed_source_pair_count": (
            "fixed_source_pair_count",
            "matched_fixed_pair_count",
            "fixed_source_resolved_count",
        ),
    }
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        aliases = counter_aliases.get(key, (key,))
        report[key] = _partial_value(key, *tuple(alias for alias in aliases if alias != key))
    for key in PROVIDER_MATCH_BOOLS:
        report[key] = _partial_value(key)
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    report.setdefault("parameter_bounds", None)
    return report


def _run_in_process_worker_with_timeout(
    *,
    worker: Callable[[], dict[str, object] | None],
    output_path: Path,
    timeout_report_factory: Callable[[], dict[str, object]],
    timeout_seconds: float,
) -> dict[str, object]:
    result_box: dict[str, object] = {}

    def _target() -> None:
        try:
            result = worker()
            if isinstance(result, Mapping):
                result_box.update(dict(result))
                if not output_path.exists():
                    _write_json(output_path, dict(result))
        except Exception as exc:
            result_box.update({"status": "error", "pass": False, "error_text": str(exc)})

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_seconds))
    if thread.is_alive():
        report = timeout_report_factory()
        _write_json(output_path, report)
        return report
    if output_path.exists():
        return _read_json(output_path)
    report = dict(result_box) if result_box else {"status": "error", "pass": False}
    _write_json(output_path, report)
    return report


def _run_solver_rung_with_timeout(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    max_nfev: int,
    timeout_seconds: float,
    rung: int,
    rung_name: str,
    feature: str | None = None,
    state_hash_before: str | None = None,
    use_subprocess: bool = True,
    context: Mapping[str, object] | None = None,
    diagnostic_logging: bool = False,
    dirty_timeout_on_timeout: bool = False,
) -> dict[str, object]:
    started = _perf_counter()
    heartbeat_path = output_path.with_suffix(".heartbeat.json")
    initial_state_hash = str(state_hash_before or _state_sha256(Path(state_path)))

    def _make_timeout_report(
        *,
        dirty_timeout_abort: bool = False,
        child_process_killed_cleanly: bool | None = None,
    ) -> dict[str, object]:
        return _timeout_report(
            rung=int(rung),
            rung_name=str(rung_name),
            active_names=active_names,
            started_at=started,
            heartbeat_path=heartbeat_path,
            timeout_seconds=float(timeout_seconds),
            state_path=Path(state_path),
            state_hash_before=initial_state_hash,
            dirty_timeout_abort=bool(dirty_timeout_abort),
            child_process_killed_cleanly=child_process_killed_cleanly,
            feature=feature,
        )

    if not use_subprocess:
        timeout_report_factory = (
            lambda: _make_timeout_report(
                dirty_timeout_abort=bool(dirty_timeout_on_timeout),
                child_process_killed_cleanly=(
                    False if bool(dirty_timeout_on_timeout) else None
                ),
            )
        )
        return _run_in_process_worker_with_timeout(
            worker=lambda: _worker_solve_once(
                state_path=state_path,
                background_index=int(background_index),
                active_names=active_names,
                output_path=output_path,
                heartbeat_path=heartbeat_path,
                max_nfev=int(max_nfev),
                rung=int(rung),
                rung_name=str(rung_name),
                feature=feature,
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
            ),
            output_path=output_path,
            timeout_report_factory=timeout_report_factory,
            timeout_seconds=float(timeout_seconds),
        )

    cmd = _solver_worker_command(
        state_path=state_path,
        background_index=int(background_index),
        active_names=active_names,
        output_path=output_path,
        heartbeat_path=heartbeat_path,
        max_nfev=int(max_nfev),
        rung=int(rung),
        rung_name=str(rung_name),
        feature=feature,
        diagnostic_logging=bool(diagnostic_logging),
    )
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not bool(diagnostic_logging):
        env["RA_SIM_DISABLE_LOGGING"] = "1"
        env["RA_SIM_DISABLE_ALL_LOGGING"] = "1"
        env["RA_SIM_LOG_INTERSECTION_CACHE"] = "0"
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_text, stderr_text = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        dirty_timeout_abort = False
        child_process_killed_cleanly = True
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            dirty_timeout_abort = True
            child_process_killed_cleanly = False
            pass
        if process.poll() is None:
            dirty_timeout_abort = True
            child_process_killed_cleanly = False
        report = _make_timeout_report(
            dirty_timeout_abort=dirty_timeout_abort,
            child_process_killed_cleanly=child_process_killed_cleanly,
        )
        _write_json(output_path, report)
        return report

    if output_path.exists():
        report = _read_json(output_path)
        if stdout_text.strip():
            report["worker_stdout_tail"] = stdout_text.strip()[-2000:]
        if stderr_text.strip():
            report["worker_stderr_tail"] = stderr_text.strip()[-2000:]
        _write_json(output_path, report)
        return report
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "error",
        "pass": False,
        "active_params": [str(name) for name in active_names],
        "candidate_param_names": [str(name) for name in active_names],
        "elapsed_seconds": float(max(0.0, _perf_counter() - started)),
        "returncode": process.returncode,
        "worker_stdout_tail": stdout_text.strip()[-2000:],
        "worker_stderr_tail": stderr_text.strip()[-2000:],
        "feature": feature,
    }
    _write_json(output_path, report)
    return report


def _finalize_one_param_report(
    report: Mapping[str, object],
    *,
    param_name: str,
    state_path: Path,
    state_hash_before: str,
    timeout_seconds: float,
) -> dict[str, object]:
    finalized = dict(report)
    finalized.setdefault("rung", 3)
    finalized.setdefault("rung_name", f"one_param_{param_name}")
    finalized["param_name"] = str(param_name)
    finalized["active_params"] = [str(param_name)]
    finalized["var_names"] = [str(param_name)]
    finalized.setdefault("candidate_param_names", [str(param_name)])
    finalized["elapsed_s"] = _metric_float(
        finalized.get("elapsed_s", finalized.get("elapsed_seconds", 0.0))
    )
    finalized.setdefault("elapsed_seconds", finalized["elapsed_s"])
    finalized["timeout_s"] = float(timeout_seconds)
    finalized.setdefault("timeout_seconds", float(timeout_seconds))
    finalized["state_sha256_before"] = str(
        finalized.get("state_sha256_before") or state_hash_before
    )
    finalized["state_sha256_after"] = str(
        finalized.get("state_sha256_after") or _state_sha256(Path(state_path))
    )
    finalized["state_hash_unchanged"] = (
        finalized["state_sha256_before"] == finalized["state_sha256_after"]
    )
    finalized["state_hash_before"] = finalized["state_sha256_before"]
    finalized["state_hash_after"] = finalized["state_sha256_after"]
    _apply_single_param_fields(finalized)
    _apply_one_param_diagnostic_aliases(finalized)
    for key in (
        "before_rms_px",
        "after_rms_px",
        "before_max_error_px",
        "after_max_error_px",
        "parameter_before",
        "parameter_after",
        "parameter_delta",
    ):
        finalized.setdefault(key, None)
    finalized.setdefault("parameter_bounds", None)
    finalized.setdefault("point_match_summary", {})
    finalized.setdefault("last_nfev", finalized.get("nfev"))
    finalized.setdefault("last_residual_norm", finalized.get("residual_norm"))
    finalized.setdefault("last_rms_px", finalized.get("after_rms_px"))
    finalized.setdefault("last_max_error_px", finalized.get("after_max_error_px"))
    finalized.setdefault("last_parameter_value", finalized.get("parameter_after"))
    finalized.setdefault("current_bounds", finalized.get("parameter_bounds"))
    finalized.setdefault(
        "last_point_match_summary",
        finalized.get("point_match_summary") if finalized.get("point_match_summary") else None,
    )
    finalized.setdefault("last_heartbeat_elapsed_s", None)
    finalized.setdefault("heartbeat_count", 0)
    finalized.setdefault("child_process_killed_cleanly", None)
    finalized.setdefault("dirty_timeout_abort", False)
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        finalized.setdefault(key, None)
    for key in PROVIDER_MATCH_BOOLS:
        finalized.setdefault(key, None)
    finalized["residuals_finite"] = bool(
        math.isfinite(_metric_float(finalized.get("before_rms_px", np.nan)))
        and math.isfinite(_metric_float(finalized.get("after_rms_px", np.nan)))
    )
    _apply_one_param_diagnostic_aliases(finalized)

    if str(finalized.get("status", "")) == "timeout":
        finalized["pass"] = False
        classification = str(finalized.get("diagnosis_classification") or "")
        if classification == "fixed_source_or_pair_integrity_lost":
            finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
        elif bool(finalized.get("dirty_timeout_abort", False)):
            finalized["failure_reason"] = "dirty_timeout_abort"
        elif not finalized.get("failure_reason"):
            finalized["failure_reason"] = "timeout"
        return finalized

    integrity_failures = _one_param_integrity_failures(finalized)
    metric_failures = _one_param_metric_failures(finalized)
    solve_flag_failures: list[str] = []
    if bool(finalized.get("least_squares_called", False)) is not True:
        solve_flag_failures.append("least_squares_not_called")
    if not (
        bool(finalized.get("optimizer_solve_called", False))
        or bool(finalized.get("real_solve_called", False))
    ):
        solve_flag_failures.append("optimizer_solve_not_called")
    if finalized.get("candidate_param_names") != [str(param_name)]:
        solve_flag_failures.append("candidate_param_names_not_singleton")
    if finalized.get("var_names") != [str(param_name)]:
        solve_flag_failures.append("var_names_not_singleton")
    if bool(finalized.get("state_hash_unchanged", False)) is not True:
        solve_flag_failures.append("state_hash_changed")

    guard_failures = integrity_failures + metric_failures + solve_flag_failures
    finalized["one_param_guard_failures"] = guard_failures
    finalized["diagnosis_classification"] = _diagnosis_classification(finalized)
    if not guard_failures and str(finalized.get("status", "")) in {"ok", "pass"}:
        finalized["status"] = "ok"
        finalized["pass"] = True
        finalized["failure_reason"] = None
        finalized["diagnosis_classification"] = "usable"
        return finalized

    finalized["status"] = "failed"
    finalized["pass"] = False
    if integrity_failures:
        finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
    elif any(reason.startswith("non_finite") for reason in metric_failures):
        finalized["failure_reason"] = "non_finite_residual"
    elif "no_matched_peak_rejection" in metric_failures:
        finalized["failure_reason"] = "no_matched_peak_rejection"
    elif metric_failures:
        finalized["failure_reason"] = "metric_guard_failed"
    elif solve_flag_failures:
        finalized["failure_reason"] = "solve_flag_guard_failed"
    else:
        finalized.setdefault("failure_reason", "one_param_solve_failed")
    return finalized


def _active_params_from_sensitivity(
    sensitivity: Mapping[str, object],
    context: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    active = _as_str_list(sensitivity.get("active_params"))
    excluded = (
        set(_as_str_list(sensitivity.get("near_zero_params")))
        | set(_as_str_list(sensitivity.get("non_finite_params")))
        | set(_as_str_list(sensitivity.get("unsafe_params")))
    )
    skipped = [name for name in active if name in excluded]
    filtered = [name for name in active if name not in excluded]
    return _coerce_active_names(context, filtered), skipped


def _rung2_green_failures(sensitivity: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if str(sensitivity.get("status", "")) != "ok" or bool(sensitivity.get("pass", False)) is not True:
        failures.append("sensitivity_status_not_ok")
    if bool(sensitivity.get("least_squares_called", False)):
        failures.append("least_squares_called")
    if bool(sensitivity.get("optimizer_solve_called", False)):
        failures.append("optimizer_solve_called")
    if "state_hash_unchanged" in sensitivity and bool(
        sensitivity.get("state_hash_unchanged", False)
    ) is not True:
        failures.append("state_hash_changed")
    failures.extend(
        _fixed_source_contract_failures(
            sensitivity,
            expected_counts=RUNG2_FIXED_SOURCE_COUNTS,
            required_bool_keys=PROVIDER_MATCH_BOOLS,
        )
    )

    params_by_name = {
        str(entry.get("param_name", entry.get("name", ""))): entry
        for entry in sensitivity.get("params", []) or []
        if isinstance(entry, Mapping)
    }
    for name in _as_str_list(sensitivity.get("active_params")):
        entry = params_by_name.get(name)
        if entry is None:
            failures.append(f"{name}_missing_sensitivity_entry")
            continue
        failures.extend(
            _fixed_source_contract_failures(
                entry,
                expected_counts=RUNG2_FIXED_SOURCE_COUNTS,
                required_bool_keys=PROVIDER_MATCH_BOOLS,
                prefix=f"{name}_",
            )
        )
    return failures


def _best_single_param(
    reports: Sequence[Mapping[str, object]],
    metric_key: str,
) -> dict[str, object] | None:
    best_report: Mapping[str, object] | None = None
    best_value = float("inf")
    for report in reports:
        if bool(report.get("pass", False)) is not True:
            continue
        value = _metric_float(report.get(metric_key, np.nan))
        if math.isfinite(value) and value < best_value:
            best_value = value
            best_report = report
    if best_report is None:
        return None
    return {
        "param_name": str(best_report.get("param_name", "")),
        metric_key: best_value,
    }


def _one_param_summary(
    *,
    run_dir: Path,
    sensitivity_report_path: Path,
    state_path: Path | None = None,
    background_index: int | None = None,
    sensitivity: Mapping[str, object],
    active_params: Sequence[str],
    attempted_reports: Sequence[Mapping[str, object]],
    skipped_params: Sequence[str],
    state_hash_before: str,
    state_hash_after: str,
    provider_after: Mapping[str, object] | None,
    dirty_timeout_abort: bool = False,
    failure_reason: str | None = None,
    reports: Sequence[Mapping[str, object]] | None = None,
    filtered_params: Sequence[str] | None = None,
) -> dict[str, object]:
    attempted_params = [str(report.get("param_name", "")) for report in attempted_reports]
    passed_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if bool(report.get("pass", False))
    ]
    passing_reports = [
        report for report in attempted_reports if bool(report.get("pass", False))
    ]
    timed_out_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if str(report.get("status", "")) == "timeout"
    ]
    failed_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if str(report.get("status", "")) == "failed"
    ]
    diagnosis_by_param = {
        str(report.get("param_name", "")): str(report.get("diagnosis_classification", ""))
        for report in attempted_reports
        if report.get("diagnosis_classification")
    }
    any_pair_loss = any(
        str(report.get("failure_reason", "")) == "fixed_source_or_pair_integrity_lost"
        or str(report.get("diagnosis_classification", ""))
        == "fixed_source_or_pair_integrity_lost"
        or (
            str(report.get("status", "")) != "timeout"
            and bool(_one_param_integrity_failures(report))
        )
        for report in attempted_reports
    )
    passing_pair_loss = any(
        str(report.get("failure_reason", "")) == "fixed_source_or_pair_integrity_lost"
        or str(report.get("diagnosis_classification", ""))
        == "fixed_source_or_pair_integrity_lost"
        or bool(_one_param_integrity_failures(report))
        for report in passing_reports
    )
    any_timeout = bool(timed_out_params)
    provider_guard_after_ok = (
        bool(provider_after.get("provider_guard_ok", False))
        and str(provider_after.get("classification", "")) == "point_provider_parity_ok"
        if isinstance(provider_after, Mapping)
        else False
    )
    state_hash_unchanged = state_hash_before == state_hash_after
    all_fixed_source_counters_clean = (
        bool(attempted_reports)
        and not any_pair_loss
        and all(
            bool(report.get("fixed_source_counters_clean_at_last_heartbeat", True))
            for report in attempted_reports
        )
    )
    all_passing_fixed_source_counters_clean = bool(passing_reports) and not any(
        _one_param_integrity_failures(report) for report in passing_reports
    )

    if failure_reason:
        status = "failed"
    elif dirty_timeout_abort:
        status = "failed"
        failure_reason = "dirty_timeout_abort"
    elif not passed_params:
        status = "failed"
        failure_reason = "no_one_param_solve_passed"
    elif passing_pair_loss:
        status = "failed"
        failure_reason = "fixed_source_or_pair_integrity_lost"
    elif (
        failed_params
        or timed_out_params
        or list(skipped_params)
        or provider_guard_after_ok is not True
        or state_hash_unchanged is not True
    ):
        status = "ok_with_failures" if provider_guard_after_ok and state_hash_unchanged else "failed"
        if status == "failed" and not failure_reason:
            failure_reason = "provider_guard_after_failed" if not provider_guard_after_ok else "state_hash_changed"
    else:
        status = "ok"

    summary = {
        "rung": 3,
        "rung_name": "one_param_summary",
        "status": status,
        "failure_reason": failure_reason,
        "run_dir": str(run_dir),
        "state_path": str(Path(state_path).expanduser().resolve()) if state_path else None,
        "background_index": int(background_index) if background_index is not None else None,
        "sensitivity_report_path": str(sensitivity_report_path),
        "active_params_from_sensitivity": list(active_params),
        "attempted_params": attempted_params,
        "passed_params": passed_params,
        "failed_params": failed_params,
        "timed_out_params": timed_out_params,
        "skipped_params": [str(name) for name in skipped_params],
        "filtered_params": [str(name) for name in (filtered_params or [])],
        "diagnosis_by_param": diagnosis_by_param,
        "diagnosis_classification": (
            next(iter(diagnosis_by_param.values()))
            if len(diagnosis_by_param) == 1
            else None
        ),
        "best_single_param_by_rms": _best_single_param(attempted_reports, "after_rms_px"),
        "best_single_param_by_max_error": _best_single_param(
            attempted_reports,
            "after_max_error_px",
        ),
        "all_fixed_source_counters_clean": bool(all_fixed_source_counters_clean),
        "all_passing_fixed_source_counters_clean": bool(
            all_passing_fixed_source_counters_clean
        ),
        "any_timeout": bool(any_timeout),
        "any_pair_loss": bool(any_pair_loss),
        "any_branch_mismatch": any(
            _safe_int(report.get("branch_mismatch_count", 0), default=0) != 0
            for report in attempted_reports
        ),
        "any_no_matched_peak_rejection": any(
            NO_MATCH_REJECTION in str(report.get("rejection_reason", ""))
            for report in attempted_reports
        ),
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "state_sha256_before": state_hash_before,
        "state_sha256_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_unchanged),
        "provider_guard_after_ok": bool(provider_guard_after_ok),
        "provider_guard_after": dict(provider_after or {}),
        "rung_0_green": True,
        "rung_1_green": True,
        "rung_2_green": not _rung2_green_failures(sensitivity),
        "reports": list(reports or []),
    }
    return summary


def _passed_params_from_one_param_reports(reports: Sequence[Mapping[str, object]]) -> list[str]:
    passed: list[str] = []
    for report in reports:
        if bool(report.get("pass", False)):
            names = [str(name) for name in report.get("active_params", []) or []]
            if len(names) == 1 and names[0] not in passed:
                passed.append(names[0])
    return passed


def _run_one_param_stage(
    *,
    state_path: Path,
    background_index: int,
    run_dir: Path,
    context: Mapping[str, object],
    sensitivity: Mapping[str, object],
    sensitivity_report_path: Path,
    reports: list[dict[str, object]],
    state_hash_before: str,
    max_nfev: int,
    timeout_seconds: float,
    one_param_filter: str | None = None,
    use_subprocess: bool = False,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    raw_active_params = _as_str_list(sensitivity.get("active_params"))
    filter_name = str(one_param_filter or "").strip()
    rung2_failures = _rung2_green_failures(sensitivity)
    if rung2_failures:
        state_hash_after = _state_sha256(state_path)
        summary = _one_param_summary(
            run_dir=run_dir,
            sensitivity_report_path=sensitivity_report_path,
            state_path=state_path,
            background_index=int(background_index),
            sensitivity=sensitivity,
            active_params=raw_active_params,
            attempted_reports=[],
            skipped_params=[],
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            provider_after=None,
            failure_reason="sensitivity_not_green",
            reports=reports,
        )
        summary["rung_2_failures"] = rung2_failures
        _write_json(run_dir / "rung_03_one_param_summary.json", summary)
        return summary

    active_params, skipped_params = _active_params_from_sensitivity(sensitivity, context)
    filtered_params: list[str] = []
    if filter_name:
        filtered_params = [name for name in raw_active_params if name != filter_name]
        if filter_name not in raw_active_params or filter_name not in active_params:
            state_hash_after = _state_sha256(state_path)
            summary = _one_param_summary(
                run_dir=run_dir,
                sensitivity_report_path=sensitivity_report_path,
                state_path=state_path,
                background_index=int(background_index),
                sensitivity=sensitivity,
                active_params=raw_active_params,
                attempted_reports=[],
                skipped_params=skipped_params,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                provider_after=None,
                failure_reason="filtered_param_not_active",
                reports=reports,
                filtered_params=filtered_params,
            )
            summary["one_param_filter"] = filter_name
            _write_json(run_dir / "rung_03_one_param_summary.json", summary)
            return summary
        active_params = [filter_name]
        skipped_params = [name for name in skipped_params if name != filter_name]
    if not active_params:
        state_hash_after = _state_sha256(state_path)
        summary = _one_param_summary(
            run_dir=run_dir,
            sensitivity_report_path=sensitivity_report_path,
            state_path=state_path,
            background_index=int(background_index),
            sensitivity=sensitivity,
            active_params=raw_active_params,
            attempted_reports=[],
            skipped_params=skipped_params,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            provider_after=None,
            failure_reason="no_active_params",
            reports=reports,
            filtered_params=filtered_params,
        )
        _write_json(run_dir / "rung_03_one_param_summary.json", summary)
        return summary

    one_param_reports: list[dict[str, object]] = []
    dirty_timeout_abort = False
    for index, name in enumerate(active_params):
        output_path = _rung_path(run_dir, 3, f"one_param_{name}")
        with _timed_report_window("3", f"one_param_{name}"):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=[name],
                output_path=output_path,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=3,
                rung_name=f"one_param_{name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
            report = _finalize_one_param_report(
                report,
                param_name=name,
                state_path=state_path,
                state_hash_before=state_hash_before,
                timeout_seconds=float(timeout_seconds),
            )
            _write_json(output_path, report)
        reports.append(report)
        one_param_reports.append(report)
        if bool(report.get("dirty_timeout_abort", False)):
            dirty_timeout_abort = True
            skipped_params.extend(active_params[index + 1 :])
            break

    provider_after: dict[str, object] | None = None
    if not dirty_timeout_abort:
        with _solver_debug_logging_scope(diagnostic_logging):
            provider_after = run_provider_guard(
                state_path=state_path,
                background_index=int(background_index),
                output_path=_rung_path(run_dir, 3, "provider_guard_after"),
            )
        reports.append(provider_after)

    state_hash_after = _state_sha256(state_path)
    summary = _one_param_summary(
        run_dir=run_dir,
        sensitivity_report_path=sensitivity_report_path,
        state_path=state_path,
        background_index=int(background_index),
        sensitivity=sensitivity,
        active_params=raw_active_params,
        attempted_reports=one_param_reports,
        skipped_params=skipped_params,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        provider_after=provider_after,
        dirty_timeout_abort=dirty_timeout_abort,
        reports=reports,
        filtered_params=filtered_params,
    )
    if filter_name:
        summary["one_param_filter"] = filter_name
    _write_json(run_dir / "rung_03_one_param_summary.json", summary)
    return summary


def _groups_for_pair_rungs(passed_params: Sequence[str], theta_name: str) -> list[tuple[str, list[str]]]:
    passed = set(str(name) for name in passed_params)
    candidates = [
        ("center_xy", ["center_x", "center_y"]),
        ("gamma_Gamma", ["gamma", "Gamma"]),
        ("chi_cor_angle", ["chi", "cor_angle"]),
        ("theta_cor_angle", [theta_name, "cor_angle"]),
        ("corto_detector", ["corto_detector"]),
        ("zs_zb", ["zs", "zb"]),
        ("a_c", ["a", "c"]),
        ("a_c_psi_z", ["a", "c", "psi_z"]),
    ]
    return [(name, params) for name, params in candidates if all(param in passed for param in params)]


def _state_hash_evidence_failures(
    report: Mapping[str, object],
    *,
    current_hash: str,
    before_key: str = "state_sha256_before",
    after_key: str = "state_sha256_after",
    unchanged_key: str = "state_hash_unchanged",
) -> list[str]:
    failures: list[str] = []
    before = str(report.get(before_key) or "")
    after = str(report.get(after_key) or "")
    if before != str(current_hash):
        failures.append(f"{before_key}_not_current")
    if after != str(current_hash):
        failures.append(f"{after_key}_not_current")
    if bool(report.get(unchanged_key, False)) is not True:
        failures.append(f"{unchanged_key}_not_true")
    return failures


def _run_input_evidence_failures(
    report: Mapping[str, object],
    *,
    state_path: Path,
    background_index: int,
) -> list[str]:
    failures: list[str] = []
    if not _same_resolved_path(report.get("state_path"), state_path):
        failures.append("state_path_mismatch")
    if _safe_int(report.get("background_index"), default=-999999) != int(background_index):
        failures.append("background_index_mismatch")
    return failures


def _load_required_json(path: Path | None, missing_reason: str) -> tuple[dict[str, object], list[str]]:
    if path is None:
        return {}, [missing_reason]
    report = _read_json(Path(path).expanduser().resolve())
    if not report:
        return {}, [f"{missing_reason}_unreadable"]
    return report, []


def _validate_one_param_summary_evidence(
    summary_path: Path | None,
    *,
    state_path: Path,
    background_index: int,
    current_hash: str,
) -> tuple[dict[str, object], list[str]]:
    summary, failures = _load_required_json(summary_path, "missing_one_param_summary")
    if failures:
        return summary, failures
    failures.extend(
        _state_hash_evidence_failures(summary, current_hash=current_hash)
    )
    failures.extend(
        _run_input_evidence_failures(
            summary,
            state_path=state_path,
            background_index=int(background_index),
        )
    )
    if bool(summary.get("provider_guard_after_ok", False)) is not True:
        failures.append("provider_guard_after_not_green")
    if bool(summary.get("dirty_timeout_abort", False)):
        failures.append("dirty_timeout_abort")
    if not _as_str_list(summary.get("passed_params")):
        failures.append("no_passed_singleton_params")
    return summary, failures


def _validate_one_param_diagnosis_evidence(
    diagnosis_path: Path | None,
    *,
    state_path: Path,
    background_index: int,
    current_hash: str,
    current_active_params: Sequence[str],
) -> tuple[dict[str, object], list[str], list[str], bool]:
    if diagnosis_path is None:
        return {}, [], [], False
    summary = _read_json(Path(diagnosis_path).expanduser().resolve())
    if not summary:
        return {}, ["one_param_diagnosis_summary_unreadable"], [], False
    fatal_failures: list[str] = []
    local_failures: list[str] = []
    final_report = _single_attempt_report(summary)
    hash_owner = summary if summary.get("state_sha256_before") else final_report
    fatal_failures.extend(
        _state_hash_evidence_failures(hash_owner, current_hash=current_hash)
    )
    path_owner = summary if summary.get("state_path") else final_report
    fatal_failures.extend(
        _run_input_evidence_failures(
            path_owner,
            state_path=state_path,
            background_index=int(background_index),
        )
    )
    if "a" not in set(str(name) for name in current_active_params):
        local_failures.append("a_not_active")
    if str(summary.get("status", "")) != "ok":
        local_failures.append("one_param_diagnosis_status_not_ok")
    diagnosis = str(
        summary.get("diagnosis_classification")
        or final_report.get("diagnosis_classification")
        or ""
    )
    if diagnosis != "usable":
        local_failures.append("a_diagnosis_not_usable")
    if bool(summary.get("dirty_timeout_abort", final_report.get("dirty_timeout_abort", False))):
        fatal_failures.append("dirty_timeout_abort")
    usable = not fatal_failures and not local_failures
    return summary, fatal_failures, local_failures, usable


def _caked_reprojection_guard_failures(
    report: Mapping[str, object],
    *,
    current_hash: str,
    background_index: int,
) -> list[str]:
    failures: list[str] = []
    expected = {
        "status": "pass",
        "point_count": EXPECTED_PROVIDER_PAIR_COUNT,
        "exact_projector_available": True,
        "theta_projector_signature_changed": True,
        "distance_projector_signature_changed": True,
        "full_background_recake_call_count": 0,
        "provider_guard_before_ok": True,
        "provider_guard_after_ok": True,
        "new4_state_hash_unchanged": True,
    }
    for key, expected_value in expected.items():
        actual = report.get(key)
        if key in {"point_count", "full_background_recake_call_count"}:
            if _safe_int(actual, default=-999999) != int(expected_value):
                failures.append(f"{key}_not_{expected_value}")
        elif actual != expected_value:
            failures.append(f"{key}_not_{expected_value}")
    if "background_index" not in report:
        failures.append("background_index_missing")
    elif _safe_int(report.get("background_index"), default=-999999) != int(background_index):
        failures.append("background_index_mismatch")
    before = str(report.get("state_hash_before") or report.get("state_sha256_before") or "")
    after = str(report.get("state_hash_after") or report.get("state_sha256_after") or "")
    if not before:
        failures.append("state_hash_before_missing")
    elif before != str(current_hash):
        failures.append("state_hash_before_not_current")
    if not after:
        failures.append("state_hash_after_missing")
    elif after != str(current_hash):
        failures.append("state_hash_after_not_current")
    return failures


def _pair_requires_caked_reprojection(pair: Sequence[str]) -> bool:
    return bool(set(str(name) for name in pair) & CAKED_REPROJECTION_REQUIRED_PARAMS)


def _base_parameter_values_for_pair(
    context: Mapping[str, object],
    pair: Sequence[str],
) -> dict[str, object]:
    prepared_run = context.get("prepared_run")
    params = getattr(prepared_run, "fit_params", {}) if prepared_run is not None else {}
    if not isinstance(params, Mapping):
        params = {}
    return {str(name): params.get(str(name)) for name in pair}


def _parameter_maps_from_deltas(
    report: Mapping[str, object],
    pair: Sequence[str],
    base_parameter_values: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    before = {str(name): base_parameter_values.get(str(name)) for name in pair}
    after: dict[str, object] = {str(name): None for name in pair}
    delta: dict[str, object] = {str(name): None for name in pair}
    bounds: dict[str, object] = {str(name): None for name in pair}
    for entry in _as_mapping_list(report.get("parameter_deltas")):
        name = str(entry.get("name", ""))
        if name not in before:
            continue
        before[name] = entry.get("start")
        after[name] = entry.get("final")
        delta[name] = entry.get("delta")
        bounds[name] = {
            "lower": entry.get("lower"),
            "upper": entry.get("upper"),
        }
    return before, after, delta, bounds


def _pair_metric_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    residual_norm = _metric_float(
        report.get("last_residual_norm", report.get("residual_norm", np.nan))
    )
    before_rms = _metric_float(report.get("before_rms_px", np.nan))
    after_rms = _metric_float(report.get("after_rms_px", np.nan))
    before_max = _metric_float(report.get("before_max_error_px", np.nan))
    after_max = _metric_float(report.get("after_max_error_px", np.nan))
    if (
        not bool(report.get("residuals_finite", False))
        or not math.isfinite(residual_norm)
        or not math.isfinite(before_rms)
        or not math.isfinite(after_rms)
    ):
        failures.append("non_finite_residual")
    if not math.isfinite(before_max) or not math.isfinite(after_max):
        failures.append("non_finite_max_error")
    if after_rms > before_rms + PAIR_RMS_TOLERANCE_PX:
        failures.append("rms_regressed")
    if after_max > before_max + PAIR_MAX_ERROR_TOLERANCE_PX:
        failures.append("max_error_regressed")
    rejection_text = json.dumps(
        _jsonable(
            {
                "rejection_reason": report.get("rejection_reason"),
                "rejection_reasons": report.get("rejection_reasons"),
            }
        ),
        sort_keys=True,
    )
    if NO_MATCH_REJECTION in rejection_text:
        failures.append("no_matched_peak_rejection")
    return failures


def _pair_timeout_integrity_dirty(report: Mapping[str, object]) -> bool:
    if bool(report.get("fixed_source_counters_dirty_seen", False)):
        return True
    failures = _as_str_list(report.get("fixed_source_counter_failures_at_last_heartbeat"))
    if failures and bool(report.get("fixed_source_counters_clean_at_last_heartbeat", True)) is not True:
        return True
    return False


def _finalize_pair_report(
    report: Mapping[str, object],
    *,
    pair_name: str,
    pair: Sequence[str],
    state_path: Path,
    state_hash_before: str,
    timeout_seconds: float,
    base_parameter_values: Mapping[str, object],
) -> dict[str, object]:
    names = [str(name) for name in pair]
    finalized = dict(report)
    finalized.setdefault("rung", 4)
    finalized.setdefault("rung_name", f"pair_{pair_name}")
    finalized["pair_name"] = str(pair_name)
    finalized["pair"] = list(names)
    finalized["active_params"] = list(names)
    finalized["var_names"] = _as_str_list(finalized.get("var_names")) or list(names)
    finalized["candidate_param_names"] = _as_str_list(
        finalized.get("candidate_param_names")
    )
    finalized["effective_var_names_seen_by_solver"] = _as_str_list(
        finalized.get("effective_var_names_seen_by_solver")
    )
    finalized.setdefault("elapsed_s", finalized.get("elapsed_seconds", 0.0))
    finalized.setdefault("elapsed_seconds", finalized.get("elapsed_s", 0.0))
    finalized["timeout_s"] = float(timeout_seconds)
    finalized.setdefault("timeout_seconds", float(timeout_seconds))
    finalized["state_sha256_before"] = str(
        finalized.get("state_sha256_before") or state_hash_before
    )
    finalized["state_sha256_after"] = str(
        finalized.get("state_sha256_after") or _state_sha256(Path(state_path))
    )
    finalized["state_hash_unchanged"] = (
        finalized["state_sha256_before"] == finalized["state_sha256_after"]
    )
    finalized["base_parameter_values"] = dict(base_parameter_values)
    before, after, delta, bounds = _parameter_maps_from_deltas(
        finalized,
        names,
        base_parameter_values,
    )
    finalized["parameter_before"] = before
    finalized["parameter_after"] = after
    finalized["parameter_delta"] = delta
    finalized["parameter_bounds"] = bounds
    finalized.setdefault("point_match_summary", {})
    finalized.setdefault(
        "last_point_match_summary",
        finalized.get("point_match_summary") if finalized.get("point_match_summary") else None,
    )
    finalized.setdefault("dirty_timeout_abort", False)
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        finalized.setdefault(key, None)
    for key in PROVIDER_MATCH_BOOLS:
        finalized.setdefault(key, None)

    if str(finalized.get("status", "")) == "timeout":
        finalized["pass"] = False
        if bool(finalized.get("dirty_timeout_abort", False)):
            finalized["failure_reason"] = "dirty_timeout_abort"
        elif _pair_timeout_integrity_dirty(finalized):
            finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
        else:
            finalized["failure_reason"] = "timeout"
        finalized["pair_guard_failures"] = [str(finalized["failure_reason"])]
        return finalized

    integrity_failures = _fixed_source_contract_failures(
        finalized,
        expected_counts=ONE_PARAM_FIXED_SOURCE_COUNTS,
        required_bool_keys=PROVIDER_MATCH_BOOLS,
    )
    metric_failures = _pair_metric_failures(finalized)
    solve_flag_failures: list[str] = []
    if bool(finalized.get("least_squares_called", False)) is not True:
        solve_flag_failures.append("least_squares_not_called")
    if not (
        bool(finalized.get("optimizer_solve_called", False))
        or bool(finalized.get("real_solve_called", False))
    ):
        solve_flag_failures.append("optimizer_solve_not_called")
    if finalized.get("candidate_param_names") != list(names):
        solve_flag_failures.append("candidate_param_names_not_pair")
    if finalized.get("var_names") != list(names):
        solve_flag_failures.append("var_names_not_pair")
    if finalized.get("effective_var_names_seen_by_solver") != list(names):
        solve_flag_failures.append("effective_var_names_seen_by_solver_not_pair")
    if bool(finalized.get("state_hash_unchanged", False)) is not True:
        solve_flag_failures.append("state_hash_changed")

    guard_failures = integrity_failures + metric_failures + solve_flag_failures
    finalized["pair_guard_failures"] = guard_failures
    if not guard_failures:
        finalized["status"] = "ok"
        finalized["pass"] = True
        finalized["failure_reason"] = None
        return finalized

    finalized["status"] = "failed"
    finalized["pass"] = False
    if integrity_failures:
        finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
    elif "no_matched_peak_rejection" in metric_failures:
        finalized["failure_reason"] = "no_matched_peak_rejection"
    elif any(reason.startswith("non_finite") for reason in metric_failures):
        finalized["failure_reason"] = "non_finite_residual"
    elif metric_failures:
        finalized["failure_reason"] = "metric_guard_failed"
    elif solve_flag_failures:
        finalized["failure_reason"] = "solve_flag_guard_failed"
    else:
        finalized.setdefault("failure_reason", "pair_solve_failed")
    return finalized


def _pair_ref(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "pair_name": str(report.get("pair_name", "")),
        "pair": [str(name) for name in report.get("pair", []) or []],
    }


def _best_pair(
    reports: Sequence[Mapping[str, object]],
    metric_key: str,
) -> dict[str, object] | None:
    best_report: Mapping[str, object] | None = None
    best_value = float("inf")
    for report in reports:
        if bool(report.get("pass", False)) is not True:
            continue
        value = _metric_float(report.get(metric_key, np.nan))
        if math.isfinite(value) and value < best_value:
            best_value = value
            best_report = report
    if best_report is None:
        return None
    payload = _pair_ref(best_report)
    payload[metric_key] = best_value
    return payload


def _pair_summary(
    *,
    run_dir: Path,
    state_path: Path,
    background_index: int,
    sensitivity: Mapping[str, object],
    allowed_params: Sequence[str],
    skipped_pairs: Sequence[Mapping[str, object]],
    pair_reports: Sequence[Mapping[str, object]],
    state_hash_before: str,
    state_hash_after: str,
    provider_after: Mapping[str, object] | None,
    evidence_failures: Sequence[str] = (),
    fatal_evidence_failures: Sequence[str] = (),
    local_usability_failures: Sequence[str] = (),
    disallowed_params: Sequence[str] = (),
    reports: Sequence[Mapping[str, object]] | None = None,
    provider_report_hash: str | None = None,
) -> dict[str, object]:
    attempted_pairs = [_pair_ref(report) for report in pair_reports]
    passed_reports = [report for report in pair_reports if bool(report.get("pass", False))]
    failed_reports = [
        report for report in pair_reports if str(report.get("status", "")) == "failed"
    ]
    timed_out_reports = [
        report for report in pair_reports if str(report.get("status", "")) == "timeout"
    ]
    improved_reports: list[Mapping[str, object]] = []
    neutral_reports: list[Mapping[str, object]] = []
    for report in passed_reports:
        before_rms = _metric_float(report.get("before_rms_px", np.nan))
        after_rms = _metric_float(report.get("after_rms_px", np.nan))
        before_max = _metric_float(report.get("before_max_error_px", np.nan))
        after_max = _metric_float(report.get("after_max_error_px", np.nan))
        improved = (
            math.isfinite(before_rms)
            and math.isfinite(after_rms)
            and after_rms < before_rms - PAIR_RMS_TOLERANCE_PX
        ) or (
            math.isfinite(before_max)
            and math.isfinite(after_max)
            and after_max < before_max - PAIR_MAX_ERROR_TOLERANCE_PX
        )
        if improved:
            improved_reports.append(report)
        else:
            neutral_reports.append(report)
    provider_guard_after_ok = (
        bool(provider_after.get("provider_guard_ok", False))
        and str(provider_after.get("classification", "")) == "point_provider_parity_ok"
        if isinstance(provider_after, Mapping)
        else False
    )
    state_hash_unchanged = state_hash_before == state_hash_after
    if evidence_failures:
        status = "failed"
        failure_reason = "pair_evidence_not_current"
    elif not passed_reports:
        status = "failed"
        failure_reason = "no_pair_solve_passed"
    elif provider_guard_after_ok is not True:
        status = "failed"
        failure_reason = "provider_guard_after_failed"
    elif state_hash_unchanged is not True:
        status = "failed"
        failure_reason = "state_hash_changed"
    elif failed_reports or timed_out_reports:
        status = "ok_with_failures"
        failure_reason = None
    else:
        status = "ok"
        failure_reason = None
    stable_pairs = [_pair_ref(report) for report in passed_reports]
    recommended_next_blocks = [
        {
            "block_name": f"{ref['pair_name']}_block",
            "params": ref["pair"],
            "source_pair": ref["pair_name"],
        }
        for ref in stable_pairs
    ]
    fatal_failures = [
        str(item) for item in (fatal_evidence_failures or evidence_failures)
    ]
    summary = {
        "rung": 4,
        "rung_name": "pair_summary",
        "status": status,
        "failure_reason": failure_reason,
        "run_dir": str(run_dir),
        "run_id": Path(run_dir).name,
        "timestamp": Path(run_dir).name,
        "state_path": str(Path(state_path).expanduser().resolve()),
        "background_index": int(background_index),
        "provider_report_hash": str(provider_report_hash or ""),
        "active_params_from_sensitivity": _as_str_list(sensitivity.get("active_params")),
        "allowed_params": [str(name) for name in allowed_params],
        "disallowed_params": [str(name) for name in disallowed_params],
        "attempted_pairs": attempted_pairs,
        "skipped_pairs": [dict(item) for item in skipped_pairs],
        "passed_pairs": [_pair_ref(report) for report in passed_reports],
        "stable_pairs": stable_pairs,
        "improved_pairs": [_pair_ref(report) for report in improved_reports],
        "neutral_pairs": [_pair_ref(report) for report in neutral_reports],
        "failed_pairs": [_pair_ref(report) for report in failed_reports],
        "timed_out_pairs": [_pair_ref(report) for report in timed_out_reports],
        "best_pair_by_rms": _best_pair(pair_reports, "after_rms_px"),
        "best_pair_by_max_error": _best_pair(pair_reports, "after_max_error_px"),
        "recommended_next_blocks": recommended_next_blocks,
        "any_pair_loss": any(
            str(report.get("failure_reason", "")) == "fixed_source_or_pair_integrity_lost"
            for report in pair_reports
        ),
        "any_branch_mismatch": any(
            _safe_int(report.get("branch_mismatch_count", 0), default=0) != 0
            for report in pair_reports
        ),
        "any_timeout": bool(timed_out_reports),
        "any_no_matched_peak_rejection": any(
            str(report.get("failure_reason", "")) == "no_matched_peak_rejection"
            for report in pair_reports
        ),
        "provider_guard_after_ok": bool(provider_guard_after_ok),
        "provider_guard_after": dict(provider_after or {}),
        "state_sha256_before": state_hash_before,
        "state_sha256_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_unchanged),
        "evidence_failures": [str(item) for item in evidence_failures],
        "fatal_evidence_failures": fatal_failures,
        "local_usability_failures": [
            str(item) for item in local_usability_failures
        ],
        "reports": list(reports or []),
    }
    return summary


def _run_pair_stage(
    *,
    state_path: Path,
    background_index: int,
    run_dir: Path,
    context: Mapping[str, object],
    sensitivity: Mapping[str, object],
    reports: list[dict[str, object]],
    state_hash_before: str,
    max_nfev: int,
    timeout_seconds: float,
    one_param_summary_path: Path | None = None,
    one_param_diagnosis_summary_path: Path | None = None,
    caked_point_reprojection_report_path: Path | None = None,
    include_psi_extension_pairs: bool = False,
    provider_report_hash: str | None = None,
    use_subprocess: bool = False,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    current_active_params, _skipped = _active_params_from_sensitivity(sensitivity, context)
    evidence_failures: list[str] = []
    one_param_summary, one_param_failures = _validate_one_param_summary_evidence(
        one_param_summary_path,
        state_path=state_path,
        background_index=int(background_index),
        current_hash=state_hash_before,
    )
    evidence_failures.extend(f"one_param:{failure}" for failure in one_param_failures)
    (
        diagnosis_summary,
        diagnosis_fatal_failures,
        diagnosis_local_failures,
        a_usable,
    ) = _validate_one_param_diagnosis_evidence(
        one_param_diagnosis_summary_path,
        state_path=state_path,
        background_index=int(background_index),
        current_hash=state_hash_before,
        current_active_params=current_active_params,
    )
    evidence_failures.extend(
        f"one_param_diagnosis:{failure}" for failure in diagnosis_fatal_failures
    )
    local_usability_failures = [
        f"one_param_diagnosis:{failure}" for failure in diagnosis_local_failures
    ]
    a_locally_unusable = bool(diagnosis_local_failures)
    singleton_passed_params = set(_as_str_list(one_param_summary.get("passed_params")))
    if a_locally_unusable:
        singleton_passed_params.discard("a")
    allowed_params = [
        name
        for name in current_active_params
        if name in singleton_passed_params
    ]
    if a_usable and "a" in current_active_params and "a" not in allowed_params:
        allowed_params.append("a")
    disallowed_params = [
        name for name in current_active_params if name not in set(allowed_params)
    ]

    if evidence_failures:
        state_hash_after = _state_sha256(state_path)
        summary = _pair_summary(
            run_dir=run_dir,
            state_path=state_path,
            background_index=int(background_index),
            sensitivity=sensitivity,
            allowed_params=allowed_params,
            skipped_pairs=[],
            pair_reports=[],
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            provider_after=None,
            evidence_failures=evidence_failures,
            fatal_evidence_failures=evidence_failures,
            local_usability_failures=local_usability_failures,
            disallowed_params=disallowed_params,
            reports=reports,
            provider_report_hash=provider_report_hash,
        )
        summary["one_param_summary"] = one_param_summary
        summary["one_param_diagnosis_summary"] = diagnosis_summary
        summary["report_path"] = str(run_dir / "rung_04_pair_summary.json")
        _write_json(run_dir / "rung_04_pair_summary.json", summary)
        return summary

    caked_report = (
        _read_json(Path(caked_point_reprojection_report_path).expanduser().resolve())
        if caked_point_reprojection_report_path is not None
        else {}
    )
    caked_failures = (
        _caked_reprojection_guard_failures(
            caked_report,
            current_hash=state_hash_before,
            background_index=int(background_index),
        )
        if caked_report
        else ["missing_caked_point_reprojection_report"]
    )

    allowed_set = set(allowed_params)
    skipped_pairs: list[dict[str, object]] = []
    runnable_pairs: list[tuple[str, list[str]]] = []
    for pair_name, raw_pair in RUNG4_DEFAULT_PAIRS:
        pair = [str(name) for name in raw_pair]
        missing = [name for name in pair if name not in allowed_set]
        if missing:
            skipped_pairs.append(
                {
                    "pair_name": pair_name,
                    "pair": pair,
                    "skip_reason": "param_not_allowed_by_singleton_evidence",
                    "missing_params": missing,
                }
            )
            continue
        runnable_pairs.append((pair_name, pair))

    pair_reports: list[dict[str, object]] = []
    dirty_abort = False
    abort_reason = ""
    extension_pairs_added = False
    index = 0
    while True:
        if index >= len(runnable_pairs):
            a_c_passed = any(
                bool(report.get("pass", False))
                and _pair_key(_as_str_list(report.get("pair"))) == _pair_key(("a", "c"))
                for report in pair_reports
            )
            if (
                include_psi_extension_pairs
                and not extension_pairs_added
                and not dirty_abort
                and not abort_reason
                and a_c_passed
                and {"a", "c", "psi_z"}.issubset(allowed_set)
            ):
                runnable_pairs.extend(
                    (pair_name, [str(name) for name in raw_pair])
                    for pair_name, raw_pair in RUNG4_PSI_EXTENSION_PAIRS
                )
                extension_pairs_added = True
                continue
            break

        pair_name, pair = runnable_pairs[index]
        current_hash = _state_sha256(state_path)
        if current_hash != state_hash_before:
            abort_reason = "state_hash_changed_before_pair"
            skipped_pairs.extend(
                {
                    "pair_name": skipped_name,
                    "pair": list(skipped_pair),
                    "skip_reason": abort_reason,
                }
                for skipped_name, skipped_pair in runnable_pairs[index:]
            )
            break
        output_path = _rung_path(run_dir, 4, f"pair_{pair_name}")
        base_parameter_values = _base_parameter_values_for_pair(context, pair)
        if _pair_requires_caked_reprojection(pair) and caked_failures:
            report = {
                "rung": 4,
                "rung_name": f"pair_{pair_name}",
                "pair_name": pair_name,
                "pair": list(pair),
                "status": "failed",
                "pass": False,
                "failure_reason": "caked_point_reprojection_guard_failed",
                "pair_guard_failures": list(caked_failures),
                "active_params": list(pair),
                "var_names": list(pair),
                "candidate_param_names": list(pair),
                "effective_var_names_seen_by_solver": [],
                "least_squares_called": False,
                "optimizer_solve_called": False,
                "real_solve_called": False,
                "base_parameter_values": base_parameter_values,
                "state_sha256_before": state_hash_before,
                "state_sha256_after": _state_sha256(state_path),
                "state_hash_unchanged": state_hash_before == _state_sha256(state_path),
                "caked_point_reprojection_report_path": (
                    str(Path(caked_point_reprojection_report_path).expanduser().resolve())
                    if caked_point_reprojection_report_path is not None
                    else None
                ),
            }
            _write_json(output_path, report)
            reports.append(report)
            pair_reports.append(report)
            index += 1
            continue

        with _timed_report_window("4", f"pair_{pair_name}"):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=pair,
                output_path=output_path,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=4,
                rung_name=f"pair_{pair_name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
            report = _finalize_pair_report(
                report,
                pair_name=pair_name,
                pair=pair,
                state_path=state_path,
                state_hash_before=state_hash_before,
                timeout_seconds=float(timeout_seconds),
                base_parameter_values=base_parameter_values,
            )
            _write_json(output_path, report)
        reports.append(report)
        pair_reports.append(report)
        timeout_integrity_dirty = (
            str(report.get("status", "")) == "timeout"
            and _pair_timeout_integrity_dirty(report)
        )
        if bool(report.get("dirty_timeout_abort", False)) or timeout_integrity_dirty:
            dirty_abort = True
            abort_reason = (
                "dirty_timeout_abort"
                if bool(report.get("dirty_timeout_abort", False))
                else "fixed_source_or_pair_integrity_lost"
            )
            skipped_pairs.extend(
                {
                    "pair_name": skipped_name,
                    "pair": list(skipped_pair),
                    "skip_reason": abort_reason,
                }
                for skipped_name, skipped_pair in runnable_pairs[index + 1 :]
            )
            break
        if bool(report.get("state_hash_unchanged", False)) is not True:
            abort_reason = "state_hash_changed_after_pair"
            skipped_pairs.extend(
                {
                    "pair_name": skipped_name,
                    "pair": list(skipped_pair),
                    "skip_reason": abort_reason,
                }
                for skipped_name, skipped_pair in runnable_pairs[index + 1 :]
            )
            break
        index += 1

    provider_after: dict[str, object] | None = None
    if not dirty_abort:
        with _solver_debug_logging_scope(diagnostic_logging):
            provider_after = run_provider_guard(
                state_path=state_path,
                background_index=int(background_index),
                output_path=_rung_path(run_dir, 4, "provider_guard_after"),
            )
        reports.append(provider_after)

    state_hash_after = _state_sha256(state_path)
    summary = _pair_summary(
        run_dir=run_dir,
        state_path=state_path,
        background_index=int(background_index),
        sensitivity=sensitivity,
        allowed_params=allowed_params,
        skipped_pairs=skipped_pairs,
        pair_reports=pair_reports,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        provider_after=provider_after,
        evidence_failures=([abort_reason] if abort_reason and not pair_reports else []),
        fatal_evidence_failures=(
            [abort_reason] if abort_reason and not pair_reports else []
        ),
        local_usability_failures=local_usability_failures,
        disallowed_params=disallowed_params,
        reports=reports,
        provider_report_hash=provider_report_hash,
    )
    summary["one_param_summary_path"] = (
        str(Path(one_param_summary_path).expanduser().resolve())
        if one_param_summary_path is not None
        else None
    )
    summary["one_param_diagnosis_summary_path"] = (
        str(Path(one_param_diagnosis_summary_path).expanduser().resolve())
        if one_param_diagnosis_summary_path is not None
        else None
    )
    summary["caked_point_reprojection_report_path"] = (
        str(Path(caked_point_reprojection_report_path).expanduser().resolve())
        if caked_point_reprojection_report_path is not None
        else None
    )
    summary["report_path"] = str(run_dir / "rung_04_pair_summary.json")
    _write_json(run_dir / "rung_04_pair_summary.json", summary)
    return summary


def _run_caked_point_reprojection_guard(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    run_id: str,
) -> dict[str, object]:
    from scripts.debug.run_new4_caked_point_reprojection_check import (
        run_new4_caked_point_reprojection_check,
    )

    return run_new4_caked_point_reprojection_check(
        state_path=Path(state_path),
        background_index=int(background_index),
        output_root=Path(output_root),
        run_id=str(run_id),
    )


def _pair_key(pair: Sequence[object]) -> frozenset[str]:
    return frozenset(str(name) for name in pair)


def _passed_pair_keys(pair_summary: Mapping[str, object]) -> set[frozenset[str]]:
    keys: set[frozenset[str]] = set()
    for raw_ref in pair_summary.get("passed_pairs", []) or []:
        if not isinstance(raw_ref, Mapping):
            continue
        pair = _as_str_list(raw_ref.get("pair"))
        if pair:
            keys.add(_pair_key(pair))
    return keys


def _validate_pair_summary_evidence(
    summary_path: Path | None,
    *,
    state_path: Path,
    background_index: int,
    current_hash: str,
    current_provider_report_hash: str,
    current_run_id: str,
) -> tuple[dict[str, object], list[str]]:
    summary, failures = _load_required_json(summary_path, "missing_pair_summary")
    if failures:
        return summary, failures
    failures.extend(
        _state_hash_evidence_failures(summary, current_hash=current_hash)
    )
    failures.extend(
        _run_input_evidence_failures(
            summary,
            state_path=state_path,
            background_index=int(background_index),
        )
    )
    if str(summary.get("status", "")) not in {"ok", "ok_with_failures"}:
        failures.append("pair_summary_status_not_usable")
    if not _passed_pair_keys(summary):
        failures.append("no_passed_pair_evidence")
    provider_hash = str(summary.get("provider_report_hash") or "")
    if provider_hash and provider_hash != str(current_provider_report_hash):
        failures.append("provider_report_hash_not_current")
    if bool(summary.get("dirty_timeout_abort", False)):
        failures.append("dirty_timeout_abort")
    for key in ("run_id", "timestamp"):
        value = str(summary.get(key) or "")
        if value and value != str(current_run_id):
            failures.append(f"{key}_not_current")
    return summary, failures


def _block_requires_caked_reprojection(block: Sequence[str]) -> bool:
    return bool(set(str(name) for name in block) & CAKED_REPROJECTION_REQUIRED_PARAMS)


def _block_dependency_missing(
    block_name: str,
    dependencies: Sequence[Sequence[str]],
    passed_pairs: set[frozenset[str]],
) -> list[list[str]]:
    missing = [
        [str(name) for name in pair]
        for pair in dependencies
        if _pair_key(pair) not in passed_pairs
    ]
    if block_name == "a_c_psi_z":
        psi_pair_ok = (
            _pair_key(("a", "psi_z")) in passed_pairs
            or _pair_key(("c", "psi_z")) in passed_pairs
        )
        if not psi_pair_ok:
            missing.append(["a", "psi_z"])
            missing.append(["c", "psi_z"])
    return missing


def _block_ref(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "block_name": str(report.get("block_name", "")),
        "block": [str(name) for name in report.get("block", []) or []],
    }


def _best_block(
    reports: Sequence[Mapping[str, object]],
    metric_key: str,
) -> dict[str, object] | None:
    best_report: Mapping[str, object] | None = None
    best_value = float("inf")
    for report in reports:
        if bool(report.get("pass", False)) is not True:
            continue
        value = _metric_float(report.get(metric_key, np.nan))
        if math.isfinite(value) and value < best_value:
            best_value = value
            best_report = report
    if best_report is None:
        return None
    payload = _block_ref(best_report)
    payload[metric_key] = best_value
    return payload


def _finalize_block_report(
    report: Mapping[str, object],
    *,
    block_name: str,
    block: Sequence[str],
    state_path: Path,
    state_hash_before: str,
    timeout_seconds: float,
    base_parameter_values: Mapping[str, object],
    provider_after: Mapping[str, object] | None,
    caked_point_reprojection_guard_ok: bool | None = None,
    caked_point_reprojection_report_path: Path | None = None,
) -> dict[str, object]:
    names = [str(name) for name in block]
    finalized = dict(report)
    finalized.setdefault("rung", 5)
    finalized.setdefault("rung_name", f"block_{block_name}")
    finalized["block_name"] = str(block_name)
    finalized["block"] = list(names)
    finalized["active_params"] = list(names)
    finalized["var_names"] = _as_str_list(finalized.get("var_names")) or list(names)
    finalized["candidate_param_names"] = _as_str_list(
        finalized.get("candidate_param_names")
    )
    finalized["effective_var_names_seen_by_solver"] = _as_str_list(
        finalized.get("effective_var_names_seen_by_solver")
    )
    finalized.setdefault("elapsed_s", finalized.get("elapsed_seconds", 0.0))
    finalized.setdefault("elapsed_seconds", finalized.get("elapsed_s", 0.0))
    finalized["timeout_s"] = float(timeout_seconds)
    finalized.setdefault("timeout_seconds", float(timeout_seconds))
    finalized["state_sha256_before"] = str(
        finalized.get("state_sha256_before") or state_hash_before
    )
    finalized["state_sha256_after"] = str(
        finalized.get("state_sha256_after") or _state_sha256(Path(state_path))
    )
    finalized["state_hash_unchanged"] = (
        finalized["state_sha256_before"] == finalized["state_sha256_after"]
    )
    finalized["base_parameter_values"] = dict(base_parameter_values)
    before, after, delta, bounds = _parameter_maps_from_deltas(
        finalized,
        names,
        base_parameter_values,
    )
    finalized["parameter_before"] = before
    finalized["parameter_after"] = after
    finalized["parameter_delta"] = delta
    finalized["parameter_bounds"] = bounds
    finalized.setdefault("point_match_summary", {})
    finalized.setdefault(
        "last_point_match_summary",
        finalized.get("point_match_summary") if finalized.get("point_match_summary") else None,
    )
    finalized.setdefault("dirty_timeout_abort", False)
    provider_after_payload = dict(provider_after or {})
    provider_after_ok = (
        bool(provider_after_payload.get("provider_guard_ok", False))
        and str(provider_after_payload.get("classification", "")) == "point_provider_parity_ok"
    )
    finalized["provider_guard_after_ok"] = bool(provider_after_ok)
    finalized["provider_guard_after"] = provider_after_payload
    if caked_point_reprojection_guard_ok is not None:
        finalized["caked_point_reprojection_guard_ok"] = bool(
            caked_point_reprojection_guard_ok
        )
    if caked_point_reprojection_report_path is not None:
        finalized["caked_point_reprojection_report_path"] = str(
            Path(caked_point_reprojection_report_path).expanduser().resolve()
        )
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        finalized.setdefault(key, None)
    for key in PROVIDER_MATCH_BOOLS:
        finalized.setdefault(key, None)

    if str(finalized.get("status", "")) == "timeout":
        finalized["pass"] = False
        if bool(finalized.get("dirty_timeout_abort", False)):
            finalized["failure_reason"] = "dirty_timeout_abort"
        elif _pair_timeout_integrity_dirty(finalized):
            finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
        else:
            finalized["failure_reason"] = "timeout"
        finalized["block_guard_failures"] = [str(finalized["failure_reason"])]
        return finalized

    integrity_failures = _fixed_source_contract_failures(
        finalized,
        expected_counts=ONE_PARAM_FIXED_SOURCE_COUNTS,
        required_bool_keys=PROVIDER_MATCH_BOOLS,
    )
    metric_failures = _pair_metric_failures(finalized)
    solve_flag_failures: list[str] = []
    if bool(finalized.get("least_squares_called", False)) is not True:
        solve_flag_failures.append("least_squares_not_called")
    if not (
        bool(finalized.get("optimizer_solve_called", False))
        or bool(finalized.get("real_solve_called", False))
    ):
        solve_flag_failures.append("optimizer_solve_not_called")
    if finalized.get("candidate_param_names") != list(names):
        solve_flag_failures.append("candidate_param_names_not_block")
    if finalized.get("var_names") != list(names):
        solve_flag_failures.append("var_names_not_block")
    if finalized.get("effective_var_names_seen_by_solver") != list(names):
        solve_flag_failures.append("effective_var_names_seen_by_solver_not_block")
    if bool(finalized.get("state_hash_unchanged", False)) is not True:
        solve_flag_failures.append("state_hash_changed")
    if provider_after_ok is not True:
        solve_flag_failures.append("provider_guard_after_failed")
    if (
        _block_requires_caked_reprojection(names)
        and bool(finalized.get("caked_point_reprojection_guard_ok", False)) is not True
    ):
        solve_flag_failures.append("caked_point_reprojection_guard_failed")

    guard_failures = integrity_failures + metric_failures + solve_flag_failures
    finalized["block_guard_failures"] = guard_failures
    if not guard_failures:
        finalized["status"] = "ok"
        finalized["pass"] = True
        finalized["failure_reason"] = None
        return finalized

    finalized["status"] = "failed"
    finalized["pass"] = False
    if integrity_failures:
        finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
    elif "no_matched_peak_rejection" in metric_failures:
        finalized["failure_reason"] = "no_matched_peak_rejection"
    elif any(reason.startswith("non_finite") for reason in metric_failures):
        finalized["failure_reason"] = "non_finite_residual"
    elif metric_failures:
        finalized["failure_reason"] = "metric_guard_failed"
    elif solve_flag_failures:
        finalized["failure_reason"] = "solve_flag_guard_failed"
    else:
        finalized.setdefault("failure_reason", "block_solve_failed")
    return finalized


def _block_summary(
    *,
    run_dir: Path,
    state_path: Path,
    background_index: int,
    pair_summary: Mapping[str, object],
    block_reports: Sequence[Mapping[str, object]],
    state_hash_before: str,
    state_hash_after: str,
    evidence_failures: Sequence[str] = (),
    fatal_evidence_failures: Sequence[str] = (),
    local_usability_failures: Sequence[str] = (),
    allowed_params: Sequence[str] = (),
    disallowed_params: Sequence[str] = (),
    reports: Sequence[Mapping[str, object]] | None = None,
    provider_report_hash: str | None = None,
) -> dict[str, object]:
    passed_reports = [report for report in block_reports if bool(report.get("pass", False))]
    skipped_reports = [
        report for report in block_reports if str(report.get("status", "")) == "skipped"
    ]
    failed_reports = [
        report for report in block_reports if str(report.get("status", "")) == "failed"
    ]
    timed_out_reports = [
        report for report in block_reports if str(report.get("status", "")) == "timeout"
    ]
    attempted_reports = [
        report for report in block_reports if str(report.get("status", "")) != "skipped"
    ]
    improved_reports: list[Mapping[str, object]] = []
    neutral_reports: list[Mapping[str, object]] = []
    for report in passed_reports:
        before_rms = _metric_float(report.get("before_rms_px", np.nan))
        after_rms = _metric_float(report.get("after_rms_px", np.nan))
        before_max = _metric_float(report.get("before_max_error_px", np.nan))
        after_max = _metric_float(report.get("after_max_error_px", np.nan))
        improved = (
            math.isfinite(before_rms)
            and math.isfinite(after_rms)
            and after_rms < before_rms - PAIR_RMS_TOLERANCE_PX
        ) or (
            math.isfinite(before_max)
            and math.isfinite(after_max)
            and after_max < before_max - PAIR_MAX_ERROR_TOLERANCE_PX
        )
        if improved:
            improved_reports.append(report)
        else:
            neutral_reports.append(report)

    if evidence_failures:
        status = "failed"
        failure_reason = "pair_evidence_not_current"
    elif not attempted_reports:
        status = "failed"
        failure_reason = (
            "no_dependency_backed_blocks" if skipped_reports else "no_block_solve_passed"
        )
    elif not passed_reports:
        status = "failed"
        failure_reason = "no_block_solve_passed"
    elif failed_reports or timed_out_reports:
        status = "ok_with_failures"
        failure_reason = None
    else:
        status = "ok"
        failure_reason = None
    provider_checked_reports = [
        report
        for report in attempted_reports
        if "provider_guard_after_ok" in report
        and (
            bool(report.get("least_squares_called", False))
            or bool(report.get("optimizer_solve_called", False))
            or bool(report.get("real_solve_called", False))
        )
    ]
    provider_guard_after_ok = bool(provider_checked_reports) and all(
        bool(report.get("provider_guard_after_ok", False))
        for report in provider_checked_reports
    )
    best_by_rms = _best_block(block_reports, "after_rms_px")
    recommended_next_full_candidate = dict(best_by_rms) if best_by_rms else None
    fatal_failures = [
        str(item) for item in (fatal_evidence_failures or evidence_failures)
    ]
    summary = {
        "rung": 5,
        "rung_name": "block_summary",
        "status": status,
        "failure_reason": failure_reason,
        "run_dir": str(run_dir),
        "run_id": Path(run_dir).name,
        "timestamp": Path(run_dir).name,
        "state_path": str(Path(state_path).expanduser().resolve()),
        "background_index": int(background_index),
        "provider_report_hash": str(provider_report_hash or ""),
        "pair_summary_path": pair_summary.get("report_path"),
        "pair_summary_status": pair_summary.get("status"),
        "allowed_params": [str(name) for name in allowed_params],
        "disallowed_params": [str(name) for name in disallowed_params],
        "attempted_blocks": [_block_ref(report) for report in attempted_reports],
        "passed_blocks": [_block_ref(report) for report in passed_reports],
        "failed_blocks": [_block_ref(report) for report in failed_reports],
        "skipped_blocks": [_block_ref(report) for report in skipped_reports],
        "timed_out_blocks": [_block_ref(report) for report in timed_out_reports],
        "best_block_by_rms": best_by_rms,
        "best_block_by_max_error": _best_block(block_reports, "after_max_error_px"),
        "improved_blocks": [_block_ref(report) for report in improved_reports],
        "neutral_blocks": [_block_ref(report) for report in neutral_reports],
        "recommended_next_blocks": [_block_ref(report) for report in passed_reports],
        "recommended_next_full_candidate": recommended_next_full_candidate,
        "full_fitter_validated": False,
        "state_sha256_before": state_hash_before,
        "state_sha256_after": state_hash_after,
        "state_hash_before": state_hash_before,
        "state_hash_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_before == state_hash_after),
        "provider_guard_after_ok": bool(provider_guard_after_ok),
        "evidence_failures": [str(item) for item in evidence_failures],
        "fatal_evidence_failures": fatal_failures,
        "local_usability_failures": [
            str(item) for item in local_usability_failures
        ],
        "reports": list(reports or []),
    }
    return summary


def _write_skipped_block_report(
    *,
    output_path: Path,
    block_name: str,
    block: Sequence[str],
    missing_pairs: Sequence[Sequence[str]],
    state_hash_before: str,
    state_path: Path,
) -> dict[str, object]:
    report = {
        "rung": 5,
        "rung_name": f"block_{block_name}",
        "block_name": str(block_name),
        "block": [str(name) for name in block],
        "status": "skipped",
        "pass": False,
        "skip_reason": "missing_pair_evidence",
        "missing_pair_evidence": [[str(name) for name in pair] for pair in missing_pairs],
        "candidate_param_names": [str(name) for name in block],
        "var_names": [str(name) for name in block],
        "effective_var_names_seen_by_solver": [str(name) for name in block],
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "real_solve_called": False,
        "state_sha256_before": state_hash_before,
        "state_sha256_after": _state_sha256(state_path),
        "state_hash_before": state_hash_before,
        "state_hash_after": _state_sha256(state_path),
        "state_hash_unchanged": state_hash_before == _state_sha256(state_path),
    }
    _write_json(output_path, report)
    return report


def _write_caked_failed_block_report(
    *,
    output_path: Path,
    block_name: str,
    block: Sequence[str],
    caked_failures: Sequence[str],
    caked_report_path: Path | None,
    state_hash_before: str,
    state_path: Path,
    base_parameter_values: Mapping[str, object],
) -> dict[str, object]:
    report = {
        "rung": 5,
        "rung_name": f"block_{block_name}",
        "block_name": str(block_name),
        "block": [str(name) for name in block],
        "status": "failed",
        "pass": False,
        "failure_reason": "caked_point_reprojection_guard_failed",
        "block_guard_failures": [str(item) for item in caked_failures],
        "active_params": [str(name) for name in block],
        "var_names": [str(name) for name in block],
        "candidate_param_names": [str(name) for name in block],
        "effective_var_names_seen_by_solver": [],
        "caked_point_reprojection_guard_ok": False,
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "real_solve_called": False,
        "base_parameter_values": dict(base_parameter_values),
        "state_sha256_before": state_hash_before,
        "state_sha256_after": _state_sha256(state_path),
        "state_hash_before": state_hash_before,
        "state_hash_after": _state_sha256(state_path),
        "state_hash_unchanged": state_hash_before == _state_sha256(state_path),
        "caked_point_reprojection_report_path": (
            str(Path(caked_report_path).expanduser().resolve())
            if caked_report_path is not None
            else None
        ),
    }
    _write_json(output_path, report)
    return report


def _run_block_stage(
    *,
    state_path: Path,
    background_index: int,
    run_dir: Path,
    context: Mapping[str, object],
    sensitivity: Mapping[str, object],
    reports: list[dict[str, object]],
    state_hash_before: str,
    max_nfev: int,
    timeout_seconds: float,
    sensitivity_report_path: Path,
    provider_report_hash: str,
    pair_summary_path: Path | None = None,
    one_param_summary_path: Path | None = None,
    one_param_diagnosis_summary_path: Path | None = None,
    caked_point_reprojection_report_path: Path | None = None,
    use_subprocess: bool = False,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    caked_report_path = caked_point_reprojection_report_path
    pair_summary: dict[str, object]
    evidence_failures: list[str] = []
    fatal_evidence_failures: list[str] = []
    local_usability_failures: list[str] = []

    if pair_summary_path is not None:
        pair_summary, pair_failures = _validate_pair_summary_evidence(
            pair_summary_path,
            state_path=state_path,
            background_index=int(background_index),
            current_hash=state_hash_before,
            current_provider_report_hash=provider_report_hash,
            current_run_id=Path(run_dir).name,
        )
        pair_failure_refs = [f"pair:{failure}" for failure in pair_failures]
        evidence_failures.extend(pair_failure_refs)
        fatal_evidence_failures.extend(pair_failure_refs)
    else:
        with _timed_report_window("3", "one_param_summary"):
            one_param_summary = _run_one_param_stage(
                state_path=state_path,
                background_index=int(background_index),
                run_dir=run_dir,
                context=context,
                sensitivity=sensitivity,
                sensitivity_report_path=sensitivity_report_path,
                reports=reports,
                state_hash_before=state_hash_before,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                use_subprocess=bool(use_subprocess),
                diagnostic_logging=bool(diagnostic_logging),
            )
        one_param_summary_path = run_dir / "rung_03_one_param_summary.json"
        current_active_params, _skipped = _active_params_from_sensitivity(sensitivity, context)
        if (
            "a" in set(current_active_params)
            and "a" not in set(_as_str_list(one_param_summary.get("passed_params")))
        ):
            with _timed_report_window("3A", "a_diagnosis"):
                with _suppress_timing_collection():
                    diagnosis = run_one_param_diagnosis_variants(
                        state_path=state_path,
                        background_index=int(background_index),
                        output_root=run_dir / "rung_03a_a_diagnosis",
                        one_param_filter="a",
                        use_subprocess=bool(use_subprocess),
                        diagnostic_logging=bool(diagnostic_logging),
                    )
            reports.append(diagnosis)
            one_param_diagnosis_summary_path = (
                run_dir / "rung_03a_a_diagnosis" / "variant_summary.json"
            )
        elif one_param_diagnosis_summary_path is None:
            one_param_diagnosis_summary_path = None

        with _solver_debug_logging_scope(diagnostic_logging):
            with _timed_report_window("3B", "caked_point_reprojection"):
                caked_report = _run_caked_point_reprojection_guard(
                    state_path=state_path,
                    background_index=int(background_index),
                    output_root=run_dir,
                    run_id="rung_03b_caked_point_reprojection",
                )
        reports.append(caked_report)
        report_path = caked_report.get("report_path")
        if report_path:
            caked_report_path = Path(str(report_path))

        with _timed_report_window("4", "pair_summary"):
            pair_summary = _run_pair_stage(
                state_path=state_path,
                background_index=int(background_index),
                run_dir=run_dir,
                context=context,
                sensitivity=sensitivity,
                reports=reports,
                state_hash_before=state_hash_before,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                one_param_summary_path=one_param_summary_path,
                one_param_diagnosis_summary_path=one_param_diagnosis_summary_path,
                caked_point_reprojection_report_path=caked_report_path,
                include_psi_extension_pairs=True,
                provider_report_hash=provider_report_hash,
                use_subprocess=bool(use_subprocess),
                diagnostic_logging=bool(diagnostic_logging),
            )
        local_usability_failures.extend(
            _as_str_list(pair_summary.get("local_usability_failures"))
        )
        pair_fatal_failures = _as_str_list(pair_summary.get("fatal_evidence_failures"))
        if pair_fatal_failures:
            pair_failure_refs = [f"pair:{failure}" for failure in pair_fatal_failures]
            evidence_failures.extend(pair_failure_refs)
            fatal_evidence_failures.extend(pair_failure_refs)
        elif (
            str(pair_summary.get("status", "")) not in {"ok", "ok_with_failures"}
            and _passed_pair_keys(pair_summary)
        ):
            evidence_failures.append("pair:pair_summary_status_not_usable")
            fatal_evidence_failures.append("pair:pair_summary_status_not_usable")

    allowed_params = _as_str_list(pair_summary.get("allowed_params"))
    if not allowed_params:
        allowed_seen: list[str] = []
        for raw_ref in pair_summary.get("passed_pairs", []) or []:
            if not isinstance(raw_ref, Mapping):
                continue
            for name in _as_str_list(raw_ref.get("pair")):
                if name not in allowed_seen:
                    allowed_seen.append(name)
        allowed_params = allowed_seen
    disallowed_params = _as_str_list(pair_summary.get("disallowed_params"))

    if caked_report_path is None:
        raw_caked_path = pair_summary.get("caked_point_reprojection_report_path")
        if raw_caked_path:
            caked_report_path = Path(str(raw_caked_path))
    caked_report = (
        _read_json(Path(caked_report_path).expanduser().resolve())
        if caked_report_path is not None
        else {}
    )
    caked_failures = (
        _caked_reprojection_guard_failures(
            caked_report,
            current_hash=state_hash_before,
            background_index=int(background_index),
        )
        if caked_report
        else ["missing_caked_point_reprojection_report"]
    )

    if evidence_failures:
        with _timed_report_window("5", "block_summary"):
            state_hash_after = _state_sha256(state_path)
            summary = _block_summary(
                run_dir=run_dir,
                state_path=state_path,
                background_index=int(background_index),
                pair_summary=pair_summary,
                block_reports=[],
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                evidence_failures=evidence_failures,
                fatal_evidence_failures=fatal_evidence_failures,
                local_usability_failures=local_usability_failures,
                allowed_params=allowed_params,
                disallowed_params=disallowed_params,
                reports=reports,
                provider_report_hash=provider_report_hash,
            )
            summary["report_path"] = str(run_dir / "rung_05_block_summary.json")
            _write_json(run_dir / "rung_05_block_summary.json", summary)
            return summary

    block_summary_window = _TimingWindow(
        rung_id="5",
        rung_name="block_summary",
        started_perf=_perf_counter(),
        started_at=_utc_now(),
    )
    passed_pairs = _passed_pair_keys(pair_summary)
    block_reports: list[dict[str, object]] = []
    dirty_abort = False
    for index, (block_name, raw_block, dependencies) in enumerate(RUNG5_BLOCKS):
        block = [str(name) for name in raw_block]
        output_path = _rung_path(run_dir, 5, f"block_{block_name}")
        missing_pairs = _block_dependency_missing(block_name, dependencies, passed_pairs)
        if missing_pairs:
            with _timed_report_window("5", f"block_{block_name}"):
                report = _write_skipped_block_report(
                    output_path=output_path,
                    block_name=block_name,
                    block=block,
                    missing_pairs=missing_pairs,
                    state_hash_before=state_hash_before,
                    state_path=state_path,
                )
            reports.append(report)
            block_reports.append(report)
            continue

        current_hash = _state_sha256(state_path)
        if current_hash != state_hash_before:
            for skipped_name, skipped_block, _deps in RUNG5_BLOCKS[index:]:
                with _timed_report_window("5", f"block_{skipped_name}"):
                    report = {
                        "rung": 5,
                        "rung_name": f"block_{skipped_name}",
                        "block_name": str(skipped_name),
                        "block": [str(name) for name in skipped_block],
                        "status": "skipped",
                        "pass": False,
                        "skip_reason": "state_hash_changed_before_block",
                        "state_sha256_before": state_hash_before,
                        "state_sha256_after": current_hash,
                        "state_hash_before": state_hash_before,
                        "state_hash_after": current_hash,
                        "state_hash_unchanged": False,
                    }
                    _write_json(_rung_path(run_dir, 5, f"block_{skipped_name}"), report)
                reports.append(report)
                block_reports.append(report)
            break

        base_parameter_values = _base_parameter_values_for_pair(context, block)
        if _block_requires_caked_reprojection(block) and caked_failures:
            with _timed_report_window("5", f"block_{block_name}"):
                report = _write_caked_failed_block_report(
                    output_path=output_path,
                    block_name=block_name,
                    block=block,
                    caked_failures=caked_failures,
                    caked_report_path=caked_report_path,
                    state_hash_before=state_hash_before,
                    state_path=state_path,
                    base_parameter_values=base_parameter_values,
                )
            reports.append(report)
            block_reports.append(report)
            continue

        with _timed_report_window("5", f"block_{block_name}"):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=block,
                output_path=output_path,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=5,
                rung_name=f"block_{block_name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
            provider_after: dict[str, object] | None = None
            if not bool(report.get("dirty_timeout_abort", False)):
                with _solver_debug_logging_scope(diagnostic_logging):
                    provider_after = _run_provider_guard_report(
                        state_path=state_path,
                        background_index=int(background_index),
                        rung=5,
                        rung_name=f"block_{block_name}_provider_guard_after",
                    )
            report = _finalize_block_report(
                report,
                block_name=block_name,
                block=block,
                state_path=state_path,
                state_hash_before=state_hash_before,
                timeout_seconds=float(timeout_seconds),
                base_parameter_values=base_parameter_values,
                provider_after=provider_after,
                caked_point_reprojection_guard_ok=(
                    not caked_failures if _block_requires_caked_reprojection(block) else None
                ),
                caked_point_reprojection_report_path=(
                    caked_report_path if _block_requires_caked_reprojection(block) else None
                ),
            )
            _write_json(output_path, report)
        reports.append(report)
        block_reports.append(report)
        timeout_integrity_dirty = (
            str(report.get("status", "")) == "timeout"
            and _pair_timeout_integrity_dirty(report)
        )
        if bool(report.get("dirty_timeout_abort", False)) or timeout_integrity_dirty:
            dirty_abort = True
            skip_reason = (
                "dirty_timeout_abort"
                if bool(report.get("dirty_timeout_abort", False))
                else "fixed_source_or_pair_integrity_lost"
            )
            for skipped_name, skipped_block, _deps in RUNG5_BLOCKS[index + 1 :]:
                with _timed_report_window("5", f"block_{skipped_name}"):
                    skipped_report = {
                        "rung": 5,
                        "rung_name": f"block_{skipped_name}",
                        "block_name": str(skipped_name),
                        "block": [str(name) for name in skipped_block],
                        "status": "skipped",
                        "pass": False,
                        "skip_reason": skip_reason,
                        "state_sha256_before": state_hash_before,
                        "state_sha256_after": _state_sha256(state_path),
                        "state_hash_before": state_hash_before,
                        "state_hash_after": _state_sha256(state_path),
                        "state_hash_unchanged": state_hash_before == _state_sha256(state_path),
                    }
                    _write_json(_rung_path(run_dir, 5, f"block_{skipped_name}"), skipped_report)
                reports.append(skipped_report)
                block_reports.append(skipped_report)
            break
        if bool(report.get("state_hash_unchanged", False)) is not True:
            for skipped_name, skipped_block, _deps in RUNG5_BLOCKS[index + 1 :]:
                with _timed_report_window("5", f"block_{skipped_name}"):
                    skipped_report = {
                        "rung": 5,
                        "rung_name": f"block_{skipped_name}",
                        "block_name": str(skipped_name),
                        "block": [str(name) for name in skipped_block],
                        "status": "skipped",
                        "pass": False,
                        "skip_reason": "state_hash_changed_after_block",
                        "state_sha256_before": state_hash_before,
                        "state_sha256_after": _state_sha256(state_path),
                        "state_hash_before": state_hash_before,
                        "state_hash_after": _state_sha256(state_path),
                        "state_hash_unchanged": False,
                    }
                    _write_json(_rung_path(run_dir, 5, f"block_{skipped_name}"), skipped_report)
                reports.append(skipped_report)
                block_reports.append(skipped_report)
            break

    state_hash_after = _state_sha256(state_path)
    summary = _block_summary(
        run_dir=run_dir,
        state_path=state_path,
        background_index=int(background_index),
        pair_summary=pair_summary,
        block_reports=block_reports,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        evidence_failures=[],
        fatal_evidence_failures=[],
        local_usability_failures=local_usability_failures,
        allowed_params=allowed_params,
        disallowed_params=disallowed_params,
        reports=reports,
        provider_report_hash=provider_report_hash,
    )
    summary["dirty_timeout_abort"] = bool(dirty_abort)
    summary["caked_point_reprojection_report_path"] = (
        str(Path(caked_report_path).expanduser().resolve())
        if caked_report_path is not None
        else None
    )
    summary["report_path"] = str(run_dir / "rung_05_block_summary.json")
    block_window_token = _ACTIVE_TIMING_WINDOW.set(block_summary_window)
    try:
        _write_json(run_dir / "rung_05_block_summary.json", summary)
    finally:
        _ACTIVE_TIMING_WINDOW.reset(block_window_token)
    return summary


def _groups_for_cumulative_rungs(passed_params: Sequence[str], theta_name: str) -> list[tuple[str, list[str]]]:
    passed = set(str(name) for name in passed_params)
    groups: list[tuple[str, list[str]]] = []
    current: list[str] = []
    for name, additions in [
        ("center", ["center_x", "center_y"]),
        ("center_primary_tilt", ["gamma", "Gamma"]),
        ("center_tilt", ["chi", "cor_angle"]),
        ("center_tilt_theta", [theta_name]),
        ("center_tilt_theta_distance", ["corto_detector"]),
        ("center_tilt_theta_distance_z", ["zs", "zb"]),
        ("center_tilt_theta_distance_lattice", ["a", "c", "psi_z"]),
    ]:
        if all(param in passed for param in additions):
            for param in additions:
                if param not in current:
                    current.append(param)
            groups.append((name, list(current)))
    return groups


def run_ladder(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    max_rung: str,
    timeout_seconds: float = 120.0,
    max_nfev: int = 20,
    timestamp: str | None = None,
    sensitivity_report: Path | None = None,
    one_param_filter: str | None = None,
    one_param_summary: Path | None = None,
    one_param_diagnosis_summary: Path | None = None,
    caked_point_reprojection_report: Path | None = None,
    pair_summary: Path | None = None,
    timing_report: Path | None = None,
    use_subprocess: bool = False,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    state_path = Path(state_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    run_dir = output_root / str(timestamp or _run_stamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    state_hash_before = _state_sha256(state_path)
    reports: list[dict[str, object]] = []
    parent_collector = _ACTIVE_TIMING_COLLECTOR.get()
    collector_owned = parent_collector is None
    collector = parent_collector or _TimingCollector(
        run_dir=run_dir,
        expected_rung_ids=_expected_rung_ids_for_run(
            max_rung,
            one_param_summary=one_param_summary,
            pair_summary=pair_summary,
            caked_point_reprojection_report=caked_point_reprojection_report,
        ),
    )
    collector_token = (
        _ACTIVE_TIMING_COLLECTOR.set(collector) if collector_owned else None
    )
    timing_report_path = (
        Path(timing_report).expanduser().resolve() if timing_report is not None else None
    )

    def _finish(result: dict[str, object]) -> dict[str, object]:
        try:
            timing_summary_path = run_dir / "rung_timing_summary.json"
            timing_summary: dict[str, object] | None = None
            if collector_owned:
                timing_summary = collector.summary()
                result["timing_summary_path"] = str(timing_summary_path)
                if timing_report_path is not None:
                    result["timing_report_path"] = str(timing_report_path)
                for key in (
                    "total_elapsed_s",
                    "timing_collection_mode",
                    "completed_rung_count",
                    "missing_expected_rungs",
                    "rung_timings",
                    "slowest_rung",
                    "slowest_rung_elapsed_s",
                    "timing_threshold_status",
                    "timing_threshold_max_s",
                    "timing_threshold_exceeded_rungs",
                ):
                    result[key] = timing_summary.get(key)
            _write_json(run_dir / "ladder_summary.json", result)
            if collector_owned and timing_summary is not None:
                with _suppress_timing_collection():
                    _write_json(timing_summary_path, timing_summary)
                    if timing_report_path is not None:
                        _write_json(timing_report_path, timing_summary)
            return result
        finally:
            if collector_token is not None:
                _ACTIVE_TIMING_COLLECTOR.reset(collector_token)

    with _solver_debug_logging_scope(diagnostic_logging):
        with _timed_report_window("0", "provider_guard"):
            provider_report = run_provider_guard(
                state_path=state_path,
                background_index=int(background_index),
                output_path=_rung_path(run_dir, 0, "provider_guard"),
            )
    provider_report_hash = _provider_report_evidence_hash(provider_report)
    reports.append(provider_report)
    if not bool(provider_report.get("provider_guard_ok", False)):
        result = {
            "status": "aborted",
            "reason": "provider_guard_failed",
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
        }
        return _finish(result)

    with _solver_debug_logging_scope(diagnostic_logging):
        context = _capture_solver_context(state_path, int(background_index))
    with _solver_debug_logging_scope(diagnostic_logging):
        with _timed_report_window("1", "objective_dry_run"):
            dry_report = run_objective_dry_run(
                context,
                output_path=_rung_path(run_dir, 1, "objective_dry_run"),
                max_nfev=int(max_nfev),
            )
    reports.append(dry_report)
    rung1_failures = _rung1_green_failures(dry_report)
    if rung1_failures:
        result = {
            "status": "aborted",
            "reason": "objective_dry_run_failed",
            "failure_reason": (
                "sensitivity_not_green"
                if str(max_rung).strip().lower()
                in {"one-param", "pair", "pairs", "block", "blocks"}
                else "objective_dry_run_failed"
            ),
            "rung_1_failures": rung1_failures,
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
            "residual_probe_called": False,
            "least_squares_called": False,
            "optimizer_solve_called": False,
        }
        return _finish(result)

    max_rung_name = str(max_rung).strip().lower()
    sensitivity_output_path = _rung_path(run_dir, 2, "sensitivity_scan")
    with _timed_report_window("2", "sensitivity_scan"):
        if sensitivity_report is not None:
            sensitivity = _read_json(Path(sensitivity_report).expanduser().resolve())
            sensitivity["debug_sensitivity_report_override"] = str(
                Path(sensitivity_report).expanduser().resolve()
            )
            _write_json(sensitivity_output_path, sensitivity)
        else:
            with _solver_debug_logging_scope(diagnostic_logging):
                sensitivity = run_sensitivity_scan(
                    context,
                    output_path=sensitivity_output_path,
                    max_nfev=int(max_nfev),
                    rung_1_report=dry_report,
                    state_path=state_path,
                    state_hash_before=state_hash_before,
                )
    reports.append(sensitivity)
    if not bool(sensitivity.get("pass", False)):
        if max_rung_name == "one-param":
            with _timed_report_window("3", "one_param_summary"):
                result = _run_one_param_stage(
                    state_path=state_path,
                    background_index=int(background_index),
                    run_dir=run_dir,
                    context=context,
                    sensitivity=sensitivity,
                    sensitivity_report_path=sensitivity_output_path,
                    reports=reports,
                    state_hash_before=state_hash_before,
                    max_nfev=int(max_nfev),
                    timeout_seconds=float(timeout_seconds),
                    one_param_filter=one_param_filter,
                    use_subprocess=bool(use_subprocess),
                    diagnostic_logging=bool(diagnostic_logging),
                )
            return _finish(result)
        result = {
            "status": "aborted",
            "reason": "sensitivity_scan_failed",
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
        }
        return _finish(result)

    if max_rung_name == "sensitivity":
        result = {
            "status": "pass",
            "run_dir": str(run_dir),
            "final_selected_params": list(sensitivity.get("active_params", []) or []),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
            "state_unchanged": state_hash_before == _state_sha256(state_path),
            "residual_probe_called": bool(sensitivity.get("residual_probe_called", False)),
            "least_squares_called": False,
            "optimizer_solve_called": False,
        }
        return _finish(result)

    if max_rung_name == "one-param":
        with _timed_report_window("3", "one_param_summary"):
            result = _run_one_param_stage(
                state_path=state_path,
                background_index=int(background_index),
                run_dir=run_dir,
                context=context,
                sensitivity=sensitivity,
                sensitivity_report_path=sensitivity_output_path,
                reports=reports,
                state_hash_before=state_hash_before,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                one_param_filter=one_param_filter,
                use_subprocess=bool(use_subprocess),
                diagnostic_logging=bool(diagnostic_logging),
            )
        return _finish(result)

    if max_rung_name in {"block", "blocks"}:
        result = _run_block_stage(
            state_path=state_path,
            background_index=int(background_index),
            run_dir=run_dir,
            context=context,
            sensitivity=sensitivity,
            reports=reports,
            state_hash_before=state_hash_before,
            max_nfev=int(max_nfev),
            timeout_seconds=float(timeout_seconds),
            sensitivity_report_path=sensitivity_output_path,
            provider_report_hash=provider_report_hash,
            pair_summary_path=pair_summary,
            one_param_summary_path=one_param_summary,
            one_param_diagnosis_summary_path=one_param_diagnosis_summary,
            caked_point_reprojection_report_path=caked_point_reprojection_report,
            use_subprocess=bool(use_subprocess),
            diagnostic_logging=bool(diagnostic_logging),
        )
        return _finish(result)

    if max_rung_name in {"pair", "pairs"}:
        with _timed_report_window("4", "pair_summary"):
            result = _run_pair_stage(
                state_path=state_path,
                background_index=int(background_index),
                run_dir=run_dir,
                context=context,
                sensitivity=sensitivity,
                reports=reports,
                state_hash_before=state_hash_before,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                one_param_summary_path=one_param_summary,
                one_param_diagnosis_summary_path=one_param_diagnosis_summary,
                caked_point_reprojection_report_path=caked_point_reprojection_report,
                provider_report_hash=provider_report_hash,
                use_subprocess=bool(use_subprocess),
                diagnostic_logging=bool(diagnostic_logging),
            )
        return _finish(result)

    sensitive = set(str(name) for name in sensitivity.get("active_parameters", []) or [])
    one_param_reports: list[dict[str, object]] = []
    center_only = max_rung_name == "center"
    params_to_run = CENTER_PARAMS if center_only else _candidate_order(context)
    for name in params_to_run:
        if name not in sensitive and not (center_only and name in CENTER_PARAMS):
            continue
        with _timed_report_window("3", f"one_param_{name}"):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=[name],
                output_path=_rung_path(run_dir, 3, f"one_param_{name}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=3,
                rung_name=f"one_param_{name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
        reports.append(report)
        one_param_reports.append(report)
        if not bool(report.get("pass", False)):
            result = {
                "status": "stopped",
                "reason": "one_param_failed",
                "failed_parameter": str(name),
                "run_dir": str(run_dir),
                "reports": reports,
                "state_sha256_before": state_hash_before,
                "state_sha256_after": _state_sha256(state_path),
            }
            return _finish(result)

    passed_params = _passed_params_from_one_param_reports(one_param_reports)
    theta_name = _active_theta_name(context) or "theta_initial"
    pair_groups = (
        [("center_xy", ["center_x", "center_y"])]
        if center_only
        else _groups_for_pair_rungs(passed_params, theta_name)
    )
    for group_name, names in pair_groups:
        with _timed_report_window("4", f"pair_{group_name}"):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=names,
                output_path=_rung_path(run_dir, 4, f"pair_{group_name}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=4,
                rung_name=f"pair_{group_name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
        reports.append(report)
        if not bool(report.get("pass", False)):
            result = {
                "status": "stopped",
                "reason": "pair_failed",
                "failed_group": str(group_name),
                "run_dir": str(run_dir),
                "reports": reports,
                "state_sha256_before": state_hash_before,
                "state_sha256_after": _state_sha256(state_path),
            }
            return _finish(result)

    final_selected_params = list(pair_groups[-1][1]) if pair_groups else list(passed_params)
    if not center_only:
        for group_name, names in _groups_for_cumulative_rungs(passed_params, theta_name):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=names,
                output_path=_rung_path(run_dir, 5, f"block_{group_name}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=5,
                rung_name=f"block_{group_name}",
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
            reports.append(report)
            if not bool(report.get("pass", False)):
                result = {
                    "status": "stopped",
                    "reason": "cumulative_block_failed",
                    "failed_group": str(group_name),
                    "run_dir": str(run_dir),
                    "reports": reports,
                    "state_sha256_before": state_hash_before,
                    "state_sha256_after": _state_sha256(state_path),
                }
                return _finish(result)
            final_selected_params = list(names)

    if max_rung_name == "features" and final_selected_params:
        for feature in FEATURE_RUNS:
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=final_selected_params,
                output_path=_rung_path(run_dir, 6, f"feature_{feature}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=6,
                rung_name=f"feature_{feature}",
                feature=feature,
                state_hash_before=state_hash_before,
                use_subprocess=bool(use_subprocess),
                context=context,
                diagnostic_logging=bool(diagnostic_logging),
                dirty_timeout_on_timeout=not bool(use_subprocess),
            )
            reports.append(report)
            if not bool(report.get("pass", False)):
                result = {
                    "status": "stopped",
                    "reason": "feature_failed",
                    "failed_feature": str(feature),
                    "run_dir": str(run_dir),
                    "reports": reports,
                    "state_sha256_before": state_hash_before,
                    "state_sha256_after": _state_sha256(state_path),
                }
                return _finish(result)

    result = {
        "status": "pass",
        "run_dir": str(run_dir),
        "final_selected_params": final_selected_params,
        "reports": reports,
        "state_sha256_before": state_hash_before,
        "state_sha256_after": _state_sha256(state_path),
        "state_unchanged": state_hash_before == _state_sha256(state_path),
    }
    return _finish(result)


def _variant_can_continue(report: Mapping[str, object]) -> bool:
    if str(report.get("failure_reason", "")) == "dirty_timeout_abort":
        return False
    if str(report.get("diagnosis_classification", "")) == "fixed_source_or_pair_integrity_lost":
        return False
    if _safe_int(report.get("last_nfev", report.get("nfev")), default=0) <= 0:
        return False
    if not (
        math.isfinite(_metric_float(report.get("last_residual_norm", np.nan)))
        and math.isfinite(_metric_float(report.get("last_rms_px", np.nan)))
        and math.isfinite(_metric_float(report.get("last_max_error_px", np.nan)))
    ):
        return False
    if bool(report.get("fixed_source_counters_clean_at_last_heartbeat", True)) is not True:
        return False
    if bool(report.get("state_hash_unchanged", False)) is not True:
        return False
    if str(report.get("status", "")) == "timeout" and report.get("child_process_killed_cleanly") is not True:
        return False
    return True


def _single_attempt_report(result: Mapping[str, object]) -> Mapping[str, object]:
    reports = result.get("reports", [])
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        for report in reversed(reports):
            if not isinstance(report, Mapping):
                continue
            if int(report.get("rung", -1) or -1) == 3 and report.get("param_name"):
                return report
    return result


def _variant_stop_reason(
    result: Mapping[str, object],
    single_report: Mapping[str, object],
) -> str | None:
    if bool(single_report.get("dirty_timeout_abort", False)):
        return "dirty_timeout_abort"
    if str(single_report.get("diagnosis_classification", "")) == (
        "fixed_source_or_pair_integrity_lost"
    ):
        return "fixed_source_or_pair_integrity_lost"
    result_reason = str(result.get("failure_reason") or result.get("reason") or "")
    single_reason = str(
        single_report.get("failure_reason") or single_report.get("reason") or ""
    )
    reason = result_reason or single_reason
    if reason in {
        "filtered_param_not_active",
        "sensitivity_not_green",
        "provider_guard_failed",
        "objective_dry_run_failed",
        "no_active_params",
    }:
        return reason
    if (
        not single_report.get("param_name")
        and str(result.get("status", single_report.get("status", "")))
        in {"failed", "aborted"}
    ):
        return reason or str(result.get("status") or single_report.get("status"))
    return None


def run_one_param_diagnosis_variants(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    one_param_filter: str = "a",
    use_subprocess: bool = False,
    diagnostic_logging: bool = False,
) -> dict[str, object]:
    state_path = Path(state_path).expanduser().resolve()
    state_hash_before = _state_sha256(state_path) if state_path.is_file() else ""
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    attempted_variants: list[str] = []
    stopped_reason: str | None = None
    for index, (label, max_nfev, timeout_seconds) in enumerate(ONE_PARAM_A_VARIANTS):
        if index == 2 and not all(
            _variant_can_continue(_single_attempt_report(report)) for report in reports[:2]
        ):
            stopped_reason = "variant_c_conditions_not_met"
            break
        result = run_ladder(
            state_path=state_path,
            background_index=int(background_index),
            output_root=output_root,
            max_rung="one-param",
            timeout_seconds=float(timeout_seconds),
            max_nfev=int(max_nfev),
            timestamp=label,
            one_param_filter=str(one_param_filter),
            use_subprocess=bool(use_subprocess),
            diagnostic_logging=bool(diagnostic_logging),
        )
        reports.append(result)
        attempted_variants.append(label)
        single_report = _single_attempt_report(result)
        stopped_reason = _variant_stop_reason(result, single_report)
        if stopped_reason:
            break
    final_report = _single_attempt_report(reports[-1]) if reports else {}
    state_hash_after = _state_sha256(state_path) if state_path.is_file() else ""
    summary = {
        "status": (
            "failed"
            if stopped_reason
            and stopped_reason != "variant_c_conditions_not_met"
            else "ok"
        ),
        "failure_reason": stopped_reason,
        "one_param_filter": str(one_param_filter),
        "state_path": str(state_path),
        "background_index": int(background_index),
        "state_sha256_before": str(final_report.get("state_sha256_before") or state_hash_before),
        "state_sha256_after": str(final_report.get("state_sha256_after") or state_hash_after),
        "state_hash_unchanged": bool(
            str(final_report.get("state_sha256_before") or state_hash_before)
            == str(final_report.get("state_sha256_after") or state_hash_after)
        ),
        "dirty_timeout_abort": bool(final_report.get("dirty_timeout_abort", False)),
        "attempted_variants": attempted_variants,
        "diagnosis_classification": final_report.get("diagnosis_classification"),
        "reports": reports,
    }
    _write_json(output_root / "variant_summary.json", summary)
    return summary


def _worker_main(args: argparse.Namespace) -> int:
    active_names = [str(name) for name in json.loads(str(args.active_params_json))]
    _worker_solve_once(
        state_path=Path(args.state).expanduser().resolve(),
        background_index=int(args.background_index),
        active_names=active_names,
        output_path=Path(args.output_json).expanduser().resolve(),
        heartbeat_path=Path(args.heartbeat_json).expanduser().resolve(),
        max_nfev=int(args.max_nfev),
        rung=int(args.rung_number),
        rung_name=str(args.rung_name),
        feature=args.feature,
        diagnostic_logging=bool(getattr(args, "diagnostic_logging", False)),
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded new4 geometry-fit optimizer ladder.",
    )
    parser.add_argument("--state", required=True, help="Path to saved GUI state JSON.")
    parser.add_argument("--background-index", required=True, type=int)
    parser.add_argument("--output-root", help="Root directory for ladder artifacts.")
    parser.add_argument(
        "--max-rung",
        choices=(
            "sensitivity",
            "one-param",
            "pair",
            "pairs",
            "block",
            "blocks",
            "center",
            "full",
            "features",
        ),
        default="center",
    )
    parser.add_argument(
        "--sensitivity-report",
        help="Explicit debug override for Rung 2 sensitivity JSON.",
    )
    parser.add_argument(
        "--one-param-filter",
        help="Run only this current-run active parameter when --max-rung=one-param.",
    )
    parser.add_argument(
        "--one-param-summary",
        help="Current Rung 3 one-param summary JSON for pair rungs.",
    )
    parser.add_argument(
        "--one-param-diagnosis-summary",
        help="Current Rung 3A one-param diagnosis variant summary JSON for pair rungs.",
    )
    parser.add_argument(
        "--caked-point-reprojection-report",
        help="Fresh Rung 3B caked-point reprojection JSON required for theta/distance pairs.",
    )
    parser.add_argument(
        "--pair-summary",
        help="Debug-only Rung 4 pair summary JSON override for block rungs.",
    )
    parser.add_argument(
        "--run-one-param-variants",
        action="store_true",
        help="Run the bounded A/B/C one-param diagnostic sequence.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument(
        "--timestamp",
        help=(
            "Debug run directory name. Use with --pair-summary when strict "
            "run_id/timestamp evidence must match."
        ),
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Run each solve rung in a separate cold worker process.",
    )
    parser.add_argument(
        "--diagnostic-logging",
        action="store_true",
        help="Keep debug and intersection-cache logging enabled during solver probes.",
    )
    parser.add_argument(
        "--timing-report",
        help="Optional path for a copy of the current-run rung timing summary JSON.",
    )
    parser.add_argument("--_worker-solve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat-json", help=argparse.SUPPRESS)
    parser.add_argument("--active-params-json", help=argparse.SUPPRESS)
    parser.add_argument("--rung-number", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rung-name", help=argparse.SUPPRESS)
    parser.add_argument("--feature", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args._worker_solve:
        return _worker_main(args)
    if not args.output_root:
        parser.error("--output-root is required.")
    if bool(getattr(args, "run_one_param_variants", False)):
        result = run_one_param_diagnosis_variants(
            state_path=Path(args.state),
            background_index=int(args.background_index),
            output_root=Path(args.output_root),
            one_param_filter=str(args.one_param_filter or "a"),
            use_subprocess=bool(args.use_subprocess),
            diagnostic_logging=bool(args.diagnostic_logging),
        )
    else:
        result = run_ladder(
            state_path=Path(args.state),
            background_index=int(args.background_index),
            output_root=Path(args.output_root),
            max_rung=str(args.max_rung),
            timeout_seconds=float(args.timeout_seconds),
            max_nfev=int(args.max_nfev),
            timestamp=str(args.timestamp or "") or None,
            sensitivity_report=Path(args.sensitivity_report) if args.sensitivity_report else None,
            one_param_filter=str(args.one_param_filter or "") or None,
            one_param_summary=(
                Path(args.one_param_summary) if args.one_param_summary else None
            ),
            one_param_diagnosis_summary=(
                Path(args.one_param_diagnosis_summary)
                if args.one_param_diagnosis_summary
                else None
            ),
            caked_point_reprojection_report=(
                Path(args.caked_point_reprojection_report)
                if args.caked_point_reprojection_report
                else None
            ),
            pair_summary=Path(args.pair_summary) if args.pair_summary else None,
            timing_report=Path(args.timing_report) if args.timing_report else None,
            use_subprocess=bool(args.use_subprocess),
            diagnostic_logging=bool(args.diagnostic_logging),
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "failure_reason": result.get("failure_reason"),
                "run_dir": result.get("run_dir"),
                "state_unchanged": result.get("state_unchanged"),
            },
            sort_keys=True,
        )
    )
    timing_summary_path = result.get("timing_summary_path")
    if timing_summary_path:
        timing_summary = _read_json(Path(str(timing_summary_path)))
        if timing_summary:
            print(_format_timing_table(timing_summary))
    return 0 if str(result.get("status")) in {"pass", "ok", "ok_with_failures", "stopped", "aborted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
