"""Run real RA-SIM GUI timing trials and summarize JSONL timing events."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TRUE_VALUES = {"1", "true", "yes", "on"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "perf" / "gui_timing"
DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "geometry_fit_gui_states" / "new4.json"


@dataclass(frozen=True)
class RunSpec:
    trial: int
    trial_group: str
    repetitions: int
    artifact_stem: str
    measured: bool = True


@dataclass(frozen=True)
class TrialResult:
    path: Path
    returncode: int | None
    timed_out: bool
    termination: str
    child_pid: int
    measured: bool = True


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _event_record(event: str, **fields: object) -> dict[str, object]:
    record: dict[str, object] = {
        "event": event,
        "perf_counter_ns": time.perf_counter_ns(),
        "wall_time_iso": datetime.now().astimezone().isoformat(timespec="microseconds"),
        "pid": os.getpid(),
        "thread_id": None,
        "scenario_id": fields.pop("scenario_id", None),
        "trial_id": fields.pop("trial_id", None),
        "update_id": fields.pop("update_id", None),
        "phase": fields.pop("phase", None),
    }
    record.update(fields)
    return record


def _append_event(path: Path, event: str, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_event_record(event, **fields), sort_keys=True) + "\n")


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _first_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == name:
            return event
    return None


def _last_event(events: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == name:
            return event
    return None


def _event_ns(event: dict[str, Any] | None) -> int | None:
    if event is None:
        return None
    try:
        return int(event["perf_counter_ns"])
    except Exception:
        return None


def _first_event_at_or_after(
    events: list[dict[str, Any]], name: str, minimum_ns: int
) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") != name:
            continue
        event_ns = _event_ns(event)
        if event_ns is not None and event_ns >= minimum_ns:
            return event
    return None


def _last_event_at_or_before(
    events: list[dict[str, Any]], name: str, maximum_ns: int
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") != name:
            continue
        event_ns = _event_ns(event)
        if event_ns is not None and event_ns <= maximum_ns:
            return event
    return None


def _duration_ms(
    events: list[dict[str, Any]],
    start_name: str,
    end_name: str,
    *,
    pairing: str = "first_after_start",
) -> float | None:
    if pairing == "last_before_end":
        end = _first_event(events, end_name)
        end_ns = _event_ns(end)
        if end is None or end_ns is None:
            return None
        start = _last_event_at_or_before(events, start_name, end_ns)
    else:
        start = _first_event(events, start_name)
        start_ns = _event_ns(start)
        if start is None or start_ns is None:
            return None
        end = _first_event_at_or_after(events, end_name, start_ns)
    start_ns = _event_ns(start)
    end_ns = _event_ns(end)
    if start_ns is None or end_ns is None:
        return None
    duration_ns = end_ns - start_ns
    if duration_ns < 0:
        return None
    return duration_ns / 1_000_000.0


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "trial_count": 0,
            "median_ms": None,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p95_ms": None,
            "std_ms": None,
        }
    ordered = sorted(clean)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "trial_count": len(clean),
        "median_ms": statistics.median(clean),
        "mean_ms": statistics.mean(clean),
        "min_ms": min(clean),
        "max_ms": max(clean),
        "p95_ms": ordered[p95_index],
        "std_ms": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def _format_ms(value: object) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.3f}"


def _redact_path(value: object) -> object:
    if isinstance(value, (str, Path)):
        text = str(value)
        if "\\" in text or "/" in text:
            return Path(text).name
    return value


def _detect_ram_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except Exception:
        return None


def _detect_display() -> dict[str, object]:
    display: dict[str, object] = {
        "width_px": "not detected",
        "height_px": "not detected",
        "scaling": "not detected",
        "refresh_rate_hz": "not detected",
    }
    if os.name != "nt":
        return display
    try:
        import ctypes

        display["width_px"] = int(ctypes.windll.user32.GetSystemMetrics(0))
        display["height_px"] = int(ctypes.windll.user32.GetSystemMetrics(1))
        try:
            dpi = int(ctypes.windll.user32.GetDpiForSystem())
            display["scaling"] = round(dpi / 96.0, 3)
        except Exception:
            pass
    except Exception:
        pass
    return display


def _detect_tk_info() -> dict[str, object]:
    try:
        import tkinter as tk

        return {
            "tk_version": str(tk.TkVersion),
            "tcl_version": str(tk.TclVersion),
        }
    except Exception:
        return {"tk_version": "not detected", "tcl_version": "not detected"}


def _detect_matplotlib_info() -> dict[str, object]:
    try:
        import matplotlib

        return {
            "version": str(getattr(matplotlib, "__version__", "not detected")),
            "backend": str(matplotlib.get_backend()),
        }
    except Exception:
        return {"version": "not detected", "backend": "not detected"}


def _detect_config_dir_basename() -> str:
    try:
        from ra_sim.config import get_config_dir

        return str(_redact_path(get_config_dir()))
    except Exception:
        return "not detected"


def _child_runtime_metadata_event(
    trial_events: list[list[dict[str, Any]]],
) -> dict[str, Any] | None:
    for events in trial_events:
        for event in events:
            if event.get("event") == "gui.runtime.metadata":
                return event
    return None


def _merge_child_runtime_metadata(
    metadata: dict[str, object],
    trial_events: list[list[dict[str, Any]]],
) -> dict[str, object]:
    merged = dict(metadata)
    child_event = _child_runtime_metadata_event(trial_events)
    if child_event is None:
        merged.setdefault("backend", "not detected")
        merged.setdefault("child_gui", {})
        return merged

    standard_fields = {
        "event",
        "perf_counter_ns",
        "wall_time_iso",
        "pid",
        "thread_id",
        "scenario_id",
        "trial_id",
        "update_id",
        "phase",
        "rss_bytes",
    }
    child_gui = {
        str(key): value for key, value in child_event.items() if str(key) not in standard_fields
    }
    backend = child_event.get("matplotlib_backend") or child_event.get("backend")
    if backend:
        merged["backend"] = str(backend)
    merged["child_gui"] = child_gui
    merged["matplotlib"] = {
        "backend": str(child_event.get("matplotlib_backend", "not detected")),
        "version": str(child_event.get("matplotlib_version", "not detected")),
    }
    merged["tk"] = {
        "tk_version": str(child_event.get("tk_version", "not detected")),
        "tcl_version": str(child_event.get("tcl_version", "not detected")),
    }
    if child_event.get("root_geometry"):
        merged["window_geometry"] = child_event.get("root_geometry")
    if child_event.get("display_width_px") or child_event.get("display_height_px"):
        parent_display = dict(merged.get("display", {}) or {})
        parent_display.update(
            {
                "child_width_px": child_event.get("display_width_px", "not detected"),
                "child_height_px": child_event.get("display_height_px", "not detected"),
                "child_scaling": child_event.get("display_scaling", "not detected"),
            }
        )
        merged["display"] = parent_display
    return merged


def _metadata(output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    def _git(*git_args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *git_args],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo": REPO_ROOT.name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "os": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _detect_ram_bytes() or "not detected",
        "display": _detect_display(),
        "scenario": args.scenario,
        "trials": int(args.trials),
        "state_file": _redact_path(args.state) if args.state else None,
        "output_dir": output_dir.name,
        "parent_tk_probe": _detect_tk_info(),
        "parent_matplotlib_probe": _detect_matplotlib_info(),
        "backend": "not detected",
        "child_gui": {},
        "ra_sim_config_dir": _detect_config_dir_basename(),
        "window_geometry": "reported by child events when available; fixed minsize 1200x760",
        "numba_cache_warm": "reported by child events when available",
        "first_run_jit_compile": "reported by child events when available",
        "power_mode": "not detected",
        "paths_redacted": True,
        "startup_grouping": (
            "startup trials 1-5=fresh_process; trials 6+=warm_fresh_process only after "
            "warmup_001 completes"
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _scenario_launch_count(scenario: str, trials: int) -> tuple[int, int]:
    normalized = scenario.lower().replace("_", "-")
    if normalized in {"theta10", "redraw-only", "cache-hit"}:
        return 1, int(trials)
    return int(trials), 1


def _trial_group(scenario: str, trial: int, warmup_completed: bool = False) -> str:
    normalized = scenario.lower().replace("_", "-")
    if normalized in {"defaults", "defaults-restored"}:
        if warmup_completed and trial > 5:
            return "warm_fresh_process"
        return "fresh_process"
    return "fresh_process"


def _build_run_specs(scenario: str, launch_count: int, repetitions: int) -> list[RunSpec]:
    normalized = scenario.lower().replace("_", "-")
    specs: list[RunSpec] = []
    warmup_added = False
    for trial in range(1, launch_count + 1):
        if normalized in {"defaults", "defaults-restored"} and trial == 6:
            specs.append(
                RunSpec(
                    trial=0,
                    trial_group="warmup_process",
                    repetitions=1,
                    artifact_stem="warmup_001",
                    measured=False,
                )
            )
            warmup_added = True
        specs.append(
            RunSpec(
                trial=trial,
                trial_group=_trial_group(
                    scenario,
                    trial,
                    warmup_completed=warmup_added,
                )
                if not (launch_count == 1 and normalized not in {"defaults", "defaults-restored"})
                else "steady_state_process",
                repetitions=repetitions,
                artifact_stem=f"trial_{trial:03d}",
                measured=True,
            )
        )
    return specs


def _run_failed(result: TrialResult) -> bool:
    return bool(result.timed_out) or result.returncode != 0


def _harness_exit_code(results: list[TrialResult]) -> int:
    return 1 if any(_run_failed(result) for result in results) else 0


def _trial_results_summary(results: list[TrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.append(
            {
                "artifact": result.path.name,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "termination": result.termination,
                "child_pid": result.child_pid,
                "measured": result.measured,
            }
        )
    return rows


def _trial_failures_summary(results: list[TrialResult]) -> list[dict[str, object]]:
    return [
        row for row in _trial_results_summary(results) if row["timed_out"] or row["returncode"] != 0
    ]


def _run_specs(
    *,
    output_dir: Path,
    scenario: str,
    run_specs: list[RunSpec],
    state: Path | None,
    timeout_s: float,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    warmup_failed = False
    for spec in run_specs:
        if warmup_failed and spec.trial_group == "warm_fresh_process":
            break
        result = _run_trial(
            output_dir=output_dir,
            scenario=scenario,
            spec=spec,
            state=state,
            timeout_s=timeout_s,
        )
        if spec.trial_group == "warmup_process" and _run_failed(result):
            warmup_failed = True
        results.append(result)
    return results


def _child_env(
    *,
    base_env: dict[str, str],
    trial_jsonl: Path,
    scenario: str,
    trial: int,
    trial_group: str,
    repetitions: int,
    state: Path | None,
) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "RA_SIM_TIMING": "1",
            "RA_SIM_TIMING_OUT": str(trial_jsonl),
            "RA_SIM_TIMING_SCENARIO": scenario,
            "RA_SIM_TIMING_TRIAL": str(trial),
            "RA_SIM_TIMING_TRIAL_GROUP": trial_group,
            "RA_SIM_TIMING_AUTOMATION": "1",
            "RA_SIM_TIMING_REPETITIONS": str(repetitions),
            "RA_SIM_FORCE_EXIT_ON_GUI_CLOSE": "1",
            "RA_SIM_EARLY_STARTUP_MODE": "simulation",
        }
    )
    if state is not None:
        env["RA_SIM_TIMING_RESTORE_STATE"] = str(state)
    return env


def _run_trial(
    *,
    output_dir: Path,
    scenario: str,
    spec: RunSpec,
    state: Path | None,
    timeout_s: float,
) -> TrialResult:
    trial_jsonl = output_dir / f"{spec.artifact_stem}.jsonl"
    env = _child_env(
        base_env=dict(os.environ),
        trial_jsonl=trial_jsonl,
        scenario=scenario,
        trial=spec.trial,
        trial_group=spec.trial_group,
        repetitions=spec.repetitions,
        state=state,
    )
    command = [sys.executable, "-m", "ra_sim", "gui"]
    _append_event(
        trial_jsonl,
        "process.launch.start",
        phase="process",
        scenario_id=scenario,
        trial_id=spec.trial,
        trial_group=spec.trial_group,
        measured=spec.measured,
        command="python -m ra_sim gui",
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _append_event(
        trial_jsonl,
        "process.launch.returned",
        phase="process",
        scenario_id=scenario,
        trial_id=spec.trial,
        trial_group=spec.trial_group,
        measured=spec.measured,
        child_pid=process.pid,
    )
    deadline = time.monotonic() + float(timeout_s)
    timed_out = False
    termination = "normal"
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if process.poll() is None:
        timed_out = True
        try:
            termination = "terminate"
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            termination = "kill"
            process.kill()
            try:
                process.wait(timeout=10)
            except Exception:
                pass
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination = "kill"
        process.kill()
        stdout, stderr = process.communicate()
    (output_dir / f"{spec.artifact_stem}.stdout.txt").write_text(stdout or "", encoding="utf-8")
    (output_dir / f"{spec.artifact_stem}.stderr.txt").write_text(stderr or "", encoding="utf-8")
    _append_event(
        trial_jsonl,
        "process.exit",
        phase="process",
        scenario_id=scenario,
        trial_id=spec.trial,
        trial_group=spec.trial_group,
        measured=spec.measured,
        returncode=process.returncode,
        timed_out=timed_out,
        termination=termination,
    )
    return TrialResult(
        path=trial_jsonl,
        returncode=process.returncode,
        timed_out=timed_out,
        termination=termination,
        child_pid=int(process.pid),
        measured=spec.measured,
    )


STARTUP_METRICS = {
    "process launch to first visible": ("process.launch.start", "first_tk.after_idle.visible"),
    "process launch overhead": ("process.launch.start", "process.launch.returned"),
    "python entry to GUI main start": ("python.entry", "gui.main.start"),
    "runtime import": ("gui.runtime.import.start", "gui.runtime_session.import.end"),
    "config load": ("config.load.start", "config.load.end"),
    "first background load": ("background.first_load.start", "background.first_load.end"),
    "first simulation compute": (
        "first_simulation.compute.start",
        "first_simulation.compute.end",
    ),
    "first GUI render after compute": (
        "first_gui_render_after_compute.start",
        "first_gui_render_after_compute.end",
    ),
    "draw event to Tk idle": ("first_canvas.draw_event", "first_tk.after_idle.visible"),
}


THETA_METRICS = {
    "input to var write": ("theta_change.input.start", "theta_change.var_write"),
    "variable write to update scheduled": (
        "theta_change.var_write",
        "theta_change.update_scheduled",
    ),
    "update queue delay": ("theta_change.queue_delay.start", "theta_change.update_begin"),
    "simulation calculation only": ("theta_change.compute.start", "theta_change.compute.end"),
    "GUI update after calculation": (
        "theta_change.compute.end",
        "theta_change.after_idle.visible",
    ),
    "result ready to canvas draw complete": (
        "theta_change.result_apply.end",
        "theta_change.canvas_draw.end",
    ),
    "canvas draw complete to Tk idle visible": (
        "theta_change.canvas_draw.end",
        "theta_change.after_idle.visible",
    ),
    "total change to visible": ("theta_change.input.start", "theta_change.after_idle.visible"),
    "total excluding automated typing": (
        "theta_change.var_write",
        "theta_change.after_idle.visible",
    ),
}


REDRAW_METRICS = {
    "input to visible": ("redraw_only.input.start", "redraw_only.after_idle.visible"),
    "rasterize": ("redraw_only.rasterize.start", "redraw_only.rasterize.end"),
    "display set_data": (
        "redraw_only.display_set_data.start",
        "redraw_only.display_set_data.end",
    ),
    "canvas draw": ("redraw_only.canvas_draw.start", "redraw_only.canvas_draw.end"),
    "canvas draw complete to Tk idle visible": (
        "redraw_only.canvas_draw.end",
        "redraw_only.after_idle.visible",
    ),
}


METRIC_PAIRING = {
    "draw event to Tk idle": "last_before_end",
    "canvas draw complete to Tk idle visible": "last_before_end",
}

SAVED_STATE_BREAKDOWN_ORDER = [
    "total process launch to first visible",
    "saved-state file read",
    "saved-state parse/import",
    "variable restore",
    "background load",
    "PONI parse/restore",
    "CIF read/restore",
    "manual geometry restore",
    "caked/overlay restore",
    "first simulation compute",
    "post-compute render",
    "draw_event to after_idle visible",
    "unexplained saved-state startup gap",
]

SAVED_STATE_NON_OVERLAP_CHILD_ROWS = [
    "saved-state file read",
    "saved-state parse/import",
    "variable restore",
    "background load",
    "PONI parse/restore",
    "CIF read/restore",
    "manual geometry restore",
    "caked/overlay restore",
    "first simulation compute",
    "post-compute render",
    "draw_event to after_idle visible",
]

STARTUP_TIMELINE_ORDER = [
    "process.launch.start -> python.entry",
    "python.entry -> gui.command.received",
    "gui.command.received -> gui.runtime.import.start",
    "gui.runtime.import.start -> gui.runtime.import.end",
    "gui.runtime.import.end -> gui.main.start",
    "gui.main.start -> tk.root.create.start",
    "tk.root.create.start -> tk.root.create.end",
    "tk.root.create.end -> widgets.build.start",
    "widgets.build.start -> widgets.build.end",
    "widgets.build.end -> saved_state.start",
    "saved_state.start -> saved_state.end",
    "saved_state.end -> first_update.requested",
    "first_update.requested -> first_update.update_begin",
    "first_update.update_begin -> first_simulation.compute.start",
    "first_simulation.compute.end -> first_gui_render_after_compute.start",
    "first_gui_render_after_compute.end -> first_canvas.draw_event",
    "first_canvas.draw_event -> first_tk.after_idle.visible",
]

STARTUP_TIMELINE_ROWS = [(metric, *metric.split(" -> ")) for metric in STARTUP_TIMELINE_ORDER]

STARTUP_TIMELINE_UPDATE_EVENTS = {
    "first_update.update_begin",
    "first_simulation.compute.start",
    "first_simulation.compute.end",
    "first_gui_render_after_compute.start",
    "first_gui_render_after_compute.end",
    "first_canvas.draw_event",
    "first_tk.after_idle.visible",
}

STARTUP_TIMELINE_STARTUP_ONLY_EVENTS = {
    "first_update.requested",
}

FIRST_SIMULATION_COMPUTE_ORDER = [
    "parameter_collection",
    "signature_build",
    "cache_lookup",
    "beam_sample_generation",
    "structure_factor_prepare",
    "q_or_rod_solving",
    "peak_processing",
    "worker_wait_or_sync_compute",
    "kernel_call",
    "image_accumulation",
    "rasterize",
    "normalization",
    "result_ready",
]

FIRST_SIMULATION_COMPUTE_EVENT_PAIRS = {
    "beam_sample_generation": (
        "first_simulation.beam_sample_generation.start",
        "first_simulation.beam_sample_generation.end",
    ),
    "kernel_call": (
        "first_simulation.kernel_call.start",
        "first_simulation.kernel_call.end",
    ),
    "result_ready": (
        "first_simulation.result_ready.start",
        "first_simulation.result_ready.end",
    ),
}

FIRST_SIMULATION_COMPUTE_OUTSIDE_ROWS = {
    "parameter_collection": "outside_first_simulation_compute_interval",
    "signature_build": "outside_first_simulation_compute_interval",
    "cache_lookup": "outside_first_simulation_compute_interval",
    "worker_wait_or_sync_compute": "synchronous_first_compute_has_no_exclusive_worker_wait_span",
    "normalization": "outside_first_simulation_compute_interval",
}

FIRST_SIMULATION_COMPUTE_KERNEL_INTERNAL_ROWS = {
    "structure_factor_prepare",
    "q_or_rod_solving",
    "peak_processing",
    "image_accumulation",
    "rasterize",
}

FIRST_SIMULATION_COMPUTE_TOTAL_ROWS = {
    "beam_sample_generation",
    "kernel_call",
    "result_ready",
}

FIRST_SIMULATION_WORKER_HANDOFF_ORDER = [
    "worker_queue_wait_ms",
    "worker_submit_to_start_ms",
    "worker_start_to_kernel_start_ms",
    "kernel_call_ms",
    "kernel_end_to_result_ready_ms",
    "result_ready_to_gui_thread_ms",
    "gui_thread_result_apply_ms",
    "worker_result_transfer_ms",
]

FIRST_SIMULATION_WORKER_HANDOFF_CATEGORY = {
    "worker_queue_wait_ms": "queue/scheduling",
    "worker_submit_to_start_ms": "queue/scheduling",
    "worker_start_to_kernel_start_ms": "worker start before kernel",
    "kernel_end_to_result_ready_ms": "post-kernel result packaging",
    "result_ready_to_gui_thread_ms": "GUI-thread scheduling",
    "gui_thread_result_apply_ms": "GUI-thread application",
    "worker_result_transfer_ms": "result transfer/future fetch",
}


def _metric_pairing(metric: str) -> str:
    return METRIC_PAIRING.get(metric, "first_after_start")


def _safe_basename(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).name


def _event_has_name(event: dict[str, Any], name: str) -> bool:
    return str(event.get("event", "")) == str(name)


def _events_named(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if _event_has_name(event, name)]


@dataclass(frozen=True)
class SpanPairingResult:
    intervals: list[tuple[int, int]]
    start_count: int
    end_count: int
    used_start_count: int
    used_end_count: int


def _span_pairing_result_ns(
    events: list[dict[str, Any]],
    start_name: str,
    end_name: str,
) -> SpanPairingResult:
    starts = [
        event
        for event in events
        if _event_has_name(event, start_name) and _event_ns(event) is not None
    ]
    ends = [
        event
        for event in events
        if _event_has_name(event, end_name) and _event_ns(event) is not None
    ]
    intervals: list[tuple[int, int]] = []
    used_start_indexes: set[int] = set()
    used_end_indexes: set[int] = set()
    for start_index, start in enumerate(starts):
        start_ns = _event_ns(start)
        if start_ns is None:
            continue
        for index, end in enumerate(ends):
            if index in used_end_indexes:
                continue
            end_ns = _event_ns(end)
            if end_ns is None or end_ns < start_ns:
                continue
            used_start_indexes.add(start_index)
            used_end_indexes.add(index)
            intervals.append((start_ns, end_ns))
            break
    return SpanPairingResult(
        intervals=intervals,
        start_count=len(starts),
        end_count=len(ends),
        used_start_count=len(used_start_indexes),
        used_end_count=len(used_end_indexes),
    )


def _duration_from_intervals_ms(intervals: list[tuple[int, int]]) -> float:
    return sum((end_ns - start_ns) / 1_000_000.0 for start_ns, end_ns in intervals)


def _non_overlapping_interval_status(
    intervals: list[tuple[int, int]],
) -> tuple[bool, str | None]:
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            return False, "overlapping_child_spans"
    return True, None


def _phase_result(
    *,
    status: str,
    duration_ms: float | None = None,
    intervals_ns: list[tuple[int, int]] | None = None,
    reason: str | None = None,
    correlation_status: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "phase_status": status,
        "duration_ms": duration_ms,
        "missing_phase_reason": reason,
        "_intervals_ns": list(intervals_ns or []),
    }
    if correlation_status is not None:
        result["correlation_status"] = correlation_status
    return result


def _phase_from_spans(
    events: list[dict[str, Any]],
    spans: list[tuple[str, str]],
    *,
    not_applicable_reason: str | None = None,
) -> dict[str, object]:
    intervals: list[tuple[int, int]] = []
    missing: list[str] = []
    observed_partial = False
    for start_name, end_name in spans:
        pairing = _span_pairing_result_ns(events, start_name, end_name)
        has_events = bool(pairing.start_count or pairing.end_count)
        has_orphan_boundaries = (
            pairing.used_start_count != pairing.start_count
            or pairing.used_end_count != pairing.end_count
        )
        if pairing.intervals and not has_orphan_boundaries:
            intervals.extend(pairing.intervals)
            continue
        if has_events:
            missing.append(f"incomplete_span:{start_name}->{end_name}")
            observed_partial = True
        else:
            missing.append(f"missing_span:{start_name}->{end_name}")
    if not intervals:
        reason = (
            ";".join(missing) if observed_partial else not_applicable_reason or ";".join(missing)
        )
        return _phase_result(
            status=(
                "not_detected"
                if observed_partial
                else "not_applicable"
                if not_applicable_reason
                else "not_detected"
            ),
            reason=reason,
        )
    ok, overlap_reason = _non_overlapping_interval_status(intervals)
    if not ok:
        return _phase_result(
            status="not_detected",
            intervals_ns=intervals,
            reason=overlap_reason,
        )
    if missing:
        return _phase_result(
            status="not_detected",
            reason=";".join(missing) if missing else "partial_phase_spans",
        )
    return _phase_result(
        status="measured",
        duration_ms=_duration_from_intervals_ms(intervals),
        intervals_ns=intervals,
    )


def _single_interval_phase(
    events: list[dict[str, Any]],
    start_name: str,
    end_name: str,
) -> dict[str, object]:
    return _phase_from_spans(events, [(start_name, end_name)])


def _event_with_update(
    events: list[dict[str, Any]],
    name: str,
    update_id: object,
    *,
    maximum_ns: int | None = None,
    reverse: bool = False,
) -> dict[str, Any] | None:
    candidates = reversed(events) if reverse else events
    for event in candidates:
        if not _event_has_name(event, name):
            continue
        if event.get("update_id") != update_id:
            continue
        event_ns = _event_ns(event)
        if event_ns is None:
            continue
        if maximum_ns is not None and event_ns > maximum_ns:
            continue
        return event
    return None


def _has_uncorrelated_events(
    events: list[dict[str, Any]],
    names: tuple[str, ...],
    update_id: object,
) -> bool:
    for event in events:
        if str(event.get("event", "")) not in names:
            continue
        if event.get("update_id") != update_id:
            return True
    return False


def _correlated_update_id(events: list[dict[str, Any]]) -> tuple[object | None, str | None]:
    visible = _first_event(events, "first_tk.after_idle.visible")
    if visible is None:
        return None, "first_visible_not_detected"
    update_id = visible.get("update_id")
    if update_id is None:
        return None, "first_visible_update_id_missing"
    return update_id, None


def _correlated_span_phase(
    events: list[dict[str, Any]],
    start_name: str,
    end_name: str,
    update_id: object | None,
    *,
    visible_end: bool = False,
) -> dict[str, object]:
    if update_id is None:
        return _phase_result(
            status="not_detected",
            reason="correlated_update_id_missing",
            correlation_status="uncorrelated",
        )
    if visible_end:
        visible = _first_event(events, end_name)
        visible_ns = _event_ns(visible)
        end_ns = visible_ns if visible and visible.get("update_id") == update_id else None
        if end_ns is None:
            reason = f"{end_name}_not_detected"
            if _has_uncorrelated_events(events, (end_name,), update_id):
                reason = "uncorrelated_update_id"
            return _phase_result(
                status="not_detected",
                reason=reason,
                correlation_status="uncorrelated" if reason == "uncorrelated_update_id" else None,
            )
        start_event = _last_event_at_or_before(events, start_name, end_ns)
        start_ns = _event_ns(start_event)
        if start_ns is None:
            reason = f"{start_name}_not_detected"
            if _has_uncorrelated_events(events, (start_name,), update_id):
                reason = "uncorrelated_update_id"
            return _phase_result(
                status="not_detected",
                reason=reason,
                correlation_status="uncorrelated" if reason == "uncorrelated_update_id" else None,
            )
        if start_event and start_event.get("update_id") != update_id:
            return _phase_result(
                status="not_detected",
                reason="uncorrelated_update_id",
                correlation_status="uncorrelated",
            )
        return _phase_result(
            status="measured",
            duration_ms=(end_ns - start_ns) / 1_000_000.0,
            intervals_ns=[(start_ns, end_ns)],
            correlation_status="correlated",
        )
    names = (start_name, end_name)
    if _has_uncorrelated_events(events, names, update_id):
        return _phase_result(
            status="not_detected",
            reason="uncorrelated_update_id",
            correlation_status="uncorrelated",
        )
    correlated_events = [
        event
        for event in events
        if _event_has_name(event, start_name) or _event_has_name(event, end_name)
        if event.get("update_id") == update_id
    ]
    pairing = _span_pairing_result_ns(correlated_events, start_name, end_name)
    if not pairing.intervals:
        reason = f"{end_name}_not_detected"
        if pairing.start_count and not pairing.end_count:
            reason = f"{end_name}_not_detected"
        elif pairing.end_count and not pairing.start_count:
            reason = f"{start_name}_not_detected"
        elif pairing.start_count or pairing.end_count:
            reason = f"phase_events_out_of_order:{start_name}->{end_name}"
        return _phase_result(
            status="not_detected",
            reason=reason,
            correlation_status="correlated",
        )
    if (
        pairing.used_start_count != pairing.start_count
        or pairing.used_end_count != pairing.end_count
    ):
        return _phase_result(
            status="not_detected",
            reason=f"incomplete_span:{start_name}->{end_name}",
            correlation_status="correlated",
        )
    ok, overlap_reason = _non_overlapping_interval_status(pairing.intervals)
    if not ok:
        return _phase_result(
            status="not_detected",
            intervals_ns=pairing.intervals,
            reason=overlap_reason,
            correlation_status="correlated",
        )
    return _phase_result(
        status="measured",
        duration_ms=_duration_from_intervals_ms(pairing.intervals),
        intervals_ns=pairing.intervals,
        correlation_status="correlated",
    )


def _saved_state_phase_results(events: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    update_id, update_reason = _correlated_update_id(events)
    results: dict[str, dict[str, object]] = {
        "total process launch to first visible": _single_interval_phase(
            events,
            "process.launch.start",
            "first_tk.after_idle.visible",
        ),
        "saved-state file read": _phase_from_spans(
            events,
            [("saved_state.file_read.start", "saved_state.file_read.end")],
        ),
        "saved-state parse/import": _phase_from_spans(
            events,
            [
                ("saved_state.json_parse.start", "saved_state.json_parse.end"),
                ("saved_state.snapshot_import.start", "saved_state.snapshot_import.end"),
            ],
        ),
        "variable restore": _phase_from_spans(
            events,
            [("saved_state.variable_restore.start", "saved_state.variable_restore.end")],
        ),
        "background load": _phase_from_spans(
            events,
            [("saved_state.background_load.start", "saved_state.background_load.end")],
        ),
        "PONI parse/restore": _phase_from_spans(
            events,
            [
                (
                    "saved_state.poni_restore_or_parse.start",
                    "saved_state.poni_restore_or_parse.end",
                )
            ],
            not_applicable_reason="saved_state_restore_does_not_parse_poni",
        ),
        "CIF read/restore": _phase_from_spans(
            events,
            [("saved_state.cif_restore_or_read.start", "saved_state.cif_restore_or_read.end")],
        ),
        "manual geometry restore": _phase_from_spans(
            events,
            [
                (
                    "saved_state.geometry_manual_pairs_restore.start",
                    "saved_state.geometry_manual_pairs_restore.end",
                ),
                (
                    "saved_state.manual_geometry_cache_restore.start",
                    "saved_state.manual_geometry_cache_restore.end",
                ),
            ],
        ),
        "caked/overlay restore": _phase_from_spans(
            events,
            [
                ("saved_state.caked_state_restore.start", "saved_state.caked_state_restore.end"),
                ("saved_state.overlay_restore.start", "saved_state.overlay_restore.end"),
            ],
        ),
        "first simulation compute": _correlated_span_phase(
            events,
            "first_simulation.compute.start",
            "first_simulation.compute.end",
            update_id,
        ),
        "post-compute render": _correlated_span_phase(
            events,
            "first_gui_render_after_compute.start",
            "first_gui_render_after_compute.end",
            update_id,
        ),
        "draw_event to after_idle visible": _correlated_span_phase(
            events,
            "first_canvas.draw_event",
            "first_tk.after_idle.visible",
            update_id,
            visible_end=True,
        ),
    }
    if update_reason:
        for metric in (
            "first simulation compute",
            "post-compute render",
            "draw_event to after_idle visible",
        ):
            results[metric]["missing_phase_reason"] = update_reason
            results[metric]["correlation_status"] = "uncorrelated"
    results["unexplained saved-state startup gap"] = _saved_state_unexplained_gap_result(results)
    return results


def _saved_state_unexplained_gap_result(
    phase_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    total_result = phase_results.get("total process launch to first visible", {})
    total_intervals = list(total_result.get("_intervals_ns") or [])
    if len(total_intervals) != 1:
        return _phase_result(
            status="not_detected",
            reason="total_first_visible_interval_not_detected",
        )
    total_start, total_end = total_intervals[0]
    child_intervals: list[tuple[int, int]] = []
    for metric in SAVED_STATE_NON_OVERLAP_CHILD_ROWS:
        result = phase_results.get(metric, {})
        if result.get("phase_status") != "measured":
            continue
        for start_ns, end_ns in list(result.get("_intervals_ns") or []):
            if start_ns < total_start or end_ns > total_end:
                return _phase_result(
                    status="not_detected",
                    reason=f"child_span_outside_total:{metric}",
                    intervals_ns=[(start_ns, end_ns)],
                )
            child_intervals.append((int(start_ns), int(end_ns)))
    ok, reason = _non_overlapping_interval_status(child_intervals)
    if not ok:
        return _phase_result(status="not_detected", reason=reason, intervals_ns=child_intervals)
    total_ms = (total_end - total_start) / 1_000_000.0
    measured_ms = _duration_from_intervals_ms(child_intervals)
    gap_ms = total_ms - measured_ms
    if gap_ms < -0.001:
        return _phase_result(
            status="not_detected",
            reason="measured_child_spans_exceed_total",
            intervals_ns=child_intervals,
        )
    return _phase_result(
        status="measured",
        duration_ms=max(0.0, gap_ms),
        intervals_ns=[],
    )


def _timeline_update_required(event_name: str) -> bool:
    return event_name in STARTUP_TIMELINE_UPDATE_EVENTS


def _timeline_startup_only(event_name: str) -> bool:
    return event_name in STARTUP_TIMELINE_STARTUP_ONLY_EVENTS


def _timeline_boundary_event(
    events: list[dict[str, Any]],
    name: str,
    *,
    minimum_ns: int | None,
    update_id: object | None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if _timeline_update_required(name) and update_id is None:
        return None, "uncorrelated", "first_visible_update_id_missing"
    saw_name = False
    saw_uncorrelated = False
    saw_wrong_update_id = False
    saw_before_minimum = False
    for event in events:
        if not _event_has_name(event, name):
            continue
        event_ns = _event_ns(event)
        if event_ns is None:
            continue
        saw_name = True
        if minimum_ns is not None and event_ns < minimum_ns:
            saw_before_minimum = True
            continue
        if _timeline_startup_only(name) and event.get("update_id") is not None:
            saw_wrong_update_id = True
            continue
        if _timeline_update_required(name) and event.get("update_id") != update_id:
            saw_uncorrelated = True
            continue
        return event, None, None
    if saw_uncorrelated:
        return None, "uncorrelated", f"uncorrelated_boundary:{name}"
    if saw_wrong_update_id:
        return None, "not_detected", f"boundary_requires_no_update_id:{name}"
    if saw_name and saw_before_minimum:
        return None, "not_detected", f"boundary_out_of_order:{name}"
    return None, "not_detected", f"missing_boundary:{name}"


def _startup_timeline_row_result(
    events: list[dict[str, Any]],
    metric: str,
    start_name: str,
    end_name: str,
    *,
    minimum_ns: int | None,
    update_id: object | None,
) -> dict[str, object]:
    start_event, start_status, start_reason = _timeline_boundary_event(
        events,
        start_name,
        minimum_ns=minimum_ns,
        update_id=update_id,
    )
    if start_event is None:
        return _phase_result(
            status=str(start_status or "not_detected"),
            reason=start_reason,
            correlation_status="uncorrelated" if start_status == "uncorrelated" else None,
        ) | {"start_event": start_name, "end_event": end_name, "metric": metric}
    start_ns = _event_ns(start_event)
    end_event, end_status, end_reason = _timeline_boundary_event(
        events,
        end_name,
        minimum_ns=start_ns,
        update_id=update_id,
    )
    if end_event is None:
        return _phase_result(
            status=str(end_status or "not_detected"),
            reason=end_reason,
            correlation_status="uncorrelated" if end_status == "uncorrelated" else None,
        ) | {"start_event": start_name, "end_event": end_name, "metric": metric}
    end_ns = _event_ns(end_event)
    if start_ns is None or end_ns is None:
        return _phase_result(
            status="not_detected",
            reason=f"missing_boundary_timestamp:{start_name}->{end_name}",
        ) | {"start_event": start_name, "end_event": end_name, "metric": metric}
    if end_ns < start_ns:
        return _phase_result(
            status="not_detected",
            reason=f"boundary_out_of_order:{start_name}->{end_name}",
        ) | {"start_event": start_name, "end_event": end_name, "metric": metric}
    return _phase_result(
        status="measured",
        duration_ms=(end_ns - start_ns) / 1_000_000.0,
        intervals_ns=[(start_ns, end_ns)],
        correlation_status=(
            "correlated"
            if _timeline_update_required(start_name) or _timeline_update_required(end_name)
            else None
        ),
    ) | {"start_event": start_name, "end_event": end_name, "metric": metric}


def _startup_timeline_results(events: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    update_id, update_reason = _correlated_update_id(events)
    results: dict[str, dict[str, object]] = {}
    previous_end_ns: int | None = None
    for metric, start_name, end_name in STARTUP_TIMELINE_ROWS:
        result = _startup_timeline_row_result(
            events,
            metric,
            start_name,
            end_name,
            minimum_ns=previous_end_ns,
            update_id=update_id,
        )
        intervals = list(result.get("_intervals_ns") or [])
        if result.get("phase_status") == "measured" and intervals:
            start_ns, end_ns = intervals[0]
            if previous_end_ns is not None and int(start_ns) < previous_end_ns:
                result = _phase_result(
                    status="overlap_error",
                    intervals_ns=[(int(start_ns), int(end_ns))],
                    reason="timeline_interval_overlap",
                    correlation_status=result.get("correlation_status"),
                ) | {"start_event": start_name, "end_event": end_name, "metric": metric}
            else:
                previous_end_ns = int(end_ns)
        if result.get("correlation_status") == "uncorrelated" and update_reason:
            result["missing_phase_reason"] = update_reason
        results[metric] = result
    return results


def _startup_timeline_gap_fields(
    events: list[dict[str, Any]],
    timeline_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    total = _single_interval_phase(
        events,
        "process.launch.start",
        "first_tk.after_idle.visible",
    )
    total_intervals = list(total.get("_intervals_ns") or [])
    if len(total_intervals) != 1:
        return {
            "startup_timeline_measured_total_ms": None,
            "startup_timeline_unexplained_gap_ms": None,
            "longest_unexplained_interval_ms": None,
            "longest_unexplained_interval_before_event": None,
            "longest_unexplained_interval_after_event": None,
            "startup_timeline_gap_status": "not_detected",
            "startup_timeline_gap_reason": "total_first_visible_interval_not_detected",
        }
    total_start, total_end = total_intervals[0]
    measured: list[tuple[int, int, str, str]] = []
    for metric in STARTUP_TIMELINE_ORDER:
        result = timeline_results.get(metric, {})
        if result.get("phase_status") != "measured":
            continue
        intervals = list(result.get("_intervals_ns") or [])
        if len(intervals) != 1:
            continue
        start_ns, end_ns = intervals[0]
        if int(start_ns) < total_start or int(end_ns) > total_end:
            continue
        measured.append(
            (
                int(start_ns),
                int(end_ns),
                str(result.get("start_event") or ""),
                str(result.get("end_event") or ""),
            )
        )
    measured.sort()
    non_overlapping: list[tuple[int, int, str, str]] = []
    previous_end_ns = total_start
    for start_ns, end_ns, start_event, end_event in measured:
        if start_ns < previous_end_ns:
            continue
        non_overlapping.append((start_ns, end_ns, start_event, end_event))
        previous_end_ns = end_ns
    measured_total_ms = sum(
        (end_ns - start_ns) / 1_000_000.0 for start_ns, end_ns, _, _ in non_overlapping
    )
    gaps: list[tuple[int, int, str, str]] = []
    cursor_ns = total_start
    before_event = "process.launch.start"
    for start_ns, end_ns, start_event, end_event in non_overlapping:
        if start_ns > cursor_ns:
            gaps.append((cursor_ns, start_ns, before_event, start_event))
        if end_ns > cursor_ns:
            cursor_ns = end_ns
            before_event = end_event
    if cursor_ns < total_end:
        gaps.append((cursor_ns, total_end, before_event, "first_tk.after_idle.visible"))
    longest = max(gaps, key=lambda gap: gap[1] - gap[0]) if gaps else None
    total_ms = (total_end - total_start) / 1_000_000.0
    unexplained_ms = max(0.0, total_ms - measured_total_ms)
    return {
        "startup_timeline_measured_total_ms": measured_total_ms,
        "startup_timeline_unexplained_gap_ms": unexplained_ms,
        "longest_unexplained_interval_ms": (
            (longest[1] - longest[0]) / 1_000_000.0 if longest else 0.0
        ),
        "longest_unexplained_interval_before_event": longest[2] if longest else None,
        "longest_unexplained_interval_after_event": longest[3] if longest else None,
        "startup_timeline_gap_status": "measured",
        "startup_timeline_gap_reason": None,
    }


def _events_named_with_update(
    events: list[dict[str, Any]],
    name: str,
    update_id: object,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if _event_has_name(event, name)
        and event.get("update_id") == update_id
        and _event_ns(event) is not None
    ]


def _events_with_update(
    events: list[dict[str, Any]],
    names: tuple[str, ...],
    update_id: object,
) -> list[dict[str, Any]]:
    name_set = set(names)
    matched = [
        event
        for event in events
        if str(event.get("event", "")) in name_set
        and event.get("update_id") == update_id
        and _event_ns(event) is not None
    ]
    return sorted(matched, key=lambda event: int(_event_ns(event) or 0))


def _first_simulation_compute_parent(
    events: list[dict[str, Any]],
    update_id: object | None,
) -> dict[str, object]:
    return _correlated_span_phase(
        events,
        "first_simulation.compute.start",
        "first_simulation.compute.end",
        update_id,
    )


def _first_simulation_compute_pair_result(
    events: list[dict[str, Any]],
    metric: str,
    update_id: object | None,
    parent_interval: tuple[int, int] | None,
) -> dict[str, object]:
    start_name, end_name = FIRST_SIMULATION_COMPUTE_EVENT_PAIRS[metric]
    if update_id is None:
        return _phase_result(
            status="uncorrelated",
            reason="first_visible_update_id_missing",
            correlation_status="uncorrelated",
        )
    if _has_uncorrelated_events(events, (start_name, end_name), update_id):
        return _phase_result(
            status="uncorrelated",
            reason="uncorrelated_update_id",
            correlation_status="uncorrelated",
        )
    starts = _events_named_with_update(events, start_name, update_id)
    ends = _events_named_with_update(events, end_name, update_id)
    if not starts and not ends:
        return _phase_result(
            status="not_detected",
            reason=f"missing_span:{start_name}->{end_name}",
            correlation_status="correlated",
        )
    if not starts or not ends:
        return _phase_result(
            status="not_detected",
            reason=f"incomplete_span:{start_name}->{end_name}",
            correlation_status="correlated",
        )
    intervals: list[tuple[int, int]] = []
    used_start_indexes: set[int] = set()
    used_end_indexes: set[int] = set()
    for start_index, start in enumerate(starts):
        start_ns = _event_ns(start)
        if start_ns is None:
            continue
        for index, end in enumerate(ends):
            if index in used_end_indexes:
                continue
            end_ns = _event_ns(end)
            if end_ns is None or end_ns < start_ns:
                continue
            used_start_indexes.add(start_index)
            used_end_indexes.add(index)
            intervals.append((start_ns, end_ns))
            break
    if not intervals:
        return _phase_result(
            status="overlap_error",
            reason=f"phase_events_out_of_order:{start_name}->{end_name}",
            correlation_status="correlated",
        )
    if len(used_start_indexes) != len(starts) or len(used_end_indexes) != len(ends):
        return _phase_result(
            status="not_detected",
            intervals_ns=intervals,
            reason=f"incomplete_span:{start_name}->{end_name}",
            correlation_status="correlated",
        )
    ok, overlap_reason = _non_overlapping_interval_status(intervals)
    if not ok:
        return _phase_result(
            status="overlap_error",
            intervals_ns=intervals,
            reason=overlap_reason,
            correlation_status="correlated",
        )
    if parent_interval is None:
        return _phase_result(
            status="not_detected",
            intervals_ns=intervals,
            reason="first_simulation_compute_parent_not_detected",
            correlation_status="correlated",
        )
    parent_start, parent_end = parent_interval
    for start_ns, end_ns in intervals:
        if start_ns < parent_start or end_ns > parent_end:
            return _phase_result(
                status="not_applicable",
                intervals_ns=intervals,
                reason="span_outside_first_simulation_compute_parent",
                correlation_status="correlated",
            )
    return _phase_result(
        status="measured",
        duration_ms=_duration_from_intervals_ms(intervals),
        intervals_ns=intervals,
        correlation_status="correlated",
    )


def _boundary_names_text(names: tuple[str, ...]) -> str:
    return "|".join(names)


def _correlated_boundary_result(
    events: list[dict[str, Any]],
    start_names: tuple[str, ...],
    end_names: tuple[str, ...],
    update_id: object | None,
    *,
    start_selector: str = "first",
    not_applicable_reason: str | None = None,
    not_applicable_without_start: bool = False,
) -> dict[str, object]:
    names = (*start_names, *end_names)
    if update_id is None:
        return _phase_result(
            status="uncorrelated",
            reason="first_visible_update_id_missing",
            correlation_status="uncorrelated",
        )
    if _has_uncorrelated_events(events, names, update_id):
        return _phase_result(
            status="uncorrelated",
            reason="uncorrelated_update_id",
            correlation_status="uncorrelated",
        )
    starts = _events_with_update(events, start_names, update_id)
    ends = _events_with_update(events, end_names, update_id)
    if not starts and not ends:
        return _phase_result(
            status="not_applicable" if not_applicable_reason else "not_detected",
            reason=not_applicable_reason or f"missing_boundary:{_boundary_names_text(start_names)}",
            correlation_status="correlated",
        )
    if not starts:
        return _phase_result(
            status="not_applicable" if not_applicable_without_start else "not_detected",
            reason=not_applicable_reason
            if not_applicable_without_start and not_applicable_reason
            else f"missing_boundary:{_boundary_names_text(start_names)}",
            correlation_status="correlated",
        )
    if not ends:
        return _phase_result(
            status="not_detected",
            reason=f"missing_boundary:{_boundary_names_text(end_names)}",
            correlation_status="correlated",
        )
    start = starts[-1] if start_selector == "last" else starts[0]
    start_ns = _event_ns(start)
    if start_ns is None:
        return _phase_result(
            status="not_detected",
            reason=f"missing_boundary_timestamp:{_boundary_names_text(start_names)}",
            correlation_status="correlated",
        )
    eligible_ends = [end for end in ends if (_event_ns(end) or -1) >= start_ns]
    if not eligible_ends:
        return _phase_result(
            status="overlap_error",
            reason=(
                "phase_events_out_of_order:"
                f"{_boundary_names_text(start_names)}->{_boundary_names_text(end_names)}"
            ),
            correlation_status="correlated",
        )
    end = eligible_ends[0]
    end_ns = _event_ns(end)
    if end_ns is None:
        return _phase_result(
            status="not_detected",
            reason=f"missing_boundary_timestamp:{_boundary_names_text(end_names)}",
            correlation_status="correlated",
        )
    return _phase_result(
        status="measured",
        duration_ms=(end_ns - start_ns) / 1_000_000.0,
        intervals_ns=[(start_ns, end_ns)],
        correlation_status="correlated",
    )


def _copy_worker_handoff_result(result: dict[str, object]) -> dict[str, object]:
    copied = _phase_result(
        status=str(result.get("phase_status", "not_detected")),
        duration_ms=(
            float(result["duration_ms"]) if result.get("duration_ms") is not None else None
        ),
        intervals_ns=[tuple(interval) for interval in list(result.get("_intervals_ns") or [])],
        reason=(
            str(result.get("missing_phase_reason")) if result.get("missing_phase_reason") else None
        ),
        correlation_status=(
            str(result.get("correlation_status")) if result.get("correlation_status") else None
        ),
    )
    copied["status"] = copied.get("phase_status")
    return copied


def _first_simulation_worker_handoff_results(
    events: list[dict[str, Any]],
    compute_results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    update_id, _update_reason = _correlated_update_id(events)
    results: dict[str, dict[str, object]] = {
        "worker_queue_wait_ms": _correlated_boundary_result(
            events,
            ("first_worker.request.queued",),
            ("first_worker.submit",),
            update_id,
            not_applicable_reason="worker_request_was_not_queued",
            not_applicable_without_start=True,
        ),
        "worker_submit_to_start_ms": _correlated_boundary_result(
            events,
            ("first_worker.submit",),
            ("first_simulation.compute.start",),
            update_id,
            not_applicable_reason="synchronous_first_compute_has_no_worker_submit",
            not_applicable_without_start=True,
        ),
        "worker_start_to_kernel_start_ms": _correlated_boundary_result(
            events,
            ("first_simulation.compute.start",),
            ("first_simulation.kernel_call.start",),
            update_id,
        ),
        "kernel_call_ms": _copy_worker_handoff_result(
            compute_results.get(
                "kernel_call",
                _phase_result(status="not_detected", reason="kernel_call_not_detected"),
            )
        ),
        "kernel_end_to_result_ready_ms": _correlated_boundary_result(
            events,
            ("first_simulation.kernel_call.end",),
            ("first_simulation.result_ready.start",),
            update_id,
            start_selector="last",
        ),
        "result_ready_to_gui_thread_ms": _correlated_boundary_result(
            events,
            ("first_simulation.result_ready.end",),
            ("simulation_worker.result.ready", "first_result.apply.start"),
            update_id,
            start_selector="last",
        ),
        "gui_thread_result_apply_ms": _correlated_boundary_result(
            events,
            ("first_result.apply.start",),
            ("first_result.apply.end",),
            update_id,
        ),
        "worker_result_transfer_ms": _correlated_boundary_result(
            events,
            ("simulation_worker.result.fetch.start",),
            ("simulation_worker.result.fetch.end",),
            update_id,
            not_applicable_reason="synchronous_first_compute_has_no_worker_result_fetch",
        ),
    }
    for result in results.values():
        result["status"] = result.get("phase_status")
    return results


def _first_simulation_compute_results(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    update_id, update_reason = _correlated_update_id(events)
    parent = _first_simulation_compute_parent(events, update_id)
    parent_intervals = list(parent.get("_intervals_ns") or [])
    parent_interval = (
        (int(parent_intervals[0][0]), int(parent_intervals[0][1]))
        if len(parent_intervals) == 1 and parent.get("phase_status") == "measured"
        else None
    )
    results: dict[str, dict[str, object]] = {}
    for metric in FIRST_SIMULATION_COMPUTE_ORDER:
        if metric in FIRST_SIMULATION_COMPUTE_OUTSIDE_ROWS:
            result = _phase_result(
                status="not_applicable",
                reason=FIRST_SIMULATION_COMPUTE_OUTSIDE_ROWS[metric],
            )
        elif update_reason:
            result = _phase_result(
                status="uncorrelated",
                reason=update_reason,
                correlation_status="uncorrelated",
            )
        elif metric in FIRST_SIMULATION_COMPUTE_EVENT_PAIRS:
            result = _first_simulation_compute_pair_result(
                events,
                metric,
                update_id,
                parent_interval,
            )
        elif metric in FIRST_SIMULATION_COMPUTE_KERNEL_INTERNAL_ROWS:
            result = _phase_result(
                status="not_detected",
                reason="kernel_internal_not_separately_instrumented",
                correlation_status="correlated" if update_id is not None else None,
            )
        else:
            result = _phase_result(
                status="not_detected",
                reason=f"missing_compute_subphase:{metric}",
            )
        result["status"] = result.get("phase_status")
        results[metric] = result
    return results


def _first_simulation_compute_gap_fields(
    events: list[dict[str, Any]],
    compute_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    update_id, update_reason = _correlated_update_id(events)
    parent = _first_simulation_compute_parent(events, update_id)
    parent_intervals = list(parent.get("_intervals_ns") or [])
    if parent.get("phase_status") != "measured" or len(parent_intervals) != 1:
        return {
            "first_simulation_compute_total_ms": None,
            "first_simulation_compute_measured_subphase_total_ms": None,
            "first_simulation_compute_unexplained_gap_ms": None,
            "first_simulation_compute_slowest_subphase": None,
            "first_simulation_compute_slowest_subphase_ms": None,
            "first_simulation_update_id": update_id,
            "first_simulation_compute_gap_status": "uncorrelated"
            if update_reason
            else "not_detected",
            "first_simulation_compute_gap_reason": update_reason
            or "first_simulation_compute_parent_not_detected",
        }
    parent_start, parent_end = parent_intervals[0]
    child_intervals: list[tuple[int, int, str]] = []
    measured_rows: list[tuple[str, float]] = []
    for metric in FIRST_SIMULATION_COMPUTE_TOTAL_ROWS:
        result = compute_results.get(metric, {})
        if result.get("phase_status") != "measured":
            continue
        duration = result.get("duration_ms")
        if duration is not None:
            measured_rows.append((metric, float(duration)))
        for start_ns, end_ns in list(result.get("_intervals_ns") or []):
            child_intervals.append((int(start_ns), int(end_ns), metric))
    child_intervals.sort()
    interval_pairs = [(start_ns, end_ns) for start_ns, end_ns, _ in child_intervals]
    ok, reason = _non_overlapping_interval_status(interval_pairs)
    if not ok:
        return {
            "first_simulation_compute_total_ms": (parent_end - parent_start) / 1_000_000.0,
            "first_simulation_compute_measured_subphase_total_ms": None,
            "first_simulation_compute_unexplained_gap_ms": None,
            "first_simulation_compute_slowest_subphase": None,
            "first_simulation_compute_slowest_subphase_ms": None,
            "first_simulation_update_id": update_id,
            "first_simulation_compute_gap_status": "overlap_error",
            "first_simulation_compute_gap_reason": reason,
        }
    for start_ns, end_ns, metric in child_intervals:
        if start_ns < parent_start or end_ns > parent_end:
            return {
                "first_simulation_compute_total_ms": (parent_end - parent_start) / 1_000_000.0,
                "first_simulation_compute_measured_subphase_total_ms": None,
                "first_simulation_compute_unexplained_gap_ms": None,
                "first_simulation_compute_slowest_subphase": None,
                "first_simulation_compute_slowest_subphase_ms": None,
                "first_simulation_update_id": update_id,
                "first_simulation_compute_gap_status": "not_applicable",
                "first_simulation_compute_gap_reason": f"child_span_outside_parent:{metric}",
            }
    parent_ms = (parent_end - parent_start) / 1_000_000.0
    measured_ms = sum((end_ns - start_ns) / 1_000_000.0 for start_ns, end_ns in interval_pairs)
    slowest = max(measured_rows, key=lambda item: item[1]) if measured_rows else (None, None)
    return {
        "first_simulation_compute_total_ms": parent_ms,
        "first_simulation_compute_measured_subphase_total_ms": measured_ms,
        "first_simulation_compute_unexplained_gap_ms": max(0.0, parent_ms - measured_ms),
        "first_simulation_compute_slowest_subphase": slowest[0],
        "first_simulation_compute_slowest_subphase_ms": slowest[1],
        "first_simulation_update_id": update_id,
        "first_simulation_compute_gap_status": "measured",
        "first_simulation_compute_gap_reason": None,
    }


def _saved_state_context_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for name in ("saved_state.metadata", "saved_state.end", "saved_state.start"):
        event = _first_event(events, name)
        if event is not None:
            return event
    return None


def _saved_state_trial_row(events: list[dict[str, Any]]) -> dict[str, object] | None:
    context_event = _saved_state_context_event(events)
    if context_event is None:
        return None
    phase_results = _saved_state_phase_results(events)
    startup_timeline_results = _startup_timeline_results(events)
    startup_timeline_gap = _startup_timeline_gap_fields(events, startup_timeline_results)
    first_simulation_compute_results = _first_simulation_compute_results(events)
    first_simulation_compute_gap = _first_simulation_compute_gap_fields(
        events,
        first_simulation_compute_results,
    )
    first_simulation_worker_handoff_results = _first_simulation_worker_handoff_results(
        events,
        first_simulation_compute_results,
    )
    first_visible_total_ms = phase_results["total process launch to first visible"].get(
        "duration_ms"
    )
    saved_state_restore_total_ms = _duration_ms(events, "saved_state.start", "saved_state.end")
    unexplained_ms = phase_results["unexplained saved-state startup gap"].get("duration_ms")
    phase_statuses = {
        metric: {
            "phase_status": result.get("phase_status"),
            "missing_phase_reason": result.get("missing_phase_reason"),
            "correlation_status": result.get("correlation_status"),
        }
        for metric, result in phase_results.items()
    }
    startup_timeline_statuses = {
        metric: {
            "phase_status": result.get("phase_status"),
            "missing_phase_reason": result.get("missing_phase_reason"),
            "correlation_status": result.get("correlation_status"),
        }
        for metric, result in startup_timeline_results.items()
    }
    first_simulation_compute_statuses = {
        metric: {
            "status": result.get("status") or result.get("phase_status"),
            "phase_status": result.get("phase_status"),
            "missing_phase_reason": result.get("missing_phase_reason"),
            "correlation_status": result.get("correlation_status"),
        }
        for metric, result in first_simulation_compute_results.items()
    }
    first_simulation_worker_handoff_statuses = {
        metric: {
            "status": result.get("status") or result.get("phase_status"),
            "phase_status": result.get("phase_status"),
            "missing_phase_reason": result.get("missing_phase_reason"),
            "correlation_status": result.get("correlation_status"),
        }
        for metric, result in first_simulation_worker_handoff_results.items()
    }
    worker_handoff_ms_fields = {
        metric: (result.get("duration_ms") if result.get("phase_status") == "measured" else None)
        for metric, result in first_simulation_worker_handoff_results.items()
    }
    return {
        "trial_id": context_event.get("trial_id"),
        "scenario_id": context_event.get("scenario_id"),
        "state_path_basename": _safe_basename(
            context_event.get("state_path_basename") or context_event.get("state_file")
        ),
        "state_file_size_bytes": context_event.get("state_file_size_bytes"),
        "restored_background_count": context_event.get("restored_background_count"),
        "restored_manual_pair_count": context_event.get("restored_manual_pair_count"),
        "restored_selection_count": context_event.get("restored_selection_count"),
        "restored_has_caked_state": context_event.get("restored_has_caked_state"),
        "restored_has_geometry_manual_pairs": context_event.get(
            "restored_has_geometry_manual_pairs"
        ),
        "metadata_status": context_event.get("metadata_status"),
        "missing_phase_reason": context_event.get("missing_phase_reason"),
        "first_visible_total_ms": first_visible_total_ms,
        "saved_state_restore_total_ms": saved_state_restore_total_ms,
        "saved_state_unexplained_gap_ms": unexplained_ms,
        "startup_timeline_measured_total_ms": startup_timeline_gap.get(
            "startup_timeline_measured_total_ms"
        ),
        "startup_timeline_unexplained_gap_ms": startup_timeline_gap.get(
            "startup_timeline_unexplained_gap_ms"
        ),
        "longest_unexplained_interval_ms": startup_timeline_gap.get(
            "longest_unexplained_interval_ms"
        ),
        "longest_unexplained_interval_before_event": startup_timeline_gap.get(
            "longest_unexplained_interval_before_event"
        ),
        "longest_unexplained_interval_after_event": startup_timeline_gap.get(
            "longest_unexplained_interval_after_event"
        ),
        "first_simulation_compute_total_ms": first_simulation_compute_gap.get(
            "first_simulation_compute_total_ms"
        ),
        "first_simulation_compute_measured_subphase_total_ms": (
            first_simulation_compute_gap.get("first_simulation_compute_measured_subphase_total_ms")
        ),
        "first_simulation_compute_unexplained_gap_ms": first_simulation_compute_gap.get(
            "first_simulation_compute_unexplained_gap_ms"
        ),
        "first_simulation_compute_slowest_subphase": first_simulation_compute_gap.get(
            "first_simulation_compute_slowest_subphase"
        ),
        "first_simulation_compute_slowest_subphase_ms": first_simulation_compute_gap.get(
            "first_simulation_compute_slowest_subphase_ms"
        ),
        "first_simulation_update_id": first_simulation_compute_gap.get(
            "first_simulation_update_id"
        ),
        "first_simulation_compute_gap_status": first_simulation_compute_gap.get(
            "first_simulation_compute_gap_status"
        ),
        "first_simulation_compute_gap_reason": first_simulation_compute_gap.get(
            "first_simulation_compute_gap_reason"
        ),
        **worker_handoff_ms_fields,
        "saved_state_phase_statuses": phase_statuses,
        "startup_timeline_statuses": startup_timeline_statuses,
        "first_simulation_compute_statuses": first_simulation_compute_statuses,
        "first_simulation_worker_handoff_statuses": (first_simulation_worker_handoff_statuses),
        "_phase_results": phase_results,
        "_startup_timeline_results": startup_timeline_results,
        "_first_simulation_compute_results": first_simulation_compute_results,
        "_first_simulation_worker_handoff_results": (first_simulation_worker_handoff_results),
    }


def _saved_state_trial_rows(
    trial_events: list[list[dict[str, Any]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for events in trial_events:
        row = _saved_state_trial_row(events)
        if row is not None:
            rows.append(row)
    return rows


def _saved_state_summary_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not rows:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for metric in SAVED_STATE_BREAKDOWN_ORDER:
        values: list[float] = []
        missing_count = 0
        statuses: list[str] = []
        reasons: list[str] = []
        correlations: list[str] = []
        for row in rows:
            phase_results = row.get("_phase_results")
            if not isinstance(phase_results, dict):
                continue
            result = phase_results.get(metric, {})
            status = str(result.get("phase_status", "not_detected"))
            statuses.append(status)
            if result.get("correlation_status"):
                correlations.append(str(result.get("correlation_status")))
            duration = result.get("duration_ms")
            if status == "measured" and duration is not None:
                values.append(float(duration))
            else:
                missing_count += 1
                reason = result.get("missing_phase_reason")
                if reason:
                    reasons.append(str(reason))
        stats = _stats(values)
        if values:
            phase_status = "measured"
        elif statuses and all(status == "not_applicable" for status in statuses):
            phase_status = "not_applicable"
        else:
            phase_status = "not_detected"
        stats.update(
            {
                "phase_status": phase_status,
                "missing_count": missing_count,
                "missing_phase_reason": ";".join(sorted(set(reasons))) or None,
                "correlation_status": ";".join(sorted(set(correlations))) or None,
            }
        )
        summary[metric] = stats
    return summary


def _startup_timeline_summary_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not rows:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for metric in STARTUP_TIMELINE_ORDER:
        values: list[float] = []
        missing_count = 0
        statuses: list[str] = []
        reasons: list[str] = []
        correlations: list[str] = []
        for row in rows:
            timeline_results = row.get("_startup_timeline_results")
            if not isinstance(timeline_results, dict):
                continue
            result = timeline_results.get(metric, {})
            status = str(result.get("phase_status", "not_detected"))
            statuses.append(status)
            if result.get("correlation_status"):
                correlations.append(str(result.get("correlation_status")))
            duration = result.get("duration_ms")
            if status == "measured" and duration is not None:
                values.append(float(duration))
            else:
                missing_count += 1
                reason = result.get("missing_phase_reason")
                if reason:
                    reasons.append(str(reason))
        stats = _stats(values)
        if "overlap_error" in statuses:
            phase_status = "overlap_error"
        elif "uncorrelated" in statuses:
            phase_status = "uncorrelated"
        elif values:
            phase_status = "measured"
        elif statuses and all(status == "not_applicable" for status in statuses):
            phase_status = "not_applicable"
        else:
            phase_status = "not_detected"
        stats.update(
            {
                "phase_status": phase_status,
                "missing_count": missing_count,
                "missing_phase_reason": ";".join(sorted(set(reasons))) or None,
                "correlation_status": ";".join(sorted(set(correlations))) or None,
            }
        )
        summary[metric] = stats
    return summary


def _startup_timeline_unattributed_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    values: list[float] = []
    missing_count = 0
    for row in rows:
        value = row.get("startup_timeline_unexplained_gap_ms")
        if value is None:
            missing_count += 1
            continue
        values.append(float(value))
    stats = _stats(values)
    stats["raw_ms"] = values
    stats["missing_count"] = missing_count
    return stats


def _first_simulation_compute_summary_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not rows:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for metric in FIRST_SIMULATION_COMPUTE_ORDER:
        values: list[float] = []
        missing_count = 0
        statuses: list[str] = []
        reasons: list[str] = []
        correlations: list[str] = []
        for row in rows:
            compute_results = row.get("_first_simulation_compute_results")
            if not isinstance(compute_results, dict):
                continue
            result = compute_results.get(metric, {})
            status = str(result.get("phase_status", "not_detected"))
            statuses.append(status)
            if result.get("correlation_status"):
                correlations.append(str(result.get("correlation_status")))
            duration = result.get("duration_ms")
            if status == "measured" and duration is not None:
                values.append(float(duration))
            else:
                missing_count += 1
                reason = result.get("missing_phase_reason")
                if reason:
                    reasons.append(str(reason))
        stats = _stats(values)
        if "overlap_error" in statuses:
            phase_status = "overlap_error"
        elif "uncorrelated" in statuses:
            phase_status = "uncorrelated"
        elif values:
            phase_status = "measured"
        elif statuses and all(status == "not_applicable" for status in statuses):
            phase_status = "not_applicable"
        else:
            phase_status = "not_detected"
        stats.update(
            {
                "status": phase_status,
                "phase_status": phase_status,
                "missing_count": missing_count,
                "missing_phase_reason": ";".join(sorted(set(reasons))) or None,
                "correlation_status": ";".join(sorted(set(correlations))) or None,
            }
        )
        summary[metric] = stats
    return summary


def _first_simulation_worker_handoff_summary_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not rows:
        return {}
    summary: dict[str, dict[str, object]] = {}
    for metric in FIRST_SIMULATION_WORKER_HANDOFF_ORDER:
        values: list[float] = []
        missing_count = 0
        statuses: list[str] = []
        reasons: list[str] = []
        correlations: list[str] = []
        for row in rows:
            handoff_results = row.get("_first_simulation_worker_handoff_results")
            if not isinstance(handoff_results, dict):
                continue
            result = handoff_results.get(metric, {})
            status = str(result.get("phase_status", "not_detected"))
            statuses.append(status)
            if result.get("correlation_status"):
                correlations.append(str(result.get("correlation_status")))
            duration = result.get("duration_ms")
            if status == "measured" and duration is not None:
                values.append(float(duration))
            else:
                missing_count += 1
                reason = result.get("missing_phase_reason")
                if reason:
                    reasons.append(str(reason))
        stats = _stats(values)
        stats["raw_ms"] = values
        if "overlap_error" in statuses:
            phase_status = "overlap_error"
        elif "uncorrelated" in statuses:
            phase_status = "uncorrelated"
        elif values:
            phase_status = "measured"
        elif statuses and all(status == "not_applicable" for status in statuses):
            phase_status = "not_applicable"
        else:
            phase_status = "not_detected"
        stats.update(
            {
                "status": phase_status,
                "phase_status": phase_status,
                "missing_count": missing_count,
                "missing_phase_reason": ";".join(sorted(set(reasons))) or None,
                "correlation_status": ";".join(sorted(set(correlations))) or None,
            }
        )
        summary[metric] = stats
    return summary


def _first_simulation_compute_unattributed_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    values: list[float] = []
    missing_count = 0
    reasons: list[str] = []
    for row in rows:
        value = row.get("first_simulation_compute_unexplained_gap_ms")
        if value is None:
            missing_count += 1
            reason = row.get("first_simulation_compute_gap_reason")
            if reason:
                reasons.append(str(reason))
            continue
        values.append(float(value))
    stats = _stats(values)
    stats["raw_ms"] = values
    stats["missing_count"] = missing_count
    stats["missing_phase_reason"] = ";".join(sorted(set(reasons))) or None
    return stats


def _summary_for_metrics(
    trial_events: list[list[dict[str, Any]]],
    metrics: dict[str, tuple[str, str]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for metric, (start, end) in metrics.items():
        values: list[float] = []
        missing_count = 0
        for events in trial_events:
            duration = _duration_ms(events, start, end, pairing=_metric_pairing(metric))
            if duration is None:
                missing_count += 1
                continue
            values.append(duration)
        summary[metric] = _stats(values)
        summary[metric]["raw_ms"] = values
        summary[metric]["missing_count"] = missing_count
        summary[metric]["pairing"] = _metric_pairing(metric)
    return summary


def _trial_group_for_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        group = event.get("trial_group")
        if group:
            return str(group)
    return "unclassified"


def _startup_summary_by_group(
    trial_events: list[list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for events in trial_events:
        grouped[_trial_group_for_events(events)].append(events)
    return {
        group: _summary_for_metrics(group_events, STARTUP_METRICS)
        for group, group_events in sorted(grouped.items())
    }


def _summary_for_grouped_theta_metrics(
    trial_events: list[list[dict[str, Any]]],
    metrics: dict[str, tuple[str, str]],
    *,
    transition: str | None = None,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for metric, (start, end) in metrics.items():
        values: list[float] = []
        for events in trial_events:
            for update_events in _events_by_update(events).values():
                if not any(
                    str(event.get("event")) == "theta_change.after_idle.visible"
                    for event in update_events
                ):
                    continue
                input_event = _first_event(update_events, "theta_change.input.start")
                if transition is not None and (input_event or {}).get("transition") != transition:
                    continue
                duration = _duration_ms(
                    update_events,
                    start,
                    end,
                    pairing=_metric_pairing(metric),
                )
                if duration is not None:
                    values.append(duration)
                else:
                    summary.setdefault(metric, {"missing_count": 0})
                    summary[metric]["missing_count"] = int(summary[metric]["missing_count"]) + 1
        summary[metric] = _stats(values)
        summary[metric]["raw_ms"] = values
        summary[metric]["missing_count"] = int(summary[metric].get("missing_count", 0))
        summary[metric]["pairing"] = _metric_pairing(metric)
    return summary


def _summary_for_grouped_redraw_metrics(
    trial_events: list[list[dict[str, Any]]],
    metrics: dict[str, tuple[str, str]],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for metric, (start, end) in metrics.items():
        values: list[float] = []
        for events in trial_events:
            for update_events in _events_by_update(events).values():
                if not any(
                    str(event.get("event")) == "redraw_only.after_idle.visible"
                    for event in update_events
                ):
                    continue
                duration = _duration_ms(
                    update_events,
                    start,
                    end,
                    pairing=_metric_pairing(metric),
                )
                if duration is not None:
                    values.append(duration)
                else:
                    summary.setdefault(metric, {"missing_count": 0})
                    summary[metric]["missing_count"] = int(summary[metric]["missing_count"]) + 1
        summary[metric] = _stats(values)
        summary[metric]["raw_ms"] = values
        summary[metric]["missing_count"] = int(summary[metric].get("missing_count", 0))
        summary[metric]["pairing"] = _metric_pairing(metric)
    return summary


def _redraw_validation_counts(trial_events: list[list[dict[str, Any]]]) -> dict[str, int]:
    compute_span_count = 0
    compute_absent_count = 0
    visible_count = 0
    for events in trial_events:
        for update_events in _events_by_update(events).values():
            if not any(
                str(event.get("event")) == "redraw_only.after_idle.visible"
                for event in update_events
            ):
                continue
            visible_count += 1
            compute_absent_count += sum(
                1
                for event in update_events
                if str(event.get("event")) == "redraw_only.compute_absent"
            )
            compute_span_count += sum(
                1 for event in update_events if str(event.get("event")).endswith(".compute.start")
            )
    return {
        "redraw_visible_count": visible_count,
        "redraw_compute_absent_count": compute_absent_count,
        "redraw_compute_span_count": compute_span_count,
    }


def _events_by_update(events: list[dict[str, Any]]) -> dict[object, list[dict[str, Any]]]:
    grouped: dict[object, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        update_id = event.get("update_id")
        if update_id is None:
            continue
        grouped[update_id].append(event)
    return grouped


def _theta_rows(trial_events: list[list[dict[str, Any]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for events in trial_events:
        trial = (events[0].get("trial_id") if events else None) or ""
        for update_id, update_events in _events_by_update(events).items():
            if not any(
                str(event.get("event")) == "theta_change.after_idle.visible"
                for event in update_events
            ):
                continue
            input_event = _first_event(update_events, "theta_change.input.start")
            var_event = _first_event(update_events, "theta_change.var_write")
            visible_event = _first_event(update_events, "theta_change.after_idle.visible")
            compute_ms = _duration_ms(
                update_events,
                "theta_change.compute.start",
                "theta_change.compute.end",
            )
            render_ms = _duration_ms(
                update_events,
                "theta_change.compute.end",
                "theta_change.after_idle.visible",
            )
            total_ms = _duration_ms(
                update_events,
                "theta_change.input.start",
                "theta_change.after_idle.visible",
            )
            before = (input_event or {}).get("display_fingerprint_before", {})
            if not before:
                before = (var_event or {}).get("display_fingerprint_before", {})
            after = (visible_event or {}).get("display_fingerprint", {})
            cache_event = _last_event(update_events, "theta_change.cache_hit")
            rows.append(
                {
                    "trial": trial,
                    "scenario": (visible_event or {}).get("scenario_id", ""),
                    "transition": (input_event or {}).get("transition", ""),
                    "old_theta": (var_event or input_event or {}).get("old_theta", ""),
                    "new_theta": (input_event or var_event or {}).get("new_theta", ""),
                    "cache_hit": (cache_event or visible_event or {}).get("cache_hit", ""),
                    "compute_ms": compute_ms,
                    "render_ms": render_ms,
                    "total_ms": total_ms,
                    "display_hash_before": before.get("hash") if isinstance(before, dict) else "",
                    "display_hash_after": after.get("hash") if isinstance(after, dict) else "",
                    "update_id": update_id,
                }
            )
    return rows


def _write_combined_csv(output_dir: Path, trial_events: list[list[dict[str, Any]]]) -> None:
    rows: list[dict[str, object]] = []
    for events in trial_events:
        for event in events:
            rows.append(
                {
                    "trial_id": event.get("trial_id", ""),
                    "scenario_id": event.get("scenario_id", ""),
                    "update_id": event.get("update_id", ""),
                    "event": event.get("event", ""),
                    "phase": event.get("phase", ""),
                    "perf_counter_ns": event.get("perf_counter_ns", ""),
                    "wall_time_iso": event.get("wall_time_iso", ""),
                    "fields_json": json.dumps(
                        {
                            key: value
                            for key, value in event.items()
                            if key
                            not in {
                                "trial_id",
                                "scenario_id",
                                "update_id",
                                "event",
                                "phase",
                                "perf_counter_ns",
                                "wall_time_iso",
                            }
                        },
                        sort_keys=True,
                    ),
                }
            )
    with (output_dir / "combined_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trial_id",
                "scenario_id",
                "update_id",
                "event",
                "phase",
                "perf_counter_ns",
                "wall_time_iso",
                "fields_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _compact_raw_ms(values: object, *, limit: int = 12) -> str:
    if not isinstance(values, list):
        return ""
    formatted = [_format_ms(value) for value in values[:limit]]
    if len(values) > limit:
        formatted.append(f"... ({len(values)} total)")
    return ", ".join(value for value in formatted if value)


def _markdown_table(summary: dict[str, dict[str, object]]) -> str:
    headers = [
        "metric",
        "trial_count",
        "missing_count",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "p95_ms",
        "std_ms",
        "raw_ms",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for metric, stats in summary.items():
        lines.append(
            " | ".join(
                [
                    metric,
                    str(stats.get("trial_count", 0)),
                    str(stats.get("missing_count", 0)),
                    _format_ms(stats.get("median_ms")),
                    _format_ms(stats.get("mean_ms")),
                    _format_ms(stats.get("min_ms")),
                    _format_ms(stats.get("max_ms")),
                    _format_ms(stats.get("p95_ms")),
                    _format_ms(stats.get("std_ms")),
                    _compact_raw_ms(stats.get("raw_ms")),
                ]
            )
        )
    return "\n".join(lines)


def _saved_state_markdown_table(summary: dict[str, dict[str, object]]) -> str:
    if not summary:
        return "No saved-state startup rows captured."
    headers = [
        "metric",
        "phase_status",
        "trial_count",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "p95_ms",
        "std_ms",
        "missing_count",
        "missing_phase_reason",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for metric, stats in summary.items():
        lines.append(
            " | ".join(
                [
                    metric,
                    str(stats.get("phase_status", "")),
                    str(stats.get("trial_count", 0)),
                    _format_ms(stats.get("median_ms")),
                    _format_ms(stats.get("mean_ms")),
                    _format_ms(stats.get("min_ms")),
                    _format_ms(stats.get("max_ms")),
                    _format_ms(stats.get("p95_ms")),
                    _format_ms(stats.get("std_ms")),
                    str(stats.get("missing_count", 0)),
                    str(stats.get("missing_phase_reason") or ""),
                ]
            )
        )
    return "\n".join(lines)


def _first_simulation_compute_markdown_table(summary: dict[str, dict[str, object]]) -> str:
    if not summary:
        return "No first simulation compute rows captured."
    headers = [
        "metric",
        "trial_count",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "p95_ms",
        "std_ms",
        "status",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for metric, stats in summary.items():
        lines.append(
            " | ".join(
                [
                    metric,
                    str(stats.get("trial_count", 0)),
                    _format_ms(stats.get("median_ms")),
                    _format_ms(stats.get("mean_ms")),
                    _format_ms(stats.get("min_ms")),
                    _format_ms(stats.get("max_ms")),
                    _format_ms(stats.get("p95_ms")),
                    _format_ms(stats.get("std_ms")),
                    str(stats.get("status") or stats.get("phase_status") or ""),
                ]
            )
        )
    return "\n".join(lines)


def _first_simulation_worker_handoff_markdown_table(
    summary: dict[str, dict[str, object]],
) -> str:
    if not summary:
        return "No first simulation worker handoff rows captured."
    headers = [
        "metric",
        "trial_count",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "p95_ms",
        "std_ms",
        "status",
        "missing_count",
        "raw_ms",
        "missing_phase_reason",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for metric, stats in summary.items():
        lines.append(
            " | ".join(
                [
                    metric,
                    str(stats.get("trial_count", 0)),
                    _format_ms(stats.get("median_ms")),
                    _format_ms(stats.get("mean_ms")),
                    _format_ms(stats.get("min_ms")),
                    _format_ms(stats.get("max_ms")),
                    _format_ms(stats.get("p95_ms")),
                    _format_ms(stats.get("std_ms")),
                    str(stats.get("status") or stats.get("phase_status") or ""),
                    str(stats.get("missing_count", 0)),
                    _compact_raw_ms(stats.get("raw_ms")),
                    str(stats.get("missing_phase_reason") or ""),
                ]
            )
        )
    return "\n".join(lines)


def _first_simulation_compute_diagnostics(
    summary: dict[str, dict[str, object]],
) -> list[str]:
    diagnostics: list[str] = []
    for metric, stats in summary.items():
        status = str(stats.get("status") or stats.get("phase_status") or "")
        if status == "measured":
            continue
        reason = str(stats.get("missing_phase_reason") or "")
        diagnostics.append(f"- {metric}: `{status}`" + (f" ({reason})" if reason else ""))
    return diagnostics


def _first_simulation_worker_handoff_conclusion(
    summary: dict[str, dict[str, object]],
) -> str:
    measured: list[tuple[str, float]] = []
    for metric in FIRST_SIMULATION_WORKER_HANDOFF_ORDER:
        if metric == "kernel_call_ms":
            continue
        stats = summary.get(metric, {})
        if stats.get("status") != "measured":
            continue
        median_ms = stats.get("median_ms")
        if median_ms is None:
            continue
        measured.append((metric, float(median_ms)))
    if not measured:
        return (
            "No direct non-kernel worker handoff rows were measured. "
            "`worker_wait_or_sync_compute` is a broad diagnostic span for these rows, "
            "not exclusive queue/sync overhead."
        )
    metric, median_ms = max(measured, key=lambda item: item[1])
    category = FIRST_SIMULATION_WORKER_HANDOFF_CATEGORY.get(metric, "worker handoff")
    return (
        "Largest direct non-kernel handoff row is "
        f"`{metric}` ({_format_ms(median_ms)} ms median), classed as {category}. "
        "`worker_wait_or_sync_compute` remains a broad diagnostic span, "
        "diagnostic-only and excluded from child-gap subtraction."
    )


def _public_saved_state_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    public_rows: list[dict[str, object]] = []
    for row in rows:
        public_row = {key: value for key, value in row.items() if not str(key).startswith("_")}
        public_rows.append(public_row)
    return public_rows


def _raw_theta_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No theta-change rows captured."
    headers = [
        "trial",
        "scenario",
        "transition",
        "old_theta",
        "new_theta",
        "cache_hit",
        "compute_ms",
        "render_ms",
        "total_ms",
        "display_hash_before",
        "display_hash_after",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("trial", "")),
                    str(row.get("scenario", "")),
                    str(row.get("transition", "")),
                    str(row.get("old_theta", "")),
                    str(row.get("new_theta", "")),
                    str(row.get("cache_hit", "")),
                    _format_ms(row.get("compute_ms")),
                    _format_ms(row.get("render_ms")),
                    _format_ms(row.get("total_ms")),
                    str(row.get("display_hash_before", ""))[:12],
                    str(row.get("display_hash_after", ""))[:12],
                ]
            )
        )
    return "\n".join(lines)


def _trial_results_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No trial results captured."
    headers = ["artifact", "measured", "returncode", "timed_out", "termination", "child_pid"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("artifact", "")),
                    str(row.get("measured", "")),
                    str(row.get("returncode", "")),
                    str(row.get("timed_out", "")),
                    str(row.get("termination", "")),
                    str(row.get("child_pid", "")),
                ]
            )
        )
    return "\n".join(lines)


def _first_line(value: object, default: str = "not detected") -> str:
    lines = str(value or default).splitlines()
    return lines[0] if lines else default


def _write_report(
    output_dir: Path,
    metadata: dict[str, object],
    startup_summary: dict[str, dict[str, object]],
    startup_by_group: dict[str, dict[str, dict[str, object]]],
    saved_state_summary: dict[str, dict[str, object]],
    startup_timeline_summary: dict[str, dict[str, object]],
    startup_timeline_unattributed: dict[str, object],
    first_simulation_compute_summary: dict[str, dict[str, object]],
    first_simulation_compute_unattributed: dict[str, object],
    first_simulation_worker_handoff_summary: dict[str, dict[str, object]],
    saved_state_rows: list[dict[str, object]],
    theta_summary: dict[str, dict[str, object]],
    theta_return_summary: dict[str, dict[str, object]],
    redraw_summary: dict[str, dict[str, object]],
    redraw_validation: dict[str, int],
    theta_rows: list[dict[str, object]],
    trial_results: list[dict[str, object]],
    trial_failures: list[dict[str, object]],
) -> None:
    report = [
        "# RA-SIM GUI Timing Report",
        "",
        f"- Scenario: `{metadata.get('scenario')}`",
        f"- Trials: `{metadata.get('trials')}`",
        f"- Git: `{metadata.get('git_branch')}` `{metadata.get('git_commit')}`",
        f"- OS: `{metadata.get('os')}`",
        f"- Python: `{_first_line(metadata.get('python'))}`",
        f"- Backend: `{metadata.get('backend')}`",
        f"- State file: `{metadata.get('state_file') or 'none'}`",
        f"- Failed runs: `{len(trial_failures)}`",
        "",
        "## Startup",
        "",
        _markdown_table(startup_summary),
        "",
        "## Startup By Group",
        "",
        *[
            f"### {group}\n\n{_markdown_table(group_summary)}\n"
            for group, group_summary in startup_by_group.items()
        ],
        *(
            [
                "## Saved-State Startup Breakdown",
                "",
                _saved_state_markdown_table(saved_state_summary),
                "",
                "## Startup Timeline",
                "",
                _saved_state_markdown_table(startup_timeline_summary),
                "",
                "- total_unattributed_gap_ms median: "
                f"`{_format_ms(startup_timeline_unattributed.get('median_ms'))}`.",
                "",
                "## First Simulation Compute Breakdown",
                "",
                _first_simulation_compute_markdown_table(first_simulation_compute_summary),
                "",
                "- total_unattributed_gap_ms median: "
                f"`{_format_ms(first_simulation_compute_unattributed.get('median_ms'))}`.",
                "",
                "Diagnostics:",
                "",
                *(
                    _first_simulation_compute_diagnostics(first_simulation_compute_summary)
                    or ["- none"]
                ),
                "",
                "## First Simulation Worker Handoff",
                "",
                _first_simulation_worker_handoff_markdown_table(
                    first_simulation_worker_handoff_summary
                ),
                "",
                "- Conclusion: "
                + _first_simulation_worker_handoff_conclusion(
                    first_simulation_worker_handoff_summary
                ),
                "",
                "- Raw saved-state trial rows are in `summary.json` under "
                "`saved_state_startup.trial_rows`.",
                "",
            ]
            if saved_state_rows
            else []
        ),
        "## Theta Change To 10 Deg",
        "",
        _markdown_table(theta_summary),
        "",
        "## Theta Return To Original",
        "",
        _markdown_table(theta_return_summary),
        "",
        "## Redraw Only",
        "",
        _markdown_table(redraw_summary),
        "",
        "- Redraw updates visible: "
        f"`{redraw_validation.get('redraw_visible_count', 0)}`; "
        "compute_absent events: "
        f"`{redraw_validation.get('redraw_compute_absent_count', 0)}`; "
        "compute spans during redraw: "
        f"`{redraw_validation.get('redraw_compute_span_count', 0)}`.",
        "",
        "## Raw Theta Rows",
        "",
        _raw_theta_table(theta_rows),
        "",
        "## Trial Results",
        "",
        _trial_results_table(trial_results),
        "",
        "## Notes",
        "",
        "- Raw trial JSONL files contain the source-of-truth `perf_counter_ns` events.",
        "- Local paths are redacted to basenames in metadata and timing event helpers.",
        "- External pyautogui-style input is not run unless added separately; in-app Tk automation is used by default.",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")


def _write_summary(
    output_dir: Path,
    metadata: dict[str, object],
    trial_events: list[list[dict[str, Any]]],
    trial_results: list[TrialResult],
    all_run_events: list[list[dict[str, Any]]] | None = None,
) -> None:
    output_events = all_run_events or trial_events
    metadata = _merge_child_runtime_metadata(metadata, output_events)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    startup_summary = _summary_for_metrics(trial_events, STARTUP_METRICS)
    startup_by_group = _startup_summary_by_group(trial_events)
    saved_state_rows = _saved_state_trial_rows(trial_events)
    saved_state_public_rows = _public_saved_state_rows(saved_state_rows)
    saved_state_summary = _saved_state_summary_from_rows(saved_state_rows)
    startup_timeline_summary = _startup_timeline_summary_from_rows(saved_state_rows)
    startup_timeline_unattributed = _startup_timeline_unattributed_summary(saved_state_rows)
    first_simulation_compute_summary = _first_simulation_compute_summary_from_rows(saved_state_rows)
    first_simulation_compute_unattributed = _first_simulation_compute_unattributed_summary(
        saved_state_rows
    )
    first_simulation_worker_handoff_summary = _first_simulation_worker_handoff_summary_from_rows(
        saved_state_rows
    )
    theta_summary = _summary_for_grouped_theta_metrics(
        trial_events,
        THETA_METRICS,
        transition="to_10",
    )
    theta_return_summary = _summary_for_grouped_theta_metrics(
        trial_events,
        THETA_METRICS,
        transition="return",
    )
    redraw_summary = _summary_for_grouped_redraw_metrics(trial_events, REDRAW_METRICS)
    redraw_validation = _redraw_validation_counts(trial_events)
    rows = _theta_rows(trial_events)
    result_rows = _trial_results_summary(trial_results)
    failure_rows = _trial_failures_summary(trial_results)
    summary = {
        "metadata": metadata,
        "startup": startup_summary,
        "startup_by_group": startup_by_group,
        "saved_state_startup": {
            "phase_breakdown": saved_state_summary,
            "startup_timeline": {
                "phase_breakdown": startup_timeline_summary,
                "total_unattributed_gap_ms": startup_timeline_unattributed,
            },
            "first_simulation_compute_breakdown": {
                "phase_breakdown": first_simulation_compute_summary,
                "total_unattributed_gap_ms": first_simulation_compute_unattributed,
            },
            "first_simulation_worker_handoff": {
                "phase_breakdown": first_simulation_worker_handoff_summary,
                "conclusion": _first_simulation_worker_handoff_conclusion(
                    first_simulation_worker_handoff_summary
                ),
            },
            "trial_rows": saved_state_public_rows,
        },
        "theta_change": theta_summary,
        "theta_return": theta_return_summary,
        "redraw_only": redraw_summary,
        "redraw_validation": redraw_validation,
        "theta_rows": rows,
        "trial_results": result_rows,
        "trial_failures": failure_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_combined_csv(output_dir, output_events)
    _write_report(
        output_dir,
        metadata,
        startup_summary,
        startup_by_group,
        saved_state_summary,
        startup_timeline_summary,
        startup_timeline_unattributed,
        first_simulation_compute_summary,
        first_simulation_compute_unattributed,
        first_simulation_worker_handoff_summary,
        saved_state_public_rows,
        theta_summary,
        theta_return_summary,
        redraw_summary,
        redraw_validation,
        rows,
        result_rows,
        failure_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="defaults",
        choices=[
            "defaults",
            "theta10",
            "redraw-only",
            "cache-hit",
            "saved-state-startup",
        ],
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_root / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = str(args.scenario)
    state = args.state
    if scenario == "saved-state-startup":
        scenario = "defaults-restored"
        state = state or DEFAULT_STATE_PATH
    metadata_args = argparse.Namespace(**vars(args))
    metadata_args.scenario = scenario
    metadata_args.state = state
    metadata = _metadata(output_dir, metadata_args)
    launch_count, repetitions = _scenario_launch_count(scenario, int(args.trials))
    run_specs = _build_run_specs(scenario, launch_count, repetitions)
    trial_results = _run_specs(
        output_dir=output_dir,
        scenario=scenario,
        run_specs=run_specs,
        state=state,
        timeout_s=float(args.timeout),
    )
    all_trial_events = [_read_events(result.path) for result in trial_results]
    measured_results = [result for result in trial_results if result.measured]
    trial_events = [_read_events(result.path) for result in measured_results]
    _write_summary(
        output_dir, metadata, trial_events, trial_results, all_run_events=all_trial_events
    )
    print(output_dir)
    return _harness_exit_code(trial_results)


if __name__ == "__main__":
    raise SystemExit(main())
