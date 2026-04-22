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


def _metric_pairing(metric: str) -> str:
    return METRIC_PAIRING.get(metric, "first_after_start")


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
