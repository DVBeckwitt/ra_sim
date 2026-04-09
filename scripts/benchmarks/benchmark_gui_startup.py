from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
IMPORT_MODULES = (
    "ra_sim.__main__",
    "ra_sim.gui.runtime",
    "ra_sim.gui._runtime.runtime_impl",
)
STARTUP_EVENT_PREFIX = "RA_SIM_STARTUP_EVENT "


def _measure_import_seconds(module_name: str) -> float:
    command = [
        sys.executable,
        "-c",
        (
            "import importlib, json, sys, time; "
            "t0 = time.perf_counter(); "
            "importlib.import_module(sys.argv[1]); "
            "print(json.dumps({'seconds': time.perf_counter() - t0}))"
        ),
        module_name,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    return float(payload["seconds"])


def _measure_gui_startup(timeout_s: float) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["RA_SIM_STARTUP_BENCHMARK"] = "1"
    env["RA_SIM_STARTUP_BENCHMARK_AUTO_EXIT"] = "1"

    command = [sys.executable, "-m", "ra_sim", "gui", "--no-excel"]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )

    captured_lines: list[str] = []
    event_times: dict[str, float] = {}
    event_payloads: dict[str, dict[str, object]] = {}

    try:
        assert process.stdout is not None
        while True:
            if (time.perf_counter() - started) > float(timeout_s):
                raise TimeoutError(f"GUI startup benchmark exceeded {timeout_s:.1f}s.")

            line = process.stdout.readline()
            if line == "" and process.poll() is not None:
                break
            if not line:
                time.sleep(0.01)
                continue

            captured_lines.append(line.rstrip())
            if line.startswith(STARTUP_EVENT_PREFIX):
                payload = json.loads(line[len(STARTUP_EVENT_PREFIX) :].strip())
                event_name = str(payload.get("event", "unknown"))
                event_times[event_name] = time.perf_counter() - started
                event_payloads[event_name] = payload
    finally:
        try:
            process.wait(timeout=max(1.0, float(timeout_s) / 4.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    return {
        "command": command,
        "returncode": int(process.returncode or 0),
        "events": event_times,
        "event_payloads": event_payloads,
        "stdout_tail": captured_lines[-40:],
    }


def _build_report(label: str | None, timeout_s: float) -> dict[str, object]:
    report = {
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "imports": {
            module_name: _measure_import_seconds(module_name)
            for module_name in IMPORT_MODULES
        },
        "gui": _measure_gui_startup(timeout_s),
    }
    return report


def _write_report(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            payload = existing
        else:
            payload = [existing]
    else:
        payload = []
    payload.append(report)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark RA-SIM GUI startup timings.")
    parser.add_argument("--label", default=None, help="Optional run label such as baseline or post-change.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to append the report to.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds to wait for the GUI benchmark subprocess.",
    )
    args = parser.parse_args(argv)

    report = _build_report(args.label, float(args.timeout))
    if args.output:
        _write_report(Path(args.output), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
