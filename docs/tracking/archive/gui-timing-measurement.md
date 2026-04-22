# RA-SIM GUI timing measurement

Status: resolved
Type: feature
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-22

## Summary

Added gated high-resolution timing for real `python -m ra_sim gui` runs and a
GUI harness that records raw JSONL, combined CSV, summary JSON, and Markdown
reports. Timing is disabled by default and only runs when `RA_SIM_TIMING=1` is
set.

## Current state

Resolved. The GUI runtime now emits startup, simulation, rendering, visible
frame, metadata, automation, and display-fingerprint events through the real
Tk/Matplotlib update path. The harness can run defaults, theta-change,
redraw-only, cache-hit, and saved-state startup scenarios.

Review findings 1-7 are fixed:

- Failed or timed-out child GUI trials now make the harness return nonzero
  after artifacts are written.
- Startup warm trials are labeled `warm_fresh_process` only after a separate
  warmup GUI process is run.
- Child GUI metadata provides the report-visible Matplotlib backend, Tk/Tcl
  versions, canvas class, display size, scaling, and window geometry.
- Markdown tables include `missing_count` and compact raw duration columns;
  full raw arrays remain in `summary.json`.
- A failed warmup stops later warm measured trials from being run or
  summarized as warm.
- Metadata can be merged from all run JSONL files, including unmeasured warmup
  runs, while metrics stay measured-only.
- `combined_events.csv` includes all attempted run events, including warmup
  and failed or unmeasured runs.

## Next actions

- Run the full requested trial matrix when machine time is available:
  defaults 10, theta10 30, redraw-only 30, cache-hit, and saved-state startup.
- Update `docs/tracking/index.md` after unrelated staged tracking edits are
  reconciled.

## Validation

- 2026-04-22: `python -m py_compile ra_sim/timing.py ra_sim/launcher.py
  ra_sim/gui/_runtime/runtime_session.py scripts/measure_gui_timing.py
  tests/test_timing.py` passed.
- 2026-04-22: `python -m ruff format --check ra_sim/timing.py
  ra_sim/launcher.py ra_sim/gui/_runtime/runtime_session.py
  scripts/measure_gui_timing.py tests/test_timing.py` passed.
- 2026-04-22: `python -m ruff check ra_sim/timing.py ra_sim/launcher.py
  ra_sim/gui/_runtime/runtime_session.py scripts/measure_gui_timing.py
  tests/test_timing.py` passed.
- 2026-04-22: `python -m pytest -q tests/test_timing.py
  tests/test_gui_runtime_update_trace.py` passed with 17 tests.
- 2026-04-22: `python scripts/measure_gui_timing.py --scenario defaults
  --trials 6` passed and produced warmup-aware artifacts.
- 2026-04-22: The generated `combined_events.csv` from that run contains
  `warmup_process` events.
- 2026-04-22: `python -m ra_sim.dev check` still fails only on unrelated
  formatting in `ra_sim/fitting/optimization.py`, which was not changed as
  part of this timing work.

## Links

- Issue: none
- Timing helper: [ra_sim/timing.py](../../../ra_sim/timing.py)
- Runtime instrumentation:
  [ra_sim/gui/_runtime/runtime_session.py](../../../ra_sim/gui/_runtime/runtime_session.py)
- Harness: [scripts/measure_gui_timing.py](../../../scripts/measure_gui_timing.py)
- Tests: [tests/test_timing.py](../../../tests/test_timing.py)
- Validation artifact:
  [artifacts/perf/gui_timing/20260422_130301](../../../artifacts/perf/gui_timing/20260422_130301)
