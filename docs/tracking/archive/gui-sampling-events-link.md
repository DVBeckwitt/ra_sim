# GUI Sampling Events Link

Status: completed
Type: feature
Owner: -
Issue: none
Priority: p2
Last updated: 2026-04-28

## Summary

The GUI sampling control now treats events per beam phase as linked to the beam
phase sample count by default. The default sample count is 75, and the initial
events-per-phase value is 150. GUI state and legacy `parameters.npy` loads that
do not include an explicit sample count also default back to 75 samples.

## Current State

Feature status: shipped in repo. The `Events per beam phase` slider is disabled
while linked and mirrors sample-count changes at two events per sample. The
`Independent` checkbutton beside the events control enables the events slider so
operators can diverge events from samples explicitly.

Bug/error status: fixed. The stale one-event-per-sample dependent default and
missing-load fallback are covered by focused regressions. The lower-level
diffraction/backend default remains unchanged; this is a GUI control and load
behavior change.

## Validation

- `python -m pytest tests/test_data_loading_parameters.py tests/test_gui_sim_signature.py tests/test_gui_views.py -q`:
  74 passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_state_io.py -q`:
  329 passed.
- `python -m ruff check ra_sim/io/data_loading.py ra_sim/gui/_runtime/runtime_session.py tests/test_data_loading_parameters.py tests/test_gui_sim_signature.py`:
  passed.
- `python -m py_compile ra_sim/gui/_runtime/runtime_session.py ra_sim/io/data_loading.py tests/test_data_loading_parameters.py tests/test_gui_sim_signature.py`:
  passed.
- `python -m ra_sim.dev check`: still blocked by existing repo-wide formatter
  drift in unrelated files and broad `runtime_session.py` formatting regions.
