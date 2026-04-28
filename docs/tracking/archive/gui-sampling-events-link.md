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
events-per-phase value is also 75.

## Current State

Feature status: shipped in repo. The `Events per beam phase` slider is disabled
while linked and mirrors sample-count changes. The `Independent` checkbutton
beside the events control enables the events slider so operators can diverge
events from samples explicitly.

Bug/error status: no open bug remains for this request. The lower-level
diffraction/backend default remains unchanged; this is a GUI control behavior
change.

## Validation

- `python -m pytest tests\test_gui_views.py tests\test_gui_sim_signature.py`:
  66 passed.
- `python -m ruff check ra_sim\gui\state.py ra_sim\gui\views.py ra_sim\gui\_runtime\runtime_session.py tests\test_gui_views.py tests\test_gui_sim_signature.py`:
  passed.
- Broader `tests\test_gui_runtime_import_safe.py` was not green in this
  worktree because two existing update-dependency/prune-cache tests fail in
  unrelated dirty changes.
