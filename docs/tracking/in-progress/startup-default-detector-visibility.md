# Startup default detector visibility regression

Status: in-progress
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-18

## Summary

Cold start in default settings can open with detector simulation missing.
Changing any simulation parameter makes it appear. Switching to caked also
reveals it. Returning to default values makes it disappear again. Prior redraw
fix did not resolve user-visible bug.

## Current state

Scope is startup/default-settings detector view only. Simulation generation
appears healthy because caked view and parameter perturbation recover it.
Likely remaining fault is detector/default-state presentation or
default-signature handling, not general simulation computation.

## Next actions

- Reproduce from cold start with untouched defaults.
- Compare detector state at cold start vs after parameter perturbation vs after
  revert-to-default.
- Trace detector artist/signature/default-parameter path in
  `ra_sim/gui/_runtime/runtime_session.py`.
- Verify whether default-value path suppresses redraw or clears visible
  detector state.

## Validation

- Cold start with defaults must show detector simulation immediately.
- Changing a parameter must keep simulation visible.
- Reverting to exact startup defaults must still keep simulation visible.
- Caked toggle must not be required.

## Links

- Issue: none
- Runtime path: [ra_sim/gui/_runtime/runtime_session.py](../../ra_sim/gui/_runtime/runtime_session.py)
- Test path: [tests/test_gui_runtime_import_safe.py](../../tests/test_gui_runtime_import_safe.py)
- Tracking index: [docs/tracking/index.md](../index.md)
