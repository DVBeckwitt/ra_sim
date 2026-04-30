# Q-space viewer fix

Status: implemented, broader check blocked
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-04-30

## Summary

Q-space viewer now owns the geometry that produced the current simulation image.
Changing detector distance invalidates simulation caches, Q-space conversion uses
the submitted simulation geometry instead of live widgets, and Q-space-only view
can update without running the caked-analysis path.

## Current state

- `distance_m` participates in the simulation signature.
- Submitted simulation Q-space geometry is copied into runtime state with copied
  center coordinates and cleared with the simulation image it describes.
- Q-space conversion routes through stored image-owner geometry, then
  detector-center remap geometry, then live controls only as fallback.
- `analysis_sig[2]` remains `q_space_geometry_sig`; request flags after it are
  plain `bool`.
- Q-space-only display uses 512 by 512 interactive bins and does not change
  caked, 1D, or export bin behavior.
- Q-space-only freshness requires matching Q-space payload/signature but not
  `last_res2_sim`.
- Display Qr filtering removes non-finite or non-positive centers and filters
  only arrays whose last axis matches the original Qr length.
- Timing logs stay gated and silent by default.

## Next actions

- Re-run `python -m ra_sim.dev check` after the unrelated formatting drift in
  `ra_sim/fitting/optimization.py` is resolved.

## Validation

- `python -m pytest tests/test_q_space_viewer_runtime.py -ra` passed.
- `python -m pytest tests/test_gui_runtime_update_actions.py -ra` passed.
- `python -m pytest tests/test_gui_sim_signature.py tests/test_gui_runtime_import_safe.py -ra` passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/state.py tests/test_q_space_viewer_runtime.py tests/test_gui_runtime_import_safe.py` passed.
- `python -m ruff format --check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/state.py tests/test_q_space_viewer_runtime.py tests/test_gui_runtime_import_safe.py` passed.
- `python -m ra_sim.dev check` did not complete because pre-existing dirty
  `ra_sim/fitting/optimization.py` needs formatting.

## Links

- Runtime: `ra_sim/gui/_runtime/runtime_session.py`
- State: `ra_sim/gui/state.py`
- Tests: `tests/test_q_space_viewer_runtime.py`
