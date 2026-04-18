# Startup default detector visibility regression

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-18

## Summary

Cold start in default settings can open with detector simulation missing.
Changing any simulation parameter makes it appear. Switching to caked also
reveals it. Returning to default values makes it disappear again. Prior redraw
fix did not resolve user-visible bug. The cached blank-detector projection path
was patched, and the user confirmed the startup/default disappearance is fixed.

## Current state

Resolved. Scope was startup/default-settings detector view only. Simulation
generation itself stayed healthy because caked view and parameter perturbation
could recover the image; the failure lived in detector/default-state
presentation and default-signature handling.

The detector redraw path was projecting `global_image_buffer` before the new
`simulation_runtime_state.unscaled_image` was copied into that buffer. Because
`_store_primary_raster_source()` assigns `global_image_buffer` the semantic
detector source signature, `display_projection.project_raster_to_view()` could
cache a detached blank projection under the startup/default signature. Later
scale application copied the correct data into `global_image_buffer`, but the
projection cache reused the blank default-signature raster. Perturbing a
parameter changed the signature and made a nonblank stale buffer visible.
Returning to exact defaults reused the blank default-signature projection.

Patch applied: `ra_sim/gui/_runtime/runtime_session.py` now refreshes the
scaled detector buffer from `unscaled_image` before any detector projection can
cache under the current detector signature. The scale-factor redraw path uses
the same helper. User follow-up confirmed this resolved the visible startup
regression, so this tracking item is archived.

## Next actions

- None for repo-local tracking.
- Reopen only if startup/default detector visibility regresses again.

## Validation

- User confirmed the detector simulation now appears correctly at cold start
  with defaults and no longer disappears when returning to exact defaults.
- 2026-04-18: `python3 -m py_compile ra_sim/gui/_runtime/runtime_session.py`
  and `tests/test_gui_runtime_import_safe.py` passed.
- 2026-04-18: Added a regression test for refreshing the detector buffer
  before projection-cache use. Full pytest collection was not completed
  in this environment because importing the runtime module hangs during heavy
  GUI/simulation dependency import.

## Links

- Issue: none
- Runtime path: [ra_sim/gui/_runtime/runtime_session.py](../../../ra_sim/gui/_runtime/runtime_session.py)
- Test path: [tests/test_gui_runtime_import_safe.py](../../../tests/test_gui_runtime_import_safe.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
