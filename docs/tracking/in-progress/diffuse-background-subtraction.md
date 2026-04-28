# Diffuse background subtraction

Status: implemented, validation-partial
Type: feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-04-28

## Summary

RA-SIM now has a shared tunable diffuse-background subtraction path for
experimental detector backgrounds. The feature is disabled by default. When
enabled, the same NumPy/SciPy model is available to GUI preview, GUI fitting,
headless geometry fitting, and CLI-driven mosaic-shape fitting.

The model estimates a robust radial two-theta halo, optionally adds a slow
caked two-dimensional residual, masks local peaks and the direct beam before
fitting, and keeps signed corrected intensities for numerical comparison.
Display clipping is limited to display paths.

## Current state

- Added `ra_sim/fitting/diffuse_background.py` with config mapping helpers,
  valid/exclusion mask builders, radial and caked residual estimators, detector
  evaluation, native fitting, and subtraction.
- Added disabled-by-default config under
  `instrument.fit.geometry.background_subtraction`.
- Added a GUI `Background` tab with controls for enable/mode/apply flags,
  scale, radial/caked model parameters, masking parameters, diagnostics, and
  raw/model/subtracted/residual/mask previews.
- Wired corrected backgrounds into GUI caking, comparison, analysis, auto-match,
  and geometry-fit payloads only when `apply_to_fit` is enabled.
- Added subtraction-aware cache signatures and invalidation so caked background
  data is not reused after model/control changes.
- Added headless and CLI support for saved/off/radial/radial-plus-caked-2d
  modes, scale overrides, and diagnostics artifact writing.
- Preserved raw background access for debugging/export and left existing raw
  behavior unchanged when subtraction is disabled.
- Added compatibility fixes around manual geometry/caked payload hydration that
  were exposed by the runtime safety suite during this cross-cutting change.

## Validation

Passed:

- `python -m compileall ra_sim tests`
- `python -m pytest tests/test_diffuse_background.py -ra`
- `python -m pytest tests/test_background_peak_matching.py -ra`
- `python -m pytest tests/test_gui_views.py -ra`
- `python -m pytest tests/test_gui_state_io.py -ra`
- `python -m pytest tests/test_cli_geometry_fit.py -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra`
- Combined targeted suite for the six files above: 419 passed.
- Manual-geometry regression selectors and full
  `tests/test_gui_runtime_import_safe.py`: 313 passed.
- Pure off-mode validation confirmed corrected output remains equal to raw.
- Synthetic diffuse-background validation passed with relative model error
  about 1.9 percent and peak centroid shift about 0.14 px.

Full-suite status:

- Plain `python -m pytest -ra` did not complete cleanly in this worktree.
  The base worktree also failed/crashed during reporting.
- Comparable `--tb=no` runs showed no candidate-only failures versus base:
  base had 290 failures, candidate had 261 failures, and candidate-only failure
  count was zero.

Not validated locally:

- Real-background smoke tests for off/radial/radial-plus-caked-2d modes.
- Manual GUI preview/orientation checks.
- Headless saved-state override checks using a real saved GUI state.
- Diagnostic artifact sanity checks on real detector data.

Those checks need a real raw detector background and saved GUI state. No such
input was present in this worktree during validation.

## Next actions

- Run the real-background smoke matrix when a detector background and saved GUI
  state are available.
- Inspect generated subtraction diagnostics on real data for valid/masked
  fraction, radial profile smoothness, and Bragg leakage into the model.
- Exercise the GUI Background tab manually for preview orientation, saved-state
  round-trip, and cache invalidation after parameter changes.

## Links

- `ra_sim/fitting/diffuse_background.py`
- `tests/test_diffuse_background.py`
