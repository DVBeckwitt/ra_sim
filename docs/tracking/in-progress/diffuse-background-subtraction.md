# Diffuse background subtraction

Status: phi-block implemented, validation-partial
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

The model estimates a robust radial two-theta halo, can add a coarse
phi/theta block residual model for broad detector-module artifacts, can then
add a slow caked two-dimensional residual, masks local peaks and the direct
beam before fitting, and keeps signed corrected intensities for numerical
comparison. Display clipping is limited to display paths.

## Current state

- Added `ra_sim/fitting/diffuse_background.py` with config mapping helpers,
  valid/exclusion mask builders, radial and caked residual estimators, detector
  evaluation, native fitting, and subtraction.
- Added disabled-by-default config under
  `instrument.fit.geometry.background_subtraction`.
- Added a GUI `Background` tab with controls for enable/mode/apply flags,
  scale, radial/caked model parameters, masking parameters, diagnostics, and
  raw/model/subtracted/residual/mask previews.
- Refined the GUI `Background` tab into a workflow-oriented panel with presets,
  explained sliders, collapsible advanced controls, preview guidance, dirty
  status feedback, debounced auto-preview, and diagnostics summaries. This is
  a UI-only refinement over the existing subtraction model.
- Tightened the visible Background tab copy into a compact control-panel style:
  shorter labels, section names, preset names, button text, slider hints,
  status text, diagnostics labels, and tooltips. This was copy-only and did
  not change subtraction math, saved-state keys, CLI flags, headless behavior,
  or cache behavior.
- Wired corrected backgrounds into GUI caking, comparison, analysis, auto-match,
  and geometry-fit payloads only when `apply_to_fit` is enabled.
- Added subtraction-aware cache signatures and invalidation so caked background
  data is not reused after model/control changes.
- Added headless and CLI support for saved/off/radial/radial-plus-caked-2d
  modes, scale overrides, and diagnostics artifact writing.
- Added radial-plus-phi-blocks and
  radial-plus-phi-blocks-plus-caked-2d modes, GUI controls for phi-block
  tuning, CLI phi-block overrides, component previews, component return keys,
  and phi-block diagnostic artifacts.
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
- `python -m pytest tests/test_gui_views.py tests/test_gui_state_io.py tests/test_gui_runtime_import_safe.py tests/test_cli_geometry_fit.py tests/test_diffuse_background.py -ra`
- Combined targeted suite for the six files above: 419 passed.
- Combined targeted UX/headless/numerical suite: 417 passed.
- Copy-tightening targeted suite:
  `python -m pytest tests/test_gui_views.py tests/test_gui_state_io.py tests/test_gui_runtime_import_safe.py -ra`
  passed with 391 tests.
- Copy-tightening broader relevant suite:
  `python -m pytest tests/test_diffuse_background.py tests/test_background_peak_matching.py tests/test_gui_views.py tests/test_gui_state_io.py tests/test_cli_geometry_fit.py tests/test_gui_runtime_import_safe.py -ra`
  passed with 426 tests.
- Copy-tightening lint/whitespace checks:
  `python -m ruff check ra_sim/gui/views.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_views.py`
  and `git diff --check -- ra_sim/gui/views.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_views.py`
  passed.
- `python -m ruff check ra_sim/gui/views.py ra_sim/gui/state.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_views.py tests/test_gui_state_io.py tests/test_gui_runtime_import_safe.py`
- `python -m ruff format --check ra_sim/gui/views.py ra_sim/gui/state.py tests/test_gui_views.py tests/test_gui_state_io.py tests/test_gui_runtime_import_safe.py`
- Manual-geometry regression selectors and full
  `tests/test_gui_runtime_import_safe.py`: 313 passed.
- Pure off-mode validation confirmed corrected output remains equal to raw.
- Synthetic diffuse-background validation passed with relative model error
  about 1.9 percent and peak centroid shift about 0.14 px.
- Phi-block extension targeted validation passed:
  `python -m compileall ra_sim tests`,
  `python -m pytest tests/test_diffuse_background.py -ra`,
  `python -m pytest tests/test_gui_views.py -ra`,
  `python -m pytest tests/test_gui_state_io.py -ra`,
  `python -m pytest tests/test_cli_geometry_fit.py -ra`,
  `python -m pytest tests/test_gui_runtime_import_safe.py -ra`, and the combined
  six-file targeted suite passed with 429 tests.

Full-suite status:

- Plain `python -m pytest -ra` did not complete cleanly in this worktree.
  The base worktree also failed/crashed during reporting.
- After the UX refinement, full `python -m pytest -ra` was attempted twice and
  timed out before completion. Per operator direction, no further full-suite
  chase was performed for this patch.
- After the copy-tightening pass, full `python -m pytest -ra` was attempted
  again and timed out after 20 minutes with no completed result.
- After the phi-block extension, full `python -m pytest -ra` was attempted
  again and timed out after 20 minutes with no completed result.
- Comparable `--tb=no` runs showed no candidate-only failures versus base:
  base had 290 failures, candidate had 261 failures, and candidate-only failure
  count was zero.

Not validated locally:

- Real-background smoke tests for off/radial/radial-plus-caked-2d,
  radial-plus-phi-blocks, and radial-plus-phi-blocks-plus-caked-2d modes.
- Manual GUI preview/orientation checks.
- Manual hover-tooltip and preset interaction checks in a live Tk session.
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
