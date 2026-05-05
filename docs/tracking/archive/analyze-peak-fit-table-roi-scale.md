# Analyze peak-fit table and ROI scale

Status: resolved
Type: bug/feature
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-05

## Summary

Analyze peak fitting reports radial and azimuthal fit metrics in a readable
table, uses a tail-aware independent-width mosaic-mix profile, and keeps
integration-region changes from changing the main caked image intensity scale.

## Current state

- Peak-fit summaries use a monospaced table instead of dense inline text.
- Fit rows include model, fit center, Gaussian FWHM, Lorentzian FWHM,
  Gaussian/Lorentzian mixture percent, and RMSE.
- The legacy Analyze Pseudo-Voigt path was removed from GUI state, controls,
  runtime selection, helper support, and focused tests.
- Analyze radial and azimuthal peak fitting now exposes no profile-family
  toggles and always uses the `Mosaic mix` profile.
- Mosaic mix rows report independent Gaussian-core and Lorentzian-tail FWHM
  values with the Gaussian/Lorentzian area split from `eta`; the fit also
  records weighted RMSE from the tail-aware objective.
- Standard integration-range refreshes skip the main caked raster repaint when
  the cached caked image and intensity mode are already current.
- The 2theta and phi 1D integration plots still recompute from the selected ROI
  and autoscale their axes after each region change.

## Status

- Feature status: complete for replacing Analyze Pseudo-Voigt with Mosaic mix.
- Bug status: fixed for standard caked integration-region updates and
  Analyze shared-width core/tail underfitting.
- Error status: no known runtime errors from the focused regression coverage.
- Compatibility: no saved-state, CLI, config, or artifact schema changes.

## Validation

- `python -m pytest tests/test_gui_analysis_peak_tools.py tests/test_gui_views.py tests/test_gui_runtime_import_safe.py -ra`
- `python -m compileall ra_sim tests`
- `python -m ruff check ra_sim/gui/analysis_peak_tools.py ra_sim/gui/views.py ra_sim/gui/state.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_analysis_peak_tools.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py`

`python -m ra_sim.dev check` still stops at the repository formatter gate
because the dirty worktree has whole-file format drift in
`ra_sim/gui/_runtime/runtime_session.py`.
The runtime file was not whole-file reformatted to avoid touching unrelated
local edits.

## Links

- `docs/gui-workflow.md`
- `CHANGELOG.md`
- `ra_sim/gui/analysis_peak_tools.py`
- `ra_sim/gui/_runtime/runtime_session.py`
- `ra_sim/gui/views.py`
- `tests/test_gui_analysis_peak_tools.py`
- `tests/test_gui_runtime_import_safe.py`
