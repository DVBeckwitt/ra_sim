# Analyze peak-fit table and ROI scale

Status: resolved
Type: bug/feature
Owner: -
Issue: none
Priority: p2
Last updated: 2026-04-30

## Summary

Analyze peak fitting now reports radial and azimuthal fit metrics in a readable
table, and integration-region changes no longer change the main caked image
intensity scale.

## Current state

- Peak-fit summaries use a monospaced table instead of dense inline text.
- Fit rows include model, fit center, Gaussian FWHM, Lorentzian FWHM,
  Gaussian/Lorentzian mixture percent, and RMSE.
- Pseudo-Voigt rows report the shared FWHM in both FWHM columns and show the
  Gaussian/Lorentzian split from area-normalized Gaussian and Lorentzian
  equations, so the Lorentzian percentage is an area fraction rather than a
  height-weighted mix.
- Standard integration-range refreshes skip the main caked raster repaint when
  the cached caked image and intensity mode are already current.
- The 2theta and phi 1D integration plots still recompute from the selected ROI
  and autoscale their axes after each region change.

## Status

- Feature status: complete.
- Bug status: fixed for standard caked integration-region updates and
  pseudo-Voigt Lorentzian-percentage reporting.
- Error status: no known runtime errors from the focused regression coverage.
- Compatibility: no saved-state, CLI, config, or artifact schema changes.

## Validation

- `python -m pytest tests/test_gui_analysis_peak_tools.py tests/test_gui_views.py -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_integration_range_drag.py -ra`
- `python -m compileall ra_sim tests`
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py tests/test_gui_runtime_import_safe.py`

`python -m ra_sim.dev check` still stops at the repository formatter gate
because the dirty worktree has whole-file format drift in
`ra_sim/fitting/optimization.py` and `ra_sim/gui/_runtime/runtime_session.py`.
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
