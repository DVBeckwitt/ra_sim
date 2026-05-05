# Remove Global Background Subtraction

Status: completed
Type: bug/feature
Owner: Codex
Issue: none
Priority: p1
Last updated: 2026-05-05

## Summary

Removed the global diffuse/background-subtraction workflow because it changed
image-wide detector and caked data outside the local peak-fit task that needed
baseline correction.

## Current state

The top-level Background tab is gone. Background loading, switching, and theta
controls remain in Setup > Backgrounds.

Detector display, caked display, matching, manual picks, geometry fit, and
headless geometry-fit inputs now use the raw loaded background image. Legacy
background-subtraction CLI options and saved GUI variables still load, but they
are ignored and warn once when requesting a non-off old mode.

Analyze peak fitting now owns the only subtraction path. The Analyze peak tools
panel has `Subtract linear background`, default enabled. When `Fit Selected
Peaks` runs, each selected source/box fits a local 2D plane over finite caked ROI
pixels with peak-core exclusion, subtracts it only from local fit data, and
stores the linear-background metadata in result rows.

Bug/error status: no known open bug from this removal after focused validation.
Feature status: complete for the requested Analyze-only local linear
background correction. Compatibility status: old saved states remain loadable;
old global subtraction settings are legacy no-op data.

## Next actions

None required for this change. Future baseline models should stay local to the
Analyze fitting operation unless a separate user-facing workflow explicitly
requires image-wide correction.

## Validation

- `python -m pytest tests/test_gui_analysis_peak_tools.py tests/test_gui_views.py tests/test_gui_state_io.py tests/test_cli_geometry_fit.py -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_runtime_fit_analysis.py -ra`
- `python -m compileall ra_sim tests`
- `python -m ra_sim.dev check`

All listed commands passed on 2026-05-05.

## Links

- `README.md`
- `docs/gui-workflow.md`
- `docs/testing-and-validation.md`
- `CHANGELOG.md`
