# Background peak-fit diagnostics

Status: resolved
Type: diagnostic workflow
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-30

## Summary

`scripts/diagnostics/all_background_peak_fits.ipynb` now supports faster
reruns across saved GUI states and uses a joint Qz peak model for Qr-rod
profile diagnostics. The notebook remains a diagnostic artifact generator, not
a GUI fitting path.

## Current state

The notebook has a parameter cell for `GUI_STATE_PATH`, `OUTPUT_DIR`,
`RUN_NAME`, fit workers, and Numba workers. It still honors
`RA_SIM_ALL_BACKGROUND_STATE`, `RA_SIM_ALL_BACKGROUND_OUT_DIR`, and related
environment variables when the parameter cell is blank.

`scripts/diagnostics/run_all_background_peak_fits.py` executes the notebook for
one or more saved GUI state files. A single-state run writes to the requested
output directory, while multi-state runs create `<state-stem>_state`
subdirectories to avoid artifact collisions.

Qr-rod Qz profile fits now fit all projected branch-point Gaussian peaks for a
rod/branch simultaneously. This replaces independent per-marker profile fitting
for the diagnostic overlay and reduces false peak-height inflation when close
projected peaks overlap.

## Bug/error/feature status

- Bug: fixed. Close Qz markers no longer get independently overfit in a way that
  fills the valley between peaks and overstates overlap.
- Error: no known notebook static-regression errors remain after focused tests.
  The notebook itself was not executed end-to-end in this patch.
- Feature: complete for scripted diagnostic reruns. Batch state execution,
  per-state output routing, and parameter-cell overrides are available.

## Validation

- 2026-04-30: `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  passed.
- 2026-04-30: `python -m compileall scripts/diagnostics/run_all_background_peak_fits.py tests/test_background_peak_fits_notebook.py`
  passed.

## Links

- Notebook: [scripts/diagnostics/all_background_peak_fits.ipynb](../../../scripts/diagnostics/all_background_peak_fits.ipynb)
- Runner: [scripts/diagnostics/run_all_background_peak_fits.py](../../../scripts/diagnostics/run_all_background_peak_fits.py)
- Tests: [tests/test_background_peak_fits_notebook.py](../../../tests/test_background_peak_fits_notebook.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
