# Analyze overlay hardening

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-20

## Summary

Analyze-mode composite peak fitting kept the intended selected-window fit-domain
behavior, but two runtime edges needed hardening:

- the overlay renderer needed to keep drawing stored fit curves even when
  `selected_peaks` was empty;
- isolated runtime-function loading in `tests/test_gui_runtime_import_safe.py`
  needed `Sequence` in its exec namespace so production annotations resolved.

This fix resolved those edges without changing composite-fit input semantics.

## Current state

- radial and azimuth composite fits still use all finite samples inside the
  current selected 1D integration window only;
- no peak-local crop was reintroduced;
- no bins outside the active Analyze selection are used for fitting;
- `_render_analysis_peak_overlays()` now treats stored radial/azimuth fit
  results as overlay content, so full-span fit curves still render even when no
  selected peaks are present;
- fit plotting still prefers `x_fit`/`y_fit`, falls back to `x_window` only
  when `x_fit is None`, skips `plot_fit is False`, and plots one line per
  `fit_group_id`;
- wrapped azimuth fit curves still break across excluded gaps;
- clearing Analyze peak selection clears both stored fit results and live
  artist references, so stale overlays do not survive a clear action;
- the isolated runtime-function loader now injects `Sequence`, matching the
  production module environment closely enough for annotation evaluation.

## Next actions

None for this bug. If future Analyze refactors touch overlay-state storage or
fit-domain selection, keep the selected-window semantics and the clear-selection
artist cleanup tests intact.

## Validation

Validated on 2026-04-20 with:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_runtime_import_safe.py::test_render_analysis_peak_overlays_breaks_wrapped_fit_curve_gap tests/test_gui_runtime_import_safe.py::test_analysis_curve_selected_window_returns_empty_when_selection_excludes_curve tests/test_gui_runtime_import_safe.py::test_clear_selected_analysis_peaks_clears_stale_fit_results`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_runtime_import_safe.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_analysis_peak_tools.py`
- `python -m py_compile ra_sim/gui/analysis_peak_tools.py ra_sim/gui/_runtime/runtime_session.py`

All passed in the local workspace after the hardening changes.

## Links

- Runtime implementation: `ra_sim/gui/_runtime/runtime_session.py`
- Regression coverage: `tests/test_gui_runtime_import_safe.py`
- Related fit helpers: `ra_sim/gui/analysis_peak_tools.py`
- Tracking hub: [docs/tracking/index.md](../index.md)
