# Analyze Linear Background Plane Fix

Status: completed
Type: bug
Owner: Codex
Issue: none
Priority: p1
Last updated: 2026-05-05

## Summary

Analyze peak fitting could oversubtract the local linear background as a
function of `2theta`. The core helper already used a 2D plane in `2theta` and
`phi`, but the runtime path projected corrected boxes through a source-wide
sparse image and the plane fit could silently include peak-contaminated pixels.

## Current state

The Analyze-only background correction now fits each selected caked ROI with a
local 2D plane and projects radial/azimuth curves from that corrected ROI's own
bounds. It no longer uses union bounds from unrelated corrected windows.

The plane fitter rejects positive peak-wing residuals before the final plane
fit. If too few non-peak background samples remain, subtraction fails closed and
the peak fit continues with the original uncorrected 1D curve while recording
that background subtraction was not applied.

Bug/error status: fixed. Feature status: complete for the existing Analyze
`Subtract linear background` workflow. No new GUI control was added.

## Next actions

None required. Future Analyze background changes should keep subtraction local
to the selected caked ROI and avoid mutating detector or cached caked images.

## Validation

- `python -m pytest tests/test_gui_analysis_peak_tools.py -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "analysis_linear_background or fit_selected_analysis_peaks_uses_selected_integration_window_only" -ra`
- `python -m pytest tests/test_gui_analysis_peak_tools.py tests/test_gui_runtime_import_safe.py -ra`
- `python -m compileall ra_sim tests`
- `python -m ra_sim.dev check`

All commands passed on 2026-05-05.

## Links

- `CHANGELOG.md`
- `ra_sim/gui/analysis_peak_tools.py`
- `ra_sim/gui/_runtime/runtime_session.py`
