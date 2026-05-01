# Detector Selected-Qr Rod ROI

Status: resolved
Type: bug/feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-01

## Summary

Selected-Qr Rod ROI had two separate requirements that were previously
conflated:

- detector view should use detector-native Qr/Qz/phi masks for overlay and drag;
- plotted Qz rod profiles should use the caked `2theta/phi` data path.

The follow-up UI/performance work also needed explicit multi-rod selection,
debounced refresh, shared detector/caked inputs, bounded caches, reusable 1D
plot artists, optional profiling, detector-view raw-sum defaults, and clearer
`delta_Qr` semantics.

## Current state

Resolved. Detector view now keeps detector-native masks for display and drag
only. Selected-Qr rod Qz profiles are integrated from caked data in both
detector and caked views; runtime selected-rod plotting and auto-match do not
fall back to detector-pixel profile integration.

Multiple rods are selected through a stable extended-selection rod list instead
of modifier-key selection or one widget per rod. `analysis_range.selected_qr_rod_keys`
stores selected rods in displayed list order. Legacy
`analysis_range.selected_qr_rod_key` mirrors the first selected key or `""`.

Overlay and drag use OR-union masks across selected rods. Numerical profiles
stay per-rod: each stacked Qz subplot uses that rod's own caked mask, not the
union mask.

Fresh detector-view Selected-Qr rod mode defaults `Rod profile intensity` to
`Raw accumulated intensity`. Fresh caked-view rod mode keeps
support-normalized density. Restored or user-edited intensity modes are marked
custom and are not overwritten.

The `Delta Qr width (A^-1)` control now means full rod width in the GUI and
saved state. Current saves write
`analysis_range.delta_qr_width_mode = "full_width"`. Older saved states without
that marker are treated as legacy half-width values and doubled on load. Low
level caked/detector mask and drag builders still receive half-width at the
runtime boundary.

Fresh selected-rod phi defaults remain `phi_min=-90`, `phi_max=90` unless a
custom/restored selected-rod phi range is active.

Performance path is now consolidated:

- selection changes update state immediately and debounce expensive refresh;
- detector overlay/drag builds shared detector geometry once per refresh;
- caked Qz profiles build shared caked inputs once per simulation/background
  profile update;
- stored-hit shape masks use a bounded signature cache instead of per-rod table
  scans;
- detector/caked union masks and per-rod profiles use dedicated bounded caches;
- selected-rod 1D plots reuse axes and `Line2D` artists when the rod count and
  keys permit;
- optional `RA_SIM_PROFILE_QR_ROD=1` timings are retained in bounded internal
  debug lists and do not affect cache signatures or debounce behavior.

## Bug/error/feature status

- Bug: fixed. Detector-native masks no longer supply plotted Selected-Qr rod
  Qz profiles.
- Bug: fixed. Detector view no longer falls back to the angular rectangular ROI
  while selected-rod mode is enabled.
- Feature: complete. Extended-list multi-selection, per-rod stacked Qz
  profiles, union overlay/drag masks, detector raw-sum default, phi defaults,
  full-width `delta_Qr` saved-state semantics, bounded caches, plot reuse, and
  optional profiling are implemented.
- Migration: complete. Legacy `selected_qr_rod_key` still loads into
  `selected_qr_rod_keys`, and legacy half-width `delta_qr` loads as full width.
- Error status: no known selected-Qr rod runtime errors remain after focused
  validation below.
- Known unrelated error: the HKL grouped-cache regression remains red in
  `tests/test_gui_runtime_import_safe.py::test_runtime_session_hkl_pick_builds_grouped_cache_from_stored_raw_peak_rows`.

## Validation

- 2026-05-01 final selected-Qr validation:
  `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py tests/test_gui_state_io.py -ra`
  completed with `545 passed` and the known unrelated HKL grouped-cache failure.
- 2026-05-01: `python -m compileall ra_sim tests` passed.
- 2026-05-01: `git diff --check` passed with CRLF normalization warnings only.
- 2026-05-01 profiling smoke:
  `RA_SIM_PROFILE_QR_ROD=1 python -m pytest tests/test_gui_runtime_import_safe.py -k "qr_rod" -ra`
  passed, `33 passed`.
- Manual GUI checklist from the implementation plan was not run in this
  automated pass.

## Links

- GUI controls: [ra_sim/gui/views.py](../../../ra_sim/gui/views.py)
- Runtime: [ra_sim/gui/_runtime/runtime_session.py](../../../ra_sim/gui/_runtime/runtime_session.py)
- Drag support: [ra_sim/gui/integration_range_drag.py](../../../ra_sim/gui/integration_range_drag.py)
- Workflow docs: [docs/gui-workflow.md](../../gui-workflow.md)
