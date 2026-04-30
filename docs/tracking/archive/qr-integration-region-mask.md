# Qr integration region mask

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-30

## Summary

Selected-Qr rod caked integration masks could still miss real high-`|phi|`
signal after trace gap interpolation and span-fill repairs, especially around
`phi = -72..-85` degrees. The caked image and exact-cake LUT had detector
contributors there, but the forward-projected analytic Qr center trace could
drop out before any fill logic could classify the bin.

## Current state

Resolved. Selected-Qr rod caked masks now use detector pixels and the
exact-cake LUT as the primary source of truth. The mask path builds cached
detector Qr/Qz maps with the same geometry/refraction convention as the Qr
cylinder projection, selects detector pixels by `Qr0 +/- delta_Qr` and Qz
bounds, splats them into caked bins with the normalized detector-to-caked LUT,
then applies the selected caked phi windows. If the LUT/Q-map path is usable,
its result is returned even when all bins are false; trace rasterization is now
fallback only when LUT context or Q-map construction is unavailable.

Selected-Qr drag Qz bounds now use the LUT transpose from dragged caked bins
back to contributing detector pixels, then filter those detector pixels by Qr
and valid Qz. This avoids using a single inverse caked-bin centroid for an
aggregate bin. The old projected-sample drag helper remains fallback only.

Selected-Qr rod 1D profiles now use detector-space numeric integration instead
of the caked display mask. The runtime selects detector pixels by Qr band, Qz
bin, valid detector Q map, finite intensity, and signed/mirrored GUI phi window,
then histograms intensity by Qz. The caked selected-Qr mask remains the overlay
and drag-preview source only.

The selected-Qr rod Qz controls now default to `0..5`. Runtime slider bounds use
`0` as the lower Qz limit and the largest positive Qz candidate from the
current caked 2theta extent as the upper limit.

## Bug/error/feature status

- Bug: fixed. Selected-Qr rod numeric 1D profiles no longer integrate caked bins
  through the display mask; they classify detector pixels by Qr/Qz/phi first.
- Error: no known selected-Qr rod detector-integration errors remain after the
  focused GUI/runtime validation below.
- Feature: complete for this workflow. Existing rod intensity mode, mirrored phi
  controls, caked overlay, and caked Qz drag behavior are preserved.

## Next actions

- None for this bug. Reopen if selected-Qr rod ROI loses caked bins that have
  exact-cake detector contributors satisfying Qr/Qz, or if selected-Qr drag Qz
  bounds depend on finite projected trace samples again.

## Validation

- 2026-04-29: `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_intersection_analysis.py tests/test_exact_cake_portable.py -ra`
  passed, `127 passed`.
- 2026-04-29: `python -m ruff check ra_sim/gui/qr_cylinder_overlay.py ra_sim/gui/integration_range_drag.py ra_sim/simulation/intersection_analysis.py ra_sim/simulation/exact_cake_portable.py tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_intersection_analysis.py tests/test_exact_cake_portable.py`
  passed.
- 2026-04-29: `python -S -m py_compile ra_sim/gui/qr_cylinder_overlay.py ra_sim/gui/integration_range_drag.py ra_sim/simulation/intersection_analysis.py ra_sim/simulation/exact_cake_portable.py`
  passed.
- 2026-04-29: `python -m pytest tests/test_gui_integration_range_drag.py tests/test_gui_bootstrap.py tests/test_gui_runtime_import_safe.py -ra`
  passed, `401 passed`.
- 2026-04-29: `python -m pytest tests/test_gui_views.py::test_create_integration_range_controls_store_vars_bindings_and_commands -ra`
  passed, `1 passed`.
- 2026-04-30: `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py -ra`
  passed, `414 passed`.
- 2026-04-30: `python -m ruff check ra_sim/gui/qr_cylinder_overlay.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_qr_cylinder_overlay.py tests/test_gui_runtime_import_safe.py`
  passed.
- 2026-04-30: `python -m py_compile ra_sim/gui/qr_cylinder_overlay.py ra_sim/gui/_runtime/runtime_session.py`
  passed.

## Links

- Runtime path: [ra_sim/gui/qr_cylinder_overlay.py](../../../ra_sim/gui/qr_cylinder_overlay.py)
- Test path: [tests/test_gui_qr_cylinder_overlay.py](../../../tests/test_gui_qr_cylinder_overlay.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
