# Qr integration region mask

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-05-01

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

2026-05-01 correction: Selected-Qr rod 1D profiles no longer use detector-space
numeric integration in runtime plotting or auto-match. Detector-native support
is now overlay/drag only. Plotted Qz profiles are computed from per-rod caked
`2theta/phi` data in both detector and caked views. Union masks are display/drag
support only and are not integrated as combined profiles.

The selected-Qr rod Qz controls now default to `0..5`. Runtime slider bounds use
`0` as the lower Qz limit and the largest positive Qz candidate from the
current caked 2theta extent as the upper limit.

Selected-Qr rod mode now plots one stacked Qz subplot per selected rod. Normal
caked radial/azimuthal integration restores the standard layout when
selected-Qr rod mode is disabled.

Selected-Qr rod mode also has an optional `Include rod shape` control. When it
is enabled, detector-backed caked masks and detector-space Qz profiles include
the selected Qr/Qz group's detector support mask in addition to the numeric
`Qr0 +/- delta_Qr` band. The shape mask participates in cache signatures so
stale rod masks are not reused across selected-shape changes.

The GUI startup TypeError from passing `listed_q_group_keys_for_picker` into the
manual-geometry cache bootstrap is fixed by adding the matching callback
parameter to the manual-geometry cache callback factory.

## Bug/error/feature status

- Bug: superseded by the 2026-05-01 Selected-Qr rod split. Runtime numeric 1D
  profiles now use per-rod caked masks/profiles, while detector masks are
  visual/drag support only.
- Error: no known selected-Qr rod detector-integration runtime path remains for
  plotting or auto-match after the focused GUI/runtime validation below.
- Error: fixed. GUI startup no longer fails with
  `make_runtime_geometry_manual_cache_callbacks() got an unexpected keyword
  argument 'listed_q_group_keys_for_picker'`.
- Feature: complete for this workflow. Existing rod intensity mode, mirrored phi
  controls, optional rod-shape support, caked overlay, caked Qz drag behavior,
  checkbox multi-rod selection, and standard caked radial/azimuthal integration
  behavior are preserved.

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
- 2026-04-30: `python -c "import importlib; m=importlib.import_module('ra_sim.gui._runtime.runtime_session'); m.ensure_runtime_controls_initialized(); print('runtime controls ok')"`
  passed.
- 2026-04-30: `python -m pytest tests/test_gui_runtime_import_safe.py -k "selected_qr_rod_1d or caked_profiles_from_sum_fields or refresh_integration" -ra`
  passed, `6 passed, 332 deselected`.
- 2026-04-30: `python -m compileall ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/manual_geometry.py tests/test_gui_runtime_import_safe.py`
  passed.
- 2026-04-30: `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py -k "selected_qr_rod or refresh_integration" -ra`
  passed, `27 passed, 399 deselected`.
- 2026-04-30: `python -m ruff check ra_sim/gui/bootstrap.py ra_sim/gui/integration_range_drag.py ra_sim/gui/qr_cylinder_overlay.py ra_sim/gui/state.py ra_sim/gui/views.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_runtime_import_safe.py`
  passed.
- 2026-04-30: `python -m pytest tests/test_gui_runtime_import_safe.py -q -x`
  currently stops outside this selected-Qr/caked startup patch at
  `test_geometry_source_snapshot_signature_tracks_sf_picker_inventory`, where
  that test expects a runtime `current_sf_prune_bias` monkeypatch target.
- 2026-05-01: `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py tests/test_gui_state_io.py -ra`
  passed, `523 passed`.
- 2026-05-01: `python -m compileall ra_sim tests` passed.

## Links

- Runtime path: [ra_sim/gui/qr_cylinder_overlay.py](../../../ra_sim/gui/qr_cylinder_overlay.py)
- Test path: [tests/test_gui_qr_cylinder_overlay.py](../../../tests/test_gui_qr_cylinder_overlay.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
