# 6H Qr Reference SF Picking

Status: implemented, focused validation green, wider manual-geometry slice red
Type: feature / picker bug fix
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-30

## Summary

PbI2 manual Qr/Qz picking can optionally add 6H reference groups when the
stacking-fault `w1` contribution is nonzero. This covers extra Qr/Qz positions
that appear in the `p=1`/`w1` state but are not present in the normal primary
inventory.

## Current State

- Added packaged reference CIF at
  `ra_sim/config/materials/PbI2_6H.cif`.
- Added saved GUI checkbox `Include 6H Qr refs`, default off.
- 6H reference rows are generated only when the checkbox is on and `w1 > 0`.
- 6H rows are tagged as `pbii_6h_ref` and flow into the same Qr/Qz listing and
  manual-pick cache path as primary/SF rows.
- Duplicate numeric Qr/Qz groups merge by tolerance before listing/picking, so
  primary and 6H duplicates share one displayed selector row and combined
  detector candidates.
- Caked fallback remains detector-backed when the exact caked projector is not
  available.

## Bug / error / feature status

- Feature status: implemented.
- Bug status: fixed for missing opt-in 6H reference Qr/Qz groups when PbI2
  stacking-fault `w1` peaks need extra manual-pick references.
- Error status: focused 6H gating, duplicate merge, state IO, detector
  fallback, compile, and ruff checks are green. The broader manual-geometry
  slice is still red on existing caked-view candidate/reverse-LUT expectations
  listed below.

## Compatibility

- No version bump.
- No CLI, config, artifact, or saved-state schema change.
- Old GUI states load with the new checkbox defaulting to off.
- Existing primary/secondary Qr behavior is unchanged unless the checkbox is
  enabled.

## Validation

Focused validation:

- `python -m py_compile ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py ra_sim/gui/controllers.py ra_sim/gui/manual_geometry.py ra_sim/gui/views.py ra_sim/gui/state.py ra_sim/gui/runtime_invalidation.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_io.py tests/test_qr_grouping.py`
  - passed.
- `python -m pytest tests/test_qr_grouping.py -k "6h_reference or duplicate" -ra`
  - 2 passed, 7 deselected.
- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "6h_reference_detector_fallback_group" -ra`
  - 1 passed, 490 deselected.
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "6h_qr_reference_gate" -ra`
  - 1 passed, 339 deselected.
- `python -m pytest tests/test_gui_state_io.py -k "6h_qr_reference_toggle" -ra`
  - 1 passed, 13 deselected.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py ra_sim/gui/controllers.py ra_sim/gui/manual_geometry.py ra_sim/gui/views.py ra_sim/gui/state.py ra_sim/gui/runtime_invalidation.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_io.py tests/test_qr_grouping.py`
  - passed.

Wider validation:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_io.py tests/test_qr_grouping.py -ra`
  - failed in `tests/test_manual_geometry_selection_helpers.py`.
- First five failures from `--maxfail=5`:
  - `test_geometry_manual_choose_group_at_honors_detector_display_frame_when_values_match_caked`
  - `test_geometry_manual_candidate_helpers_prefer_caked_coords_in_caked_view`
  - `test_caked_background_pick_refines_locally_binds_lut_and_separates_display`
  - `test_refresh_geometry_manual_pair_entry_recomputes_stale_detector_anchor_once`
  - `test_refresh_geometry_manual_pair_entry_sim_replay_uses_current_projected_caked_point_without_detector_alias_fallback`

Remaining validation:

- Resolve or triage the broader caked-view manual-geometry failures before
  claiming full manual-geometry slice closure.
