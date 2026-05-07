# Manual Qr Picker SF Coverage

Status: implemented; targeted validation passed
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-07

## Summary

Manual Qr/Qz set picking could miss groups that appear only when stacking-fault
probability is nonzero. The Qr list was built from SF-expanded hit-table
inventory, but the manual picker cache could still reuse stale or partial
detector candidates. In caked view with no exact caked projector, this showed
detector fallback status while clicks still missed the new SF groups.

## Current state

Picker cache now audits enabled listed Qr/Qz keys against detector picker
groups. If listed keys are missing and current hit-table inventory exists, it
rebuilds detector picker rows through a manual-pick coverage path. That path
forces detector projection for manual-pick fallback and keeps exact caked
projector failures strict for geometry-fit/preflight paths.

Unknown SF members now keep stable source identity slots based on source label,
reflection index, source row, and best sample index, so variable-count SF groups
are not collapsed into stale `+x/-x` or generic unknown buckets. Cache
signatures include SF picker inventory state: `p0/p1/p2`, stacking weights,
active/requested contribution keys, filter signatures, prune stats, and Q-group
content.

Imported GUI states that save `state.geometry.q_group_rows` without
`state.geometry.peak_records` now restore those Qr/Qz selector rows directly
into the selector cache. The imported rows are marked as pending live refresh,
survive a transient empty refresh while no live hit tables exist, and count as
manual-pick rebuild artifacts for the current background. This fixes the
reported Bi2Se3, Bi2Te3, and PbI2 imports where the manual picker showed:
`Manual Qr picker has no detector source rows. Update simulation or refresh the Qr/Qz list.`

## Bug / error / feature status

- Bug status: fixed for saved GUI states that contain nonempty
  `state.geometry.q_group_rows` and empty `state.geometry.peak_records`.
- Error status: the specific no-detector-source-rows warning is no longer
  expected for those imported states after restore.
- Feature status: no new operator-facing feature, schema migration, or public
  API. This is compatibility hardening for existing saved GUI states.

## Next actions

- Manual GUI smoke test the imported Bi2Se3, Bi2Te3, and PbI2 states through
  one Qr set selection each.
- If full helper-test runtime remains too long, split or mark the slow existing
  tests so full targeted validation gives useful output before timeout.

## Validation

- `python -m pytest tests/test_manual_geometry_selection_helpers.py::test_build_geometry_manual_pick_cache_rebuilds_detector_rows_for_listed_sf_groups tests/test_manual_geometry_selection_helpers.py::test_caked_qr_picker_starts_sf_detector_fallback_group_with_variable_count -ra`:
  passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py::test_manual_pick_cache_coverage_rebuild_bypasses_partial_snapshot tests/test_gui_runtime_import_safe.py::test_geometry_source_snapshot_signature_tracks_sf_picker_inventory -ra`:
  passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra`: 338 passed.
- `python -m pytest tests/test_manual_geometry_selection_helpers.py::test_build_geometry_manual_pick_cache_matches_active_group_multiset_for_cache_and_rebuild tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_starts_two_branch_session_for_qr_qz_group tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_group_target_count_uses_single_bg_peak_for_00l -ra`:
  passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/manual_geometry.py ra_sim/gui/geometry_q_group_manager.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_runtime_import_safe.py`:
  passed.
- Real saved-state restore proof for Bi2Se3, Bi2Te3, and PbI2:
  `q_group_rows` restored as `35`, `31`, and `39` rows with `pending=True`,
  `refresh=True`, and no warnings.
- `python -m pytest tests/test_gui_state_restore_helpers.py tests/test_gui_geometry_q_group_manager.py tests/test_gui_runtime_import_safe.py tests/test_manual_geometry_selection_helpers.py -k "apply_gui_state_geometry or preserves_imported_rows_during_empty_refresh or runtime_snapshot_capture_refreshes_open_window or manual_pick_cache_source_rows_rebuild_allowed or picker_candidates_only_skips_refinement_and_fresh_simulation"`:
  11 passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py ra_sim/gui/state.py ra_sim/gui/state_io.py tests/test_gui_geometry_q_group_manager.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_restore_helpers.py`:
  passed.
- Full `python -m pytest tests/test_manual_geometry_selection_helpers.py -ra`
  timed out after 15 minutes in this workspace. Focused
  `-k "manual_pick_cache or caked_qr_picker"` showed existing cache-churn
  expectation failures outside the new SF coverage regressions.

## Links

- Saved-state example: `C:\Users\Kenpo\.local\share\ra_sim\PbI2.json`
