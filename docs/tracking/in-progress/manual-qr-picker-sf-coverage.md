# Manual Qr Picker SF Coverage

Status: implemented; targeted validation passed
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-27

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

For the PbI2 modified-CIF disorder path, restored `disordered_phase` Qr rows now
rebuild detector picker source rows from stored generated-disordered hit-table
maxima when live preview rows are empty or stale. This keeps the Qr set peaks
pickable without falling back to primary or packaged 6H rows.

## Bug / error / feature status

- 2026-05-27 detector background-click patch: fixed the regression where a
  selected single Qr/Qz group could fail to start manual placement when warm
  detector-picker cache rows still contained additional groups. Fallback
  selection now scopes candidates to the active grouped cache first, then the
  listed Qr/Qz group keys if no active grouped scope is available. The same
  detector-view release is preserved for local background-peak refinement, and
  branch assignment remains based on the refined background point.
- Bug status: fixed for saved GUI states that contain nonempty
  `state.geometry.q_group_rows` and empty `state.geometry.peak_records`,
  including PbI2 `disordered_phase` rows produced by the modified-CIF disorder
  technique.
- Error status: the specific "No Qr/Qz set found" miss is no longer expected
  for one active selected Qr/Qz group when extra stale detector-picker cache
  groups are present.
- Error status: the specific no-detector-source-rows warning is no longer
  expected for those imported states after restore when source rows or stored
  generated-disordered hit tables are available.
- Feature status: no new operator-facing feature, schema migration, public API,
  dependency, CI workflow, or deprecation/migration path. This is compatibility
  hardening for existing manual Qr/Qz background picking and saved GUI states.
- Launch status: focused runtime/import gates passed locally. Rollback is a
  normal revert of the source-row restore fix or this detector background-click
  patch.

## Next actions

- Manual GUI smoke test the imported Bi2Se3, Bi2Te3, and PbI2 states through
  one Qr set selection each.
- If full helper-test runtime remains too long, split or mark the slow existing
  tests so full targeted validation gives useful output before timeout.

## Validation

- `pytest -q tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_starts_single_active_group_from_background_click tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_fallback_uses_single_listed_group tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_single_group_background_click_uses_refined_peak_for_branch_assignment tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_rejects_background_click_with_multiple_active_groups tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_starts_two_branch_session_for_qr_qz_group tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_toggle_selection_at_tags_clicked_seed_within_group tests/test_gui_canvas_interactions.py::test_canvas_detector_view_single_group_fallback_places_on_release tests/test_gui_canvas_interactions.py::test_canvas_first_manual_pick_click_does_not_immediately_place_background_point tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_place_selection_at_uses_refined_click_nearest_candidate tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_place_selection_at_uses_refined_click_branch_representative tests/test_manual_geometry_selection_helpers.py::test_caked_background_branch_association_uses_refined_peak_before_save`:
  11 passed.
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
- `python -m pytest tests/test_gui_state_restore_helpers.py::test_apply_gui_state_geometry_restores_saved_q_group_rows_without_peak_records tests/test_gui_runtime_import_safe.py::test_manual_pick_cache_source_rows_rebuild_allowed_for_restored_q_group_rows tests/test_gui_runtime_import_safe.py::test_manual_pick_cache_rebuild_uses_stored_disordered_rows_for_restored_q_groups tests/test_disordered_phase_live_runtime_regression.py::test_live_regression_disordered_q_groups_are_clickable_candidates tests/test_disordered_phase_picker_to_fitter_end_to_end.py tests/test_disordered_phase_q_group_cache.py -q`:
  10 passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_restore_helpers.py`:
  passed.
- `python -m py_compile ra_sim/gui/_runtime/runtime_session.py tests/test_gui_runtime_import_safe.py tests/test_gui_state_restore_helpers.py`:
  passed.
- Full `python -m pytest tests/test_manual_geometry_selection_helpers.py -ra`
  timed out after 15 minutes in this workspace. Focused
  `-k "manual_pick_cache or caked_qr_picker"` showed existing cache-churn
  expectation failures outside the new SF coverage regressions.

## Links

- Saved-state example: `C:\Users\Kenpo\.local\share\ra_sim\PbI2.json`
