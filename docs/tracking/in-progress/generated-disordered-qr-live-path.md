# Generated Disordered Qr Live Path

Status: implemented, focused live-picker, fit-handoff, and branch-switch validation green
Type: bug fix / GUI picker feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-03

## Summary

The live GUI could generate a HT-shifted disordered PbI2 CIF but still refresh
the Qr/Qz picker from primary-only cached rows. The reported symptom was
`w(p≈1)` nonzero with the same or fewer picker groups and no clickable
disordered-phase candidates.

## Current State

- Generated disordered Qr refs are controlled by a visible checkbox:
  `Include generated disordered-phase Qr refs`.
- The checkbox defaults on, persists in GUI state, and is independent from
  `Include packaged 6H Qr refs`.
- Runtime enable status now reports enabled/disabled reasons for generated
  disordered refs: checkbox off, stacking disorder off, zero disordered weight,
  or missing active primary CIF.
- The generated disordered source signature participates in Qr/Qz freshness,
  while the primary rendered-image cache uses a disabled disordered signature so
  Qr refs do not force a primary image redraw.
- Live Qr/Qz refresh evaluates the generated inventory from the active primary
  CIF, never from packaged `PbI2_6H.cif`.
- Hit-table-only updates schedule generated disordered collection when stored
  disordered rows are missing or stale, even when primary hit tables/images are
  reusable.
- Disordered hit-table collection runs with `accumulate_image=False`.
- Current-simulation Qr/Qz refresh publishes stored disordered rows into the
  active picker cache and keeps `source_label="disordered_phase"`.
- Ordered and disordered rows with the same numeric Qr/Qz remain separate by
  source label.
- Manual Qr/Qz selection and placement carry the selected source through
  candidates, detector hit-table coordinates, assigned HKL rows, saved manual
  pairs, and geometry-fit inputs.
- Geometry-fit preflight now logs a live handoff marker
  `geometry_fit_live_handoff_patch_marker=phase4d1`, reports job-build
  live-row counts, and can build a job-local fit-source-row fallback from the
  active picker/Q-group cache when preview rows are empty.
- The fresh source-row rebuild wrapper logs
  `fresh_rebuild_consumer_wrapper=deduped` and removes duplicate `consumer`
  keyword forwarding, preventing the prior fit fallback crash.
- Geometry-fit live-cache validation now separates live-row fit usability from
  trusted full-reflection identity. Source-matched Q-group rows can validate by
  `source + group + hkl`, `source + group`, or `source + hkl` when the match is
  unambiguous and detector coordinates are finite.
- The live-cache validator now reports exact validator predicates and cannot
  emit `status=invalid reason=ready`. Invalid rows report a concrete reason
  such as `required_source_missing`, `ambiguous_candidate`, or
  `no_finite_detector_position`.
- Detector-origin manual pairs keep detector fit-space even when caked
  diagnostic fields are backfilled. Exact-caked projector setup remains
  required only for caked-origin pairs.
- Fresh fallback is source-safe for `disordered_phase`: primary-only fallback
  rows cannot satisfy disordered saved pairs and report
  `required_source_rows_unavailable` instead.
- The geometry correspondence resolver now runs the existing stale source-row
  proof and saved-detector/source-switch fallback before returning
  `locked_qr_row_unavailable` for locked/stale QR rows. This restores
  `prediction_branch_source_switched` when a deterministic alternate source is
  proven, while ambiguous alternate evidence keeps the rejection reason.

## Bug / Error / Feature Status

- Bug status: fixed for the user-reported primary-only picker cache reuse, the
  `status=invalid reason=ready` live-cache contradiction, detector-origin
  caked-projector fallthrough, primary-only disordered fallback, and the
  follow-up locked/stale QR branch-switch regression.
- Error status: the active fallback crash
  `_build_source_rows_for_rebuild() got multiple values for keyword argument
  'consumer'` is covered by regression tests and should no longer occur on the
  patched live path. The later broad geometry failure that returned
  `locked_qr_row_unavailable` instead of `prediction_branch_source_switched`
  is also covered by regression tests.
- Feature status: implemented with explicit GUI control, live runtime logging,
  cache invalidation, hit-table scheduling, publishing, source-aware picker
  placement, fit-handoff diagnostics/tests, source-cache rung tests, and
  detector/caked fit-space classification tests.
- Remaining manual check: relaunch the GUI from the committed branch and confirm
  the live fit trace includes the `phase4d1` marker, nonzero
  `live_rows_raw_count`, and either `source_cache_live_runtime_cache_accepted`
  or a specific handoff-drop reason.
- Remaining risk: full `ra_sim.dev check` is still blocked by pre-existing
  formatting drift in `ra_sim/fitting/optimization.py`, outside this bug fix.
  The broad local selector
  `python -m pytest tests -q -k "geometry_fit or manual_geometry or manual_picker or disordered_phase"`
  timed out after five minutes during final validation without failure output;
  focused required gates passed.

## Diagnostics

Every live Qr/Qz listing refresh that emits `Updated listed Qr/Qz peaks:` also
logs exactly one generated-disordered decision:

- `Disordered Qr refs enabled: true`
- `Disordered Qr refs skipped: <reason>`

Generated-disordered diagnostics also report:

- inventory source/generated CIF paths
- unavailable inventory skip reason
- collected hit-table count
- published group/peak count
- source counts in the Qr/Qz listing message, for example
  `sources: primary=22, disordered_phase=22`

Geometry-fit handoff diagnostics report:

- `geometry_fit_job_live_rows_build`
- `q_group_cached_entries`
- `manual_picker_candidates`
- `live_preview_rows_count`
- `live_rows_by_background_current_count`
- `live_rows_source_counts`
- `live_rows_signature_match`
- `fresh_rebuild_consumer_wrapper=deduped`
- `validator_rows_nonempty`
- `validator_signature_match`
- `validator_required_sources_present`
- `validator_required_groups_present`
- `validator_required_hkls_present`
- `validator_finite_detector_rows`
- `validator_finite_caked_rows`
- `validator_canonical_row_count`
- `validator_row_schema_valid_count`
- `validator_failure_reason`
- accepted live-cache `source_counts`

## Validation

Focused validation:

- `python -m pytest tests/test_disordered_phase_user_report_live_path.py -q`
  - 1 passed.
- `python -m pytest tests/test_disordered_phase_live_q_group_refresh.py tests/test_disordered_phase_ui_enable.py tests/test_disordered_phase_inventory_live_path.py tests/test_disordered_phase_hit_table_scheduling.py tests/test_disordered_phase_current_refresh.py tests/test_disordered_phase_user_report_live_path.py -q`
  - 29 passed.
- `python -m pytest tests/test_disordered_phase_user_report_live_path.py tests/test_disordered_phase_q_group_cache.py tests/test_disordered_phase_hit_tables.py tests/test_disordered_phase_logging.py -q`
  - 15 passed.
- `python -m compileall ra_sim tests`
  - passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py tests/test_disordered_phase_user_report_live_path.py`
  - passed.
- `python -m ruff format --check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py tests/test_disordered_phase_user_report_live_path.py`
  - passed.
- `python -m pytest tests/test_geometry_fit_fresh_rebuild_consumer.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_cache_diagnostics.py -q`
  - 25 passed.
- `python -m compileall ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py tests`
  - passed.
- `python -m pytest tests/test_geometry_fitting.py::test_geometry_fit_correspondence_saved_detector_waits_for_row_proof -q`
  - passed.
- `python -m pytest tests/test_geometry_fitting.py -k "prediction_branch_source_switched or locked_qr_row_unavailable" -q`
  - 1 passed, 200 deselected.
- `python -m pytest tests/test_geometry_fitting.py -q`
  - 201 passed.
- `python -m pytest tests/test_geometry_fit_live_cache_validation_acceptance.py tests/test_geometry_fit_source_cache_rungs.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_geometry_fit_live_cache_diagnostics.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_job_live_rows_handoff.py -q`
  - 46 passed.
- `python -m compileall ra_sim tests`
  - passed.
- `git diff --check`
  - passed.

Blocked validation:

- `python -m ra_sim.dev check`
  - blocked by existing `ra_sim/fitting/optimization.py` formatting drift.
- `python -m pytest tests -q -k "geometry_fit or manual_geometry or manual_picker or disordered_phase"`
  - timed out after five minutes without failure output.
