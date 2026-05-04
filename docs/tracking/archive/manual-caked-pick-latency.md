# Manual Caked Pick Latency

## Status

- Bug status: fixed for the targeted latency path. Normal caked Qr/Qz clicks no
  longer print live visual-source ledger rows by default.
- Error status: no new targeted-test failures found. Broader known failures
  remain in refresh-session and New4 geometry-fit tests outside this patch.
- Feature status: implemented. Live caked visual-source tracing is now opt-in
  with explicit environment flags.

## What Changed

- `RA_SIM_LIVE_CAKED_TRACE=1` enables `[ra-sim] live_caked_visual_source`
  rows.
- `RA_SIM_LIVE_CAKED_TRACE_ALL=1` includes unchanged duplicate trace rows.
- `RA_SIM_SUPPRESS_LIVE_CAKED_TRACE=1` suppresses trace output even when enable
  flags are set.
- Manual Qr/Qz simulated-candidate refinement now stores a cheap refinement
  signature in the pick cache and skips row-level refinement on warm-cache
  calls when the signature is unchanged.
- Refined lookup rebuilds are also signature-gated, but still rebuild when
  required lookup keys are missing.
- `_get_pick_cache()` resolves runtime values once, handles the
  `next_cache_data is cache_data` case once, and keeps picker behavior
  unchanged.

## Validation

Passed:

- `python -m compileall ra_sim tests`
- `python -m ruff check ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py tests/test_manual_geometry_live_peak_cache.py tests/test_manual_geometry_selection_helpers.py`
- `python -m pytest -q tests/test_manual_geometry_live_peak_cache.py`
- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "live_caked_visual_trace or refined_qr_cache"`
- `python -m pytest -q tests/test_geometry_fit_manual_fit_space_classification.py`

Known unrelated failures observed:

- `python -m ra_sim.dev format-check` still reports pre-existing formatting
  drift in `ra_sim/dev_doctor.py` and `ra_sim/fitting/optimization.py`.
- `tests/test_manual_geometry_selection_helpers.py -k "... refresh_geometry_manual_pick_session_candidates"`
  still fails
  `test_refresh_geometry_manual_pick_session_candidates_keeps_one_branch_row_without_branch_hints`.
- `tests/test_gui_geometry_fit_workflow.py` still has New4/dynamic-reanchor
  failures in geometry-fit/optimization paths.

## Compatibility

- Manual pair schema, source/branch identity, detector picker behavior, caked
  visual coordinate semantics, and the 80 degree second-peak assignment behavior
  were not changed.
