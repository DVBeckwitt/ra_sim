# Selected-Qr ROI Recompute Cache

Status: implemented. Cache-crash follow-up fixed and validated on 2026-05-12.

## Problem

Changing selected-Qr ROI controls such as phi min/max or delta_Qr rebuilt the selected-rod
Qz profile from the full caked mask each time. The caked result was already reusable, but the
profile summing path did not reuse previous per-bin components.

## Change

- Vectorized `binned_caked_mask_profile` with `np.searchsorted` and `np.bincount`.
- Added an internal selected-Qr profile component cache keyed by stable caked result inputs,
  selected rod key, axes, Qz edges, projection signature, and source arrays.
- Excluded phi windows and delta_Qr from the component-cache base key so ROI width/window
  edits update from added/removed mask pixels instead of doing a second full profile scan.
- Cached the selected-Qr caked profile payload per caked result so ROI-only refreshes keep
  stable caked image, axis, and sum-array identities.
- Preserved numerical semantics for raw sums, density, acceptance, pixel counts, inclusive
  final Qz edge behavior, and caked-only selected-Qr Qz profile generation.

## Validation

- Passed focused selected-Qr and profile tests.
- `python -m compileall ra_sim tests` passed.
- `git diff --check` passed.
- Full requested target set had one unrelated existing failure:
  `test_runtime_session_hkl_pick_builds_grouped_cache_from_stored_raw_peak_rows`.

## 2026-05-12 Cache-Crash Follow-Up

Bug/error status:

- Fixed: restored GUI states can enable Selected-Qr rod ROI when the caked
  projection signature contains nested projection-token dictionaries.
- Root cause: the profile and component cache keys used the nested
  `caked_projection_signature` directly, so `OrderedDict` lookup raised
  `TypeError: unhashable type: 'dict'`.
- Change: the selected-Qr profile and component cache signatures now digest
  the caked projection signature with the existing timing-signature helper
  before cache lookup.
- Public interfaces: unchanged. No GUI control, saved-state field, CLI flag,
  config key, artifact schema, dependency, CI pipeline, or migration path was
  added.
- Deprecation/migration: none. Existing saved Bi2Te3/Bi2S3-style states keep
  the same state format; only the internal runtime cache key is normalized.
- Shipping status: ready for local use after focused validation. Rollback is
  the two-line cache-key digest change plus the matching regression-test update.

Validation:

- Reproduced the original failure in
  `test_runtime_session_selected_qr_rod_profile_cache_semantics` before the
  runtime change.
- `python -m pytest tests/test_gui_runtime_import_safe.py::test_runtime_session_selected_qr_rod_profile_cache_semantics -ra`
  passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py -k selected_qr_rod -ra`
  passed: 39 selected-Qr tests.
- `python -m pytest tests/test_gui_views.py tests/test_gui_state_io.py -ra`
  passed: 78 GUI view/state tests.
- `python -m ra_sim.dev check` passed: ruff format/check, ruff lint, fast
  pytest tier, and mypy.
