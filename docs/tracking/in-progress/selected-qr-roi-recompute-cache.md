# Selected-Qr ROI Recompute Cache

Status: implemented and partially validated on 2026-05-01.

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
