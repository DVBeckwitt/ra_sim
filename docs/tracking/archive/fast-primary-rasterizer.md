# Fast Primary Rasterizer

Status: completed
Type: refactor
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-22

## Summary

Primary-cache rematerialization no longer rasterizes cached detector hit tables
with a Python per-hit loop. The rasterizer now uses vectorized NumPy bilinear
splat accumulation while preserving the existing public API.

An adjacent safe-wrapper bug was fixed: `process_peaks_parallel_safe(...)` now
skips building the optional last-intersection cache when last-cache retention is
disabled, avoiding wasted cache construction and log side effects.

## Current state

Feature status: implemented. The rasterizer keeps prior behavior for malformed
tables, one-row arrays, non-finite rows, duplicate hits, clipped detector edges,
and bilinear fractional pixels. Regression status: fixed for enormous finite
off-detector coordinates; those rows are now filtered before integer casts, so
they do not emit warnings or contribute pixels.

Bug status: fixed. The last-intersection cache guard is covered on the direct
runner path and leaves `get_last_intersection_cache()` empty when retention is
disabled.

Engine behavior is unchanged by this item; hidden-rerun and fit-loop cache
changes remain separate work.

## Next actions

None for this item. Follow-up speed work can continue with the remaining
simulation-speed audit items.

## Validation

- `python -m pytest -q tests/test_gui_runtime_primary_cache.py` passed: 15 passed.
- `python -m pytest -q tests/test_simulation_engine.py tests/test_diffraction_safe_wrapper.py` passed: 51 passed.
- `python -m ra_sim.dev check` passed: format check, lint, 239 fast tests, and mypy.

## Links

- Issue: none
