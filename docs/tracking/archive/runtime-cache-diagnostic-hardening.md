# Runtime Cache Diagnostic Hardening

Status: completed
Type: bug
Owner: -
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-22

## Summary

Hardened runtime cache and geometry diagnostic paths found during the review
cycle. The fixes keep debug-only disk cache lookups behind the intersection
cache logging toggle, make QR raw-row generation tolerant of missing best-sample
buffers, and prevent newer empty intersection-cache logs from hiding older valid
logs.

## Current state

Completed. No open bug from this bundle remains after review.

- Runtime geometry fit logged-cache loaders now skip disk lookup when
  intersection-cache logging is disabled.
- Runtime simulation generation returns empty best-sample arrays when a runner
  does not populate `best_sample_indices_out`.
- Primary-cache rematerialization treats invalid cached best-sample values as
  missing instead of crashing.
- Intersection-cache log loading orders timestamp-named directories without
  stat-sorting and keeps scanning past empty recent logs.
- New4 caked point reprojection diagnostics now fail when provider guard,
  state hash, or full-recake guard checks fail during context setup.
- Generated New4 verification and Q-group sensitivity artifacts are ignored.

## Next actions

None for this bundle.

## Validation

- `python -m pytest -q tests/test_peak_sensitivity.py tests/test_diffraction_constraints.py tests/test_gui_runtime_primary_cache.py tests/test_simulation_engine.py tests/test_diffraction_safe_wrapper.py`: 138 passed.
- `python -m ra_sim.dev check`: format, ruff, fast tests, and mypy subset passed; 239 fast tests passed.
- `git diff --check`: passed.

## Links

- Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
