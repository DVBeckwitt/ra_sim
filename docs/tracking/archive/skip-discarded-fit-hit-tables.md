# Skip Discarded Fit Hit Tables

Status: completed
Type: refactor
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-22

## Summary

Mosaic fitting image simulations no longer request hit tables that are thrown
away. The width-fit helper still collects hit tables only for the seed capture
path that records hits.

The adjacent simulation-engine cache regression was also fixed. Intersection
cache builds now collect required hit tables during the first kernel pass
instead of triggering a hidden second pass, while public results still hide hit
tables when `collect_hit_tables=False`.

## Current state

Feature status: implemented. Image-only mosaic width, legacy mosaic-shape, and
profile mosaic-shape simulation paths pass `collect_hit_tables=False` unless a
caller explicitly records hits.

Bug/error status: fixed. The stale cache rerun helper and stale staged index
state were removed from this patch scope. Forward and QR simulations now share
the single-pass cache behavior.

Tracking status: fixed. The archive tracking item is now listed in
`docs/tracking/index.md` for repo-local visibility.

Regression status: clean. The patch preserves public hit-table behavior and
keeps explicit `best_sample_indices_out` forwarding when callers provide a
buffer.

## Next actions

None for this item. Remaining simulation-speed audit items can proceed as
separate patches.

## Validation

- `python -m pytest -q tests/test_simulation_engine.py::test_simulate_collects_hidden_hit_tables_once_to_build_intersection_cache tests/test_simulation_engine.py::test_simulate_skips_hidden_rerun_when_intersection_cache_is_disabled tests/test_simulation_engine.py::test_simulate_omits_auto_best_sample_buffer_when_intersection_cache_is_disabled tests/test_simulation_engine.py::test_simulate_forwards_explicit_best_sample_buffer_when_intersection_cache_is_disabled tests/test_simulation_engine.py::test_simulate_forwards_extended_kernel_options tests/test_simulation_engine.py::test_simulate_qr_rods_collects_hidden_hit_tables_once_to_build_intersection_cache tests/test_simulation_engine.py::test_simulate_qr_rods_skips_hidden_rerun_when_intersection_cache_is_disabled tests/test_simulation_engine.py::test_simulate_qr_rods_omits_auto_best_sample_buffer_when_intersection_cache_is_disabled tests/test_simulation_engine.py::test_simulate_qr_rods_forwards_extended_kernel_options` passed: 9 passed.
- `python -m pytest -q tests/test_mosaic_width_optimization.py::test_fit_mosaic_widths_separable_recovers_true_widths tests/test_fit_cache_controls.py::test_optimization_mosaic_image_cache_respects_retention_gate` passed: 3 passed.
- `python -m pytest -q tests/test_mosaic_shape_optimization.py` passed: 13 passed.
- `python -m ra_sim.dev check` passed: format check, lint, 239 fast tests, and mypy.

## Links

- Tracking hub: [docs/tracking/index.md](../index.md)
