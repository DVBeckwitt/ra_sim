# Weighted-event representative cache carry-through

Status: in-progress
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-24

## Summary

Fast weighted-event runtime was keeping sampled-event semantics correct, but the
representative cache could still lose the deterministic closest/highest-mosaic
ray for one final Qr-set branch before QR click, caked click, or geometry-fit
consumers read `get_last_intersection_cache()`.

This follow-up keeps fast weighted-event path active and moves representative
selection fully into raw candidate enumeration:

`raw candidate -> final Qr-branch slot -> deterministic winner -> cache carry-through`

Representative identity is now final Qr-set branch slot, not `(peak_idx,
branch_id)`, so multiple HKLs in one shared `Qr/L/branch` fold to one stored
representative.

## Current state

Implemented in [diffraction.py](../../../ra_sim/simulation/diffraction.py) with
targeted regression coverage in
[test_diffraction_weighted_events.py](../../../tests/test_diffraction_weighted_events.py),
[test_diffraction_constraints.py](../../../tests/test_diffraction_constraints.py),
and
[test_manual_geometry_selection_helpers.py](../../../tests/test_manual_geometry_selection_helpers.py).

What changed:

- added `_build_weighted_event_representative_slot_map(miller, av)` to map each
  peak/branch to one final representative slot;
- fast weighted-event representative buffers now allocate per final slot, not
  per peak;
- `_weighted_event_update_representative(...)` now ranks by
  `(-sample_weight, top_distance, -mass, sample_idx, peak_idx, q_idx)` with
  finite-positive sample weight fallback to `1.0`;
- representative rows now keep explicit provenance
  `[peak_idx, q_idx, sample_idx]` in hit-row columns `7/8/9`;
- representative hit-table emission is one row per valid final slot in stable
  slot-key order;
- `build_intersection_cache(...)` now preserves finite hit-row provenance into
  cache columns `14/15/16`, with fallback only when provenance is missing;
- `build_branch_representative_intersection_cache(...)` is passthrough-only and
  no longer reselects, recollapses, merges, or deduplicates preselected
  representative rows;
- `get_last_intersection_cache()` stays representative-facing while
  `get_last_intersection_cache_views()` still exposes both sampled-event rows
  and branch-representative rows.

Bug/error status:

- requested representative carry-through fix is implemented on fast path;
- targeted weighted-event and cache tests are green;
- requested caked-Qr representative-pick proof is green;
- broader manual-geometry replay/workflow suites were already red in this
  worktree and remain red for adjacent replay/finalizer paths not changed in
  this patch;
- full-suite run is not green, with additional unrelated failures in CLI mock
  expectations, CIF-hash/reference docs, testing-index drift, and local Tk
  runtime availability.

Feature status:

- no weighted-event sampling/statistics behavior change is intended;
- sampled rows still preserve duplicates and remain separate from
  representative-facing cache rows;
- representative cache is hardened for QR click, caked click, and geometry-fit
  source selection.

## Validation

Passed in this worktree:

- `python -m pytest tests/test_diffraction_weighted_events.py -q`
- `python -m pytest tests/test_diffraction_constraints.py -q`
- `python -m pytest tests/test_source_template_cache.py tests/test_peak_multiplicity_cache.py tests/test_diffraction_safe_wrapper.py -q`

Still failing in this worktree:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py tests/test_gui_geometry_fit_workflow.py -q`
- `python -m pytest`

Current failure buckets outside this patch:

- manual geometry detector/caked replay refresh and replay-display tests;
- geometry-fit workflow New4 finalizer/preflight tests;
- CLI shared-runner mock expecting no `seed_policy`;
- CIF/reference hash drift and testing-index drift;
- local Tk backend availability for one projection-alignment test.

## Next actions

- decide whether to treat remaining manual-geometry replay failures as separate
  follow-up or same bug family;
- if replay work continues, re-run the requested GUI/manual suite after replay
  fixes;
- once unrelated red suites are cleared, archive this note and move the fix out
  of active tracking.

## Links

- [Tracking hub](../index.md)
- [Simulation and fitting reference](../../simulation_and_fitting.md)
- [Testing and validation index](../../testing-and-validation.md)
