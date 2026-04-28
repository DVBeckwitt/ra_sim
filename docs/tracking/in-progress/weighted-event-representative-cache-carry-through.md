# Weighted-event representative cache carry-through

Status: implemented; broad-suite cleanup remains separate
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-28

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
- `_weighted_event_update_representative(...)` now ranks representatives by
  true mosaic weight first, then mosaic-top angular distance, beam/wavelength
  distance, candidate mass, and deterministic provenance;
- representative rows now keep explicit provenance
  `[peak_idx, q_idx, sample_idx]` in hit-row columns `7/8/9`;
- representative hit-row column `9` is the authoritative mosaic-top beam
  sample index for Qr/cache selection and takes precedence over table-level
  sampled-event `best_sample_indices_out` fallbacks during cache construction;
- GUI `mosaic_top_rank_key(...)` now follows the same representative fallback
  order as simulation selection: highest mosaic weight, angular distance, beam
  distance, wavelength distance from profile center, intensity/mass, then
  stable source order;
- the weighted-event fast parallel path now uses explicit Python
  `ThreadPoolExecutor` chunks calling `_weighted_event_sample_chunk_kernel`
  compiled as `njit(nogil=True)`, with no monolithic `parallel=True` sample
  kernel;
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
- Qr selection tests now exercise representative rows through
  `build_intersection_cache(...) -> intersection_cache_to_hit_tables(...) ->
  build_geometry_fit_simulated_peaks(...) -> collapse_qr_qz_selection_peaks(...)`
  and prove sampled-event rows are not used when representative rows exist.
- threaded chunk stats report `parallel_backend="threaded_njit_chunks"` and
  `parallel_worker_count`.

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

- `python -m pytest tests/test_diffraction_weighted_events.py::test_solve_q_real_jit_does_not_crash_allocate_sched -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_compute_intensity_array_is_serial_njit -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_representative_choice_uses_true_mosaic_weight_before_mass -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_representative_choice_preserves_mosaic_top_sample_index_in_hit_row -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_mosaic_top_representative_survives_even_when_unsampled -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_events_dispatcher_path_matrix -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_controlled_backend -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_events_original_plan_compliance_matrix -q -s`
  printed `original_plan_validation_incomplete=no`;
- `python -m pytest tests/test_diffraction_weighted_events.py -q`
- `python -m pytest tests/test_diffraction_weighted_events.py tests/test_source_template_cache.py tests/test_intersection_cache_schema.py -q`
- `python -m pytest tests/test_gui_geometry_q_group_manager.py::test_collapse_geometry_fit_simulated_peaks_prefers_mosaic_top_per_branch -q`
- `python -m pytest tests/test_gui_peak_selection.py::test_select_peak_by_hkl_prefers_mosaic_top_candidate_over_brighter_duplicate -q`
- `python -m pytest tests/test_gui_geometry_q_group_manager.py tests/test_gui_peak_selection.py tests/test_intersection_cache_schema.py -q`
- `python -m ruff check ra_sim/simulation/diffraction.py ra_sim/simulation/intersection_cache_schema.py ra_sim/gui/mosaic_top_selection.py ra_sim/gui/geometry_q_group_manager.py tests/test_diffraction_weighted_events.py tests/test_source_template_cache.py tests/test_intersection_cache_schema.py tests/test_gui_geometry_q_group_manager.py tests/test_gui_peak_selection.py scripts/benchmarks/benchmark_weighted_events_parallel.py`
- `python -m py_compile ra_sim/simulation/diffraction.py ra_sim/simulation/intersection_cache_schema.py ra_sim/gui/mosaic_top_selection.py ra_sim/gui/geometry_q_group_manager.py tests/test_diffraction_weighted_events.py tests/test_source_template_cache.py tests/test_intersection_cache_schema.py tests/test_gui_geometry_q_group_manager.py tests/test_gui_peak_selection.py scripts/benchmarks/benchmark_weighted_events_parallel.py`
- `python scripts/benchmarks/benchmark_weighted_events_parallel.py --runs 1 --threads 1 2 --n-samp 8 --events 2`
  reported `threads_1_parallel_backend: fast_serial`,
  `threads_2_parallel_backend: threaded_njit_chunks`, and no
  `weighted_events_python` fallback;
- `python scripts/benchmarks/benchmark_weighted_events_parallel.py --samples 2 --workers 1 2 --iterations 1 --events 1`
- `python -m pytest tests/test_diffraction_constraints.py -q`
- `python -m pytest tests/test_source_template_cache.py tests/test_peak_multiplicity_cache.py tests/test_diffraction_safe_wrapper.py -q`

Still failing in this worktree:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py -q`
- `python -m pytest tests/test_manual_geometry_selection_helpers.py tests/test_gui_geometry_fit_workflow.py -q`
- `python -m pytest tests -q --durations=20`

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
