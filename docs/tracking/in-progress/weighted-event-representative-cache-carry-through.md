# Weighted-event representative cache carry-through

Status: implemented; committed validation pending
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-21

## Summary

Fast weighted-event runtime keeps sampled-event semantics for images, but the
geometry fitter needs deterministic branch anchors that are independent of the
stochastic sampled rays. Each final Qr/L/branch now gets a zero-intensity ghost
representative at beam center, with zero divergence/beam offsets and the
default wavelength, and the fitter-facing cache uses those rows instead of the
closest sampled row.

Representative identity remains the final Qr-set branch slot, not `(peak_idx,
branch_id)`, so multiple HKLs in one shared `Qr/L/branch` fold to one stored
ghost representative.

## Current state

Implemented in [diffraction.py](../../../ra_sim/simulation/diffraction.py) with
targeted regression coverage in
[test_diffraction_weighted_events.py](../../../tests/test_diffraction_weighted_events.py),
[test_diffraction_constraints.py](../../../tests/test_diffraction_constraints.py),
[test_gui_runtime_primary_cache.py](../../../tests/test_gui_runtime_primary_cache.py),
[test_gui_runtime_invalidation.py](../../../tests/test_gui_runtime_invalidation.py),
and
[test_gui_structure_factor_pruning.py](../../../tests/test_gui_structure_factor_pruning.py).

What changed:

- added `_build_weighted_event_representative_slot_map(miller, av)` to map each
  peak/branch to one final representative slot;
- added extended hit rows carrying explicit zero beam/divergence/wavelength
  context for ghost representatives;
- added `_build_weighted_event_ghost_representative_hit_tables(...)` to solve
  branch anchors at beam center/default wavelength without sampling intensity;
- removed the old sampled mosaic-top representative ranking/update/merge hot
  path from the weighted-event kernels;
- representative rows now keep explicit provenance `[peak_idx, q_idx]` in
  hit-row columns `7/8`, leave best sample as `NaN`, and write zero context
  offsets in hit-row columns `10:15`;
- the weighted-event fast parallel path now uses explicit Python
  `ThreadPoolExecutor` chunks calling `_weighted_event_sample_chunk_kernel`
  compiled as `njit(nogil=True)`, with no monolithic `parallel=True` sample
  kernel;
- representative hit-table emission is one ghost row per valid final slot in
  stable slot-key order;
- `build_intersection_cache(...)` now preserves finite hit-row provenance into
  cache columns `14/15/16`, with fallback only when provenance is missing;
- `build_intersection_cache(...)` accepts explicit hit-row context offsets, so
  ghost rows with no best sample survive caking/geometry cache construction;
- `build_branch_representative_intersection_cache(...)` is passthrough-only and
  no longer reselects, recollapses, merges, or deduplicates preselected
  representative rows;
- `get_last_intersection_cache()` stays representative-facing while
  `get_last_intersection_cache_views()` still exposes both sampled-event rows
  and branch-representative rows.
- GUI primary-fill stores representative intersection-cache entries per
  contribution key, drops stale representative entries when raw hit rows are
  replaced, and translates representative detector coordinates during
  detector-center remaps;
- Qr selection tests now exercise representative rows through
  `build_intersection_cache(...) -> intersection_cache_to_hit_tables(...) ->
  build_geometry_fit_simulated_peaks(...) -> collapse_qr_qz_selection_peaks(...)`
  and prove sampled-event rows are not used when representative rows exist.
- threaded chunk stats report `parallel_backend="threaded_njit_chunks"` and
  `parallel_worker_count`;
- weighted-event worker count now resolves through explicit API
  `numba_thread_count`, `RA_SIM_WEIGHTED_EVENT_WORKERS`, then a conservative
  Auto heuristic, and stats report requested/effective worker counts plus the
  source;
- headless geometry fitting exposes the same control with
  `--weighted-event-workers`;
- added `scripts/diagnostics/validate_weighted_event_merge.py` as the
  branch-local focused merge gate;
- benchmark smoke now asserts it imported `ra_sim` from the current checkout,
  not a stale installed package.

Bug/error status:

- requested zero-intensity branch ghost fix is implemented on fast and Python
  weighted-event paths;
- targeted weighted-event, cache, invalidation, structure-factor-pruning, and
  Numba-disable compatibility tests are green;
- old sampled mosaic-top representative ranking code has been removed from the
  production hot path;
- full `tests/test_gui_runtime_primary_cache.py` still has the known unrelated
  detector-center relative-coordinate expectation mismatch in
  `test_store_primary_cache_payload_can_store_detector_relative_hit_tables`.

Feature status:

- no weighted-event sampling/statistics behavior change is intended;
- sampled rows still preserve duplicates and remain separate from
  representative-facing cache rows;
- representative cache is hardened for QR click, caked click, geometry-fit
  source selection, primary-fill reuse, and detector-center remap;
- manual weighted-event worker control is available through Python API,
  config, environment variable, headless CLI, and benchmark flags;
- future weighted-event merges can be validated with one focused command before
  broad-suite triage.

Committed branch status:

- current branch head is `f7a93f3 feat(diffraction): add worker control`;
- the worker-control feature is committed and focused weighted-event diagnostics
  were green before commit;
- the later core-scaling benchmark experiment was discarded by resetting back to
  `HEAD`, so it is not part of the committed branch;
- remaining known failures are broader project/test-data issues, not
  weighted-event worker-control regressions.

## Validation

## Manual worker control

Python/API:

```python
diffraction.process_peaks_parallel(..., numba_thread_count=4)
```

Environment fallback:

```bash
RA_SIM_WEIGHTED_EVENT_WORKERS=4 python scripts/benchmarks/benchmark_weighted_events_parallel.py --runs 3 --threads 1 2 4 8 --n-samp 512 --events 10
```

Config fallback:

```yaml
instrument:
  simulation:
    weighted_event_worker_count: auto
```

Headless geometry-fit CLI:

```bash
python -m ra_sim fit-geometry state.json --weighted-event-workers 4
```

Benchmark:

```bash
python scripts/benchmarks/benchmark_weighted_events_parallel.py --runs 3 --threads 1 2 4 8 --n-samp 512 --events 10
```

Auto mode is conservative: tiny weighted-event runs stay serial, larger runs
scale up gradually, and requested counts are clamped to available CPU capacity.
Small simulations may be faster with one worker; more workers are not always
faster.

## Merge diagnostics

Run focused diagnostics before treating a full-suite result as meaningful:

```bash
python scripts/diagnostics/validate_weighted_event_merge.py --skip-full-pytest
```

To include the broader suite after the focused gate:

```bash
python scripts/diagnostics/validate_weighted_event_merge.py --full-pytest
```

The diagnostics runner checks compliance-matrix markers, backend names,
benchmark import hygiene, representative cache behavior, Qr selection behavior,
and the weighted-event benchmark smoke output.

Passed in this worktree:

- `python scripts/diagnostics/validate_weighted_event_merge.py --skip-full-pytest`
  passed and printed the required compliance and backend markers;
- `python scripts/diagnostics/validate_weighted_event_merge.py --full-pytest`
  passed the focused diagnostics and benchmark phases, then failed during the
  broader `python -m pytest tests -q --durations=20` phase on unrelated
  full-suite failures;
- `python -m pytest tests -q --durations=20 -x` first failed at
  `tests/test_compare_bi2se3_reference_tool.py::test_compare_tool_default_uses_two_theta_balanced_wavelength`
  on CIF reference hash drift after 92 passing tests and 2 skipped tests;
- `python -m pytest tests/test_diffraction_weighted_events.py::test_solve_q_real_jit_does_not_crash_allocate_sched -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_compute_intensity_array_is_serial_njit -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_center_ghost_representative_survives_even_when_unsampled -q`
- `python -m pytest tests/test_gui_runtime_primary_cache.py::test_store_primary_cache_payload_drops_stale_representative_intersection_cache -q`
- `python -m pytest tests/test_gui_runtime_primary_cache.py::test_translate_intersection_cache_entry_cache_for_center_delta -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_manual_worker_count_one_routes_serial -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_manual_worker_count_two_routes_threaded_chunks -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_manual_worker_count_four_reports_four_workers_when_enough_samples -q`
- `python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_event_worker_count_config_override -q`
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
- `python scripts/benchmarks/benchmark_weighted_events_parallel.py --runs 1 --threads 1 2 4 --n-samp 512 --events 2`
  reported `threads_1_parallel_backend: fast_serial`,
  `threads_2_parallel_backend: threaded_njit_chunks`,
  `threads_4_parallel_backend: threaded_njit_chunks`, and no
  `weighted_events_python` fallback;
- `python scripts/benchmarks/benchmark_weighted_events_parallel.py --runs 1 --threads 1 2 4 8 --n-samp 512 --events 2`
  verifies manual requested/effective worker reporting and CPU clamping;
- `python scripts/benchmarks/benchmark_weighted_events_parallel.py --samples 2 --workers 1 2 --iterations 1 --events 1`
- `python -m pytest tests/test_diffraction_constraints.py -q`
- `python -m pytest tests/test_source_template_cache.py tests/test_peak_multiplicity_cache.py tests/test_diffraction_safe_wrapper.py -q`

Still failing in this worktree:

- `python scripts/diagnostics/validate_weighted_event_merge.py --full-pytest`
  only because the optional full-suite phase fails after the focused gate;
- `python -m pytest tests/test_manual_geometry_selection_helpers.py -q`
- `python -m pytest tests/test_manual_geometry_selection_helpers.py tests/test_gui_geometry_fit_workflow.py -q`
- `python -m pytest tests -q --durations=20`

Current failure buckets outside this patch:

- manual geometry detector/caked replay refresh and replay-display tests;
- geometry-fit workflow New4 finalizer/preflight tests;
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
