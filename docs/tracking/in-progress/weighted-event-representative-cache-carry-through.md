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

## 2026-05-21 operator Q&A summary

This summarizes the current GUI/manual Qr geometry-fit behavior discussed while
triaging the `Geometry Fit Rejected` dialog.

- `Pick Qr Sets` should choose deterministic representative rows when the
  current main simulation/cache is available. Those rows are zero-intensity
  ghost rays at beam center, with zero divergence/beam offsets and the scalar
  default wavelength. They are stable source anchors, not closest sampled
  mosaic rays.
- A ghost ray fixes source identity, not a detector pixel. During fitting, the
  same source branch is evaluated under each trial geometry, so predicted
  detector and caked coordinates can move as fit parameters move.
- Manual background points for the saved Qr/Qz fixed-source fit are fixed
  observations. The fit does not re-pick, drag, or move those measured anchors;
  only the simulated/predicted source position is recomputed.
- Residual space depends on objective space. A `caked_deg` objective must
  compare finite observed and predicted caked angular anchors in degrees or
  fail preflight. Detector-space/manual point mode can still use pixel-space
  residuals.
- A branch is the detector-side physical side of the same HKL/Qr/L reflection.
  For non-`00l` rows, branch identity is normally resolved from signed `phi`:
  negative `phi` is branch `0`, positive `phi` is branch `1`, and near-zero
  `phi` is ambiguous. The same HKL appearing in branch `0` and branch `1` is
  expected.
- Branch identity for manual Qr rows must be treated as
  `HKL + q_group_key + source_branch_index`, not just `HKL`.

Resolved GUI-route blockers from the 2026-05-21 rejected GUI runs:

- The trace preserved both clicked background points, found two required
  `[-1,0,10]` branch candidates, validated the live cache, and reported
  `branch_mismatch_count=0`.
- The final acceptance matcher then dropped both fixed-source pairs with
  `nested_full_identity_branch_ambiguous` and
  `provider_local_duplicate_hkl_unproven`, leaving `matched_pair_count=0`.
- That first rejection was fixed by treating
  `HKL + q_group_key + source_branch_index + source_reflection_index/source_row_index`
  as the effective identity for branch-proven locked Qr rows.
- A later run preserved both pairs through preflight but then fell back to
  detector `point-match` and finalized as `central_point_match, matched=1`.
  That route-loss bug is fixed by forcing detector-origin locked Qr/Qz pairs
  through the exact-caked dynamic/manual route and by rejecting any final
  validation that loses locked pair identity as `locked_manual_qr_identity_loss`.
- Remaining expected behavior: if exact caked projection cannot be prepared,
  the fit fails before optimization with a locked-Qr missing-projector or route
  invariant reason instead of reporting a pixel RMS/outlier.

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
- `python -m pytest tests/test_diffraction_weighted_events.py::test_branch_representative_cache_uses_zero_intensity_center_ghost -ra`
  proves the ghost representative precompute uses the scalar default
  wavelength, not the mosaic wavelength mean, with zero beam/divergence
  offsets;
- `python -m pytest tests/test_diffraction_weighted_events.py::test_center_ghost_representative_survives_even_when_unsampled tests/test_diffraction_weighted_events.py::test_branch_representative_cache_uses_zero_intensity_center_ghost tests/test_diffraction_weighted_events.py::test_get_last_intersection_cache_is_representative_facing_after_weighted_events tests/test_diffraction_weighted_events.py::test_get_last_intersection_cache_views_split_sampled_and_representative_rows tests/test_diffraction_constraints.py::test_build_intersection_cache_keeps_explicit_zero_offset_representative_without_sample -ra`
  passed as the focused core proof for ghost rows, fitter-facing cache, split
  sampled/representative views, and explicit zero-offset representatives;
- `python -m pytest tests/test_gui_runtime_primary_cache.py::test_rematerialize_primary_artifacts_prefers_cached_representative_intersection_cache tests/test_gui_runtime_primary_cache.py::test_store_primary_cache_payload_stores_representative_intersection_cache tests/test_gui_runtime_primary_cache.py::test_store_primary_cache_payload_drops_stale_representative_intersection_cache tests/test_gui_runtime_primary_cache.py::test_translate_intersection_cache_entry_cache_for_center_delta tests/test_gui_runtime_invalidation.py tests/test_gui_structure_factor_pruning.py -ra`
  passed as the GUI/runtime cache carry-through and invalidation proof;
- `python -m ra_sim.dev check`
  passed with ruff format check, ruff lint, fast pytest, and mypy;
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
