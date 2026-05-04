# Weighted-event projected-candidate reuse

Status: fixed
Type: optimization
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-04

## Summary

The fast weighted-event diffraction path projected the same Q candidates twice:
once to build the per-sample weighted-event mass distribution and again to emit
selected events. The second projection pass is now replaced by stored-candidate
emission in the default fast path.

Follow-up memory-cap fix: the projected-candidate buffer cap now applies to the
total worker-local allocation in the threaded weighted-event path, not to one
worker's buffer.

## Root cause

`_process_peaks_parallel_weighted_events_fast(...)` walked candidates in
sample/peak/Q order for pass 1, then walked the same Q sets again through
`_weighted_event_pass2_for_qset(...)`. That made
`_project_weighted_candidate_fast(...)` do duplicate detector projection and
optics work for candidates whose validity, detector position, phi, and mass
were already known.

The threaded follow-up bug was in the Python-side allocation decision. The
parallel path compared only
`max_sample_candidate_capacity * _WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES`
against `weighted_event_candidate_buffer_max_bytes`, then allocated five
candidate arrays with shape `(active_worker_count, chunk_candidate_capacity)`.
That let N workers allocate up to N times the configured cap.

## Fix

Implemented in `ra_sim/simulation/diffraction.py`:

- added bounded per-sample candidate buffers for mass, row, col, phi, and peak
  index;
- added `_weighted_event_project_store_for_qset(...)`, preserving pass-1
  representative updates and `save_flag == 1` `q_data/q_count` writes while
  storing valid projected candidates;
- added `_weighted_event_emit_from_stored_candidates(...)`, which streams the
  stored candidates in the same cumulative mass order and writes hit rows with
  the existing schema;
- refactored the serial fast path and threaded chunk kernel to use stored
  candidates by default;
- kept `_weighted_event_pass2_for_qset(...)` as the memory fallback and debug
  baseline path.

Follow-up fix in `ra_sim/simulation/diffraction.py`:

- added `_weighted_event_candidate_buffer_memory_policy(...)`;
- normalized `weighted_event_candidate_buffer_max_bytes` through one helper;
- changed threaded reuse eligibility to require
  `active_worker_count * max_sample_candidate_capacity
  * _WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES <= cap`;
- kept the serial path equivalent by using `worker_count=1`;
- added candidate-buffer byte diagnostics for per-worker bytes, total bytes,
  effective cap, and worker-multiplier disablement;
- left `_WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES`,
  `candidate_buffer_capacity_max`, event selection, hit-table schema, and Numba
  kernel parameters unchanged.

## Status

- Bug/error status: no user-facing behavior change intended beyond runtime and
  statistics.
- Feature status: default weighted-event fast path now avoids duplicate
  projection work.
- Fallback status: old two-projection pass remains reachable with
  `reuse_weighted_event_projected_candidates=False` or a too-small
  `weighted_event_candidate_buffer_max_bytes`.
- Memory-cap status: fixed. In serial mode, the cap still applies to one
  candidate buffer. In threaded mode, the cap applies to all worker-local
  candidate buffers combined.
- Statistics status: `n_project_candidate_calls` now counts projection calls
  only; `n_stored_projected_candidates`, `candidate_buffer_capacity_max`, and
  `candidate_buffer_fallback_count` expose candidate-buffer behavior. The
  follow-up also reports requested per-worker bytes, requested total bytes,
  effective max bytes, and whether reuse was disabled only by the worker memory
  multiplier.
- Formatter-error status: fixed. The broader `ra_sim.dev check` blocker in
  `ra_sim/fitting/optimization.py` was formatter-only drift and is now
  formatted.

Hard invariants kept:

- candidate traversal remains sample ascending, peak ascending, Q ascending;
- weighted-event selection keeps the same cumulative mass order;
- representative hit-table behavior is unchanged;
- sampled hit-table row schema is unchanged;
- candidate record layout remains five arrays: mass, row, col, phi, peak index;
- `candidate_buffer_capacity_max` remains a candidate count, not bytes;
- `q_data/q_count` behavior for `save_flag == 1` is unchanged;
- `pass1_total_mass`, `pass2_total_mass`, and pass-2 mass mismatch diagnostics
  remain reported.

## Validation

Passed:

```powershell
python -m py_compile ra_sim/simulation/diffraction.py tests/test_diffraction_weighted_events.py
python -m pytest tests/test_diffraction_weighted_events.py -q
python -m pytest tests/test_diffraction_inner_loop_optimizations.py -q
python -m pytest tests/test_diffraction_local_arc.py -q
git diff --check
```

2026-05-04 follow-up validation:

```powershell
python -m compileall -q ra_sim tests
python -m ruff check ra_sim/simulation/diffraction.py tests/test_diffraction_weighted_events.py ra_sim/fitting/optimization.py
python -m ruff format --check ra_sim/simulation/diffraction.py tests/test_diffraction_weighted_events.py ra_sim/fitting/optimization.py
python -m pytest -q tests/test_diffraction_weighted_events.py::test_weighted_event_candidate_buffer_memory_policy_counts_parallel_workers tests/test_diffraction_weighted_events.py::test_weighted_event_candidate_buffer_memory_policy_serial_one_worker_still_fits tests/test_diffraction_weighted_events.py::test_weighted_event_candidate_buffer_memory_policy_allows_zero_capacity tests/test_diffraction_weighted_events.py::test_fast_outer_loop_candidate_reuse_reduces_projection_calls tests/test_diffraction_weighted_events.py::test_fast_outer_loop_candidate_reuse_memory_fallback_uses_old_pass2 tests/test_diffraction_weighted_events.py::test_fast_outer_loop_candidate_reuse_preserves_q_data_save_flag tests/test_diffraction_weighted_events.py::test_parallel_candidate_reuse_memory_cap_counts_workers
python -m pytest -q tests/test_diffraction_weighted_events.py
python -m ra_sim.dev check
```

Results:

- targeted candidate-reuse tests: 7 passed;
- full weighted-event test file: 89 passed;
- dev check: ruff format/check passed, fast pytest gate passed with 279 tests,
  and mypy passed.
