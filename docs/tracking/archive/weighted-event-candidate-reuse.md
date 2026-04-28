# Weighted-event projected-candidate reuse

Status: fixed
Type: optimization
Owner: -
Issue: none
Priority: p2
Last updated: 2026-04-28

## Summary

The fast weighted-event diffraction path projected the same Q candidates twice:
once to build the per-sample weighted-event mass distribution and again to emit
selected events. The second projection pass is now replaced by stored-candidate
emission in the default fast path.

## Root cause

`_process_peaks_parallel_weighted_events_fast(...)` walked candidates in
sample/peak/Q order for pass 1, then walked the same Q sets again through
`_weighted_event_pass2_for_qset(...)`. That made
`_project_weighted_candidate_fast(...)` do duplicate detector projection and
optics work for candidates whose validity, detector position, phi, and mass
were already known.

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

## Status

- Bug/error status: no user-facing behavior change intended beyond runtime and
  statistics.
- Feature status: default weighted-event fast path now avoids duplicate
  projection work.
- Fallback status: old two-projection pass remains reachable with
  `reuse_weighted_event_projected_candidates=False` or a too-small
  `weighted_event_candidate_buffer_max_bytes`.
- Statistics status: `n_project_candidate_calls` now counts projection calls
  only; `n_stored_projected_candidates`, `candidate_buffer_capacity_max`, and
  `candidate_buffer_fallback_count` expose candidate-buffer behavior.

Hard invariants kept:

- candidate traversal remains sample ascending, peak ascending, Q ascending;
- weighted-event selection keeps the same cumulative mass order;
- representative hit-table behavior is unchanged;
- sampled hit-table row schema is unchanged;
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

Additional attempted validation:

```powershell
python -m pytest tests/test_geometry_fitting.py tests/test_fit_cache_controls.py -q
```

That optional geometry/cache command is still red for unrelated geometry-fitting
failures, including `NameError: name 'qr_fit_expected_count' is not defined` in
`ra_sim/fitting/optimization.py`.
