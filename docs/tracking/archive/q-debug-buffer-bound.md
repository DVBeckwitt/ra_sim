# Bounded Q-debug allocation

Status: fixed
Type: bug
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-03

## Summary

`save_flag == 1` no longer allocates a fixed 2,000,000-row Q-debug
buffer per peak before simulation work begins. Q-debug capture is now capped
per peak, configurable by callers, and reports dropped debug rows when the cap
is reached.

## Root cause

`ra_sim/simulation/diffraction.py` and the legacy debug helper used a stale
fixed allocation sized for 2,000,000 Q rows per peak. With many peaks, this
could allocate multiple GB even for small diagnostic runs.

## Fix

Implemented in `ra_sim/simulation/diffraction.py`:

- added `DEFAULT_Q_DEBUG_MAX_SOLUTIONS_PER_PEAK = 8192`;
- added `q_debug_max_solutions_per_peak=None` through the weighted-event and
  public `process_peaks_parallel(...)` call paths;
- normalized `None` to the default, clamped integer values below 1 to 1, and
  rejected non-integer values with `TypeError`;
- replaced fixed Q-debug allocations with `_allocate_q_debug_buffers(...)`;
- kept `save_flag == 0` placeholder behavior unchanged;
- kept `q_count` as stored-row count and never incremented it past capacity;
- added `q_debug_truncated_solution_count` to weighted-event stats.

Also updated `ra_sim/simulation/diffraction_debug.py` to use the same bounded
allocator.

## Status

- Bug status: fixed.
- Error status: no known runtime error remains for the bounded Q-debug path.
- Feature/API status: `q_debug_max_solutions_per_peak` is available as an
  optional diagnostic cap; default behavior stores up to 8192 rows per peak.
- Compatibility status: return tuple order, Q-data columns, and `save_flag == 0`
  placeholder behavior are unchanged.

## Validation

Passed:

```powershell
python -m pytest tests/test_diffraction_weighted_events.py::test_fast_outer_loop_candidate_reuse_preserves_q_data_save_flag tests/test_diffraction_weighted_events.py::test_q_debug_max_solutions_per_peak_normalizes_and_rejects_invalid tests/test_diffraction_weighted_events.py::test_save_flag_q_data_allocation_is_bounded tests/test_diffraction_weighted_events.py::test_save_flag_q_data_reports_truncation_when_debug_buffer_full tests/test_diffraction_weighted_events.py::test_diffraction_debug_uses_bounded_q_debug_allocation -ra
python -m pytest tests/test_diffraction_constraints.py -ra
python -m pytest tests/test_diffraction_weighted_events.py::test_weighted_events_parallel_from_bound_matches_serial_real_solve_q_small -ra
python -m compileall ra_sim/simulation/diffraction.py ra_sim/simulation/diffraction_debug.py tests/test_diffraction_weighted_events.py
python -m ruff check ra_sim/simulation/diffraction.py ra_sim/simulation/diffraction_debug.py tests/test_diffraction_weighted_events.py
git diff --check
```

Broader validation:

```powershell
python -m ra_sim.dev check
```

Still blocked by unrelated formatting drift in
`ra_sim/fitting/optimization.py`.
