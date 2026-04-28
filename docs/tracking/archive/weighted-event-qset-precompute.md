# Weighted-event Q-set precompute

Status: fixed
Type: optimization
Owner: -
Issue: none
Priority: p2
Last updated: 2026-04-28

## Summary

The weighted-event fast serial path no longer builds Q sets lazily through a
Python dict cache during the per-sample/per-peak pass. It now precomputes all
unique Q sets needed for a run into flat NumPy arrays and uses integer lookup
tables during both weighted-event passes.

## Root cause

The previous serial fast path kept `all_q_cache = {}` inside
`_process_peaks_parallel_weighted_events_fast_serial(...)`. Pass 1 built or
reused Q sets using `(peak_idx, rep_idx)` dict keys, while pass 2 reused the same
Python-owned arrays indirectly through the per-sample `sample_qsets` list. That
preserved physics, but kept cache construction tied to the hot traversal loop
and made future geometry-level Q reuse harder.

## Fix

Implemented `_precompute_weighted_event_qsets(...)` in
`ra_sim/simulation/diffraction.py`.

The helper:

- skips invalid samples and invalid sample weights using the same rules as the
  existing weighted-event pass;
- uses the exact existing key, `(peak_idx, rep_idx)`;
- calls `solve_q(...)` with the same k-vector, reflection, solver, and trig-table
  arguments as the old dict-cache path;
- flattens Q rows into `q_values`;
- records `qset_offsets`, `qset_lengths`, `qset_status`, and
  `qset_id_by_sample_peak`.

The fast serial pass now sets `all_status` from `qset_status`, slices Q rows by
offset/length, and passes those views into the unchanged weighted-event
projection and emission helpers.

## Status

- Bug/error status: no output behavior change intended or accepted.
- Feature status: exact-preserving Q-set table is active for the fast serial
  weighted-event path.
- Follow-up status: geometry-level reuse by `(Gr, Gz)` or another validated
  Miller-invariant key is explicitly not implemented here.
- Statistics status: `n_solve_q_calls` remains the unique `(peak_idx, rep_idx)`
  solve count, while `n_qsets_precomputed`, `n_qset_lookup_entries`,
  `n_qset_reuse_hits`, and `time_qset_index` describe the new table.

## Validation

Passed:

```powershell
python -m py_compile ra_sim/simulation/diffraction.py tests/test_diffraction_weighted_events.py
python -m pytest tests/test_diffraction_weighted_events.py -q
python -m pytest tests/test_diffraction_inner_loop_optimizations.py tests/test_diffraction_local_arc.py -q
```

Focused coverage now checks representative Q-set reuse solve counts, invalid
sample status preservation (`-10`), invalid sample-weight status preservation
(`-12`), helper skip behavior, and empty-Q status propagation.

Manual warm stat sanity run:

```text
n_solve_q_calls 1
n_qsets_precomputed 1
n_qset_lookup_entries 4
n_qset_reuse_hits 3
pass2_mass_mismatch_count 0
```

Attempted broad subset:

```powershell
python -m pytest tests -q -k "weighted or diffraction or solve_q or cache"
```

The broad subset exceeded the 120 second command timeout before returning
results.
