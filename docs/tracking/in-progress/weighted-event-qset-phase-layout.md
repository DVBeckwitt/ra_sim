# Weighted-event Q-set and phase layout

Status: implemented for packed threaded Q storage and scalar-equivalent phase counts; clustering and geometry-level Q reuse remain planned
Type: bug/performance/feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-28

## Summary

The weighted-event diffraction fast path had three separate issues after the
first scaling pass:

- the threaded path still materialized dense Q tables as
  `all_q_data[num_peaks, n_samp, q_width, 4]` plus `all_q_counts`;
- threaded timing charged whole chunk runtime to `time_project`, even though
  that work included selection, emission, and cache work;
- beam phases were still implicit samples, making future clustering and
  per-phase event policies harder to test.

This patch keeps the existing physics and event semantics, but makes memory
layout and diagnostics honest before approximate beam-phase clustering.

## Current state

Implemented in `ra_sim/simulation/diffraction.py` with regression coverage in
`tests/test_diffraction_weighted_events.py`.

What changed:

- removed the threaded dense Q-table precompute path;
- made threaded weighted-event chunks consume the same packed Q-set layout as
  serial:
  `q_values`, `qset_offsets`, `qset_lengths`, `qset_status`, and
  `qset_id_by_sample_peak`;
- split threaded chunk wall time into `time_chunk_compute`;
- left threaded `time_project` as projection-only accounting rather than
  pretending it covers the full chunk;
- added internal `phase_event_counts`, initialized from the scalar
  `events_per_beam_phase`;
- changed flat hit-table capacity calculations to use
  `sum(phase_event_counts)`;
- added beam-phase diagnostics:
  `n_raw_beam_phases`, `n_effective_beam_phases`,
  `n_exact_solve_q_phase_groups`, `phase_weight_sum`, and
  `phase_event_count_total`;
- added a direct worker-count clamp regression that proves process stats report
  the clamped effective worker count;
- documented the change in `CHANGELOG.md` under Unreleased.

## Bug/error status

Fixed:

- dense threaded Q storage is removed from the weighted-event fast path;
- threaded branch no longer reports full chunk wall time as `time_project`;
- scalar `events_per_beam_phase` behavior remains unchanged and still means
  events per full beam phase, not events per peak;
- threaded and serial weighted-event paths still match in the focused suite;
- process stats expose the new chunk/phase diagnostics.

Not fixed in this patch:

- opt-in clustered beam phases are still disabled;
- `(Gr, Gz)` geometry-level Q reuse across peaks is still not implemented;
- benchmark output still needs the planned expanded correctness/scaling
  counters.

## Feature status

Done:

- packed Q-set memory layout shared by serial and threaded weighted-event
  execution;
- scalar-equivalent internal phase event counts;
- exact solve-Q phase reuse remains active through existing
  `_annotate_solve_q_sample_reuse(...)` and packed Q-set keying;
- first phase visibility stats are available for later clustering work.

Planned:

- explicit beam-phase table object outside Numba boundaries;
- public opt-in `beam_phase_mode` choices for full, exact reuse, and clustered
  preview/validation modes;
- clustered event policies: constant, weighted, and square-root weighted;
- cross-peak Q reuse keyed by solve-Q geometry.

## Validation

Passed:

```bash
python -m pytest tests/test_diffraction_weighted_events.py -ra
python -m compileall ra_sim tests
python -m ruff check ra_sim/simulation/diffraction.py tests/test_diffraction_weighted_events.py
```

Blocked:

```bash
python -m ra_sim.dev check
```

The dev check is blocked by pre-existing format drift in unrelated files:

- `ra_sim/fitting/optimization.py`
- `ra_sim/gui/_runtime/primary_cache_helpers.py`
- `ra_sim/gui/_runtime/runtime_session.py`
- `tests/test_gui_runtime_primary_cache.py`
- `tests/test_timing.py`
