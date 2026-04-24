# Fast weighted-event tail-fill repair

Status: fixed
Type: bug
Priority: p1
Last updated: 2026-04-24

## Summary

The fast weighted-event runtime stopped drawing the simulation because leftover
draw targets were filled inside each q-set pass. Early low-mass peaks could
consume phase-local event targets before later high-mass peaks were scanned.

## Root cause

`_weighted_event_pass2_for_qset` performed tail-fill locally. That made the CDF
act like a per-q-set or per-peak CDF instead of the required whole beam-phase
CDF.

## Fix

Tail-fill now runs only once in `_process_peaks_parallel_weighted_events_fast`
after every peak and q-set for the beam phase has streamed CDF crossings.

`_weighted_event_pass2_for_qset` now only streams crossings and returns updated
state plus last-valid candidate metadata. The outer fast path compares pass2
cumulative mass with pass1 total mass before doing any leftover fill.

If the pass2 mass differs beyond tolerance, the kernel records diagnostics and
does not hide the mismatch with tail-fill.

## Diagnostics

Internal weighted-event stats now include:

- `pass2_mass_mismatch_count`
- `pass2_mass_mismatch_max_abs`
- `tail_fill_events`

## Validation

Passed:

```powershell
python -m pytest tests/test_diffraction_weighted_events.py -q
python -m pytest tests/test_simulation_engine.py tests/test_gui_runtime_primary_cache.py -q
```

Regression coverage proves q-set-local tail-fill is gone, whole-phase draw
targets are preserved across peaks, and mass mismatch diagnostics prevent silent
tail-fill masking.
