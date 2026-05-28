# Diagnostics Scripts

This directory contains diagnostic and validation entry points. Scripts are
classified so cleanup work can distinguish maintained diagnostics from
compatibility or removal candidates.

## Maintained Diagnostics

- `all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py` - notebook-derived background peak-fit diagnostic entry point retained until the shared core is extracted.
- `background_peak_fit_worker.py` - multiprocessing worker used by the background peak-fit diagnostic flow.
- `check_geometry_fit_handoff.py` - geometry-fit handoff validation helper used by regression tests and manual diagnostics.
- `run_all_background_peak_fits.py` - orchestration wrapper for background peak-fit diagnostic runs.
- `summarize_geometry_fit_overlay_diagnostics.py` - summary tool for geometry-fit overlay diagnostic payloads.
- `validate_weighted_event_merge.py` - validation helper for weighted-event merge diagnostics.

## Archived Diagnostics

- No scripts are archived yet.

## Delete Candidates

- `comparison.py` - legacy compatibility wrapper around the background peak-fit diagnostic script; keep until external workflow usage is checked.
