# Background peak fit detector Qr rod panel

Type: bug/feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-07
Status: implemented locally, focused validation passing

## Problem

The parallel background peak-fit diagnostics workflow was mixing detector-space
geometry overview, Qr rod calibration, profile integration, and fit-quality
diagnostics in ways that made the detector view hard to interpret. Rod overlays
could be inconsistent with plotted markers when multiple target-Qr sources
shared one HK value, and the detector figure did not expose a direct pixel-space
curve-distance diagnostic. The workflow also needed a quick raw detector crop
showing the beam center through the `HK=0`, `L=1` / `00L` peak without opening
the full detector-region figure. Direct Windows runs also left global peak
fitting on the slow thread backend, so the CPU-heavy stage used one process
instead of all available worker processes.

## Change

Updated the parallel diagnostic workflow, centered on
`scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`,
to make the Qr rod detector and integration figures source-consistent:

- Detector rod calibration is Qr-driven by default with `FIT_QZ_WEIGHT = 0.0`;
  Qz residuals remain diagnostics.
- Specular `(0,L)` anchors are skipped from detector-rotation fitting.
- Fit anchors are rod-balanced and filtered through a shared
  `accept_marker_for_plotted_rod(...)` predicate.
- Mixed target-Qr identities are rejected by `(target_source_label, m)` or, if
  needed, `(target_source_label, m, branch)` instead of being forced into one
  plotted rod.
- Rejected rods are omitted from fit anchors, placed stars, profile samples,
  Qr-center fitting, integration profiles, and detector rod plotting.
- Detector-space debug now reports `curve_distance_px` using point-to-polyline
  segment distance on the same projected arrays used for plotting.
- Detector view uses linear grayscale intensity, projected rod lines, accepted
  placed-star diagnostics, HK branch labels at the low-L rod base, and
  transparent Delta-Qr bands including the central `HK=0` rod.
- Integrated Qr rod figure centers `HK=0`, labels its x-axis as `L`, uses
  `Intensity (a.u.)` only on the HK=0 row, aligns non-specular x ranges from
  `L=2`, and places the Data/Simulation legend in the top-right panel.
- Earlier per-tilt background-vs-fit plots now label peaks directly with compact
  `(HK,L)` text instead of numbered labels with a side key and branch suffix.
- The script now writes `hk0_l1_star.png`, a raw detector-intensity crop from
  the beam center through and above the drawable `HK=0`, `L=1` / `00L` marker.
  Missing marker, beam, or crop inputs skip the image without failing the run.
- The guarded runner now accepts the generated `.py` diagnostic through the
  existing `--notebook` flag and sets the internal process guard needed for
  Windows process-pool fitting.
- The generated `.py` diagnostic allows the process backend only when that
  guard is present. Direct Windows execution still falls back to threads and
  prints the guarded-runner command.
- Runtime logging now includes the process guard state next to worker and
  Numba-thread settings.

## Status

Implemented in the parallel diagnostic script. The patch is intended as a local
diagnostic/publication-workflow fix, not a package release. No version bump,
schema change, CLI change, or runtime package API change was made.

Bug/error status:

- Missing beam center, detector image, crop bounds, or drawable `HK=0`, `L=1`
  marker is handled as a skipped diagnostic image, not a run failure.
- The Qr-rod marker-label helper ordering bug is fixed; profile annotation and
  redraw paths no longer call `rod_marker_annotation_label(...)` before it is
  defined.
- Existing unrelated non-parallel notebook test failures remain out of scope.

Feature status:

- `hk0_l1_star.png` is implemented in the diagnostic `.py` as a raw detector
  crop from the beam center through and above the `HK=0`, `L=1` / `00L`
  marker.
- Windows CPU parallelization is implemented through the guarded runner path.
  A Bi2Se3 run on 2026-05-07 reported `backend=process_pool`, `pids=28`, and
  `global peak fitting elapsed=22.83s`, versus the direct-thread report of
  `backend=thread_pool`, `pids=1`, and `elapsed=220.07s`.
- The helper interface is internal to the diagnostic script; no CLI, config,
  saved-state, or package API surface changed.
- No dead Qr-rod helper or cache code was removed in this slice:
  `marker_qz_values_for_profile(...)` and the Qr-rod pickle cache are still
  referenced in the current `.py` diagnostic.

## Validation

Passing checks:

- `.py` parse and compile.
- Guarded runner `.py` source execution and Windows process-backend guard tests.
- Targeted `hk0_l1_star.png` helper and wiring coverage for crop bounds,
  edge clipping, invalid inputs, `HK=0`, `L=1` marker selection, synthetic PNG
  save, and diagnostic call-site wiring.
- Fast project check tier:
  `python -m ra_sim.dev check` passed with `280 passed`.
- Parallel-notebook pytest checks:
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_process_pool_worker`
  and
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_gaussian_core_lorentzian_tail_model`
  both pass.

Known validation limits:

- Full `tests/test_background_peak_fits_notebook.py` remains red in this
  checkout because the non-parallel notebook expectations/failure path are
  unrelated to the patched ignored notebook.
- `hk0_l1_star.png` skipped in the guarded-process Bi2Se3 benchmark because a
  drawable `HK=0`, `L=1` marker was unavailable in that run.
- Visual acceptance still needs manual script-output review: grayscale detector
  background, HK labels near low-L rod bases, central `HK=0` Delta-Qr band,
  the `hk0_l1_star.png` crop fully containing the L=1 intensity, no misleading
  mixed-target rods, and reasonable `curve_distance_px` values.

## Follow-up

Run the script against the current sample state and inspect the detector,
`hk0_l1_star.png`, and Qr-integration figures before using them in a manuscript.
If a mixed target-Qr rejection removes data needed for publication, split that
source into separate plotted rod identities rather than drawing one centerline
through multiple target Qr values.
