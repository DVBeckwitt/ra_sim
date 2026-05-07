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
showing the beam center through the intended `HK=0`, `L=3` / `00L` peak without
opening the full detector-region figure. The Qr-rod point organization popup was
missing from the active `.py` workflow, so marker edits could not be made before
the final joint Qz fit. Direct Windows runs also left global peak fitting on the
slow thread backend, so the CPU-heavy stage used one process instead of all
available worker processes.

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
- The script now writes `hk0_l3_star.png`, a raw detector-intensity crop from
  the beam center through and above the drawable `HK=0`, `L=3` / `00L` marker.
  The crop uses the detector colormap with log intensity scaling, isolated to
  this image. Missing marker, beam, or crop inputs skip the image without
  failing the run.
- The generated `.py` diagnostic opens the Qr-rod peak marker editor by default
  before the final joint Qz fit. `RA_SIM_QR_ROD_PEAK_EDIT_MODE=skip` disables
  the popup for unattended runs, `popup` is the default-on mode, and accepted
  edits participate in the final-fit cache key.
- The editor marker table now includes the dynamically projected `HK=0` /
  `00L` specular markers before cache lookup and fitting, so those peaks appear
  in the same organization popup as the non-specular Qr rods.
- The editor `Snap` action now snaps all markers in the selected rod panel to
  nearby local profile peaks rather than only moving one selected marker.
- The editor plots each rod panel on fitted `L` coordinates instead of raw Qz,
  with integer x-axis ticks. Dragging/clicking still saves marker positions as
  Qz through the panel's Qz-to-L mapping.
- Each Qr-rod peak marker now has an editable final-figure title. Blank titles
  fall back to `L=<rounded display_l>`, and the editor `Label` text box sets
  the exact text used when the final Qr-rod figure is drawn.
- Final Qr-rod figure labels are placed above and to the right of each peak,
  with a leader arrow pointing back to the marked peak.
- The joint Qz peak refinement now keeps weak labeled marker components instead
  of dropping them at the old 1% initial-amplitude gate, so labeled HK=0 peaks
  such as `006` remain included in the final fit.
- Final Qr-rod fit cache keys now include
  `fit_signature=joint_qz_labeled_marker_fit_v2`; old cached fits without that
  signature are recomputed.
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

- Missing beam center, detector image, crop bounds, or drawable `HK=0`, `L=3`
  marker is handled as a skipped diagnostic image, not a run failure.
- Missing or invalid Qr-rod peak edit JSON is reported and ignored; accepted
  popup/imported edits invalidate the final Qr-rod fit cache by marker-table
  hash.
- Weak labeled Qr-rod markers are now preserved through nonlinear joint Qz
  refinement. This fixes labeled HK=0 peaks such as `006` being shown as
  labels but absent from the fitted components.
- Missing `HK=0` / `00L` markers in the editor is fixed by merging specular
  marker rows into the editable marker table before the editor opens.
- The Qr-rod marker-label helper ordering bug is fixed; profile annotation and
  redraw paths no longer call `rod_marker_annotation_label(...)` before it is
  defined.
- Existing unrelated non-parallel notebook test failures remain out of scope.

Feature status:

- `hk0_l3_star.png` is implemented in the diagnostic `.py` as a raw detector
  crop from the beam center through and above the `HK=0`, `L=3` / `00L`
  marker, rendered with detector-style color and log intensity normalization.
- Qr-rod peak marker editing is implemented in the diagnostic `.py` and is on
  by default with `RA_SIM_QR_ROD_PEAK_EDIT_MODE=popup`; `skip` disables it for
  unattended runs and optional JSON round trip is available through
  `RA_SIM_QR_ROD_PEAK_EDITS`.
- Direct `.py` runs can override only the sample label/filename stem with
  `SAMPLE_NAME_OVERRIDE` or `RA_SIM_ALL_BACKGROUND_SAMPLE_NAME`; `RUN_NAME`
  continues to control the run directory.
- `HK=0` specular markers are included in the editable Qr-rod marker table and
  deduplicated after final-fit cache reuse.
- Panel-level marker snapping is implemented for the Qr-rod editor's `Snap`
  button and `s` key.
- Editor x-axes are implemented in fitted integer `L` units; Qz remains the
  stored marker coordinate and fit input.
- Editable marker titles are implemented for the Qr-rod editor and final
  Qr-rod figure; accepted title edits are included in JSON round trip and
  final-fit cache-key hashing. Generated fallback L labels are rounded to
  integers.
- Final Qr-rod peak label placement is implemented as upper-right annotations
  with leader arrows.
- Labeled weak-peak inclusion is implemented for final Qr-rod joint fits; the
  cache signature prevents reuse of stale final fits from the older component
  gate.
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
- Targeted `hk0_l3_star.png` helper and wiring coverage for crop bounds,
  edge clipping, invalid inputs, `HK=0`, `L=3` marker selection, synthetic PNG
  save, detector-style color/log rendering, and diagnostic call-site wiring.
- Targeted Qr-rod marker edit coverage for per-rod marker replacement,
  marker-table cache-key hashing, headless/interactive mode resolution, JSON
  edit round trip including `marker_title`, final-label override behavior,
  sample-name override wiring, `HK=0` specular marker inclusion, and editor
  call ordering before the joint-fit cache lookup.
- Targeted weak-peak regression coverage verifies a labeled HK=0 weak marker
  at `006`-like relative intensity remains present in the final joint-fit
  component list.
- Targeted snap coverage verifies that all markers in a selected profile panel
  move to nearby local maxima.
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
- The L3 star crop and interactive Qr-rod marker editor still need a real
  Bi2Se3 script run after this slice.
- Visual acceptance still needs manual script-output review: grayscale detector
  background, HK labels near low-L rod bases, central `HK=0` Delta-Qr band,
  the `hk0_l3_star.png` crop fully containing the L=3 intensity, the Qr-rod
  crop's log color scaling, the Qr-rod popup appearing on an interactive
  backend, edited peak titles appearing in the final Qr-rod figure, no
  misleading mixed-target rods, and reasonable `curve_distance_px` values.

## Follow-up

Run the script against the current sample state and inspect the detector,
`hk0_l3_star.png`, and Qr-integration figures before using them in a manuscript.
If a mixed target-Qr rejection removes data needed for publication, split that
source into separate plotted rod identities rather than drawing one centerline
through multiple target Qr values.
