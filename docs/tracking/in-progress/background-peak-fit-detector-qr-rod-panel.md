# Background peak fit detector Qr rod panel

Type: bug/feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-06
Status: implemented locally, validation partial

## Problem

The parallel background peak-fit diagnostics notebook was mixing detector-space
geometry overview, Qr rod calibration, profile integration, and fit-quality
diagnostics in ways that made the detector view hard to interpret. Rod overlays
could be inconsistent with plotted markers when multiple target-Qr sources
shared one HK value, and the detector figure did not expose a direct pixel-space
curve-distance diagnostic.

## Change

Updated
`scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.ipynb`
in notebook cell 14, plus the earlier per-tilt peak-label helper cell, to make
the Qr rod detector and integration figures source-consistent:

- Detector rod calibration is Qr-driven by default with `FIT_QZ_WEIGHT = 0.0`;
  Qz residuals remain diagnostics.
- Specular `(0,L)` anchors are skipped from detector-rotation fitting.
- Fit anchors are rod-balanced and filtered through a shared
  `accept_marker_for_plotted_rod(...)` predicate.
- Mixed target-Qr identities are rejected by `(target_source_label, m)` or, if
  needed, `(target_source_label, m, branch)` instead of being forced into one
  plotted rod.
- Rejected rods are omitted from fit anchors, profile samples, Qr-center
  fitting, integration profiles, and detector rod plotting.
- Detector-space debug now reports `curve_distance_px` using point-to-polyline
  segment distance on the same projected arrays used for plotting.
- Detector view uses the existing magma detector colormap, projected rod
  lines, HK branch labels at the low-L rod base, and transparent Delta-Qr
  bands including the central `HK=0` rod. Placed-peak star markers were
  removed from the final figure.
- Integrated Qr rod figure centers `HK=0`, labels its x-axis as `L`, uses
  `Intensity (a.u.)` only on the HK=0 row, aligns non-specular x ranges from
  `L=2`, and places the Data/Simulation legend in the top-right panel.
- Integrated Qr rod profiles now annotate HK locations with arrows whose tips
  point to the interpolated `Data` trace at the plotted L coordinate. `HK=0`
  rod-profile panels use a log y-axis with positive limits derived from the
  plotted data/simulation values.
- `HK=0` Qr rod center integrations no longer subtract the shared linear
  baseline from plotted real data. For this rod only, plotted `Data` remains
  raw `background_density`, and the shared linear baseline is added to plotted
  `Simulation` as `joint_peak_density + joint_linear_baseline_density`.
- The central `00L` detector-region rod line now uses a separate linewidth at
  half the non-specular HK rod centerline width.
- Earlier per-tilt background-vs-fit plots now label peaks directly with compact
  `(HK,L)` text instead of numbered labels with a side key and branch suffix.

## Status

Implemented in the ignored parallel notebook. The patch is intended as a local
diagnostic/publication-notebook fix, not a package release. No version bump,
changelog finalization, schema change, CLI change, or runtime package API change
was made.

## Validation

Passing checks:

- Notebook JSON parse.
- `nbformat.validate`.
- AST parse and compile for all code cells.
- Static checks for removed placed-star markers, restored magma detector
  colormap, HK arrow annotations on the plotted line, `HK=0` log y-axis,
  `curve_distance_px`, mixed-Qr rejection reason, `FIT_QZ_WEIGHT = 0.0`, and
  shared predicate use across fit/profile/marker paths.
- Parallel-notebook pytest checks:
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_process_pool_worker`
  and
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_gaussian_core_lorentzian_tail_model`
  pass, along with the parallel source/behavior checks for HK arrows, removed
  stars, magma detector color, and `HK=0` log scaling.
- Targeted parallel-notebook regression checks pass for the `HK=0` raw-data /
  baseline-added-simulation behavior, existing `HK=0` log scaling, and removed
  detector-region placed-star markers:
  `python -m pytest tests/test_background_peak_fits_notebook.py::test_parallel_qr_rod_profile_hk_zero_adds_baseline_to_simulation_only tests/test_background_peak_fits_notebook.py::test_parallel_qr_rod_profile_hk_zero_uses_log_y_axis tests/test_background_peak_fits_notebook.py::test_parallel_detector_region_final_figure_omits_placed_peak_stars`.
- Parallel-only notebook test subset passes:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k parallel`.

Known validation limits:

- The target notebook section was not rerun with local data during this patch.
- Full `tests/test_background_peak_fits_notebook.py` remains red in this
  checkout because the non-parallel notebook expectations/failure path are
  unrelated to the patched ignored notebook.
- Visual acceptance still needs manual notebook regeneration: magma detector
  background, no placed-star markers, HK arrows landing on the rod-profile
  data lines, `HK=0` log-scale readability, central `HK=0` Delta-Qr band, no
  misleading mixed-target rods, and reasonable `curve_distance_px` values.

## Follow-up

Run the patched notebook section against the current sample state and inspect
the detector and Qr-integration figures before using them in a manuscript. If a
mixed target-Qr rejection removes data needed for publication, split that source
into separate plotted rod identities rather than drawing one centerline through
multiple target Qr values.
