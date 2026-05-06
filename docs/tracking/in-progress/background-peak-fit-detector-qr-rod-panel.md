# Background peak fit detector Qr rod panel

Type: bug/feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-06
Status: implemented locally, validated

## Problem

The parallel background peak-fit diagnostics notebook was mixing detector-space
geometry overview, Qr rod calibration, profile integration, and fit-quality
diagnostics in ways that made the detector view hard to interpret. Rod overlays
could be inconsistent with plotted markers when multiple target-Qr sources
shared one HK value, and the detector figure did not expose a direct pixel-space
curve-distance diagnostic.

Follow-up work found the generated `.py` diagnostic had drifted from the
intended notebook fixes and the notebook had been reverted. The working
diagnostic surface is now the tracked Python script
`scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`.
The notebook is not the source of truth for this repair.

## Change

Recreated and updated
`scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
to make the Qr rod detector and integration figures source-consistent:

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
- Final Qr-rod joint fits are cached by GUI-state filename in
  `*_qr_rod_profile_cache.pkl`; reruns of the same state filename can reuse the
  final rod-profile fit instead of refining again, and
  `RA_SIM_RESET_QR_ROD_PROFILE_CACHE=1` clears the cache.
- The cache file keeps the existing filename pattern but is read as a JSON
  envelope. Legacy pickle payloads are ignored and regenerated rather than
  deserialized.
- Final-fit cache hits are accepted only when the cached marker table includes
  `m`, `branch`, `qz_marker`, `display_l`, and either `fit_l` or `l`; older
  caches without these fields are treated as stale.
- Manual or imported `L` values are display overrides only. `fit_l` remains the
  fitted coordinate used for Qz-to-L mapping and fitting; `display_l` controls
  the visible `(HK,L)` labels.
- Rod-profile figures now draw the actual fit marker positions on the plotted
  data trace and annotate labels from `display_l`.
- Local peak snapping is bounded to each marker's local two-theta/phi window and
  requires a true local maximum. If no local peak top exists in that window, the
  projected marker position is kept instead of jumping to another peak.
- The detector selected-Q-region figure uses `specular_l_marker_table` for
  specular Qz support bounds, so cache-hit runs no longer depend on
  cache-miss-only `specular_qz_values`.
- On Windows, the script normalizes `process`/`auto` fit backend requests to
  `thread` because the generated top-level diagnostic cannot safely be imported
  by `multiprocessing.spawn` child processes.

## Status

Implemented in the recreated diagnostic script and covered by regression tests.
The patch is a local diagnostic/publication workflow fix, not a package
release. No version bump, config schema change, CLI change, or runtime package
API change was made.

## Validation

Passing checks:

- `python -m py_compile scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
  passes.
- `python -m pytest tests/test_background_peak_fits_notebook.py -ra` passes,
  `29 passed`.
- `python -m ra_sim.dev check` passes, including ruff, `280` fast tests, and
  mypy.
- Runtime reproduction with
  `python all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
  from `scripts/diagnostics` completed successfully on 2026-05-06. The run used
  `fit_backend=thread`, fit `79/79` background peaks with `0` failures, reused
  `Bi2Se3_qr_rod_profile_cache.pkl`, and produced the rod-profile and detector
  selected-Q-region figures without the prior Windows child-process traceback.
- Runtime artifact check confirmed
  `figure7_bi2se3_qr_rod_qz_profiles_peak_markers_5deg.csv` has `32` rows and
  includes `fit_l` and `display_l`.
- Runtime artifact check confirmed
  `figure7_bi2se3_qr_rod_qz_profiles.png` exists in the manuscript figure
  directory, is nonempty, and was regenerated by the validation run.

Known validation limits:

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
