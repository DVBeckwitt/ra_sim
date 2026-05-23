# Background peak fit detector Qr rod panel

Type: bug/feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-22
Status: implemented locally, Qr-rod editor startup and L-bound callback crash
fixed, detector label editing/import/export restored through a responsive Tk
canvas popup, detector companion preview/deferred Delta Qr validation passing,
PbI2 HK=0/nonzero L defaults aligned across editor/preview/final plots, legacy
notebook consumers migrated to the maintained `.py` diagnostic, HK=0 editor
phase real-profile seeding restored, PbI2 compressed nonzero marker-L mapping
guard added, PbI2 final HK=0 row restored, split editor persistence/cache
guarded, PbI2 manuscript figures routed to `results_pbi2`, and HK=4 minus
marker edits preserve the active editor panel range, HK=0/specular editor
refreshes are phase-scoped, and PbI2 `m=7` Qr-rod rows are hidden before
artifacts; real HK=0/qz profile rebuilding is restored from active 00L markers,
phi/2theta ROI bounds, and detector Q maps independent of cached base rows;
optional pre-fit central-phi caked plane background reduction is implemented with
cache-explicit diagnostics and raw detector preservation; PbI2 now defaults the
active Qr-rod editor Delta Qr to `0.13`, and saved marker edit files are
rejected when their stored background/profile policy no longer matches the
current prefit subtraction settings; Qr-rod peak marker editing now coalesces
Delta Qr preview refreshes and caches per-panel plot inputs for smoother marker
dragging; nonzero plus/minus branches now share an L grid and common finite
support, HK=0/specular no longer exposes or applies L min/max controls, and
final Qr-rod profile figures now skip peak fitting and draw only the
GUI-integrated trace plus a slider-controlled smoothed copy

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

The PbI2 case also exposed a material-specific plotting failure. PbI2 rods use
different lattice constants than the Bi2Se3/Bi2Te3 states, and the weak
nonzero PbI2 rod profiles were showing peak-only Qz fit components as
`Simulation` even when the fit relied on a large negative Qz-linear nuisance
baseline. That made `m=3` and `m=4` overlays look physically wrong while the
sideband-corrected data still carried the rod-centered diffuse signal.

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
- Detector view uses a linear detector-intensity colormap, projected rod lines, accepted
  placed-star diagnostics, HK branch labels at the low-L rod base, and
  transparent Delta-Qr bands including the central `HK=0` rod.
- The detector selected-region figure now gives the central `HK=0` / `00L`
  rod a dedicated high-contrast sky-blue style, same-width centerline,
  stronger band fill, and expanded Delta-Qr boundary so the actual selected
  region remains visible over high-intensity detector pixels without making the
  centerline dominate the region.
- The detector selected-region Qr-label editor now runs as a Tk canvas popup
  over a saved detector-figure snapshot instead of using Matplotlib text-artist
  dragging. Labels are selected and dragged in detector pixel coordinates, the
  active label text and font size can be adjusted in the popup, and
  import/export still use the existing detector-label JSON schema.
- Integrated Qr rod figure centers `HK=0`, labels its x-axis as `L`, labels the
  HK=0 row and left nonzero subplot axes with `Intensity (a.u.)`, aligns
  non-specular x ranges from `L=2`, and places the Data/Simulation legend in
  the top-right panel.
- Earlier per-tilt background-vs-fit plots now label peaks directly with compact
  `(HK,L)` text instead of numbered labels with a side key and branch suffix.
- The script now writes `hk0_l3_star.png`, a raw detector-intensity crop from
  the beam center through and above the drawable `HK=0`, `L=3` / `00L` marker.
  The crop uses the detector colormap with log intensity scaling, isolated to
  this image. Missing marker, beam, or crop inputs skip the image without
  failing the run.
- Optional pre-fit central-phi caked plane background reduction now runs before
  caking, peak fitting, and Qr-rod profile extraction when enabled through
  `RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_*` overrides or the popup. The raw
  detector image remains available, the detector-evaluated plane model and
  corrected detector are saved as diagnostics, and the pre-editor/final Qr-rod
  cache signatures include the subtraction policy. The editor uses a Tk/Pillow
  canvas popup instead of Matplotlib widgets, fits a 2D linear plane in caked
  `(phi, 2theta)` space using only `-90 <= phi <= 90`, shows labeled
  raw/model/corrected previews cropped to that phi band, and keeps scale changes
  on the cached plane model.
- PbI2 Qr-rod marker editing now starts with active Delta Qr `0.13` when the
  saved state still carries the old generic default source value. Saved
  Qr-rod peak edit JSON now records the active lattice, Q-group, rod-reference,
  and background/profile policy signature, and automatic loading rejects stale
  files instead of placing old marker points on newly background-corrected
  profiles.
- The Qr-rod marker editor now keeps the high-frequency UI path lighter:
  Delta Qr slider movement marks the detector companion preview dirty and
  flushes one latest-state redraw on mouse release or Accept, marker/profile
  arrays are cached per editor panel, and detector-preview callback errors are
  stored in editor state without closing the popup.
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
- Marker add, drag, delete, and snap operations preserve the active Matplotlib
  panel x/y limits. Nonzero HK rods expose Delta Qr, `L Min`, `L Max`, and
  `theta_i`; HK=0/specular exposes only phi/2theta ROI bounds.
- The same Qr-rod editor now includes nonzero Delta Qr, `L Min`, `L Max`, and
  `theta_i` controls. Delta Qr slider movement updates the detector companion
  preview immediately and marks integrated profile rows dirty; the expensive
  detector Qr/Qz profile accumulator is run once on slider release or accept.
  `L Min`, `L Max`, and `theta_i` submissions update the preview and defer the
  expensive profile-table refresh until Accept. Accepted Delta Qr/L/theta_i
  values are applied to the nonzero profile rows used by the final joint Qz fit
  and cache key.
- Popup-mode Qr-rod marker editing now opens a read-only detector companion
  figure titled `Qr rod detector region preview` before the blocking marker
  editor show call. The preview uses the current detector image, Qr/Qz maps,
  selected Qr-rod Delta-Qr bands, and the central `HK=0` / `00L` band so marker
  edits, Delta Qr, and L-window adjustments can be made while the detector
  regions are visible.
- The detector companion preview now refreshes its Qr-region overlays whenever
  the unified editor's nonzero Delta Qr/L/theta_i controls or HK=0 phi/2theta
  controls change. The detector image remains static; only the existing overlay
  artists are removed and redrawn with the current profile scope.
- Delta Qr drag now queues one companion detector overlay redraw until the
  slider is released or the editor is accepted. This keeps the high-frequency
  UI path responsive while preserving final profile/fitting correctness.
- HK=0/specular editor refresh callbacks now carry the active editor phase.
  Phi/2theta ROI changes redraw only the HK=0 detector band and rebuild only
  specular profile rows, so nonzero Qr rods keep their accepted Delta Qr, L
  bounds, and theta_i.
- PbI2 configured-hidden Qr-rod rows are filtered before editor/cache/final
  artifact tables and plots. The current hidden set removes `m=7`, so that rod
  no longer appears in PbI2 Qr-rod outputs.
- The marker editor now explicitly calls `show(warn=False)` on companion
  figures before entering the blocking Matplotlib event loop, so the detector
  preview window is mapped as well as redrawn.
- Each Qr-rod peak marker now has an editable final-figure title. Blank titles
  fall back to `L=<rounded display_l>`, and the editor `Label` text box sets
  the exact text used when the final Qr-rod figure is drawn.
- Final Qr-rod figure labels are placed above and to the right of each peak,
  with a leader arrow pointing back to the marked peak.
- The joint Qz peak refinement now keeps weak labeled marker components instead
  of dropping them at the old 1% initial-amplitude gate, so labeled HK=0 peaks
  such as `006` remain included in the final fit.
- Final Qr-rod fit cache keys now include
  `fit_signature=joint_qz_labeled_marker_fit_specular_theta_i0_l8_v9`; old
  cached fits without that signature are recomputed.
- The guarded runner now accepts the generated `.py` diagnostic through the
  existing `--notebook` flag and sets the internal process guard needed for
  Windows process-pool fitting.
- Direct Windows execution of the generated `.py` diagnostic now relaunches
  itself through the guarded runner when the requested backend is `process` or
  `auto`, so the default command uses process-pool fitting instead of silently
  falling back to threads. Explicit `BACKGROUND_FIT_BACKEND=thread` and
  `BACKGROUND_FIT_BACKEND=serial` remain direct opt-outs.
- Runtime logging now includes the process guard state next to worker and
  Numba-thread settings.
- The generated `.py` diagnostic now writes a pre-editor cache keyed by the
  GUI-state filename, background filenames, geometry, orientation, fit settings,
  and peak-job signature. Matching reruns reuse completed global peak fits,
  line-profile fits, and Qr-rod profile/marker-table construction before the
  manual marker editor opens.
- Post-editor Qr-rod marker export now preserves manually added rows even when
  they inherit duplicate `hkl` labels from an existing marker. The final
  specular `HK=0` marker table is rebuilt from the post-editor marker table,
  so edited Qz positions propagate into the marker CSV, detector
  selected-region figure, and `hk0_l3_star.png` crop.
- Final Qr-rod profile figure row selection now filters rods through the
  drawable profile data and marker-derived positive-L mapping before allocating
  subplot rows. Empty rods, such as the observed Bi2Te3 `HK=7` row, are skipped
  instead of producing blank figure rows.
- PbI2 nonzero Qr-rod extraction uses same-Qz transverse Qr sideband
  subtraction. The central raw density remains in `background_density_raw`, the
  sideband estimate is written as `qr_sideband_background_density`, and
  sideband-corrected `background_density` remains the fitted/audit profile.
- PbI2 nonzero profile plots now use a plot-model decision helper. Branches
  show raw central `background_density_raw` as `Data`; available dashed `Fit`
  overlays plot `joint_fit_density + qr_sideband_background_density`.
  Marker/L mapping and Qz-baseline cancellation checks remain diagnostics in
  the markdown table instead of suppressing m=3 or m=4 overlays.
- PbI2 Qr-rod profile panels are capped at `L=3`, without trimming the
  exported CSV/profile rows. Only `HK=0` remains log-scaled; nonzero HK panels
  use the shared linear intensity axis so signed sideband-corrected values stay
  visible.
- PbI2 no-background debug mode is available through
  `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION=1`. It forces transverse sideband
  subtraction off for PbI2 rods, records the mode in cache signatures, and plots
  raw `background_density` against full `joint_fit_density`.
- Background image subtraction is temporarily disabled by default for saved
  background images. The generated `peak_subtracted` detector/caked images stay
  raw while fitted peak models remain saved separately for inspection.
- PbI2 nonzero HK rods use the shared `0.5 <= L <= 3.0` display window and
  linear y axes. HK=0/specular uses a log y axis and derives its editable
  profile/preview support from the phi/2theta ROI, not an L min/max helper.
- Qr-rod marker editing now runs in explicit phases: nonzero HK rods first,
  `HK=0`/`00L` second, and detector-label placement last. The accepted `HK=0`
  phase state drives the detector specular band and `00L_region.png` mask.
- Final Qr-rod profile figures no longer run or draw the joint Qz peak fit.
  The final panels use the same accepted `background_density` traces shown in
  the GUI and overlay a second dashed trace smoothed by the GUI `Smooth` slider.
  The smoothing amount is stored in Qr-rod edit JSON with the nonzero/HK=0
  region parameters and participates in final cache/policy identity.
- Saved HK=0 Qr-rod profile figures now match the GUI's no-L-window behavior:
  the accepted phi/2theta ROI trace is not clipped by stale specular L bounds,
  and the final HK=0 x-axis is left to autoscale from the plotted ROI data.
- The legacy `scripts/diagnostics/comparison.py` duplicate no longer carries its
  own stale Qr-rod plotting implementation. It delegates to the maintained
  parallel diagnostic, preventing old entry points from rewriting manuscript
  figures with the obsolete `Simulation` overlay and forced HK=0 L axis.
- The batch runner default and the Bi2Te3/Bi2Se3/PbI2 batch file now point at
  `all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`,
  so reruns do not fall back to the removed notebook or a stale generated copy.
- Configured-hidden rods such as `HK=7` are skipped in the editor/support/final
  Qr-rod plots without changing the generated profile CSV tables.
- The legacy `all_background_peak_fits.ipynb` diagnostic artifact is removed
  from the active path. Regression coverage that still exercised joint-Qz
  helpers now loads the maintained parallel `.py` script instead.
- Headless PbI2 debug runs now use the default `auto` Qr-rod marker edit mode,
  which still opens the popup on interactive Matplotlib backends but skips it
  when `RA_SIM_HEADLESS` or CI mode is active.
- PbI2 generated rods remain disabled by default and final profile rows still
  require complete detector support, so the unsupported `m=7` row is skipped.
- The `HK=0` / `00L` marker-editor phase now receives the pre-editor specular
  profile rows even after the nonzero-HK phase has replaced the active profile
  table. If those real rows are unavailable, the popup still opens with
  marker-only fallback rows and prints an explicit diagnostic.
- L-bound profile refreshes now reject tables that have drawable rows only for
  the wrong editor phase. This keeps the `HK=0` phase on its last valid
  specular intensity curve when the callback returns only nonzero rows.
- PbI2 nonzero marker-derived L mappings now require a minimum L-span relative
  to their Qz span before they can drive the profile x-axis. Collapsed mappings
  like the observed `m=4 -` branch fall back to the lattice/Qz L axis, matching
  the behavior of branches whose markers do not carry usable L references.
- PbI2 `HK=0` marker rows now honor the same active specular L window before
  pre-editor integration. Stale below-window rows such as `L=1` are filtered
  out, and active-lattice fallback markers are generated only through the
  active HK=0 display maximum, so the final figure can include the `m=0`
  integration row below the nonzero rods.
- Real HK=0/qz profile construction now uses a dedicated builder that merges
  active edited markers with specular lattice markers, derives fresh qz bins
  from the active detector phi/2theta ROI, and integrates detector pixels in
  that ROI. The recompute path calls this builder for `m=0`, `branch="qz"`
  before iterating cached/base nonzero profile groups, and edited `00L` markers
  are passed into the recompute callback.
- Split Qr-rod marker editing now imports saved edit JSON before launching the
  nonzero and `HK=0` phases, keeps edit-file writes until the final specular
  phase is accepted, carries the detector companion preview across both phases,
  records nonzero L/theta_i controls, and records HK=0/specular phi/2theta ROI
  bounds in final-fit cache identity.
- PbI2 manuscript figure outputs now default to
  `C:\Users\Kenpo\OneDrive\Documents\GitHub\PhD Work\2D-Manuscript-Draft\figures\results_pbi2`.
  Non-PbI2 samples still default to `results_ordered`, and
  `RA_SIM_ALL_BACKGROUND_FIGURE_OUT_DIR` / `FIGURE_OUTPUT_DIR` remain explicit
  overrides. Intermediate cache/output files under `OUT_DIR` are unchanged.

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
- Stale `HK=0` marker projection in exported images is fixed by rebuilding
  specular detector angles from the final post-editor Qz positions before the
  detector and L3 star images are written.
- Empty final Qr-rod profile rows are fixed by excluding rod entries whose
  branches have no drawable positive-L profile data.
- PbI2 manuscript figure misrouting is fixed: when the active sample resolves
  to a `pbi2` stem after state/CIF sample detection, final figure artifacts are
  written under `results_pbi2`. Other samples keep `results_ordered`, and an
  explicit figure-output override still wins.
- Misleading PbI2 nonzero dashed overlays are fixed by plotting raw central rod
  data, adding the Qr sideband background back to available fits, and recording
  marker/L mapping or Qz-baseline cancellation as diagnostics instead of
  suppressing `m=3`/`m=4` overlays.
- The Qr-rod marker-label helper ordering bug is fixed; profile annotation and
  redraw paths no longer call `rod_marker_annotation_label(...)` before it is
  defined.
- The Qr-rod L-axis helper ordering bug is fixed in the generated `.py`
  diagnostic; cached pre-editor reruns now define `qz_values_to_l_axis(...)`
  before the editor L-window setup calls `rod_profile_l_window_from_table(...)`.
  This resolves the observed PbI2 `NameError` before the marker editor can
  open.
- The Qr-rod L-window textbox crash path is fixed by validating profile-refresh
  tables before the popup replaces its last good profile table. If a refresh
  triggered by `L Min` or `L Max` returns no drawable rows, invalid `m`,
  invalid Qz, invalid density, or blank branch labels, the editor records
  `profile_update_error`, keeps the previous drawable profile table, refreshes
  the detector preview where possible, and remains open.
- The `HK=0` / `00L` profile-blanking path is fixed by treating nonzero-only
  refresh output as invalid for the specular phase. The editor keeps the real
  pre-editor HK=0 intensity rows instead of replacing them with unrelated
  nonzero rows or marker-only dummy intensity.
- The PbI2 `m=4 -` final-profile compression is fixed by rejecting
  near-flat marker L mappings for nonzero PbI2 branches. The full detector
  profile data remains in the CSV; only the plotted/profile-window L mapping is
  corrected, and the mapping guard is recorded in the Qr-rod profile cache
  policy signature.
- The missing PbI2 final `m=0` integration row is fixed by deriving the active
  specular support from the HK=0 phi/2theta ROI before computing the detector
  specular profile. HK=0 marker rows are seeded from the same specular detector
  Qz map used for integration, and the pre-editor Qr-rod cache signature was
  advanced so stale cached marker tables are not reused. If the direct `Qr=0`
  detector band has no pixels in the active ROI, the HK=0 profile falls back to
  the same detector-space specular ROI strip so the final m=0 row is still
  drawn.
- Partial edit-file overwrite is fixed. The nonzero phase no longer writes
  `RA_SIM_QR_ROD_PEAK_EDITS` output by itself; final persistence happens only
  after the HK=0/specular phase accepts the combined marker table.
- Stale final-fit cache reuse across different HK=0/specular editor bounds is
  fixed by adding the specular phi/2theta ROI to the final Qr-rod fit cache
  policy.
- Detector companion preview availability across the split editor is fixed by
  showing it during the nonzero phase and keeping it alive until the final
  specular phase closes.
- The L-window textbox redraw boundary is also guarded. Matplotlib widget
  callbacks are dispatched through the backend event loop, so redraw exceptions
  can otherwise be printed by Matplotlib as callback tracebacks while leaving
  the operator with a broken-feeling popup. `L Min` / `L Max` redraw failures
  are now recorded as `redraw_error` in the editor region state, and the popup
  remains open for recovery.
- The real non-headless failure was the editor's global Enter key shortcut
  racing the Matplotlib `TextBox` submit path. Pressing Enter inside `L Min` or
  `L Max` submitted the text field, then the figure-level key handler also
  treated that same keypress as Accept and closed the popup. The key handler
  now ignores active editor text boxes, including the L-bound controls, so the
  field submit is owned by the TextBox and Accept remains explicit.
- The detector selected-region label editor was accidentally bypassed by the
  unified Qr-rod marker-editor flow. The final detector-region save path now
  calls `edit_detector_region_label_positions(...)` before the last label draw
  and save, passing the existing settings path, Matplotlib backend, and
  environment so interactive backends still open the popup and headless/CI runs
  still skip it.
- The detector label editor popup then froze on click because the Matplotlib
  text-artist drag path redrew the detector figure during label movement. The
  editor now uses a Tk `Canvas` with independent text items over a temporary PNG
  snapshot, so dragging updates the small canvas item and converts back through
  the existing Matplotlib data transform only when coordinates change.
- Gated callback diagnostics are available with
  `RA_SIM_QR_ROD_EDITOR_DEBUG=1`. Set
  `RA_SIM_QR_ROD_EDITOR_DEBUG_LOG=<path>` to append JSON-line phase records for
  L-bound submits, rejected profile refreshes, accepted profile refreshes, and
  redraw errors. These diagnostics are off by default and do not change fitting
  inputs or artifact schemas.
- PbI2 `HK=4 -` marker edits no longer change the visible panel range/scope.
  The fix is display-only inside the diagnostic marker popup: marker edits
  preserve the current axis limits, while accepted marker coordinates and final
  fit inputs still update exactly as before.
- Existing unrelated non-parallel notebook test failures remain out of scope.

Feature status:

- `hk0_l3_star.png` is implemented in the diagnostic `.py` as a raw detector
  crop from the beam center through and above the `HK=0`, `L=3` / `00L`
  marker, rendered with detector-style color and log intensity normalization.
- The detector selected-region figure uses dedicated high-contrast styling for
  the central `HK=0` / `00L` rod. This is a display-only change; Qr/Qz maps,
  Delta-Qr values, selected masks, integration, fitting, and cache identities
  are unchanged.
- The detector selected-region label editor now supports Tk-canvas picking and
  dragging for Qr-region labels. The helper signature and
  `ra_sim.detector_label_settings.v1` JSON payload are unchanged, and the
  temporary PNG/window are cleaned up before the final figure is saved.
- The final detector selected-region save path now invokes that existing label
  editor after Qr-rod region edits and before drawing the saved labels. This
  restores the operator opportunity to drag/nudge labels on the actual detector
  figure and to import/export the existing detector-label settings JSON.
- The unified Qr-rod editor owns marker editing, Delta Qr, and L-window changes
  before the final joint Qz fit. Detector label placement remains a separate
  detector-figure step after that editor, using the existing label editor and
  `ra_sim.detector_label_settings.v1` import/export interface.
- The unified Qr-rod editor now also receives a detector companion figure in
  popup mode. This is display-only context for the operator and does not alter
  Qr/Qz maps, fit inputs, cache signatures, exported artifact schemas, or
  headless/skip behavior.
- The detector companion figure is now wired to the unified editor's existing
  region-control refresh path. Delta Qr edits redraw the preview overlays using
  the current marker table and defer profile reintegration until release or
  accept; L-window text submissions perform one immediate profile refresh.
  This is still display-only/workflow behavior and does not add a CLI/config/
  saved-state surface.
- The companion preview show path now explicitly maps the preview figure before
  `plt.show(block=True)`, while the marker editor close event still closes the
  companion window.
- 2026-05-12 closeout: the final simplification pass only collapsed duplicate
  detector-preview band rendering into a local helper. It did not add files,
  dependencies, public interfaces, config keys, saved-state fields, artifact
  fields, CI jobs, or deployment automation.
- Qr-rod peak marker editing is implemented in the diagnostic `.py` with
  default `RA_SIM_QR_ROD_PEAK_EDIT_MODE=auto`; `popup` forces the editor, `skip`
  disables it for unattended runs, and optional JSON round trip is available through
  `RA_SIM_QR_ROD_PEAK_EDITS`.
- The marker editor has `Import` and `Export` buttons that round-trip the same
  JSON marker table as `RA_SIM_QR_ROD_PEAK_EDITS`, including adjusted
  positions and final-figure label text.
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
- Bi2Se3 is now the default local state for the parallel diagnostic script.
  The final Qz fit cache signature is `joint_qz_labeled_marker_fit_specular_theta_i0_l8_v9`
  so cached fits from the older marker gate, baseline-sensitive weak-marker
  gate, pre-log-residual full-profile objective, and pre-PbI2 sideband plot
  policy are recomputed.
- Supported weak Bi2Se3 low-L specular markers are preserved through
  aggregation and nonlinear refinement even when their initial component
  amplitude is below the relative model threshold. The current bug fix also
  seeds a labeled weak component from raw-profile local prominence when the
  preliminary shared baseline is too high at the shoulder. Unsupported
  projected markers in the same local profile window remain excluded, and the
  Bi2Te3 weak marker at the same nominal location remains fitted.
- Final nonlinear Pearson-VII Qz refinement now appends a bounded log-intensity
  residual to the existing intensity-weighted residual. This improves the
  full-profile fit seen on the log-scaled Qr integration plot while preserving
  the existing `fit_joint_qz_peak_sum(qz_center, background_density, qz_markers)`
  helper interface.
- PbI2 raw-data/additive-sideband plot policy is implemented for the diagnostic
  `.py`. The generated markdown includes a `Plot model decisions` table with
  the plotted data source, fit source, added-background source, and diagnostic
  flags per branch.
- PbI2 nonzero-branch L-axis display now rejects missing, conflicting, or
  non-monotonic marker-derived `qz -> L` mappings and falls back to the active
  lattice `c` spacing. This fixes the case where detector overlays were in the
  right locations but same-`m` `+/-` profile panels looked unrelated because one
  branch used raw Qz or a bad marker fit as the displayed L axis.
- PbI2 no-background debug mode is implemented for the diagnostic `.py` through
  `PBI2_DISABLE_BACKGROUND_SUBTRACTION_OVERRIDE` or
  `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION`. It is diagnostic-only and does
  not add a GUI control, CLI flag, saved-state field, or package API surface.
- Tail-component aggregation now fails closed when `x`, target, finite-mask,
  and model shapes are inconsistent instead of partially replacing only one
  mask and risking a mismatched array index.
- CI/deployment automation was reviewed and left unchanged for this local
  diagnostic/publication slice. The applicable quality gate remains the focused
  diagnostics pytest, scoped ruff, compile, whitespace check, and guarded
  headless script path before a manuscript run or PR; no workflow files,
  deployment jobs, secrets, or branch protections were changed.
- No deprecation path or user migration is required. The only compatibility
  action is the final-fit cache signature bump to v8, which forces stale cached
  Qr-rod final fits to be recomputed.
- Shipping status is local-only. The rollback path is to revert the focused
  diagnostic script/test/doc commits (`ec09c152` for PbI2 plot stabilization and
  `a261a8b` for the detector-label editor), delete regenerated local artifacts
  if needed, and allow older cache signatures to be regenerated by the previous
  fit logic.
- Windows CPU parallelization is implemented through the guarded runner path.
  A Bi2Se3 run on 2026-05-07 reported `backend=process_pool`, `pids=28`, and
  `global peak fitting elapsed=22.83s`, versus the direct-thread report of
  `backend=thread_pool`, `pids=1`, and `elapsed=220.07s`.
- Direct Windows `.py` execution now enters that guarded runner path
  automatically for the default process backend, while preserving the existing
  thread/serial opt-outs for debugging.
- Pre-editor cache reuse is implemented in the diagnostic `.py`. Use
  `RA_SIM_RESET_PRE_EDITOR_CACHE=1` to force recomputation of cached global
  peak fits, line-profile fits, and Qr-rod pre-marker profile data. The cache
  does not bypass the marker editor or reuse final joint Qz fits across changed
  marker edits. Sample/output label overrides are excluded from this cache
  identity, so changing only the output stem from `Bi2Se3` to `Bi2Te3` can reuse
  the same cached fit data.
- The helper interface is internal to the diagnostic script; no CLI, config,
  saved-state, or package API surface changed.
- The obsolete internal `edit_qr_rod_peak_markers(...)` wrapper was removed
  after the unified editor became the only active marker/region edit path.
  `marker_qz_values_for_profile(...)` and the Qr-rod pickle cache are still
  referenced in the current `.py` diagnostic and remain in place.
- 2026-05-11 closeout: the unified editor remains the sole Qr-region and
  peak-marker editing path. The deprecation was internal only: no operator
  migration, CLI/config change, saved-state change, or cache-schema change is
  required beyond the previously documented Qr-rod fit-cache invalidations.

## Validation

Passing checks:

- `.py` parse and compile.
- Guarded runner `.py` source execution and Windows process-backend guard tests.
- Direct Windows `.py` process-backend re-entry coverage verifies default
  process/auto runs launch the guarded runner before preparation or backend
  normalization, while explicit thread/serial backends remain direct opt-outs.
- Targeted `hk0_l3_star.png` helper and wiring coverage for crop bounds,
  edge clipping, invalid inputs, `HK=0`, `L=3` marker selection, synthetic PNG
  save, detector-style color/log rendering, and diagnostic call-site wiring.
- Targeted detector selected-region source/helper coverage verifies the
  central `HK=0` / `00L` rod uses the dedicated high-contrast style,
  same-width centerline without a halo, stronger Delta-Qr band styling,
  unchanged Delta-Qr mask-builder input, and expanded boundary mask.
- Targeted detector label-editor coverage verifies Tk canvas controls, complete
  label event bindings, data-space coordinate conversion, Tk mainloop use
  without Matplotlib event-loop blocking, cleanup of the temporary PNG/window,
  runtime-mode handling, settings round trip, and final-save wiring.
- Targeted Qr-rod marker edit coverage for per-rod marker replacement,
  marker-table cache-key hashing, headless/interactive mode resolution, JSON
  edit round trip including `marker_title`, final-label override behavior,
  in-popup import/export wiring, Delta Qr/L-window control state, live
  integrated-profile refresh callback wiring, accepted profile-table handoff,
  sample-name override wiring, `HK=0` specular marker inclusion, and editor call
  ordering before the joint-fit cache lookup.
- Targeted figure-output routing coverage verifies PbI2 samples default to
  `results_pbi2`, non-PbI2 samples stay on `results_ordered`, explicit
  figure-output overrides win, and figure-output refresh occurs after state/CIF
  sample detection.
- Targeted weak-peak regression coverage verifies a labeled HK=0 weak marker
  at `006`-like relative intensity remains present in the final joint-fit
  component list.
- Targeted snap coverage verifies that all markers in a selected profile panel
  move to nearby local maxima.
- Earlier fast project check tier:
  `python -m ra_sim.dev check` passed with `280 passed`.
- Parallel-notebook pytest checks:
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_process_pool_worker`
  and
  `tests/test_background_peak_fits_notebook.py::test_parallel_background_peak_fits_notebook_uses_gaussian_core_lorentzian_tail_model`
  both pass.
- Pre-editor cache regression coverage verifies filename/signature identity,
  pickle-envelope stage round trip, stage validation, reset behavior, and cache
  lookup ordering before the expensive fit/profile/editor stages.
- Post-editor marker export regression coverage verifies duplicate-HKL manual
  rows are retained and edited `HK=0` Qz markers are converted to the updated
  detector theta/phi used by downstream exported images.
- Final Qr-rod profile row regression coverage verifies empty rods such as the
  observed Bi2Te3 `HK=7` case are excluded before subplot rows are allocated.
- Bi2Se3/Bi2Te3 low-L specular marker regression coverage uses embedded
  artifact-shaped m=0 profiles. It verifies the Bi2Se3 weak marker is included
  despite the over-high preliminary shared baseline, the unsupported nearby
  Bi2Se3 marker is excluded, Bi2Te3 still keeps the matching weak marker, the
  final component counts stay fixed, and the Bi2Se3 m=0 full-profile log RMS
  stays below the regression threshold.
- Measured embedded-profile status after the log-residual slice: Bi2Se3 m=0
  full-profile log RMS is `0.0473` and low-L tail-window log RMS is `0.0613`;
  Bi2Te3 m=0 full-profile log RMS is `0.0591`.
- Tail-component shape-mismatch coverage verifies the helper returns no
  components instead of raising or synthesizing unsupported data.
- PbI2 sideband and plot-policy coverage verifies same-Qz Qr sideband
  subtraction keeps raw, sideband, and corrected densities available; PbI2
  final plots use raw central profiles as data and add the sideband background
  back to available total `joint_fit_density` overlays; invalid marker/L
  mappings fall back to the active lattice for display while Qz-baseline
  cancellation remains diagnostic-only; non-PbI2 model selection is unchanged;
  unsupported detector-incomplete rows such as `m=7` stay hidden; nonzero PbI2
  panels use linear y axes while `HK=0` stays log-scaled.
- PbI2 no-background debug coverage verifies the environment/local override
  disables transverse sideband subtraction, changes both pre-editor and final
  Qr-rod cache signatures, and makes the plot policy compare raw
  `background_density` to full `joint_fit_density` without data-minus-baseline
  subtraction.
- Pre-fit background coverage verifies the legacy lower-percentile annulus
  fallback, the central-phi caked plane fit, scale-zero no-op behavior,
  scale-one subtraction, optional negative-value clipping, popup wiring before
  background preparation, cached model reuse on scale changes, corrected caking
  and Qr-rod profile inputs, raw detector preservation, saved diagnostics, and
  pre-editor/final Qr-rod cache keys that change with the subtraction policy.
- Focused PbI2 acceptance command passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "qr_rod_peak_editor_is_wired_before_joint_fit_cache or qr_rod_peak_edit_runtime_mode_respects_headless or qr_sideband or pbi2_plot_policy or pbi2_debug or background_debug_policy or final_profile_plot_uses_model_decisions or pbi2_rod_profile_l_axis or pbi2_final_profile or shared_nonzero_rod_profile_y_axis_limits" -ra`
  with `16 passed`, including the no-background debug flag and default-auto
  headless editor coverage.
- Headless PbI2 script execution passed with
  `RA_SIM_HEADLESS=1 RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION=1`; it skipped
  marker and detector-label popups through default `auto` mode, regenerated the
  PbI2 Qr-rod profile artifacts, recorded sideband subtraction disabled, kept
  `background_density == background_density_raw`, used log axes on all PbI2
  panels, and skipped unsupported `m=7` in the final figure.
- Focused command passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "joint_qz_fit or rod_profile_panels_use_centered_m_labels or qr_rod_marker_hash_changes_cache_key or marker_title_changes_cache_key or qr_rod_final_cache_requires_fit_signature or tail_component_aggregation_rejects_shape_mismatch" -ra`
- Compile check passed for package, tests, and the diagnostic script:
  `python -m compileall -q ra_sim tests scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
- Focused unified-editor command passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_qr_rod_region_editor or unified_editor or detector_region_label_editor_wires_before_final_save or saved_figures_do_not_include_panel_letters or initial_placement_uses_default_geometry or axis_tick_labels_use_bottom_left_origin or qr_rod_peak_editor_is_wired_before_joint_fit_cache or pre_editor_cache_is_checked_before_expensive_stages or qr_rod_peak_editor_uses_l_axis" -ra`
  with `13 passed`.
- Focused simplification closeout command passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_qr_rod_region_editor or unified_editor or qr_rod_peak_editor_uses_l_axis or import_export_buttons or qr_rod_peak_editor_shows_hk0_in_log_view or qr_rod_peak_editor_is_wired_before_joint_fit_cache or pre_editor_cache_is_checked_before_expensive_stages" -ra`
  with `11 passed`.
- Qr-rod editor startup regression passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py::test_parallel_script_qz_l_axis_helper_is_defined_before_editor_l_window_setup -ra`.
- Runtime-safe PbI2 diagnostic script execution passed with
  `RA_SIM_QR_ROD_PEAK_EDIT_MODE=skip`; it reached
  `Qr-rod region editor: mode=skip source=last_cached` and exited `0`, proving
  the former `NameError` path is cleared without requiring a blocking popup.
- Compile closeout passed for the touched diagnostic/test files:
  `python -m compileall -q scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py tests/test_background_peak_fits_notebook.py`.
- Popup companion-preview regression coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_qr_preview_is_passed or companion_figures_before_show" -ra`
  with `2 passed`.
- Live companion-preview refresh regression coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "region_controls_update_preview or detector_qr_preview_is_passed_to_unified_editor" -ra`
  with `2 passed`.
- Companion-window show regression coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "companion_figures_before_show or region_controls_update_preview or detector_qr_preview_is_passed_to_unified_editor" -ra`
  with `3 passed`.
- Delta Qr deferred-integration regression coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "region_controls_update_preview or delta_qr_refreshes_profile_table_on_release or accept_flushes_pending_profile_refresh" -ra`
  with `3 passed`.
- Compile closeout passed for the touched diagnostic/test files:
  `python -m compileall scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py tests/test_background_peak_fits_notebook.py`.
- Focused popup/editor regression coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "qr_rod_peak_editor or detector_qr_preview or companion_figures or unified_editor" -ra`
  with `11 passed`.
- Focused popup/editor live-preview closeout passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_editor or detector_qr_preview or companion_figures" -ra`
  with `8 passed`.
- Focused popup/editor deferred-integration closeout passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_editor or detector_qr_preview or companion_figures" -ra`
  with `9 passed`.
- Focused L-bound crash regression passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "l_bounds_reject_invalid_profile_refresh or l_bounds_keep_popup_on_redraw_error or l_bound_enter_does_not_accept_editor or region_controls_update_preview or delta_qr_refreshes_profile_table_on_release or accept_flushes_pending_profile_refresh" -ra`
  with `6 passed`.
- Non-headless QtAgg editor repro passed by opening the real Matplotlib event
  loop, scheduling `L Min=0.5` and `L Max=2.5` textbox submissions through a
  GUI timer, refreshing profiles twice, and closing the popup accepted without
  callback traceback. This used Matplotlib `3.10.3` with backend `qtagg`.
- The non-headless QtAgg repro was rerun with TextBox Enter events for
  `L Min=0.5`, `L Max=2.5`, `L Min=0.25`, and `L Max=2.75`. The popup stayed
  open through each L-bound Enter keypress, accepted only through the Accept
  button, and returned final state `l_min=0.25`, `l_max=2.75`.
- Focused detector-label restoration coverage passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_region_label_editor or detector_region_labels or detector_region_label_settings or unified_editor or l_bound" -ra`
  with `24 passed`.
- Non-headless QtAgg detector-label editor validation passed by opening the
  real Matplotlib event loop, dragging a label from `(40,50)` to `(60,70)`,
  increasing font size, exporting detector-label settings, moving the label
  again, importing the saved settings, and accepting. The editor returned
  `text="m = 7 moved"`, `label_xy=(60,70)`, `fontsize=10`, and saved schema
  `ra_sim.detector_label_settings.v1`.
- Focused detector-label visibility regression passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_region_label_editor_shows_figure_before_event_loop" -ra`
  with `1 passed`.
- Focused detector-label/unified-editor regression coverage passed after the
  visibility fix:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_region_label_editor or detector_region_labels or detector_region_label_settings or unified_editor or l_bound" -ra`
  with `25 passed`.
- Non-headless QtAgg detector-label visibility validation passed with backend
  `qtagg`: the live window reported `visible=True`, printed
  `detector label editor: popup open`, dragged a label from `(40,50)` to
  `(60,70)`, exported/imported detector-label settings, and accepted with
  schema `ra_sim.detector_label_settings.v1`.
- Final local shipping-gate closeout for this slice passed the same focused
  popup/editor regression command with `12 passed`, compiled the touched
  diagnostic/test files, and passed
  `git diff --check -- scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py tests/test_background_peak_fits_notebook.py`.
- Earlier project check passed for this worktree after the detector
  companion-preview closeout: `python -m ra_sim.dev check` reported
  `All checks passed!`, `281 passed`, and no mypy issues.
- Project check was rerun after the L-bound callback fix:
  `python -m ra_sim.dev check` reported `All checks passed!`, `281 passed`,
  and no mypy issues.
- Final detector-label restoration closeout passed:
  `python -m compileall scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py tests/test_background_peak_fits_notebook.py`,
  `git diff --check -- CHANGELOG.md docs/tracking/in-progress/background-peak-fit-detector-qr-rod-panel.md scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py tests/test_background_peak_fits_notebook.py`,
  and `python -m ra_sim.dev check` with `All checks passed!`, `281 passed`,
  and no mypy issues.
- Visibility-fix closeout passed compileall for the touched diagnostic/test
  files, scoped ruff on those files, scoped `git diff --check`, and the
  focused detector-label/unified-editor regression command with `25 passed`.
- Tk-canvas detector-label closeout passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_region_label_editor" -ra`
  with `9 passed`; scoped ruff and compileall passed for the touched
  diagnostic/test files; a real Tk smoke opened the popup, found the
  `detector_label` canvas item, dragged it, accepted, and returned label
  coordinates changed from `[40.0, 50.0]` to approximately `[59.481, 59.74]`.
- `python -m ra_sim.dev check` was rerun after the Tk-canvas label-editor fix
  and is currently blocked before tests by unrelated dirty work outside this
  slice: `ra_sim/fitting/optimization.py` would be reformatted by
  `ruff format --check`.
- Runtime-safe PbI2 diagnostic script execution passed again with
  `RA_SIM_HEADLESS=1 RA_SIM_QR_ROD_PEAK_EDIT_MODE=skip`; it reached
  `Qr-rod region editor: mode=skip source=last_cached` and completed in the
  guarded runner without constructing the popup-only detector preview.
- Focused PbI2 L-axis/y-axis regression passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "pbi2 or rod_profile_l_axis or final_profile_plot" -ra`
  with `11 passed`, plus scoped `compileall` and `ruff` on the touched
  diagnostic/test files.
- Read-only PbI2 artifact sanity using the patched helper and active
  `c=6.78 A` showed the formerly divergent branch display ranges now agree:
  `m=3 +/-` both span about `L=0.49..2.98`, and `m=4 +/-` both span about
  `L=0.53..3.0`.
- 2026-05-12 shipping closeout: no release/version bump, public API change,
  saved-state change, artifact-schema change, CI workflow change, deprecation
  notice, or user migration is required. The bug is fixed in the diagnostic
  script/tests/docs, with broader `python -m ra_sim.dev check` still blocked by
  unrelated dirty formatting in `ra_sim/fitting/optimization.py`.
- Focused whitespace check passed:
  `git diff --check`
- 2026-05-12 PbI2/no-subtraction/notebook-migration closeout passed focused
  Qr-rod regressions, full `tests/test_background_peak_fits_notebook.py`, full
  `tests/test_geometry_fitting.py`, scoped compile, scoped Ruff, and
  `git diff --check`; `python -m ra_sim.dev check` also passed with
  `281 passed`. No release/version bump, CI workflow change, public API change,
  config schema change, saved-state migration, or artifact schema migration is
  required. Rollback is a normal git revert of the diagnostic script/test/doc
  changes.
- 2026-05-20 HK=4 minus marker-range closeout passed:
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "hk4_minus_drag_preserves_panel_limits or qr_rod_peak_editor_click_preserves_panel_limits" -ra`
  with `2 passed`. Bug/error status: fixed for the observed marker-editor
  range/scope mutation. Feature status: no new operator control, CLI/config
  key, saved-state field, artifact schema, dependency, CI workflow, release
  version, migration, or ADR. Rollback is a normal git revert of the diagnostic
  script/test/doc changes.
- 2026-05-20 HK=4 minus follow-up: the remaining editor-only range/scope
  change was traced to the popup's local Qz-to-L conversion still using raw
  marker-row regression, while the final/profile path already rejected the
  compressed PbI2 `HK=4 -` mapping and fell back to the active lattice L axis.
  The marker editor now uses the same validated `qz_to_l_linear_coeff(...,
  marker_source=edited)` policy, and marker press/drag/release snapshots the
  active panel limits so marker-only interactions cannot mutate the visible
  panel range. Focused validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "qr_rod_peak_editor or unified_editor or marker_group or qz_l_axis or marker_l_mapping" -ra`
  (`29 passed`), scoped compileall, scoped Ruff format/check, and
  `git diff --check`; full `python -m ra_sim.dev check` is blocked by an
  unrelated dirty `ra_sim/gui/_runtime/runtime_session.py` format change.
- 2026-05-20 HK=0 L Min follow-up: this earlier slice deferred specular L-bound
  profile refresh work until Accept. The 2026-05-21 follow-up supersedes that
  control path: HK=0/specular no longer exposes `L Min` / `L Max`, and
  phi/2theta ROI edits define the specular profile, detector preview, final
  rows, and cache identity.
- 2026-05-20 HK=0 phase-isolation/PbI2-hidden-rod closeout: changing the HK=0
  Qr-rod marker editor controls now updates only HK=0/specular preview/profile
  state, and PbI2 `m=7` rows are filtered before artifacts. Focused validation
  passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "hk0 or specular or qr_rod_peak_editor or region_preview or final_profile_l_filter or detector_hk0 or 00l_region or unified_editor or pbi2_hidden_rods" -ra`
  (`56 passed`), scoped compileall, `git diff --check`, and
  `python -m ra_sim.dev check` (`294 passed`). Bug/error status: fixed for
  HK=0 editor edits mutating nonzero rod scope and for PbI2 `m=7` appearing in
  Qr-rod outputs. Feature status: no new operator control, dependency, config
  key, saved-state field, artifact schema, CI workflow, version bump, ADR, or
  migration. Rollback is a normal git revert.
- 2026-05-20 HK=0 lower-L support follow-up: the broad blue dashed region was
  traced to the previous fallback-mask workaround being drawn as the HK=0
  detector region. This path has since been superseded by the phi/2theta ROI
  specular path; profile integration, editor preview, and the final
  `00L_region` detector figure now share the same specular ROI instead of an L
  window or Delta-Qr band. The normal `specular_profile_mask_override` and broad
  fallback-mask path were removed.
  Cache signatures were advanced to `PRE_EDITOR_QR_ROD_STAGE_SIGNATURE=v11` and
  `QR_ROD_FINAL_FIT_CACHE_SIGNATURE=v12` so stale truncated or broad-fallback
  profiles/fits are recomputed automatically. A follow-up HK=0 disappearance
  regression fixed the detector-profile Qr lower bound so negative-Qr detector
  pixels inside the symmetric m=0 band are integrated instead of being clipped
  at `Qr=0`. Validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`172 passed, 8 skipped`), focused HK=0/cache tests (`5 passed`), scoped
  compileall, scoped Ruff, and `git diff --check`. `python -m ra_sim.dev check`
  was not clean because an unrelated staged `ra_sim/fitting/optimization.py`
  formatting change predates this slice. Bug/error status: fixed for HK=0
  integrated-region calculations not extending below the observed cutoff, fixed
  for the HK=0 detector region rendering as a broad fallback wedge, and fixed
  for the m=0 profile disappearing when only negative-Qr detector pixels remain.
  Feature status: no new operator control, dependency, config key, saved-state
  field, artifact schema, CI workflow, version bump, ADR, or user migration is
  required. Rollback is a normal git revert.
- 2026-05-21 real HK=0/qz rebuild closeout: the blank PbI2 HK=0 editor path was
  traced to marker-only fallback rows, recompute depending on pre-existing
  `qr_rod_editor_base_profiles`, and edited `00L` markers not being passed into
  profile recompute. The fix builds `m=0`, `branch="qz"` profiles directly from
  active marker/L-window state and detector Q maps, filters stale specular
  markers above and below the active window, advances
  `PRE_EDITOR_QR_ROD_STAGE_SIGNATURE=v12` and
  `QR_ROD_FINAL_FIT_CACHE_SIGNATURE=v13`, and rejects stale caches that contain
  HK=0 markers without positive-pixel real HK=0 profile rows. Focused
  validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "hk0 or specular or qr_rod_peak_editor or final_profile_l_filter or final_cache" -ra`
  (`54 passed, 133 deselected`), scoped compileall, scoped Ruff, and
  `git diff --check`. A headless PbI2 diagnostic rerun with stale caches reset
  completed under `C:\asr_work\ra_sim_hk0_final_20260521_015538`; the
  final rod profile table contained 96 HK=0/qz rows, 69 positive-pixel HK=0/qz
  rows, and the new "skipped HK=0 final figure row" diagnostic did not appear.
  Bug/error status: fixed for missing real HK=0 intensity traces, fixed for
  HK=0 controls being unable to create real rows when base profiles lacked
  HK=0, and fixed for stale cache reuse of no-m0 states. Feature status: no new
  operator control, dependency, config key, saved-state field, artifact schema,
  CI workflow, version bump, ADR, or user migration is required. Rollback is a
  normal git revert.
- 2026-05-21 nonzero import responsiveness follow-up: importing nonzero-HK
  Qr-rod marker edits no longer refreshes the detector companion once per
  imported branch group. The import path now batches the marker-table dirty
  mark and runs a single preview refresh before redraw; Accept still performs
  the deferred profile recompute once. Focused validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "import_refreshes_nonzero_preview_once or marker_changes_dirty" -ra`
  (`2 passed`) and the broader Qr-rod marker/HK0 selection
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "qr_rod_peak_editor or hk0 or specular or final_profile_l_filter or final_cache" -ra`
  (`55 passed, 133 deselected`). Bug/error status: fixed for the nonzero-HK
  import path appearing to hang after the HK=0 marker-table plumbing change.
- 2026-05-21 pre-fit background feature closeout: the parallel diagnostic now
  has a separate pre-fit background reducer independent from the existing
  post-fit/sideband background flags. It is off by default, can be previewed in
  a Tk/Pillow popup, accepts environment overrides for mode/scale and legacy
  radial policy fields, preserves the raw detector image, saves plane
  model/corrected detector arrays, and invalidates peak-fit/Qr-rod caches when
  the subtraction policy changes. Focused validation
  passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "radial_background or background_subtraction or qr_rod_peak_editor or hk0 or final_cache" -ra`
  (`46 passed, 155 deselected`), scoped compileall, scoped Ruff, and
  `git diff --check`. Bug/error status: no active crash path; this is a new
  optional feature. Feature status: implemented locally, automated source/helper
  coverage is passing, and manual popup acceptance against a real detector run
  remains the launch check.
- 2026-05-21 caked plane background follow-up: the pre-fit editor no longer
  uses the lower-envelope radial model for this workflow. It now fits a full 2D
  linear background plane in caked `(phi, 2theta)` space from `-90 <= phi <= 90`
  rows, shows that phi band with `2theta`/`phi` axis labels, and subtracts the
  detector-evaluated plane before caking, peak fitting, and Qr-rod profile
  extraction. The popup still uses Pillow `ImageTk.PhotoImage` on Tk canvases
  and keeps the accepted config interface stable. Focused
  validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "radial_background or background_subtraction or qr_rod_peak_editor or hk0 or final_cache" -ra`
  (`48 passed, 155 deselected`), after the narrower radial check
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "radial_background or background_subtraction" -ra`
  (`9 passed, 192 deselected`), scoped compileall, scoped Ruff, and
  `git diff --check`; the central-phi plane follow-up added focused caked-plane
  helper/source coverage and reran the related selector
  (`51 passed, 155 deselected`). Bug/error status: fixed for the editor using
  the wrong radial model and for showing detector images where the operator
  needed a central-phi caked plane preview. The clip-zero preview still re-cakes
  the detector-corrected image so it matches the fit pipeline semantics.
  Migration status: stale prefit-background caches are rejected by bumped
  pre-editor and final Qr-rod cache signatures; no saved-state migration is
  required. Feature status: implemented locally; manual real-data popup
  validation remains pending.
- 2026-05-21 marker-editor responsiveness closeout: Delta Qr slider motion now
  defers detector companion preview work until release or accept, the marker
  editor caches per-panel plot inputs across redraws, and preview callback
  errors stay in editor state instead of closing the popup. Full notebook-script
  validation passed with `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`203 passed, 8 skipped`), scoped compileall, scoped Ruff, and
  `git diff --check`. Bug/error status: fixed for the observed laggy marker
  editor interactions. Migration status: no saved-state, CLI/config, artifact
  schema, dependency, CI workflow, or version bump is required.
- 2026-05-21 HK=0 phi/2theta ROI closeout: the HK=0 / `00L` Qr-rod picker no
  longer uses theta_i or a Delta Qr band. The specular phase is defined by
  `phi_min`, `phi_max`, `2theta_min`, and `2theta_max`, the detector companion
  draws the same ROI as an arc/region overlay, the profile builder integrates
  that phi/2theta ROI, and the final cache policy records `specular_roi` rather
  than a no-op specular Delta Qr value. Focused validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "hk0_roi or specular_hk0 or delta_qr or final_cache or detector_region_specular_visual" -ra`
  (`16 passed, 202 deselected`) and full notebook-script validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`210 passed, 8 skipped`), followed by scoped compileall, scoped Ruff,
  `git diff --check`, and `python -m ra_sim.dev check` (`294 passed`, Ruff
  clean, mypy clean). Bug/error status: fixed for the obsolete HK=0
  theta_i/specular-Delta-Qr path and for stale no-op Delta Qr cache identity.
  Migration status: stale final/pre-editor Qr-rod caches are rejected by the
  bumped signatures; saved-state, CLI/config, artifact schema, dependency, CI
  workflow, and version changes are not required. Feature status: implemented
  locally; manual real-data visual review of the HK=0 ROI arc and profile row
  still remains before manuscript use.
- 2026-05-21 nonzero HK theta_i editor slice: the nonzero Qr-rod marker editor
  now exposes a numeric `theta_i` text field. Submitting it updates the editor
  state, detector companion preview, deferred profile refresh, nonzero detector
  Q-map rebuild, and final Qr-rod cache policy; the HK=0 phase stays on the
  phi/2theta ROI path. Focused validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_editor_result_updates_final_profile_table or unified_editor_has_region_controls or nonzero_editor_theta_i_textbox or recompute_nonzero_theta_i_changes_q_map_cache_key or recompute_creates_hk0_when_base_profiles_lack_hk0" -ra`
  (`5 passed, 215 deselected`), full notebook-script validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`212 passed, 8 skipped`), and `python -m ra_sim.dev check` passed
  (`294 passed`, Ruff clean, mypy clean). Feature status: implemented locally;
  manual real-data visual review remains before manuscript use.
- 2026-05-21 nonzero branch L-start/specular-L-control fix: nonzero Qr-rod
  profile refresh now rebuilds each plus/minus branch from a shared active L
  grid and trims to common finite detector support for that `m`, so `m=1 +` and
  `m=1 -` keep aligned displayed L starts even when branch-specific Qz marker
  mappings or empty detector-support bins differ. The HK=0/specular editor no
  longer exposes or applies `L Min` / `L Max`; specular profile, preview, panel
  range, final rows, and cache identity come from phi/2theta ROI data. Focused
  validation passed with `python -m pytest tests/test_background_peak_fits_notebook.py -k "recompute_nonzero_branches_use_same_l_start or visible_branch_starts or recompute_specular_ignores_l_bounds or hk0_uses_specular_rows_without_l_controls or hk0_two_theta_submit or unified_editor_roi_refresh_keeps_specular_rows" -ra`
  (`6 passed, 217 deselected`), full notebook-script validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`215 passed, 8 skipped`), `python -m ra_sim.dev check` passed (`294 passed`,
  Ruff clean, mypy clean), scoped compileall passed, and `git diff --check`
  passed. Feature/bug status: implemented locally; no dependency, CI workflow,
  version, migration, public API, saved-state, config, or artifact-schema change
  is required.
- 2026-05-21 review/ship closeout: the residual pre-editor HK=0/specular path
  no longer clips detector support through an L-derived Qz window before the
  marker editor opens. The obsolete specular Qz-edge helper was removed, the
  specular profile builder now depends only on phi/2theta ROI inputs, and
  failure diagnostics report ROI support instead of obsolete L/Qz-window
  fields. Validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -k "specular_hk0 or specular_profile or recompute_specular_ignores_l_bounds or hk0_uses_specular_rows_without_l_controls or 00l_region_uses_specular_two_theta_roi or nonzero_branches_share_qz_bounds or visible_branch_starts" -ra`
  (`8 passed, 215 deselected`),
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`215 passed, 8 skipped`), scoped compileall, scoped Ruff
  format/check, `git diff --check`, and `python -m ra_sim.dev check`
  (`295 passed`, Ruff clean, mypy clean). Bug/error status: fixed for the
  observed nonzero branch L-start drift and for m=0 using obsolete L bounds
  instead of 2theta/phi ROI. Shipping status: local quality gates passed; no
  dependency, CI workflow, version, migration, public API, saved-state, config,
  or artifact-schema change is required.
- 2026-05-22 accepted L-axis closeout: final Qr-rod profile recomputation now
  freezes the GUI-accepted per-branch L-axis coefficients, includes them in the
  final profile cache identity, bypasses stale editor-cache rows after accept,
  and audits the GUI profile rows against the final profile rows. Final figure
  plotting uses the accepted GUI L mapping and `background_density` profile data
  consistently, while HK=0 axis limits follow the accepted specular controls.
  Focused validation passed with
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`240 passed, 8 skipped`). Bug/error status: fixed for stale final
  Qr-rod L-axis/profile rows after marker edits; no dependency, CI workflow,
  version, migration, public API, saved-state, config, or artifact-schema change
  is required.
- 2026-05-22 accepted L-axis simplification closeout: removed the unused
  `freeze_qr_rod_gui_state()` wrapper and its unused assignment after verifying
  that final Figure 7 state is consumed directly through
  `accepted_l_axis_coefficients`. Validation passed with scoped compileall,
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`240 passed, 8 skipped`), and `git diff --check`. Bug/error/feature status:
  behavior preserved; cleanup only. No CI workflow, deprecation, migration,
  version, public API, saved-state, config, artifact-schema, or launch-procedure
  change is required.
- 2026-05-22 post-cache Figure 7 source-of-truth closeout: final Qr-rod
  detector-region overlays now rebuild after final-fit cache hit/miss
  resolution and final marker/specular normalization, so saved region specs and
  detector-region masks consume the same post-cache `rod_profile_table` and
  marker table as the final profile figure. The GUI-vs-final audit now compares
  accepted GUI profile rows against post-cache final rows by `(m, branch,
  qz_center)` instead of comparing the final table to itself. Validation passed
  with `python -m pytest --assert=plain tests/test_background_peak_fits_notebook.py -ra`
  (`243 passed, 8 skipped`), scoped compileall, `git diff --check`, and
  `python -m ra_sim.dev check` (`295 passed`, Ruff clean, mypy clean).
  Bug/error status: fixed for stale final detector/profile images after cache
  reuse; feature status: no new operator control; migration/deprecation status:
  no dependency, CI workflow, version, public API, saved-state, config, or
  artifact-schema change is required.
- 2026-05-22 Figure 7 diagnostic source-state closeout: final Qr-rod output
  artifacts now pass through one internal output-state builder that derives the
  final detector overlays, saved region specs, marker plotting table, and
  GUI-vs-final audit from the same accepted post-cache profile state. Regression
  tests now poison pre-editor overlays, mutate post-cache profile rows, and
  exercise GUI fallback controls for `Delta Qr`, L bounds, `theta_i`, HK=0
  phi/2theta ROI, and marker aliases. Focused validation passed with
  `python -m pytest --assert=plain tests/test_background_peak_fits_notebook.py -k "final_output_state or final_outputs_ignore_poison_pre_editor_region_overlays or final_qr_rod_region_overlays_are_rebuilt_after_gui_editor or final_region_overlays_are_rebuilt_after_final_fit_cache_resolution or final_qr_rod_region_specs_are_saved_with_gui_fields or final_profile_audit_call_receives_accepted_gui_rows" -ra`
  (`7 passed, 247 deselected`), scoped compileall, and `git diff --check`.
  Earlier full notebook-script validation for the same code change passed with
  `python -m pytest --assert=plain tests/test_background_peak_fits_notebook.py -ra`
  (`246 passed, 8 skipped`); a later full rerun after documentation edits hit
  a Matplotlib access violation in the existing marker-editor close/preview
  tests, and an `MPLBACKEND=Agg` full rerun exposed an older backend-sensitive
  draw-count assertion outside this source-state slice.
  Bug/error status: diagnostic coverage is in place for final images drifting
  away from GUI-accepted regions after cache reuse; feature status: no new
  operator control; migration/deprecation status: no dependency, CI workflow,
  version, public API, saved-state, config, or artifact-schema change is
  required. Shipping status: ready for local use, with manual real-output
  visual review still recommended before manuscript use.
- 2026-05-22 radial background auto-scale closeout: blank
  `RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_SCALE` now estimates a robust
  sideband subtraction scale from held-out/off-peak detector pixels before the
  caked-plane popup opens. Explicit numeric scale overrides still win, and
  explicit `RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_MODE=off` keeps subtraction
  disabled. The popup fits its background model from the full caked phi range
  and only crops to `-90 <= phi <= 90` for display. Cache policy diagnostics
  now record the auto scale source and algorithm. Validation passed with scoped
  compileall, `python -m pytest --assert=plain tests/test_background_peak_fits_notebook.py -k "radial_background or auto_scale" -ra`
  (`16 passed, 244 deselected`), `git diff --check`, and
  `python -m ra_sim.dev check` (`295 passed`, Ruff clean, mypy clean).
  Feature status: complete for the caked-plane radial subtraction default-scale
  path. Bug/error status: no open failure in the auto-scale test slice.
  Migration/deprecation status: no dependency, CI workflow, version, public API,
  saved-state schema, or artifact-schema migration is required because explicit
  saved and environment overrides remain stable. Shipping status: local quality
  gates passed; real detector-output visual review remains recommended before
  manuscript use.
- 2026-05-22 output save-folder chooser: interactive diagnostic reruns now
  offer a save-folder chooser after the sample name is detected and before
  output/cache files are written. When accepted, the chosen directory becomes
  both `OUT_DIR` and `FIGURE_OUT_DIR` so CSV/JSON/NPY diagnostics and final
  figure files land together. Headless/CI runs skip the chooser, and explicit
  `RA_SIM_ALL_BACKGROUND_OUT_DIR` or `RA_SIM_ALL_BACKGROUND_FIGURE_OUT_DIR`
  overrides keep their existing deterministic behavior unless the chooser is
  forced with `RA_SIM_ALL_BACKGROUND_SAVE_DIR_EDIT_MODE=popup`. Focused
  validation passed with
  `python -m pytest --assert=plain tests/test_background_peak_fits_notebook.py -k "final_output_dir_choice or save_folder_chooser or radial_background or auto_scale" -ra`
  (`18 passed, 244 deselected`), scoped compileall, `git diff --check`, and
  `python -m ra_sim.dev check` (`295 passed`, Ruff clean, mypy clean).
  Feature status: complete for interactive output-directory selection.
  Migration/deprecation status: no schema, dependency, CLI, or CI workflow
  migration required.
- 2026-05-22 guarded-runner and settled-overlay closeout: direct Windows
  guarded background peak-fit reruns now skip the final save-folder chooser in
  default `auto` mode, preventing a hidden Tk folder dialog from stalling the
  child process after startup logging. Explicit
  `RA_SIM_ALL_BACKGROUND_SAVE_DIR_EDIT_MODE=popup` still forces the chooser and
  prints a waiting message first. Startup logging now reports the selected
  state path and background-file count before output-directory selection, so a
  future stall identifies its phase. Settled geometry-overlay redraws also
  discard view-bound `initial_pairs_display` coordinates and rebuild current
  manual-pair markers while preserving durable fitted overlay records.
  Validation passed with scoped compileall,
  `python -m pytest tests/test_background_peak_fits_notebook.py -ra`
  (`266 passed, 8 skipped`),
  `python -m pytest tests/test_gui_runtime_import_safe.py -k "refresh_settled_overlays or apply_main_caked_view_toggle" -ra`
  (`6 passed, 463 deselected`), `git diff --check`, and
  `python -m ra_sim.dev check` (`295 passed`, Ruff clean, mypy clean).
  Bug/error status: fixed for guarded child-process apparent hangs caused by
  default save-folder dialogs and for stale view-bound geometry-fit initial
  markers after simulation changes. Feature status: no new operator control;
  existing explicit popup/output override behavior is preserved. Migration and
  deprecation status: no dependency, CI workflow, version, public API,
  saved-state schema, config, or artifact-schema migration is required.
  Shipping status: ready for local use; rollback is a normal git revert of this
  commit because no data/schema migration is involved.

Known validation limits:

- Full `tests/test_background_peak_fits_notebook.py` is passing after the
  legacy notebook consumer migration.
- The 2026-05-10 Bi2Se3 weak-marker fix was validated with focused tests using
  embedded real-profile values, not a full real-sample script run.
- The PbI2 headless diagnostic path has been rerun. The L3 star crop,
  cache reuse, and interactive Qr-rod marker editor still need a real
  Bi2Se3/Bi2Te3 script run after this slice.
- The caked plane background popup still needs a manual interactive run with
  `RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_EDIT_MODE=popup`, blank or `auto`
  `RA_SIM_RADIAL_BACKGROUND_SUBTRACTION_SCALE`, and cache resets to verify the
  auto-scale slider default, labeled `-90 <= phi <= 90` caked plane preview,
  and scale-slider acceptance on current raw data.
- Visual acceptance still needs manual script-output review: colored detector
  background, HK labels near low-L rod bases, the central `HK=0` specular ROI,
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
