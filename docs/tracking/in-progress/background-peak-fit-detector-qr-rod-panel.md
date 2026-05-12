# Background peak fit detector Qr rod panel

Type: bug/feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-12
Status: implemented locally, Qr-rod editor startup fixed, focused validation passing

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
- The detector selected-region Qr-label editor now runs on the same Matplotlib
  detector figure instead of a separate Tk static-image canvas. Labels are
  selected and dragged directly in detector pixel coordinates, the active label
  text and font size can be adjusted in-figure, and import/export still use the
  existing detector-label JSON schema.
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
- The same Qr-rod editor now includes Delta Qr, `L Min`, and `L Max` controls.
  Delta Qr changes refresh the integrated profile table drawn in every subplot
  by reusing the existing detector Qr/Qz profile accumulator, and accepted
  Delta Qr/L values are applied to the profile rows used by the final joint Qz
  fit and cache key.
- Each Qr-rod peak marker now has an editable final-figure title. Blank titles
  fall back to `L=<rounded display_l>`, and the editor `Label` text box sets
  the exact text used when the final Qr-rod figure is drawn.
- Final Qr-rod figure labels are placed above and to the right of each peak,
  with a leader arrow pointing back to the marked peak.
- The joint Qz peak refinement now keeps weak labeled marker components instead
  of dropping them at the old 1% initial-amplitude gate, so labeled HK=0 peaks
  such as `006` remain included in the final fit.
- Final Qr-rod fit cache keys now include
  `fit_signature=joint_qz_labeled_marker_fit_specular_theta_i0_l8_v8`; old
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
- PbI2 Qr-rod profile panels are displayed on log-scaled intensity axes and
  capped at `L=3`, without trimming the exported CSV/profile rows.
- PbI2 no-background debug mode is available through
  `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION=1`. It forces transverse sideband
  subtraction off for PbI2 rods, records the mode in cache signatures, and plots
  raw `background_density` against full `joint_fit_density`.
- Headless PbI2 debug runs now use the default `auto` Qr-rod marker edit mode,
  which still opens the popup on interactive Matplotlib backends but skips it
  when `RA_SIM_HEADLESS` or CI mode is active.
- PbI2 generated rods remain disabled by default and final profile rows still
  require complete detector support, so the unsupported `m=7` row is skipped.

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
- Existing unrelated non-parallel notebook test failures remain out of scope.

Feature status:

- `hk0_l3_star.png` is implemented in the diagnostic `.py` as a raw detector
  crop from the beam center through and above the `HK=0`, `L=3` / `00L`
  marker, rendered with detector-style color and log intensity normalization.
- The detector selected-region figure uses dedicated high-contrast styling for
  the central `HK=0` / `00L` rod. This is a display-only change; Qr/Qz maps,
  Delta-Qr values, selected masks, integration, fitting, and cache identities
  are unchanged.
- The detector selected-region label editor now supports in-figure picking and
  dragging for Qr-region labels. The helper signature and
  `ra_sim.detector_label_settings.v1` JSON payload are unchanged, and the
  temporary editor controls/artists are removed before the final figure is
  saved.
- The unified Qr-rod editor now owns the operator-facing Qr-region adjustment
  order for the parallel diagnostic: marker editing, Delta Qr, and L-window
  changes happen in the same Matplotlib popup before the final joint Qz fit.
  The older detector-label editor helper remains for schema compatibility, but
  the final detector selected-region save path consumes unified-editor label
  entries when present and otherwise draws the generated defaults without
  opening a second popup.
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
  The final Qz fit cache signature is `joint_qz_labeled_marker_fit_specular_theta_i0_l8_v8`
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
- PbI2 no-background debug mode is implemented for the diagnostic `.py` through
  `PBI2_DISABLE_BACKGROUND_SUBTRACTION_OVERRIDE` or
  `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION`. It is diagnostic-only and does
  not add a GUI control, CLI flag, saved-state field, or package API surface.
- Tail-component aggregation now fails closed when `x`, target, finite-mask,
  and model shapes are inconsistent instead of partially replacing only one
  mask and risking a mismatched array index.
- CI/deployment automation was not changed for this slice. The applicable gate
  remains the local diagnostics pytest/compile path before a manuscript run or
  PR; no workflow files, deployment jobs, secrets, or branch protections were
  changed.
- No deprecation path or user migration is required. The only compatibility
  action is the final-fit cache signature bump to v8, which forces stale cached
  Qr-rod final fits to be recomputed.
- Shipping status is local-only. The rollback path is to revert the diagnostic
  script/test/doc commit, delete regenerated local artifacts if needed, and
  allow older cache signatures to be regenerated by the previous fit logic.
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
- Targeted detector label-editor coverage verifies Matplotlib in-figure
  controls, label selection through canvas events, data-space dragging, event
  loop use without closing the final figure, cleanup of temporary editor
  artifacts, runtime-mode handling, settings round trip, and final-save wiring.
- Targeted Qr-rod marker edit coverage for per-rod marker replacement,
  marker-table cache-key hashing, headless/interactive mode resolution, JSON
  edit round trip including `marker_title`, final-label override behavior,
  in-popup import/export wiring, Delta Qr/L-window control state, live
  integrated-profile refresh callback wiring, accepted profile-table handoff,
  sample-name override wiring, `HK=0` specular marker inclusion, and editor call
  ordering before the joint-fit cache lookup.
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
  mappings and Qz-baseline cancellation are retained as diagnostics instead of
  overlay gates; non-PbI2 model selection is unchanged; and unsupported
  detector-incomplete rows such as `m=7` stay hidden.
- PbI2 no-background debug coverage verifies the environment/local override
  disables transverse sideband subtraction, changes both pre-editor and final
  Qr-rod cache signatures, and makes the plot policy compare raw
  `background_density` to full `joint_fit_density` without data-minus-baseline
  subtraction.
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
- Focused whitespace check passed:
  `git diff --check`
  The touched diagnostic/test files still contain older unrelated ruff-format
  churn outside this slice, so this patch leaves those hunks untouched instead
  of mixing a broad formatting cleanup into the PbI2 plotting fix.

Known validation limits:

- Full `tests/test_background_peak_fits_notebook.py` was rerun on 2026-05-11
  after the unified Qr-rod editor slice. It still has four unrelated
  notebook/script source-token expectation failures, with `130 passed`, `2
  skipped`, and `4 failed`. The failures are
  the older missing `ROD_PROFILE_MAX_TWO_THETA_DEG = 60.0` and
  `"fit_model": "rotated_gaussian_plane"` notebook tokens plus the
  specular-region and sample-name override script-token expectations.
- Current full project check is not green from this dirty worktree: focused
  Qr-rod tests pass, but broad format/check gates still report unrelated
  formatting drift outside this cleanup slice.
- The 2026-05-10 Bi2Se3 weak-marker fix was validated with focused tests using
  embedded real-profile values, not a full real-sample script run.
- The PbI2 headless diagnostic path has been rerun. The L3 star crop,
  cache reuse, and interactive Qr-rod marker editor still need a real
  Bi2Se3/Bi2Te3 script run after this slice.
- Visual acceptance still needs manual script-output review: colored detector
  background, HK labels near low-L rod bases, emphasized central `HK=0`
  Delta-Qr band, the `hk0_l3_star.png` crop fully containing the L=3
  intensity, the Qr-rod crop's log color scaling, the Qr-rod popup appearing
  on an interactive backend, edited peak titles appearing in the final Qr-rod
  figure, no misleading mixed-target rods, and reasonable `curve_distance_px`
  values.

## Follow-up

Run the script against the current sample state and inspect the detector,
`hk0_l3_star.png`, and Qr-integration figures before using them in a manuscript.
If a mixed target-Qr rejection removes data needed for publication, split that
source into separate plotted rod identities rather than drawing one centerline
through multiple target Qr values.
