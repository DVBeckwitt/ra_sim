# Changelog

## Unreleased (since `e11bec1` on 2026-02-13)

- **Release and versioning**
  - Set the active development version to `1.0.0.dev0`, documented the 1.0 release sequence, and surfaced the resolved package version in the simulation GUI title and Help tab.

- **Simulation performance**
  - Added a Numba compatibility layer so simulation modules can import with Python fallbacks when Numba itself is unavailable or fails during import.
  - Hardened Q-debug truncation plumbing so threaded weighted-event chunks use distinct truncation counters and legacy diffraction debug exposes bounded-buffer truncation stats without changing return tuples.
  - Fixed parallel weighted-event candidate reuse so the configured candidate-buffer memory cap applies to total worker-local buffers instead of one worker's buffer.
  - Bounded `save_flag=1` Q-debug buffers with a configurable per-peak cap and truncation diagnostics to avoid multi-GB upfront allocations.
  - Made best-sample provenance buffers lazy for image-only/no-cache forward and QR rod simulations while preserving explicit caller buffers and cache-building provenance.
  - Vectorized primary-cache hit-table rematerialization, filtered impossible off-detector rows before integer casts, and skipped optional last-intersection cache builds when retention is disabled.
  - Optimized diffraction simulation with low-discrepancy antithetic beam sampling, weighted beam clustering, nominal detector culling, local-arc `solve_q`, sparse bilinear accumulation, and a fast-mode optics lookup table.
  - Moved the normal weighted-event diffraction path back onto a compiled runtime in `ra_sim.simulation.diffraction`: `_process_peaks_parallel_impl(...)` now drives Numba pass-1/pass-2 candidate rescans, off-detector or bilinear-unsupported candidates are excluded before entering the event PDF, solve-`q` reuse is limited to pure `All_Q` geometry within one call, image deposits can aggregate duplicate ordinals without collapsing duplicate hit/cache rows, and weighted-event debug stats now expose solve/project/select counters plus pass-1/pass-2 mass totals.
  - Replaced the threaded weighted-event dense Q table with the serial packed Q-set layout, split threaded chunk wall time into `time_chunk_compute`, and added scalar-equivalent beam-phase/event-count diagnostics.
  - Restored the Python cache/stat compatibility surface around `process_peaks_parallel_safe(...)` in `ra_sim.simulation.diffraction`, including `_PHASE_SPACE_CACHE`, `_SOURCE_TEMPLATE_CACHE`, `_Q_VECTOR_CACHE`, and `get_last_process_peaks_safe_stats()`, so the source-template cache regression coverage passes again.
  - Scoped the default Numba cache directory by Python cache tag and moved diagnostic intersection analysis off the unstable compiled `solve_q`/intensity path to avoid stale-cache and full-suite compile failures.

- **Config and path migration**
  - Renamed background-path keys in config from `dark_image`/`osc_files` to `simulation_dark_osc_file`/`simulation_background_osc_files`.
  - Added geometry solver tuning block under `instrument.fit.geometry.solver`.
  - Added Windows YAML path fallback parsing and `get_path_first(...)` in `ra_sim/path_config.py`.
  - Canonicalized config loading in `ra_sim.config.loader` for file paths, directories, materials, and instrument config, and kept `ra_sim/path_config.py` as a reloadable compatibility shim on top of it.
  - Migrated packaged config call sites in the CLI, GUI runtime/controllers, hBN workflow, diffraction debug helpers, and shared calculations to import from `ra_sim.config`.
  - Moved `get_temp_dir()` into `ra_sim.config`, migrated the remaining non-compat config imports off `ra_sim.path_config`, and added direct loader/shim coverage for temp-dir behavior.
  - Removed the obsolete `ra_sim/path_config.py` shim and moved the remaining config compatibility coverage onto `tests/test_config_loader.py`.

- **hBN bundle geometry mapping and metadata**
  - Added canonical hBN bundle metadata keys and strict validation in `ra_sim/hbn.py`.
  - Added hBN-to-simulation geometry conversion helpers for detector angles and center mapping:
    - `convert_hbn_bundle_geometry_to_simulation`
    - `build_hbn_geometry_debug_trace`
    - `format_hbn_geometry_debug_trace`
  - Updated `load_tilt_hint` to return converted simulation-space tilt/center/distance hints.

- **Fitting and optimization**
  - Fixed locked manual Qr/Qz geometry fits whose reduced hit-table rows lost
    top-level full-reflection fields even though the canonical provider identity
    still carried them; the matcher now recovers unique `m,L` branch rows from
    nested source identity instead of rejecting with
    `prediction_branch_source_switched`.
  - Fixed default GUI/manual caked Qr/Qz geometry fits so exact-caked,
    fixed-source manual `m,L` rows automatically use the dynamic angular
    point-match objective and bounded point-only projection path instead of
    falling back to static detector-pixel matching with `matched=0`.
  - Blocked detector-only source-cache fallback for manual caked Qr/Qz fits:
    active caked intent now requires matched finite caked fit-space rows and
    exact projection/storage before dataset build, with an optimizer guard that
    rejects missing exact-caked handoff instead of running `central_point_match`.
  - Split manual Qr/Qz pick provenance from required fit-space: detector-origin
    manual picks can remain detector provenance while a requested caked
    objective now carries an explicit caked-required handoff flag and fails
    with `manual_caked_fit_space_missing` instead of producing
    `central_point_match`/`metric_unit=px`; the runtime worker now uses the
    same caked-required flag for caked payload handoff and mixed-provenance
    preflight, and CI checks trace logs for impossible caked/pixel states.
  - Fixed manual caked Qr/Qz geometry-fit handoff for detector-origin picks by
    projecting observed detector anchors into exact-caked fit space during
    dataset build and extending the trace checker to reject live text-log
    caked/pixel fallback signatures.
  - Fixed PbI2 Qr-rod marker editor edits so `HK=4 -` marker
    add/move/delete/snap interactions preserve the current panel x/y limits;
    explicit Delta-Qr and L-window controls remain the scope-changing path.
  - Fixed PbI2 HK=0 Qr-rod editor phase isolation so specular Delta-Qr/L
    changes rebuild only HK=0 previews and profiles, leave nonzero rods'
    L minima unchanged, and filter configured-hidden `m=7` rows before
    artifact tables and plots.
  - Added opt-in exact manual-pair exclusion and dry-run parameter-combo
    sweeps for headless Bi2Se3 geometry recovery fits, including per-combo
    JSON/PNG artifacts, top-level sweep reports, fail-closed combo results,
    and a guarded GUI apply helper for accepted, state-matching results.
  - Hardened headless parameter-combo sweeps so CLI runs isolate each combo in
    a child process and record nonzero child exits, including native Windows
    Python crashes, as rejected combo results with required artifacts instead
    of terminating the sweep parent, without keeping transient child
    request/result/log files in the artifact tree.
  - Added explicit Bi2Se3 sweep-result application through `fit-geometry`
    `--apply-sweep-result` plus repeated `--approve-excluded-pair-id` flags,
    with state-hash/exclusion/QR-contract guards, preserved manual pairs,
    recorded geometry-fit exclusions, actionable CLI success/failure messages,
    and `04_applied_geometry_overlay.{png,json}` artifacts for accepted
    `gamma,Gamma` results.
  - Hardened the sweep-result apply boundary so combo counters must be real
    integers, accepted overlay PNGs must come from the combo artifact directory
    and pass a PNG-header check before geometry is mutated, failed overlay or
    rebuild steps roll back GUI geometry variables, variable setter failures
    fail closed instead of silently returning a partial apply, and output-state
    hashes remain pending until a saved state file exists.
  - Expanded parameter-combo sweep reports with selected-combo QR counts,
    dry-run/update flags, artifact status, native child exit classification,
    durable bounded stdout/stderr tail files for child crashes, and top missing
    pair identities for `qr_fit_objective_incomplete` failures.
  - Wired Bi2Se3 headless `gamma,Gamma` recovery fits to write the single-step QR
    coordinate audit, full-fit before/after overlay, and rejected-fit worst-row
    PNG/JSON artifacts into the fit output folder, record their paths in the
    progress sidecar, and fail closed when any required PNG is missing.
  - Changed the dynamic caked QR objective sensitivity diagnostic to probe
    angular variables with a bounded `0.1` to `5` degree ladder and report the
    first meaningful step plus caked-prediction and residual-vector deltas,
    preventing `objective_param_insensitive` from being inferred from only a
    sub-0.1-degree finite difference while keeping cache reuse distinct from
    rebuilt source rows.
  - Hardened the New4 single-step QR coordinate audit so proof artifacts fail
    closed when any row has a failing or missing QR fit surface contract, and so
    the diagnostic visual/objective surface-divergence flag cannot mask
    non-surface failures.
  - Fixed caked click-pick QR trial source-row diagnostics so the public
    `missing_dynamic_trial_source_row_count` is recomputed after live caked
    candidate rows are merged, and scoped New4 single-step `proof_status=pass`
    as a caked-space contract proof when the detector panel is diagnostic only.
  - Simplified GUI manual Qr/Qz geometry fits so detector-origin picks stay on
    the fixed detector-pixel direct-LSQ path instead of inheriting seed
    multistart or being auto-promoted into the caked angular objective when
    `gamma,Gamma` are selected.
  - Made Qr/Qz group identity explicit as `m=H^2+H*K+K^2, L` while preserving
    the saved `("q_group", source, m, L)` key shape, updated selector/export
    labels to expose `m` and `L`, and fixed locked manual Qr/Qz source-row
    resolution so same-`m,L` HKL equivalents such as `(-1,0,10)` and
    `(1,0,10)` do not reject the fitted solution as
    `prediction_branch_source_switched`.
  - Changed GUI fit-background activation so multi-background geometry-fit
    sessions start with only the initial image selected; later backgrounds are
    added by saved enabled manual Qr/Qz pairs, and selected later backgrounds
    with no enabled pairs are skipped instead of blocking the fit. Manual-pair
    undo now uses the same reconciliation so undone later-background points do
    not leave stale active fit backgrounds behind.
  - Added a popup-mode detector Qr-region companion preview beside the Qr-rod peak marker editor in the parallel background peak-fit diagnostic script, explicitly showing the companion window, refreshing detector overlays live during Delta Qr/L-window edits, and deferring expensive Delta Qr profile reintegration until slider release or accept.
  - Fixed the parallel Qr-rod marker editor so `L Min` / `L Max` submissions reject malformed profile-refresh tables, keep redraw errors inside the editor callback, and no longer let the global Enter shortcut close the popup while an L-bound text box is being edited.
  - Fixed PbI2 Qr-rod profile/editor defaults so `HK=0` uses `L_min=1.5` and `theta_i=40°`, nonzero HK rods use `0.5 <= L <= 3.0` with linear y axes, the detector companion preview uses the same HK-specific L bounds, and configured-hidden rows such as `HK=7` stay out of editor/support/final plots.
  - Fixed the PbI2 `00L` detector/editor region fallback so the specular rod still appears when cached marker rows are missing, split Qr-rod marker editing into nonzero-HK and HK=0 phases before detector-label placement, and kept Qr-rod marker clicks from changing subplot min/max limits.
  - Fixed the HK=0 Qr-rod editor phase so it seeds from the pre-editor real intensity profile, keeps that last valid specular profile when an L-bound refresh returns only nonzero rows, and reports marker-only fallback explicitly when real HK=0 intensity rows are unavailable.
  - Fixed PbI2 nonzero Qr-rod profile plotting so collapsed marker-derived L mappings, such as the observed `m=4 -` branch, are rejected and the panel falls back to the lattice/Qz L axis instead of compressing the full profile into a narrow L window.
  - Restored the PbI2 final HK=0 integration row by filtering stale specular markers below the active `L_min=1.5` window, generating fallback specular marker rows only through the active HK=0 display maximum, seeding HK=0 markers from the same theta=40 detector Qz map used for integration, and falling back to the detector-space specular L-window strip when the theta=40 `Qr=0` band has no pixels.
  - Routed PbI2 background peak-fit manuscript figures to `figures/results_pbi2` by default while keeping other samples on `figures/results_ordered` and preserving `RA_SIM_ALL_BACKGROUND_FIGURE_OUT_DIR` as the explicit override.
  - Fixed the split Qr-rod marker editor persistence and cache contract so imported edits load before the nonzero/HK=0 phase split, intermediate phases no longer write partial edit JSON, the detector companion preview is available across both phases, and final fit cache keys include the active HK=0/specular L bounds.
  - Disabled saved background-image peak subtraction by default in the parallel background peak-fit diagnostic while keeping fitted peak models saved separately.
  - Removed the stale `all_background_peak_fits.ipynb` diagnostic artifact from the active path and migrated remaining joint-Qz regression coverage to the maintained parallel `.py` diagnostic.
  - Restored the detector selected-region label-position editor before final detector-region save in the parallel background peak-fit diagnostic script, replacing the freezing Matplotlib label-drag path with a Tk canvas popup that supports drag/nudge/text/font editing plus import/export through the existing detector-label settings JSON.
  - Fixed geometry-fit live-cache preflight so source-matched q-group rows can satisfy generated-disordered manual pairs without falling through to fresh simulation or caked-projector setup, and restored locked/stale QR prediction-branch source switching before `locked_qr_row_unavailable` is returned.
  - Fixed manual Qr/Qz caked replay and fit handoff so detector-origin saved rows redraw through detector projection, visual caked aliases stay separate from fit/cache aliases, required caked projector errors stay precise, and cold detector clicks fail fast instead of rebuilding picker caches on the UI path.
  - Fixed caked-to-detector Qr replay fallback rotation so source-backed caked replay uses `display_rotate_k`, not its inverse, when the bound detector display callback is unavailable.
  - Fixed import-safe geometry-fit overlay invalidation, detector fallback source labels, and the caked-select -> detector-rearm manual Qr/Qz click workflow so saved visual identity survives view changes without accepting caked projection rows as detector candidates.
  - Fixed detector-view manual Qr/Qz and HKL picker payloads to prefer explicit detector-picker cache rows over stale grouped candidates, so visible detector markers remain clickable after simulation/caked refreshes without changing geometric-fit handoff interfaces.
  - Fixed manual caked Qr/Qz geometry-fit diagnostics so observed detector-display fields no longer contain caked `(2theta, phi)` values, and made non-legacy dynamic caked fits fail closed with `dynamic_objective_not_sensitive_to_fit_variables` instead of reporting complete when the objective is insensitive to all fit variables.
  - Fixed GUI geometry-fit preflight so `All`-background runs can refine from selected backgrounds that have saved manual Qr/Qz pairs while logging empty selected backgrounds as skipped; all-empty selections and mixed detector/caked fit spaces still fail closed.
  - Fixed GUI saved-manual caked Qr geometry fits to default to the direct fixed-source solve used by headless runs instead of implicitly entering ladder multistart; explicit `ladder-multistart` seed policy remains available.
  - Fixed saved manual background origin replay so explicit detector-origin rows win over stale caked frame tokens, and new manual placements persist the matching origin/frame contract.
  - Fixed detector-origin manual background redraw in caked view so failed live detector-to-caked projection leaves the marker unresolved instead of falling back to stale saved caked fields.
  - Fixed detector-origin manual Qr/Qz geometry-fit handoff so saved origin/frame metadata survives dataset orientation and exact caked reprojection replaces stale saved caked aliases before the optimizer and QR handoff audit consume the pair.
  - Hardened manual Qr/Qz simulated caked fit/cache resolution so bare `caked_x/y` is used only for explicit simulated caked projection rows, not background/replay-shaped rows.
  - Fixed beam-center defaults to keep PONI-derived centers in native row/col order in GUI startup, headless `fit-geometry`, and headless `simulate` paths, and kept beam-center picking out of the center-dependent caked wrapper.
  - Fixed beam-center picking to use the clicked detector-display point exactly, avoiding local peak snapping before applying the GUI contract `row = display_row`, `col = detector_width - display_col`.
  - Untangled `Pick Beam Center` coordinate handling so the pick writes one canonical GUI Row/Col pair through the visible sliders and entries, projects the marker back into detector/caked views from that same pair, and keeps detector-center remap reads on the same values.
  - Added gated beam-center JSONL tracing behind `RA_SIM_TRACE_BEAM_CENTER=1`, including widget-chain, scheduled-update, marker, remap, and overwrite-guard records in `debug/beam_center_trace.jsonl`.
  - Fixed the default clockwise pick mapping to commit `row = display_row`, `col = detector_width - display_col` without a slider-only correction path.
  - Changed Refine tab panels to start collapsed by default, including geometry, beam, lattice, CIF, and ordered-structure sections.
  - Backfilled legacy manual Qr/Qz pairs that have detector/background pixels but no saved caked `(2theta, phi)` anchors before headless geometry-fit preparation, and carried the repaired `manual_pairs` into the returned saved-state snapshot.
  - Fixed generated disordered-phase Qr/Qz references so nonzero disordered stacking weights invalidate live picker caches, schedule disordered hit-table collection even when primary hit tables are reusable, and preserve `source_label="disordered_phase"` in pickable Qr/Qz groups.
  - Added live runtime trace diagnostics for generated disordered-phase Qr/Qz references, including source counts, skip reasons, collection counts, and published group/peak counts.
  - Added an explicit `Include generated disordered-phase Qr refs` GUI toggle, enabled by default and independent from the packaged 6H reference toggle.
  - Fixed imported PbI2 generated-disordered Qr/Qz sets so restored `disordered_phase` q-group rows rebuild manual-picker detector source rows from stored modified-CIF hit tables when live preview rows are empty or stale.
  - Fixed source-consistent generated-disordered manual Qr/Qz handoff through detector picker candidates, placement, saved geometry pairs, and geometry-fit preflight, including a job-local picker/Q-group source-row fallback and a deduped fresh rebuild `consumer` wrapper.
  - Fixed GUI geometry-fit saved-manual-pair handoff so selected-background jobs prefer saved refined simulated coordinates over stale live-preview, picker-cache, or legacy `sim_col/sim_row` rows, while allowing non-current worker backgrounds to consume signature-matched job-local live rows.
  - Added trial caked axes-only payload support so refined geometry probes can recompute dynamic Qr/Qz source rows without rasterizing a full caked image.
  - Added New4 ladder worker phase/partial-report telemetry, residual-evaluation timing, cache-rebuild counters, and timeout diagnostics; singleton solve rungs can skip the duplicate initial dry-run objective and use the first solver evaluation instead.
  - Added a mixed-update geometry-fitter cache regression suite covering unsafe mixed fast-path fallbacks, stale worker result handling, deferred q-group refresh, projection handoff validity, and objective-cache reject reasons.
  - Fixed GUI geometry-fit overlay redraws in caked point-only fits so green `fit sim` markers use the current fitted prediction point, including legacy progress rows whose caked prediction is stored under `fit_prediction_detector_display_px`, without treating detector-space display pixels as caked angles.
  - Added GUI geometry-fit coordinate-lineage diagnostics that join cached handoff points, fit-dataset predictions, residual-eval snapshots, final point matches, and drawn green-marker probes by the same Qr/Qz source identity.
  - Fixed caked point-only Qr/Qz geometry-fit objectives so locked simulated points resolve from current trial hit tables, then project detector coordinates through the exact caked projector instead of reusing stale visual caked aliases.
  - Fixed caked point-only Qr/Qz geometry-fit objective resolution so hit-table misses and stale provider-local hit rows fall back to current trial source rows before the `qr_fit_objective_incomplete` guard fires; missing-pair diagnostics now report hit-table, source-row, and objective-cache lineage.
  - Fixed manual caked Qr/Qz geometry-fit handoff so current simulated caked rows can drive the dynamic objective without detector reprojection, manual observed caked rows are not reused as simulated sources, closer saved refined predictions are retained when live completion would worsen residuals, and acceptable insensitive objectives become no-op successes instead of drifting geometry parameters.
  - Added a dry-run New4 single-step QR coordinate visual audit that evaluates one bounded `gamma,Gamma` trial step from live dynamic projections, writes JSON/CSV plus one two-panel PNG, marks mixed detector display/native rows invalid instead of plotting them, and records the dynamic objective sensitivity proof used by caked-fit fail-closed checks.
  - Tightened the New4 single-step QR visual audit so proof mode fails when GUI visual simulation QR points and optimizer objective QR points use different caked coordinate surfaces, while keeping divergent GUI points as explicit diagnostic markers.
  - Fixed caked-display QR objective source resolution so live `sim_visual_caked_deg` rows are not reprojected through point-only detector coordinates; the New4 single-step proof artifact now records source-authority matches and passes when visual and objective caked surfaces are identical.
  - Warmed detector-native and detector-display coordinates immediately for caked manual Qr/Qz picks, so saved caked selections no longer require a detector-view toggle before fitting or replay.
  - Added a repeatable geometry-fitter cache regression gate script with local and strict modes, fast cache/handoff/objective coverage, workflow-slice validation, slow-geometry strict coverage, and optional New4 artifact handling.
  - Added a fast end-to-end QR selector to geometry-fitter handoff scenario covering fast-path invalidation sequencing, point-provider parity, projection-cache invalidation, and objective-cache reuse/reject behavior without requiring New4 artifacts.
  - Hardened geometry-objective cache signatures so center-only reuse is gated by unchanged physics, dataset, point-provider, QR branch identity, source-row identity, manual selection, refined peak, objective mode, and active fit-parameter signatures.
  - Avoided dense detector-image allocation in hit-table-only geometry fitting paths by using an empty simulation buffer, disabling image accumulation, and keeping locked-Qr detector-shape fallback behavior intact.
  - Removed the global diffuse/background-subtraction workflow from detector, caked, matching, geometry-fit, manual-pick, and headless fit inputs while keeping legacy CLI flags accepted as no-op compatibility options.
  - Expanded `hbn_fitter/fitter.py` with uncertainty-aware ellipse refinement and point-sigma handling.
  - Added projective tilt optimization path with fallback to legacy optimization.
  - Extracted hBN fitter bundle-payload assembly into a shared helper used by `save_bundle()`.
  - Extended bundle save/load fields to include confidence, ring sigma, optimizer metadata, coordinate-frame metadata, and compatibility keys.
  - Updated `ra_sim/fitting/optimization.py` with robust solver config, restart support, one-to-one point matching, weighted residuals, and missing-pair penalties.
  - Kept New4 ladder lean solve rungs fast by disabling identifiability diagnostics by default, preserving the explicit `identifiability_features` diagnostic run, and throttling running heartbeat writes while keeping full residual traces in final reports.
  - Fixed the `(-1,0,10)` Qr/Qz geometric-fitter objective handoff so provider-local fixed-source rows resolve without HKL fallback, selected caked residuals enter the optimizer vector, saved detector-point shortcuts cannot bypass ambiguous row rejection, and baseline fit-space offsets are primed from explicit baseline params instead of the first objective evaluation.
  - Fixed New4 Mode A Qr geometric-fit prediction so all 14 saved first-image branches resolve by locked dynamic identity, trial params rebuild detector-space source rows, simulated caked peaks are refined before residual calculation, and partial 8/28 residual objectives fail closed instead of silently using stale visual/caked coordinates.
  - Fixed a manual Qr diagnostic/fit-contract bug where saved caked `(2theta, phi)` targets could move under exact reprojection; cached manual targets now remain fixed while trial simulated sources are recomputed dynamically.
  - Fixed async GUI geometry-fit worker manual-pair dataset building so worker jobs no longer carry GUI/Tk refresh or display callbacks, use job-local detector/caked coordinate fallbacks, and reuse cached projected caked rows without stale worker markers forcing a second projection.
  - Finalized the saved-manual-caked Qr headless fit contract: manual caked Qr targets stay fixed in cached `2θ/φ`, simulated sources use dynamic `sim_visual_caked_deg`, saved-manual-caked Qr fits use the bounded point-only solve policy, and default headless now infers the validated saved-manual-caked policy without enabling `c`.
  - Fixed GUI saved-manual caked Qr fits that only vary detector tilts `gamma` and `Gamma`: trial simulated rows now reuse the prebuilt source cache through the same exact caked projector used by import/redraw, avoiding the removed parallel detector-tilt projection shim; the final status reports visual overlay distance against the saved background picks and distinguishes dynamic-fit RMS from full-beam overlay RMS. Validated on the Bi2Se3 and Bi2Te3 saved gamma/Gamma GUI states.
  - Fixed GUI geometry-fit caked `fit sim` overlays to prefer the same rendered fitted caked simulation points as the final image, report marker-vs-render deltas, and log detector/caked holistic image residuals after the fitted redraw.
  - Added GUI geometry-fit visual-probe diagnostics that compare the actual drawn green `fit sim` marker artist coordinates against the visible simulation image peak, so stale overlay markers can be detected even when cached point distances or holistic residuals look plausible.
  - Added `--active-vars` forwarding to the geometry-fit quality baseline runner so the Bi2Se3/Bi2Te3 headless `gamma,Gamma` direct-fit residual-improvement gate is repeatable from the existing debug script.
  - Limited `all_background_peak_fits.ipynb` Qr-rod Qz profiles and caked overlays to caked support at or below `60°` 2theta.
  - Changed PbI2 Qr-rod profile plotting to show raw central rod data, add the Qr sideband background back to available nonzero-rod fits, report marker/L and Qz-baseline cancellation checks as diagnostics instead of overlay gates, use a logarithmic intensity axis only for `HK=0`, and cap the displayed L range at 3.
  - Fixed PbI2 Qr-rod profile L-axis display so missing, conflicting, or non-monotonic marker-derived branch mappings fall back to the active lattice `c` spacing instead of plotting raw Qz or a bad linear marker fit.
  - Added `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION` for PbI2 diagnostic runs that need Qr-rod sideband subtraction disabled while plotting raw data against the full fit.
  - Fixed headless parallel background peak-fit diagnostics so the default Qr-rod marker edit mode is `auto`, preserving the popup on interactive Matplotlib backends while respecting `RA_SIM_HEADLESS` and CI skip behavior.
  - Fixed diagnostic Qz rod profiles to plot acceptance-normalized intensity density instead of raw integrated sums, removing false high-2θ support ramps.
  - Replaced `all_background_peak_fits.ipynb` pseudo-Voigt peak fits with rotated 2D Gaussian-plus-plane fits, then fit each Qr-rod Qz profile jointly as a simultaneous sum of all projected branch-point Gaussian peaks to avoid overlap overestimation between close peaks.
  - Added parameter-cell state selection and a batch runner for `all_background_peak_fits.ipynb`, with per-GUI-state output directories by default.
  - Fixed the parallel background Qr-rod final fit to keep supported weak Bi2Se3 low-L specular markers even when the preliminary shared baseline runs above the shoulder, while preserving the matching Bi2Te3 weak marker, rejecting unsupported nearby markers, and failing closed on inconsistent tail-component aggregation shapes.
  - Improved the parallel background Qr-rod final joint Qz fit by adding a log-scale residual term to the nonlinear Pearson-VII refinement, reducing the Bi2Se3 m=0 full-profile mismatch while preserving the Bi2Te3 weak specular peak and invalidating older final-fit caches.
  - Added `hk0_l3_star.png` to the parallel background peak-fit diagnostic script as a colored, log-scaled raw detector crop from the beam center through the `HK=0`, `L=3` / `00L` marker.
  - Improved the parallel diagnostic detector selected-region figure so the central `HK=0` / `00L` rod uses high-contrast styling and a more prominent Delta-Qr band.
  - Adjusted the detector selected-region highlight so the central `HK=0` / `00L` rod keeps the same-width centerline while the visible Delta-Qr region becomes easier to inspect.
  - Changed the detector selected-region label editor to pick and drag Qr-region labels on a responsive Tk canvas snapshot, while keeping the existing detector-label JSON settings schema.
  - Added `SAMPLE_NAME_OVERRIDE` / `RA_SIM_ALL_BACKGROUND_SAMPLE_NAME` to the parallel `.py` diagnostic so direct runs can replace only the sample label and filename stem, such as `Bi2Se3` to `Bi2Te3`, without changing the run directory.
  - Added a filename-keyed pre-editor cache to the parallel `.py` diagnostic so repeated runs with the same state/background filenames can reuse completed global peak fits, line-profile fits, and Qr-rod profile construction before the manual marker editor opens.
  - Restored default-on Qr-rod peak marker editing in the generated `.py` diagnostic with `RA_SIM_QR_ROD_PEAK_EDIT_MODE=popup|skip|auto`, JSON round trip through `RA_SIM_QR_ROD_PEAK_EDITS`, and marker-table cache-key invalidation before final joint Qz fitting.
  - Added Qr-rod marker editor `Import` and `Export` buttons for the same JSON marker-table format used by `RA_SIM_QR_ROD_PEAK_EDITS`.
  - Fixed the Qr-rod peak marker editor so dynamically projected `HK=0` / `00L` specular markers are included before final-fit cache lookup and fitting.
  - Changed the Qr-rod peak marker editor Snap action to snap all markers in the selected rod panel to nearby local profile peaks.
  - Changed the Qr-rod peak marker editor plots to use fitted integer `L` x-axes while still saving marker positions as Qz.
  - Changed the parallel Qr-rod editor so Delta Qr, `L Min`, and `L Max` controls live in the same marker-editor popup; Delta Qr drag updates the detector preview immediately and refreshes integrated profile intensities on slider release or accept, accepted values feed the final joint-fit table and cache key, and final detector-label placement remains a separate label-position popup before save.
  - Added editable per-peak Qr-rod marker titles so the popup `Label` field controls the final Qr-rod figure text, with blank titles falling back to `L=<rounded display_l>`.
  - Rounded generated Qr-rod peak fallback L labels to integer values while preserving user-edited marker titles.
  - Moved final Qr-rod peak labels to the upper-right of each marked peak with leader arrows pointing back to the peak.
  - Fixed final Qr-rod joint fitting so weak labeled peaks such as HK=0 `006` are preserved through nonlinear refinement, and invalidated older final-fit caches that could omit those components.
  - Fixed Qr-rod marker export propagation so manually added duplicate-HKL marker rows are preserved and edited `HK=0` marker Qz positions update the marker CSV, detector selected-region figure, and `hk0_l3_star.png`.
  - Changed direct Windows execution of the parallel `.py` diagnostic to relaunch through the existing guarded runner for the default process backend, so global peak fitting uses the process pool unless `BACKGROUND_FIT_BACKEND=thread` or `serial` is explicitly requested.
  - Fixed the final Qr-rod profile figure so rods with no drawable branch data, such as the empty Bi2Te3 `HK=7` row, are skipped instead of reserving blank figure rows.
  - Added `Intensity (a.u.)` y-axis labels to the left nonzero Qr-rod integration subplot axes, matching the existing HK=0 row label.
  - Fixed the parallel background peak-fit diagnostic script so Qr-rod marker labels are defined before profile annotation/redraw code can call them.
  - Fixed the parallel background peak-fit diagnostic runner so the generated `.py` diagnostic can run through the guarded Windows process backend, restoring process-pool CPU use while keeping direct top-level `.py` execution on the safe thread fallback.
  - Fixed manual Q-set simulated peak refinement propagation so refined detector/caked Qr rows rebuild lookup maps before redraw and fit handoff, and Q-set objective rows stay on the dynamic resolver instead of falling back to nominal direct projections.
  - Decoupled manual Qr/Qz click paths from picker-cache construction: clicks now consume warm caches only, background Qr references avoid simulated inventory, redraw skips simulated lookup for background-only references, and cache diagnostics expose build/refine/lookup timing.
  - Preserved warm manual Qr/Qz picker caches after source-entry click placement so the next group click can reuse the existing cache instead of cold-failing.
  - Fixed manual Qr/Qz caked picker cache projection so simulation source identity stays view-independent, caked sidecars are normalized and token-checked, saved caked redraw prefers source-matched current simulated projections over stale saved fields, and cold caked clicks reuse prewarmed sidecars instead of reporting cache-not-ready.
  - Fixed active manual Qr/Qz pick session refresh so detector/caked view toggles replace `group_entries` from the current view's source-matched rows while preserving only visual caked aliases by identity.
  - Fixed a detector-view wrong-frame cache leak where caked Qr/Qz projection rows with detector-looking fields could bypass picker-only recovery, keeping stale caked rows in detector selection/session refresh paths.
  - Fixed manual Qr/Qz detector picking to prefer visual detector display/native authority over refined/cache coordinates, treat wrong-frame grouped rows as empty for bounded detector recovery, and keep visual detector-to-caked aliases from overwriting fit/cache caked fields.
  - Split saved manual Qr/Qz caked pair fields so fit/cache caked coordinates and visual caked aliases stay separate, forced required caked geometry-fit projection through per-background projectors, narrowed view-toggle overlay invalidation to stale view-bound markers, and stopped swallowing unexpected cleanup errors on the caked-view toggle path.
  - Fixed Qr/Qz Q-set branch collapse so detector-distinct non-00l rows without explicit branch metadata keep separate branch representatives, hit-row phi branches survive source-row canonicalization, branch repair uses stable detector-native/refined coordinates before display aliases, and the Qr/Qz collapse wrapper honors explicit whole-group collapse.
  - Recreated the parallel background peak-fit diagnostic script, restored final Qr-rod pickle-envelope cache reuse by state filename with schema validation, separated `fit_l` from display-only `display_l` marker labels, drew used markers on Qr-rod profile figures, kept local peak snapping bounded to each marker window, fixed cache-hit specular support bounds, and normalized direct Windows script execution to threads to avoid `multiprocessing.spawn` re-running top-level diagnostic code.
  - Hardened New4 targeted source-row preflight diagnostics with drop-stage HKL/branch inventories, a separate source-coverage gate, Q-group HKL alias matching after canonical source-row construction, and deferred raw hit-table branch filtering. Provider-only parity remains green; full preflight is still red at 5/7 source pairs pending missing live rows for collapsed `00l` and q16 branch 1.
  - Added New4 refined-center diagnostics proving observed caked centers and simulated refined caked centers are recomputed under trial geometry, while classifying the current objective as bin-limited because simulated caked refinement is integer-bin argmax without subpixel peak refinement.
  - Hardened the geometry-fit integration workflow tests and diagnostics by keeping in-memory source-resolution diagnostics dict-shaped, preserving raw-derived simulated native coordinates during orientation setup, routing optional New4 artifact tests through the shared skip gate, and refreshing synthetic New4 dynamic-source coverage fixtures without changing the production dynamic-source gate.
  - Fixed cold-start caked manual geometry-fit preflight so exact caked projection metadata is cached separately from display intensity images, allowing zero-support density NaNs without discarding the geometric projector.
  - Fixed Analyze local linear background subtraction so peak fits project each corrected caked ROI locally, reject peak-wing outliers before fitting the 2D plane, and fall back to uncorrected 1D curves instead of fitting through peak-contaminated pixels.
  - Changed Analyze peak-fit overlays so applied linear background subtraction hides the raw/background-bearing 1D radial and azimuthal lines and displays only the corrected data curve plus fit overlays for that axis.
  - Fixed generated and noncurrent caked manual-fit projection payload handling so empty payloads can use generated fallback, invalid axes-only payloads fail closed, and caked row rebuilds never receive a missing exact projector.
  - Fixed live GUI caked geometry-fit source-row fallback so first-fit preflight logs unambiguous internal/UI background indices, validates live rows before targeted cache lookup, accepts current-hit fallback only from `stored_max_positions_local` provenance, keeps `last_intersection_cache` memory-only, rejects stale/forged row caches with explicit reasons, and reports late fresh-simulation status instead of appearing stuck.
  - Validated Bi2Se3 and Bi2Te3 saved manual-caked geometry states through the direct fixed-manual path: Bi2Se3 matched 82/82 fixed pairs and Bi2Te3 matched 84/84 fixed pairs, with zero missing pairs, zero branch mismatches, exact-caked fit-space projectors, and reduced direct residuals.
  - Fixed runtime exact-caked projection digests to hash DetectorCakeLUT matrix/count content and recompute stored `projection_content_token_v3` values instead of trusting stale incoming tokens.
  - Hardened exact-caked projection-token trust so token-only payloads are absent, stored projection bundles are private read-only copies, and manual caked warm-cache reuse accepts only runtime-sourced verified tokens.
  - Fixed Bi2Se3/Bi2Te3 saved manual caked geometry fitting so provider-backed fixed pairs rebind without HKL fallback, run direct fixed-pair least squares by default, match all fixed pairs, and reduce the saved-state residuals in the active quality baseline.
  - Fixed headless `fit-geometry` per-background caked projector compatibility so Bi2Se3 and Bi2Te3 saved manual-caked states can run constrained `gamma,Gamma` direct fits through the shared dataset builder.

- **GUI and UX updates**
  - Added a Setup `Pick Beam Center` control that uses the current detector/background image, zoomed preview, and GUI Row/Col mapping to update the beam-center sliders and entries.
  - Made live caked Qr/Qz visual-source ledger output opt-in and cached manual-pick simulated-candidate refinement by stable simulation/exact-projection signatures so repeated warm-cache caked clicks do not reprint trace rows or rerun full row refinement; display-image sanitization and copied-but-equivalent caked axes no longer churn caked pick caches, explicit projection signatures survive normalize/hydrate/digest handoff, lookup rebuild completion is tracked explicitly, failed lookup builders are retried, no-signature direct calls clear stale skip metadata, and the New4 exact-caked finalizer repairs stale polish fields only for clean selected summaries.
  - Fixed detector-view Selected-Qr rod ROI mode so it displays a detector-native Qr/Qz support mask, suppresses the legacy detector `2theta/phi` angular ROI while enabled, and sets Qz bounds from detector pixels during rod drags.
  - Fixed Selected-Qr rod ROI profiles so detector view uses detector-native masks only for overlay/drag, while plotted Qz profiles always integrate from caked `2theta/phi` data.
  - Added Selected-Qr rod multi-selection with a stable extended-selection rod list, union overlay/drag masks, and vertically stacked per-rod Qz subplots; saved GUI state now records both `selected_qr_rod_keys` and legacy `selected_qr_rod_key`.
  - Changed fresh Selected-Qr rod ROI phi defaults to `-90..90` without overwriting restored/custom phi windows.
  - Changed detector-view Selected-Qr rod profile defaults to raw accumulated intensity while preserving restored/custom rod intensity modes and caked-view density defaults.
  - Fixed Selected-Qr rod ROI profile caching so nested caked projection signatures no longer crash the GUI callback when enabling the ROI from restored Bi2Te3/Bi2S3-style states.
  - Fixed restored PbI2-style Qr/Qz geometry rows so they seed Selected-Qr rod ROI options even when live Bragg-Qr simulation rod inventory is empty.
  - Optimized Selected-Qr rod interactions with debounced selection refreshes, shared detector/caked inputs, bounded per-feature caches, reusable 1D plot axes/lines, and optional `RA_SIM_PROFILE_QR_ROD=1` timing hooks.
  - Replaced modifier-key Selected-Qr rod multi-selection with an extended-selection rod list; selected keys remain saved in displayed rod order.
  - Changed the Selected-Qr rod `delta_Qr` control and saved `analysis_range.delta_qr` to mean full rod width, with legacy half-width saved states migrated on load and low-level mask builders still receiving half-width.
  - Changed Selected-Qr rod integration controls to display L min/max bounds, plot per-rod integrated intensity against L when lattice `c` is available, and show the live fitted geometry values used for the caked mask/profile path while preserving saved Qz bounds.
  - Added a Match-tab `Place Background Qr Set` control for saving background-only Qr reference peaks with local peak-top refinement and `2theta,phi` labels instead of HKL values.
  - Fixed `Pick Beam Center` conversion to use the detector display extent instead of raw pixel-index inversion, with Row following clicked display row and Col mirrored across displayed detector width.
  - Fixed Q-space viewer geometry ownership so detector distance participates in simulation cache identity, Q-space conversion uses the geometry that produced the current image, Q-space-only display skips caking, and displayed Qr centers are finite and positive.
  - Fixed full GUI-state import so legacy manual placements with detector pixels but missing caked `2theta,phi` anchors rebuild those anchors from the exact caked projection before geometry figures or fits consume them.
  - Fixed GUI-state JSON import and timing startup restore so saved-state restore suppresses intermediate simulation updates, defers caked-view refresh, Qr/Qz selector repaint, and manual-pair source-row cache rebuilds during import, and schedules one post-restore update instead of hanging during partially restored state.
  - Warmed caked Qr/Qz projection cache data immediately after detector-mode Qr/Qz selector changes, so manual picking can use caked sim/background coordinates without first switching to caked view.
  - Added a GUI control and saved-state flag for hiding or showing the simulation overlay independently from the background image.
  - Added an Analyze `Caked image intensity` control so full caked images and standard radial/azimuthal integrations can switch between support-normalized density and raw accumulated caked-bin signal, saved as `analysis_range.caked_intensity_mode`.
  - Fixed the main GUI `phi x 2theta` caked density convention so simulation/background caking and caked 1D profiles use detector-count density without solid-angle normalization, matching notebook `sum_signal / pixel_support` behavior while preserving raw-sum mode and Q-space conversion.
  - Fixed the Analyze caked intensity toggle so switching density/raw modes repaints the main caked figure raster instead of reusing stale projected pixels.
  - Changed Analyze peak-fit results from compact summary lines to a monospaced table showing Gaussian FWHM, Lorentzian FWHM, Gaussian/Lorentzian mix percent, center, and RMSE for radial and azimuthal fits.
  - Added Analyze-only `Subtract linear background` peak fitting, enabled by default, using one local 2D plane per selected source box without mutating cached detector or caked images.
  - Cropped standard Analyze radial/azimuthal 1D integration curves to the selected integration box and added an Analyze Fit Axes `Log y-scale` toggle for those 1D plot y-axes.
  - Replaced Analyze radial and azimuthal Pseudo-Voigt peak fitting with a `Mosaic mix` profile that fits independent Gaussian-core and Lorentzian-tail widths with tail-aware residual weighting.
  - Kept the Analyze main caked figure intensity scale fixed during integration-region changes while radial and azimuthal 1D plots still rescale to the selected region.
  - Added an Analyze selected-Qr rod `Rod profile intensity` control with density-first default and raw accumulated intensity opt-in, saved as `analysis_range.rod_profile_intensity_mode`.
  - Hid the standard azimuthal integration subplot while selected-Qr rod ROI profiles are active, leaving the Qz rod profile as the only 1D plot for that mode.
  - Fixed GUI startup after selected-Qr rod picker wiring by threading `listed_q_group_keys_for_picker` through the manual-geometry cache callbacks.
  - Changed selected-Qr rod Qz controls to default to `0..5` and clamp slider bounds to the positive caked-Qz candidate range.
  - Changed the selected-Qr rod half-width default to `0.1 A^-1`.
  - Added an Analyze selected-Qr rod `Include rod shape` option, saved as `analysis_range.include_selected_qr_rod_shape`, so rod overlay/drag masks and per-rod caked profile masks can include the selected rod hit-cloud footprint outside the numeric Qr band.
  - Fixed the selected-Qr rod ROI toggle so it remains selectable in detector view; only Q-space view disables it.
  - Added an opt-in `Include 6H Qr refs` stacking control that loads the packaged PbI2 6H reference CIF when `w1` is nonzero, merges duplicate numeric Qr/Qz groups, and makes 6H-only groups available to manual Qr picking.
  - Reorganized Match-tab peak tools so `Drag Move Placed Peaks` stays visible beside the manual pick control, removed the auto-search radius slider from the peak tool row, and renamed the point-removal toggle to `Click Remove Placed Peaks`.
  - Made the Setup tab expose an expanded primary-CIF import control, with replacement CIF loads routed through full simulation, optics, picker, and analysis cache invalidation.
  - Added Match-tab `Add All Qr Set Peaks` and `Remove Qr Set Peaks` controls so enabled Qr/Qz selector groups can be auto-saved through the manual-pick refinement path or removed from the current background.
  - Added an auto-add-only Qr branch-pair length restraint so non-`00l` Qr sets require branch 0 and branch 1 to agree with the same-frame simulated branch length, while `00l` sets keep collapsed/single-branch behavior.
  - Skipped the origin `(0,0,0)` reflection during Qr/Qz auto-add so it is not auto-selected as a collapsed branch.
  - Added manual-geometry move support for already placed Qr/Qz background points: clicking a saved point arms a one-point replacement and the new placement is refined locally before saving; the Match tab also has a runtime-only `Drag Move Placed Peaks` option for click-drag-release replacement.
  - Fixed caked-view manual Qr/Qz picking so normal pick mode ignores saved-placement move hits unless the explicit drag-move tool is enabled, preventing an already selected caked Qr set from stealing later clicks.
  - Saved detector-origin manual Qr/Qz placements with projected caked `(2theta, phi)` coordinates and backfilled those cache fields when importing legacy GUI state files that do not contain them.
  - Reduced Qr/HKL image tags to one smaller, more transparent label per Qr set so placed and candidate peaks remain visible.
  - Parallelized the post-placement geometry refinement pass for auto-added Qr/Qz peaks with a bounded CPU worker pool.
  - Added selected-Qr rod `Mirror +/-phi band` integration so caked high-azimuth lobes can be selected as a symmetric `|phi|` band without filling the central phi rows.
  - Fixed selected-Qr rod caked integration masks so valid high-`|phi|` bins are selected from detector Qr/Qz pixels through the exact-cake LUT instead of depending on finite forward-projected Qr trace samples; selected-Qr drag Qz bounds now use the LUT transpose from dragged caked bins back to detector contributors.
  - Fixed selected-Qr rod 1D profiles to stay on the caked `2theta/phi` integration path; detector-native Qr/Qz/phi masks are reserved for detector overlay and drag support.
  - Added an optional selected-Qr rod shape mask so rod ROI masks and per-rod caked profiles can include detector-derived support from the selected Qr/Qz group shape in addition to the numeric Qr band.
  - Hardened GUI detector-center remap cache handling so exact remaps retain QR/Qz identity, invalidate stale manual/caked/q-space projection caches, report projection and handoff trace state, and fall back to full simulation for missing secondary exact caches or center-plus-physics changes.
  - Fixed manual geometry refresh so detector-coordinate truth is not replaced by stale caked fields and refreshed caked coordinates update raw caked fields.
  - Hardened GUI runtime prune reuse/fill QR selector cache handling so explicit QR/Qz masks persist, stale source-row snapshots are not retained across incompatible hit-table identity, and runtime traces report selector retention/deferred refresh/fitter handoff validity.
  - Added linked GUI sampling controls: beam phase samples now default to 75 on startup and legacy state/parameter loads, events per beam phase tracks two events per sample by default, and an `Independent` toggle enables separate event-count control when needed.
  - Made simulation GUI startup default to diagnostics-off, with saved debug settings and one-run debug mode kept as explicit opt-in choices.
  - Stopped creating per-run debug bundles on default diagnostics-off launches; simulation now starts bundle capture only after the chosen normal/saved/debug startup mode is known.
  - Resolved manual Qr/Qz and HKL picker alignment across detector and caked `(2theta, phi)` views. Qr/Qz group availability now comes from CIF/lattice simulation hit-table state, caked Qr/Qz positions are mapped from simulation-native detector pixels through the live caked simulation frame, and HKL picking now reuses the same current-view candidate payload as the Qr picker.
  - Routed caked manual background picks through the detector background oracle: caked clicks convert to `(2theta, phi)`, reverse through the existing detector LUT, refine in detector space, and save background truth as `background_two_theta_deg`/`background_phi_deg` while simulated Qr projection remains separate.
  - Fixed caked Qr picking to use detector-native simulated source rows as canonical truth. Caked Qr selection now hit-tests only the detector-to-caked projection cache, saved caked redraw resolves `source identity -> projection lookup -> sim_display`, alias-only simulated points fail visible as unresolved, and detector display transform state invalidates the caked Qr cache.
  - Fixed detector-view Qr picking so detector picker rows are validated before source selection stops, invalid caked/current-view rows fall through to detector-stable rows, detector mode no longer applies caked/current-view projection, and explicit `q_group_key` rows remain listed even when HKL aliases are missing.
  - Fixed the remaining detector-view Qr manual-picker cache population bug so matching empty detector caches are rebuilt instead of reused, `manual_pick_cache` source snapshots can rebuild for the current background when simulation artifacts exist, and detector-picker diagnostics report cache source, stale reason, source-snapshot status, rebuild attempts, rebuild row counts, and source-row counts by source.
  - Fixed a detector-view Qr manual-placement gap where caked-only preview rows could stop cache rebuild before detector picker candidates were materialized. Detector mode now falls through to detector-stable source rows, rebuilds snapshots that cannot project into detector rows, and reports caked-only versus missing detector-coordinate source diagnostics.
  - Fixed detector-view Qr picking after saving or clearing one set so background array churn can reuse detector-stable picker rows, stale existing-cache reprojection falls through to source rows again, and subsequent Qr sets remain selectable without switching to caked mode.
  - Fixed detector-view Qr picker cache poisoning after a placement-time empty rebuild by retaining detector-capable cached rows from grouped/simulated cache fields and surfacing source-snapshot diagnostics even when the final cache source is unavailable.
  - Fixed caked Qr manual picking so signed and unknown provenance rows for the same physical branch collapse to one placement target, preventing non-`00l` groups from asking for four background points.
  - Hardened Qr/Qz group cache signatures against recursive or exception-throwing mapping/sequence values so malformed row payloads cannot crash cache comparison.
  - Fixed GUI-state import for saved Qr/Qz selector rows that have `q_group_rows` but no `peak_records`, so imported Bi2Se3, Bi2Te3, and PbI2 states keep listed Qr sets available and can rebuild manual-picker detector source rows instead of reporting that the picker has no detector source rows.
  - Fixed caked manual preflight probing so caked display coordinates stay in caked frame and grouped candidates come from the caked projection cache or current-view caked projection.
  - Restored main detector and caked figure right-drag panning after stale left-drag suppression or missing press `xdata`/`ydata` could prevent pan startup; canvas button handling now normalizes Matplotlib/Tk integer, enum, and string button values, and right-drag release commits preview limits before ending the live interaction while preserving wheel zoom.
  - Fixed the GUI Geometric Fit source-cache handoff after source preflight. Projection-view signatures are normalized before `.get(...)` access so legacy list signatures no longer crash manual geometry rendering, preflight-built source/projected rows are persisted into source snapshots for dataset builds, empty `snapshot_hit` results are reported as exact snapshot/projection/filter statuses, and the New4 button-path regression now reaches dataset build with 15 source rows and Mode A resolving 14 paired branches / 28 components. Remaining known failure is downstream optimizer coverage (`qr_fit_objective_incomplete` 13/15 in `point_consistency_rungs`), not source snapshot availability.
  - Made GUI manual selected-point geometric fits use a fast interactive runtime by default: serial solver execution, `cfg["solver"]["max_nfev"]` capped at 30, and identifiability diagnostics disabled unless an explicit diagnostic path is requested.
  - Fixed the Qr/Qz peak sensitivity tool so perturbed finite-difference rows restore trusted reflection provenance only when stable identity fields match the baseline; `new4.json` now reports trusted `ok` sensitivity rows instead of all rows being `identity_changed`.
  - Extended the Qr/Qz peak sensitivity tool with ray-cloud and image-ROI center-of-mass shape metrics, including covariance/axis outputs, COM-vs-refined-max offsets, shape-specific long/matrix artifact exports, and ray-cloud COM as the default metric.
  - Restored the GUI selection-cache path for Qr/Qz and HKL picking so selection jobs build `stored_intersection_cache` from main-run hit tables with `collect_hit_tables=True` and `build_intersection_cache=True`, while raw-only and image-only updates keep detector cache builds disabled.
  - Fixed the startup/default detector-view regression where a blank detached projection could be cached until a parameter change or caked-view toggle forced a new detector signature.
  - Updated `main.py` and `ra_sim/gui/app.py` to use shared hBN geometry conversion helpers.
  - Routed the import-safe `ra_sim/gui/app.py` geometry-fit UI/value readers through the same shared value-callback bundle used by the main runtime.
  - Routed runtime geometry-fit undo/redo callback assembly through a shared history-callback builder in `ra_sim/gui/geometry_fit.py`.
  - Routed runtime geometry-fit constraint name/state normalization through shared helpers in `ra_sim/gui/geometry_fit.py`.
  - Routed runtime geometry-fit constraint domain/default calculations through shared helpers in `ra_sim/gui/geometry_fit.py`.
  - Shared the import-safe lazy wrapper behavior used by `ra_sim/gui/runtime.py` and `ra_sim/gui/app.py` through `ra_sim/gui/lazy_runtime.py`, including failure-safe cleanup when path-loading the heavy runtime implementation.
  - Routed the lazy background cache read/update workflow used by the main runtime through shared wrappers in `ra_sim/gui/background_manager.py`.
  - Routed initial background-cache boot and shared background-runtime normalization through `ra_sim/gui/background_manager.py`.
  - Routed background display defaults, transparency, and range/default-refresh helpers through `ra_sim/gui/background_manager.py`.
  - Corrected center-axis mappings used in detector angle-space/intersection geometry paths.
  - Improved sliders (`ra_sim/gui/sliders.py`) with entry sync, snapping, optional range expansion, and `min`/`max` typed values.
  - Added background file browser/status controls in `main.py`.
  - Removed the unused `ra_sim/gui/main_app.py` compatibility shim and standardized the package GUI entrypoint on `ra_sim.gui.app.main`.
  - Made `ra_sim/gui/runtime.py` import-safe by turning it into a lazy compatibility wrapper around the heavy GUI implementation in `ra_sim/gui/_runtime/runtime_impl.py`.
  - Extracted runtime background/bootstrap assembly into `ra_sim/gui/runtime_background.py`, leaving the internal runtime implementation with thinner background workflow wiring.
  - Extracted selected-peak / HKL lookup / manual-geometry / geometry-tool action runtime assembly into `ra_sim/gui/runtime_geometry_interaction.py`, replacing the matching internal-runtime AST checks with direct helper tests.
  - Extracted Bragg-Qr pruning/control wiring, integration-range update runtime assembly, and geometry-fit action runtime assembly into `ra_sim/gui/runtime_fit_analysis.py`, replacing the matching internal-runtime AST checks with direct helper tests.
  - Extracted geometry Q-group runtime assembly plus the cross-feature canvas interaction runtime assembly into `ra_sim/gui/runtime_geometry_preview.py`, replacing the matching internal-runtime AST checks with direct helper tests.
  - Extracted geometry-fit runtime value callback assembly, manual-dataset/config factory assembly, and geometry-fit action assembly into `ra_sim/gui/runtime_geometry_fit.py`, replacing the remaining geometry-fit runtime ordering check with direct helper tests.
  - Extracted the Qr-cylinder overlay runtime assembly into `ra_sim/gui/runtime_qr_cylinder_overlay.py`, moving the active-entry factory, render-config factory, and bound overlay runtime/toggle surface behind one import-safe helper seam.
  - Replaced the remaining GUI runtime AST checks around pruning-default wiring and selected-peak maintenance wiring with direct helper coverage in `tests/test_gui_runtime_fit_analysis.py` and `tests/test_gui_runtime_geometry_interaction.py`, and removed `tests/test_gui_runtime_bootstrap.py`.
  - Moved the primary-CIF / diffuse-HT control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the optional CIF-weight control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers and reused a controller helper for weighted intensity recompute.
  - Moved the fit-geometry parameter checklist out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the 1D integration-range control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the top-level GUI shell and bottom status panel out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers, including notebook state sync and compact console-backed status labels.
  - Wired `ra_sim/gui/runtime.py` through the shared `AppState` container for extracted GUI view state plus background/HKL interaction runtime state.
  - Extracted the remaining GUI runtime-owned state in `ra_sim/gui/runtime.py` into explicit `AppState` slices covering background selection/orientation, geometry interaction caches/artists, simulation/update/caking/peak caches, atom-site override cache bookkeeping, Bragg-Qr disabled state, hBN debug-report text, sampling count, and the caked-view override flag.
  - Reduced the remaining `runtime.py` module globals to structure-model / diffuse-HT rebuild state and the legacy `write_excel` flag.
  - Moved the structure-model / diffuse-HT bootstrap and rebuild workflow out of `ra_sim/gui/runtime.py` into `ra_sim/gui/structure_model.py`, including initial HT cache setup, weighted intensity recompute, and the live occupancy/stacking rebuild path.
  - Replaced the remaining inline primary-CIF / atom-site helper implementations in `ra_sim/gui/runtime.py` with thin delegates to `ra_sim/gui/structure_model.py` for occupancy metadata, atom-site override CIF generation, and CIF numeric parsing.
  - Expanded `ra_sim/gui/structure_model.py` to own the primary-CIF browse dialog plus the diffuse-HT open/export dialog workflow and status handling, leaving `ra_sim/gui/runtime.py` with thin delegate wrappers plus control rebuild callbacks.
  - Moved the Bragg-Qr / structure-factor pruning filter pipeline out of `ra_sim/gui/runtime.py` into `ra_sim/gui/controllers.py`, including Bragg-Qr source/L-key normalization, disabled-filter pruning, and filtered rod/HKL rebuild helpers.
  - Expanded `ra_sim/gui/structure_factor_pruning.py` with zero-arg runtime binding/callback factories plus normalized pruning / solve-q default and current-value helpers, leaving `ra_sim/gui/runtime.py` with bound pruning callbacks and a few live call sites.
  - Moved the remaining Bragg-Qr manager list-building workflow out of `ra_sim/gui/runtime.py` into `ra_sim/gui/controllers.py`, including group entry formatting, L-value mapping, and list-model/status construction for the manager window.
  - Expanded `ra_sim/gui/bragg_qr_manager.py` with shared runtime helpers for live lattice-value normalization, Bragg-Qr entry/L-value construction, active Qr-cylinder overlay entry derivation, the zero-arg overlay-entry factory, zero-arg runtime binding/refresh/open callbacks, and manager action wiring, leaving `ra_sim/gui/runtime.py` with one bound Bragg-Qr runtime factory value plus the remaining manager/overlay call sites.
  - Expanded `ra_sim/gui/qr_cylinder_overlay.py` with analytic Qr-cylinder overlay render config, the zero-arg render-config factory, cache-signature, path-construction, and runtime binding/refresh/toggle helpers, leaving `ra_sim/gui/runtime.py` with the remaining live overlay call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with a zero-arg runtime binding/callback bundle for selector open/refresh/toggle/include/exclude/save/load/update actions, leaving `ra_sim/gui/runtime.py` with one bound geometry-selector callback bundle plus the remaining call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with cached-entry snapshot replacement/capture helpers and the preview-exclusion open/status runtime helper, leaving `ra_sim/gui/runtime.py` with a narrowed update-cycle refresh call site plus geometry-tool toolbar wiring.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with shared propagated-hit filtering, reflection Qr/Qz group metadata, stable group-key reconstruction, and selector-entry snapshot assembly, leaving `ra_sim/gui/runtime.py` with thin fit-preview/cached-hit value sources plus the remaining selector call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with simulated-peak assembly from cached hit tables and geometry-fit hit-table exports, leaving `ra_sim/gui/runtime.py` with thin image-shape/display-coordinate value plumbing plus the remaining selector call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with geometry-fit hit-table simulation, max-position center aggregation, and preview-style simulated peak helpers, leaving `ra_sim/gui/runtime.py` with thinner geometry-fit parameter/lattice/display wiring around those call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with a runtime callback bundle for the geometry-fit hit-table and simulated-peak helpers, leaving `ra_sim/gui/runtime.py` with a bound geometry-fit simulation surface instead of local wrapper functions.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with a runtime value callback bundle for cached preview peaks, listed-Q-group state, export rows, selector status text, and Qr/Qz entry snapshots, leaving `ra_sim/gui/runtime.py` with bound geometry Q-group value surfaces instead of local wrapper functions.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with live-preview exclusion key/HKL/filter helpers plus runtime value callbacks for applying preview exclusions, leaving `ra_sim/gui/runtime.py` without local live-preview exclusion wrapper functions.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with shared geometry-fit seed filtering and degenerate-collapse helpers, leaving `ra_sim/gui/runtime.py` with preview-exclusion/live-preview orchestration plus thin geometry selector value plumbing.
  - Expanded `ra_sim/gui/background_manager.py` with a zero-arg runtime binding/callback bundle for background status refresh plus browse/load/switch actions, leaving `ra_sim/gui/runtime.py` with one bound background callback bundle plus the remaining call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with live-preview auto-match config normalization, cached overlay-state/status helpers, and preview-exclusion clear/toggle workflow helpers, leaving `ra_sim/gui/runtime.py` with thin preview delegate wrappers plus the remaining preview availability/fallback simulation call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with the live-preview enable/disable action workflow used by the geometry tool controls, leaving `ra_sim/gui/runtime.py` with a thin delegate around that toggle path.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with runtime callback-bundle entries for preview-exclusion window open, preview-exclude-mode toggling, preview-exclusion point toggling, exclusion clearing, and the live-preview checkbox action flow, leaving `ra_sim/gui/runtime.py` with bound preview-action callbacks instead of local wrapper functions.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with runtime live-preview enabled/render helpers plus callback-bundle entries for the bound redraw/status path, leaving `ra_sim/gui/runtime.py` without local live-preview enabled/render wrapper functions.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with the runtime live-preview simulated-peak acquisition/fallback helper, leaving `ra_sim/gui/runtime.py` without the local cached/fallback preview-peak resolution branch.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with the runtime live-preview seed filter/collapse helper and its empty-state exits, leaving `ra_sim/gui/runtime.py` without the local no-selected-Qr/Qz and no-seeds-after-collapse branches.
  - Fixed Qr/Qz preview and manual seed selection so raw cache rows, manual toggle, refresh/view-change, and place setup retain one mosaic-top simulated seed per normalized branch for each real Qr/Qz group while preserving branch/reflection/ray provenance.
  - Fixed caked-mode Qr/Qz manual picks so selected `2theta,phi` seeds map back to detector view through the same detector-display path as simulation markers, including stale-session refresh.
  - Fixed cross-view Qr/Qz manual picking so detector and caked clicks both resolve the same visible simulated marker after switching views, including stale caked-cache rows that still carry detector provenance.
  - Replaced the independent caked-native Qr simulated-picking path with detector-native source identity plus the existing detector-to-caked projection cache, so direct caked picks and detector-origin picks resolve the same simulated marker and alias-only saved caked sim points become unresolved instead of truth.
  - Hardened Qr/Qz selection after the lazy best-sample patch so detector-view peak records use detector-display projection, raw simulation cache rows stay simulation-native, and source-backed saved caked manual picks refresh detector/display coordinates from their saved `(2theta, phi)` values.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with the runtime live-preview match-result application helper, leaving `ra_sim/gui/runtime.py` without the local overlay-state build/store/render branch after auto-match completes.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with the runtime live-preview availability/background gate, leaving `ra_sim/gui/runtime.py` without the local disabled, caked-view, and hidden-background preview exits.
  - Expanded `ra_sim/gui/background_manager.py` with the background visibility toggle workflow used by the workspace action controls and wired it into the runtime callback bundle, leaving `ra_sim/gui/runtime.py` without a standalone background-toggle helper.
  - Expanded `ra_sim/gui/background_manager.py` with backend-orientation debug status plus rotate/flip/reset runtime helpers and callback wiring, leaving `ra_sim/gui/runtime.py` with thin background debug status-refresh and control-wiring call sites.

- **Repository cleanup**
  - Removed tracked root artifacts that did not belong in the long-term project layout: `ig_graph.sqlite`, `ig_graph.sqlite-shm`, `ig_graph.sqlite-wal`, `session.json`, `oneline`, `et --hard a485e65`, and the legacy root `hbn.py`.
  - Added root-level ignore rules so those local artifacts stay out of version control.
  - Expanded `ra_sim/gui/manual_geometry.py` with background-path matching, placement export-row/snapshot assembly, placement snapshot apply, and placement export/import dialog workflow helpers, leaving `ra_sim/gui/runtime.py` with thin delegates plus file-dialog dir and caked-view callback wiring.
  - Expanded `ra_sim/gui/geometry_fit.py` with runtime fit-history button-state, undo snapshot capture, runtime restore/redraw, and undo/redo transition helpers, leaving `ra_sim/gui/runtime.py` with thin delegates plus live state/controller/view callback wiring for that workflow.
  - Added `ra_sim/gui/integration_range_drag.py` plus explicit `IntegrationRangeDragState` in `ra_sim/gui/state.py` for raw/caked canvas drag-selection of 1D integration ranges, and expanded that module with the drag/region rectangle helpers plus current integration-region visual refresh used by the live GUI, leaving `ra_sim/gui/runtime.py` with manual-geometry canvas branches, top-level event dispatch, and thin live call sites.
  - Expanded `ra_sim/gui/peak_selection.py` to own the selected-peak HKL-pick toggle and raw-image click workflow, the Bragg/Ewald intersection analysis path, the ideal-center probe helper, the selected-peak config builders, and the runtime binding/callback bundle for HKL-pick labels, mode toggles, selected-peak refresh, HKL-control selection, and canvas-click selection, leaving `ra_sim/gui/runtime.py` with live GUI scalar getters plus the bound peak-selection wiring.
  - Expanded `ra_sim/gui/peak_selection.py` with runtime config-factory helpers for canvas-pick, Bragg/Ewald intersection, and ideal-center probe setup plus the shared selected-peak runtime config-factory bundle helper, leaving `ra_sim/gui/runtime.py` with thinner selected-peak value-source wiring around the bound runtime callbacks.
  - Expanded `ra_sim/gui/peak_selection.py` with the simulated peak-overlay cache builder and zero-arg runtime callback helper, leaving `ra_sim/gui/runtime.py` with a bound peak-overlay callback plus the remaining selected-peak call sites.
  - Added `ra_sim/gui/canvas_interactions.py` to own the top-level raw-image canvas event arbitration between manual-geometry placement, preview exclusion, HKL picking, and integration-range dragging, leaving `ra_sim/gui/runtime.py` with one bound canvas callback bundle plus thin event-hook wiring.
  - Moved the runtime binding/callback bootstrap assembly for structure-factor pruning, Bragg-Qr manager workflow, Qr-cylinder overlay, selected-peak workflow, integration-range dragging, canvas interactions, background workflow, and geometry Q-group workflow out of `ra_sim/gui/runtime.py` into `ra_sim/gui/bootstrap.py`, leaving `runtime.py` with local value-source/config builders plus the bound bundle variables used by the live call sites.
  - Moved the geometry-fit action binding/callback bootstrap out of `ra_sim/gui/runtime.py` into `ra_sim/gui/bootstrap.py`, leaving `runtime.py` with one bound geometry-fit action runtime bootstrap value plus the live fit-button config call site.
  - Expanded `ra_sim/gui/geometry_fit.py` with a shared manual-pair dataset bindings structure/factory plus a shared runtime-config factory for live geometry-fit preparation, leaving `ra_sim/gui/runtime.py` with one bound manual-dataset factory and one shared runtime-config factory instead of threading that geometry-fit prep wiring inline.
  - Preserved the live theta value when geometry-fit background selection is applied without per-background theta overrides.
  - Kept detector hit-table collection enabled when visible manual-geometry overlays need peak metadata for redraws.
  - Added primary CIF browse/apply workflow and dynamic occupancy control rebuild in `main.py`.
  - CIF parsing now handles numeric/scalar forms robustly and no longer multiplies `c` by 3.
  - Added a top-right red/green responsiveness indicator in the simulation GUI that turns red before blocking loads/fits/updates and green again once Tk is responsive.
  - Reworked the responsiveness indicator into a canvas-anchored block that stays positioned correctly across window and canvas resizes.

- **CLI updates**
  - Updated `ra_sim/cli.py` CIF parsing (raw `a,c` values; no forced `c*3`) and tilt-hint application using converted degree fields.
  - Added `fit-geometry-correlations`/`fit-geometry-correlation` for headless geometry-fit parameter correlation exports.

- **Tests**
  - Added `tests/test_cli_cif_parse.py` for CIF numeric parsing behavior.
  - Added `tests/test_gui_runtime_import_safe.py` and removed the import-smoke skip for `ra_sim.gui.runtime`.
  - Replaced the `ra_sim.gui.app` AST/source-extraction helper tests with direct behavioral imports.
  - Replaced the hBN fitter bundle-export AST/source-extraction regression with direct helper and save-path coverage.
  - Replaced the background portion of the internal runtime bootstrap AST checks with direct helper-module coverage.
  - Added `tests/test_gui_runtime_qr_cylinder_overlay.py` for direct coverage of the extracted Qr-cylinder overlay runtime assembly helper.
  - Extended `tests/test_gui_geometry_fit_workflow.py` with direct coverage for the extracted geometry-fit history snapshot/restore and undo/redo runtime helpers in `ra_sim.gui.geometry_fit`.
  - Extended `tests/test_manual_geometry_selection_helpers.py` with direct coverage for placement export-row/snapshot apply helpers and the export/import dialog workflow in `ra_sim.gui.manual_geometry`.
  - Added `tests/test_gui_structure_model.py` for the extracted structure-model helpers and rebuild workflow.
  - Extended `tests/test_gui_structure_model.py` with primary-CIF dialog workflow plus diffuse-HT open/export dialog workflow coverage in addition to the existing reload snapshot/restore and request-packaging tests.
  - Extended `tests/test_gui_controllers.py` with Bragg-Qr / structure-factor pruning filter pipeline coverage.
  - Extended `tests/test_gui_structure_factor_pruning.py` with direct coverage for the extracted structure-factor pruning / solve-q runtime-binding, default-normalization, current-value, factory, and callback workflow.
  - Extended `tests/test_gui_controllers.py` with Bragg-Qr manager entry/L-value/list-model coverage.
  - Extended `tests/test_gui_bragg_qr_manager.py` with direct coverage for the extracted Bragg-Qr runtime value helpers, overlay-entry derivation, zero-arg binding/callback factories, runtime-binding, action, and window-lifecycle helpers.
  - Added `tests/test_gui_qr_cylinder_overlay.py` for the extracted analytic Qr-cylinder overlay config, signature, detector/caked path-construction helpers, and the live runtime overlay refresh/toggle workflow.
  - Extended `tests/test_gui_qr_cylinder_overlay.py` with direct coverage for the extracted live render-config factory helper.
  - Extended `tests/test_gui_bragg_qr_manager.py` with direct coverage for the extracted active Qr-cylinder overlay entries factory helper.
  - Extended `tests/test_gui_bootstrap.py` with direct coverage for the extracted runtime callback/bootstrap helper surfaces.
  - Added `tests/test_gui_runtime_bootstrap.py` to guard import-time ordering of the pruning and HKL-pick bootstrap constants used by `ra_sim/gui/runtime.py`.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted geometry-fit Qr/Qz selector runtime binding/callback bundle, propagated-hit/group-metadata/entry-aggregation/simulated-peak helpers, and the snapshot replacement/capture and preview-exclusion runtime helpers in addition to the side-effect and dialog workflow helpers.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted geometry-fit hit-table simulation, max-position center aggregation, and preview-style simulated peak helpers.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted geometry-fit runtime simulation callback bundle.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted geometry Q-group runtime value callback bundle.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted live-preview exclusion key/HKL/filter helpers and their runtime value callbacks.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted live-preview auto-match config, cached overlay-state/status rendering, and preview-exclusion clear/toggle helpers.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted live-preview enable/disable action helper.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime preview-action helpers and their geometry Q-group callback-bundle wiring.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime live-preview enabled/render helpers and their callback-bundle wiring.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime live-preview simulated-peak acquisition/fallback helper.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime live-preview seed filter/collapse helper and its empty-state exits.
  - Added Qr/Qz branch-seed regression coverage for default per-branch collapse, explicit whole-group collapse, ungrouped rows, raw-cache preview before view change, manual toggle, refresh, place setup, and structural marker/display counts.
  - Added structural caked-to-detector regression coverage that verifies the same simulated `2theta,phi` seed returns to the same detector marker position and keeps branch/reflection/ray provenance.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime live-preview match-result application helper.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted runtime live-preview availability/background gate.
  - Extended `tests/test_gui_background_manager.py` with direct coverage for the extracted background visibility toggle helper and callback-bundle wiring.
  - Extended `tests/test_gui_background_manager.py` with direct coverage for the extracted background runtime binding/callback bundle plus the backend-orientation status and rotate/flip/reset helpers in addition to the background status refresh and post-load/post-switch workflow helpers.
  - Extended `tests/test_gui_peak_selection.py` with direct coverage for the extracted selected-peak runtime config-factory helpers in addition to the existing selected-peak workflow, ideal-center probe, and callback-bundle coverage.
  - Extended `tests/test_gui_peak_selection.py` with direct coverage for the shared selected-peak runtime config-factory bundle helper.
  - Extended `tests/test_gui_peak_selection.py` with direct coverage for the extracted simulated peak-overlay cache builder and runtime callback helper.
  - Added `tests/test_gui_integration_range_drag.py` for the extracted integration-range drag helpers, rectangle construction, integration-region visual refresh, runtime-binding factory, and callback bundle, and extended `tests/test_gui_controllers.py` to cover the new shared drag-state slice.
  - Extended `tests/test_gui_peak_selection.py` with direct coverage for the extracted HKL-pick toggle, raw-image click selection, config builders, ideal-center probe helper, runtime binding/callback bundle, and selected-peak Bragg/Ewald intersection workflow in addition to the existing HKL lookup and selected-peak state helpers.
  - Added `tests/test_gui_canvas_interactions.py` for the extracted cross-feature canvas click/drag runtime binding factory, callback bundle, and manual-pick/preview/HKL routing behavior.
  - Extended `tests/test_gui_geometry_fit_workflow.py` with direct coverage for the extracted geometry-fit manual-dataset bindings factory, runtime-config factory, and narrowed preparation/action binding bundles.
  - Added `tests/test_hbn_geometry_mapping.py` for geometry mapping math, metadata validation, sign handling, and startup/import consistency.
  - Added regression coverage for blank background-theta selections preserving the active live theta value.
  - Added regression coverage for manual-geometry overlay redraws requesting hit tables only when the overlay is visible.

- **Notes**
  - `ra_sim/simulation/diffraction.py` and `ra_sim/simulation/diffraction_debug.py` currently show line-ending-only modifications (no content diff).

## feat: reorganize package and add advanced diffraction/mosaic features

- **Folder Structure & Organization**  
  - Moved `main.py` to the top-level directory  
  - Created new file `ra_sim/gui/update.py`  
  - Added `ra_sim.egg-info/` folder for packaging metadata  
  - Extended or reorganized code in `ra_sim/fitting`, `ra_sim/gui`, `ra_sim/io`, `ra_sim/simulation`, and `ra_sim/utils`

- **File Updates**  
  - **`file_parsing.py`**: Introduced `Open_ASC` function for loading `.asc` files and rotating them  
  - **`mosaic_profiles.py`**: Enhanced pseudo-Voigt sampling logic in `generate_random_profiles`  
  - **`diffraction.py`**: 
    - `process_peaks_parallel` now returns `(image, max_positions)` tracking max intensities for each reflection sign  
    - Additional logic for mapping reflection maxima onto the detector image  
  - **`simulation.py`**: New `simulate_diffraction` workflow integrates mosaic profiles, beam profiles, and reflection calculations  
  - **`optimization.py`**: 
    - Introduced or extended Bayesian Optimization pipeline  
    - `run_optimization` sets slider states and updates a `progress_label` for improved GUI feedback  
    - `objective_function_bayesian` dynamically sets mosaic parameters before simulating diffraction  
  - **`gui/sliders.py`**: Updated `create_slider` to accept a callback for slider value changes  
  - **`main.py` (top-level)**: 
    - Tkinter-based GUI enhancements (sliders for mosaic/detector geometry, background toggling, colorbar, etc.)  
    - Incorporates `Open_ASC` to load `.asc` background images  
    - Detects nearest Bragg peak positions by referencing new `bragg_pixel_positions`  
    - Integrates Bayesian optimization

- **Notable Changes**  
  - Usage of new `simulate_diffraction` function to handle the full mosaic+beam profile plus reflection mapping  
  - `process_peaks_parallel` updated to store reflection peak intensities in `max_positions`  
  - Code now includes additional references (e.g. `detect_blobs`) and a `view_azimuthal_radial` helper in `tools.py`  
  - The package metadata in `setup.py` and `ra_sim.egg-info/` was revised to reflect the new structure  
  - `main.py` (top-level) orchestrates the new advanced GUI-based analysis with the refactored modules  
