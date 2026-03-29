# Refactor Migration Notes

See also: `docs/refactor_plan.md` for the active tracking plan covering what is
done, what is left, and the recommended order for finishing the refactor.

## Scope

This document summarizes the maintainability refactor delivered for RA-SIM while preserving user-facing behavior.

## Compatibility Guarantees

- `python main.py` still launches the GUI.
- `python -m ra_sim gui|simulate|hbn-fit` is unchanged.
- `ra-sim` console script now points to `ra_sim.cli:main`.
- `ra_sim.gui.app.main` is the supported package GUI entrypoint.
- Existing config access helpers remain:
  - `ra_sim.path_config.get_path`
  - `ra_sim.path_config.get_dir`
  - `ra_sim.path_config.get_instrument_config`
  - `ra_sim.path_config.get_temp_dir`
- Top-level utility script names remain:
  - `plot_excel_scatter.py`
  - `compare_intensity.py`

## New Components

- Typed simulation API:
  - `ra_sim.simulation.types`
  - `ra_sim.simulation.engine`
- Config package:
  - `ra_sim.config.models`
  - `ra_sim.config.loader`
  - `ra_sim.config.validation`
- GUI scaffolding modules:
  - `ra_sim.gui.state`
  - `ra_sim.gui.controllers`
  - `ra_sim.gui.views`
- Extracted GUI feature modules:
  - `ra_sim.gui.bootstrap`
  - `ra_sim.gui.background`
  - `ra_sim.gui.background_manager`
  - `ra_sim.gui.background_theta`
  - `ra_sim.gui.bragg_qr_manager`
  - `ra_sim.gui.canvas_interactions`
  - `ra_sim.gui.geometry_fit`
  - `ra_sim.gui.geometry_overlay`
  - `ra_sim.gui.geometry_q_group_manager`
  - `ra_sim.gui.manual_geometry`
  - `ra_sim.gui.overlays`
  - `ra_sim.gui.peak_selection`
  - `ra_sim.gui.structure_factor_pruning`
  - `ra_sim.gui.structure_model`
  - `ra_sim.gui.state_io`
- Utility package namespace:
  - `ra_sim.tools.plot_excel_scatter`
  - `ra_sim.tools.compare_intensity`

## Recent Progress

- Root launch paths are now packaged:
  - `main.py` is a compatibility wrapper.
  - `ra_sim.cli` launches the GUI through package code.
- `ra_sim.gui.app` is now an import-safe entrypoint that lazy-loads
  `ra_sim.gui.runtime`.
- `ra_sim.gui.runtime` is now also an import-safe compatibility wrapper that
  lazy-loads the heavy GUI implementation from
  `ra_sim/gui/_runtime/runtime_impl.py`.
- `tests/test_import_smoke.py` no longer skips `ra_sim.gui.runtime`.
- The `ra_sim.gui.app` helper and sim-signature tests now import the live
  module directly instead of using AST/source extraction.
- The hBN fitter bundle-export regression test now imports
  `hbn_fitter.fitter` directly, and the NPZ payload assembly used by
  `save_bundle()` now flows through one shared helper there.
- The background runtime/bootstrap assembly now also flows through a dedicated
  import-safe helper module, and the matching regression coverage now uses
  direct helper tests instead of AST assertions against the internal runtime
  implementation.
- The selected-peak / HKL lookup / manual-geometry / geometry-tool action
  runtime/bootstrap assembly now also flows through the import-safe helper
  module `ra_sim.gui.runtime_geometry_interaction`, and the matching
  regression coverage now uses direct helper tests instead of AST assertions
  against the internal runtime implementation.
- The Bragg-Qr pruning/control wiring, integration-range update runtime
  assembly, and geometry-fit action runtime assembly now also flow through
  the import-safe helper module `ra_sim.gui.runtime_fit_analysis`, and the
  matching regression coverage now uses direct helper tests instead of AST
  assertions against the internal runtime implementation.
- The geometry Q-group runtime assembly plus the cross-feature canvas
  interaction runtime assembly now also flow through the import-safe helper
  module `ra_sim.gui.runtime_geometry_preview`, and the matching regression
  coverage now uses direct helper tests instead of AST assertions against the
  internal runtime implementation.
- The geometry-fit runtime value callback assembly, manual-dataset/config
  factory assembly, and geometry-fit action runtime assembly now also flow
  through the import-safe helper module `ra_sim.gui.runtime_geometry_fit`,
  and the matching regression coverage now uses direct helper tests instead
  of AST assertions against the internal runtime implementation.
- The active Qr-cylinder overlay entry factory, overlay render-config
  factory, and bound overlay runtime/toggle assembly now also flow through
  the import-safe helper module
  `ra_sim.gui.runtime_qr_cylinder_overlay`, and the matching regression
  coverage now uses direct helper tests for that composition seam.
- The remaining GUI-runtime AST checks around pruning defaults and
  selected-peak maintenance wiring now also flow through import-safe helper
  seams in `ra_sim.gui.runtime_fit_analysis` and
  `ra_sim.gui.runtime_geometry_interaction`, and the matching regression
  coverage now uses direct helper tests instead of parsing
  `runtime_impl.py`.
- `ra_sim.config.loader` now also owns the canonical file-path, directory,
  materials, and instrument-config helper surface, while `ra_sim.path_config`
  delegates to it as a reloadable compatibility shim.
- Several packaged modules now import config helpers from `ra_sim.config`
  directly instead of going through `ra_sim.path_config`.
- `ra_sim.config.loader` now also owns `get_temp_dir()`, and the remaining
  non-compat imports were migrated off `ra_sim.path_config`.
- `ra_sim.gui.main_app` was removed now that the package, launcher, tests, and
  docs all converge on `ra_sim.gui.app.main` as the canonical GUI entrypoint.
- Manual geometry was split out of the runtime monolith in stages:
  - pure helpers, serialization, placement snapshot/apply helpers, and the
    placement export/import dialog workflow moved into
    `ra_sim.gui.manual_geometry`
  - session/preview/manual-pick orchestration moved into
    `ra_sim.gui.manual_geometry`
  - backing manual-geometry state and undo/session mutations now flow through
    `ra_sim.gui.state` and `ra_sim.gui.controllers`
- Geometry-fit history and Qr/Qz selector state have also started moving out of
  runtime-owned globals:
  - geometry-fit undo/redo history now uses shared GUI state
  - geometry-fit undo snapshot capture, runtime restore/redraw, and undo/redo
    transition helpers now also live in `ra_sim.gui.geometry_fit`
  - Qr/Qz selector cached-entry and refresh mutations now flow through
    `ra_sim.gui.controllers`
  - selector formatting/status/save-load workflow now also flows through
    `ra_sim.gui.geometry_q_group_manager`
- Live geometry preview and Qr/Qz selector view state have also started moving
  out of runtime-owned globals:
  - preview exclusion state, preview skip-once state, and auto-match
    background cache state now use shared GUI state
  - the Qr/Qz selector window lifecycle and row rendering now flow through
    `ra_sim.gui.views`
- Live geometry preview exclusion migration has continued:
  - excluded preview-pair keys, exclude-mode armed state, and cached preview
    overlay summary now use shared GUI state
  - preview exclusion toggles and overlay snapshot replacement now flow
    through `ra_sim.gui.controllers`
  - dead preview-selector scaffolding was removed from `ra_sim.gui.runtime`
- Live geometry preview / Qr/Qz workflow extraction has also advanced:
  - preview auto-match config normalization, cached overlay-state payload
    helpers, live-preview status rendering, and preview exclusion clear/toggle
    helpers now also live in `ra_sim.gui.geometry_q_group_manager`
  - the live-preview enable/disable action flow now also lives in
    `ra_sim.gui.geometry_q_group_manager`
  - the runtime callback bundle for preview-exclusion window open, preview-
    exclude-mode toggling, preview-exclusion point toggling, exclusion
    clearing, and the live-preview checkbox action now also lives in
    `ra_sim.gui.geometry_q_group_manager`
  - the runtime live-preview enabled/render helpers now also live in
    `ra_sim.gui.geometry_q_group_manager`
  - the runtime live-preview simulated-peak acquisition/fallback helper now
    also lives in `ra_sim.gui.geometry_q_group_manager`
  - the runtime live-preview seed filter/collapse helper and its empty-state
    exits now also live in `ra_sim.gui.geometry_q_group_manager`
  - the runtime live-preview match-result application helper now also lives
    in `ra_sim.gui.geometry_q_group_manager`
  - the runtime live-preview availability/background gate now also lives in
    `ra_sim.gui.geometry_q_group_manager`
  - `ra_sim.gui.runtime` now delegates those paths through bound callbacks and
    thin wrappers
- Selected-peak / HKL-pick bootstrap cleanup has also advanced:
  - the live selected-peak canvas-pick/intersection config factories,
    peak-overlay callback wiring, and selected-peak runtime callback bundle
    now assemble through one shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer wires that selected-peak setup inline
- A shared Tk after-token cancellation helper now lives in `ra_sim.gui.controllers`
  and is used by runtime glue paths (manual-geometry caked-view switch and
  geometry-update scheduling) to centralize duplicate timer cancellation and
  harden cancellation error handling.
- Integration-range drag scheduling now also uses that shared helper for pending
  token cleanup before queuing range-update callbacks.
- Geometry-fit undo restore in `ra_sim.gui.app` now uses that shared helper when
  clearing `update_pending` before redrawing restored geometry state.
- The app restore path is now covered by `tests/test_gui_app_helpers.py` to
  assert that `clear_tk_after_token` is invoked during undo restoration before
  rerunning the redraw flow.
- `app.py` now delegates undo restore to
  `gui_geometry_fit.restore_runtime_geometry_fit_undo_state`, bringing it in line
  with the runtime implementation’s shared restore orchestration.
- HKL lookup / Bragg-Qr open-control cleanup has also advanced:
  - the HKL lookup control cluster, its initial HKL-pick button refresh, and
    the shared Bragg-Qr manager open action now assemble through one shared
    helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer wires that control cluster directly or
    keeps a standalone Bragg-Qr open-control alias
- Integration-range drag bootstrap cleanup has also advanced:
  - the drag-selection rectangle, integration-region rectangle, live
    integration-region refresh callback, and runtime drag callback bundle now
    assemble through one shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer creates those drag/region rectangles
    inline or owns the thin integration-region refresh wrapper
- Bragg-Qr / structure-factor-pruning bootstrap cleanup has also advanced:
  - the live pruning callback surface, Bragg-Qr manager callback surface, and
    the refresh/apply-filters link between them now assemble through one
    shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer manually threads the live Bragg-Qr manager
    refresh callback into the pruning workflow or re-wires the manager
    apply-filters callback inline
  - the pruning control-cluster default normalization, control construction,
    and live solve-q / prune-bias trace hookup now also assemble through one
    shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.structure_factor_pruning` now also owns the helper that
    reapplies normalized pruning / solve-q defaults back to the live GUI vars
  - `ra_sim.gui.runtime` no longer builds that pruning control cluster or
    binds those trace callbacks inline
- Geometry-fit manual-pair action-binding cleanup has also advanced:
  - the shared prepare-bundle factory, execution-bundle builder, and top-level
    action-binding builder for the live manual-pair geometry fit now live in
    `ra_sim.gui.geometry_fit`
  - `ra_sim.gui.runtime` no longer nests that geometry-fit action binding
    assembly inline
  - `ra_sim.gui.geometry_fit` now also owns the live action-bindings factory
    and zero-arg action callback helper used by the runtime fit button
  - `ra_sim.gui.runtime` no longer rebuilds that geometry-fit action bindings
    bundle inline on each click
- Geometry-fit manual-pair preview/action cleanup has also advanced:
  - `ra_sim.gui.manual_geometry` now also owns the bound runtime callback
    bundle for current-pair rendering, Qr/Qz group toggle selection, manual
    point placement, preview refresh, and pick-session cancelation
  - `ra_sim.gui.runtime` no longer calls those manual-geometry helper surfaces
    directly inline; it now keeps one bound callback bundle that is threaded
    through canvas interaction, geometry-tool actions, background refresh, and
    GUI-state restore call sites
- Bragg Qr manager migration has also started:
  - Bragg-Qr selection/index bookkeeping now uses shared GUI state
  - Bragg-Qr selection mapping and group/L-value toggle mutations now flow
    through `ra_sim.gui.controllers`
  - the Bragg Qr manager window lifecycle and listbox rendering now flow
    through `ra_sim.gui.views`
  - Bragg-Qr source/L-key normalization and the filtered rod/HKL rebuild
    helpers now also flow through `ra_sim.gui.controllers`
  - the Bragg-Qr group-entry listing, L-value mapping, and manager list-model
    formatting now also flow through `ra_sim.gui.controllers`
  - the Bragg-Qr manager selection and list-refresh workflow now also flows
    through `ra_sim.gui.bragg_qr_manager`
- hBN geometry debug viewer migration has also landed:
  - shared widget references for the debug viewer now use shared GUI state
  - the hBN geometry debug window lifecycle and report-text rendering now flow
    through `ra_sim.gui.views`
- Geometry-fit constraints panel migration has also landed:
  - shared widget references and row-control state for the constraints panel
    now use shared GUI state
  - the scrollable constraints panel construction and mouse-wheel routing now
    flow through `ra_sim.gui.views`
- Background-theta / geometry-fit background control migration has also landed:
  - shared widget references and `StringVar` state for those controls now use
    shared GUI state
  - the background-theta panel construction and event bindings now flow
    through `ra_sim.gui.views`
  - the live runtime value readers plus sync/apply callback bundle for theta
    metadata and fit-background selection now also live in
    `ra_sim.gui.background_theta`
  - the runtime assembly of that background-theta workflow now also boots
    through one shared helper in `ra_sim.gui.bootstrap`
- Workspace panel / background-backend debug control migration has also landed:
  - shared widget references for the workspace action/background/session
    panels and the background backend/orientation debug controls now use
    shared GUI state
  - workspace panel construction, background-file status updates, reusable
    stacked-button rendering, and background backend/orientation debug control
    construction now flow through `ra_sim.gui.views`
- Manual-geometry / geometry-fit-history alias cleanup has also advanced:
  - `ra_sim.gui.runtime` no longer keeps direct module-level aliases to the
    manual-geometry pair/session/undo stores or the geometry-fit undo/redo
    stacks
  - those reads now go straight through the shared state containers and
    existing controller helpers
- Geometry-tool action control migration has also landed:
  - shared widget references and `StringVar` state for the fit-history,
    manual-placement, and preview-exclusion action controls now use shared GUI
    state
  - geometry-tools action-control construction and the related label/state
    update helpers now flow through `ra_sim.gui.views`
- HKL lookup control migration has also landed:
  - shared widget references and `StringVar` state for the HKL lookup /
    peak-selection panel now use shared GUI state
  - HKL lookup control construction, its entry bindings, and helper updates
    for the HKL values and image-pick button label now flow through
    `ra_sim.gui.views`
- Geometry overlay / analysis-view control migration has also landed:
  - shared widget references and `BooleanVar` state for the Qr-cylinder
    overlay, fit-mosaic, and 1D/caked/log analysis toggles now use shared GUI
    state
  - overlay/action and analysis-view control construction now flow through
    `ra_sim.gui.views`
- Analysis export control migration has also landed:
  - shared widget references for the analysis export buttons now use shared
    GUI state
  - snapshot/Q-space/grid export control construction now flows through
    `ra_sim.gui.views`
- Sampling-resolution / optics-mode control migration has also landed:
  - shared widget references and `StringVar` state for those controls now use
    shared GUI state
  - sampling/optics control construction, custom-sample widget-state updates,
    and summary-label updates now flow through `ra_sim.gui.views`
  - sampling-count parsing, resolution-choice normalization, and summary
    formatting now flow through `ra_sim.gui.controllers`
- Finite-stack control migration has also landed:
  - shared widget references and Tk var state for the finite-stack toggle,
    layer count, phi-L divisor, and phase-delta equation controls now use
    shared GUI state
  - finite-stack control construction, layer-widget enable/disable updates,
    scale-range growth, and entry-text updates now flow through
    `ra_sim.gui.views`
  - finite-stack layer-count and Hendricks-Teller input normalization /
    formatting now flow through `ra_sim.gui.controllers`
- Display-control panel migration has also landed:
  - shared widget references plus display-limit override/callback bookkeeping
    for the background/simulation intensity controls now use shared GUI state
  - display-control panel construction and scale-factor entry discovery now
    flow through `ra_sim.gui.views`
  - shared display-intensity range validation and display scale-factor
    normalization now flow through `ra_sim.gui.controllers`
- Structure-factor pruning / arc-integration control migration has also landed:
  - shared widget references and Tk var state for the prune-bias status/mode
    controls and solve-q inputs now use shared GUI state
  - pruning / arc-integration control construction plus the adaptive
    relative-tolerance enabled-state helper now flow through `ra_sim.gui.views`
  - prune-bias clipping and solve-q mode/step/tolerance normalization now flow
    through `ra_sim.gui.controllers`
- Beam/mosaic parameter slider migration has also landed:
  - shared widget references and Tk var state for the main geometry, detector,
    lattice, mosaic, Debye, beam-center, and bandwidth sliders now use shared
    GUI state
  - beam/mosaic slider construction, range-expanding slider configuration, and
    callback wiring now flow through `ra_sim.gui.views`
  - the remaining psi-z range guard now reuses a shared slider-bounds clamp
    helper from `ra_sim.gui.controllers`
- Stacking-probability / occupancy / atom-site control migration has also
  landed:
  - shared widget references and Tk var state for the stacking probability
    sliders, occupancy panels, and atom-site fractional-coordinate table now
    use shared GUI state
  - stacking/occupancy panel construction plus the dynamic occupancy and
    atom-site rebuild helpers now flow through `ra_sim.gui.views`
  - occupancy clamping and stacking weight normalization now flow through
    `ra_sim.gui.controllers`
- Primary-CIF / diffuse-HT control migration has also landed:
  - shared widget references and `StringVar` state for the primary CIF path
    entry and diffuse-HT action buttons now use shared GUI state
  - primary-CIF path / diffuse-HT control construction and entry/button
    bindings now flow through `ra_sim.gui.views`
- CIF-weight control migration has also landed:
  - shared widget references and Tk var state for the optional secondary-CIF
    weight sliders now use shared GUI state
  - CIF-weight slider construction now flows through `ra_sim.gui.views`
  - weighted-CIF intensity recompute now reuses a helper from
    `ra_sim.gui.controllers`
- Geometry-fit parameter checklist migration has also landed:
  - shared widget references and Tk var state for the fit-geometry parameter
    checklist now use shared GUI state
  - fit-geometry checklist construction and checkbutton layout now flow
    through `ra_sim.gui.views`
- 1D integration-range control migration has also landed:
  - shared widget references and Tk var state for the 1D integration-range
    sliders, labels, and entry widgets now use shared GUI state
  - integration-range control construction and entry bindings now flow through
    `ra_sim.gui.views`
- GUI shell / status-panel migration has also landed:
  - shared widget references for the top-level notebook shell, scrolled panel
    bodies, parameter columns, analysis containers, and status panel now use
    shared GUI state
  - top-level shell construction, notebook state synchronization, compact
    console-backed status labels, and status-panel construction now flow
    through `ra_sim.gui.views`
- Shared app-state ownership of extracted GUI/runtime state has also landed:
  - `ra_sim.gui.runtime` now initializes the extracted view/state containers
    through one shared `AppState` instead of instantiating them piecemeal as
    separate runtime globals
  - background-cache/orientation state and HKL image-pick interaction state
    now also use explicit runtime-state containers in `ra_sim.gui.state`
- Final GUI runtime-state extraction has also landed:
  - background file/current-orientation state, geometry interaction/cache
    state, simulation/update/caking/peak cache state, atom-site override cache
    state, Bragg-Qr disabled-toggle state, hBN debug-report text, sampling
    count state, and the caked-view override flag now use explicit
    `ra_sim.gui.state` containers
  - the remaining `global` lines in `ra_sim.gui.runtime` are now limited to
    structure-model / diffuse-HT rebuild state plus the legacy `write_excel`
    flag
- Structure-model / diffuse-HT workflow extraction has also landed:
  - CIF metadata parsing, atom-site override CIF generation, HT bootstrap /
    rebuild helpers, and weighted intensity recompute now live in
    `ra_sim.gui.structure_model`
  - `ra_sim.gui.runtime` now delegates the initial structure-model boot path
    and live occupancy / stacking rebuild flow through that module
  - the remaining primary-CIF occupancy / atom-site helper wrappers in
    `ra_sim.gui.runtime` now also defer to `ra_sim.gui.structure_model`
  - the primary-CIF reload state transition and diffuse-HT request packaging
    now also defer to `ra_sim.gui.structure_model`, leaving runtime with the
    Tk-facing control rebuild, file-dialog, and status-label work
  - the primary-CIF browse dialog plus the diffuse-HT open/export dialog
    workflow now also defer to `ra_sim.gui.structure_model`, leaving runtime
    with thin delegate wrappers plus control-var rebuild callbacks
- Bragg-Qr / structure-factor pruning workflow extraction has also advanced:
  - Bragg-Qr source/L-key normalization, disabled-filter pruning, filtered
    rod/HKL rebuild helpers, and status-text formatting now live in
    `ra_sim.gui.controllers`
  - the remaining structure-factor pruning / solve-q control callback
    workflow, pruning-status refresh, Bragg-Qr filter side effects, and
    adaptive-control sync now live in `ra_sim.gui.structure_factor_pruning`
  - `ra_sim.gui.structure_factor_pruning` now also owns the zero-arg runtime
    binding/callback factories used by pruning-status refresh, Bragg-Qr
    filter application, and solve-q control traces, plus the normalized
    pruning / solve-q default and current-value helpers used by the runtime
  - `ra_sim.gui.runtime` now keeps only bound structure-factor-pruning
    callback values plus a few live value-source and default-reset call sites
    around that workflow
  - the Bragg-Qr manager list-building workflow now also delegates through
    those helpers, leaving runtime with listbox selection reads and the
    enable/disable action callbacks
  - the Bragg-Qr manager selection/list-refresh flow now also delegates
    through `ra_sim.gui.bragg_qr_manager`, leaving runtime with the manager
    window lifecycle and enable/disable action callbacks
  - the remaining Bragg-Qr manager enable/disable/toggle action workflow now
    also delegates through `ra_sim.gui.bragg_qr_manager`, and the runtime-side
    callback wiring for that manager now also flows through one shared
    runtime-binding context in that module
  - `ra_sim.gui.bragg_qr_manager` now also owns the live lattice-value
    normalization, Bragg-Qr entry/L-value construction, active
    Qr-cylinder overlay entry derivation, and the zero-arg overlay-entry
    factory used by that workflow
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    refresh/open callback helpers used by the live filter pipeline and HKL
    lookup controls
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    binding-factory helper used by the live filter pipeline, HKL lookup
    controls, and manager callbacks
  - `ra_sim.gui.qr_cylinder_overlay` now also owns the analytic Qr-cylinder
    overlay render config, zero-arg render-config factory, cache-signature,
    path-construction helpers, and the runtime binding/refresh/toggle helpers
    used by the live detector/caked overlay workflow
  - `ra_sim.gui.runtime` now keeps only one bound Bragg-Qr runtime factory
    value plus the remaining live manager/overlay call sites for that
    workflow
- Geometry-fit Qr/Qz selector workflow extraction has also advanced:
  - selector line/status formatting, window refresh, checkbox/bulk include-
    exclude side effects, update-listed-peaks request flow, and save/load
    dialog workflow now live in
    `ra_sim.gui.geometry_q_group_manager`
  - shared propagated-hit filtering, reflection Qr/Qz group metadata,
    stable group-key reconstruction, and selector-entry snapshot assembly
    now also live in `ra_sim.gui.geometry_q_group_manager`
  - simulated-peak assembly from cached hit tables and geometry-fit hit-
    table exports now also lives in `ra_sim.gui.geometry_q_group_manager`
  - geometry-fit hit-table simulation, max-position center aggregation, and
    preview-style simulated-peak helpers used by the live preview, geometry
    fit, and manual-pick workflows now also live in
    `ra_sim.gui.geometry_q_group_manager`
  - the runtime callback bundle that binds those geometry-fit simulation
    helpers to live lattice/cached-peak/display sources now also lives in
    `ra_sim.gui.geometry_q_group_manager`
  - the runtime value callback bundle for cached preview peaks, listed-
    Q-group state, export rows, selector status text, and live Qr/Qz entry
    snapshot helpers now also lives in `ra_sim.gui.geometry_q_group_manager`
  - the live-preview exclusion key/HKL/filter helpers plus the runtime value
    callbacks that apply those exclusions to cached preview matches now also
    live in `ra_sim.gui.geometry_q_group_manager`
  - shared geometry-fit seed filtering and degenerate-collapse helpers used
    by the live preview, geometry fit, and manual-pick grouping workflows
    now also live in `ra_sim.gui.geometry_q_group_manager`
  - cached-entry snapshot replacement/capture plus the preview-exclusion
    open/status helper used by the live update cycle and geometry tool
    controls now also live in `ra_sim.gui.geometry_q_group_manager`
  - live-preview auto-match config normalization, cached overlay-state
    payload helpers, status rendering, and preview exclusion clear/toggle
    helpers now also live in `ra_sim.gui.geometry_q_group_manager`
  - the live-preview enable/disable action flow now also lives in
    `ra_sim.gui.geometry_q_group_manager`
  - `ra_sim.gui.geometry_q_group_manager` now also owns the zero-arg runtime
    binding/callback bundle used for selector open/refresh/toggle/include/
    exclude/save/load/update actions
  - `ra_sim.gui.runtime` now keeps only one bound geometry-selector factory/
    callback bundle plus thin fit-preview/cached-hit value sources, live-
    preview availability/fallback simulation orchestration, a narrowed
    update-cycle refresh call site, and thin toolbar wiring around that
    workflow
- Geometry-fit manual-pair fitting workflow extraction has also advanced:
  - `ra_sim.gui.geometry_fit` now owns manual-pair dataset assembly, fit
    preflight validation, fitted-value application, and matched-peak export
    row helpers used by the live geometry-fit action
  - `ra_sim.gui.geometry_fit` now also owns optimizer diagnostic formatting,
    RMS/result summary helpers, fitted-parameter merge helpers, overlay-
    diagnostic formatting, pixel-offset analysis/formatting, and final
    geometry-fit status-text assembly used after the live solver returns
  - `ra_sim.gui.geometry_fit` now also owns the pure post-solver analysis
    bundle that runs simulated/measured peak comparison, aggregates matched
    centers, filters overlay diagnostics, assembles overlay-state payloads,
    and packages the matched-peak export/save payload used by the live GUI
  - `ra_sim.gui.geometry_fit` now also owns the geometry-fit profile-cache
    merge helper used after one successful fit
  - `ra_sim.gui.geometry_fit` now also owns the runtime result-application
    helper that orchestrates optimizer-diagnostic logging, undo/profile-cache
    updates, preview-refresh scheduling, postprocess application, overlay-state
    persistence, export-save callbacks, and final success-status reporting
    through one callback bundle
  - `ra_sim.gui.geometry_fit` now also owns the prepared optimizer dataset-
    spec packaging, the geometry-fit start-log/console prelude assembly, and
    the runtime result-binding factory that snapshots current UI values into
    fitted-parameter and profile-cache updates
  - `ra_sim.gui.geometry_fit` now also owns the live solver-request
    packaging, log-file lifecycle, execution/failure reporting, and the
    runtime execution helper that runs one prepared geometry-fit through the
    solver and extracted postprocess/apply workflow
  - `ra_sim.gui.geometry_fit` now also owns the runtime preparation wrapper
    that packages the live dataset-builder/runtime-config callbacks around the
    extracted preflight helper, plus the runtime execution-setup builder that
    packages live state/callback sources into the extracted execution helper
  - `ra_sim.gui.geometry_fit` now also owns the shared live geometry-fit
    value-source bundle for selected fit variables, current parameter reads, UI
    snapshots, and the runtime var-map reused by undo/execute flows
  - `ra_sim.gui.geometry_fit` now also owns the top-level runtime action helper
    that drives one live geometry fit from current values through prepare,
    execution-setup packaging, solver execution, and the top-level preflight
    failure/status path
  - `ra_sim.gui.geometry_fit` now also owns the shared manual-pair dataset
    bindings structure/factory plus the live runtime-config factory used
    during geometry-fit preparation
  - `ra_sim.gui.geometry_fit` now also owns the live action-bindings factory
    and zero-arg action callback helper used by the runtime fit button
  - the geometry-fit action binding/callback bootstrap used by the live fit
    button now also flows through one shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` now keeps one bound geometry-fit action bootstrap
    value, one bound manual-dataset factory, one shared runtime-config
    factory, and the remaining Tk-side control wiring around the extracted
    helper
- Background-file workflow extraction has also advanced:
  - background-file state transition, file-dialog initial-dir selection,
    background status refresh, and the post-load/post-switch redraw/reset
    sequencing now live in `ra_sim.gui.background_manager`
  - backend-orientation debug status plus the rotate/flip/reset runtime
    helpers used by the live debug controls now also live in
    `ra_sim.gui.background_manager`
  - the background visibility toggle workflow used by the workspace action
    controls now also lives in `ra_sim.gui.background_manager`
  - `ra_sim.gui.background_manager` now also owns the zero-arg runtime
    binding/callback bundle for background status refresh, backend debug
    status, visibility/browse/load/switch actions, and backend rotate/flip/
    reset actions
  - the workspace background action/file controls plus the backend-debug
    control cluster now also boot through one shared helper in
    `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` now keeps only one bound background callback bundle
    plus two late-bound status-refresh helpers that are reused by the
    remaining geometry-fit/manual-geometry/background-theta call sites
- Geometry-fit manual-pair preview/action cleanup has also advanced:
  - the manual-geometry runtime cache/display callback bundle now also owns
    the manual-pick cache-signature assembly, cache reuse, and initial-pair
    overlay display assembly used by the live GUI
  - the manual-geometry runtime projection callback bundle now also owns the
    caked/raw view selection, manual-pair entry projection, simulated-peak
    projection, candidate grouping, and lookup assembly used by the live GUI
  - the manual-geometry runtime cache/display callback bundle now also boots
    through one shared helper in `ra_sim.gui.bootstrap`
  - the manual-geometry runtime projection callback bundle now also boots
    through one shared helper in `ra_sim.gui.bootstrap`
  - the manual-geometry runtime callback bundle now also boots through one
    shared helper in `ra_sim.gui.bootstrap`
  - the geometry-tool action callback bundle used by the manual-pair tools
    now also boots through one shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer constructs the manual-pair cache-signature,
    cache-reuse, and initial-pair overlay helpers directly inline
  - `ra_sim.gui.runtime` no longer constructs the manual-pair live
    view/projection helpers directly inline
  - `ra_sim.gui.runtime` no longer constructs those manual-geometry /
    geometry-tool callback bundles directly inline; it now keeps bound
    bootstrap results that are threaded through canvas interaction,
    geometry-tool actions, background refresh, and GUI-state restore call
    sites
- Integration-range drag workflow extraction has also advanced:
  - explicit live drag-selection state for 1D integration-range picking now
    lives in `ra_sim.gui.state`
  - canvas clamp helpers, detector-angle lookup, raw preview mask rendering,
    current integration-region visual refresh, drag/region rectangle
    construction, range-apply logic, and the runtime binding/callback bundle
    for raw/caked integration-range dragging now live in
    `ra_sim.gui.integration_range_drag`
  - `ra_sim.gui.runtime` now keeps only the manual-geometry-specific canvas
    branches, top-level event dispatch, and thin live call sites around that
    workflow
- Integration-range update/control workflow extraction has also advanced:
  - debounced redraw-only range-update scheduling, analysis-view toggle
    callbacks, and the runtime slider/entry callback wiring for the 1D
    integration-range controls now live in
    `ra_sim.gui.integration_range_drag`
  - the runtime assembly of that update/control workflow now also boots
    through one shared helper in `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` no longer wires the range-control cluster,
    `schedule_range_update`, or the analysis-view toggle handlers inline
- HKL lookup / selected-peak workflow extraction has also advanced:
  - HKL lookup parsing, selected-peak summary text, degenerate-HKL lookup,
    selection-by-index/HKL, clear-selection state transitions, HKL-pick
    toggle / raw-image click selection workflow, and selected-peak
    Bragg/Ewald intersection analysis now live in
    `ra_sim.gui.peak_selection`
  - the selected-peak config builders, the ideal-center probe helper used by
    raw-image HKL picking, the shared runtime config-factory bundle helper,
    the simulated peak-overlay cache builder/runtime callback, and the
    runtime config-factory helpers that resolve live GUI values now also live
    in `ra_sim.gui.peak_selection`
  - `ra_sim.gui.peak_selection` now also owns the runtime binding/callback
    bundle for HKL-pick button labels, mode toggles, selected-peak refresh,
    HKL-control selection, raw-image click selection, and the selected-peak
    Bragg/Ewald action
  - `ra_sim.gui.peak_selection` now also owns the remaining selected-peak
    update-cycle refresh and restored-target application callbacks used by the
    live runtime
  - the selected-peak bootstrap bundle now also carries that maintenance
    callback surface
  - `ra_sim.gui.runtime` now keeps only cross-feature click-mode dispatch,
    thin live value-source wiring, and the bound peak-selection runtime
    wiring around that workflow
- Canvas interaction workflow extraction has also advanced:
  - top-level raw-image canvas click/drag arbitration between
    manual-geometry placement, preview exclusion, HKL picking, and
    integration-range dragging now lives in `ra_sim.gui.canvas_interactions`
  - `ra_sim.gui.runtime` now keeps only one bound canvas callback bundle plus
    thin event-hook wiring around that workflow
- Runtime callback/bootstrap extraction has also advanced:
  - assembly of the runtime binding/callback bundles for structure-factor
    pruning, Bragg-Qr manager workflow, Qr-cylinder overlay, selected-peak
    workflow, geometry-fit action workflow, integration-range dragging,
    canvas interactions, background workflow, and the geometry Q-group
    workflow now lives in
    `ra_sim.gui.bootstrap`
  - `ra_sim.gui.runtime` now keeps the live value-source/config factories for
    those workflows plus the bound bundle variables consumed by the remaining
    call sites
- Direct tests were added for extracted controller/state behavior.
  - this now includes preview-state controller coverage, Bragg-Qr controller
  coverage, and direct Qr/Qz/workspace/Bragg/hBN/constraints/
  background-theta/background-debug/geometry-tool-action/HKL-lookup/
  overlay-action/analysis-view/analysis-export/sampling-optics/finite-stack/
  display-controls/pruning-controls/beam-mosaic-slider/stacking-parameter/
  primary-CIF/CIF-weight/fit-checklist/integration-range/app-shell/status
  helper coverage plus direct structure-model coverage for primary-CIF reload
  snapshot/restore, diffuse-HT request assembly, Bragg-Qr / SF-pruning filter
  application, Bragg-Qr manager list-model assembly, Bragg-Qr manager
  selection/refresh/action workflow, structure-factor pruning / solve-q
  workflow, geometry-fit Qr/Qz selector workflow, background-file workflow,
  HKL lookup / selected-peak workflow, selected-peak HKL-pick click
  workflow, and selected-peak Bragg/Ewald intersection workflow

## Remaining Migration Focus

- As of 2026-03-29, the refactor is leaving open-ended `runtime.py`
  decomposition mode.
- `ra_sim.gui.runtime` is now an import-safe public wrapper.
- `ra_sim.gui.runtime` now carries a small explicit lazy attribute contract
  (`main` and `write_excel` as stable local exports, dunder rejection, and lazy
  `__dir__`), with companion tests so startup safety stays enforced.
- The remaining large integration monolith now lives in
  `ra_sim/gui/_runtime/runtime_impl.py`, and most of the remaining inline code
  there is glue rather than high-value extractable feature logic.
- Going forward, that internal runtime implementation should be treated as an
  integration shell.
- New runtime refactors should be driven by concrete wins in import/startup
  safety, testability, duplicated-logic removal, bug-prone workflow
  simplification, or direct feature delivery.
- Thin value-source rewiring, one-off callback extraction, and line-count
  reduction are no longer goals by themselves.
- The public runtime import/startup boundary is now in much better shape.
- The remaining GUI-runtime shape tests are now gone.
- The config compatibility migration and GUI entrypoint cleanup are now
  effectively closed out.
- Repository-root cleanup has now landed: the tracked sqlite/session/log
  artifacts are gone, the legacy root `hbn.py` script has been removed, and
  `.gitignore` now keeps those local files out of version control.
- The small diffraction safe-wrapper cache/stat compatibility surface expected
  by `tests/test_source_template_cache.py` has also been restored in
  `ra_sim.simulation.diffraction`, so that separate simulation regression is no
  longer open.
- There is no new broad cleanup tranche queued ahead of feature-driven work; new
  changes must be justified by reliability, testability, or direct feature impact.
- Targeted runtime cleanup can still happen in structure-model, pruning,
  Bragg-Qr, background, selected-peak, geometry-fit, and integration-range
  workflows, but only when it materially supports those goals or unblocks
  active work.
- The newer `state` / `controllers` / `views` boundary should keep expanding
  where it simplifies shared workflows, but it does not need to absorb every
  thin adapter left in the internal runtime implementation.
- `ra_sim.config` is now the canonical config helper surface, while
  `ra_sim.path_config` remains only as a compatibility shim for older call
  sites.
- `ra_sim.gui.main_app` is gone; `ra_sim.gui.app.main` is the canonical
  package GUI entrypoint.

## Configuration Changes

- Added `RA_SIM_CONFIG_DIR` override to select an alternate configuration directory.
- Added `config/file_paths.example.yaml` as a portable template.
- Consolidated index-of-refraction parameter source to `config/ior_params.yaml`.

## Deprecation Timeline

### Current release

- Legacy positional simulation kernel calls remain available and are wrapped by typed APIs.
- Root utility scripts remain as wrappers around `ra_sim.tools`.

### Next release target

- Prefer typed simulation requests (`SimulationRequest`) internally and in new extension points.
- Keep wrappers but emit deprecation warnings for direct internal positional kernel usage.

### Future major release

- Remove legacy internal wrappers that bypass typed simulation models.
- Remove compatibility aliases that are no longer used by project tests or docs.
