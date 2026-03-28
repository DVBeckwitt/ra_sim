# RA-SIM Refactor Plan

## Purpose

This document tracks the maintainability refactor work in the repository:

- what has already been completed
- what is still left
- why the remaining work matters
- the recommended order for finishing it

This is an active tracking document. `docs/refactor_migration.md` remains the
high-level migration summary; this file is the working plan.

## Current Position

As of 2026-03-28, the refactor has made real progress. The legacy root-script
problem is largely solved, the last broad GUI runtime-state extraction pass has
landed, the structure-model / diffuse-HT rebuild workflow plus the primary-CIF
reload state transition have moved into an extracted helper module, the shared
canvas click/drag arbitration now also lives in an extracted helper module, the
live-preview overlay status/config plus preview-exclusion clear/toggle workflow
now also lives in the extracted geometry Q-group manager module, the runtime
binding/callback bootstrap for the extracted GUI workflows now also lives in
`ra_sim.gui.bootstrap`, the selected-peak / HKL-pick runtime setup now also
boots through one shared helper there instead of being assembled inline, the
integration-range drag rectangle setup plus live region-refresh callback now
also boot there instead of being assembled inline, the live structure-factor-
pruning and Bragg-Qr manager workflow now also boots there through one shared
helper instead of being manually threaded together in `runtime.py`, the
geometry-fit manual-pair action binding assembly now also delegates through
`ra_sim.gui.geometry_fit` instead of being nested inline in `runtime.py`, the
top-level geometry-fit action callback now also resolves through a shared live
bindings factory and zero-arg helper in `ra_sim.gui.geometry_fit` instead of
being rebuilt inline in `runtime.py`, the runtime assembly of that geometry-fit
action bootstrap now also flows through one shared helper in
`ra_sim.gui.bootstrap` instead of being manually staged inline in
`runtime.py`, the
HKL lookup control surface plus its shared Bragg-Qr open action now also boot
through one shared helper in `ra_sim.gui.bootstrap` instead of being wired
directly in `runtime.py`, and the main remaining
cleanup target is now the
residual workflow/orchestration logic still inline in `ra_sim/gui/runtime.py`,
not `main.py` or `mosaic_profiles.py`.

### What Is Already Done

- Root launch path is now packaged.
  - `main.py` is a thin compatibility wrapper around `ra_sim.launcher`.
  - `ra_sim.cli` launches the GUI via package code, not a root-script monolith.
- Package import safety improved.
  - `ra_sim.gui.app` is now an import-safe entrypoint that lazy-loads
    `ra_sim.gui.runtime`.
  - `tests/test_import_smoke.py` is green except for the intentional skip of
    `ra_sim.gui.runtime`, which still performs heavy GUI startup work on import.
- Large GUI feature slices have already been extracted from `runtime.py`.
  - `ra_sim.gui.bootstrap`
  - `ra_sim.gui.background`
  - `ra_sim.gui.background_theta`
  - `ra_sim.gui.bragg_qr_manager`
  - `ra_sim.gui.canvas_interactions`
  - `ra_sim.gui.geometry_fit`
  - `ra_sim.gui.geometry_overlay`
  - `ra_sim.gui.manual_geometry`
  - `ra_sim.gui.overlays`
  - `ra_sim.gui.structure_model`
  - `ra_sim.gui.state_io`
- Manual geometry migration advanced substantially.
  - Selection helpers, caked-coordinate helpers, serialization helpers,
    placement snapshot/apply helpers, placement export/import dialog workflow,
    preview/session flow, and the remaining manual-pick/Q-group orchestration
    now live in `ra_sim.gui.manual_geometry`.
  - `runtime.py` now delegates those paths through thin wrappers.
- Manual-geometry state migration has started.
  - Shared manual-geometry state now lives in `ra_sim.gui.state`.
  - Manual-geometry undo/session/pair-map mutations now flow through
    `ra_sim.gui.controllers`.
  - Direct controller tests cover in-place state replacement and undo restore.
- Geometry-fit history and Q-group selector state migration has started.
  - Geometry-fit undo/redo history now has shared state in `ra_sim.gui.state`.
  - Q-group selector cached rows and refresh state now have shared state in
    `ra_sim.gui.state`.
  - Controller helpers now own the corresponding stack/list/refresh mutations.
  - The selector formatting/status/save-load workflow now also lives in
    `ra_sim.gui.geometry_q_group_manager`.
- Live geometry preview and Q-group view migration has started.
  - Preview exclusion state, the one-shot preview skip flag, and the
    auto-match background cache now live in `ra_sim.gui.state`.
  - Q-group selector window/widget ownership now flows through
    `ra_sim.gui.views`.
  - Direct tests now cover the new preview controller helpers and extracted
    Q-group view rendering helpers.
- Live geometry preview exclusion migration advanced further.
  - Excluded preview-pair keys, exclude-mode armed state, and cached preview
    overlay summary metrics now live in `ra_sim.gui.state`.
  - Controller helpers now own preview exclusion toggles and preview overlay
    snapshot replacement.
  - Dead preview-selector scaffolding was removed from `runtime.py`.
- Live geometry preview / Qr-Qz workflow extraction advanced further.
  - Preview auto-match config normalization, cached overlay-state payload
    helpers, live-preview status rendering, and preview exclusion clear/toggle
    helpers now live in `ra_sim.gui.geometry_q_group_manager`.
  - The live-preview enable/disable action flow now also lives in
    `ra_sim.gui.geometry_q_group_manager`.
  - `runtime.py` now delegates those paths through thin wrappers.
- Selected-peak / HKL-pick bootstrap cleanup has advanced further.
  - The live selected-peak canvas-pick/intersection config factories, peak-
    overlay callback wiring, and selected-peak runtime callback bundle now
    assemble through one shared helper in `ra_sim.gui.bootstrap`.
  - `runtime.py` no longer wires that selected-peak setup inline.
- HKL lookup / Bragg-Qr open-control cleanup has advanced further.
  - The HKL lookup control cluster, its initial HKL-pick button refresh, and
    the shared Bragg-Qr manager open action now assemble through one shared
    helper in `ra_sim.gui.bootstrap`.
  - `runtime.py` no longer wires that control cluster directly or keeps a
    standalone Bragg-Qr open-control alias.
- Geometry-fit action bootstrap cleanup has advanced further.
  - The live geometry-fit action binding factory and fit-button callback now
    assemble through one shared helper in `ra_sim.gui.bootstrap`.
  - `runtime.py` no longer assembles that live geometry-fit action bootstrap
    inline.
- Integration-range drag bootstrap cleanup has advanced further.
  - The drag-selection rectangle, integration-region rectangle, live
    integration-region refresh callback, and runtime drag callback bundle now
    assemble through one shared helper in `ra_sim.gui.bootstrap`.
  - `runtime.py` no longer creates those drag/region rectangles inline or owns
    the thin integration-region refresh wrapper around that workflow.
- Bragg-Qr / structure-factor-pruning bootstrap cleanup has advanced further.
  - The live pruning callback surface, Bragg-Qr manager callback surface, and
    the refresh/apply-filters link between them now assemble through one
    shared helper in `ra_sim.gui.bootstrap`.
  - `runtime.py` no longer manually threads the live Bragg-Qr manager refresh
    callback into the pruning workflow or re-wires the manager apply-filters
    callback inline.
- Bragg Qr manager migration has started.
  - Bragg-Qr selection/index bookkeeping now lives in `ra_sim.gui.state`.
  - Controller helpers now own Bragg-Qr selection mapping and group/L-value
    toggle mutations.
  - `ra_sim.gui.views` now owns the Bragg Qr manager window lifecycle and
    listbox rendering helpers.
  - The Bragg-Qr source/L-key normalization and filtered rod/HKL rebuild
    pipeline now also live in `ra_sim.gui.controllers`.
  - The Bragg-Qr group-entry listing, L-value mapping, and list-model/status
    formatting for that manager now also live in `ra_sim.gui.controllers`.
  - The Bragg-Qr manager selection and list-refresh workflow now also lives in
    `ra_sim.gui.bragg_qr_manager`.
- hBN geometry debug viewer migration has landed.
  - Shared widget references for the debug viewer now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the debug window lifecycle and report-text
    rendering helper.
  - Direct view tests now cover the extracted debug viewer helpers.
- Geometry-fit constraints panel migration has landed.
  - Shared widget references and row-control state for the constraints panel
    now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the scrollable constraints panel construction
    and mouse-wheel routing helpers.
  - Direct view tests now cover the extracted constraints panel helpers.
- Background-theta / geometry-fit background control migration has landed.
  - Shared widget references and `StringVar` state for those control surfaces
    now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the panel construction and event binding for
    those background-theta controls.
  - Direct view tests now cover the extracted background-theta control helpers.
- Workspace panel / background-backend debug control migration has landed.
  - Shared widget references for the workspace action/background/session
    panels and the background backend/orientation debug controls now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the workspace panel construction,
    background-file status text wiring, reusable stacked-button rendering, and
    background backend/orientation debug control construction.
  - Direct view tests now cover the extracted workspace/debug control helpers.
- Manual-geometry / geometry-fit-history alias cleanup has advanced.
  - `runtime.py` no longer keeps direct module-level aliases to the
    manual-geometry pair/session/undo stores or the geometry-fit undo/redo
    stacks.
  - Those reads now go straight through the shared state containers and the
    existing controller helpers.
- Geometry-tool action control migration has landed.
  - Shared widget references and `StringVar` state for the fit-history,
    manual-placement, and preview-exclusion action controls now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of that geometry-tools action
    cluster, plus helper updates for the fit-history button enabled state and
    the manual-pick / preview-exclude button labels.
  - Direct view tests now cover the extracted geometry-tool action helpers.
- HKL lookup control migration has landed.
  - Shared widget references and `StringVar` state for the HKL lookup /
    peak-selection controls now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the HKL lookup panel construction, its entry
    bindings, and helper updates for the HKL values and image-pick button
    label.
  - Direct view tests now cover the extracted HKL lookup helpers.
- Geometry overlay / analysis-view control migration has landed.
  - Shared widget references and `BooleanVar` state for the Qr-cylinder
    overlay, fit-mosaic, and 1D/caked/log analysis toggles now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of those overlay/action and
    analysis-view control clusters.
  - Direct view tests now cover the extracted overlay/action and analysis-view
    helpers.
- Analysis export control migration has landed.
  - Shared widget references for the analysis export buttons now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of the snapshot/Q-space/grid
    export control cluster.
  - Direct view tests now cover the extracted analysis export helpers.
- Sampling-resolution / optics-mode control migration has landed.
  - Shared widget references and `StringVar` state for those controls now live
    in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of that control cluster plus the
    custom-sample enable/disable and summary-label helper updates.
  - `ra_sim.gui.controllers` now owns the sampling-count parsing,
    resolution-choice normalization, and summary formatting helpers.
  - Direct controller/view tests now cover the extracted sampling/optics
    helpers.
- Finite-stack control migration has landed.
  - Shared widget references and Tk var state for the finite-stack toggle,
    layer count, phi-L divisor, and phase-delta equation controls now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of that control cluster plus the
    layer-widget enable/disable, scale-range growth, and entry-text helpers.
  - `ra_sim.gui.controllers` now owns the finite-stack layer-count and
    Hendricks-Teller input normalization / formatting helpers.
  - Direct controller/view tests now cover the extracted finite-stack helpers.
- Display-control panel migration has landed.
  - Shared widget references plus display-limit override/callback bookkeeping
    for the background/simulation intensity controls now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the display-control panel construction and
    slider/entry reference capture.
  - `ra_sim.gui.controllers` now owns the shared display-intensity range and
    scale-factor normalization helpers used by that control surface.
  - Direct controller/view tests now cover the extracted display-control
    helpers.
- Structure-factor pruning / arc-integration control migration has landed.
  - Shared widget references and Tk var state for the prune-bias status/mode
    controls and solve-q inputs now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of that control cluster plus the
    adaptive relative-tolerance enable/disable helper.
  - `ra_sim.gui.controllers` now owns the prune-bias clipping and solve-q
    mode/step/tolerance normalization helpers.
  - Direct controller/view tests now cover the extracted pruning / arc-
    integration helpers.
- Beam/mosaic parameter slider migration has landed.
  - Shared widget references and Tk var state for the main geometry, detector,
    lattice, mosaic, Debye, beam-center, and bandwidth sliders now live in
    `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns construction of that slider cluster, including
    the range-expanding lattice/detector sliders and mosaic-update wiring.
  - `ra_sim.gui.controllers` now owns the shared slider-bounds clamp helper
    used by the remaining psi-z range guard.
  - Direct controller/view tests now cover the extracted beam/mosaic slider
    helpers.
- Stacking-probability / occupancy / atom-site control migration has landed.
  - Shared widget references and Tk var state for the stacking probability
    sliders, occupancy panels, and atom-site fractional-coordinate table now
    live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns those panel constructors and the dynamic
    occupancy / atom-site rebuild helpers.
  - `ra_sim.gui.controllers` now owns the occupancy clamp and stacking
    weight-normalization helpers used by that control family.
  - Direct controller/view tests now cover the extracted stacking / occupancy /
    atom-site helpers.
- Primary-CIF / diffuse-HT control migration has landed.
  - Shared widget references and `StringVar` state for the primary CIF path
    entry and diffuse-HT action buttons now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns that control-cluster construction and the
    corresponding entry/button bindings.
  - Direct view tests now cover the extracted primary-CIF control helpers.
- CIF-weight control migration has landed.
  - Shared widget references and Tk var state for the optional secondary-CIF
    weight sliders now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns that control-cluster construction.
  - `ra_sim.gui.controllers` now owns the weighted-CIF intensity combination
    helper reused by the runtime recompute path.
  - Direct controller/view tests now cover the extracted CIF-weight helpers.
- Geometry-fit parameter checklist migration has landed.
  - Shared widget references and Tk var state for the fit-geometry parameter
    checklist now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns that checklist construction and checkbutton
    layout.
  - Direct view tests now cover the extracted fit-parameter checklist helper.
- 1D integration-range control migration has landed.
  - Shared widget references and Tk var state for the 1D integration-range
    sliders, labels, and entry widgets now live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns that control-cluster construction and entry
    bindings.
  - Direct view tests now cover the extracted integration-range helper.
- GUI shell / status-panel migration has landed.
  - Shared widget references for the top-level notebook shell, scrolled panel
    bodies, parameter columns, analysis containers, and bottom status panel now
    live in `ra_sim.gui.state`.
  - `ra_sim.gui.views` now owns the top-level shell construction, notebook
    state synchronization, compact console-backed status labels, and status
    panel construction.
  - Direct view tests now cover the extracted app-shell and status-panel
    helpers.
- Shared app-state ownership of extracted GUI/runtime state has landed.
  - `ra_sim.gui.runtime` now initializes the extracted view/state containers
    through one shared `AppState` instance instead of instantiating them
    piecemeal as separate runtime globals.
  - Background-cache/orientation state and HKL image-pick interaction state now
    also have explicit runtime-state containers in `ra_sim.gui.state` and are
    synchronized into that shared app state.
- Remaining GUI runtime-state extraction has landed.
  - `ra_sim.gui.state` now also owns explicit runtime-state containers for
    background file/current-orientation state, geometry interaction/cache
    state, simulation/update/caking/peak cache state, atom-site override cache
    state, Bragg-Qr disabled-toggle state, hBN debug-report text, sampling
    count state, and the caked-view override flag.
  - The remaining `global` lines in `ra_sim.gui.runtime` are now limited to
    structure-model / diffuse-HT rebuild state plus the legacy `write_excel`
    flag.
- Structure-model / diffuse-HT workflow extraction has landed.
  - `ra_sim.gui.structure_model` now owns the primary CIF metadata parsing,
    atom-site override CIF generation, HT-cache bootstrap/rebuild helpers, and
    weighted intensity recompute used by the live GUI.
  - `ra_sim.gui.runtime` now delegates the initial structure-model boot path
    and the occupancy / stacking rebuild flow through that extracted module.
  - `ra_sim.gui.runtime` no longer keeps separate inline copies of the
    primary-CIF occupancy/atom-site helper logic; those thin wrappers now also
    delegate to `ra_sim.gui.structure_model`.
  - The primary-CIF reload state transition now also flows through
    `ra_sim.gui.structure_model`, including snapshot/restore helpers for the
    structure-model state slice and diffuse-HT request packaging reused by the
    open/export actions.
  - The primary-CIF browse dialog plus the diffuse-HT open/export dialog
    workflow now also flow through `ra_sim.gui.structure_model`, leaving
    `ra_sim.gui.runtime` with thin delegate wrappers plus control-var rebuild
    callbacks.
- Bragg-Qr / structure-factor pruning workflow extraction has advanced.
  - `ra_sim.gui.controllers` now owns the Bragg-Qr source/L-key normalization,
    disabled-filter pruning, filtered rod/HKL rebuild helpers, and status-text
    formatting used by the live GUI.
  - `ra_sim.gui.structure_factor_pruning` now owns the remaining
    structure-factor pruning / solve-q control callback workflow, pruning-
    status refresh, Bragg-Qr filter side effects, and adaptive-control sync
    used by the live GUI.
  - `ra_sim.gui.structure_factor_pruning` now also owns the zero-arg runtime
    binding/callback factories used by pruning-status refresh, Bragg-Qr
    filter application, and solve-q control traces, plus the normalized
    pruning / solve-q default and current-value helpers used by the runtime.
  - `ra_sim.gui.runtime` now keeps only bound structure-factor-pruning
    callback values plus a few live call sites around that workflow.
  - The Bragg-Qr manager list-building workflow now also delegates through
    controller helpers, leaving runtime with listbox selection reads and the
    enable/disable action callbacks.
  - The Bragg-Qr manager selection/list-refresh flow now also delegates
    through `ra_sim.gui.bragg_qr_manager`, leaving runtime with the manager
    window lifecycle and enable/disable action callbacks.
  - The remaining Bragg-Qr manager enable/disable/toggle action workflow now
    also lives in `ra_sim.gui.bragg_qr_manager`, and the runtime-side callback
    wiring for that manager now also flows through one shared runtime-binding
    context in that module.
  - `ra_sim.gui.bragg_qr_manager` now also owns the live lattice-value
    normalization, Bragg-Qr entry/L-value construction, active
    Qr-cylinder overlay entry derivation, and the zero-arg overlay-entry
    factory used by the runtime workflow.
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    refresh/open callback helpers used by the live filter pipeline and HKL
    lookup controls.
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    binding-factory helper used by the live filter pipeline, HKL lookup
    controls, and manager callbacks.
  - `ra_sim.gui.qr_cylinder_overlay` now also owns the analytic Qr-cylinder
    overlay render config, zero-arg render-config factory, cache-signature,
    path-construction helpers, and the runtime binding/refresh/toggle helpers
    used by the live detector/caked overlay workflow.
  - `ra_sim.gui.runtime` now keeps only one bound Bragg-Qr runtime factory
    value plus the remaining live manager/overlay call sites for that
    workflow.
- Geometry-fit Qr/Qz selector workflow extraction has advanced.
  - `ra_sim.gui.geometry_q_group_manager` now owns the selector line/status
    formatting, window refresh, checkbox/bulk include-exclude side effects,
    update-listed-peaks request flow, and save/load dialog workflow used by
    the live GUI.
  - `ra_sim.gui.geometry_q_group_manager` now also owns shared propagated-
    hit filtering, reflection Qr/Qz group metadata, stable group-key
    reconstruction, and selector-entry snapshot assembly used by the live
    geometry workflows.
  - `ra_sim.gui.geometry_q_group_manager` now also owns simulated-peak
    assembly from cached hit tables and geometry-fit hit-table exports used
    by the live preview and fit workflows.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the geometry-fit
    hit-table simulation, max-position center aggregation, and preview-style
    simulated-peak helper bundle used by the live preview, geometry fit, and
    manual-pick workflows.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime callback
    bundle that binds those geometry-fit simulation helpers to live lattice/
    cached-peak/display sources.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime value
    callback bundle for cached preview peaks, listed-Q-group state, export
    rows, selector status text, and the live Qr/Qz entry snapshot helpers.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the live-preview
    exclusion key/HKL/filter helpers plus the runtime value callbacks that
    apply those exclusions to cached preview matches.
  - `ra_sim.gui.geometry_q_group_manager` now also owns shared geometry-fit
    seed filtering and degenerate-collapse helpers used by the live preview,
    geometry fit, and manual-pick grouping workflows.
  - `ra_sim.gui.geometry_q_group_manager` now also owns cached-entry
    snapshot replacement/capture plus the preview-exclusion open/status
    helper used by the live update cycle and geometry tool controls.
  - `ra_sim.gui.geometry_q_group_manager` now also owns live-preview
    auto-match config normalization, cached overlay-state payload helpers,
    status rendering, and preview exclusion clear/toggle helpers used by the
    live preview workflow.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the live-preview
    enable/disable action flow used by the geometry tool controls.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime callback
    bundle entries for preview-exclusion window open, preview-exclude-mode
    toggling, preview-exclusion point toggling, exclusion clearing, and the
    live-preview checkbox action flow.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime live-
    preview enabled/render helpers plus the callback-bundle entries that
    expose those bound redraw/status paths.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime live-
    preview simulated-peak acquisition/fallback helper used before the
    remaining preview filter/collapse/auto-match orchestration.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime live-
    preview seed filter/collapse helper, including the empty-state exits for
    no selected Qr/Qz groups and no remaining seeds after collapse.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime live-
    preview match-result application helper that stores overlay state and
    redraws the cached preview after the remaining auto-match call.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the runtime live-
    preview availability/background gate used before preview config assembly,
    including the disabled, caked-view, and hidden-background exits.
  - `ra_sim.gui.geometry_q_group_manager` now also owns the zero-arg runtime
    binding/callback bundle used for selector open/refresh/toggle/include/
    exclude/save/load/update actions.
  - `ra_sim.gui.runtime` now keeps only one bound geometry-selector factory/
    callback bundle plus thin fit-preview/cached-hit value sources, live-
    preview availability/fallback simulation orchestration, a narrowed
    update-cycle refresh call site, and thin geometry-tool button wiring
    around that workflow.
- Geometry-fit manual-pair fitting workflow extraction has advanced.
  - `ra_sim.gui.geometry_fit` now owns manual-pair dataset assembly, fit
    preflight validation, fitted-value application, and matched-peak export
    row helpers used by the live geometry-fit action.
  - `ra_sim.gui.geometry_fit` now also owns optimizer diagnostic formatting,
    RMS/result summary helpers, fitted-parameter merge helpers, overlay-
    diagnostic formatting, pixel-offset analysis/formatting, and final
    geometry-fit status-text assembly used after the live solver returns.
  - `ra_sim.gui.geometry_fit` now also owns the pure post-solver analysis
    bundle that runs simulated/measured peak comparison, aggregates matched
    centers, filters overlay diagnostics, assembles overlay-state payloads,
    and packages the matched-peak export/save payload used by the live GUI.
  - `ra_sim.gui.geometry_fit` now also owns the geometry-fit profile-cache
    merge helper used after one successful fit.
  - `ra_sim.gui.geometry_fit` now also owns the runtime result-application
    helper that orchestrates optimizer-diagnostic logging, undo/profile-cache
    updates, preview-refresh scheduling, postprocess application, overlay-state
    persistence, export-save callbacks, and final success-status reporting
    through one callback bundle.
  - `ra_sim.gui.geometry_fit` now also owns the prepared optimizer dataset-
    spec packaging, the geometry-fit start-log/console prelude assembly, and
    the runtime result-binding factory that snapshots current UI values into
    fitted-parameter and profile-cache updates.
  - `ra_sim.gui.geometry_fit` now also owns the live solver-request packaging,
    log-file lifecycle, execution/failure reporting, and the runtime execution
    helper that runs one prepared geometry-fit through the solver and extracted
    postprocess/apply workflow.
  - `ra_sim.gui.geometry_fit` now also owns the runtime preparation wrapper
    that packages the live dataset-builder/runtime-config callbacks around the
    extracted preflight helper, plus the runtime execution-setup builder that
    packages live state/callback sources into the extracted execution helper.
  - `ra_sim.gui.geometry_fit` now also owns the shared live geometry-fit
    value-source bundle for selected fit variables, current parameter reads, UI
    snapshots, and the runtime var-map reused by undo/execute flows.
  - `ra_sim.gui.geometry_fit` now also owns the top-level runtime action helper
    that drives one live geometry fit from current values through prepare,
    execution-setup packaging, solver execution, and the top-level preflight
    failure/status path.
  - `ra_sim.gui.geometry_fit` now also owns the shared prepare-bundle factory,
    execution-bundle builder, and top-level action-binding builder used by the
    live geometry-fit action callback.
  - `ra_sim.gui.geometry_fit` now also owns the live action-bindings factory
    and zero-arg action callback helper used by the runtime fit button.
  - `ra_sim.gui.runtime` no longer rebuilds that geometry-fit action bindings
    bundle inline on each click; it now keeps only the remaining Tk-side
    control wiring around the extracted helper.
- Background-file workflow extraction has advanced.
  - `ra_sim.gui.background_manager` now owns the background-file state
    transition, file-dialog initial-dir selection, background status refresh,
    and the post-load/post-switch redraw/reset sequencing used by the live
    GUI.
  - `ra_sim.gui.background_manager` now also owns the backend-orientation
    debug status plus the rotate/flip/reset runtime helpers used by the live
    debug controls.
  - `ra_sim.gui.background_manager` now also owns the background visibility
    toggle workflow used by the workspace action controls.
  - `ra_sim.gui.background_manager` now also owns the zero-arg runtime
    binding/callback bundle for background status refresh, backend debug
    status, visibility/browse/load/switch actions, and backend rotate/flip/
    reset actions.
  - `ra_sim.gui.runtime` now keeps only one bound background callback bundle
    plus thin status-refresh and control-wiring call sites around that
    workflow.
- Integration-range drag workflow extraction has advanced.
  - `ra_sim.gui.state` now also owns explicit live drag-selection state for
    1D integration-range picking.
  - `ra_sim.gui.integration_range_drag` now owns the canvas clamp helpers,
    detector-angle lookup, raw preview mask rendering, current
    integration-region visual refresh, drag/region rectangle construction,
    range-apply logic, and the runtime binding/callback bundle for raw/caked
    integration-range dragging.
  - `ra_sim.gui.runtime` now keeps only the manual-geometry-specific canvas
    branches, top-level event dispatch, and thin live call sites around that
    workflow.
- HKL lookup / selected-peak workflow extraction has advanced.
  - `ra_sim.gui.peak_selection` now owns HKL lookup parsing, selected-peak
    summary text, degenerate-HKL lookup, selection-by-index/HKL,
    clear-selection state transitions, HKL-pick toggle / raw-image click
    selection workflow, and selected-peak Bragg/Ewald intersection analysis
    used by the live GUI.
  - `ra_sim.gui.peak_selection` now also owns the selected-peak config
    builders, the ideal-center probe helper used by raw-image HKL picking,
    the shared runtime config-factory bundle helper, the simulated peak-
    overlay cache builder/runtime callback, and the runtime config-factory
    helpers that resolve live GUI values for that workflow.
  - `ra_sim.gui.peak_selection` now also owns the runtime binding/callback
    bundle used for HKL-pick button labels, mode toggles, selected-peak
    refresh, HKL-control selection, raw-image click selection, and the
    selected-peak Bragg/Ewald action.
  - `ra_sim.gui.runtime` now keeps only cross-feature click-mode dispatch,
    thin live value-source wiring, and the bound peak-selection runtime
    wiring around that workflow.
- Canvas interaction workflow extraction has advanced.
  - `ra_sim.gui.canvas_interactions` now owns the top-level raw-image canvas
    click/drag arbitration between manual-geometry placement, preview
    exclusion, HKL picking, and integration-range dragging.
  - `ra_sim.gui.runtime` now keeps only one bound canvas callback bundle plus
    thin event-hook wiring around that workflow.
- Runtime callback/bootstrap extraction has advanced.
  - `ra_sim.gui.bootstrap` now owns the assembly of the runtime binding/
    callback bundles for structure-factor pruning, Bragg-Qr manager workflow,
    Qr-cylinder overlay, selected-peak workflow, integration-range dragging,
    canvas interactions, background workflow, and the geometry Q-group
    workflow.
  - `ra_sim.gui.runtime` now keeps the live value-source/config factories for
    those workflows plus the bound bundle variables consumed by the remaining
    call sites.
- Several tests were moved off monolith-coupled runtime behavior and onto
  extracted modules.

### What This Means

- The original goal of turning `main.py` into a wrapper is effectively done.
- The main unfinished work is the final stage of breaking down
  `ra_sim/gui/runtime.py` and turning the GUI scaffolding modules into real
  application structure.

## Workstreams

### 1. GUI Runtime Decomposition

Status: Completed

What is done:

- The runtime no longer owns every extracted subsystem directly.
- Overlay, geometry-fit, background, state-IO, background-theta, and most
  manual-geometry logic now live in dedicated modules.
- Manual-geometry state is no longer just loose runtime-owned storage.
  - Shared state now has a `ManualGeometryState` container.
  - Undo snapshots and state replacement now flow through controller helpers.
- Geometry-fit history and Q-group selector state are no longer only loose
  runtime-owned storage.
  - Shared state now has geometry-fit history and Q-group selector containers.
  - Controller helpers now own fit-history stack transitions and Q-group
    snapshot/refresh mutations.
- Live geometry preview filter/cache state is no longer just loose runtime
  storage.
  - Shared state now has a preview-state container for excluded Q-groups, the
    skip-once flag, excluded live-preview pairs, exclude-mode state, cached
    overlay summary data, and auto-match background cache data.
- The Q-group selector window lifecycle is no longer built directly in
  `runtime.py`.
  - `ra_sim.gui.views` now owns the selector window/widget construction and row
    rendering helpers.
- The Bragg Qr manager is no longer built directly in `runtime.py`.
  - `ra_sim.gui.views` now owns its window construction and listbox refresh
    helpers.
- The hBN geometry debug viewer is no longer built directly in `runtime.py`.
  - `ra_sim.gui.views` now owns its window construction and report-text
    rendering.
- The geometry-fit constraints panel is no longer assembled directly in
  `runtime.py`.
  - `ra_sim.gui.views` now owns its scrollable panel construction and
    mouse-wheel routing helpers.
- The background-theta and geometry-fit background controls are no longer built
  directly in `runtime.py`.
  - `ra_sim.gui.views` now owns their panel construction and entry/button
    bindings.
- The workspace action/background/session panels are no longer built directly
  in `runtime.py`.
  - `ra_sim.gui.views` now owns their panel construction, background-file
    status label wiring, and reusable button-stack rendering.
- The background backend/orientation debug controls are no longer built
  directly in `runtime.py`.
  - `ra_sim.gui.views` now owns their widget construction and the corresponding
    Tk var/status-label view state.
- `runtime.py` no longer keeps direct aliases to the shared manual-geometry
  state stores or geometry-fit history stacks.
  - Remaining runtime reads now go through the shared state containers
    themselves and the existing controller helpers.
- The geometry-tools fit-history/manual-placement/preview-exclusion action
  controls are no longer assembled directly in `runtime.py`.
  - `ra_sim.gui.views` now owns that action-control construction and the
    related button-var/button-state helpers.
- The HKL lookup / peak-selection controls are no longer assembled directly in
  `runtime.py`.
  - `ra_sim.gui.views` now owns that panel construction and the related HKL
    entry/button-var helpers.
- The Qr-cylinder / fit-mosaic action controls and the 1D/caked/log analysis
  toggles are no longer assembled directly in `runtime.py`.
  - `ra_sim.gui.views` now owns those control-cluster constructors and the
    corresponding `BooleanVar` view state.
- The analysis export buttons are no longer assembled directly in `runtime.py`.
  - `ra_sim.gui.views` now owns that export-control construction.
- The sampling-resolution / optics-mode controls are no longer assembled
  directly in `runtime.py`.
  - `ra_sim.gui.state` now owns their shared widget/`StringVar` view state.
  - `ra_sim.gui.views` now owns that control-cluster construction and the
    custom-sample widget-state / summary-label helpers.
  - `ra_sim.gui.controllers` now owns the sampling-count parsing,
    resolution-choice normalization, and summary formatting helpers.
- The finite-stack controls are no longer assembled directly in `runtime.py`.
  - `ra_sim.gui.state` now owns their shared widget/Tk-var view state.
  - `ra_sim.gui.views` now owns that control-cluster construction and the
    layer-widget enable/disable, scale-range, and entry-text helpers.
  - `ra_sim.gui.controllers` now owns the finite-stack layer-count and
    Hendricks-Teller input normalization / formatting helpers.
- The background/simulation display controls are no longer assembled directly
  in `runtime.py`.
  - `ra_sim.gui.state` now owns their widget refs plus display-limit
    override/callback state.
  - `ra_sim.gui.views` now owns the display-control panel construction and
    scale-factor entry lookup.
  - `ra_sim.gui.controllers` now owns the shared display-intensity range and
    scale-factor normalization helpers.
- The structure-factor pruning / arc-integration controls are no longer
  assembled directly in `runtime.py`.
  - `ra_sim.gui.state` now owns their widget refs and Tk-var view state.
  - `ra_sim.gui.views` now owns that control-cluster construction and the
    relative-tolerance enabled-state helper.
  - `ra_sim.gui.controllers` now owns the prune-bias clipping and solve-q
    mode/step/tolerance normalization helpers.
- The main beam/mosaic parameter slider cluster is no longer assembled
  directly in `runtime.py`.
  - `ra_sim.gui.state` now owns the corresponding slider var/scale view state.
  - `ra_sim.gui.views` now owns that slider construction and callback wiring.
  - `ra_sim.gui.controllers` now owns the shared slider-bounds clamp helper
    used by the remaining psi-z guard.
- The stacking-probability / occupancy / atom-site control cluster is no longer
  assembled directly in `runtime.py`.
  - `ra_sim.gui.state` now owns the corresponding slider/widget/Tk-var view
    state.
  - `ra_sim.gui.views` now owns that panel construction plus the dynamic
    occupancy and atom-site control rebuild helpers.
  - `ra_sim.gui.controllers` now owns the occupancy clamp and stacking
    weight-normalization helpers reused by that control family.
- The 1D integration-range controls are no longer assembled directly in
  `runtime.py`.
  - `ra_sim.gui.state` now owns their shared widget/Tk-var view state.
  - `ra_sim.gui.views` now owns that control-cluster construction and entry
    bindings.
- The top-level GUI shell is no longer assembled directly in `runtime.py`.
  - `ra_sim.gui.state` now owns the notebook-shell/status-panel view state.
  - `ra_sim.gui.views` now owns the top-level pane/notebook/scrolled-frame
    construction and notebook selection synchronization helpers.
- The bottom status panel is no longer built directly in `runtime.py`.
  - `ra_sim.gui.views` now owns the compact console-backed status-label wrapper
    and the shared status-panel construction helper.
- The extracted GUI state containers are no longer instantiated ad hoc in
  `runtime.py`.
  - `ra_sim.gui.runtime` now builds them through one shared `AppState`
    instance.
- Background-cache/orientation state and HKL image-pick interaction state are
  no longer only loose runtime globals.
  - `ra_sim.gui.state` now has explicit runtime-state containers for those
    slices, and `runtime.py` synchronizes them through the shared app state.
- Long-lived GUI runtime state is no longer scattered across `runtime.py`.
  - `ra_sim.gui.state` now owns explicit runtime-state containers for
    background file/current-orientation state, geometry interaction/cache
    state, simulation/update/caking/peak cache state, atom-site override cache
    state, Bragg-Qr disabled-toggle state, hBN debug-report text, sampling
    count state, and the caked-view override flag.
  - The remaining runtime `global` lines are limited to structure-model /
    diffuse-HT rebuild state plus the legacy `write_excel` flag.

What is left:

- No further GUI runtime-state extraction blockers remain.
- `ra_sim/gui/runtime.py` is still very large and still coordinates too many
  cross-feature workflow transitions and Tk widgets inline.
- The follow-on runtime cleanup is now the remaining controller/view workflow
  glue still inline in `runtime.py`, not another round of GUI runtime-state
  extraction or the structure-model / diffuse-HT rebuild path.
- The remaining structure-model runtime code is now mostly thin delegate
  wrappers, progress-label wiring, and control-var rebuild callbacks.
- The remaining structure-factor-pruning runtime code is now mostly control
  trace hookups plus a few value-source call sites around the extracted
  pruning module and shared bootstrap wiring.
- The remaining Bragg-Qr runtime code is now mostly a few manager/overlay call
  sites around the extracted controller/view modules and shared bootstrap/
  config helpers.
- The remaining Bragg-Qr manager runtime code is now mostly a few manager
  refresh/action call sites around the extracted manager helpers and shared
  bootstrap wiring.
- The remaining geometry-fit Qr/Qz selector runtime code is now mostly thin
  fit-preview parameter/value sources plus a couple of delegated call sites
  around the extracted manager/view helpers, the bound geometry-fit
  simulation/value callback bundles, and shared bootstrap wiring.
- The remaining background runtime code is now mostly one bound callback
  surface plus thin status-refresh and control-wiring call sites around the
  extracted background manager and shared bootstrap helpers.
- The remaining selected-peak runtime code is now mostly a few refresh/action
  call sites around the extracted peak-selection helpers and shared bootstrap
  helpers.
- The remaining geometry-fit manual-pair runtime code is now mostly the
  remaining fit-history/manual-pick mode-clear and preview-action control
  wiring around the extracted `ra_sim.gui.geometry_fit` and
  `ra_sim.gui.manual_geometry` helper surfaces.
- The remaining integration-range drag runtime code is now mostly the
  remaining cross-feature canvas event handoff around the extracted drag and
  canvas-interaction helpers.

Why it matters:

- The highest-risk runtime-state ownership gap is now closed.
- The codebase still has one oversized integration module, but the remaining
  work can now focus on application boundaries and workflow orchestration
  instead of more GUI-global cleanup.

Definition of done:

- `runtime.py` becomes a composition/integration layer rather than the owner of
  feature logic and long-lived feature state.
- Feature state is grouped into explicit structures instead of scattered module
  globals.

### 2. Controllers / State / Views Migration

Status: In progress

What is done:

- `ra_sim.gui.state`, `ra_sim.gui.controllers`, and `ra_sim.gui.views` exist.
- `state.py` now owns real manual-geometry state structures.
- `controllers.py` now owns real manual-geometry state/undo mutations.
- `state.py` now also owns geometry-fit history and Q-group selector state.
- `controllers.py` now also owns geometry-fit history and Q-group selector
  mutations.
- `state.py` now also owns live geometry preview state and Q-group view state.
- `controllers.py` now also owns preview exclusion/skip/cache mutations and
  preview overlay snapshot replacement.
- `views.py` now owns the Q-group selector window lifecycle and row rendering.
- `state.py` now also owns Bragg-Qr manager state.
- `controllers.py` now also owns Bragg-Qr manager selection/toggle helpers.
- `views.py` now also owns the Bragg Qr manager window lifecycle and list
  rendering.
- `state.py` now also owns hBN geometry debug view state.
- `views.py` now also owns the hBN geometry debug viewer lifecycle and
  report-text rendering.
- `state.py` now also owns geometry-fit constraints view state and its
  row-control registry.
- `views.py` now also owns the geometry-fit constraints panel construction and
  scroll helpers.
- `state.py` now also owns the background-theta control view state and its
  `StringVar` references.
- `views.py` now also owns the background-theta and geometry-fit background
  control construction helpers.
- `state.py` now also owns workspace panel view state and background
  backend/orientation debug control view state.
- `views.py` now also owns workspace panel helpers, background-file status
  updates, reusable stacked-button rendering, and background
  backend/orientation debug control construction.
- The runtime now reads manual-geometry state and geometry-fit history
  directly from shared state containers instead of keeping separate alias
  globals for those stores.
- `state.py` now also owns geometry-tool action view state for the fit-history
  and manual-geometry control cluster.
- `views.py` now also owns the geometry-tools action-control construction and
  helper updates for those button refs and `StringVar`s.
- `state.py` now also owns HKL lookup view state for the peak-selection panel.
- `views.py` now also owns the HKL lookup control construction and helper
  updates for its entry vars and pick-button label.
- `state.py` now also owns the geometry overlay / fit-mosaic action view state
  and the analysis-view toggle view state.
- `views.py` now also owns those overlay/action and analysis-view control
  constructors.
- `state.py` now also owns analysis export control view state.
- `views.py` now also owns the analysis export control constructor.
- `state.py` now also owns sampling-resolution / optics-mode control view
  state.
- `views.py` now also owns the sampling/optics control construction and helper
  updates for the custom-sample widgets and summary label.
- `controllers.py` now also owns the sampling-count parsing,
  resolution-choice normalization, and summary formatting helpers.
- `state.py` now also owns finite-stack control view state.
- `views.py` now also owns the finite-stack control construction and helper
  updates for the layer widgets and entry text.
- `controllers.py` now also owns the finite-stack layer-count and
  Hendricks-Teller input normalization / formatting helpers.
- `state.py` now also owns display-control panel view state and the associated
  display-limit override/callback state container.
- `views.py` now also owns the background/simulation display-control
  construction and scale-factor entry discovery.
- `controllers.py` now also owns shared display-intensity range validation and
  display scale-factor normalization helpers.
- `state.py` now also owns structure-factor pruning / arc-integration control
  view state.
- `views.py` now also owns that control-cluster construction and the adaptive
  relative-tolerance enabled-state helper.
- `controllers.py` now also owns the prune-bias clipping and solve-q
  mode/step/tolerance normalization helpers.
- `state.py` now also owns the main beam/mosaic parameter slider view state.
- `views.py` now also owns that slider construction and callback wiring.
- `controllers.py` now also owns the shared slider-bounds clamp helper used by
  the remaining psi-z guard.
- `state.py` now also owns the stacking probability / occupancy / atom-site
  control view state.
- `views.py` now also owns those control constructors plus the dynamic
  occupancy / atom-site rebuild helpers.
- `controllers.py` now also owns the occupancy clamp and stacking
  weight-normalization helpers.
- `state.py` now also owns the primary-CIF / diffuse-HT control view state.
- `views.py` now also owns the primary-CIF path / diffuse-HT control
  construction and entry/button bindings.
- `state.py` now also owns the optional CIF-weight control view state.
- `views.py` now also owns the optional CIF-weight slider construction.
- `controllers.py` now also owns the weighted-CIF intensity combination helper.
- `state.py` now also owns the fit-geometry parameter checklist view state.
- `views.py` now also owns that checklist construction and checkbutton layout.
- `state.py` now also owns the 1D integration-range control view state.
- `views.py` now also owns that control-cluster construction and entry
  bindings.
- `state.py` now also owns the top-level app-shell and status-panel view
  state.
- `views.py` now also owns the top-level pane/notebook shell construction,
  notebook state synchronization, compact status-label wrapper, and shared
  status-panel construction.
- `state.py` now also owns explicit background-runtime and peak-selection state
  containers used by `runtime.py`.
- `controllers.build_initial_state()` now returns the shared `AppState` used by
  `runtime.py` to initialize the extracted GUI state containers.

What is left:

- The GUI still does not flow broadly through an explicit
  state/controller/view boundary.
- The new controller/state boundary is now real for a larger share of the GUI,
  but it is not yet the dominant app structure across the rest of the runtime.
- The next practical boundary targets are the remaining cross-feature
  runtime-owned helpers plus the remaining controller/view workflow
  transitions that still live inline in `runtime.py`.

Why it matters:

- Until these modules become real, extracted feature helpers are still anchored
  to the runtime monolith instead of a maintainable app structure.
- This is the clearest next step for finishing the migration rather than just
  extracting more helpers.

Definition of done:

- `state.py` owns meaningful GUI state structures.
- `controllers.py` owns user-action orchestration and workflow transitions.
- `views.py` owns actual Tk construction helpers beyond a minimal root factory.

### 3. Compatibility Cleanup

Status: Partially done

What is done:

- `main.py` is now a compatibility wrapper.
- `ra_sim.gui.main_app.main` is a working compatibility alias.
- Import smoke around stale `ra_sim.gui.main_app` behavior is fixed.

What is left:

- `ra_sim.gui.main_app` still exists only as a compatibility shim.
- `docs/refactor_migration.md` still describes that alias as temporary.

Why it matters:

- Compatibility wrappers are useful during migration, but they create confusion
  once the real launch path has stabilized.
- The codebase should eventually converge on one canonical GUI entrypoint.

Definition of done:

- One supported GUI entrypoint is documented and used by tests/docs.
- Compatibility shims are either removed or explicitly marked as permanent.

### 4. Config Unification

Status: Partially done

What is done:

- `ra_sim.config.loader` exists as a newer cached config loader.
- `ra_sim.path_config` was restored to compatibility-safe behavior, including
  reloadability.

What is left:

- The project still has two config-loading systems:
  - `ra_sim.path_config`
  - `ra_sim.config.loader`
- Both still resolve config directories and cache config state.

Why it matters:

- Duplicated config systems create drift, duplicate tests, and ambiguous
  extension points.
- This is a structural cleanup item, not just style cleanup.

Definition of done:

- One config loading system is canonical.
- The other becomes a short-lived shim or is removed.
- Duplicate configuration data/files are eliminated where possible.

### 5. Test Architecture Cleanup

Status: Partially done

What is done:

- Several tests were moved away from runtime-heavy extraction and toward direct
  module coverage.
- The old `main.py` AST lock-in is no longer the primary issue it once was.

What is left:

- Some tests still use AST extraction against implementation files such as
  `ra_sim/gui/app.py`.
- `ra_sim.gui.runtime` still cannot participate in normal import-smoke testing
  because it performs heavy startup work on import.

Why it matters:

- AST-based tests are brittle and reward preserving file shape rather than
  preserving behavior.
- Import-heavy runtime behavior still makes normal module testing harder than it
  should be.

Definition of done:

- Tests primarily import stable modules and validate behavior directly.
- AST-only tests are reserved for narrow structural assertions, not as a
  workaround for import problems.

### 6. Repository Cleanup

Status: Not started

What is left:

- The repo root still contains stale or non-source artifacts that do not belong
  in the long-term project layout.
- Examples currently present at the repo root include:
  - `ig_graph.sqlite`
  - `ig_graph.sqlite-shm`
  - `ig_graph.sqlite-wal`
  - `session.json`
  - `oneline`
  - `et --hard a485e65`
  - the legacy root `hbn.py`

Why it matters:

- Root clutter makes packaging, onboarding, and repository intent harder to
  understand.
- It also obscures what is part of the product versus what is an accident or a
  local artifact.

Definition of done:

- The root contains packaging, docs, config, tests, maintained wrappers, and
  intentional top-level scripts only.

### 7. Unrelated Simulation Test Debt

Status: Still open

What is left:

- `tests/test_source_template_cache.py` still fails against
  `ra_sim.simulation.diffraction`.
- The failure is due to tests expecting cache internals such as
  `_PHASE_SPACE_CACHE` and `_Q_VECTOR_CACHE` that are not exposed by the module.

Why it matters:

- This is not part of the GUI migration, but it is still active technical debt.
- It should be tracked separately so it does not get conflated with GUI
  refactor progress.

Definition of done:

- The test is either rewritten around supported behavior or the missing cache
  surface is intentionally restored and documented.

## Recommended Order

### Phase A: Finish Runtime State Extraction

Goal:

- Stop letting `ra_sim/gui/runtime.py` own major feature state directly.

Tasks:

- Group more runtime globals into explicit state containers.
- Move remaining long-lived runtime-owned widget/state coordination into shared
  state containers and `views.py` helpers where it still exists.
- Reduce direct feature coordination in `runtime.py` where controller logic or
  shared state can own it instead.
- Keep moving Tk-owned widget references out of runtime and into `views.py`
  state/helpers.

Why first:

- This is the highest-value remaining cleanup item.
- It unlocks the scaffold modules and reduces the risk of further extraction
  work.

Current status:

- Completed. The remaining GUI work has moved on to workflow/controller/view
  boundaries rather than long-lived runtime-state extraction.

### Phase B: Turn Scaffolding Into Real Modules

Goal:

- Make `state.py`, `controllers.py`, and `views.py` real application
  boundaries.

Tasks:

- Define app-level state containers.
- Move workflow orchestration into controllers.
- Move Tk construction helpers and widget assembly into views.

Why second:

- Once state ownership is clearer, controller/view boundaries become practical
  rather than decorative.

### Phase C: Finish Compatibility Cleanup

Goal:

- Converge on a single documented GUI entrypoint.

Tasks:

- Decide whether `ra_sim.gui.main_app` stays as a permanent compatibility alias
  or is removed later.
- Update docs/tests to reflect the canonical entrypoint.

Why third:

- This is easiest to finalize after the runtime structure is stable.

### Phase D: Unify Config Loading

Goal:

- Eliminate the dual config-system split.

Tasks:

- Migrate call sites from `ra_sim.path_config` to `ra_sim.config`.
- Keep a compatibility shim only as long as needed.
- Remove duplicated config responsibilities.

Why fourth:

- This is a repo-wide change and should happen after the GUI boundaries are less
  volatile.

### Phase E: Finish Test Cleanup

Goal:

- Make tests target imported modules and stable behavior.

Tasks:

- Replace remaining AST-extraction tests where practical.
- Reduce remaining reasons to skip import-smoke coverage.

Why fifth:

- The best time to simplify tests is after the target module boundaries stop
  moving.

### Phase F: Clean the Repository Root

Goal:

- Remove stale artifacts and clarify project layout.

Tasks:

- Remove stray local artifacts from version control.
- Review root-level legacy scripts and wrappers for retention or removal.

Why last:

- This is important, but it is lower leverage than finishing the runtime and
  config migration.

## Suggested Next Concrete Step

The next best step is:

- build on the completed runtime-state extraction by moving the remaining
  cross-feature workflow/orchestration helpers out of `ra_sim/gui/runtime.py`
- focus next on controller-owned user-action flows plus the remaining
  cross-feature runtime workflow glue that still lives inline in `runtime.py`
- keep turning `state.py`, `controllers.py`, and `views.py` into the dominant
  application boundary rather than leaving them as helper scaffolding

Immediate checklist after the HKL lookup / Bragg-Qr open-control cleanup:

- trim the remaining geometry-fit fit-history/manual-pick control wiring
- then trim the remaining selected-peak refresh / action call sites
- then take the remaining Bragg-Qr manager refresh / action call sites
- defer config unification, compatibility cleanup, and repo-root cleanup until
  after those runtime workflow slices stop moving

That is the point where the refactor stops being "runtime state cleanup" and
starts becoming a real application-structure finish.

## Tracking Notes

When updating this file:

- move completed items from “what is left” into “what is done”
- keep the “why it matters” sections intact unless the rationale changes
- update the recommended order only when the dependency structure changes
- keep GUI migration work separate from unrelated simulation-layer test debt
