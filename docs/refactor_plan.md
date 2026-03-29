# RA-SIM Refactor Plan

## Purpose

This document tracks the maintainability refactor work in the repository:

- what has already been completed
- what targeted follow-up remains, if any
- why the remaining work matters
- the recommended order for finishing it

This is an active tracking document. `docs/refactor_migration.md` remains the
high-level migration summary; this file is the working plan.

## Current Position

As of 2026-03-29, the broad maintainability refactor is functionally complete.
The legacy root-script
problem is largely solved, the last broad GUI runtime-state extraction pass has
landed, the structure-model / diffuse-HT rebuild workflow plus the primary-CIF
reload state transition have moved into an extracted helper module, the shared
canvas click/drag arbitration now also lives in an extracted helper module, the
live-preview overlay status/config plus preview-exclusion clear/toggle workflow
now also lives in the extracted geometry Q-group manager module, the runtime
binding/callback bootstrap for the extracted GUI workflows now also lives in
`ra_sim.gui.bootstrap`, the selected-peak / HKL-pick runtime setup now also
boots through one shared helper there instead of being assembled inline, the
remaining selected-peak update-cycle refresh plus restored-target application
callbacks now also assemble through one shared helper in
`ra_sim.gui.peak_selection` and flow through that bootstrap bundle instead of
being wired inline in `runtime.py`, the
integration-range drag rectangle setup plus live region-refresh callback now
also boot there instead of being assembled inline, the live structure-factor-
pruning and Bragg-Qr manager workflow now also boots there through one shared
helper instead of being manually threaded together in `runtime.py`, the
integration-range update/control workflow now also boots there through one
shared helper instead of being wired inline in `runtime.py`, the debounced
range-update scheduler plus analysis-toggle callbacks now also live in
`ra_sim.gui.integration_range_drag`, the workspace background action/file
controls plus backend-debug control wiring now also boot through one shared
helper in `ra_sim.gui.bootstrap`, the background-theta / fit-background-
selection runtime surface now also boots there through one shared helper and
the remaining live theta/background-selection callbacks now also live in
`ra_sim.gui.background_theta`, the
geometry-fit manual-pair action binding assembly now also delegates through
`ra_sim.gui.geometry_fit` instead of being nested inline in `runtime.py`, the
top-level geometry-fit action callback now also resolves through a shared live
bindings factory and zero-arg helper in `ra_sim.gui.geometry_fit` instead of
being rebuilt inline in `runtime.py`, the runtime assembly of that geometry-fit
action bootstrap now also flows through one shared helper in
`ra_sim.gui.bootstrap` instead of being manually staged inline in
`runtime.py`, the remaining geometry-fit manual-pair dataset/value-source
wiring now also assembles through shared manual-dataset/config factories in
`ra_sim.gui.geometry_fit` instead of being threaded inline in `runtime.py`, the
HKL lookup control surface plus its shared Bragg-Qr open action now also boot
through one shared helper in `ra_sim.gui.bootstrap` instead of being wired
directly in `runtime.py`, the geometry-tool action control cluster now also
boots through one shared helper in `ra_sim.gui.bootstrap`, the remaining
fit-history button-state plus manual-pick label/mode/clear runtime callbacks
now also assemble through one shared helper in `ra_sim.gui.geometry_fit`
instead of being wired inline in `runtime.py`, the public
`ra_sim.gui.runtime` module is now also an import-safe wrapper that lazy-loads
the heavy GUI body from `ra_sim/gui/_runtime/runtime_impl.py`, the
Qr-cylinder overlay runtime assembly now also flows through the import-safe
helper module `ra_sim.gui.runtime_qr_cylinder_overlay`, and the project is now
leaving open-ended runtime decomposition mode: the remaining inline code in
that internal implementation module is mostly integration glue, so future
refactors there should be driven by concrete wins rather than file-size
reduction.
The import-safe `ra_sim.gui.runtime` and `ra_sim.gui.app` wrappers now also
share one lazy wrapper helper in `ra_sim.gui.lazy_runtime`, so their local
`main`/`write_excel` forwarding, guarded dunder behavior, lazy `__dir__`
surface, and failure-safe runtime loading no longer drift independently.
The main runtime implementation now also routes lazy background cache
load/current-read state application through `ra_sim.gui.background_manager`
instead of hand-applying those cache payloads inline.
The main runtime implementation now also routes initial background-cache boot
plus shared background-state normalization through `ra_sim.gui.background_manager`
instead of keeping that synchronization logic inline.
The main runtime implementation now also routes background display-control
defaults, transparency application, range application, and slider-default
refresh through `ra_sim.gui.background_manager` instead of keeping that
background display workflow inline.

As part of the next concrete step, `ra_sim.gui.controllers` now owns a shared
Tk after-token cancellation helper used by runtime glue paths that previously
duplicated cancellation logic across manual-geometry and geometry-update flows.
The shared helper is now also used by integration-range scheduling flow to keep
timer-cancel behavior consistent across another runtime workflow.
Compatibility undo handling in the import-safe app shim now also uses this helper
when restoring geometry-fit state before redraw.
Geometry-fit undo restore now also assembles through one shared lazy callback
builder in `ra_sim.gui.geometry_fit`, so both `ra_sim.gui.app` and
`ra_sim/gui/_runtime/runtime_impl.py` no longer hand-build that runtime
restore delegate separately. Direct behavior tests now also cover the live
diffraction Fresnel call paths and the packaged hBN conversion helper instead
of reading source text to infer those contracts.
The import-safe app shim now also resolves geometry-fit UI parameter reads,
selected-fit variable-name reads, and undo-restore var-map lookup through the
same shared geometry-fit value-callback bundle used by the primary runtime
implementation instead of hand-building those value readers locally.
The primary runtime implementation now also assembles its geometry-fit undo and
redo callbacks through one shared history-callback builder in
`ra_sim.gui.geometry_fit` instead of hand-wiring two near-identical history
transitions inline.
The primary runtime implementation now also resolves geometry-fit constraint
parameter/control name mapping plus live constraint-state reads through shared
helpers in `ra_sim.gui.geometry_fit` instead of keeping those readers inline in
`runtime_impl.py`.
The primary runtime implementation now also resolves geometry-fit constraint
parameter domains plus default window/pull calculations through shared helpers
in `ra_sim.gui.geometry_fit` instead of keeping those calculations inline in
`runtime_impl.py`.

### How We Know It Is Done

- `main.py`, `ra_sim.gui.app`, and `ra_sim.gui.runtime` are now import-safe
  entry boundaries instead of launch-time monoliths.
- `ra_sim.gui.runtime` is now a thin wrapper, and
  `ra_sim/gui/_runtime/runtime_impl.py` is now treated as an accepted
  integration shell rather than a target for open-ended line-count reduction.
- `ra_sim.gui.state`, `ra_sim.gui.controllers`, and `ra_sim.gui.views` now own
  meaningful shared state, workflow orchestration, and Tk construction
  boundaries.
- `ra_sim.config` is now the canonical and only documented config helper
  surface.
- New refactor edits now require a concrete reliability, testability,
  duplicated-logic removal, bug-risk reduction, or feature-delivery reason.
- Once those conditions hold, remaining changes are targeted maintenance, not
  another broad refactor tranche.

### What Is Already Done

- Root launch path is now packaged.
  - `main.py` is a thin compatibility wrapper around `ra_sim.launcher`.
  - `main.py` has explicit import-safety coverage to keep launcher delegation
    lazy and avoid unintended GUI-module loading during root entrypoint import.
  - `ra_sim.cli` launches the GUI via package code, not a root-script monolith.
- Package import safety improved.
  - `ra_sim.gui.app` is now an import-safe entrypoint that lazy-loads
    `ra_sim.gui.runtime`.
  - `ra_sim.gui.app` now has direct import-safety coverage ensuring runtime
    bootstrap is lazy and `main()` argument forwarding stays centralized.
  - `ra_sim.gui.app` now also exposes a guarded lazy attribute surface via
    stable local exports, dunder rejection, and a lazy `__dir__` contract that
    mirrors the runtime shim boundary.
  - `ra_sim.gui.runtime` is now also an import-safe compatibility wrapper that
    lazy-loads `ra_sim/gui/_runtime/runtime_impl.py`.
  - `ra_sim.gui.runtime` now also exposes an explicit lazy attribute surface via
    stable local exports, guarded dunder behavior, and a `__dir__` contract that
    avoids eager implementation import.
  - `tests/test_import_smoke.py` now covers `ra_sim.gui.runtime` directly.
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
  - Geometry-fit undo snapshot capture, runtime restore/redraw, and undo/redo
    transition helpers now also live in `ra_sim.gui.geometry_fit`.
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
- Selected-peak refresh / restore cleanup has advanced further.
  - The remaining selected-peak update-cycle refresh and restored-target
    application callbacks now also assemble through one shared helper in
    `ra_sim.gui.peak_selection`.
  - The selected-peak bootstrap bundle now also carries that maintenance
    callback surface.
  - `runtime.py` no longer keeps inline selected-peak overlay refresh or
    restored-target/HKL-control refresh call sites for that workflow.
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
- Geometry-fit fit-history / manual-pick control cleanup has advanced further.
  - The geometry-tool action control cluster now assembles through one shared
    helper in `ra_sim.gui.bootstrap`.
  - The remaining fit-history button-state and manual-pick label/mode/clear
    runtime callbacks now also assemble through one shared helper in
    `ra_sim.gui.geometry_fit`.
  - `runtime.py` no longer wires that control cluster directly or keeps
    standalone inline implementations for those callback surfaces.
- Geometry-interaction runtime/bootstrap cleanup has advanced further.
  - The selected-peak / HKL lookup / manual-geometry / geometry-tool action
    runtime assembly now also flows through the import-safe helper module
    `ra_sim.gui.runtime_geometry_interaction`.
  - `runtime_impl.py` no longer calls those geometry-interaction bootstrap
    helpers directly.
  - Direct helper tests now cover that assembly instead of AST assertions
    against the internal runtime implementation.
- Fit/analysis runtime/bootstrap cleanup has advanced further.
  - The Bragg-Qr pruning/control wiring, integration-range update runtime
    assembly, and geometry-fit action runtime assembly now also flow through
    the import-safe helper module `ra_sim.gui.runtime_fit_analysis`.
  - `runtime_impl.py` no longer calls those fit/analysis bootstrap helpers
    directly.
  - Direct helper tests now cover that assembly instead of AST assertions
    against the internal runtime implementation.
- Geometry-preview runtime/bootstrap cleanup has advanced further.
  - The geometry Q-group runtime assembly plus the cross-feature canvas
    interaction runtime assembly now also flow through the import-safe helper
    module `ra_sim.gui.runtime_geometry_preview`.
  - `runtime_impl.py` no longer calls those geometry-preview bootstrap helpers
    directly or wires the raw canvas event hooks inline.
  - Direct helper tests now cover that assembly instead of AST assertions
    against the internal runtime implementation.
- Geometry-fit runtime/bootstrap cleanup has advanced further.
  - The geometry-fit runtime value callback assembly, manual-dataset/config
    factory assembly, and geometry-fit action runtime assembly now also flow
    through the import-safe helper module `ra_sim.gui.runtime_geometry_fit`.
  - `runtime_impl.py` no longer stages that geometry-fit runtime assembly
    block directly.
  - Direct helper tests now cover that assembly instead of AST assertions
    against the internal runtime implementation.
- Qr-cylinder overlay runtime/bootstrap cleanup has advanced further.
  - The active-entry factory, render-config factory, and bound overlay
    runtime/toggle assembly now also flow through the import-safe helper
    module `ra_sim.gui.runtime_qr_cylinder_overlay`.
  - `runtime_impl.py` no longer stages that overlay assembly block directly.
  - Direct helper tests now cover that assembly through the extracted helper
    seam.
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
  - The pruning control-cluster default normalization, control construction,
    and live solve-q / prune-bias trace hookup now also assemble through one
    shared helper in `ra_sim.gui.bootstrap`.
  - `ra_sim.gui.structure_factor_pruning` now also owns the helper that
    reapplies normalized pruning / solve-q defaults back to the live GUI vars.
  - `runtime.py` no longer builds that pruning control cluster or binds those
    trace callbacks inline.
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
  - `ra_sim.gui.background_theta` now also owns the live runtime value readers
    plus sync/apply callback bundle for theta metadata and fit-background
    selection.
  - `ra_sim.gui.bootstrap` now also assembles that runtime bundle instead of
    leaving `runtime.py` to wire those callbacks inline.
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
    callback values plus a few live value-source and default-reset call sites
    around that workflow.
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
  - `ra_sim.gui.geometry_fit` now also owns the shared manual-pair dataset
    bindings structure/factory plus the live runtime-config factory used during
    geometry-fit preparation.
  - `ra_sim.gui.geometry_fit` now also owns the live action-bindings factory
    and zero-arg action callback helper used by the runtime fit button.
  - `ra_sim.gui.runtime` no longer rebuilds that geometry-fit action bindings
    bundle inline on each click or threads the manual-pair dataset/value-source
    bundle inline; it now keeps one bound manual-dataset factory, one shared
    runtime-config factory, and the remaining Tk-side control wiring around the
    extracted helper.
- Geometry-fit manual-pair preview/action cleanup has advanced.
  - `ra_sim.gui.manual_geometry` now also owns the bound runtime callback
    bundle for current-pair rendering, Qr/Qz group toggle selection, manual
    point placement, preview refresh, and pick-session cancelation.
  - `ra_sim.gui.manual_geometry` now also owns the bound runtime cache/display
    callback bundle for manual-pick cache-signature assembly, cache reuse,
    and initial-pair overlay display assembly.
  - `ra_sim.gui.manual_geometry` now also owns the bound runtime projection
    callback bundle for caked/raw view selection, manual-pair entry
    projection, simulated-peak projection, candidate grouping, and lookup
    assembly.
  - the manual-geometry runtime callback bundle and the geometry-tool action
    callback bundle now also boot through shared helpers in
    `ra_sim.gui.bootstrap`.
  - the manual-geometry cache/display callback bundle now also boots through
    one shared helper in `ra_sim.gui.bootstrap`.
  - the manual-geometry projection callback bundle now also boots through one
    shared helper in `ra_sim.gui.bootstrap`.
  - `ra_sim.gui.runtime` no longer constructs the manual-pair cache-signature,
    cache-reuse, and initial-pair overlay helpers directly inline.
  - `ra_sim.gui.runtime` no longer constructs the manual-pair live
    view/projection helpers directly inline.
  - `ra_sim.gui.runtime` no longer constructs those manual-geometry /
    geometry-tool callback bundles directly inline; it now keeps bound
    bootstrap results that are threaded through canvas interaction,
    geometry-tool actions, background refresh, GUI-state restore, and the
    geometry-fit manual-pair workflow.
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
  - the workspace background action/file controls plus the backend-debug
    control cluster now also boot through one shared helper in
    `ra_sim.gui.bootstrap`.
  - `ra_sim.gui.runtime` now keeps only one bound background callback bundle
    plus two late-bound status-refresh helpers that are reused by the
    remaining geometry-fit/manual-geometry/background-theta call sites.
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
- The broad GUI runtime-state extraction phase is effectively done.
- The public `ra_sim.gui.runtime` import/startup boundary is now also
  stabilized.
- The remaining runtime-implementation work is now lower-ROI integration glue
  rather than another high-value extraction wave.
- The strategy from here should be to cap open-ended runtime rewiring,
  prioritize import/startup safety, test cleanup, and config unification, and
  only keep refactoring runtime call sites when they buy a concrete win.

## Workstreams

### 1. GUI Runtime Decomposition

Status: Completed (targeted follow-up only)

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
  - `ra_sim.gui.integration_range_drag` now owns the remaining runtime slider/
    entry callbacks, debounced redraw scheduling, and analysis-toggle action
    handlers used by that workflow.
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
- `ra_sim.gui.runtime` and `ra_sim.gui.app` no longer hand-maintain duplicate
  lazy wrapper behavior.
  - `ra_sim.gui.lazy_runtime` now owns their shared `main` forwarding,
    `write_excel` attribute surface, guarded dunder handling, lazy `__dir__`,
    and failure-safe path loading for `runtime_impl.py`.
- `ra_sim.gui.background_manager` now also owns the in-place lazy background
  cache update wrappers used by the main runtime implementation.
  - `ra_sim.gui._runtime.runtime_impl` no longer hand-applies the cache-list
    payloads returned by per-index load/current background reads.
- `ra_sim.gui.background_manager` now also owns the initial background-cache
  boot helper plus the shared background-runtime normalization helper.
  - `ra_sim.gui._runtime.runtime_impl` no longer initializes that cache or
    normalizes background runtime/file-path state inline.
- `ra_sim.gui.background_manager` now also owns the background display-control
  defaults and live background range/transparency helpers.
  - `ra_sim.gui._runtime.runtime_impl` no longer keeps that background display
    limits/default-refresh workflow inline.

What is left:

- No further GUI runtime-state extraction blockers remain.
- `ra_sim.gui.runtime` is now a thin import-safe wrapper, while the remaining
  large integration module lives in `ra_sim/gui/_runtime/runtime_impl.py`.
- Most of the remaining inline code there is now integration glue rather than
  high-value extractable feature logic.
- The remaining structure-model, structure-factor-pruning, Bragg-Qr,
  geometry-fit selector, background, selected-peak, manual-pair, and
  integration-range runtime code is now mostly thin delegate wiring,
  value-source readers, default-reset helpers, and Tk-side coordination around
  the extracted modules.
- Further runtime decomposition should now be opt-in, not the default next
  task.
- A runtime refactor should only be taken when it improves import/startup
  safety, materially improves testability, removes duplicated behavior used by
  multiple workflows, simplifies a bug-prone flow, or directly unblocks
  feature work.
- Thin value-source rewiring or callback extraction done only to shrink
  the internal runtime implementation is now out of scope by default.
- As of 2026-03-29, the next refactor step is complete: no open-ended runtime
  decomposition is planned, and new work is scoped to feature-blocking, duplicated,
  or testability/reliability-driven runtime edits.

Why it matters:

- The highest-risk runtime-state ownership gap is now closed.
- The remaining runtime work has diminishing returns if it is pursued for
  purity alone.
- Treating `runtime.py` as an integration shell preserves the earlier
  extraction wins without spending more days on low-payoff plumbing.

Definition of done:

- `ra_sim.gui.runtime` remains an import-safe public wrapper.
- The internal runtime implementation is accepted as a
  composition/integration layer rather than a target for open-ended
  line-count reduction.
- Remaining runtime edits are justified by concrete reliability,
  maintainability, testability, or feature-delivery wins.
- Feature state is grouped into explicit structures instead of scattered module
  globals.

### 2. Controllers / State / Views Migration

Status: Completed (targeted follow-up only)

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
- `controllers.py` now also owns a shared Tk `after`-token cancellation helper,
  which is now used by manual-geometry and runtime update-scheduling paths.
- `controllers.py` now also owns shared timer-cancel coverage via
  `test_gui_controllers` to guard behavior when Tk callbacks are missing or
  failing.
- `app.py` now also uses the shared Tk timer-cancel helper during geometry-fit
  undo restore to clear pending update callbacks without inline try/except logic.
- `test_gui_app_helpers` now also verifies the geometry-fit undo restore path uses
  the shared timer-cancel helper before rerunning geometry updates.
- `app.py` now delegates geometry-fit undo restore orchestration through
  `gui_geometry_fit.restore_runtime_geometry_fit_undo_state`, consolidating it with
  the same shared orchestration path used by the primary runtime implementation.
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
- `ra_sim.gui.geometry_fit` now also owns a shared lazy undo-restore callback
  builder used by both `ra_sim.gui.app` and `ra_sim.gui._runtime.runtime_impl`.
  - Geometry-fit undo restore no longer hand-assembles duplicate runtime glue
    across those two entry surfaces.
- `ra_sim.gui.app` now also reads geometry-fit UI params, selected fit-variable
  names, and undo-restore var-map state through the same shared geometry-fit
  value-callback bundle used by the primary runtime implementation.
- `ra_sim.gui.geometry_fit` now also owns a shared runtime history-callback
  builder for geometry-fit undo/redo transitions.
  - `ra_sim.gui._runtime.runtime_impl` no longer hand-assembles separate
    controller/history wiring for those two callbacks.
- `ra_sim.gui.geometry_fit` now also owns shared helpers for geometry-fit
  constraint parameter/control name mapping and live constraint-state reads.
  - `ra_sim.gui._runtime.runtime_impl` no longer keeps those normalization
    helpers inline.
- `ra_sim.gui.geometry_fit` now also owns shared helpers for geometry-fit
  constraint parameter-domain lookup plus default window/pull calculation.
  - `ra_sim.gui._runtime.runtime_impl` no longer keeps those defaulting
    calculations inline.

What is left:

- No broad controller/state/view migration blockers remain.
- The `state` / `controllers` / `views` boundary is now real enough to count as
  the default shared app structure.
- Future movement into those modules should be driven by duplicated workflow
  simplification, import/startup safety, test-friction reduction, or direct
  feature work rather than by forcing every remaining callback through the new
  boundary.
- Thin one-off Tk adapters can remain in the internal runtime implementation
  when moving them would be mostly mechanical.

Why it matters:

- These modules should become the default home for shared stateful behavior and
  reusable workflow orchestration.
- They do not need to absorb every last adapter function to provide a real,
  maintainable app structure.

Definition of done:

- `state.py` owns meaningful GUI state structures.
- `controllers.py` owns user-action orchestration and workflow transitions.
- `views.py` owns actual Tk construction helpers beyond a minimal root factory.
- Remaining thin adapters are accepted as integration glue when moving them
  would be mostly mechanical.

### 3. Compatibility Cleanup

Status: Completed

What is done:

- `main.py` is now a compatibility wrapper.
- `ra_sim.gui.app.main` is the canonical package GUI entrypoint.
- `ra_sim.gui.main_app` was removed once tests/docs no longer depended on it.

What is left:

- No package-level GUI entrypoint compatibility blockers remain.
- Follow-up here is now limited to documentation clarity and any future
  external deprecation communication.

Why it matters:

- Compatibility wrappers are useful during migration, but they create confusion
  once the real launch path has stabilized.
- The codebase should eventually converge on one canonical GUI entrypoint.

Definition of done:

- One supported GUI entrypoint is documented and used by tests/docs.
- Compatibility shims are either removed or explicitly marked as permanent.

### 4. Config Unification

Status: Completed

What is done:

- `ra_sim.config.loader` exists as a newer cached config loader.
- `ra_sim.config.loader` now also owns the canonical file-path, directory,
  materials, and instrument-config loading helpers, including the Windows YAML
  path fallback and the active-config cache.
- `ra_sim.config.loader` now also owns `get_temp_dir()`, so the full
  packaged config helper surface lives under `ra_sim.config`.
- Several packaged call sites now import config helpers from `ra_sim.config`
  directly instead of reaching through the old shim surface.
- The remaining non-compat imports were migrated off `ra_sim.path_config`
  before the shim was removed.
- The obsolete `ra_sim.path_config` compatibility shim and its dedicated
  compatibility-only test module are now gone.
- The remaining useful regression coverage for config switching/missing-key
  behavior now lives in `tests/test_config_loader.py`.

What is left:

- No config-surface compatibility blocker remains.
- Follow-up here is now limited to doc clarity and future config-surface
  feature work on top of `ra_sim.config`.

Why it matters:

- Duplicated config systems create drift, duplicate tests, and ambiguous
  extension points.
- This is a structural cleanup item, not just style cleanup.

Definition of done:

- One config loading system is canonical.
- The other becomes a short-lived shim or is removed.
- Duplicate configuration data/files are eliminated where possible.

### 5. Test Architecture Cleanup

Status: Completed (narrow structural compile/import checks only)

What is done:

- `ra_sim.gui.runtime` is now an import-safe public wrapper around the internal
  GUI implementation module.
- `tests/test_import_smoke.py` now covers `ra_sim.gui.runtime` directly.
- The `ra_sim.gui.app` helper and sim-signature tests now import the live
  module directly instead of extracting functions from source text.
- The hBN fitter bundle-export test now imports `hbn_fitter.fitter` directly
  and exercises a shared bundle-payload helper instead of extracting
  `save_bundle()` from source text.
- The background runtime/bootstrap assembly now flows through an import-safe
  helper module, and the matching background runtime-shape checks now use
  direct helper tests instead of AST assertions against `runtime_impl.py`.
- The selected-peak / HKL lookup / manual-geometry / geometry-tool action
  runtime/bootstrap assembly now also flows through an import-safe helper
  module, and the matching regression coverage now uses direct helper tests
  instead of AST assertions against `runtime_impl.py`.
- The Bragg-Qr pruning/control wiring, integration-range update runtime
  assembly, and geometry-fit action runtime assembly now also flow through an
  import-safe helper module, and the matching regression coverage now uses
  direct helper tests instead of AST assertions against `runtime_impl.py`.
- The geometry Q-group runtime assembly plus the cross-feature canvas
  interaction runtime assembly now also flow through an import-safe helper
  module, and the matching regression coverage now uses direct helper tests
  instead of AST assertions against `runtime_impl.py`.
- The geometry-fit runtime value callback assembly, manual-dataset/config
  factory assembly, and geometry-fit action runtime assembly now also flow
  through an import-safe helper module, and the matching regression coverage
  now uses direct helper tests instead of AST assertions against
  `runtime_impl.py`.
- The remaining GUI-runtime AST checks around pruning-default wiring and
  selected-peak maintenance wiring were replaced with direct helper tests
  against `ra_sim.gui.runtime_fit_analysis` and
  `ra_sim.gui.runtime_geometry_interaction`.
- `tests/test_fresnel_calls.py` now exercises the live diffraction helper call
  paths through Python-callable `numba` entrypoints instead of parsing
  `ra_sim/simulation/diffraction.py`.
- `tests/test_hbn_geometry_mapping.py` now exercises the packaged GUI hBN
  conversion helper directly instead of reading `ra_sim/gui/app.py` source.
- Several tests were moved away from runtime-heavy extraction and toward direct
  module coverage.
- The old `main.py` AST lock-in is no longer the primary issue it once was.

What is left:

- No AST-based tests remain in `tests/`.
- The remaining structural test surface is limited to narrow compile/import
  checks where the contract itself is still the point of the test.

Why it matters:

- AST-based tests are brittle and reward preserving file shape rather than
  preserving behavior.
- The public runtime import boundary is now fixed, so the remaining test work
  can focus on reducing file-shape coupling instead of compensating for startup
  side effects.

Definition of done:

- Tests primarily import stable modules and validate behavior directly.
- Remaining structural tests are narrow compile/import contracts, not
  source-shape stand-ins for behavior.

### 6. Repository Cleanup

Status: Completed (2026-03-28)

What landed:

- Removed the tracked root artifacts that did not belong in the long-term
  project layout:
  - `ig_graph.sqlite`
  - `ig_graph.sqlite-shm`
  - `ig_graph.sqlite-wal`
  - `session.json`
  - `oneline`
  - `et --hard a485e65`
  - the legacy root `hbn.py`
- Added root-level ignore rules so those local artifacts do not get recommitted.

Why it matters:

- Root clutter makes packaging, onboarding, and repository intent harder to
  understand.
- It also obscures what is part of the product versus what is an accident or a
  local artifact.

Definition of done:

- The root contains packaging, docs, config, tests, maintained wrappers, and
  intentional top-level scripts only.

### 7. Unrelated Simulation Test Debt

Status: Completed (2026-03-28)

What landed:

- Restored the small Python cache/stat compatibility surface expected by
  `tests/test_source_template_cache.py` in `ra_sim.simulation.diffraction`.
- The module now again exposes `_PHASE_SPACE_CACHE`,
  `_SOURCE_TEMPLATE_CACHE`, `_Q_VECTOR_CACHE`, and
  `get_last_process_peaks_safe_stats()`.
- The direct regression tests for that surface now pass again.

Why it matters:

- This is not part of the GUI migration, but it is still active technical debt.
- It should be tracked separately so it does not get conflated with GUI
  refactor progress.

Definition of done:

- The test is either rewritten around supported behavior or the missing cache
  surface is intentionally restored and documented.

## Recommended Order

### Phase A: Stabilize Import / Startup Boundaries

Goal:

- Make GUI startup behavior easier to reason about and reduce import-time
  coupling.

Tasks:

- Identify `ra_sim.gui.runtime` work that still happens on import instead of
  at launch/call time.
- Move launch-only initialization behind explicit call boundaries where
  practical.
- Make the public runtime module safe to import without launching the GUI.

Why first:

- This yields concrete reliability and testability gains.
- It is higher ROI than more line-count-driven runtime slicing.

Current status:

- Landed for the public runtime boundary.
- `ra_sim.gui.runtime` now lazy-loads the heavy GUI implementation from
  `ra_sim/gui/_runtime/runtime_impl.py`.

### Phase B: Finish Test Cleanup

Goal:

- Make tests target imported modules and stable behavior.

Tasks:

- Replace remaining AST-extraction tests where practical.
- Add or extend direct module tests around GUI bootstrap and controller/view
  behavior.
- Keep import-smoke coverage moving once startup side effects are contained.

Why second:

- Better tests are the main force multiplier for the next phase of feature
  work.
- This is now a better use of time than more mechanical callback extraction.

### Phase C: Unify Config Loading

Goal:

- Eliminate the dual config-system split.

Tasks:

- Migrate packaged call sites from `ra_sim.path_config` to `ra_sim.config`.
- Keep `ra_sim.path_config` as a compatibility shim only as long as needed.
- Remove duplicated config responsibilities.

Why third:

- This is repo-wide technical debt with a clear duplication cost.
- It no longer needs to wait on more GUI rewiring.

Current status:

- Landed for the packaged codebase.
- `ra_sim.config` is now the canonical and only supported helper surface.

### Phase D: Targeted Runtime / Boundary Cleanup Only Where It Pays Off

Goal:

- Treat the internal runtime implementation as an integration shell and only
  refactor the remaining inline glue when there is concrete payoff.

Tasks:

- Only move runtime-owned logic when it improves import/startup safety,
  materially improves testability, removes duplicated behavior, simplifies a
  bug-prone workflow, or directly unblocks feature work.
- Prefer deleting duplicated cross-feature logic over relocating one-off
  value-source plumbing.
- Leave thin Tk wiring in the internal runtime implementation when extracting
  it would be mostly mechanical.

Why fourth:

- The remaining runtime work is now lower-ROI glue.
- This preserves momentum without abandoning useful cleanup opportunities.

### Phase E: Finish Compatibility Cleanup

Goal:

- Converge on a single documented GUI entrypoint once the higher-leverage work
  stabilizes.

Tasks:

- Remove obsolete package-level GUI compatibility aliases once the codebase no
  longer depends on them.
- Update docs/tests to reflect the canonical entrypoint.

Why fifth:

- This is easier to finalize after startup/testing/config boundaries are
  clearer.

Current status:

- Landed for the package GUI entrypoint.
- `ra_sim.gui.app.main` is now canonical, and `ra_sim.gui.main_app` has been
  removed.

### Phase F: Clean the Repository Root

Goal:

- Remove stale artifacts and clarify project layout.

Tasks:

- Remove stray local artifacts from version control.
- Review root-level legacy scripts and wrappers for retention or removal.

Why last:

- This is important, but it is lower leverage than finishing the runtime and
  config migration.

Current status:

- Landed on 2026-03-28.
- The tracked root artifacts were removed, the legacy root `hbn.py` was
  deleted, and `.gitignore` now keeps those local files out of version
  control.

## Refactor Guardrails

- Do not take open-ended internal-runtime shrink tasks just to reduce line
  count or move one-off plumbing.
- Prefer work that improves import safety, startup behavior, test coverage,
  config coherence, or active feature delivery.
- If a proposed runtime refactor does not remove duplicated logic, reduce bug
  risk, improve testability, or unblock a concrete workflow, do not do it.
- Treat `state.py`, `controllers.py`, and `views.py` as the place for shared
  logic and reusable boundaries, not as a destination for every thin adapter.

## Suggested Next Concrete Step

The broad refactor now meets its exit criteria. The next concrete step is to
keep it closed out by default:

- keep the runtime/config/entrypoint/root-cleanup refactor closed out and stop
  expanding it by default
- treat the restored diffraction cache/stat surface as compatibility coverage,
  not as a reason to reopen broad simulation-kernel reshaping
- only return to `ra_sim/gui/_runtime/runtime_impl.py` for specific bug-prone,
  duplicated, or feature-blocking workflows that still justify the move
- let future cleanup be driven by active product work rather than another
  open-ended structural sweep

Operationally, the standing rule is:

- add/maintain the doc-level gate: no runtime change lands without a concrete
  reliability, testability, or feature-delivery reason.

Immediate checklist after the strategy pivot:

- keep import-smoke and direct helper coverage green as feature work lands
- make targeted runtime or simulation changes only when they buy testability,
  reliability, or concrete workflow simplification
- keep `ra_sim.config` as the only documented config helper surface
- keep GUI refactor work focused on concrete reliability or feature-delivery
  wins
- defer thin callback/value-source rewiring unless it directly advances one of
  those items

## Tracking Notes

When updating this file:

- move completed items from “what is left” into “what is done”
- keep the “why it matters” sections intact unless the rationale changes
- update the recommended order only when the dependency structure changes
- do not add new runtime-only cleanup tasks unless they clear the refactor
  guardrails above
- keep GUI migration work separate from unrelated simulation-layer test debt
