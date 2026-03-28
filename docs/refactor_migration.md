# Refactor Migration Notes

See also: `docs/refactor_plan.md` for the active tracking plan covering what is
done, what is left, and the recommended order for finishing the refactor.

## Scope

This document summarizes the maintainability refactor delivered for RA-SIM while preserving user-facing behavior.

## Compatibility Guarantees

- `python main.py` still launches the GUI.
- `python -m ra_sim gui|simulate|hbn-fit` is unchanged.
- `ra-sim` console script now points to `ra_sim.cli:main`.
- Existing config access helpers remain:
  - `ra_sim.path_config.get_path`
  - `ra_sim.path_config.get_dir`
  - `ra_sim.path_config.get_instrument_config`
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
- Manual geometry was split out of the runtime monolith in stages:
  - pure helpers and serialization moved into `ra_sim.gui.manual_geometry`
  - session/preview/manual-pick orchestration moved into
    `ra_sim.gui.manual_geometry`
  - backing manual-geometry state and undo/session mutations now flow through
    `ra_sim.gui.state` and `ra_sim.gui.controllers`
- Geometry-fit history and Qr/Qz selector state have also started moving out of
  runtime-owned globals:
  - geometry-fit undo/redo history now uses shared GUI state
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
  - `ra_sim.gui.runtime` now delegates those paths through thin wrappers
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
    callback values plus a few live call sites around that workflow
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
    normalization, Bragg-Qr entry/L-value construction, and active
    Qr-cylinder overlay entry derivation used by that workflow
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    refresh/open callback helpers used by the live filter pipeline and HKL
    lookup controls
  - `ra_sim.gui.bragg_qr_manager` now also owns the zero-arg runtime
    binding-factory helper used by the live filter pipeline, HKL lookup
    controls, and manager callbacks
  - `ra_sim.gui.qr_cylinder_overlay` now also owns the analytic Qr-cylinder
    overlay render config, cache-signature, path-construction helpers, and
    the runtime binding/refresh/toggle helpers used by the live detector/
    caked overlay workflow
  - `ra_sim.gui.runtime` now keeps only one bound Bragg-Qr runtime factory
    value, thin active-entry/render-config value sources, and the remaining
    live manager/overlay call sites for that workflow
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
  - `ra_sim.gui.runtime` now keeps only one bound background callback bundle
    plus thin status-refresh and control-wiring call sites around that
    workflow
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
- HKL lookup / selected-peak workflow extraction has also advanced:
  - HKL lookup parsing, selected-peak summary text, degenerate-HKL lookup,
    selection-by-index/HKL, clear-selection state transitions, HKL-pick
    toggle / raw-image click selection workflow, and selected-peak
    Bragg/Ewald intersection analysis now live in
    `ra_sim.gui.peak_selection`
  - the selected-peak config builders and the ideal-center probe helper used
    by raw-image HKL picking plus the runtime config-factory helpers that
    resolve live GUI values now also live in `ra_sim.gui.peak_selection`
  - `ra_sim.gui.peak_selection` now also owns the runtime binding/callback
    bundle for HKL-pick button labels, mode toggles, selected-peak refresh,
    HKL-control selection, raw-image click selection, and the selected-peak
    Bragg/Ewald action
  - `ra_sim.gui.runtime` now keeps only cross-feature click-mode dispatch,
    thin value-source wiring, and the bound peak-selection runtime wiring
    around that workflow
- Canvas interaction workflow extraction has also advanced:
  - top-level raw-image canvas click/drag arbitration between
    manual-geometry placement, preview exclusion, HKL picking, and
    integration-range dragging now lives in `ra_sim.gui.canvas_interactions`
  - `ra_sim.gui.runtime` now keeps only one bound canvas callback bundle plus
    thin event-hook wiring around that workflow
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

- `ra_sim.gui.runtime` is still the largest remaining integration monolith.
- The remaining runtime work is now the cross-feature workflow glue and the
  remaining controller/view orchestration that still live inline in
  `ra_sim.gui.runtime`, not long-lived GUI runtime-state ownership or the
  extracted structure-model rebuild path.
- The remaining structure-model runtime code is now mostly thin delegate
  wrappers, progress-label wiring, and control-var rebuild callbacks.
- The remaining structure-factor-pruning runtime code is now mostly bound
  callback values plus a few live call sites around the extracted pruning
  module.
- The remaining Bragg-Qr runtime code is now mostly one bound runtime-factory
  value plus thin active-entry/render-config value sources and a few
  manager/overlay call sites around the extracted controller/view modules.
- The remaining geometry-fit Qr/Qz selector runtime code is now mostly one
  bound factory/callback bundle plus thin fit-preview/cached-hit value
  sources, live-preview availability/fallback simulation orchestration,
  image-shape/display-coordinate value plumbing, and a couple of delegated
  call sites around the extracted manager/view helpers.
- The remaining Bragg-Qr manager runtime code is now mostly the bound
  factory wiring used by the live filter pipeline and HKL lookup controls
  around the extracted manager helpers.
- The remaining background runtime code is now mostly one bound callback
  bundle plus thin status-refresh and control-wiring call sites around the
  extracted background manager.
- The remaining selected-peak runtime code is now mostly thin value-source
  wiring plus a few call sites around the extracted peak-selection and
  canvas-interaction modules.
- The remaining integration-range drag runtime code is now mostly the
  bound runtime-factory/canvas wiring plus thin integration-region visual
  call sites around the extracted drag and canvas-interaction modules.
- The next bounded GUI targets are the remaining cross-feature workflow/state
  transitions that still bypass the newer state/controller/view modules.
- `ra_sim.path_config` and `ra_sim.config.loader` still overlap and need
  eventual unification.
- `ra_sim.gui.main_app.main` still exists as a compatibility alias pending
  final entrypoint cleanup.

## Configuration Changes

- Added `RA_SIM_CONFIG_DIR` override to select an alternate configuration directory.
- Added `config/file_paths.example.yaml` as a portable template.
- Consolidated index-of-refraction parameter source to `config/ior_params.yaml`.

## Deprecation Timeline

### Current release

- Legacy positional simulation kernel calls remain available and are wrapped by typed APIs.
- `ra_sim.gui.main_app.main` remains available as a compatibility alias.
- Root utility scripts remain as wrappers around `ra_sim.tools`.

### Next release target

- Prefer typed simulation requests (`SimulationRequest`) internally and in new extension points.
- Keep wrappers but emit deprecation warnings for direct internal positional kernel usage.

### Future major release

- Remove legacy internal wrappers that bypass typed simulation models.
- Remove compatibility aliases that are no longer used by project tests or docs.
