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
  - `ra_sim.gui.runtime` now delegates the Bragg-Qr / SF-pruning control
    workflow through that module and keeps only thin delegate wrappers plus
    current-value helpers inline
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
  - `ra_sim.gui.runtime` now keeps only the shared runtime-binding factory
    plus thin manager/overlay call sites for that workflow
- Geometry-fit Qr/Qz selector workflow extraction has also advanced:
  - selector line/status formatting, window refresh, checkbox/bulk include-
    exclude side effects, update-listed-peaks request flow, and save/load
    dialog workflow now live in
    `ra_sim.gui.geometry_q_group_manager`
  - `ra_sim.gui.runtime` now delegates that selector workflow through the
    extracted module and keeps only thin delegate wrappers plus window
    lifecycle wiring inline
- Background-file workflow extraction has also advanced:
  - background-file state transition, file-dialog initial-dir selection,
    background status refresh, and the post-load/post-switch redraw/reset
    sequencing now live in `ra_sim.gui.background_manager`
  - `ra_sim.gui.runtime` now delegates that workflow through the extracted
    module and keeps only the file-dialog plus progress/error text inline
- HKL lookup / selected-peak workflow extraction has also advanced:
  - HKL lookup parsing, selected-peak summary text, degenerate-HKL lookup,
    selection-by-index/HKL, clear-selection state transitions, HKL-pick
    toggle / raw-image click selection workflow, and selected-peak
    Bragg/Ewald intersection analysis now live in
    `ra_sim.gui.peak_selection`
  - `ra_sim.gui.runtime` now delegates that workflow through the extracted
    module and keeps only cross-feature click-mode dispatch plus current
    control-value delegation inline
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
- The remaining Bragg-Qr runtime code is now mostly the shared runtime-binding
  factory plus thin manager/overlay call sites around the extracted
  controller/view modules.
- The remaining geometry-fit Qr/Qz selector runtime code is now mostly thin
  delegate wrappers and window lifecycle wiring around the extracted
  manager/view helpers.
- The remaining Bragg-Qr manager runtime code is now mostly the shared
  binding-factory wiring used by the live filter pipeline and HKL lookup
  controls around the extracted manager helpers.
- The remaining background runtime code is now mostly file-dialog plumbing and
  progress/error text around the extracted background manager.
- The remaining selected-peak runtime code is now mostly cross-feature
  event-dispatch and click-mode arbitration around the extracted peak
  selection helpers.
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
