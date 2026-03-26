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

As of 2026-03-25, the refactor has made real progress. The legacy root-script
problem is largely solved, and the main remaining cleanup target is now the
packaged GUI monolith in `ra_sim/gui/runtime.py`, not `main.py` or
`mosaic_profiles.py`.

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
  - `ra_sim.gui.geometry_fit`
  - `ra_sim.gui.geometry_overlay`
  - `ra_sim.gui.manual_geometry`
  - `ra_sim.gui.overlays`
  - `ra_sim.gui.state_io`
- Manual geometry migration advanced substantially.
  - Selection helpers, caked-coordinate helpers, serialization helpers,
    preview/session flow, and the remaining manual-pick/Q-group orchestration
    have been moved into `ra_sim.gui.manual_geometry`.
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
- Bragg Qr manager migration has started.
  - Bragg-Qr selection/index bookkeeping now lives in `ra_sim.gui.state`.
  - Controller helpers now own Bragg-Qr selection mapping and group/L-value
    toggle mutations.
  - `ra_sim.gui.views` now owns the Bragg Qr manager window lifecycle and
    listbox rendering helpers.
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
- Several tests were moved off monolith-coupled runtime behavior and onto
  extracted modules.

### What This Means

- The original goal of turning `main.py` into a wrapper is effectively done.
- The main unfinished work is the final stage of breaking down
  `ra_sim/gui/runtime.py` and turning the GUI scaffolding modules into real
  application structure.

## Workstreams

### 1. GUI Runtime Decomposition

Status: In progress

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

What is left:

- `ra_sim/gui/runtime.py` is still very large and still owns too much mutable
  GUI state.
- The runtime still coordinates too many cross-feature globals and Tk widgets.

Why it matters:

- The codebase still has one oversized integration module that is hard to test,
  reason about, and safely change.
- The remaining state ownership is the main reason the scaffolding modules have
  not become real boundaries yet.

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

What is left:

- The GUI still does not flow broadly through an explicit
  state/controller/view boundary.
- The new controller/state boundary is real for manual geometry, but it is not
  yet the dominant app structure across the rest of the runtime.
- Other Tk-heavy surfaces still need to move behind `views.py` helpers.

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

- Finish removing runtime-owned aliases around manual-geometry state.
- Group more runtime globals into explicit state containers.
- Reduce direct feature coordination in `runtime.py` where controller logic can
  own it instead.
- Keep moving Tk-owned widget references out of runtime and into `views.py`
  state/helpers.

Why first:

- This is the highest-value remaining cleanup item.
- It unlocks the scaffold modules and reduces the risk of further extraction
  work.

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

- build on the new manual-geometry / geometry-fit-history / preview /
  Q-group / Bragg-Qr / hBN-debug / geometry-fit-constraints /
  background-theta-control / workspace-panel / background-debug slices by
  moving the next runtime-owned GUI
  workflows into explicit state + controller + view boundaries
- focus next on the remaining runtime-owned workflow/state coordination in
  `runtime.py`, especially the scattered cross-feature globals and other
  still-inline Tk-heavy helpers that prevent the extracted scaffolding modules
  from becoming the dominant app structure

That is the point where the migration stops being “more helper extraction” and
starts becoming a real architectural finish.

## Tracking Notes

When updating this file:

- move completed items from “what is left” into “what is done”
- keep the “why it matters” sections intact unless the rationale changes
- update the recommended order only when the dependency structure changes
- keep GUI migration work separate from unrelated simulation-layer test debt
