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
  - `ra_sim.gui.background_theta`
  - `ra_sim.gui.geometry_fit`
  - `ra_sim.gui.geometry_overlay`
  - `ra_sim.gui.manual_geometry`
  - `ra_sim.gui.overlays`
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
- Direct tests were added for extracted controller/state behavior.
  - this now includes preview-state controller coverage, Bragg-Qr controller
    coverage, and direct Qr/Qz/workspace/Bragg/hBN/constraints/
    background-theta/background-debug view helper coverage

## Remaining Migration Focus

- `ra_sim.gui.runtime` is still the largest remaining integration monolith.
- `ra_sim.gui.views` is now active for the Qr/Qz selector, but other Tk-heavy
  surfaces still need the same treatment, especially the remaining widget-heavy
  runtime-owned helpers and cross-feature workflow glue that still assemble
  long-lived Tk references or own mutable GUI state inline.
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
