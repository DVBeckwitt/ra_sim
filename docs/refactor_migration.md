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
- Direct tests were added for extracted controller/state behavior.

## Remaining Migration Focus

- `ra_sim.gui.runtime` is still the largest remaining integration monolith.
- `ra_sim.gui.views` is still mostly placeholder scaffolding.
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
