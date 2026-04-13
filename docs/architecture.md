# RA-SIM architecture

This page is the short architectural map for contributors and coding agents. It
points to where major responsibilities live without repeating the full
implementation-level detail from the canonical reference.

See also:

- [docs index](index.md)
- [GUI workflow](gui-workflow.md)
- [Debug and cache guide](debug-and-cache.md)
- [Canonical reference code map](simulation_and_fitting.md#code-map)

## Package Layout

- `ra_sim/simulation/`: forward simulation engine, diffraction kernel, typed requests/responses, detector geometry
- `ra_sim/fitting/`: geometry fitting, mosaic fitting, objective assembly, peak matching
- `ra_sim/fitting/optimization_runtime.py`: shared worker, cache, and process-peaks runtime helpers extracted from the large optimizer
- `ra_sim/gui/`: Tk application, controllers, runtime state, overlays, operator workflows
- `ra_sim/gui/_runtime/live_cache_helpers.py`: cache-inventory and overlay-reset helpers extracted from the main runtime session
- `ra_sim/io/`: OSC readers, file parsing, GUI state persistence
- `ra_sim/config/`: config loading, models, and validation
- `ra_sim/utils/`: optics helpers, stacking-fault support, parallel helpers, generic utilities
- `ra_sim/hbn.py` and `ra_sim/hbn_geometry.py`: hBN calibrant workflow and geometry conversion helpers
- `tests/`: regression coverage for config, simulation, fitting, CLI, and GUI helpers

## System Shape

Typical flow:

1. Config is loaded from `config/` or `RA_SIM_CONFIG_DIR`.
2. The calibrant path estimates geometry and tilt from hBN rings when needed.
3. The main GUI or CLI assembles beam, mosaic, geometry, and material inputs.
4. Simulation code produces detector-space predictions.
5. Fitting code compares predicted and observed peaks or images.
6. GUI/runtime code manages interaction state, cached datasets, and analysis views.

## Where To Edit

- New CLI behavior: `ra_sim/cli.py`
- Config semantics or defaults: `ra_sim/config/` plus versioned templates in `config/`
- Diffraction or detector geometry logic: `ra_sim/simulation/`
- Peak matching or fit objectives: `ra_sim/fitting/`
- Shared fitting worker/cache plumbing: `ra_sim/fitting/optimization_runtime.py`
- GUI interactions and workflow controls: `ra_sim/gui/`
- GUI cache inventory/reset helpers: `ra_sim/gui/_runtime/live_cache_helpers.py`
- hBN calibrant workflow: `ra_sim/hbn.py`, `ra_sim/hbn_geometry.py`, `ra_sim/hbn_fitter/`

## Docs Strategy

- Use this page for package routing and edit targeting.
- Use [gui-workflow.md](gui-workflow.md) for operator flow.
- Use [debug-and-cache.md](debug-and-cache.md) for debug/log behavior.
- Use [simulation_and_fitting.md](simulation_and_fitting.md) when you need exact defaults, equations, or function-level mapping.
