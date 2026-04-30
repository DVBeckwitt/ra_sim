# RA-SIM docs

This directory is the short-form map for the repository. Use these pages to
find the right entry point quickly, then drop into the canonical reference when
you need implementation-level detail.

## Start Here

- [README.md](../README.md): install, configuration, CLI entry points, screenshots
- [CONTRIBUTING.md](../CONTRIBUTING.md): development workflow, validation, PR expectations
- [simulation_and_fitting.md](simulation_and_fitting.md): canonical reference for the live pipeline, including the mosaic profile and structure-factor fitting math contracts

## Task-Focused Guides

- [tracking/index.md](tracking/index.md): live work notes for substantial recoveries, investigations, and planned efforts
- [testing-and-validation.md](testing-and-validation.md): test, validation, timing, benchmark, fixture, and automation index
- [gui-workflow.md](gui-workflow.md): operator workflow through calibrant, GUI, and refinement stages
- [hbn-fitter.md](hbn-fitter.md): script-level guide for the hBN calibrant fitter, click snapping, ellipse refinement, and tilt export
- [architecture.md](architecture.md): package layout, major subsystems, and where to edit
- [release-versioning.md](release-versioning.md): package version source of truth and the 1.0 release sequence
- [adr/index.md](adr/index.md): accepted architecture decision records
- [debug-and-cache.md](debug-and-cache.md): debug controls, output locations, and cache retention
- [troubleshooting.md](troubleshooting.md): common setup, config, and workflow failures

## Quick Routing

If you need to:

- run or launch the app: start with [README.md](../README.md#usage)
- change simulation/fitting code: read [architecture.md](architecture.md) and then the code map in [simulation_and_fitting.md](simulation_and_fitting.md#code-map)
- understand stable architecture decisions: read [adr/index.md](adr/index.md)
- understand operator flow: read [gui-workflow.md](gui-workflow.md)
- edit or explain the hBN calibrant fitter: read [hbn-fitter.md](hbn-fitter.md)
- check release version policy: read [release-versioning.md](release-versioning.md)
- inspect live recovery work or substantial investigations: start with [tracking/index.md](tracking/index.md)
- inspect the current fitter project order: start with [tracking/index.md](tracking/index.md)
- implement the selected-pair mosaic profile plan: read [tracking/in-progress/mosaic-fitter.md](tracking/in-progress/mosaic-fitter.md) and [the canonical mosaic section](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- implement the global multi-image structure-factor fitter: read [tracking/planned-features/structure-factor-fitter.md](tracking/planned-features/structure-factor-fitter.md) and [the ordered-structure section](simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
- choose tests, validation tools, timing tools, or benchmarks: read [testing-and-validation.md](testing-and-validation.md)
- debug the geometric fitter saved-state baseline: start with [tracking/index.md](tracking/index.md)
- debug rotated or drifting simulated peak overlays: start with [tracking/index.md](tracking/index.md)
- debug logs or retained caches: read [debug-and-cache.md](debug-and-cache.md)
- fix local setup or path issues: read [troubleshooting.md](troubleshooting.md)

## Canonical Reference

The big reference stays authoritative for names, defaults, equations, and file
paths:

- [simulation_and_fitting.md](simulation_and_fitting.md)

The focused guides in this directory summarize and route. The canonical
reference goes deep.
