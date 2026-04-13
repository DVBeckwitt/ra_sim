# RA-SIM docs

This directory is the short-form map for the repository. Use these pages to
find the right entry point quickly, then drop into the canonical reference when
you need implementation-level detail.

## Start Here

- [README.md](../README.md): install, configuration, CLI entry points, screenshots
- [CONTRIBUTING.md](../CONTRIBUTING.md): development workflow, validation, PR expectations
- [simulation_and_fitting.md](simulation_and_fitting.md): canonical reference for the live pipeline

## Task-Focused Guides

- [gui-workflow.md](gui-workflow.md): operator workflow through calibrant, GUI, and refinement stages
- [architecture.md](architecture.md): package layout, major subsystems, and where to edit
- [debug-and-cache.md](debug-and-cache.md): debug controls, output locations, and cache retention
- [troubleshooting.md](troubleshooting.md): common setup, config, and workflow failures

## Quick Routing

If you need to:

- run or launch the app: start with [README.md](../README.md#usage)
- change simulation/fitting code: read [architecture.md](architecture.md) and then the code map in [simulation_and_fitting.md](simulation_and_fitting.md#code-map)
- understand operator flow: read [gui-workflow.md](gui-workflow.md)
- debug logs or retained caches: read [debug-and-cache.md](debug-and-cache.md)
- fix local setup or path issues: read [troubleshooting.md](troubleshooting.md)

## Canonical Reference

The big reference stays authoritative for names, defaults, equations, and file
paths:

- [simulation_and_fitting.md](simulation_and_fitting.md)

The focused guides in this directory summarize and route. The canonical
reference goes deep.
