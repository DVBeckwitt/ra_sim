# 0002: Developer command surface

Status: Accepted

Date: 2026-04-21

## Context

RA-SIM has several external tools for local development: pip, ruff, pytest,
mypy, pre-commit, lock refresh, and optional validation helpers. Running those
tools through one Python module keeps command names stable across shells and CI.

## Decision

`python -m ra_sim.dev ...` is the canonical developer command surface. The
installed `ra-sim-dev` script exposes the same entry point after installation.

## Consequences

Developer docs, CI, and agent guidance should prefer `python -m ra_sim.dev`
commands. New dev-only checks should be added there when they need a stable
cross-platform wrapper. Optional reports, such as coverage and package build,
should remain outside `check` unless they become required gates.

## Related docs/tests

- [Development commands](../../README.md#developer-entry-points)
- [Contributing validation](../../CONTRIBUTING.md#validation)
- `tests/test_dev_cli.py`
