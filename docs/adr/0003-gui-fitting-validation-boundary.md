# 0003: GUI fitting validation boundary

Status: Accepted

Date: 2026-04-21

## Context

GUI-assisted fitting combines Tk state, detector-space selections, headless fit
reruns, and real experiment files that are not bundled with the repository.
Most contributors and agents can run fast headless checks, while real-data
diagnostics remain opt-in because they require local inputs.

## Decision

GUI and fitting validation is split across fast headless tests, integration
workflow tests, and explicit opt-in real-data diagnostics. The fast tier checks
portable contracts. The integration tier exercises workflow-heavy paths. Local
diagnostics can inspect real saved states and experiment files when available.

## Consequences

Changes to GUI/fitting boundaries should update nearby fast or integration
tests when behavior changes. Real-data diagnostics can support investigation,
but should not become required for default agent checks unless the needed data
is portable and documented.

## Related docs/tests

- [GUI workflow](../gui-workflow.md)
- [Architecture](../architecture.md)
- [Canonical simulation and fitting reference](../simulation_and_fitting.md)
- `tests/test_gui_geometry_fit_workflow.py`
- `tests/test_cli_geometry_fit.py`
