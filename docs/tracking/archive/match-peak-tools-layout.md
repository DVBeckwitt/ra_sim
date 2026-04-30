# Match peak tools layout

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-30

## Summary

The Match-tab peak tools were crowded into one row. The `Drag Move Placed
Peaks` checkbox could be hidden after the auto-search radius slider and adjacent
Qr-set controls, which made the move tool hard to discover. The auto-search
slider also duplicated a low-level tuning option in a workflow that should keep
the common pick, move, remove, and placement actions visible.

## Current state

Resolved. The peak tool controls are now grouped by task:

- Pick/move/remove modes live together, with `Drag Move Placed Peaks` visible
  beside the manual pick button.
- The click-removal toggle is named `Click Remove Placed Peaks` to distinguish
  it from the Qr-set removal command.
- Qr-set actions live on their own row.
- Placement cleanup and import/export actions are separated from pick-mode
  toggles.
- The auto-search radius slider is no longer shown in the peak tools.

The underlying auto-add search radius runtime value remains available to the
auto-add implementation; this patch only removes the exposed Match-tab slider.

## Validation

- 2026-04-30: `python -S -m py_compile ra_sim/gui/views.py ra_sim/gui/manual_geometry.py tests/test_gui_views.py`
  passed.
- 2026-04-30: `python -m pytest tests/test_gui_views.py -ra`
  passed, `63 passed`.
- 2026-04-30: `python -m pytest tests/test_gui_bootstrap.py::test_build_runtime_geometry_tool_action_controls_bootstrap_wires_add_all -ra`
  passed.
- 2026-04-30: `python -m ruff check ra_sim/gui/views.py ra_sim/gui/manual_geometry.py tests/test_gui_views.py`
  passed.

## Links

- Runtime path: [ra_sim/gui/views.py](../../../ra_sim/gui/views.py)
- Status text path: [ra_sim/gui/manual_geometry.py](../../../ra_sim/gui/manual_geometry.py)
- Test path: [tests/test_gui_views.py](../../../tests/test_gui_views.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
