# Main figure right-drag pan regression

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-29

## Summary

Right-button drag panning stopped working on the main 2D detector and caked
figures. Wheel zoom still worked. The regression was in canvas interaction
dispatch: a stale left-drag suppression flag could consume the next right-button
press before pan session setup ran.

## Current state

Resolved. `ra_sim/gui/canvas_interactions.py` now normalizes mouse button values
from Matplotlib, Tk, integer, enum, and string sources before routing click,
press, motion, release, and wheel-scroll events. Right-button press starts the
pan session before stale left-click drag suppression can consume it. The stale
suppression flag is cleared for right-button pan and remains limited to the
left-click drag path it was created for.

The fix applies to detector and caked main-figure views. Wheel zoom routing is
unchanged except for the shared normalized scroll-token helper.

## Next actions

- None. Reopen only if right-drag panning fails again in detector or caked view.

## Validation

- 2026-04-29: `python -m pytest tests/test_gui_canvas_interactions.py -ra`
  passed, `34 passed`.
- 2026-04-29: `python -m pytest tests/test_gui_runtime_geometry_preview.py -ra`
  passed, `3 passed`.
- 2026-04-29: `python -m ruff format --check ra_sim/gui/canvas_interactions.py tests/test_gui_canvas_interactions.py`
  passed.
- 2026-04-29: `python -m pytest tests/test_gui_runtime_import_safe.py -ra -q`
  still failed in four unrelated manual-geometry/source-boundary tests present
  before this fix.
- 2026-04-29: `python -m ra_sim.dev format-check` still reported unrelated
  pre-existing formatting drift in files outside this patch.

## Links

- Runtime path: [ra_sim/gui/canvas_interactions.py](../../../ra_sim/gui/canvas_interactions.py)
- Test path: [tests/test_gui_canvas_interactions.py](../../../tests/test_gui_canvas_interactions.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
