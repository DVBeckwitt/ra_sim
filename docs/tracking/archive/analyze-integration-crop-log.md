# Analyze Integration Crop and Log View

Status: completed
Type: bug/feature
Owner: Codex
Issue: none
Priority: p2
Last updated: 2026-05-05

## Summary

Standard Analyze radial and azimuthal 1D integration plots were showing bins
outside the selected integration box. The 1D integration axes also needed a
dedicated log-scale control near the Fit Axes controls instead of sharing the
image-display log toggle.

## Current state

Standard caked radial and azimuthal profile generation now returns only bins
inside the rectangular `2theta/phi` integration box. Wrapped phi selections keep
their sorted plotting order, but out-of-box phi bins are no longer returned as
zero-intensity tail data.

Analyze > Fit Axes now includes `Log y-scale`, which switches the 1D
integration plot y-axes between linear and log intensity. It applies to the
standard radial/azimuthal layout and selected-Qr rod profile axes.

Bug/error status: fixed. Feature status: complete for the requested cropped
radial/azimuthal integration plots and Fit Axes log-y view. No known open issue
remains from this change.

## Next actions

None required. Future integration-plot modes should keep display curves cropped
to the active integration support instead of plotting masked bins as zeros.

## Validation

- `python -m pytest tests/test_gui_runtime_import_safe.py -ra`
- `python -m compileall ra_sim/gui/_runtime/runtime_session.py tests/test_gui_runtime_import_safe.py`
- `python -m pytest tests/test_gui_views.py tests/test_gui_runtime_import_safe.py -ra`
- `python -m compileall ra_sim tests`
- `python -m ra_sim.dev check`

All commands passed on 2026-05-05.

## Links

- `docs/gui-workflow.md`
- `CHANGELOG.md`
