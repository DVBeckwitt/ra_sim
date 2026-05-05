# Manual Qr/Qz Caked Picker Cache Projection

Status: completed
Type: bug
Owner: Codex
Issue: none
Priority: p1
Last updated: 2026-05-05

## Summary

Manual Qr/Qz simulation picking now keeps source identity view-independent and
treats detector and caked display coordinates as derived projections. A
simulation generated in detector view can be selected directly in caked view,
and detector -> caked -> detector redraw keeps the same simulation source row.

## What changed

- Added a forced caked projection callback so the cache can build caked sidecars
  without depending on the currently active GUI view.
- Made reuse-only caked clicks accept a detector-warmed cache only when the
  source/mask signatures match and a current caked sidecar is present.
- Made caked sidecar rows normalize to `caked_display`, with
  `display_col/display_row`, `caked_x/caked_y`, and `two_theta_deg/phi_deg`
  coming from the selected caked axis point.
- Changed saved caked redraw so source-matched current simulated projection
  wins over stale saved simulated caked fields.
- Kept measured/background caked fields separate from simulated caked overlay
  points.
- Tightened caked sidecar reuse so a requested sidecar misses when the current
  caked projection token changed or is unavailable. Detector-only reuse is
  unchanged.
- Broadened prewarm after simulation and on caked-view entry so clicks remain
  no-build while cold caked selection is warm by the time the user clicks.

## Status

- Bug status: fixed for targeted detector/caked Qr/Qz selection, saved caked
  redraw, sidecar reuse, and detector replay paths.
- Error status: fixed for the cold caked click path that reported
  `Manual Qr picker cache is not ready; update simulation or move the mouse to
  warm it.` after simulation generation.
- Feature status: implemented as cache/projection hardening only. No new GUI
  controls, saved-state fields, CLI flags, or version bump.
- Compatibility status: manual-pair source identity stays canonical; display
  fields remain active-view output and are not persisted as source truth.

## Validation

Passed:

- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "manual_qr_caked_saved or caked_qr_projection or detector_replay or sidecar or cache_is_not_ready or reuse_only"`
  (`31 passed, 501 deselected`)
- `python -m pytest -q tests/test_gui_runtime_geometry_interaction.py`
  (`15 passed`)
- `python -m pytest -q tests/test_gui_runtime_import_safe.py -k "prewarm or reuse_only or sidecar or projection"`
  (`55 passed, 353 deselected`)
- `python -m compileall -q ra_sim/gui tests`
- `python -m ruff check ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_geometry_interaction.py`

## Links

- [Debug and cache guide](../../debug-and-cache.md#geometry-qrqz-and-hkl-picker-state)
- [Changelog](../../../CHANGELOG.md)
