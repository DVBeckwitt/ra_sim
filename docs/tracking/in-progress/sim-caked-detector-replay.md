# Sim caked detector replay

Status: in-progress
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-23

## Summary

Manual geometry had a caked-to-detector replay bypass for source-backed
simulated Qr rows. Caked selection was already using the current caked
projection cache correctly, but detector redraw could still reuse saved
detector/native aliases instead of replaying through the reverse LUT. This
follow-up keeps source identity as truth, keeps caked sim display/picking on
the current projection lookup, and routes detector replay only through:

`source identity -> current caked projection -> reverse LUT -> detector replay cache`

## Current state

Implemented in [manual_geometry.py](../../../ra_sim/gui/manual_geometry.py)
with focused regression coverage in
[test_manual_geometry_selection_helpers.py](../../../tests/test_manual_geometry_selection_helpers.py).

What changed:

- added `resolve_sim_detector_replay_from_caked_projection(...)`;
- detector replay now triggers only for source-backed sim rows with current
  caked projection evidence;
- detector replay cache is stored in `sim_detector_anchor_x/y`,
  `sim_detector_display_col/row`, and `sim_detector_frame_provenance`;
- detector redraw/runtime projection no longer falls back to direct
  `refined_sim_*`, `native_*`, `sim_*`, or stale detector display aliases for
  caked-resolved replay rows;
- stale replay anchors are closure-checked against the current projected caked
  point and refreshed once when needed;
- stale replay detector display cache is recomputed from the anchor/current
  detector display transform instead of being trusted blindly;
- replay helper no longer preserves stale saved detector display aliases when
  only a valid replay anchor is available and no current detector-display
  projection can be built;
- initial saved detector overlay now prefers source identity plus current caked
  projection replay over conflicting detector lookup rows;
- background replay path was left separate.

Bug/error status:

- code path updated across helper refresh, initial saved redraw, and runtime
  detector projection;
- adjacent replay regressions found in review were patched in the same blast
  zone;
- helper-level stale-display regressions and initial-build detector-lookup
  conflicts are covered in targeted tests;
- validation is still pending.

Feature status:

- no new operator feature;
- this is behavior hardening for existing manual Qr/caked replay.

## Next actions

- run targeted replay tests in
  `tests/test_manual_geometry_selection_helpers.py`;
- run the normal touched-area quality gate if the environment is clean enough;
- do one manual detector -> caked -> detector GUI replay check and confirm no
  post-finish jump;
- update the external GitHub issue/project if this follow-up is being tracked
  there.

## Validation

Not yet run in this worktree:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "sim_replay or detector_replay or initial_pairs"`
- `python -m ra_sim.dev check`
- manual GUI replay check

## Links

- [Tracking hub](../index.md)
- [Resolved picker alignment history](../archive/sim-peak-overlay-recovery.md)
- [Resolved detector-oracle caked background picks](../archive/detector-oracle-caked-background-picks.md)
- [GUI workflow](../../gui-workflow.md)
- [Debug and cache guide](../../debug-and-cache.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
