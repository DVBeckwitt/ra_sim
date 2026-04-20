# Manual Qr/Qz and HKL picker alignment

Status: resolved
Type: bug
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-04-19

This page records the resolved detector/caked manual-picking problem for Qr/Qz
group selection and HKL selection. It replaces the earlier in-progress
`Simulated peak overlay recovery` tracker for this issue.

See also:

- [tracking hub](../index.md)
- [docs index](../../index.md)
- [investigation history](sim-peak-overlay-recovery-history.md)
- [GUI workflow](../../gui-workflow.md)
- [Debug and cache guide](../../debug-and-cache.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)

## Resolution Summary

Manual Qr/Qz group selection and HKL selection now behave the same way in
detector view and caked `(2theta, phi)` view. The final solution separated
structural simulation truth from current-view projection, then made both
pickers consume the same current-view simulated-candidate payload.

User confirmation on 2026-04-19: Qr selection is working correctly, and the
entire Qr/HKL picker problem is resolved.

## Implementation Summary

The fix landed in four connected steps:

- move Qr/Qz group membership onto structural simulation state keyed by active
  CIF/lattice hit tables instead of live detector/caked preview rows;
- rebuild caked manual-pick candidates from stored simulation rows and
  `stored_max_positions_local` when current live rows are empty after caked
  integration refresh;
- keep detector/display aliases in detector space, keep caked aliases in
  angular space, and project caked targets from simulation-native detector
  branch pixels through the live caked simulation transform;
- route HKL hit testing and selected-marker placement through the same shared
  simulated candidate payload already used by the corrected Qr/manual picker.

## Qr/Qz Root Causes

The Qr/Qz picker had three separate failure modes that looked related but needed
different fixes.

First, caked mode could fail to list all Qr/Qz groups because the selectable
group universe was being rebuilt from live `peak_records` or caked/intersection
rows. Those rows are view-dependent and can be filtered or emptied by caked
projection or background caked-integration refresh. Detector view still had
usable rows, so the bug looked caked-specific.

Second, after the group list was restored, caked Qr/Qz click targets could still
be unavailable after background caked refresh because the picker asked the live
preview rows for candidates. The stored simulation hit tables still contained
the correct simulation state, but the caked picker was not using them as a
fallback.

Third, after caked Qr/Qz candidates were rebuilt, their `(2theta, phi)` positions
could be wrong because simulation-native detector pixels were sometimes sent
through a background/display detector adapter before caked conversion. That
applied the wrong detector frame to simulated seed pixels.

## Qr/Qz Solution

The Qr/Qz group universe is now cached from the active CIF/lattice simulation
hit-table state rather than from view-filtered live rows. The cache is tied to
the CIF, unit-cell/lattice values, and simulation hit-table content. Detector
and caked view switches do not invalidate or shrink it.

When caked live rows are empty but the simulation state is still valid, manual
Qr/Qz picking can rebuild selectable candidates from `stored_max_positions_local`
and the stored lattice/hit-table metadata. That keeps background caked
integration refreshes from erasing the Qr/Qz picker.

Caked Qr/Qz target positions are now projected from simulation-native detector
branch pixels through the same live caked simulation transform used to render
the caked simulation image. Detector/display aliases stay detector/display
coordinates. Caked angular coordinates stay in caked fields such as `caked_x`,
`caked_y`, `raw_caked_x`, `raw_caked_y`, `two_theta_deg`, and `phi_deg`.

## HKL Root Cause

The HKL picker was not following the fixed Qr/Qz picker path. Runtime HKL clicks
could bypass the Qr/manual candidate payload, and caked HKL hit testing could
fall back to detector/display coordinates when it needed the current-view caked
coordinates for the same simulated branch.

That made detector view appear correct while caked HKL selection disagreed with
the now-correct Qr/Qz selection.

## HKL Solution

The HKL picker now uses the same simulated candidate payload as the Qr/manual
picker. Runtime HKL clicks are wired through the HKL simulation-point factory
that reads from the Qr picker cache, so both pickers use the same source-row
identity and current-view projected coordinates.

In caked mode, HKL hit testing prefers true caked/angular fields from that
candidate payload and does not treat detector/display `display_col` and
`display_row` as caked positions. The selected HKL marker also uses the active
caked candidate point, matching the click hit test and the rendered caked
simulation spot.

## Guardrails

Preserve these rules when touching manual picking again:

- Qr/Qz group membership is structural simulation state, not current-view state.
- Detector/caked view switches must not change the Qr/Qz group universe.
- Simulation-native detector seed pixels must be mapped through the live caked
  simulation transform before caked hit testing.
- Detector aliases must remain detector/display coordinates.
- Caked aliases must remain caked/angular coordinates.
- HKL and Qr/Qz picking should share candidate source identity and current-view
  projection wherever possible.

## Main Code Paths

- `ra_sim/gui/geometry_q_group_manager.py`
- `ra_sim/gui/manual_geometry.py`
- `ra_sim/gui/peak_selection.py`
- `ra_sim/gui/_runtime/runtime_session.py`
- `ra_sim/gui/state.py`

Main regression coverage lives near:

- `tests/test_gui_geometry_q_group_manager.py`
- `tests/test_manual_geometry_selection_helpers.py`
- `tests/test_gui_peak_selection.py`
- `tests/test_gui_canvas_interactions.py`

## Acceptance Criteria Reached

- Qr/Qz detector-view selection chooses the expected group and branch.
- Qr/Qz caked-view selection chooses the expected group and branch.
- Caked Qr/Qz click targets agree with the rendered caked simulation spot.
- Background caked-integration refresh does not erase the selectable Qr/Qz group
  universe while simulation hit tables are still valid.
- HKL detector-view selection chooses the expected simulated branch.
- HKL caked-view selection uses the same current-view candidate frame as Qr/Qz
  selection.
- HKL caked markers agree with the rendered caked simulation spot.
