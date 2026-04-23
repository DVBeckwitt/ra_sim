# Detector-Oracle Caked Qr/Background Picks

Status: completed
Type: bug
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-04-23

## Summary

Caked manual Qr picking now treats detector-native simulated source rows as the
only simulated truth. Caked Qr candidates are built by sending canonical detector
source rows through the same detector-to-caked projection path used by the
working detector-selection flow. Caked background picks stay separate from
simulated Qr display and refine on the active caked background path before
round-tripping through the existing LUT callbacks for detector display.

## Current state

- Simulated Qr truth is detector-native provenance:
  `source_table_index`, `source_row_index`, reflection/branch/ray identity, and
  native detector coordinates.
- Caked Qr clicks hit-test only `caked_qr_projection_grouped_candidates`; there
  is no `grouped_candidates` fallback in caked Qr selection.
- Active and saved caked `sim_display` resolve from
  `source identity -> caked_qr_projection_lookup -> projected sim_display`.
- Alias-only saved simulated points are marked unresolved instead of being used
  as simulated truth.
- Detector display transform state participates in caked Qr cache invalidation.
- Background display remains view-local: caked background picks refine on caked
  background coordinates, detector background picks refine in detector display
  coordinates, and cross-view display uses existing LUT callbacks.

## Next actions

- Keep issue `#248` open only for unrelated remaining picker/projection work.
- Do not reintroduce caked-native simulated Qr truth or caked click-to-detector
  simulated reconstruction.
- If future caked Qr drift appears, debug the detector-to-caked projection cache
  and cache signature first.
- If future caked background drift appears, debug the background resolver/LUT
  path separately from simulated Qr display.

## Validation

- `python -m pytest tests/test_manual_geometry_selection_helpers.py -q`: PASS
- `python -m pytest tests/test_manual_geometry_live_peak_cache.py -q`: PASS
- `python -m pytest tests/test_gui_geometry_fit_workflow.py -q`: PASS
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_coordinate_diagnostics.py ra_sim/gui/manual_geometry.py scripts/debug/run_new4_caked_point_reprojection_check.py tests/test_gui_geometry_fit_workflow.py tests/test_manual_geometry_live_peak_cache.py tests/test_manual_geometry_selection_helpers.py`: PASS
- `python -m ruff format --check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_coordinate_diagnostics.py ra_sim/gui/manual_geometry.py scripts/debug/run_new4_caked_point_reprojection_check.py tests/test_gui_geometry_fit_workflow.py tests/test_manual_geometry_live_peak_cache.py tests/test_manual_geometry_selection_helpers.py`: PASS
- `git diff --check`: PASS
- `python -m ra_sim.dev check`: BLOCKED at format check by unrelated formatter drift in `ra_sim/fitting/optimization.py` and `tests/test_timing.py`.

## Links

- Related issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
- Tracking index: [docs/tracking/index.md](../index.md)
