# Background Qr Reference Picks

## Status

Implemented and targeted validation passed on 2026-04-30.

## Problem

Manual Qr/Qz placement required a simulated group identity. Some diagnostic
background peaks need the same local measured-peak refinement and export path,
but they should not pretend to have an HKL or feed the geometry solver.

## Change

- Added Match-tab `Place Background Qr Set`.
- The button arms a one-click background-only placement session.
- The clicked background point is refined to the local peak top using the manual
  Qr/Qz measured-peak path.
- The saved row omits HKL and Qr group identity.
- The saved label is the refined measured position:
  `2theta=<value>,phi=<value>`.
- The row is marked as a background Qr reference and disabled for geometry
  solving.
- Manual pair import/export keeps the row, so diagnostic notebooks that read
  manual pairs can consume it alongside normal Qr/Qz picks.

## Validation

Passed:

- `python -m compileall ra_sim/gui/manual_geometry.py ra_sim/gui/views.py ra_sim/gui/bootstrap.py ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_views.py tests/test_gui_bootstrap.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_geometry_fit_workflow.py`
- `python -m pytest tests/test_gui_views.py::test_geometry_tool_action_controls_store_refs_and_support_updates tests/test_gui_views.py::test_geometry_tool_action_controls_can_add_all_qr_set_button tests/test_gui_bootstrap.py::test_build_runtime_geometry_tool_action_controls_bootstrap_wires_add_all tests/test_manual_geometry_selection_helpers.py::test_geometry_manual_place_selection_at_saves_background_qr_reference_without_hkl tests/test_gui_geometry_fit_workflow.py::test_geometry_manual_pair_enabled_for_geometry_fit_skips_background_qr_reference -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py::test_runtime_session_import_stays_headless_until_tk_backend_is_needed -ra`

Known blocker:

- `python -m ra_sim.dev check` still fails because pre-existing
  `ra_sim/fitting/optimization.py` formatting is out of sync. The touched files
  format clean.

## Compatibility

No package version bump. The new saved rows are additive manual-pair metadata and
do not alter normal HKL/Qr/Qz placement semantics. Geometry fitting filters these
background-reference rows out before dataset construction.
