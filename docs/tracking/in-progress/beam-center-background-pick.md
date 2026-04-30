# Beam Center Background Pick

Type: feature
Status: implemented, targeted validation green
Last updated: 2026-04-30

## Problem

Setup beam-center entry was slider-only. Operators could refine Qr/Qz background
placements from image peaks, but could not use that same detector/background
peak-refinement path to set the direct beam center.

## Change

- Added `Setup > Beam Controls > Pick Beam Center`.
- The mode uses the current detector/background image. It switches back to
  detector view when needed, shows the background if hidden, and errors if no
  background is loaded.
- Left press starts a local zoom window, motion previews raw to refined point,
  and release commits the refined location.
- Right click cancels and restores the pre-pick view.
- Commit maps display `(col,row)` back to native detector coords, then sets
  `center_x = native_row` and `center_y = native_col`.
- The picker reuses the manual Qr/Qz local refiner with detector-space forcing,
  plus the same preview markers, throttle constants, and zoom window size.
- Runtime state is transient only; saved-state schema is unchanged.

## Validation

- `python -m pytest tests/test_gui_canvas_interactions.py tests/test_gui_views.py tests/test_beam_center_pick_helpers.py -ra`
  - 116 passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra`
  - 323 passed.
- `python -m pytest tests/test_gui_state_io.py -ra`
  - 13 passed.
- Selected canvas route regressions:
  - 5 passed.
- Targeted `compileall` on touched GUI/test files passed.

## Current Status

Feature path is implemented and targeted tests are green. Full
`python -m ra_sim.dev check` is not green in this dirty worktree because the
formatter gate reports pre-existing formatting drift in
`ra_sim/fitting/optimization.py` plus current local formatting drift in
`ra_sim/gui/_runtime/runtime_session.py`. A broader
`tests/test_gui_geometry_fit_workflow.py` run also sees existing geometry-fit
expectation drift around newly present dataset-spec fields, unrelated to this
button path.

No package version bump.
