# Beam Center Background Pick

Type: feature / bug fix
Status: implemented, coordinate bug fixed, targeted validation green
Last updated: 2026-04-30

## Problem

Setup beam-center entry was slider-only. Operators could refine Qr/Qz background
placements from image peaks, but could not use that same detector/background
peak-refinement path to set the direct beam center.

After the picker was added, the preview marker landed visually on the intended
beam spot, but release wrote the wrong slider values in the default state. A
known example was the real beam center `row=1596`, `col=1453` being committed
as approximately `row=1546`, `col=1595`.

## Change

- Added `Setup > Beam Controls > Pick Beam Center`.
- The mode uses the current detector/background image. It switches back to
  detector view when needed, shows the background if hidden, and errors if no
  background is loaded.
- Left press starts a local zoom window, motion previews raw to refined point,
  and release commits the refined location.
- Right click cancels and restores the pre-pick view.
- Commit maps the refined detector-view display `(col,row)` into the same
  simulation beam-center frame used by the default/hBN geometry path, then
  sets `center_x = center_row` and `center_y = center_col`.
- Added shared display-rotation helpers so inverse display rotation uses the
  rotated display shape for non-square detector images.
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
- `python -m pytest tests/test_beam_center_pick_helpers.py -ra`
  - 4 passed, including the default-style `row=1596`, `col=1453` repro.
- `python -m pytest tests/test_gui_canvas_interactions.py tests/test_gui_views.py -ra`
  - 114 passed.
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra`
  - 329 passed.

## Current Status

Feature path is implemented, the wrong coordinate commit bug is fixed, and
targeted tests are green. Full
`python -m ra_sim.dev check` is not green in this dirty worktree because the
formatter gate reports pre-existing formatting drift in
`ra_sim/fitting/optimization.py` plus current local formatting drift in
`ra_sim/gui/_runtime/runtime_session.py`. A broader
`tests/test_gui_geometry_fit_workflow.py` run also sees existing geometry-fit
expectation drift around newly present dataset-spec fields, unrelated to this
button path.

No package version bump.
