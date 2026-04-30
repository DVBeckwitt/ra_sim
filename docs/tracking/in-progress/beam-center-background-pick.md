# Beam Center Background Pick

Type: feature / bug fix
Status: implemented, coordinate bug fixed, targeted validation green
Last updated: 2026-04-30

## Problem

Setup beam-center entry was slider-only. Operators could refine Qr/Qz background
placements from image peaks, but could not use that same detector/background
peak-refinement path to set the direct beam center.

After the picker was added, the preview marker landed visually on the intended
beam spot, but release still treated the refined point as detector-display
coordinates. In the default repro, the visually correct point is
`display_col=1404`, `display_row=1453`; writing that point directly gives the
bad slider result `row=1453`, `col=1404`.

The correct slider conversion is not the raw pixel-index inverse. Beam-center
sliders use the detector-geometry extent convention, so a 3000 px detector must
invert the default display rotation with `3000 - display_col`, not
`2999 - display_col`. That maps `display_col=1404`, `display_row=1453` to
`row=1596`, `col=1453`.

## Change

- Added `Setup > Beam Controls > Pick Beam Center`.
- The mode uses the current detector/background image. It switches back to
  detector view when needed, shows the background if hidden, and errors if no
  background is loaded.
- Left press starts a local zoom window, motion previews raw to refined point,
  and release commits the refined location.
- Right click cancels and restores the pre-pick view.
- Commit maps the refined detector-view display `(col,row)` through
  `beam_center_row_col_from_detector_display(...)`, an extent-based
  beam-center transform kept separate from pixel-index transforms, then sets
  `center_x = center_row` and `center_y = center_col`.
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
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_beam_center_pick_helpers.py tests/test_gui_canvas_interactions.py -q -s`
  - 58 passed.
- Targeted `compileall` on
  `ra_sim/gui/geometry_overlay.py`, `ra_sim/gui/manual_geometry.py`,
  `ra_sim/gui/_runtime/runtime_session.py`,
  `ra_sim/gui/canvas_interactions.py`, `ra_sim/gui/state.py`,
  `ra_sim/gui/views.py`, `tests/test_beam_center_pick_helpers.py`, and
  `tests/test_gui_canvas_interactions.py` passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_beam_center_pick_helpers.py tests/test_gui_canvas_interactions.py tests/test_gui_views.py -q -s`
  - 120 passed, 1 failed.
  - The failing test is
    `tests/test_gui_views.py::test_stacking_parameter_panels_and_slider_refs_are_stored`
    with Tk's `RuntimeError: Too early to create variable: no default root window`;
    this is in the stacking-panel test path, not the beam-center picker path.

## Current Status

Feature path is implemented, the extent-based beam-center conversion fixes the
known default repro (`1404,1453 -> row=1596,col=1453`), and targeted
beam-center/canvas tests are green. The broader requested view suite is still
blocked by an existing Tk default-root failure in the stacking-panel test. Full
`python -m ra_sim.dev check` is not green in this dirty worktree because the
formatter gate reports pre-existing formatting drift in
`ra_sim/fitting/optimization.py` plus current local formatting drift in
`ra_sim/gui/_runtime/runtime_session.py`. A broader
`tests/test_gui_geometry_fit_workflow.py` run also sees existing geometry-fit
expectation drift around newly present dataset-spec fields, unrelated to this
button path.

No package version bump.
