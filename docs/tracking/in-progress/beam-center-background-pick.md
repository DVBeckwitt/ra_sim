# Beam Center Background Pick

Type: feature / bug fix
Status: implemented, click-placement bug fixed, targeted validation green
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

The second bug was in the default center seed itself. GUI startup and headless
geometry-fit defaults rebuilt PONI centers with a rotated-display formula:
`row = poni2 / pixel_size`, `col = image_size - poni1 / pixel_size`. That made
the initial beam center disagree with pyFAI/native detector semantics before
the user picked anything. PONI defaults now stay in native slider order:
`row = poni1 / pixel_size`, `col = poni2 / pixel_size`.

Beam-center preview also no longer uses the center-dependent caked manual-pick
wrapper. It now avoids local peak refinement entirely, so the committed point is
the exact display point the user clicked.

The placement path was still wrong for exact click placement because the
beam-center preview could snap the clicked point to a nearby detector-image
maximum before commit. Beam-center picking now uses the clicked detector-display
point exactly. For the default 3000 px clockwise display, the committed center
is `row = 3000 - display_col`, `col = display_row`.

## Change

- Added `Setup > Beam Controls > Pick Beam Center`.
- The mode uses the current detector/background image. It switches back to
  detector view when needed, shows the background if hidden, and errors if no
  background is loaded.
- Left press starts a local zoom window, motion previews the clicked display
  point, and release commits that clicked location.
- Right click cancels and restores the pre-pick view.
- Commit maps the clicked detector-view display `(col,row)` through
  `beam_center_row_col_from_detector_display(...)`, an extent-based
  beam-center transform kept separate from pixel-index transforms, then sets
  `center_x = center_row` and `center_y = center_col`.
- Added shared display-rotation helpers so inverse display rotation uses the
  rotated display shape for non-square detector images.
- The picker no longer uses local peak refinement. It keeps the same preview
  markers, throttle constants, and zoom window size.
- Runtime state is transient only; saved-state schema is unchanged.
- Added `beam_center_row_col_from_poni(...)` and used it from GUI runtime and
  headless geometry-fit defaults so both paths share the same native row/col
  PONI convention.
- Beam-center preview now bypasses all local peak refinement and stores the
  clicked display point exactly before applying the detector-extent transform.

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
- `python -m pytest tests/test_beam_center_pick_helpers.py -ra`
  - 12 passed, including PONI row/col defaults, exact clicked-point placement,
    headless default source checks, and no-refiner preview checks.
- `python -m compileall ra_sim/gui/geometry_overlay.py ra_sim/gui/_runtime/runtime_session.py ra_sim/headless_geometry_fit.py tests/test_beam_center_pick_helpers.py`
  - passed.
- `git diff --check -- CHANGELOG.md ra_sim/gui/geometry_overlay.py ra_sim/gui/_runtime/runtime_session.py ra_sim/headless_geometry_fit.py tests/test_beam_center_pick_helpers.py`
  - passed.
- `python -m ra_sim.dev check`
  - blocked by pre-existing formatting drift in
    `ra_sim/fitting/optimization.py`; touched files pass the formatter gate.

## Current Status

Feature path is implemented. First coordinate bug is fixed by the
detector-extent display-to-center transform
(`1404,1453 -> row=1596,col=1453`). Second default-center bug is fixed by
keeping PONI-derived centers in native row/col order for both GUI and headless
geometry fit. Third click-placement bug is fixed by disabling beam-center
auto-refine: the clicked display point is transformed directly as
`row = height - display_col`, `col = display_row`.

Targeted beam-center tests, targeted compile, and touched-file diff checks are
green. Full `python -m ra_sim.dev check` was not rerun after the focused fix.
No saved-state, CLI, or artifact schema change. No package version bump.
