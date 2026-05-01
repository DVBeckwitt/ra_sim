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

The runtime commit path also wrote the mapped pick values into the visible GUI
beam-center sliders in the wrong order. The helper still computes the
detector-display mapping as row/col, but the pick commit now reverses those two
values when updating the GUI row/col slider pair. For the default 3000 px view,
the visible Beam Center Col field receives `3000 - display_col`.

The visible slider/entry row also needs to be driven through the Tk Scale widget,
not only the backing DoubleVar. Pick commit now calls the Beam Center Row/Col
slider `set(...)` methods with a DoubleVar fallback so the visible entries
refresh immediately.

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
  beam-center transform kept separate from pixel-index transforms, then writes
  the two mapped values into the visible GUI Beam Center Row/Col sliders in the
  pick path's required order.
- Added shared display-rotation helpers so inverse display rotation uses the
  rotated display shape for non-square detector images.
- The picker reuses the manual Qr/Qz local refiner with detector-space forcing,
  plus the same preview markers, throttle constants, and zoom window size.
- Runtime state is transient only; saved-state schema is unchanged.
- Added `beam_center_row_col_from_poni(...)` and used it from GUI runtime and
  headless geometry-fit defaults so both paths share the same native row/col
  PONI convention.
- Beam-center preview now bypasses all local peak refinement and stores the
  clicked display point exactly before applying the detector-extent transform.
- Beam-center commit now reverses the mapped pick row/col values when assigning
  the GUI row/col slider pair, matching the visible Beam Controls fields.
- The pick path now names the visible GUI values explicitly so Beam Center Col
  remains the detector-extent inverse (`3000 - display_col` in the default
  detector view).
- Pick commit now writes those values through the visible slider widgets, with
  fallback to the backing variables only if the widget write is unavailable.

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
  - previously blocked by
    `tests/test_gui_views.py::test_stacking_parameter_panels_and_slider_refs_are_stored`
    creating Tk variables without fake widget bindings.
- `python -m pytest tests/test_gui_views.py::test_stacking_parameter_panels_and_slider_refs_are_stored -ra`
  - 1 passed after adding the missing fake `Checkbutton` and `BooleanVar`
    bindings in the test.
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
`row = height - display_col`, `col = display_row`. Fourth row/column commit
bug is fixed by reversing the mapped values into the GUI slider fields during
Pick Beam Center assignment.

Targeted beam-center tests, targeted compile, the previously failing stacking
panel view test, and touched-file diff checks are green. Full
`python -m ra_sim.dev check` was not rerun after the focused fix. No
saved-state, CLI, or artifact schema change. No package version bump.
