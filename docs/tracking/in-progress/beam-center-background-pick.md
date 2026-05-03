# Beam Center Background Pick

Type: feature / bug fix
Status: implemented, coordinate ownership untangled, targeted validation green
Last updated: 2026-05-01

## Problem

Setup beam-center entry was slider-only. Operators could refine Qr/Qz background
placements from image peaks, but could not use that same detector/background
peak-refinement path to set the direct beam center.

After several formula-level fixes, the live GUI still did not consistently show
the expected Beam Center Row/Col values after a pick. The working hypothesis is
not one more row/col swap, but duplicate ownership or a later stale writer after
the pick commit.

The required GUI contract is fixed: for a default 3000 px clockwise detector
display, `display_col=1456`, `display_row=1607` must commit Beam Center
`row=1607`, `col=1544`. Row follows clicked display row; Col is mirrored across
the displayed detector width.

This patch adds gated beam-center tracing and an overwrite guard so the first
post-pick divergence is visible in `debug/beam_center_trace.jsonl` when
`RA_SIM_TRACE_BEAM_CENTER=1` is set. The trace covers the pick callback,
visible Tk Scale/DoubleVar/entry chain, scheduled update, marker projection, and
detector-remap center reads. No root cause is documented here until a trace
record identifies the specific overwriter.

## Change

- Added `Setup > Beam Controls > Pick Beam Center`.
- The mode uses the current detector/background image. It switches back to
  detector view when needed, shows the background if hidden, and errors if no
  background is loaded.
- Left press starts a local zoom window, motion previews the exact display point,
  and release commits that point after one detector-extent transform.
- Right click cancels and restores the pre-pick view.
- Commit maps the clicked detector-view display `(col,row)` through
  `beam_center_row_col_from_detector_display(...)`, then writes the mapped
  Beam Center Row/Col values directly into the visible GUI sliders and entries.
- Added shared display-rotation helpers so inverse display rotation uses the
  rotated display shape for non-square detector images.
- The picker shares the detector preview markers, throttle constants, and zoom
  window size, but does not run the manual Qr/Qz local peak refiner.
- Runtime state is transient only; saved-state schema is unchanged.
- Added `beam_center_row_col_from_poni(...)` and used it from GUI runtime,
  headless geometry-fit defaults, and headless simulation defaults so all three
  paths share the same native row/col PONI convention.
- Beam-center preview now bypasses all local peak refinement and stores the
  clicked display point exactly before applying the detector-extent transform.
- Beam-center commit now writes one canonical GUI `(row, col)` pair through
  `_set_beam_center_row_col_sliders(...)`; the helper drives Scale, DoubleVar,
  and entry text consistently.
- The pick path uses the displayed detector extent instead of hard-coded
  dimensions or raw pixel-index assumptions.
- The beam-center marker projects slider row/col back into the rotated detector
  display for visual feedback and hides in q-space.
- `RA_SIM_TRACE_BEAM_CENTER=1` records pick, widget, scheduled-update, marker,
  remap, and overwrite-guard diagnostics to `debug/beam_center_trace.jsonl`.

## Validation

- `python -m compileall -q ra_sim/gui/geometry_overlay.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_detector_remap_cache.py ra_sim/headless_geometry_fit.py ra_sim/cli.py tests/test_beam_center_pick_helpers.py tests/test_gui_runtime_detector_remap_cache.py tests/test_cli_headless.py`
  - passed.
- Direct smoke check for display click `(1456, 1607)` on a 3000 px clockwise
  detector:
  - mapped to GUI `row=1607`, `col=1544`;
  - projected the marker back to display `(1456, 1607)`;
  - kept runtime, simulation, and remap center reads on `(1607, 1544)` after
    Tk events drained.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gui_runtime_detector_remap_cache.py -q`
  - 9 passed.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_beam_center_pick_helpers.py -q`
  - 16 passed.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gui_canvas_interactions.py::test_beam_center_pick_press_motion_release_uses_priority_canvas_route -q`
  - 1 passed.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_cli_headless.py -q`
  - 2 passed.

## Current Status

Feature path is implemented with a GUI-first beam-center convention: Pick Beam
Center maps a clicked display point once, writes Beam Center Row/Col through the
visible slider and entry widgets, and projects the marker back from the same
pair. Runtime, simulation, and detector-remap reads are traced against that
same canonical pair after Tk events drain.

For the default clockwise detector view, a click at `display_col=1456`,
`display_row=1607` commits `row=1607`, `col=1544`. No saved-state, CLI argument,
or artifact schema changed. No package version bump.
