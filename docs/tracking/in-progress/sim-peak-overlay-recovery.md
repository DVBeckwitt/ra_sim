# Simulated peak overlay recovery status

Status: in-progress
Type: bug
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-04-18

This page tracks the unresolved simulated-peak overlay bug in the GUI. Use it
as the short resume document if chat history is lost.

See also:

- [tracking hub](../index.md)
- [docs index](../../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Architecture guide](../../architecture.md)
- [Canonical simulation/fitting reference](../../simulation_and_fitting.md)

## Purpose

- Keep continuity for the detector/caked simulated-peak placement bug.
- Separate the visible user bug from adjacent cache and Q-group cleanup work.
- Record what has already been changed so the next pass does not repeat it.

## Resume Here

- User-visible symptom:
  simulated detector-space markers can appear about 90 degrees clockwise from
  where they should be, while background points are correct.
- In detector mode, the error looks like a clean 90 degree rotation.
- In caked mode, simulated point overlays are also wrong, but not by the same
  clean 90 degree rotation.
- Important non-symptom:
  the full caked simulation image itself is reported to be correct.
- High-confidence consequence:
  the bug is in per-peak overlay identity/projection, not the whole simulation
  image generation path.
- Most relevant recent commits:
  - `9be9c9b` `fix(gui): split detector peak frame prep`
  - `1b62e0d` `fix(q-group): trust normalized peak records`
  - `1519e90` `fix(runtime): use exact caked cache angles`
  - `09c7be7` `fix(manual-geometry): sync refined sim aliases`
  - `7a66a05` `fix(gui): reproject refined sim peaks`
- Main code paths:
  - `ra_sim/gui/manual_geometry.py`
  - `ra_sim/gui/geometry_fit.py`
  - `ra_sim/gui/peak_selection.py`
  - `ra_sim/gui/geometry_q_group_manager.py`
  - `ra_sim/gui/_runtime/runtime_session.py`
- Main tests already covering nearby behavior:
  - `tests/test_manual_geometry_selection_helpers.py`
  - `tests/test_gui_geometry_fit_workflow.py`
  - `tests/test_gui_overlays.py`
  - `tests/test_gui_peak_selection.py`
  - `tests/test_gui_geometry_q_group_manager.py`

## Problem Statement

The GUI has two different detector-space display frames in play:

- background detector display frame
- simulation/native detector display frame

The background image and background-picked points are displayed correctly, but
some simulated-peak consumers still appear to reuse detector/caked alias fields
from the wrong frame instead of reprojecting from canonical detector-native
coordinates.

The visible result is:

- detector view:
  simulated marker lands in the wrong rotated location, typically looking like
  a 90 degree clockwise offset from the correct point
- caked point overlays:
  simulated point can also drift, but the caked-space error does not look like
  the same clean 90 degree detector-space rotation, even though the full caked
  image is correct

## Expected Behavior

For the same simulated source peak:

- detector-view simulated marker should land near the corresponding background
  point in the current displayed detector frame
- caked simulated marker should land at the same `(2theta_deg, phi_deg)` that
  the full caked simulation image implies for that detector branch point
- HKL picking, QR/manual picking, and geometry-fit overlays should all agree on
  the same source row and current-view position

## What We Know

- The background image is rotated for display.
- The full caked simulation image is already correct.
- That means the remaining bug is likely not in the full detector-to-cake image
  pipeline.
- The remaining bug is most likely in one or more per-peak overlay consumers
  that still trust stored detector/caked alias fields instead of canonical
  detector-native truth.

## Current State

- Two focused implementation attempts now exist in current repo state:
  the earlier manual-geometry native-projector handoff and the later
  detector-frame split through peak-selection/Q-group cache paths.
- `9be9c9b` threaded detector-view peak-frame callbacks through
  `peak_selection`, `bootstrap`, and `geometry_q_group_manager` so live
  detector-view overlay/cache rows rebuild in the displayed background-detector
  frame instead of the simulator-detector frame.
- The same detector-frame work also hardened detector-view click selection so
  detector-only inverse callbacks fail safe when they return `None` or
  non-finite values and no simulator inverse is available.
- Adjacent cache-schema helper work now keeps detector/caked layout handling
  explicit during replay, normalization, and focused regression coverage.
- Targeted automated validation was green:
  - `python -m pytest tests/test_gui_peak_selection.py -q` -> `47 passed`
  - `python -m pytest tests/test_gui_runtime_import_safe.py tests/test_intersection_cache_schema.py tests/test_gui_runtime_primary_cache.py tests/test_manual_geometry_live_peak_cache.py -q` -> `234 passed`
- Manual GUI verification still showed detector-view `Pick Qr Set` and related
  simulated overlays failing to align with `Pick HKL` for the same simulated
  reflection.
- So the detector-frame work, click guard, and focused regressions changed real
  code paths but still did not resolve the user-visible bug or produce a
  visible behavior change in the GUI.

## What Has Already Been Changed

### 1. Q-group / peak-record trust cleanup

- Q-group code was updated to stop trusting arbitrary raw `peak_records`.
- Mixed valid/stale peak-record lists are now filtered through normalized live
  candidates before reuse.
- Peak-record rebuild paths were hardened so QR/manual preview can repopulate
  from the overlay/cache path instead of silently reusing garbage rows.

What this helped:

- reduced stale-row pollution in Q-group state
- improved QR/manual preview rebuild behavior

What it did not prove:

- that detector-view simulated marker placement is now correct

### 2. Exact caked angle cache work

- Prepared caked cache columns `17/18` were moved toward exact branch-point
  angle storage instead of cake-bin center approximation.
- Worker/cached-result geometry sourcing was updated so angle cache prep does
  not depend on live sliders when replaying a result.
- Cache-schema handling is now centralized around the four supported layouts:
  detector `14`/`17` and caked `16`/`19`, with shared helpers for provenance
  and cached-angle extraction so overlay/rebuild consumers stop inferring
  semantics from scattered raw width checks.

What this helped:

- exact `(2theta, phi)` cache provenance
- worker-result vs live-slider geometry consistency

What it did not prove:

- that point overlays are using those exact cached angles as their truth source

### 3. Manual-geometry refined alias sync

- Refined detector/native/caked aliases were synchronized more aggressively.
- `geometry_manual_apply_refined_simulated_override()` now carries refined
  fields onto the resolved row and clears stale native aliases when only
  refined detector/caked coords exist.
- `project_peaks_to_current_view()` now:
  - prefers refined fields when present
  - can invert refined detector display through background display space
  - uses current background shape for detector-view reprojection

What this helped:

- closed one real bug where refined rows could snap back to stale native truth
- closed one real bug where non-square background shape could move refined
  detector points after reprojection

What it did not prove:

- that every consumer of simulated rows actually goes through this projector

### 4. HKL/Qr-set detector projection handoff attempt

- `ra_sim/gui/manual_geometry.py` was updated so
  `project_peaks_to_current_view()` first tries
  `native_sim_to_display_coords(...)` whenever a live candidate row already has
  canonical native simulated detector coordinates.
- `ra_sim/gui/_runtime/runtime_session.py` now passes
  `_native_sim_to_display_coords` into the manual-geometry runtime projection
  workflow.
- `tests/test_manual_geometry_live_peak_cache.py` now covers the live
  peak-cache fallback case and asserts detector-view manual-geometry candidates
  keep the same projected detector coordinates as HKL lookup.

What this helped:

- removed one real mismatch between HKL lookup projection and the manual
  projector code path for native simulated detector coords
- proved the focused projector handoff works in targeted automated coverage

What it did not prove:

- that the failing GUI row actually goes through that code path end to end
- that no later consumer or redraw step overwrites the corrected detector-view
  coordinates
- that this projector handoff is the main seam causing the remaining visible
  bug

### 5. Detector-frame peak/cache split attempt

- Detector-view peak-selection cache assembly was updated so live
  `peak_positions`, restored detector rows, and nearest-peak detector lookups
  rebuild in the displayed background-detector frame instead of trusting
  simulator-frame detector aliases.
- Runtime/bootstrap wiring now carries detector-native to displayed-detector
  callbacks into the peak-selection and Q-group preview/rebuild seams rather
  than forcing those detector rows through the simulator-display projector.
- Detector-view click inversion now returns safely when a detector-only inverse
  callback yields `None` or non-finite values and no simulator inverse exists,
  instead of falling through to a `TypeError`.
- Cache-schema helpers and focused tests now make the supported detector/caked
  layouts explicit for overlay/cache replay and rebuild consumers instead of
  relying on scattered raw-width guesses.

What this helped:

- removed one real wrong-frame seam in detector-view HKL nearest-peak search,
  overlay cache rebuild, and Q-group preview row normalization
- closed one real detector-only click crash path
- added deterministic regression coverage for the legacy/current cache-layout
  replay seams touched by this attempt

What it did not prove:

- that the final visible overlay consumer keeps using those corrected
  detector-frame coordinates instead of overwriting them later
- that the remaining visible bug is explained by cache-schema replay rather
  than a later overlay assembly seam
- that manual GUI behavior changed

## What Is Most Likely Still Wrong

The remaining bug is probably one of these:

1. A consumer still reads `sim_col/sim_row`, `display_col/display_row`,
   `caked_x/caked_y`, or `two_theta_deg/phi_deg` directly from a shared row
   without first reprojecting from canonical detector-native coordinates.
2. A row is being reprojected, but the wrong source frame is still chosen for
   that row before or after the projector runs.
3. HKL overlay, QR/manual overlay, and geometry-fit overlay are not all using
   the same canonical source-row truth for the same picked peak.
4. The failing detector-view row may never exercise the new
   `native_sim_to_display_coords(...)` handoff, or a later overlay assembly
   path may replace the corrected detector coordinates before draw.

Latest evidence:

- the manual-geometry native-projector handoff was implemented and the focused
  tests passed
- `9be9c9b` detector-frame split landed with focused peak/Q-group/cache
  regressions
- `python -m pytest tests/test_gui_peak_selection.py -q` passed with
  `47 passed`
- `python -m pytest tests/test_gui_runtime_import_safe.py tests/test_intersection_cache_schema.py tests/test_gui_runtime_primary_cache.py tests/test_manual_geometry_live_peak_cache.py -q`
  passed with `234 passed`
- manual GUI behavior still did not change
- so the bug is not explained solely by missing
  `native_sim_to_display_coords(...)` inside
  `project_peaks_to_current_view()`, nor solely by the later detector-frame
  rewiring in peak-selection/Q-group cache paths

The highest-value suspicion is still:

- some later per-peak overlay path is still treating corrected detector-frame
  coordinates as disposable and is either rebuilding from simulation-frame
  aliases or overwriting the row after the peak-selection/Q-group seams already
  did the right thing

## Most Relevant Places To Recheck

### Canonical overlay truth

- `ra_sim/gui/peak_selection.py`
  - `ensure_runtime_peak_overlay_data()`
  - anything building `simulation_runtime_state.peak_records`

Questions:

- Does this path always emit trustworthy `native_col/native_row`?
- Does any row keep detector display fields that are only valid in the sim
  image frame?

### Manual / QR overlay reprojection

- `ra_sim/gui/manual_geometry.py`
  - `build_geometry_manual_initial_pairs_display()`
  - `geometry_manual_apply_refined_simulated_override()`
  - `make_runtime_geometry_manual_projection_callbacks()`
  - `project_peaks_to_current_view()`

Questions:

- Does every detector/caked simulated point shown to the user come from this
  projector?
- Are any rows allowed to bypass it?

### Geometry-fit overlay assembly

- `ra_sim/gui/geometry_fit.py`
  - `_project_source_entry_for_current_view()`
  - `build_geometry_manual_fit_dataset()`
  - legacy dense-source resolution paths

Questions:

- Are `fit_source_entry` and `overlay_source_entry` using the same truth row?
- Is any legacy path still carrying stale detector/caked aliases into the final
  `initial_pairs_display` payload?

### Q-group preview and source-row reuse

- `ra_sim/gui/geometry_q_group_manager.py`

Questions:

- Are Q-group candidates always derived from projected `peak_records`?
- Can detector-only rows still reach a caked consumer before caked coords are
  rebuilt?

## What To Do Next

The next debugging pass should be deterministic and row-based:

1. Pick one concrete bad peak in the GUI.
2. Confirm whether that exact bad row is already corrected inside
   `simulation_runtime_state.peak_records` / detector-view peak overlay cache
   after `ensure_runtime_peak_overlay_data()`, and whether any detector-only
   inverse guard is even relevant for the failing interaction.
3. Log or breakpoint the exact source identity for that peak:
   `source_table_index`, `source_row_index`, `source_peak_index`,
   `source_branch_index`, `source_reflection_index`.
4. For that exact row, compare fields at each seam:
   - `native_col/native_row`
   - `sim_col_raw/sim_row_raw`
   - `display_col/display_row`
   - `caked_x/caked_y`
   - `two_theta_deg/phi_deg`
   - `refined_sim_*`
5. Trace that same row through:
   - HKL overlay
   - QR/manual overlay
   - Q-group preview / candidate rebuild
   - geometry-fit `initial_pairs_display`
6. Verify which field each consumer actually draws and whether any later step
   overwrites or ignores the corrected detector-frame coords after the new
   peak-selection/Q-group handoff.

Do not start from aggregate caches or grouped previews first. Start from one
known bad row and follow it end to end.

## Acceptance Criteria

This bug is only done when all of these are true for the same source row:

- detector-view simulated marker lands near the matching background point
- caked simulated marker lands at the same `(2theta, phi)` implied by the full
  caked image
- HKL overlay, QR/manual overlay, and geometry-fit overlay all agree on the
  same detector/caked position
- figure-facing regressions pass for:
  - detector view
  - caked view
  - non-square background display
  - refined manual override without saved refined native coordinates

## Adjacent Work That Is Useful But Not The Main Visible Bug

These issues matter, but they should not be confused with the main detector
marker rotation bug:

- prepared caked cache using live slider geometry instead of result geometry
- exact angle cache vs cake-bin-center cache
- stale Q-group peak-record reuse
- rebuilt peak records missing caked coordinates
- Numba warm-marker bookkeeping

Those paths can affect reproducibility or preview stability, but the persistent
visible user bug is still the wrong per-peak overlay position.
