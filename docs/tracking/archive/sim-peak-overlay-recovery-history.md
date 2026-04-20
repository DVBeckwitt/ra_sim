# Simulated peak overlay recovery history

Status: archived
Type: investigation
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-04-19

This page preserves the original in-progress investigation log for the
simulated-peak overlay bug behind issue `#248`. It is archived as debugging
history now that the finished resolution lives in
`Manual Qr/Qz and HKL picker alignment`.

See also:

- [tracking hub](../index.md)
- [docs index](../../index.md)
- [resolved summary](sim-peak-overlay-recovery.md)
- [GUI workflow](../../gui-workflow.md)
- [Architecture guide](../../architecture.md)
- [Canonical simulation/fitting reference](../../simulation_and_fitting.md)

## Purpose

- Keep continuity for the detector/caked simulated-peak placement bug.
- Separate the visible user bug from adjacent cache and Q-group cleanup work.
- Record what has already been changed so the next pass does not repeat it.

## Resume Here

- User-visible symptom:
  selected `Pick Qr Set` simulated markers were landing in the wrong place even
  when the chosen Qr branch identity was already correct.
- Detector-view redraw was the first confirmed failure mode:
  selected simulated branch candidates were being redrawn through the
  background/manual geometry refresh path instead of from the chosen simulated
  candidate row itself.
- Detector-view selected-Qr redraw is now fixed.
- Caked-view selected-Qr redraw is still wrong and is the next active target.
- The HKL peak issue is still unresolved, but it is intentionally not the
  active target until the Qr redraw path is correct in both detector and caked
  views.
- Important non-symptom:
  the full caked simulation image itself is reported to be correct.
- High-confidence consequence:
  the bug is in per-peak overlay identity/projection, not the whole simulation
  image generation path.
- Most relevant recent commits:
  - `4620434` `fix(gui): keep selected Qr markers aligned`
  - `9be9c9b` `fix(gui): split detector peak frame prep`
  - `1b62e0d` `fix(q-group): trust normalized peak records`
  - `1519e90` `fix(runtime): use exact caked cache angles`
  - `09c7be7` `fix(manual-geometry): sync refined sim aliases`
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

- The active bug has now been narrowed from “Qr picker wrong” to
  “selected-Qr redraw wrong after correct branch identification.”
- `4620434` fixed the detector-view selected-Qr redraw path.
- The key change was in
  `geometry_manual_session_initial_pairs_display()`:
  selected simulated branch candidates are now treated as the source of truth
  for display instead of being refreshed through the background/manual geometry
  path by default.
- In detector view, the chosen simulated candidate now draws from its own
  simulation display pixel (`sim_col`, `sim_row`) instead of from refreshed
  background-frame `x`, `y`.
- In caked view, the selected candidate now prefers
  `project_peaks_to_current_view(...)`, and background/manual geometry refresh
  is only a fallback when no usable current-view simulated coordinate exists.
- Runtime wiring now passes both `refresh_entry_geometry` and
  `project_peaks_to_current_view` into the selected-Qr overlay path.
- Targeted validation for the redraw fix was green:
  - `python -S -m py_compile ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_runtime_import_safe.py`
  - focused pytest regressions for detector redraw, caked fallback, and runtime
    wiring -> `6 passed`
- Detector-view `Pick Qr Set` redraw is now considered fixed.
- The next active bug is caked-view selected-Qr redraw.
- HKL mismatch is still unresolved, but it is intentionally deferred until the
  Qr redraw path is correct in both detector and caked views.

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

### 6. Selected-Qr redraw source-of-truth fix

- `geometry_manual_session_initial_pairs_display()` was updated so selected
  simulated branch candidates are no longer refreshed through
  `refresh_entry_geometry()` before drawing by default.
- The selected simulated candidate row is now the source of truth for redraw:
  - detector view uses the chosen candidate's simulation display point
  - caked view prefers the chosen candidate projected through
    `project_peaks_to_current_view(...)`
- `refresh_entry_geometry()` remains in the path only as a fallback when the
  chosen simulated candidate does not expose a usable current-view simulated
  coordinate.
- Focused regressions now cover:
  - projection of selected active simulated display geometry
  - preserving detector simulation pixels instead of background-rotated
    detector coordinates
  - caked-view fallback when projection returns detector-only fields
  - runtime wiring for projection + refresh callbacks

What this helped:

- fixed the detector-view selected-Qr redraw bug caused by routing chosen
  simulated candidates through the background/manual display frame
- proved the selected-Qr overlay now rejects detector-only coordinates in caked
  mode and falls back to refreshed caked coordinates instead

What it did not prove:

- that the live caked-view redraw is now correct end to end
- that the separate HKL overlay issue is resolved

## What Is Most Likely Still Wrong

The active remaining bug is probably one of these:

1. A caked-view consumer still accepts detector-space fields from the chosen
   Qr branch row instead of requiring the projected caked-space fields.
2. `project_peaks_to_current_view(...)` is not yet returning the authoritative
   caked coordinates for the exact chosen row used by redraw.
3. A later caked-view overlay assembly step still overwrites or ignores the
   corrected projected row before draw.
4. The HKL overlay mismatch is a separate remaining consumer bug, but it is not
   the active target until the Qr redraw path is correct in both views.

Latest evidence:

- detector-view selected-Qr redraw is now fixed
- focused redraw regressions passed after the selected-source-of-truth change
- the full caked simulation image is still reported correct
- the live remaining target is caked-view selected-Qr redraw, not branch
  identification
- HKL mismatch still exists, but work is intentionally paused there until the
  Qr redraw path is stable in both views

The highest-value suspicion is still:

- some caked-view redraw consumer is still treating the chosen projected
  simulated row as disposable and is either accepting detector-frame aliases or
  overwriting the row after the manual-geometry projector already did the right
  thing

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

The next debugging pass should stay tightly scoped to caked-view selected-Qr
redraw:

1. Reproduce one selected Qr branch in caked view whose detector-view redraw is
   already confirmed fixed.
2. Confirm the exact chosen candidate identity for that row:
   `source_table_index`, `source_row_index`, `source_peak_index`,
   `source_branch_index`, `source_reflection_index`.
3. Verify whether `project_peaks_to_current_view([candidate])` returns the
   authoritative `caked_x/caked_y` for that exact chosen row.
4. If projection is wrong or incomplete, fix that caked projection seam first.
5. If projection is correct, trace where later overlay assembly or redraw
   discards or overwrites that projected caked row before draw.
6. Do not expand back into HKL work until detector + caked Qr redraw share the
   same source-of-truth path.

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
