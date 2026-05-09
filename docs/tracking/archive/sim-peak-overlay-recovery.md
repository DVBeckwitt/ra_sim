# Manual Qr/Qz and HKL picker alignment

Status: resolved
Type: bug
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-05-07

This page records the resolved detector/caked manual-picking problem for Qr/Qz
group selection and HKL selection. It replaces the earlier in-progress
`Simulated peak overlay recovery` tracker for this issue.

See also:

- [tracking hub](../index.md)
- [docs index](../../index.md)
- [investigation history](sim-peak-overlay-recovery-history.md)
- [GUI workflow](../../gui-workflow.md)
- [Debug and cache guide](../../debug-and-cache.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)

## Resolution Summary

Manual Qr/Qz group selection and HKL selection now behave the same way in
detector view and caked `(2theta, phi)` view. The final solution separated
structural simulation truth from current-view projection, then made both
pickers consume the same current-view simulated-candidate payload.

User confirmation on 2026-04-19: Qr selection is working correctly, and the
entire Qr/HKL picker problem is resolved.

Follow-up status on 2026-04-23: the caked-origin Qr saved-redraw drift is fixed
by routing source-backed caked Qr hit testing, active selected markers, and
saved-pair redraw through one detector-to-caked projection cache. Automated
helper coverage is green; live GUI recheck is still not recorded in repo docs.

Follow-up status on 2026-05-07: bare `caked_x/y` is no longer accepted as
simulated fit/cache truth unless the row has explicit simulated caked-projection
provenance. The remaining allowed use of bare `caked_x/y` is display-only:
caked-view candidate distance may read the current-view caked point when visual
aliases are absent, but saved fit/cache fields stay separate. Bug/error status:
fixed in the helper path. Feature status: no new operator workflow, no schema
migration, and no deprecation path required.

## Implementation Summary

The fix landed in four connected steps:

- move Qr/Qz group membership onto structural simulation state keyed by active
  CIF/lattice hit tables instead of live detector/caked preview rows;
- rebuild caked manual-pick candidates from stored simulation rows and
  `stored_max_positions_local` when current live rows are empty after caked
  integration refresh;
- keep detector/display aliases in detector space, keep caked aliases in
  angular space, and project caked targets from simulation-native detector
  branch pixels through the live caked simulation transform;
- route HKL hit testing and selected-marker placement through the same shared
  simulated candidate payload already used by the corrected Qr/manual picker.

## Qr/Qz Root Causes

The Qr/Qz picker had three separate failure modes that looked related but needed
different fixes.

First, caked mode could fail to list all Qr/Qz groups because the selectable
group universe was being rebuilt from live `peak_records` or caked/intersection
rows. Those rows are view-dependent and can be filtered or emptied by caked
projection or background caked-integration refresh. Detector view still had
usable rows, so the bug looked caked-specific.

Second, after the group list was restored, caked Qr/Qz click targets could still
be unavailable after background caked refresh because the picker asked the live
preview rows for candidates. The stored simulation hit tables still contained
the correct simulation state, but the caked picker was not using them as a
fallback.

Third, after caked Qr/Qz candidates were rebuilt, their `(2theta, phi)` positions
could be wrong because simulation-native detector pixels were sometimes sent
through a background/display detector adapter before caked conversion. That
applied the wrong detector frame to simulated seed pixels.

## Qr/Qz Solution

The Qr/Qz group universe is now cached from the active CIF/lattice simulation
hit-table state rather than from view-filtered live rows. The cache is tied to
the CIF, unit-cell/lattice values, and simulation hit-table content. Detector
and caked view switches do not invalidate or shrink it.

When caked live rows are empty but the simulation state is still valid, manual
Qr/Qz picking can rebuild selectable candidates from `stored_max_positions_local`
and the stored lattice/hit-table metadata. That keeps background caked
integration refreshes from erasing the Qr/Qz picker.

Caked Qr/Qz target positions are now projected from simulation-native detector
branch pixels through the same live caked simulation transform used to render
the caked simulation image. Detector/display aliases stay detector/display
coordinates. Caked angular coordinates stay in caked fields such as `caked_x`,
`caked_y`, `raw_caked_x`, `raw_caked_y`, `two_theta_deg`, and `phi_deg`.

## HKL Root Cause

The HKL picker was not following the fixed Qr/Qz picker path. Runtime HKL clicks
could bypass the Qr/manual candidate payload, and caked HKL hit testing could
fall back to detector/display coordinates when it needed the current-view caked
coordinates for the same simulated branch.

That made detector view appear correct while caked HKL selection disagreed with
the now-correct Qr/Qz selection.

## HKL Solution

The HKL picker now uses the same simulated candidate payload as the Qr/manual
picker. Runtime HKL clicks are wired through the HKL simulation-point factory
that reads from the Qr picker cache, so both pickers use the same source-row
identity and current-view projected coordinates.

In caked mode, HKL hit testing prefers true caked/angular fields from that
candidate payload and does not treat detector/display `display_col` and
`display_row` as caked positions. The selected HKL marker also uses the active
caked candidate point, matching the click hit test and the rendered caked
simulation spot.

## Guardrails

Preserve these rules when touching manual picking again:

- Qr/Qz group membership is structural simulation state, not current-view state.
- Detector/caked view switches must not change the Qr/Qz group universe.
- Simulation-native detector seed pixels must be mapped through the live caked
  simulation transform before caked hit testing.
- Detector aliases must remain detector/display coordinates.
- Caked aliases must remain caked/angular coordinates.
- HKL and Qr/Qz picking should share candidate source identity and current-view
  projection wherever possible.
- Source-backed caked Qr/Qz hit testing, active `sim_display`, and saved
  `sim_display` must resolve through the detector-to-caked projection cache.
  Background `bg_display` may use clicked/refined caked coordinates, but those
  measured fields must not move the simulated branch marker.

## Caked Qr Projection Cache Closure

On 2026-04-23, the saved/live caked Qr redraw path was closed by making
detector-native source rows the only simulated Qr provenance accepted in caked
mode. Bug/error status: fixed in the manual-geometry helper path, with scoped
runtime wiring validated. Feature status: no new operator feature; this is a
behavioral hardening of the existing manual Qr picker.

The caked Qr projection cache is built by sending canonical detector-native
source rows through the same detector-to-caked projection path used by the
working detector-selection flow. It strips stale display/refined/caked aliases
before projection and stores projected entries keyed by:

- `source_table_index`
- `source_row_index`
- `source_reflection_index`
- `source_branch_index`
- `source_ray_id`
- `branch_id`

Caked Qr hit testing now requires `caked_qr_projection_grouped_candidates`;
there is no `grouped_candidates` fallback in caked Qr mode. Active caked
selected markers draw from projection-cache entries. Saved caked redraw resolves
`source identity -> caked_qr_projection_lookup -> sim_display`; if lookup is
missing, the cache is rebuilt from canonical source rows. If the projection is
still unavailable, `sim_display` is omitted and the row is marked unresolved
internally. Alias-only saved simulated points are not used as truth.

Background placement remains separate. Caked background points refine on the
visible caked image and use saved/refined caked background fields for
`bg_display`; detector background points continue to use detector display and
the existing reverse-LUT path when crossing views.

Regression coverage in `tests/test_manual_geometry_selection_helpers.py` now
proves:

- detector-origin Qr selection projected into caked view is the oracle for
  direct caked Qr selection;
- poisoned `refined_sim_*`, `display_*`, and `sim_*` aliases do not beat native
  detector provenance;
- direct caked picking uses the caked projection cache for hit testing;
- source-backed saved redraw resolves through the caked projection cache and
  does not move the simulated Qr marker after finish;
- missing caked projection lookup becomes unresolved instead of falling back to
  stale saved aliases;
- empty caked projection cache blocks caked simulated Qr selection only, while
  detector picking and background placement remain available;
- caked background refinement moves only `bg_display`; the simulated Qr marker
  stays on the cached projected source candidate;
- changing caked axes/binning changes the caked projection cache signature.

Additional 2026-05-07 coverage proves:

- background-shaped `caked_x/y` does not populate simulated fit/cache caked
  fields on saved manual pair creation;
- explicit simulated caked projection rows may still use bare `caked_x/y` as
  fit/cache input because their provenance is unambiguous;
- explicit simulated fit/cache fields beat conflicting bare `caked_x/y`;
- caked-view candidate distance keeps a display-only fallback to current-view
  caked coordinates when visual aliases are absent.

The same validation pass kept related manual-picker guardrails green: stale
background detector anchors refresh from caked authority only when provenance
allows it, projection-cache poison rows do not overwrite current caked visual
coordinates, mask-filter failures publish diagnostic source rows without active
picker rows, and cache metadata is stored before cache state publication.

Validation recorded for this closure:

- red tests for strict cache use, alias poisoning, unresolved saved redraw, and
  axes/binning invalidation failed before the fix and passed after it;
- `python -m pytest tests/test_manual_geometry_selection_helpers.py` passed
  with 256 tests;
- `python -m pytest tests/test_gui_runtime_geometry_interaction.py` passed
  with 9 tests;
- touched-file `ruff format --check`, touched-file `ruff check`, and
  `git diff --check` passed;
- `python -m ra_sim.dev check` remains blocked by unrelated formatting drift in
  `ra_sim/fitting/optimization.py` and `tests/test_timing.py`;
- broader `tests/test_gui_geometry_fit_workflow.py` still has legacy caked
  alias/diagnostic expectations and is not claimed fixed by this manual-picker
  closure.

## Regression Contract

On 2026-04-21, `tests/test_projection_alignment_contract.py` was added as a
coordinate-level guard for the resolved projection-alignment bug. It checks both
detector and caked views without screenshot diffs:

- the HKL point and analytic Qr path land on the same transformed background
  matrix cell;
- the Matplotlib `Line2D` artist coordinates for HKL points and Qr paths match
  the displayed background axes;
- detector HKL overlays are forced through the detector-display coordinate
  adapter, while caked HKL overlays are forced through cached caked
  `(two_theta, phi)` values;
- caked Qr paths are forced through the caked projection path instead of
  leaking detector-display coordinates.

Status: the original bug remains resolved, and this work adds regression
coverage only. No production code change was needed. Negative-control source
mutations killed all six required coordinate regressions: detector rotation
off-by-one, detector HKL fallback, detector Qr raw coordinates, caked HKL
detector coordinates, caked Qr detector-display leakage, and shifted Qr
`Line2D` x data.

## Selection Cache Restore Guard

On 2026-04-21, the GUI picker path was corrected to rebuild the reduced
selection cache from main-run hit tables when Qr/Qz or HKL selection is active.
The raw-row-only plan was too weak for this workflow because it did not prove
the closest-mosaic ray choice or preserve both detector-side branch candidates.

Current runtime contract:

- selection workflows request `collect_hit_tables=True` and
  `build_intersection_cache=True` for each serialized active side;
- the reduced `stored_intersection_cache` is built from the main simulation
  result, with one simulation runner call per active side and no hidden rerun;
- raw-capture-only GUI runs keep `collect_hit_tables=True` and
  `build_intersection_cache=False`;
- preview, headless, and non-selection image-only updates keep
  `build_intersection_cache=False`;
- `build_*_intersection_cache=True` must never be sent with
  `collect_*_hit_tables=False`;
- raw rows and `peak_records` are fallback picker sources only. If selection is
  armed and the reduced cache is empty or stale, they must not suppress one
  selection-cache refresh;
- stale stored primary/secondary detector caches are not republished unless
  every serialized active side has a current detector-cache signature;
- serialized `active_peak_row_sides` from the job/result is authoritative for
  result-apply paths, with GUI weight/image recomputation allowed only for
  legacy callers that pass `None`;
- source snapshots depend on geometry/image and row content, not on volatile
  UI hit-table request state.

Regression coverage in `tests/test_gui_runtime_import_safe.py` now proves:
selection jobs request collect/build per active side, raw-only jobs request
collect/no-build, repeated arming does not enqueue duplicate equivalent work,
raw rows plus `peak_records` do not block a stale reduced-cache refresh, the
stored reduced cache feeds Qr/Qz and HKL picker rows first, branch `+/-`
candidates are preserved, detector/display/caked pixels are finite, Qr/Qz
groups are clickable at candidate pixels, and HKL nearest-pixel selection
recovers the candidate.

## Detector Wrong-Frame Cache Leak Closure

Status: fixed on 2026-05-09. Detector-mode manual Qr/Qz picking now fails
closed on caked projection rows, including rows that carry detector-looking
fields such as `sim_refined_detector_display_px`. The removed behavior was an
internal fallback from detector picker grouping to raw `grouped_candidates`;
there is no public API, saved-state schema, CLI, or config migration.

What changed:

- `geometry_manual_detector_picker_row(...)` rejects caked projection
  provenance and current caked display frames before resolving detector display
  points.
- `geometry_manual_detector_picker_grouped_candidates_from_cache(...)` no
  longer returns raw grouped rows that did not pass detector picker conversion.
- Detector clicks that first see only caked grouped rows now treat the detector
  candidate set as empty, run the existing bounded `picker_candidates_only`
  recovery, and select from rebuilt detector picker rows.
- `_refresh_geometry_manual_pick_session(reuse_only=True)` can refresh detector
  `group_entries` from picker-ready rows when `cache_ready=False` but
  `picker_candidates_ready=True`, while caked refresh remains conservative.

Bug/error status: the reported `No Qr/Qz set found within a ... px window`
wrong-frame path is covered by regression tests. The active session should now
preserve source identity and refresh into the current view instead of staying a
caked row after a detector toggle.

Validation recorded for this closure:

- direct probe: a caked projection row with `sim_refined_detector_display_px`
  returns `None` from detector picker row conversion and `{}` from detector
  grouped cache conversion;
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_picker_grouped_candidates_does_not_fallback_to_caked_projection_rows or detector_picker_rejects_caked_projection_row_even_with_refined_detector_display or detector_click_retries_picker_only_when_cache_contains_only_caked_grouped_rows" -ra`
  passed with 3 tests;
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_picker or picker_candidates_only or refresh_session_replaces_caked_projection_rows" -ra`
  passed with 8 tests;
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_runtime_import_safe.py -k "reuse_only_refresh or runtime_refresh_detector_session_uses_picker_ready_cache_even_when_refinement_cold" -ra`
  passed with 2 tests;
- `python -m compileall -q ra_sim/gui tests`, `git diff --check`, and
  `python -m ra_sim.dev check` passed.

## Main Code Paths

- `ra_sim/gui/geometry_q_group_manager.py`
- `ra_sim/gui/manual_geometry.py`
- `ra_sim/gui/peak_selection.py`
- `ra_sim/gui/_runtime/runtime_session.py`
- `ra_sim/gui/state.py`

Main regression coverage lives near:

- `tests/test_gui_geometry_q_group_manager.py`
- `tests/test_manual_geometry_selection_helpers.py`
- `tests/test_gui_peak_selection.py`
- `tests/test_gui_canvas_interactions.py`
- `tests/test_projection_alignment_contract.py`

## Acceptance Criteria Reached

- Qr/Qz detector-view selection chooses the expected group and branch.
- Qr/Qz caked-view selection chooses the expected group and branch.
- Caked Qr/Qz click targets agree with the rendered caked simulation spot.
- Background caked-integration refresh does not erase the selectable Qr/Qz group
  universe while simulation hit tables are still valid.
- HKL detector-view selection chooses the expected simulated branch.
- HKL caked-view selection uses the same current-view candidate frame as Qr/Qz
  selection.
- HKL caked markers agree with the rendered caked simulation spot.
- Detector/caked HKL and Qr overlays share displayed background coordinates at
  the matrix layer and Matplotlib artist layer.
- Selection full GUI simulations publish a reduced `stored_intersection_cache`
  built from main-run hit tables while raw-only full GUI simulations keep
  detector cache publication empty.
- Stale per-side detector caches can remain stored, but active combined detector
  cache publication is all-or-nothing across serialized active sides.
