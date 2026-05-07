# Sim caked detector replay

Status: implemented; automated validation passed; manual GUI smoke pending
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-07

## Summary

Manual geometry had a caked-to-detector replay bypass for source-backed
simulated Qr rows. Caked selection was already using the current caked
projection cache correctly, but detector redraw could still reuse saved
detector/native aliases instead of replaying through the reverse LUT. This
follow-up keeps source identity as truth, keeps caked sim display/picking on
the current projection lookup, and routes detector replay only through:

`source identity -> current caked projection -> reverse LUT -> detector replay cache`

## Current state

Implemented in [manual_geometry.py](../../../ra_sim/gui/manual_geometry.py)
with focused regression coverage in
[test_manual_geometry_selection_helpers.py](../../../tests/test_manual_geometry_selection_helpers.py).

2026-05-07 final coordinate-authority update: detector-origin background rows
rendered in caked view now fail closed when live detector/native-to-caked
projection is unavailable. The runtime resolver no longer falls through to
stale saved `caked_x/y` or `background_two_theta_deg/background_phi_deg` for
known detector-origin rows. Manual pair creation also treats bare `caked_x/y`
as ambiguous by default: the values can populate simulated fit/cache caked
fields only when the candidate has explicit simulated caked projection
provenance, source identity, and no manual/background authority. Explicit
simulated fit/cache fields still beat bare caked aliases.

Bug/error status: fixed and covered by focused regressions for runtime redraw,
initial saved-pair redraw, background-shaped candidates, explicit simulated
caked projection candidates, and explicit simulated caked field precedence.
Feature status: no new operator control, CLI flag, saved-state schema, or
artifact format. Migration/deprecation status: no migration required; legacy
ambiguous rows are handled conservatively at read/save time and stale derived
fields remain data, not authority. Shipping status: automated local gates are
green; manual detector/caked GUI smoke remains the final operator acceptance
step. Rollback is a normal git revert; no cleanup job, data repair, or
feature-flag migration is required.

2026-05-07 round-trip replay update: detector-origin and caked-origin manual
background rows now resolve display coordinates from provenance first during
runtime view toggles. Detector-origin rows no longer let saved caked
`background_two_theta_deg/background_phi_deg` or stale refreshed `x/y` become
detector-view authority; detector redraw prefers detector/native anchors and
then projects through the current detector-display callback. Caked-origin rows
entering detector view project from the saved visual caked point through the
current caked-to-detector path instead of trusting stale detector aliases.
Background-reference redraw now fails closed as unresolved when a known-origin
row cannot be projected, instead of falling back to finite but wrong-frame
fields.

Bug/error status: fixed in focused runtime tests, including poisoned stale
`x/y`/display aliases. Feature status: no new UI, public API, cache layer,
file format, or schema migration. Migration/deprecation status: existing saved
rows remain compatible because provenance classification is handled at read and
redraw time. Shipping status: automated local gates are green; manual
detector -> caked -> detector and caked -> detector -> caked GUI smoke remains
the final operator acceptance step. Rollback is a normal git revert; no cleanup
job or data repair is required.

2026-05-07 provenance update: detector-clicked background rows no longer save
derived caked coordinates as input provenance. The explicit
`manual_background_input_origin` field is now authoritative over
`manual_background_input_frame`, so already-saved contradictory rows like
`origin="detector", frame="caked_2theta_phi"` classify as detector-origin and
take the live detector-to-caked redraw path. New detector-view placements stamp
`manual_background_input_frame="detector_display"` while still preserving
derived `background_two_theta_deg`, `background_phi_deg`, `caked_x/y`, and raw
caked replay fields. Caked-view placements continue to stamp
`origin="caked", frame="caked_2theta_phi"`.

Bug/error status: root provenance bug is fixed in code and covered by focused
regressions. Existing contradictory saved rows do not need a schema migration
because the classifier now treats explicit origin as the compatibility source
of truth. Feature status: no new user-facing feature, public API, state schema,
or cache redesign. Shipping status: targeted manual-geometry, workflow/runtime,
compile, ruff, and diff checks pass locally; manual GUI detector/caked smoke
remains the only acceptance item not run in this worktree. Rollback is a normal
git revert of this provenance patch; no data migration or cleanup job is
required.

2026-05-07 update: the cross-view manual background/simulation visual-cache
drift is fixed at helper/runtime level. Detector-origin saved background rows
entering caked view now reproject from detector/native or detector-display
truth through the live caked projection path before drawing `bg_display`; if
that projection is unavailable, the row stays unresolved instead of falling
back to stale saved caked fields. Caked-origin manual placement also drops
stale simulated caked cache aliases after saving the pair, preserving the
separate visual caked source instead of letting cache fields become visual
truth after a detector/caked view switch.

2026-05-07 follow-up: saved manual background origin detection now gives
`manual_background_input_origin` precedence over stale or conflicting
`manual_background_input_frame` values. New detector and caked placements write
both fields, so detector-origin rows remain detector-origin through
detector -> caked -> detector replay, while caked-origin rows keep their visual
caked anchor and still project to detector display when needed.

Bug/error status: focused automated coverage now proves detector-origin rows do
not redraw from stale caked fields, unresolved detector-origin caked rows fail
closed, and caked-origin saved rows return to detector view without stale
simulation-caked cache aliases. The latest follow-up also proves conflicting
legacy origin/frame rows prefer the explicit origin. Feature status: no new
operator control or public API; this is a correction to existing manual Qr/Qz
replay and display cache behavior. Scope cleanup status: unrelated generated
artifacts remain uncommitted.

2026-05-06 update: the manual Qr/Qz caked picker slice now preserves the
separate fit/cache and visual caked coordinate contracts when saving manual
pairs. `geometry_manual_pair_entry_from_candidate(...)` keeps refined/cache
caked values in `simulated_two_theta_deg`, `simulated_phi_deg`,
`refined_sim_caked_x/y`, `sim_refined_caked_deg`, and `sim_caked_display`,
while visual aliases stay on `sim_visual_caked_deg`, `sim_visual_deg`, and
`sim_caked` when present. Required caked geometry-fit projection now fails
closed unless the per-background projector can receive forced caked kwargs, and
the runtime projector callbacks accept/forward those kwargs. The main
detector/caked view toggle now clears manual pick artists, drops stale
`initial_pairs_display`, preserves only redraw-safe fitted overlay records, and
propagates unexpected cleanup errors.

Review status for this patch: focused behavior, runtime, compile, lint, diff,
and `ra_sim.dev check` validation are green. A follow-up review pass identified
three remaining cleanup items before merge-quality handoff: keep
`_geometry_fit_put_simulated_point_fields(...)` from overwriting existing
visual caked aliases, avoid wrapping compatible projector body `TypeError`s as
projector-contract errors, and trim the single-use retained GUI canvas helper.
Those review items are documented but not fixed in this commit.

2026-04-30 update: detector-mode Qr/Qz selector changes now warm the caked
projection sidecar immediately, without toggling the GUI into caked view. The
manual-pick cache can carry detector picker rows plus caked Qr projection
entries/lookups at the same time, and warmed sidecars are retained by later
compatible detector-cache reads.

What changed:

- added `resolve_sim_detector_replay_from_caked_projection(...)`;
- detector replay now triggers only for source-backed sim rows with current
  caked projection evidence;
- replay eligibility no longer depends on saved background angles or caked
  background fields;
- detector replay cache is stored in `sim_detector_anchor_x/y`,
  `sim_detector_display_col/row`, and `sim_detector_frame_provenance`;
- detector redraw/runtime projection no longer falls back to direct
  `refined_sim_*`, `native_*`, `sim_*`, or stale detector display aliases for
  caked-resolved replay rows;
- reverse-LUT failure for replay-eligible rows now clears replay aliases and
  leaves detector sim replay unresolved instead of falling through to detector,
  native, display, or background fields;
- stale replay anchors are closure-checked against the current projected caked
  point and refreshed once when needed;
- stale replay detector display cache is recomputed from the anchor/current
  detector display transform instead of being trusted blindly;
- replay helper no longer preserves stale saved detector display aliases when
  only a valid replay anchor is available and no current detector-display
  projection can be built;
- initial saved detector overlay now prefers source identity plus current caked
  projection replay over conflicting detector lookup rows;
- detector initial-build replay no longer lets detector candidate ranking
  overwrite the current caked-projection replay source once replay is eligible;
- background replay path was left separate.
- detector-mode manual-pick cache can request `build_caked_projection_sidecar`;
- sidecar cache signatures include the caked projection signature even while
  detector mode remains the primary pick mode;
- runtime Qr/Qz checkbox, bulk include/exclude, and loaded selection changes
  call a best-effort hidden caked-cache warm callback;
- the hidden warm path resolves/generated exact-caked payload metadata and
  projects Qr rows through the same caked projection machinery used by caked
  view switching, without setting `show_caked_2d_var`;
- the warmed cache reuses detector picker rows and keeps caked projection
  entries/grouped candidates/lookups available for manual picking and replay.

Bug/error status:

- original detector-mode Qr-set cache miss is fixed at helper and runtime
  wiring level;
- saved manual caked fit/cache fields and visual caked aliases are now split at
  manual-pair save time;
- required caked geometry-fit projection now fails closed instead of silently
  falling back to detector projection;
- caked-view toggle cleanup no longer swallows unexpected artist/state errors;
- code path updated across helper refresh, initial saved redraw, runtime
  detector projection, and detector-mode Qr selector cache warming;
- adjacent replay regressions found in review were patched in the same blast
  zone;
- helper-level stale-display regressions and initial-build detector-lookup
  conflicts are covered in targeted tests;
- saved-background-gating and reverse-LUT-failure replay regressions are now
  covered in targeted tests;
- focused detector-mode sidecar and Qr selector warm regressions pass;
- broader validation remains red/noisy in the current dirty worktree for
  pre-existing geometry-fit workflow and formatting failures listed below.

Feature status:

- no new visible operator control;
- detector mode stays detector while the caked sim/background coordinates are
  warmed as hidden cache data;
- this is behavior hardening for existing manual Qr/caked replay and
  detector-mode Qr-set selection.

## Next actions

- do one manual detector -> caked -> detector GUI replay check and confirm
  detector-origin background points stay anchored;
- do one manual caked -> detector -> caked GUI replay check and confirm
  visual caked background points stay anchored;
- update the external GitHub issue/project if this follow-up is being tracked
  there.

## Validation

Latest local validation, 2026-05-07:

- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "background_caked_xy_as_simulated or explicit_simulated_caked_projection_candidate or visual_caked or manual_qr_caked_saved" -ra`
  passed (`12 passed`).
- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "runtime_entry_display_coords or background_reference_detector_origin or background_reference_caked_origin or detector_caked_detector or caked_detector_caked or caked_to_detector_replay" -ra`
  passed (`9 passed`).
- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_first_qr_selection or picker_candidates_only or sidecar_prewarmed_detector_rows or cold_cache or warm_manual_qr" -ra`
  passed (`8 passed`).
- `python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "simulated_point_fields_preserves_visual_caked_aliases or internal_type_error_propagates or manual_fit_dataset and caked and projector" -ra`
  passed (`7 passed`).
- `python -m pytest -q tests/test_gui_runtime_import_safe.py -k "manual_pick_cache_wrapper or prewarm or apply_main_caked_view_toggle or toggle_caked_2d" -ra`
  passed (`17 passed`).
- `python -m compileall -q ra_sim/gui tests` passed.
- `python -m ruff check ra_sim/gui/manual_geometry.py tests/test_manual_geometry_selection_helpers.py`
  passed.
- `git diff --check` passed, with existing LF/CRLF warnings only.

Earlier local validation, 2026-05-07:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "saved_background_origin_prefers_detector_origin_over_caked_frame or detector_origin_caked_display_prefers_projection_over_conflicting_frame or detector_origin_caked_display_does_not_fallback_to_stale_caked_fields or detector_caked_detector_replay_preserves_detector_origin_anchor or caked_detector_caked_replay_preserves_visual_caked_anchor or place_selection_at_detector_pick_saves_projected_caked or saves_background_qr_reference_without" -ra`
  passed (`7 passed`).
- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_first_qr_selection or picker_candidates_only or sidecar_prewarmed_detector_rows or cold_cache or warm_manual_qr or saved_background_origin or detector_origin_initial_pairs_display or runtime_entry_display_reprojects_detector_origin_before_stale_caked_fields or manual_qr_caked_saved or caked_to_detector_replay or visual_caked or runtime_projection_uses_sim_detector_adapter_after_caked_view or caked_then_detector_with_stale_caked_cache or caked_manual_seed_returns_to_same_detector_visual_position or project_peaks_to_current_view_detector_replay" -ra`
  passed (`29 passed`).
- `python -m pytest -q tests/test_gui_runtime_import_safe.py -k "manual_pick_cache_wrapper or prewarm or apply_main_caked_view_toggle or toggle_caked_2d" -ra`
  passed (`17 passed`).
- `python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "simulated_point_fields_preserves_visual_caked_aliases or internal_type_error_propagates or manual_fit_dataset and caked and projector" -ra`
  passed (`7 passed`).
- `python -m compileall -q ra_sim/gui tests` passed.
- `python -m ruff check ra_sim/gui/manual_geometry.py tests/test_manual_geometry_selection_helpers.py`
  passed.
- `git diff --check` passed, with existing LF/CRLF warnings only.

Additional provenance validation, 2026-05-07:

- `python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "saved_background_origin or detector_origin_initial_pairs_display or manual_qr_caked_saved or detector_caked_detector_replay or caked_detector_caked_replay or caked_to_detector_replay or visual_caked or detector_pick_saves_projected_caked_coordinates or saves_background_qr_reference_without_hkl or back_projects_caked_pick_to_detector_space or caked_background_refines_to_peak_top" -ra`
  passed (`21 passed`).
- `python -m ruff check ra_sim/gui/manual_geometry.py tests/test_manual_geometry_selection_helpers.py`
  passed.
- `python -m compileall -q ra_sim/gui tests` passed.
- `git diff --check` passed, with existing LF/CRLF warnings only.

Latest local validation, 2026-04-30:

- `python -m compileall ra_sim tests` passed.
- `python -m ruff check ra_sim/gui/manual_geometry.py ra_sim/gui/geometry_q_group_manager.py ra_sim/gui/_runtime/runtime_session.py tests/test_manual_geometry_selection_helpers.py tests/test_gui_geometry_q_group_manager.py` passed.
- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "detector_manual_pick_cache_sidecar or detector_manual_pick_cache_without_sidecar or detector_manual_pick_cache_reuses_warmed_sidecar" -ra` passed (`3 passed`).
- `python -m pytest tests/test_gui_geometry_q_group_manager.py -k "warm_caked_cache or checkbox_side_effects_update_status or bulk_enable_side_effects_cover_empty_and_live_refresh" -ra` passed (`4 passed`).
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra` passed
  (`321 passed`).
- `git diff --check` passed, with existing CRLF warnings only.

Known remaining local blockers:

- Full `tests/test_gui_geometry_q_group_manager.py -ra` has one unrelated
  collapse-helper failure; the same `collapsed_count=0` behavior reproduces
  from `HEAD:ra_sim/gui/geometry_q_group_manager.py`.
- Full `tests/test_manual_geometry_selection_helpers.py -ra` timed out locally
  after 304 seconds.
- Full `tests/test_gui_geometry_fit_workflow.py -ra` remains red in this
  dirty tree; last-failed rerun reported nine geometry-fit/finalizer/trace
  failures.
- `python -m ra_sim.dev check` stops at `ruff format --check` because the
  current worktree has formatting drift in pre-existing touched files.

Older pending validation:

- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "sim_replay or detector_replay or initial_pairs"`
- manual GUI replay check

## Links

- [Tracking hub](../index.md)
- [Resolved picker alignment history](../archive/sim-peak-overlay-recovery.md)
- [Resolved detector-oracle caked background picks](../archive/detector-oracle-caked-background-picks.md)
- [GUI workflow](../../gui-workflow.md)
- [Debug and cache guide](../../debug-and-cache.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
