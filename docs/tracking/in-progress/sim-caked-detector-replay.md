# Sim caked detector replay

Status: in-progress
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-06

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

- run targeted replay tests in
  `tests/test_manual_geometry_selection_helpers.py`;
- run the normal touched-area quality gate if the environment is clean enough;
- do one manual detector -> caked -> detector GUI replay check and confirm no
  post-finish jump;
- update the external GitHub issue/project if this follow-up is being tracked
  there.

## Validation

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
