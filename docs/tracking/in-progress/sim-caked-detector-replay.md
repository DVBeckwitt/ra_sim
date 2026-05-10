# Sim caked detector replay

Status: implemented; automated validation passed; manual GUI smoke pending
Type: bug
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-10

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

2026-05-10 detector-origin fit handoff launch note: the detector-origin
manual Qr/Qz fit bug is fixed in the dataset handoff path and covered by
focused geometry-fit workflow tests. Bug/error status: resolved for stale
display/native aliases moving observed fit targets off the clicked background
points; manual GUI smoke remains pending before this tracker should close.
Feature status: no new feature, GUI control, CLI flag, public API, saved-state
field, artifact field, dependency, CI workflow, migration, deprecation, or
version bump. CI/CD status: local targeted tests, Ruff, compile, and diff
hygiene passed for the changed files; the broad dev check is still blocked by
unrelated formatter drift outside this bug slice. Shipping status: safe to ship
as a normal bug-fix stack with rollback by git revert.

2026-05-10 detector-origin caked fit target update: GUI geometry-fit dataset
handoff now reprojects detector-origin manual Qr/Qz background anchors from the
saved native detector point through the exact caked projector before installing
`fit_space_anchor_override`. The optimizer and fit overlay now receive the same
fixed caked target that the detector-to-caked projection reports for the saved
manual background point, instead of trusting stale or backfilled
`background_two_theta_deg/background_phi_deg` aliases on detector-origin rows.
Caked-origin rows keep their explicit caked target behavior.

Bug/error status: fixed for the case where amber background triangles and the
optimizer's observed target moved to stale caked coordinates while the fitted
source point stayed near the real detector-origin projection. Feature status:
no new GUI control, public API, CLI flag, saved-state schema, artifact format,
dependency, CI workflow, migration, deprecation, or version bump. Interface
status: the existing internal dataset and optimizer interfaces are preserved;
only the cached fit-space anchor source for detector-origin rows changes to
`detector_origin_exact_caked_projection`. Shipping status: focused dataset,
fit-space classification, dynamic reanchor, and optimizer point-only tests pass
locally. Rollback is a normal git revert; no data cleanup or migration is
required.

2026-05-10 detector-origin cleanup/review update: the follow-up simplification
keeps the detector-origin projection sequence behavior-preserving by collecting
all exact caked anchor projections before mutating measured, initial, and
display entries. Review status: no avoidable new file or abstraction; no public
interface, saved-state schema, artifact schema, dependency, CI/CD workflow, or
deployment process change. Migration/deprecation status: none required. Launch
status: ship as a normal bug-fix/refactor commit after targeted local gates;
rollback remains a normal git revert of the cleanup commit.

2026-05-10 detector-origin display/native frame update: detector-origin manual
Qr/Qz rows that only carry saved display-space `detector_x/detector_y` aliases
now derive their native `background_detector_x/background_detector_y` anchor
from the existing `unrotate_display_peaks(...)` result before exact caked
projection. Generic `detector_x/detector_y` fields are no longer promoted to
native detector coordinates. This fixes the remaining handoff path where the
visual detector-to-caked trace stayed near the clicked background triangle, but
the fit handoff projected the display alias as a native detector point and sent
the observed caked target toward the opposite/off-detector azimuth. Interface
status: no new public API, GUI control, saved-state field, artifact field,
dependency, migration, deprecation, or version bump; the existing internal
`background_detector_input_frame="native_detector"` contract is preserved.
Shipping status: focused detector-origin anchor, exact-projector, fit-space
classification, Ruff, compile, and diff hygiene checks pass locally. Manual GUI
smoke remains pending for the broader tracker.

2026-05-10 GUI fit-space preflight update: selected backgrounds with no
enabled manual Qr/Qz pairs now classify as `missing` instead of `detector` for
geometry-fit preflight. This keeps the GUI `All`-background fit path from
misreporting a state like `caked, empty, empty` as mixed detector/caked
fit-space. If at least one selected background has valid saved manual pairs,
the async GUI fit now filters the empty selected backgrounds out of the actual
fit job and emits a preflight skip message naming them. If every selected
background is empty, the user-facing failure still says to save manual Qr/Qz
pairs first. Real mixed detector/caked manual pairs are still rejected,
including mixed pairs within one background.

Bug/error status: fixed for the misleading GUI dialog shown when included
backgrounds have zero enabled pairs, and fixed for the blocked `All`-background
fit where the current background has saved pairs but other selected backgrounds
are still empty. Feature status: no new control, public API, saved-state
schema, CLI flag, dependency, or migration. Interface status: the internal
helper `geometry_manual_pairs_fit_space_kind(...)` has a `missing`
classification for empty enabled-pair sets, and async geometry-fit jobs include
additive `skipped_manual_pair_backgrounds` metadata for operator diagnostics.
Shipping status: focused classification, runtime async-job, and workflow
missing/mixed tests pass locally. Rollback is a normal git revert.

2026-05-10 GUI saved-manual caked solve update: GUI geometry fits for saved
manual caked Qr pairs now use the same direct fixed-source default as headless
saved-manual caked runs. The implicit GUI default no longer switches
`gamma,Gamma` fits into ladder multistart solely because fixed manual caked rows
exist; operators can still request `ladder-multistart` explicitly through the
existing seed-policy path. This keeps the common two-point GUI refine path on
the bounded direct solve instead of appearing to stall at the normalized-u
multistart phase.

Bug/error status: fixed for the slow GUI default shown by
`running normalized-u multistart solve` after a saved-manual caked preflight.
Feature status: no new GUI control, public API, CLI flag, saved-state schema,
artifact format, dependency, migration, or version bump. Interface status: the
existing internal `seed_policy` contract is preserved; only the default when it
is omitted changes to match headless. Shipping status: the import-safe GUI
runtime override regression now proves direct default and explicit ladder
opt-in behavior; full import-safe GUI runtime coverage and geometry-fit workflow
coverage pass locally, along with compile, Ruff, and diff-hygiene checks.
Rollback is a normal git revert.

2026-05-10 repo-clean follow-up: runtime geometry-fit overlay invalidation is
now import-safe when `geometry_fit_history_state` is absent, refined-only
detector fallback rows report their actual detector source label, and the
caked-select -> detector -> clear/rearm -> detector-click workflow has direct
coverage for selecting the same source identity without treating caked
projection rows as detector candidates. The QR handoff audit also records
provider-local exact-source row proof when the resolver trace contains the
matching source row, keeping the saved caked `(-1,0,10)` residual path on the
explicit source row instead of falling back to HKL ambiguity.

Bug/error status: fixed for the import-safe runtime failure, the misleading
refined-only detector source label, and the detector rearm no-hit path after a
caked Qr selection. Feature status: no operator control, public API, CLI flag,
saved-state schema, artifact format, dependency, CI workflow, migration, or
version bump. Shipping status: `tests/test_gui_runtime_import_safe.py`,
`tests/test_gui_geometry_fit_workflow.py`, and the focused manual-helper
rotation/source/rearm/`(-1,0,10)` slice pass locally. Manual detector/caked GUI
smoke remains the only broader tracker item still pending. Rollback is a normal
git revert with no data cleanup.

2026-05-10 replay rotation invariant update: caked-to-detector Qr replay now
uses the same native-detector-to-display rotation as direct detector
projection when the bound `native_detector_coords_to_detector_display_coords`
callback is unavailable. The fallback in `_project_peaks_to_view(...)` now uses
`display_rotate_k` directly instead of negating it for caked replay. The
regression covers native `(1080.0, 1900.0)`, detector shape `(3000, 3000)`, and
`display_rotate_k=-1`, and proves the replayed detector display point matches
`rotate_point_for_display(..., display_rotate_k)` while rejecting the old
inverse-rotation point.

Bug/error status: fixed in the caked replay fallback path. Feature status: no
new operator control, public API, CLI flag, saved-state schema, artifact
format, dependency, CI workflow, or release/version bump. Migration/deprecation
status: no migration required; existing saved rows are replayed through the
same runtime provenance path with corrected fallback rotation. Shipping status:
automated focused gates are green; manual detector/caked GUI smoke remains
pending for the broader tracker. Rollback is a normal git revert; no data
cleanup, feature flag, or migration step is required.

2026-05-10 detector visual authority update: detector-mode manual Qr/Qz
picking now resolves detector display points from `sim_visual_detector_display_px`
before refined/cache detector coordinates, and pairs that visual display point
with visual detector native coordinates when present. Wrong-frame rows still
fail closed: rows marked `_caked_qr_projection_cache` or `display_frame="caked"`
are not detector picker candidates even if they carry detector-looking fields.
When the grouped detector candidates contain only those invalid rows, the click
path treats the group as empty and runs bounded detector-only picker recovery
with `picker_candidates_only=True`, `reuse_only=False`, and no caked sidecar
build.

Visual detector-to-caked projection also stays in the visual alias lane:
`sim_visual_caked_deg`, `sim_visual_deg`, `sim_caked`, and visual source fields
can update, while fit/cache fields such as `sim_refined_caked_deg`,
`refined_sim_caked_x/y`, and `simulated_two_theta_deg/phi` remain fit/cache
authority.

Bug/error status: fixed in focused helper/runtime tests and direct source
authority assertions. Feature status: no new operator control, public API, CLI
flag, saved-state schema, artifact format, dependency, or CI workflow.
Migration/deprecation status: no migration required; stale saved/runtime rows
are rejected or rebound by provenance and identity at runtime. Shipping status:
local automated gates cover picker authority, wrong-frame recovery, active
session refresh, caked/detector round trips, compile, lint, and diff hygiene;
manual detector/caked GUI smoke remains pending. Rollback is a normal git
revert; no data cleanup, feature flag, or migration step is required.

2026-05-09 detector picker hard-reject update: detector-mode manual Qr/Qz
picking now treats caked projection rows as invalid detector candidates before
reading detector-looking coordinate fields. A caked projection row with
`sim_refined_detector_display_px` must return no detector picker row and no
detector grouped candidate. That keeps caked projection provenance from
blocking the bounded picker-only detector recovery path. The runtime refresh
path also keeps the picker-ready detector session path concise without changing
behavior.

Bug/error status: the wrong-frame detector admission path is fixed in focused
helper/runtime tests and direct probe evidence. Feature status: no new operator
control, public API, CLI flag, saved-state schema, artifact format, dependency,
or CI workflow. Migration/deprecation status: no migration required; existing
saved rows are interpreted at runtime by provenance/frame. Shipping status:
automated local gates are green; manual detector/caked GUI smoke remains
pending before closing this active tracker. Rollback is a normal git revert; no
data cleanup, feature flag, or migration step is required.

2026-05-08 active manual pick session refresh update: detector/caked view
toggles now refresh the active session from the current view's grouped
candidates instead of letting caked projection rows stay sticky in
`pick_session["group_entries"]`. The session refresh keeps source identity as
the match key, lets detector rows replace caked projection rows on detector
refresh, and preserves only visual caked aliases such as
`sim_visual_caked_deg` by identity. Fit/cache caked fields are no longer copied
by the visual-preservation helper.

Bug/error status: fixed in focused helper and runtime refresh tests. This
specifically covers detector -> caked -> detector and caked -> detector active
session refresh, where the prior runtime path could select detector grouped
candidates but still keep caked projection rows in session state. Feature
status: no new operator control, public API, CLI flag, saved-state schema,
artifact format, dependency, or CI workflow. Migration/deprecation status: no
migration required; existing saved rows remain compatible because the change is
runtime session refresh only. Shipping status: targeted refresh tests are
green; manual GUI smoke remains pending. Rollback is a normal git revert; no
data cleanup, feature flag, or migration step is required.

2026-05-08 visual caked resolver update: caked-view candidate display/ranking
now resolves visual caked points in the intended authority order:
explicit visual aliases first, safe current-view caked display aliases second,
and fit/cache caked fields last. Safe current-view fallback is limited to
display caked aliases such as `caked_x/y` or `raw_caked_x/y`; refined/cache
fields like `refined_sim_caked_x/y` stay fit/cache fallback and are not
reported as `current_view_caked`. Background/manual-authority rows with
background caked or detector fields are rejected as simulated visual caked
rows.

Bug/error status: fixed in focused helper tests for stale fit/cache caked
fields, background-shaped current-view rows, explicit simulated caked
projection candidates, and refined-only fit/cache fallback source authority.
Feature status: no new operator control, public API, CLI flag, saved-state
schema, artifact format, dependency, or CI workflow. Migration/deprecation
status: no migration required; old ambiguous rows remain readable and are
handled conservatively at resolver time. Shipping status: focused automated
local gates are green; the later 2026-05-09 validation pass records
`python -m ra_sim.dev check` passing after the runtime-session formatting drift
was cleaned up.
Rollback is a normal git revert; no data cleanup, feature flag, or migration
step is required.

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

Latest local validation, 2026-05-10:

- pre-fix regression evidence:
  `tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_reprojects_detector_origin_anchor_for_caked_fit`
  failed because the measured fit target stayed at stale caked aliases
  `(999.0, -999.0)` instead of the exact detector projection `(22.5, -35.5)`.
- `python -m pytest tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_reprojects_detector_origin_anchor_for_caked_fit -ra`
  passed (`1 passed`).
- `python -m pytest tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py -k "auto_caked_detector_origin or detector_origin_anchor or exact_projector or dynamic_reanchor_projects_detector_click or dynamic_reanchor_uses_exact_caked_bundle" -ra`
  passed (`10 passed, 624 deselected`).
- `python -m pytest tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py::test_stale_caked_fields_do_not_classify_or_override_detector_target tests/test_gui_geometry_fit_workflow.py::test_valid_caked_fields_classify_saved_bi_pair_as_caked_despite_detector_xy tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_refreshes_manual_pairs_from_saved_caked_angles tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_uses_saved_refined_caked_coords_without_live_source -ra`
  passed (`13 passed`).
- `python -m pytest tests/test_geometry_fitting.py -k "fit_space_anchor or qr_fit_point_only_projection" -ra`
  passed (`11 passed, 200 deselected`).
- `python -m compileall -q ra_sim/gui tests`, `python -m ruff check ra_sim/gui/geometry_fit.py tests/test_gui_geometry_fit_workflow.py`,
  `python -m ruff format --check tests/test_gui_geometry_fit_workflow.py`,
  and `git diff --check -- ra_sim/gui/geometry_fit.py tests/test_gui_geometry_fit_workflow.py docs/tracking/in-progress/sim-caked-detector-replay.md`
  passed.
- `python -m ruff format --check ra_sim/gui/geometry_fit.py` would reformat
  existing unrelated lines in that large module, so this slice left the file's
  surrounding formatting unchanged and used scoped `ruff check` plus
  `git diff --check`.
- `python -m ra_sim.dev check` is still blocked by pre-existing formatting
  drift in `ra_sim/fitting/optimization.py` and
  `ra_sim/gui/_runtime/runtime_session.py`; those files are outside this slice.
- pre-fix regression evidence: `test_caked_to_detector_replay_rotation_uses_display_rotate_k_without_detector_callback`
  failed with the old inverse point `(1900.0, 1919.0)` instead of the expected
  display-rotation point `(1099.0, 1080.0)`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "project_peaks_to_current_view_detector_replay or caked_to_detector_replay_rotation" -ra`
  passed (`4 passed, 568 deselected`).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "caked_to_detector_replay_rotation or project_peaks_to_current_view_detector_replay_drops_stale_display_cache_without_detector_display_callback or detector_picker_uses_visual_detector_display_before_refined_detector_display or detector_choose_group_uses_visual_detector_display_point or detector_picker_rejects_caked_projection_row_even_with_refined_detector_display or detector_click_retries_picker_only_when_cache_contains_only_caked_grouped_rows or apply_sim_visual_detector_fields_preserves_fit_caked_fields" -ra`
  passed (`7 passed, 565 deselected`).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_runtime_import_safe.py -k "runtime_refresh_detector_session_uses_picker_ready_cache_even_when_refinement_cold" -ra`
  passed (`1 passed, 418 deselected`).
- `python -m compileall -q ra_sim/gui tests`, `python -m ruff check ra_sim/gui/manual_geometry.py tests/test_manual_geometry_selection_helpers.py`, and `git diff --check`
  passed.

Latest local validation, 2026-05-09:

- direct probe: a caked projection row with `sim_refined_detector_display_px`
  returns `None` from `geometry_manual_detector_picker_row(...)` and `{}` from
  detector grouped-cache conversion.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_picker_grouped_candidates_does_not_fallback_to_caked_projection_rows or detector_picker_rejects_caked_projection_row_even_with_refined_detector_display or detector_click_retries_picker_only_when_cache_contains_only_caked_grouped_rows" -ra`
  passed (`3 passed, 565 deselected`).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "detector_picker or picker_candidates_only or refresh_session_replaces_caked_projection_rows" -ra`
  passed (`8 passed, 560 deselected`).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gui_runtime_import_safe.py -k "reuse_only_refresh or runtime_refresh_detector_session_uses_picker_ready_cache_even_when_refinement_cold" -ra`
  passed (`2 passed, 417 deselected`).
- `python -m compileall -q ra_sim/gui tests`, `git diff --check`, and
  `python -m ra_sim.dev check` passed.
- Full `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -x -ra`
  still stops at the unrelated
  `test_minus_1_0_10_fit_step_reduces_qr_residual` source assertion after
  `467 passed`.

Latest local validation, 2026-05-08:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "geometry_manual_candidate_helpers_prefer_caked_coords_in_caked_view or background_caked_xy_as_simulated or explicit_simulated_caked_projection_candidate or explicit_simulated_caked_fields_beat_bare_caked_xy or current_caked_display_before_stale_fit_cache or visual_caked_helper_rejects_background or refined_only_candidate_as_fit_cache_fallback" -ra`
  passed (`7 passed, 556 deselected`).
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_manual_geometry_selection_helpers.py -k "runtime_entry_display_coords or background_reference_detector_origin or background_reference_caked_origin or detector_caked_detector or caked_detector_caked or caked_to_detector_replay" -ra`
  passed (`9 passed, 554 deselected`).
- `python -m compileall -q ra_sim/gui tests` passed.
- `python -m ruff check ra_sim/gui/manual_geometry.py tests/test_manual_geometry_selection_helpers.py`
  passed.
- `python -m ra_sim.dev check` did not pass because
  `ra_sim/gui/_runtime/runtime_session.py` would be reformatted. That file was
  outside this slice and was not changed here.

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
