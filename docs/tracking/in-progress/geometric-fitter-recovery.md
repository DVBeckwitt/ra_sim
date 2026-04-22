# Geometric Fitter Recovery

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-22

## Summary

Point-provider parity is fixed for the manual-geometry handoff layer, and the
next validation step is a bounded `new4` optimizer ladder. The manual Qr picker
saved/refined pair, the geometry-fit point provider pair, and the actual
dataset handoff row must agree before any optimizer entrypoint runs.

This closes the current point-provider bug/error scope: stale source locators
are diagnostic when the saved picker assignment still resolves semantically,
picker-owned saved/refined simulated points overwrite live/caked prefill, and
`new4` reports 7/7 provider pairs without launching the optimizer.

The Qr/Qz branch-seed bug/error scope is closed for the picker boundary:
raw-cache preview, manual toggle, manual refresh/view-change, and manual place
setup now retain one mosaic-top simulated seed per normalized branch for each
real Qr/Qz group. The lower-level collapse helper still keeps its legacy
per-branch default, and whole-group collapse remains explicit-only for fitting
callers that request it.

The caked-mode Qr/Qz detector-return bug/error scope is also closed: manual
picks made in caked `2theta,phi` space now convert back to detector view through
the same detector-display projection path used by simulation markers, including
stale-session refresh and refined simulated seed redraw. The adjacent
cross-view selection regression is closed too: detector-mode Qr/Qz hit-testing
uses the visible detector marker coordinates first, rejects stale caked
active-view display fields using explicit display-frame metadata before the
legacy numeric alias fallback, and then falls back to detector provenance, so
picking works after detector-to-caked and caked-to-detector view changes.

The detector-view Qr/Qz simulation-frame bug/error scope is now closed:
simulation hit-table rows with `native_col/native_row` are treated as
simulation-native detector pixels and projected with
`native_sim_to_display_coords`, while the background detector display adapter is
reserved for rows explicitly marked as background/native-detector. Guard tests
lock caked selection and caked-to-detector conversion behavior so this fix does
not reopen those known-good paths.

The adjacent startup/list-refresh bug is closed: normal simulation updates now
request the same hit-table/selection cache needed by Qr/Qz picking and refresh
the listed Qr/Qz peaks automatically when the grouped row content changes, so
operators no longer need to press Update Listed Peaks after simulation. Manual
Qr/Qz refresh requests now remain pending while a hit-table refresh is in
flight instead of being consumed before rows can be captured.

The `new4` rung 1 fixed-source handoff bug/error scope is closed for objective
dry-run: provider-local fixed-source rows keep provider provenance through
subset remap, the resolver accepts only branch-proven singleton stale-row
repairs for duplicate-HKL local rows, and all seven fixed rows resolve without
HKL fallback. The review hardening also keeps duplicate-HKL local rows without
branch provenance in fallback/fail state instead of silently accepting an
assigned singleton table. This is not a solve-rung feature; center fit, full
solve, and baseline runs remain blocked until a later solve-rung project starts.

The geometric optimizer hang/convergence problem is now handled by
`scripts/debug/run_new4_geometry_fit_ladder.py`, not by the old full baseline
as the first debug tool. Rung 1 objective dry-run is green with provider-fixed
source handoff, and solve rungs remain a separate next project.

Rung 2 sensitivity scan is now implemented as a residual-probe-only ladder stop
behind `--max-rung sensitivity`. It requires the rung 1 green counters first,
hashes `new4.json` before and after, reports fixed-source counters for each
plus/minus residual probe, distinguishes patched residual probing from real
`least_squares`, and writes no center/solve rung artifacts. The rung 2 review
hardening is closed: direct function calls now fail closed when rung 1 is not
green, and per-eval counters must come from live point-match summaries rather
than request-level fallback. The adjacent review bugs are also closed:
malformed rung 1 reports cannot make the aborted rung 2 report raise, `None`,
NaN, or non-numeric per-eval counters are dirty instead of zero, and
fixed-correspondence branch mismatch counts are measured from resolver payloads
instead of being hard-coded. Abort reports now also preserve strict boolean
semantics, so malformed truthy strings cannot be reported as provider identity
or point-match success.

Rung 3 one-parameter solves are now implemented behind `--max-rung one-param`.
The ladder runs fresh same-run rungs 0/1/2, reads current-run
`rung_02_sensitivity_scan.json` `active_params`, runs singleton solve requests
only, writes one JSON per attempted parameter plus a summary, and stops before
any center, paired, block, full, feature, or baseline rung. The 2026-04-21 real
run completed with partial success: eight active parameters passed, `a` timed
out cleanly, passing parameters preserved fixed-source counters, `new4.json`
was unchanged, and the provider-only guard remained green after the run. This
is not full geometric fitter validation. Rung 3 review hardening is also
closed: Rung 3 cannot start from dirty or malformed Rung 2 top-level or
per-active fixed-source counters, boolean counter payloads are rejected as
malformed counters, timeout reports emit the full one-param schema with partial
values or nulls, and clean one-param reports without heartbeat summaries fall
back to their top-level counters instead of being misclassified as pair loss.

Rung 3A `a` timeout diagnosis is complete under
`artifacts/geometry_fit_ladder/new4_a_diagnose/`. The filtered `a` runs used
fresh provider guard, Rung 1, and Rung 2 inputs, then attempted only singleton
`a`. Variants `a_nfev5_t120`, `a_nfev10_t120`, and `a_nfev20_t300` all
completed before timeout with `diagnosis_classification == "usable"`,
`last_nfev == 6`, finite residual/RMS/max-error metrics, clean fixed-source
counters, `dirty_timeout_abort == false`, and unchanged `new4.json`. No child
kill was needed because no timeout occurred. No center, paired, block, full,
feature, baseline, or non-`a` tuning artifacts were written. This justified
including `a` in the bounded Rung 4 candidate set.

Rung 4 paired solves are complete for the initial bounded pair set. Latest
result: `status == "ok"`, 5 attempted pairs, 5 passed pairs, 0 failed or
timed-out pairs, provider guard after green, and `new4.json` unchanged. Best
pair by both RMS and max error was `[corto_detector, theta_initial]`. The run
correctly stopped at Rung 4 and did not run full fit, feature rung, baseline,
GUI fit button, block solve, or any higher rung.

The repeated cold-start speed bug is fixed for ladder solve rungs. One solver
context is captured once per ladder session, then reused by one-param, pair,
block, and feature solve probes through the warm in-process worker path. Normal
solver probes also suppress debug/intersection-cache logging unless
`--diagnostic-logging` is requested. The previous cold subprocess path remains
available with `--use-subprocess` for isolation diagnostics. Measured Rung 4
pair solves improved from 315.74 seconds total across five pairs to 5.91
seconds total, about 53.4x faster. First residual time improved from a 62.07
second average to a 0.35 second average. End-to-end pair ladder runtime is
still not fully solved because one-time context capture and pre-solve setup
remain expensive.

## Validated Ladder State

The active `new4` recovery question has moved from "is the point handoff
correct?" to "which bounded solve rung should run next?" The point-provider,
fixed-source request handoff, sensitivity, singleton solve, `a` diagnosis,
caked-point reprojection, and initial paired-solve guards below are already
validated and should not be repeated unless they regress.

Speed status as of 2026-04-22:

- Bug/error: repeated per-solve cold setup is fixed for the ladder path.
- Feature: warm in-process solver reuse is implemented and covered by tests.
- Still open: initial context capture, provider guard, objective dry-run, and
  sensitivity setup still dominate whole-run wall time.

Point-provider parity is complete. The manual Qr picker saved/refined pairs,
geometry-fit `provider_pairs`, and actual fitter handoff rows match exactly for
`new4`. The provider-only report is green with
`classification == "point_provider_parity_ok"`, 7/7 manual/provider pairs, zero
dataset/provider mismatch, zero fallback, and no optimizer call. This proves
the fitter receives the correct manual-picked points. It does not prove the
optimizer converges.

Rung 1 fixed-source handoff is complete. Provider/dataset rows survive into
`GeometryFitSolverRequest` and the optimizer subset/objective as fixed-source
rows. Required green counters are:

- `provider_pair_count == 7`
- `dataset_pair_count == 7`
- `optimizer_request_pair_count == 7`
- `fixed_source_pair_count == 7`
- `fallback_row_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `missing_fixed_source_count == 0`
- `fixed_source_resolved_count == 7`
- `fallback_entry_count == 0`
- `provider_to_optimizer_identity_match == true`
- `provider_to_optimizer_point_match == true`
- `objective_eval_called == true`
- `objective_dry_run_residual_finite == true`
- `least_squares_called == false`
- `optimizer_solve_called == false`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`

Root cause fixed: local provider-fixed rows were previously falling back because
local source identity was not preserved correctly through optimizer
request/subset/remap. The duplicate-HKL provider-local remap/resolver path now
keeps all 7 rows fixed.

Rung 2 sensitivity scan is complete. The ladder can perturb each candidate
parameter without running a real solve and identify active, near-zero, and
unsafe parameters. Current validation result: `status == "ok"`, Rung 1 stayed
green, `provider_pair_count == 7`, `fixed_source_pair_count == 7`,
`fallback_entry_count == 0`, `residual_probe_called == true`,
`least_squares_called == false`, `optimizer_solve_called == false`, and
`state_hash_unchanged == true`. Current classification is 9 active, 4
near-zero, 0 non-finite, and 0 unsafe. Active parameters:

- `chi`
- `cor_angle`
- `theta_initial`
- `corto_detector`
- `zs`
- `zb`
- `a`
- `c`
- `psi_z`

Rung 3 one-parameter solves are complete enough to proceed. Each active
parameter was attempted as a singleton real solve with all other parameters
frozen. Result: `status == "ok_with_failures"`. Attempted parameters were
`chi`, `cor_angle`, `theta_initial`, `corto_detector`, `zs`, `zb`, `a`, `c`,
and `psi_z`. Passing parameters were `chi`, `cor_angle`, `theta_initial`,
`corto_detector`, `zs`, `zb`, `c`, and `psi_z`; `a` timed out; failed
parameters were none. There was no pair loss, no branch mismatch, no
no-matched-pair rejection, passing params kept fixed-source counters clean,
`new4.json` was unchanged, and the provider guard after the run was green. This
does not prove full geometric fitter validation. It only proves singleton solve
viability for the listed passing params.

Rung 3A diagnosed `a`. The previous `a` timeout was isolated with
heartbeat/timeout instrumentation. Variants `a_nfev5_t120`,
`a_nfev10_t120`, and `a_nfev20_t300` completed with `last_nfev == 6`; all
completed before timeout, no child kill was needed,
`dirty_timeout_abort == false`, fixed-source counters stayed clean,
`diagnosis_classification == "usable"`, and `new4.json` hash stayed
`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`. Key
metrics: `last_residual_norm == 2666.27241841688`,
`last_rms_px == 995.892462440596`,
`last_max_error_px == 1856.9210531158`, and
`last_parameter_value == 4.15299999995151`. This justified including `a` in
the bounded Rung 4 candidate set. Do not treat `a` as a solver pathology.

Rung 3B caked point reprojection is complete. Changing `theta_initial` or
`corto_detector` recomputes caked/fit-space coordinates for the selected
detector/background points through the live exact point projector, without
recaking the full background image. Result: `status == "pass"`,
`point_count == 7`, `exact_projector_available == true`, theta projector
signature changed, distance projector signature changed, theta-perturbed
points shifted, distance-perturbed points shifted,
`full_background_recake_call_count == 0`, provider guard before/after green,
and `new4_state_hash_unchanged == true`. Raw detector pixels do not move. The
geometry transform changes, and the selected detector/native points are
reprojected into caked/fit space. This proves the selected-point residual path
responds to `theta_initial` and `corto_detector` changes without recaking the
whole image.

Rung 4 paired solves are complete for the initial bounded pair list. The latest
run attempted 5 pairs and passed all 5 with no failures and no timeouts.
Provider guard after the run was green, `new4.json` was unchanged, and the best
pair by both RMS and max error was `[corto_detector, theta_initial]`. This is
still not full geometric fitter validation: full fit, feature rung, baseline,
GUI fit button, block solve, and higher rungs were not run and remain
unclaimed.

Rung 4 warm-path performance is validated for solve-rung overhead. The old
Rung 4 pair artifacts under `artifacts/geometry_fit_ladder/new4/20260421_235235`
showed individual pair solves at 56.95, 62.50, 59.47, 58.48, and 78.34 seconds.
The warm-path measurement under
`artifacts/geometry_fit_ladder/new4/20260422_004012` shows the same initial
pair set at 1.17, 1.17, 1.31, 1.16, and 1.11 seconds. All five warm pair solves
reported `solver_context_reused == true`, clean pass status, provider guard
after green, and unchanged `new4.json`.

Rung 5 small cumulative blocks are implemented behind
`--max-rung block|blocks`. The debug pair-backed caveat is resolved for New4:
fresh same-run run `20260422_105016` rebuilt same-run evidence and passed
Rungs 1-5. Rung 5 wrote `rung_05_block_summary.json` with `status == "ok"`,
four attempted blocks, four passed blocks, zero failed blocks, zero skipped
blocks, provider guard after blocks green, fixed-source counters clean on
passing blocks, and unchanged `new4.json`
(`F5BF185EBCFBFA8B32F161CC4BD781E177175DAD84B6FCE4D563F23CA021EF36`).
No full, feature, baseline, GUI fit, dynamic reanchor, multistart, polish,
feature rung, or Rung 6 solve was run, and `full_fitter_validated == false`.

Rung 5 status by work type:

- Feature: `--max-rung block|blocks` is implemented with per-block JSON and
  `rung_05_block_summary.json`.
- Bug/error fixed: debug `--pair-summary` can use a matching `--timestamp` for
  strict run-id evidence, local parameter usability failures no longer invalidate
  unrelated pair/block evidence, skipped dependency blocks preserve the
  solver-variable schema, and fresh same-run Rung 5 is green.
- Still open: any full/feature/baseline/GUI validation and the separately
  approved Rung 6 selected combined solve / full-candidate dry run.

## Do Not Redo

Do not redo these completed validations unless their guard output regresses:

- Point-provider parity for manual Qr picker pairs, provider pairs, and fitter
  handoff rows.
- Provider-only `new4` report with no optimizer call.
- Rung 1 fixed-source request/objective dry-run.
- Rung 2 sensitivity scan.
- Rung 3 singleton solves for active parameters.
- Rung 3A `a` timeout diagnosis.
- Rung 3B caked point reprojection guard.
- Rung 4 initial paired solves.
- Rung 5 fresh same-run cumulative blocks.

## Next Rung

Rung 5 fresh same-run is green for New4 ladder validation. The next approved
rung is separate: Rung 6 selected combined solve / full-candidate dry run. Do
not run full, feature, baseline, GUI fit button, dynamic reanchor, multistart,
polish, broad parameter tuning, or a higher rung until its own gate is explicit.

Allowed parameter set for Rung 4:

- `chi`
- `cor_angle`
- `theta_initial`
- `corto_detector`
- `zs`
- `zb`
- `a`
- `c`
- `psi_z`

Validated initial pair list:

- `[a, c]`
- `[chi, cor_angle]`
- `[theta_initial, cor_angle]`
- `[corto_detector, theta_initial]`
- `[zs, zb]`

Optional later pairs were not part of the green initial Rung 4 result. Treat
them as unvalidated candidates for a future explicit pair-expansion step, not
as repeat work:

- `[c, psi_z]`
- `[a, psi_z]`
- `[corto_detector, c]`

Do not run `[a, c, psi_z]` yet. It remains a dependency-blocked Rung 5 block.

Rung 4 pass requirements per pair:

- `least_squares_called == true`
- `optimizer_solve_called == true`
- `fixed_source_pair_count == 7`
- `fixed_source_resolved_count == 7`
- `fallback_row_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `missing_fixed_source_count == 0`
- `fallback_entry_count == 0`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`
- `provider_to_optimizer_identity_match == true`
- `provider_to_optimizer_point_match == true`
- finite residuals
- no "No matched peak pairs were available for the fitted solution." rejection
- `after_rms_px <= before_rms_px + 0.25`
- `after_max_error_px <= before_max_error_px + 1.0`
- `new4.json` unchanged
- if pair contains `theta_initial` or `corto_detector`,
  `caked_point_reprojection_guard_ok == true`

## Still Not Validated

Full geometric fitter validation is not yet claimed. Baseline completion is not
yet claimed. RMS/max global improvement is not yet claimed. Feature rungs,
the `[a, c, psi_z]` block, and full solves remain unclaimed. The GUI fit button
is not the validation path. `run_geometry_fit_quality_baseline.py` is not the
immediate next step.

## Current State

- Provider selection preserves picker-owned saved/refined background and
  simulated coordinates as provider coordinates and dataset handoff rows.
- A saved source identity resolves when it has a picker-owned saved simulated
  point and the live row matches normalized HKL plus branch/group semantics.
  Exact locator mismatch is recorded as `stale_source_identity_diagnostic`,
  not fallback.
- Missing saved simulated point still forces explicit fallback with
  `fallback_reason == "missing_saved_simulated_point"`.
- Picker-owned provider simulated points always overwrite live overlay/caked
  prefill in `initial_pairs_display`; non-picker-owned live projection keeps
  the existing no-overwrite behavior.
- The caked overwrite regression uses only `refined_sim_caked_x/y` for the
  saved simulated point, so it exercises the caked-frame branch directly.
- The canonical `new4` parity test passes with seven manual pairs, seven
  provider pairs, no fallback, matching identities, matching saved/refined
  points, matching frames, and optimizer call count zero.
- The ladder runs the provider guard first, then blocks objective dry-run before
  `least_squares` when the request or objective resolver would use fallback
  fixed-source rows.
- Real `new4` rung 1 objective dry-run is green as of 2026-04-21: 7
  provider/dataset/request/fixed rows, 0 request fallback/missing rows, 7
  fixed-source objective resolutions, 0 objective fallback entries, exact
  provider-to-optimizer identity/point match, finite dry-run residual,
  `matched_pair_count == 7`, `missing_pair_count == 0`,
  `branch_mismatch_count == 0`, `least_squares_called == false`, and
  `optimizer_solve_called == false`.
- Coordinate parity is closed through the optimizer request as of 2026-04-22:
  visual truth, provider pairs, manual point pairs, initial display pairs,
  `measured_for_fit`, `spec["measured_peaks"]`, and
  `GeometryFitSolverRequest.measured_peaks` all match for seven `new4` pairs
  without `least_squares`, solver entrypoints, or `new4.json` mutation.
- Optimizer-request capture failure is now an incomplete diagnostic state, not
  a visual/backend frame mismatch. Failed capture leaves the optimizer request
  un-compared, keeps `optimizer_request.measured_peaks` out of
  `surfaces_compared`, records `solver_request_capture_failed`, recommends
  `optimizer_request_capture`, and returns `ok == false`. Runs without
  `--include-optimizer-request` report `not_requested` and continue to judge
  provider/dataset surfaces normally.
- The dataset-to-optimizer bridge copies provider canonical identity and
  measured-point fields into the optimizer request, and the optimizer
  subset/resolver preserves provider-local fixed-source rows through objective
  matching without HKL fallback.
- Duplicate-HKL provider-local subset remap now records whether the assignment
  came from branch-aware allocation. Singleton stale-row repair is limited to
  that branch-proven path and reports actual row diagnostics separately from
  requested branch/peak identity.
- Qr/Qz UI preview and manual seed paths keep both detector-side branch
  representatives for each real Qr/Qz group, selecting only the mosaic-top row
  within each branch and preserving branch/reflection/ray provenance on the
  kept rows.
- Raw cache rows that share a Qr/Qz group but differ by
  `branch_id`, `source_branch_index`, or `source_reflection_index` collapse to
  branch representatives before initial drawing, before manual session storage,
  and after refresh. Ungrouped rows with `q_group_key is None` remain separate.
- `collapse_geometry_fit_simulated_peaks(..., one_per_q_group=True)` remains
  available for explicit whole-Qr/Qz-group collapse, but default and Qr/Qz UI
  wrapper behavior remain branch-aware.
- Caked-mode manual Qr/Qz placements store detector display/native/caked fields
  that round-trip through the same LUT/rotation path as the live simulation
  marker projection. Refresh now trusts authoritative caked `2theta,phi` fields
  over stale detector fields.
- Detector-mode Qr/Qz selection now clicks the same visible detector marker
  position that simulation draws, while rejecting stale caked display fields as
  detector click coordinates. Structural cross-view tests cover detector to
  caked, caked to detector, stale caked-cache candidates with valid detector
  provenance, and visible detector-display coordinates that differ from raw
  provenance while carrying caked metadata. Projected rows tag `display_frame`
  as detector or caked so equal numeric detector/caked values do not hide a
  valid detector marker. Peak-selection hit testing now shares the manual
  detector-coordinate resolver, so legacy `x/y` and `simulated_x/y` caked
  aliases cannot be matched as detector pixels.
- Simulation-native Qr/Qz detector rows now keep the simulated detector image
  frame through manual projection, peak overlay restoration, and geometry
  fallback normalization. With divergent detector/simulation rotations on a
  non-square image, the sim projection wins and the background-detector rotated
  point is ignored unless the row is explicitly tagged as a background/native
  detector row.
- Listed Qr/Qz peak rows update automatically after simulation row content
  changes. The manual Update Listed Peaks action remains available, but it is
  no longer required before detector-mode Qr/Qz picking, and pending manual
  refresh is not consumed until the listing snapshot is actually captured.
- Rung 2 residual sensitivity scan runs with `--max-rung sensitivity` only
  after rung 1 reports 7 fixed rows, zero fallback/missing rows, a finite dry
  residual, and no `least_squares` or optimizer solve call. It reports active,
  near-zero, non-finite, and unsafe parameters without mutating `new4.json`.
  Direct `run_sensitivity_scan` calls now abort before probing if rung 1 is
  missing or not green. Each moved base/plus/minus residual eval must carry a
  live `point_match_summary`; missing or dirty counters make the parameter
  unsafe. Fixed-correspondence summaries now report real branch mismatch counts
  from the resolved branch. The 2026-04-21 real `new4` scan at
  `artifacts/geometry_fit_ladder/new4/20260421_183827` reports rung 1 green,
  rung 2 `status == "ok"`, 9 active parameters, 4 near-zero parameters, 0
  non-finite parameters, 0 unsafe parameters, `state_hash_unchanged == true`,
  and no center/solve rung artifacts. Abort-report booleans are strict
  `is True` checks, so malformed string values remain failed in both
  `rung_1_failures` and the aborted report body.
- Rung 3 one-parameter solve run
  `artifacts/geometry_fit_ladder/new4/20260421_193603` used the current-run
  rung 2 active list only: `chi`, `cor_angle`, `theta_initial`,
  `corto_detector`, `zs`, `zb`, `a`, `c`, and `psi_z`. Rung 2 was green with 9
  active, 4 near-zero, 0 non-finite, and 0 unsafe parameters. Rung 3 summary
  status is `ok_with_failures`: passed params are `chi`, `cor_angle`,
  `theta_initial`, `corto_detector`, `zs`, `zb`, `c`, and `psi_z`; failed params
  are none; timed-out params are `a`; skipped params are none.
- Every passing one-param solve reported `least_squares_called == true`,
  `optimizer_solve_called == true`, 7 fixed-source pairs, 0 fallback rows, 0
  fixed-source resolution fallback, 0 missing fixed source, 7 resolved fixed
  sources, 0 fallback entries, 7 matched pairs, 0 missing pairs, 0 branch
  mismatches, provider identity/point match true, and
  `state_hash_unchanged == true`. `a` wrote a timeout partial JSON after
  120.09 seconds and the ladder continued cleanly.
- Rung 3 best single parameter by RMS and max error was `corto_detector`
  (`after_rms_px == 704.4849611916295`,
  `after_max_error_px == 1243.9093211467562`). Summary flags:
  `any_timeout == true`, `any_pair_loss == false`,
  `any_branch_mismatch == false`, `any_no_matched_peak_rejection == false`,
  `state_hash_unchanged == true`, and `provider_guard_after_ok == true`.
- Rung 3 review findings are closed in the ladder/test blast zone. The
  one-param entry gate now requires the full fixed-source/provider contract at
  both the Rung 2 top level and each active parameter entry, missing or boolean
  counter fields fail as `sensitivity_not_green`, timeout partial JSON keeps
  all required report fields present, clean top-level one-param reports no
  longer become false pair-loss failures when heartbeat/point-summary data is
  absent, and all-active metric failures keep
  `failure_reason == "no_one_param_solve_passed"`.
- Rung 3A `a` timeout diagnosis is complete:
  `artifacts/geometry_fit_ladder/new4_a_diagnose/variant_summary.json` reports
  `status == "ok"` and `diagnosis_classification == "usable"`. Attempted
  variants were `a_nfev5_t120`, `a_nfev10_t120`, and `a_nfev20_t300`; all three
  report `param_name == "a"`, `status == "ok"`, `last_nfev == 6`,
  `heartbeat_count == 6`, finite `last_residual_norm`, `last_rms_px`, and
  `last_max_error_px`, clean fixed-source counters at the last heartbeat, no
  fixed-source counter failures, `dirty_timeout_abort == false`, and
  `state_hash_unchanged == true`. The solve completed before timeout in all
  variants, so `child_process_killed_cleanly` is not applicable rather than
  dirty. Non-selected active params were recorded as `filtered_params`, not
  failures. The diagnose directory contains no rung 4/5/6, center, paired,
  block, full, feature, or baseline artifacts.
- Rung 3B caked point reprojection is complete. The live exact point projector
  recomputes fit-space coordinates for selected detector/native points when
  `theta_initial` or `corto_detector` changes, without recaking the full
  background image. The guard passed with 7 points, projector signatures and
  perturbed points changed, full background recake count 0, provider guards
  green before/after, and unchanged `new4.json`.
- Rung 4 paired solves are complete for the initial bounded pair list:
  `[a, c]`, `[chi, cor_angle]`, `[theta_initial, cor_angle]`,
  `[corto_detector, theta_initial]`, and `[zs, zb]`. Summary status was
  `ok`: attempted 5, passed 5, failed 0, timed out 0. Provider guard after was
  green, `new4.json` was unchanged, and best pair by both RMS and max error was
  `[corto_detector, theta_initial]`. The run intentionally did not create full
  fit, feature rung, baseline, GUI fit, block, or higher-rung validation.

## Next Actions

- Treat Rung 3 one-parameter solves, the Rung 3A `a` diagnosis, Rung 3B caked
  point reprojection, and Rung 4 initial paired solves as complete. The next
  patch should use the recorded Rung 4 result instead of repeating the same
  pair discovery run.
- Treat warm solve-rung reuse as implemented. Do not reintroduce one Python
  subprocess or one fresh solver context per candidate unless explicitly
  running `--use-subprocess` for diagnostics.
- Profile the remaining one-time setup cost before more speed work. Current
  whole-run wall time is still dominated by context capture, provider guard,
  objective dry-run, and sensitivity setup rather than pair optimizer math.
- Keep provider logic closed unless the provider-only parity gate regresses.
- Keep Qr/Qz branch seed behavior closed unless raw-cache preview, manual
  toggle, refresh, or place setup regresses to either every raw ray or one
  whole-group-only ray.
- Keep Qr/Qz listing/selection behavior closed unless a simulation update stops
  refreshing listed Qr/Qz peaks automatically, or detector-mode clicks again
  report no Qr/Qz set when clicking a visible simulated Qr/Qz marker.
- Keep detector-view Qr/Qz simulation-frame behavior closed unless simulation
  hit-table rows again project through the background detector adapter by
  default, or stale caked display fields become detector click targets.
- Keep caked-to-detector Qr/Qz return behavior closed unless the same
  simulated `2theta,phi` seed no longer redraws at the same detector marker
  position after switching view or refreshing, or Qr/Qz seed selection stops
  recognizing the same visible marker after switching detector/caked views.
- Keep objective rung 1 as a guard: any request/objective fallback row must stop
  before `least_squares`.
- Do not run block, full, feature, baseline, GUI fit button, broad parameter
  tuning, higher rungs, or loosen fallback rules without an explicit next-rung
  gate.

## Point-Provider Stop Criteria

- `point_provider_parity_gate.ok == true`.
- Manual picker pair count equals provider pair count.
- Provider pairs match manual picker truth on selected source identity,
  normalized HKL, branch/group, background point, simulated point, and frame.
- Actual dataset handoff rows match provider pairs.
- `new4` has 7 manual pairs and 7 provider pairs.
- No missing pairs, branch mismatches, silent fallback, or optimizer call.
- Targeted branch-group counters remain zero for unrelated projected/scored
  rows.

## Targeted Preflight Gate

- Use `manual_geometry_targeted` mode when saved manual geometry picks exist.
- Collect required branch-group keys before source-cache rebuild.
- Pass required branch-group keys to the targeted source-generation path.
- Use targeted fresh simulation when supported.
- Diagnose full fresh fallback and do not let it pass
  `targeted_performance_gate`.
- Reuse targeted projected cache on unchanged repeated preflight.
- On unchanged repeated preflight, do not fresh-simulate, rebuild full source
  rows, or project the full table.
- Report `targeted_performance_gate.ok == true` for an accepted canonical run.

For accepted targeted performance, `targeted_performance_gate.ok` may be true
only when all of these are true:

- `preflight_mode == "manual_geometry_targeted"`.
- `targeted_preflight_enabled == true`.
- `unrelated_projected_row_count_for_rebinding == 0`.
- `unrelated_scored_row_count_for_rebinding == 0`.
- `full_source_rows_built_for_rebinding == false`.
- `full_source_rows_projected_for_rebinding == false`.
- `targeted_cache_hit == true` or `targeted_simulation_used == true`.

Targeted preflight remains a secondary performance gate. It should not obscure
the primary point-provider parity result.

## Red Flags

- Point-provider tests launch the geometric optimizer.
- Ladder starts before the provider-only parity report is green.
- Rung 1 reaches `least_squares` while `fallback_row_count > 0`.
- Acceptance depends on the old full baseline before the bounded ladder finds a
  stable parameter set.
- Provider chooses a different simulated source identity than the picker saved.
- Provider silently rebinds a stale source identity without fallback
  diagnostics.
- Provider replaces saved/refined picker coordinates with live source-row
  projection in normal saved-value parity.
- `targeted_simulation_fallback_reason == "simulator_filter_not_supported"`
  during uncached fresh preflight.
- Full `733181`-row source build/projection during unchanged targeted
  preflight.
- `source_rows_projected_for_rebinding` approximately equals
  `total_source_rows_available`.
- `candidate_rows_scored_for_background_distance` includes unrelated HKL or
  branch rows.
- `source_cache_build_ready` waits for caked-view work.
- Runtime/debug selected candidates disagree.
- `resolved_source_pair_count == 0`.
- Huge candidate distances.

## Validation

### Guard Commands

Provider parity:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv
```

Provider-only report:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --point-provider-report-only --report-path artifacts/geometry_fit_gui_states/new4_point_provider_report.json
```

Rung 1 objective dry-run:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py::test_new4_rung1_direct_objective_dry_run_green_or_fail_before_solve -vv
```

Rung 2 sensitivity:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4 --max-rung sensitivity
```

Rung 3 one-param:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4 --max-rung one-param --max-nfev 20 --timeout-seconds 120
```

Rung 3A `a` diagnosis:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 5 --timeout-seconds 120

python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 10 --timeout-seconds 120

python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 20 --timeout-seconds 300
```

Rung 3B caked point reprojection:

```powershell
python scripts/debug/run_new4_caked_point_reprojection_check.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4
```

Focused point-provider gate:

```powershell
python -m py_compile ra_sim/gui/manual_geometry.py ra_sim/gui/geometry_fit.py scripts/debug/validate_geometry_preflight_rebind.py scripts/debug/run_new4_geometry_fit_ladder.py
pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv
```

Pinpoint regression gate:

```powershell
pytest tests/test_gui_geometry_fit_workflow.py::test_point_provider_stale_locator_is_diagnostic_when_saved_assignment_resolves tests/test_gui_geometry_fit_workflow.py::test_point_provider_marks_stale_saved_identity_as_fallback tests/test_gui_geometry_fit_workflow.py::test_point_provider_saved_refined_sim_point_overwrites_caked_prefill -q
```

Qr/Qz branch-seed regression gate:

```powershell
pytest tests/test_gui_geometry_q_group_manager.py tests/test_manual_geometry_selection_helpers.py -q
git diff --check
```

Caked-to-detector return and cross-view Qr/Qz selection regressions are covered
in the same gate by structural marker/candidate tests in
`tests/test_manual_geometry_selection_helpers.py`.
Automatic listed-Qr/Qz refresh is covered by runtime source guards in
`tests/test_gui_geometry_q_group_manager.py`.
Detector-view simulation-frame selection is covered by the divergent-rotation
regression in `tests/test_manual_geometry_selection_helpers.py` and the overlay
alignment contract in `tests/test_projection_alignment_contract.py`.

Provider-only validator check:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --mode full `
  --point-provider-report-only `
  --report-path artifacts/geometry_fit_gui_states/new4_preflight_report.json
```

Bounded optimizer ladder:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --output-root artifacts/geometry_fit_ladder/new4 `
  --max-rung sensitivity `
  --max-nfev 20 `
  --timeout-seconds 120
```

Rung 2 review hardening validation, 2026-04-21:

- `py_compile`: passed for manual geometry, GUI fitting, optimization, preflight
  validator, and ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after scan:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/provider-local/resolver/live-update tests: 12 passed.
- Direct real rung 1 dry-run test: passed.
- Rung 2 sensitivity tests: 15 passed.
- Real `new4 --max-rung sensitivity`: passed, wrote only rung 0/1/2 JSON,
  `status == "ok"`, `residual_probe_called == true`,
  `least_squares_called == false`, `optimizer_solve_called == false`,
  `state_hash_unchanged == true`, 9 active, 4 near-zero, 0 non-finite, 0 unsafe,
  and every moved eval used `counter_source == "point_match_summary"` with clean
  fixed-source counters.

Rung 3 one-parameter validation, 2026-04-21:

- `py_compile`: passed for manual geometry, GUI fitting, optimization, preflight
  validator, and ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after Rung 3:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/provider-local/resolver tests: 11 passed.
- Direct real rung 1 dry-run test: passed.
- Real `new4 --max-rung sensitivity`:
  `artifacts/geometry_fit_ladder/new4/20260421_193458`, status `pass`, current
  rung 2 `status == "ok"`, 9 active, 4 near-zero, 0 non-finite, 0 unsafe,
  `least_squares_called == false`, `optimizer_solve_called == false`.
- Real `new4 --max-rung one-param --max-nfev 20 --timeout-seconds 120`:
  `artifacts/geometry_fit_ladder/new4/20260421_193603`, status
  `ok_with_failures`, attempted only current-run active params, passed
  `chi`, `cor_angle`, `theta_initial`, `corto_detector`, `zs`, `zb`, `c`, and
  `psi_z`, timed out `a`, no failed params, no skipped params, no pair loss,
  no branch mismatch, no no-matched-pair rejection, `new4.json` unchanged, and
  provider guard after green.

Rung 3 review hardening validation, 2026-04-22:

- Focused one-param/review suite passed with 30 tests: stale sensitivity
  default avoidance, no-active abort, strict Rung 2 dirty/missing/bool counter
  gates, singleton `candidate_param_names`/`var_names`, same base state per
  param, schema-complete timeout partial JSON, clean timeout continuation,
  dirty timeout abort, fallback-row failure, all-fail summary reason,
  partial-success visibility, provider guard after one-param, and clean
  top-level report fallback when heartbeat/point-summary data is absent.

Rung 3A `a` timeout diagnosis validation, 2026-04-22:

- `py_compile`: passed for optimization, GUI fitting, preflight validator, and
  ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after diagnosis:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/Rung 1 guards: fixed-source tests passed, and direct real Rung 1
  dry-run tests passed.
- `--max-rung sensitivity` guard passed under
  `artifacts/geometry_fit_ladder/new4_a_diagnose/sensitivity` with
  `state_unchanged == true`.
- Focused `--one-param-filter`/timeout tests passed: 35 passed, covering parser
  acceptance, singleton `a` attempt filtering, `filtered_params`, inactive
  filter fail-before-solve, no higher-rung artifacts, timeout schema, slow/hang
  classification, dirty fixed-source heartbeat classification, dirty child kill
  abort, and unchanged `new4` hash.
- Real filtered variants attempted `a_nfev5_t120`, `a_nfev10_t120`, and
  `a_nfev20_t300`. All completed before timeout with
  `diagnosis_classification == "usable"`, `last_nfev == 6`,
  clean fixed-source counters, no dirty child kill, no dirty timeout abort,
  finite residual metrics, and unchanged `new4.json`.

Rung 4 paired-solve validation, 2026-04-22:

- Real `new4 --max-rung pairs` completed with `status == "ok"`.
- Attempted initial pairs were `[a, c]`, `[chi, cor_angle]`,
  `[theta_initial, cor_angle]`, `[corto_detector, theta_initial]`, and
  `[zs, zb]`.
- Results: 5 attempted, 5 passed, 0 failed, 0 timed out.
- Provider guard after was green, `new4.json` was unchanged, and best pair by
  both RMS and max error was `[corto_detector, theta_initial]`.
- The run intentionally did not run full fit, feature rung, baseline, GUI fit
  button, block solve, or any higher rung.

Coordinate parity closure, 2026-04-22:

- Focused coordinate diagnostic tests: 9 passed.
- `scripts/debug/diagnose_new4_visual_backend_coordinates.py
  --include-optimizer-request` reports `ok == true`,
  `classification == "visual_backend_parity_ok"`,
  `optimizer_request_compared == true`, `optimizer_request_pair_count == 7`,
  `optimizer_request_visual_parity_ok == true`, no first mismatching surface,
  `optimizer_called == false`, `least_squares_called == false`,
  `optimizer_entrypoints_called == []`, and `state_hash_unchanged == true`.
- Optimizer-request diagnostic failure semantics are closed. Focused
  `visual_backend or coordinate_diagnostic or new4_visual_backend` suite passed
  with 14 tests. Capture failure now returns
  `diagnostic_incomplete_optimizer_request_unavailable` instead of
  `frame_mismatch_detected`, and absent optimizer-request comparison reports
  `not_requested` while provider/dataset parity remains green.

Do not use `run_geometry_fit_quality_baseline.py` as the first optimizer debug
tool. Run it only after the ladder identifies a stable parameter set.

## Links

- [New4 geometric fitter recovery handoff](new4-geometric-fitter-recovery-handoff.md)
- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
