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

## Next Actions

- Treat rung 3 one-parameter solves as partial-success complete. Next
  recommended rung is a separate bounded follow-up that either retries or
  diagnoses the `a` timeout, then designs the next solve rung from the passing
  singleton set only. Do not run center, paired, block, full, feature, baseline,
  or parameter tuning inside this completed rung 3 scope.
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
- Do not run solve rungs, tune parameters, loosen fallback rules, or run the old
  baseline until the separate solve-rung project begins.

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

- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
