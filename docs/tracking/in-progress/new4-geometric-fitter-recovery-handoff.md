# New4 Geometric Fitter Recovery Handoff

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-05-11

## Current status

- 2026-05-11 saved manual-pair geometry-fit handoff patch completed. Async
  geometry-fit jobs now build job-local live rows from per-background saved
  manual pairs before falling back to picker or Q-group caches, ignore warmed
  manual-picker cache data stamped for another background, and allow
  non-current worker backgrounds to consume job-local live rows when their
  requested signatures match. Saved manual pairs with refined simulated
  detector/native/caked coordinates now materialize those refined fields onto
  source-row keys before normalization, so stale legacy `sim_col/sim_row` values
  cannot override newer `refined_sim_x/refined_sim_y` data.
- Bug/error fixed: selected-background geometry fits could reuse stale live
  preview, manual-picker, or legacy source-coordinate rows instead of saved
  manual-pair refined source coordinates, especially when the selected fit
  background differed from the currently displayed GUI background. The focused
  regression now covers the stale `sim_col/sim_row` plus fresh
  `refined_sim_x/refined_sim_y` case.
- Feature/status: this is an internal handoff/cache correctness fix. It does
  not change CLI flags, config keys, saved-state schema, artifact schema, or
  public geometry-fit interfaces. No deprecation or migration is required.
- Validation status for this pass: the focused failing regression first failed
  with stale `sim_col/sim_row`, then passed after the fix. The selected
  geometry-fit handoff subset passes (`5 passed, 427 deselected`), and
  `python -m py_compile ra_sim/gui/_runtime/runtime_session.py` passes. Broader
  GUI/runtime and integration suites were not rerun for this patch.
- 2026-05-03 narrow integration-hardening pass completed. Runtime
  diagnostics now keep mappings as dicts for user-facing/in-memory payloads and
  trace records, while cache signature canonicalization is unchanged. Raw
  `sim_col_raw/sim_row_raw` derived `sim_native` is preserved through provider
  point install and orientation setup when it comes from live source rows;
  saved/refined caked display authority can still override when no finite
  raw-derived live native point exists. Optional New4 tests now use the shared
  `require_new4_state()` fixture gate, and synthetic New4 mocks include 7/7
  dynamic trial source rows without weakening the production dynamic-source
  gate.
- Bug/error status: the scoped stale exact-dict assertion failures,
  `legacy_chosen_live_row` dict-shape regression, raw display-to-native
  overwrite regression, optional New4 fixture hard-fail class, trace
  list-shaped record regression, and synthetic Rung 1 dynamic-source fixture
  failures are fixed. The broad integration marker is still not green:
  `tests/test_gui_geometry_fit_workflow.py -m integration` reports
  `586 passed, 1 skipped, 22 failed`. Remaining failures are intentionally left
  for follow-up because they touch frozen/out-of-scope areas: real New4
  Rung 1/CLI source resolution, exact-caked guard expectation updates,
  dynamic reanchor matching, headless `_signature_numeric`, saved-state
  compatibility probe callbacks, exact projector local-parameter authority,
  live-cache validator reason/count semantics, trial source row caked pool,
  targeted fallback scoring, dual-path diff expectations, and caked ROI
  fallback reason precedence.
- Validation status for this pass: `tests/test_geometry_fitting.py` passes
  (`201 passed`), the live-cache/source-rung/fit-space/disordered subset passes
  (`46 passed`), the PowerShell fast tier passes (`279 passed`),
  `python -m compileall ra_sim tests` passes, and `git diff --check` passes
  with only CRLF normalization warnings.
- 2026-05-01 manual point audit contract updated to match the final coordinate
  authority. `fit_observed_caked_deg` is the cached caked target from
  `cached_fit_space_anchor` and must match `manual_saved_caked_deg`.
  `sim_refined_x0_caked_deg` must overlap that target and residual `2θ/φ`
  components must stay near zero. The detector-native reprojection formerly
  reported as `manual_fitspace_caked_deg` is now
  `manual_detector_native_reprojected_caked_deg` and is diagnostic only.
  A mismatch between that reprojection and the cached target is no longer a
  failure.
- Bug/error fixed: the stale diagnostic assertion
  `manual_fitspace_caked_deg == fit_observed_caked_deg` was removed from the
  manual-point audit tests. The fitter was not changed. The audit now rejects
  stale visual prediction sources, asserts cached-target authority, and keeps
  generated `manual_point_audit` figures ignored/untracked.
- Validation status for this audit update: targeted projection and
  manual-point-audit tests pass (`6 passed, 485 deselected`), the focused
  export test passes, `ruff format --check` passes for
  `tests/test_manual_geometry_selection_helpers.py`, and `py_compile` passes.
  Full `python -m ra_sim.dev check` remains blocked only by pre-existing
  formatting drift in `ra_sim/fitting/optimization.py`.
- 2026-05-01 New4 manual caked Qr geometry-fit contract is validated through
  Rung 7, explicit C2, explicit 12-active headless, and default headless.
  Manual caked Qr targets stay fixed in cached `2θ/φ`; optimizer sources use
  dynamic `sim_visual_caked_deg`; residuals are degree-space
  `[Δ2θ, wrapped Δφ] = source - target`; and saved-manual-caked headless fits
  use the bounded point-only solve policy without exposing a public flag or
  re-enabling saved/manual source-coordinate fallback.
- Bug/error fixed: the historical mixed coordinate-authority failure is closed
  for the saved New4 caked Qr path. The objective now reports 7/7 fixed Qr
  pairs, `fixed_source_resolved_count == 7`, `matched_pair_count == 7`,
  `missing_pair_count == 0`, clean fallback counters, target source
  `cached_fit_space_anchor`, source `sim_visual_caked_deg`, metric
  `raw_angular_rms_deg`, unit `deg`, point-only projection enabled, `c`
  fixed/excluded, and `gamma/Gamma` bounded inside `[-90, +90]`.
- Feature/status: the coordinate audit visualizer now emits machine-checkable
  JSON plus diagnostic PNGs for objective rows, perturbations, and
  after-objective-step residual checks. The audit proves visual/manual cached
  target identity, immutable optimizer measured targets, dynamic simulated
  source authority, q-group/HKL/branch identity, and the angular residual
  contract.
- Headless status: explicit C2, explicit 12-active, and default
  `python -m ra_sim fit-geometry artifacts/geometry_fit_gui_states/new4.json`
  now accept the saved-manual-caked point-only contract. Default headless
  infers the validated policy for saved New4 caked Qr states, keeps `c`
  excluded, and uses `ladder-multistart`.
- Validation status: focused coordinate tests, manual caked helper tests,
  protected Rung workflow slices, final CLI geometry-fit tests, and compileall
  pass. `python -m ra_sim.dev check` is still blocked only by pre-existing
  formatting drift in `ra_sim/fitting/optimization.py`, which is outside this
  patch. Generated ladder/audit/temp artifacts remain ignored and uncommitted.
- 2026-04-30 New4 caked Qr fit checkpoint: coordinate contract is fixed and
  validated through Rung 5. Manual caked Qr targets now use fixed cached
  `(2theta, phi)` anchors in degrees, even when an exact caked projector is
  available. Trial simulated sources are recomputed dynamically from
  `sim_visual_caked_deg`, and caked Qr residuals are `source - target` with
  wrapped phi.
- Bug/error fixed: `_measured_fit_space_anchor()` no longer reprojects manual
  caked Qr targets through changing trial geometry. This closes the moving
  measured-target failure class where stale visual/detector aliases could leak
  into the fit path.
- Feature/status: `scripts/debug/visualize_new4_qr_fit_coordinates.py` now emits
  machine-checkable JSON plus a PNG overlay. Base, `center_x`, and
  `theta_initial` visualizer gates pass: optimizer measured anchors match cached
  targets, optimizer sources match dynamic rows, targets stay fixed under
  perturbation, and sources move when expected.
- Feature/status: private runtime-only point projection is used for solver rungs
  at and after Rung 3. It is not a public CLI/config flag, is not persisted, and
  Rung 1/Rung 2 still use the normal validation path. Full caked image/refinement
  work is skipped only on this solver path while strict q-group/HKL/branch and
  dynamic `sim_visual_caked_deg` checks remain active.
- Bug/error fixed: transient Windows child crashes before heartbeat now receive
  one narrow subprocess retry. Normal solver failures, timeouts, dirty
  fixed-source failures, non-finite residuals, and post-heartbeat crashes are not
  retried into a pass. Reports record child exit code, heartbeat/partial status,
  parameter/block name, and retry count.
- Bug/error fixed: ladder JSON writing now guards recursive payloads so the
  parent process cannot crash while writing large block summaries.
- Rung 3 status: green at
  `artifacts/geometry_fit_ladder/new4_full_validation/20260430_112053/`.
  Summary: Rung 0 pass; Rung 1 pass with 7/7 dynamic Qr rows and anchor mismatch
  0; Rung 2 pass with 12 active parameters, `c` fixed/excluded, and 0 near-zero;
  Rung 3 one-param pass with 12/12 solves, 0 failed, 0 dirty timeout.
- Rung 3A/3B status: Rung 3A `a` variants have no unresolved regression. Rung
  3B caked point reprojection passes with 7 points, native-detector projection
  input, exact caked bundle projector, and no stale aliases.
- Rung 4 status: green at
  `artifacts/geometry_fit_ladder/new4_full_validation/20260430_133237/`.
  Required pairs pass: `chi_cor_angle`, `theta_initial_cor_angle`,
  `corto_detector_theta_initial`, and `zs_zb`. `a_c` is skipped because `c` is
  fixed by the new Rung 2 contract.
- Rung 5 status: green at
  `artifacts/geometry_fit_ladder/new4_full_validation/20260430_152300/`.
  Required blocks pass: `corto_detector_theta_initial_cor_angle`,
  `chi_cor_angle_theta_initial`, `corto_detector_theta_initial_zs_zb`, and
  `a_c_psi_z`. The legacy `a_c_psi_z` block label now solves only `a, psi_z`
  and records `c` as fixed with `fixed_param_policy = rung2_inactive_fixed`.
- Remaining work for this contract patch: none. Rung 1-7, the coordinate
  contract, Rung 2 active/fixed policy, point-only projection contract, and
  headless acceptance policy are frozen unless a future audit proves a shared
  regression.
- 2026-04-29 strict full-validation checkpoint: Rung 0/provider parity remains
  green on the restored historical 7-pair fixture. The active state is
  `artifacts/geometry_fit_gui_states/new4.json`, preserved hash
  `4B59F99CA88F7DFC8BE91EB9325DFF61DAC282782AFA15C5EB4E718A671DE129`.
  The accidental local 15-pair state was preserved separately and is not New4
  ladder-compatible.
- Bug/error fixed: zero-Qr / `00l` manual source rebinding now treats branch
  identity as collapsed only for `hkl=(0,0,L)` or
  `q_group_key=("q_group","primary",0,L)`. A saved legacy branch `0` can bind
  to the live collapsed branch `1` without counting as identity drift, while
  non-00l branch mismatches still fail.
- Feature/status: headless targeted preflight now carries required manual-fit
  targets, branch-group keys, and source locators into targeted hit-table
  simulation and filtering. The performance gate is green for the latest full
  preflight attempt (`targeted_performance_gate.ok == true`) without broad
  full-source fallback.
- Bug/error fixed: full-preflight live source coverage now agrees with
  provider-resolved targets for the restored New4 7-pair fixture. The dataset
  materializes provider-backed live source rows when targeted source rows are
  otherwise absent, preserves a collapsed `00l` coverage alias for `(0,0,3)`,
  and retains non-00l branch identity for q16 branch 1. The latest full
  preflight reports `dataset_resolved_source_pair_count == 7`,
  `targeted_source_coverage_gate.ok == true`, `matched_required_branch_group_count == 7`,
  `provider_backed_source_coverage_row_count == 5`, and
  `coverage_source_present_point_missing_count == 0`.
- Bug/error still open: after source coverage passes, full preflight now fails
  later as `classification == "seam_failure"` with background/candidate
  distance gates red. Provider-only parity remains green, so the remaining
  failure is downstream of live-row coverage.
- Validation status: focused New4 source-coverage tests, focused 00l visual
  collapse tests, focused caked physical-branch collapse tests, Qr/Qz signature
  hardening tests, provider-only preflight, and the source-coverage portion of
  full preflight are green. Full preflight is no longer blocked at
  `targeted_source_coverage_failed`, but it is still not fully green because of
  the downstream seam failure. The current local reports are
  `temp/codex_new4_provider_only_live_coverage.json` and
  `temp/codex_new4_full_preflight_live_coverage.json`. Do not mark the
  geometric fitter fully validated until the seam gate and ladder reach the
  requested green gates.
- Point-provider parity is closed.
- Visual/backend coordinate parity is closed for new4.
- `GeometryFitSolverRequest.measured_peaks` coordinate parity is closed when optimizer-request capture succeeds.
- Rung 1 objective dry-run is green.
- Rung 2 sensitivity scan is green.
- Rung 3 one-parameter solves are green for bounded ladder validation.
- Rung 3A `a` diagnosis is usable.
- Rung 3B caked-point reprojection guard is green.
- Rung 4 initial paired solves are green.
- Target `(-1,0,10)` Qr/Qz point-consistency rungs are green for branches 0
  and 1 across detector visual/native, caked `2theta,phi`,
  manual/background observed, visual simulation, fit observed, and fit
  prediction values.
- Target `(-1,0,10)` optimizer objective rung is green: handoff/audit,
  optimizer dry-run, and solver callback use the same locked Qr prediction
  resolver at x0; the Qr residual block is present in caked degrees, residuals
  are `predicted - observed`, Qr weights are `[1.0]`, dry-run evaluates the
  objective without `least_squares`, and branch identity stays fixed during
  solve evaluation.
- Earlier target `(-1,0,10)` and early full Mode A Qr-only reducer evidence
  remains historical. The latest refined-center diagnostic starts after 14/14
  branch coverage and 28/28 Qr components, runs `nfev=7`, keeps branch identity
  stable, and accepts no parameter step. Theta, phi, and total norms are
  unchanged.
- Earlier full-fit decomposition remains ladder evidence for the prior
  fixed-source path. Current full Mode A claim is narrower: objective coverage,
  refined caked residual use, no stale prediction cache, and fail-closed partial
  objective gating are green. Full GUI/baseline convergence remains
  unvalidated.
- Current Qr pipeline structural classification is complete, but the current
  center-objective classification is `refinement_bin_limited`. Observed caked
  centers are recomputed from fixed detector/native points under trial geometry,
  simulated caked centers are recomputed from trial simulation output, and
  branch identity stays stable. The simulated caked peak refinement itself is
  integer-bin only, with no subpixel method.
- 2026-04-29 Q-set refinement propagation bug is fixed at the cache and
  objective boundaries. Manual refinement now rebuilds `simulated_lookup` from
  refined active rows and `caked_qr_projection_lookup` from refined caked
  projection rows before runtime cache replacement, saved-pair redraw, and fit
  handoff. Q-set fixed-source objective rows are classified as dynamic Qr rows
  without requiring visual alias fields, block nominal
  `direct_fit_space_projection` fallback, and prefer refined detector-native
  coordinates before stale nominal native/display aliases.
- Bug/error status for the Q-set refinement patch: focused detector/caked
  lookup regressions, detector-picker refined-field regressions, Q-set objective
  refined-caked residual checks, q-group branch coverage, and compile checks are
  green. The broad `-k "refined or detector_picker"` selector timed out locally
  because it includes heavy diagnostics. Repo-level `ra_sim.dev check` still
  stops at pre-existing format-check drift in `optimization.py` and unrelated
  runtime/test files, so full-tree cleanliness is not claimed by this patch.
- 2026-04-29 Q-set branch-collapse bug is fixed in the Qr/Qz selection path.
  Hit-row phi branch indices are stamped before live-source canonicalization and
  restored if canonicalization strips `source_branch_index` or
  `source_peak_index`. Mirrored-branch repair now clusters on stable detector
  coordinates, preferring refined/native detector fields before display aliases.
  Non-00l rows that still lack explicit branch metadata no longer collapse into
  one `unknown:<q_group>` bucket; collapse keys use branch, source-peak,
  source-row/reflection, then detector-native cluster identity, while 00l remains
  one canonical branch. The Qr/Qz collapse wrapper now forwards
  `one_per_q_group=True` for explicit whole-group collapse.
- Bug/error status for the Q-set branch-collapse patch: compile checks and the
  targeted Q-group branch/collapse regression slice are green. The broad
  `tests/test_manual_geometry_selection_helpers.py -k "minus_1_0_10"` selector
  timed out after 10 minutes locally. The isolated
  `test_detector_mode_qr_picker_selects_minus_1_0_10_branch_clicks` subcase
  still fails on the exact detector click/row expectation, and the same failure
  reproduces on clean HEAD `b481ee0`, so it is tracked as pre-existing rather
  than introduced by this branch-collapse patch.
- 2026-04-29 caked Qr manual physical-branch collapse is fixed for signed and
  unknown provenance rows. Caked Qr projection grouping now collapses `+x` with
  source branch `0`, `-x` with source branch `1`, and `00l` rows to one
  physical slot, so non-`00l` groups ask for two background targets instead of
  four while zero-Qr/`00l` groups still ask for one.
- Bug/error status for the caked physical-branch collapse patch: focused
  caked-branch selection, caked projection grouping, pending replacement, and
  `00l` collapse regressions are green locally. This closes the specific
  caked-picker symptom where duplicate signed/unknown provenance rows for the
  same physical branch inflated the manual-pick target count.
- 2026-04-29 Qr/Qz group cache signature hardening is fixed. Signature
  generation now handles recursive containers, mapping/sequence checks that
  raise, and iterators/indexers that fail by encoding the failure in the
  signature payload instead of crashing cache comparison.
- 2026-04-29 caked manual preflight probing is fixed to stay in caked display
  space. Finite `two_theta_deg`/`phi_deg` rows report `caked_display`, caked
  probes prefer `caked_qr_projection_grouped_candidates`, and live source-row
  fallback projects to the current caked view before building grouped
  candidates.
- 2026-04-29 detector-view Qr-set recognition bug is fixed at the picker cache
  boundary. Detector picker source selection now validates each cache source
  before returning it, prefers detector picker rows, and falls through when a
  non-empty source contains only caked/current-view rows or rows without usable
  detector display pixels. Manual pick cache construction preserves detector
  picker source rows before any caked/current-view projection, detector-mode
  cache reprojection leaves detector display/native fields intact, and projected
  geometry-fit rows only call current-view projection when caked view is active.
  Q-group entry construction now accepts explicit `q_group_key` rows even when
  HKL normalization is unavailable, using row/key Qr/Qz metadata instead of
  dropping the detector selector entry.
- Bug/error status for the detector-view Qr-set recognition patch: compile,
  ruff, diff-check, focused detector fallback/no-caked-cache tests, explicit
  q-group-without-HKL listing, and detector-display projection-gating tests are
  green. Requested `q_group_entries` and `detector_display` slices pass. The
  requested `detector_mode_qr_picker` slice is 6/7 passing; the remaining
  `test_detector_mode_qr_picker_selects_minus_1_0_10_branch_clicks` exact-click
  failure is the same clean-HEAD `b481ee0` pre-existing diagnostic noted above.
- 2026-04-29 remaining detector-view Qr picker cache-population bug is fixed.
  `build_geometry_manual_pick_cache()` no longer reuses a matching detector
  manual-pick cache unless that cache can actually produce detector picker
  source rows/candidates. Empty matching detector caches record stale reason
  `cached manual-pick detector source rows were empty; rebuilding.` and rebuild
  from source snapshots or fresh simulated rows. Runtime
  `_geometry_manual_source_rows_for_background(..., consumer="manual_pick_cache")`
  may now rebuild source-row snapshots for the current background when stored
  simulation artifacts exist, and detector manual-pick rebuilds force detector
  projection mode unless the manual picker is explicitly in caked space.
- Bug/error status for the remaining detector-view Qr picker cache-population
  patch: fixed and locally validated. Focused detector manual-pick cache
  regressions, startup/clean-start/no-caked-cache detector picker regressions,
  the user-log-shaped empty-prior-cache regression, manual-pick source snapshot
  rebuild gating, q-group manager detector/listing slices, adjacent cache reuse
  tests, and compileall are green. This closes the reported symptom where
  `Updated listed Qr/Qz peaks: N groups` could coexist with an armed detector
  manual picker that still reported no Qr/Qz source rows because it reused an
  empty manual-pick cache or refused to rebuild the missing source snapshot.
- Latest post-hardening verification run `20260422_codex_final_rungs_1_4_v5`
  passed Rungs 1->4 again after the lazy best-sample and Qr/Qz selection
  fixes; caked reprojection reported `failures: []`.
- Rung 5 small cumulative blocks are green for fresh same-run New4 ladder
  validation. Run `20260422_115256` passed four attempted blocks with zero
  failed/skipped blocks.
- Fresh Rung 7 feature-gate prerequisites are green. Run
  `20260422_rung7_feature_gate_blocks` passed 4/4 Rung 5 blocks, and
  `20260422_rung7_feature_gate_combined` passed both Rung 6 combined
  candidates, including C2
  `corto_detector/theta_initial/cor_angle/chi/zs/zb`.
- Controlled Rung 7 feature gate is green end-to-end as bounded ladder
  evidence. The passing on-disk feature-sequence comparator is
  `codex_restore_rung7_features_fix_20260423`, mirrored under
  `codex_final_features_fullseq_20260423`; the older
  `codex_final_features_20260423` artifact is stale and still carries the
  pre-fix `full_beam_polish` failure. Provider/caked/Rung 5/Rung 6 guards
  stayed green, all five Rung 7 features passed in the fixed comparator, exact
  caked evidence stayed present, and the finalizer normalized
  `residuals_finite` without masking unrelated guard failures.
- Rung 0-5 timing observability is implemented. Each current-run rung report
  gets finite timing metadata, the run directory gets `rung_timing_summary.json`,
  `--timing-report` can write an explicit copy, and timing thresholds are
  diagnostic only. Real opt-in timing run `20260422_123330` finished with
  ladder `status == "ok"`, total `26.612s`, slowest rung
  `caked_point_reprojection` at `9.572s`, no missing expected rungs, no Rung 6/7
  timing records, and zero non-finite elapsed values.
- Fast manual selected-point fit defaults are implemented. The GUI manual-point
  runtime now caps `cfg["solver"]["max_nfev"]` at 30, preserves lower valid caps,
  forces serial `workers=1` and `parallel_mode="off"` unless unsafe runtime is
  explicitly enabled, and disables identifiability diagnostics by default.
- Fast ladder lean diagnostics are implemented. Lean ladder solve rungs disable
  identifiability diagnostics unless `feature="identifiability_features"` is
  requested, and running heartbeat JSON writes sparse residual progress without
  rewriting the growing full residual trace on every residual evaluation. Final
  reports still keep the full `residual_eval_trace`.
- Manual caked geometry-fit drift is fixed for solver routing and Rung 6
  validation. Caked manual picks keep `dynamic_point_geometry_fit=True`, require
  exact caked fit-space rows, report `fit_space_projector_kind ==
  "exact_caked_bundle"`, and reject fallback/analytic-detector rows. Headless
  Rung 6 now seeds C2 from accepted C1 plus the accepted Rung 5 z/zb block.
  Real validation chain:
  `temp/codex_caked_manual_blast/final_caked_reprojection/current`,
  `temp/codex_caked_manual_blast/final_ladder_full_retry/20260422_195846`, and
  `temp/codex_caked_manual_blast/final_ladder_combined_noop/20260422_201042`.
  Rungs 0-5 passed, Rung 6 C1 improved caked metrics
  `57.6813/99.4711 -> 37.9420/99.2608`, and Rung 6 C2 accepted the seeded
  initial caked state because the optimizer candidate regressed
  `37.8529/98.9712 -> 37.8940/99.1367`; accepted final C2 metrics stayed
  `37.8529/98.9712`.
- The exact-caked preflight boundary is closed. Current Rung 2 expected
  baseline is `active_param_count=11`, `near_zero_param_count=2`; `center_x`
  and `center_y` are active because `residual_norm_base` dropped `17.32x`,
  shrinking the unchanged classifier threshold faster than their delta norms
  fell (`1.37x` and `2.15x`). This is expected, not a fitting regression. Do
  not mix it with the exact-caked path.
- New4 Mode A dynamic/refined Qr prediction is implemented and verified for
  saved state `C:\Users\Kenpo\.local\share\ra_sim\new4.json`, background index
  `0` (`Bi2Se3_5m_5d.osc`). The optimizer regenerates trial detector-space
  source rows from trial params, resolves locked Qr branch identity by durable
  key, projects through the trial caked projector, refines in simulated caked
  intensity, and computes residuals as `refined_sim_caked - observed_caked`.
  Mode A resolves 14/14 branches and 28/28 caked residual components; partial
  Qr objectives fail closed with `qr_fit_objective_incomplete=yes`.
- New4 refined-center diagnostics are green for recomputation but red for
  subpixel numerical resolution. Caked bins are `0.071355959` degrees in
  `2theta` and `0.5` degrees in `phi`; all 14 simulated refinements report
  `subpixel_refinement_method=none`,
  `subpixel_refinement_status=integer_bin_argmax`, and
  `refined_bin_center_only=True`. This explains the snapped refined values such
  as `(40.132644,-36.750000)` and `(57.472142,-10.250000)`.
- New4 dynamic baseline anchor validation is green for all 14 Mode A branches.
  The pre-fix first bad branch was `(-1,0,10)` branch 1, source table 160 row
  120. After detector-native row correction, the divergence localized to
  `B. caked_projection_mismatch`: regenerated detector/native coordinates were
  anchored, but baseline caked projection used the wrong frame. The fix keeps
  original x0 fit params as `baseline_fit_params`, prefers saved refined sim
  caked anchors at fit prep, applies a baseline caked alignment offset, uses
  native detector coordinates for source-row projection, constrains simulated
  caked refinement to a local one-bin window, and blocks optimizer start with
  `optimizer_start_blocked_reason=dynamic_baseline_anchor_mismatch` if x0
  dynamic predictions drift outside saved-anchor tolerance.
- Real full headless geometric-fit smoke was run for
  `artifacts/geometry_fit_gui_states/new4.json`, background `0`. Earlier
  seed/start-state split evidence remains useful for baseline/full-beam
  comparison, but current dynamic/refined Qr evidence is limited to complete
  Mode A prediction coverage and residual correctness, not full GUI/baseline
  convergence.
- Baseline, GUI fit button, and unrestricted feature-combination runs should
  still be treated as unvalidated.

This handoff is the bounded-through-Rung-7 feature-gate recovery state for
`new4` plus the current Qr resolver/full-objective diagnostic state. Do not use
it as approval for GUI, baseline, or unrestricted feature-combination solves.

Status by work type:

- Bug/error: multi-branch New4 Mode A Qr identity resolution is fixed. The
  earlier regenerated hit-table resolver resolved only 4/14 saved branches and
  produced 8/28 caked residual components. The durable `fit_qr_branch_key` now
  resolves all 14 branches with one dynamic candidate each and no branch-only
  fallback.
- Bug/error: stale saved visual/caked fallback is removed from active Qr
  prediction. Trial detector source rows, caked projection signatures, and
  simulated caked image signatures are tied to the objective trial params; stale
  baseline cache reuse under changed params is rejected.
- Feature: shared dynamic Qr prediction helper returns locked branch identity,
  nominal detector/native/caked coordinates, refined simulated caked
  coordinates, refinement status, params signature, detector-source signature,
  and caked-simulation signature. Handoff audit, objective dry-run, and solver
  callback use this same helper at x0.
- Feature/status: caked refinement is applied to the objective for all 28 Mode
  A components. Residual units are weighted caked degrees, with residuals
  computed as `sim_refined_caked_deg - observed_caked_deg`.
- Bug/error: baseline dynamic Qr predictions now reproduce the saved refined
  simulated peak centers for all 14 Mode A branches before any fit can start.
  The previous wrong-peak/large-phi symptom is fixed by native source-row
  projection, baseline caked anchor alignment, local refinement limits, and a
  fail-closed baseline anchor gate.
- Feature/status: Qr-only fit now starts only after full Mode A coverage and
  baseline anchor validation pass. Current refined-center objective result is
  theta `20.415070959 -> 20.415070959`, phi
  `4.053221387 -> 4.053221387`, total
  `20.813546691 -> 20.813546691`, `nfev=7`, branch identity stable, and
  accepted parameter changes `<none>`. Classification is
  `refinement_bin_limited`, not theta/phi improvement.
- Full-fit status: the complete dynamic/refined objective includes all 28 Qr
  components. Current full fit reports total
  `40.029486770 -> 40.029486770`, Qr
  `20.813546691 -> 20.813546691`, theta
  `20.415070959 -> 20.415070959`, phi
  `4.053221387 -> 4.053221387`, and non-Qr
  `31.031629300 -> 31.031629300`. It accepts no parameter step; exact reason
  for no Qr improvement is `refinement bin limited`.
- Bug/error: target `(-1,0,10)` Qr/Qz objective absence and prediction-resolver
  split are fixed. Handoff/audit, optimizer dry-run, and solver callback call
  the shared fixed-manual Qr fit resolver and agree at x0. If they diverge,
  optimizer start is blocked with
  `optimizer_start_blocked_reason=prediction_resolver_mismatch`.
- Bug/error: fixed provider-local request rows now stay locked through the
  optimizer. The resolver preserves provider-local proof, fails closed for
  ambiguous duplicate-HKL rows, and only uses saved detector-native simulation
  points after stale-row proof or canonical saved-source identity proof. Raw
  native saved pixels without canonical display/native proof still require
  stale-row proof.
- Review hardening: saved-simulation fit-space offset caching is baseline
  primed before seed scoring or least-squares solve, so seed/multistart order
  cannot decide the Qr/Qz residual alignment offset.
- Feature: objective dry-run and residual-vector audit tests now prove Qr
  residual-vector membership before solve. Full-fit decomposition reports total,
  Qr, non-Qr, line, and prior block norms before/after; current evidence is
  total `6.847163064 -> 6.731263668` and Qr
  `2.819315157 -> 2.644004804`.
- Feature/status: multi-group Qr diagnostics for `(-1,0,5)`, `(-1,0,10)`, and
  `(-1,0,16)` keep branch/source identity stable. Qr residual improvements are
  mostly in 2theta; phi residuals remain nearly unchanged because active params
  have little phi leverage.
- Feature: controlled Rung 7 passed `dynamic_reanchor`, `discrete_modes`,
  `seed_multistart`, `full_beam_polish`, and `identifiability_features` in the
  fixed comparator `codex_restore_rung7_features_fix_20260423`; the exact-caked
  path is green through bounded Rung 7 ladder evidence.
- Bug/error: exact-caked preflight ordering/harness blockers are closed; current
  Rung 2 expected baseline is `11/2` under the unchanged threshold rule and
  does not require solver, residual, runtime, or caked-routing changes.
- Full fit bug/error/status: request construction is clean, the Qr block is not
  silently dropped, and partial Qr objectives now fail closed. Current
  dynamic/refined evidence does not claim full GUI/baseline convergence.
  Remaining Qr issue is active-parameter phi sensitivity, not source identity,
  detector-space reporting, stale caked coordinates, or objective membership.
- Timing feature: current-run Rung 0-5 timing JSON and stdout table are
  available for opt-in ladder runs.
- Timing bug/error: review follow-up is closed. Timing collection excludes
  Rung 6/7 path mappings and expected IDs, Rung 5 skipped reports are timed,
  and `RA_SIM_NEW4_LADDER_TIMING_MAX_S` never gates status or exit code.
- Manual selected-point fit bug/error: default GUI fits no longer inherit
  `max_nfev: 400`, parallel orchestration, or identifiability diagnostics for a
  few selected spots. Unsafe parallel runtime and richer dynamic point fitting
  remain explicit paths.
- Manual caked fit bug/error: solver-path drift is fixed and Rung 6 validates
  same-coordinate exact-caked rows without detector fallback. Preflight
  fail-closed ordering still has four focused import-safe failures, so the
  operator-facing GUI preflight path remains in-progress.
- Ladder lean bug/error: finite-difference identifiability diagnostics no
  longer run on every fast ladder solve. The identifiability feature run remains
  the explicit diagnostic path.
- Ladder heartbeat bug/error: running heartbeat files no longer rewrite stale or
  growing `residual_eval_trace` payloads on every evaluation. Timeout progress
  keeps `last_residual_eval`, counters, timing, bounds, and solver context flags.
- GUI timing harness: gated smoke and 10-trial evidence was collected under
  `artifacts/gui_timing/20260422_130625`; 30-trial evidence stopped at a
  focused `theta10` child timeout after `defaults_30` passed.
- Not validated: baseline, GUI fit, and unrestricted feature combinations
  remain unclaimed. Full-beam validation is green only as bounded ladder
  evidence; current full-fit claim is limited to locked-source Qr contribution
  and objective decomposition diagnostics.

## GUI timing harness checkpoint

Artifact root: `artifacts/gui_timing/20260422_130625`

Prechecks passed:

- `python -m ruff check ra_sim/timing.py scripts/measure_gui_timing.py tests/test_timing.py tests/test_gui_runtime_update_trace.py`
- `python -m pytest tests/test_timing.py tests/test_gui_runtime_update_trace.py -q` -> `17 passed`
- `python -m mypy ra_sim/timing.py`

CLI source of truth:

- Uses `--scenario`, not `--preset`.
- Restored New4 scenario is `saved-state-startup --state artifacts/geometry_fit_gui_states/new4.json`; summaries record it as `defaults-restored`.
- Per-run artifacts are `summary.json`, `metadata.json`, `combined_events.csv`,
  `README.md`, and per-trial `trial_*.jsonl`/stdout/stderr files. The current
  harness output does not emit top-level `events.jsonl` or `report.csv`.
- Current summaries/events do not record RSS samples, so RSS peak and RSS growth
  were not available from these artifacts.

Batch status:

- Smoke passed for `defaults`, `theta10`, `redraw-only`, `cache-hit`, and restored New4.
- 10-trial batch passed for all five scenarios with zero `trial_failures`.
- 30-trial batch: `defaults_30` passed; `theta10_30` timed out on
  `trial_001.jsonl`, so `redraw-only_30`, `cache-hit_30`, and restored New4 30
  were not run.

10-trial timing comparison:

| Scenario | Primary span | median ms | p95 ms | max ms | events |
| --- | --- | ---: | ---: | ---: | ---: |
| `defaults_10` | startup/process launch to first visible | 6443.5 | 6636.6 | 6636.6 | 1473 |
| `theta10_10` | theta change/total change to visible | 3991.0 | 4515.6 | 4515.6 | 2281 |
| `theta10_10` | theta return/total change to visible | 1675.1 | 2000.1 | 2000.1 | 2281 |
| `redraw_only_10` | redraw/input to visible | 146.2 | 175.2 | 271.2 | 458 |
| `cache_hit_10` | theta change/total change to visible | 4805.3 | 4805.3 | 4805.3 | 325 |
| `restored_new4_10` | startup/process launch to first visible | 16562.6 | 16739.6 | 16739.6 | 1887 |

Observed timing shape:

- Slowest 10-trial spans were restored New4 startup (`~16.6s`), default startup
  (`~6.4s`), and theta change visible latency (`~4.0s`).
- Restored New4 adds about `10.1s` startup/restore overhead versus default
  startup median.
- `cache-hit_10` did not come out faster than `theta10_10`; its measured theta
  change was one counted span at `4805.3ms`, with cache-hit events showing both
  false and true states. Treat cache-hit evidence as needing focused follow-up
  before using it as a speed claim.
- `redraw_only_10` is fast and compute-free in the summary: redraw visible
  median `146.2ms`, p95 `175.2ms`.
- Repeated update IDs mostly match repeated scenario work. The focused
  `theta10_30` timeout ended after partial measurements (`14` theta changes,
  `13` returns); last events show repeated render callbacks for update `56` and
  overlay update `57`, then no stderr and a `151.849s` event gap until the 240s
  harness timeout.

## What is proven

The point handoff chain is proven:

```text
manual Qr picker saved/refined pairs
==
provider_pairs
==
manual_point_pairs
==
initial_pairs_display
==
measured_for_fit
==
spec["measured_peaks"]
==
GeometryFitSolverRequest.measured_peaks
```

This was proven without running `least_squares` or the optimizer.

Also proven:

- Visual drawn pairs match backend handoff rows.
- Optimizer request coordinate comparison passes in the dev environment.
- If optimizer-request capture fails in another environment, the diagnostic now reports `diagnostic_incomplete_optimizer_request_unavailable`, not a fake frame mismatch.

Final coordinate diagnostic fields:

- `ok == true`
- `classification == "visual_backend_parity_ok"`
- `optimizer_request_compared == true`
- `optimizer_request_pair_count == 7`
- `optimizer_request_visual_parity_ok == true`
- `optimizer_called == false`
- `least_squares_called == false`
- `state_hash_unchanged == true`

## Important historical findings

1. Stale saved simulated/source identity was originally winning over measured-background-nearest candidates.
2. Runtime and debug validator initially diverged.
3. Visual points looked right, but backend coordinates still had to be proven against them.
4. `new4_fresh_all.json` is diagnostic only, not visual truth.
5. Visual truth comes from draw-path coordinates, not provider rows.
6. Full solve was avoided until point handoff and request handoff were proven.
7. Duplicate-HKL provider-local fixed rows caused Rung 1 fallback until subset/remap/resolver handling was fixed.
8. Rung 2 now finds 11 active parameters and 2 near-zero parameters, with 0 unsafe and 0 non-finite; `center_x` and `center_y` crossed from near-zero to active because lower `residual_norm_base` shrank the unchanged classifier threshold faster than their delta norms fell.
9. Rung 5 excludes `[a, c]` as a block; `[a, c]` is Rung 4 prerequisite evidence only.
10. `[a, c, psi_z]` remains dependency-blocked until `[a, psi_z]` or `[c, psi_z]` passes.

## Artifacts

`artifacts/geometry_fit_gui_states/new4.json`

- Canonical input state.

`artifacts/geometry_fit_gui_states/new4_point_provider_report.json`

- Provider-only parity artifact.

`artifacts/geometry_fit_coordinate_diagnostics/new4/coordinate_transform_diagnosis.json`

- Visual/backend parity report.

`artifacts/geometry_fit_coordinate_diagnostics/new4/coordinate_transform_pairs.csv`

- Per-pair visual/backend coordinate deltas.

`artifacts/geometry_fit_coordinate_diagnostics/new4/coordinate_transform_overlay.png`

- Visual/backend overlay diagnostic.

`artifacts/geometry_fit_coordinate_diagnostics/new4/coordinate_transform_vectors.png`

- Vector diagnostic.

`artifacts/geometry_fit_ladder/new4/<latest>/rung_01_objective_dry_run.json`

- Rung 1 proof. Green exemplar: `artifacts/geometry_fit_ladder/new4/20260421_183827/rung_01_objective_dry_run.json`.

`artifacts/geometry_fit_ladder/new4/20260422_105016/rung_05_block_summary.json`

- Fresh same-run Rung 5 block proof. Status `ok`; attempted blocks `4`, passed
  blocks `4`, failed/skipped blocks `0`, provider guard after blocks green,
  and `new4.json` unchanged. This supersedes the earlier debug pair-backed-only
  caveat. `full_fitter_validated == false`.

`artifacts/geometry_fit_ladder/new4/20260422_115256/rung_05_block_summary.json`

- Fresh same-run Rung 5 block proof. Status `ok`; attempted blocks `4`, passed
  blocks `4`, failed/timed-out blocks `0`, provider guard after blocks green,
  caked reprojection guard path present and green, and `new4.json` unchanged.
  `full_fitter_validated == false`.

`artifacts/geometry_fit_ladder/new4/<latest>/rung_02_sensitivity_scan.json`

- Rung 2 current proof: `temp/rungs_1_7_verify/codex_final_blocks_20260423/rung_02_sensitivity_scan.json`.
- Historical pre-threshold-shrink baseline: `artifacts/geometry_fit_ladder/new4/20260421_183827/rung_02_sensitivity_scan.json`.

Old `new4_preflight_report.json` and `new4_fresh_all.json` may be stale or diagnostic only unless regenerated by the current scripts.

## Tests and commands that passed

Current New4 refined-center objective gate:

```powershell
python -m py_compile ra_sim/fitting/optimization.py ra_sim/gui/geometry_fit.py ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py
pytest tests/test_manual_geometry_selection_helpers.py -k "caked_refinement_bin_resolution or observed_trial_caked_recomputed or sim_trial_caked_recomputed or refined_objective_theta_phi_decomposition or full_fit_with_dynamic_refined_center_objective" -s -q
pytest tests/test_gui_runtime_import_safe.py -k "toggle_caked_2d" -q
```

Expected/current summary:

- `test_new4_caked_refinement_bin_resolution_and_subpixel_status`: all 14 Mode
  A branches print nominal/refined caked centers, caked bin size, nominal/local
  max pixel indices, local max intensity, window size, subpixel status, and
  refinement delta. Current status is
  `caked_refinement_integer_bin_only=yes` and
  `caked_subpixel_refinement_missing=yes`.
- `test_new4_observed_trial_caked_recomputed_from_detector_center`: observed
  detector/native pixels stay fixed while observed trial caked centers move
  under finite `corto_detector` change. Current status is
  `observed_caked_static_under_trial_geometry=no`.
- `test_new4_sim_trial_caked_recomputed_from_detector_sim`: dynamic simulated
  detector image signatures change for all 14 branches, nominal/refined caked
  centers move under trial params, and branch identity stays stable. Current
  status is `sim_refined_caked_static_under_trial_params=no`.
- `test_new4_refined_objective_theta_phi_decomposition_after_pipeline_fix`:
  Qr-only theta, phi, and total norms are unchanged over `nfev=7`; accepted
  parameter changes are `<none>`; classification is `refinement_bin_limited`.
- `test_new4_solver_with_dynamic_refined_center_objective`: full fit includes
  all 28 Qr components, preserves branch identity, accepts no parameter step,
  and reports no Qr improvement because `refinement bin limited`.
- Command results: `py_compile` passed, targeted New4 refined-center tests
  `5 passed, 406 deselected`, runtime import-safe toggle test
  `4 passed, 309 deselected`.

Provider parity:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv
```

Expected summary:

- `manual_picker_pair_count == 7`
- `point_provider_pair_count == 7`
- `missing_pair_count == 0`
- `fallback_pair_count == 0`
- `optimizer_called == false`
- `classification == "point_provider_parity_ok"`

Provider-only report:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --point-provider-report-only `
  --report-path artifacts/geometry_fit_gui_states/new4_point_provider_report.json
```

Expected summary:

- `ok == true`
- `classification == "point_provider_parity_ok"`
- `manual_picker_pair_count == 7`
- `point_provider_pair_count == 7`
- `manual_point_pair_count == 7`
- `initial_pairs_display_count == 7`
- `measured_for_fit_count == 7`
- `spec_measured_peaks_count == 7`
- `fallback_pair_count == 0`
- `optimizer_call_count == 0`

Coordinate parity:

```powershell
python scripts/debug/diagnose_new4_visual_backend_coordinates.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --provider-report artifacts/geometry_fit_gui_states/new4_point_provider_report.json `
  --background-index 0 `
  --include-optimizer-request `
  --output-dir artifacts/geometry_fit_coordinate_diagnostics/new4
```

Expected summary:

- `ok == true`
- `classification == "visual_backend_parity_ok"`
- `optimizer_request_compared == true`
- `optimizer_request_pair_count == 7`
- `optimizer_request_visual_parity_ok == true`
- `optimizer_called == false`
- `least_squares_called == false`
- `state_hash_unchanged == true`

Rung ladder through sensitivity:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --output-root artifacts/geometry_fit_ladder/new4 `
  --max-rung sensitivity
```

Expected Rung 1 summary:

- `status == "ok"`
- `pass == true`
- `objective_dry_run_residual_finite == true`
- `fixed_source_pair_count == 7`
- `fixed_source_resolved_count == 7`
- `fallback_row_count == 0`
- `provider_row_fallback_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `fallback_entry_count == 0`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`
- `least_squares_called == false`
- `optimizer_solve_called == false`

Expected Rung 2 summary:

- `status == "ok"`
- `pass == true`
- `active_param_count == 11`
- `near_zero_param_count == 2`
- `non_finite_param_count == 0`
- `unsafe_param_count == 0`
- `residual_probe_called == true`
- `least_squares_called == false`
- `optimizer_solve_called == false`
- `state_hash_unchanged == true`
- `provider_pair_count == 7`
- `fixed_source_pair_count == 7`
- `fallback_entry_count == 0`
- `center_x` and `center_y` are active because lower `residual_norm_base`
  shrank the unchanged classifier threshold `17.32x`
  (`2.6651915227297353e-4 -> 1.538907308516619e-5`) while their delta norms
  dropped only `1.37x` and `2.15x`

Opt-in Rung 0-5 timing report:

```bash
python scripts/debug/run_new4_geometry_fit_ladder.py \
  --state artifacts/geometry_fit_gui_states/new4.json \
  --background-index 0 \
  --output-root artifacts/geometry_fit_ladder/new4 \
  --max-rung blocks \
  --max-nfev 20 \
  --timeout-seconds 120 \
  --timing-report artifacts/geometry_fit_ladder/new4/latest_timing_summary.json
```

Inspect timing JSON with Python:

```bash
python -c "import json; r=json.load(open('artifacts/geometry_fit_ladder/new4/latest_timing_summary.json')); print(r['rung_timings']); print(r['slowest_rung'], r['slowest_rung_elapsed_s'])"
```

Full workflow checkpoint:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py -q
```

Latest reported local result: `316 passed` after class-A/class-C cleanup. This documentation handoff did not rerun the suite.

## Rung status table

| Rung | Scope | Status | Notes |
| --- | --- | --- | --- |
| Rung 0 | provider-only parity | green | no optimizer |
| Rung 1 | objective dry-run | green | finite residual, 7 fixed rows, 0 fallback |
| Rung 2 | sensitivity scan | green | 11 active, 2 near-zero, 0 non-finite, 0 unsafe |
| Rung 3 | one-parameter solves | green | singleton evidence usable for pair/block work |
| Rung 4 | paired solves | green | initial pair set passed |
| Rung 5 | cumulative blocks | green | fresh run `20260422_rung7_feature_gate_blocks`, 4/4 blocks passed |
| Rung 6 | selected combined solve / full-candidate dry run | green | fresh run `20260422_rung7_feature_gate_combined`, C2 passed |
| Rung 7 | controlled feature gate | green | `dynamic_reanchor`, `discrete_modes`, `seed_multistart`, `full_beam_polish`, and `identifiability_features` passed |

## Active and near-zero parameters from Rung 2

Active:

- `center_x`
- `center_y`
- `chi`
- `cor_angle`
- `theta_initial`
- `corto_detector`
- `zs`
- `zb`
- `a`
- `c`
- `psi_z`

Near-zero:

- `gamma`
- `Gamma`

`non_finite`: none.

`unsafe`: none.

`center_x` and `center_y` changed near_zero -> active because
`residual_norm_base` dropped `17.32x`
(`2665.1915227297354 -> 153.89073085166189`), shrinking the unchanged active
threshold `17.32x`. Their delta norms dropped less: `center_x` `1.37x`,
`center_y` `2.15x`. This is expected under the current rule, not a fitting
regression.

## Closed issues, do not reopen unless a guard fails

Do not reopen:

- manual visual placement
- saved picker point propagation
- provider_pairs vs dataset rows
- visual/backend coordinate transforms
- `GeometryFitSolverRequest` coordinate parity
- Rung 1 fixed-source handoff
- Rung 2 residual sensitivity

Do not add:

- backend coordinate transforms
- visual point movement
- nearest-candidate rebinding changes
- source identity loosening
- fallback acceptance

unless a focused parity/coordinate/rung guard fails.

## Known environment caveat

On some environments, optimizer-request capture can fail because headless execution setup is unavailable.

That is not a coordinate mismatch. The diagnostic must classify it as:

```text
classification == "diagnostic_incomplete_optimizer_request_unavailable"
```

not `frame_mismatch_detected`.

Fixed behavior: failed optimizer-request capture leaves the optimizer request un-compared and reports the diagnostic-incomplete classification. Focused visual/backend, coordinate diagnostic, and `new4` visual/backend tests cover this path.

## Remaining work

Next project: compare the real headless start-state/feature-toggle contract
against the passing 6-variable ladder candidate. Current expected Rung 2
baseline is `active_param_count=11`, `near_zero_param_count=2` under the
unchanged threshold rule. Do not reopen the exact-caked preflight, 3B harness,
or Rung 7 finalizer in this track unless a guard regresses.

Final frozen New4 chain:
`codex_final_blocks_20260423`, `codex_final_combined_20260423`, and passing
feature-sequence comparator `codex_restore_rung7_features_fix_20260423`
(`codex_final_features_fullseq_20260423`) kept provider/caked/Rung 5/Rung 6
guards green and passed `dynamic_reanchor`, `discrete_modes`,
`seed_multistart`, `full_beam_polish`, and `identifiability_features`. The
older `codex_final_features_20260423` artifact is stale and still shows the
pre-fix `full_beam_polish` failure. Exact-caked evidence stayed present and the
finalizer normalized `residuals_finite` without masking other guard failures.
`new4.json` stayed unchanged
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`), and
`full_fitter_validated == false` because the full fitter itself is still not
claimed.

Real full headless smoke `python -m ra_sim fit-geometry artifacts/geometry_fit_gui_states/new4.json`
was run separately on 2026-04-23 for background `0`. Exact-caked request
invariants stayed green, but the run rejected with `accepted == false`,
`detector_rms_px == 914.4948551954421`, and `unweighted_peak_max_px ==
1698.2499036720524`. The first divergence versus the passing ladder comparator
is a seed/start-state split, not request construction, acceptance-threshold
logic, candidate selection, or detector-space reporting.

Opt-in timing check `20260422_123330` measured the approved fresh Rung 5 blocks
path only (`--max-rung blocks`, `--timing-report`). It wrote
`rung_timing_summary.json` plus `latest_timing_summary.json`, listed Rungs
0/1/2/3/3B/4/5 only, and left threshold diagnostics `not_configured`.

## Do not run as acceptance

Do not use the old full baseline as the first next step.

Do not run full fitter, baseline, GUI fit button, unrestricted feature
combinations, auto-freeze/selective thaw, or feature-combo solves as full
acceptance. Current Rung 2 expected baseline is `11/2` under the unchanged
threshold rule; do not reopen the exact-caked path for it.

Do not treat RMS/max baseline or full fitter behavior as validated yet.

## GitHub issue note

Issue [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) was updated with this checkpoint:

```text
New4 bounded ladder plus real headless-fit checkpoint:

- Provider/caked/Rung 5/Rung 6 guards stayed green across the final frozen
  ladder chain.
- Passing Rung 7 feature-sequence comparator is
  `codex_restore_rung7_features_fix_20260423`
  (`codex_final_features_fullseq_20260423`); the older
  `codex_final_features_20260423` artifact is stale and still shows the
  pre-fix `full_beam_polish` failure.
- Passed: `dynamic_reanchor`, `discrete_modes`, `seed_multistart`,
  `full_beam_polish`, `identifiability_features`.
- Exact-caked evidence stayed present and `residuals_finite` normalized
  without masking unrelated guard failures.
- Real `fit-geometry` smoke on `artifacts/geometry_fit_gui_states/new4.json`
  background `0` still rejected with `accepted == false`,
  `detector_rms_px == 914.4948551954421`, and `unweighted_peak_max_px ==
  1698.2499036720524`.
- First divergence versus the passing ladder comparator is a seed/start-state
  split: real headless fit uses the 9-variable GUI/runtime contract and
  selected `axis:zb-1`, while the passing ladder comparator uses the 6-variable
  New4 candidate bundle and a different seed family.
- `full_beam_polish` is disabled in the real headless run, so
  candidate-selection is not the first divergence.
- `new4.json` hash stayed
  `f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`.
- `full_fitter_validated == false`; current Rung 2 expected baseline is `11/2`
  under the unchanged threshold rule, so do not reopen the exact-caked path for
  it.

```

## Links

- [Geometric fitter recovery](geometric-fitter-recovery.md)
- [Tracking hub](../index.md)
