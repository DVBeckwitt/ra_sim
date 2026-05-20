# New4 Geometric Fitter Recovery Handoff

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-05-20

## Current status

- 2026-05-20 shipping/status follow-up completed. The current branch contains
  the detector-origin observed-caked projection fix and the follow-up regression
  pinning commit. The cached live traces
  `geometry_fit_trace_20260520_145342.jsonl` and
  `geometry_fit_trace_20260520_155154.jsonl` still reproduce the broken
  signature when checked, but both are pre-fix evidence; the newer cached trace
  was written before the current fix commits. Bug/error status: fixed in the
  source and regression suite for the caked-objective handoff path that reached
  detector-pixel `central_point_match` with missing `fit_observed_caked_deg`.
  Remaining runtime proof: rerun the real Tk GUI workflow after the current
  commits and require either finite observed/predicted caked anchors with
  `dynamic_angular_point_match` in degrees, or a preflight
  `manual_caked_fit_space_missing` block before optimizer start. Feature
  status: no GUI control, CLI flag, config key, saved-state field, artifact
  schema, dependency, migration, deprecation, ADR, or version bump. CI/CD
  status: no workflow change; local validation remains the project gate.
  Validation status: focused handoff tests passed, the trace checker rejects
  the known broken traces, `python -m ra_sim.dev check` passed, and the GUI
  runtime/view/state tier passed on rerun after one non-repeatable Python
  access violation. Shipping status: safe as an internal bug-fix slice with
  rollback by git revert.
- 2026-05-20 detector-origin observed caked projection follow-up completed.
  Detector-origin manual picks in explicit caked-required fits now project
  their observed detector anchor through the exact-caked projector during
  dataset build, so the caked objective receives finite observed caked rows
  without treating stale saved caked aliases as truth. The async runtime no
  longer performs the duplicate observed-caked preflight before dataset
  projection can run, and the handoff checker now parses live text logs for
  missing observed-caked anchors, pixel RMS, and caked/pixel fallback
  signatures. Bug/error status: fixed for the detector-origin caked-required
  handoff loop reaching `central_point_match` with unavailable observed caked
  coordinates. Feature status: no GUI control, CLI flag, config key,
  saved-state field, artifact schema, dependency, or version change.
  CI/CD status: no workflow change; local `ra_sim.dev check` remains the
  shipping gate. Deprecation/migration status: none required. ADR status: no
  ADR needed for this private validation-boundary fix. Shipping status: safe
  as an internal bug fix with rollback by git revert.
- 2026-05-20 explicit caked-required handoff follow-up completed. The latest
  GUI trace after the previous patch proved the invariant was still not fully
  encoded: `manual_caked_fit_space_required=false`,
  `validator_finite_caked_rows=0`, `exact_fit_space_projector_available=false`,
  and the solver still completed as detector-pixel `central_point_match`.
  Root cause: `manual_fit_space_by_background` was still being used as both
  pick provenance and fit-space requirement, so detector-origin manual picks
  could suppress the caked-objective guard. The handoff now carries an explicit
  `manual_caked_fit_space_required_by_background` map derived from requested
  caked projection/objective intent, while preserving detector-origin
  provenance. Dataset preparation accepts `manual_fit_requires_caked_space`,
  source-row validation fails closed when required caked rows are absent, and a
  small trace checker rejects the broken signature
  (`manual_caked_fit_space_required=false`, `validator_finite_caked_rows=0`,
  `final_metric_name=central_point_match`, `metric_unit=px`). Bug/error
  status: fixed for the circular caked-objective-to-pixel-central-match
  fallback signature; remaining proof is a real GUI rerun showing either
  `manual_caked_fit_space_missing` before optimizer start or
  `dynamic_angular_point_match` in degrees with finite observed/predicted
  caked anchors. Feature status: no GUI control, saved-state schema, CLI flag,
  dependency, or version change.
- 2026-05-20 review follow-up completed. The worker now also uses
  `manual_fit_requires_caked_space` when copying the current exact caked
  payload into the async job, and mixed detector/caked provenance is rejected
  only for detector-space fits, not for explicit caked-required fits. The trace
  checker now detects split-record contradictions where one trace row declares
  `objective_space=caked_deg` and a later run row reports
  `central_point_match` or `metric_unit=px`. Validation status: focused handoff
  regressions pass, the known broken 20260520_145342 trace is rejected by the
  checker, and `python -m ra_sim.dev check` passes. Live GUI rerun remains the
  final operator smoke test.
- 2026-05-20 caked handoff simplification pass completed. The follow-up
  refactor preserved behavior while reducing duplicate fit-space classification
  code: worker manual-pair extraction now flows through one local helper, and
  manual caked-coordinate detection uses one finite-coordinate loop instead of
  repeated per-field branches. Bug/error status: the manual caked Qr/Qz fallback
  loop remains fixed; this pass only reduced maintenance risk around the guard.
  Feature status: unchanged. CI/CD status: no workflow change; local
  `ra_sim.dev check` remains the quality gate. Deprecation/migration status:
  none required. Shipping status: safe as an internal refactor with rollback by
  reverting the refactor commit.
- 2026-05-20 manual caked fit-space handoff gate completed. Root cause for the
  latest GUI loop was upstream of the optimizer: a selected caked Qr/Qz fit
  could still accept detector-only/live `peak_records` source rows, time out
  caked view storage, and then reach the solver as detector-pixel
  `central_point_match`. The handoff now treats active caked intent from
  `projection_view_mode`, `pick_uses_caked_space`, or
  `manual_fit_space_by_background` as a hard requirement for exact caked
  fit-space rows. Detector-origin rows with explicit detector provenance remain
  detector-space fits. Runtime source-cache validation rejects required matches
  whose matched candidate lacks finite caked coordinates, manual caked preflight
  fails on caked projection/storage timeout, prepared dataset specs carry an
  internal manual-caked-required flag, and the optimizer returns
  `manual_caked_fit_space_missing` instead of silently falling back to
  `central_point_match` if the GUI handoff regresses. Bug/error status: fixed
  for the caked-mode fallback loop; remaining proof is an actual Tk GUI rerun
  of default `m=1,L=10`, `gamma,Gamma` with finite caked rows. Feature status:
  no new GUI control, CLI flag, config key, saved-state schema, artifact
  schema, dependency, or version bump. CI/CD status: no workflow changes; local
  focused pytest and diff hygiene are the quality gate for this slice.
  Deprecation/migration status: none required because no public interface or
  stored schema changed. ADR status: no ADR needed for this private validation
  boundary hardening. Validation: focused rebuild, runtime, optimizer, matched
  caked-row validator, and detector/caked compatibility regressions passed.
- 2026-05-20 nested full-identity simplification reviewed and ready to ship.
  Behavior is unchanged: full-reflection trust detection now uses one private
  predicate, and fixed-source matching uses one local row-record cache helper
  instead of repeating the same cache-loading branch. Bug/error status: the
  GUI/default `m=1,L=10`, `gamma,Gamma` `prediction_branch_source_switched`
  failure remains fixed. Feature status: no user-facing feature change.
  Compatibility status: no public API, GUI control, CLI flag, config key,
  saved-state field, artifact schema, dependency, or version change. CI/CD
  status: no workflow changes; local quality gates remain the merge signal.
  Deprecation/migration status: none required because no public behavior or
  storage contract changed. ADR status: no ADR needed for this private
  behavior-preserving refactor. Validation: targeted locked-Qr regressions
  passed (`5 passed`), `tests/test_geometry_fitting.py` passed (`271 passed`),
  `python -m ra_sim.dev check` passed (`294 passed` plus ruff/mypy), actual Tk
  GUI startup reached `Simulation ready.`, and the saved Bi2Se3 GUI-state fit
  with `--active-vars gamma,Gamma` still resolved `15/15` fixed pairs with
  `prediction_branch_source_switched_count=0`. Shipping status: ready as an
  internal cleanup with rollback by git revert.
- 2026-05-20 nested full-identity fixed-source recovery completed. The
  `m=1,L=10` locked manual Qr/Qz pair no longer loses its source authority when
  final fit correspondence rows retain the full reflection identity only inside
  `provider_selected_source_identity_canonical`. The fixed-source matcher now
  recovers a unique branch row from the current reduced hit tables using the
  nested full identity plus the `m=H^2+H*K+K^2,L` Q-group identity, and final
  correspondence resolution uses the same nested identity when top-level
  `source_reflection_*` fields were stripped. Bug/error status: fixed for the
  GUI/default `gamma,Gamma` two-tilt rejection path that reported
  `prediction_branch_source_switched`, `matched=0`, and no available matched
  peak pairs after preflight had resolved the manual points. Feature status:
  no GUI control, CLI flag, config key, saved-state schema, artifact schema, or
  dependency change. Deprecation/migration status: none required; existing
  saved manual-pick payloads remain compatible. CI/CD status: no workflow files
  changed. Validation: `tests/test_geometry_fitting.py` passed (`271 passed`),
  GUI/runtime smoke tests passed (`528 passed`), `python -m ra_sim.dev check`
  passed (`294 passed` plus ruff/mypy), actual Tk GUI startup reached
  `Simulation ready.`, and the saved Bi2Se3 GUI-state fit with
  `--active-vars gamma,Gamma` resolved `15/15` fixed pairs with
  `prediction_branch_source_switched_count=0` and weighted RMS `1.2063 deg`.
  Shipping status: ready as a normal bug-fix slice with rollback by git revert.
- 2026-05-20 same-`m,L` resolver cleanup completed. Bug/error status remains
  fixed for the locked manual Qr/Qz `prediction_branch_source_switched`
  rejection described below. The follow-up code change is behavior-preserving:
  it removes the single-use `_hkl_matches_q_group_ml` wrapper, keeps the same
  explicit `m,L` comparison in the fixed-source resolver, and reuses the
  already-sliced simulated row head during stale-row matching. Feature status:
  no new GUI control, CLI flag, config key, artifact schema, or saved-state
  field. Deprecation/migration status: none required because public interfaces
  remain stable. CI/CD status: no workflow files changed; the existing local
  quality gates remain the merge signal. Validation: focused same-`m,L`
  geometry-fit regressions passed, the Qr/Qz manager file passed, `python -m
  ra_sim.dev check` passed, and `git diff --check` passed. Shipping status:
  ready as an internal cleanup with rollback by git revert.
- 2026-05-20 same-`m,L` Qr/Qz identity slice completed. Qr/Qz selector rows
  now expose the canonical `m=H^2+H*K+K^2, L` identity in helper contracts,
  row labels, and exported rows while preserving the serialized
  `("q_group", source, m, L)` saved-state key. Locked manual Qr/Qz source-row
  resolution now treats signed-HK representatives from the same `m,L` family
  as the same Q-group when branch identity still matches. Bug/error status:
  fixed for the follow-up GUI rejection where a direct two-branch `m=1,L=10`
  fit could run direct LSQ and then fail final acceptance with
  `prediction_branch_source_switched` / `matched=0` after `(-1,0,10)` was
  compared against an equivalent current row such as `(1,0,10)`. Feature
  status: label/export clarity only; no new operator control. Compatibility
  status: saved-state key shape, CLI flags, config keys, and artifact schemas
  remain compatible; additive `m_index` and `l_index` fields are included in
  Qr/Qz export rows beside legacy `gz_index`. CI/CD status: no workflow files
  changed. Validation so far: focused q-group manager tests and locked-Qr
  source-row resolver regressions passed. Remaining verification: rerun the
  user's GUI default caked-mode `m=1,L=10` `gamma,Gamma` fit to confirm the
  dialog no longer reports `prediction_branch_source_switched`.
- 2026-05-20 detector-origin simplification slice completed. Manual Qr/Qz rows
  picked in detector view now stay on the fixed detector-pixel LSQ path,
  including `gamma,Gamma`, instead of being auto-promoted into the exact-caked
  angular objective. The GUI manual-point runtime now disables seed multistart
  for these detector-origin rows, so a two-branch fit reaches the direct LSQ
  solver instead of failing at seed prescore with `selected=0`. Explicit
  caked-origin rows still require the exact caked projector and fail closed when
  it is unavailable. Bug/error status: fixed for the GUI rejection where a
  simple two-branch detector-origin fit could report optimizer failure,
  non-finite caked angular residuals, seed-search candidate rejection, and no
  matched fitted peak pairs even though the detector-space manual pairs were
  present. Feature status: no new operator control; this removes the internal
  auto-caked detector-origin path and unnecessary GUI manual-point seed search
  while preserving public helper signatures for compatibility.
  Migration/deprecation status: no saved-state schema, CLI flag, config key, or
  artifact schema changed; saved caked aliases on detector-origin rows remain
  display/replay cache data. CI/CD status: no workflow files changed; the local
  quality gate is the existing project gate. Validation: focused detector/caked
  fit-space and GUI runtime tests passed, a runtime probe showed detector-origin
  rows do not request caked projection while caked-origin rows still do, and
  `python -m ra_sim.dev check` passed. Shipping status: ready as a bug-fix
  commit with rollback by git revert; the full real-data New4 excluded-row
  proof remains outside this slice.
- 2026-05-16 reporting hardening slice completed. Caked click-pick QR trial
  source-row diagnostics now treat merged live caked candidate rows as final
  dynamic coverage, so `missing_dynamic_trial_source_row_count` reports the
  post-rebinding count instead of a stale pre-supplement count. The New4
  single-step QR audit now labels `proof_status` with
  `proof_scope="caked_space_contract"` and reports detector-panel scope/status
  separately, so `proof_status=pass` with zero detector-panel plotted rows is
  explicitly caked-space-only proof. The skipped-background OSC-name runtime
  regression was rechecked locally and is green in this checkout, so no runtime
  label code changed. Status: this fixes reporting diagnostics only; it does
  not prove the full 82-row Bi2Se3 fit or regenerate accepted excluded-row
  artifacts. A direct excluded `gamma,Gamma` rerun wrote
  `artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/direct_00_gamma_gamma_20260516/`
  and resolved 79/79 QR rows at `0.7688782306077002 deg` RMS, but the strict
  caked-space single-step proof failed with nonzero source-authority/surface
  mismatches and `detector_panel_status="not_plotted"`.
- 2026-05-14 Bi2Se3 first-image caked manual fit fail-closed slice completed.
  QR handoff audit rows now keep caked observed targets in
  `fit_observed_caked_deg` and reconstruct `fit_observed_detector_display_px`
  only from detector-native coordinates through the dataset's native-to-display
  conversion. Saved caked `(2theta, phi)` clicks no longer appear as detector
  pixels in the audit. Non-legacy dynamic caked fits that prove
  `objective_param_sensitivity_status=all_fit_vars_insensitive` now return
  `success=false`, `status=-9`, and
  `dynamic_objective_not_sensitive_to_fit_variables`, and the optimizer status
  line says failed instead of complete. Bug/error status: fixed for the
  misleading detector-display audit fields and for the Bi2Se3 two-point/three
  caked manual fit path being reported as complete when the objective cannot
  move under the requested fit variables. Feature status: behavior hardening
  only; no GUI control, CLI flag, config key, dependency, migration,
  deprecation, or release version change. Interface/artifact status: one
  additive diagnostic reason field may appear beside
  `fit_observed_detector_display_px`; existing saved states remain compatible.
  Validation: the new regressions passed (`2 passed`), the full geometry
  fitting and GUI geometry-fit workflow files passed (`959 passed, 2 skipped`),
  and `python -m ra_sim.dev check` passed (`292 passed`, ruff clean, mypy
  clean). Shipping status: ready as a normal bug-fix slice; rollback is a git
  revert.
- 2026-05-14 sweep-apply review hardening completed. The explicit
  `--apply-sweep-result` path now treats `combo_result.json` as an external
  boundary: QR and mismatch counters must be real integer JSON values, accepted
  overlay PNGs must resolve inside the combo artifact directory, and the overlay
  source must pass a PNG-header check before any geometry variables are updated.
  Missing, non-gettable, or non-settable active variable targets fail before
  mutation. Setter failures, overlay destination failures, overlay temp-file
  failures, and rebuild exceptions now restore pre-apply GUI geometry values
  and remove temporary overlay artifacts before re-raising.
  `04_applied_geometry_overlay.json` now leaves `output_state_sha256` pending
  until the CLI finalizes it from the actual `--out-state` file. Bug/error
  status: closes the review findings for arbitrary overlay-path copying,
  malformed counter coercion, pre-save output hash reporting, and non-atomic
  failed applies, including silent partial variable-set failures. Feature
  status: hardening only; no threshold tuning, combo acceptance change,
  migration, deprecation, or new public flag. Shipping status: opt-in CLI apply
  remains the rollout boundary, rollback is the unchanged input state or a
  previous saved output state, and failed applies leave geometry unchanged.
  Validation: focused GUI apply tests passed (`30 passed`), CLI geometry-fit
  tests passed (`36 passed`), the selected fitting regression target passed
  (`9 passed`), the full GUI geometry-fit workflow file passed
  (`699 passed, 2 skipped`), the full geometry fitting file passed
  (`260 passed`), touched-file `compileall` passed, and
  `python -m ra_sim.dev check` passed (`292 passed`, ruff clean, mypy clean).
  CI/CD status: no workflow files changed; this used the existing local quality
  gate as the merge signal. Deprecation/migration status: no CLI flags,
  saved-state schema, artifact field, or compatibility path changed. ADR
  status: no new ADR is needed because the decision stays inside the existing
  guarded apply architecture. Shipping status: ready as a normal bug-fix commit
  with rollback by git revert or by using the unchanged input state; the real
  Bi2Se3 apply was not run by this hardening slice.
- 2026-05-14 Bi2Se3 controlled exclusion and parameter-combo sweep completed.
  Headless `fit-geometry` now supports exact `--exclude-pair-id` removal before
  objective assembly, `--parameter-combo-sweep` dry runs, and explicit
  `--apply-sweep-result` application with repeated
  `--approve-excluded-pair-id` approvals. The exclusion is exact by `pair_id`,
  not HKL, and records `excluded_pair_ids`, `excluded_rows`,
  `original_qr_fit_expected_count`, `qr_fit_expected_count`,
  `qr_fit_resolved_count`, `qr_fit_missing_pairs`,
  `excluded_rows_do_not_count_as_missing=true`, and
  `saved_gui_state_mutated=false`. The apply helper only applies an accepted
  sweep result after matching the saved-state SHA, approved exclusion list,
  caked residual thresholds, complete QR contract, mismatch counters, and
  objective sensitivity. Bug/error status: the three known outlier rows
  `bg1:pair15`, `bg0:pair20`, and `bg2:pair17` are removed before fitting and
  are no longer counted as missing; they are preserved in the saved state and
  recorded under `geometry_fit_excluded_pair_ids` when the accepted result is
  applied. Unsupported, incomplete, and native-crashing combos now fail closed
  with JSON/PNG artifacts and durable bounded stdout/stderr tail files instead
  of aborting the sweep. Feature status: additive CLI/report/apply workflow
  only; no threshold tuning, forced acceptance, branch remap, saved-state
  mutation during dry runs, or `qr_fit_objective_incomplete` removal. Latest
  sweep status: `00_gamma_Gamma` accepted; `03_gamma_Gamma_center_x_center_y`
  accepted; `01_gamma_Gamma_theta_initial` and
  `02_gamma_Gamma_corto_detector` failed closed with native child return code
  `3221225477`; `04_gamma_Gamma_theta_initial_corto_detector` and
  `05_gamma_Gamma_theta_initial_center_x_center_y` failed closed with
  `qr_fit_objective_incomplete`; therefore
  `all_supported_required_combos_pass=false`. Best accepted combo:
  `00_gamma_Gamma`, RMS `0.8083400569700655 deg`, max
  `2.5981580333851113 deg`, exclusions
  `bg0:pair20`, `bg1:pair15`, `bg2:pair17`, `qr_fit_expected_count=79`,
  `qr_fit_resolved_count=79`, and `qr_fit_missing_pairs=[]`. Successful apply
  writes `04_applied_geometry_overlay.{png,json}` and an output state with
  `geometry_updated=true`. Validation for this slice: focused GUI sweep/apply
  tests passed, CLI geometry-fit tests passed (`36 passed`), the full geometry
  fitting file passed (`260 passed`), the full GUI geometry-fit workflow file
  passed (`683 passed, 2 skipped`), the manual fit-space classification file
  passed (`9 passed`), and `python -m ra_sim.dev check` passed (`292 passed`).
  The real `%LOCALAPPDATA%\ra_sim\Bi2Se3.json` apply gate was not run in this
  workspace because that user-local state file is absent.
- 2026-05-13 review hardening cleanup completed. Versioned tracking docs now
  refer to user-local GUI states as `%LOCALAPPDATA%\ra_sim\*.json` instead of
  committing machine-specific absolute paths. The headless progress sanitizer
  now reuses the optimizer's QR caked-source contract list instead of carrying
  a second copy in `headless_geometry_fit.py`, and the progress tests now guard
  that true detector-display predictions with detector/native proof remain
  unchanged. Bug/error status: closes the review findings for docs hygiene,
  avoidable duplicated source-list code, and missing detector-preservation test
  coverage. Feature status: behavior-preserving hardening only; no threshold
  tuning, forced acceptance, manual-pick edits, saved-state schema changes,
  deprecation, migration, or public CLI changes. Validation: focused headless
  recovery/progress tests passed (`5 passed`) and `python -m ra_sim.dev check`
  passed (`281 passed`).
- 2026-05-13 Bi2Se3 recovery provenance/reporting cleanup completed. Headless
  gamma/Gamma progress and recovery overlay JSON now record `input_state_path`
  and `input_state_sha256`, so the artifact folder proves whether it used the
  user-local `%LOCALAPPDATA%\ra_sim\Bi2Se3.json` state or another
  saved state. Live-cache progress rows no longer leave caked `(2theta, phi)`
  values under `fit_prediction_detector_display_px` for caked-display aliases:
  the detector-display field is `null` with
  `caked_degrees_not_detector_display_px`, and the same coordinate is reported
  as `fit_prediction_caked_deg`. Bug/error status: reporting-only fix for the
  misleading progress JSON field name; fitter acceptance, thresholds, manual
  picks, GUI state, branch repair, and image generation behavior are unchanged.
  Validation: focused recovery/progress artifact tests passed (`5 passed`),
  focused QR contract/objective tests passed (`11 passed`), overlay legacy
  compatibility tests passed (`2 passed`), `python -m ra_sim.dev check` passed,
  and a full headless run from
  `%LOCALAPPDATA%\ra_sim\Bi2Se3.json` completed with artifact
  status `pass`. That run stayed fail-closed with `full_fit_success=false`,
  `geometry_updated=false`, `qr_fit_resolved_count=82`,
  `qr_fit_expected_count=82`, `objective_param_sensitivity_status=sensitive`,
  `failure_classification=manual_outliers_or_physical_bad_fit`, and worst row
  `bg1:pair15` / `hkl=[-2,0,5]` at `70.99 deg`.
- 2026-05-13 headless recovery artifact wiring completed. Bi2Se3 headless
  `gamma,Gamma` fits now write `01_single_step_qr_coordinate_audit.{json,csv,png}`,
  `02_full_fit_initial_vs_final_qr_overlay.{json,png}`, and, on rejection,
  `03_worst_residual_rows.{json,png}` into the same output folder as
  `bi2se3_gamma_gamma_fit.progress.json`. The progress sidecar records all
  generated paths under `geometry_fit_recovery_artifacts` and at the common
  top-level keys. Bug/error status: the prior run saved only JSON progress, so
  a rejected `branch_source_pairing_mismatch` fit lacked the required visual
  approval layer. The headless run now fails closed if the single-step PNG, the
  full-fit overlay PNG, or the rejected-fit worst-row PNG is missing. Feature
  status: additive local recovery artifacts only; no threshold tuning, forced
  acceptance, manual-pick edits, saved-state schema changes, config changes,
  public CLI flag changes, deprecation, or migration. Validation: the focused
  artifact writer slice passed (`3 passed, 637 deselected`), the QR
  contract/single-step slice passed (`23 passed`), the broader QR/sensitivity
  target passed (`37 passed`), the full geometry fitting file passed
  (`260 passed`), the full GUI geometry-fit workflow file passed
  (`637 passed, 2 skipped`), the manual fit-space classification file passed
  (`9 passed`), and `python -m ra_sim.dev check` passed. A local verification
  regenerated ignored images from the existing Bi2Se3 progress JSON and updated
  that ignored progress sidecar with the artifact paths; a fresh full optimizer
  rerun remains optional for overnight validation.
- 2026-05-13 proof-status hardening completed. The single-step QR coordinate
  audit now fails closed when any row has a failing or missing QR fit contract
  status, so `proof_status=pass` cannot coexist with a failed contract. The
  `--allow-visual-objective-surface-divergence` diagnostic flag no longer masks
  non-surface failures: row-count, identity, branch/hkl, bounded `gamma/Gamma`,
  and contract failures keep the final artifact status at `fail`. Generated
  recovery artifacts under `artifacts/geometry_fit_recovery/` are ignored as
  local diagnostic output. Bug/error status: the proof image/JSON gate no longer
  presents a human approval artifact as pass/diagnostic when the contract layer
  failed. Feature status: this is an additive debug-artifact hardening slice; no
  fit thresholds, manual picks, GUI state, config keys, saved-state schema, or
  accepted geometry outputs changed. No deprecation or migration is required.
  Validation: the focused QR contract/single-step audit slice passed
  (`23 passed`), the broader QR/sensitivity target passed (`37 passed`), the
  full geometry fitting file passed (`260 passed`), the full GUI geometry-fit
  workflow file passed (`635 passed, 2 skipped`), the manual fit-space
  classification file passed (`9 passed`), and `python -m ra_sim.dev check`
  passed.
- 2026-05-13 sensitivity ladder slice completed. The dynamic caked QR objective
  sensitivity diagnostic no longer classifies `gamma`/`Gamma` from a
  sub-0.1-degree probe alone; angular fit variables are probed at
  `0.1, 0.25, 0.5, 1, 2, 5` degrees and each variable records the first
  meaningful step plus maximum predicted-caked and residual-vector deltas.
  Bug/error status: this guards the reported `objective_param_insensitive`
  failure mode from tiny finite differences. Feature status: additive
  diagnostic fields only; no thresholds, branch identity, manual picks, GUI
  state, config, or saved-state schema changed in this slice. Existing
  fail-closed insensitive-objective rejection now requires the full ladder proof
  instead of a tiny finite difference.
  Additional propagation guards prove `gamma` reaches the source-row builder,
  `Gamma` reaches the exact caked projector, and changed `gamma/Gamma`
  signatures miss the shared prediction source-row cache instead of reusing base
  rows. Review hardening fixed cache-reuse diagnostics so
  `reused_for_same_params_signature` is not reported as rebuilt, and the
  five-degree cap is asserted against emitted probes. Validation: the targeted
  sensitivity/threading/cache slice passed (`9 passed`), the dynamic-source GUI
  workflow target passed (`6 passed`), `tests/test_geometry_fitting.py` passed
  (`255 passed`), and `python -m ra_sim.dev check` passed. Full New4
  gamma/Gamma rerun and branch repair remain next steps.
- 2026-05-13 maintenance cleanup: full-fit QR contract assembly now reuses the
  already-normalized manual and objective caked point payloads, and the contract
  helper resolves the GUI-drawn simulation source once before writing both
  legacy-compatible source fields. Bug/error/feature status is unchanged: this
  is a behavior-preserving cleanup of the diagnostic contract path, with no
  CLI, config, saved-state, artifact-schema, deprecation, migration, or launch
  impact. The narrow QR contract/audit test slice and `python -m ra_sim.dev
  check` passed after formatting. A longer New4 headless gamma/Gamma recheck
  was interrupted before completion, so no new runtime artifact status is
  claimed from that aborted run and generated recovery artifacts remain
  unversioned.
- 2026-05-13 full gamma/Gamma headless gate reproduced. The New4 fit now
  exposes top-level QR contract fields in the public point-match summary:
  `qr_fit_expected_count=7`, `qr_fit_resolved_count=7`,
  `qr_fit_missing_pairs=[]`, `qr_fit_contract_status=pass`,
  `source_authority_mismatch_count=0`,
  `visual_objective_surface_mismatch_count=0`,
  `source_authority_match_all_caked_display_rows=true`,
  `visual_objective_surface_match_all_rows=true`, and
  `objective_param_sensitivity_status=sensitive`. The fit still rejects, but
  the failure is now meaningful and classified as
  `branch_source_pairing_mismatch` with worst rows and
  `recommended_next_fix=repair_locked_branch_identity`.
- 2026-05-13 maintenance cleanup: single-step QR visual audit proof checks now
  use shared row-value and row-predicate helpers instead of repeated inline
  loops. Bug/error/feature status is unchanged: this is a behavior-preserving
  cleanup of debug audit bookkeeping, with no public CLI, config, saved-state,
  artifact-schema, deprecation, or migration impact.
- 2026-05-13 QR surface contract slice completed. The optimizer now exposes a
  central `_build_qr_fit_point_surface_contract(...)` diagnostic helper, the
  single-step audit writes per-row contract payloads and top-level contract
  counters, and the caked audit rows record `gui_drawn_sim_caked_source`. The
  regenerated New4 proof reports `qr_fit_contract_status=pass`,
  `qr_fit_contract_failure_count=0`, `source_authority_mismatch_row_count=0`,
  and `surface_mismatch_row_count=0`. Remaining work is the full gamma/Gamma
  GUI-fit gate and overnight recovery report, not threshold tuning.
- 2026-05-13 single-step QR coordinate visual audit completed. The New4
  coordinate visualizer now has a dry-run `--single-step-detector-angle-audit`
  mode that evaluates base dynamic residuals, finite-differences only
  `gamma,Gamma`, applies one clipped least-squares trial step, and writes
  `01_single_step_qr_coordinate_audit.json`, `.csv`, and one two-panel `.png`
  under `artifacts/geometry_fit_recovery/latest` without calling the full
  geometry optimizer, updating GUI state, or accepting geometry.
- Bug/error fixed: the proof no longer trusts stale saved `sim_refined_caked`
  fields for simulated QR points. Original and trial simulation caked points
  come from the live caked objective surface. Caked-display QR source rows now
  prefer live `sim_visual_caked_deg`/caked fields over point-only detector
  reprojection, making the bad state
  `objective_source_authority=sim_visual_caked_deg` plus
  `optimizer_source_source=point_only_detector_projection` fail-closed instead
  of masquerading as proof.
- Feature/status: this is an additive debug-script interface and test-only
  fitting diagnostic. Public CLI commands, saved-state schema, config keys, and
  accepted geometry artifacts are unchanged. No deprecation or migration is
  required.
- Validation status: `tests/test_geometry_fitting.py` passes (`244 passed`),
  `tests/test_gui_geometry_fit_workflow.py` passes (`634 passed, 2 skipped`),
  `tests/test_geometry_fit_manual_fit_space_classification.py` passes
  (`9 passed`), and `python -m ra_sim.dev check` passes. The full manual
  selection helper file was too slow for this local run, but the targeted manual
  QR helper slice passed. The generated New4 proof reports `status=pass`,
  `proof_status=pass`, `row_count=7`, `plotted_row_count=0`,
  `invalid_detector_display_row_count=7`,
  `source_authority_match_all_caked_display_rows=true`,
  `surface_mismatch_row_count=0`, `json_authoritative=true`, and
  `png_diagnostic_only=true`.
- Shipping/rollback status: the mode is opt-in, debug-only, and produces local
  artifacts under the requested output root. Rollback is limited to removing the
  debug flag path and its tests; no data migration or user-facing compatibility
  cleanup is needed.
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
  saved state `%LOCALAPPDATA%\ra_sim\new4.json`, background index
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

## 2026-05-15 Manual Caked Qr/Qz Fit Handoff Fix

Bug/error status: fixed for the targeted GUI manual caked Qr/Qz handoff path that rejected visually aligned points with large angular residuals and `dynamic_objective_not_sensitive_to_fit_variables`.

Root cause:
- Caked-display source rows that already carried current simulated `(2theta, phi)` were accepted only in point-only projection mode, so the non-point-only dynamic objective could fall back to stale locked prediction fields.
- Manual observed/background caked rows could look like caked simulated source rows during trial source-row resolution.
- When a live dynamic completion row was worse than the saved refined simulated caked prediction, the fitter could replace the better handoff prediction and still report success.
- Acceptable-but-insensitive dynamic caked objectives could accept arbitrary parameter drift even though the Qr/Qz residual did not move.

Fix:
- The shared locked-Qr prediction resolver now consumes current caked simulated rows directly for dynamic caked objectives, without detector reprojection.
- Source-row resolution filters out rows whose live caked point is the manual observed/background target.
- The resolver retains closer saved refined simulated caked predictions when a live completion candidate would worsen the caked residual.
- If the dynamic caked objective is already aligned or within the caked acceptance limits but insensitive to every fit variable, the fit is a no-op success and keeps the starting geometry instead of applying unrelated parameter movement.
- The GUI progress text keeps separate dynamic-fit and full-beam overlay RMS lines for dynamic angular fits.

Validation:
- `python -m pytest tests/test_geometry_fitting.py -ra` -> `265 passed`.
- `python -m pytest tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_prediction_source_is_explicit tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_sim_refined_caked_uses_real_projection tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_handoff_audit_sim_refined_caked_uses_real_projection tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_step_reduces_qr_residual -ra` -> `4 passed`.
- `python -m pytest tests/test_gui_runtime_import_safe.py::test_geometry_fit_progress_text_reports_dynamic_fit_rms_separately -ra` -> `1 passed`.
- `python -m compileall ra_sim tests` -> passed.
- `python -m ra_sim.dev check` -> passed.
- `git diff --check` -> passed.

Follow-up validation fix:
- `python -m pytest tests/test_gui_runtime_import_safe.py::test_raw_only_full_update_restores_qr_and_hkl_picker_rows -ra` now passes after updating its stale source-signature assertion to the detector-picker path. The runtime behavior already returned `detector_picker_grouped`, matching the newer HKL picker tests and detector-picker row preference policy.

Follow-up simplification and shipping status:
- Bug/error status remains fixed for the GUI manual caked Qr/Qz handoff path. The follow-up change is behavior-preserving cleanup only.
- The locked-Qr prediction resolver now names the live-caked objective authority and refinement policy once before writing the diagnostic payload, keeping the public diagnostic fields unchanged.
- GUI manual-fit dataset building now streams candidate manual caked target points instead of allocating a temporary list while preserving the same target search order.
- Live-caked QR geometry-fit tests now share the repeated dataset-spec and refinement-config setup, reducing duplicated fixture code without adding new files or dependencies.
- Review status: approved for correctness, simplification, security, performance, and test quality; no avoidable new files or abstractions were found.
- CI/shipping status: no CI configuration, deprecation path, migration, release version, or public workflow change is required for this cleanup. Rollback is the commit revert; the previously validated user-visible fix stays unchanged.
- Validation: `python -m pytest tests/test_geometry_fitting.py -ra` -> `265 passed`; `python -m pytest tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_step_reduces_qr_residual -ra` -> `1 passed`; `python -m ra_sim.dev check` -> passed; `git diff --check` -> passed.

## 2026-05-20 Default Manual Caked Qr/Qz Dynamic Fit Fix

Bug/error status: fixed for the reported default GUI `m=1,L=10` manual caked
Qr/Qz fit rejection where the direct fixed-source solve reached the static
`central_point_match` objective and reported `matched=0`.

Root cause:
- The default manual point-fit solver selected the pixel/static point-match
  evaluator unless `dynamic_point_geometry_fit` was already set in runtime
  config.
- Exact-caked fixed-source manual Qr/Qz dataset rows already had the information
  needed for the dynamic angular objective, but that dataset evidence was not
  used when choosing the evaluator.

Fix:
- `fit_geometry_parameters(...)` now detects fixed-source manual Qr/Qz rows with
  exact-caked fit-space anchors in `dataset_specs` and auto-enables the dynamic
  angular point-match path for that solve.
- The same auto-detected path enables the existing bounded Qr/Qz point-only
  projection flag, matching the intended saved-manual-caked runtime policy
  without adding GUI controls, config keys, dependencies, or new files.
- Existing handoff diagnostics and optimizer prediction agreement were kept
  unchanged after an attempted audit change failed the end-to-end handoff
  regression.

Validation:
- `python -m pytest tests/test_geometry_fitting.py::test_manual_caked_qr_fit_auto_enables_dynamic_point_path -ra` -> passed.
- `python -m pytest tests/test_geometry_fitting.py::test_manual_caked_qr_fit_auto_enables_dynamic_point_path tests/test_geometry_fitting.py::test_fit_geometry_parameters_manual_point_fit_with_cached_sources_defaults_to_central_point_match tests/test_geometry_fitting.py::test_dynamic_angular_point_match_final_metric_uses_caked_deg_not_pixel_rms -ra` -> `3 passed`.
- `python -m pytest tests/test_geometry_fitting.py -ra` -> `272 passed`.
- `python -m pytest tests/test_gui_geometry_fit_workflow.py::test_prepare_geometry_fit_run_caked_pairs_use_existing_exact_projector_no_recake tests/test_gui_geometry_fit_workflow.py::test_new4_ladder_lean_runtime_config_preserves_caked_manual_path tests/test_gui_geometry_fit_workflow.py::test_objective_rejects_pixel_residual_for_manual_caked_qr_fit tests/test_gui_geometry_fit_workflow.py::test_gui_dynamic_caked_metric_rejects_large_angular_residual -ra` -> `4 passed`.
- `python -m pytest tests/test_gui_runtime_import_safe.py -ra` -> `450 passed`.
- `python -m pytest tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_optimizer_prediction_matches_fit_handoff_prediction tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_handoff_audit_sim_refined_caked_uses_real_projection tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_handoff_audit_fit_prediction_source_is_explicit tests/test_manual_geometry_selection_helpers.py::test_minus_1_0_10_fit_handoff_audit_no_caked_values_printed_as_detector_px -ra` -> `4 passed`.
- `python -m compileall ra_sim tests` -> passed.
- `python -m ra_sim.dev check` -> passed.
- `git diff --check` -> passed.

Shipping status:
- Review completed across correctness, simplification, security, performance,
  and test quality with no blocking findings.
- No CI configuration, dependency, public API, saved-state schema, CLI, or GUI
  control migration is required for this internal fitter-path selection fix.
- No package version bump is required because this is not release prep; the
  release-facing summary is tracked under `CHANGELOG.md` Unreleased.
- Rollback plan is a normal commit revert. The pre-fix behavior is isolated to
  the static point-match fallback for default exact-caked manual Qr/Qz rows.

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
