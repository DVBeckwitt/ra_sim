# New4 Geometric Fitter Recovery Handoff

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-23

## Current status

- Point-provider parity is closed.
- Visual/backend coordinate parity is closed for new4.
- `GeometryFitSolverRequest.measured_peaks` coordinate parity is closed when optimizer-request capture succeeds.
- Rung 1 objective dry-run is green.
- Rung 2 sensitivity scan is green.
- Rung 3 one-parameter solves are green for bounded ladder validation.
- Rung 3A `a` diagnosis is usable.
- Rung 3B caked-point reprojection guard is green.
- Rung 4 initial paired solves are green.
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
- Controlled Rung 7 feature gate is now green end-to-end. The final frozen
  chain `codex_final_blocks_20260423`, `codex_final_combined_20260423`, and
  `codex_final_features_20260423` kept provider/caked/Rung 5/Rung 6 guards
  green and passed `dynamic_reanchor`, `discrete_modes`,
  `seed_multistart`, `full_beam_polish`, and `identifiability_features`.
  Exact-caked evidence stayed present in the feature report, and the finalizer
  normalized `residuals_finite` without masking unrelated guard failures.
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
- The exact-caked preflight boundary is closed. The follow-up is separate
  Rung 2 sensitivity drift (`active_param_count=11`,
  `near_zero_param_count=2`); do not mix it with the exact-caked path.
- No full fitter, baseline, GUI fit, or unrestricted feature combination run
  should be treated as validated.

This handoff is the bounded-through-Rung-7 feature-gate recovery state for
`new4`. Do not use it as approval for full fitter, GUI, baseline, or unrestricted
feature-combination solves.

Status by work type:

- Feature: controlled Rung 7 passed `dynamic_reanchor`, `discrete_modes`,
  `seed_multistart`, `full_beam_polish`, and `identifiability_features`; the
  exact-caked path is green through Rung 7.
- Bug/error: exact-caked preflight ordering/harness blockers are closed; Rung 2
  sensitivity drift remains the separate follow-up.
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
- Not validated: full fitter, baseline, GUI fit, and unrestricted feature
  combinations remain unclaimed. Full-beam validation is now green as bounded
  ladder evidence, but it is not the same as full fitter acceptance.

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
8. Rung 2 found 9 active parameters and 4 near-zero parameters, with 0 unsafe and 0 non-finite.
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

- Rung 2 proof. Green exemplar: `artifacts/geometry_fit_ladder/new4/20260421_183827/rung_02_sensitivity_scan.json`.

Old `new4_preflight_report.json` and `new4_fresh_all.json` may be stale or diagnostic only unless regenerated by the current scripts.

## Tests and commands that passed

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
- `active_param_count == 9`
- `near_zero_param_count == 4`
- `non_finite_param_count == 0`
- `unsafe_param_count == 0`
- `residual_probe_called == true`
- `least_squares_called == false`
- `optimizer_solve_called == false`
- `state_hash_unchanged == true`
- `provider_pair_count == 7`
- `fixed_source_pair_count == 7`
- `fallback_entry_count == 0`

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
| Rung 2 | sensitivity scan | green | 9 active, 4 near-zero, 0 non-finite, 0 unsafe |
| Rung 3 | one-parameter solves | green | singleton evidence usable for pair/block work |
| Rung 4 | paired solves | green | initial pair set passed |
| Rung 5 | cumulative blocks | green | fresh run `20260422_rung7_feature_gate_blocks`, 4/4 blocks passed |
| Rung 6 | selected combined solve / full-candidate dry run | green | fresh run `20260422_rung7_feature_gate_combined`, C2 passed |
| Rung 7 | controlled feature gate | green | `dynamic_reanchor`, `discrete_modes`, `seed_multistart`, `full_beam_polish`, and `identifiability_features` passed |

## Active and near-zero parameters from Rung 2

Active:

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

- `center_x`
- `center_y`
- `gamma`
- `Gamma`

`non_finite`: none.

`unsafe`: none.

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

Next project: the separate Rung 2 sensitivity drift follow-up
(`active_param_count=11`, `near_zero_param_count=2`). Do not reopen the
exact-caked preflight, 3B harness, or full_beam_polish paths in this track
unless a guard regresses.

Final frozen New4 chain:
`codex_final_blocks_20260423`, `codex_final_combined_20260423`, and
`codex_final_features_20260423` kept provider/caked/Rung 5/Rung 6 guards green
and passed `dynamic_reanchor`, `discrete_modes`, `seed_multistart`,
`full_beam_polish`, and `identifiability_features`. Exact-caked evidence stayed
present and the finalizer normalized `residuals_finite` without masking other
guard failures. `new4.json` stayed unchanged
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`), and
`full_fitter_validated == false` because the full fitter itself is still not
claimed.

Opt-in timing check `20260422_123330` measured the approved fresh Rung 5 blocks
path only (`--max-rung blocks`, `--timing-report`). It wrote
`rung_timing_summary.json` plus `latest_timing_summary.json`, listed Rungs
0/1/2/3/3B/4/5 only, and left threshold diagnostics `not_configured`.

## Do not run as acceptance

Do not use the old full baseline as the first next step.

Do not run full fitter, baseline, GUI fit button, unrestricted feature
combinations, auto-freeze/selective thaw, or feature-combo solves as full
acceptance. Rung 2 drift remains a separate follow-up.

Do not treat RMS/max baseline or full fitter behavior as validated yet.

## GitHub issue note

Issue [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) was updated with this checkpoint:

```text
New4 Rung 7 final frozen feature-gate checkpoint:

- Provider/caked/Rung 5/Rung 6 guards stayed green across the final frozen
  ladder chain.
- Rung 7 `codex_final_features_20260423`: full controlled feature sequence,
  no `--feature`, no full fit, no baseline, no GUI fit.
- Passed: `dynamic_reanchor`, `discrete_modes`, `seed_multistart`,
  `full_beam_polish`, `identifiability_features`.
- Exact-caked evidence stayed present and `residuals_finite` normalized
  without masking unrelated guard failures.
- `new4.json` hash stayed
  `f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`.
- `full_fitter_validated == false`; next work is the separate Rung 2
  sensitivity drift, not the exact-caked path.

```

## Links

- [Geometric fitter recovery](geometric-fitter-recovery.md)
- [Tracking hub](../index.md)
