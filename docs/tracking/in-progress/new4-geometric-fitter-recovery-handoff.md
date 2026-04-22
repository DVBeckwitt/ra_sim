# New4 Geometric Fitter Recovery Handoff

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-22

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
- No full, feature, baseline, GUI fit, dynamic reanchor, multistart, polish, or
  feature rung should be treated as validated.

This handoff is the bounded-through-Rung-5 recovery state for `new4`. Do not
use it as approval for full, feature, GUI, or baseline solves.

Status by work type:

- Feature: fresh same-run Rung 5 blocks are green for New4 ladder validation.
- Bug/error: Rung 5 evidence handling is fixed so fatal evidence aborts stay
  fatal, local `a` usability failures stay local, and missing dependencies skip
  only affected blocks.
- Timing feature: current-run Rung 0-5 timing JSON and stdout table are
  available for opt-in ladder runs.
- Timing bug/error: review follow-up is closed. Timing collection excludes
  Rung 6/7 path mappings and expected IDs, Rung 5 skipped reports are timed,
  and `RA_SIM_NEW4_LADDER_TIMING_MAX_S` never gates status or exit code.
- Manual selected-point fit bug/error: default GUI fits no longer inherit
  `max_nfev: 400`, parallel orchestration, or identifiability diagnostics for a
  few selected spots. Unsafe parallel runtime and richer dynamic point fitting
  remain explicit paths.
- Ladder lean bug/error: finite-difference identifiability diagnostics no
  longer run on every fast ladder solve. The identifiability feature run remains
  the explicit diagnostic path.
- Ladder heartbeat bug/error: running heartbeat files no longer rewrite stale or
  growing `residual_eval_trace` payloads on every evaluation. Timeout progress
  keeps `last_residual_eval`, counters, timing, bounds, and solver context flags.
- Not validated: full fitter, feature rung, baseline, GUI fit, dynamic reanchor,
  multistart, and polish remain unclaimed.

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
| Rung 5 | cumulative blocks | green | fresh same-run run `20260422_115256`, 4/4 blocks passed |
| Rung 6 | selected combined solve / full-candidate dry run | not started | separate approval required |

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

Next project: Rung 6 selected combined solve / full-candidate dry run.

Fresh same-run Rung 5 is accepted for New4 ladder validation. The acceptance run
was `20260422_115256`: Rung 5 `status == "ok"`, four attempted blocks, four
passed blocks, zero failed/skipped blocks, provider guard after blocks green,
and unchanged `new4.json`
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`).
`full_fitter_validated == false`.

Rung 6 remains separate and unstarted. Do not run dynamic reanchor, multistart,
polish, GUI fit, baseline, feature rung, freeze/thaw, or full fitter validation
until Rung 6 is separately approved.

Opt-in timing check `20260422_123330` measured the approved fresh Rung 5 blocks
path only (`--max-rung blocks`, `--timing-report`). It wrote
`rung_timing_summary.json` plus `latest_timing_summary.json`, listed Rungs
0/1/2/3/3B/4/5 only, and left threshold diagnostics `not_configured`.

## Do not run as acceptance

Do not use the old full baseline as the first next step.

Do not run full, feature, baseline, GUI fit button, multistart, full-beam polish,
dynamic reanchor, auto-freeze/selective thaw, or feature rung as acceptance.

Do not treat RMS/max baseline or full fitter behavior as validated yet.

## GitHub issue note

Issue [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) was updated with this checkpoint:

```text
New4 geometric fitter recovery checkpoint:

- Provider parity closed.
- Visual/backend coordinate parity closed.
- Rung 1 objective dry-run green.
- Rung 2 sensitivity scan green.
- Rung 3 one-parameter solves have usable bounded evidence.
- Rung 3A `a` diagnosis is usable.
- Rung 3B caked-point reprojection guard is green.
- Rung 4 initial pairs are green.
- Rung 5 fresh same-run blocks are green: run `20260422_115256`, 4/4 blocks
  passed, provider guard after blocks green, `new4.json` unchanged.
- No full, feature, baseline, GUI fit, dynamic reanchor, multistart, polish, or
  feature rung validated yet.
```

## Links

- [Geometric fitter recovery](geometric-fitter-recovery.md)
- [Tracking hub](../index.md)
