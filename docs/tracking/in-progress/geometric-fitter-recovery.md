# Geometric Fitter Recovery

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-21

## Summary

Point-provider parity is fixed for the manual-geometry handoff layer. The
manual Qr picker saved/refined pair, the geometry-fit point provider pair, and
the actual dataset handoff row now agree before any optimizer entrypoint runs.

This closes the current point-provider bug/error scope: stale source locators
are diagnostic when the saved picker assignment still resolves semantically,
picker-owned saved/refined simulated points overwrite live/caked prefill, and
`new4` reports 7/7 provider pairs without launching the optimizer.

The geometric optimizer hang/convergence problem remains explicitly out of
scope for this work item. RMS/max-error improvement and
`run_geometry_fit_quality_baseline.py` are optional later diagnostics, not the
stop gate for point-provider parity.

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
- Acceptance depends on RMS/max-error improvement or baseline convergence.
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
python -m py_compile ra_sim/gui/manual_geometry.py ra_sim/gui/geometry_fit.py scripts/debug/validate_geometry_preflight_rebind.py
pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer or canonical_geometry_source_identity" -vv
```

Pinpoint regression gate:

```powershell
pytest tests/test_gui_geometry_fit_workflow.py::test_point_provider_stale_locator_is_diagnostic_when_saved_assignment_resolves tests/test_gui_geometry_fit_workflow.py::test_point_provider_marks_stale_saved_identity_as_fallback tests/test_gui_geometry_fit_workflow.py::test_point_provider_saved_refined_sim_point_overwrites_caked_prefill -q
```

Provider-only validator check:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --mode full `
  --point-provider-report-only `
  --report-path artifacts/geometry_fit_gui_states/new4_preflight_report.json
```

Do not use `tests/test_geometry_fit_quality_baseline.py` or
`run_geometry_fit_quality_baseline.py` as acceptance for this point-provider
task. Optimizer hang/convergence belongs to a later optimizer issue.

## Links

- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
