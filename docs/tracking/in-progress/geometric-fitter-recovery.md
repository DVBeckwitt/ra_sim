# Geometric Fitter Recovery

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-21

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
uses detector provenance (`sim_col_raw/sim_row_raw` or detector-projected
coordinates) instead of stale caked active-view display fields, so picking works
after detector-to-caked and caked-to-detector view changes.

The geometric optimizer hang/convergence problem is now handled by
`scripts/debug/run_new4_geometry_fit_ladder.py`, not by the old full baseline
as the first debug tool. The ladder is currently stopped at rung 1 until the
optimizer request has zero fallback rows.

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
  any optimizer call when the request would use fallback rows. Current `new4`
  rung 1 reports 7 provider pairs, 7 optimizer request pairs, 7 fallback rows,
  7 missing fixed-source rows, and `optimizer_called == false`.
- The dataset-to-optimizer bridge now copies provider canonical identity and
  measured-point fields into the optimizer request, emits handoff counters on
  every ladder report, and rejects request-side fallback before `least_squares`.
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
- Detector-mode Qr/Qz selection now rejects stale caked display fields as
  detector click coordinates. Structural cross-view tests cover detector to
  caked, caked to detector, and stale caked-cache candidates with valid
  detector provenance.
- Solve rungs are disabled operationally until objective dry-run reports zero
  `fallback_row_count` and zero `fixed_source_resolution_fallback_count`.

## Next Actions

- Fix only the dataset-to-optimizer request bridge so the validated provider
  handoff supplies the fixed-source fields required by the optimizer resolver.
- Keep provider logic closed unless the provider-only parity gate regresses.
- Keep Qr/Qz branch seed behavior closed unless raw-cache preview, manual
  toggle, refresh, or place setup regresses to either every raw ray or one
  whole-group-only ray.
- Keep caked-to-detector Qr/Qz return behavior closed unless the same
  simulated `2theta,phi` seed no longer redraws at the same detector marker
  position after switching view or refreshing, or Qr/Qz seed selection stops
  recognizing the same visible marker after switching detector/caked views.
- Do not run solve rungs, tune parameters, loosen fallback rules, or run the old
  baseline until rung 1 has seven fixed-source pairs and zero fallback rows.

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
  --max-rung center
```

Do not use `run_geometry_fit_quality_baseline.py` as the first optimizer debug
tool. Run it only after the ladder identifies a stable parameter set.

## Links

- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
