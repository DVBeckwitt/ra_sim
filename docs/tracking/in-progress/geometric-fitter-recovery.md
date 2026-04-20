# Geometric fitter recovery

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-20

## Summary

The active project is to get the geometric fitter working reliably from the
current GUI saved state. The current acceptance path is no longer the older
`new2` or `new3` diagnostic states. The canonical state for this pass is
`new4.json`, followed by a freshly regenerated `new4_fresh_all.json` export.

The saved measured background point is the target. The live simulated point is
the movable candidate. For saved-pair recovery, select the live simulated row
that minimizes target-to-candidate display-space distance after normalized HKL
and branch filtering. Saved source identity is stale after regeneration and may
only break true ties or serve as diagnostics.

Source rebinding is now fixed for the canonical `new4` path: preflight resolves
`7/7` manual pairs, `background_distance_gate_ok` passes, and
`dataset_resolved_source_pair_count == 7`. Do not reopen source rebinding
unless a focused regression test proves it broke again.

The remaining blocker is manual caked fit-space projection provenance. Manual
caked residual rows must use the exact dataset projector built from the same
detector-to-caked conversion path as manual selection. When an exact caked
projector is available, `analytic_detector_fit_space` is forbidden for those
rows.

The next projects are intentionally blocked behind this one:

1. geometric fitter
2. mosaic fitter
3. structure-factor fitter
4. stacking-fault fitter

Manual Qr/Qz and HKL picker alignment is considered resolved and archived. Do
not reopen Qr selection, Q-group selection, caked peak selection, or manual GUI
selection behavior while completing this geometry baseline unless a new failure
proves that one of those paths is the direct cause.

## Current state

Use `new4.json` as the only live saved-state artifact for geometric-fit
validation.

Known `new4.json` contract:

- GUI state type is `ra_sim.gui_state`, version `1`.
- Saved timestamp is `2026-04-19T21:32:27`.
- `current_background_index` is `0`.
- There are 3 stored background image path entries.
- Stored background entries are saved paths, not guaranteed existing files.
- Only background index `0`, `Bi2Se3_5m_5d.osc`, has saved manual geometry
  pairs.
- The geometry manual-pair group has 7 entries: `(0,0,3)` and paired branches
  for `(-1,0,5)`, `(-1,0,10)`, and `(-1,0,16)`.
- `state.geometry.peak_records` is empty.
- `state.geometry.q_group_rows` is empty.

The empty transient rows are expected. The validator and baseline runner must
regenerate live simulation/source rows from the saved manual pairs rather than
requiring saved GUI peak rows.

Current root cause for the remaining `new4` fitter mismatch:

- manual caked selection already records exact caked `(2theta, phi)` through
  `native_detector_coords_to_caked_display_coords(...)`,
- optimizer residual code historically mixed in analytic
  `_detector_pixels_to_fit_space()` conversion,
- good correspondence can therefore coexist with wrong geometric-fit residual
  endpoints unless measured and simulated residual rows prove
  `dataset_fit_space_projector` provenance.

Current live-GUI blocker under verification is fit-space provenance, not source
rebinding. The canonical `new4` run must show endpoint-specific projection
diagnostics for every manual caked residual row and must report
`exact_fit_space_projector_available == true`.

Temporary test policy:

- Real `new4` geometry baseline fitting tests are skipped by default because the
  fitting path can hang or timeout.
- Enable those tests explicitly with `RA_SIM_RUN_SLOW_BASELINE_FITS=1`.
- This is not acceptance of the final baseline proof.
- Default CI and local default validation still cover correspondence, projector
  provenance, frame ownership, no analytic fallback, and serialization.
- The long-running optimizer-quality baseline remains a manual or opt-in gate
  until the runtime hang is resolved.

Next action: keep correspondence fixed, validate projector provenance on manual
caked residual rows, and classify any remaining `new4` failures as fit-space
projection failure, optimizer quality failure, or missing local
fixture/configuration.

The older `new2.json`, `new3.json`, `new2_fresh_all.json`, and
`new3_fresh_all.json` artifacts are retired as live acceptance gates. They may
remain only as historical diagnostics or archived notes.

## Required immediate work

1. Install `new4.json` as the repo-local canonical fixture under
   `artifacts/geometry_fit_gui_states/new4.json`.
2. If copying from the reviewed local `new4.json`, verify this SHA256 first:
   `F5BF185EBCFBFA8B32F161CC4BD781E177175DAD84B6FCE4D563F23CA021EF36`.
3. Hard-rename `scripts/debug/validate_new2_preflight_rebind.py` to
   `scripts/debug/validate_geometry_preflight_rebind.py`.
4. Delete the old helper path. Do not keep a shim, alias, wrapper, or
   deprecated entry point.
5. Update every active reference to the generic helper name, including tests,
   loader names, CLI help, temp prefixes, output keys, docs commands, and
   tracking commands.
6. Remove active new2/new3 defaults from the baseline runner and tests.
7. Regenerate `new4_fresh_all.json` from `new4.json` before running the saved
   state baseline.

## Preflight regeneration command

After the hard rename exists, regenerate the canonical fresh export with:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --mode full `
  --export-fresh-state artifacts/geometry_fit_gui_states/new4_fresh_all.json `
  > artifacts/geometry_fit_gui_states/new4_preflight_report.json
```

Preflight must prove:

- background index `0` is selected,
- 7 bg0 manual entries are processed,
- all 7 bg0 manual entries bind to stable simulated candidates,
- live simulation/source rows are regenerated despite empty saved
  `peak_records` and `q_group_rows`,
- branch mismatches are `0`,
- `background_distance_gate_ok` is true using saved measured background points
  against live simulated display points in the same frame,
- the fresh export is written cleanly.

Define `preflight_valid_count` as the number of bg0 manual entries with
successful selected-candidate binding. For canonical `new4.json`, the expected
value is `7`.

## Saved-state baseline gate

Run the baseline only after `new4_fresh_all.json` has been regenerated from the
canonical `new4.json` fixture:

```powershell
python scripts/debug/run_geometry_fit_quality_baseline.py `
  artifacts/geometry_fit_gui_states/new4_fresh_all.json
```

Capture at least:

- `accepted`,
- `rejection_reason`,
- `before_rms_px`,
- `after_rms_px`,
- before-fit and after-fit max error,
- fixed-pair counts before and after,
- missing fixed-pair counts before and after,
- outside-radius counts before and after,
- `branch_mismatch_count`,
- `full_beam_polish_*`,
- `full_beam_start_vector_source`,
- `seed_correspondence_count`,
- `final_metric_name`,
- `nfev`,
- `elapsed_s`,
- `stage_timing_s`.

Gate requirements:

- Correspondence acceptance:
  `matched_fixed_pair_count_after == preflight_valid_count`.
- `missing_fixed_pair_count_after <= missing_fixed_pair_count_before`.
- `branch_mismatch_count == 0`.
- `resolved_source_pair_count == preflight_valid_count`.
- Top-level rejection reason is not
  `No matched peak pairs were available for the fitted solution.`
- Fit-quality acceptance:
  `after_rms_px` stays close to or improves the measured before-fit RMS.
- After-fit max error stays close to or improves the measured before-fit max
  error. If correspondence is correct but fit quality regresses, keep artifacts
  and treat that as a separate fitter-quality issue, not a rebinding failure.

For canonical `new4.json`, expected passing fixed-pair shape is:

- `matched_fixed_pair_count_after == 7`,
- `missing_fixed_pair_count_after == 0`.

Do not reuse the old 6 to 8 px RMS band. Measure `new4` first and compare the
fit against its own fresh before-fit values.

## Alignment-first gate

`new4.json` and its regenerated `new4_fresh_all.json` export are the live gate.
The older `new2` and `new3` states are historical diagnostics only and must not
drive current acceptance decisions.

Treat the saved-state baseline as passed only when background-to-live-simulated
display-space correspondence is correct and detector-space evidence then shows
either:

- a real residual improvement against the fresh before-fit detector baseline, or
- a proven no-op optimum where the best valid raw-detector candidate is the
  retained start and that retained start already satisfies the fixed-pair gate.

Keep a candidate ledger for every saved-state run. Final selection must be
explainable from the saved background point to the live simulated candidate in
the active display frame. Report `background_distance_px`,
`max_background_distance_px`, and `background_distance_gate_ok`. Old
`candidate_*` / `detector_*` gate names may remain only as exact deprecated
aliases while downstream consumers migrate.

`retained_start_safe_fallback` is not a fit-quality pass by default. It is only
acceptable when the raw-detector candidate ledger proves the retained start is
the best valid no-op optimum instead of a masked timeout or degraded solve.

## Stop criteria for #249

- Focused correspondence tests pass.
- Runtime/debug parity tests pass.
- Frame-safety and tie/duplicate tests pass.
- Non-gating source-cache/caked-view tests pass.
- Logged-cache `params_mismatch` is fast by construction: meta/signature
  rejection before heavy hit-table load.
- Geometry-fit preflight uses `manual_geometry_targeted` mode when saved manual
  geometry picks exist.
- Geometry-fit preflight collects required branch-group keys before
  source-cache rebuild.
- Geometry-fit preflight passes required branch-group keys to the targeted
  source-generation path.
- If targeted fresh simulation is supported, only required branch groups are
  simulated.
- If cached full data already exists, preflight may filter it, but must not
  reproject or rescore unrelated rows.
- Uncached full fresh simulation fallback is diagnosed and does not pass the
  targeted performance gate.
- Repeated unchanged preflight reuses targeted projected cache and does not
  fresh-simulate, rebuild full source rows, or project the full source table.
- `new4` preflight reports `7/7` bound, zero missing, zero branch mismatch,
  `background_distance_gate_ok=true`, `runtime_prepare_ok=true`,
  `fresh_export_ok=true`, `resolved_source_pair_count=7`, and
  `targeted_performance_gate.ok=true`.
- Manual caked residual rows report
  `simulated_fit_space_source == "dataset_fit_space_projector"`.
- Manual caked measured rows with known detector/native frame and no explicit
  override report
  `measured_fit_space_source == "dataset_fit_space_projector"`.
- `analytic_detector_fit_space` is absent from exact-projector manual caked
  rows.
- Canonical `new4` baseline reports
  `exact_fit_space_projector_available == true`.
- `new4` baseline reports `7` matched fixed pairs, zero missing fixed pairs,
  zero branch mismatch, no `No matched peak pairs were available` rejection,
  `after_rms_px <= before_rms_px + 0.25`, and
  `after_max_error_px <= before_max_error_px + 1.0`.
- GUI logs reach `source_cache_build_ready` and do not block that event on
  caked-view work.

## Targeted preflight stop criteria

- Geometry-fit preflight source generation, expansion, projection, and
  background-distance scoring are limited to the branch groups required by the
  saved manual geometry picks.
- The targeted projected source-row cache is keyed by
  `required_hkl_branch_keys_digest`, not by measured click coordinates.
- Moving only the saved measured background point changes the manual-target
  scoring digest and rescoring work, but does not force a targeted source-row
  rebuild or reprojection.
- `required_branch_group_key` uses the richest stable identity available:
  normalized HKL, `source_branch_index` when present, and `q_group_key` /
  `source_q_group_key` / `branch_group_key` when available.
- Canonical `new4` pairs must not degrade to
  `branch_constraint_status == "unconstrained_missing_branch"`.
- `candidate_rows_scored_for_background_distance` contains only required branch
  groups.
- `unrelated_projected_row_count_for_rebinding == 0`.
- `unrelated_scored_row_count_for_rebinding == 0`.
- `full_source_rows_built_for_rebinding == false`.
- `full_source_rows_projected_for_rebinding == false`.

## Red flags

- Candidate distances in hundreds or thousands of pixels.
- Internal use of `candidate_distance_*` instead of `background_distance_*`.
- `resolved_source_pair_count == 0`.
- Runtime/debug selected candidates disagree.
- `params_mismatch` takes seconds or minutes.
- Fresh simulation runs on unchanged repeated preflight.
- Full-table source-row build or projection runs on unchanged repeated
  preflight.
- `source_rows_projected_for_rebinding` is approximately equal to
  `total_source_rows_available` on unchanged targeted preflight.
- `candidate_rows_scored_for_background_distance` includes unrelated HKL or
  branch rows.
- `source_cache_build_ready` waits for caked-view work.
- `targeted_simulation_fallback_reason ==
  "simulator_filter_not_supported"` during an uncached fresh preflight.
- Full `733181`-row source build or projection occurs during unchanged targeted
  geometry-fit preflight.
- Baseline rejects with
  `No matched peak pairs were available for the fitted solution.`

Once all stop criteria pass, stop changing fitter or rebinding logic. Future
changes require a newly failing test or a measured performance regression.

## Out of scope for this pass

Do not touch these areas while completing the `new4` saved-state baseline:

- Qr selection,
- Q-group selection,
- GUI pick behavior,
- caked peak selection,
- `ra_sim/gui/peak_selection.py`,
- `ra_sim/gui/geometry_q_group_manager.py`,
- manual GUI selection code,
- mosaic fitting,
- structure-factor fitting,
- stacking-fault fitting.

## Validation

Focused checks:

```powershell
pytest tests/test_geometry_fit_quality_baseline.py
pytest tests/test_gui_geometry_fit_workflow.py
pytest tests/test_geometry_fitting.py
pytest tests/test_manual_geometry_live_peak_cache.py
pytest tests/test_gui_runtime_import_safe.py
pytest tests/test_gui_bootstrap.py
```

Canonical saved-state gate:

```powershell
python scripts/debug/run_geometry_fit_quality_baseline.py `
  artifacts/geometry_fit_gui_states/new4_fresh_all.json
```

Full repo gates:

```powershell
python -m ra_sim.dev check
python -m ra_sim.dev lint
python -m ra_sim.dev typecheck
python -m ra_sim.dev test-all
```

Final stale-reference check:

Run a repository grep for the retired new2/new3 helper names, fixture names,
loader names, temp prefixes, output keys, and fresh-state artifacts. The only
acceptable hits are migration notes that explicitly mark the names retired,
archival notes, and old diagnostic artifacts. Active tests, script defaults,
helper names, imports, loaders, CLI help, tracking commands, and fixture paths
must not use the retired names.

## Links

- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Mosaic fitter plan](../planned-features/mosaic-fitter.md)
- [Structure-factor fitter plan](../planned-features/structure-factor-fitter.md)
- [Stacking-fault fitter plan](../planned-features/stacking-fault-fitter.md)
