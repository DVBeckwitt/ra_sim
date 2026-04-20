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

- `matched_fixed_pair_count_after == preflight_valid_count`.
- `missing_fixed_pair_count_after <= missing_fixed_pair_count_before`.
- `branch_mismatch_count == 0`.
- Top-level rejection reason is not
  `No matched peak pairs were available for the fitted solution.`
- `after_rms_px` stays close to or improves the measured before-fit RMS.
- After-fit max error stays close to or improves the measured before-fit max
  error.

For canonical `new4.json`, expected passing fixed-pair shape is:

- `matched_fixed_pair_count_after == 7`,
- `missing_fixed_pair_count_after == 0`.

Do not reuse the old 6 to 8 px RMS band. Measure `new4` first and compare the
fit against its own fresh before-fit values.

## Solver patch rule

Do not touch `ra_sim/fitting/optimization.py` unless `new4_fresh_all.json`
reproduces the exact all-missing fixed-correspondence failure shape:

- the requested or retained start vector has fixed-source matches,
- the current/final vector has zero fixed-source matches under the
  fixed-correspondence evaluator,
- full-beam polish candidate is rejected,
- top-level failure says no matched peak pairs.

If that exact shape appears, add the retained-start regression first, then patch
only the retained-start path. If `new4` passes, stop without applying the old
new2/new3 retained-start patch. If `new4` fails differently, diagnose that new
failure shape first.

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
