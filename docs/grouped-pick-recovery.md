# Grouped-pick recovery status

This page tracks current recovery work for the grouped-pick manual geometry
workflow. It explains what is already stable, what is currently broken, and
what must be true before end-to-end geometry fitting is trusted again.

See also:

- [docs index](index.md)
- [Architecture guide](architecture.md)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)

## Purpose

- Use this page as the short project-status map for grouped-pick recovery.
- Use this page as the resume document if chat history is lost.
- Use the canonical reference for implementation detail and code routing.
- Treat full geometry fit as a late acceptance signal, not the first diagnostic.

## Resume Here

- Primary raw diagnostic inputs:
  - `C:\Users\Kenpo\.local\share\ra_sim\new2.json`
  - `C:\Users\Kenpo\.local\share\ra_sim\new3.json`
- Primary fresh canonical exports:
  - `C:\Users\Kenpo\.local\share\ra_sim\new2_fresh_all.json`
  - `C:\Users\Kenpo\.local\share\ra_sim\new3_fresh_all.json`
- Sentinel one-pair artifacts:
  - `C:\Users\Kenpo\.local\share\ra_sim\Sentinel.json`
  - `C:\Users\Kenpo\.local\share\ra_sim\Sentinel_fresh_pair.json`
  - `C:\Users\Kenpo\.local\share\ra_sim\Sentinel_fresh_slot0.json`
- Most relevant commits so far:
  - `de26d3b` add preflight rebind validator
  - `06a88f7` add fresh pair preflight rebind test coverage
  - `a65b3f3` validate fresh-all pair identity
  - `5bf415e` align milestone-6 gate so `--mode full` means `fresh-all`
  - `0dc99e8` add downstream identity harness and expose current last seam
  - `516d965` remap trusted full-reflection replay into subset-local tables
- Main script to resume from:
  - `scripts/debug/validate_new2_preflight_rebind.py`
- Main code paths touched by this recovery:
  - `ra_sim/gui/geometry_q_group_manager.py`
  - `ra_sim/gui/geometry_fit.py`
  - `ra_sim/fitting/optimization.py`
  - `tests/test_gui_geometry_fit_workflow.py`
  - `tests/test_geometry_fitting.py`
- If you only remember one sentence:
  raw `new2/new3` are diagnostic sources, fresh-all exports are the acceptance
  artifacts, identity transport and trusted full-reflection replay are green,
  and both `new2_fresh_all.json` and `new3_fresh_all.json` now end at
  `full_beam_fixed_correspondence`.

## What Changed In Approach

- `new.json` is no longer the primary truth source. Keep it as a legacy control
  because it still carries dense legacy ids and the old failure mode.
- Canonical identity is now `(source_reflection_index, source_branch_index)`.
- Full-reflection trust is only valid when
  `source_reflection_namespace="full_reflection"` and
  `source_reflection_is_full=True`.
- Optimizer trust stays fail-closed.
- Validation now follows a phase-trace / seam-testing strategy so each boundary
  can be checked directly instead of inferred from a final fit result.

## What Is Already Done

- Downstream identity plumbing is in place. Once canonical identity exists, the
  current code can carry it through later stages.
- Targeted seam validation already proved:
  - cache-preferred vs rebuild active-pair parity on the cache-provenance seam
  - subset reduction preserves trusted reflection and branch identity
  - optimizer-bridge metadata continuity into diagnostics and seed
    correspondence
- The old bad run is now understood as a multi-seam failure, not one isolated
  bug. It could show healthy-looking fit counters and still fail in detector
  space.
- Because of that, seam validation now comes before end-to-end fit acceptance.

## Current State

- Raw `new2.json` is primary frozen input and diagnostic source for grouped-pick
  workflow.
- It already stores manual pairs with trusted full-reflection provenance and
  branch ids.
- It intentionally does not store transient `peak_records` or `q_group_rows`.
  Current code must regenerate those during validation.
- The milestone-6 green artifact is not raw `new2.json` by itself. It is the
  fresh canonical export derived from raw `new2.json` after all saved slots are
  re-emitted from current code and that export survives the saved-state
  compatibility / preflight sweep.
- `new2_fresh_all.json` is current primary green artifact. Raw `new2.json`
  re-emits `9/9` fresh slots, exported canonical state passes compatibility /
  preflight, and that export is acceptance target for milestone 6.
- `new3_fresh_all.json` is current secondary green control. Raw `new3.json`
  re-emits `11/11` fresh slots, including old slot-8 seam, and exported
  canonical state also passes compatibility / preflight.
- Fresh current-code sentinel validation proved one mirrored sentinel pair can
  survive:
  - fresh emitted pair row creation
  - detector/caked no-fit redraw
  - session refresh/rebind
  - fresh save -> reload -> preflight
  - mirrored two-entry compatibility validation
- Milestone 6 is complete. Upstream source-row / branch-stamping seam that broke
  `new3` slot 8 is fixed in live source-row emission, before grouped-candidate
  collapse.
- Next active work is downstream identity validation from preflight-normalized
  pairs through solver input, subset mapping, seed correspondence, and
  full-beam correspondence.
- The downstream harness now exists and runs on fresh canonical exports.
  The old full-beam replay failure is fixed: trusted full-reflection seed
  records now remap into subset-local full-beam tables instead of dying with
  `source_table_out_of_range`.
- Latest diagnostics-only pass now retains both
  `start_point_match_diagnostics` / `candidate_point_match_diagnostics` and
  both summaries inside `full_beam_polish_summary`.
- The downstream validator now reads those optimizer-captured diagnostics
  directly, pairs them by stable identity key, carries weight fields through
  comparison output, and promotes coverage failure to explicit top-level
  classification `identity_key_coverage_mismatch`.
- Full-beam behavior pass now keeps resolved fixed correspondences resolved
  during full-beam polish even when detector distance exceeds the full-beam
  radius, and marks those rows with
  `resolution_reason="outside_match_radius"`.
- Downstream validator now accepts `match_kind="full_beam_fixed"` as valid
  full-beam fixed correspondence output.
- Final optimizer pass now promotes full-beam polish using the all-resolved
  fixed-correspondence point-match objective basis, excluding priors. Raw
  detector-space peak metrics stay in `full_beam_polish_summary` as
  diagnostic-only fields.
- After that final pass:
  - `new2_fresh_all.json` passes downstream identity fully and ends with
    `final_metric_name=full_beam_fixed_correspondence`
  - `new3_fresh_all.json` also passes downstream identity fully and ends with
    `final_metric_name=full_beam_fixed_correspondence`

## Implemented Recovery Work

- Replaced raw `new.json` as primary acceptance target with fresh canonical
  exports derived from current code.
- Added canonical identity handling around
  `source_reflection_index`, `source_reflection_namespace`,
  `source_reflection_is_full`, `source_branch_index`, and
  `source_peak_index`.
- Added seam-first validator modes:
  - single-slot `fresh`
  - all-slot `fresh-all`
  - saved-state `compatibility`
  - downstream chain `downstream-identity`
- Added saved-to-selected identity delta reporting that ignores `pair_id` and
  distinguishes already-canonical rows from legacy rows that canonicalize
  cleanly.
- Fixed upstream live source-row / branch-stamping seam for `new3` slot 8 in
  source-row emission before grouped-candidate collapse.
- Added trusted single-row source fallback in downstream source resolution so
  branch-ambiguous but canonical trusted rows do not fail early just because
  later branch recovery cannot split a one-row table.
- Fixed trusted full-reflection replay at full-beam stage so frozen global
  reflection ids remap into subset-local `hit_tables` before correspondence
  resolution.
- Added fail-closed rejection for trusted full-reflection ids that are not
  present in the active subset:
  `trusted_full_reflection_index_not_in_subset`.
- Added full-beam replay diagnostics that preserve
  `frozen_table_index`, `frozen_table_namespace`,
  `resolved_table_index`, and `trusted_full_reflection_remapped`.
- Added diagnostics retention for rejected full-beam polish so
  `full_beam_polish_summary` now keeps:
  `start_point_match_diagnostics`, `candidate_point_match_diagnostics`,
  `start_point_match_summary`, and `candidate_point_match_summary`.
- Updated downstream validator to pair optimizer-captured start/candidate
  full-beam diagnostics directly instead of recomputing groups, and to emit
  weighted residual fields in paired comparison output.
- Added explicit validator classification
  `identity_key_coverage_mismatch` for selected-side full-beam coverage
  failures before fixed-correspondence promotion.
- Kept resolved full-beam fixed correspondences as matched rows during
  full-beam polish even when they exceed the full-beam radius, while exposing
  `outside_match_radius` in diagnostics.
- Updated downstream validator to treat `match_kind="full_beam_fixed"` as valid
  full-beam fixed-correspondence output.
- Added regression coverage for:
  - stale sentinel-id canonicalization reporting
  - fresh-all export ordering and empty transient lists
  - mirrored live source-row branch emission
  - downstream input-contract rejection
  - downstream earliest-failure stop behavior
  - downstream happy-path identity preservation
  - downstream optimizer-captured full-beam start/candidate comparison
  - downstream explicit coverage-mismatch classification promotion
  - full-beam resolved fixed correspondence retained outside radius
  - trusted deadband source-row fallback
  - trusted full-reflection replay remap into subset-local tables
  - fail-closed trusted full-reflection replay miss
  - rejected full-beam polish retaining both start/candidate diagnostic sets

## Validation Commands

- Rebuild fresh-all export from raw `new2.json`:
  ```powershell
  python scripts/debug/validate_new2_preflight_rebind.py --state "C:\Users\Kenpo\.local\share\ra_sim\new2.json" --background-index 0 --mode full --export-fresh-state "C:\Users\Kenpo\.local\share\ra_sim\new2_fresh_all.json"
  ```
- Rebuild fresh-all export from raw `new3.json`:
  ```powershell
  python scripts/debug/validate_new2_preflight_rebind.py --state "C:\Users\Kenpo\.local\share\ra_sim\new3.json" --background-index 0 --mode full --export-fresh-state "C:\Users\Kenpo\.local\share\ra_sim\new3_fresh_all.json"
  ```
- Run downstream identity harness on fresh canonical export:
  ```powershell
  python scripts/debug/validate_new2_preflight_rebind.py --state "C:\Users\Kenpo\.local\share\ra_sim\new2_fresh_all.json" --background-index 0 --mode downstream-identity
  ```
- Current targeted test slices:
  ```powershell
  python -m pytest tests/test_gui_geometry_fit_workflow.py -k "probe_main_aliases_full_to_fresh_all or downstream_identity_validation_rejects_non_canonical_input or downstream_identity_validation_stops_at_subset_drift or downstream_identity_validation_preserves_canonical_identity or solver_request_to_subset_mapping_and_seed_correspondence_preserves_trusted_identity"
  python -m pytest tests/test_geometry_fitting.py -k "trusted_deadband_source_row_fallback or legacy_branch_alias or keeps_distinct_branches or preserves_trusted_identity_payload"
  python -m pytest tests/test_geometry_fitting.py -k "full_beam_polish_remaps_trusted_full_reflection_indices_into_subset_local_tables or trusted_full_reflection_index_missing_from_subset"
  python -m pytest tests/test_geometry_fitting.py -k "full_beam_polish_rejects_unweighted_rms_regression or full_beam_polish_rejection_preserves_central_point_match_result"
  python -m pytest tests/test_gui_geometry_fit_workflow.py -k "downstream_identity_validation_uses_optimizer_captured_full_beam_diagnostics or downstream_identity_validation_promotes_coverage_mismatch_classification"
  ```

## What To Not Forget

- `--mode full` is now an alias for `fresh-all`, not the old single-sentinel
  path.
- `downstream-identity` is intentionally fail-closed and must only consume
  fresh canonical exports with empty `peak_records` and `q_group_rows`.
- Canonical equality excludes `pair_id`. Keep `pair_id` as diagnostic only.
- Raw full-beam diagnostic list order is not the acceptance order. Coverage is
  checked first; fixed-correspondence comparison is deterministic.
- On rejected full-beam polish, truth source is
  `full_beam_polish_summary.start_point_match_diagnostics` /
  `candidate_point_match_diagnostics`, not recomputed validator groups.
- `identity_key_coverage_mismatch` now means selected-side full-beam output
  still contains unresolved or duplicate fixed correspondences, even if
  optimizer-captured start/candidate sets remain stable.
- Do not use full end-to-end fit as the primary gate again until the final
  full-beam fixed-correspondence acceptance seam is green.

## What Must Be True Next

- Fresh canonical exports must preserve these fields without loss:
  `hkl`, `source_reflection_index`, `source_reflection_namespace`,
  `source_reflection_is_full`, `source_branch_index`, or
  `source_peak_index`.
- Boundaries that must preserve identity:
  - session refresh / rebind
  - save -> reload
  - preflight rebinding
  - solver input
  - subset mapping
  - seed correspondence
  - full-beam correspondence
- Only after those invariants hold on fresh canonical exports should full
  geometry fit return as primary acceptance test.

## Next Sequential Milestone Order

1. Fresh live candidate -> emitted pair row: done for sentinel
2. Emitted pair row -> detector/caked no-fit redraw: done for sentinel
3. Emitted pair row -> refresh/rebind: done for sentinel
4. Fresh one-pair save -> reload -> preflight: done for sentinel
5. Mirrored mate: done for fresh sentinel pair
6. Fresh-all sweep for all 9 pairs in raw `new2.json`, then compatibility /
   preflight on exported fresh canonical state: done
7. `preflight_normalized_pairs -> solver input -> subset mapping -> seed
   correspondence -> full-beam correspondence`: active

## Phase Exit Criteria

Milestone 6 completed when:

- one fresh mirrored sentinel pair is canonical at emission time
- that pair survives save/load and preflight unchanged
- raw `new2.json` can be re-emitted slot-by-slot into a fresh canonical export
- that exported `new2` state survives compatibility / preflight cleanly
- the same upstream invariants hold for the broader `new3.json` diagnostic path
- only then downstream identity seam chain becomes next active gate

Milestone 7 completes when:

- `new2_fresh_all.json` preserves canonical identity through
  `preflight_normalized_pairs`
- same canonical identity survives solver request measured peaks
- same canonical identity survives subset mapping
- same canonical identity survives seed correspondence
- same canonical identity survives full-beam fixed correspondence
- `new3_fresh_all.json` passes same downstream identity chain
- only then full geometry fit is reused as meaningful acceptance check

Current Milestone 7 status:

- `new2_fresh_all.json`: passes input contract, preflight, solver request,
  subset mapping, seed correspondence, and trusted full-reflection replay
  remap and now reaches
  `final_metric_name=full_beam_fixed_correspondence`
- `new3_fresh_all.json`: passes input contract, preflight, solver request,
  subset mapping, seed correspondence, trusted full-reflection replay remap,
  and full-beam fixed correspondence and now also reaches
  `final_metric_name=full_beam_fixed_correspondence`
- Milestone 7 is complete on current code. The next objective is no longer seam
  repair; it is end-to-end geometry-fit quality on the background images.

With milestone 7 closed, seam integrity is established. Treat accepted
end-to-end geometry fit and background-overlay quality as the primary signal
again.
