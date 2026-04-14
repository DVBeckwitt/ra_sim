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
- Use the canonical reference for implementation detail and code routing.
- Treat full geometry fit as a late acceptance signal, not the first diagnostic.

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
  Current earliest failing seam on both `new2_fresh_all.json` and
  `new3_fresh_all.json` is the last gate:
  `final_metric_name` stays `central_point_match` instead of reaching
  `full_beam_fixed_correspondence`, even though canonical identity now survives
  through seed correspondence and full-beam identity coverage.

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
  subset mapping, seed correspondence, and full-beam identity coverage; fails
  final full-beam fixed-correspondence acceptance because
  `final_metric_name=central_point_match`
- `new3_fresh_all.json`: same current failure shape
- earliest remaining seam is now full-beam fixed-correspondence acceptance, not
  earlier identity transport

Until then, treat fit output as secondary evidence. Primary signal remains seam
integrity at each boundary.
