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

- Raw `new2.json` is the primary frozen input and diagnostic source for the
  grouped-pick workflow.
- It already stores manual pairs with trusted full-reflection provenance and
  branch ids.
- It intentionally does not store transient `peak_records` or `q_group_rows`.
  Current code must regenerate those during validation.
- The milestone-6 green artifact is not raw `new2.json` by itself. It is the
  fresh canonical export derived from raw `new2.json` after all saved slots are
  re-emitted from current code and that export survives the saved-state
  compatibility / preflight sweep.
- Fresh current-code sentinel validation now proves that one mirrored sentinel
  pair can survive:
  - fresh emitted pair row creation
  - detector/caked no-fit redraw
  - session refresh/rebind
  - fresh save -> reload -> preflight
  - mirrored two-entry compatibility validation
- The remaining live blocker is therefore narrower than before. Current code can
  create and preserve a fresh canonical sentinel pair, but legacy saved states
  with stale source-row identity still fail at grouped-candidate regeneration /
  source-row identity alignment.
- Real-state harness work on `new2.json` still shows the first meaningful
  failure at that upstream producer/preflight seam, not in subset mapping and
  not in solver math.
- `new3.json` is a red diagnostic path for the same seam. It is useful because
  it exercises more saved pairs, but it is not an acceptance artifact until the
  upstream source-row / branch-stamping break is fixed.
- Current `new3.json` triage shows the slot-8 failure begins in live source-row
  emission. Branch 1 is missing before grouped-candidate collapse, so the next
  code change must target upstream source-row / branch stamping first and only
  revisit collapse if branch 1 later proves to be present pre-collapse and lost
  post-collapse.

## What Must Be True Next

- Fresh current-code success for one mirrored sentinel pair must now scale to
  all 9 saved pairs in `new2.json`.
- Every surviving pair must preserve these fields without loss:
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
- Only after those invariants hold should full geometry fit return as an
  acceptance test.

## Next Sequential Milestone Order

1. Fresh live candidate -> emitted pair row: done for sentinel
2. Emitted pair row -> detector/caked no-fit redraw: done for sentinel
3. Emitted pair row -> refresh/rebind: done for sentinel
4. Fresh one-pair save -> reload -> preflight: done for sentinel
5. Mirrored mate: done for fresh sentinel pair
6. Fresh-all sweep for all 9 pairs in raw `new2.json`, then compatibility /
   preflight on the exported fresh canonical state
7. Only then `preflight_normalized_pairs -> solver input` and the downstream
   fit seams

## Phase Exit Criteria

This recovery phase is complete when:

- one fresh mirrored sentinel pair is canonical at emission time
- that pair survives save/load and preflight unchanged
- raw `new2.json` can be re-emitted slot-by-slot into a fresh canonical export
- that exported `new2` state survives compatibility / preflight cleanly
- the same upstream invariants hold for the broader `new3.json` diagnostic path
- only then full geometry fit is reused as a meaningful acceptance check

Until then, treat fit output as secondary evidence. Primary signal remains seam
integrity at each boundary.
