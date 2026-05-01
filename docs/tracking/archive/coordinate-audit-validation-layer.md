# Coordinate Audit Validation Layer

Status: complete
Type: bug fix and validation feature
Last updated: 2026-05-01

## Problem

The New4 manual-point audit still enforced an old invariant:
detector-native reprojection had to equal the objective target. That was stale.
The final fitter contract uses cached caked 2theta/phi as the fixed objective
target, dynamic `sim_visual_caked_deg` as the source, and angular
source-minus-target residuals in degrees with wrapped phi.

The stale check made a detector-native diagnostic mismatch look like a
coordinate failure even when objective target, dynamic source, and residuals
agreed.

## Change

- Promoted coordinate audit JSON to the authoritative validation artifact.
- Kept PNG output diagnostic-only.
- Relabeled detector-native reprojection as diagnostic, not pass/fail target.
- Added explicit cached target, optimizer measured target, dynamic source,
  optimizer source, residual, frame/unit/identity, and diagnostic reprojection
  fields.
- Added residual vector gates proving flat residual slots are
  `[source_2theta - target_2theta, wrapped(source_phi - target_phi)]`.
- Added fit-improvement audit mode:
  `--fit-improvement-audit`, `--perturb-start`, and `--active-vars`.
- Added controlled perturb solve audit to separate coordinate correctness from
  fitter fitness improvement.

## Status

Bug/error status: resolved. The stale
`manual_fitspace_caked_deg == fit_observed_caked_deg` gate is removed. Cached
target remains authoritative. Detector-native reprojection mismatch is still
reported, but only as diagnostic evidence.

Feature status: implemented. Audit JSON now proves:

- manual visual/saved caked point resolves to cached target,
- optimizer measured target stays `cached_fit_space_anchor`,
- optimizer source equals dynamic `sim_visual_caked_deg`,
- objective residual is angular source-minus-target in degrees,
- detector-native reprojection is diagnostic,
- fit-improvement audit can show a perturbed start improves raw angular RMS.

Generated artifacts stay untracked under
`artifacts/geometry_fit_ladder/new4_coordinate_audit/`.

## Validation

Focused tests and protected gates passed:

- manual point audit tests: 2 passed
- geometry fitting objective slice: 16 passed
- GUI coordinate audit slice: 15 passed
- manual caked helper slice: 4 passed
- protected rung slice: 64 passed
- compileall: passed for `ra_sim`, `scripts`, and `tests`

Audit runs passed:

- Phase 3 base target audit: 7/7 pairs, target/cache deltas 0
- Phase 4 target immutability: target unchanged under center/theta/gamma/Gamma
- Phase 5 source authority: optimizer source and dynamic source both
  `sim_visual_caked_deg`; fallback/source counters 0
- Phase 6 residual vector: residual contract error 0 for 2theta and wrapped phi
- Phase 7 improvement: controlled `center_x=1.0` start improved raw angular RMS
  from `72.3890090176 deg` to `72.2927313400 deg`
- Phase 9 protected C2 smoke: accepted true, metric `raw_angular_rms_deg`, unit
  `deg`, cached target count 7, sim visual source row count 7, fallback counters
  0

No Rung 1-7 fitter math was changed for this audit work.
