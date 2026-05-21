# 0004: Manual geometry fit-space boundary

Status: Accepted, amended 2026-05-21

Date: 2026-05-20

## Context

Manual Qr/Qz geometry fitting receives point pairs from two different user
origins: detector-view clicks and explicit caked-view clicks. A previous repair
tried to promote detector-origin rows into the caked angular objective whenever
`gamma,Gamma` were active. That made the simple two-point detector fit depend
on exact-caked projection cache state and stale caked aliases, and could reject
the GUI fit with non-finite angular residuals even though the detector pixels
were matched to the intended simulated sources.

A later live Tk trace proved the opposite failure can also happen: detector
provenance can remain valid while the requested fit objective is explicitly
caked. In that case finite observed and predicted caked anchors were present,
but the optimizer still entered the fixed manual detector-pixel branch and
completed as `central_point_match` with pixel RMS and `matched=0`.

## Decision

The manual pick origin is provenance only. The requested fit objective is the
fit-space requirement.

For detector objectives, detector-origin manual Qr/Qz rows stay in
detector-pixel fit space, including two-tilt `gamma,Gamma` fits. For
`objective_space=caked_deg`, or the equivalent internal
`_manual_caked_fit_space_required` handoff, every manual pair must have finite
observed and predicted caked anchors. Ready caked manual pairs are evaluated in
degree space by the caked angular evaluator. Missing caked fit-space fails
closed before optimization with `manual_caked_fit_space_missing`.

The old detector-origin auto-caked promotion path remains removed, but detector
origin no longer authorizes a caked objective to use detector-pixel matching.
Saved caked display aliases on detector-origin rows remain display/replay cache
data; only explicit fitted observed/predicted caked anchors can satisfy the
caked objective. A caked objective must not finalize as `central_point_match`,
`metric_unit=px`, pixel `weighted_rms`, or post-optimization `matched=0`.

## Alternatives Considered

### Keep auto-caked detector-origin promotion

- Pros: preserved the previous attempt to make detector tilts use angular
  residuals.
- Cons: required extra projection state for a two-point detector problem,
  allowed stale aliases to override the clicked detector pixels, and produced
  confusing GUI rejections.
- Rejected: it made the fitter harder to reason about and did not match the
  operator's detector-view workflow.

### Let detector provenance choose detector-pixel residuals

- Pros: simple for detector-view manual picks and matched the first repair.
- Cons: failed the live caked Tk route where detector-origin picks had been
  projected into finite caked fit-space for a requested Qr/Qz caked objective.
- Rejected: provenance and objective requirement must stay separate.

### Add a parallel manual-caked solver while keeping the old fallback

- Pros: could make the caked route name explicit.
- Cons: would leave the defective caked-to-pixel escape path available.
- Rejected for this slice: the correct minimal change is to alter or bypass the
  existing fixed manual branch for caked objectives and fail closed if it leaks.

## Consequences

The immediate GUI behavior is simpler: detector-origin picks fit the nearest
saved detector background points to their selected simulated sources when the
objective is detector-space. Caked Qr/Qz manual fits use caked degree residuals
when finite anchors exist, regardless of detector-origin provenance, or fail
before optimization if required caked fit-space is missing. Existing saved
states remain compatible because no schema field changed. No CLI flag, config
key, artifact field, dependency, or migration is introduced. Rollback is a
normal git revert of the fitter-route commit.

## Related docs/tests

- [README manual geometry-fit behavior](../../README.md)
- [Debug and cache notes](../debug-and-cache.md)
- [New4 geometric fitter handoff](../tracking/in-progress/new4-geometric-fitter-recovery-handoff.md)
- `tests/test_geometry_fit_manual_fit_space_classification.py`
- `tests/test_geometry_fitter_cache_regression_gate_script.py`
- `tests/test_geometry_fitting.py`
- `tests/test_gui_geometry_fit_workflow.py`
- `tests/test_gui_runtime_import_safe.py`
