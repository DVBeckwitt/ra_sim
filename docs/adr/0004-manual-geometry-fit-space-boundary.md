# 0004: Manual geometry fit-space boundary

Status: Accepted

Date: 2026-05-20

## Context

Manual Qr/Qz geometry fitting receives point pairs from two different user
origins: detector-view clicks and explicit caked-view clicks. A previous repair
tried to promote detector-origin rows into the caked angular objective whenever
`gamma,Gamma` were active. That made the simple two-point detector fit depend
on exact-caked projection cache state and stale caked aliases, and could reject
the GUI fit with non-finite angular residuals even though the detector pixels
were matched to the intended simulated sources.

## Decision

The manual fit space is determined by the saved pick origin.

Detector-origin manual Qr/Qz rows stay in detector-pixel fit space, including
two-tilt `gamma,Gamma` fits. Explicit caked-origin rows use the caked angular
objective and must have an exact per-background caked projector before fitting.
Mixed detector/caked fit spaces continue to fail closed during preflight.

The old detector-origin auto-caked promotion path is removed. Saved caked fields
on detector-origin rows remain display/replay cache data and are not optimizer
anchors.

## Alternatives Considered

### Keep auto-caked detector-origin promotion

- Pros: preserved the previous attempt to make detector tilts use angular
  residuals.
- Cons: required extra projection state for a two-point detector problem,
  allowed stale aliases to override the clicked detector pixels, and produced
  confusing GUI rejections.
- Rejected: it made the fitter harder to reason about and did not match the
  operator's detector-view workflow.

### Refine manual points between simulations

- Pros: could chase local maxima as parameters move.
- Cons: expands scope and adds image-dependent behavior before the basic fixed
  correspondence fit is reliable.
- Rejected for this slice: nearest saved background points are enough for the
  current bug and keep the objective deterministic.

## Consequences

The immediate GUI behavior is simpler: detector-origin picks fit the nearest
saved detector background points to their selected simulated sources. Caked
fits still use the caked projector, but only when the operator picked in caked
space. Existing saved states remain compatible because no schema field changed.
Rollback is a normal git revert of the simplification commit.

## Related docs/tests

- [README manual geometry-fit behavior](../../README.md)
- [Debug and cache notes](../debug-and-cache.md)
- [New4 geometric fitter handoff](../tracking/in-progress/new4-geometric-fitter-recovery-handoff.md)
- `tests/test_geometry_fit_manual_fit_space_classification.py`
- `tests/test_gui_geometry_fit_workflow.py`
- `tests/test_gui_runtime_import_safe.py`
