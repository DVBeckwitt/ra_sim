# hBN Fitter Documentation

Status: completed
Type: feature
Owner: Codex
Issue: none
Priority: p2
Last updated: 2026-04-30

## Summary

Added a focused maintainer/operator documentation page for
`ra_sim/hbn_fitter/fitter.py`. The page explains the hBN calibrant workflow,
click snapping, full-resolution coordinate handling, ellipse refinement,
projective tilt optimization, bundle save/load schema, overlays, UI actions,
fallbacks, maintenance invariants, and validation expectations.

## Current state

Complete as a documentation feature. The guide is linked from:

- `docs/index.md`
- `docs/gui-workflow.md`
- `docs/architecture.md`
- `docs/simulation_and_fitting.md`

Bug/error status:

- No runtime bug was fixed.
- No code path was changed.
- Main risk addressed is documentation drift around snapping semantics,
  coordinate scaling, tilt sign conventions, and legacy NPZ bundle
  compatibility.

## Next actions

- Update this guide when `ra_sim/hbn_fitter/fitter.py` changes click snapping,
  coordinate frames, tilt export, or NPZ schema.
- Add regression tests before changing the documented maintenance invariants.

## Validation

- Documentation link/path sanity checks passed.
- `git diff --check` passed for the touched existing docs.
- No Python tests were run because the patch is documentation-only.

## Links

- [hBN fitter guide](../../hbn-fitter.md)
- [canonical hBN calibrant reference](../../simulation_and_fitting.md#hbn-calibrant-fitting)
