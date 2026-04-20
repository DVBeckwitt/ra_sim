# Mosaic fitter recovery

Status: planned
Type: feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-20

## Summary

The mosaic fitter is the next project after the geometric fitter is green. It
must stay geometry-locked and consume the cached manual-geometry dataset from
the last successful geometry fit. It should not compensate for unresolved beam
center, detector distance, shared-theta, or picked-peak identity errors.

Project order:

1. finish the geometric fitter,
2. get the mosaic fitter working,
3. get the structure-factor fitter working,
4. get the stacking-fault fitter working.

## Current state

The repository already documents a geometry-fit-cached mosaic-shape path. The
planned recovery should treat that existing path as the starting point rather
than designing a parallel fitter.

Key dependency: the mosaic fitter is blocked until the `new4` geometric-fitter
saved-state baseline passes. If geometry changes, manual pairs change, selected
backgrounds change, or shared-theta metadata changes, rerun the geometry fit and
refresh the cached geometry dataset before mosaic fitting.

## Next actions

After the geometric fitter passes its `new4_fresh_all.json` gate:

1. Identify the exact GUI and headless mosaic entry points that should be live.
2. Confirm the mosaic fitter reads the latest successful geometry-fit dataset
   bundle and does not reopen geometry parameters.
3. Confirm selected specular and off-specular peak lists are preserved from the
   geometry-fit cache.
4. Add or refresh a canonical saved-state mosaic baseline that starts from the
   passing geometry output.
5. Gate on image/profile agreement without allowing geometry drift.
6. Add focused regressions for stale or missing geometry-fit cache rejection.

## Acceptance criteria

- The mosaic fitter refuses to run when the geometry-fit cache is missing or
  stale.
- The mosaic objective keeps geometry fixed.
- The fit changes only mosaic/profile parameters that are explicitly in scope.
- The selected peak universe matches the latest accepted geometry-fit dataset.
- The fitter reports before and after image/profile metrics.
- The fit improves or preserves the measured mosaic objective without materially
  degrading point-match geometry quality.
- Headless and GUI entry points use the same cached dataset contract.

## Validation

Expected validation shape after implementation:

```powershell
pytest tests/test_geometry_fitting.py
pytest tests/test_gui_geometry_fit_workflow.py
pytest tests/test_cli_headless.py
python -m ra_sim fit-mosaic-shape artifacts/geometry_fit_gui_states/new4_fresh_all.json
python -m ra_sim.dev check
```

Adjust the exact command once the canonical mosaic baseline fixture exists.

## Links

- [Geometric fitter recovery](../in-progress/geometric-fitter-recovery.md)
- [GUI workflow](../../gui-workflow.md)
- [Mosaic-shape fitting reference](../../simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
