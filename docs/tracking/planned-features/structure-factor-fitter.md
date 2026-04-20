# Structure-factor fitter recovery

Status: planned
Type: feature
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-20

## Summary

The structure-factor fitter is planned after geometry and mosaic fitting are
working. Its job is to refine structural intensity terms only after peak
positions and peak shapes are stable enough that intensity residuals are
meaningful.

Project order:

1. geometric fitter,
2. mosaic fitter,
3. structure-factor fitter,
4. stacking-fault fitter.

## Current state

The repository contains structure-factor code and comparison tests, but this
planned project still needs a current fitter contract. Do not use structure
factor parameters to mask unresolved geometry or mosaic-profile errors.

## Next actions

After mosaic fitting is green:

1. Define the exact parameter set that the structure-factor fitter may vary.
2. Define the measured intensity inputs and normalization strategy.
3. Lock geometry and mosaic parameters from the previous accepted stages.
4. Decide which reflections enter the canonical acceptance set.
5. Add synthetic and saved-state regressions for intensity-only fitting.
6. Document how GUI and headless runs share the same fitted structure-factor
   state.

## Acceptance criteria

- Geometry is fixed from the accepted geometric fitter output.
- Mosaic/profile parameters are fixed from the accepted mosaic fitter output.
- The fitter reports before and after intensity residuals.
- The selected reflection set is explicit and reproducible.
- The result improves or preserves the intensity objective without moving peak
  positions or mosaic widths.
- The fitted structure-factor state is serializable and reloadable.

## Validation

Expected validation shape after implementation:

```powershell
pytest tests/test_bi2se3_structure_factor.py
pytest tests/test_compare_intensity.py
pytest tests/test_diffraction_constraints.py
python -m ra_sim.dev check
```

Add a fitter-specific headless command once the structure-factor fitter entry
point is selected.

## Links

- [Geometric fitter recovery](../in-progress/geometric-fitter-recovery.md)
- [Mosaic fitter recovery](mosaic-fitter.md)
- [Canonical simulation and fitting reference](../../simulation_and_fitting.md)
