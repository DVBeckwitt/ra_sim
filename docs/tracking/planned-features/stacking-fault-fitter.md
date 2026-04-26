# Stacking-fault fitter recovery

Status: planned
Type: feature
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-24

## Summary

The stacking-fault fitter is planned after geometry, mosaic, and
structure-factor fitting are working. It should refine stacking-disorder terms
only after detector alignment, peak shapes, and baseline structural intensities
are stable.

Project order:

1. geometric fitter,
2. mosaic fitter,
3. structure-factor fitter,
4. stacking-fault fitter.

## Current state

Stacking-related controls and simulation concepts exist in the project, but the
planned fitter still needs a current acceptance contract. This project must not
start by reopening geometry, mosaic, or structure-factor terms.

## Next actions

After the structure-factor fitter is green:

1. Define the stacking-fault parameters that are allowed to vary.
2. Define the reflections or detector regions most sensitive to stacking
   disorder.
3. Lock accepted geometry, mosaic, and structure-factor outputs from the
   preceding stages.
4. Build a canonical synthetic case where the true stacking fault is known.
5. Build a saved-state acceptance case from a real GUI state or exported fit
   bundle.
6. Add GUI/headless serialization and reload tests for fitted stacking state.

## Acceptance criteria

- Geometry, mosaic, and structure-factor parameters are fixed from accepted
  upstream stages.
- The selected stacking-sensitive reflections or regions are explicit.
- The fitter reports before and after stacking objective values.
- The result improves or preserves the stacking objective without degrading the
  upstream geometry, mosaic, or intensity gates.
- The fitted stacking-fault state is serializable and reproducible.

## Validation

Expected validation shape after implementation:

```powershell
pytest tests/test_diffraction_constraints.py
pytest tests/test_stacking_fault_diffuse_f2_q.py
pytest tests/test_stacking_fault_cache_invalidation.py
pytest tests/test_stacking_fault_redundancy.py
pytest tests/test_ht_analytical.py
pytest tests/test_qr_grouping.py
python -m ra_sim.dev check
```

Add a fitter-specific headless command once the stacking-fault fitter entry
point is selected.

## Links

- [Geometric fitter recovery](../in-progress/geometric-fitter-recovery.md)
- [Mosaic fitter recovery](../in-progress/mosaic-fitter.md)
- [Structure-factor fitter recovery](structure-factor-fitter.md)
- [Canonical simulation and fitting reference](../../simulation_and_fitting.md)
