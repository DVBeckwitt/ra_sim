# Mosaic fitter recovery

Status: in-progress
Type: feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-22

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
recovery should treat that existing path as the starting point rather than
designing a parallel fitter.

A first scaffold now exists in
`ra_sim/fitting/optimization_mosaic_profiles.py`. It adds pure helper APIs for
selected-Qr/background point pairing, local ROI integration into intensity versus
`phi`, additive Lorentzian plus Gaussian peak fitting, peak-centered profile
comparison, and a callback-driven least-squares wrapper that varies only the
chosen mosaic parameters. The scaffold is not yet wired into the GUI or headless
entry point.

Key dependency: the mosaic fitter remains geometry-locked. If geometry changes,
manual pairs change, selected backgrounds change, or shared-theta metadata
changes, rerun the geometry fit and refresh the cached geometry dataset before
mosaic fitting.

Agreement with the existing documented path:

- It consumes selected manual/Qr peaks and measured background anchors.
- It compares profile shapes after local ROI extraction.
- It keeps geometry fixed and exposes only explicitly chosen fit parameters.
- It supports the existing selected-Qr grouping metadata rather than replacing
  it.

Differences from the current documented/implemented path:

- The scaffold emphasizes every user-selected Qr/background pair rather than
  reducing off-specular picks to the top three HKL/Qr groups.
- It currently focuses on `phi` profiles for the selected pairs. The existing
  production path still treats specular peaks as `2theta` profiles.
- Each extracted profile is first fit with an additive Lorentzian plus Gaussian
  with independent amplitudes and widths. The current forward simulation and
  documented width parameters still use a pseudo-Voigt-style mixture controlled
  by `eta`.
- The parameter fit is callback-driven scaffolding. It does not yet run the full
  simulator, refresh GUI state, or enforce cache-staleness guards by itself.

## Next actions

1. Wire the selected-Qr/background phi-profile scaffold to the geometry-fit
   cached dataset bundle.
2. Decide whether specular selected peaks should enter this scaffold as `phi`
   profiles, retain the existing `2theta` treatment, or use both terms.
3. Map chosen physical parameters onto the simulation callback while keeping
   detector geometry fixed.
4. Decide whether the production mosaic parameterization should remain
   pseudo-Voigt-style or move toward the additive Lorentzian plus Gaussian
   profile fit requested for the extracted integrations.
5. Add GUI and headless entry-point wiring only after stale-cache rejection and
   selected-pair provenance checks are explicit.
6. Add or refresh a canonical saved-state mosaic baseline that starts from the
   passing geometry output.

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

Scaffold validation added:

```powershell
pytest tests/test_mosaic_shape_optimization.py::test_pair_selected_qr_and_background_points_matches_duplicate_q_groups_by_peak_index
pytest tests/test_mosaic_shape_optimization.py::test_fit_lorentzian_plus_gaussian_profile_recovers_center_and_widths
pytest tests/test_mosaic_shape_optimization.py::test_integrate_selected_qr_phi_profiles_uses_background_anchor_roi_and_fit
pytest tests/test_mosaic_shape_optimization.py::test_compare_centered_phi_profiles_aligns_fitted_peak_centers
pytest tests/test_mosaic_shape_optimization.py::test_fit_mosaic_parameters_from_centered_phi_profiles_refines_active_width
```

Expected validation shape after entry-point wiring:

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
