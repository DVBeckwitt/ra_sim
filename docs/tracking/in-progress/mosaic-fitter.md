# Mosaic fitter recovery

Status: in-progress
Type: feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-24

## Summary

The mosaic fitter is the next project after the geometric fitter is green. It
must stay geometry-locked and consume the accepted geometry-fit cache from the
last successful manual geometry fit. It should not compensate for unresolved
beam center, detector distance, shared-theta interpretation, detector geometry,
or picked-peak identity errors.

Project order:

1. finish the geometric fitter,
2. get the mosaic fitter working,
3. get the structure-factor fitter working,
4. get the stacking-fault fitter working.

## Problem

The repository already has a geometry-fit-cached mosaic-shape path and a newer
selected-pair profile scaffold, but production wiring is still incomplete. The
missing piece is an explicit contract that says which cache is consumed, which
selected pairs define the measured universe, which profile math is used, and
which parameters may move.

## Intended behavior

After the scaffold is fully wired, mosaic fitting should do this:

1. Load the last accepted geometry-fit cache.
2. Take the selected Qr/background pairs from that cache.
3. Extract one measured local intensity-versus-phi profile per pair.
4. Fit each measured profile with an additive Lorentzian plus Gaussian model to
   get a robust peak center.
5. Simulate the same pair universe for a trial mosaic parameter vector while
   keeping geometry fixed.
6. Extract and fit the simulated profiles with the same rules.
7. Center measured and simulated profiles at their fitted centers.
8. Compare centered profile shapes on a common phi grid.
9. Optimize only the explicit active mosaic/profile parameters.

The fitter must preserve pair identity and ordering between measured and
simulated profiles. It must refuse to run when the cache or selected-pair
provenance is stale.

## Current behavior and gap

A first scaffold exists in
`ra_sim/fitting/optimization_mosaic_profiles.py`. It provides pure helper APIs
for selected-Qr/background pairing, local ROI integration into `I(phi)`,
additive Lorentzian plus Gaussian fitting, centered-profile comparison, and a
callback-driven least-squares wrapper that varies only active parameters. These
helpers are re-exported through `ra_sim/fitting/optimization.py`.

The scaffold is not yet wired into the production GUI or headless mosaic entry
point. The canonical geometry-cache to mosaic-cache contract is only partially
documented. The production decision for specular handling is still open.

The existing production path still treats specular `(00l)` peaks as `2theta`
profiles and off-specular peaks as `phi` profiles. The new selected-pair
scaffold currently describes a direct `phi`-profile route for selected pairs.

## Root cause hypothesis

If geometry cache identity, selected-pair identity, and profile centering are
left implicit, mosaic fitting can move the wrong quantities. It can hide stale
geometry, compare different measured and simulated peak universes, or optimize
profile position instead of profile shape. The fitter then appears to improve
while silently absorbing geometry or selection errors.

## Fix strategy

Wire the selected-pair scaffold downstream of the accepted geometry-fit cache.
Keep geometry immutable. Make stale-cache rejection explicit. Add a dry-run rung
before production optimization. Keep the objective centered on profile shapes,
not on fitted profile parameters.

The production fork for specular peaks must be resolved explicitly:

1. keep the existing hybrid behavior: specular goes to `2theta`, off-specular
   goes to `phi`,
2. move all selected pairs to the selected-pair `phi` route,
3. use both terms with explicit weights and reporting.

Option 1 is the narrowest production patch because it preserves existing
specular behavior. Option 2 is cleaner conceptually, but it changes the meaning
of the residual vector. Option 3 is most complete, but it needs the strongest
weighting and diagnostics.

## Math contract

For selected pair `p`, the measured ROI is centered on the measured/background
anchor, not on the simulated point. Let `R_p` be the square local detector ROI.
Pixels near the local center line in the orthogonal direction form signal sets
`S_p(g)` for phi bin `g`; sideband pixels form `B_p(g)`.

The background-subtracted measured profile bin is

\[
m_p(g)
= \sum_{u \in S_p(g)} D(u)
- \frac{|S_p(g)|}{|B_p(g)|}\sum_{v \in B_p(g)}D(v).
\]

Bins with insufficient signal or sideband support are rejected. The simulated
profile `s_p(g; x)` is extracted by the same rule from the simulated detector
image produced by active mosaic parameter vector `x`.

Each measured and simulated profile is fit with

\[
I(\phi)
= A_G\exp\!\left[-\frac{1}{2}\left(\frac{\phi-\phi_0}{\sigma}\right)^2\right]
+ \frac{A_L}{1 + \left((\phi-\phi_0)/\gamma\right)^2}
+ b .
\]

This is an additive Lorentzian plus Gaussian model. It is not a pseudo-Voigt
mixture. `A_G`, `A_L`, `sigma`, `gamma`, `phi_0`, and `b` are diagnostics and
centering aids. The objective does not directly minimize differences between
those fitted parameters.

For each profile, center the local coordinate by the fitted center:

\[
\phi' = \phi - \hat\phi_0 .
\]

Interpolate measured and simulated centered profiles onto a common grid `g`.
Optionally normalize each profile by positive area when the production decision
is shape-only comparison. Then form

\[
r_p(g; x) = \sqrt{w_p}\,[s_p(g; x) - m_p(g)]
\]

and solve

\[
\min_x \frac{1}{2}\left\|\operatorname{concat}_p r_p(g; x)\right\|_2^2 .
\]

Only names in `active_parameter_names` may be changed by `x`. Detector geometry,
beam center, detector distance, shared-theta interpretation, and selected-pair
identity are invariants.

## Ordered implementation steps

1. Enforce the accepted geometry-cache contract before mosaic fitting starts.
   Reject missing, stale, or provenance-mismatched caches.
2. Build a no-optimizer dry-run command or rung that loads the cache, pairs
   selected Qr/background points, extracts measured profiles, fits centers, and
   reports rejected pairs.
3. Add selected-pair parity checks for pair keys, duplicate ordering, HKL labels,
   Q-group keys, dataset labels, and measured/simulated counts.
4. Decide and document the production specular policy.
5. Wire the simulation callback so trial parameter dictionaries change only
   active mosaic/profile parameters and keep geometry fixed.
6. Extract simulated profiles with the same pair list, ROI rules, phi binning,
   and profile-fit model used by the measured side.
7. Assemble the centered-profile residual vector and least-squares call.
8. Add before/after diagnostics per profile: center, width diagnostics, area,
   residual norm, weight, rejection reason, and pair identity.
9. Wire GUI and headless entry points to the same cache contract and objective.
10. Add or refresh a canonical saved-state mosaic baseline that starts from the
    passing geometry output.

## Acceptance criteria and tests

- The mosaic fitter refuses to run when the geometry-fit cache is missing or
  stale.
- Manual picks, selected backgrounds, Qr/Qz grouping, shared-theta metadata, and
  accepted geometry-fit dataset identity are part of the cache contract.
- The no-optimizer dry run proves pairing, ROI extraction, Lorentzian plus
  Gaussian fits, centered self-comparison, and rejected-pair reporting.
- The objective keeps geometry fixed.
- Only explicitly active mosaic/profile parameters change.
- Measured and simulated pair keys match exactly, including duplicate ordering.
- Before and after diagnostics are reported per profile.
- The fit improves or preserves the measured mosaic objective without materially
  degrading point-match geometry quality.
- Headless and GUI entry points use the same cached dataset contract.
- Phi-bin labels use true bin centers, not edge-like endpoints, before
  width-sensitive production fitting is claimed.

## Validation

Scaffold validation already added:

```powershell
pytest tests/test_mosaic_shape_optimization.py::test_pair_selected_qr_and_background_points_matches_duplicate_q_groups_by_peak_index
pytest tests/test_mosaic_shape_optimization.py::test_fit_lorentzian_plus_gaussian_profile_recovers_center_and_widths
pytest tests/test_mosaic_shape_optimization.py::test_integrate_selected_qr_phi_profiles_uses_background_anchor_roi_and_fit
pytest tests/test_mosaic_shape_optimization.py::test_compare_centered_phi_profiles_aligns_fitted_peak_centers
pytest tests/test_mosaic_shape_optimization.py::test_fit_mosaic_parameters_from_centered_phi_profiles_refines_active_width
```

Expected validation shape after entry-point wiring:

```powershell
pytest tests/test_mosaic_shape_optimization.py
pytest tests/test_geometry_fitting.py
pytest tests/test_gui_geometry_fit_workflow.py
pytest tests/test_cli_headless.py
python -m ra_sim fit-mosaic-shape artifacts/geometry_fit_gui_states/new4_fresh_all.json
python -m ra_sim.dev check
```

Adjust the exact command once the canonical mosaic baseline fixture exists.

## Risks, invariants, and things not to break

- Do not re-fit detector geometry in the mosaic step.
- Do not let profile center mismatch become a geometry correction.
- Do not compare different measured and simulated selected-pair universes.
- Do not silently drop selected pairs without a reported rejection reason.
- Do not mix pseudo-Voigt kernel parameters with Lorentzian plus Gaussian
  profile-fit diagnostics without naming the adapter boundary.
- Do not claim full fitter validation from scaffold tests alone.

## Links

- [Geometric fitter recovery](../in-progress/geometric-fitter-recovery.md)
- [Structure-factor fitter recovery](../planned-features/structure-factor-fitter.md)
- [GUI workflow](../../gui-workflow.md)
- [Mosaic-shape fitting reference](../../simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
