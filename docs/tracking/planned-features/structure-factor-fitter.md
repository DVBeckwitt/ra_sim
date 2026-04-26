# Structure-factor fitter recovery

Status: planned
Type: feature
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-24

## Summary

The structure-factor fitter comes after geometry and mosaic fitting are green.
Its job is to infer one shared set of structure-factor parameters from relative
peak intensities across multiple detector images. It must not use
structure-factor parameters to hide unresolved geometry, mosaic-profile,
background, exposure, or display-normalization errors.

Project order:

1. geometric fitter,
2. mosaic fitter,
3. structure-factor fitter,
4. stacking-fault fitter.

## Problem

Infer one global parameter vector `theta` for structure-factor terms that
explains the relative intensities of selected peaks across many images. The fit
should use detector-space observations, not raw `|F|^2`, not GUI display
intensity, and not a full-image pixel objective.

## Intended behavior

Each image contributes one observation per accepted peak ROI. The observation is
the background-subtracted integrated detector count in that ROI. The prediction
is the sum of ray-carried simulated intensity that lands in the same ROI after
geometry, mosaic, optics, and detector projection have already been applied.

The optimizer varies one shared structure-factor parameter vector across all
images and one small set of nuisance terms per image. The first nuisance should
be one log scale `alpha_m` per image. Geometry and mosaic/profile parameters are
fixed from the accepted earlier stages.

## Current behavior and gap

The repository contains structure-factor code, parity tests, detector transport,
ROI/mask machinery, and local ordered-structure detector fitting in
`ra_sim/gui/ordered_structure_fit.py`. What is missing is a single global
fitting contract that defines the observation, detector-space prediction,
branch/overlap accounting, image normalization, weighting, objective, staging,
and acceptance tests.

## Root cause hypothesis

If those design choices stay implicit, the fitter can drift for the wrong
reasons. Common failure modes are per-image exposure differences, specular peak
dominance, branch double-counting, overlap double-counting, display
normalization leaks, and fitting raw structure-factor intensity before detector
transport.

## Fix strategy

Use detector-space peak ROI areas as the data. Use ray-cast detector-carried
intensity summed into the same ROI as the prediction. Add one analytic per-image
log scale. Keep structure-factor parameters shared across all images. Fit
non-`00l` peaks first, then add `00l` peaks with lower weight or a dedicated
nuisance model.

## Observation model

For ROI observation `i` in image `m(i)`, let `R_i` be the peak ROI and `B_i` be
a local background ring or sideband. With detector counts `D_m(p)` at pixel `p`,
use

\[
\hat b_i = \frac{1}{|B_i|}\sum_{q \in B_i}D_{m(i)}(q),
\qquad
 y_i = \sum_{p \in R_i}D_{m(i)}(p)-|R_i|\hat b_i .
\]

A practical initial variance estimate is

\[
\sigma_{y,i}^{2}
\approx
\sum_{p \in R_i}\max(D_{m(i)}(p),0)
+ |R_i|^2\operatorname{Var}(\hat b_i).
\]

Use direct integration for isolated peaks. For overlapping peaks, either fit a
local positive-amplitude multi-peak model with a shared background or remove the
ROI from the main fit. Never count the same detector pixel as two separate
observations.

## Prediction model

The prediction is detector-space intensity in the same ROI:

\[
\mu_i(\theta)=\sum_{r \in C_i}q_{ir}W_r|F_{hkl(r)}(\theta)|^2 .
\]

`C_i` is the set of simulated ray or subpixel contributors that deposit into
`R_i`. `q_ir` is the fractional bilinear or subpixel deposition inside `R_i`.
`W_r` contains fixed beam, mosaic, footprint, optics, attenuation, detector, and
sampling factors from the accepted geometry/mosaic state.

The varying structure-factor part is

\[
F_{hkl}(\theta)
=
\sum_a o_a(\theta) f_a(Q)
\exp\!\left[2\pi i\,\mathbf G_{hkl}\cdot\mathbf r_a(\theta)\right]
\exp\!\left[-2\pi^2\mathbf G_{hkl}\cdot\mathbf U_a(\theta)\cdot\mathbf G_{hkl}\right].
\]

Branch and multiplicity rules:

- The unit of comparison is one ROI in one image.
- If branches land in different ROIs, keep the observations separate.
- If branches merge into one ROI, sum their predicted contributions before the
  comparison.
- If ray casting already expands symmetry or multiplicity into separate hits,
  do not apply another multiplicity factor.
- Do not compare one branch prediction to an ROI that contains multiple branches
  unless the prediction has summed the same contributors.

## Objective

Use robust log-intensity residuals:

\[
e_i(\theta,\alpha)
=\log(y_i^+ + \epsilon)-\alpha_{m(i)}-\log(\mu_i(\theta)+\epsilon).
\]

Then solve

\[
\min_{\theta,\alpha}\sum_i \rho\!\left(\sqrt{w_i}\,e_i\right),
\]

where `rho` is Huber or Student-t and `epsilon` is a small positive count floor.
The preferred gate is to exclude peaks with unusable or nonpositive
background-subtracted area rather than let negative areas drive the log fit.

For fixed `theta` and weighted least squares, the per-image scale is analytic:

\[
\alpha_m(\theta)
=
\frac{\sum_{i:m(i)=m}w_i[\log(y_i^+ + \epsilon)-\log(\mu_i(\theta)+\epsilon)]}
{\sum_{i:m(i)=m}w_i}.
\]

With Huber or Student-t loss, update `alpha_m` using the same formula inside an
iteratively reweighted least-squares loop with robust effective weights.

Use composite weights

\[
w_i=w_{\mathrm{var},i}w_{\mathrm{family},i}w_{\mathrm{image},i}w_{\mathrm{quality},i}.
\]

`w_var` is inverse variance from ROI statistics. `w_family` balances reflection
families or `q` bins. `w_image` equalizes total weight per image. `w_quality`
downweights saturated, overlapped, clipped, or low-support peaks. Cap large
weights so one giant peak cannot dominate the solution.

## Ordered implementation steps

1. Build a peak catalog over all fitted images. Each row should contain image
   id, ROI id, reflection key, branch id if resolved, detector ROI bounds,
   observed integrated area, uncertainty, quality flags, and selected-pair or
   geometry-cache provenance.
2. Define one canonical prediction path. For each ROI, ray cast with fixed
   geometry/mosaic parameters, keep only rays that hit the detector, and sum the
   carried intensity entering that ROI.
3. Add per-image log scale `alpha_m` and solve it analytically for each trial
   `theta`.
4. Start with identifiable global parameters: occupancies first, isotropic
   Debye-Waller terms second, then anisotropic or coordinate-like terms only
   after the earlier fit is stable.
5. Fit non-`00l` peaks first.
6. Add `00l` peaks with lower weight or a dedicated specular nuisance only after
   the off-spec fit is stable.
7. Turn on overlap decomposition only where needed. Do not force local
   multi-peak fits on every ROI.
8. Add held-out image and held-out reflection-family validation before GUI or
   headless production wiring is claimed.
9. Serialize the fitted structure-factor state and per-image scales so GUI and
   headless reruns use the same state.

## Acceptance criteria and tests

- Geometry is fixed from the accepted geometric fitter output.
- Mosaic/profile parameters are fixed from the accepted mosaic fitter output.
- Multiplying one image by a constant leaves `theta` unchanged and moves only
  that image's `alpha_m`.
- The model reproduces peak-strength rank ordering on held-out images.
- A held-out image can be predicted from a fit on the remaining images.
- A held-out reflection family can be predicted from a fit that excluded it.
- Overlap tests recover integrated areas within tolerance for synthetic or
  curated merged peaks.
- Branch accounting tests prove summed ROI prediction equals the sum of included
  ray-carried contributors with no hidden multiplicity factor.
- Fits with and without `00l` peaks do not materially move core non-specular
  parameters.
- Changing GUI display normalization does not change fitted parameters.
- The selected reflection set is explicit and reproducible.
- The fitted structure-factor state is serializable and reloadable.

## Validation

Expected validation shape after implementation:

```powershell
pytest tests/test_bi2se3_structure_factor.py
pytest tests/test_compare_intensity.py
pytest tests/test_diffraction_constraints.py
pytest tests/test_raw_structure_factor_api.py
pytest tests/test_structure_factor_sites.py
pytest tests/test_structure_factor_switch_sweep.py
python -m ra_sim.dev check
```

Add fitter-specific tests for scale invariance, branch accounting, overlap
accounting, held-out image prediction, held-out family prediction, and display
normalization independence when the entry point is selected.

## Risks, invariants, and things not to break

- Do not fit to GUI-normalized display images.
- Do not compare detector ROI area to raw `|F|^2` before detector transport.
- Do not apply multiplicity twice.
- Do not let one intense peak dominate the objective.
- Do not let overlapped pixels contribute to two independent observations.
- Do not use `00l` peaks as ordinary peaks until the specular weighting or
  nuisance policy is validated.
- Do not release many weakly identifiable structural parameters in the first
  optimizer pass.

## Strongest counterargument and rebuttal

Counterargument: fit the full detector image directly and avoid ROI selection.

Rebuttal: full-image fitting overweights empty pixels, mixes background and
geometry errors into the structure-factor fit, and makes overlap debugging much
harder. For structure-factor parameter inference, peak-ROI fitting is the
cleaner abstraction because it compares the measured and predicted quantities at
the same detector-space support.

## Links

- [Geometric fitter recovery](../in-progress/geometric-fitter-recovery.md)
- [Mosaic fitter recovery](../in-progress/mosaic-fitter.md)
- [Canonical simulation and fitting reference](../../simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
