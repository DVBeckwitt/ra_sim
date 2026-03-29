# Simulation and Fitting Internals

This note documents the current implementation in this repository. It is meant
to answer "what does the code actually do?" rather than "what would an ideal
paper derivation look like?".

## Code Map

| Area | Main files | What they do |
| --- | --- | --- |
| Typed simulation entrypoints | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py), [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Build typed requests and call the legacy diffraction kernel. |
| Legacy simulation wrapper | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Generates beam/mosaic samples and launches one detector simulation. |
| Beam and mosaic sampling | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Generates low-discrepancy beam samples and optionally clusters them for speed. |
| Core physics kernel | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Builds sample/detector geometry, solves reciprocal-space intersections, applies optics, and deposits intensity on the detector. |
| Material and optical helpers | [`ra_sim/utils/calculations.py`](../ra_sim/utils/calculations.py) | Computes d-spacing, Bragg angle, index of refraction, and Fresnel transmission helpers. |
| Geometry, mosaic, and image fitting | [`ra_sim/fitting/optimization.py`](../ra_sim/fitting/optimization.py) | Main geometry-fit solver, separable mosaic-width fit, and multi-stage image refinement. |
| Background peak auto-matching | [`ra_sim/fitting/background_peak_matching.py`](../ra_sim/fitting/background_peak_matching.py) | Finds local background peaks and assigns simulated seeds to measured summits. |
| GUI geometry-fit preparation | [`ra_sim/gui/geometry_fit.py`](../ra_sim/gui/geometry_fit.py), [`ra_sim/gui/manual_geometry.py`](../ra_sim/gui/manual_geometry.py), [`ra_sim/gui/geometry_overlay.py`](../ra_sim/gui/geometry_overlay.py) | Builds fit datasets from manual pairs, chooses orientation transforms, and formats outputs. |
| Stacking-fault / rod intensity preprocessing | [`ra_sim/utils/stacking_fault.py`](../ra_sim/utils/stacking_fault.py) | Generates analytical Hendricks-Teller rod intensities and groups them into Qr rods. |
| hBN calibrant fitter | [`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py), [`ra_sim/hbn_fitter/optimizer.py`](../ra_sim/hbn_fitter/optimizer.py) | Fits ring ellipses, estimates detector tilt/circularity, and exports geometry bundles. |
| Legacy integrated-profile cost helper | [`ra_sim/fitting/objective.py`](../ra_sim/fitting/objective.py) | Simple MSE over reduced profiles; currently not used by the active fit workflows. |

## 1. Forward Physics Simulation

### 1.1 Entry points and overall flow

The two public entry layers are:

- [`simulate`](../ra_sim/simulation/engine.py) / [`simulate_qr_rods`](../ra_sim/simulation/engine.py): typed APIs that accept a `SimulationRequest`.
- [`simulate_diffraction`](../ra_sim/simulation/simulation.py): legacy convenience wrapper used by the GUI and tests.

The actual detector image is produced by
[`process_peaks_parallel_safe`](../ra_sim/simulation/diffraction.py), which:

1. optionally reuses cached/source-template work,
2. optionally clusters similar beam samples to reduce cost,
3. prefers the Numba kernel,
4. falls back to the Python version if JIT execution fails.

The physical kernel is still [`process_peaks_parallel`](../ra_sim/simulation/diffraction.py).

### 1.2 Lattice, reflection, and material model

The hexagonal lattice calculations in
[`ra_sim/utils/calculations.py`](../ra_sim/utils/calculations.py) use

```text
1 / d_hkl^2 = (4/3) * (h^2 + h*k + k^2) / a^2 + l^2 / c^2
2theta = 2 * asin(lambda / (2*d))
```

The reciprocal-space vector used by the diffraction kernel is

```text
G = [0,
     (4*pi/a) * sqrt((h^2 + h*k + k^2) / 3),
     2*pi*l/c]
```

The active material index of refraction in
[`IndexofRefraction`](../ra_sim/utils/calculations.py) is

```text
n = 1 - delta + i*beta
delta = (r_e * lambda^2 * rho_e) / (2*pi)
beta  = (mu * lambda) / (4*pi)
```

where `rho_e` comes from the configured stoichiometry and density, and `mu`
comes from the configured mass attenuation coefficients.

### 1.3 Beam and mosaic sampling

[`generate_random_profiles`](../ra_sim/simulation/mosaic_profiles.py) generates
five aligned sample arrays:

- `beam_x_array`
- `beam_y_array`
- `theta_array`
- `phi_array`
- `wavelength_array`

The implementation is not plain Monte Carlo. It uses:

- Latin-hypercube sampling in 5 dimensions,
- antithetic pairing (`u` and `1-u`) to reduce variance,
- inverse-normal mapping (`ndtri`) so each sampled dimension is Gaussian.

That means the beam divergence, beam-position jitter, and wavelength spread are
sampled with lower variance than independent random draws.

For speed, [`cluster_beam_profiles`](../ra_sim/simulation/mosaic_profiles.py)
can compress the raw samples into weighted representative clusters. That changes
how the kernel is evaluated, but it is intended to preserve the same weighted
beam distribution.

### 1.4 Detector and sample geometry

[`process_peaks_parallel`](../ra_sim/simulation/diffraction.py) builds:

- detector rotations from `gamma` and `Gamma`,
- sample base rotations from `chi` and `psi`,
- a center-of-rotation tilt from `theta_initial`,
- a configurable CoR axis from `cor_angle` and `psi_z`.

The CoR tilt is applied in
[`_build_sample_rotation`](../ra_sim/simulation/diffraction.py) with Rodrigues'
formula. The axis starts along `+x`, is pitched toward `+z` by `cor_angle`, then
yawed about `+z` by `psi_z`, then normalized. This matches
[docs/cor_rotation_math.md](./cor_rotation_math.md).

The detector frame is defined by:

- `n_det_rot`: detector normal after detector rotations,
- `e1_det`: in-plane detector axis projected from `unit_x`,
- `e2_det`: orthogonal detector axis from `n_det_rot x e1_det`.

### 1.5 Per-beam precomputation

[`_precompute_sample_terms`](../ra_sim/simulation/diffraction.py) computes all
beam/sample terms that do not depend on the current reflection:

1. Intersect the incident beam ray with the tilted sample plane.
2. Compute the incident grazing angle `th_i_prime`.
3. Compute the incident in-plane azimuth `phi_i_prime`.
4. Compute transmitted in-sample wavevector components and transmission power.
5. Store entry-point coordinates, in-plane scattered-wave components, `Re(kz)`,
   `Im(kz)`, transmission power `Ti2`, and path length `L_in`.

There are two optics branches:

- `OPTICS_MODE_EXACT`
  - Uses exact complex-`kz` Fresnel transmission helpers
    (`_fresnel_t_exact`, `_fresnel_power_t_exact`).
  - Uses the decaying branch of the complex square root explicitly.
- `OPTICS_MODE_FAST`
  - Uses grazing-angle Snell transmission (`transmit_angle_grazing`),
    approximate `kz` decomposition (`ktz_components`), and a precomputed outgoing
    optics lookup table.

Both branches ultimately produce transmission power factors and absorption-like
path-length damping terms.

### 1.6 Solving the reciprocal-space intersection

For each reflection and beam sample,
[`solve_q`](../ra_sim/simulation/diffraction.py) constructs the Bragg-sphere
circle defined by

```text
|Q| = |G|
|Q + k_in_crystal| = k_scat
```

It does this by intersecting:

- a sphere of radius `|G|` centered at the origin,
- a sphere of radius `k_scat` centered at `-k_in_crystal`.

The resulting circle has center

```text
A = -k_in_crystal
c = (|G|^2 + |A|^2 - k_scat^2) / (2*|A|)
O = c * A_hat
```

and radius

```text
circle_r = sqrt(|G|^2 - c^2)
```

Two orthonormal directions `e1`, `e2` span the circle plane.

### 1.7 Mosaic broadening model

The mosaic broadening kernel is implemented in
[`compute_intensity_array`](../ra_sim/simulation/diffraction.py) and
[`_mosaic_density_scalar`](../ra_sim/simulation/diffraction.py). It depends only
on the angular offset between the trial `Q` direction and the nominal `G`
direction:

```text
theta0 = atan2(Gz, sqrt(Gx^2 + Gy^2))
theta  = atan2(Qz, sqrt(Qx^2 + Qy^2))
dtheta = wrapped(theta - theta0)

Gaussian   = [1 / (sigma * sqrt(2*pi))] * exp(-0.5 * (dtheta/sigma)^2)
Lorentzian = [1 / (pi * gamma)]         * 1 / (1 + (dtheta/gamma)^2)

omega(dtheta) = (1-eta)*Gaussian + eta*Lorentzian
I_mosaic      = omega / (2*pi*|G|^2)
```

So the mosaic model is a 1D pseudo-Voigt in angular deviation, normalized by a
geometry factor `2*pi*|G|^2`.

### 1.8 Uniform and adaptive `solve_q`

There are two deterministic `solve_q` modes:

- Uniform mode:
  - [`_solve_q_uniform`](../ra_sim/simulation/diffraction.py)
  - Samples the circle at fixed angular increments.
- Adaptive mode:
  - [`_solve_q_adaptive`](../ra_sim/simulation/diffraction.py)
  - Starts from coarse subintervals and recursively splits where Simpson-vs-
    trapezoid mass error is largest.

The code does not always sample the whole circle. It first builds local arc
windows with [`_build_local_arc_windows`](../ra_sim/simulation/diffraction.py),
using the pseudo-Voigt angular width around the nominal `G` angle. Broad peaks
fall back to full-circle sampling; narrow peaks only integrate the arcs that can
carry non-negligible mosaic weight.

The adaptive mode stores interval mass rather than pointwise weight, which is
why it remains deterministic and handles Lorentzian tails better than random
sampling.

### 1.9 Outgoing optics, detector intersection, and deposited intensity

[`_calculate_phi_from_precomputed`](../ra_sim/simulation/diffraction.py) is the
reflection-level core.

For each surviving `Q` solution it:

1. Forms the transmitted outgoing wavevector inside the sample.
2. Applies outgoing interface transmission (`Tf2`).
3. Applies propagation attenuation from incoming and outgoing `Im(kz)` terms.
4. Rotates the outgoing ray back to the laboratory frame.
5. Intersects that ray with the detector plane.
6. Deposits the resulting intensity bilinearly into the four surrounding pixels.

The deposited value is

```text
val =
    reflection_intensity
  * sample_weight
  * I_Q
  * prop_fac
  * exp(-Qz^2 * debye_x^2)
  * exp(-(Qx^2 + Qy^2) * debye_y^2)
```

where:

- `reflection_intensity` is the input structure-factor intensity for that
  reflection or rod point,
- `I_Q` is the integrated mosaic density returned by `solve_q`,
- `prop_fac` is the optics and absorption factor,
- `debye_x` damps out-of-plane scattering,
- `debye_y` damps in-plane scattering.

Auxiliary outputs:

- `hit_tables`: per-reflection hit rows `[I, col, row, phi, H, K, L]`,
- `miss_tables`: outgoing rays that missed the detector,
- `q_data`: optional saved `Q` solutions when `save_flag == 1`.

[`hit_tables_to_max_positions`](../ra_sim/simulation/diffraction.py) later
merges nearby hit rows into subpixel centroids and returns up to the two
strongest peak centers per reflection. That postprocessing matters because the
fitters do not work directly on raw hit rows.

### 1.10 Stacking-fault rods and Qr rods

[`ra_sim/utils/stacking_fault.py`](../ra_sim/utils/stacking_fault.py) implements
analytical Hendricks-Teller rod intensities.

For one `(h, k)` pair, [`analytical_ht_intensity_for_pair`](../ra_sim/utils/stacking_fault.py):

- uses the precomputed `F^2(L)` curve from the CIF,
- flips the user parameter `p` to `1-p` to stay consistent with the legacy
  diffuse code,
- builds the complex parameter `z = (1-p) + p*exp(i*delta)`,
- forms the infinite- or finite-layer HT correction `R`,
- returns `AREA * F2 * R`.

Then:

- [`ht_dict_to_arrays`](../ra_sim/utils/stacking_fault.py) turns `(h, k)` rods
  into `(h, k, L)` rows,
- [`ht_dict_to_qr_dict`](../ra_sim/utils/stacking_fault.py) groups rods by
  `m = h^2 + h*k + k^2`,
- [`qr_dict_to_arrays`](../ra_sim/utils/stacking_fault.py) returns grouped rod
  intensities and their degeneracy,
- [`process_qr_rods_parallel`](../ra_sim/simulation/diffraction.py) forwards
  those grouped intensities to the same detector kernel.

So rod simulations reuse the same detector physics; only the input intensity
model changes.

### 1.11 Reduced-coordinate integration

[`ra_sim/simulation/geometry.py`](../ra_sim/simulation/geometry.py) and
[`ra_sim/utils/tools.py`](../ra_sim/utils/tools.py) build pyFAI
`AzimuthalIntegrator` objects from detector metadata. These helpers support the
GUI's radial, azimuthal, and caked views, but they are separate from the core
detector-space forward model.

## 2. Geometry Fitting

### 2.1 What the GUI actually fits

The live GUI geometry fit is assembled in
[`ra_sim/gui/geometry_fit.py`](../ra_sim/gui/geometry_fit.py) and solved in
[`fit_geometry_parameters`](../ra_sim/fitting/optimization.py).

The GUI does not fit directly from arbitrary clicks. It first builds one or more
prepared datasets from saved manual Qr/Qz pair groups:

- [`build_geometry_manual_fit_dataset`](../ra_sim/gui/geometry_fit.py)
- [`prepare_geometry_fit_run`](../ra_sim/gui/geometry_fit.py)

For each selected background:

1. The saved manual pairs are loaded.
2. The background image is normalized into the fit orientation.
3. The current simulation is used to recover initial simulated peak positions.
4. An orientation transform is chosen by
   [`select_fit_orientation`](../ra_sim/gui/geometry_overlay.py).

The orientation search tries detector-indexing variants (`xy` vs `yx`), 90 deg
rotations, and axis flips, then keeps the best transform only if it improves the
simulated-vs-measured RMS enough and stays below the configured RMS ceiling.

Multi-background geometry fit is supported through `dataset_specs`. In that
mode, each dataset has its own `theta_initial` base value and the fit can use a
shared `theta_offset` parameter.

### 2.2 Legacy geometry-fit residual

If `fit_geometry_parameters` is called without `experimental_image` or
`dataset_specs`, it uses the older angle-based residual path:

- [`compute_peak_position_error_geometry_local`](../ra_sim/fitting/optimization.py)
- [`simulate_and_compare_hkl`](../ra_sim/fitting/optimization.py)

That path:

1. simulates the requested reflections,
2. extracts up to two peak centers per reflection,
3. converts detector pixels to `(2theta, phi)`,
4. matches simulated and measured peaks by HKL,
5. returns absolute radial and azimuthal angular differences.

This is not a pixel-intensity fit. It is a peak-position fit in angular space.

### 2.3 Main point-match geometry-fit residual

When `experimental_image` or `dataset_specs` is present,
[`fit_geometry_parameters`](../ra_sim/fitting/optimization.py) switches to
point-match mode and evaluates
[`_evaluate_geometry_fit_dataset_point_matches`](../ra_sim/fitting/optimization.py).

For each dataset, that function:

1. simulates only the reflection subset needed for the measured entries,
2. extracts simulated peak centers with `hit_tables_to_max_positions`,
3. resolves "fixed source" matches first when a measured point already knows
   which simulated hit-table row it came from,
4. falls back to HKL-group matching for the remaining points,
5. returns one fixed `(dx, dy)` residual slot per measured point.

The fixed residual shape is deliberate. The code comment states that SciPy's
finite-difference Jacobian estimation becomes unstable if the residual length
changes while points switch between matched and unmatched states.

Fallback matching is done per HKL group:

- simulated candidates are the one or two peak centers returned for that HKL,
- measured candidates are the saved points with the same HKL,
- assignment uses the global one-to-one matcher
  [`_build_global_point_matches`](../ra_sim/fitting/optimization.py), which is
  a Hungarian-style minimum-distance assignment under `pixel_tol`.

Unmatched measured points do not disappear. They receive a penalty residual
whose first component is `missing_pair_penalty` and whose second component is
zero.

### 2.4 Peak weighting and measurement uncertainty

Residual weighting is centralized in
[`_weight_measurement_residual`](../ra_sim/fitting/optimization.py).

There are two independent weighting ideas:

- Distance reweighting:

```text
w_dist = 1 / sqrt(1 + (pair_dist / solver_f_scale)^2)
```

  This downweights long simulated-to-measured pairings when
  `weighted_matching=True`.

- Measurement-uncertainty weighting:
  - off: residuals are just multiplied by `w_dist`,
  - isotropic: residual vector is divided by one scalar `sigma_px`,
  - anisotropic: residual is decomposed into detector-radial and tangential
    components about the beam center, then divided by
    `sigma_radial_px` and `sigma_tangential_px`.

In anisotropic mode the code effectively applies an inverse square-root
covariance in the radial/tangential basis and then rotates back to `(dx, dy)`.

Missing-pair penalties are also scaled by the same sigma-weight logic, so a
point with tighter declared uncertainty contributes a larger effective penalty.

### 2.5 Solver, priors, and search heuristics

The main nonlinear solver is
[`scipy.optimize.least_squares`](../ra_sim/fitting/optimization.py) with:

- explicit lower and upper bounds,
- per-parameter `x_scale`,
- robust losses `linear`, `soft_l1`, `huber`, `cauchy`, or `arctan`,
- optional parameter priors appended as residuals:

```text
r_prior_i = (x_i - prior_center_i) / prior_sigma_i
```

The geometry fitter wraps that base solve with several optional search stages.

Before the main solve it can run:

- pair reparameterization:
  - converts supported parameter pairs into mean/half-difference coordinates,
  - accepts the seed only if cost does not regress beyond the configured limit;
- staged release:
  - solves reduced subproblems with only selected parameter groups active,
  - then progressively releases more parameters;
- adaptive regularization:
  - computes an identifiability summary at the start point,
  - adds temporary Gaussian priors on weak or strongly correlated parameters,
  - accepts the regularized seed only if identifiability improves without
    unacceptable cost increase,
  - optionally runs one "release" solve after removing the temporary priors.

During and after the main solve it can also run:

- restart seeds from corners, center, axis probes, quasi-random global samples,
  and local jitter;
- stagnation probing along coordinate, pairwise, and optional random
  directions;
- ridge refinement;
- ROI/image refinement;
- auto-freeze of weak or highly correlated parameters;
- selective thaw of a limited number of previously frozen parameters.

The important point is that the geometry fit is not one plain
`least_squares(x0)` call. It is a layered solver pipeline with several
accept/reject gates.

### 2.6 Ridge refinement

[`_maybe_run_ridge_refinement`](../ra_sim/fitting/optimization.py) is only
available in point-match mode. It reuses
[`_stage_one_initialize`](../ra_sim/fitting/optimization.py), which works on
downsampled ridge maps rather than peak coordinates.

Stage 1 does:

1. downsample measured and simulated images,
2. convert each to a binary ridge map using Sobel-gradient magnitude plus a
   threshold,
3. compute Euclidean distance transforms of the ridge complements,
4. build a symmetric Chamfer-like residual by sampling:
   - simulated distance at measured ridge pixels,
   - measured distance at simulated ridge pixels.

The ridge refinement result is accepted only if:

- the ridge-map objective decreases,
- the point-match robust cost does not regress past its allowed limit,
- the displayed point RMS does not regress past its allowed limit,
- the matched-pair count does not decrease.

So ridge refinement is a secondary image-structure polish, not a replacement
for peak matching.

### 2.7 ROI/image refinement inside geometry fitting

[`_maybe_run_image_refinement`](../ra_sim/fitting/optimization.py) is also a
point-match-mode add-on and currently requires exactly one dataset with no
shared `theta_offset` parameter.

It:

1. checks that enough matched pairs exist,
2. builds tube ROIs from the current fitted geometry,
3. runs the stage-2 image optimizer
   [`_stage_two_refinement`](../ra_sim/fitting/optimization.py),
4. converts the refined parameters back into the geometry-fit parameter vector,
5. reevaluates the point-match diagnostics.

Acceptance criteria are analogous to ridge refinement:

- image objective must improve,
- point-match cost must stay within the allowed increase,
- point RMS must stay within the allowed increase,
- matched-pair count must not decrease.

In other words, the image-space refinement is subordinate to the point-match
objective; it is accepted only when it helps image agreement without breaking
the peak correspondence that the user actually curated.

### 2.8 Identifiability diagnostics

[`_build_identifiability_summary`](../ra_sim/fitting/optimization.py) computes
a local identifiability report at the final solution.

It:

1. finite-differences the residual with respect to each active parameter,
2. builds a Jacobian,
3. computes its SVD,
4. estimates rank and condition number,
5. approximates covariance with `pinv(J^T J) * residual_variance`,
6. converts that into a correlation matrix.

From those quantities it derives:

- `parameter_entries`: column norms and relative sensitivities,
- `group_sensitivity`: sensitivity grouped into center/tilt/distance/lattice/
  other buckets,
- `weak_parameters`: very low-sensitivity or invalid Jacobian columns,
- `high_correlation_pairs`: strongly correlated parameter pairs,
- `freeze_recommendations`: parameters recommended for fixing,
- `recommended_fixed_parameters` / `recommended_fixed_indices`,
- `warning_flags` such as `underconstrained`, `high_correlation`,
  `weak_sensitivity`.

In point-match mode it also computes `top_peak_sensitivity`, which records the
matched peaks whose `(dx, dy)` residuals change the most under each parameter's
finite-difference perturbation.

### 2.9 Geometry-fit outputs

After solving, the GUI calls
[`postprocess_geometry_fit_result`](../ra_sim/gui/geometry_fit.py).

That postprocessing step:

1. reruns `simulate_and_compare_hkl` at the fitted parameters,
2. aggregates matched peak centers across duplicate HKL entries,
3. computes per-HKL pixel offsets `(dx, dy, |Delta|)`,
4. builds overlay records for the fitted display,
5. computes overlay frame diagnostics,
6. saves the matched peak export table,
7. formats the progress/log text shown in the GUI.

The displayed RMS is resolved by
[`geometry_fit_result_rms`](../ra_sim/gui/geometry_fit.py):

- if `result.rms_px` exists, it uses that;
- otherwise it falls back to RMS of `result.fun`.

In point-match mode the fitter explicitly overwrites `result.rms_px` with
`unweighted_peak_rms_px` from the final diagnostics when available, so the GUI
shows an unweighted peak-position RMS rather than the raw weighted residual RMS.

## 3. Mosaic-Width Fitting

[`fit_mosaic_widths_separable`](../ra_sim/fitting/optimization.py) fits only
the pseudo-Voigt mosaic parameters:

- `sigma_mosaic_deg`
- `gamma_mosaic_deg`
- `eta`

Everything geometric is held fixed.

### 3.1 Reflection and candidate selection

The fitter first reduces the reflection list:

- only reflections with `(2*h + k) % 3 == 0` are allowed,
- `(0, 0, l)` reflections are handled specially,
- other reflections are dropped if `2theta > 65 deg`.

Candidate peak centers come from one of three sources:

- `geometry`
  - use measured peaks already established by geometry fitting;
- `auto`
  - use simulated hit-table peaks scored by simulated intensity;
- `hybrid`
  - union of the two, deduplicated.

For `geometry` and `hybrid`, candidate score is local measured-image SNR from
[`_local_peak_snr`](../ra_sim/fitting/optimization.py). For `auto`, score is
simulated intensity.

Optional stratification can enforce diversity across:

- distinct `L` values, or
- equal-width `2theta` bins.

### 3.2 ROI construction and separable amplitude/background elimination

The fitter selects non-overlapping square ROIs centered on candidate peaks.
Each ROI stores:

- its HKL and reflection index,
- the measured pixel vector,
- its simulated center and score,
- the ROI pixel coordinates.

For a trial `(sigma, gamma, eta)`, the code simulates only the selected
reflection subset and solves each ROI's affine intensity nuisance parameters in
closed form:

```text
model = amp * template + bkg

tt = t dot t
to = sum(t)
ty = t dot y
oy = sum(y)
oo = N
det = tt*oo - to^2

amp = (ty*oo - oy*to) / det
bkg = (tt*oy - to*ty) / det
```

If the template is empty, has negligible energy, or the affine system is
ill-conditioned, the code falls back to `observed - mean(observed)`.

Residual blocks can optionally be normalized by `sqrt(num_pixels)`, which is
the default `roi_normalization="sqrt_npix"`.

### 3.3 Optimization and acceptance

The nonlinear parameters are optimized with `least_squares` under bounds:

```text
sigma in [0.03, 3.0] deg
gamma in [0.03, 3.0] deg
eta   in [0.0, 1.0]
```

The fitter then optionally runs jittered restart solves around the current best
solution.

Diagnostics include:

- per-ROI RMS,
- outlier flags from robust sigma-clipping on ROI RMS values,
- `outlier_fraction`,
- worst ROIs,
- restart history,
- a `boundary_warning` if the optimum lands on a bound.

The result attribute `acceptance_passed` is `True` only if:

- cost reduction is at least `20%`,
- outlier fraction is at most `25%`,
- no parameter finished on a bound.

So the mosaic-width fitter is intentionally conservative about declaring a fit
"good".

## 4. Image-Statistics Refinement

[`iterative_refinement`](../ra_sim/fitting/optimization.py) is a separate
three-stage image-based optimizer. It is conceptually broader than the GUI
geometry fit, although some of its stages are reused inside that workflow.

### 4.1 Stage 1: coarse ridge alignment

Stage 1 is exactly the Chamfer-style ridge-map alignment described in
Section 2.6, just run as a standalone first pass.

### 4.2 Tube ROIs

[`build_tube_rois`](../ra_sim/fitting/optimization.py) constructs elongated
ROIs around each reflection trajectory.

Tube width is estimated from three broadening terms projected to detector
pixels:

```text
width_px ~= mosaic_fwhm * detector_distance / pixel_size
          + divergence * detector_distance / pixel_size
          + bandwidth  * detector_distance / pixel_size
```

The ROI centerline is the interpolated polyline through:

- simulated one- or two-peak centers for that reflection,
- any measured points for the same HKL.

The result is a detector-space tube mask, a centerline mask, and an off-tube
mask for background sampling.

### 4.3 Sensitivity-driven active pixels

[`compute_sensitivity_weights`](../ra_sim/fitting/optimization.py) estimates
which ROI pixels are informative for the currently active parameters.

For each parameter it:

1. perturbs that parameter by a small finite-difference step,
2. resimulates the image,
3. accumulates squared image gradients.

It then forms an importance map approximately proportional to

```text
importance ~ |dI/dx|^2 / I_base
```

clips the heaviest tails, upsamples back to detector resolution, and keeps:

- high-importance on-tube pixels,
- at least a fixed quota per reflection,
- a small random off-tube fraction for background control.

### 4.4 Local background model

[`fit_local_background`](../ra_sim/fitting/optimization.py) fits a quadratic
surface inside each ROI bounding box:

```text
b(x, y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
```

Those local backgrounds are blended across overlaps, smoothed, and then lightly
anchored outside the ROIs using a very blurred global background image.

### 4.5 Robust image residual

[`robust_residuals`](../ra_sim/fitting/optimization.py) combines:

- Poisson deviance,
- Anscombe-transformed residuals,
- Huber clipping,
- a mixture weight that suppresses the effect of clear outliers.

The returned residual is roughly

```text
[0.5*sqrt(|Poisson deviance|) + 0.5*Huber(Anscombe residual)]
* sqrt(weights) / sampling_probability
* sqrt(mixture_weight)
```

### 4.6 Stage 2 and Stage 3

[`_stage_two_refinement`](../ra_sim/fitting/optimization.py) runs on a reduced
pixel set with:

- downsampled sensitivity estimation,
- a limited reflection budget,
- tile sampling inside each ROI,
- more exploratory sampling.

[`_stage_three_refinement`](../ra_sim/fitting/optimization.py) reruns the same
idea at higher fidelity:

- native-resolution sensitivity maps,
- larger pixel quotas,
- lower exploratory fraction,
- tighter ROI refresh threshold,
- slightly less outlier mixture.

This makes the image refinement increasingly local and data-driven as it
converges.

## 5. Background Peak Auto-Matching

[`build_background_peak_context`](../ra_sim/fitting/background_peak_matching.py)
and
[`match_simulated_peaks_to_peak_context`](../ra_sim/fitting/background_peak_matching.py)
implement the automatic matching between simulated seed peaks and peaks found in
one background image.

### 5.1 Peak-context construction

The reusable peak context is built by:

1. replacing invalid pixels with the median valid intensity,
2. applying a fine Gaussian blur and a broader Gaussian blur,
3. defining `peakness = fine - broad`,
4. finding local maxima on `peakness`,
5. thresholding by robust prominence:

```text
candidate_floor = median(peakness) + min_prominence_sigma * robust_sigma
```

6. falling back to a high percentile threshold if that yields no candidates,
7. labeling candidate summits and storing their prominence/background values,
8. refining each summit center to subpixel precision with a quadratic fit and
   centroid fallback.

### 5.2 Seed-to-summit assignment

For each simulated seed, the matcher:

1. finds nearby summit candidates with a KD-tree,
2. optionally assigns each summit an owning seed using nearest-seed logic,
3. optionally rejects ambiguous summit ownership,
4. walks uphill on the `peakness` map from the seed to a nearby local summit,
5. scores viable seed-summit edges with distance, prominence, seed weight,
   uphill ascent, and a walk-to-same-summit bonus,
6. solves the one-to-one assignment with the Hungarian algorithm,
7. sigma-clips the final match distances.

The score is built by `_assignment_score`, which is an additive combination of:

- a distance term,
- a prominence term,
- a seed-intensity term,
- an ascent term,
- a bonus when the summit found by uphill walking matches the assigned summit.

So the background matcher is neither pure nearest-neighbor nor pure local-max
detection. It is a constrained, scored, one-to-one assignment.

## 6. hBN Calibrant Fitter

The hBN calibrant tools live in
[`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py) and
[`ra_sim/hbn_fitter/optimizer.py`](../ra_sim/hbn_fitter/optimizer.py).

### 6.1 Ellipse residual and robust seed fit

The point-to-ellipse residual in
[`ellipse_residuals_px`](../ra_sim/hbn_fitter/fitter.py) is:

```text
u = rotated x coordinate
v = rotated y coordinate
q = sqrt((u^2/a^2) + (v^2/b^2))
residual_px = |q - 1| * 0.5*(a + b)
```

So the code uses a radialized ellipse residual in pixel units.

[`robust_fit_ellipse`](../ra_sim/hbn_fitter/fitter.py):

- runs RANSAC with `EllipseModel`,
- then performs two sigma-clipping passes on the ellipse residuals,
- refits the ellipse after each successful clip.

[`weighted_refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) then refines that seed
with a Powell optimizer using:

- per-point weights `1/sigma_px^2`,
- a pseudo-Huber-like loss on ellipse residuals,
- weak regularization that keeps the refined ellipse near the robust seed.

### 6.2 Snapping clicks and iterative ellipse refinement

[`snap_points_to_ring`](../ra_sim/hbn_fitter/fitter.py) searches around each clicked
point:

- `u`: along the local radial direction,
- `v`: across the local tangent direction.

For each candidate, it samples an intensity profile, finds the best radial peak
with a pseudo-Voigt fit or a fast quadratic fallback, refines the 2D peak by a
subpixel centroid, and scores the candidate by:

- peak SNR,
- ellipse residual,
- tangent offset,
- click distance from the user's original point.

[`refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) then iterates:

1. sample many radial profiles around the current ellipse,
2. pick the best radial peak on each profile,
3. optionally add snapped user anchor points,
4. robust-fit and weighted-refine a new ellipse,
5. stop when the ellipse update is small.

### 6.3 Confidence scoring

[`compute_fit_confidence`](../ra_sim/hbn_fitter/fitter.py) combines:

- ellipse residual,
- ring SNR measured along the fitted ellipse,
- angular coverage of the clicked points,
- number of clicked points,
- downsampling penalty.

It returns per-ring and overall confidence scores, plus the intermediate
metrics.

### 6.4 Tilt optimization / circularization

There are two tilt models.

Legacy model:

- [`apply_tilt_xy`](../ra_sim/hbn_fitter/fitter.py)
- scales `x` and `y` by `1/cos(tilt)` about a shared center,
- optimized by [`optimize_tilts`](../ra_sim/hbn_fitter/fitter.py) using:
  - coarse grid search,
  - local Nelder-Mead,
  - optional Powell joint refinement of center and tilts.

Projective model:

- [`apply_tilt_projective`](../ra_sim/hbn_fitter/fitter.py)
- rotates a detector-plane basis in 3D and reprojects through a finite
  detector distance,
- optimized by [`optimize_tilts_projective`](../ra_sim/hbn_fitter/fitter.py) using:
  - coarse tilt grid search at nominal distance,
  - Powell refinement of `(tilt_x, tilt_y, log(distance))`,
  - optional Powell joint refinement of `(center_x, center_y, tilt_x, tilt_y,
    log(distance))`,
  - optional center prior and center drift limit.

The projective circularity objective minimizes the average squared robust
fractional ring-width:

```text
cost ~ mean_over_rings[(robust_sigma(r) / median(r))^2]
```

The legacy model uses the simpler `std(r) / mean(r)` metric.

## 7. Legacy and Auxiliary Notes

- [`compute_cost`](../ra_sim/fitting/objective.py) is a reduced-profile mean
  squared error helper. A repository search shows it is not part of the active
  geometry, mosaic, or image-statistics workflows.
- pyFAI integration helpers in
  [`ra_sim/simulation/geometry.py`](../ra_sim/simulation/geometry.py) and
  [`ra_sim/utils/tools.py`](../ra_sim/utils/tools.py) are for radial/caked GUI
  views, not for detector-space fitting.
- Rod and stacking-fault simulations reuse the same detector physics kernel as
  ordinary HKL peak simulations. Only the input intensity model differs.
- In the current geometry-fit implementation, point matching is the primary
  objective. Ridge refinement and ROI/image refinement are optional secondary
  stages with strict non-regression checks.

## 8. Short Summary

The repository's forward model is a deterministic beam-sampled diffraction
simulation with explicit sample and detector rotations, Fresnel-style
transmission factors, reciprocal-space circle integration, pseudo-Voigt mosaic
broadening, and pixel-space bilinear deposition.

The main fitters then sit on top of that forward model:

- geometry fitting matches simulated and measured peak positions in pixel space,
  with robust weighting, priors, restarts, and identifiability analysis;
- mosaic-width fitting holds geometry fixed and fits only mosaic widths by
  solving ROI amplitude/background analytically;
- image-statistics refinement uses ridge maps, tube ROIs, sensitivity-driven
  active pixels, quadratic local backgrounds, and robust count-like residuals;
- background auto-matching detects measured summits from a peakness image and
  assigns simulated seeds with a scored Hungarian matching step;
- the hBN calibrant fitter is an ellipse-and-circularization pipeline with
  robust ring fitting and optional projective detector-tilt optimization.
