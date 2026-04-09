# RA-SIM reference

This is the canonical RA-SIM reference for GUI workflow, forward simulation, fitting, debug/log controls, cache retention, and the detailed center of rotation derivation. It stays implementation-faithful: names, defaults, equations, and file paths are taken from the current code. If an older note, README pointer, or inline comment disagrees with this file, prefer this file and then audit the linked functions directly.

The scope is the current live pipeline:

- GUI work areas and the order they are usually used
- forward HKL and rod simulation through the typed API and legacy wrapper
- exact detector and sample geometry used by the diffraction kernel
- reciprocal-space integration and mosaic broadening
- manual point-pick geometry fitting
- automatic background peak matching
- geometry-fit-cached mosaic-shape fitting, legacy mosaic-width fitting, and image-space refinement
- hBN calibrant ellipse fitting and projective tilt correction
- debug/logging controls and optional cache-retention policy
- the full center-of-rotation axis derivation used by the diffraction kernel

No new API is defined here. All names, defaults, equations, and file paths below are taken from the current code. The older standalone notes for GUI views, logging/debug controls, and CoR math are now folded into this file.

## Index

- [GUI workflow and views](#gui-workflow-and-views)
- [Read This First: The Physical Story of the Pipeline](#read-this-first-the-physical-story-of-the-pipeline)
- [Code map](#code-map)
- [Notation and units](#notation-and-units)
- [Simulation parameter inventory](#simulation-parameter-inventory)
- [Forward simulation: strict algorithm](#forward-simulation-strict-algorithm)
- [Pedagogical view of the optics modes](#pedagogical-view-of-the-optics-modes)
- [Exact per-sample ray tracing and entry optics](#exact-per-sample-ray-tracing-and-entry-optics)
- [Reciprocal-space circle solve and mosaic weighting](#reciprocal-space-circle-solve-and-mosaic-weighting)
- [Outgoing ray tracing, detector projection, and deposition](#outgoing-ray-tracing-detector-projection-and-deposition)
- [Implementation details that materially affect behavior](#implementation-details-that-materially-affect-behavior)
- [Rods and stacking disorder](#rods-and-stacking-disorder)
- [Geometry fitting from picked spots](#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](#automatic-background-peak-matching)
- [Mosaic-shape fitting, legacy mosaic-width fitting, and image-space refinement](#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- [hBN calibrant fitting](#hbn-calibrant-fitting)
- [Logging, debug, and cache controls](#logging-debug-and-cache-controls)
- [Appendix A: Center of rotation axis math](#appendix-a-center-of-rotation-axis-math)
- [Appendix B: equation-to-code map](#appendix-b-equation-to-code-map)

## GUI workflow and views

These are the main RA-SIM work areas in the order they are usually used.

### Simulation view

The simulation view is the main detector-space workspace. Use it to answer the
first-order questions:

- Are the main arcs, caps, and ring fragments in the right place?
- Is the detector geometry roughly correct?
- Are the broadening and intensity trends physically believable?

This is the best place to establish global agreement before checking reduced
coordinates.

### Integration views

The integration views reduce the 2D detector image into easier diagnostics:

- radial intensity versus `2theta`
- azimuthal intensity versus detector angle
- caked maps for localized mismatch

Use these views after the detector-space pattern is roughly aligned. They are
the quickest way to see whether the model has the correct radial positions,
azimuthal widths, and intensity balance.

### Calibrant view

The calibrant view is the hBN ellipse-fitting workflow used to estimate beam
center, detector tilt, and related geometry terms from ring data.

Typical use:

1. Load a calibrant frame and any associated dark/background image.
2. Mark or edit ring points.
3. Fit ellipses and refine the geometry.
4. Save the bundle and use it as the starting point for the main simulation.

Use this view when detector geometry is uncertain or when you want a stronger
initial geometry before refining diffraction parameters.

### Parameters panel

The parameters panel is the control surface for geometry, lattice, mosaic,
beam, stacking, occupancy, and fitting controls.

In practice:

1. Use the geometry controls until detector-space features land correctly.
2. Check integrations to verify widths and intensity trends.
3. Refine mosaic and structural terms only after the geometry is stable.

The `Fit Mosaic Shapes` action is geometry-locked. It reuses the exact
multi-background dataset bundle from the last successful manual geometry fit,
so any change to manual picks, selected backgrounds, or shared-theta metadata
requires rerunning geometry fit before the mosaic-shape step.

Within that cached bundle, the mosaic fitter keeps the selected specular
`(00l)` family, then reduces the off-specular side to the top three current
HKL/Qr groups so paired reflections with identical `Qr` are not both simulated.
Specular picks contribute both `2theta` line-shape terms and relative-intensity
constraints across the selected specular family, while the retained
off-specular groups contribute `phi` line-shape terms only.

Saving parameter snapshots regularly is the easiest way to keep iterations
reproducible.

## Read This First: The Physical Story of the Pipeline

If you are new to the project, read this section first and then return to the
reference sections below. The rest of the file stays close to the code; this
overview explains the physical story those equations are serving. In this
document, "best" always means "best tradeoff for RA-SIM's current goals":
enough physics fidelity to place and weight peaks correctly in the regimes we
study, enough determinism for fitting and debugging, and enough speed to run
many forward evaluations inside optimizers. It does not mean universally best
for every scattering experiment or every optics regime.

1. RA-SIM models a grazing-incidence x-ray experiment. A beam arrives at a
   shallow angle, interacts with a tilted sample, and produces scattered rays
   that land on a finite detector. The core simulation question is: for a given
   geometry, lattice, beam distribution, and optics model, which outgoing rays
   are allowed and how bright are they?
2. The code uses separate laboratory, sample, and detector frames because each
   frame has one clean physical job. The lab frame is where hardware lives, the
   sample frame is where reciprocal-space and in-sample optics are simplest,
   and the detector frame is where 3D intersections become 2D pixels. Keeping
   those jobs separated is easier to reason about than carrying one giant
   rotated expression through the entire pipeline.
3. The simulator does not treat the beam as one perfect ray because real data
   are broadened by beam footprint, divergence, wavelength spread, and beam
   position spread. RA-SIM therefore propagates a bundle of sampled beam states
   rather than a single nominal trajectory. That is why peak shapes, split
   maxima, and visibility changes can be captured at all.
4. Entry and exit optics are handled separately from reciprocal-space
   scattering because they answer different physical questions. The optics say
   how strongly the wave enters, decays within, and exits the sample, while
   `solve_q(...)` says which outgoing in-sample directions satisfy the
   scattering geometry. This separation is what lets RA-SIM keep one geometric
   scattering core while swapping between fast and exact optics transport.
5. The reciprocal-space solve is a circle problem because the allowed `Q`
   vectors satisfy two sphere constraints with known radii and centers. Once
   that geometry is written down analytically, the code can integrate mosaic
   density along a one-parameter circle instead of asking a generic nonlinear
   solver to search 3D space from scratch. That is both faster and more stable
   inside repeated fit evaluations.
6. Detector hits are generated from outgoing directions and then deposited
   bilinearly because the detector sees a continuous ray intersection, not a
   pre-quantized pixel. The outgoing direction determines where the ray lands;
   the optics determine how much intensity that hit carries. Bilinear
   deposition then turns a subpixel hit into a smooth image that moves
   continuously when geometry parameters change.
7. The fitting workflow is staged because not all parameters are equally
   identifiable at the same time. Geometry is anchored first from peak
   positions, then matching logic organizes correspondences, and only after that
   do mosaic-shape and image-space refinements spend effort on broader profile
   details. This prevents width, background, or intensity nuisance terms from
   compensating for basic geometry mistakes.
8. The hBN calibrant path is separate because rings constrain detector geometry
   differently from sparse Bragg spots. Ring distortion is a global geometric
   signal, so ellipse fitting and projective tilt correction are the natural
   tools there. Sparse spot fitting, by contrast, is fundamentally a
   correspondence problem between predicted and measured peak locations.

## Code map

| Area | Main files | Role |
| --- | --- | --- |
| Typed simulation entrypoints | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py), [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Public request/response API for forward simulation. |
| Legacy positional wrapper | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Backward-compatible `simulate_diffraction(...)` entrypoint. |
| Beam and profile sampling | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Latin-hypercube beam sampling and optional beam-state clustering. |
| Diffraction kernel | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Sample/detector geometry, optics, `solve_q`, detector projection, hit-table generation. |
| Material and optics helpers | [`ra_sim/utils/calculations.py`](../ra_sim/utils/calculations.py) | Refractive index, Fresnel transmission, Bragg-angle helper math. |
| Geometry and image fitting | [`ra_sim/fitting/optimization.py`](../ra_sim/fitting/optimization.py) | Geometry fitting, geometry-cached mosaic-shape fitting, legacy mosaic-width fitting, image refinement, identifiability analysis. |
| Background peak matching | [`ra_sim/fitting/background_peak_matching.py`](../ra_sim/fitting/background_peak_matching.py) | Measured-summit detection and scored one-to-one simulated/measured matching. |
| Geometry-fit dataset preparation | [`ra_sim/gui/geometry_fit.py`](../ra_sim/gui/geometry_fit.py), [`ra_sim/gui/geometry_overlay.py`](../ra_sim/gui/geometry_overlay.py), [`ra_sim/gui/_runtime/runtime_impl.py`](../ra_sim/gui/_runtime/runtime_impl.py) | Manual picking, orientation resolution, dataset assembly, overlay bookkeeping. |
| Rod intensity preprocessing | [`ra_sim/utils/stacking_fault.py`](../ra_sim/utils/stacking_fault.py) | Hendricks-Teller rod intensity generation and Qr grouping. |
| hBN calibrant fitter | [`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py) | Click snapping, robust ellipse fitting, iterative ring refinement, projective tilt optimization. |
| Instrument defaults | [`config/instrument.yaml`](../config/instrument.yaml) | GUI defaults, solver config, and simulation defaults exposed to users. |

## Notation and units

Why this choice: The first difficulty in this codebase is not one specific
equation, but keeping several coordinate systems and unit systems straight at
once. Writing the frames and units down explicitly is the best tradeoff for
RA-SIM because the same event appears as a ray in the lab, a scattering
constraint in the sample, and a residual in detector pixels. The notation here
mirrors the code's actual data layout: lattice-scale quantities stay in
angstrom-based units, hardware geometry stays in meters, and fitting errors stay
in pixels. That separation keeps later formulas readable and keeps unit
conversions from being hidden inside the algebra.

### Coordinate frames

- Laboratory frame: the kernel launches rays and places the detector in this frame. The nominal detector center is at `D = (0, distance_m, 0)`.
- Sample frame: reciprocal vectors and in-sample optics are handled in the rotated sample frame. `R_sample` maps sample-frame vectors into the laboratory frame.
- Detector plane frame: detector hits are resolved into the orthonormal in-plane axes `e1_det` and `e2_det`.
- Pixel frame: detector images are indexed as `(row, col)`, with `row` increasing downward and `col` increasing to the right.

### Pixel versus metric detector coordinates

If a detector-plane hit is `P_hit`, then

\[
x_{\mathrm{det}} = (P_{\mathrm{hit}} - D)\cdot e_{1,\mathrm{det}},
\qquad
y_{\mathrm{det}} = (P_{\mathrm{hit}} - D)\cdot e_{2,\mathrm{det}}.
\]

These are metric detector-plane coordinates in meters. They are converted to floating pixel coordinates by

\[
\mathrm{row}_f = \mathrm{center}_{\mathrm{row}} - \frac{y_{\mathrm{det}}}{p},
\qquad
\mathrm{col}_f = \mathrm{center}_{\mathrm{col}} + \frac{x_{\mathrm{det}}}{p},
\]

with `p = pixel_size_m`.

### Units

- Lattice parameters `a`, `c`, wavelengths `lambda_angstrom`, and reciprocal magnitudes are in angstrom-based units.
- Detector distance, pixel pitch, finite sample dimensions, beam offsets used as physical positions, and attenuation path lengths are in meters.
- Most user-facing rotations are stored in degrees and converted to radians inside the kernel.
- Pixel residuals in the fitting layer are in detector pixels.

### Lattice and reciprocal-space formulas

These formulas answer two recurring questions for the simulator: which
reciprocal vector belongs to a chosen HKL, and what nominal Bragg scale that
reflection lives on before beam spread, optics, and mosaic broadening reshape
it.

For the hexagonal lattice used by the simulator,

\[
\frac{1}{d_{hkl}^2}
= \frac{4}{3}\frac{h^2+hk+k^2}{a^2} + \frac{l^2}{c^2},
\qquad
2\theta_B = 2\arcsin\left(\frac{\lambda}{2d_{hkl}}\right).
\]

The nominal reciprocal vector inserted into the kernel is

\[
\mathbf{G} =
\begin{bmatrix}
0 \\
\frac{4\pi}{a}\sqrt{\frac{h^2 + hk + k^2}{3}} \\
\frac{2\pi l}{c}
\end{bmatrix}
=
\begin{bmatrix}
0 \\
G_r \\
G_z
\end{bmatrix}.
\]

The radial term `Gr` and out-of-plane term `Gz` matter because source geometries are reused when `(Gr, Gz, forced_sample_idx)` match exactly.

### Refractive index

This is the optics input, not the reciprocal-space input. It tells the code how
the wave is transmitted and attenuated at interfaces, which is why the same
`n2` value reappears in the entry and exit transport equations later on.

The simulator uses the complex refractive index

\[
n = 1 - \delta + i\beta,
\qquad
\delta = \frac{r_e \lambda^2 \rho_e}{2\pi},
\qquad
\beta = \frac{\mu \lambda}{4\pi}.
\]

In the code this is the sample refractive index `n2`, or the wavelength-dependent override `n2_sample_array` when the material path builds one sample-by-sample. The dielectric constant is `eps2 = n2^2`.

### Meaning of the geometry symbols

- `gamma`: detector tilt about the laboratory x axis in the detector rotation `R_x(gamma)`.
- `Gamma`: detector tilt about the laboratory z axis in the detector rotation `R_z(Gamma)`.
- `theta_initial`: additional sample tilt angle applied about the configured center-of-rotation axis.
- `cor_angle`: elevation of the center-of-rotation axis before the optional `psi_z` azimuthal rotation.
- `psi`: in-plane sample azimuth used in `R_z(psi)`.
- `psi_z`: azimuth used when rotating the center-of-rotation axis itself.
- `chi`: sample tilt in `R_y(chi)` before the extra `theta_initial` rotation.
- `zs`: sample-plane reference shift along the laboratory/sample geometry path; it enters the sample plane anchor point `P0 = (0, 0, -zs)`.
- `zb`: vertical beam-origin offset used when building the beam start point `beam_start = [beam_x, -20 mm, beam_y - zb]`.
- `debye_x`: phenomenological damping scale applied to `Qz` through `exp(-Qz^2 debye_x^2)`.
- `debye_y`: phenomenological damping scale applied to `Qx^2 + Qy^2` through `exp(-(Qx^2 + Qy^2) debye_y^2)`.
- `n2`: complex sample refractive index, used in both entry and exit optics.
- `solve_q_steps`: integration budget for the reciprocal-space circle solve. In uniform mode it is the number of deterministic angle samples. In adaptive mode it is the maximum interval count.
- `solve_q_rel_tol`: adaptive relative error tolerance for Simpson-versus-trapezoid interval refinement.
- `solve_q_mode`: arc integration mode, `0 = uniform`, `1 = adaptive`.

### Rotation construction used by the kernel

The detector normal starts from `n_detector` and is rotated by

\[
n_{\mathrm{det,rot}} = R_z(\Gamma)\,R_x(\gamma)\,n_{\mathrm{det}}.
\]

The sample frame first applies

\[
R_{ZY} = R_z(\psi)\,R_y(\chi),
\]

with the code's `R_z` sign convention

\[
R_z(\psi) =
\begin{bmatrix}
\cos\psi & \sin\psi & 0 \\
-\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}.
\]

The center-of-rotation axis is built from `cor_angle = \alpha` and `psi_z` as

\[
a_0 = (\cos\alpha, 0, \sin\alpha),
\qquad
a = (\cos\psi_z\cos\alpha,\; -\sin\psi_z\cos\alpha,\; \sin\alpha),
\]

then normalized and inserted into Rodrigues' formula for the extra `theta_initial` rotation. The final sample rotation is

\[
R_{\mathrm{sample}} = R_{\mathrm{cor}}R_{ZY}.
\]

The sample surface normal is the rotated sample normal,

\[
n_{\mathrm{surf}} = \mathrm{normalize}(R_{\mathrm{cor}}\,n_{ZY}),
\]

and the sample-plane anchor point is rotated as `P0_rot = R_sample @ P0`, then forced to `P0_rot[0] = 0`.

Appendix A expands this into the exact axis construction, Rodrigues form, and reference-point convention used by the kernel.

## Simulation parameter inventory

The simulation surface is split across dataclasses, the legacy wrapper, and the GUI config defaults. The tables below state the parameter name, units, default or source, where it enters the code, and what part of the math it controls.

### Instrument defaults from `config/instrument.yaml`

These are GUI and default-profile values, not hard kernel constants unless the GUI passes them through unchanged.

| Name | Units | Default | Code source | Mathematical role |
| --- | --- | --- | --- | --- |
| `instrument.detector.image_size` | px | `3000` | [`config/instrument.yaml`](../config/instrument.yaml) | Output image dimension. |
| `instrument.detector.pixel_size_m` | m | `1.0e-4` | [`config/instrument.yaml`](../config/instrument.yaml) | Detector pixel pitch used in projection `row_f`, `col_f`. |
| `instrument.geometry_defaults.distance_m` | m | `0.075` | [`config/instrument.yaml`](../config/instrument.yaml) | Detector distance `D`. |
| `instrument.geometry_defaults.rot1` | rad-style GUI field | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI detector tilt default; mapped downstream into detector angles. |
| `instrument.geometry_defaults.rot2` | rad-style GUI field | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI detector tilt default; mapped downstream into detector angles. |
| `instrument.geometry_defaults.poni1_m` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI detector-center helper default. |
| `instrument.geometry_defaults.poni2_m` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI detector-center helper default. |
| `instrument.geometry_defaults.wavelength_m` | m | `1.0e-10` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI wavelength default. |
| `instrument.beam.wavelength_angstrom` | A | `1.54` | [`config/instrument.yaml`](../config/instrument.yaml) | Nominal incident wavelength. |
| `instrument.beam.divergence_fwhm_deg` | deg FWHM | `0.05` | [`config/instrument.yaml`](../config/instrument.yaml) | Beam angular spread used to build `theta_array` and `phi_array`. |
| `instrument.beam.sigma_mosaic_fwhm_deg` | deg FWHM | `0.8` | [`config/instrument.yaml`](../config/instrument.yaml) | Gaussian mosaic width shown in the GUI. |
| `instrument.beam.gamma_mosaic_fwhm_deg` | deg FWHM | `0.7` | [`config/instrument.yaml`](../config/instrument.yaml) | Lorentzian mosaic width shown in the GUI. |
| `instrument.beam.eta` | unitless | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Pseudo-Voigt mixing fraction. |
| `instrument.beam.solve_q_steps` | samples or intervals | `1000` | [`config/instrument.yaml`](../config/instrument.yaml) | Reciprocal-space arc integration budget. |
| `instrument.beam.solve_q_rel_tol` | unitless | `5.0e-4` | [`config/instrument.yaml`](../config/instrument.yaml) | Adaptive arc integration tolerance. |
| `instrument.beam.solve_q_mode` | enum | `uniform` | [`config/instrument.yaml`](../config/instrument.yaml) | Chooses uniform or adaptive `solve_q`. |
| `instrument.beam.bandwidth_percent` | % | `0.7` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI wavelength-spread control. |
| `instrument.beam.bandwidth_sigma_fraction` | fractional sigma | `5.0e-05` | [`config/instrument.yaml`](../config/instrument.yaml) | Alternate bandwidth default used in some profile-generation paths. |
| `instrument.sample_orientation.theta_initial_deg` | deg | `6.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Extra sample tilt about the CoR axis. |
| `instrument.sample_orientation.cor_deg` | deg | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | CoR-axis elevation. |
| `instrument.sample_orientation.chi_deg` | deg | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Sample `R_y(chi)` tilt. |
| `instrument.sample_orientation.psi_deg` | deg | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Sample in-plane azimuth. |
| `instrument.sample_orientation.psi_z_deg` | deg | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | CoR-axis azimuth. |
| `instrument.sample_orientation.zb` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Beam-origin z offset. |
| `instrument.sample_orientation.zs` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Sample-plane anchor shift. |
| `instrument.sample_orientation.width_m` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Finite sample width; zero keeps the legacy unbounded width. |
| `instrument.sample_orientation.length_m` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Finite sample length; zero keeps the legacy unbounded length. |
| `instrument.sample_orientation.depth_m` | m | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | GUI slab-thickness default. |
| `instrument.debye_waller.x` | reciprocal-length damping scale | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Damps `Qz`. |
| `instrument.debye_waller.y` | reciprocal-length damping scale | `0.0` | [`config/instrument.yaml`](../config/instrument.yaml) | Damps in-plane `Qr`. |

### `DetectorGeometry`

| Field | Units | Default | Used in | Role |
| --- | --- | --- | --- | --- |
| `image_size` | px | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Square detector image size. |
| `av` | A | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | In-plane lattice parameter `a`. |
| `cv` | A | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Out-of-plane lattice parameter `c`. |
| `lambda_angstrom` | A | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Nominal wavelength sent to the kernel; per-sample wavelengths can differ. |
| `distance_m` | m | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Detector center position `(0, distance_m, 0)`. |
| `gamma_deg` | deg | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Detector x-tilt. |
| `Gamma_deg` | deg | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Detector z-tilt. |
| `chi_deg` | deg | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Sample `R_y(chi)` tilt. |
| `psi_deg` | deg | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Sample `R_z(psi)` azimuth. |
| `psi_z_deg` | deg | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | CoR-axis azimuth. |
| `zs` | m | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Sample-plane reference point offset. |
| `zb` | m | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Beam-origin offset in the incident-ray start point. |
| `center` | px | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Detector beam center `(row, col)` used in metric-to-pixel projection. |
| `theta_initial_deg` | deg | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Extra sample tilt about the CoR axis. |
| `cor_angle_deg` | deg | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | CoR-axis elevation before `psi_z` rotation. |
| `unit_x` | unit vector | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Reference axis projected into the detector plane to define `e1_det`. |
| `n_detector` | unit vector | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Unrotated detector normal. |
| `pixel_size_m` | m | `100e-6` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Pixel pitch in the detector projection. |
| `sample_width_m` | m | `0.0` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Optional finite sample width clip. |
| `sample_length_m` | m | `0.0` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Optional finite sample length clip. |

### `BeamSamples`

All beam arrays are aligned by sample index.

| Field | Units | Default | Used in | Role |
| --- | --- | --- | --- | --- |
| `beam_x_array` | m | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Lateral beam start coordinate in `beam_start`. |
| `beam_y_array` | m | required | [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py) | Vertical beam start coordinate in `beam_start`. |
| `theta_array` | rad | required | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Incident angular deviation controlling `k_in_z`. |
| `phi_array` | rad | required | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Incident angular deviation controlling `k_in_x` and `k_in_y`. |
| `wavelength_array` | A | required | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Per-sample wavelength. |
| `sample_weights` | unitless | `None` | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Representative-beam weight after clustering. |
| `n2_sample_array` | complex | `None` | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Optional wavelength-dependent refractive index override. |

### `MosaicParams` and `DebyeWallerParams`

| Field | Units | Default | Used in | Role |
| --- | --- | --- | --- | --- |
| `sigma_mosaic_deg` | deg | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Gaussian pseudo-Voigt width for the mosaic density. |
| `gamma_mosaic_deg` | deg | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Lorentzian pseudo-Voigt width. |
| `eta` | unitless | required | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Gaussian/Lorentzian mixture fraction. |
| `solve_q_steps` | samples or intervals | `1000` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Uniform sample count or adaptive interval budget. |
| `solve_q_rel_tol` | unitless | `5.0e-4` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Adaptive relative tolerance. |
| `solve_q_mode` | enum | `0` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | `0 = uniform`, `1 = adaptive`. |
| `DebyeWallerParams.x` | reciprocal-length damping scale | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Applies `exp(-Qz^2 x^2)`. |
| `DebyeWallerParams.y` | reciprocal-length damping scale | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Applies `exp(-(Qx^2+Qy^2) y^2)`. |

Kernel-side hard limits from [`ra_sim/simulation/diffraction.py`](../ra_sim/simulation/diffraction.py):

- `DEFAULT_SOLVE_Q_STEPS = 1000`
- `MIN_SOLVE_Q_STEPS = 32`
- `MAX_SOLVE_Q_STEPS = 8192`
- `DEFAULT_SOLVE_Q_BASE_INTERVALS = 48`
- `MIN_SOLVE_Q_BASE_INTERVALS = 8`
- `DEFAULT_SOLVE_Q_REL_TOL = 5.0e-4`
- `MIN_SOLVE_Q_REL_TOL = 1.0e-6`
- `MAX_SOLVE_Q_REL_TOL = 5.0e-2`
- `DEFAULT_SOLVE_Q_MODE = 0`

### `SimulationRequest`

| Field | Units | Default | Used in | Role |
| --- | --- | --- | --- | --- |
| `miller` | integer triplets | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Reflection list `(H, K, L)`. |
| `intensities` | arbitrary intensity units | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Reflection amplitudes sent to detector accumulation. |
| `geometry` | dataclass | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | All detector/sample geometry inputs. |
| `beam` | dataclass | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Per-sample beam states and optional weights. |
| `mosaic` | dataclass | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Reciprocal-space broadening and integration settings. |
| `debye_waller` | dataclass | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Phenomenological damping factors. |
| `n2` | complex | required | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Constant sample refractive index if no per-sample override exists. |
| `image_buffer` | image array | `None` | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Optional output buffer; otherwise a zeroed `float64` image is allocated. |
| `save_flag` | int | `0` | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | Legacy flag propagated into the kernel. |
| `record_status` | bool | `False` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Enables per-sample status diagnostics. |
| `thickness` | m | `0.0` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | If positive, overrides evanescent decay lengths with a fixed slab thickness. |
| `optics_mode` | enum or `None` | `None` | [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) | `None` means use the kernel default `OPTICS_MODE_FAST`; otherwise force fast or exact optics. |
| `collect_hit_tables` | bool | `True` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Enables per-reflection subpixel hit tables. |
| `exit_projection_mode` | enum string | `"internal"` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Chooses whether detector geometry uses the normalized solved outgoing direction (`"internal"`, default) or the legacy refracted-angle reconstruction (`"refracted"`). The internal default avoids the near-critical dead band that produced the horizontal empty stripe. |
| `single_sample_indices` | integer array or `None` | `None` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Forces selected beam samples per reflection. |
| `best_sample_indices_out` | integer array or `None` | `None` | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py) | Optional output buffer for best-sample tracking. |

### Legacy `simulate_diffraction(...)` wrapper

The legacy wrapper packs positional arguments into the typed request. Its extra profile-generation inputs are:

| Name | Units | Default | Used in | Role |
| --- | --- | --- | --- | --- |
| `num_samples` | count | required | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Number of generated beam states if `profile_samples` is absent. |
| `divergence_sigma` | rad | required | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Standard deviation for `theta_array` and `phi_array`. |
| `bw_sigma` | m | required | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Standard deviation for `beam_x_array` and `beam_y_array`. |
| `sigma_mosaic_var` | deg | required | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Passed into `MosaicParams.sigma_mosaic_deg`. |
| `gamma_mosaic_var` | deg | required | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Passed into `MosaicParams.gamma_mosaic_deg`. |
| `eta_var` | unitless | required | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Passed into `MosaicParams.eta`. |
| `bandwidth` | fractional sigma | `0.007` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Used in `wavelength = lambda0 + lambda0 * bandwidth * z`. |
| `profile_samples` | five aligned arrays or dict | `None` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Bypasses random profile generation. |
| `profile_rng` | RNG or seed | `None` | [`ra_sim/simulation/mosaic_profiles.py`](../ra_sim/simulation/mosaic_profiles.py) | Seed source for low-discrepancy profile generation. |
| `pixel_size_m` | m | `100e-6` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Forwarded into `DetectorGeometry.pixel_size_m`. |
| `sample_width_m` | m | `0.0` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Forwarded into finite sample clipping. |
| `sample_length_m` | m | `0.0` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Forwarded into finite sample clipping. |
| `thickness` | m | `0.0` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Forwarded into slab entry/exit attenuation lengths. |
| `n2_sample_array` | complex array | `None` | [`ra_sim/simulation/simulation.py`](../ra_sim/simulation/simulation.py) | Per-sample refractive index override. |

### Beam-profile generation and clustering

When the wrapper generates beam profiles internally, [`generate_random_profiles`](../ra_sim/simulation/mosaic_profiles.py) uses Latin-hypercube sampling, antithetic pairing, and inverse-normal mapping:

\[
\theta_i = \sigma_{\mathrm{div}} z_0,
\quad
\phi_i = \sigma_{\mathrm{div}} z_1,
\quad
x_i = \sigma_{\mathrm{bw}} z_2,
\quad
y_i = \sigma_{\mathrm{bw}} z_3,
\quad
\lambda_i = \lambda_0 + \lambda_0\,\mathrm{bandwidth}\,z_4,
\]

where `z_j` are deterministic Gaussianized low-discrepancy samples.

When clustering is enabled, [`cluster_beam_profiles`](../ra_sim/simulation/mosaic_profiles.py) compresses the 5D beam states `(beam_x, beam_y, theta, phi, wavelength)` into weighted representatives and returns:

- clustered beam arrays
- `sample_weights`
- `raw_to_cluster`
- `cluster_to_rep`

Those weights enter the final detector intensity linearly.

## Forward simulation: strict algorithm

Why this choice: The forward model is written as a fixed pipeline because each
stage makes one different physical decision: where a beam sample hits, which
`Q` states are allowed, how the outgoing wave is weighted, and where it lands
on the detector. That is the best tradeoff for RA-SIM because fitting,
debugging, and caching all benefit from inspectable intermediate products
instead of a single opaque black-box simulator. The beam bundle is also sampled
deterministically, with low-discrepancy profiles and optional clustering,
instead of with fresh Monte Carlo draws at every call. In practice that keeps
optimization objectives, status codes, and regression tests reproducible.

The live forward path is `simulate(...)` or `simulate_qr_rods(...)` through [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py), which forwards typed inputs into [`process_peaks_parallel_safe`](../ra_sim/simulation/diffraction.py). The algorithm is:

1. Build detector and sample rotations from `gamma`, `Gamma`, `chi`, `psi`, `psi_z`, `theta_initial`, and `cor_angle`.
2. Build or receive aligned beam-sample arrays `(beam_x, beam_y, theta, phi, wavelength)` and optional `sample_weights`.
3. For each beam sample, intersect the incident ray with the sample plane, apply finite sample clipping if `sample_width_m` or `sample_length_m` are nonzero, and precompute entry optics quantities that do not depend on `(H, K, L)`.
4. For each reflection, build `G_vec = (0, Gr, Gz)` from `(H, K, L, a, c)`.
5. For each valid beam sample, solve the reciprocal-space circle problem `solve_q(...)`.
6. For each accepted `Q` solution, build the solved outgoing in-sample wavevector, apply exit optics and attenuation to the intensity, project the outgoing direction to the detector using the selected exit-projection mode, convert to `(row_f, col_f)`, and deposit intensity bilinearly.
7. Optionally emit hit tables, `Q` tables, miss tables, and per-sample status codes.

The simulation is deterministic once the beam arrays are fixed. There is no Monte Carlo scattering inside the kernel itself; even the beam sampling is low-discrepancy and can be clustered into deterministic weighted representatives.

## Pedagogical view of the optics modes

Why this choice: RA-SIM keeps two optics branches because there is no single
transport model that is best for every use case the project cares about. The
fast branch is best when the dominant constraint is throughput inside fitting
loops, while the exact slab branch is best when near-critical-angle transport
and complex attenuation materially affect intensities. Both branches
deliberately reuse the same reflection list, `solve_q(...)` machinery, and
default detector projection so that changing optics does not silently change the
rest of the physics. In implementation terms, the mode switch changes entry and
exit transport quantities, not the reciprocal-space search itself.

The code has two optics branches:

- `OPTICS_MODE_FAST`, labeled in the GUI as `Original Fast Approx (Fresnel + Beer-Lambert)`
- `OPTICS_MODE_EXACT`, labeled in the GUI as `Complex-k DWBA slab optics (Precise)`

Pedagogically, both modes do the same three-stage calculation:

1. Build an in-sample incoming wave from the incident beam.
2. Scatter that wave through the same reciprocal-space `solve_q(...)` machinery.
3. Build an outgoing wave, transmit it back out of the sample, and project it to the detector.

The difference is therefore not in the reflection list, the `solve_q(...)`
geometry, or the structure-factor model. In the default
`exit_projection_mode = "internal"` path, both optics modes also use the same
solved outgoing direction for detector geometry. The difference is in how the
code computes the entry and exit transport through the sample and therefore the
intensity weights attached to each hit.

### Shared intensity skeleton

The next product is the bookkeeping identity for one accepted hit. It says the
deposited intensity is built from a reflection amplitude, a beam-sample weight,
a reciprocal-space weight, an optics transport factor, and phenomenological
broadening terms.

For any accepted reciprocal-space solution `Q = (Q_x, Q_y, Q_z)`, both modes
ultimately use the same detector intensity structure:

\[
I_{\mathrm{hit}}
= I_{\mathrm{reflection}}
\cdot w_{\mathrm{sample}}
\cdot I_Q
\cdot \mathrm{prop\_fac}
\cdot \exp(-Q_z^2 d_x^2)
\cdot \exp(-(Q_x^2 + Q_y^2)d_y^2),
\]

with

\[
\mathrm{prop\_fac}
= T_i^2 T_f^2
\cdot \exp(-2\,\Im(k_{z,\mathrm{in}})\,L_{\mathrm{in}})
\cdot \exp(-2\,\Im(k_{z,\mathrm{out}})\,L_{\mathrm{out}}).
\]

So the optics mode only changes the four quantities

\[
T_i^2,\quad T_f^2,\quad k_{z,\mathrm{in}},\quad k_{z,\mathrm{out}},
\]

and the corresponding in-sample and outgoing wavevectors built from them.

### Fast mode: refracted-angle plus exponential attenuation

The fast branch treats the sample optics as a transmitted grazing-angle problem
plus exponential damping through an absorbing medium.

For entry, let `k_0 = 2\pi/\lambda` and let `\theta_i'` be the local grazing
angle on the sample. The code first computes an approximate transmitted angle

\[
\theta_t' = \mathrm{transmit\_angle\_grazing}(\theta_i', n_2),
\]

then sets an in-sample wavevector magnitude

\[
k_{\mathrm{scat}} = k_0 \sqrt{\max(\Re(\epsilon_2), 0)}.
\]

Using the local in-plane azimuth `\phi_i'`, the in-sample incoming components are

\[
k_{x,\mathrm{scat}} = k_{\mathrm{scat}}\cos\theta_t' \sin\phi_i',
\qquad
k_{y,\mathrm{scat}} = k_{\mathrm{scat}}\cos\theta_t' \cos\phi_i'.
\]

The normal component is taken from the helper

\[
(\Re(k_z), \Im(k_z)) = \mathrm{ktz\_components}(k_0, n_2, \theta_t'),
\]

with the implementation flipping the real part so the wave points into the
sample:

\[
\Re(k_z) \leftarrow -\Re(k_z).
\]

The entry transmission factor is an approximate polarization average:

\[
T_i^2 \approx \frac{|t_s|^2 + |t_p|^2}{2},
\]

where `t_s` and `t_p` come from `fresnel_transmission(...)`.

For exit, the code takes the absolute in-sample exit angle derived from the
candidate outgoing wavevector,

\[
|2\theta_t'| = \left|\operatorname{atan}\left(\frac{k_{tz}'}{k_r}\right)\right|,
\]

and looks up four precomputed quantities as a function of that angle:

- `T_f^2`
- `\Im(k_{z,\mathrm{out}})`
- `L_{\mathrm{out}}`
- the absolute external exit angle

So the fast mode is best thought of as:

- use a refracted transmitted angle to define the in-sample ray,
- approximate Fresnel transmission with squared amplitudes,
- attenuate with exponential damping,
- and reuse a lookup table for the exit optics so the inner loop stays cheap.

### Exact mode: complex-`k_z` slab transport

The exact branch keeps the same scattering geometry but replaces the helper
angle model with a complex wavevector construction that enforces the slab
dispersion relation and interface matching more directly.

For entry, the in-plane wavevector is conserved:

\[
k_{\parallel,i} = k_0 |\cos\theta_i'|.
\]

The code then forms the vacuum-side and sample-side normal wavevector
components

\[
k_{z1,i} = \mathrm{kz\_branch\_decay}(k_0^2 - k_{\parallel,i}^2),
\qquad
k_{z2,i} = \mathrm{kz\_branch\_decay}(\epsilon_2 k_0^2 - k_{\parallel,i}^2).
\]

The branch choice is the decaying one, so `\Im(k_{z2,i}) \ge 0`. The stored
incoming in-sample wavevector components become

\[
k_{x,\mathrm{scat}} = k_{\parallel,i}\sin\phi_i',
\qquad
k_{y,\mathrm{scat}} = k_{\parallel,i}\cos\phi_i',
\qquad
\Re(k_z) = -|\Re(k_{z2,i})|,
\qquad
\Im(k_z) = |\Im(k_{z2,i})|.
\]

Instead of using `|t|^2` directly, the code converts the exact Fresnel
transmission amplitudes into transmitted power using the appropriate flux
ratio:

\[
T_i^2 = \frac{1}{2}\left(T_{i,s}^{\mathrm{power}} + T_{i,p}^{\mathrm{power}}\right).
\]

For exit, after adding `Q` to the in-sample incident wavevector, the radial
in-plane magnitude is

\[
k_r = \sqrt{k_{tx}'^2 + k_{ty}'^2}.
\]

The outgoing slab and vacuum normal components are then

\[
k_{z2,f} = \mathrm{kz\_branch\_decay}(\epsilon_2 k_0^2 - k_r^2),
\qquad
k_{z3,f} = \mathrm{kz\_branch\_decay}(k_0^2 - k_r^2),
\]

with an exact polarization-average exit transmission

\[
T_f^2 = \frac{1}{2}\left(T_{f,s}^{\mathrm{power}} + T_{f,p}^{\mathrm{power}}\right).
\]

The external exit angle is then reconstructed from the vacuum dispersion:

\[
2\theta_t = \arccos\left(\mathrm{clamp}\left(\frac{k_r}{k_0}, -1, 1\right)\right)\operatorname{sign}(2\theta_t').
\]

So the exact mode is best thought of as:

- conserve the in-plane component of the wavevector at the interface,
- solve for the complex normal component from the dispersion relation,
- use that same complex `k_z` for both transmission and attenuation,
- and recompute the exit optics per accepted scattering solution instead of
  interpolating a cached table.

### What "more exact" means here

In this repository, "exact" does not mean a different diffraction model. It
means the transport through the air/sample interfaces is computed from the
complex slab wavevector rather than from a transmitted-angle shortcut.

That is why the exact mode is slower but usually more faithful near critical
angles, strong refraction regions, or other situations where the normal
component of the wavevector becomes delicate.

The current exact path is still a single air/sample/air slab, not a general
multilayer reflectivity stack. The detailed implementation-faithful formulas
for each step appear in the next sections.

## Exact per-sample ray tracing and entry optics

Why this choice: This stage traces each sampled beam state through the real
sample geometry before any reciprocal-space solve is attempted. That is the
best tradeoff for RA-SIM because local incidence angle, finite footprint, and
beam-position offsets all change which parts of the sample contribute and how
strongly they contribute. A single nominal ray would be cheaper, but it would
erase the same visibility and shape effects the later fitting stages need to
explain. In implementation terms, the kernel intersects each beam sample with
the sample plane, applies finite sample clipping, and only then precomputes the
entry-optics quantities reused across reflections.

The beam-sample precompute lives in [`_precompute_sample_terms`](../ra_sim/simulation/diffraction.py). This section describes the exact formulas used there.

### Incident-ray start point and direction

For beam sample `i`,

\[
P_{\mathrm{beam},i} =
\begin{bmatrix}
\mathrm{beam\_x}_i \\
-20\times 10^{-3} \\
\mathrm{beam\_y}_i - z_b
\end{bmatrix},
\]

and with `dtheta = theta_array[i]`, `dphi = phi_array[i]`,

\[
k_{\mathrm{in},i} =
\begin{bmatrix}
\cos(d\theta)\sin(d\phi) \\
\cos(d\theta)\cos(d\phi) \\
\sin(d\theta)
\end{bmatrix}.
\]

The incident ray is intersected with the sample plane anchored at `P0_rot` and normal to `n_surf` using [`intersect_line_plane`](../ra_sim/simulation/diffraction.py):

\[
t = \frac{(P_{\mathrm{plane}} - P_0)\cdot n_{\mathrm{plane}}}{k\cdot n_{\mathrm{plane}}},
\qquad
P_{\mathrm{hit}} = P_0 + t k.
\]

The implementation-specific validity rules are:

- if `|k . n_plane| < 1e-14`, the ray is treated as parallel to the plane
- if the start point already lies on the plane within `1e-6`, the start point is accepted as the hit
- otherwise a parallel ray is rejected
- if `t < -1e-9`, the hit is rejected as lying behind the ray start
- if `-1e-9 <= t < 0`, the code clamps `t = 0`

So grazing or almost-on-plane rays are retained instead of being spuriously dropped.

### Finite sample clipping

After the sample-plane hit `I_plane = (ix, iy, iz)` is found, the code projects the relative hit position onto the rotated sample axes

\[
\Delta P = I_{\mathrm{plane}} - P_{0,\mathrm{rot}},
\qquad
x_{\mathrm{local}} = \Delta P \cdot \hat{s}_x,
\qquad
y_{\mathrm{local}} = \Delta P \cdot \hat{s}_y,
\]

where `sample_axis_x` and `sample_axis_y` are the first two columns of `R_sample`.

Rejection rules:

- if `sample_width_m > 0`, require `|x_local| <= sample_width_m / 2`
- if `sample_length_m > 0`, require `|y_local| <= sample_length_m / 2`
- if either test fails, the beam sample is invalid for all reflections

With both lengths zero, the code keeps the legacy effectively unbounded surface.

### Local incidence angles on the tilted sample

The code computes

\[
\cos\theta_{\mathrm{surf}} = k_{\mathrm{in}}\cdot n_{\mathrm{surf}},
\qquad
\theta_i' = \frac{\pi}{2} - \arccos(\mathrm{clamp}(k_{\mathrm{in}}\cdot n_{\mathrm{surf}}, -1, 1)).
\]

This `theta_i_prime` is the grazing-angle convention used by the optics helpers.

The incident direction is then projected into the sample plane:

\[
k_{\parallel} = k_{\mathrm{in}} - (k_{\mathrm{in}}\cdot n_{\mathrm{surf}})n_{\mathrm{surf}}.
\]

The code normalizes this projected vector when possible, expresses it in the local in-plane basis `(e1_temp, e2_temp)`, and defines

\[
p_1 = k_{\parallel}\cdot e_1,
\qquad
p_2 = k_{\parallel}\cdot e_2,
\qquad
\phi_i' = \frac{\pi}{2} - \operatorname{atan2}(p_2, p_1).
\]

The `e1_temp` basis is built from `cross(n_surf, [0,0,-1])`, with fallbacks to `[1,0,0]`, `[0,1,0]`, or `[0,0,1]` if that cross product is degenerate.

### Exact entry optics branch

When `optics_mode == OPTICS_MODE_EXACT`, the code uses the complex-k slab path:

\[
k_0 = \frac{2\pi}{\lambda_i},
\qquad
k_{\parallel,i} = k_0 |\cos\theta_i'|.
\]

Then

\[
k_{z1,i} = \mathrm{kz\_branch\_decay}(k_0^2 - k_{\parallel,i}^2),
\qquad
k_{z2,i} = \mathrm{kz\_branch\_decay}(\epsilon_2 k_0^2 - k_{\parallel,i}^2).
\]

The transmitted in-sample incident wavevector components used downstream are

\[
k_{x,\mathrm{scat}} = k_{\parallel,i}\sin\phi_i',
\qquad
k_{y,\mathrm{scat}} = k_{\parallel,i}\cos\phi_i',
\qquad
\Re(k_z) = -| \Re(k_{z2,i}) |,
\qquad
\Im(k_z) = | \Im(k_{z2,i}) |.
\]

The magnitude stored as `k_scat` is

\[
k_{\mathrm{scat}} = \sqrt{\max(k_{\parallel,i}^2 + \Re(k_{z2,i})^2,\; 0)}.
\]

Entry transmission is the polarization-average exact Fresnel power transmission:

\[
T_i^2 = \frac{1}{2}\left(T_{i,s}^{\mathrm{power}} + T_{i,p}^{\mathrm{power}}\right).
\]

The in-slab incoming path length is

\[
L_{\mathrm{in}} =
\begin{cases}
\mathrm{thickness}, & \mathrm{thickness} > 0 \\
\frac{1}{2\,\Im(k_z)}, & \text{otherwise.}
\end{cases}
\]

### Fast entry optics branch

When `optics_mode == OPTICS_MODE_FAST`, the code uses the grazing-angle transmitted angle

\[
\theta_t' = \mathrm{transmit\_angle\_grazing}(\theta_i', n_2)
\]

and sets

\[
k_{\mathrm{scat}} = k_0 \sqrt{\max(\Re(\epsilon_2), 0)}.
\]

The in-sample wavevector is

\[
k_{x,\mathrm{scat}} = k_{\mathrm{scat}}\cos\theta_t' \sin\phi_i',
\qquad
k_{y,\mathrm{scat}} = k_{\mathrm{scat}}\cos\theta_t' \cos\phi_i'.
\]

The z component comes from `ktz_components(k0, n2_samp, th_t)` and is then sign-flipped:

\[
(\Re(k_z), \Im(k_z)) = \mathrm{ktz\_components}(k_0, n_2, \theta_t'),
\qquad
\Re(k_z) \leftarrow -\Re(k_z).
\]

The code computes the approximate polarization-average Fresnel power transmission from the complex amplitudes `fresnel_transmission(..., s)` and `fresnel_transmission(..., p)`, then uses the same `L_in` rule as the exact path.

### What is stored per sample

Each valid beam sample contributes the following reflection-invariant quantities:

- the sample-plane hit point `I_plane`
- `k_x_scat`, `k_y_scat`, `re_k_z`, `im_k_z`
- `k_scat`
- `k0`
- `Ti2`
- `L_in`
- `Re(n2)`

Invalid samples simply retain `valid = 0` and are skipped for every reflection.

The routine also chooses `best_idx`, the most central beam sample, by minimizing

\[
\theta_i^2 + \phi_i^2
\]

and then breaking ties by minimizing

\[
\mathrm{beam\_x}_i^2 + \mathrm{beam\_y}_i^2.
\]

That sample is later used for the nominal visibility screen and for hit-table recording when no sample is forced.

## Reciprocal-space circle solve and mosaic weighting

Why this choice: Once the incoming in-sample wave is fixed, the scattering
constraints are geometric enough that RA-SIM can solve them analytically rather
than by launching a generic nonlinear search in 3D. That is the best tradeoff
for this codebase because the sphere-sphere circle construction is fast, stable,
and turns the remaining problem into a one-parameter integration of mosaic
density. It also gives clean deterministic failure codes when the geometry is
impossible or degenerate. The uniform and adaptive modes then choose how to
integrate along that analytic circle, not how to discover the circle in the
first place.

The reciprocal-space solve is [`solve_q`](../ra_sim/simulation/diffraction.py). It does not solve a generic nonlinear system; it constructs the exact circle of intersection of two spheres and then integrates the mosaic density along that circle.

### Sphere-sphere intersection geometry

The next equations define the exact set of allowed `Q` vectors for one
reflection and one sampled beam state. They are the reason `solve_q(...)` can
be both geometric and deterministic instead of a black-box numerical search.

For one reflection, the kernel sets

\[
\mathbf{A} = -k_{\mathrm{in}}^{\mathrm{crystal}},
\qquad
|\mathbf{Q}| = |\mathbf{G}|,
\qquad
|\mathbf{Q} + k_{\mathrm{in}}^{\mathrm{crystal}}| = k_{\mathrm{scat}}.
\]

This is the intersection of:

- a sphere of radius `|G|` centered at the origin
- a sphere of radius `k_scat` centered at `A`

The circle center and radius are

\[
c = \frac{|\mathbf{G}|^2 + |\mathbf{A}|^2 - k_{\mathrm{scat}}^2}{2|\mathbf{A}|},
\qquad
\mathbf{O} = c\,\hat{\mathbf{A}},
\qquad
r_{\mathrm{circle}} = \sqrt{|\mathbf{G}|^2 - c^2}.
\]

The solver constructs orthonormal vectors `e1`, `e2` spanning the plane normal to `A_hat`, and parameterizes the full circle as

\[
\mathbf{Q}(\phi) = \mathbf{O} + r_{\mathrm{circle}}\left(\cos\phi\,e_1 + \sin\phi\,e_2\right).
\]

### Failure status codes

Negative `solve_q` status codes are exact kernel conditions:

- `-1`: `|G|` is too small
- `-2`: `|A|` is too small
- `-3`: the two spheres do not intersect, so `r_circle^2 < 0`
- `-4`: failure constructing `e1`
- `-5`: failure constructing `e2`

Status `0` means the circle construction succeeded, even if no detector hit is eventually deposited.

### Mosaic density on the circle

At each candidate `Q`,

\[
\theta_0 = \operatorname{atan2}(G_z,\sqrt{G_x^2+G_y^2}),
\qquad
\theta = \operatorname{atan2}(Q_z,\sqrt{Q_x^2+Q_y^2}),
\qquad
\Delta\theta = \mathrm{wrap\_to\_pi}(\theta - \theta_0).
\]

The pseudo-Voigt mosaic density is

\[
\omega(\Delta\theta)
= (1-\eta)\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\Delta\theta/\sigma)^2}
+ \eta\frac{1}{\pi\gamma}\frac{1}{1+(\Delta\theta/\gamma)^2}.
\]

The kernel then uses

\[
I_Q = \frac{\omega(\Delta\theta)}{2\pi |\mathbf{G}|^2}.
\]

This is the actual shell-density factor deposited downstream.

### Local arc restriction

The solver usually does not integrate the entire circle. It first estimates a useful polar-angle window

\[
\theta_{\mathrm{window}} = \max(10\sigma,\; 24\gamma,\; 5\times 10^{-4}),
\]

then caps it at `pi`.

If this window is at least `0.75*pi`, the solver uses the full circle. Otherwise it:

1. scans the circle to locate roots or near-roots of `theta(phi) - theta0`
2. expands local windows until `|theta(phi) - theta0|` crosses `theta_window`
3. merges overlapping windows
4. samples only those windows

This is a speed optimization, but it changes which parts of the circle are evaluated, so it belongs in the physics description.

### Uniform mode

In uniform mode, each local window is sampled at equally spaced midpoints. On the full circle the code uses

\[
\Delta\phi = \frac{2\pi}{N_{\mathrm{steps}}},
\qquad
ds = r_{\mathrm{circle}}\Delta\phi,
\qquad
I_{\mathrm{segment}} = I_Q ds.
\]

Only segments with intensity above `_INTENSITY_CUTOFF = exp(-100)` are retained.

### Adaptive mode

In adaptive mode, each interval `[phi_a, phi_b]` is assigned a Simpson mass and a trapezoid estimate:

\[
\mathrm{mass} = r_{\mathrm{circle}}\Delta\phi\frac{f_a + 4f_m + f_b}{6},
\qquad
\mathrm{trap} = r_{\mathrm{circle}}\Delta\phi\frac{f_a + f_b}{2},
\qquad
\mathrm{err} = |\mathrm{mass} - \mathrm{trap}|.
\]

Intervals are repeatedly split at the largest-error region until either:

- the interval budget `solve_q_steps` is exhausted, or
- the total error falls below

\[
\mathrm{abs\_tol} + \mathrm{solve\_q\_rel\_tol}\cdot |\mathrm{total\_mass}|,
\]

with `abs_tol = 1e-20`.

The retained output rows are `(Qx, Qy, Qz, mass_i)`.

## Outgoing ray tracing, detector projection, and deposition

Why this choice: After `solve_q(...)`, the important separation is between
direction and intensity. Detector geometry should come from the solved outgoing
direction, while interface optics should remain in transmission and attenuation
weights; using optics to bend the detector ray itself is what created the old
near-critical dead band. That is the best tradeoff for RA-SIM because hit
placement stays geometrically faithful while optics modes still change
brightness. The final bilinear deposition is also intentional: it turns a
continuous subpixel hit into a smooth image that can move continuously under
fitting instead of jumping from pixel to pixel.

The reflection kernel is [`_calculate_phi_from_precomputed`](../ra_sim/simulation/diffraction.py). It turns the precomputed entry terms plus one `G_vec` into detector hits.

### Nominal visibility screen

Before looping over all beam samples, the code calls [`_nominal_reflection_visible`](../ra_sim/simulation/diffraction.py). This is a real optimization, not just bookkeeping.

It uses the preferred central beam sample or the forced sample and builds the nominal outgoing wavevector by replacing `Q` with the nominal reciprocal vector `G_vec` itself:

\[
k_t' =
\begin{bmatrix}
G_x + k_{x,\mathrm{scat}} \\
G_y + k_{y,\mathrm{scat}} \\
G_z + \Re(k_z)
\end{bmatrix}.
\]

It converts that to an approximate outgoing angle, computes the detector `2theta` range spanned by the detector corners and center line, and rejects the reflection early if the nominal outgoing angle lies outside the padded range.

The exact padding used by the kernel is

\[
\sigma_{\mathrm{pad}} = 10\sigma,
\qquad
\gamma_{\mathrm{pad}} = 24\gamma,
\qquad
\mathrm{angle\_pad} = \max(\sigma_{\mathrm{pad}}, \gamma_{\mathrm{pad}}, 4\times 5\times 10^{-4}).
\]

The detector-space padding used for the final pixel-box test is

\[
\mathrm{pixel\_pad}
= d_{\mathrm{det}}\tan(\min(\mathrm{angle\_pad}, 0.45\pi))\frac{1}{p},
\]

clamped from below to `24` pixels.

Early return codes from this screen:

- if no valid samples exist, the caller returns sample status `-10`
- if the nominal reflection is not visible, the caller returns sample status `-11`

### Outgoing in-sample wavevector

For each retained `Q` row from `solve_q`,

\[
k_{tx}' = Q_x + k_{x,\mathrm{scat}},
\qquad
k_{ty}' = Q_y + k_{y,\mathrm{scat}},
\qquad
k_{tz}' = Q_z + \Re(k_z).
\]

Define

\[
k_r = \sqrt{{k_{tx}'}^2 + {k_{ty}'}^2}.
\]

The code stores the in-sample grazing-like exit angle as

\[
2\theta_t' =
\begin{cases}
0, & k_r < 10^{-12} \\
\operatorname{atan}(k_{tz}' / k_r), & \text{otherwise.}
\end{cases}
\]

### Exact exit optics

In `OPTICS_MODE_EXACT`,

\[
k_{z2,f} = \mathrm{kz\_branch\_decay}(\epsilon_2 k_0^2 - k_r^2),
\qquad
k_{z3,f} = \mathrm{kz\_branch\_decay}(k_0^2 - k_r^2).
\]

The exit transmission is the polarization-average exact Fresnel power

\[
T_f^2 = \frac{1}{2}\left(T_{f,s}^{\mathrm{power}} + T_{f,p}^{\mathrm{power}}\right),
\]

and

\[
\Im(k_{z,f}) = |\Im(k_{z2,f})|,
\qquad
L_{\mathrm{out}} =
\begin{cases}
\mathrm{thickness}, & \mathrm{thickness} > 0 \\
\frac{1}{2\,\Im(k_{z,f})}, & \text{otherwise.}
\end{cases}
\]

The exact branch also constructs a refracted-angle representation

\[
2\theta_t = \arccos\left(\mathrm{clamp}(k_r/k_0, -1, 1)\right)\operatorname{sign}(2\theta_t'),
\qquad
|k_f| = k_0.
\]

Those quantities are still available for diagnostics and for the legacy
`exit_projection_mode = "refracted"` path, but they are not the default
detector-projection inputs.

### Fast exit optics

In `OPTICS_MODE_FAST`, the code lazily builds one lookup table per beam sample. The table stores

- `Tf2`
- `im_k_z_f`
- `L_out`
- absolute exit angle

as a function of the absolute in-sample exit angle. The lookup uses a quadratic parameterization in angle index, then linearly interpolates the stored values.

The downstream formulas are

\[
2\theta_t = |2\theta_t|_{\mathrm{lut}}\operatorname{sign}(2\theta_t'),
\qquad
|k_f| = k_{\mathrm{scat}}.
\]

In older fast-mode detector projection, that remapped `2\theta_t` was used to
build the outgoing detector ray. The current default keeps the LUT-derived
`T_f^2`, `\Im(k_{z,f})`, and `L_{\mathrm{out}}` for intensity, but it does not
use the remapped angle to place the hit on the detector.

### Why the detector projection default changed

Older saved states could project the detector ray from the refracted-angle
reconstruction, especially in `OPTICS_MODE_FAST`. That path first solved the
outgoing in-sample direction

\[
k_f' = (k_{tx}', k_{ty}', k_{tz}'),
\qquad
k_r = \sqrt{{k_{tx}'}^2 + {k_{ty}'}^2},
\qquad
2\theta_t' = \operatorname{atan}(k_{tz}' / k_r),
\]

then remapped that already-solved direction to a refracted exit angle

\[
2\theta_t
= \arccos\left(\mathrm{clamp}(n_{\mathrm{real}}\cos 2\theta_t', -1, 1)\right)\operatorname{sign}(2\theta_t').
\]

For x-rays, `n_real < 1`. That means the remap enforces

\[
|2\theta_t| \ge \alpha_c,
\qquad
\alpha_c = \arccos(n_{\mathrm{real}}).
\]

So even when the solved outgoing direction had `2\theta_t' \approx 0`, the
projected detector ray was forced away from the sample plane by at least the
critical angle. That creates a forbidden angular strip in which no detector
hits can ever be placed. On the detector this appears as the horizontal empty
line that moves as the sample plane rotates with `\theta_i`.

The detector intersection code itself was not the problem. The problem was that
the old path intersected the wrong outgoing vector. The current default keeps
the optics factors for intensity but projects the detector geometry from the
solved internal direction.

### Propagation factor and intensity

The entry/exit attenuation factor is the product of the incoming and outgoing evanescent-decay exponentials:

\[
\mathrm{prop\_att}
= \exp(-2\,\Im(k_{z,\mathrm{in}})\,L_{\mathrm{in}})
\times
\exp(-2\,\Im(k_{z,\mathrm{out}})\,L_{\mathrm{out}}).
\]

Then

\[
\mathrm{prop\_fac} = T_i^2 T_f^2 \,\mathrm{prop\_att}.
\]

Any non-finite or non-positive `prop_att` or `prop_fac` is rejected.

The detector intensity associated with one `Q` sample is

\[
I_{\mathrm{hit}}
= I_{\mathrm{reflection}}
\cdot w_{\mathrm{sample}}
\cdot I_Q
\cdot \mathrm{prop\_fac}
\cdot \exp(-Q_z^2 d_x^2)
\cdot \exp(-(Q_x^2 + Q_y^2)d_y^2).
\]

This is the exact multiplicative structure of the implementation.

### Final outgoing ray and detector intersection

The outgoing in-sample azimuth is

\[
\phi_f = \operatorname{atan2}(k_{tx}', k_{ty}').
\]

By default, `exit_projection_mode = "internal"` and the detector direction is
the normalized solved outgoing vector

\[
\hat{k}_{f,\mathrm{proj}} =
\frac{1}{\sqrt{{k_{tx}'}^2 + {k_{ty}'}^2 + {k_{tz}'}^2}}
\begin{bmatrix}
k_{tx}' \\
k_{ty}' \\
k_{tz}'
\end{bmatrix}.
\]

If the legacy `exit_projection_mode = "refracted"` path is requested, the code
instead rebuilds the outgoing direction from the refracted-angle
representation:

\[
k_{f,\mathrm{proj}} =
\begin{bmatrix}
|k_f|\cos(2\theta_t)\sin\phi_f \\
|k_f|\cos(2\theta_t)\cos\phi_f \\
|k_f|\sin(2\theta_t)
\end{bmatrix}.
\]

The chosen projection vector is rotated back into the laboratory frame by

\[
k_f^{\mathrm{lab}} = R_{\mathrm{sample}}\,k_{f,\mathrm{proj}}.
\]

The detector intersection is then

\[
P_{\mathrm{det}} = \mathrm{intersect\_line\_plane}(I_{\mathrm{plane}}, k_f^{\mathrm{lab}}, D, n_{\mathrm{det,rot}}).
\]

Rejection conditions at this stage:

- no valid detector intersection from `intersect_line_plane`
- non-finite detector-plane coordinates
- non-finite pixel coordinates
- non-finite or non-positive deposited intensity
- bilinear deposit falling completely outside the image

Missed outgoing directions can be recorded into the miss table when auxiliary capture is enabled.

### Detector projection and bilinear deposition

The detector-plane coordinates are

\[
x_{\mathrm{det}} = (P_{\mathrm{det}} - D)\cdot e_{1,\mathrm{det}},
\qquad
y_{\mathrm{det}} = (P_{\mathrm{det}} - D)\cdot e_{2,\mathrm{det}}.
\]

These become floating detector pixels

\[
\mathrm{row}_f = \mathrm{center}_{\mathrm{row}} - y_{\mathrm{det}}/p,
\qquad
\mathrm{col}_f = \mathrm{center}_{\mathrm{col}} + x_{\mathrm{det}}/p.
\]

[`_accumulate_bilinear_hit`](../ra_sim/simulation/diffraction.py) and its cached variant split the value into the four neighboring pixels around

\[
(\lfloor \mathrm{row}_f \rfloor,\; \lfloor \mathrm{col}_f \rfloor)
\]

with the usual weights

\[
w_{00} = (1-d_r)(1-d_c),\;
w_{01} = (1-d_r)d_c,\;
w_{10} = d_r(1-d_c),\;
w_{11} = d_r d_c.
\]

This is why even a single geometric hit becomes a small four-pixel footprint in the raw image.

## Implementation details that materially affect behavior

These are not the core scattering equations, but they do change runtime, determinism, or what the fitting layer sees.

### Beam-state clustering

[`cluster_beam_profiles`](../ra_sim/simulation/mosaic_profiles.py) can compress many beam states into weighted representatives in the 5D feature space

\[
(\mathrm{beam\_x}, \mathrm{beam\_y}, \theta, \phi, \lambda).
\]

When clustering is used, the kernel simulates only the representatives and multiplies each deposited intensity by `sample_weight`.

### Reflection-source reuse

The diffraction driver groups sources by identical

\[
(G_r, G_z, \mathrm{forced\_sample\_idx}),
\]

simulates one source geometry, and scales or reuses the resulting hit tables, miss tables, and `Q` rows for every reflection that shares that geometry.

### Numba-safe fallback

[`process_peaks_parallel_safe`](../ra_sim/simulation/diffraction.py) is the live entry used by the engine. It can reuse a safe cache, cluster beam samples, run the Numba kernel, and fall back to a Python-safe path if JIT execution fails. So the public API is robust to environments where Numba compilation is not available.

### Thread-local image accumulation

The parallel kernel uses per-thread local image buffers and a local pixel cache to avoid excessive atomic-like contention. The cache is flushed to the image when it fills; if the cache still cannot accept a deposit, the code falls back to direct bilinear accumulation.

### Hit tables and subpixel peak centers

When `collect_hit_tables=True`, the kernel records selected nominal or best-candidate subpixel hits with columns

- intensity
- `col_f`
- `row_f`
- `phi_f`
- `H`
- `K`
- `L`

The fitting layer later reduces those rows with [`hit_tables_to_max_positions`](../ra_sim/simulation/diffraction.py), which can return up to two subpixel peak centers per reflection. Those two centers are what geometry fitting uses when one reflection splits into two visible detector maxima.

### Per-sample status codes used by fitting diagnostics

At the reflection-evaluation level the geometry fitter can see:

- `-10`: no valid sample or invalid precompute for that sample
- `-11`: nominal reflection not visible on the detector after the visibility screen
- `-12`: invalid or non-positive `sample_weight`
- `solve_q` negative codes: reciprocal-space construction failed for that sample

These codes matter when diagnosing missing fits or apparently disappearing reflections.

## Rods and stacking disorder

Rod simulations reuse the same detector physics kernel. The difference is the intensity model supplied before the detector trace.

[`analytical_ht_intensity_for_pair`](../ra_sim/utils/stacking_fault.py) implements the analytical Hendricks-Teller intensity for one `(h, k)` rod. Starting from a precomputed `F^2(L)` curve, it flips the user parameter as

\[
p_{\mathrm{flipped}} = 1 - p_{\mathrm{user}},
\]

forms

\[
z = (1-p_{\mathrm{flipped}}) + p_{\mathrm{flipped}} e^{i\delta},
\qquad
f = \min(|z|,\; 1 - P_{\mathrm{CLAMP}}),
\qquad
\psi = \arg(z),
\]

then uses

\[
\phi = \delta + \frac{2\pi L}{\phi_{L,\mathrm{divisor}}}
\]

inside either the infinite-layer correction

\[
R_{\infty}
= \frac{1-f^2}{1+f^2-2f\cos(\phi-\psi)}
\]

or the finite-layer form. The returned rod intensity is `AREA * F^2 * R`.

[`ht_dict_to_qr_dict`](../ra_sim/utils/stacking_fault.py) then groups rods by

\[
m = h^2 + hk + k^2
\]

so Qr-equivalent rods can be traced together by the same detector machinery.

## Geometry fitting from picked spots

Why this choice: The first fitting stage uses peak positions rather than full
image intensities because geometry errors move peaks more reliably than they
change detailed local brightness. That is the best tradeoff for RA-SIM because
it anchors detector and sample geometry before weaker nuisance effects such as
mosaic broadening, local background, and intensity scaling are allowed to move.
The residual vector is kept at fixed length, weighted by optional measurement
sigma models, and solved with priors, restarts, and staged release so the
optimizer remains stable even when picks are missing or uncertain. In practice
this makes point matching the geometry anchor and leaves image refinement for
later stages.

The live geometry fitter is built around manual point pairs assembled in the GUI, then optimized by [`fit_geometry_parameters`](../ra_sim/fitting/optimization.py).

### Manual picking behavior in the GUI

Manual picking is implemented in [`ra_sim/gui/_runtime/runtime_impl.py`](../ra_sim/gui/_runtime/runtime_impl.py).

The interaction is exactly:

1. click in the simulated overlay view
2. the GUI snaps that click to the nearest currently simulated peak by Euclidean distance using `_nearest_simulated_peak`
3. click in the measured background
4. the GUI searches a local square patch and snaps to the brightest display pixel using `_peak_maximum_near(..., search_radius=6)`
5. right-click or double-click finishes the collection and launches the fit

So the stored "real" point is not the literal mouse coordinate. It is the local brightness maximum near the click.

The saved pair contains:

- the HKL label
- the simulated display-frame peak
- the measured display-frame peak
- bookkeeping indices for overlay reconstruction

### Dataset preparation and orientation resolution

[`build_geometry_manual_fit_dataset`](../ra_sim/gui/geometry_fit.py) prepares one dataset for the optimizer.

For each selected background it:

1. loads the saved manual pairs
2. loads the native and display background images
3. applies the backend orientation used by the geometry-fit path
4. rebuilds simulated peaks for the current parameter set, including `theta_initial = theta_base + theta_offset`
5. maps any saved fixed source references through `(source_table_index, source_row_index)` into the current simulated peak list
6. unrotates the measured display picks back to the native frame
7. constructs paired simulated/native and measured/native coordinates
8. runs [`select_fit_orientation`](../ra_sim/gui/geometry_overlay.py)
9. applies the chosen orientation transform to both the measured peaks and the experimental image

The orientation search considers detector indexing mode, 90 degree rotations, and x/y flips. The default acceptance logic in [`select_fit_orientation`](../ra_sim/gui/geometry_overlay.py) is:

- orientation search enabled unless config disables it
- require at least one matched simulated/measured pair set
- require finite best RMS
- reject if best RMS exceeds the configured ceiling
- reject if the RMS improvement is less than `min_improvement_px`, default `0.25`

If those criteria fail, the fitter falls back to the identity transform.

### Reflection subsetting before each simulation

[`_prepare_reflection_subset`](../ra_sim/fitting/optimization.py) reduces the simulated reflection list before each fit evaluation. It keeps:

- reflections referenced by fixed saved source indices `(source_table_index, source_row_index)`
- reflections needed by HKL fallback matching

The fixed-source path matters because saved manual picks can outlive changes in reflection ordering. The HKL fallback matters because source-table references can become stale after clearing or re-adding picks.

### Point-match residual construction

The next equations define the geometric error for one matched peak pair. They
are intentionally simple: the fitter first wants to know whether simulated and
measured peak centers coincide before it worries about line shape or local
intensity structure.

The live residual is [`_evaluate_geometry_fit_dataset_point_matches`](../ra_sim/fitting/optimization.py).

For each dataset, the solver:

1. simulates only the reduced reflection subset
2. converts hit tables into up to two subpixel simulated centers per reflection with `hit_tables_to_max_positions`
3. resolves fixed-source matches first
4. groups remaining measured points by HKL
5. matches fallback points against the simulated centers for that HKL
6. writes one fixed two-component residual slot `(dx, dy)` per measured point

The fixed residual length is important. The code explicitly keeps the residual vector shape constant because SciPy finite-difference Jacobians become unstable if points appear and disappear between evaluations.

For one matched point,

\[
r = \begin{bmatrix}\Delta x \\ \Delta y\end{bmatrix}
=
\begin{bmatrix}
x_{\mathrm{sim}} - x_{\mathrm{meas}} \\
y_{\mathrm{sim}} - y_{\mathrm{meas}}
\end{bmatrix}.
\]

If `weighted_matching` is enabled, the residual first gets the distance downweight

\[
w_{\mathrm{dist}} = \frac{1}{\sqrt{1 + (d/f_{\mathrm{scale}})^2}},
\]

where `d = sqrt(dx^2 + dy^2)` and `f_scale = solver_f_scale`.

By default in [`config/instrument.yaml`](../config/instrument.yaml), point matching uses:

- `loss = linear`
- `f_scale_px = 8.0`
- `max_nfev = 140`
- `restarts = 4`
- `restart_jitter = 0.15`
- `missing_pair_penalty_px = 20.0`
- `weighted_matching = false`
- `use_measurement_uncertainty = true`
- `anisotropic_measurement_uncertainty = false`

### Isotropic and anisotropic measurement-uncertainty weighting

[`_weight_measurement_residual`](../ra_sim/fitting/optimization.py) handles the measurement sigma model.

If uncertainty weighting is disabled, the residual is just

\[
r_{\mathrm{weighted}} = w_{\mathrm{dist}} r.
\]

If isotropic weighting is enabled, the code divides by `sigma_px`:

\[
r_{\mathrm{weighted}} = \frac{w_{\mathrm{dist}}}{\sigma_{\mathrm{px}}} r.
\]

If anisotropic weighting is enabled and radial/tangential information is available, the code builds the detector-local radial/tangential basis

\[
\hat{u}_r = \frac{(x_{\mathrm{meas}} - x_c,\; y_{\mathrm{meas}} - y_c)}{\|(x_{\mathrm{meas}} - x_c,\; y_{\mathrm{meas}} - y_c)\|},
\qquad
\hat{u}_t = (-u_{r,y},\; u_{r,x}),
\]

stacks them into a basis matrix `B = [u_r, u_t]`, and applies

\[
\Sigma^{-1/2} = B
\begin{bmatrix}
1/\sigma_r & 0 \\
0 & 1/\sigma_t
\end{bmatrix}
B^T.
\]

The residual becomes

\[
r_{\mathrm{weighted}} = w_{\mathrm{dist}} \Sigma^{-1/2} r.
\]

The diagnostics also report the radial and tangential components separately.

When only one scalar sigma is provided but anisotropic mode is requested, the code synthesizes

\[
\sigma_r = \sigma_{\mathrm{px}}\cdot \mathrm{radial\_sigma\_scale},
\qquad
\sigma_t = \sigma_{\mathrm{px}}\cdot \mathrm{tangential\_sigma\_scale}.
\]

### Missing-pair penalties

If a measured point cannot be paired to a simulated point, it still remains in the residual vector. The penalty inserted is

\[
\begin{bmatrix}
\mathrm{missing\_pair\_penalty}\cdot \sigma_{\mathrm{weight}} \\
0
\end{bmatrix},
\]

with

\[
\sigma_{\mathrm{weight}}
= \frac{1}{\sqrt{\sigma_r \sigma_t}}
\]

for the sigma-equivalent used by the implementation.

This is why unmatched points still influence the optimizer instead of silently disappearing.

### Matching order and fallback logic

The matching logic is intentionally not a single nearest-neighbor pass.

1. resolve fixed source matches using the saved source-table references
2. group unresolved measured points by HKL
3. collect up to two simulated centers per HKL from the hit tables
4. greedily pair measured and simulated points within `pixel_tol`
5. assign missing-pair penalties to any remaining measured points
6. assign a final unresolved penalty slot to any measured entries that somehow escaped both paths

If point-match inputs are absent entirely, the fitter falls back to the older [`simulate_and_compare_hkl`](../ra_sim/fitting/optimization.py) angle-based path that compares radial and azimuthal angular errors in degrees.

### Solver, priors, restarts, staged release, and identifiability

The main nonlinear solve uses `scipy.optimize.least_squares`.

Optional Gaussian priors are appended as extra residuals:

\[
r_{\mathrm{prior},i} = \frac{x_i - \mu_i}{\sigma_i}.
\]

Around that local solve, the fitter can apply:

- restart seeds from corners, center, axis probes, quasi-random seeds, and local jitter
- staged release of parameter blocks
- pair reparameterization
- stagnation probes along coordinate and pairwise directions
- adaptive regularization based on identifiability
- optional ridge refinement
- optional image refinement
- auto-freeze and selective thaw based on identifiability warnings

Current live behavior from the code:

- `use_single_ray = False`
- `single_ray_polish_enabled = False`

So the single-ray polish path exists but is inactive.

[`_build_identifiability_summary`](../ra_sim/fitting/optimization.py) finite-differences the final residual, with probe step

\[
\Delta x_i = \max(\mathrm{fd\_min\_step},\; \mathrm{fd\_step\_fraction}\cdot \mathrm{probe\_scale}_i),
\]

then clipped to each parameter span. It reports:

- SVD-based numerical rank
- condition number
- covariance approximation `pinv(J^T J) * residual_variance`
- strong parameter correlations
- weak Jacobian columns
- per-parameter top-contributing peaks

The GUI then overwrites `result.rms_px` with the final unweighted point-position RMS when point matching is active. So the user-facing RMS is not the raw weighted least-squares norm.

## Automatic background peak matching

Why this choice: Automatic matching is treated as a detection-and-assignment
problem, not as a single nearest-neighbor shortcut. That is the best tradeoff
for RA-SIM because real detector backgrounds can contain clutter, split maxima,
missing peaks, and local artifacts that would make naive nearest matching
fragile. The code therefore separates summit detection from scored one-to-one
assignment and post-match clipping so the optimizer receives a cleaner
correspondence set. This stage sits between forward simulation and later
refinement because match quality matters more here than full image completeness.

Automatic matching is a different workflow from manual geometry fitting. It is implemented in [`build_background_peak_context`](../ra_sim/fitting/background_peak_matching.py) and [`match_simulated_peaks_to_peak_context`](../ra_sim/fitting/background_peak_matching.py).

Important live defaults from [`config/instrument.yaml`](../config/instrument.yaml) and [`ra_sim/fitting/background_peak_matching.py`](../ra_sim/fitting/background_peak_matching.py):

- `search_radius_px = 24.0`
- `local_max_size_px = 5`
- `smooth_sigma_px = 3.0`
- `min_prominence_sigma = 2.0`
- `min_match_prominence_sigma = 2.5`
- `k_neighbors = 12`
- `distance_sigma_clip = 3.5`
- `ambiguity_ratio_min = 1.15`
- `ambiguity_margin_px = 2.0`
- `max_candidate_peaks = 1200`
- `walk_max_steps = 24`
- `walk_step_min_gain_sigma = 0.0`
- `require_candidate_ownership = True`
- `walk_match_score_bonus = 1.0`
- `ascent_score_weight = 0.15`

### Background summit detection

The reusable peak context is built in this exact sequence:

1. convert the background to a `float32` image
2. replace invalid pixels with the median valid intensity
3. compute a fine image with Gaussian blur `climb_sigma_px` and a broader image with Gaussian blur `smooth_sigma_px`
4. define

\[
\mathrm{peakness} = \mathrm{fine} - \mathrm{broad}
\]

5. find local maxima by comparing `peakness` to a max-filter window
6. estimate robust peakness scale with

\[
\sigma_{\mathrm{est}} = 1.4826 \cdot \mathrm{MAD}
\]

falling back to standard deviation if needed
7. threshold candidates at

\[
\mathrm{candidate\_floor}
= \mathrm{median}(\mathrm{peakness})
+ \mathrm{min\_prominence\_sigma}\cdot \sigma_{\mathrm{est}}
\]

8. if no candidates survive, fall back to a high percentile threshold
9. label connected local-max regions
10. choose one representative summit per labeled region
11. refine the summit center with [`_refine_peak_center`](../ra_sim/fitting/background_peak_matching.py)

The summit-center refinement first computes a weighted centroid on a `3 x 3` patch, then attempts a quadratic surface fit. If the fitted Hessian corresponds to a valid local maximum and the subpixel offset stays within `1.25` pixels in each direction, the quadratic maximum is accepted.

### Seed-to-summit candidate generation

For each simulated seed peak, the matcher:

1. filters summit candidates by `min_match_prominence_sigma`
2. limits the candidate pool to `max_candidate_peaks`
3. uses a KD-tree to find nearby summit centers inside `search_radius_px`
4. optionally assigns summit ownership to the nearest seed
5. rejects ambiguous candidates when a competing seed is too close, controlled by `ambiguity_ratio_min` and `ambiguity_margin_px`
6. runs an uphill walk on the `peakness` image from the simulated seed location via [`_walk_seed_to_summit`](../ra_sim/fitting/background_peak_matching.py)

The uphill walk moves to the best neighbor in a `3 x 3` patch until either:

- no higher neighbor exists
- the gain is below `step_min_gain_sigma * sigma_est`
- `walk_max_steps` is reached

### Assignment score and one-to-one solve

For one seed-summit candidate edge, the score is

\[
\mathrm{score}
= w_d \max\left(0, 1 - \frac{d}{r_{\mathrm{match}}}\right)
+ w_p \max(0, \mathrm{prominence}_\sigma)
+ w_s \max(0, \mathrm{seed\_weight})
+ w_a \max(0, \mathrm{net\_ascent}_\sigma)
+ \mathrm{walk\_bonus}\cdot \mathbf{1}_{\mathrm{walk\_summit\_match}}.
\]

The implementation defaults are:

- `distance_score_weight = 2.5`
- `prominence_score_weight = 0.05`
- `seed_weight_score_scale = 1e-12`
- `ascent_score_weight = 0.15`
- `walk_match_score_bonus = 1.0`

Each candidate also gets a quick confidence estimate

\[
\mathrm{confidence}
= \frac{\max(0,\mathrm{prominence}_\sigma)}{1 + \max(0,d_{\mathrm{px}})}.
\]

The score matrix is converted into a cost matrix and solved with the Hungarian algorithm via `linear_sum_assignment`, so the final matching is globally one-to-one.

### Post-match sigma clipping

After Hungarian assignment, the code sigma-clips the match distances using the robust rule

\[
\mathrm{clip\_limit}
= \min\left(
\mathrm{search\_radius},
\mathrm{median}(d) + \mathrm{distance\_sigma\_clip}\cdot \sigma_d
\right),
\]

with robust `sigma_d` from the MAD where possible. Matches outside that limit are removed.

So the automatic workflow is:

- image preprocessing
- summit finding
- ownership-constrained candidate generation
- uphill-walk evidence
- scored one-to-one assignment
- robust distance clipping

not plain nearest-neighbor snapping.

## Mosaic-shape fitting, legacy mosaic-width fitting, and image-space refinement

Why this choice: Once geometry is anchored, the remaining mismatch is mostly
about profile width, asymmetry, relative intensity, and local background, so
the objective changes accordingly. That is the best tradeoff for RA-SIM because
reopening the full geometry problem here would let weakly constrained nuisance
terms absorb shape errors. The separable ROI formulation analytically solves
local amplitude and baseline terms so the nonlinear optimizer can focus on a
small set of physically meaningful width parameters. The broader image-refining
stages then add robust losses and acceptance guards so bad pixels, outliers,
and background drift do not dominate the fit.

### Geometry-fit-cached detector-shape fit

The GUI action `Fit Mosaic Shapes` calls
[`fit_mosaic_shape_parameters`](../ra_sim/fitting/optimization.py). This
workflow is geometry-locked: it only runs after a successful manual geometry
fit and it reuses that run's cached background bundle, including the exact
selected backgrounds, per-background `theta_initial`, measured peak anchors,
and oriented detector images.

The optimizer refines:

- `sigma_mosaic_deg`
- `gamma_mosaic_deg`
- `eta`
- `theta_initial` for a single cached background, a shared `theta_offset`, or
  per-background `theta_i` values depending on the current background-selection
  mode

The objective is also different from the legacy separable fitter. For each
cached measured peak it first attempts to build one fixed angular-profile ROI.
The inclusion rule is:

- every selected cached peak with `(h, k) = (0, 0)` is treated as specular
- every other selected cached peak with `(h, k) != (0, 0)` is treated as
  off-specular

The family assignment controls which residual block that peak contributes to;
it is not meant to exclude valid selected rods just because `h != k`.

For specular `(00l)` peaks the fitter:

- compares the measured and simulated `2theta` line shape directly
- solves one shared per-dataset scale factor for the whole specular bundle
- optionally adds a relative-intensity term across the selected specular peaks

For every selected off-specular peak with `(h, k) != (0, 0)` the fitter:

- extracts a `phi` profile at the measured anchor
- compares normalized line shape only
- does not add an inter-peak relative-intensity term

A selected peak can still be dropped before optimization if ROI preparation
fails, for example because the measured anchor is missing, the ROI is
degenerate, the local angular transform is invalid, there are too few support
pixels in the profile window, or the measured profile is empty after background
subtraction.

Each dataset block is scaled by `1 / sqrt(num_rois_in_dataset)` so one
selected background cannot dominate just because it contributed more peaks.

The GUI step refuses to run if the geometry-fit cache is missing or stale. Any
change to manual picks, selected backgrounds, shared-theta mode, or stored
background theta values requires rerunning geometry fit first.

### Legacy separable affine ROI fit

[`fit_mosaic_widths_separable`](../ra_sim/fitting/optimization.py) keeps geometry fixed and fits only the pseudo-Voigt width parameters:

- `sigma_mosaic_deg`
- `gamma_mosaic_deg`
- `eta`

### Reflection selection

The fitter first prunes the reflection list:

- require `(2h + k) % 3 == 0`
- skip `(0, 0, 0)`
- drop non-`(00l)` reflections above `65 deg` in `2theta`

Candidate peaks can come from `geometry`, `auto`, or `hybrid` sourcing, and optional stratification can enforce diversity across `L` or `2theta`.

### Separable affine ROI fit

The next equations solve the local nuisance terms exactly inside each ROI. That
lets the nonlinear optimizer spend its effort on mosaic-width parameters instead
of repeatedly relearning a trivial local scale and offset.

Inside each selected ROI, the simulated template `t` is not compared directly to the measured vector `y`. The code solves the local affine correction

\[
\mathrm{model} = a t + b
\]

analytically at every nonlinear iteration.

With

\[
tt = t\cdot t,\;
to = \sum t,\;
ty = t\cdot y,\;
oy = \sum y,\;
oo = N,\;
\det = tt\,oo - to^2,
\]

the best affine coefficients are

\[
a = \frac{ty\,oo - oy\,to}{\det},
\qquad
b = \frac{tt\,oy - to\,ty}{\det}.
\]

If the template is empty, too weak, or the affine system is ill-conditioned, the code falls back to `observed - mean(observed)`.

This is why the nonlinear fit only has to solve for mosaic widths while local amplitude and baseline are handled exactly.

### Nonlinear solve and restart acceptance

The mosaic parameters are optimized with `least_squares` under the default bounds:

- `sigma in [0.03, 3.0] deg`
- `gamma in [0.03, 3.0] deg`
- `eta in [0.0, 1.0]`

The fitter can then launch jittered restarts around the best solution. A restart outcome is accepted only when the current code's `acceptance_passed` guard is satisfied:

- cost reduction at least `20%`
- outlier fraction at most `25%`
- no final parameter on a bound

### Tube ROIs and sensitivity-guided image refinement

For broader image refinement, [`iterative_refinement`](../ra_sim/fitting/optimization.py) uses ridge-following ROIs rather than square spots.

[`build_tube_rois`](../ra_sim/fitting/optimization.py) estimates the tube width approximately as

\[
\mathrm{width}_{px}
\approx
\mathrm{mosaic\_fwhm}\frac{D}{p}
+
\mathrm{divergence}\frac{D}{p}
+
\mathrm{bandwidth}\frac{D}{p},
\]

where `D` is the detector distance and `p` is the pixel pitch.

Within these ROIs, the code builds sensitivity weights from finite-difference image derivatives so that pixels where the image responds strongly to parameter changes carry more leverage. In shorthand,

\[
\mathrm{importance} \propto \frac{|\partial I / \partial x|^2}{I_{\mathrm{base}}}
\]

captures the intent, though the implementation uses finite-difference image perturbations rather than a closed-form derivative.

### Quadratic local background model

Inside each ROI, the local background is modeled as

\[
b(x,y) = c_0 + c_1 x + c_2 y + c_3 x^2 + c_4 xy + c_5 y^2.
\]

That removes slow local intensity drift so the optimizer spends its effort on reflection shape and position rather than on background offsets.

### Robust image residual composition

[`robust_residuals`](../ra_sim/fitting/optimization.py) mixes several robustification layers:

- Poisson-deviance style residuals for count-like behavior
- Anscombe-space residuals for variance stabilization
- Huber clipping
- an outlier mixture that suppresses obviously bad pixels

The ROI/image stage is accepted only if it improves the image-space objective without materially degrading the point-match fit, matched count, or point RMS beyond the configured thresholds.

## hBN calibrant fitting

Why this choice: The hBN calibrant path is separate because rings constrain
detector geometry through global shape distortion rather than through sparse
one-to-one peak correspondences. That is the best tradeoff for RA-SIM because
ellipse geometry and projective tilt correction extract detector center, tilt,
and distance information very efficiently from ring data. The fitter therefore
combines robust point handling, optional snap refinement, and projective
circularization instead of reusing the Bragg-spot residual. In implementation
terms, the final tilt solve minimizes ring circularity after projective
reprojection, which is the geometry signal the calibrant actually provides.

The hBN fitter in [`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py) is a separate calibration workflow. It is built around manually selected ring points, optional snap-to-ring refinement, robust ellipse fitting, and projective tilt correction.

Important live defaults from [`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py):

- `DEFAULT_REFINE_ITERS = 5`
- `DEFAULT_CLICK_SEARCH_ALONG = 12.0`
- `DEFAULT_CLICK_SEARCH_ACROSS = 3.0`
- `DEFAULT_CLICK_SEARCH_STEP = 0.5`
- `DEFAULT_PV_MIN_SAMPLES = 7`
- `DEFAULT_PRECISION_PICK_SIZE_DS = 40.0`
- `DEFAULT_CENTER_DRIFT_LIMIT_PX = 35.0`
- `DEFAULT_CENTER_PRIOR_SIGMA_PX = 12.0`

### Manual click coordinates and quantization floor

The GUI usually works on a downsampled display image. A click therefore has an irreducible quantization uncertainty of about one downsampled pixel cell, modeled in the confidence code as

\[
\sigma_{\mathrm{click}} = \frac{\mathrm{downsample}}{\sqrt{12}}.
\]

This uncertainty is not the full fitting sigma, but it becomes a hard floor in the confidence score and explains why very coarse downsample factors are penalized.

### Ellipse residual used everywhere

The next formula is the basic geometric error for one candidate ring point. It
measures how far the point sits from the current ellipse in a scale-aware
pixel-like way, which is why the same quantity can be reused in robust fitting,
reporting, and confidence scoring.

For ellipse parameters `(xc, yc, a, b, theta)`, a point is rotated into ellipse coordinates `(u, v)` and evaluated with

\[
q = \sqrt{\frac{u^2}{a^2} + \frac{v^2}{b^2} + 10^{-12}},
\qquad
r_{\mathrm{ellipse}} = |q - 1| \cdot \frac{a+b}{2}.
\]

This residual is in pixels and is the core quantity used by robust ellipse fitting and reporting.

### Robust seed fitting

[`robust_fit_ellipse`](../ra_sim/hbn_fitter/fitter.py) is the first ellipse fit.

It:

1. requires at least `5` points
2. runs RANSAC with `EllipseModel`, `min_samples = 5`
3. uses a residual threshold in pixels
4. performs two sigma-clipping passes after RANSAC

The robust scale estimate is

\[
\sigma = \max(1.4826 \cdot \mathrm{MAD}, 10^{-6}),
\]

and the clipping threshold is

\[
\mathrm{thr} = \max(\mathrm{residual\_threshold\_px}, 3\sigma).
\]

If RANSAC fails entirely, the code falls back to a direct ellipse estimate.

### Weighted Powell refinement

[`weighted_refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) refines the seed with a Powell optimizer.

Per-point weights are

\[
w_i = \frac{1}{\sigma_i^2},
\]

then normalized by their median.

The robust loss is based on

\[
u_i = \frac{r_i}{\delta},
\qquad
\rho(u_i) = (\sqrt{1 + u_i^2} - 1)\delta^2,
\]

with `delta` taken from the median point sigma and clipped into a reasonable range.

The objective also adds weak regularization toward the robust seed:

- small center drift penalty
- small axis-change penalty in log-axis space

So the Powell stage is intentionally local, not a free global refit.

### Snap-to-ring search for clicked spots

[`snap_points_to_ring`](../ra_sim/hbn_fitter/fitter.py) refines manual ring points by scanning around each click.

For a click `(px, py)` and current ellipse parameters, the code builds:

- a radial direction from the ellipse center to the click
- a tangential direction perpendicular to that radial direction

It then searches tangential offsets `v` and, for each one, samples a radial profile over offsets `u`.

Along each radial profile it estimates the local peak either by:

- pseudo-Voigt profile fitting, or
- fast quadratic refinement fallback

and then applies a small 2D subpixel centroid correction at the candidate peak.

Each candidate receives the posterior-like score

\[
\log p
= w_{\mathrm{peak}}\cdot \mathrm{peak\_snr}
- \frac{1}{2}w_{\mathrm{resid}}\left(\frac{r_{\mathrm{ellipse}}}{\sigma_{\mathrm{resid}}}\right)^2
- \frac{1}{2}w_{\mathrm{across}}\left(\frac{v}{\sigma_v}\right)^2
- \frac{1}{2}w_{\mathrm{click}}\left(\frac{d_{\mathrm{click}}}{\sigma_d}\right)^2.
\]

The best candidate wins, but optional confidence gates can reject the snap if:

- the posterior is too low
- the posterior margin to the second-best candidate is too small
- the peak SNR is too weak
- the click-to-snap distance or ellipse residual is too large

If snapping is rejected, the fitter keeps the manual point.

### Snap uncertainty estimation

The GUI also estimates a pointwise snap uncertainty. Around the original click it launches nearby seed snaps on concentric radii and multiple angles, then combines them with weights derived from the candidate posteriors and a radial falloff.

The weighted mean `(mu_x, mu_y)` is used to compute:

- scatter of the nearby snapped seeds around the weighted mean
- bias between the weighted mean and the accepted snap

The final uncertainty is

\[
\sigma_{\mathrm{snap}} = \sqrt{\mathrm{scatter}^2 + \mathrm{bias}^2},
\]

then clipped into `[0.20,\; 12.0]` pixels by default.

If this estimate fails, the GUI falls back to a heuristic based on the best residual and click distance.

### Iterative ellipse refinement from radial profiles

[`refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) repeatedly improves an ellipse by sampling radial profiles around the current ellipse.

For each angular parameter `t`, it constructs the current ellipse point

\[
(x_e, y_e) =
\left(
x_c + a\cos t \cos\theta - b\sin t \sin\theta,
\;
y_c + a\cos t \sin\theta + b\sin t \cos\theta
\right),
\]

then scans along the local radial direction through offsets `s in [-dr, dr]`.

At each angle the code picks the best radial peak, converts that to a detector point, and accumulates the resulting peak cloud. It can also augment this cloud with snapped seed points from the manually picked anchors.

That peak cloud is then passed back through:

1. robust ellipse fitting
2. weighted Powell refinement

The loop stops early when the center shift and axis shift are both below `0.05` pixels in norm.

### Confidence scoring

[`compute_fit_confidence`](../ra_sim/hbn_fitter/fitter.py) scores each fitted ring using:

- ellipse residual
- ring SNR
- angular coverage
- number of points
- downsample penalty from click quantization

The angular coverage is based on the largest gap in ellipse-normalized point angles:

\[
\mathrm{coverage} = 1 - \frac{\mathrm{largest\_gap}}{2\pi}.
\]

The residual score uses the downsample-aware effective residual

\[
r_{\mathrm{eff}} = \sqrt{r_{\mathrm{ellipse}}^2 + \sigma_{\mathrm{click}}^2}.
\]

The code then combines the terms as

\[
\mathrm{score}_{\mathrm{core}}
= 0.45\,\mathrm{residual\_score}
+ 0.35\,\mathrm{snr\_score}
+ 0.15\,\mathrm{coverage\_score}
+ 0.05\,\mathrm{points\_score},
\]

and multiplies by the downsample score

\[
\mathrm{downsample\_score} = \frac{1}{1 + (\sigma_{\mathrm{click}}/1.5)^2}.
\]

The final reported confidence is `100 * clip(score, 0, 1)`.

### Projective detector-tilt optimization

The tilt-correction model is [`apply_tilt_projective`](../ra_sim/hbn_fitter/fitter.py). It embeds detector-plane points into 3D:

1. build a rotation from `tilt_x_deg`, `tilt_y_deg`
2. rotate the detector basis vectors
3. place the detector center at `(0, 0, distance_px)`
4. express each 2D point as a 3D point on the tilted detector plane
5. reproject back to the nominal flat detector with scale

\[
s = \frac{d}{z}.
\]

The optimization objective [`circularity_cost_projective`](../ra_sim/hbn_fitter/fitter.py) computes the corrected radius for each ring and minimizes the weighted mean squared robust fractional spread

\[
\left(\frac{\sigma_{\mathrm{robust}}(r)}{\mathrm{median}(r)}\right)^2.
\]

[`optimize_tilts_projective`](../ra_sim/hbn_fitter/fitter.py) solves this in stages:

1. estimate an initial detector distance from the median ring radius
2. run a coarse tilt grid
3. refine `(tilt_x, tilt_y, log distance)` with Powell
4. optionally refine `(center_x, center_y, tilt_x, tilt_y, log distance)` with Powell

Important implementation constants are:

- `OPT_GRID_SIZE_COLD = 31`
- `OPT_GRID_SPAN_COLD = 10 deg`
- `OPT_GRID_SIZE_WARM = 17`
- `OPT_GRID_SPAN_WARM = 4 deg`
- `PROJECTIVE_TILT_MAX_DEG = 25`
- `PROJECTIVE_DIST_MIN_PX = 120`
- `PROJECTIVE_DIST_MAX_MULT = 20`
- `PROJECTIVE_DIST_PRIOR_SIGMA_LOG = 0.9`
- `DEFAULT_CENTER_PRIOR_SIGMA_PX = 12`
- `DEFAULT_CENTER_DRIFT_LIMIT_PX = 35`

So the hBN fitter is not just "fit an ellipse". It is:

- manual or snapped point selection
- per-point uncertainty estimation
- robust seed fitting
- iterative profile-based ellipse refinement
- confidence scoring
- projective circularization of the corrected rings

## Logging, debug, and cache controls

RA-SIM now uses `config/debug.yaml` as the primary control surface for user-facing debug/logging output and optional cache retention.

The main kill switch is:

```yaml
debug:
  global:
    disable_all: true
```

When `debug.global.disable_all` is `true`, every debug/log output path documented here is disabled, regardless of the other entries in `debug.yaml`.

Legacy environment variables still work as compatibility overrides. They are no longer the primary interface.

### Primary config

The repo default is `config/debug.yaml`:

```yaml
debug:
  global:
    disable_all: false
  console:
    enabled: false
  runtime_update_trace:
    enabled: true
  geometry_fit:
    log_files: true
    extra_sections: true
  mosaic_fit:
    log_files: true
  projection_debug:
    enabled: true
  diffraction_debug_csv:
    enabled: true
  intersection_cache:
    enabled: true
    log_dir: null
  cache:
    default_retention: auto
    families:
      primary_contribution: auto
      source_snapshots: auto
      caking: auto
      peak_overlay: auto
      background_history: auto
      manual_pick: auto
      geometry_fit_dataset: auto
      qr_cylinder_overlay: auto
      diffraction_safe: auto
      diffraction_last_intersection: never
      fit_simulation: auto
      stacking_fault_base: auto
```

Meaning of each key:

1. `debug.global.disable_all`
   Global kill switch for all user-facing debug/log output covered by this document.

2. `debug.console.enabled`
   Enables console debug printing and Numba logging.

3. `debug.runtime_update_trace.enabled`
   Enables the GUI runtime trace log.

4. `debug.geometry_fit.log_files`
   Enables geometry-fit log file creation.

5. `debug.geometry_fit.extra_sections`
   Enables the more verbose geometry-fit diagnostic sections inside those logs.

6. `debug.mosaic_fit.log_files`
   Enables mosaic-shape fit log file creation in both GUI and CLI paths.

7. `debug.projection_debug.enabled`
   Enables projection-debug JSON logging.

8. `debug.diffraction_debug_csv.enabled`
   Enables the explicit diffraction debug CSV dump written by `dump_debug_log()`.

9. `debug.intersection_cache.enabled`
   Enables intersection-cache dump folders.

10. `debug.intersection_cache.log_dir`
    Optional root directory for intersection-cache dumps. `null` means use `debug_log_dir`.

11. `debug.cache.default_retention`
    Default policy for optional retained caches. Valid values are `never`, `auto`, and `always`.

12. `debug.cache.families.<name>`
    Per-cache-family override for optional retained caches. Valid values are `never`, `auto`, and `always`.

### Cache policy

The cache section controls only optional retained caches. It does not control active simulation state.

Three categories matter:

1. Mandatory current state
   Current simulation images, current peak tables, current active intersection caches, current integration payloads, the active background image, and the active beam/profile bundle. These are required for the current UI/runtime state and are not gated by the cache policy.

2. Optional retained caches
   Recomputable data kept only for reuse or debug convenience. These are controlled by `debug.cache`.

3. Per-call scratch buffers
   Temporary hot-loop work arrays. These are not retained caches and are not controlled here.

Current optional cache families:

1. `primary_contribution`
   Per-contribution primary hit-table cache used by incremental SF-prune reuse.

2. `source_snapshots`
   Stored source-row snapshots used by manual-geometry/source-row reuse flows.

3. `caking`
   Retained caked-analysis payloads reused across repeated analysis refreshes.

4. `peak_overlay`
   Reusable peak-overlay records and click-index payloads.

5. `background_history`
   Inactive background-image history. The currently selected background image remains mandatory.

6. `manual_pick`
   Geometry manual-pick candidate/match cache.

7. `geometry_fit_dataset`
   Cached geometry-fit dataset bundle for follow-on geometry-fit workflows.

8. `qr_cylinder_overlay`
   Cached analytic Qr-cylinder overlay paths.

9. `diffraction_safe`
   Retained diffraction safe-cache internals such as Q-vector reuse state.

10. `diffraction_last_intersection`
    Retained global last-intersection snapshot. Default is `never`.

11. `fit_simulation`
    Reusable fitting simulation/image caches.

12. `stacking_fault_base`
    Retained HT base-curve cache in stacking-fault generation.

Retention modes:

1. `never`
   Build on demand if needed for the current action, then discard.

2. `auto`
   Retain only when the active feature benefits from reuse. This is the default balanced mode.

3. `always`
   Retain whenever built.

Important detail:

- `debug.global.disable_all` disables logging/debug output only.
- `debug.global.disable_all` does not disable optional caches.
- Tiny infrastructure/compile caches such as config bundle loading, CIF parsing, and compiled expression helpers stay always-on and are not controlled by `debug.cache`.

### Resolution order

Debug control resolution follows this order:

1. Global disable is active if either `debug.global.disable_all` is `true` or `RA_SIM_DISABLE_ALL_LOGGING` / `RA_SIM_DISABLE_LOGGING` is truthy.
2. If global disable is active, all subsystem debug/log outputs are disabled.
3. Otherwise, existing environment variables override the matching `debug.yaml` entry.
4. Otherwise, `debug.yaml` provides the value.
5. For `debug.geometry_fit.extra_sections` only, if that key is absent, the code falls back to the legacy instrument config keys `instrument.fit.geometry.debug_logging` and then `instrument.fit.geometry.debug_mode`.

Important detail:

- `RA_SIM_DEBUG=1` does not bypass the global kill switch.
- Some debug keys are config-only because there was no legacy env var for them.
- Cache retention has no environment-variable overrides in v1.

### Compatibility environment variables

These env vars are still honored:

1. `RA_SIM_DISABLE_ALL_LOGGING=0/1`
   Compatibility override for the global kill switch.

2. `RA_SIM_DISABLE_LOGGING=0/1`
   Legacy alias for the same global kill switch.

3. `RA_SIM_DEBUG=0/1`
   Compatibility override for `debug.console.enabled`.

4. `RA_SIM_DISABLE_PROJECTION_DEBUG=0/1`
   Negative compatibility override for `debug.projection_debug.enabled`.
   `1` disables projection-debug logging.

5. `RA_SIM_LOG_INTERSECTION_CACHE=0/1`
   Compatibility override for `debug.intersection_cache.enabled`.

6. `RA_SIM_INTERSECTION_CACHE_LOG_DIR=/path/to/dir`
   Compatibility override for `debug.intersection_cache.log_dir`.

Legacy geometry-fit compatibility:

1. `instrument.fit.geometry.debug_logging`
   Fallback for `debug.geometry_fit.extra_sections` when the new key is absent.

2. `instrument.fit.geometry.debug_mode`
   Older fallback alias used only if `debug_logging` is absent.

### What is covered by the global kill switch

`debug.global.disable_all: true` disables all of these:

1. Console debug output and Numba logging.
2. GUI runtime update trace logging.
3. Geometry-fit log file creation.
4. Geometry-fit verbose diagnostic sections.
5. Mosaic-shape fit log file creation in GUI and CLI flows.
6. Projection-debug JSON output.
7. Diffraction debug CSV output from `dump_debug_log()`.
8. Intersection-cache dump folders.

This includes the older direct geometry-fit and mosaic-fit writers in the GUI runtime and CLI paths. They are now routed through the centralized resolver.

### Output files and directories

Default output locations still come from `config/dir_paths.yaml`.

Relevant directory keys:

1. `downloads`
2. `debug_log_dir`
3. `overlay_dir`
4. `temp_root`
5. `file_dialog_dir`

Default directory values:

1. `downloads`: `~/Downloads`
2. `debug_log_dir`: `~/.cache/ra_sim/logs`
3. `overlay_dir`: `~/.cache/ra_sim/overlays`
4. `temp_root`: `~/.cache/ra_sim`
5. `file_dialog_dir`: `~/.local/share/ra_sim`

Current debug/log outputs:

1. GUI runtime update trace
   File: `runtime_update_trace_<YYYYMMDD>.log`
   Location: `downloads`
   Controlled by: `debug.runtime_update_trace.enabled`

2. Geometry-fit logs
   File: `geometry_fit_log_<stamp>.txt`
   Location: `debug_log_dir`
   Controlled by: `debug.geometry_fit.log_files`
   Verbosity controlled by: `debug.geometry_fit.extra_sections`

3. Mosaic-shape fit logs
   File: `mosaic_shape_fit_log_<stamp>.txt`
   Location: `debug_log_dir`
   Controlled by: `debug.mosaic_fit.log_files`

4. Projection-debug JSON
   File: `projection_debug_<stamp>.json`
   Location: `debug_log_dir`
   Controlled by: `debug.projection_debug.enabled`

5. Diffraction debug CSV
   File: `mosaic_full_debug_log.csv`
   Location: `debug_log_dir`
   Controlled by: `debug.diffraction_debug_csv.enabled`

6. Intersection-cache dumps
   Directory pattern: `intersection_cache_<stamp>_<pid>`
   Root location: `debug.intersection_cache.log_dir` when set, otherwise `debug_log_dir`
   Controlled by: `debug.intersection_cache.enabled`

`get_dir(...)` still creates missing configured directories automatically. Changing a directory setting redirects output, but it does not enable or disable output by itself.

### Practical examples

Disable all debug/log output in config:

```yaml
debug:
  global:
    disable_all: true
```

Keep everything on except console spam:

```yaml
debug:
  global:
    disable_all: false
  console:
    enabled: false
```

Disable only projection-debug JSON and intersection-cache dumps:

```yaml
debug:
  projection_debug:
    enabled: false
  intersection_cache:
    enabled: false
```

Disable geometry-fit extra sections but keep the log files:

```yaml
debug:
  geometry_fit:
    log_files: true
    extra_sections: false
```

Redirect intersection-cache dumps:

```yaml
debug:
  intersection_cache:
    enabled: true
    log_dir: /tmp/ra-sim-cache-dumps
```

Temporary compatibility override from PowerShell:

```powershell
$env:RA_SIM_DISABLE_ALL_LOGGING = "1"
```

Temporary console-debug override from PowerShell:

```powershell
$env:RA_SIM_DEBUG = "1"
```

### Bottom line

Use `config/debug.yaml` for normal project configuration.

If you need a master OFF switch, set:

```yaml
debug:
  global:
    disable_all: true
```

If you need a temporary shell-level override, use:

```powershell
$env:RA_SIM_DISABLE_ALL_LOGGING = "1"
```

The config kill switch and the env kill switches both disable every debug/log output covered by this section.

## Appendix A: Center of rotation axis math

This appendix expands the short rotation summary in [Rotation construction used by the kernel](#rotation-construction-used-by-the-kernel).

RA-SIM applies the sample tilt `theta_initial` about a configurable
center-of-rotation axis rather than always about laboratory `+x`. The axis is
controlled by:

- `cor_angle`: pitch away from `+x` toward `+z`
- `psi_z`: yaw of that pitched axis about laboratory `z`

This note matches the exact sign convention used in
`ra_sim/simulation/diffraction.py`.

### Baseline sample rotation

Before the CoR tilt is applied, the code builds

\[
R_{ZY} = R_z(\psi)\,R_y(\chi),
\]

with

\[
R_y(\chi)=
\begin{bmatrix}
\cos\chi & 0 & \sin\chi \\
0 & 1 & 0 \\
-\sin\chi & 0 & \cos\chi
\end{bmatrix},
\qquad
R_z(\psi)=
\begin{bmatrix}
\cos\psi & \sin\psi & 0 \\
-\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}.
\]

With this `R_z` convention, positive `psi` rotates `+x` toward `-y`.

### CoR axis

The unyawed CoR axis lives in the `x-z` plane:

\[
\mathbf{a}_0 =
\begin{bmatrix}
\cos\varphi \\
0 \\
\sin\varphi
\end{bmatrix},
\qquad
\varphi = \mathrm{radians}(\text{cor\_angle}).
\]

Yawing that axis by `psi_z` gives

\[
\mathbf{a} = R_z(\psi_z)\,\mathbf{a}_0
=
\begin{bmatrix}
\cos\psi_z \cos\varphi \\
-\sin\psi_z \cos\varphi \\
\sin\varphi
\end{bmatrix}.
\]

So:

- `cor_angle = 0`, `psi_z = 0` gives `a = +x`
- positive `cor_angle` tips the axis toward `+z`
- positive `psi_z` rotates the `+x` direction toward `-y`

The implementation normalizes `a` before using it.

### Rodrigues rotation

Let

\[
\theta = \mathrm{radians}(\text{theta\_initial}).
\]

The CoR tilt is applied with Rodrigues' formula:

\[
R_{\mathrm{cor}}
= \cos\theta\,I
+ (1-\cos\theta)\,\mathbf{a}\mathbf{a}^\top
+ \sin\theta\,[\mathbf{a}]_\times,
\]

where

\[
[\mathbf{a}]_\times =
\begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}.
\]

The final sample rotation used by the kernel is

\[
R_{\mathrm{sample}} = R_{\mathrm{cor}}\,R_{ZY}.
\]

The rotated sample normal is

\[
\mathbf{n}_{\mathrm{surf}} = R_{\mathrm{cor}}\,(R_{ZY}\,\hat{\mathbf{z}}).
\]

### Reference point

The sample reference point starts at

\[
P_0 = (0,\,0,\,-z_s).
\]

The code rotates it with `R_sample` and then explicitly sets `P0_rot[0] = 0`.
That keeps the reference point on the laboratory CoR plane used by the detector
intersection geometry.

### Where this appears

The same construction is used in:

- `_build_sample_rotation()` in `ra_sim/simulation/diffraction.py`
- the detector-path debug helper in the same module

That keeps the main simulation and the debug path on the same axis convention.

## Appendix B: equation-to-code map

| Topic | Main function(s) |
| --- | --- |
| Typed simulation request/response surface | [`ra_sim/simulation/types.py`](../ra_sim/simulation/types.py), [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py) |
| Legacy positional simulation wrapper | [`simulate_diffraction`](../ra_sim/simulation/simulation.py) |
| Low-discrepancy beam profile generation | [`generate_random_profiles`](../ra_sim/simulation/mosaic_profiles.py) |
| Beam-state clustering and weights | [`cluster_beam_profiles`](../ra_sim/simulation/mosaic_profiles.py) |
| Sample rotation and sample-plane setup | [`_build_sample_rotation`](../ra_sim/simulation/diffraction.py), [`_precompute_sample_terms`](../ra_sim/simulation/diffraction.py) |
| Real-space ray/plane intersection | [`intersect_line_plane`](../ra_sim/simulation/diffraction.py), [`intersect_infinite_line_plane`](../ra_sim/simulation/diffraction.py) |
| Nominal detector visibility screen | [`_nominal_reflection_visible`](../ra_sim/simulation/diffraction.py) |
| Reciprocal-space sphere intersection and arc integration | [`solve_q`](../ra_sim/simulation/diffraction.py), [`_solve_q_uniform`](../ra_sim/simulation/diffraction.py), [`_solve_q_adaptive`](../ra_sim/simulation/diffraction.py) |
| Mosaic density and local arc restriction | [`_mosaic_density_scalar`](../ra_sim/simulation/diffraction.py), [`_build_local_arc_windows`](../ra_sim/simulation/diffraction.py) |
| Outgoing ray build and detector projection | [`_calculate_phi_from_precomputed`](../ra_sim/simulation/diffraction.py) |
| Bilinear detector deposition | [`_accumulate_bilinear_hit`](../ra_sim/simulation/diffraction.py), cached variants in the same file |
| Hit-table reduction to up to two simulated centers | [`hit_tables_to_max_positions`](../ra_sim/simulation/diffraction.py) |
| Geometry dataset preparation | [`build_geometry_manual_fit_dataset`](../ra_sim/gui/geometry_fit.py) |
| Orientation auto-selection | [`select_fit_orientation`](../ra_sim/gui/geometry_overlay.py) |
| Point-match residual assembly | [`_evaluate_geometry_fit_dataset_point_matches`](../ra_sim/fitting/optimization.py), [`_weight_measurement_residual`](../ra_sim/fitting/optimization.py) |
| Geometry nonlinear solve and identifiability | [`fit_geometry_parameters`](../ra_sim/fitting/optimization.py) |
| Background summit detection and matching | [`build_background_peak_context`](../ra_sim/fitting/background_peak_matching.py), [`match_simulated_peaks_to_peak_context`](../ra_sim/fitting/background_peak_matching.py) |
| Geometry-cached mosaic-shape fitting | [`fit_mosaic_shape_parameters`](../ra_sim/fitting/optimization.py) |
| Legacy mosaic-width fitting | [`fit_mosaic_widths_separable`](../ra_sim/fitting/optimization.py) |
| Image-space refinement | [`build_tube_rois`](../ra_sim/fitting/optimization.py), [`robust_residuals`](../ra_sim/fitting/optimization.py), [`iterative_refinement`](../ra_sim/fitting/optimization.py) |
| hBN robust ellipse fitting | [`robust_fit_ellipse`](../ra_sim/hbn_fitter/fitter.py), [`weighted_refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) |
| hBN snap-to-ring and uncertainty estimate | [`snap_points_to_ring`](../ra_sim/hbn_fitter/fitter.py), `_estimate_snap_uncertainty_px` in the same file |
| hBN iterative ring refinement | [`refine_ellipse`](../ra_sim/hbn_fitter/fitter.py) |
| hBN confidence scoring | [`compute_fit_confidence`](../ra_sim/hbn_fitter/fitter.py) |
| hBN projective tilt optimization | [`apply_tilt_projective`](../ra_sim/hbn_fitter/fitter.py), [`optimize_tilts_projective`](../ra_sim/hbn_fitter/fitter.py) |
