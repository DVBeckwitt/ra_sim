# Simulation and Fitting Reference

This document is the implementation-faithful reference for the forward simulation and fitting code in this repository. It is still the place to check what the code does today, but it now also spells out the physics assumptions, the exact geometry and ray-tracing conventions, the fitting residuals, and the statistical procedures used by the GUI workflows. If another note disagrees with this file, prefer this file and then audit the linked functions directly.

The scope is the current live pipeline:

- forward HKL and rod simulation through the typed API and legacy wrapper
- exact detector and sample geometry used by the diffraction kernel
- reciprocal-space integration and mosaic broadening
- manual point-pick geometry fitting
- automatic background peak matching
- geometry-fit-cached mosaic-shape fitting, legacy mosaic-width fitting, and image-space refinement
- hBN calibrant ellipse fitting and projective tilt correction

No new API is defined here. All names, defaults, and equations below are taken from the current code.

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
- `zs`: sample-plane reference shift along the laboratory/sample geometry path; it enters the sample plane anchor point `P0 = (0, 0, zs)`.
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

The live forward path is `simulate(...)` or `simulate_qr_rods(...)` through [`ra_sim/simulation/engine.py`](../ra_sim/simulation/engine.py), which forwards typed inputs into [`process_peaks_parallel_safe`](../ra_sim/simulation/diffraction.py). The algorithm is:

1. Build detector and sample rotations from `gamma`, `Gamma`, `chi`, `psi`, `psi_z`, `theta_initial`, and `cor_angle`.
2. Build or receive aligned beam-sample arrays `(beam_x, beam_y, theta, phi, wavelength)` and optional `sample_weights`.
3. For each beam sample, intersect the incident ray with the sample plane, apply finite sample clipping if `sample_width_m` or `sample_length_m` are nonzero, and precompute entry optics quantities that do not depend on `(H, K, L)`.
4. For each reflection, build `G_vec = (0, Gr, Gz)` from `(H, K, L, a, c)`.
5. For each valid beam sample, solve the reciprocal-space circle problem `solve_q(...)`.
6. For each accepted `Q` solution, build the outgoing wavevector, apply exit optics and attenuation, intersect the outgoing ray with the detector plane, convert to `(row_f, col_f)`, and deposit intensity bilinearly.
7. Optionally emit hit tables, `Q` tables, miss tables, and per-sample status codes.

The simulation is deterministic once the beam arrays are fixed. There is no Monte Carlo scattering inside the kernel itself; even the beam sampling is low-discrepancy and can be clustered into deterministic weighted representatives.

## Pedagogical view of the optics modes

The code has two optics branches:

- `OPTICS_MODE_FAST`, labeled in the GUI as `Original Fast Approx (Fresnel + Beer-Lambert)`
- `OPTICS_MODE_EXACT`, labeled in the GUI as `Complex-k DWBA slab optics (Precise)`

Pedagogically, both modes do the same three-stage calculation:

1. Build an in-sample incoming wave from the incident beam.
2. Scatter that wave through the same reciprocal-space `solve_q(...)` machinery.
3. Build an outgoing wave, transmit it back out of the sample, and project it to the detector.

The difference is therefore not in the reflection list, the `solve_q(...)`
geometry, the structure-factor model, or the detector mapping. The difference
is in how the code computes the entry and exit transport through the sample.

### Shared intensity skeleton

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

The reciprocal-space solve is [`solve_q`](../ra_sim/simulation/diffraction.py). It does not solve a generic nonlinear system; it constructs the exact circle of intersection of two spheres and then integrates the mosaic density along that circle.

### Sphere-sphere intersection geometry

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

The final exit angle and outgoing magnitude are

\[
2\theta_t = \arccos\left(\mathrm{clamp}(k_r/k_0, -1, 1)\right)\operatorname{sign}(2\theta_t'),
\qquad
|k_f| = k_0.
\]

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

The outgoing in-sample wavevector is

\[
k_f =
\begin{bmatrix}
|k_f|\cos(2\theta_t)\sin\phi_f \\
|k_f|\cos(2\theta_t)\cos\phi_f \\
|k_f|\sin(2\theta_t)
\end{bmatrix}.
\]

It is rotated back into the laboratory frame by

\[
k_f^{\mathrm{lab}} = R_{\mathrm{sample}}\,k_f.
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

## Appendix: equation-to-code map

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

