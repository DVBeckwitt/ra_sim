# hBN Fitter

This page documents [`ra_sim/hbn_fitter/fitter.py`](../ra_sim/hbn_fitter/fitter.py)
as a workflow. The broader math reference remains
[`simulation_and_fitting.md`](simulation_and_fitting.md#hbn-calibrant-fitting).
This page focuses on operator flow, GUI state, click snapping, bundle
compatibility, and maintenance invariants.

The hBN fitter is the calibrant workflow for estimating detector geometry from
hBN diffraction rings. It loads an hBN OSC image and a dark OSC image, prepares
a background-subtracted log image, lets the user pick points on several rings,
snaps those clicks to local intensity maxima, fits robust ellipses to the
rings, refines those ellipses from radial intensity profiles, solves a detector
tilt correction by circularizing the rings, and saves an NPZ bundle for reload
or downstream RA-SIM use.

## Entry Points

Use the packaged GUI through the main CLI:

```bash
python -m ra_sim calibrant
python -m ra_sim calibrant --bundle path/to/hbn_bundle.npz
python -m ra_sim calibrant-fit --bundle path/to/hbn_bundle.npz
```

The fitter module can also be launched directly:

```bash
python -m ra_sim.hbn_fitter.fitter --bundle path/to/hbn_bundle.npz
```

The GUI class is `HBNFitterGUI`. The script entry point parses `--bundle`,
creates the Tk root, restores launch-window affinity, builds `HBNFitterGUI`,
optionally loads the startup bundle, then starts the Tk event loop.

The non-GUI counterpart is:

```bash
python -m ra_sim hbn-fit
```

## Current Status

Status: documented.

Type: documentation feature.

What changed:

- Added this maintainer/operator guide for the hBN fitter script.
- Documented current full-resolution GUI coordinate handling and historical
  `points_ds` naming.
- Documented raw-click versus snapped-point storage, final-click snapping,
  uncertainty estimates, ellipse refinement, projective tilt optimization, and
  NPZ bundle export contracts.
- Linked the guide from the docs index, GUI workflow guide, architecture map,
  and canonical simulation/fitting reference.

Bug/error status:

- No runtime bug was fixed by this documentation patch.
- No code path was changed.
- The documented risk is future regression risk: point scaling, click snapping,
  tilt sign export, and legacy NPZ bundle compatibility should not drift.

Validation status:

- Documentation link/path sanity checks passed.
- `git diff --check` passed for the touched existing docs.
- No Python tests were run because this patch is documentation-only.

## Operator Workflow

1. Select the hBN OSC file and the dark OSC file.
2. Let the GUI load the image when both paths exist.
3. Optionally pick or type the beam center.
4. Enable pick mode and collect points around each ring.
5. Let the GUI snap each raw click to a likely ring maximum.
6. Fit ellipses from the snapped points.
7. Optionally run high-resolution refinement.
8. Optimize detector tilt/circularization.
9. Save the NPZ bundle and/or overlay PNG.

`Run Full Calibration` performs the initial ellipse fit and tilt optimization.
`Further Refine (High-Res)` runs a more expensive high-resolution pass after an
initial fit exists.

The core data flow is:

```text
OSC hBN + dark
  -> dark subtraction and broad background removal
  -> log image for sampling and display
  -> user clicks ring points
  -> live fast snap preview
  -> final snap with confidence gates and sigma estimate
  -> robust seed ellipse per ring
  -> profile-based iterative ellipse refinement
  -> fit confidence scoring
  -> projective circularity tilt optimization
  -> NPZ bundle export
```

## Coordinate Frames

The viewer displays the image at full resolution. Mouse coordinates, stored
interactive points, ellipses, centers, and tilt optimization point sets are in
full-resolution detector pixels during normal GUI operation.

Some names still contain `ds`, such as `points_ds`, because older code used
downsampled display coordinates. Current in-memory arrays are full-resolution
GUI coordinates. The downsample setting is used as a fitting/control parameter:
`_points_for_fit()` divides points by `self.down`, and `fit_ellipses()`
multiplies them back by `downsample`. This preserves the existing fit API and
bundle semantics.

Bundle persistence is different from in-memory GUI state. `save_bundle()` writes
point arrays in the declared `point_coord_frame = "downsampled"` form through
`_points_for_fit()`. `load_bundle()` reads the declared frame and rescales
legacy or downsampled coordinates back to full-resolution in-memory
coordinates. Do not double-scale these points.

Coordinate invariants:

- Ellipse parameters are full-resolution pixels.
- Centers are full-resolution pixels.
- `img_bgsub` and `img_log` are full-resolution arrays.
- `img_disp` is displayed full-resolution in current GUI operation.
- Saved `ell_points_ds` can be downsample-frame data even though internal
  `points_ds` is full-resolution after loading.
- The plot uses `origin="lower"`; the input array is not rotated.

## Image Preparation

Image input is handled by three helpers:

- `load_and_bgsub(hbn_path, dark_path)` reads the dark and raw OSC arrays,
  subtracts dark, removes a broad Gaussian-blurred background, clips negative
  values, and returns a float32 background-subtracted image.
- `make_log_image(img)` converts the background-subtracted image to log
  intensity with a small epsilon.
- `build_display(img_log, downsample)` normalizes the log image by robust
  percentiles and can downsample it. Current GUI usage passes `downsample = 1`,
  so the displayed image remains full-resolution.

Log intensity is used because ring peaks have high dynamic range. Log scaling
makes weak rings visible while keeping strong rings usable for snapping and
profile sampling. Display normalization is not a physical intensity model.

## GUI Data Model

Important state fields inside `HBNFitterGUI`:

- `img_bgsub`: full-resolution background-subtracted data.
- `img_log`: full-resolution log intensity used for fitting and snapping.
- `img_disp`: normalized display image.
- `points_raw_ds`: raw user click positions, despite the historical suffix.
- `points_ds`: snapped point positions used for fitting, despite the suffix.
- `points_sigma_ds`: per-point snap uncertainty in pixels.
- `ellipses`: fitted ring ellipses, one dict per ring with `xc`, `yc`, `a`,
  `b`, `theta`, and `ring_index`.
- `fit_quality`: confidence and diagnostic arrays per ring.
- `optim`: tilt/circularization result dict.
- `corrected`: corrected point sets after tilt correction.
- Mode flags: `pick_mode`, `edit_mode`, `center_pick_mode`,
  `_precision_active`, `_drag_edit_active`, and `_pan_active`.

Raw clicks and snapped points are deliberately stored separately. Raw clicks are
drawn as small x markers for transparency. Snapped points are drawn as open
circles and used for fitting.

## Clicking And Snapping

Point clicking is the most important interaction contract. A raw click is not
directly treated as the fitting point. The GUI previews a fast snapped point
during mouse motion, then performs a full final snap on release, stores both raw
and snapped positions, estimates snap uncertainty, and refreshes the overlay.

### Pick Mode

Pick-mode call path:

```text
on_click(left) -> start precision pick
on_motion(left held) -> fast preview snap
on_release(left) -> _commit_point()
_commit_point() -> _snap_click_point_ds(preview_fast=False)
_snap_click_point_ds() -> snap_points_to_ring()
_commit_point() stores raw point, snapped point, snap sigma, status text, and refreshes plot
```

A left click in pick mode opens a 40 x 40 pixel zoom around the click instead
of immediately committing the point. While the user drags inside that zoom, the
GUI displays the raw cursor point, the preview-snapped point, and a line between
them. Mouse motion preview uses `preview_fast=True`, with a smaller search
window, coarser step, and quadratic peak detection instead of pseudo-Voigt
fitting. On mouse release, the GUI runs the full final snap and stores the
result.

### Edit Mode

Edit-mode call path:

```text
toggle_edit_mode()
on_click(left) -> _start_drag_edit()
on_motion(left held) -> fast preview snap
on_release(left) -> _move_existing_point()
_move_existing_point() -> final snap and state update
```

Edit mode selects the nearest existing point within a dynamic radius, zooms to
the point, previews a snapped replacement during drag, then replaces the
existing point on release.

### Snap Algorithm

`snap_points_to_ring()` is the core snapping engine. For each clicked point it:

1. Chooses a provisional ring model. It uses an existing fitted ellipse when
   available, tries an initial ellipse from existing points plus the candidate
   when possible, and otherwise falls back to a circular model around the
   current/default center.
2. Defines a local radial direction from the ring center to the click and a
   tangential direction perpendicular to it.
3. Builds a local search grid with radial offsets `along` and tangential
   offsets `across`.
4. Samples log intensity on that grid with bilinear interpolation.
5. Finds the best radial peak for each tangential offset. Full snapping uses
   pseudo-Voigt fitting when enough samples are available. Preview snapping uses
   the faster quadratic peak estimator.
6. Optionally refines the candidate peak with a small 2D centroid.
7. Scores candidates with a posterior-like value that rewards peak SNR and
   penalizes ellipse residual, tangential displacement, and click distance.
8. Picks the best-scoring candidate.
9. Rejects weak or ambiguous snaps when confidence gating is enabled, keeping
   the raw click instead.
10. Returns the snapped point and metadata.

Snapping metadata includes:

- `used_snap`
- `reason`
- `best_posterior`
- `posterior_margin`
- `best_peak_snr`
- `best_resid_px`
- `best_tangent_offset_px`
- `best_click_distance_px`
- `candidate_count`

### Snap Uncertainty

`_estimate_snap_uncertainty_px()` estimates final-click uncertainty by
perturbing the seed point over a small set of radii and angles, snapping those
seeds in fast mode, then measuring posterior-weighted scatter and bias relative
to the committed snap. The result is clamped between the configured minimum and
maximum and stored in `points_sigma_ds`.

This is a practical repeatability estimate for the local snapping operation,
not a statistical covariance estimate.

### Click-Speed Notes

The click path is optimized without changing final snap semantics:

- `bilinear_sample_many()` vectorizes local grid sampling.
- `find_profile_peak_fast()` uses the quadratic peak estimator for previews and
  uncertainty seeds.
- `subpixel_quadratic_peak_1d()` uses a closed-form 3-point parabola instead of
  a general polynomial fit.
- `subpixel_centroid_2d()` converts only the small local patch to float.
- `_estimate_snap_uncertainty_px()` snaps all uncertainty seeds in one batch.
- Final clicking still runs full pseudo-Voigt snapping and uncertainty
  estimation, so it is intentionally slower than live preview.

## Ellipse Fitting

Ellipse fitting starts from snapped point sets:

- `ellipse_residuals_px()` computes a radialized residual from points to an
  ellipse.
- `robust_fit_ellipse()` performs RANSAC ellipse fitting followed by sigma
  clipping.
- `_sanitize_point_sigma()` fills missing or invalid point uncertainties with a
  stable default.
- `weighted_refine_ellipse()` runs Powell refinement with point uncertainty
  weights and a robust loss.
- `fit_initial_ellipse()` performs robust initial fitting with fallback direct
  or bounding-box estimates.
- `refine_ellipse()` samples radial profiles around the ellipse, detects ring
  peaks, refits the ellipse, and stops when updates are small.
- `fit_ellipses()` orchestrates ring-level fitting, warm starts from prior
  ellipses when valid, snaps seed points, fits/refines each ring, and returns
  ellipse dicts.

Ring fitting lifecycle:

1. A ring needs at least 5 points to fit an ellipse.
2. A valid previous ellipse can warm-start the fit.
3. User points are snapped to the current ring model.
4. The algorithm samples intensity around the ellipse and finds local radial
   peaks.
5. User-picked anchor points remain included so the result does not drift to
   unrelated rings.
6. Point uncertainties can weight the final ellipse update.

The fitter uses robust estimation, sigma clipping, profile refinement, and
weighted local refinement. Do not describe it as ordinary least squares.

## Fit Confidence

Fit confidence is computed by:

- `ring_signal_snr()`
- `click_angular_coverage()`
- `compute_fit_confidence()`

Confidence is a heuristic diagnostic, not a formal probability. It combines
point-to-ellipse residual, ring signal-to-noise, angular coverage around the
ellipse, number of points, and downsample penalty. The optimizer can use these
confidence values to filter or weight weak rings.

Use confidence to compare fit quality within the same calibration workflow. Do
not interpret it as an absolute detector-geometry uncertainty.

## Center Handling

Center-related helpers:

- `ellipse_center()`
- `_get_center()`
- `_default_center_full()`
- `toggle_center_pick()`
- `center_from_fit()`

Center precedence:

1. User-entered or user-picked center if available.
2. Robust weighted center from fitted ellipses.
3. Image center fallback for early snapping and suggestions.

A user-provided center becomes a prior during tilt optimization. It can drift
within the configured limit instead of being completely fixed.

## Suggested Regions

Point suggestions use:

- `add_more_points_action()`
- `_suggest_region_for_ring()`
- `_best_snap_from_seeds()`
- `_update_suggested_regions()`
- `_largest_angle_gap_mid()`

Suggestions try to fill the largest angular gap in each incomplete ring. When
an ellipse exists, seeds are placed on that ellipse and snapped to image
intensity. Without an ellipse, the fallback uses the current/default center and
estimated radius.

## High-Resolution Refinement

High-resolution refinement uses:

- `_highres_refine_settings()`
- `further_refine_action()`
- `_ellipse_solution_delta()`

The refinement switches fitting to downsample 1, estimates radial step and
angular density from current residuals and largest ring radius, runs a refined
fit, then runs a stricter verification pass. If the stricter pass changes the
result only negligibly, the primary refined fit is treated as a plateau.
Otherwise, the stricter result is accepted.

## Tilt And Circularization

The final geometry signal is projective circularization, not only ellipse
shape. After applying a candidate tilt correction, each ring should be as
circular as possible around the optimized center. The cost is based on radial
scatter normalized by radius and averaged across rings with weights.

Projective path:

- `apply_tilt_projective()`
- `circularity_cost_projective()`
- `circularity_metrics_projective()`
- `optimize_tilts_projective()`

Projective optimizer stages:

1. Build dense point samples from fitted ellipses when available.
2. Weight rings by fit confidence and snap uncertainty.
3. Run a coarse grid over tilts with fixed center and nominal distance.
4. Locally refine tilt and projective distance.
5. Optionally refine center, tilt, and distance jointly with center prior/drift
   constraints.
6. Return corrected points, before/after circularity, radii, costs, center,
   tilts, and optimizer metadata.

Legacy path:

- `apply_tilt_xy()`
- `circularity_cost()`
- `circularity_metrics()`
- `optimize_tilts()`

Legacy optimizer stages:

1. Run a coarse tilt grid.
2. Run Nelder-Mead local tilt refinement.
3. Optionally run Powell center-plus-tilt refinement.
4. Return the same general diagnostics, with `distance_px` as NaN.

`optimize_action()` tries the projective path when ellipses are available and
falls back to the legacy optimizer if the projective path fails. Optimization
produces corrected point sets and tilt metadata; it should not mutate clicked
points or ellipse fits.

## Bundle Schema

Bundle helpers:

- `build_hbn_fitter_bundle_payload()`
- `save_bundle()`
- `load_bundle()`
- `npz_scalar()`, `npz_string()`
- `pts_to_obj()`, `obj_to_pts_list()`
- `scalars_to_obj()`, `obj_to_scalar_lists()`, `obj_to_ndarrays()`
- `ellipses_to_array()`, `array_to_ellipses()`
- `ellipse_ring_indices()`, `apply_ellipse_ring_indices()`

NPZ bundles store enough state to reload the GUI and enough geometry/tilt
metadata for downstream RA-SIM use.

Important keys:

- `npz_format_version`
- `img_bgsub`
- `img_log`
- `downsample_factor`
- `point_coord_frame`
- `point_sigma_coord_frame`
- `ell_points_ds`
- `ell_points_raw_ds`
- `ell_points_sigma_px`
- `ellipse_params`
- `ellipse_ring_indices`
- `detector_center`
- `center_source`
- `tilt_x_deg`, `tilt_y_deg`
- `tilt_x_deg_internal`, `tilt_y_deg_internal`
- `cost_zero`, `cost_final`
- `circ_before`, `circ_after`
- `radii_before`, `radii_after`
- `ring_weights`
- `ring_snap_sigma_px`
- `optimizer_kind`
- `projective_distance_px`
- `ell_points_corrected`
- `input_hbn_path`, `input_dark_path`
- Legacy compatibility keys: `center`, `tilt_correction`, `tilt_hint`,
  `distance_estimate_m`, and `expected_peaks`

Tilt sign convention:

- Internal optimizer tilt values are stored in `tilt_x_deg_internal` and
  `tilt_y_deg_internal`.
- Exported `tilt_x_deg` and `tilt_y_deg` use the NPZ exchange convention,
  currently the opposite sign from the internal optimizer state.
- The bundle also stores sign convention flags for simulation gamma/Gamma
  mapping.

Do not remove legacy keys or change sign conventions without updating bundle
import/export and downstream simulation docs.

## Plot Refresh And Overlays

`refresh_plot()` redraws:

- normalized hBN log image,
- raw clicks as small x markers,
- snapped points as open circles,
- lines from raw click to snapped point,
- fitted ellipses,
- suggested target regions,
- current center,
- optimized center,
- before/after circularized rings,
- corrected points.

View limits are preserved across most refreshes unless `reset_view=True`. This
matters for precision picking and edit mode.

## UI Actions

File selection and auto-load:

- `browse_hbn()`
- `browse_dark()`
- `_schedule_auto_load()`
- `_auto_load_if_ready()`
- `load_from_osc()`

Mode toggles:

- `toggle_pick()`
- `toggle_center_pick()`
- `toggle_edit_mode()`

Point management:

- `undo_last()`
- `reset_points()`
- `clear_fits_keep_points()`
- `add_more_points_action()`

Fitting and optimization:

- `fit_action()`
- `further_refine_action()`
- `solve_all_action()`
- `optimize_action()`

Persistence:

- `load_bundle_dialog()`
- `save_bundle_dialog()`
- `save_overlay_dialog()`
- `save_bundle()`
- `load_bundle()`

## Error Handling And Fallbacks

Main fallback behavior:

- Image load errors are shown as message boxes and status text.
- A ring with fewer than 5 points is skipped by ellipse fitting.
- Initial ellipse fitting falls back from RANSAC to direct estimation to a
  bounding-box estimate.
- Snapping falls back to raw click when no candidate passes gates.
- Projective tilt optimization falls back to legacy tilt optimization.
- Bundle loading tolerates legacy missing fields where practical.

## Maintenance Invariants

Do not break these contracts:

- Preserve raw clicked points separately from snapped fitting points.
- Preserve full-resolution internal coordinates after loading or clicking.
- Preserve bundle coordinate-frame declarations and scaling behavior.
- Preserve tilt sign convention on export.
- Preserve confidence metadata names in saved bundles.
- Preserve ability to reload legacy NPZ bundles.
- Preserve toolbar pan/zoom guard before point interactions.
- Preserve final-click full snap accuracy, even if preview snapping is
  approximate.
- Preserve fit weighting from `points_sigma_ds`.
- Preserve view restoration after precision pick and edit drag.

## Validation

For documentation-backed code changes to `fitter.py`, start with:

```bash
python -m compileall -q ra_sim/hbn_fitter/fitter.py
python -m pytest tests/test_hbn_fitter_bundle_export.py tests/test_hbn_geometry_import_safe.py tests/test_gui_bootstrap.py -ra
```

For click/snapping changes, add or run parity checks that compare snapped
coordinates and metadata on representative synthetic rings. At minimum check:

- fast snap preview returns finite points near the ring,
- final pseudo-Voigt snapping returns stable coordinates within round-off
  tolerance,
- uncertainty fields are present and finite when snapping succeeds,
- bundle save/load preserves point coordinates, sigmas, ellipses, center, tilt
  metadata, and ring indices.

Acceptance criteria for future fitter patches:

- Saved bundles load in existing downstream RA-SIM tools.
- Raw clicked points and snapped points are both preserved.
- Snap confidence gates are unchanged unless explicitly documented.
- Point order and ring indices are unchanged.
- Final accepted snap coordinates match previous behavior within floating-point
  tolerance on regression fixtures.
- Final click latency improves without reducing fit confidence.

## Function Map

| Area | Functions or methods | Purpose |
|---|---|---|
| Input and display | `load_and_bgsub`, `make_log_image`, `build_display` | Read OSC files, subtract dark/background, prepare log and display images. |
| Bundle format | `build_hbn_fitter_bundle_payload`, `save_bundle`, `load_bundle` | Persist and restore calibration state. |
| Ellipse math | `ellipse_residuals_px`, `robust_fit_ellipse`, `weighted_refine_ellipse`, `fit_initial_ellipse` | Fit robust ellipses from clicked/snapped points. |
| Peak localization | `bilinear_sample_many`, `find_profile_peak_pseudovoigt`, `find_profile_peak_fast`, `subpixel_centroid_2d` | Locate ring intensity maxima near clicks or refinement samples. |
| Click snapping | `snap_points_to_ring`, `_snap_click_point_ds`, `_estimate_snap_uncertainty_px` | Convert raw clicks into snapped ring points and sigma estimates. |
| Ring refinement | `refine_ellipse`, `fit_ellipses`, `further_refine_action` | Improve ellipses from image radial profiles. |
| Confidence | `ring_signal_snr`, `click_angular_coverage`, `compute_fit_confidence` | Score ring and overall fit quality. |
| Tilt optimization | `apply_tilt_projective`, `circularity_cost_projective`, `optimize_tilts_projective` | Find detector tilt that circularizes rings. |
| GUI state and events | `HBNFitterGUI`, `on_click`, `on_motion`, `on_release`, `refresh_plot` | Manage interaction, preview, editing, and overlays. |
| CLI | `parse_args`, `main` | Launch the standalone Tk GUI. |

## Common Pitfalls

- Do not describe the fitter as only an ellipse fitter. The final geometry
  signal is projective ring circularization.
- Do not imply display normalization is used for physical intensity fitting.
- Do not assume `points_ds` means current in-memory display pixels. In current
  GUI operation it is effectively full-resolution storage with historical
  naming.
- Do not remove legacy NPZ keys.
- Do not change tilt sign conventions without updating import/export and
  downstream simulation docs.
- Do not treat fit confidence as a calibrated probability.
- Do not make preview snapping authoritative. Final committed clicks rerun the
  full snap path.
