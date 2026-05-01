# RA-SIM GUI workflow

This page is the short operator map for the interactive workflow. It summarizes
the usual order of operations and points into the canonical reference when you
need the detailed implementation story.

See also:

- [docs index](index.md)
- [Architecture guide](architecture.md)
- [Troubleshooting guide](troubleshooting.md)
- [Canonical GUI workflow section](simulation_and_fitting.md#gui-workflow-and-views)
- [hBN fitter guide](hbn-fitter.md)

## Main Work Areas

- Simulation view: primary detector-space workspace for global geometric agreement
- Integration views: radial, azimuthal, and caked diagnostics after the detector pattern is roughly aligned
- Calibrant view: hBN ellipse-fitting workflow used to estimate beam center and detector tilt; see the [hBN fitter guide](hbn-fitter.md) for click snapping, coordinate frames, and bundle export details
- Parameters panel: controls for geometry, mosaic, structure factor, stacking fault, occupancy, beam, and fit settings

## Typical Workflow

1. Start from the calibrant workflow if detector geometry is uncertain.
2. Launch the main GUI and load the experimental background.
3. Align detector-space features before trusting reduced-coordinate plots.
4. Use integration views to validate widths, radial positions, and intensity balance.
5. Fit mosaic shapes only after accepting the geometry-fit cache. The implementation target is selected Qr/background pairing, local `I(phi)` extraction, Lorentzian plus Gaussian profile centering, and centered measured/simulated profile comparison.
6. Fit structure-factor terms after mosaic is stable. The implementation target is a global multi-image detector-ROI intensity fit with one shared structure-factor parameter vector and one scale nuisance per image.
7. Save parameter snapshots so iterations stay reproducible.

The Refine tab starts with its parameter sections collapsed. Open only the
geometry, detector, beam, lattice, mosaic, CIF, or ordered-structure section
needed for the current step.

## Headless Counterparts

The same workflows have CLI entry points when you need automation or a
non-interactive path:

- `python -m ra_sim gui`
- `python -m ra_sim calibrant`
- `python -m ra_sim simulate --out output.png`
- `python -m ra_sim hbn-fit`
- `python -m ra_sim fit-geometry <state.json>`
- `python -m ra_sim fit-mosaic-shape <state.json>`

## Workflow Notes

- Geometry-fit-cached mosaic fitting depends on the latest successful manual geometry dataset.
- Mosaic fitting must refuse stale geometry, stale selected backgrounds, stale Qr/Qz grouping, or changed shared-theta metadata.
- Structure-factor fitting must not use GUI display normalization. It should compare background-subtracted detector ROI counts to ray-carried simulated intensity summed into the same ROI.
- Detector-space agreement comes first; 1D views are validation tools, not the primary fitting target.
- Use Setup > Beam Controls > `Pick Beam Center` to set detector `center_x`/`center_y` directly from the loaded background image. The pick temporarily returns to detector view, shows the background if hidden, zooms around the press point, and writes the mapped click through the visible Beam Center Row/Col slider widgets in the GUI order on release.
- Use `Toggle Simulation` to hide or show the simulated detector/caked overlay without changing the loaded background image, fit inputs, or generated simulation data.
- Diffuse background subtraction remains off by default. When enabled, the Background Subtraction panel's `Use before fit/pick` option uses the signed subtracted background for Qr picking, auto-match, and fit comparison while keeping the raw background available.
- Selected-Qr rod ROI mode can be selected in detector view or caked view. Q-space view disables it because the ROI is defined from detector-space Qr/Qz support.
- In detector view, selected-Qr rod ROI mode displays the detector-native Qr/Qz support mask directly from detector Q maps, clipped by Qz, phi, finite-Q, and detector validity. This detector-native mask is visual/drag support only; it does not provide the plotted Qz profile. It does not show the old detector `2theta/phi` angular ROI while rod mode is enabled.
- In caked view, the selected-Qr rod ROI overlay uses the caked Qr/Qz/phi mask. Selected-Qr rod Qz profiles are always integrated from the caked `2theta/phi` result, even while the main view is detector mode.
- Use `Mirror +/-phi band` in selected-Qr rod ROI mode when the intended integration is a symmetric high-`|phi|` band such as `72.5 <= |phi| <= 85`; normal rectangular and signed-interval integrations ignore this flag. Fresh selected-Qr rod mode uses `phi_min=-90` and `phi_max=90` unless a restored/custom selected-rod phi range is already active. Selected-Qr drag Qz bounds use the active view's rod support, so detector-view drags use detector pixels and caked-view drags use caked-bin support.
- Use `Include rod shape` in selected-Qr rod ROI mode when the selected Qr/Qz group's shape support should clip the selected rod mask in addition to the numeric Qr band. The toggle is saved as `analysis_range.include_selected_qr_rod_shape`.
- `Caked image intensity` controls full caked-image pixels plus standard radial/azimuthal integrations. It defaults to support-normalized density and can show raw accumulated caked-bin signal for inspection. The choice is saved as `analysis_range.caked_intensity_mode`, and changing it repaints the main caked figure with the selected pixel semantics.
- Changing the standard integration region no longer rescales the main caked image. The radial 2theta and azimuthal phi 1D plots still recompute and autoscale to the selected region.
- Peak-fit results in Analyze are shown as a monospaced table with center, Gaussian FWHM, Lorentzian FWHM, Gaussian/Lorentzian mixture percent, model, and RMSE for radial and azimuthal fits. Pseudo-Voigt percentages use area-normalized Gaussian/Lorentzian equations, so the Lorentzian value is the fitted Lorentzian area fraction.
- `Rod profile intensity` defaults to `Raw accumulated intensity` when fresh Selected-Qr rod ROI mode is enabled in detector view, and to `Intensity density (support-normalized)` in caked view. Restored or user-edited values are preserved and saved as `analysis_range.rod_profile_intensity_mode`.
- Multiple Qr rods can be selected at once with the rod checkboxes. Checkbox selection is stored in displayed rod order in `analysis_range.selected_qr_rod_keys`; `analysis_range.selected_qr_rod_key` remains the first selected rod for legacy state compatibility.
- When selected-Qr rod ROI mode is active, the 1D analysis panel shows one Qz subplot per selected rod, stacked vertically. The standard radial/azimuthal integration layout is restored when rod mode is turned off.
- `Delta Qr width (A^-1)` is the full selected-rod width in the GUI and saved state. Current saves include `analysis_range.delta_qr_width_mode="full_width"`; older saved half-width values are converted to full width on load.
- Selected-Qr rod Qz controls default to `0..5`; the Qz slider lower bound stays at `0`, and the upper bound tracks the largest positive Qz candidate from the active caked axes or detector Q maps.
- Manual Qr/Qz group selection now works in both detector view and caked `(2theta, phi)` view. The selectable Qr/Qz groups come from active CIF/lattice simulation hit-table state, not from view-filtered live peak rows.
- `Include generated disordered-phase Qr refs` is on by default beside the stacking controls. When enabled with a nonzero disordered stacking component, the GUI generates Qr/Qz references from the active PbI2 CIF, publishes them as `disordered_phase` picker rows, and does not add disordered intensity to the rendered simulation image.
- Enable `Include packaged 6H Qr refs` beside the stacking controls only when the manual picker should also include the legacy packaged 6H reference Qr/Qz groups. The control is off by default, is saved in GUI state, and merges duplicate numeric Qr/Qz groups into one displayed and pickable group with combined detector candidates.
- Use Match tab > `Place Background Qr Set` to click any background peak without selecting a simulated Qr/Qz group. The click still runs local peak-top refinement, saves the measured `(2theta, phi)` as the label, omits HKL identity, exports/imports with manual pairs, and is ignored by geometry solving.
- In the Match tab, `Add All Qr Set Peaks` saves placements for every enabled Qr/Qz selector group by seeding each background refinement from the refined simulated spot position, matching the same measured-point path used by manual clicks. Auto-add ignores the origin `(0,0,0)` reflection. For non-`00l` Qr sets, auto-add also requires branch 0 and branch 1 to keep the same-frame branch-pair length predicted by the refined simulated spots; `00l` sets keep their collapsed/single-branch behavior. `Remove Qr Set Peaks` deletes saved placements for the enabled Qr/Qz groups on the current background.
- To move one already placed Qr/Qz background point, arm manual picking, click the saved background point, then click the new local peak. Or enable the visible `Drag Move Placed Peaks` checkbox next to the manual pick control and drag a saved point directly. The replacement keeps that point's Qr/Qz identity and refines locally again before saving. Use `Click Remove Placed Peaks` for click-to-remove mode; the old auto-search radius slider is no longer shown in the peak tools.
- HKL picking now follows the same current-view candidate frame as Qr/Qz picking. In caked view, simulated HKL targets agree with the rendered caked simulation spot, not detector/display aliases.
- Main viewport stays embedded in Tk through Matplotlib.

## Manual Geometry Status

- Fixed: caked-view Qr/Qz picking no longer lets an already saved placement intercept normal pick clicks. Move/edit behavior now requires the explicit drag-move tool, so selecting a later set such as `006` is not redirected to an earlier saved set such as `003`.
- Fixed: detector-view manual Qr/Qz placements now save the projected caked `(2theta, phi)` cache values, and imported legacy GUI states missing those fields are backfilled when a caked transform is available.
- Fixed: headless `fit-geometry` now performs the same legacy manual-pair caked-coordinate backfill before preparing the geometry fit and returns an updated saved-state snapshot when rows were repaired.
- Added: optional PbI2 6H reference Qr/Qz groups are generated from the packaged 6H CIF when `Include packaged 6H Qr refs` is enabled and `w1` is nonzero; duplicate numeric groups merge before listing and manual picking.
- Added: `Place Background Qr Set` stores locally refined background-only reference peaks with `2theta,phi` labels instead of HKL identity; these rows stay portable with manual placement exports and diagnostic notebooks while staying disabled for geometry solving.
- Added: auto-add uses a same-frame branch-pair length restraint for non-`00l` Qr sets, skips `(0,0,0)`, runs a final local refinement pass after placement, and can parallelize that refinement on CPU workers.
- Added: placed Qr/Qz background peaks can be moved and locally refined again, and whole Qr sets can be removed from one click through the Match-tab tools under `Pick Qr Sets`.
- Added: selected-Qr rod ROI mode has an `Include rod shape` option that clips selected rod masks and profiles with the selected Qr/Qz group's shape support.
- Fixed: detector-view selected-Qr rod ROI mode now displays the detector-native Qr/Qz support mask for overlay/drag only, while Qz profiles stay on the caked `2theta/phi` integration path.
- Added: multiple selected Qr rods plot as vertically stacked Qz profiles, with union masks used only for overlay and drag support.
- Changed: selected-Qr rod selection uses checkboxes, detector-view rod profiles default to raw accumulated intensity, and `Delta Qr width` is stored as full width with legacy half-width migration.
- Default: diffuse background subtraction is still off by default, including notebook startup state. Enable `Use before fit/pick` only when subtraction should feed picking and fitting.

For deeper physical and implementation detail, use:

- [GUI workflow and views](simulation_and_fitting.md#gui-workflow-and-views)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)
- [Mosaic-shape fitting and image refinement](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- [Ordered-structure and structure-factor fitting](simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
