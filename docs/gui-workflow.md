# RA-SIM GUI workflow

This page is the short operator map for the interactive workflow. It summarizes
the usual order of operations and points into the canonical reference when you
need the detailed implementation story.

See also:

- [docs index](index.md)
- [Architecture guide](architecture.md)
- [Troubleshooting guide](troubleshooting.md)
- [Canonical GUI workflow section](simulation_and_fitting.md#gui-workflow-and-views)

## Main Work Areas

- Simulation view: primary detector-space workspace for global geometric agreement
- Integration views: radial, azimuthal, and caked diagnostics after the detector pattern is roughly aligned
- Calibrant view: hBN ellipse-fitting workflow used to estimate beam center and detector tilt
- Parameters panel: controls for geometry, mosaic, structure factor, stacking fault, occupancy, beam, and fit settings

## Typical Workflow

1. Start from the calibrant workflow if detector geometry is uncertain.
2. Launch the main GUI and load the experimental background.
3. Align detector-space features before trusting reduced-coordinate plots.
4. Use integration views to validate widths, radial positions, and intensity balance.
5. Fit mosaic shapes only after accepting the geometry-fit cache. The implementation target is selected Qr/background pairing, local `I(phi)` extraction, Lorentzian plus Gaussian profile centering, and centered measured/simulated profile comparison.
6. Fit structure-factor terms after mosaic is stable. The implementation target is a global multi-image detector-ROI intensity fit with one shared structure-factor parameter vector and one scale nuisance per image.
7. Save parameter snapshots so iterations stay reproducible.

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
- Use Setup > Beam Controls > `Pick Beam Center` to set detector `center_x`/`center_y` directly from the loaded background image. The pick temporarily returns to detector view, shows the background if hidden, zooms around the press point, previews the same local refinement used by manual Qr/Qz placement, and commits native detector row/col on release.
- Use `Toggle Simulation` to hide or show the simulated detector/caked overlay without changing the loaded background image, fit inputs, or generated simulation data.
- Diffuse background subtraction remains off by default. When enabled, the Background Subtraction panel's `Use before fit/pick` option uses the signed subtracted background for Qr picking, auto-match, and fit comparison while keeping the raw background available.
- In selected-Qr rod ROI mode, the caked ROI mask is detector-backed: detector pixels are classified by Qr/Qz, splatted through the exact-cake LUT, and then clipped to the selected caked phi window. The analytic Qr traces remain display overlays and are only a mask fallback when the LUT context is unavailable.
- Use `Mirror +/-phi band` in selected-Qr rod ROI mode when the intended caked integration is a symmetric high-`|phi|` band such as `72.5 <= |phi| <= 85`; normal rectangular and signed-interval integrations ignore this flag. Selected-Qr drag Qz bounds use the same caked-bin to detector-pixel LUT support, so dragging one lobe stays local to that lobe.
- `Caked image intensity` controls full caked-image pixels plus standard radial/azimuthal integrations. It defaults to support-normalized density and can show raw accumulated caked-bin signal for inspection. The choice is saved as `analysis_range.caked_intensity_mode`, and changing it repaints the main caked figure with the selected pixel semantics.
- Changing the standard integration region no longer rescales the main caked image. The radial 2theta and azimuthal phi 1D plots still recompute and autoscale to the selected region.
- Peak-fit results in Analyze are shown as a monospaced table with center, Gaussian FWHM, Lorentzian FWHM, Gaussian/Lorentzian mixture percent, model, and RMSE for radial and azimuthal fits.
- `Rod profile intensity` defaults to `Intensity density (support-normalized)` for selected-Qr rod ROI plots. `Raw accumulated intensity` is available for support/compatibility inspection and is saved as `analysis_range.rod_profile_intensity_mode`.
- When selected-Qr rod ROI mode is active, the 1D analysis panel shows only the Qz rod profile. The standard azimuthal integration subplot is hidden until the mode is turned off.
- Selected-Qr rod Qz controls default to `0..5`; the Qz slider lower bound stays at `0`, and the upper bound tracks the largest positive Qz candidate calculated from the current caked 2theta extent.
- Manual Qr/Qz group selection now works in both detector view and caked `(2theta, phi)` view. The selectable Qr/Qz groups come from active CIF/lattice simulation hit-table state, not from view-filtered live peak rows.
- In the Match tab, `Add All Qr Set Peaks` saves placements for every enabled Qr/Qz selector group by seeding each background refinement from the refined simulated spot position, matching the same measured-point path used by manual clicks. Auto-add ignores the origin `(0,0,0)` reflection. For non-`00l` Qr sets, auto-add also requires branch 0 and branch 1 to keep the same-frame branch-pair length predicted by the refined simulated spots; `00l` sets keep their collapsed/single-branch behavior. `Remove Qr Set Peaks` deletes saved placements for the enabled Qr/Qz groups on the current background.
- To move one already placed Qr/Qz background point, arm manual picking, click the saved background point, then click the new local peak. Or enable the visible `Drag Move Placed Peaks` checkbox next to the manual pick control and drag a saved point directly. The replacement keeps that point's Qr/Qz identity and refines locally again before saving. Use `Click Remove Placed Peaks` for click-to-remove mode; the old auto-search radius slider is no longer shown in the peak tools.
- HKL picking now follows the same current-view candidate frame as Qr/Qz picking. In caked view, simulated HKL targets agree with the rendered caked simulation spot, not detector/display aliases.
- Main viewport stays embedded in Tk through Matplotlib.

## Manual Geometry Status

- Fixed: caked-view Qr/Qz picking no longer lets an already saved placement intercept normal pick clicks. Move/edit behavior now requires the explicit drag-move tool, so selecting a later set such as `006` is not redirected to an earlier saved set such as `003`.
- Fixed: detector-view manual Qr/Qz placements now save the projected caked `(2theta, phi)` cache values, and imported legacy GUI states missing those fields are backfilled when a caked transform is available.
- Fixed: headless `fit-geometry` now performs the same legacy manual-pair caked-coordinate backfill before preparing the geometry fit and returns an updated saved-state snapshot when rows were repaired.
- Added: auto-add uses a same-frame branch-pair length restraint for non-`00l` Qr sets, skips `(0,0,0)`, runs a final local refinement pass after placement, and can parallelize that refinement on CPU workers.
- Added: placed Qr/Qz background peaks can be moved and locally refined again, and whole Qr sets can be removed from one click through the Match-tab tools under `Pick Qr Sets`.
- Default: diffuse background subtraction is still off by default, including notebook startup state. Enable `Use before fit/pick` only when subtraction should feed picking and fitting.

For deeper physical and implementation detail, use:

- [GUI workflow and views](simulation_and_fitting.md#gui-workflow-and-views)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)
- [Mosaic-shape fitting and image refinement](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- [Ordered-structure and structure-factor fitting](simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
