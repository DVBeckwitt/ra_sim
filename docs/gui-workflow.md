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
- Structure-factor fitting must not use GUI display normalization. It should compare detector ROI counts to ray-carried simulated intensity summed into the same ROI.
- Detector-space agreement comes first; 1D views are validation tools, not the primary fitting target.
- Use Setup > Beam Controls > `Pick Beam Center` to set the GUI beam center from the loaded background image. The pick temporarily returns to detector view, shows the background if hidden, zooms around the press point, maps the clicked detector-display point once at the displayed detector extent, then writes `center_x` as Beam Center Row and `center_y` as Beam Center Col through the visible slider and entry widgets on release. In the default 3000 px clockwise detector view, `row = display_row` and `col = detector_width - display_col`; `RA_SIM_TRACE_BEAM_CENTER=1` records the widget/update/marker/remap path to `debug/beam_center_trace.jsonl`.
- Use `Toggle Simulation` to hide or show the simulated detector/caked overlay without changing the loaded background image, fit inputs, or generated simulation data.
- Setup > Backgrounds still controls background loading, switching, and theta metadata. There is no global background-subtraction path; loaded backgrounds remain raw for display, matching, geometry fitting, manual picks, and headless fitting.
- Selected-Qr rod ROI mode can be selected in detector view or caked view. Q-space view disables it because the ROI is defined from detector-space Qr/Qz support.
- In detector view, selected-Qr rod ROI mode displays the detector-native Qr/Qz support mask directly from detector Q maps, clipped by Qz, phi, finite-Q, and detector validity. This detector-native mask is visual/drag support only; it does not provide the plotted Qz profile. It does not show the old detector `2theta/phi` angular ROI while rod mode is enabled.
- In caked view, the selected-Qr rod ROI overlay uses the caked Qr/Qz/phi mask. Selected-Qr rod Qz profiles are always integrated from the caked `2theta/phi` result, even while the main view is detector mode.
- Use `Mirror +/-phi band` in selected-Qr rod ROI mode when the intended integration is a symmetric high-`|phi|` band such as `72.5 <= |phi| <= 85`; normal rectangular and signed-interval integrations ignore this flag. Fresh selected-Qr rod mode uses `phi_min=-90` and `phi_max=90` unless a restored/custom selected-rod phi range is already active. Selected-Qr drag Qz bounds use the active view's rod support, so detector-view drags use detector pixels and caked-view drags use caked-bin support.
- Use `Include rod shape` in selected-Qr rod ROI mode when the selected Qr/Qz group's shape support should clip each selected rod mask in addition to the numeric Qr band. The toggle is saved as `analysis_range.include_selected_qr_rod_shape`.
- `Caked image intensity` controls full caked-image pixels plus standard radial/azimuthal integrations. It defaults to support-normalized density and can show raw accumulated caked-bin signal for inspection. The choice is saved as `analysis_range.caked_intensity_mode`, and changing it repaints the main caked figure with the selected pixel semantics.
- Changing the standard integration region no longer rescales the main caked image. The radial 2theta and azimuthal phi 1D plots recompute, crop their displayed curves to the selected region, and autoscale to that cropped data. Use Analyze > Fit Axes > `Log y-scale` to switch these 1D plot y-axes between linear and log intensity.
- Peak-fit results in Analyze are shown as a monospaced table with center, Gaussian FWHM, Lorentzian FWHM, Gaussian/Lorentzian mixture percent, model, and RMSE for radial and azimuthal fits. Radial and azimuthal fitting now uses one profile model only: `Mosaic mix`, an independent-width Gaussian-core/Lorentzian-tail area mixture with tail-aware residual weighting. There are no Gaussian-only, Lorentzian-only, or Pseudo-Voigt profile toggles. `Subtract linear background` is enabled by default for `Fit Selected Peaks`; it fits one local 2D plane inside the selected caked box and subtracts it only from the peak-fit data. When the subtraction is applied, the radial/azimuthal fit axes hide the raw and background-bearing 1D lines and show the corrected data curve for that axis.
- `Rod profile intensity` defaults to `Raw accumulated intensity` when fresh Selected-Qr rod ROI mode is enabled in detector view, and to `Intensity density (support-normalized)` in caked view. Restored or user-edited values are preserved and saved as `analysis_range.rod_profile_intensity_mode`.
- Multiple Qr rods can be selected at once with the extended-selection rod list. Multi-select order is the rod-list display order in `analysis_range.selected_qr_rod_keys`; `analysis_range.selected_qr_rod_key` remains the first selected rod for legacy state compatibility.
- When selected-Qr rod ROI mode is active, the 1D analysis panel shows one L subplot per selected rod, stacked vertically, when the active lattice `c` value is available. The profile data still retains the canonical Qz centers for saved-state and internal compatibility. The standard radial/azimuthal integration layout is restored when rod mode is turned off.
- Selected-Qr rod union masks are display/drag support only. Per-rod Qz profiles use per-rod caked masks and never use the union overlay mask.
- Restored GUI states can enable Selected-Qr rod ROI with nested caked projection cache signatures; the runtime normalizes that internal signature before profile-cache lookup and does not change saved-state fields.
- Restored Qr/Qz geometry rows, including PbI2-style state rows keyed as `q_group/source/m/L`, can seed Selected-Qr rod choices when fresh simulation rod inventory is not available. Saved-state fields and selected-rod key format are unchanged.
- Optional selected-Qr rod profiling is available with `RA_SIM_PROFILE_QR_ROD=1`; timing records are retained in bounded internal debug lists instead of GUI status spam.
- `Delta Qr width (A^-1)` is the full selected-rod width in the GUI and saved state. Current saves include `analysis_range.delta_qr_width_mode="full_width"`; older saved half-width values are converted to full width on load.
- Selected-Qr rod L controls are backed by the existing saved Qz bounds. The GUI displays `L Min` and `L Max` using the active lattice `c` conversion, while saved GUI state continues to write `analysis_range.qz_min` and `analysis_range.qz_max`. The lower bound stays at Qz `0` and the upper bound tracks the largest positive Qz candidate from the active caked axes or detector Q maps.
- The Selected-Qr rod ROI panel shows the live geometry values used by the caked mask/profile path, including theta, chi, psi_z, detector tilts, detector distance, and detector center. Geometry-fit results update the same underlying controls; the ROI panel does not introduce a separate geometry source.
- Manual Qr/Qz group selection now works in both detector view and caked `(2theta, phi)` view. The selectable Qr/Qz groups come from active CIF/lattice simulation hit-table state, not from view-filtered live peak rows.
- `Include generated disordered-phase Qr refs` is on by default beside the stacking controls. When enabled with a nonzero disordered stacking component, the GUI generates Qr/Qz references from the active PbI2 CIF, publishes them as `disordered_phase` picker rows, and does not add disordered intensity to the rendered simulation image.
- Disordered-phase Qr/Qz manual picks remain source-specific through detector candidates, placement, saved geometry pairs, and geometry-fit preflight. Geometry-fit diagnostics include `geometry_fit_live_handoff_patch_marker=phase4d1` and live-row source counts when the fit consumes those picker rows.
- Enable `Include packaged 6H Qr refs` beside the stacking controls only when the manual picker should also include the legacy packaged 6H reference Qr/Qz groups. The control is off by default, is saved in GUI state, and merges duplicate numeric Qr/Qz groups into one displayed and pickable group with combined detector candidates.
- Use Match tab > `Place Background Qr Set` to click any background peak without selecting a simulated Qr/Qz group. The click still runs local peak-top refinement, saves the measured `(2theta, phi)` as the label, omits HKL identity, exports/imports with manual pairs, and is ignored by geometry solving.
- In the Match tab, `Add All Qr Set Peaks` saves placements for every enabled Qr/Qz selector group by seeding each background refinement from the refined simulated spot position, matching the same measured-point path used by manual clicks. Auto-add ignores the origin `(0,0,0)` reflection. For non-`00l` Qr sets, auto-add also requires branch 0 and branch 1 to keep the same-frame branch-pair length predicted by the refined simulated spots; `00l` sets keep their collapsed/single-branch behavior. `Remove Qr Set Peaks` deletes saved placements for the enabled Qr/Qz groups on the current background.
- To move one already placed Qr/Qz background point, arm manual picking, click the saved background point, then click the new local peak. Or enable the visible `Drag Move Placed Peaks` checkbox next to the manual pick control and drag a saved point directly. The replacement keeps that point's Qr/Qz identity and refines locally again before saving. Use `Click Remove Placed Peaks` for click-to-remove mode; the old auto-search radius slider is no longer shown in the peak tools.
- HKL picking now follows the same current-view candidate frame as Qr/Qz picking. In caked view, simulated HKL targets agree with the rendered caked simulation spot, not detector/display aliases.
- Main viewport stays embedded in Tk through Matplotlib.

## Manual Geometry Status

- Fixed: Qr/Qz group identity is now displayed and exported as `m,L`, where
  `m = H^2 + H*K + K^2` and `L` is the Miller L index. Existing saved
  `q_group/source/m/L` keys remain compatible. Locked manual Qr/Qz geometry
  fits now accept current simulated rows with the same `m,L` identity even when
  the signed HKL representative changes, so the `m=1, L=10` Bi2Se3 pair
  `(-1,0,10)` / `(1,0,10)` does not fail as
  `prediction_branch_source_switched`.
- Fixed: caked-view Qr/Qz picking no longer lets an already saved placement intercept normal pick clicks. Move/edit behavior now requires the explicit drag-move tool, so selecting a later set such as `006` is not redirected to an earlier saved set such as `003`.
- Fixed: detector-view manual Qr/Qz placements now save the projected caked `(2theta, phi)` cache values, and imported legacy GUI states missing those fields are backfilled when a caked transform is available.
- Fixed: headless `fit-geometry` now performs the same legacy manual-pair caked-coordinate backfill before preparing the geometry fit and returns an updated saved-state snapshot when rows were repaired.
- Added: optional PbI2 6H reference Qr/Qz groups are generated from the packaged 6H CIF when `Include packaged 6H Qr refs` is enabled and `w1` is nonzero; duplicate numeric groups merge before listing and manual picking.
- Added: `Place Background Qr Set` stores locally refined background-only reference peaks with `2theta,phi` labels instead of HKL identity; these rows stay portable with manual placement exports and diagnostic notebooks while staying disabled for geometry solving.
- Added: auto-add uses a same-frame branch-pair length restraint for non-`00l` Qr sets, skips `(0,0,0)`, runs a final local refinement pass after placement, and can parallelize that refinement on CPU workers.
- Added: placed Qr/Qz background peaks can be moved and locally refined again, and whole Qr sets can be removed from one click through the Match-tab tools under `Pick Qr Sets`.
- Added: selected-Qr rod ROI mode has an `Include rod shape` option that clips each selected rod mask with that rod's Qr/Qz group shape support.
- Fixed: detector-view selected-Qr rod ROI mode now displays the detector-native Qr/Qz support mask for overlay/drag only, while Qz profiles stay on the caked `2theta/phi` integration path.
- Added: multiple selected Qr rods plot as vertically stacked Qz profiles, with union masks used only for overlay and drag support.
- Changed: selected-Qr rod selection uses an extended-selection list, detector-view rod profiles default to raw accumulated intensity, and `Delta Qr width` is stored as full width with legacy half-width migration.
- Fixed: async geometry-fit workers build manual-pair datasets without GUI/Tk refresh or display callbacks, resolve detector/caked entry display points from job-local data, and reuse already projected caked rows without a second projection pass.
- Fixed: saved manual caked Qr/Qz fits that only vary detector tilts `gamma` and `Gamma` now keep the fast prebuilt source-row path while using the same exact caked projector as import/redraw instead of a parallel detector-tilt projection shim. Locked two-branch Qr/Qz groups also add an explicit branch-line angle residual alongside the endpoint residuals. Status: validated through settled GUI harness runs for the Bi2Se3 and Bi2Te3 saved gamma/Gamma states; the completion text reports the visual overlay median distance against saved background picks, keeps `Dynamic fit RMS` separate from the full-beam overlay RMS, and reports finite `Branch-line angle RMS` when available.
- Default: Analyze peak fitting locally subtracts a linear 2D background plane unless `Subtract linear background` is unchecked. This does not mutate detector, caked, matching, manual-pick, geometry-fit, or headless inputs.

For deeper physical and implementation detail, use:

- [GUI workflow and views](simulation_and_fitting.md#gui-workflow-and-views)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)
- [Mosaic-shape fitting and image refinement](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- [Ordered-structure and structure-factor fitting](simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
