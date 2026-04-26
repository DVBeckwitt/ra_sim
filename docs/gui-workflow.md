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
- Manual Qr/Qz group selection now works in both detector view and caked `(2theta, phi)` view. The selectable Qr/Qz groups come from active CIF/lattice simulation hit-table state, not from view-filtered live peak rows.
- HKL picking now follows the same current-view candidate frame as Qr/Qz picking. In caked view, simulated HKL targets agree with the rendered caked simulation spot, not detector/display aliases.
- Main viewport stays embedded in Tk through Matplotlib.

For deeper physical and implementation detail, use:

- [GUI workflow and views](simulation_and_fitting.md#gui-workflow-and-views)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)
- [Mosaic-shape fitting and image refinement](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
- [Ordered-structure and structure-factor fitting](simulation_and_fitting.md#ordered-structure-intensity-model-and-detector-space-refinement)
