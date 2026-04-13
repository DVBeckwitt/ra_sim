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
- Parameters panel: controls for geometry, mosaic, stacking, occupancy, beam, and fit settings

## Typical Workflow

1. Start from the calibrant workflow if detector geometry is uncertain.
2. Launch the main GUI and load the experimental background.
3. Align detector-space features before trusting reduced-coordinate plots.
4. Use integration views to validate widths, radial positions, and intensity balance.
5. Refine mosaic, stacking, and structure after geometry is stable.
6. Save parameter snapshots so iterations stay reproducible.

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
- Detector-space agreement comes first; 1D views are validation tools, not the primary fitting target.
- The default main viewport is the Tk-native canvas path, with a Matplotlib fallback available through `RA_SIM_PRIMARY_VIEWPORT=matplotlib`.

For deeper physical and implementation detail, use:

- [GUI workflow and views](simulation_and_fitting.md#gui-workflow-and-views)
- [Geometry fitting from picked spots](simulation_and_fitting.md#geometry-fitting-from-picked-spots)
- [Automatic background peak matching](simulation_and_fitting.md#automatic-background-peak-matching)
- [Mosaic-shape fitting and image refinement](simulation_and_fitting.md#mosaic-shape-fitting-legacy-mosaic-width-fitting-and-image-space-refinement)
