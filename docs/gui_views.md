# GUI Views Reference

These are the main RA-SIM work areas in the order they are usually used.

## 1. Simulation

The simulation view is the main detector-space workspace. Use it to answer the
first-order questions:

- Are the main arcs, caps, and ring fragments in the right place?
- Is the detector geometry roughly correct?
- Are the broadening and intensity trends physically believable?

This is the best place to establish global agreement before checking reduced
coordinates.

## 2. Integration

The integration views reduce the 2D detector image into easier diagnostics:

- radial intensity versus `2theta`
- azimuthal intensity versus detector angle
- caked maps for localized mismatch

Use these views after the detector-space pattern is roughly aligned. They are
the quickest way to see whether the model has the correct radial positions,
azimuthal widths, and intensity balance.

## 3. Calibrant

The calibrant view is the hBN ellipse-fitting workflow used to estimate beam
center, detector tilt, and related geometry terms from ring data.

Typical use:

1. Load a calibrant frame and any associated dark/background image.
2. Mark or edit ring points.
3. Fit ellipses and refine the geometry.
4. Save the bundle and use it as the starting point for the main simulation.

Use this view when detector geometry is uncertain or when you want a stronger
initial geometry before refining diffraction parameters.

## 4. Parameters

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
