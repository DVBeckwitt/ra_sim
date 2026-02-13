# GUI Views Reference

This page documents the main RA-SIM GUI views in practical order:

1. Simulation
2. Phi-vs-Theta and Integration
3. Calibrant
4. Parameters

## 1) Simulation

The simulation view is the primary diffraction image workspace. It renders the simulated detector pattern so you can:

- Visually compare modeled diffraction features to measured background data.
- Check whether ring/cap intensity, arc positions, and peak structure are physically plausible.
- Iterate quickly while tuning geometry, mosaic broadening, and structural inputs.

Use this view first to establish overall agreement before fitting fine details.

## 2) Phi-vs-Theta and Integration

After simulation, use integration views to compare model vs experiment in reduced coordinates:

- **Radial integration (2theta)** shows intensity vs scattering angle.
- **Azimuthal integration (phi)** shows intensity vs detector azimuth.
- **2D caked/integration map** helps inspect selected angular regions and orientation-dependent mismatch.

In practice, this view answers:

- Are simulated peak positions aligned in 2theta?
- Are azimuthal widths/intensities correct?
- Are there region-specific mismatches hidden in the full 2D detector image?

## 3) Calibrant

The calibrant view is the hBN fitter workflow used to establish detector geometry/tilt from ring data:

- Load calibrant + dark frames.
- Pick center/points or use edit mode.
- Fit rings/ellipses, refine, and optimize tilt.
- Save/load NPZ bundles and overlay outputs.

Use calibrant mode when geometry is uncertain or when you need a reliable starting point for simulation refinement.

## 4) Parameters

The parameter/control panel is where you drive the model:

- Geometry (`theta`, `Gamma`, detector rotation, `chi`, `zs`, `zb`, etc.).
- Lattice parameters (`a`, `c`).
- Mosaic broadening and beam center controls.
- Stacking probabilities and site occupancies.
- Fit toggles and save/load parameter actions.

Treat this panel as the control surface for all simulation and fitting passes. Save parameter snapshots regularly so iterations are reproducible.
