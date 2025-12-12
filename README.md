# RA Simulation

RA Simulation is an open-source Distorted-Wave Born Approximation (DWBA) forward model for quantitative analysis of diffraction from two-dimensional oriented powders. The code refines full area-detector images, jointly modeling detector geometry, mosaic orientation distributions, stacking disorder, and crystallographic structure factors. It accompanies the manuscript "Quantitative simulation and refinement of diffraction from 2D oriented powders."

## Features

- **DWBA forward model** with refraction, footprint and divergence corrections for grazing-incidence diffraction.
- **Geometry and detector calibration** using a 3D powder standard and transfer to samples.
- **Mosaic orientation distributions** parameterized by pseudo-Voigt functions for hybrid ring–cap patterns.
- **Structure-factor refinement** including fractional occupancies, atomic positions, and anisotropic Debye–Waller factors.
- **Tk-based GUI** for loading images, running simulations and refining parameters.
- **Debug utilities** and helpers for stacking-fault analysis.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<user>/ra_sim.git
cd ra_sim
pip install -e .
```

## Quick start

Launch the GUI:

```bash
python -m ra_sim gui
# or
python main.py
```

The application loads example images specified in `config/dir_paths.yaml`. Refine detector geometry with a calibrant, then adjust mosaic and structural parameters to fit sample data. The `tests` folder contains unit tests that can be run with `pytest`.

## Command-line hBN ellipse fitting

You can run the hBN ellipse fitting workflow without the GUI through the project CLI. The workflow understands a YAML/JSON paths file so you do not have to repeat calibrant and dark frame paths each time.

1. Update `config/hbn_paths.yaml` with your calibrant, dark, and optional bundle/profile paths (or point `--paths-file` to a custom YAML/JSON).
2. Launch the workflow at full 3000×3000 resolution from the command prompt:

```bash
# Use the default config/hbn_paths.yaml and process at full resolution
python -m ra_sim hbn-fit

# Load the bundle listed in the paths file without retyping its path
python -m ra_sim hbn-fit --load-bundle

# Recompute a fresh background/fit at full resolution using stored ellipses as starting guesses
python -m ra_sim hbn-fit --load-bundle --highres-refine --osc /path/to/calibrant.osc --dark /path/to/dark.osc

# Collect a fresh set of 5 points on each of the 5 rings even when a bundle exists
python -m ra_sim hbn-fit --reclick --osc /path/to/calibrant.osc --dark /path/to/dark.osc

# Supply an alternate paths file
python -m ra_sim hbn-fit --paths-file /path/to/custom_hbn_paths.yaml
```

When a bundle NPZ is provided in the paths file (or via `--load-bundle`), `--highres-refine` will rebuild the background and refit using the saved ellipses as starting guesses at full resolution.

**Why five clicks per ring?** An unconstrained ellipse has five free parameters (center `xc, yc`, semi-axes `a, b`, and rotation `θ`), so you need at least five non-collinear points to uniquely define it. If you click fewer than five points the fit becomes underdetermined; the workflow therefore requires five points unless you are reusing a saved click profile or bundle that already contains the necessary geometry.

After each run, the overlay figure shows the fitted ellipses on top of the background-subtracted image and annotates the fitted parameters (xc, yc, a, b, θ). The saved fit profile also records an estimated detector tilt; the GUI and the `simulate` CLI subcommand will use this tilt as their starting Rot1/Rot2 defaults the next time you launch a simulation.

## Troubleshooting

Set `RA_SIM_DEBUG=1` to enable verbose logging and additional diagnostic plots:

```bash
export RA_SIM_DEBUG=1  # Linux/macOS
# or
set RA_SIM_DEBUG=1     # Windows CMD
```

## Citation

If you use this software in published work, please cite the corresponding paper:

> D. V. Beckwitt *et al.*, "Quantitative simulation and refinement of diffraction from 2D oriented powders," (in preparation, 2024).

## License

This project is distributed under the terms of the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

