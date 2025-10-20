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
python main.py
```

The application loads example images specified in `config/dir_paths.yaml`. Refine detector geometry with a calibrant, then adjust mosaic and structural parameters to fit sample data. The `tests` folder contains unit tests that can be run with `pytest`.

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

