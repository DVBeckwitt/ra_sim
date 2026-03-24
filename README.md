<div align="center">
  <h1>RA-SIM</h1>
  <p><strong>DWBA forward simulation and refinement for 2D oriented powder diffraction</strong></p>
  <p>
    RA-SIM is research software for modeling grazing-incidence diffraction on area
    detectors, calibrating detector geometry, and refining mosaic, stacking-disorder,
    and crystallographic parameters against experimental images.
  </p>
  <p>
    <a href="https://github.com/DVBeckwitt/ra_sim/actions/workflows/ci.yml">
      <img src="https://img.shields.io/github/actions/workflow/status/DVBeckwitt/ra_sim/ci.yml?label=CI" alt="CI status" />
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/github/license/DVBeckwitt/ra_sim" alt="License" />
    </a>
  </p>
  <p>
    <a href="#installation">Installation</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#usage">Usage</a> •
    <a href="#screenshots">Screenshots</a> •
    <a href="#development">Development</a>
  </p>
</div>

![RA-SIM simulation view](docs/images/simulation.png)

> [!NOTE]
> RA-SIM is active research software. The repository does not bundle raw detector
> data, so you will need to point the configuration files at local `.osc`, `.poni`,
> `.cif`, and measured-peak inputs before launching the workflows.

## Highlights

- Full 2D DWBA forward modeling with refraction, footprint, divergence, and detector
  geometry handling for grazing-incidence diffraction.
- Calibration workflow for fitting hBN ring ellipses and transferring detector
  geometry into simulation runs.
- Refinement controls for mosaic orientation distributions, stacking disorder,
  lattice parameters, occupancies, and Debye-Waller terms.
- Desktop GUI for image-based analysis plus CLI entry points for headless
  simulation and calibrant fitting.

## Installation

RA-SIM supports Python 3.10+.

```bash
git clone https://github.com/DVBeckwitt/ra_sim.git
cd ra_sim
python -m pip install -e .
```

If you prefer the installed console script, the package also exposes `ra-sim`.

> [!TIP]
> The GUI uses Tkinter. Most Python distributions include it already, but some
> Linux environments require installing the system `tk` package separately.

## Configuration

RA-SIM reads project settings from `config/`.

1. Create a local paths file from the example.
2. Update the file so it points at your experiment-specific data.
3. Optionally move your config out of the repository and set `RA_SIM_CONFIG_DIR`
   to that folder.

```bash
cp config/file_paths.example.yaml config/file_paths.yaml
# Windows PowerShell: Copy-Item config/file_paths.example.yaml config/file_paths.yaml
```

At minimum, review these files:

- `config/file_paths.yaml` for detector images, `.poni` geometry, CIF input,
  measured peaks, and output artifacts.
- `config/hbn_paths.yaml` for the headless hBN ellipse-fit workflow.
- `config/instrument.yaml` and `config/materials.yaml` for instrument defaults
  and material-specific constants.

## Usage

Common entry points:

| Task | Command |
| --- | --- |
| Interactive launcher | `python -m ra_sim` |
| Main GUI | `python -m ra_sim gui` |
| Windows launcher | `run_ra_sim.bat` |
| Calibrant GUI | `python -m ra_sim calibrant` |
| Headless simulation | `python -m ra_sim simulate --out output.png` |
| Headless hBN ellipse fit | `python -m ra_sim hbn-fit` |

A typical workflow looks like this:

1. Fit detector geometry with the calibrant workflow.
2. Launch the main GUI and load your experimental background.
3. Match 2D detector features first, then validate radial and azimuthal
   integrations.
4. Refine mosaic, stacking, and structural parameters and save parameter
   snapshots for reproducibility.

Further GUI notes: [docs/gui_views.md](docs/gui_views.md)

<details>
<summary>Advanced CLI examples</summary>

```bash
# Launch the GUI through the installed script
ra-sim gui

# Save a headless simulation image
python -m ra_sim simulate --out output.png --samples 2000 --image-size 3000

# Use the default hBN paths file
python -m ra_sim hbn-fit --load-bundle

# Use a custom hBN paths file
python -m ra_sim hbn-fit --paths-file /path/to/custom_hbn_paths.yaml

# Refit the hBN workflow at full resolution
python -m ra_sim hbn-fit --load-bundle --highres-refine
```

</details>

## Screenshots

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/simulation.png" alt="Simulation view" width="100%" /><br />
      <strong>Simulation</strong><br />
      Main diffraction workspace for matching detector-space features.
    </td>
    <td align="center" width="50%">
      <img src="docs/images/integration.png" alt="Integration view" width="100%" /><br />
      <strong>Integration</strong><br />
      Reduced-coordinate views for radial and azimuthal comparisons.
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/calibrant.png" alt="Calibrant view" width="100%" /><br />
      <strong>Calibrant</strong><br />
      Ring fitting and geometry transfer from hBN calibrant data.
    </td>
    <td align="center" width="50%">
      <img src="docs/images/parameters.png" alt="Parameters view" width="100%" /><br />
      <strong>Parameters</strong><br />
      Central control surface for geometry, mosaic, beam, and structure refinement.
    </td>
  </tr>
</table>

## Project Layout

- `ra_sim/` contains the core simulation, fitting, GUI, and CLI code.
- `config/` contains default instrument, material, and path configuration files.
- `docs/` contains GUI notes, figures, and supporting technical writeups.
- `tests/` contains regression tests for the simulation, fitting, GUI helpers,
  and CLI behavior.

## Development

The CI workflow runs on Python 3.10 through 3.13. Local checks:

```bash
ruff check .
pytest -q
python -m mypy ra_sim/config ra_sim/simulation ra_sim/fitting ra_sim/gui
```

Set `RA_SIM_DEBUG=1` to enable verbose debug output and extra diagnostics.

## Citation

If RA-SIM contributes to published work, please cite the accompanying manuscript:

> D. V. Beckwitt *et al.*, "Quantitative simulation and refinement of diffraction from 2D oriented powders," (in preparation, 2024).

## License

RA-SIM is distributed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for details.
