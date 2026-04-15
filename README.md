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
      <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+" />
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/github/license/DVBeckwitt/ra_sim" alt="License" />
    </a>
  </p>
  <p>
    <a href="#getting-started">Getting Started</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#usage">Usage</a> •
    <a href="#documentation-map">Documentation</a> •
    <a href="#development">Development</a> •
    <a href="#troubleshooting">Troubleshooting</a>
  </p>
</div>

![RA-SIM simulation view](docs/images/simulation.png)

> [!NOTE]
> RA-SIM does not bundle raw detector data or private experiment files. You
> must point the configuration at local `.osc`, `.poni`, `.cif`, measured-peak,
> and artifact paths before the GUI and headless workflows can run successfully.

## Table of Contents

- [What RA-SIM Does](#what-ra-sim-does)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Runtime Toggles](#runtime-toggles)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Detector-to-Angle Backend](#detector-to-angle-backend)
- [Documentation Map](#documentation-map)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## What RA-SIM Does

RA-SIM is a local desktop and headless toolkit for diffraction workflows built
around 2D detector images. It combines detector-space forward simulation,
geometry calibration, GUI-assisted fitting, and reproducible headless reruns
from saved GUI states.

The repository is shaped around three core workflows:

- detector-space forward simulation for 2D grazing-incidence diffraction
- hBN calibrant fitting to estimate beam center, tilt, and geometry hints
- iterative geometry, mosaic-shape, and structural refinement against measured images

### Highlights

- Full 2D grazing-incidence forward modeling with refraction, footprint,
  divergence, detector geometry handling, and multiple optics transport modes.
- Interactive Tk/Matplotlib GUI for detector-space alignment, integration
  checks, and parameter refinement.
- In-repo exact-cake detector-to-angle backend for caked views, manual QR/HKL
  picks, and fit-space point conversion with no `pyFAI` dependency.
- Headless CLI entry points for simulation, hBN fitting, geometry fitting, and
  geometry-locked mosaic-shape fitting.
- Config-driven workflow with versioned templates under `config/` and machine-local
  overrides kept out of git.
- Extensive regression coverage across CLI, config loading, simulation, fitting,
  and GUI helper paths.

## Tech Stack

- **Language**: Python 3.11+
- **Core numerics**: NumPy, SciPy, Numba
- **Diffraction/science stack**: Dans_Diffraction, PyCifRW, spglib, xraydb
- **Detector-angle backend**: in-repo flat exact-cake integrator with sparse LUT reuse
- **GUI**: Tkinter + Matplotlib
- **Image/data tools**: Pillow, OpenCV, scikit-image, pandas, openpyxl
- **Optional acceleration**: Qt + PyQtGraph fast viewer path
- **Packaging**: setuptools with console scripts for `ra-sim` and `ra-sim-dev`
- **Validation**: pytest, ruff, mypy
- **CI**: GitHub Actions on Python 3.11, 3.12, and 3.13

## Prerequisites

- Python 3.11 or newer
- Git if you plan to clone and update the repository
- Local experiment inputs referenced by config:
  - detector/background `.osc` or GUI image inputs
  - detector geometry `.poni`
  - material `.cif`
  - measured peaks and output artifact locations
- Tkinter for GUI entry points
  - Windows and macOS Python builds usually include it
  - Linux often needs a system package such as `python3-tk` or `python3.11-tk`
- Optional Qt binding plus `pyqtgraph` if you want the fast-viewer acceleration path

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/DVBeckwitt/ra_sim.git
cd ra_sim
```

### 2. Install RA-SIM

Preferred developer-friendly setup:

```bash
python -m ra_sim.dev bootstrap
```

That helper upgrades `pip`, installs the package in editable mode, and brings
in the dev tooling used by CI.

Manual fallback:

```bash
python -m pip install --group dev -e .
# Older pip fallback:
# python -m pip install -e ".[dev]"
```

After install, these console scripts are available:

- `ra-sim` for the shared launcher and headless CLI
- `ra-sim-dev` for bootstrap, checks, tests, and lockfile refreshes

### 3. Create Local Config Files

Copy the versioned templates and keep the machine-local copies untracked:

```bash
cp config/file_paths.example.yaml config/file_paths.yaml
cp config/hbn_paths.example.yaml config/hbn_paths.yaml
# Windows PowerShell:
# Copy-Item config/file_paths.example.yaml config/file_paths.yaml
# Copy-Item config/hbn_paths.example.yaml config/hbn_paths.yaml
```

### 4. Point Config at Your Data

At minimum, update the local config with paths to your experiment files:

- `config/file_paths.yaml`
  - detector backgrounds
  - `.poni` geometry
  - active CIF
  - measured peaks
  - artifact output locations
- `config/hbn_paths.yaml`
  - calibrant OSC
  - dark frame OSC
  - optional bundle and click-profile locations

### 5. Launch RA-SIM

Interactive launcher dialog:

```bash
python -m ra_sim
```

Direct simulation GUI:

```bash
python -m ra_sim gui
```

Installed-script equivalent:

```bash
ra-sim gui
```

## Configuration

RA-SIM resolves configuration from:

1. `RA_SIM_CONFIG_DIR` when set
2. the repository `config/` directory otherwise

This matters for relative paths:

- when using the repository `config/`, relative paths resolve from the repo root
- when using `RA_SIM_CONFIG_DIR`, relative paths resolve from that custom config directory

### Config Files

| File | Purpose |
| --- | --- |
| `config/file_paths.example.yaml` | Versioned template for simulation and GUI file inputs |
| `config/hbn_paths.example.yaml` | Versioned template for hBN calibrant CLI inputs |
| `config/instrument.yaml` | Detector, beam, fit, and runtime defaults |
| `config/materials.yaml` | Material definitions and shared constants |
| `config/debug.yaml` | Debug/log/cache toggles |
| `config/dir_paths.yaml` | Default output and working directories |

### Local-Only Files

These should stay untracked:

- `config/file_paths.yaml`
- `config/hbn_paths.yaml`

If you need per-machine or per-dataset config outside the repository, point
`RA_SIM_CONFIG_DIR` at another folder that contains the same config filenames.

### Key Path Settings

Common keys in `config/file_paths.yaml`:

- `simulation_background_osc_files`
- `geometry_poni`
- `cif_file`
- `measured_peaks`
- `parameters_file`
- `gui_background_image`
- `overlay_output`

Common keys in `config/hbn_paths.yaml`:

- `calibrant`
- `dark`
- `beam_center`
- `bundle`
- `click_profile`
- `fit_profile`

### Default Output Locations

From `config/dir_paths.yaml`:

| Key | Default |
| --- | --- |
| `downloads` | `~/Downloads` |
| `overlay_dir` | `~/.cache/ra_sim/overlays` |
| `debug_log_dir` | `~/.cache/ra_sim/logs` |
| `file_dialog_dir` | `~/.local/share/ra_sim` |
| `temp_root` | `~/.cache/ra_sim` |

`debug_log_dir` is also where per-run artifact bundles are written as
`run_bundle_<stamp>.zip`.

## Usage

### Launcher vs Shared CLI

RA-SIM has two closely related command surfaces:

- `python -m ra_sim`
  - lightweight launcher first
  - opens the startup dialog when called without subcommands
  - still forwards non-launcher subcommands like `simulate` and `hbn-fit` into the shared CLI
- `ra-sim` or `python -m ra_sim.cli`
  - exposes the full shared CLI directly

Because of that split:

- `python -m ra_sim --help` shows launcher-oriented help only
- `python -m ra_sim simulate --help` works
- `python -m ra_sim.cli --help` or `ra-sim --help` is the clearest way to inspect the full CLI

### Common Entry Points

| Task | `python -m ra_sim ...` | Installed script |
| --- | --- | --- |
| Startup chooser | `python -m ra_sim` | `ra-sim` |
| Main GUI | `python -m ra_sim gui` | `ra-sim gui` |
| Calibrant GUI | `python -m ra_sim calibrant --bundle bundle.npz` | `ra-sim calibrant --bundle bundle.npz` |
| Mosaic visualizer | `python -m ra_sim mosaic` | `ra-sim mosaic` |
| Headless image render | `python -m ra_sim simulate --out output.png` | `ra-sim simulate --out output.png` |
| Headless hBN fit | `python -m ra_sim hbn-fit --load-bundle` | `ra-sim hbn-fit --load-bundle` |
| Headless geometry fit from saved GUI state | `python -m ra_sim fit-geometry state.json` | `ra-sim fit-geometry state.json` |
| Headless geometry + mosaic-shape fit | `python -m ra_sim fit-mosaic-shape state.json` | `ra-sim fit-mosaic-shape state.json` |

### Typical Workflow

1. Run the calibrant workflow if detector geometry is unknown or stale.
2. Launch the main GUI and load the experimental background.
3. Match detector-space features first.
4. Use radial, azimuthal, and caked views to validate the alignment.
5. Refine mosaic, stacking, and structural parameters once geometry is stable.
6. Save parameter snapshots or GUI state files so the run can be reproduced headlessly.

### CLI Examples

```bash
# Launch the simulation GUI
python -m ra_sim gui

# Save a headless simulation image
python -m ra_sim simulate --out output.png --samples 2000 --image-size 3000

# Load the calibrant GUI with an existing bundle
python -m ra_sim calibrant --bundle artifacts/hbn_ellipse_bundle.npz

# Run the hBN workflow using defaults from config/hbn_paths.yaml
python -m ra_sim hbn-fit --load-bundle

# Reload a specific bundle and perform a high-resolution refine pass
python -m ra_sim hbn-fit --load-bundle artifacts/hbn_ellipse_bundle.npz --highres-refine

# Fit detector geometry from a saved GUI state
python -m ra_sim fit-geometry saved_state.json

# Fit geometry and then mosaic shape from a saved GUI state
python -m ra_sim fit-mosaic-shape saved_state.json
```

<details>
<summary>Optics transport modes</summary>

RA-SIM currently exposes two optics transport models in the GUI and headless paths:

- `Original Fast Approx (Fresnel + Beer-Lambert)`
  - uses a grazing-angle transmitted-angle approximation
  - uses Fresnel transmission factors and exponential depth attenuation
  - caches exit optics in a lookup table
  - is the default throughput-oriented mode for interactive fitting
- `Complex-k DWBA slab optics (Precise)`
  - keeps the same reflection list, `solve_q` search, structure factors, and detector projection
  - replaces entry and exit transport with a complex-`k_z` slab treatment
  - computes in-sample and exit wavevectors from the slab dispersion relation
  - uses exact Fresnel power transmission at the air/sample interfaces

The practical difference is mostly how the beam is refracted, transmitted, and
attenuated on the way into and out of the sample. It is not a separate detector
model or a different structure-factor engine.

Detector placement now uses the solved outgoing direction itself rather than a
refracted exit-angle remap. The older fast-projection path could create a
forbidden strip near the sample plane because `n_real < 1` for x-rays. The
current path keeps the optics weights in the intensity model while intersecting
the detector with the normalized solved outgoing vector.

The exact path is currently an air/sample/air slab model rather than a general
multilayer stack. Use the fast mode for throughput and the exact mode when
near-critical-angle transport matters more than speed.

</details>

<details>
<summary>Manual geometry-fit behavior</summary>

The GUI manual geometry-fit workflow keeps one objective throughout the solve:

- manual picks keep a detector-native background anchor plus the chosen simulated source identity
- saved or restored runs rebuild missing source rows from live caches, retained intersection caches, or a fresh simulation before failing
- source identity stays pinned to the saved source row or peak when possible, then falls back through q-group or HKL rebinding when reflection ordering changes
- refinement recomputes observed and simulated points in the same flat detector-derived angular space and minimizes residuals in `(2theta, phi)`
- geometry, lattice, wavelength, and shared-theta updates re-anchor both simulated source rows and measured/background maxima during the solve
- beam center, detector distance, and pixel geometry move both sides of the comparison; detector `gamma/Gamma` tilts are intentionally ignored in the detector-to-angle remap
- shared sample-rotation variables move only the simulated side
- rematching, robust reweighting, and post-polish stages are disabled for this path so it stays on the manual correspondence workflow

</details>

## Runtime Toggles

| Variable | What it does | Notes |
| --- | --- | --- |
| `RA_SIM_CONFIG_DIR` | Use an external config directory | Relative paths resolve from that directory |
| `RA_SIM_PRIMARY_VIEWPORT` | Choose the primary 2D viewport backend | Current default is `matplotlib`; set `tk_canvas` to request the Tk-native viewport |
| `RA_SIM_FAST_VIEWER` | Request the Qt/PyQtGraph fast-viewer path | Defaults to enabled when available; set `0` to keep it off |
| `RA_SIM_DEBUG` | Enable verbose diagnostics | Does not override the global logging kill switch |
| `RA_SIM_DISABLE_ALL_LOGGING` | Disable user-facing logging/debug output | Preferred master kill switch |
| `RA_SIM_DISABLE_LOGGING` | Legacy logging disable switch | Still honored for compatibility |
| `RA_SIM_DISABLE_PROJECTION_DEBUG` | Disable projection-debug artifacts | Useful when trimming debug noise |
| `RA_SIM_LOG_INTERSECTION_CACHE` | Enable retained intersection-cache logging | Separate from active runtime state |
| `RA_SIM_INTERSECTION_CACHE_LOG_DIR` | Override the retained intersection-cache log directory | Falls back to config-controlled locations otherwise |
| `NUMBA_CACHE_DIR` | Directory for Numba on-disk compilation cache | Default is `~/.cache/ra_sim/numba` unless already set by user |
| `NUMBA_DEBUG_CACHE` | Print Numba cache compilation/write activity for inspection | Set to `1` to trace cache hit/miss behavior |

For the exact logging, debug, and cache semantics, use
[`docs/debug-and-cache.md`](docs/debug-and-cache.md) and the canonical
[`docs/simulation_and_fitting.md`](docs/simulation_and_fitting.md).

### Per-Run Debug Bundles

Each full RA-SIM process run now writes one zip bundle into `debug_log_dir`
when the process exits normally:

- `run_bundle_<stamp>.zip`

The bundle is meant to keep one run's debug/cache context together. It can
include:

- geometry-fit logs
- mosaic-shape fit logs
- projection-debug JSON
- diffraction debug CSV
- retained intersection-cache dump folders
- tracked run inputs such as `.poni`, measured-peak, saved-state, and parameter files
- tracked outputs such as matched-peaks exports, saved GUI states, and headless simulation images
- `manifest.json` describing bundled files, entrypoint, and omitted inputs

To avoid copying the large raw experiment sources by default, detector `.osc`
files and material `.cif` files are tracked as omitted inputs and are not
copied into the zip.

## Architecture at a Glance

### Package Layout

- `ra_sim/simulation/`: forward simulation engine, diffraction kernel, detector geometry
- `ra_sim/fitting/`: geometry fitting, mosaic fitting, objectives, peak matching
- `ra_sim/gui/`: Tk application, controllers, overlays, runtime workflows
- `ra_sim/io/`: GUI state persistence, file parsing, OSC readers
- `ra_sim/config/`: config loading, validation, material/instrument accessors
- `ra_sim/hbn.py` and `ra_sim/hbn_geometry.py`: calibrant workflow and geometry conversion helpers
- `tests/`: regression coverage for config, CLI, simulation, fitting, and GUI helpers

### System Flow

1. Config is loaded from `config/` or `RA_SIM_CONFIG_DIR`.
2. The calibrant path can estimate beam center, tilt, and geometry hints from hBN rings.
3. The GUI or CLI assembles beam, geometry, mosaic, and material inputs.
4. The simulation engine produces detector-space predictions.
5. Fitting code compares predictions against measured peaks or images.
6. GUI/runtime code manages interaction state, retained caches, and analysis views.

## Detector-to-Angle Backend

RA-SIM no longer uses `pyFAI` for detector-to-angle conversion. The GUI caked
button, manual QR/HKL selection helpers, geometry-fit point conversion, and the
diffraction utility views all go through the repo-local
`FastAzimuthalIntegrator` compatibility wrapper in
`ra_sim/simulation/exact_cake_portable.py`.

That wrapper intentionally implements only the subset RA-SIM actually uses:

- `integrate2d(...)` for caked images
- `twoThetaArray(...)` for detector-wide `2theta` maps
- `chiArray(...)` for detector-wide raw azimuth maps

### Geometry Inputs

The exact-cake backend is flat-detector only and currently requires square
pixels. Geometry is defined by:

- detector distance `d`
- pixel size `p`
- beam center in either pixels `(center_row_px, center_col_px)` or PONI meters

When PONI values are given, the beam center is reconstructed as:

```math
\mathrm{center\_row\_px} = \frac{\mathrm{poni1\_m}}{p},
\qquad
\mathrm{center\_col\_px} = \frac{\mathrm{poni2\_m}}{p}.
```

### Point Conversion Math

For manual picks and geometry-fit correspondences, detector coordinates
`(col, row)` are converted to flat-detector `(2theta, phi)` with:

```math
dx = (col - c_x)\,p,
\qquad
dy = (c_y - row)\,p,
\qquad
r = \sqrt{dx^2 + dy^2}
```

```math
2\theta = \operatorname{degrees}\!\left(\operatorname{atan2}(r, d)\right)
```

```math
\phi = \operatorname{wrap}_{[-180,180)}\!\left(
\operatorname{degrees}\!\left(\operatorname{atan2}(dx, dy)\right)
\right)
```

where `(c_y, c_x)` is the detector center in pixels, `p` is the pixel size, and
`d` is the sample-to-detector distance.

This is intentionally flat-detector math. `gamma` and `Gamma` are ignored in
the detector-to-`(2theta, phi)` remap. They can still affect simulated detector
hits upstream, but they do not enter the fit-space angle conversion for manual
or geometric correspondence points.

### Full-Image Caking

For the caked view, RA-SIM performs exact pixel splitting from detector space
into `(2theta, phi)` bins using the same flat geometry. Internally the backend
builds detector-wide maps at pixel-center coordinates and computes:

```math
y = (row + 0.5 - c_y)\,p,
\qquad
x = (col + 0.5 - c_x)\,p
```

```math
2\theta_{\mathrm{pixel}} =
\operatorname{degrees}\!\left(\operatorname{atan2}(\sqrt{x^2+y^2}, d)\right)
```

```math
\phi_{\mathrm{raw}} =
\operatorname{degrees}\!\left(\operatorname{atan2}(y, x)\right)
```

The GUI keeps its historical display convention by remapping the raw azimuth to
the plotted caked-axis angle:

```math
\phi_{\mathrm{gui}} =
\operatorname{wrap}_{[-180,180)}\!\left(-90^\circ - \phi_{\mathrm{raw}}\right)
```

So the internal `chiArray(...)` result is the raw detector azimuth, while the
displayed caked view uses the GUI remap above.

### Performance Behavior

The exact-cake path is optimized for repeated interactive use:

- detector-wide `(2theta, raw azimuth)` maps are cached as readonly process-level
  LRU entries keyed by flat geometry plus detector shape
- exact-cake LUTs are cached as process-level LRU entries keyed by flat
  geometry, detector shape, and output binning/range
- solid-angle normalization is cached per integrator instance by detector shape
- the Numba kernel is warmed once on a tiny dummy image in a background thread
  when the GUI runtime starts
- once the GUI knows the live geometry and detector shape, it schedules an idle
  background warmup for the real detector maps, solid-angle array, and the
  default caked `1000 x 720` LUT
- auto worker selection defaults to `8` and clamps to both host CPU count and
  the number of work chunks
- repeated `integrate2d(...)` calls reuse those warmed caches instead of
  rebuilding them on the first button press

Rebuild the LUT when any of these change:

- detector distance
- beam center
- pixel size
- image shape
- `2theta` bin count or range
- `phi` bin count or range

Detector-map and LUT caches are shared across integrator instances, so GUI cache
resets and integrator recreation do not discard previously built geometry work
for the same flat-detector setup. The LUT speeds repeated full-image caking.
Sparse point conversion for geometry fitting does not use the LUT because that
path only converts selected detector coordinates rather than reintegrating the
full image.

## Documentation Map

Use these entry points depending on the question:

- [docs/index.md](docs/index.md): short navigation hub
- [docs/gui-workflow.md](docs/gui-workflow.md): operator workflow and headless equivalents
- [docs/architecture.md](docs/architecture.md): package layout and edit routing
- [docs/debug-and-cache.md](docs/debug-and-cache.md): logging, output locations, and cache policy
- [docs/troubleshooting.md](docs/troubleshooting.md): setup and config failures
- [docs/simulation_and_fitting.md](docs/simulation_and_fitting.md): canonical implementation reference

If you need exact defaults, equations, or code-map-level detail, the canonical
reference in `docs/simulation_and_fitting.md` is authoritative.

## Development

CI runs fast checks on Python 3.11 through 3.13 and the slower integration tier
on Python 3.11.

### Developer Entry Points

| Command | What it does |
| --- | --- |
| `python -m ra_sim.dev bootstrap` | Install editable package with dev tooling |
| `python -m ra_sim.dev format` | Format the current formatter frontier |
| `python -m ra_sim.dev format-check` | Check formatting on the formatter frontier |
| `python -m ra_sim.dev hooks` | Install local pre-commit hooks |
| `python -m ra_sim.dev lint` | Run `ruff check .` |
| `python -m ra_sim.dev typecheck` | Run the current mypy frontier |
| `python -m ra_sim.dev test-fast` | Run the `fast` pytest tier |
| `python -m ra_sim.dev test-integration` | Run the slower `integration` tier |
| `python -m ra_sim.dev test-all` | Run the full pytest suite |
| `python -m ra_sim.dev check` | Run format-check + lint + fast tests + typecheck |
| `python -m ra_sim.dev lock` | Refresh `pylock.toml` |

Installed-script equivalents work too:

```bash
ra-sim-dev format-check
ra-sim-dev check
ra-sim-dev test-integration
ra-sim-dev hooks
ra-sim-dev lock
```

### Validation Notes

- `pytest` markers:
  - `fast`: quick local-feedback tier
  - `integration`: slower workflow-heavy tests
  - `benchmark`: hardware-sensitive performance coverage
- current mypy frontier targets:
  - `ra_sim/config/`
  - `ra_sim/dev.py`
  - `ra_sim/fitting/optimization_runtime.py`
  - `ra_sim/gui/_runtime/live_cache_helpers.py`

### CI and Security Automation

- CI bootstrap uses `python -m ra_sim.dev bootstrap`
- CI fast checks use `python -m ra_sim.dev check`
- integration tests run on Python 3.11
- security automation rejects tracked machine-local paths, runs `pip-audit`,
  and scans the repo with `gitleaks`

## Troubleshooting

### Tkinter Missing

Symptom:

- `python -m ra_sim gui` or `python -m ra_sim calibrant` fails because Tkinter is unavailable

Fix:

- install the system Tk package for your Python version, commonly `python3-tk` or `python3.11-tk` on Linux

### Local Path Config Missing

Symptom:

- startup or headless runs fail because detector images, `.poni`, CIFs, or measured peaks are not configured

Fix:

1. Copy `config/file_paths.example.yaml` to `config/file_paths.yaml`.
2. Copy `config/hbn_paths.example.yaml` to `config/hbn_paths.yaml` if you use the hBN workflow.
3. Update the local files for your machine and experiment.
4. Keep those overrides untracked.

### Config Outside the Repository

Symptom:

- you need different config roots per machine or dataset

Fix:

- move the config files into another folder and set `RA_SIM_CONFIG_DIR` to that folder

### hBN Bundle or Paths Confusion

Symptom:

- `python -m ra_sim hbn-fit --load-bundle` cannot find the bundle you expect

Fix:

- pass `--load-bundle /path/to/bundle.npz`
- or set `bundle` in your local `config/hbn_paths.yaml`
- or pass `--paths-file /path/to/custom_hbn_paths.yaml`

### Security Workflow Rejects Local Paths

Symptom:

- CI rejects tracked files containing `/Users/`, `/home/`, or `C:\Users\`

Fix:

- keep machine-local values in ignored config files
- keep only portable examples in `config/*.example.yaml`
- scrub local absolute paths from tracked docs and JSON/YAML before pushing

Need more detail:

- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/debug-and-cache.md](docs/debug-and-cache.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

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

## Contributing

Contributor workflow, validation expectations, and local-config rules live in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If RA-SIM contributes to published work, please cite the accompanying manuscript:

> D. V. Beckwitt *et al.*, "Quantitative simulation and refinement of diffraction from 2D oriented powders," (in preparation, 2024).

## License

RA-SIM is distributed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for details.
