# Changelog

## Unreleased (since `e11bec1` on 2026-02-13)

- **Config and path migration**
  - Renamed background-path keys in config from `dark_image`/`osc_files` to `simulation_dark_osc_file`/`simulation_background_osc_files`.
  - Added geometry solver tuning block under `instrument.fit.geometry.solver`.
  - Added Windows YAML path fallback parsing and `get_path_first(...)` in `ra_sim/path_config.py`.

- **hBN bundle geometry mapping and metadata**
  - Added canonical hBN bundle metadata keys and strict validation in `ra_sim/hbn.py`.
  - Added hBN-to-simulation geometry conversion helpers for detector angles and center mapping:
    - `convert_hbn_bundle_geometry_to_simulation`
    - `build_hbn_geometry_debug_trace`
    - `format_hbn_geometry_debug_trace`
  - Updated `load_tilt_hint` to return converted simulation-space tilt/center/distance hints.

- **Fitting and optimization**
  - Expanded `hbn_fitter/fitter.py` with uncertainty-aware ellipse refinement and point-sigma handling.
  - Added projective tilt optimization path with fallback to legacy optimization.
  - Extended bundle save/load fields to include confidence, ring sigma, optimizer metadata, coordinate-frame metadata, and compatibility keys.
  - Updated `ra_sim/fitting/optimization.py` with robust solver config, restart support, one-to-one point matching, weighted residuals, and missing-pair penalties.

- **GUI and UX updates**
  - Updated `main.py` and `ra_sim/gui/app.py` to use shared hBN geometry conversion helpers.
  - Corrected center-axis mappings used in pyFAI/intersection geometry paths.
  - Improved sliders (`ra_sim/gui/sliders.py`) with entry sync, snapping, optional range expansion, and `min`/`max` typed values.
  - Added background file browser/status controls in `main.py`.
  - Added primary CIF browse/apply workflow and dynamic occupancy control rebuild in `main.py`.
  - CIF parsing now handles numeric/scalar forms robustly and no longer multiplies `c` by 3.

- **CLI updates**
  - Updated `ra_sim/cli.py` CIF parsing (raw `a,c` values; no forced `c*3`) and tilt-hint application using converted degree fields.

- **Tests**
  - Added `tests/test_cli_cif_parse.py` for CIF numeric parsing behavior.
  - Added `tests/test_hbn_geometry_mapping.py` for geometry mapping math, metadata validation, sign handling, and startup/import consistency.

- **Notes**
  - `ra_sim/simulation/diffraction.py` and `ra_sim/simulation/diffraction_debug.py` currently show line-ending-only modifications (no content diff).

## feat: reorganize package and add advanced diffraction/mosaic features

- **Folder Structure & Organization**  
  - Moved `main.py` to the top-level directory  
  - Created new file `ra_sim/gui/update.py`  
  - Added `ra_sim.egg-info/` folder for packaging metadata  
  - Extended or reorganized code in `ra_sim/fitting`, `ra_sim/gui`, `ra_sim/io`, `ra_sim/simulation`, and `ra_sim/utils`

- **File Updates**  
  - **`file_parsing.py`**: Introduced `Open_ASC` function for loading `.asc` files and rotating them  
  - **`mosaic_profiles.py`**: Enhanced pseudo-Voigt sampling logic in `generate_random_profiles`  
  - **`diffraction.py`**: 
    - `process_peaks_parallel` now returns `(image, max_positions)` tracking max intensities for each reflection sign  
    - Additional logic for mapping reflection maxima onto the detector image  
  - **`simulation.py`**: New `simulate_diffraction` workflow integrates mosaic profiles, beam profiles, and reflection calculations  
  - **`optimization.py`**: 
    - Introduced or extended Bayesian Optimization pipeline  
    - `run_optimization` sets slider states and updates a `progress_label` for improved GUI feedback  
    - `objective_function_bayesian` dynamically sets mosaic parameters before simulating diffraction  
  - **`gui/sliders.py`**: Updated `create_slider` to accept a callback for slider value changes  
  - **`main.py` (top-level)**: 
    - Tkinter-based GUI enhancements (sliders for mosaic/detector geometry, background toggling, colorbar, etc.)  
    - Incorporates `Open_ASC` to load `.asc` background images  
    - Detects nearest Bragg peak positions by referencing new `bragg_pixel_positions`  
    - Integrates Bayesian optimization

- **Notable Changes**  
  - Usage of new `simulate_diffraction` function to handle the full mosaic+beam profile plus reflection mapping  
  - `process_peaks_parallel` updated to store reflection peak intensities in `max_positions`  
  - Code now includes additional references (e.g. `detect_blobs`) and a `view_azimuthal_radial` helper in `tools.py`  
  - The package metadata in `setup.py` and `ra_sim.egg-info/` was revised to reflect the new structure  
  - `main.py` (top-level) orchestrates the new advanced GUI-based analysis with the refactored modules  
