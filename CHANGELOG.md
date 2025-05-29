# Changelog

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
