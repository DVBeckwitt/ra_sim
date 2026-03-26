# Changelog

## Unreleased (since `e11bec1` on 2026-02-13)

- **Simulation performance**
  - Optimized diffraction simulation with low-discrepancy antithetic beam sampling, weighted beam clustering, nominal detector culling, local-arc `solve_q`, sparse bilinear accumulation, and a fast-mode optics lookup table.

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
  - Moved the primary-CIF / diffuse-HT control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the optional CIF-weight control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers and reused a controller helper for weighted intensity recompute.
  - Moved the fit-geometry parameter checklist out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the 1D integration-range control cluster out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers.
  - Moved the top-level GUI shell and bottom status panel out of `ra_sim/gui/runtime.py` into shared GUI state/view helpers, including notebook state sync and compact console-backed status labels.
  - Wired `ra_sim/gui/runtime.py` through the shared `AppState` container for extracted GUI view state plus background/HKL interaction runtime state.
  - Extracted the remaining GUI runtime-owned state in `ra_sim/gui/runtime.py` into explicit `AppState` slices covering background selection/orientation, geometry interaction caches/artists, simulation/update/caking/peak caches, atom-site override cache bookkeeping, Bragg-Qr disabled state, hBN debug-report text, sampling count, and the caked-view override flag.
  - Reduced the remaining `runtime.py` module globals to structure-model / diffuse-HT rebuild state and the legacy `write_excel` flag.
  - Moved the structure-model / diffuse-HT bootstrap and rebuild workflow out of `ra_sim/gui/runtime.py` into `ra_sim/gui/structure_model.py`, including initial HT cache setup, weighted intensity recompute, and the live occupancy/stacking rebuild path.
  - Replaced the remaining inline primary-CIF / atom-site helper implementations in `ra_sim/gui/runtime.py` with thin delegates to `ra_sim/gui/structure_model.py` for occupancy metadata, atom-site override CIF generation, and CIF numeric parsing.
  - Expanded `ra_sim/gui/structure_model.py` to own the primary-CIF browse dialog plus the diffuse-HT open/export dialog workflow and status handling, leaving `ra_sim/gui/runtime.py` with thin delegate wrappers plus control rebuild callbacks.
  - Moved the Bragg-Qr / structure-factor pruning filter pipeline out of `ra_sim/gui/runtime.py` into `ra_sim/gui/controllers.py`, including Bragg-Qr source/L-key normalization, disabled-filter pruning, and filtered rod/HKL rebuild helpers.
  - Expanded `ra_sim/gui/structure_factor_pruning.py` with zero-arg runtime binding/callback factories plus normalized pruning / solve-q default and current-value helpers, leaving `ra_sim/gui/runtime.py` with bound pruning callbacks and a few live call sites.
  - Moved the remaining Bragg-Qr manager list-building workflow out of `ra_sim/gui/runtime.py` into `ra_sim/gui/controllers.py`, including group entry formatting, L-value mapping, and list-model/status construction for the manager window.
  - Expanded `ra_sim/gui/bragg_qr_manager.py` with shared runtime helpers for live lattice-value normalization, Bragg-Qr entry/L-value construction, active Qr-cylinder overlay entry derivation, zero-arg runtime binding/refresh/open callbacks, and manager action wiring, leaving `ra_sim/gui/runtime.py` with one bound Bragg-Qr runtime factory value plus thin manager/overlay call sites.
  - Expanded `ra_sim/gui/geometry_q_group_manager.py` with a zero-arg runtime binding/callback bundle for selector open/refresh/toggle/include/exclude/save/load/update actions, leaving `ra_sim/gui/runtime.py` with one bound geometry-selector callback bundle plus the remaining call sites.
  - Expanded `ra_sim/gui/background_manager.py` with a zero-arg runtime binding/callback bundle for background status refresh plus browse/load/switch actions, leaving `ra_sim/gui/runtime.py` with one bound background callback bundle plus the remaining call sites.
  - Added `ra_sim/gui/integration_range_drag.py` plus explicit `IntegrationRangeDragState` in `ra_sim/gui/state.py` for raw/caked canvas drag-selection of 1D integration ranges, including the runtime binding/callback bundle for clamp, detector-angle lookup, preview-mask rendering, and range-apply behavior, leaving `ra_sim/gui/runtime.py` with manual-geometry canvas branches plus top-level event dispatch.
  - Expanded `ra_sim/gui/peak_selection.py` to own the selected-peak HKL-pick toggle and raw-image click workflow, the Bragg/Ewald intersection analysis path, the ideal-center probe helper, the selected-peak config builders, and the runtime binding/callback bundle for HKL-pick labels, mode toggles, selected-peak refresh, HKL-control selection, and canvas-click selection, leaving `ra_sim/gui/runtime.py` with live GUI scalar getters plus the bound peak-selection wiring.
  - Added `ra_sim/gui/canvas_interactions.py` to own the top-level raw-image canvas event arbitration between manual-geometry placement, preview exclusion, HKL picking, and integration-range dragging, leaving `ra_sim/gui/runtime.py` with the drag-rectangle construction plus one bound canvas callback bundle.
  - Preserved the live theta value when geometry-fit background selection is applied without per-background theta overrides.
  - Kept detector hit-table collection enabled when visible manual-geometry overlays need peak metadata for redraws.
  - Added primary CIF browse/apply workflow and dynamic occupancy control rebuild in `main.py`.
  - CIF parsing now handles numeric/scalar forms robustly and no longer multiplies `c` by 3.
  - Added a top-right red/green responsiveness indicator in the simulation GUI that turns red before blocking loads/fits/updates and green again once Tk is responsive.
  - Reworked the responsiveness indicator into a canvas-anchored block that stays positioned correctly across window and canvas resizes.

- **CLI updates**
  - Updated `ra_sim/cli.py` CIF parsing (raw `a,c` values; no forced `c*3`) and tilt-hint application using converted degree fields.

- **Tests**
  - Added `tests/test_cli_cif_parse.py` for CIF numeric parsing behavior.
  - Added `tests/test_gui_structure_model.py` for the extracted structure-model helpers and rebuild workflow.
  - Extended `tests/test_gui_structure_model.py` with primary-CIF dialog workflow plus diffuse-HT open/export dialog workflow coverage in addition to the existing reload snapshot/restore and request-packaging tests.
  - Extended `tests/test_gui_controllers.py` with Bragg-Qr / structure-factor pruning filter pipeline coverage.
  - Extended `tests/test_gui_structure_factor_pruning.py` with direct coverage for the extracted structure-factor pruning / solve-q runtime-binding, default-normalization, current-value, factory, and callback workflow.
  - Extended `tests/test_gui_controllers.py` with Bragg-Qr manager entry/L-value/list-model coverage.
  - Extended `tests/test_gui_bragg_qr_manager.py` with direct coverage for the extracted Bragg-Qr runtime value helpers, overlay-entry derivation, zero-arg binding/callback factories, runtime-binding, action, and window-lifecycle helpers.
  - Extended `tests/test_gui_geometry_q_group_manager.py` with direct coverage for the extracted geometry-fit Qr/Qz selector runtime binding/callback bundle in addition to the side-effect and dialog workflow helpers.
  - Extended `tests/test_gui_background_manager.py` with direct coverage for the extracted background runtime binding/callback bundle in addition to the background status refresh and post-load/post-switch workflow helpers.
  - Added `tests/test_gui_integration_range_drag.py` for the extracted integration-range drag helpers, runtime-binding factory, and callback bundle, and extended `tests/test_gui_controllers.py` to cover the new shared drag-state slice.
  - Extended `tests/test_gui_peak_selection.py` with direct coverage for the extracted HKL-pick toggle, raw-image click selection, config builders, ideal-center probe helper, runtime binding/callback bundle, and selected-peak Bragg/Ewald intersection workflow in addition to the existing HKL lookup and selected-peak state helpers.
  - Added `tests/test_gui_canvas_interactions.py` for the extracted cross-feature canvas click/drag runtime binding factory, callback bundle, and manual-pick/preview/HKL routing behavior.
  - Added `tests/test_hbn_geometry_mapping.py` for geometry mapping math, metadata validation, sign handling, and startup/import consistency.
  - Added regression coverage for blank background-theta selections preserving the active live theta value.
  - Added regression coverage for manual-geometry overlay redraws requesting hit tables only when the overlay is visible.

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
