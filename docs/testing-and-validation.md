# Testing and validation

## Purpose

This is the central index for RA-SIM tests, validation tools, timing tools, benchmarks, fixtures, developer commands, and automation gates. Use it to find the smallest useful check for a change without reorganizing tests or relying on local generated artifacts.

Inventory in this page is based on tracked repository files from `git ls-files`. Transient files such as `__pycache__`, local logs, generated bundles, and untracked artifacts are intentionally excluded.

## What should I run?

| Change type | First check | Deeper check |
|---|---|---|
| Config, paths, or local environment | `python -m ra_sim.dev doctor` | `python -m ra_sim.dev test-fast` |
| CLI or launcher behavior | Targeted CLI tests | `python -m ra_sim.dev test-all` |
| GUI helper or controller behavior | Targeted GUI tests | `python -m ra_sim.dev test-all` |
| Geometry fitting or projection behavior | Targeted geometry tests | Integration tests plus relevant debug script |
| Mosaic selected-pair profile fitting | `python -m pytest tests/test_mosaic_shape_optimization.py` | CLI/headless mosaic smoke plus stale-cache guards after wiring |
| Structure-factor intensity fitting | Structure-factor parity tests plus synthetic ROI tests when added | Held-out image and held-out reflection-family regressions |
| Simulation or diffraction behavior | Targeted simulation tests | Full tests plus benchmark when performance-sensitive |
| Timing or performance behavior | Timing or benchmark script | Compare generated artifact summaries |
| Docs-only change | Path and link sanity | No full code run required |

## Dev validation commands

`ra_sim/dev.py` exposes these developer commands. `ra-sim-dev` is the installed-script equivalent after installation.

| Command | Purpose |
|---|---|
| `python -m ra_sim.dev bootstrap` | Install editable package with dev tooling. |
| `python -m ra_sim.dev doctor` | Report Python, config, local paths, writable dirs, Tkinter, and dev-tool health. |
| `python -m ra_sim.dev format` | Format the current formatter frontier. |
| `python -m ra_sim.dev format-check` | Check formatting on the formatter frontier. |
| `python -m ra_sim.dev check` | Run format-check, ruff, fast tests, and mypy frontier. |
| `python -m ra_sim.dev test-fast` | Run the fast pytest manifest from `ra_sim/test_tiers.py`. |
| `python -m ra_sim.dev test-integration` | Run the integration pytest manifest from `ra_sim/test_tiers.py`. |
| `python -m ra_sim.dev test-all` | Run full pytest suite. |
| `python -m ra_sim.dev test-coverage-fast` | Run fast tests with coverage and write `coverage.xml` as generated output. |
| `python -m ra_sim.dev build` | Build package distributions. |
| `python -m ra_sim.dev hooks` | Install pre-commit hooks. |
| `python -m ra_sim.dev lint` | Run `ruff check .`. |
| `python -m ra_sim.dev typecheck` | Run mypy on the current frontier. |
| `python -m ra_sim.dev lock` | Refresh `pylock.toml`. |

## Pytest tiers and markers

- `ra_sim/test_tiers.py` is the source of truth for fast and integration tier manifests.
- `pyproject.toml` defines pytest markers: `fast`, `integration`, `benchmark`, and `slow_baseline_fit`.
- `tests/conftest.py` adds tier markers from the manifests, marks benchmark-directory files as benchmark tests, and resets shared debug state around tests.
- The tests/benchmarks directory is the benchmark test area.
- Fast and integration manifests are partial in the current checkout; untiered tests still run by direct pytest or `python -m ra_sim.dev test-all`.

## Test file index

| Category | Path | Run with | What it validates | Notes |
|---|---|---|---|---|
| CLI and launcher | `tests/test_cli_cif_parse.py` | `python -m ra_sim.dev test-fast` | CIF cell parsing used by CLI paths. | Fast manifest in ra_sim/test_tiers.py. |
| CLI and launcher | `tests/test_cli_geometry_fit.py` | `python -m ra_sim.dev test-fast` | Geometry and mosaic-shape CLI parser and saved-state fit behavior. | Fast manifest in ra_sim/test_tiers.py. |
| CLI and launcher | `tests/test_cli_headless.py` | `python -m ra_sim.dev test-integration` | Headless simulation CLI request construction. | Integration manifest in ra_sim/test_tiers.py. |
| CLI and launcher | `tests/test_compare_bi2se3_reference_tool.py` | `python -m pytest tests/test_compare_bi2se3_reference_tool.py` | Bi2Se3 VESTA comparison CLI defaults and wavelength reporting. | Untiered; direct pytest or full suite. |
| CLI and launcher | `tests/test_compare_single_hkl_debug_script.py` | `python -m pytest tests/test_compare_single_hkl_debug_script.py` | Single-HKL debug script defaults and payload reporting. | Untiered; direct pytest or full suite. |
| CLI and launcher | `tests/test_dev_cli.py` | `python -m ra_sim.dev test-fast` | Dev command construction, tier markers, coverage, build, lock, and hook routing. | Fast manifest in ra_sim/test_tiers.py. |
| CLI and launcher | `tests/test_launcher_routing.py` | `python -m pytest tests/test_launcher_routing.py` | GUI, calibrant, mosaic, root launcher forwarding, and force-exit paths. | Untiered; direct pytest or full suite. |
| CLI and launcher | `tests/test_main_entrypoint_import_safe.py` | `python -m pytest tests/test_main_entrypoint_import_safe.py` | Main module import laziness and Windows helper isolation. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_config_loader.py` | `python -m ra_sim.dev test-fast` | Active config bundle loading, path resolution, examples fallback, and material copy isolation. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_data_loading_parameters.py` | `python -m ra_sim.dev test-fast` | Saved parameter compatibility after removed sampling fields and default sample counts. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_gui_state_io.py` | `python -m pytest tests/test_gui_state_io.py` | GUI state serialization, legacy load wrappers, ignored legacy background-subtraction fields, and schema errors. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_gui_state_restore_helpers.py` | `python -m pytest tests/test_gui_state_restore_helpers.py` | Saved-state file reload decisions, missing CIF warnings, background caches, and UI restore callbacks. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_hbn_path_resolution.py` | `python -m ra_sim.dev test-fast` | HBN bundle path discovery from active config, explicit YAML, and config-relative paths. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_osc_reader.py` | `python -m pytest tests/test_osc_reader.py` | OSC detector image reading, signature validation, and truncated payload errors. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_user_paths.py` | `python -m pytest tests/test_user_paths.py` | User path defaults rooted under the home directory. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_debug_controls.py` | `python -m ra_sim.dev test-fast` | Debug YAML, environment overrides, kill switches, startup log paths, and debug context state. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_debug_utils.py` | `python -m ra_sim.dev test-fast` | Numba logging helper defaults and explicit log levels. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_dependency_metadata.py` | `python -m ra_sim.dev test-fast` | Declared dependencies against runtime imports and base-install dependency boundaries. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_dev_doctor.py` | `python -m ra_sim.dev test-fast` | Doctor warning/failure behavior for local files, dev tools, and strict mode. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_import_smoke.py` | `python -m ra_sim.dev test-fast` | Package import smoke coverage across modules. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_install_prereqs.py` | `python -m pytest tests/test_install_prereqs.py` | Tkinter prerequisite imports and actionable missing-module errors. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_logging_controls.py` | `python -m pytest tests/test_logging_controls.py` | Global logging-disable aliases and debug logging gates for runtime traces and projections. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_numba_cache_env.py` | `python -m pytest tests/test_numba_cache_env.py` | Stable Numba cache env setup and lazy import boundaries for CLI/headless modules. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_parallel_utils.py` | `python -m pytest tests/test_parallel_utils.py` | Reserved CPU worker counts, Numba thread splits, and detached daemon thread pools. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_sitecustomize.py` | `python -m pytest tests/test_sitecustomize.py` | Default pycache prefix setup and explicit environment preservation. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_structure_factor_environment.py` | `python -m pytest tests/test_structure_factor_environment.py` | Structure-factor parity environment snapshots and explicit wavelength checks. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_testing_validation_index.py` | `python -m pytest tests/test_testing_validation_index.py` | Static guard for this testing and validation index. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_timing.py` | `python -m pytest tests/test_timing.py` | Timing JSONL events and GUI timing summary helpers. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_utils_notifications.py` | `python -m pytest tests/test_utils_notifications.py` | Completion chime alias selection, default sound, and background playback. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_background_peak_matching.py` | `python -m ra_sim.dev test-fast` | Simulated-to-background peak matching, one-to-one ownership, ambiguity rejection, and subpixel refinement. | Fast manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_fit_cache_controls.py` | `python -m pytest tests/test_fit_cache_controls.py` | Fit simulation-kernel cache inputs, CIF source normalization, and stale array rejection. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_correlation.py` | `python -m pytest tests/test_geometry_fit_correlation.py` | Geometry-fit active parameter selection and exported correlation artifacts. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_landscape.py` | `python -m ra_sim.dev test-integration` | Geometry landscape CLI arguments, worker defaults, sweep ranges, and saved-state reports. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_lattice_ui_helpers.py` | `python -m pytest tests/test_geometry_fit_lattice_ui_helpers.py` | Lattice fit parameter ordering, UI capture, and undo restore helpers. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_quality_baseline.py` | `python -m ra_sim.dev test-integration` | Geometry quality report rows, overlay preview, New4 provenance gates, and saved-state baselines. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_safe_runtime.py` | `python -m pytest tests/test_geometry_fit_safe_runtime.py` | Geometry-fit safe simulation callbacks and manual preflight source coverage checks. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fitter_cache_regression_gate_script.py` | `python -m pytest tests/test_geometry_fitter_cache_regression_gate_script.py` | Cache-regression gate command builder modes, optional New4 artifact handling, and Python executable use. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fitting.py` | `python -m ra_sim.dev test-integration` | Geometry fitter matching, caked residuals, point providers, cache handoff, and optimizer behavior. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_line_orientation.py` | `python -m pytest tests/test_geometry_line_orientation.py` | Detector/display orientation transforms and fit orientation recovery. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_objective_cache.py` | `python -m pytest tests/test_geometry_objective_cache.py` | Geometry objective cache signatures, center-remap reuse, reject reasons, and trace payloads. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_overlay.py` | `python -m pytest tests/test_geometry_overlay.py` | Geometry-fit overlay record construction, duplicate HKLs, orientation inversion, and caked locks. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_geometry_fit_workflow.py` | `python -m ra_sim.dev test-integration` | GUI geometry fit workflow setup, point providers, saved-state cases, and workflow slices. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_gui_geometry_q_group_manager.py` | `python -m pytest tests/test_gui_geometry_q_group_manager.py` | Q-group manager grouping, row refresh, disabled-set state, and candidate selection. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_qr_cylinder_overlay.py` | `python -m pytest tests/test_gui_qr_cylinder_overlay.py` | GUI Qr cylinder overlay controls and caked-coordinate rendering helpers. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_fit.py` | `python -m pytest tests/test_gui_runtime_geometry_fit.py` | Runtime geometry-fit callbacks, active vars, state restore, and result application. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_fitter_cache_handoff.py` | `python -m pytest tests/test_gui_runtime_geometry_fitter_cache_handoff.py` | Geometry fitter handoff caches, branch identity, manual rows, and objective cache validity. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_fitter_handoff_fast.py` | `python -m pytest tests/test_gui_runtime_geometry_fitter_handoff_fast.py` | Fast update handoff retention for Qr masks, source rows, intersections, and manual picks. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_interaction.py` | `python -m pytest tests/test_gui_runtime_geometry_interaction.py` | Runtime manual geometry interactions, point selection, and fitter preview handoff. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_preview.py` | `python -m pytest tests/test_gui_runtime_geometry_preview.py` | Geometry preview overlay generation and cached projection update behavior. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_qr_cylinder_overlay.py` | `python -m pytest tests/test_gui_runtime_qr_cylinder_overlay.py` | Runtime Qr cylinder overlay refresh, trace state, and projection invalidation. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_hbn_geometry_import_safe.py` | `python -m pytest tests/test_hbn_geometry_import_safe.py` | HBN geometry import without eager calibrant stack imports. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_hbn_geometry_mapping.py` | `python -m pytest tests/test_hbn_geometry_mapping.py` | HBN detector tilt/center sign conversions, rotation metadata, and bundle routing. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_main_geometry_fit_helpers.py` | `python -m pytest tests/test_main_geometry_fit_helpers.py` | Main-window geometry fit runtime config, parameter bounds, safe runtime overrides, and caked ROI defaults. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_manual_geometry_live_peak_cache.py` | `python -m pytest tests/test_manual_geometry_live_peak_cache.py` | Manual geometry live peak caches, native/display frame preservation, and reprojected pick candidates. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_manual_geometry_selection_helpers.py` | `python -m pytest tests/test_manual_geometry_selection_helpers.py` | Manual geometry selection, caked axes, source row identity, New4 fitting, and Qr/Qz residual guards. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_peak_sensitivity.py` | `python -m pytest tests/test_peak_sensitivity.py` | Peak sensitivity callbacks, trusted reflection provenance, and finite-difference diagnostics. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_peak_transport_validation.py` | `python -m pytest tests/test_peak_transport_validation.py` | Peak transport shortcut validation against full recompute and sparse fallback rejection. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_projection_alignment_contract.py` | `python -m pytest tests/test_projection_alignment_contract.py` | Detector, caked, refined, and Qr overlay coordinate alignment contracts. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_qr_cylinder_overlay.py` | `python -m pytest tests/test_qr_cylinder_overlay.py` | Qr cylinder trace interpolation, projection context checks, wrap breaks, and theta clipping. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_qr_grouping.py` | `python -m pytest tests/test_qr_grouping.py` | Qr grouping math, HT curve units, Bragg shape tracking, and rod intensity preservation. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_source_template_cache.py` | `python -m pytest tests/test_source_template_cache.py` | Disabled source-template and grouping cache paths when weighted events are active. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_fast_manual_point_runtime_profiles.py` | `python -m pytest tests/test_fast_manual_point_runtime_profiles.py` | Fast manual-point runtime overrides, New4 ladder lean config, and residual heartbeat throttling. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_figure_controls.py` | `python -m pytest tests/test_gui_analysis_figure_controls.py` | Analysis figure autoscale, pan, zoom, double-click reset, and cursor fallback behavior. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_peak_tools.py` | `python -m pytest tests/test_gui_analysis_peak_tools.py` | Analysis peak alignment, wrapped integration regions, curve sampling, and fit summary text. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_quick_controls.py` | `python -m pytest tests/test_gui_analysis_quick_controls.py` | Quick-control clearing for analysis integration regions and callback failures. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_visibility.py` | `python -m pytest tests/test_gui_analysis_visibility.py` | Analysis output visibility across tabs, popout windows, and missing state fallback. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_app_helpers.py` | `python -m pytest tests/test_gui_app_helpers.py` | GUI app HKL normalization, geometry-fit config helpers, and undo-state delegation. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_app_import_safe.py` | `python -m pytest tests/test_gui_app_import_safe.py` | Lazy GUI app imports, main forwarding, dunder behavior, and runtime attribute forwarding. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_background.py` | `python -m pytest tests/test_gui_background.py` | Background image loading, lazy cache reuse, shape validation, history cache, and orientation mapping. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_background_manager.py` | `python -m pytest tests/test_gui_background_manager.py` | Background manager apply, list, selection, correction mode, and operator control helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_bootstrap.py` | `python -m pytest tests/test_gui_bootstrap.py` | GUI bootstrap path setup, startup config, and launcher initialization helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_bragg_qr_manager.py` | `python -m pytest tests/test_gui_bragg_qr_manager.py` | Bragg Qr manager row lists, filters, disabled states, and refresh behavior. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_canvas_interactions.py` | `python -m pytest tests/test_gui_canvas_interactions.py` | Main canvas click, pick, drag, hover, and event binding helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_controllers.py` | `python -m pytest tests/test_gui_controllers.py` | GUI controller state transitions, callback wiring, and view update helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_diffuse_cif_toggle.py` | `python -m pytest tests/test_gui_diffuse_cif_toggle.py` | Diffuse CIF toggle state, structure switching, and associated UI callbacks. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_display_projection.py` | `python -m pytest tests/test_gui_display_projection.py` | Display projection signatures, image remap decisions, and coordinate-frame helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_fast_plot_viewer.py` | `python -m pytest tests/test_gui_fast_plot_viewer.py` | Fast plot viewer data binding, redraw scheduling, and view-state helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_fit2d_error_sound.py` | `python -m pytest tests/test_gui_fit2d_error_sound.py` | FIT2D error sound routing and notification suppression behavior. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_integration_range_drag.py` | `python -m pytest tests/test_gui_integration_range_drag.py` | Integration-range drag handles, wrapped intervals, and analysis region updates. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_lazy_runtime.py` | `python -m pytest tests/test_gui_lazy_runtime.py` | Lazy runtime loading and deferred GUI module access. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_main_figure_chrome.py` | `python -m pytest tests/test_gui_main_figure_chrome.py` | Main figure chrome, toolbar state, and figure layout helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_main_matplotlib_interaction.py` | `python -m pytest tests/test_gui_main_matplotlib_interaction.py` | Matplotlib interaction handlers for main detector, caked, and analysis views. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_ordered_structure_fit.py` | `python -m pytest tests/test_gui_ordered_structure_fit.py` | Ordered-structure fit UI state, parameter collection, and runtime callbacks. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_overlays.py` | `python -m pytest tests/test_gui_overlays.py` | Overlay record construction, visibility toggles, and drawing helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_peak_selection.py` | `python -m pytest tests/test_gui_peak_selection.py` | Peak selection records, nearest-hit logic, and selected-peak UI state. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_background.py` | `python -m pytest tests/test_gui_runtime_background.py` | Runtime background loading, switching, subtraction state, and cache invalidation. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_detector_remap_cache.py` | `python -m pytest tests/test_gui_runtime_detector_remap_cache.py` | Detector-center remap cache translation, relative hit tables, clipping safety, and full-sim parity. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_display_acceleration.py` | `python -m pytest tests/test_gui_runtime_display_acceleration.py` | Display-only fast paths, image scaling reuse, and redraw avoidance. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_fit_analysis.py` | `python -m pytest tests/test_gui_runtime_fit_analysis.py` | Runtime fit-analysis caking, ROI data, and result refresh helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_import_safe.py` | `python -m pytest tests/test_gui_runtime_import_safe.py` | Runtime import safety plus focused GUI update helpers that avoid eager Tk/Matplotlib work. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_invalidation.py` | `python -m pytest tests/test_gui_runtime_invalidation.py` | Runtime cache invalidation policy for display, prune, remap, analysis, and full simulation actions. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_mixed_update_regressions.py` | `python -m pytest tests/test_gui_runtime_mixed_update_regressions.py` | Mixed update regression cases for remap, prune, display changes, stale workers, and fitter handoff. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_optimization_scenarios.py` | `python -m pytest tests/test_gui_runtime_optimization_scenarios.py` | Runtime optimization update sequences, worker call counts, prune reuse, and center remap scenarios. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_position_preview.py` | `python -m pytest tests/test_gui_runtime_position_preview.py` | Runtime position preview signatures, rendering state, and preview invalidation. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_primary_cache.py` | `python -m pytest tests/test_gui_runtime_primary_cache.py` | Primary contribution cache reuse, pruning, rasterization, and filter signatures. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_primary_viewport.py` | `python -m pytest tests/test_gui_runtime_primary_viewport.py` | Runtime primary viewport state, figure canvas refresh, and visible-image update helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_update_actions.py` | `python -m pytest tests/test_gui_runtime_update_actions.py` | Runtime update actions for display-only, combine-only, analysis-only, prune reuse/fill, full sim, and center remap. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_update_dependencies.py` | `python -m pytest tests/test_gui_runtime_update_dependencies.py` | Dependency signature classifier for full simulation, remap, prune, combine, analysis, and display updates. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_update_trace.py` | `python -m pytest tests/test_gui_runtime_update_trace.py` | Runtime update trace fields, debug gating, and emitted action diagnostics. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_sim_signature.py` | `python -m pytest tests/test_gui_sim_signature.py` | GUI simulation signature inputs for geometry, sampling, background, and event settings. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_sliders.py` | `python -m pytest tests/test_gui_sliders.py` | Slider keyboard, wheel, home/end, and final-update event behavior. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_structure_factor_pruning.py` | `python -m pytest tests/test_gui_structure_factor_pruning.py` | Structure-factor pruning UI helpers, runtime bindings, status text, and cache preservation. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_structure_model.py` | `python -m pytest tests/test_gui_structure_model.py` | Structure model site metadata, CIF override temp files, HT caches, and diffraction input rebuilds. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_tk_primary_viewport.py` | `python -m pytest tests/test_gui_tk_primary_viewport.py` | Tk primary viewport canvas class, redraw scheduling, flush helpers, and widget fallbacks. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_views.py` | `python -m pytest tests/test_gui_views.py` | GUI view construction, theme binding, manager windows, controls, and widget state helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_window_affinity.py` | `python -m pytest tests/test_gui_window_affinity.py` | Window launch context capture, monitor metadata, geometry placement, and clamp behavior. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_ctr_fast_attenuation.py` | `python -m ra_sim.dev test-fast` | CTR attenuation for finite and semi-infinite stacks, absorption limits, and interference minima. | Fast manifest in ra_sim/test_tiers.py. |
| Simulation and diffraction engine | `tests/test_diffraction_constraints.py` | `python -m pytest tests/test_diffraction_constraints.py` | Diffraction kernel constraints, fallback rules, debug kwargs, wavelength N2, and invalid sample guards. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_inner_loop_optimizations.py` | `python -m pytest tests/test_diffraction_inner_loop_optimizations.py` | Local pixel cache and fast optics LUT parity against direct calculations. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_local_arc.py` | `python -m pytest tests/test_diffraction_local_arc.py` | Local arc windows, broad-profile fallback, mass preservation, and nominal visibility culling. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_safe_wrapper.py` | `python -m pytest tests/test_diffraction_safe_wrapper.py` | Safe diffraction wrapper defaults, backend selection, event normalization, and beam replacement rules. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_subpixel.py` | `python -m pytest tests/test_diffraction_subpixel.py` | Bilinear hit accumulation, subpixel centroids, local peak merging, and cache-to-hit-row mapping. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_tools_module.py` | `python -m pytest tests/test_diffraction_tools_module.py` | Lazy utility exports and diffraction helper reexports from tools/simulation modules. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_tools_view.py` | `python -m pytest tests/test_diffraction_tools_view.py` | Azimuthal/radial viewer preserves full cake azimuth range. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_weighted_events.py` | `python -m pytest tests/test_diffraction_weighted_events.py` | Weighted diffraction event targets, sampling, deposits, mass accounting, and deterministic bounds. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_exact_cake_portable.py` | `python -m pytest tests/test_exact_cake_portable.py` | Exact-cake detector maps, cache reuse, scalar/vector angle conversion, and transform semantics. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_exact_qspace_portable.py` | `python -m pytest tests/test_exact_qspace_portable.py` | Exact q-space detector mapping, direct/LUT parity, specular ridge behavior, and cache reuse. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_fresnel_calls.py` | `python -m pytest tests/test_fresnel_calls.py` | Fresnel and sample-term precompute calls use explicit boolean direction flags. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intensities.py` | `python -m pytest tests/test_intensities.py` | HKL intensity generation against Miller generator output. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intersection_analysis.py` | `python -m pytest tests/test_intersection_analysis.py` | Intersection arc values, nearest-index determinism, sphere shape, and sample-frame psi yaw. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intersection_cache_schema.py` | `python -m pytest tests/test_intersection_cache_schema.py` | Intersection cache layout classification, coercion, provenance extraction, and caked-angle fields. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_mosaic_intensity_conservation.py` | `python -m pytest tests/test_mosaic_intensity_conservation.py` | Mosaic normalization and pre-detector intensity conservation validation. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_peak_multiplicity_cache.py` | `python -m pytest tests/test_peak_multiplicity_cache.py` | Duplicate peak enumeration, original-row intensities, and per-sample processing. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_simulation.py` | `python -m ra_sim.dev test-integration` | Diffraction simulation uses supplied profile samples. | Integration manifest in ra_sim/test_tiers.py. |
| Simulation and diffraction engine | `tests/test_simulation_engine.py` | `python -m ra_sim.dev test-integration` | Simulation engine typed requests, kernel options, N2 buffers, safe-runner kwargs, and caches. | Integration manifest in ra_sim/test_tiers.py. |
| Materials, structure factors, and mosaic models | `tests/test_atomic_scattering_factors.py` | `python -m pytest tests/test_atomic_scattering_factors.py` | Atomic scattering factor helpers and debug table coverage. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_residual_diagnostics.py` | `python -m pytest tests/test_bi2se3_residual_diagnostics.py` | Bi2Se3 VESTA residual summary and worst-HKL diagnostics. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_structure_factor.py` | `python -m pytest tests/test_bi2se3_structure_factor.py` | Bi2Se3 structure-factor labels, atom iteration, attenuation, and occupancy validation. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_validation_doc_consistency.py` | `python -m pytest tests/test_bi2se3_validation_doc_consistency.py` | Bi2Se3 validation documentation consistency with fixture metadata. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_vesta_geometry.py` | `python -m pytest tests/test_bi2se3_vesta_geometry.py` | Bi2Se3 d-spacing and two-theta parity against VESTA references. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_vesta_reference_regression.py` | `python -m pytest tests/test_bi2se3_vesta_reference_regression.py` | Bi2Se3 VESTA reference regression guards and debug artifacts. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_ht_analytical.py` | `python -m pytest tests/test_ht_analytical.py` | Analytical Hendricks-Teller backend, custom phase expressions, L divisors, and finite-stack limits. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_profiles_rng.py` | `python -m pytest tests/test_mosaic_profiles_rng.py` | Random mosaic profile reproducibility, default shapes, antithetic samples, and clustering weights. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_shape_optimization.py` | `python -m pytest tests/test_mosaic_shape_optimization.py` | Mosaic shape fitting, selected reflection focus, theta refinement, residual stability, and family handling. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_width_optimization.py` | `python -m pytest tests/test_mosaic_width_optimization.py` | Mosaic width fitting, measured-peak requirements, geometry sources, and kernel kwargs propagation. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_raw_structure_factor_api.py` | `python -m pytest tests/test_raw_structure_factor_api.py` | Raw complex structure-factor API semantics and debug payload rules. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_single_hkl_debug_payload.py` | `python -m pytest tests/test_single_hkl_debug_payload.py` | Single-HKL debug payload terms, JSON serialization, and NaN two-theta cases. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_cache_invalidation.py` | `python -m pytest tests/test_stacking_fault_cache_invalidation.py` | Stacking-fault CIF-derived iodine z, CIF change recompute, and cache-disable behavior. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_diffuse_f2_q.py` | `python -m pytest tests/test_stacking_fault_diffuse_f2_q.py` | Stacking-fault diffuse F2 Q axes and legacy shape parity. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_redundancy.py` | `python -m pytest tests/test_stacking_fault_redundancy.py` | Stacking-fault form-factor cache reuse by radial class. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_atomic_coordinates.py` | `python -m ra_sim.dev test-fast` | Fractional-site wrapping, symmetry-image deduplication, and label preservation. | Fast manifest in ra_sim/test_tiers.py. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_sites.py` | `python -m pytest tests/test_structure_factor_sites.py` | Expanded structure-factor site list determinism and occupancy checks. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_switch_sweep.py` | `python -m pytest tests/test_structure_factor_switch_sweep.py` | Structure-factor one-knob sweep diagnostics and error summaries. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_vesta_reference_parser.py` | `python -m pytest tests/test_vesta_reference_parser.py` | VESTA TXT parser, fixture metadata, and implied wavelength checks. | Untiered; direct pytest or full suite. |
| Utilities and smoke tests | `tests/test_background_theta_helpers.py` | `python -m ra_sim.dev test-fast` | Background theta parsing, shared offsets, fit background selection, and serialization. | Fast manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_compare_intensity.py` | `python -m ra_sim.dev test-fast` | Simple intensity comparison metrics. | Fast manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_hbn_fitter_bundle_export.py` | `python -m ra_sim.dev test-integration` | HBN fitter bundle payload export and detector tilt sign conventions. | Integration manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_plot_excel_scatter.py` | `python -m pytest tests/test_plot_excel_scatter.py` | Excel scatter helper column detection and normalization. | Untiered; direct pytest or full suite. |
| Utilities and smoke tests | `tests/test_runtime_qr_selector_cache_policy.py` | `python -m pytest tests/test_runtime_qr_selector_cache_policy.py` | Qr selector cache retention policy for display, combine, analysis, prune, remap, and full simulation actions. | Untiered; direct pytest or full suite. |
| Benchmarks | `tests/benchmarks/test_benchmark_mosaic_profiles.py` | `python -m pytest tests/benchmarks/test_benchmark_mosaic_profiles.py -m benchmark` | Mosaic profile generation benchmark gate. | Benchmark marker from tests/conftest.py. |

## Fixture and reference data index

| Path | Type | Used by | Purpose |
|---|---|---|---|
| `tests/Diffuse/PbI2_2H.cif` | CIF fixture | Diffuse CIF viewer support | 2H PbI2 reference structure. |
| `tests/Diffuse/PbI2_6H.cif` | CIF fixture | Diffuse CIF viewer support | 6H PbI2 reference structure. |
| `tests/config.json` | JSON fixture | Config loader and CLI tests | Minimal configuration fixture. |
| `tests/fixtures/Bi2Se3.cif` | CIF fixture | Bi2Se3 VESTA parity tests | Bi2Se3 reference structure fixture. |
| `tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.metadata.json` | JSON fixture | Bi2Se3 VESTA parity tests | VESTA export metadata and expected validation settings. |
| `tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.txt` | TXT reference | Bi2Se3 VESTA parity tests | VESTA exported structure-factor reference rows. |
| `tests/fixtures/bi2se3_reference_comparison_baseline.csv` | CSV baseline | Bi2Se3 residual diagnostics | Baseline comparison output for no-silent-update guards. |
| `tests/local_geometry.poni` | PONI fixture | Geometry and CLI tests | Local detector geometry sample. |
| `tests/local_test.cif` | CIF fixture | CIF parsing and simulation tests | Small local structure sample. |
| `tests/local_test2.cif` | CIF fixture | CIF parsing and structure tests | Second local structure sample. |
| `tests/obs_f2_comparison.csv` | CSV reference | Structure-factor tests | Observed F2 comparison data. |
| `tests/obs_f2_comparison_3c_mapping.csv` | CSV reference | Structure-factor mapping tests | 3C mapping comparison data. |
| `tests/obs_f2_comparison_vesta.csv` | CSV reference | VESTA comparison tests | Observed F2 VESTA reference. |
| `tests/obs_intensity_comparison_vesta_configured_2H.csv` | CSV reference | Intensity comparison tests | Configured 2H intensity reference. |
| `tests/obs_intensity_comparison_vesta_pbi2_ref.csv` | CSV reference | Intensity comparison tests | PbI2 intensity reference. |
| `tests/obs_structure_factor_comparison_vesta_configured_2H.csv` | CSV reference | Structure-factor tests | Configured 2H structure-factor reference. |
| `tests/obs_structure_factor_comparison_vesta_pbi2_ref.csv` | CSV reference | Structure-factor tests | PbI2 structure-factor reference. |
| `tests/README.md` | Documentation pointer | Test directory navigation | Points to this full testing and validation index. |

## Test-adjacent support tools

| Path | Type | Command shape | Purpose |
|---|---|---|---|
| `tests/Diffuse/diffuse_cif_toggle.py` | Support script | `python tests/Diffuse/diffuse_cif_toggle.py --l-max <value>` | Interactive PbI2 diffuse CIF viewer used as test-adjacent support, not a pytest file. |
| `tests/helpers/__init__.py` | Support package marker | Not direct CLI. | Package marker for shared test helpers. |
| `tests/helpers/vesta_reference.py` | Support helper | Imported by tests. | Test import shim for VESTA parity helpers. |

## Validation, diagnostic, benchmark, and timing scripts

| Category | Path | Command shape | Output artifacts | Purpose |
|---|---|---|---|---|
| Other scripts | `scripts/__init__.py` | `Not direct CLI.` | None. | Package marker for development scripts. |
| Benchmarks | `scripts/benchmarks/__init__.py` | `Not direct CLI.` | None. | Package marker for benchmark scripts. |
| Benchmarks | `scripts/benchmarks/benchmark_gui_startup.py` | `python scripts/benchmarks/benchmark_gui_startup.py --label <label> --output <json>` | Optional JSON output. | Benchmarks GUI startup import and launch timing. |
| Benchmarks | `scripts/benchmarks/benchmark_mosaic_profiles.py` | `python scripts/benchmarks/benchmark_mosaic_profiles.py --samples <n> --iterations <n>` | Stdout timing summary. | Benchmarks random mosaic profile generation. |
| Benchmarks | `scripts/benchmarks/benchmark_weighted_events_parallel.py` | `python scripts/benchmarks/benchmark_weighted_events_parallel.py --help` | Stdout timing summary. | Benchmarks weighted-event parallel execution. |
| Weighted-event diagnostics | `scripts/diagnostics/validate_weighted_event_merge.py` | `python scripts/diagnostics/validate_weighted_event_merge.py` | Focused pytest and benchmark output. | Runs the weighted-event merge diagnostics gate. |
| Geometry landscape and fitting performance | `scripts/debug/analyze_geometry_fit_jacobians.py` | `python scripts/debug/analyze_geometry_fit_jacobians.py <scenario_file> --output <report>` | JSON or YAML aggregate report. | Aggregates geometry-fit Jacobian diagnostics. |
| Debug and analysis tools | `scripts/debug/analyze_simulation_debug.py` | `python scripts/debug/analyze_simulation_debug.py` | Reads generated artifact scripts/debug/simulation.npz and shows a plot. | Analyzes missed diffracted ray directions. |
| Validation probes | `scripts/debug/bi2se3_vesta_switch_sweep.py` | `python scripts/debug/bi2se3_vesta_switch_sweep.py --help` | JSON switch-sweep report on stdout. | Runs Bi2Se3 structure-factor switch-sweep diagnostics. |
| Validation probes | `scripts/debug/compare_single_hkl_vesta_vs_sim.py` | `python scripts/debug/compare_single_hkl_vesta_vs_sim.py --help` | JSON comparison output or optional output file. | Compares one HKL between VESTA reference data and simulated raw factors. |
| Geometry recovery diagnostics | `scripts/debug/diagnose_new4_visual_backend_coordinates.py` | `python scripts/debug/diagnose_new4_visual_backend_coordinates.py --output-dir <dir>` | coordinate_transform_diagnosis.json. | Compares New4 visual and backend coordinate transforms. |
| Debug and analysis tools | `scripts/debug/optimization.py` | `python scripts/debug/optimization.py` | Configured plots and fit summaries. | Legacy global/local diffraction optimization diagnostic. |
| Debug and analysis tools | `scripts/debug/plot_mosaic_omega_and_bragg_sphere.py` | `python scripts/debug/plot_mosaic_omega_and_bragg_sphere.py --H <h> --K <k> --L <l>` | Interactive Matplotlib plot. | Visualizes mosaic omega distribution and Bragg sphere geometry. |
| Validation probes | `scripts/debug/run_diffraction_test.py` | `python scripts/debug/run_diffraction_test.py` | Generated artifact scripts/debug/simulation.npz. | Runs one headless diffraction simulation for downstream inspection. |
| Geometry landscape and fitting performance | `scripts/debug/run_geometry_fit_parameter_correlation.py` | `python scripts/debug/run_geometry_fit_parameter_correlation.py --state <state.json> --outdir <dir>` | Correlation artifacts in output directory. | Exports geometry-fit parameter correlation maps. |
| Geometry recovery diagnostics | `scripts/debug/run_geometry_fit_quality_baseline.py` | `python scripts/debug/run_geometry_fit_quality_baseline.py <state.json> [--active-vars gamma,Gamma]` | Baseline report artifacts under output root. | Runs frozen-state geometry-fit quality baselines, including constrained active-variable gates. |
| Geometry recovery diagnostics | `scripts/debug/run_geometry_fitter_cache_regression_gate.py` | `python scripts/debug/run_geometry_fitter_cache_regression_gate.py --mode local` | Focused compile, pytest, and optional New4 artifacts. | Runs geometry fitter cache regression gates in local or strict mode. |
| Geometry recovery diagnostics | `scripts/debug/run_new4_caked_point_reprojection_check.py` | `python scripts/debug/run_new4_caked_point_reprojection_check.py --state <state.json>` | rung_03b_caked_point_reprojection.json. | Checks New4 exact-cake point reprojection. |
| Geometry recovery diagnostics | `scripts/debug/run_new4_geometry_fit_ladder.py` | `python scripts/debug/run_new4_geometry_fit_ladder.py --state <state.json> --output-root <dir>` | Rung JSON reports and ladder summary. | Runs bounded New4 geometry-fit optimizer probes. |
| Validation probes | `scripts/debug/run_q_group_peak_sensitivity.py` | `python scripts/debug/run_q_group_peak_sensitivity.py --state <state.json> --group-key <key> --outdir <dir>` | Sensitivity matrix artifacts. | Exports finite-difference Q-group peak sensitivity evidence. |
| Validation probes | `scripts/debug/run_q_group_peak_transport_validation.py` | `python scripts/debug/run_q_group_peak_transport_validation.py --state <state.json> --group-key <key> --outdir <dir>` | Transport validation artifacts. | Compares Q-group transport shortcuts against recomputation. |
| Geometry recovery diagnostics | `scripts/debug/validate_geometry_preflight_rebind.py` | `python scripts/debug/validate_geometry_preflight_rebind.py --state <state.json> --mode <mode>` | Optional report JSON or fresh state export. | Validates grouped-pick preflight rebinding behavior. |
| Debug and analysis tools | `scripts/debug/view_sim_image.py` | `python scripts/debug/view_sim_image.py` | Reads generated artifact scripts/debug/simulation.npz and shows a plot. | Views the simulated image written by run_diffraction_test.py. |
| Geometry landscape and fitting performance | `scripts/geometry_fit_landscape.py` | `python scripts/geometry_fit_landscape.py --state <state.json> --outdir <dir>` | landscape_runs.csv, baseline_metadata.json, landscape_figure.png. | Sweeps geometry-fit parameters around a saved GUI baseline. |
| GUI timing and performance | `scripts/measure_gui_timing.py` | `python scripts/measure_gui_timing.py --scenario <name> --trials <n>` | Generated artifacts under artifacts/perf/gui_timing/<stamp>/. | Runs GUI timing trials and summarizes JSONL timing events. |

## Developer validation modules

| Path | Purpose |
|---|---|
| `ra_sim/dev.py` | Canonical bootstrap, format, lint, typecheck, test, coverage, build, hook, and lock command surface. |
| `ra_sim/dev_doctor.py` | Warning-first setup checks for Python, config, local paths, writable dirs, Tkinter, and dev tools. |
| `ra_sim/test_tiers.py` | Fast and integration pytest manifest used by dev tooling and test collection. |
| `ra_sim/timing.py` | Gated JSONL timing events and spans for GUI performance measurement. |

## User utility CLIs

| Path | Command shape | Purpose |
|---|---|---|
| `ra_sim/tools/compare_bi2se3_reference.py` | `python -m ra_sim.tools.compare_bi2se3_reference --help` | Compares Bi2Se3 VESTA reference exports against raw structure factors. |
| `ra_sim/tools/compare_intensity.py` | `python -m ra_sim.tools.compare_intensity <excel_path> --sheet <name>` | Compares CIF intensities against numeric Hendricks-Teller areas and plots metrics. |
| `ra_sim/tools/plot_excel_scatter.py` | `python -m ra_sim.tools.plot_excel_scatter <excel_path> --sheet <name> --intensity <column>` | Plots Excel intensity columns against L with optional interactive controls. |

## Validation support and dependency modules

These modules are not stand-alone test commands, but they are validation-critical support code used by the indexed parity and conservation checks.

| Path | Purpose |
|---|---|
| `ra_sim/simulation/mosaic_normalization.py` | Pre-detector mosaic mass conservation validation helpers. |
| `ra_sim/structure_factors/__init__.py` | Package exports for raw structure-factor helpers. |
| `ra_sim/structure_factors/options.py` | Options dataclass for raw structure-factor calculations. |
| `ra_sim/structure_factors/raw_f.py` | Raw complex structure-factor calculation and debug payloads. |
| `ra_sim/structure_factors/sweep.py` | One-knob structure-factor switch-sweep diagnostics. |
| `ra_sim/structure_factors/vesta_like_atomic_factors.py` | VESTA-parity atomic factor helpers. |
| `ra_sim/validation/__init__.py` | Package exports for validation helpers. |
| `ra_sim/validation/environment.py` | Environment snapshots for structure-factor parity tests. |
| `ra_sim/validation/vesta_reference.py` | VESTA TXT parsing and HKL comparison helpers. |

## Automation gates

| Path | Gate | Purpose |
|---|---|---|
| `.pre-commit-config.yaml` | Local pre-commit | Runs ruff format, ruff check, and `python -m ra_sim.dev typecheck`. |
| `.github/workflows/ci.yml` | GitHub Actions CI | Bootstraps dev env, verifies spglib, runs `python -m ra_sim.dev check` on Python 3.11-3.13, and runs integration tests on Python 3.11. |
| `.github/workflows/security.yml` | GitHub Actions security | Rejects tracked machine-local paths, runs pip-audit, and runs gitleaks. |

## Checked index guard

`tests/test_testing_validation_index.py` reads this file and `git ls-files` to fail when tracked tests, fixtures, scripts, tools, developer modules, workflows, pre-commit config, or agent-support scripts are missing from the index. It also rejects placeholder test-file descriptions. Its reverse path check is intentionally conservative and allows documented generated/example artifact paths.

## Focused geometric-fitter Qr/Qz validation

The `(-1,0,10)` Qr/Qz fitter regression is covered by focused tests in
`tests/test_manual_geometry_selection_helpers.py` and provider-local resolver
unit tests in `tests/test_geometry_fitting.py`.

## Focused two-rotation geometry baseline

Use the constrained baseline runner when the Bi2Se3 and Bi2Te3 saved states
need to prove only detector rotations move fitted points closer together:

```powershell
python scripts/debug/run_geometry_fit_quality_baseline.py `
  --active-vars gamma,Gamma `
  --output-root <output-root> `
  --state-timeout-seconds 900
```

The debug-script option is additive. Omitting `--active-vars` keeps the default
baseline command unchanged, while providing it forwards the comma-separated
active-variable override to `ra_sim.cli fit-geometry`.

Current status, 2026-05-10: `gamma,Gamma` remains the smallest detector-rotation
baseline that moves both Bi saved states closer together. Both material runs
preserved exact/point-only caked fit-space provenance, passed the saved-state
gate, and kept zero missing fixed pairs and zero branch mismatches. Bi2Se3
matched 82/82 fixed pairs and reduced direct RMS from 34.5307 px to 15.701942
px; Bi2Te3 matched 84/84 fixed pairs and reduced direct RMS from 36.8629 px to
36.661839 px.

The broader local matrix showed `gamma,Gamma,corto_detector` as the best tested
residual tradeoff for both materials, reducing Bi2Se3 to 10.922948 px and
Bi2Te3 to 35.581 px, but that is a three-variable exploratory candidate rather
than the focused two-rotation proof. Single-variable `gamma` and `Gamma`,
`gamma,Gamma,theta_initial`, and `corto_detector,theta_initial` also preserved
the saved-state gate and reduced direct RMS for both materials, but each was
weaker on at least one material. `theta_initial` is reported in artifacts as
canonical `theta_offset`.

Narrow regression gate:

```powershell
python -m pytest tests/test_geometry_fit_quality_baseline.py tests/test_cli_geometry_fit.py -ra
```

CI status: the normal GitHub Actions gates remain `python -m ra_sim.dev check`
and integration tests. The real Bi material baseline is intentionally local
and opt-in because it depends on user-root saved states and has a long runtime.
No deprecation, migration, saved-state schema change, artifact schema change,
or production rollout flag is required; rollback is a normal git revert.

Focused commands:

```powershell
python -m py_compile ra_sim/fitting/optimization.py ra_sim/gui/geometry_fit.py ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py
pytest tests/test_geometry_fitting.py -k "provider_local_stale_row or duplicate_hkl_ambiguous or duplicate_hkl_saved_px or saved_detector_uses_stale_row_proof or does_not_upgrade_local_to_full_reflection" -q
pytest tests/test_manual_geometry_selection_helpers.py -k "rung1_objective_dry_run_uses_qr_residuals or fitter_objective_matches_residual_audit or prediction_identity_stable_during_fit or qr_only_fit_reduces_residual_after_correspondence_fix or qr_only_objective_does_not_accept_worse_solution or full_fit_reports_qr_contribution" -s -q
pytest tests/test_gui_runtime_import_safe.py -k "toggle_caked_2d" -q
```

Expected status:

- dry-run objective calls the production residual helper and does not call
  `least_squares`,
- all seven provider-local fixed rows resolve without fallback,
- the target Qr residual block is present with nonempty weights,
- branch identity remains fixed through optimizer evaluations,
- Qr-only fit either reduces the target residual norm or reports an exact
  rejected-step reason without treating the rejected state as an accepted fit,
- full fit reports total, Qr, non-Qr, line, and prior objective blocks.

## Appendix: Additional tracked validation entrypoints

These tracked paths are part of the validation/debug inventory but are not the
primary command examples above. Bug/error status, 2026-05-10: the docs-index
guard is fixed by listing these tracked entrypoints. Feature status:
documentation-only inventory coverage; no runtime behavior, CLI, saved-state,
config, or artifact schema changed.

| Path | Purpose |
|---|---|
| `scripts/debug/visualize_new4_qr_fit_coordinates.py` | Visualizes Bi2Se3 Qr fit-coordinate diagnostics. |
| `scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py` | Runs parallel background peak-fit diagnostics with a shared linear baseline. |
| `scripts/diagnostics/background_peak_fit_worker.py` | Worker helper for background peak-fit diagnostic batches. |
| `scripts/diagnostics/run_all_background_peak_fits.py` | Launches the background peak-fit diagnostic batch runner. |
| `tests/test_background_peak_fits_notebook.py` | Checks notebook-oriented background peak-fit workflows. |
| `tests/test_beam_center_pick_helpers.py` | Checks beam-center pick helper behavior. |
| `tests/test_disordered_phase_current_refresh.py` | Checks disordered-phase current refresh behavior. |
| `tests/test_disordered_phase_end_to_end.py` | Checks disordered-phase end-to-end workflow coverage. |
| `tests/test_disordered_phase_hit_table_scheduling.py` | Checks disordered-phase hit-table scheduling. |
| `tests/test_disordered_phase_hit_tables.py` | Checks disordered-phase hit-table construction. |
| `tests/test_disordered_phase_invalidation.py` | Checks disordered-phase cache invalidation. |
| `tests/test_disordered_phase_inventory.py` | Checks disordered-phase inventory reporting. |
| `tests/test_disordered_phase_inventory_live_path.py` | Checks live-path disordered-phase inventory reporting. |
| `tests/test_disordered_phase_live_q_group_refresh.py` | Checks live Q-group refresh behavior. |
| `tests/test_disordered_phase_live_runtime_regression.py` | Checks live-runtime disordered-phase regressions. |
| `tests/test_disordered_phase_logging.py` | Checks disordered-phase diagnostic logging. |
| `tests/test_disordered_phase_picker_to_fitter_end_to_end.py` | Checks picker-to-fitter disordered-phase handoff. |
| `tests/test_disordered_phase_q_group_cache.py` | Checks disordered-phase Q-group cache behavior. |
| `tests/test_disordered_phase_real_inventory.py` | Checks real-data disordered-phase inventory handling. |
| `tests/test_disordered_phase_source_labels.py` | Checks disordered-phase source labels. |
| `tests/test_disordered_phase_state.py` | Checks disordered-phase state behavior. |
| `tests/test_disordered_phase_state_io.py` | Checks disordered-phase state serialization. |
| `tests/test_disordered_phase_ui_enable.py` | Checks disordered-phase UI enablement gates. |
| `tests/test_disordered_phase_user_report_live_path.py` | Checks live-path user reports for disordered-phase data. |
| `tests/test_geometry_fit_disordered_phase.py` | Checks geometry fitting with disordered-phase data. |
| `tests/test_geometry_fit_disordered_preflight.py` | Checks disordered-phase geometry-fit preflight behavior. |
| `tests/test_geometry_fit_fresh_rebuild_consumer.py` | Checks fresh geometry-fit rebuild consumers. |
| `tests/test_geometry_fit_job_live_rows_handoff.py` | Checks live-row handoff for geometry-fit jobs. |
| `tests/test_geometry_fit_live_cache_diagnostics.py` | Checks live-cache geometry-fit diagnostics. |
| `tests/test_geometry_fit_live_cache_validation_acceptance.py` | Checks live-cache validation acceptance. |
| `tests/test_geometry_fit_live_rows_signature_handoff.py` | Checks live-row signature handoff. |
| `tests/test_geometry_fit_manual_fit_space_classification.py` | Checks manual fit-space classification for geometry fitting. |
| `tests/test_geometry_fit_source_cache_rungs.py` | Checks source-cache rung behavior. |
| `tests/test_manual_detector_to_caked_refresh.py` | Checks manual detector-to-caked refresh behavior. |
| `tests/test_manual_geometry_disordered_state_io.py` | Checks manual-geometry disordered-state serialization. |
| `tests/test_manual_picker_disordered_detector_positions.py` | Checks manual-picker disordered detector positions. |
| `tests/test_manual_picker_disordered_phase.py` | Checks manual-picker disordered-phase behavior. |
| `tests/test_manual_picker_source_diagnostics.py` | Checks manual-picker source diagnostics. |
| `tests/test_manual_placement_disordered_source.py` | Checks manual placement of disordered sources. |
| `tests/test_new4_rung2_contract.py` | Checks New4 rung 2 contract behavior. |
| `tests/test_numba_compat.py` | Checks Numba compatibility guards. |
| `tests/test_pbi2_ht_shift_cif.py` | Checks PbI2 high-temperature shift CIF handling. |
| `tests/test_q_group_duplicate_source_identity.py` | Checks duplicate source identity handling for Q groups. |
| `tests/test_q_space_viewer_runtime.py` | Checks Q-space viewer runtime behavior. |
| `tests/test_rod_profiles.py` | Checks rod-profile helpers. |

## Appendix: Agent-skill support tools

These files support local Codex agent skills. They are not RA-SIM runtime validation entry points.

| Path | Type | Purpose |
|---|---|---|
| `.agents/skills/compress/scripts/__init__.py` | Agent-support tooling | Package marker for compress skill scripts. |
| `.agents/skills/compress/scripts/__main__.py` | Agent-support tooling | Module entry point for the compress skill CLI. |
| `.agents/skills/compress/scripts/benchmark.py` | Agent-support tooling | Benchmarks compression token savings for skill memory files. |
| `.agents/skills/compress/scripts/cli.py` | Agent-support tooling | CLI wrapper for the compress skill workflow. |
| `.agents/skills/compress/scripts/compress.py` | Agent-support tooling | Compression orchestrator for natural-language memory files. |
| `.agents/skills/compress/scripts/detect.py` | Agent-support tooling | Detects whether files are natural-language compression candidates. |
| `.agents/skills/compress/scripts/validate.py` | Agent-support tooling | Validates compressed memory content preservation. |
