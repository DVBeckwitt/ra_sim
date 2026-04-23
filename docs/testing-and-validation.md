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
| CLI and launcher | `tests/test_dev_cli.py` | `python -m ra_sim.dev test-fast` | Regression coverage for dev cli. | Fast manifest in ra_sim/test_tiers.py. |
| CLI and launcher | `tests/test_launcher_routing.py` | `python -m pytest tests/test_launcher_routing.py` | Regression coverage for launcher routing. | Untiered; direct pytest or full suite. |
| CLI and launcher | `tests/test_main_entrypoint_import_safe.py` | `python -m pytest tests/test_main_entrypoint_import_safe.py` | Regression coverage for main entrypoint import safe. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_config_loader.py` | `python -m ra_sim.dev test-fast` | Regression coverage for config loader. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_data_loading_parameters.py` | `python -m ra_sim.dev test-fast` | Regression coverage for data loading parameters. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_gui_state_io.py` | `python -m pytest tests/test_gui_state_io.py` | Regression coverage for gui state io. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_gui_state_restore_helpers.py` | `python -m pytest tests/test_gui_state_restore_helpers.py` | Regression coverage for gui state restore helpers. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_hbn_path_resolution.py` | `python -m ra_sim.dev test-fast` | Regression coverage for hbn path resolution. | Fast manifest in ra_sim/test_tiers.py. |
| Config, paths, and I/O | `tests/test_osc_reader.py` | `python -m pytest tests/test_osc_reader.py` | Regression coverage for osc reader. | Untiered; direct pytest or full suite. |
| Config, paths, and I/O | `tests/test_user_paths.py` | `python -m pytest tests/test_user_paths.py` | Regression coverage for user paths. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_debug_controls.py` | `python -m ra_sim.dev test-fast` | Regression coverage for debug controls. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_debug_utils.py` | `python -m ra_sim.dev test-fast` | Regression coverage for debug utils. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_dependency_metadata.py` | `python -m ra_sim.dev test-fast` | Regression coverage for dependency metadata. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_dev_doctor.py` | `python -m ra_sim.dev test-fast` | Regression coverage for dev doctor. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_import_smoke.py` | `python -m ra_sim.dev test-fast` | Package import smoke coverage across modules. | Fast manifest in ra_sim/test_tiers.py. |
| Developer tooling and runtime infrastructure | `tests/test_install_prereqs.py` | `python -m pytest tests/test_install_prereqs.py` | Regression coverage for install prereqs. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_logging_controls.py` | `python -m pytest tests/test_logging_controls.py` | Regression coverage for logging controls. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_numba_cache_env.py` | `python -m pytest tests/test_numba_cache_env.py` | Regression coverage for numba cache env. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_parallel_utils.py` | `python -m pytest tests/test_parallel_utils.py` | Regression coverage for parallel utils. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_sitecustomize.py` | `python -m pytest tests/test_sitecustomize.py` | Regression coverage for sitecustomize. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_structure_factor_environment.py` | `python -m pytest tests/test_structure_factor_environment.py` | Structure-factor parity environment snapshots and explicit wavelength checks. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_testing_validation_index.py` | `python -m pytest tests/test_testing_validation_index.py` | Static guard for this testing and validation index. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_timing.py` | `python -m pytest tests/test_timing.py` | Timing JSONL events and GUI timing summary helpers. | Untiered; direct pytest or full suite. |
| Developer tooling and runtime infrastructure | `tests/test_utils_notifications.py` | `python -m pytest tests/test_utils_notifications.py` | Regression coverage for utils notifications. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_background_peak_matching.py` | `python -m ra_sim.dev test-fast` | Regression coverage for background peak matching. | Fast manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_fit_cache_controls.py` | `python -m pytest tests/test_fit_cache_controls.py` | Regression coverage for fit cache controls. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_correlation.py` | `python -m pytest tests/test_geometry_fit_correlation.py` | Regression coverage for geometry fit correlation. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_landscape.py` | `python -m ra_sim.dev test-integration` | Regression coverage for geometry fit landscape. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_lattice_ui_helpers.py` | `python -m pytest tests/test_geometry_fit_lattice_ui_helpers.py` | Regression coverage for geometry fit lattice ui helpers. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_quality_baseline.py` | `python -m ra_sim.dev test-integration` | Regression coverage for geometry fit quality baseline. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_fit_safe_runtime.py` | `python -m pytest tests/test_geometry_fit_safe_runtime.py` | Regression coverage for geometry fit safe runtime. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_fitting.py` | `python -m ra_sim.dev test-integration` | Regression coverage for geometry fitting. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_geometry_line_orientation.py` | `python -m pytest tests/test_geometry_line_orientation.py` | Regression coverage for geometry line orientation. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_geometry_overlay.py` | `python -m pytest tests/test_geometry_overlay.py` | Regression coverage for geometry overlay. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_geometry_fit_workflow.py` | `python -m ra_sim.dev test-integration` | Regression coverage for gui geometry fit workflow. | Integration manifest in ra_sim/test_tiers.py. |
| Geometry fitting and projection validation | `tests/test_gui_geometry_q_group_manager.py` | `python -m pytest tests/test_gui_geometry_q_group_manager.py` | Regression coverage for gui geometry q group manager. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_qr_cylinder_overlay.py` | `python -m pytest tests/test_gui_qr_cylinder_overlay.py` | Regression coverage for gui qr cylinder overlay. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_fit.py` | `python -m pytest tests/test_gui_runtime_geometry_fit.py` | Regression coverage for gui runtime geometry fit. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_interaction.py` | `python -m pytest tests/test_gui_runtime_geometry_interaction.py` | Regression coverage for gui runtime geometry interaction. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_geometry_preview.py` | `python -m pytest tests/test_gui_runtime_geometry_preview.py` | Regression coverage for gui runtime geometry preview. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_gui_runtime_qr_cylinder_overlay.py` | `python -m pytest tests/test_gui_runtime_qr_cylinder_overlay.py` | Regression coverage for gui runtime qr cylinder overlay. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_hbn_geometry_import_safe.py` | `python -m pytest tests/test_hbn_geometry_import_safe.py` | Regression coverage for hbn geometry import safe. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_hbn_geometry_mapping.py` | `python -m pytest tests/test_hbn_geometry_mapping.py` | Regression coverage for hbn geometry mapping. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_main_geometry_fit_helpers.py` | `python -m pytest tests/test_main_geometry_fit_helpers.py` | Regression coverage for main geometry fit helpers. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_manual_geometry_live_peak_cache.py` | `python -m pytest tests/test_manual_geometry_live_peak_cache.py` | Regression coverage for manual geometry live peak cache. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_manual_geometry_selection_helpers.py` | `python -m pytest tests/test_manual_geometry_selection_helpers.py` | Regression coverage for manual geometry selection helpers. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_peak_sensitivity.py` | `python -m pytest tests/test_peak_sensitivity.py` | Regression coverage for peak sensitivity. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_peak_transport_validation.py` | `python -m pytest tests/test_peak_transport_validation.py` | Regression coverage for peak transport validation. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_projection_alignment_contract.py` | `python -m pytest tests/test_projection_alignment_contract.py` | Regression coverage for projection alignment contract. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_qr_cylinder_overlay.py` | `python -m pytest tests/test_qr_cylinder_overlay.py` | Regression coverage for qr cylinder overlay. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_qr_grouping.py` | `python -m pytest tests/test_qr_grouping.py` | Regression coverage for qr grouping. | Untiered; direct pytest or full suite. |
| Geometry fitting and projection validation | `tests/test_source_template_cache.py` | `python -m pytest tests/test_source_template_cache.py` | Regression coverage for source template cache. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_fast_manual_point_runtime_profiles.py` | `python -m pytest tests/test_fast_manual_point_runtime_profiles.py` | Regression coverage for fast manual point runtime profiles. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_figure_controls.py` | `python -m pytest tests/test_gui_analysis_figure_controls.py` | Regression coverage for gui analysis figure controls. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_peak_tools.py` | `python -m pytest tests/test_gui_analysis_peak_tools.py` | Regression coverage for gui analysis peak tools. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_quick_controls.py` | `python -m pytest tests/test_gui_analysis_quick_controls.py` | Regression coverage for gui analysis quick controls. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_analysis_visibility.py` | `python -m pytest tests/test_gui_analysis_visibility.py` | Regression coverage for gui analysis visibility. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_app_helpers.py` | `python -m pytest tests/test_gui_app_helpers.py` | Regression coverage for gui app helpers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_app_import_safe.py` | `python -m pytest tests/test_gui_app_import_safe.py` | Regression coverage for gui app import safe. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_background.py` | `python -m pytest tests/test_gui_background.py` | Regression coverage for gui background. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_background_manager.py` | `python -m pytest tests/test_gui_background_manager.py` | Regression coverage for gui background manager. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_bootstrap.py` | `python -m pytest tests/test_gui_bootstrap.py` | Regression coverage for gui bootstrap. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_bragg_qr_manager.py` | `python -m pytest tests/test_gui_bragg_qr_manager.py` | Regression coverage for gui bragg qr manager. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_canvas_interactions.py` | `python -m pytest tests/test_gui_canvas_interactions.py` | Regression coverage for gui canvas interactions. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_controllers.py` | `python -m pytest tests/test_gui_controllers.py` | Regression coverage for gui controllers. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_diffuse_cif_toggle.py` | `python -m pytest tests/test_gui_diffuse_cif_toggle.py` | Regression coverage for gui diffuse cif toggle. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_display_projection.py` | `python -m pytest tests/test_gui_display_projection.py` | Regression coverage for gui display projection. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_fast_plot_viewer.py` | `python -m pytest tests/test_gui_fast_plot_viewer.py` | Regression coverage for gui fast plot viewer. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_fit2d_error_sound.py` | `python -m pytest tests/test_gui_fit2d_error_sound.py` | Regression coverage for gui fit2d error sound. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_integration_range_drag.py` | `python -m pytest tests/test_gui_integration_range_drag.py` | Regression coverage for gui integration range drag. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_lazy_runtime.py` | `python -m pytest tests/test_gui_lazy_runtime.py` | Regression coverage for gui lazy runtime. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_main_figure_chrome.py` | `python -m pytest tests/test_gui_main_figure_chrome.py` | Regression coverage for gui main figure chrome. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_main_matplotlib_interaction.py` | `python -m pytest tests/test_gui_main_matplotlib_interaction.py` | Regression coverage for gui main matplotlib interaction. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_ordered_structure_fit.py` | `python -m pytest tests/test_gui_ordered_structure_fit.py` | Regression coverage for gui ordered structure fit. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_overlays.py` | `python -m pytest tests/test_gui_overlays.py` | Regression coverage for gui overlays. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_peak_selection.py` | `python -m pytest tests/test_gui_peak_selection.py` | Regression coverage for gui peak selection. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_background.py` | `python -m pytest tests/test_gui_runtime_background.py` | Regression coverage for gui runtime background. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_display_acceleration.py` | `python -m pytest tests/test_gui_runtime_display_acceleration.py` | Regression coverage for gui runtime display acceleration. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_fit_analysis.py` | `python -m pytest tests/test_gui_runtime_fit_analysis.py` | Regression coverage for gui runtime fit analysis. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_import_safe.py` | `python -m pytest tests/test_gui_runtime_import_safe.py` | Regression coverage for gui runtime import safe. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_position_preview.py` | `python -m pytest tests/test_gui_runtime_position_preview.py` | Regression coverage for gui runtime position preview. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_primary_cache.py` | `python -m pytest tests/test_gui_runtime_primary_cache.py` | Regression coverage for gui runtime primary cache. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_primary_viewport.py` | `python -m pytest tests/test_gui_runtime_primary_viewport.py` | Regression coverage for gui runtime primary viewport. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_runtime_update_trace.py` | `python -m pytest tests/test_gui_runtime_update_trace.py` | Regression coverage for gui runtime update trace. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_sim_signature.py` | `python -m pytest tests/test_gui_sim_signature.py` | Regression coverage for gui sim signature. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_sliders.py` | `python -m pytest tests/test_gui_sliders.py` | Regression coverage for gui sliders. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_structure_factor_pruning.py` | `python -m pytest tests/test_gui_structure_factor_pruning.py` | Regression coverage for gui structure factor pruning. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_structure_model.py` | `python -m pytest tests/test_gui_structure_model.py` | Regression coverage for gui structure model. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_tk_primary_viewport.py` | `python -m pytest tests/test_gui_tk_primary_viewport.py` | Regression coverage for gui tk primary viewport. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_views.py` | `python -m pytest tests/test_gui_views.py` | Regression coverage for gui views. | Untiered; direct pytest or full suite. |
| GUI helpers and runtime behavior | `tests/test_gui_window_affinity.py` | `python -m pytest tests/test_gui_window_affinity.py` | Regression coverage for gui window affinity. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_ctr_fast_attenuation.py` | `python -m ra_sim.dev test-fast` | Regression coverage for ctr fast attenuation. | Fast manifest in ra_sim/test_tiers.py. |
| Simulation and diffraction engine | `tests/test_diffraction_constraints.py` | `python -m pytest tests/test_diffraction_constraints.py` | Regression coverage for diffraction constraints. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_inner_loop_optimizations.py` | `python -m pytest tests/test_diffraction_inner_loop_optimizations.py` | Regression coverage for diffraction inner loop optimizations. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_local_arc.py` | `python -m pytest tests/test_diffraction_local_arc.py` | Regression coverage for diffraction local arc. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_safe_wrapper.py` | `python -m pytest tests/test_diffraction_safe_wrapper.py` | Regression coverage for diffraction safe wrapper. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_subpixel.py` | `python -m pytest tests/test_diffraction_subpixel.py` | Regression coverage for diffraction subpixel. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_tools_module.py` | `python -m pytest tests/test_diffraction_tools_module.py` | Regression coverage for diffraction tools module. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_diffraction_tools_view.py` | `python -m pytest tests/test_diffraction_tools_view.py` | Regression coverage for diffraction tools view. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_exact_cake_portable.py` | `python -m pytest tests/test_exact_cake_portable.py` | Regression coverage for exact cake portable. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_exact_qspace_portable.py` | `python -m pytest tests/test_exact_qspace_portable.py` | Regression coverage for exact qspace portable. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_fresnel_calls.py` | `python -m pytest tests/test_fresnel_calls.py` | Regression coverage for fresnel calls. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intensities.py` | `python -m pytest tests/test_intensities.py` | Regression coverage for intensities. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intersection_analysis.py` | `python -m pytest tests/test_intersection_analysis.py` | Regression coverage for intersection analysis. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_intersection_cache_schema.py` | `python -m pytest tests/test_intersection_cache_schema.py` | Regression coverage for intersection cache schema. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_mosaic_intensity_conservation.py` | `python -m pytest tests/test_mosaic_intensity_conservation.py` | Mosaic normalization and pre-detector intensity conservation validation. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_peak_multiplicity_cache.py` | `python -m pytest tests/test_peak_multiplicity_cache.py` | Regression coverage for peak multiplicity cache. | Untiered; direct pytest or full suite. |
| Simulation and diffraction engine | `tests/test_simulation.py` | `python -m ra_sim.dev test-integration` | Regression coverage for simulation. | Integration manifest in ra_sim/test_tiers.py. |
| Simulation and diffraction engine | `tests/test_simulation_engine.py` | `python -m ra_sim.dev test-integration` | Regression coverage for simulation engine. | Integration manifest in ra_sim/test_tiers.py. |
| Materials, structure factors, and mosaic models | `tests/test_atomic_scattering_factors.py` | `python -m pytest tests/test_atomic_scattering_factors.py` | Atomic scattering factor helpers and debug table coverage. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_residual_diagnostics.py` | `python -m pytest tests/test_bi2se3_residual_diagnostics.py` | Bi2Se3 VESTA residual summary and worst-HKL diagnostics. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_structure_factor.py` | `python -m pytest tests/test_bi2se3_structure_factor.py` | Regression coverage for bi2se3 structure factor. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_validation_doc_consistency.py` | `python -m pytest tests/test_bi2se3_validation_doc_consistency.py` | Bi2Se3 validation documentation consistency with fixture metadata. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_vesta_geometry.py` | `python -m pytest tests/test_bi2se3_vesta_geometry.py` | Bi2Se3 d-spacing and two-theta parity against VESTA references. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_bi2se3_vesta_reference_regression.py` | `python -m pytest tests/test_bi2se3_vesta_reference_regression.py` | Bi2Se3 VESTA reference regression guards and debug artifacts. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_ht_analytical.py` | `python -m pytest tests/test_ht_analytical.py` | Regression coverage for ht analytical. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_profiles_rng.py` | `python -m pytest tests/test_mosaic_profiles_rng.py` | Regression coverage for mosaic profiles rng. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_shape_optimization.py` | `python -m pytest tests/test_mosaic_shape_optimization.py` | Regression coverage for mosaic shape optimization. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_mosaic_width_optimization.py` | `python -m pytest tests/test_mosaic_width_optimization.py` | Regression coverage for mosaic width optimization. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_raw_structure_factor_api.py` | `python -m pytest tests/test_raw_structure_factor_api.py` | Raw complex structure-factor API semantics and debug payload rules. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_single_hkl_debug_payload.py` | `python -m pytest tests/test_single_hkl_debug_payload.py` | Single-HKL debug payload terms, JSON serialization, and NaN two-theta cases. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_cache_invalidation.py` | `python -m pytest tests/test_stacking_fault_cache_invalidation.py` | Regression coverage for stacking fault cache invalidation. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_diffuse_f2_q.py` | `python -m pytest tests/test_stacking_fault_diffuse_f2_q.py` | Regression coverage for stacking fault diffuse f2 q. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_stacking_fault_redundancy.py` | `python -m pytest tests/test_stacking_fault_redundancy.py` | Regression coverage for stacking fault redundancy. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_atomic_coordinates.py` | `python -m ra_sim.dev test-fast` | Regression coverage for structure factor atomic coordinates. | Fast manifest in ra_sim/test_tiers.py. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_sites.py` | `python -m pytest tests/test_structure_factor_sites.py` | Expanded structure-factor site list determinism and occupancy checks. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_structure_factor_switch_sweep.py` | `python -m pytest tests/test_structure_factor_switch_sweep.py` | Structure-factor one-knob sweep diagnostics and error summaries. | Untiered; direct pytest or full suite. |
| Materials, structure factors, and mosaic models | `tests/test_vesta_reference_parser.py` | `python -m pytest tests/test_vesta_reference_parser.py` | VESTA TXT parser, fixture metadata, and implied wavelength checks. | Untiered; direct pytest or full suite. |
| Utilities and smoke tests | `tests/test_background_theta_helpers.py` | `python -m ra_sim.dev test-fast` | Regression coverage for background theta helpers. | Fast manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_compare_intensity.py` | `python -m ra_sim.dev test-fast` | Regression coverage for compare intensity. | Fast manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_hbn_fitter_bundle_export.py` | `python -m ra_sim.dev test-integration` | Regression coverage for hbn fitter bundle export. | Integration manifest in ra_sim/test_tiers.py. |
| Utilities and smoke tests | `tests/test_plot_excel_scatter.py` | `python -m pytest tests/test_plot_excel_scatter.py` | Regression coverage for plot excel scatter. | Untiered; direct pytest or full suite. |
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
| Geometry landscape and fitting performance | `scripts/debug/analyze_geometry_fit_jacobians.py` | `python scripts/debug/analyze_geometry_fit_jacobians.py <scenario_file> --output <report>` | JSON or YAML aggregate report. | Aggregates geometry-fit Jacobian diagnostics. |
| Debug and analysis tools | `scripts/debug/analyze_simulation_debug.py` | `python scripts/debug/analyze_simulation_debug.py` | Reads generated artifact scripts/debug/simulation.npz and shows a plot. | Analyzes missed diffracted ray directions. |
| Validation probes | `scripts/debug/bi2se3_vesta_switch_sweep.py` | `python scripts/debug/bi2se3_vesta_switch_sweep.py --help` | JSON switch-sweep report on stdout. | Runs Bi2Se3 structure-factor switch-sweep diagnostics. |
| Validation probes | `scripts/debug/compare_single_hkl_vesta_vs_sim.py` | `python scripts/debug/compare_single_hkl_vesta_vs_sim.py --help` | JSON comparison output or optional output file. | Compares one HKL between VESTA reference data and simulated raw factors. |
| Geometry recovery diagnostics | `scripts/debug/diagnose_new4_visual_backend_coordinates.py` | `python scripts/debug/diagnose_new4_visual_backend_coordinates.py --output-dir <dir>` | coordinate_transform_diagnosis.json. | Compares New4 visual and backend coordinate transforms. |
| Debug and analysis tools | `scripts/debug/optimization.py` | `python scripts/debug/optimization.py` | Configured plots and fit summaries. | Legacy global/local diffraction optimization diagnostic. |
| Debug and analysis tools | `scripts/debug/plot_mosaic_omega_and_bragg_sphere.py` | `python scripts/debug/plot_mosaic_omega_and_bragg_sphere.py --H <h> --K <k> --L <l>` | Interactive Matplotlib plot. | Visualizes mosaic omega distribution and Bragg sphere geometry. |
| Validation probes | `scripts/debug/run_diffraction_test.py` | `python scripts/debug/run_diffraction_test.py` | Generated artifact scripts/debug/simulation.npz. | Runs one headless diffraction simulation for downstream inspection. |
| Geometry landscape and fitting performance | `scripts/debug/run_geometry_fit_parameter_correlation.py` | `python scripts/debug/run_geometry_fit_parameter_correlation.py --state <state.json> --outdir <dir>` | Correlation artifacts in output directory. | Exports geometry-fit parameter correlation maps. |
| Geometry recovery diagnostics | `scripts/debug/run_geometry_fit_quality_baseline.py` | `python scripts/debug/run_geometry_fit_quality_baseline.py <state.json>` | Baseline report artifacts under output root. | Runs frozen-state geometry-fit quality baselines. |
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

`tests/test_testing_validation_index.py` reads this file and `git ls-files` to fail when tracked tests, fixtures, scripts, tools, developer modules, workflows, pre-commit config, or agent-support scripts are missing from the index. Its reverse path check is intentionally conservative and allows documented generated/example artifact paths.

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
