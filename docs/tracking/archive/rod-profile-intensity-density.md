# Rod profile intensity density

Status: completed
Type: bug | feature
Owner: -
Issue: none
Priority: p1
Last updated: 2026-04-30

## Summary

Qr rod and caked intensity displays no longer conflate scattering strength with
variable caked-bin support by default. Rod profiles, full caked-image pixels,
and standard caked radial/azimuthal integrations now have density-first
semantics with raw accumulated intensity available for inspection.

## Current state

- Added support-normalized rod profile helpers in `ra_sim.fitting.rod_profiles`.
- Diagnostic Qz rod profiles now plot `background_density` and `fit_density`;
  raw sums stay available for audit columns.
- Analyze UI has `Caked image intensity` and `Rod profile intensity` radio
  groups with `Intensity density (support-normalized)` as the default and
  `Raw accumulated intensity` as opt-in.
- Saved GUI state preserves `analysis_range.caked_intensity_mode` and
  `analysis_range.rod_profile_intensity_mode`.
- The caked visual toggle updates the visible caked image pixels by making
  raster cache signatures mode-aware, rebuilding cached caked payloads, and
  pushing the corrected raster into the main figure before redraw.

## Next actions

- None for this patch.
- Future physical corrections such as Lorentz, polarization, footprint,
  absorption, and exit-angle corrections remain out of scope.

## Validation

- `python -m pytest tests/test_rod_profiles.py tests/test_background_peak_fits_notebook.py -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "caked_intensity or prepare_caked_display_payload or refresh_integration" -ra`
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "scales_cached_background_payload or semantic_analysis_source_signatures" -ra`
- `python -m pytest tests/test_gui_integration_range_drag.py -k "create_runtime_integration_range_controls" -ra`
- `python -m pytest tests/test_gui_views.py tests/test_gui_state_io.py -k "caked_intensity or integration_range" -ra`
- `python -m pytest tests/test_gui_state_io.py -k "selected_qr_rod_analysis_range" -ra`
- `python -m pytest tests/test_gui_controllers.py -k "q_space or caked_axes" -ra`
- `python -m pytest tests/test_gui_bootstrap.py -k "integration_range_update_bootstrap" -ra`
- `python -m compileall ra_sim scripts tests`

## Links

- `docs/gui-workflow.md`
- `CHANGELOG.md`
