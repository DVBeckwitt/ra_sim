# Cold-Start Caked Manual Fit Projector

Status: implemented, validation partial
Type: bug
Owner: -
Issue: #249
Priority: p1
Last updated: 2026-05-04

## Summary

Cold-start manual geometry fitting in caked view could fail before the
optimizer started. The runtime reached `source_cache_full_cake_ready`, then
lost the exact caked fit-space projector during exact payload
hydration/storage and surfaced `exact caked projector unavailable` for the
selected background.

The fix splits geometric exact-caked projection metadata from display
intensity data. Exact projector hydration now uses detector shape, caked axes,
raw azimuth axis, row permutation, and `CakeTransformBundle`; display
background finiteness is validated only for image-facing paths such as caked
display storage and peak-refinement image use.

## Current state

- `geometry_fit.py` accepts projection-only exact caked payloads when
  `require_background=False`, while `require_background=True` remains strict
  for image use.
- `geometry_fit_caked_projection_payload` strips caked payloads down to the
  geometric fields needed for exact projector hydration.
- Caked display cache storage sanitizes only zero-support nonfinite density
  bins to `0.0`; supported-bin nonfinite values fail with
  `nonfinite_supported_caked_background`.
- Runtime caked preflight stores and hydrates
  `projection_payload_by_background[bg_idx]` before row projection or dataset
  build.
- Caked projection signatures are computed from projection payloads, not
  display caked images.
- Manual-fit dataset prep uses projection readiness for
  `fit_space_projector_kind=exact_caked_bundle` and keeps image readiness
  separate.
- Worker/manual fit code uses a separate caked projection accessor; the
  image-facing `geometry_manual_caked_view_for_index` accessor is not fed
  axes-only projection payloads.
- Targeted fresh simulation now emits timeout diagnostics and marks later
  ready/failed completion as late when a timeout fired.

## Bug/error/feature status

- Bug status: fixed for the projector hydration/cache split covered by focused
  tests.
- Error status: `invalid_exact_caked_payload` is replaced at the touched
  storage/preflight sites by field-level statuses including
  `projection_payload_ready`, `missing_exact_caked_bundle`,
  `projection_payload_missing_axes`, `projection_payload_axis_mismatch`,
  `nonfinite_supported_caked_background`, and
  `empty_bin_nan_density_sanitized`.
- Feature status: cold-start caked manual fitting can build the exact
  fit-space projector without requiring a caked display image to be finite in
  undefined empty bins.

## Validation

Passing targeted checks:

- `python -m py_compile ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_geometry_fit_workflow.py tests/test_gui_runtime_import_safe.py`
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "caked_view or exact_caked or manual_prepare" -ra`
- `python -m pytest tests/test_gui_geometry_fit_workflow.py -k "exact_caked_hydration_ignores_nonfinite or caked_display_sanitizer or projection_payload_without_image_ready or targeted_fresh_simulation_timeout" -ra`
- `python -m pytest tests/test_geometry_fitting.py -k "exact_caked" -ra`
- `python -m ruff format --check ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_geometry_fit_workflow.py tests/test_gui_runtime_import_safe.py`
- `git diff --check -- CHANGELOG.md ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py tests/test_gui_geometry_fit_workflow.py tests/test_gui_runtime_import_safe.py`

Remaining validation gaps:

- Manual GUI acceptance still needs a fresh-start run: generate once, pick two
  caked manual points, run fit, no parameter change.
- The broader workflow selector
  `tests/test_gui_geometry_fit_workflow.py -k "exact_caked or caked_manual"`
  still has two failing New4 ladder finalizer cases outside the touched
  projector storage path.
- `python -m ra_sim.dev check` is currently blocked before tests by format
  drift in `ra_sim/dev_doctor.py` and dirty pre-existing
  `ra_sim/fitting/optimization.py`.

## Next actions

- Run the manual cold-start GUI acceptance path on the real workflow.
- Triage the two New4 ladder finalizer failures separately from the exact
  caked projector cache fix.
- Clean or intentionally format the broader dirty worktree before running the
  full project check.

## Links

- Related tracker: [new4-geometric-fitter-recovery-handoff.md](new4-geometric-fitter-recovery-handoff.md)
- Related tracker: [deterministic-geometry-runtime-fix-pass.md](deterministic-geometry-runtime-fix-pass.md)
