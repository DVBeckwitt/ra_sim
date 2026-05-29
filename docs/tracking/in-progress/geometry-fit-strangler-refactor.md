# Geometry Fit Strangler Refactor

Status: in-progress
Type: refactor
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-29

## Summary

Refactor the geometry-fit runtime, dataset, source-row, coordinate, and optimizer paths by keeping the current public wrappers in place and moving behavior behind them in independently revertible slices. The first wave must not change solver math, payload schemas, UI callbacks, log fields, env flags, or saved-state fields.

## Slice status

Status: ready for commit
Bug/error/feature status: internal refactor guardrails only; no user-facing geometry-fit behavior, saved-state schema, CLI, environment flag, solver math, UI callback, or diagnostic log-field change is intended in this slice.
Compatibility status: `ra_sim.gui.geometry_fit` remains the compatibility surface for moved contracts, and existing monkeypatch paths used by optimizer and caked reanchor tests remain available.
Migration/deprecation status: no public API is deprecated or removed. The new modules are internal extraction targets for the strangler refactor.
Shipping status: no runtime rollout or feature flag is needed because behavior is preserved behind existing public wrappers. Rollback is a normal commit revert.

## Current state

- Added canonical snapshot normalization for geometry-fit comparison tests.
- Pinned normalized async job handoff output for `_build_geometry_fit_async_job()`.
- Pinned normalized dataset contract output for `build_geometry_manual_fit_dataset()`.
- Pinned normalized optimizer result and diagnostic output for `fit_geometry_parameters()`.
- Moved geometry-fit contract dataclasses and callback aliases into `ra_sim/gui/geometry_fit_contracts.py`.
- Re-exported the moved contract names from `ra_sim/gui/geometry_fit.py`.
- Restored the `geometry_fit._detector_pixels_to_fit_space` monkeypatch alias required by existing caked dynamic-reanchor tests.
- Added `ra_sim/gui/geometry_fit_coordinates.py`.
- Moved finite value, finite pair, entry-frame, detector anchor, and caked-angle helper behavior behind imported pure helpers.
- Added `ra_sim/gui/_runtime/geometry_fit_job.py`.
- Extracted async job selection/theta decisions into `resolve_geometry_fit_selection()`.
- Fixed snapshot normalization so integer background keys and string keys cannot collapse.
- Fixed snapshot normalization to fail closed when any normalized mapping keys collide, including typed aliases such as `0` versus `"int:0"` and unstable timestamp-like string keys.
- Removed the unused snapshot JSON helper.
- Added a caked projection-authority snapshot that pins exact-projector use over stale saved caked aliases.
- Switched the extracted coordinate finite check to stdlib `math.isfinite()` after float coercion.
- Patch B extracted pure caked projection-anchor helpers into `ra_sim/gui/geometry_fit_coordinates.py`:
  `observed_detector_anchor_for_caked_projection()`,
  `simulated_detector_anchor_for_caked_projection()`,
  and `project_detector_anchor_to_caked_fit_space()`.
- Wired `build_geometry_manual_fit_dataset()` to use the observed, simulated, and exact-projector helper paths while leaving public dataset assembly in `ra_sim/gui/geometry_fit.py`.
- Preserved public dataset payloads, saved-state shape, optimizer request shape, solver math, UI callbacks, CLI/env flags, log fields, and caked projection-authority behavior.
- Patch B.1 restored the previous caked projection diagnostic ordering so `valid=False` exact-projector failures preserve `invalid_reason` even when returned caked coordinates are nonfinite.
- Patch C0 added direct coverage for `resolve_geometry_fit_selection()` no-selection, theta-metadata-not-applied, background-theta-error, and skipped-empty-background branches.
- Patch C extracted the async job builder background/manual/source input snapshots, live-row handoff, runtime config snapshot, caked projection payload capture, projection-view signatures, current hit-table cache payload, and final job dict assembly into `ra_sim/gui/_runtime/geometry_fit_job.py`.
- Patch C kept `_build_geometry_fit_async_job()` as the runtime entry point, kept the final worker job as a plain dict, and did not move worker, optimizer, dataset, source-row rebuild, saved-state, CLI/env/debug-flag, solver, or UI callback behavior.
- Post-Patch-C size report: `_build_geometry_fit_async_job()` is 420 lines, `ra_sim/gui/_runtime/runtime_session.py` is 46,186 lines, and `ra_sim/gui/_runtime/geometry_fit_job.py` is 1,131 lines.

## Review status

- Reviewed the current diff for correctness, bloat, security, performance, test quality, and unnecessary new abstractions.
- No required correctness, security, or performance blockers were found.
- Follow-up before deeper job/dataset extraction was completed in Patch C0 by adding direct helper coverage for the remaining `resolve_geometry_fit_selection()` edge branches.
- Snapshot helper now raises on normalized mapping-key collisions instead of silently dropping entries.
- Patch B simplification removed the unused `resolve_fit_space_anchor()` helper and its self-only tests because it did not remove production dataset assembly logic in this slice.
- Patch C added an import-boundary guard proving `geometry_fit_job.py` does not import `runtime_session.py`, Tk, worker modules, or GUI mutation modules.

## Next actions

1. Start worker boundary preparation with an import-boundary test and worker context/event/snapshot utilities.
2. Keep optimizer, dataset, source-row rebuild, and update-policy extraction out of the next worker-boundary slice.
3. Revisit `_build_geometry_fit_async_job()` size after worker-boundary helpers are in place; do not add hard debt gates yet.

## Validation

Initial snapshot slice:

```bash
python -m pytest -q tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_canonical_snapshot_pins_live_row_handoff_contract tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload tests/test_geometry_fitting.py::test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray
```

Contract extraction slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m ruff check ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_contracts.py tests/helpers/geometry_fit_snapshots.py tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fitting.py tests/test_gui_geometry_fit_workflow.py
```

Coordinate helper slices:

```bash
python -m pytest -q tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload tests/test_geometry_fit_snapshot_contracts.py
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m ruff check ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_contracts.py ra_sim/gui/geometry_fit_coordinates.py tests/helpers/geometry_fit_snapshots.py tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fitting.py tests/test_gui_geometry_fit_workflow.py
```

Async selection slice:

```bash
python -m pytest -q tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_canonical_snapshot_pins_live_row_handoff_contract tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_builds_q_group_rows_for_noncurrent_required_background
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m ruff check ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_contracts.py ra_sim/gui/geometry_fit_coordinates.py ra_sim/gui/_runtime/geometry_fit_job.py ra_sim/gui/_runtime/runtime_session.py tests/helpers/geometry_fit_snapshots.py tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fitting.py tests/test_gui_geometry_fit_workflow.py
```

Cleanup guard slice:

```bash
python -m pytest -q tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py::test_caked_point_reprojection_uses_detector_pixel_path tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_canonical_snapshot_pins_live_row_handoff_contract tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload tests/test_geometry_fitting.py::test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m ruff check ra_sim/gui/geometry_fit.py ra_sim/gui/geometry_fit_contracts.py ra_sim/gui/geometry_fit_coordinates.py ra_sim/gui/_runtime/geometry_fit_job.py ra_sim/gui/_runtime/runtime_session.py tests/helpers/geometry_fit_snapshots.py tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fitting.py tests/test_gui_geometry_fit_workflow.py
git diff --check
```

Patch B caked projection-anchor slice:

```bash
python -m pytest -q tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py::test_caked_point_reprojection_uses_detector_pixel_path tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_canonical_snapshot_pins_live_row_handoff_contract tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload tests/test_geometry_fitting.py::test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray
python -m pytest -q tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py::test_caked_point_reprojection_uses_detector_pixel_path tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload
python -m pytest -q tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py::test_caked_point_reprojection_uses_detector_pixel_path tests/test_gui_geometry_fit_workflow.py::test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload tests/test_geometry_fitting.py::test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray
python -m pytest -q tests/test_geometry_fit_coordinates.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or caked or coordinate or optimizer_request"
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_snapshot_contracts.py tests/test_geometry_fit_coordinates.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or caked or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m ruff check ra_sim/gui/geometry_fit_coordinates.py ra_sim/gui/geometry_fit.py tests/test_geometry_fit_coordinates.py tests/test_gui_geometry_fit_workflow.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_geometry_fit_disordered_preflight.py
git diff --check
python -m ra_sim.dev check
```

Patch C0 async selection edge-coverage slice:

```bash
python -m pytest -q tests/test_geometry_fit_job_selection.py
python -m pytest -q tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py::test_geometry_fit_job_canonical_snapshot_pins_live_row_handoff_contract
```

Patch C async job-builder extraction slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_job.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_geometry_fit.py
git diff --check
python -m ra_sim.dev check
```

Known baseline issue:

```bash
python -m ra_sim.dev check
```

fails on pre-existing formatting in `ra_sim/fitting/optimization.py`, `ra_sim/gui/_runtime/runtime_session.py`, and `ra_sim/test_tiers.py`; those files were not reformatted in this slice.

Current commit readiness:

- Narrow focused tests, compileall, targeted geometry/runtime/import-safe suites, geometry fitting route tests, GUI workflow route tests, Ruff on touched files, and `git diff --check` passed for this slice.
- `python -m ra_sim.dev check` remains blocked only by the documented pre-existing formatting drift above.
- No generated artifacts, raw data, local config, notebook output, dependency changes, release version changes, or public migration files are included.

## Links

- Related plan: geometry-fit strangler refactor with detector/caked/live-row/projection authority characterization.
