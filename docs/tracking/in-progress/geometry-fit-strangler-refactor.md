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

Status: Patch D3.3a background cache bundle construction extraction complete; ready for review
Bug/error/feature status: internal worker refactor only; no user-facing geometry-fit behavior, saved-state schema, CLI, environment flag, solver math, UI callback, or diagnostic log-field change is intended in this slice.
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
- Patch C.1 removed an unused private helper parameter from `snapshot_geometry_fit_background_inputs()` and its runtime call site.
- Post-Patch-C.1 size report: `_build_geometry_fit_async_job()` is 419 lines, `ra_sim/gui/_runtime/runtime_session.py` is 46,185 lines, and `ra_sim/gui/_runtime/geometry_fit_job.py` is 1,130 lines.
- Patch D1 added `ra_sim/gui/_runtime/geometry_fit_worker.py` and moved worker context setup, event emission, source-cache generation helpers, diagnostic helpers, and background image snapshot loading behind `GeometryFitWorkerContext`.
- Patch D1 kept `_run_async_geometry_fit_worker_job(job)` as the runtime worker wrapper and did not move caked payload loading, caked view hydration, source-row projection, cache prebuild, manual validation, dataset calls, solver calls, optimizer behavior, saved-state, CLI/env/debug flags, or UI callbacks.
- Post-Patch-D1 size report: `_run_async_geometry_fit_worker_job()` is 3,485 lines, `ra_sim/gui/_runtime/runtime_session.py` is 46,129 lines, and `ra_sim/gui/_runtime/geometry_fit_worker.py` is 114 lines.
- Patch D2 moved worker caked payload status/loading helpers behind `GeometryFitWorkerContext`:
  `caked_projection_payload_status()`, `projection_candidate_state()`,
  `load_caked_view_by_index_snapshot()`,
  `load_caked_projection_by_index_snapshot()`, and
  `ensure_worker_caked_projection_payload()`.
- Patch D2 injects caked payload dependencies from `runtime_session.py` and keeps
  `geometry_fit_worker.py` free of runtime, Tk, GUI, optimizer, matplotlib, and
  saved-state imports.
- Patch D2 kept `_run_async_geometry_fit_worker_job(job)` as the runtime worker
  wrapper and did not move source-row projection, cache prebuild, manual
  fit-space validation, dataset calls, solver calls, optimizer behavior,
  saved-state behavior, CLI/env/debug flags, result packaging, or UI callbacks.
- Post-Patch-D2 size report: `_run_async_geometry_fit_worker_job()` is 3,255 lines,
  `ra_sim/gui/_runtime/runtime_session.py` is 45,936 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 365 lines.
- Patch D3.0 added test/docs guardrails for source-row projection and cache reuse
  before moving any D3 production helpers.
- Patch D3.0 did not move production behavior. Source-row projection, cache
  marking, bundle storage, cache prebuild, manual validation, dataset, solver,
  optimizer, saved-state, CLI/env/debug flags, result packaging, and UI callbacks
  remain in their existing locations.
- Patch D3.1 moved only the worker source-row projection helpers behind
  `GeometryFitWorkerContext`:
  `project_source_rows_for_background()` and
  `project_source_rows_by_row_background()`.
- Patch D3.1 injects source projection dependencies from `runtime_session.py`,
  keeps `_run_async_geometry_fit_worker_job(job)` as the runtime worker wrapper,
  and aliases the moved methods back to the old local helper names.
- Patch D3.1 kept cached projection-row marking, projection-row matching, bundle
  row reuse, bundle storage, cache bundle build/prebuild, manual validation,
  dataset calls, solver calls, optimizer behavior, saved-state behavior,
  CLI/env/debug flags, result packaging, and UI callbacks in their existing
  locations.
- Post-Patch-D3.1 size report: `_run_async_geometry_fit_worker_job()` is 3,046
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 45,727 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 676 lines.
- Patch D3.1.1 reviewed the detector-mode empty-projection fallback. Removing
  it breaks the existing live-row signature handoff guard, so the fallback
  remains narrowly limited to current-background live rows with source
  provenance and finite caked anchors; stored caked rows without live source
  provenance are still not promoted.
- Current post-Patch-D3.1.1 size report: `_run_async_geometry_fit_worker_job()` is 3,084
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 45,727 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 678 lines.
- Patch D3.2 moved cached projection-row marking, cached projection-row
  matching, bundle row reuse, and worker background cache bundle storage behind
  `GeometryFitWorkerContext`.
- Patch D3.2 injects only a background-cache-bundle predicate and source-row
  copy callback from `runtime_session.py`; no D3.3 prebuild, bundle factory,
  manual-validation, dataset, solver, optimizer, saved-state, CLI/env/debug,
  or UI behavior moved.
- Post-Patch-D3.2 size report: `_run_async_geometry_fit_worker_job()` is 2,993
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 45,636 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 815 lines.
- Patch D3.3a moved background cache bundle construction behind
  `GeometryFitWorkerContext`:
  `build_geometry_fit_background_cache_bundle()`.
- Patch D3.3a injects only the existing background-cache-bundle constructor and
  optional-value copier from `runtime_session.py`; cache prebuild orchestration,
  source-row cache lookup/rebuild, manual validation, dataset, solver,
  optimizer, saved-state, CLI/env/debug, result packaging, and UI behavior did
  not move.
- Post-Patch-D3.3a size report: `_run_async_geometry_fit_worker_job()` is 2,847
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 45,528 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 939 lines.

## Review status

- Reviewed the current diff for correctness, bloat, security, performance, test quality, and unnecessary new abstractions.
- No required correctness, security, or performance blockers were found.
- Follow-up before deeper job/dataset extraction was completed in Patch C0 by adding direct helper coverage for the remaining `resolve_geometry_fit_selection()` edge branches.
- Snapshot helper now raises on normalized mapping-key collisions instead of silently dropping entries.
- Patch B simplification removed the unused `resolve_fit_space_anchor()` helper and its self-only tests because it did not remove production dataset assembly logic in this slice.
- Patch C added an import-boundary guard proving `geometry_fit_job.py` does not import `runtime_session.py`, Tk, worker modules, or GUI mutation modules.
- Patch D1 added an import-boundary guard proving `geometry_fit_worker.py` does not import `runtime_session.py`, Tk, GUI mutation modules, `geometry_fit.py`, `manual_geometry.py`, optimizer modules, or matplotlib.
- Patch D2 extended the duplicate-helper guard so caked payload helpers cannot remain
  nested in `_run_async_geometry_fit_worker_job()` while also existing on the
  worker context.
- Patch D3.0 added a no-production-movement guard proving the D3 source-row
  projection/cache helper names have not been introduced in
  `geometry_fit_worker.py` yet.
- Patch D3.1 updated the movement guard so only the D3.1 source-row projection
  helpers may exist in `geometry_fit_worker.py`; D3.2/D3.3 cache helpers remain
  forbidden there until their slices.
- Patch D3.1.1 resolved the D3.1 review question by validating that removing
  the detector-mode empty-projection fallback regresses accepted live-row
  handoff behavior. The fallback is now documented in code and remains covered
  by worker helper tests.
- Patch D3.2 updated the movement guard so source-row projection and cache row
  bundle helpers may exist in `geometry_fit_worker.py`; D3.3 cache
  construction/prebuild helpers remain forbidden there until their slices.
- Patch D3.3a updated the movement guard so background cache bundle construction
  may exist in `geometry_fit_worker.py`; prebuild/source-row cache lookup helpers
  remain forbidden there until their later slices.

## D3 contract map

D3.1 source-row projection helpers:

- `_project_source_rows_for_background(background_index, raw_rows, *, mode_override=None, strict_caked_projection=True, params_override=None) -> Sequence[object]`
  - Inputs: background index, raw source rows, optional projection mode override, strict caked failure mode, and optional parameter override.
  - Return shape: list-like projected row payloads from `_geometry_fit_rows_for_background(...)`; returns `[]` when there are no rows, when q-space is requested for a noncurrent background, or when q-space projection cannot run.
  - `job_data` mutations: for caked mode, may update `job_data["projection_payload_by_background"][background_index]` with a stored exact projection payload.
  - Cache map mutations: none outside the caked projection payload map.
  - Source-cache generation side effects: none.
  - Event emissions: none.
  - Fallback order: normalize rows, normalize mode, q-space current-background projector path, detector/caked background payload lookup, caked projection payload load/hydrate when mode is caked, projection callback construction, then `project_peaks_to_current_view`.
  - Empty/error behavior: detector mode falls back to normalized rows if projection callback raises; q-space returns `[]`; caked mode re-raises only when `strict_caked_projection` is true, otherwise returns `[]`.
- `_project_source_rows_by_row_background(raw_rows) -> list[dict[str, object]]`
  - Inputs: rows that may span multiple row-background identities.
  - Return shape: copied projected row dicts sorted by the temporary `__ra_sim_projection_row_order__` key, with that temporary key removed before return.
  - `job_data` mutations: only those caused by `_project_source_rows_for_background` for each group.
  - Cache map mutations: only the caked projection payload map through the per-background helper.
  - Source-cache generation side effects: none.
  - Event emissions: none.
  - Fallback order: group rows by background with current background as default, project each group, restore original row order.
  - Empty/error behavior: returns `[]` when grouping yields no rows.

D3.2 cached projection-row and bundle-storage helpers:

- `_mark_worker_cached_projection_rows(rows, *, background_index, mode) -> list[dict[str, object]]`
  - Inputs: row dicts, background index, projection mode.
  - Return shape: the same list object, with rows mutated only when mode is `caked` or `q_space`.
  - Row marker fields: `_geometry_fit_worker_cached_projection=True`, `_geometry_fit_worker_projection_mode=<mode>`, `_geometry_fit_worker_projection_background_index=<int background_index>`.
  - Empty/error behavior: detector or unknown modes return rows unchanged.
- `_worker_cached_projection_rows_match(rows, *, background_index, mode) -> bool`
  - Inputs: row mappings, background index, projection mode.
  - Return shape: boolean.
  - Freshness rules: mode must be `caked` or `q_space`; rows must be nonempty; every row must have cached marker true, matching mode, and matching background index.
  - Empty/error behavior: returns `False` for empty rows, detector/unknown modes, missing/noninteger background marker, or any stale row.
- `_bundle_rows(bundle, *, mode_override=None, params_override=None) -> list[dict[str, object]]`
  - Inputs: `GeometryFitBackgroundCacheBundle`, optional mode override, optional params override.
  - Return shape: copied or projected row dicts for the bundle background; returns `[]` for non-bundles or failed strict projection paths.
  - Reuse rules: valid cached projected rows are reused before considering `params_override`; any nonmatching projected rows are marked and reused when `params_override is None` and requested mode matches current base mode; caked/q-space can reproject stored rows when projected rows are absent/nonmatching and params override or mode mismatch prevents reuse.
  - `job_data` mutations: through projection helpers only.
  - Source-cache generation side effects: none.
- `_store_worker_background_cache_bundle(bundle) -> int`
  - Inputs: `GeometryFitBackgroundCacheBundle`.
  - Return shape: new source-cache generation id.
  - Mutations: stores bundle in `worker_background_cache_by_index`, writes a deep-copied source snapshot into `worker_source_row_snapshots`, and advances source-cache generation through `_advance_source_cache_generation`.
  - Snapshot fields: background index, requested signatures/summaries, stored rows, projected rows, row counts, diagnostics, projection view signature, and picker/dataset validity booleans.

D3.3a background cache bundle construction helper:

- `_build_geometry_fit_background_cache_bundle(...) -> GeometryFitBackgroundCacheBundle`
  - Inputs: background metadata, requested signature data, theta values, stored rows, optional projected rows, cache source, diagnostics, peak/hit/intersection caches, and cache metadata.
  - Return shape: `GeometryFitBackgroundCacheBundle` with copied stored/projected rows and copied optional table/cache payloads.
  - Projection rules: when mode is caked or q-space, projected rows are generated when projected rows are absent; stale projected-row rejection remains upstream in cache reuse logic. Projection failure status is recorded in diagnostics.
  - Mutations: may update projection payload map through projection helpers; otherwise returns a bundle.

D3.3b/D3.3c cache prebuild helpers:

- `_prebuild_background_cache_bundle_worker(...) -> GeometryFitBackgroundCacheBundle | None`
  - Inputs: background index, theta base, optional parameter set, consumer, prior diagnostics, required pairs, and stage callback.
  - Return shape: cache bundle or `None`.
  - Mutations/events: emits worker/stage diagnostics while collecting live rows, cache rows, fresh simulation rows, caked payload status, and projection results; can update worker diagnostics and projection payload maps.
  - Fallback order: live/job-local rows, runtime/source snapshots, targeted/manual fallback, then fresh simulation when needed.
  - Empty/error behavior: returns `None` with diagnostics for missing rows, timeout, stale signature, or projection failure conditions according to current status strings.
- `_prebuild_required_background_caches() -> None`
  - Inputs: required indices and job-local manual pairs from `job_data`.
  - Return shape: none.
  - Mutations/events: emits source-cache build/bundle start, ready, failed, and locked-Qr readiness diagnostics; stores successful bundles; advances cache generation through storage.
  - Fallback/error behavior: continues after ordinary failed bundle builds, but raises runtime timeout errors for targeted/fresh simulation timeout statuses.

## Next actions

1. Patch D3.3b should move single-background cache prebuild orchestration only:
   `_prebuild_background_cache_bundle_worker`.
2. Keep required-background prebuild, source-row cache lookup/rebuild, manual
   validation, dataset, solver, optimizer, saved-state, CLI/env, and UI behavior
   out of D3.3b.
3. Do not add hard debt gates yet.

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

Patch D1 worker context/event/snapshot boundary slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py
python -m pytest -q tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py
git diff --check
python -m ra_sim.dev check
```

Patch D2 worker caked payload loading slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_selection.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py
python -m pytest -q tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py
git diff --check
python -m ra_sim.dev check
```

Patch D3.0 worker projection/cache contract coverage slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py
python -m ruff check tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py tests/test_geometry_fit_job_live_rows_handoff.py
git diff --check
```

Patch D3.1 worker source-row projection helper slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py tests/test_geometry_fit_job_live_rows_handoff.py tests/test_gui_runtime_import_safe.py
git diff --check
python -m ra_sim.dev check
```

Patch D3.1.1 worker source-row projection cleanup slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py tests/test_geometry_fit_worker.py
git diff --check
python -m ra_sim.dev check
```

Patch D3.2 worker cache row bundle helper slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_import_safe.py
git diff --check
python -m ra_sim.dev check
```

Patch D3.3a background cache bundle construction slice:

```bash
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_geometry_fit_worker.py
python -m pytest -q tests/test_geometry_fit_safe_runtime.py -k "geometry_fit_worker or geometry_fit_job"
python -m pytest -q tests/test_geometry_fit_job_live_rows_handoff.py tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_gui_runtime_geometry_fit.py tests/test_gui_runtime_import_safe.py
python -m pytest -q tests/test_gui_geometry_fit_workflow.py -k "build_geometry_manual_fit_dataset or prepare_geometry_fit_run or caked or locked_qr or coordinate or optimizer_request"
python -m pytest -q tests/test_geometry_fitting.py -k "fit_geometry_parameters or caked or manual_qr or locked_qr or dynamic_point or objective_insensitive or full_beam"
python -m ruff check ra_sim/gui/_runtime/geometry_fit_worker.py ra_sim/gui/_runtime/runtime_session.py tests/test_geometry_fit_worker.py tests/test_geometry_fit_safe_runtime.py tests/test_gui_runtime_import_safe.py
git diff --check
python -m ra_sim.dev check
```

Known baseline issue:

```bash
python -m ra_sim.dev check
```

fails on pre-existing formatting in `ra_sim/fitting/optimization.py`, `ra_sim/gui/_runtime/runtime_session.py`, and `ra_sim/test_tiers.py`; those files were not reformatted in this slice.

Current validation status:

- Patch C broad validation previously passed: geometry fitting route tests, GUI workflow route tests, job/runtime/import-safe suites, Ruff on touched files, and `git diff --check`.
- Patch C.1 focused validation passed: compileall, job-selection/live-row tests, geometry-fit job import-boundary test, GUI runtime geometry test, Ruff on touched files, and `git diff --check`.
- Patch D1 validation passed: compileall, worker context tests, worker/job import-boundary tests, job/live-row/runtime/import-safe suites, GUI workflow route tests, geometry fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch D2 validation passed: compileall, worker caked payload tests, worker/job import-boundary tests, job/live-row/runtime/import-safe suites, GUI workflow route tests, geometry fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch D3.0 validation passed: compileall, worker context/source-cache tests, worker/job import-boundary tests, live-row/runtime guard tests, Ruff on touched test files, and `git diff --check`.
- Patch D3.1 validation passed: compileall, worker source-row projection tests, worker/job import-boundary tests, live-row/runtime/import-safe guard tests, GUI workflow route tests, geometry fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch D3.1.1 validation passed: compileall, worker source-row projection tests, worker/job import-boundary tests, live-row/runtime/import-safe guard tests, Ruff on touched files, and `git diff --check`.
- Patch D3.2 validation passed: compileall, worker cache row bundle tests, worker/job import-boundary tests, live-row/runtime/import-safe guard tests, GUI workflow route tests, geometry fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch D3.3a validation passed: compileall, worker background cache bundle
  construction tests, worker/job import-boundary tests,
  live-row/runtime/import-safe guard tests, GUI workflow route tests, geometry
  fitting route tests, Ruff on touched files, and `git diff --check`.
- `python -m ra_sim.dev check` remains blocked only by the documented pre-existing formatting drift above.
- No generated artifacts, raw data, local config, notebook output, dependency changes, release version changes, or public migration files are included.

## Links

- Related plan: geometry-fit strangler refactor with detector/caked/live-row/projection authority characterization.
