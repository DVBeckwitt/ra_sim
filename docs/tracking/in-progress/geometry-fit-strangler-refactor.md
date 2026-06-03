# Geometry Fit Strangler Refactor

Status: in-progress
Type: refactor
Owner:
Issue: none
Priority: p1
Last updated: 2026-06-03

## Summary

Refactor the geometry-fit runtime, dataset, source-row, coordinate, and optimizer paths by keeping the current public wrappers in place and moving behavior behind them in independently revertible slices. The first wave must not change solver math, payload schemas, UI callbacks, log fields, env flags, or saved-state fields.

## Slice status

Status: G1 dataset input-normalization extraction implemented, post-review ordering fix applied, and validated; ready for next-slice planning
Bug/error/feature status: Patch F0 audited `_run_async_geometry_fit_worker_job()` after E8 and deleted proven-dead worker-context alias assignments. Patch F0.1 then inlined the remaining one-use worker-context method aliases that were only wrapper glue. Patch F0.2 fixed the stale import-safe guard that still expected the deleted `_rebuild_source_rows_for_background_worker` alias. Patch F1 moved caked-storage event emission and non-gating caked-view task support into `GeometryFitWorkerContext` and deleted the corresponding runtime-local helpers and dependency callback seam. The post-F1 cleanup renamed two unused local unpack placeholders in the runtime wrapper to `_` without changing calls, call order, side effects, or behavior. Patch G0 added characterization coverage for `build_geometry_manual_fit_dataset()` public payload keys, detector-space objective selection, and caked-objective locked-Qr handoff audit fields before extracting dataset helpers. Patch G1 moved dataset input normalization only into `ra_sim/gui/geometry_fit_dataset.py`: caked-display input selection, enabled manual-pair filtering, manual-entry refresh snapshots, source labels, required-pair callback payload shape, manual picker truth rows, theta/base parameter copies, and reference scalar extraction. The post-review fix restored the original `load_background_by_index()` before manual-entry refresh/truth callback ordering and added regression coverage for failed background loads. Provider coverage, source candidate resolution, dynamic trial rows, caked dynamic reanchor, final dataset assembly, source-row rebuild, worker/runtime, optimizer, saved-state, CLI/env/debug behavior, UI behavior, solver math, and diagnostic log fields did not move.
Compatibility status: `ra_sim.gui.geometry_fit` remains the compatibility surface for moved contracts, and existing monkeypatch paths used by optimizer and caked reanchor tests remain available.
Migration/deprecation status: no public API is deprecated or removed. The new modules are internal extraction targets for the strangler refactor.
Shipping status: no runtime rollout or feature flag is needed because behavior is preserved behind existing public wrappers. Rollback is a normal commit revert.

## Patch F1 handoff

What was done: moved caked-storage event emission and non-gating
caked-view task support from the runtime worker closure into
`GeometryFitWorkerContext`, then deleted the matching runtime-local helpers and
required-cache dependency callbacks.

Bug/error/feature status: this is an internal refactor slice. It should not
change geometry-fit results, caked fit-space behavior, saved GUI-state format,
CLI behavior, environment flags, diagnostics, or user-visible GUI behavior.

CI/shipping status: targeted worker, runtime guard, GUI workflow, geometry
fitting, Ruff, compileall, and whitespace checks passed. No CI configuration,
dependency, release-version, rollout, feature-flag, or migration work is needed.

Deprecation/migration status: only an internal callback seam was removed. No
public API, artifact schema, config key, command, or saved-state field is
deprecated or migrated.

## Patch F1 post-review cleanup handoff

What was done: replaced two unused local unpack names inside
`_run_async_geometry_fit_worker_job()` with `_`.

Bug/error/feature status: this is a readability-only internal cleanup. It
preserves projection candidate-state checks, manual-targeted hit-table
filtering, and all wrapper side effects exactly.

CI/shipping status: targeted worker/runtime tests, Ruff on the touched runtime
file, and whitespace checks passed. No CI config, dependency, version, rollout,
feature flag, migration, or release work is needed.

Deprecation/migration status: no public API, artifact schema, config key,
command, saved-state field, or compatibility surface changed.

## Patch G0 dataset extraction contract map handoff

What was done: added characterization assertions around
`build_geometry_manual_fit_dataset()` before the next extraction slice. The
coverage pins the stable dataset payload key inventory, the detector-native
objective request in the baseline dataset path, and the caked-objective
handoff audit contract for a detector-origin locked-Qr anchor.

Bug/error/feature status: this is coverage and documentation only. It does not
move dataset code, change solver inputs, change caked fit-space behavior,
alter UI callbacks, or modify public payload schemas.

Interface/API status: no public API changed. The pinned interface remains the
existing dataset dictionary plus nested `spec` and `fit_handoff_audit_rows`
fields consumed by geometry-fit worker, optimizer, diagnostics, and saved
state paths.

CI/shipping status: targeted dataset workflow, GUI route-pack, and geometry
fitting route-pack tests passed. No CI config, dependency, release-version,
rollout, feature flag, migration, or launch work is needed for this guardrail
slice.

Deprecation/migration status: no public API, artifact schema, config key,
command, saved-state field, or compatibility surface is deprecated or
migrated.

## Patch G1 dataset input-normalization handoff

What was done: added `ra_sim/gui/geometry_fit_dataset.py` and moved only the
front input-normalization block used by `build_geometry_manual_fit_dataset()`
behind `collect_geometry_manual_dataset_inputs()`. The helper returns an
internal frozen snapshot and uses injected callables for geometry-fit-specific
behavior so it does not import `geometry_fit.py`, runtime, worker, optimizer,
Tk, Matplotlib, saved-state modules, or UI modules.

Bug/error/feature status: this is an internal refactor slice. It preserves
dataset payload keys, nested `spec` keys, objective-space selection,
manual-pair order, source identity fields, caked handoff audit rows, solver
inputs, saved-state behavior, CLI/env/debug behavior, UI callbacks, and
diagnostic log fields. The post-review fix preserves the previous failure
ordering where background load failure aborts before manual-entry refresh and
truth-pair callbacks run.

Interface/API status: no public API changed. The new dataclass and helper are
internal module-boundary tools for the next dataset extraction slices.

CI/shipping status: direct helper tests, dataset workflow tests, GUI route-pack
tests, geometry fitting caked route-pack tests, compileall, Ruff on touched
files, and whitespace checks passed. The post-review ordering regression,
focused build-dataset selector, helper tests, compileall, Ruff, and whitespace
checks also passed. `python -m ra_sim.dev check` remains blocked only by
pre-existing formatter drift in unrelated files. No CI config, dependency,
release-version, rollout, feature flag, migration, or launch work is needed for
this slice.

Deprecation/migration status: no public API, artifact schema, config key,
command, saved-state field, or compatibility surface is deprecated or
migrated.

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
- Moved caked-storage event/task support into `GeometryFitWorkerContext`.
- Cleaned unused local unpack placeholder names in the runtime wrapper.
- Patch G0 pinned `build_geometry_manual_fit_dataset()` payload keys,
  detector-native objective-space selection, and caked-objective handoff audit
  fields before moving any dataset assembly behavior.
- Patch G1 moved only `build_geometry_manual_fit_dataset()` input
  normalization into `ra_sim/gui/geometry_fit_dataset.py`; provider coverage,
  source candidate resolution, dynamic trial rows, caked dynamic reanchor, and
  final dataset assembly remain in `ra_sim/gui/geometry_fit.py`.
- Patch G1 post-review fix restored background-load-before-refresh callback
  ordering and pinned that failure behavior with a workflow regression test.
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
- Patch D3.3a.1 clarified that the D3.3a bundle helper generates
  caked/q-space projected rows only when projected rows are absent, and kept
  source-row cache lookup/rebuild helpers explicitly pending in the movement
  guard.
- Patch D3.3b moved single-background cache prebuild orchestration behind
  `GeometryFitWorkerContext`: `prebuild_background_cache_bundle_worker()`.
- Patch D3.3b injects only the runtime-local callbacks needed by the moved
  helper; required-background prebuild, source-row cache lookup/rebuild, manual
  validation, dataset, solver, optimizer, saved-state, CLI/env/debug, result
  packaging, and UI behavior did not move.
- Post-Patch-D3.3b size report: `_run_async_geometry_fit_worker_job()` is 2,465
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 45,146 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 1,421 lines.
- Patch D3.3c moved source-row cache lookup/rebuild helpers behind
  `GeometryFitWorkerContext`: `rebuild_source_rows_for_background_worker()` and
  `source_rows_for_background_worker()`.
- Patch D3.3c injects only the runtime-local callbacks needed by those moved
  helpers; required-background prebuild, manual validation, dataset, solver,
  optimizer, saved-state, CLI/env/debug, result packaging, and UI behavior did
  not move.
- Post-Patch-D3.3c.1 size report: `_run_async_geometry_fit_worker_job()` is 2,206
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,887 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 1,711 lines.
- Patch D3.3d moved required-background source-cache prebuild orchestration
  behind `GeometryFitWorkerContext.prebuild_required_background_caches()`.
- Patch D3.3d injects only the runtime-local callbacks needed for locked-Qr
  readiness, caked-view storage event emission, and caked-view wait/timeout
  handling; manual validation, dataset, solver request, solver execution,
  result packaging, optimizer, saved-state, CLI/env/debug, and UI behavior did
  not move.
- Post-Patch-D3.3d size report: `_run_async_geometry_fit_worker_job()` is 1,594
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,275 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,367 lines.
- Patch D3.3d.1 removed the one-use runtime wrapper lambda for required-cache
  prebuild and calls `worker_context.prebuild_required_background_caches()`
  directly with the same stage callback.
- Post-Patch-D3.3d.1 size report: `_run_async_geometry_fit_worker_job()` is
  1,590 lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,271 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,367 lines.
- Patch E1 moved manual fit-space validation helpers behind
  `GeometryFitWorkerContext`: `worker_manual_pairs_for_background()`,
  `worker_manual_fit_space_by_background()`,
  `worker_manual_caked_fit_space_required_for_background()`,
  `worker_validate_required_source_rows_for_fit_space()`, and
  `reject_worker_mixed_manual_fit_spaces()`.
- Patch E1 injects only the runtime-local manual fit-space classification and
  validation callbacks; caked-view ensure, dataset call, solver request, solver
  execution, result packaging, optimizer, saved-state, CLI/env/debug, and UI
  behavior did not move.
- Post-Patch-E1 size report: `_run_async_geometry_fit_worker_job()` is 1,516
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,197 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,510 lines.
- Patch E1.1 removed dead runtime aliases left by the E1 extraction and
  collapsed review-found blank-line bloat near the remaining caked-view ensure
  helpers. No behavior, public payload, saved-state, CLI/env, UI, dataset,
  solver, optimizer, or diagnostic contract changed.
- Post-Patch-E1.1 size report: `_run_async_geometry_fit_worker_job()` is 1,507
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,188 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,510 lines.
- Patch E2 moved caked-view readiness and ensure helpers behind
  `GeometryFitWorkerContext`: `worker_caked_view_payload_ready()` and
  `ensure_worker_geometry_fit_caked_view()`.
- Patch E2 uses existing caked payload and manual fit-space dependency
  injection only; caked-view storage, display-coordinate adapters,
  background-view source-row projection, dataset call, solver request, solver
  execution, result packaging, optimizer, saved-state, CLI/env/debug, and UI
  behavior did not move.
- Post-Patch-E2 size report: `_run_async_geometry_fit_worker_job()` is 1,467
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,148 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,562 lines.
- Patch E2.1 removed the dead `_worker_caked_view_payload_ready` runtime alias
  left after Patch E2 moved caked-view readiness into
  `GeometryFitWorkerContext`. No behavior, public payload, saved-state,
  CLI/env, UI, dataset, solver, optimizer, or diagnostic contract changed.
- Post-Patch-E2.1 size report: `_run_async_geometry_fit_worker_job()` is 1,466
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,147 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,562 lines.
- Patch E3 moved display/projection adapter helpers behind
  `GeometryFitWorkerContext`:
  `worker_native_detector_coords_to_detector_display_coords_for_background()`,
  `worker_geometry_manual_entry_display_coords()`, and
  `project_source_rows_for_background_view_worker()`.
- Patch E3 injects only the existing source-projection finite-pair and display
  rotation callbacks; caked-view storage, dataset call, solver request, solver
  execution, result packaging, optimizer, saved-state, CLI/env/debug, and UI
  behavior did not move.
- Post-Patch-E3 size report: `_run_async_geometry_fit_worker_job()` is 1,335
  lines, `ra_sim/gui/_runtime/runtime_session.py` is 44,016 lines, and
  `ra_sim/gui/_runtime/geometry_fit_worker.py` is 2,715 lines.
- Patch E3.1 moved source-projection dependency lookups behind the early
  invalid-input guard returns in the display adapter helpers. Invalid
  background identifiers and non-mapping manual entries now return `None`
  before dependency checks, matching the old nested helper edge behavior.
- Patch E3.1 added focused worker tests for those guard paths. No behavior,
  public payload, saved-state, CLI/env, UI, dataset, solver, optimizer, or
  diagnostic contract changed.
- Patch E3.2 moved the source-projection dependency lookup in
  `worker_geometry_manual_entry_display_coords()` behind the caked-pick scalar
  fallback. Caked manual entries with finite `caked_x`/`caked_y` now return
  before dependency checks, matching the old nested helper edge behavior.
- Patch E3.2 added a focused worker test for that caked scalar guard path. No
  public payload, saved-state, CLI/env, UI, dataset, solver, optimizer, or
  diagnostic contract changed.
- Patch E3.2 post-commit review found no correctness, bloat, security,
  performance, test-quality, unnecessary-file, or avoidable-abstraction issues.
  No migration, deprecation, rollout, feature flag, CI pipeline, or public
  documentation change is required beyond this tracking-note status update.
- Patch E4 moved `_store_worker_caked_view_for_background` into
  `GeometryFitWorkerContext.store_worker_caked_view_for_background()` behind
  explicit injected dependencies. `_run_async_geometry_fit_worker_job(job)`
  remains the runtime wrapper and aliases the moved method back to the old
  local helper name.
- Patch E4 moved caked-view storage only. Dataset construction, solver request
  assembly, solver execution, result packaging, optimizer code, saved-state
  code, CLI/env/debug behavior, UI behavior, manual validation, and display
  adapters did not move.
- Patch E4 added direct worker coverage for missing native backgrounds, invalid
  caked payloads, stale source-cache generations, successful payload/side-store
  writes, stage event keys/order, caked-mode projection-signature refresh, and
  projected-row cache refresh.
- Current measured size after Patch E4: `runtime_session.py` 43,538 lines;
  `geometry_fit_worker.py` 3,313 lines; `_run_async_geometry_fit_worker_job()`
  857 lines.
- Patch E4.1 removed unused fake caked-view storage integrator state from the
  worker tests and moved the missing-integrator control to the fake caked
  payload dependency, matching the production dependency owner.
- Patch E4.1 added direct worker coverage for the caked-view storage
  `missing_integrator` failure result shape and status. No production code,
  public payload, saved-state, CLI/env/debug behavior, UI behavior, dataset,
  solver, optimizer, or migration surface changed.
- Patch E5 moved the worker dataset preparation boundary into
  `GeometryFitWorkerContext.prepare_geometry_fit_run_for_worker()` behind
  explicit injected dataset dependencies.
- Patch E5 constructs worker manual dataset bindings with the same job-local
  callbacks, calls the injected `prepare_geometry_fit_run(...)`, and delegates
  the `build_dataset` callback to the injected
  `build_geometry_manual_fit_dataset(...)`.
- Patch E5 kept preflight exception log persistence, solver request assembly,
  solver execution, result packaging, optimizer code, saved-state code,
  CLI/env/debug behavior, and UI behavior in `runtime_session.py`.
- Post-Patch-E5 size report: `runtime_session.py` 43,428 lines;
  `geometry_fit_worker.py` 3,488 lines; `_run_async_geometry_fit_worker_job()`
  747 lines.
- Patch E5.1 removed unused `FakeManualDatasetBindings.kwargs` test fixture
  state. No production code, public payload, saved-state, CLI/env/debug
  behavior, UI behavior, dataset, solver, optimizer, migration surface, or
  rollout surface changed.
- Patch E6 moved the worker-side solver-phase keyword assembly into
  `GeometryFitWorkerContext.build_solver_phase_kwargs_for_worker()`.
- Patch E6 preserves the exact solver-phase call site in `runtime_session.py`
  via `gui_geometry_fit.execute_runtime_geometry_fit_solver_phase(...)`;
  solver request construction, solver execution, action-result packaging,
  optimizer behavior, saved-state behavior, CLI/env/debug behavior, and UI
  behavior did not move.
- Post-Patch-E6 size report: `runtime_session.py` 43,411 lines;
  `geometry_fit_worker.py` 3,521 lines; `_run_async_geometry_fit_worker_job()`
  730 lines.
- Patch E7 moved the worker-side solver execution call behind
  `GeometryFitWorkerContext.execute_solver_phase_for_worker()` and an injected
  `GeometryFitWorkerSolverDeps.execute_solver_phase` callable.
- Patch E7 keeps solver request construction inside `geometry_fit.py` and keeps
  worker cache clearing plus `_geometry_fit_worker_action_result(...)` result
  packaging in `runtime_session.py`.
- Patch E7 post-implementation review approved the production boundary:
  `geometry_fit_worker.py` remains import-clean, the injected solver dependency
  is minimal, and result packaging still lives in `runtime_session.py`.
- Post-Patch-E7 size report: `runtime_session.py` 43,411 lines;
  `geometry_fit_worker.py` 3,537 lines; `_run_async_geometry_fit_worker_job()`
  730 lines.
- Patch E8 moved only the worker-side result packaging call boundary behind
  `GeometryFitWorkerContext.build_action_result_for_worker()` and an injected
  `GeometryFitWorkerResultDeps.build_action_result` callable. The
  `_geometry_fit_worker_action_result(...)` builder remains in
  `runtime_session.py`.
- Patch E8 keeps worker cache clearing order, preflight failure log
  persistence, `prepared_run is None` handling, async polling, UI result
  application, solver execution, optimizer behavior, saved-state behavior,
  CLI/env/debug behavior, and UI behavior in their existing locations.
- Post-Patch-E8 size report: `runtime_session.py` 43,411 lines;
  `geometry_fit_worker.py` 3,562 lines; `_run_async_geometry_fit_worker_job()`
  730 lines.
- Patch E8 post-review cleanup renamed the safe-runtime worker helper movement
  guard to avoid stale wording after the result-boundary helper moved. No
  production code, runtime behavior, payload schema, migration path, or launch
  gate changed.
- Patch F0 audited `_run_async_geometry_fit_worker_job()` for stale
  worker-context alias glue and deleted only assignment-only aliases with direct
  search/AST evidence of no use. Still-used aliases for caked projection
  loading, caked storage, source-cache generation matching, manual fit-space
  validation, and result packaging were left in place.
- Post-Patch-F0 size report: `runtime_session.py` 43,373 lines;
  `geometry_fit_worker.py` 3,562 lines; `_run_async_geometry_fit_worker_job()`
  692 lines.
- Patch F0.1 inlined the remaining one-use worker-context method aliases in
  `_run_async_geometry_fit_worker_job()` and updated the runtime import-safety
  guard to assert the direct mixed-manual-space rejection call. No worker helper
  extraction, result payload, solver, dataset, optimizer, saved-state, CLI/env,
  UI, diagnostic, migration, or release-version behavior changed.
- Post-Patch-F0.1 size report: `runtime_session.py` 43,356 lines;
  `geometry_fit_worker.py` 3,562 lines; `_run_async_geometry_fit_worker_job()`
  675 lines.

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
- Patch D3.3c updated the movement guard so source-row cache lookup/rebuild
  helpers may exist in `geometry_fit_worker.py`; required-background prebuild
  remains nested in `runtime_session.py` until its later slice.

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
  - Snapshot-hit path: if the source snapshot signature matches, rows exist, and required-pair validation is absent or valid, builds/stores a background cache bundle from snapshot rows, updates source and simulation diagnostics from the bundle, and returns that bundle without a fresh rebuild.
  - Param override path: starts from `job_data["params"]`, injects per-background `theta_initial`, then applies the override. Trial signatures use the `geometry_fit_worker_trial_source_rows` tag plus digests of local params and the base requested signature.
  - Live rows payload behavior: live rows are forwarded only when the live-row signature matches the requested signature; mismatches return an empty rows payload with `requested_signature_mismatch` metadata.
  - Required-pair validation failure behavior: matching snapshots that fail required-pair validation record `background_cache_pair_validation_failed` source diagnostics before falling through to rebuild.
  - Caked payload order: ensure an active caked payload, use a ready stored projection payload when valid, hydrate it, then fall back to generated snapshot loading only when the candidate was absent.
  - Rebuild handling: calls the pure rebuild helper with equivalent cache, projection, live-row, targeted-cache, and stage callbacks; rebuild diagnostics replace worker source/simulation diagnostics.
  - Bundle storage: rebuild results with stored rows are converted through `_build_geometry_fit_background_cache_bundle`, stored through `_store_worker_background_cache_bundle`, and returned.
  - Empty/error behavior: empty rebuild results return `None`; caked projection failures still raise the existing runtime errors from the moved caked-payload helpers.
- `_prebuild_required_background_caches() -> None`
  - Inputs: required indices and job-local manual pairs from `job_data`.
  - Return shape: none.
  - Mutations/events: emits source-cache build/bundle start, ready, failed, and locked-Qr readiness diagnostics; stores successful bundles; advances cache generation through storage.
  - Fallback/error behavior: continues after ordinary failed bundle builds, but raises runtime timeout errors for targeted/fresh simulation timeout statuses.

E2 caked-view ensure helpers:

- `_worker_caked_view_payload_ready(background_index) -> bool`
  - Inputs: job-local background image snapshot, caked projection payload maps,
    caked view maps, generated payload resolver, and params.
  - Detector shape: derived from
    `job_data["background_images"][idx]["native"]` when available.
  - Payload lookup: calls projection snapshot loading with
    `allow_generated_payload=True`.
  - Exact bundle check: hydrates the caked payload with
    `require_background=False` and returns whether the injected transform-bundle
    predicate accepts the hydrated payload.
- `_ensure_worker_geometry_fit_caked_view() -> None`
  - Inputs: required indices, manual fit-space map, caked-required classifier,
    and caked-view readiness helper.
  - Ordering: rejects unsupported mixed manual fit spaces before caked readiness
    checks.
  - Filter: checks only required backgrounds that need caked fit-space.
  - Error behavior: raises
    `exact caked projector unavailable for background N` with one-based labels
    when a required exact caked payload is unavailable.

E5 dataset preparation boundary:

- `worker_manual_dataset_bindings` construction
  - Inputs: job-local osc files, current background index, image size, display
    rotation, manual-pair maps, background snapshots, orientation callbacks,
    source-row cache callbacks, caked payload loaders, and display/projection
    adapters.
  - Return shape: `GeometryFitRuntimeManualDatasetBindings` or equivalent
    runtime binding object from the injected constructor.
  - Mutations/events: none; callbacks may later read or update worker-local
    caches when `prepare_geometry_fit_run(...)` invokes them.
- `prepare_geometry_fit_run(...)` call boundary
  - Inputs: copied job-local params, var names, fit config, osc files,
    background selection state, theta state, manual-pair lookup, caked-view
    ensure callback, runtime config builder, caked-pick flags, and stage
    callback.
  - Return shape: the injected preparation result object unchanged.
  - Selection/theta behavior: preserves existing selection-error and
    background-theta-error RuntimeError paths from job snapshots.
  - Dataset callback: delegates to the injected
    `build_geometry_manual_fit_dataset(...)` with the same background index,
    theta base, base fit params, manual dataset bindings, orientation config,
    caked fit-space requirement, and stage callback.
  - Out-of-scope behavior: preflight exception logging, solver request
    assembly, solver execution, result packaging, optimizer, saved-state,
    CLI/env, and UI behavior remain in `runtime_session.py`.

E6 solver-phase input boundary:

- `build_solver_phase_kwargs_for_worker(prepared_run)`
  - Inputs: worker job data and the already prepared runtime dataset object.
  - Mutations: copies `fit_solver_mosaic_params` into
    `prepared_run.fit_params["mosaic_params"]` only when the job value is a
    mapping, matching the prior inline worker behavior.
  - Return shape: keyword dictionary for the existing
    `execute_runtime_geometry_fit_solver_phase(...)` call:
    `prepared_run`, `var_names`, `solve_fit`, `solver_inputs`, `stamp`,
    `log_path`, `event_callback`, and `live_update_callback`.
  - Event behavior: stage events flow through
    `GeometryFitWorkerContext.emit_event` with the same `job_id`, `kind`, and
    deep-copied `payload` shape; live cache update events are emitted only when
    `enable_live_update_events` is true.
  - Out-of-scope behavior: solver request construction, solver execution,
    result/action packaging, preflight failure handling, optimizer,
    saved-state, CLI/env, and UI behavior remain outside
    `geometry_fit_worker.py`.

E7 solver execution boundary:

- `GeometryFitWorkerSolverDeps`
  - Inputs: one injected `execute_solver_phase` callable owned by
    `runtime_session.py`.
  - Import boundary: the worker module does not import
    `gui_geometry_fit.execute_runtime_geometry_fit_solver_phase`; the runtime
    wrapper injects that callable.
- `execute_solver_phase_for_worker(prepared_run)`
  - Inputs: the already prepared runtime dataset object.
  - Behavior: builds the solver-phase kwargs through
    `build_solver_phase_kwargs_for_worker(prepared_run)` and calls only the
    injected executor.
  - Return shape: returns the executor result unchanged.
  - Out-of-scope behavior: worker cache clearing, action-result packaging,
    preflight failure handling, solver request construction, optimizer,
    saved-state, CLI/env, and UI behavior remain outside
    `geometry_fit_worker.py`.

E3 display/projection adapter helpers:

- `_worker_native_detector_coords_to_detector_display_coords_for_background(background_index)`
  - Inputs: job-local background image snapshots and display rotation.
  - Shape behavior: returns `None` when the background cannot be resolved or
    has no positive 2D native image shape.
  - Output: returns a named callback that maps native detector coordinates into
    display coordinates using the injected display-rotation callback and the
    background-specific shape.
- `_worker_geometry_manual_entry_display_coords(entry)`
  - Inputs: manual entry mapping, pick-space mode, current background index,
    finite-pair parser, and native-to-display callback.
  - Caked-pick behavior: preserves scalar caked/raw-caked/two-theta/phi/display
    fallback order before detector display fields.
  - Detector-pick behavior: preserves tuple display fields, scalar display
    fields, then native detector tuple/scalar projection fallback.
  - Error behavior: returns `None` for non-mapping entries, nonfinite values,
    missing native backgrounds, and invalid projection results.
- `_project_source_rows_for_background_view_worker(background_index, rows, **kwargs)`
  - Inputs: background index, source rows, optional `mode_override`, and optional
    `strict_caked_projection`.
  - Cache behavior: reuses matching cached projection rows before calling the
    source-row projection helper.
  - Projection behavior: falls back to
    `project_source_rows_for_background()` with the same mode override and
    default `strict_caked_projection=True`.

## Next actions

1. Resolve unrelated plotting files before the next production refactor by
   removing them from this branch or committing them separately.
2. Plan the next worker-runtime slice after F1. Do not move dataset internals,
   solver request/execution, result packaging, optimizer, saved-state,
   CLI/env/debug behavior, or UI behavior without a fresh plan.
3. Current measured size after F0: `runtime_session.py` 43,373 lines;
   `geometry_fit_worker.py` 3,562 lines; `_run_async_geometry_fit_worker_job()`
   692 lines.
4. Do not add hard debt gates yet.

## Patch F1 behavior map

Patch F1 may move only caked-storage event/task support from the runtime
worker closure into `GeometryFitWorkerContext`.

Behavior to preserve:

- Event kinds remain `source_cache_caked_view_ready`,
  `source_cache_caked_view_timeout`, and `source_cache_caked_view_failed`.
- Event payload defaults remain `background_index`, `background_label`,
  `source_cache_generation_id`, `row_count`, `elapsed_s`, and `message`.
- Timeout events still set `status=timeout` and use the existing
  `caked_view_timeout_s` wait text.
- Ready and failure message text still reports the current background number
  and the stored/failed status fallback.
- `late=True` still adds `late` to the emitted payload only for late events.
- Background label and row-count fallbacks still use the cache bundle fields.
- Non-gating storage still calls `store_worker_caked_view_for_background()`
  with the original `stage_callback` and `source_cache_generation_id`.
- Storage exceptions still map to a failed outcome with
  `exception:<ExceptionClass>` status fields.
- Initial result polling still waits in 0.01 second intervals until timeout.
- Timeout announcement still happens through the returned callback.
- Late events still emit only after timeout was observed, timeout was
  announced, and the source-cache generation still matches.
- Stale-generation late outcomes are still suppressed.

Patch F1 must not move dataset internals, solver request/execution, result
packaging, optimizer behavior, saved-state handling, CLI/env/debug behavior,
or UI behavior.

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
- Patch D3.3b validation passed: compileall, worker prebuild/cache tests,
  worker/job import-boundary tests, live-row/runtime/import-safe guard tests,
  GUI workflow route tests, geometry fitting route tests, Ruff on touched files,
  and `git diff --check`.
- Patch D3.3c validation passed: compileall, worker source-row cache
  lookup/rebuild tests, worker/job import-boundary tests,
  live-row/runtime/import-safe guard tests, GUI workflow route tests, geometry
  fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch D3.3c.1 validation passed: worker source-row cache tests,
  worker/job import-boundary tests, GUI runtime geometry tests, Ruff on
  `runtime_session.py`, and `git diff --check`.
- Patch D3.3d validation passed: compileall, worker required-cache tests,
  worker/job import-boundary tests, live-row/runtime/import-safe guard tests,
  GUI workflow route tests, geometry fitting route tests, Ruff on touched files,
  and `git diff --check`.
- Patch D3.3d.1 validation passed: worker required-cache tests,
  worker/job import-boundary tests, GUI runtime geometry tests, the targeted
  runtime import-safe source-row guard, Ruff on touched files, and
  `git diff --check`.
- Patch E1 validation passed: compileall, worker manual fit-space tests,
  worker/job import-boundary tests, live-row/runtime/import-safe guard tests,
  GUI workflow route tests, geometry fitting route tests, Ruff on touched files,
  and `git diff --check`.
- Patch E1.1 validation passed: worker/job import-boundary tests, targeted
  runtime import-safe guards, Ruff on `runtime_session.py`, and
  `git diff --check`.
- Patch E2 validation passed: compileall, worker caked-view readiness/ensure
  tests, worker/job import-boundary tests, live-row/runtime/import-safe guard
  tests, GUI workflow route tests, geometry fitting route tests, Ruff on touched
  files, and `git diff --check`.
- Patch E2.1 validation passed: worker/job import-boundary tests, GUI runtime
  geometry tests, Ruff on `runtime_session.py`, and `git diff --check`.
- Patch E3 validation passed: compileall, worker display/projection adapter
  tests, worker/job import-boundary tests, live-row/runtime/import-safe guard
  tests, GUI workflow route tests, geometry fitting route tests, Ruff on touched
  files, and `git diff --check`.
- Patch E3.1 validation passed: worker display/projection adapter tests,
  full worker tests, worker/job import-boundary tests, Ruff on touched files,
  and `git diff --check`.
- Patch E3.2 validation passed: worker display/projection adapter tests,
  full worker tests, worker/job import-boundary tests, Ruff on touched files,
  and `git diff --check`.
- Patch E4 validation passed: compileall, full worker tests, worker/job
  import-boundary tests, GUI runtime tests, GUI runtime import-safe tests,
  live-row/signature handoff tests, GUI workflow caked/dataset route tests,
  geometry fitting route tests, Ruff on touched files, and `git diff --check`.
- Patch E4.1 validation passed: focused caked-view storage worker tests,
  full worker tests, worker/job import-boundary tests, the targeted GUI runtime
  import-safe caked-storage ordering guard, Ruff on touched tests, and
  `git diff --check`.
- Patch E5 validation passed: compileall, worker dataset-boundary tests, full
  worker tests, worker/job import-boundary tests, GUI runtime tests, full GUI
  runtime import-safe tests, live-row/signature handoff tests, GUI workflow
  caked/dataset route tests, geometry fitting route tests, Ruff on touched
  files, and `git diff --check`.
- Patch E5.1 validation passed: focused worker dataset-boundary tests, full
  worker tests, Ruff on touched tests, and `git diff --check`.
- Patch E6 validation passed: compileall, focused solver-phase input tests,
  full worker tests, worker/job import-boundary tests, GUI runtime tests, full
  GUI runtime import-safe tests, live-row/signature handoff tests, GUI workflow
  caked/dataset route tests, geometry fitting route tests, Ruff on touched
  files, and `git diff --check`.
- Patch E7 validation passed: compileall, focused solver execution-boundary
  tests, full worker tests, worker/job import-boundary tests, GUI runtime
  tests, full GUI runtime import-safe tests, live-row/signature handoff tests,
  GUI workflow caked/dataset route tests, geometry fitting route tests, Ruff on
  touched files, and `git diff --check`.
- Patch E8 validation passed: compileall, focused result packaging-boundary
  tests, full worker tests, worker/job import-boundary tests, GUI runtime
  tests, full GUI runtime import-safe tests, live-row/signature handoff tests,
  GUI workflow caked/dataset route tests, geometry fitting route tests, Ruff on
  touched files, and `git diff --check`.
- Patch E8 post-review cleanup validation passed: worker/job import-boundary
  tests and Ruff on `tests/test_geometry_fit_safe_runtime.py`.
- Patch F0 validation passed: compileall, full worker tests, worker/job
  import-boundary tests, and GUI runtime geometry tests.
- Patch F0.1 validation passed: compileall, full worker tests, worker/job
  import-boundary tests, targeted GUI runtime import-safe worker guards, GUI
  runtime geometry tests, Ruff on touched files, and `git diff --check`.
- Patch F0.2 validation passed: focused stale worker-wrapper guard,
  targeted GUI runtime import-safe worker guards, full worker tests, worker/job
  import-boundary tests, GUI runtime geometry tests, Ruff on touched tests, and
  `git diff --check`.
- Patch F1 validation passed: compileall, full worker tests, worker/job
  import-boundary tests, live-row/runtime/import-safe guard tests, GUI workflow
  caked/dataset route tests, geometry fitting route tests, Ruff on touched
  files, and `git diff --check`.
- Patch F1 post-review cleanup validation passed: targeted worker/job
  import-boundary tests, focused worker projection/prebuild/rebuild tests, Ruff
  on `runtime_session.py`, and `git diff --check`.
- Patch G0 validation passed: targeted
  `tests/test_gui_geometry_fit_workflow.py` dataset contract tests, the GUI
  workflow caked/dataset/locked-Qr route pack, and the geometry fitting
  caked/manual-Qr/locked-Qr route pack.
- Patch G1 validation passed: direct `tests/test_geometry_fit_dataset.py`
  helper tests, targeted dataset workflow tests, the GUI workflow
  caked/dataset/locked-Qr route pack, the geometry fitting caked route pack,
  compileall, Ruff on touched files, and `git diff --check`.
- Patch G1 post-review fix validation passed: focused
  `load_failure_precedes_refresh_and_truth` workflow regression, full
  `build_geometry_manual_fit_dataset` selector, direct helper tests, compileall,
  Ruff on touched files, and `git diff --check`.
- `python -m ra_sim.dev check` remains blocked only by the documented pre-existing formatting drift above.
- No generated artifacts, raw data, local config, notebook output, dependency changes, release version changes, or public migration files are included.

## Links

- Related plan: geometry-fit strangler refactor with detector/caked/live-row/projection authority characterization.
