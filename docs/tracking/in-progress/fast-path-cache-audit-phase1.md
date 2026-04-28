# Fast Path Cache Audit and QR Selector Policy

Date: 2026-04-28

Scope: Phases 1-3.5 of fast-path cache verification, selective invalidation,
and local validation gating.

## Current Status

- Feature status: implemented and validated for QR selector cache policy,
  selective runtime invalidation, local optional-artifact skips, slow/manual
  geometry markers, fast geometry-fitter handoff checks, and geometry objective
  source-row reuse baselines.
- Bug status: fixed for overbroad fast-path invalidation of QR selector entries,
  source-row snapshots, intersection caches, and fitter handoff state; fixed
  local checkpoint failures caused by absent optional New4 artifacts.
- Error status: no known failing local Phase 3.5 gate tests.
- Compatibility status: QR disabled/enabled masks remain explicit user/state
  data and are not cleared by cache invalidation.

## File existence

- `ra_sim/gui/runtime_qr_selector_cache_policy.py`: present.
- `tests/test_runtime_qr_selector_cache_policy.py`: present.
- `tests/test_gui_runtime_invalidation.py`: present.
- `tests/test_gui_runtime_optimization_scenarios.py`: present.
- `tests/test_geometry_objective_cache.py`: missing. Equivalent baseline added to `tests/test_fit_cache_controls.py`.

## Mutation Map

The table below records the Phase 1 pre-policy audit. Current post-policy
runtime status is summarized in the Phase 2, Phase 3, and Phase 3.5 sections.

| Classification | Mutation or reuse sites |
| --- | --- |
| DISPLAY_ONLY | `runtime_update_dependencies.classify_update` returns `DISPLAY_ONLY` for display-only signature changes. `runtime_session.do_update` fast path calls `_invalidate_for_update_action(DISPLAY_ONLY)`, which is a no-op in `runtime_invalidation.invalidate_for_update_action`. QR selector masks, source snapshots, hit tables, and manual pick cache state are retained. |
| COMBINE_ONLY | `classify_update` returns `COMBINE_ONLY` for combine signature changes. Runtime fast path calls `_invalidate_for_update_action(COMBINE_ONLY)`, also a no-op. Combined image is republished from stored primary/secondary images; QR selector masks and source snapshots are retained. |
| PRIMARY_PRUNE_REUSE | Runtime image-change path first calls `_invalidate_geometry_manual_pick_cache`, which clears manual pick cache data, `_hkl_pick_simulation_points_payload_cache`, and `geometry_q_group_entries_cache`. Then `_invalidate_for_update_action(PRIMARY_PRUNE_REUSE)` clears combined artifacts including `stored_q_group_content_signature` and `stored_intersection_cache`, but retains primary contribution caches. Runtime rematerializes primary artifacts from `primary_hit_table_cache`, stores primary side cache artifacts, refreshes the primary relative remap cache signature, and captures a new source-row snapshot after publish. |
| PRIMARY_PRUNE_FILL | Runtime image-change path first calls `_invalidate_geometry_manual_pick_cache`. `_invalidate_for_update_action(PRIMARY_PRUNE_FILL)` is a no-op, then a `primary_fill` worker job fills missing contribution keys. When the ready result is applied, runtime stores the new primary contribution payload, applies primary artifacts, updates `stored_hit_table_signature`, and captures a new source-row snapshot. |
| DETECTOR_CENTER_REMAP | Runtime center-remap fast path calls `_invalidate_geometry_manual_pick_cache`, rematerializes primary/secondary artifacts from detector-relative hit tables, updates side intersection cache signatures, then calls `_invalidate_analysis_cache(clear_visuals=True)` and `_invalidate_for_update_action(DETECTOR_CENTER_REMAP)`. The action invalidation clears analysis/caked/q-space caches and `source_row_snapshots`; the later restore path calls `_invalidate_peak_picker_caches(clear_source_snapshot=True)` and then captures a fresh source-row snapshot when rows are available. |
| ANALYSIS_ONLY | `classify_update` can return `ANALYSIS_ONLY`; `runtime_invalidation` clears only analysis/caked/q-space caches for that action. Current `runtime_session.do_update` does not include `ANALYSIS_ONLY` in its fast-path allow-list, so the runtime path fails closed to `FULL_SIMULATION` rather than using the helper action directly. |
| FULL_SIMULATION | `runtime_invalidation.invalidate_for_update_action(FULL_SIMULATION)` broadly clears combined artifacts, side artifacts, primary contribution caches, secondary remap caches, analysis caches, source snapshots, and peak selection/Q-group entry caches. Current `runtime_session.do_update` does not call this helper directly; full regeneration paths clear manual picker caches before requesting or applying full simulation work, and then overwrite stored side/combined artifacts from the result. |
| GEOMETRY_OBJECTIVE_CACHE_REUSE | `optimization._build_trial_qr_source_rows_payload` reuses `fit_context["prediction_source_rows_cache"]` by `(dataset_index, params_signature, builder_kind)`, marking reused payloads as `source_rows_rebuilt_or_reused="reused_for_same_params_signature"` and `reuse_valid_for_same_params_signature=True`. `_resolve_saved_sim_caked_alignment` also reuses `dataset_ctx._qr_fit_saved_sim_caked_alignment_cache` by locked branch key and saved anchor. |
| GEOMETRY_OBJECTIVE_FULL | `optimization._build_trial_qr_source_rows_payload` rebuilds rows on cache miss and stores the payload under the same key. `_resolve_qr_fit_prediction_from_trial_params` records rebuilt trial source rows and `stale_prediction_cache_used_for_trial_params=False`. Simulation image caches use `SimulationCache` and `_simulate_with_cache`, gated by `_retain_fit_simulation_cache`. |
| STATE_LOAD | `state_io.apply_geometry_state_snapshot` applies explicit saved Q-group masks with `replace_geometry_q_group_masks`, requests a Q-group refresh, and calls `invalidate_geometry_manual_pick_cache`. Legacy saved rows queue pending legacy disabled Qz sections, request refresh, and invalidate manual pick cache. Runtime restored peak records also call `_invalidate_geometry_manual_pick_cache`. |
| USER_SELECTION_CHANGE | `controllers.replace_geometry_q_group_masks`, `set_geometry_q_group_row_enabled`, `clear_geometry_q_group_masks`, `prune_geometry_q_group_masks`, and `resolve_pending_geometry_q_group_legacy_masks` are the explicit mask mutation sites for `disabled_qr_sets`, `disabled_qz_sections`, and `pending_legacy_disabled_qz_sections`. They update `mask_revision`; cache invalidation helpers do not mutate these explicit user selections. |
| EXPLICIT_RESET | `runtime_session._initialize_runtime_controls_block_28`, `runtime_invalidation.invalidate_for_update_action(FULL_SIMULATION)`, and `primary_cache_helpers.reset_combined_simulation_artifacts` reset stored simulation/combined artifacts. `controllers.consume_geometry_q_group_refresh_request` explicitly clears only `refresh_requested`; update-action invalidation does not clear it. |

## Key Symbols

- `_invalidate_peak_picker_caches` clears manual picker cache data, payload cache, `geometry_q_group_entries_cache_signature`, `geometry_q_group_entries_cache`, and optionally the current background entry in `source_row_snapshots`.
- `_invalidate_geometry_manual_pick_cache` delegates to `_invalidate_peak_picker_caches(clear_source_snapshot=False)`.
- `geometry_q_group_entries_cache` and `geometry_q_group_entries_cache_signature` are written in `geometry_q_group_manager.capture_runtime_geometry_q_group_entries_snapshot` and cleared by picker/full/hit-table invalidation paths.
- `stored_q_group_content_signature` is written when combined or reconstructed rows are published and cleared by combined/full resets.
- `source_row_snapshots` are populated by runtime source snapshot capture/rebuild paths and by headless geometry-fit snapshot export; detector remap and full invalidation clear GUI runtime snapshots before fresh capture.
- `stored_primary_*` and `stored_secondary_*` side caches are applied from worker results, primary prune reuse/fill, and detector-center remap. They are cleared by full invalidation or secondary-side absence.

## Baseline Tests Added

- `tests/test_gui_runtime_invalidation.py` now asserts the update-action QR selector cache policy and verifies update-action invalidation never mutates explicit Q-group user selection state.
- `tests/test_fit_cache_controls.py` now asserts geometry objective trial source rows are reused only for the same params signature and rebuilt for a new params signature.

## Phase 2 Policy Work

Added `ra_sim/gui/runtime_qr_selector_cache_policy.py` as the pure policy
boundary for QR selector and geometry fitter handoff cache retention. The policy
returns explicit retain/refresh/defer decisions for each `UpdateAction` and
fails closed for unknown or conflicting inputs.

Required behavior now covered by `tests/test_runtime_qr_selector_cache_policy.py`:

- `DISPLAY_ONLY`, `COMBINE_ONLY`, and `ANALYSIS_ONLY` retain selector and
  handoff caches.
- `PRIMARY_PRUNE_REUSE` keeps masks, keeps q-group entries when content is
  unchanged, and requests refresh when q-group content changes.
- `PRIMARY_PRUNE_FILL` keeps old q-group entries while replacement rows are
  pending and defers refresh until rows are available.
- `DETECTOR_CENTER_REMAP` retains branch/source identity and marks projection
  geometry as refresh-needed when detector geometry changes.
- `FULL_SIMULATION` retains QR masks but does not reuse stale source rows across
  physics or hit-table signature changes.

## Phase 3 Runtime Invalidation Work

Runtime invalidation now routes fast paths through the QR selector policy before
clearing picker-facing caches. Broad invalidation remains available for full
simulation refreshes, but selector masks are asserted unchanged.

New narrow invalidation operations:

- q-group entries cache only
- source-row snapshots only
- intersection caches only
- manual pick projection cache only

Runtime behavior after Phase 3:

- `DISPLAY_ONLY`, `COMBINE_ONLY`, and `ANALYSIS_ONLY` preserve QR selector and
  fitter handoff caches.
- `PRIMARY_PRUNE_REUSE` preserves selector state when content signatures are
  unchanged and schedules refresh when q-group content changes.
- `PRIMARY_PRUNE_FILL` keeps old QR entries until replacement rows apply.
- `DETECTOR_CENTER_REMAP` preserves branch/source identity across exact remaps
  while refreshing detector/caked projection geometry as needed.
- `FULL_SIMULATION` clears stale rows and cache payloads when physics or
  hit-table signatures change, while retaining QR masks.

Phase 3 validation:

- `python -m pytest tests/test_runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py -q`
  passed, `23 passed`
- `python -m pytest tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_optimization_scenarios.py -q`
  passed, `22 passed`
- `python -m py_compile ra_sim/gui/runtime_invalidation.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py`
  passed

## Phase 3.5 Local Validation Gate

Phase 3.5 converted optional and slow geometry gates into explicit local-gate
categories without changing production runtime behavior.

Feature status:

- New4 artifact-backed tests skip cleanly when
  `artifacts/geometry_fit_gui_states/new4.json` is absent.
- Long New4 caked/refined geometry diagnostics are marked `slow_geometry`.
- `tests/test_gui_runtime_geometry_fitter_handoff_fast.py` provides fast
  synthetic QR selector and geometry-fitter handoff coverage without running
  the optimizer or requiring New4 artifacts.

Bug/error status:

- Missing New4 fixture now reports `skipped` instead of `FileNotFoundError`.
- No stale QR selector, source-row, intersection-cache, or manual-pick handoff
  defect was exposed by the fast substitute tests.
- Slow/manual caked/refined objective slice remains excluded from the local gate
  by instruction and is not counted as a pass.

Phase 3.5 validation:

- `python -m pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv`
  passed with `26 passed, 2 skipped`
- `python -m pytest tests/test_gui_runtime_geometry_fitter_handoff_fast.py -q`
  passed, `5 passed`
- local Phase 3.5 pytest gate passed, `438 passed`
- second manual geometry gate passed, `5 passed`
- py-compile gate passed
- static audits found no cache invalidation mutation of QR/Qz masks and no
  broad picker helper reachability from non-physics fast paths
