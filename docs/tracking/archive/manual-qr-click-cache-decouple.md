# Manual Qr/Qz Click Cache Decouple

Status: completed
Type: bug/performance/feature
Owner: Codex
Issue: none
Priority: p1
Last updated: 2026-05-05

## Summary

Manual Qr/Qz mouse clicks no longer synchronously build, fully refine, or
rebuild picker caches. Update/prewarm paths may build caches; click paths
consume warm caches or use local no-build refinement sentinels. Background Qr
reference placement and redraw no longer require simulated Qr inventory.

## What Changed

- Added manual-pick cache timing/count metadata for build, source-row provider,
  fresh-simulation fallback, QR refinement, lookup rebuild, and total
  `_get_pick_cache()` time.
- Added `geometry_manual_no_build_cache_sentinel(...)` and
  `allow_cache_build=False` support so mouse-click refinement cannot trigger a
  hidden picker-cache build through `build_cache_data`.
- Split background Qr reference placement from simulated picker inventory.
  Background references use local peak refinement, save no HKL, stay disabled
  for geometry fitting, and redraw directly from saved coordinates.
- Added `_get_pick_cache(reuse_only=True)`. Reuse-only misses return a
  temporary `cache_ready=False` payload and never replace warm cache state.
  Warm hits require grouped candidates plus matching refinement and lookup
  signatures.
- Added `prewarm_pick_cache()` to manual cache callbacks/workflow/runtime and
  call it only when simulation/background/view state is stable.
- Replaced fragile ndarray/callback object-ID signature parts with sampled
  content, generation, callable code/default/closure, and semantic
  simulation/projection tokens.
- Deduplicated QR simulated-candidate refinement by view mode plus source,
  branch, group, and HKL identity before coordinate fallback.
- Added end-to-end guards proving warm QR selection/placement clicks do not
  call cache build, fresh simulation, QR refinement, or lookup rebuild.

## Status

- Bug status: fixed for the targeted 30 s manual Qr/Qz click-latency path.
- Error status: fixed after follow-up hardening. Click-triggered session refresh,
  saved-pair redraw, active-session redraw, and saved-pair refinement now stay
  reuse-only/no-build during mouse clicks.
- Feature status: implemented. Prewarm/update paths own cache construction;
  QR/Qz click paths consume warm cache only; background Qr references avoid
  simulated inventory for placement and redraw.
- Compatibility status: no package version bump and no persisted manual-pair
  schema change. Background reference rows remain additive fit-disabled manual
  metadata.

## Follow-up hardening

- Runtime placement refresh uses reuse-only cache access and preserves the active
  session exactly on reuse-only misses.
- Click-triggered redraw forwards `reuse_only=True` into both saved-pair display
  and active-session display callbacks, so redraw side effects cannot rebuild
  source rows, fresh simulation rows, caked projection lookups, QR refinement, or
  QR lookup maps.
- Saved-pair refinement accepts an explicit `reuse_only` mode. Mouse-click
  placement passes `reuse_only=True`, and provided source entries bypass picker
  cache access entirely, including empty source-entry dicts.
- Detector-mode prewarm still warms the detector cache when the optional hidden
  caked sidecar is unavailable, but prewarm skips active, queued, or pending
  simulation/integration updates.

## Validation

Passed:

- `python -m compileall -q ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_geometry_interaction.py`
- `python -m pytest tests/test_manual_geometry_live_peak_cache.py -q`
  (`61 passed`)
- `python -m pytest -p no:ddtrace tests/test_manual_geometry_selection_helpers.py -k "(reuse_only or toggle or sidecar or placement or caked_picker) and not headless_replay" -q`
  (`38 passed, 489 deselected`)
- `python -m pytest -p no:ddtrace tests/test_gui_runtime_geometry_interaction.py tests/test_runtime_qr_selector_cache_policy.py -q`
  (`23 passed`)
- `python -m pytest -p no:ddtrace tests/test_gui_runtime_import_safe.py -k "prewarm or reuse_only or source_entry or sidecar" -q`
  (`12 passed, 391 deselected`)

Not run:

- `python -m ra_sim.dev check` because `ruff` is not installed in the validation
  environment.

## Links

- [Debug and cache guide](../../debug-and-cache.md#geometry-qrqz-and-hkl-picker-state)
- [Changelog](../../../CHANGELOG.md)
