# GUI runtime selective update fast paths

Type: optimization
Status: implemented
Last updated: 2026-04-28
Issue: none

## Summary

The GUI runtime now separates update classification from execution and uses
safe fast paths for cache-valid presentation, combination, primary-prune, and
detector-center changes. Conservative full simulation remains the fallback for
physics-affecting or mixed changes.

## What changed

- Added pure update dependency signatures and `classify_update(...)`.
- Added trace fields for action, reason, worker requirement, missing primary
  contribution count, detector-center remap use, and primary-prune cache mode.
- Added selective invalidation helper for display, combine, prune, detector
  remap, analysis, hit-table refresh, and full simulation actions.
- Added primary-prune reuse and fill routing through the existing primary
  contribution cache.
- Added exact detector-center remap helpers and runtime branch. Remap requires
  detector-relative or otherwise exact/unclipped hit tables.
- Added display-only and combine-only runtime branches guarded by valid cached
  images.
- Added end-to-end scenario tests for full -> prune -> display -> center ->
  physics update sequences and stale-worker protection.

## Status

- Feature status: implemented for the explicit fast paths listed above.
- Bug status: stale worker overwrite cases covered for display-only,
  prune-reuse, and detector-center-remap fast paths.
- Error status: no known failing targeted GUI runtime fast-path tests in this
  worktree after Phase 8 validation.
- Fallback status: conservative. Missing exact remap cache, clipped-only remap
  cache, incompatible cache signatures, secondary-active remap without exact
  secondary cache, source/lattice/physics changes, and mixed physics changes
  still request full simulation.

## Validation

Final pre-merge requested suite passed:

```bash
python -m pytest -ra tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_import_safe.py
```

Result: `408 passed`.

Manual GUI-runtime trace smoke passed:

| Update | Trace action | Worker |
| --- | --- | --- |
| Initial update | `full_simulation` | `requires_worker=True` |
| Prune change with cached keys | `primary_prune_reuse` | `requires_worker=False` |
| Display-only change | `display_only` | `requires_worker=False` |
| Combine/visibility change | `combine_only` | `requires_worker=False` |
| Detector center shift with exact cache | `detector_center_remap` | `requires_worker=False` |
| Detector center shift without exact cache | `full_simulation` | `requires_worker=True` |
| Detector distance change | `full_simulation` | `requires_worker=True` |
| Detector rotation/tilt change | `full_simulation` | `requires_worker=True` |

Passed:

```bash
python -m pytest tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py -ra
```

Result: `92 passed`.

Passed broad GUI runtime slice by expanding `tests/test_gui_runtime_*.py` in
PowerShell and adding `tests/test_gui_sim_signature.py`.

Result: `448 passed`.

Passed:

```bash
python -m compileall ra_sim/gui tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_update_dependencies.py -q
```

## Notes

No wall-clock performance assertion was added. Performance regression coverage
uses deterministic worker-call counts instead.
