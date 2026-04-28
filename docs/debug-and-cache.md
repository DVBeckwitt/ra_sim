# Debug and cache guide

This page is the short operational guide for logging, debug output, and cache
retention. Use it when you need to understand where RA-SIM writes diagnostics
or why a debug path is noisy or silent.

See also:

- [docs index](index.md)
- [Troubleshooting guide](troubleshooting.md)
- [Canonical logging section](simulation_and_fitting.md#logging-debug-and-cache-controls)

## Main Control Surface

RA-SIM uses `config/debug.yaml` as the primary debug and logging configuration.

Important points:

- `debug.global.disable_all: true` is the master kill switch for user-facing debug/log output
- the simulation GUI launcher can temporarily force diagnostics off or on for one run without rewriting `config/debug.yaml`
- `debug.console.enabled` controls console debug printing
- geometry-fit and mosaic-fit log file creation have dedicated toggles
- optional retained caches are controlled separately from logging output

## Output Locations

The default output directories come from `config/dir_paths.yaml`.

Common outputs:

- runtime trace logs
- geometry-fit logs
- mosaic-shape fit logs
- projection-debug JSON
- diffraction debug CSV
- intersection-cache dump folders

## Cache Policy

The cache policy only affects optional retained caches. It does not turn off the
active runtime state needed for the current simulation or GUI session.

Retention modes:

- `never`: build on demand, then discard
- `auto`: retain when the active feature benefits from reuse
- `always`: retain whenever built

## GUI Runtime Update Fast Paths

The GUI runtime records a pure update decision before it executes an update.
The trace fields are:

- `update_action`
- `update_reason`
- `requires_worker`
- `missing_contribution_count`
- `center_remap_used`
- `primary_prune_cache_mode`

Current action status:

- `full_simulation`: feature status implemented; still the conservative
  fallback for source, lattice, detector distance, detector rotation, pixel
  size, wavelength, beam sampling, mosaic sampling, solve-q, or mixed physics
  changes
- `primary_prune_reuse`: feature status implemented; rematerializes the
  primary image from cached contribution hit tables and does not request a full
  worker when all active keys are cached
- `primary_prune_fill`: feature status implemented; requests only missing
  primary contribution keys and preserves already cached keys
- `detector_center_remap`: feature status implemented; runs only with exact
  detector-relative/unclipped hit-table caches and falls back to full simulation
  for missing, clipped-only, stale, incompatible, or secondary-missing remap
  caches
- `display_only`: feature status implemented; redraws existing stored image
  state only when a valid stored image exists
- `combine_only`: feature status implemented; recombines cached primary and
  secondary images only when matching component images exist
- `analysis_only`: classifier status implemented; runtime execution remains
  conservative outside the explicit fast paths

Bug/error status:

- stale full-worker results are discarded before newer display, prune-reuse, or
  detector-center-remap fast-path state can be overwritten
- `last_dependency_signatures` is updated only after the applied state reflects
  the current request, with display/combine paths allowed to advance dependency
  signatures when the stored numeric image is current and no worker is pending
- detector-center remap invalidates caking/q-space/detector-geometry analysis
  caches while preserving reusable source/physics contribution caches
- broad full-simulation invalidation remains the fallback for physics changes
  and signature-incompatible cache states

Validation status as of 2026-04-28:

- `python -m pytest tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py -ra`
  passed, `92 passed`
- broad GUI runtime slice
  `python -m pytest tests/test_gui_runtime_*.py tests/test_gui_sim_signature.py -ra`
  passed via explicit PowerShell file expansion, `448 passed`
- `python -m compileall ra_sim/gui tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_update_dependencies.py -q`
  passed

## Geometry Qr/Qz and HKL Picker State

Manual Qr/Qz picking uses a structural group cache derived from the active
simulation hit tables and CIF/lattice state. It is not a detector-view or
caked-view cache. Detector/caked view switches must not invalidate or filter
the Qr/Qz group universe. CIF, unit-cell, or simulation-hit-table changes do
invalidate it.

Resolved picker behavior depends on keeping structural simulation truth
separate from current-view projection. Group membership comes from structural
state; detector or caked coordinates are derived later for the active view.

Caked manual picking uses two different coordinate responsibilities:

- simulated Qr/Qz and HKL seed positions start from simulation-native detector
  branch pixels;
- caked click targets map those simulation-native pixels through the live caked
  simulation transform into `(2theta, phi)`;
- detector aliases such as `sim_col`, `sim_row`, `display_col`, and
  `display_row` remain detector/display coordinates;
- caked aliases such as `caked_x`, `caked_y`, `raw_caked_x`, `raw_caked_y`,
  `two_theta_deg`, and `phi_deg` hold current-view angular coordinates.

For source-backed caked Qr/Qz selection, the detector-to-caked projection cache
is the authority for hit testing, active selected markers, and saved-pair
redraw. The cache is keyed by stable source/branch identity and stores the
native detector point, detector display point, caked visual point, and caked
`(2theta, phi)`. Saved or refined aliases can still describe the measured
background point, but they must not override the simulated Qr/Qz marker for a
source-backed saved pair.

The HKL picker intentionally shares the corrected Qr/manual picker candidate
payload for hit testing and selected-marker placement. If either picker
regresses, first check whether the failing path bypassed that shared candidate
payload or treated detector/display aliases as caked coordinates.

## Weighted-event diffraction status

Current status:

- feature status: active normal runtime path
- optimization status: fixed for duplicate weighted-candidate projection and
  exact-preserving Q-set precompute
- bug status: slow Python raw-candidate enumeration fixed
- error status: covered invariants green in weighted-event regression tests

What changed:

- `process_peaks_parallel(...)` and `process_peaks_parallel_safe(...)` now run the
  weighted-event path through `_process_peaks_parallel_impl(...)`, which calls
  Numba pass-1/pass-2 helpers instead of enumerating raw candidates with Python
  lists and dicts in the hot loop
- off-detector, non-finite, or bilinear-unsupported candidates are rejected
  before they enter `V`, so they cannot affect event selection or image mass
- duplicate sampled ordinals still produce duplicate hit rows, duplicate sampled
  cache rows, and duplicate best-sample event counts; only image deposition may
  aggregate repeated ordinals internally
- branch representatives stay separate from sampled events; fitter-facing
  `get_last_intersection_cache()` still returns representative rows, while
  `get_last_intersection_cache_views()` exposes both sampled event rows and
  branch representative rows
- `get_last_process_peaks_weighted_event_stats()` exposes weighted-event debug
  counters and timers, including solve/project/select counts and pass-1/pass-2
  mass totals for test assertions
- the fast weighted-event path stores valid projected candidates during the
  first pass and emits selected events from those stored buffers during the
  second phase, so `_project_weighted_candidate_fast(...)` is not called twice
  for the same candidate in the default path
- memory-bounded fallback keeps the old `_weighted_event_pass2_for_qset(...)`
  projection path available for debugging and oversized samples
- candidate-reuse stats are reported as
  `n_stored_projected_candidates`, `candidate_buffer_capacity_max`, and
  `candidate_buffer_fallback_count`; `n_project_candidate_calls` now counts
  projection calls only, not stored-candidate emission
- the fast serial weighted-event path precomputes unique `(peak_idx, rep_idx)`
  Q sets into flat NumPy tables before pass 1, then both pass 1 and pass 2 use
  integer `qset_id` lookups instead of the previous Python dict cache
- Q-set precompute is intentionally exact-preserving: it does not group by
  `(Gr, Gz)`, does not change `solve_q(...)` inputs, and does not alter
  projection, event selection, image deposition, or hit-table semantics
- Q-set precompute stats are reported as `n_qsets_precomputed`,
  `n_qset_lookup_entries`, `n_qset_reuse_hits`, and `time_qset_index`

Still intentionally disabled for weighted events:

- source-template replay
- clustered beam replacement
- grouped event emission by `(Gr, Gz)`
- sampling from representative/cache rows

## Numba on-disk compilation cache

RA-SIM sets `NUMBA_CACHE_DIR` at package import time for stable startup behavior.
When `NUMBA_CACHE_DIR` is unset, RA-SIM defaults to `~/.cache/ra_sim/numba`.
If a value is already set, RA-SIM leaves it unchanged.

RA-SIM does not force a Numba CPU mode. `NUMBA_CPU_NAME=generic` is not
enabled by default.

To inspect cache activity:

1. `set NUMBA_DEBUG_CACHE=1` before launch (or `export NUMBA_DEBUG_CACHE=1` on
   macOS/Linux)
2. run one simulation (`python -m ra_sim simulate ...` or `ra-sim simulate ...`)
3. watch console output for cache write/hit events from the Numba cache loader

Known limitations:

- first import/first run of a new kernel still compiles once before reuse
- cache hit behavior can vary across hosts, Python versions, and Numba releases
- if the default path is not writable, RA-SIM keeps startup best-effort and may not use on-disk caching

Manual verification recipe:

1. remove `~/.cache/ra_sim/numba` (or platform equivalent configured in `NUMBA_CACHE_DIR`)
2. run one simulation with `NUMBA_DEBUG_CACHE=1`
3. confirm new cache artifacts appear in `NUMBA_CACHE_DIR`
4. rerun same simulation with `NUMBA_DEBUG_CACHE=1` and confirm cache read hits in output

## Developer Tool Caches

When you run RA-SIM development commands from the repository, tool caches stay
under the user cache root instead of the worktree.

- Python bytecode: `~/.cache/ra_sim/dev/pycache`
- `mypy`: `~/.cache/ra_sim/dev/mypy`
- `pytest`: `~/.cache/ra_sim/dev/pytest`
- `ruff`: `~/.cache/ra_sim/dev/ruff`

The tool-specific `mypy`/`pytest`/`ruff` cache dirs apply to both
`python -m ra_sim.dev ...` and direct `pytest`/`mypy`/`ruff` runs from the
repository root. Python bytecode redirection applies when the repo
`sitecustomize.py` is importable, which is guaranteed for `ra_sim.dev` and
`python -m ...` launches from the repository root.

Existing repo-local cache folders are not migrated or removed automatically.
It is safe to delete stale `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, and
`__pycache__/` directories manually when they are no longer needed.

## Good Debug Hygiene

- Prefer config-based toggles over ad hoc path edits.
- Keep machine-local output paths in ignored local config, not versioned files.
- If you add a new debug artifact, document its toggle and output location.
- When sharing logs or screenshots, avoid leaking local absolute paths or private data.

For the exact key-by-key reference and current defaults, use the canonical doc:

- [Logging, debug, and cache controls](simulation_and_fitting.md#logging-debug-and-cache-controls)
