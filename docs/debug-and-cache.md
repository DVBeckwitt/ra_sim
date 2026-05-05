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

- final pre-merge requested suite
  `python -m pytest -ra tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_import_safe.py`
  passed, `408 passed`
- manual GUI-runtime trace smoke passed with these action labels:
  `full_simulation` for initial update, detector-center without exact cache,
  detector distance change, and detector rotation/tilt change;
  `primary_prune_reuse` for prune with cached keys; `display_only` for a
  display-only change; `combine_only` for the combine/visibility change; and
  `detector_center_remap` for center shift with exact cache
- `python -m pytest tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py -ra`
  passed, `92 passed`
- broad GUI runtime slice
  `python -m pytest tests/test_gui_runtime_*.py tests/test_gui_sim_signature.py -ra`
  passed via explicit PowerShell file expansion, `448 passed`
- `python -m compileall ra_sim/gui tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_update_dependencies.py -q`
  passed

Repository hygiene status:

- fast-path implementation lives in `e756ba0`
  (`perf(gui): add update fast paths`)
- validation status lives in `32c6ac9`
  (`docs(gui): record fast-path validation`)
- final audit confirmed the fast-path commit contains only fast-path
  docs/code/tests; unrelated dirty files were left unstaged
- commit-time Git worktree cleanup warnings were permission warnings only and
  did not block commit creation
- `git diff --check` found no whitespace errors, only CRLF normalization
  warnings in the dirty worktree

## Geometry Qr/Qz and HKL Picker State

Manual Qr/Qz picking uses a structural group cache derived from the active
simulation hit tables and CIF/lattice state. It is not a detector-view or
caked-view cache. Detector/caked view switches must not invalidate or filter
the Qr/Qz group universe. CIF, unit-cell, or simulation-hit-table changes do
invalidate it.

Generated disordered-phase Qr/Qz groups are structural rows, not live display
artifacts. When `Include generated disordered-phase Qr refs` is enabled and a
nonzero disordered stacking component is active, the runtime generates the
HT-shifted disordered CIF from the active PbI2 CIF, builds hit-table rows with
`accumulate_image=False`, tags them as `disordered_phase`, includes that source
signature in picker-cache validity checks, and publishes them into the active
Qr/Qz picker cache during current-simulation refreshes.

Optional PbI2 6H reference Qr/Qz groups are opt-in legacy structural rows, not
live display artifacts. When `Include packaged 6H Qr refs` is enabled and `w1` is nonzero,
the runtime loads the packaged `ra_sim.config/materials/PbI2_6H.cif`, builds
6H hit-table rows with the current wavelength/window/HKL limits, and tags them
as `pbii_6h_ref`. The q-group and manual-pick signatures include the toggle,
`w1`, HKL limit, wavelength, 2theta window, and intensity threshold so stale
picker rows are not reused across 6H-reference changes.

Duplicate Qr/Qz identities from primary, secondary, stacking-fault, or 6H
reference rows are merged numerically before listing and picking. The merge
tolerance is `atol=1e-6`, `rtol=1e-8`; one selector key remains, intensity and
candidate points are combined, and aux source aliases are kept for diagnostics.

Resolved picker behavior depends on keeping structural simulation truth
separate from current-view projection. Group membership comes from structural
state; detector or caked coordinates are derived later for the active view.

Detector-view manual Qr/Qz picking must not reuse a manual-pick cache that
matches the current signature but contains no detector source rows or detector
picker candidates. A matching empty detector cache is stale and must rebuild
from the current source snapshot, live peak records, or fresh simulation rows.
Only caked mode may accept `caked_qr_projection_grouped_candidates` as the
cache-reuse gate.

`manual_pick_cache` source-row lookup may rebuild a missing, stale, or empty
source snapshot only for the current background and only when stored simulation
artifacts exist (`stored_max_positions_local`, `stored_intersection_cache`, or
`peak_records`). Detector manual-pick rebuilds use detector projection mode
unless the manual picker is explicitly in caked space. This keeps detector
picking independent of caked integration while preserving the existing
geometry-fit dataset rebuild path for non-current backgrounds and targeted
preflight.

Detector picker diagnostics should distinguish these cases:

- matching empty manual-pick cache rebuilt instead of reused
- source snapshot missing, stale, empty, or rebuilt
- source rows present but missing `q_group_key`
- source rows present but missing detector display pixels
- detector candidates present by source row family

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

Live caked visual-source ledger rows are disabled by default. Set
`RA_SIM_LIVE_CAKED_TRACE=1` to print `[ra-sim] live_caked_visual_source` rows,
and add `RA_SIM_LIVE_CAKED_TRACE_ALL=1` to include unchanged duplicate rows.
`RA_SIM_SUPPRESS_LIVE_CAKED_TRACE=1` suppresses the ledger regardless of the
enable flags.

Warm-cache simulated-candidate refinement relies on cache, simulation, and
exact caked-projection signatures plus stable full-content value tokens. Projection-only
payloads are valid geometry inputs even when the display/background caked image
is absent. Caked pick-cache signatures intentionally ignore image-facing
background payload identity, so display density sanitization, including
zero-support `NaN -> 0.0` storage cleanup, must not churn the geometric
projection cache. Stable projection signatures use the verified
`projection_content_token_v3` recomputed during projection-payload
storage/hydration, plus full-content value tokens for axes/permutations
(`axis_content_v3` / `perm_content_v3`), not array object IDs, sampled axis
probes, legacy `signature` fields, full image hashes, or click-path LUT hashes.
DetectorCakeLUT-style projection tokens include `image_shape`, `n_rad`, `n_az`,
the detector-to-cake matrix content, and `count_flat` content. Projection
payload storage copies axes, permutations, and `CakeTransformBundle` LUT content
into private read-only objects before attaching a trusted
`projection_content_token_source`. Shared incoming bundles are not frozen.
Source-less or legacy projection tokens are not correctness keys for warm-cache
reuse. Do not mutate simulation or caked image arrays in place without also
bumping the corresponding simulation or projection signature.

Status as of 2026-05-05: live caked trace output is opt-in, unchanged trace rows
are suppressed unless explicitly requested, warm caked pick-cache calls skip
row-level refinement when simulation/projection signatures are unchanged,
zero-support display sanitization no longer invalidates caked pick caches,
equivalent copied axes and projection payloads keep the same signature/digest,
token-only projection payloads are absent and can still generate fallback
payloads, explicit trusted projection signatures survive
normalize/hydrate/digest handoff, failed
lookup rebuilds retry, and no-signature direct refine/rebuild calls clear stale
skip metadata. The exact-caked cold-start path accepts projection-only payloads
without requiring a display image. The New4 ladder finalizer now repairs stale
exact-caked report/polish fields only when the selected exact-caked summary is
clean; real missing/lost manual-pair cases still fail the fixed-source gate.
Active saved-state validation now runs Bi2Se3 and Bi2Te3 from the RA-SIM user
data root with direct fixed-pair solves; both pass all fixed-pair matching and
residual-reduction gates.

Live GUI source-row fallback status as of 2026-05-05: after
`projection_payload_ready`, caked geometry-fit preflight now logs both
`background_index_internal` and a 1-based background label, records manual-pair
backgrounds and required/matched pair counts, and reports reject reasons for
live, targeted, memory, logged, and current-hit-table cache paths. Live rows and
current hit/intersection tables are accepted only when the requested background,
fit space, exact caked projection payload, simulation signature, source IDs,
q-group keys, branch IDs, and fixed-pair count all validate without ambiguous
candidate expansion. If current live rows or signature-matched current hit
tables are usable, fresh simulation must not start. If the slow fresh-simulation
fallback does run, timeout now emits a visible late/still-running status before
the eventual ready or failed terminal event.

Validation status: runtime-level regressions cover the observed miss sequence
(`projection_payload_ready`, targeted projected miss, memory miss, logged miss)
and prove current hit tables are consumed before `simulate_hit_tables`. A
background-index regression covers manual pairs on UI background 2 building
internal index 1 and not background 1. The current Bi quality baseline ran the
same direct fixed-manual path as the smokes: Bi2Se3 accepted with 82/82 fixed
pairs matched, 0 missing, 0 branch mismatches, and direct RMS 34.5307 -> 31.078112;
Bi2Te3 accepted with 84/84 fixed pairs matched, 0 missing, 0 branch mismatches,
and direct RMS 36.8629 -> 34.394414. The live GUI manual acceptance smoke still
requires operator verification on a fresh session: import a Bi state, generate
once, run Fit Geometry without changing parameters, and confirm no first-fit
timeout, `invalid_exact_caked_payload`, or `exact caked projector unavailable`.

The HKL picker intentionally shares the corrected Qr/manual picker candidate
payload for hit testing and selected-marker placement. If either picker
regresses, first check whether the failing path bypassed that shared candidate
payload or treated detector/display aliases as caked coordinates.

## QR Selector Fast-Path Cache Policy

QR selector cache retention is centralized in
`ra_sim/gui/runtime_qr_selector_cache_policy.py`. Runtime invalidation calls the
policy before clearing selector entries, source-row snapshots, intersection
caches, or manual-pick projection payloads.

Status as of 2026-04-28:

- feature status: implemented for display-only, combine-only, analysis-only,
  primary-prune reuse, primary-prune fill, detector-center remap, and full
  simulation update actions; local Phase 3.5 validation also adds fast
  geometry-fitter handoff tests and optional New4 fixture skips; Phase 8 adds
  `scripts/debug/run_geometry_fitter_cache_regression_gate.py` as the repeatable
  local/strict cache regression gate; Phase 9 adds mixed-update and stale-worker
  sequence coverage to that gate
- bug status: fixed for overbroad fast-path invalidation that could clear QR
  selector entries or fitter handoff data before replacement rows were ready;
  fixed local New4 validation failures caused by absent optional artifacts
- error status: targeted cache-policy, runtime-invalidation, and fast handoff
  tests pass; slow/manual caked-refined geometry diagnostics are excluded from
  local mode and included in strict mode
- compatibility status: `disabled_qr_sets`, `disabled_qz_sections`, and
  `pending_legacy_disabled_qz_sections` remain explicit user/state selections
  and are not cleared by cache invalidation

Retention rules:

- display-only, combine-only, and analysis-only actions retain selector and
  fitter handoff caches
- primary-prune reuse keeps q-group entries when content signatures are
  unchanged and requests refresh when q-group content changes
- primary-prune fill keeps old q-group entries until replacement rows apply
- detector-center remap retains branch/source identity for exact remaps while
  refreshing projection geometry when detector geometry changes
- full simulation clears stale source rows and cache payloads when physics or
  hit-table signatures change, but retains QR masks

Validation:

- `python -m pytest tests/test_gui_runtime_geometry_fitter_handoff_fast.py -q`
  passed, `5 passed`
- local Phase 3.5 gate passed, `438 passed`
- New4 workflow slice passed locally with `26 passed, 2 skipped` when the
  optional `artifacts/geometry_fit_gui_states/new4.json` fixture was absent
- `python -m pytest tests/test_runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py -q`
  passed, `23 passed`
- `python -m pytest tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_optimization_scenarios.py -q`
  passed, `22 passed`
- `python -m py_compile ra_sim/gui/runtime_invalidation.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py`
  passed
- `python scripts/debug/run_geometry_fitter_cache_regression_gate.py --mode local`
  passed; untracked local New4 artifacts are skipped by default unless
  `RA_SIM_ALLOW_UNTRACKED_NEW4=1` is set
- Phase 9 local gate passed with the mixed-update suite included: fast gate
  `497 passed`, manual identity `5 passed, 423 deselected`, workflow slice
  `26 passed, 2 skipped`

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
