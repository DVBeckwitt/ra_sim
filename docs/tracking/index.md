# Live tracking

This subtree keeps long-form repo-local context for substantial efforts.
GitHub Issues are the unit of work. GitHub Projects hold live backlog, status,
and roadmap views. Do not use this page as a second backlog.

## Current fitter roadmap

The active sequence is:

1. get the geometric fitter working,
2. get the mosaic fitter working,
3. get the structure-factor fitter working,
4. get the stacking-fault fitter working.

Geometric fitter active validation is green as of 2026-05-05. Exact-caked
cold-start fitting, warm caked manual-pick cache stability, and Bi2Se3/Bi2Te3
headless saved-state residual validation now pass with strict fixed-pair
matching. A first mosaic-fitter scaffold is in progress, with a documented
selected-pair profile math contract. GUI/headless wiring must stay
geometry-locked and downstream of accepted geometry-cache provenance. The
structure-factor fitter now has a planned global multi-image detector-ROI
intensity contract, but remains downstream of green mosaic fitting.

## In progress

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Geometric fitter recovery | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-23 | [geometric-fitter-recovery.md](in-progress/geometric-fitter-recovery.md) |
| New4 geometric fitter recovery handoff | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-05-03 | [new4-geometric-fitter-recovery-handoff.md](in-progress/new4-geometric-fitter-recovery-handoff.md) |
| Deterministic geometry runtime fix pass | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-05-04 | [deterministic-geometry-runtime-fix-pass.md](in-progress/deterministic-geometry-runtime-fix-pass.md) |
| Qr/Qz shape sensitivity | feature | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [q-group-shape-sensitivity.md](in-progress/q-group-shape-sensitivity.md) |
| Q-space viewer fix | bug | - | none | p1 | 2026-04-30 | [q-space-viewer-fix.md](in-progress/q-space-viewer-fix.md) |
| Sim caked detector replay | bug | - | none | p1 | 2026-05-09 | [sim-caked-detector-replay.md](in-progress/sim-caked-detector-replay.md) |
| Background peak fit detector Qr rod panel | bug/feature | - | none | p1 | 2026-05-11 | [background-peak-fit-detector-qr-rod-panel.md](in-progress/background-peak-fit-detector-qr-rod-panel.md) |
| Beam center background pick | feature | - | none | p1 | 2026-05-01 | [beam-center-background-pick.md](in-progress/beam-center-background-pick.md) |
| Background Qr reference picks | feature | - | none | p2 | 2026-04-30 | [background-qr-reference-picks.md](in-progress/background-qr-reference-picks.md) |
| 6H Qr reference SF picking | feature/bug | - | none | p1 | 2026-04-30 | [6h-qr-reference-sf-picking.md](in-progress/6h-qr-reference-sf-picking.md) |
| Generated disordered Qr live path | bug/feature | - | none | p1 | 2026-05-03 | [generated-disordered-qr-live-path.md](in-progress/generated-disordered-qr-live-path.md) |
| Mosaic fitter recovery | feature | - | none | p1 | 2026-04-24 | [mosaic-fitter.md](in-progress/mosaic-fitter.md) |
| Weighted-event representative cache carry-through | bug | - | none | p1 | 2026-04-24 | [weighted-event-representative-cache-carry-through.md](in-progress/weighted-event-representative-cache-carry-through.md) |
| Fast-path cache audit and QR selector policy | bug | - | none | p2 | 2026-04-28 | [fast-path-cache-audit-phase1.md](in-progress/fast-path-cache-audit-phase1.md) |

Q-space viewer fix status note:
Implemented runtime geometry ownership for Q-space display without backend
algorithm changes. Detector distance now invalidates simulation caches, Q-space
conversion uses submitted simulation geometry, Q-space-only skips caking and can
stay current without `last_res2_sim`, and displayed Qr centers are finite and
positive. Focused runtime/import-safe/update tests pass. Full dev check remains
blocked by pre-existing formatting drift in dirty
`ra_sim/fitting/optimization.py`.

Replay status note: `Sim caked detector replay` has focused helper/runtime
validation green for detector-origin and caked-origin manual background replay.
Latest provenance patch makes `manual_background_input_origin` authoritative
over conflicting frame metadata, stops detector-view clicks from saving
`manual_background_input_frame="caked_2theta_phi"`, and keeps derived caked
fields as replay/cache data only. The follow-up runtime patch also prevents
stale refreshed `x/y` or detector aliases from winning cross-view redraw:
detector-origin rows return to detector anchors, caked-origin rows project from
the saved visual caked point, and provenance-known background references fail
closed when projection is unavailable. The final coordinate-authority patch
also makes bare `caked_x/y` opt-in for simulated fit/cache truth: only explicit
simulated caked projection rows can promote those aliases, while
background/replay-shaped rows stay unresolved for sim fit/cache fields.
Already-saved contradictory rows are compatible without state migration. No new
operator control, public API, cache redesign, or schema migration. Automated
local gates are green. The latest visual resolver patch keeps caked-view
display/ranking authority separate from fit/cache truth: explicit visual
aliases win first, safe current caked display aliases win second, refined-only
fit/cache fields remain fallback, and background-shaped rows do not become
simulated visual caked points. The active manual-pick session refresh now also
lets detector rows replace sticky caked projection rows on detector refresh while
preserving only visual caked aliases by identity. The 2026-05-09 detector-picker
closure hard-rejects caked projection rows before detector-looking coordinate
fields are considered, so `sim_refined_detector_display_px` on a caked
projection row no longer blocks picker-only detector recovery. No migration,
deprecation, CI workflow, feature flag, or release version change is required.
Focused detector/runtime tests, compile, diff check, and
`python -m ra_sim.dev check` pass. The 2026-05-10 repo-clean follow-up also
guards import-safe overlay invalidation when runtime history state is absent,
labels refined-only detector fallback rows with the real fallback source, and
adds a deterministic caked-select -> detector -> clear/rearm -> detector-click
identity proof. Full runtime import-safe and geometry-fit workflow suites pass
locally; manual detector/caked GUI smoke remains pending before closing the
tracking item.

Background peak fit detector Qr rod panel status note:
The ignored parallel diagnostics notebook now treats the detector Qr-rod panel
as a source-consistent geometry/overlay diagnostic. Cell 14 uses Qr-driven rod
rotation fitting (`FIT_QZ_WEIGHT = 0.0`), skips specular anchors, balances
anchors by rod identity, rejects mixed target-Qr identities by source/HK (or
branch if needed), applies the same acceptance predicate to fit anchors,
markers, profile samples, and branch items, and reports detector-space
`curve_distance_px` from point-to-polyline segment distance. The detector panel
keeps accepted placed-star diagnostics, low-L `HK=<m> +/-` labels, projected
centerlines, and transparent Delta-Qr bands including `HK=0`. The integrated
Qr figure centers `HK=0`, labels the HK=0 row and left nonzero subplot axes
with `Intensity (a.u.)`, aligns non-specular L axes from `L=2`, and places the
Data/Simulation legend in the top-right panel. JSON, nbformat, compile, static checks, and the two parallel
notebook pytest checks pass. Full notebook-section rerun and visual acceptance
remain pending; full `tests/test_background_peak_fits_notebook.py` is still red
in unrelated non-parallel notebook expectations. The 2026-05-10 Bi2Se3 update
sets the parallel diagnostic default state to Bi2Se3, bumps the final Qz fit
cache signature to v8, keeps supported weak low-L specular markers through
nonlinear refinement even when the preliminary shared baseline runs above the
shoulder, adds a log-scale residual term so the Bi2Se3 m=0 full profile fits
the log-scaled Qr integration plot more closely, preserves the matching Bi2Te3
weak marker, rejects unsupported nearby markers, and makes tail component
aggregation fail closed on shape mismatch. Focused Bi2Se3/Bi2Te3 marker,
shape-mismatch, cache-signature, compile, and format checks pass locally. Full
diagnostics test-file status is still red in unrelated notebook/script
source-token assertions; `python -m ra_sim.dev check` is blocked by pre-existing
formatting drift in `ra_sim/fitting/optimization.py`. No CI workflow, public
API, saved-state schema, or deprecation/migration path changed. The
2026-05-11 PbI2 update keeps PbI2-specific lattice/rod state dynamic, applies
same-Qz transverse Qr sideband subtraction to nonzero PbI2 profiles, hides
unsupported `m=7`, and gates misleading nonzero model overlays when marker/L
mapping or Qz-baseline cancellation diagnostics fail. Focused PbI2 acceptance,
compile, and headless script execution pass; the regenerated PbI2 figure keeps
`m=1` `Fit` overlays and omits misleading `m=3`/`m=4` overlays. Full
diagnostics test-file status is `116 passed`, `2 skipped`, `6 failed` in the
same unrelated source-token checks. No CI workflow, deployment, package API,
saved-state schema, or deprecation/migration path changed; rollback is to
revert the diagnostic script/test/doc commit and regenerate affected local
caches/artifacts.

Beam center background pick status note:
`Pick Beam Center` is implemented in Setup > Beam Controls. The mode uses the
loaded detector/background image, switches out of caked/q-space views, shows the
background if hidden, and commits the clicked detector-display point as GUI
Beam Center Row/Col on release. The current GUI contract is
`row = display_row`, `col = detector_width - display_col`; for the default
3000 px clockwise view, `display_col=1456`, `display_row=1607` maps to
`row=1607`, `col=1544`. `RA_SIM_TRACE_BEAM_CENTER=1` writes
`debug/beam_center_trace.jsonl` with widget-chain, scheduled-update, marker,
remap, and overwrite-guard diagnostics. Targeted beam-center, remap-cache,
canvas-route, smoke, and compile checks pass.

Background Qr reference pick status note:
`Place Background Qr Set` is implemented in the Match tab. The mode places a
background-only manual reference point, locally refines to the measured peak
top, saves the refined `(2theta, phi)` as the row label, omits HKL/Qr group
identity, preserves the row in manual-pair export/import for diagnostic
notebooks, and marks it disabled for geometry solving. Targeted GUI wiring,
manual placement serialization, geometry-fit filtering, runtime import, and
targeted compile checks pass. Full `ra_sim.dev check` remains blocked by
pre-existing formatting drift in `ra_sim/fitting/optimization.py`.

6H Qr reference SF picking status note:
`Include 6H Qr refs` is implemented as an opt-in stacking control. When enabled
and `w1 > 0`, the GUI loads the packaged PbI2 6H CIF, adds `pbii_6h_ref`
source rows to Qr/Qz selector inventory, merges duplicate numeric Qr/Qz groups
before listing/picking, and saves the checkbox in GUI state with legacy states
defaulting off. Focused 6H compile, duplicate-merge, detector-fallback,
runtime-gate, state-IO, and ruff checks pass. Wider manual-geometry validation
is still red in existing caked-view candidate/reverse-LUT expectations, so the
feature remains in progress until those broader failures are triaged.

Generated disordered Qr live path status note:
The user-reported primary-only live picker reuse is fixed. With nonzero
generated-disordered stacking weight, the live runtime evaluates the active-CIF
generated inventory, schedules hit-table-only disordered collection when rows
are missing or stale, publishes stored `disordered_phase` rows into the active
Qr/Qz picker cache, and exposes `Include generated disordered-phase Qr refs`
as a saved checkbox defaulting on. The path logs enable/skip decisions,
inventory paths, collection counts, published group/peak counts, and final
source counts. Manual selection, placement, saved pairs, and geometry-fit
preflight now preserve `source_label="disordered_phase"` through the active
handoff. Geometry-fit preflight also logs
`geometry_fit_live_handoff_patch_marker=phase4d1`, job-build live-row counts,
and `fresh_rebuild_consumer_wrapper=deduped`; if live preview rows are empty,
the job can build source rows from the active picker/Q-group cache without
falling back to primary-only rows. The Phase 4E/4F live-cache validator now
accepts unambiguous source-matched q-group rows with finite detector
coordinates, cannot emit `status=invalid reason=ready`, keeps detector-origin
manual pairs out of exact-caked projector fallback despite caked diagnostic
backfill, and fails source-safely when disordered rows are missing. The
follow-up locked/stale QR correspondence regression is fixed too: the resolver
runs existing stale-row proof and saved-detector/source-switch fallback before
returning `locked_qr_row_unavailable`, restoring
`prediction_branch_source_switched` for deterministic alternate evidence while
ambiguous evidence remains rejected. Focused user-report, live-refresh,
inventory, scheduling, current-refresh, UI enable, q-group cache, hit-table,
logging, source-aware picker/fitter, source-cache rung, fit-space
classification, disordered preflight, branch-switch, compile, and diff-check
gates pass. Manual GUI validation should confirm the committed build emits
`source_cache_live_runtime_cache_accepted` with nonzero `live_rows_raw_count`.
Full `ra_sim.dev check` is still blocked by existing
`ra_sim/fitting/optimization.py` formatting drift; broad geometry/manual
selector timed out locally after five minutes without failure output.

Weighted-event representative status note:
`Weighted-event representative cache carry-through` is implemented on fast
weighted-event path and targeted weighted-event/cache tests are green. Manual
geometry replay/workflow suites and full-suite status remain red in this
worktree for adjacent replay/finalizer and unrelated fixture/doc/env failures,
so the follow-up stays tracked as in-progress until broader tree health is
clean.

Fast-path cache audit and QR selector policy status note:
Phases 1-9 are implemented and locally validated. QR/Qz masks remain explicit
user/state data and are not cleared by runtime cache invalidation. Selective
invalidation is policy-gated, optional New4 tests skip cleanly when artifacts
are absent, prune reuse/fill report QR selector retention/deferred refresh and
handoff validity, and detector-center remap now reports projection invalidation,
branch/source identity retention, and full-simulation fallback reasons in the
runtime trace. Phase 6 adds objective-cache signature gating so center-only
reuse fails closed across QR branch/source-row/manual/refined peak,
point-provider, objective-mode, active-fit-parameter, dataset, and physics
changes. Phase 7 adds a synthetic end-to-end QR selector to geometry fitter
handoff scenario covering fast-path sequencing, point-provider parity,
projection invalidation, and objective-cache reuse/reject behavior. Phase 8
adds `scripts/debug/run_geometry_fitter_cache_regression_gate.py` for repeatable
local/strict cache regression validation. Phase 9 adds the final mixed-update
and stale-worker regression suite. Bug/error status: no known failing local
Phase 9 gate tests after focused mixed gate `67 passed`, local script fast gate
`497 passed`, manual identity `5 passed, 423 deselected`, and workflow slice
`26 passed, 2 skipped`.

Current emphasis for [#249](https://github.com/DVBeckwitt/ra_sim/issues/249):
New4 provider handoff, fixed-source request handoff, sensitivity scan,
one-param solves, `a` diagnosis, caked point reprojection, initial Rung 4 paired
solves, fresh same-run Rung 5 blocks, Rung 6 C2, and the bounded Rung 7 feature
chain are now validated for ladder work. Final frozen block/combined runs
`codex_final_blocks_20260423` and `codex_final_combined_20260423` stayed green,
and the passing on-disk Rung 7 feature comparator is
`codex_restore_rung7_features_fix_20260423`
(`codex_final_features_fullseq_20260423`). The older
`codex_final_features_20260423` artifact is stale and still shows the pre-fix
`full_beam_polish` failure. Current New4 Rung 2
expected baseline is `active_param_count=11`, `near_zero_param_count=2`;
`center_x` and `center_y` are active under the unchanged threshold rule because
`residual_norm_base` dropped `17.32x`, shrinking the classifier threshold
faster than their delta norms fell. This is expected, not a fitting regression;
do not reopen the exact-caked path for it.
`new4.json` stayed unchanged
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`).
Timing observability is now available for current-run Rungs 0-5:
`rung_timing_summary.json` is written in the run directory, optional
`--timing-report` writes the same machine-readable summary elsewhere, stdout
prints `Rung | Status | elapsed_s | report_path`, and
`RA_SIM_NEW4_LADDER_TIMING_MAX_S` is diagnostic-only. Real opt-in timing run
`20260422_123330` completed the approved `--max-rung blocks` path with
`status == "ok"`, total `26.612s`, slowest rung `caked_point_reprojection` at
`9.572s`, no missing expected rungs, no Rung 6/7 timing records, and zero
non-finite elapsed values. Fast manual selected-point fit defaults are now
implemented: GUI manual fits cap `cfg["solver"]["max_nfev"]` at 30, run serial
by default, and keep identifiability diagnostics off unless an explicit
diagnostic path is requested. Lean ladder rungs also keep identifiability off by
default, while `feature="identifiability_features"` remains the diagnostic
feature path. Running ladder heartbeat writes are sparse and omit the growing
full residual trace; final reports still include the full trace. Bug/error
status: manual caked geometry fits now require exact hydrated per-background
caked projectors and fail closed instead of using detector/current-view raw-row
fallbacks in the solver/ladder path. Fresh validation under
`temp/codex_caked_manual_blast` passed explicit Rungs 0-5 and Rung 6 combined;
Rung 6 C1 improved caked metrics, while C2 accepted the seeded caked initial
state because its optimizer output regressed caked RMS/max. Remaining blocker:
four `tests/test_gui_runtime_import_safe.py` fail-closed boundary tests still
show caked prepare/worker paths can reach dataset build before exact-payload
ensure/failure handling, so full GUI/preflight closure is not claimed. Rung 6/7
path mappings and expected timing IDs are excluded from timing
collection, Rung 5 skipped reports get timing metadata, fatal evidence still
aborts, local `a` usability failures stay local, missing dependencies skip only
affected blocks, stale external evidence remains rejected, manual selected-point
fits no longer inherit heavy solver defaults, and stale heartbeat traces are
reset before solve-rung writes. No full fitter, baseline, GUI fit, unrestricted
feature combination, feature-combo solve, identifiability feature run, or full
fitter validation was run; GUI manual selected-point runtime behavior is
updated, but it does not validate full GUI fitter convergence. Real headless
`python -m ra_sim fit-geometry artifacts/geometry_fit_gui_states/new4.json`
smoke is now run and failing. Exact-caked request invariants stayed green, but
the run rejected with `accepted == false`, `detector_rms_px ==
914.4948551954421`, and `unweighted_peak_max_px == 1698.2499036720524`. First
divergence versus the passing ladder comparator is a seed/start-state split:
real headless fit uses the 9-variable GUI/runtime contract and selected
`axis:zb-1`, while the passing ladder comparator uses the 6-variable New4
candidate bundle and a different seed family. `full_beam_polish` is disabled in
the real headless path, so candidate-selection is not the first divergence. No
full-headless baseline convergence patch is claimed yet. 2026-04-27 update:
New4 Mode A dynamic/refined Qr prediction now resolves 14/14 saved first-image
branches and 28/28 caked residual components using a durable locked identity
key. Partial Qr objective coverage fails closed. Latest refined-center
diagnostics prove observed caked centers and simulated refined caked centers
are recomputed under trial geometry, but simulated refinement is integer
caked-bin argmax only (`2theta` bin `0.071355959 deg`, `phi` bin `0.5 deg`).
Qr-only and full dynamic/refined fits accepted no parameter step over `nfev=7`;
theta, phi, Qr, and full objective norms stayed unchanged. Current reason:
`refinement_bin_limited`, not branch identity, stale observed caked values, or
stale simulated refined caked reuse. 2026-04-29 strict validation update:
historical 7-pair New4 fixture restoration and provider-only/Rung 0 parity are
green; zero-Qr `00l` branch rebinding is now collapsed only for 00l targets;
headless targeted preflight reaches the performance gate without full-source
fallback. Follow-up fixes now collapse caked signed/unknown provenance rows by
physical branch, harden Qr/Qz group cache signatures against recursive or
failing mapping/sequence payloads, and keep caked manual preflight probes on
caked projection candidates. Full-preflight live source coverage now reaches
7/7 for the restored fixture, but full validation remains red downstream as
`classification == "seam_failure"` with background/candidate distance gates
still failing. Full fitter validation is not claimed.

## Known bugs

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | - |

## Planned features

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Structure-factor fitter recovery | feature | - | none | p2 | 2026-04-24 | [structure-factor-fitter.md](planned-features/structure-factor-fitter.md) |
| Stacking-fault fitter recovery | feature | - | none | p2 | 2026-04-24 | [stacking-fault-fitter.md](planned-features/stacking-fault-fitter.md) |

## Archive

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Analyze dual-source box peak fit | feature | - | none | p1 | 2026-04-20 | [analyze-dual-source-box-peak-fit.md](archive/analyze-dual-source-box-peak-fit.md) |
| Analyze integration crop and log view | bug/feature | Codex | none | p2 | 2026-05-05 | [analyze-integration-crop-log.md](archive/analyze-integration-crop-log.md) |
| Analyze overlay hardening | bug | - | none | p2 | 2026-04-20 | [analyze-overlay-hardening.md](archive/analyze-overlay-hardening.md) |
| Analyze peak-fit table and ROI scale | bug/feature | - | none | p2 | 2026-04-30 | [analyze-peak-fit-table-roi-scale.md](archive/analyze-peak-fit-table-roi-scale.md) |
| Background peak-fit diagnostics | bug/feature | - | none | p2 | 2026-04-30 | [background-peak-fit-diagnostics.md](archive/background-peak-fit-diagnostics.md) |
| Cold-start caked manual fit projector | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-05-05 | [cold-start-caked-manual-fit-projector.md](archive/cold-start-caked-manual-fit-projector.md) |
| Detector Selected-Qr Rod ROI | bug/feature | - | none | p1 | 2026-05-01 | [detector-selected-qr-rod-roi.md](archive/detector-selected-qr-rod-roi.md) |
| Detector-oracle caked Qr/background picks | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [detector-oracle-caked-background-picks.md](archive/detector-oracle-caked-background-picks.md) |
| Fast primary rasterizer | refactor | - | none | p2 | 2026-04-22 | [fast-primary-rasterizer.md](archive/fast-primary-rasterizer.md) |
| Full geometry fitter repair and Bi validation | bug/feature | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-05-05 | [full-geometry-fitter-repair-bi-validation.md](archive/full-geometry-fitter-repair-bi-validation.md) |
| GUI runtime selective update fast paths | optimization | - | none | p2 | 2026-04-28 | [gui-runtime-selective-update-fast-paths.md](archive/gui-runtime-selective-update-fast-paths.md) |
| GUI sampling events link | feature | - | none | p2 | 2026-04-28 | [gui-sampling-events-link.md](archive/gui-sampling-events-link.md) |
| hBN fitter documentation | feature | Codex | none | p2 | 2026-04-30 | [hbn-fitter-documentation.md](archive/hbn-fitter-documentation.md) |
| Hit-table-only fit simulation | optimization | - | none | p2 | 2026-05-04 | [hit-table-only-fit-simulation.md](archive/hit-table-only-fit-simulation.md) |
| Lazy best-sample and Qr selection hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [lazy-best-sample-and-qr-selection-hardening.md](archive/lazy-best-sample-and-qr-selection-hardening.md) |
| Main figure right-drag pan regression | bug | - | none | p2 | 2026-04-29 | [main-figure-right-drag-pan.md](archive/main-figure-right-drag-pan.md) |
| Manual Qr/Qz caked picker cache projection | bug | Codex | none | p1 | 2026-05-05 | [manual-qr-caked-picker-cache-projection.md](archive/manual-qr-caked-picker-cache-projection.md) |
| Match peak tools layout | bug | - | none | p2 | 2026-04-30 | [match-peak-tools-layout.md](archive/match-peak-tools-layout.md) |
| Mosaic mass conservation validation | feature | - | none | p1 | 2026-04-22 | [mosaic-mass-conservation-validation.md](archive/mosaic-mass-conservation-validation.md) |
| Numba import fallback | bug | - | none | p2 | 2026-05-04 | [numba-import-fallback.md](archive/numba-import-fallback.md) |
| Qr integration region mask | bug | - | none | p2 | 2026-04-30 | [qr-integration-region-mask.md](archive/qr-integration-region-mask.md) |
| Remove global background subtraction | bug/feature | Codex | none | p1 | 2026-05-05 | [remove-global-background-subtraction.md](archive/remove-global-background-subtraction.md) |
| Rod profile intensity density | bug/feature | - | none | p1 | 2026-04-30 | [rod-profile-intensity-density.md](archive/rod-profile-intensity-density.md) |
| Skip discarded fit hit tables | refactor | - | none | p2 | 2026-04-22 | [skip-discarded-fit-hit-tables.md](archive/skip-discarded-fit-hit-tables.md) |
| Startup default detector visibility regression | bug | - | none | p1 | 2026-04-18 | [startup-default-detector-visibility.md](archive/startup-default-detector-visibility.md) |
| Manual Qr/Qz and HKL picker alignment | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [sim-peak-overlay-recovery.md](archive/sim-peak-overlay-recovery.md) |
| Runtime cache diagnostic hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [runtime-cache-diagnostic-hardening.md](archive/runtime-cache-diagnostic-hardening.md) |
| Simulated peak overlay recovery history | investigation | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [sim-peak-overlay-recovery-history.md](archive/sim-peak-overlay-recovery-history.md) |
| Weighted-event projected-candidate reuse | optimization | - | none | p2 | 2026-04-28 | [weighted-event-candidate-reuse.md](archive/weighted-event-candidate-reuse.md) |
| Weighted-event Q-set precompute | optimization | - | none | p2 | 2026-04-28 | [weighted-event-qset-precompute.md](archive/weighted-event-qset-precompute.md) |
