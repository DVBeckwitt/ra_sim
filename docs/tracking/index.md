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

The geometric fitter remains the primary recovery project. A first mosaic-fitter
scaffold is now in progress, with a documented selected-pair profile math
contract. GUI/headless wiring must stay geometry-locked and downstream of
accepted geometry-cache provenance. The structure-factor fitter now has a
planned global multi-image detector-ROI intensity contract, but remains
downstream of green mosaic fitting.

## In progress

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Geometric fitter recovery | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-23 | [geometric-fitter-recovery.md](in-progress/geometric-fitter-recovery.md) |
| New4 geometric fitter recovery handoff | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-27 | [new4-geometric-fitter-recovery-handoff.md](in-progress/new4-geometric-fitter-recovery-handoff.md) |
| Qr/Qz shape sensitivity | feature | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [q-group-shape-sensitivity.md](in-progress/q-group-shape-sensitivity.md) |
| Sim caked detector replay | bug | - | none | p1 | 2026-04-23 | [sim-caked-detector-replay.md](in-progress/sim-caked-detector-replay.md) |
| Mosaic fitter recovery | feature | - | none | p1 | 2026-04-24 | [mosaic-fitter.md](in-progress/mosaic-fitter.md) |
| Weighted-event representative cache carry-through | bug | - | none | p1 | 2026-04-24 | [weighted-event-representative-cache-carry-through.md](in-progress/weighted-event-representative-cache-carry-through.md) |
| Diffuse background subtraction | feature | - | none | p1 | 2026-04-28 | [diffuse-background-subtraction.md](in-progress/diffuse-background-subtraction.md) |
| Fast-path cache audit and QR selector policy | bug | - | none | p2 | 2026-04-28 | [fast-path-cache-audit-phase1.md](in-progress/fast-path-cache-audit-phase1.md) |

Replay status note: `Sim caked detector replay` remains in progress. Latest
replay-only patch removed saved-background gating, tightened replay eligibility
to current caked-projection evidence, and leaves replay-eligible sim rows
unresolved on reverse-LUT failure. Validation is still pending.

Weighted-event representative status note:
`Weighted-event representative cache carry-through` is implemented on fast
weighted-event path and targeted weighted-event/cache tests are green. Manual
geometry replay/workflow suites and full-suite status remain red in this
worktree for adjacent replay/finalizer and unrelated fixture/doc/env failures,
so the follow-up stays tracked as in-progress until broader tree health is
clean.

Diffuse background subtraction status note:
The shared radial/caked subtraction model, GUI Background tab, headless/CLI
overrides, diagnostics, cache invalidation, and off-mode safeguards are
implemented. The Background tab UX is now workflow-oriented with presets,
explained sliders, collapsible advanced sections, dirty status feedback,
debounced auto-preview, diagnostics summaries, and compact copy for labels,
buttons, hints, statuses, presets, tooltips, and diagnostics. The current
feature extension adds phi-block residual subtraction after the radial model,
with `radial_plus_phi_blocks` and
`radial_plus_phi_blocks_plus_caked_2d` modes, GUI controls, CLI overrides,
component diagnostics, and phi-block artifact exports. Targeted
GUI/state/import-safe/headless/numerical tests pass, including the phi-block
combined relevant suite with 429 tests. Full-suite runs timed out locally after
the UX/copy passes and again after the phi-block extension. Real detector smoke
tests for radial/phi-block/slow-caked mode comparisons, manual GUI
preview/orientation checks, tooltip/preset interaction checks, saved-state
headless override checks, and real diagnostic artifact inspection still need
project input data.

Fast-path cache audit and QR selector policy status note:
Phases 1-7 are implemented and locally validated. QR/Qz masks remain explicit
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
projection invalidation, and objective-cache reuse/reject behavior. Bug/error
status: no known failing local Phase 7 gate tests after `47 passed`, local
Phase 3.5 gate `471 passed`, and workflow slice `26 passed, 2 skipped`.

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
stale simulated refined caked reuse.

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
| Analyze overlay hardening | bug | - | none | p2 | 2026-04-20 | [analyze-overlay-hardening.md](archive/analyze-overlay-hardening.md) |
| Detector-oracle caked Qr/background picks | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [detector-oracle-caked-background-picks.md](archive/detector-oracle-caked-background-picks.md) |
| Fast primary rasterizer | refactor | - | none | p2 | 2026-04-22 | [fast-primary-rasterizer.md](archive/fast-primary-rasterizer.md) |
| GUI runtime selective update fast paths | optimization | - | none | p2 | 2026-04-28 | [gui-runtime-selective-update-fast-paths.md](archive/gui-runtime-selective-update-fast-paths.md) |
| GUI sampling events link | feature | - | none | p2 | 2026-04-28 | [gui-sampling-events-link.md](archive/gui-sampling-events-link.md) |
| Lazy best-sample and Qr selection hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [lazy-best-sample-and-qr-selection-hardening.md](archive/lazy-best-sample-and-qr-selection-hardening.md) |
| Mosaic mass conservation validation | feature | - | none | p1 | 2026-04-22 | [mosaic-mass-conservation-validation.md](archive/mosaic-mass-conservation-validation.md) |
| Skip discarded fit hit tables | refactor | - | none | p2 | 2026-04-22 | [skip-discarded-fit-hit-tables.md](archive/skip-discarded-fit-hit-tables.md) |
| Startup default detector visibility regression | bug | - | none | p1 | 2026-04-18 | [startup-default-detector-visibility.md](archive/startup-default-detector-visibility.md) |
| Manual Qr/Qz and HKL picker alignment | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [sim-peak-overlay-recovery.md](archive/sim-peak-overlay-recovery.md) |
| Runtime cache diagnostic hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [runtime-cache-diagnostic-hardening.md](archive/runtime-cache-diagnostic-hardening.md) |
| Simulated peak overlay recovery history | investigation | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-23 | [sim-peak-overlay-recovery-history.md](archive/sim-peak-overlay-recovery-history.md) |
| Weighted-event projected-candidate reuse | optimization | - | none | p2 | 2026-04-28 | [weighted-event-candidate-reuse.md](archive/weighted-event-candidate-reuse.md) |
| Weighted-event Q-set precompute | optimization | - | none | p2 | 2026-04-28 | [weighted-event-qset-precompute.md](archive/weighted-event-qset-precompute.md) |
