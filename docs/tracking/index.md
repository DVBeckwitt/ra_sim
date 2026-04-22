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

The geometric fitter is the only active implementation project. The other
fitters are planned work and must stay blocked until their upstream acceptance
gates are green.

## In progress

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Geometric fitter recovery | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [geometric-fitter-recovery.md](in-progress/geometric-fitter-recovery.md) |
| New4 geometric fitter recovery handoff | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [new4-geometric-fitter-recovery-handoff.md](in-progress/new4-geometric-fitter-recovery-handoff.md) |
| Qr/Qz shape sensitivity | feature | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [q-group-shape-sensitivity.md](in-progress/q-group-shape-sensitivity.md) |

Current emphasis for [#249](https://github.com/DVBeckwitt/ra_sim/issues/249):
New4 provider handoff, fixed-source request handoff, sensitivity scan,
one-param solves, `a` diagnosis, caked point reprojection, initial Rung 4 paired
solves, and fresh same-run Rung 5 blocks are validated for ladder work. Fresh
Rung 5 run `20260422_115256` passed Rungs 1-5; Rung 5 had `status == "ok"`,
four attempted blocks, four passed blocks, zero failed/timed-out blocks,
provider guard after blocks green, and unchanged `new4.json`
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
status: Rung 6/7 path mappings and expected timing IDs are excluded from timing
collection, Rung 5 skipped reports get timing metadata, fatal evidence still
aborts, local `a` usability failures stay local, missing dependencies skip only
affected blocks, stale external evidence remains rejected, manual selected-point
fits no longer inherit heavy solver defaults, and stale heartbeat traces are
reset before solve-rung writes. `full_fitter_validated == false`; no full,
feature, baseline, dynamic reanchor, multistart, polish, freeze/thaw, or feature
rung was run. GUI manual selected-point runtime behavior is updated, but it does
not validate full GUI fitter convergence. Next solve project remains separate
and unstarted.

## Known bugs

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | - |

## Planned features

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Mosaic fitter recovery | feature | - | none | p1 | 2026-04-20 | [mosaic-fitter.md](planned-features/mosaic-fitter.md) |
| Structure-factor fitter recovery | feature | - | none | p2 | 2026-04-20 | [structure-factor-fitter.md](planned-features/structure-factor-fitter.md) |
| Stacking-fault fitter recovery | feature | - | none | p2 | 2026-04-20 | [stacking-fault-fitter.md](planned-features/stacking-fault-fitter.md) |

## Archive

| Title | Type | Owner | Issue | Priority | Last updated | Path |
| --- | --- | --- | --- | --- | --- | --- |
| Analyze dual-source box peak fit | feature | - | none | p1 | 2026-04-20 | [analyze-dual-source-box-peak-fit.md](archive/analyze-dual-source-box-peak-fit.md) |
| Analyze overlay hardening | bug | - | none | p2 | 2026-04-20 | [analyze-overlay-hardening.md](archive/analyze-overlay-hardening.md) |
| Fast primary rasterizer | refactor | - | none | p2 | 2026-04-22 | [fast-primary-rasterizer.md](archive/fast-primary-rasterizer.md) |
| Lazy best-sample and Qr selection hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [lazy-best-sample-and-qr-selection-hardening.md](archive/lazy-best-sample-and-qr-selection-hardening.md) |
| Skip discarded fit hit tables | refactor | - | none | p2 | 2026-04-22 | [skip-discarded-fit-hit-tables.md](archive/skip-discarded-fit-hit-tables.md) |
| Startup default detector visibility regression | bug | - | none | p1 | 2026-04-18 | [startup-default-detector-visibility.md](archive/startup-default-detector-visibility.md) |
| Manual Qr/Qz and HKL picker alignment | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-21 | [sim-peak-overlay-recovery.md](archive/sim-peak-overlay-recovery.md) |
| Runtime cache diagnostic hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [runtime-cache-diagnostic-hardening.md](archive/runtime-cache-diagnostic-hardening.md) |
| Simulated peak overlay recovery history | investigation | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-19 | [sim-peak-overlay-recovery-history.md](archive/sim-peak-overlay-recovery-history.md) |
