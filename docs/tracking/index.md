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

Current emphasis for [#249](https://github.com/DVBeckwitt/ra_sim/issues/249):
New4 provider handoff, fixed-source request handoff, sensitivity scan,
one-param solves, `a` diagnosis, caked point reprojection, and initial Rung 4
paired solves are validated. Rung 5 small cumulative blocks are implemented and
have debug pair-backed pass evidence for three theta/distance blocks, but fresh
same-run Rung 5 did not complete because prerequisite singleton/diagnosis
evidence timed out. `[a, c, psi_z]` is skipped until a psi pair passes. The
repeated cold-start speed bug is fixed for solve rungs by reusing one warm
solver context in-process; Rung 4 pair solves measured about 53x faster, from
315.74 seconds total to 5.91 seconds total. Next: decide whether to run the
optional psi pair extension for `[a, c, psi_z]` or hold for a separately
approved next rung; no full-fit validation claimed yet.

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
| Skip discarded fit hit tables | refactor | - | none | p2 | 2026-04-22 | [skip-discarded-fit-hit-tables.md](archive/skip-discarded-fit-hit-tables.md) |
| Startup default detector visibility regression | bug | - | none | p1 | 2026-04-18 | [startup-default-detector-visibility.md](archive/startup-default-detector-visibility.md) |
| Manual Qr/Qz and HKL picker alignment | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-21 | [sim-peak-overlay-recovery.md](archive/sim-peak-overlay-recovery.md) |
| Qr/Qz shape sensitivity | feature | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [q-group-shape-sensitivity.md](archive/q-group-shape-sensitivity.md) |
| Runtime cache diagnostic hardening | bug | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-22 | [runtime-cache-diagnostic-hardening.md](archive/runtime-cache-diagnostic-hardening.md) |
| Simulated peak overlay recovery history | investigation | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-19 | [sim-peak-overlay-recovery-history.md](archive/sim-peak-overlay-recovery-history.md) |
