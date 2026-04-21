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
| Geometric fitter recovery | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-20 | [geometric-fitter-recovery.md](in-progress/geometric-fitter-recovery.md) |

Current emphasis for [#249](https://github.com/DVBeckwitt/ra_sim/issues/249):
validate the live `new4.json` GUI preflight now that source-cache row
readiness is split from caked-view support-state work. Nearest-candidate
rebinding after normalized HKL and branch filtering remains the intended rule;
the active blocker is source-cache versus caked-view gating and observability.
The old `new2` and `new3` saved-state gates are retired. Manual Qr/Qz and HKL
picker alignment for [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
is resolved and archived below.

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
| Startup default detector visibility regression | bug | - | none | p1 | 2026-04-18 | [startup-default-detector-visibility.md](archive/startup-default-detector-visibility.md) |
| Manual Qr/Qz and HKL picker alignment | bug | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-21 | [sim-peak-overlay-recovery.md](archive/sim-peak-overlay-recovery.md) |
| Simulated peak overlay recovery history | investigation | - | [#248](https://github.com/DVBeckwitt/ra_sim/issues/248) | p1 | 2026-04-19 | [sim-peak-overlay-recovery-history.md](archive/sim-peak-overlay-recovery-history.md) |
