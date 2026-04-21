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
| Geometric fitter recovery | investigation | - | [#249](https://github.com/DVBeckwitt/ra_sim/issues/249) | p1 | 2026-04-21 | [geometric-fitter-recovery.md](in-progress/geometric-fitter-recovery.md) |

Current emphasis for [#249](https://github.com/DVBeckwitt/ra_sim/issues/249):
validate `new4.json` point-provider parity before optimizer execution. The
manual Qr picker selected/refined pairs, geometry-fit provider pairs, and
dataset manual point rows must match exactly. The Qr/Qz branch-seed bug/error
scope is closed at the UI boundary: initial raw-cache preview, manual toggle,
refresh/view-change, and place setup keep one mosaic-top seed per normalized
branch for each real Qr/Qz group while preserving branch/reflection/ray
provenance. The caked-mode detector-return bug is also closed: selected
`2theta,phi` Qr/Qz seeds redraw in detector view through the same
detector-display projection path as simulation markers. The Qr/Qz picker now
also stays clickable after switching detector to caked or caked to detector,
even when cached candidate rows still carry stale active-view coordinates.
Detector-view Qr/Qz hit-table rows now use the simulated detector display frame
for click targets; the background detector adapter applies only to explicitly
tagged background/native-detector rows. Caked selection and caked-to-detector
conversion remain locked by regression tests.
Optimizer validation stops at bounded ladder rung 1 when the request would use
fallback rows;
solve rungs wait until objective dry-run reports zero fallback and zero missing
fixed-source rows. The bridge currently copies provider identity into the
optimizer request and fails before `least_squares` when fixed-source fields are
incomplete. The old full quality baseline runner remains blocked until the
ladder finds a stable parameter set. The old `new2` and `new3` saved-state gates
are retired.

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
