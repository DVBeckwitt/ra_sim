# Lazy Best-Sample And Qr Selection Hardening

Status: completed
Type: bug
Owner: -
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-22

## Summary

Completed the lazy best-sample buffer performance patch and hardened the
adjacent Qr/Qz selection and geometry-fit diagnostic paths found during review.
Image-only, no-cache simulation requests do not need implicit
`best_sample_indices_out` provenance buffers, while cache-building and explicit
caller buffers still retain provenance. The follow-up review closed regressions
around selected Qr rows, caked manual picks, Numba cache reuse, and New4 pair
rung diagnostics.

## Current state

Bug/error status: fixed.

- Selected Qr/Qz runtime rows that come from detector-view peak records are
  projected through the detector-display adapter, while raw simulation cache
  rows still use `native_sim_to_display_coords`.
- Source-backed saved caked manual picks refresh their detector/display point
  from saved `(2theta, phi)` values; generic stale caked fields still preserve
  existing detector truth.
- Qr/Qz sensitivity rows keep canonical branch/source identity when stable.
- The New4 pair-rung caked guard no longer loops forever after a guarded
  failure.
- Diagnostic intersection analysis avoids the unstable compiled
  `solve_q`/intensity path and uses a NumPy diagnostic solver instead.
- The default Numba cache directory is Python-cache-tag scoped to avoid stale
  cross-version compiled artifacts.

Feature/performance status: shipped in code.

- Best-sample provenance buffers are lazy for image-only/no-cache forward and
  QR rod simulations, but still present for intersection-cache builds and
  explicit caller-provided buffers.

## Next actions

None for this bundle. Continue any higher geometry-fit ladder work under the
active New4 fitter tracking items.

## Validation

- `python -m pytest -q -p no:cacheprovider tests/test_simulation_engine.py`: 37 passed.
- GUI Qr/peak/projection chunk: 225 passed.
- `python -m ra_sim.dev check`: format, ruff, 239 fast tests, and mypy passed.
- `python -m pytest -q -p no:cacheprovider`: 2449 passed, 3 skipped.
- New4 final rungs run `20260422_codex_final_rungs_1_4_v5`:
  - one-param: `status: ok`
  - caked point reprojection: `status: pass`, `failures: []`
  - pair/rung 4: `status: ok`

## Links

- Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
