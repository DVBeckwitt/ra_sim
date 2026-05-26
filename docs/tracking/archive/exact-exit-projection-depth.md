# Exact external exit projection and depth

Status: fixed
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-26

## Summary

Exact-mode diffraction now treats film-air exit projection as a boundary
wavevector problem instead of rebuilding the outgoing air angle with a clamped
`arccos(k_parallel / k0)` expression. The external projection path conserves
the tangential wavevector and rejects candidates whose in-plane wavevector is
larger than the air wavevector magnitude.

Zero sample thickness also now means no explicit Beer-path attenuation. The old
hidden fallback used an evanescent penetration depth and added one attenuation
factor per leg even when the user supplied `thickness = 0`.

## Root cause

The external projection blocks clamped `k_parallel / k0` into a propagating
angle even when `k_parallel > k0`. That could create detector hits for air-exit
waves that are evanescent and should not propagate to the detector.

The attenuation path also reused `1 / (2 Im(k_z))` when thickness was zero,
making zero thickness behave like an implicit finite depth instead of disabling
explicit slab-path attenuation.

## Fix

Implemented in `ra_sim/simulation/diffraction.py`:

- added an exact external air-exit helper that returns
  `(kx, ky, kz) = (kx_prime, ky_prime, sign(kz_prime) *
  sqrt(k0^2 - k_parallel^2))`;
- rejects external exits with `k_parallel > k0` and records the new
  `external_evanescent` projection-debug reason;
- shared the internal/external exit projection decision through one private
  compiled helper used by direct projection, fast weighted-event projection,
  nominal visibility checks, and the inlined weighted-event loop;
- replaced zero-thickness penetration-depth fallbacks with a helper that
  returns the positive effective optical depth in angstrom or `0.0`.

Implemented in `ra_sim/simulation/projection_debug.py`:

- added the `n_external_evanescent` counter and `external_evanescent` reason
  label.

Documentation was updated in `docs/simulation_and_fitting.md` and
`CHANGELOG.md`.

## Status

- Bug status: fixed for the exact external projection and zero-thickness
  attenuation behavior.
- Error status: evanescent air exits are now rejected instead of being forced
  onto the detector by clamp-based angle reconstruction.
- Feature status: no new user control, CLI flag, config key, dependency,
  saved-state schema, artifact schema, or GUI surface was added.
- Fast-optics status: unchanged. Fast optics remains rejected at compute entry
  points and no fast-LUT runtime call sites were added.
- Migration/deprecation status: no migration required. The removed behavior was
  internal exact-mode bug behavior, not a supported legacy mode.
- CI/automation status: no workflow changes required. Local quality gates cover
  the affected simulation paths.
- Shipping status: ready as a normal bug-fix commit. Rollback is a normal git
  revert; old regression images may lose artificial detector intensity from
  formerly accepted evanescent exits, which is the intended physics correction.

## Validation

Passed:

```powershell
python -m pytest tests/test_diffraction_constraints.py -k "external_air_exit or thickness or evanescent" -ra
python -m pytest tests/test_diffraction_constraints.py tests/test_diffraction_weighted_events.py tests/test_simulation_engine.py -ra
python -m ra_sim.dev check
git diff --check
rg "cos_out = _clamp\(kr /" ra_sim/simulation/diffraction.py
rg "1\.0 / np\.maximum\(2\.0 \* im_k_z" ra_sim/simulation/diffraction.py
rg "_build_fast_optics_lut_row\(" ra_sim
rg "_lookup_fast_optics" ra_sim
```

The acceptance greps returned no matches. A full `python -m pytest` run was not
clean in this workspace because of unrelated sampled failures and a Windows
fatal access violation outside the touched simulation files, so full-suite
green status is not claimed for this commit.
