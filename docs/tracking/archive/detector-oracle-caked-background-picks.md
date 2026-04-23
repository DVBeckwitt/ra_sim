# Detector-Oracle Caked Background Picks

Status: completed
Type: bug
Owner:
Issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
Priority: p1
Last updated: 2026-04-23

## Summary

Caked manual background picks now use the detector background-pick path as the
oracle. A caked click is converted from caked display coordinates into
`(2theta, phi)`, reversed through the existing detector LUT callback, refined by
the detector background helper, converted back to caked angles, and stored as
background truth.

## Current state

- `background_two_theta_deg` and `background_phi_deg` are the authoritative
  background point.
- Caked `bg_display` is rendered from those stored angles.
- Detector `bg_display` is rendered through the existing reverse-LUT callback.
- Raw caked/display fields and `caked_x/y` remain cache or legacy aliases.
- `refined_sim_*`, `sim_display`, and simulated Qr source identity do not move
  background display points.
- The simulated Qr projection cache was not changed for this fix.

## Next actions

- Keep issue `#248` open only for unrelated remaining picker/projection work.
- Do not reintroduce an independent caked-background peak finder.
- If future caked-background drift appears, debug the detector-oracle resolver
  first, not the simulated Qr projection cache.

## Validation

- `python -m py_compile ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py`: PASS
- `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "caked or background or reverse"`: PASS
- `python -m pytest tests/test_manual_geometry_live_peak_cache.py -k "background or caked"`: PASS
- `python -m ruff check ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py tests/test_manual_geometry_selection_helpers.py tests/test_manual_geometry_live_peak_cache.py`: PASS
- `git diff --check -- ra_sim/gui/manual_geometry.py ra_sim/gui/_runtime/runtime_session.py tests/test_manual_geometry_selection_helpers.py tests/test_manual_geometry_live_peak_cache.py`: PASS
- `python -m ra_sim.dev check`: BLOCKED at format check by unrelated formatter drift in `ra_sim/fitting/optimization.py` and `tests/test_timing.py`.

## Links

- Related issue: [#248](https://github.com/DVBeckwitt/ra_sim/issues/248)
- Tracking index: [docs/tracking/index.md](../index.md)
