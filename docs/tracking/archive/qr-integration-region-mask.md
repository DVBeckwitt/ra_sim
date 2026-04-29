# Qr integration region mask

Status: resolved
Type: bug
Owner:
Issue: none
Priority: p2
Last updated: 2026-04-29

## Summary

Selected-Qr rod caked integration masks could drop the filled ROI when the
projected Qr boundary traces contained isolated nonfinite caked samples, such
as the +/-90 degree caked azimuth singularity. The finite samples on either
side were split into one-point runs, so polygon rasterization did not fill the
surrounding Qz/Qr slice. Changing the QR half-width could update only sparse
boundary points instead of changing the displayed mask shape.

## Current state

Resolved. `ra_sim/gui/qr_cylinder_overlay.py` now builds paired finite trace
runs with a one-sample bridge for isolated nonfinite bins, while still splitting
at wrapped-phi seams and non-bridgeable selection gaps. Caked polygon
rasterization filters nonfinite trace samples before constructing the path, so a
single bad boundary point no longer poisons the ROI. Boundary grid points are
included in the mask fill to keep sparse caked grids from dropping valid edge
rows or columns.

Both selected-Qr Qz ROI masks and active Qr cylinder caked band masks use the
same run-bridging behavior. Wrapped-phi seams remain real discontinuities and
are not filled across.

## Next actions

- None for this bug. Reopen only if a selected-Qr caked ROI vanishes around an
  isolated nonfinite trace sample or QR half-width changes no longer change the
  filled caked mask shape.
- Separate repo health item: `python -m ra_sim.dev check` still stops on
  unrelated formatting issues outside the QR overlay files.

## Validation

- 2026-04-29: `python -S -m py_compile ra_sim/gui/qr_cylinder_overlay.py tests/test_gui_qr_cylinder_overlay.py`
  passed.
- 2026-04-29: `python -m ruff check ra_sim/gui/qr_cylinder_overlay.py tests/test_gui_qr_cylinder_overlay.py`
  passed.
- 2026-04-29: `python -m pytest tests/test_gui_qr_cylinder_overlay.py -ra`
  passed, `19 passed`.
- 2026-04-29: `python -m ra_sim.dev check` failed on unrelated formatting
  issues in `ra_sim/fitting/optimization.py`,
  `ra_sim/gui/_runtime/primary_cache_helpers.py`,
  `ra_sim/gui/_runtime/runtime_session.py`,
  `tests/test_gui_runtime_primary_cache.py`, and `tests/test_timing.py`.

## Links

- Runtime path: [ra_sim/gui/qr_cylinder_overlay.py](../../../ra_sim/gui/qr_cylinder_overlay.py)
- Test path: [tests/test_gui_qr_cylinder_overlay.py](../../../tests/test_gui_qr_cylinder_overlay.py)
- Changelog: [CHANGELOG.md](../../../CHANGELOG.md)
- Tracking index: [docs/tracking/index.md](../index.md)
