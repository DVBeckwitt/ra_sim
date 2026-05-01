# Selected Qr Rod Band Redesign

Status: implemented, targeted validation complete.
Type: GUI feature and visual bug fix.
Last updated: 2026-05-01.

## Problem

The selected Qr rod overlay worked numerically but did not read clearly. The
delta_Qr region looked like a generic filled mask or thick region without a
precise center. Users also had to tune delta_Qr through entry text only.

## Change

- Render selected Qr rods as a core centerline, translucent ribbon, and dashed
  band limits at Qr0 - delta_Qr and Qr0 + delta_Qr.
- Use the same selected-rod visual language in detector and caked views.
- Draw fill first, limits second, and centerline last so the real Qr cylinder
  line stays readable.
- Mute non-primary selected rods visually while keeping their numerical masks
  and profile semantics unchanged.
- Add a visible delta_Qr thickness slider near the selected-Qr controls. The
  slider and manual entry stay synchronized and write to the existing delta_qr
  state used by overlays, masks, profiles, save, and load.
- Add a compact delta_Qr cue label so the active band thickness is visible.

## Preserved Semantics

- The centerline marks the true Qr rod.
- The ribbon means Qr0 +/- delta_Qr.
- Profiles and masks still use the same numerical delta_Qr.
- Detector selected-Qr Qz profiles remain caked-only.
- Detector union masks remain overlay and drag-support only.
- Per-rod profiles still use per-rod caked masks.
- Legacy saved states still load when slider metadata is absent.

## Validation

- `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py tests/test_gui_state_io.py -ra`
  - Result: 552 passed, 1 failed.
  - Known unrelated failure:
    `tests/test_gui_runtime_import_safe.py::test_runtime_session_hkl_pick_builds_grouped_cache_from_stored_raw_peak_rows`
    still expects the old HKL grouped-cache tuple keys.
- `python -m compileall ra_sim tests`
  - Result: passed.
- `git diff --check`
  - Result: passed, with CRLF warnings only.
- `ruff check` and `ruff format --check` on the selected-Qr touched files
  passed.

## Current Status

The selected-Qr band redesign and slider feature are complete. The remaining
red test is an existing HKL grouped-cache expectation mismatch, not part of the
selected-Qr rod work.
