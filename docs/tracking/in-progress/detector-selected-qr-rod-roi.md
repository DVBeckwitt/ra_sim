# Detector Selected-Qr Rod ROI

## Status

Completed and validated on 2026-04-30.

## Problem

Detector view enabled Selected-Qr Rod ROI but still displayed the legacy detector
`2theta/phi` integration overlay. Dragging in detector view also used the old
angular range path instead of selecting Qz bounds from detector pixels.

## Change

- Added a shared detector-native Qr rod support mask built from detector Qr/Qz
  maps, detector validity, finite Q, phi windows, optional Qz clipping, and
  optional rod shape support.
- Reused that support for detector overlay display, detector Qz profiles,
  detector drag preview, and detector drag release.
- Made detector rod mode hide invalid/no-mask overlays instead of falling back
  to the angular detector ROI.
- Let detector-view rod controls work without prior caked axes by using
  detector Qz extent and detector-profile bin defaults.

## Validation

- `python -m pytest tests/test_gui_qr_cylinder_overlay.py tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py tests/test_gui_state_io.py -ra`
  passed with 514 tests.
- `python -m compileall ra_sim tests` passed.
- `python -m ra_sim.dev check` still stops on pre-existing formatting drift in
  `ra_sim/fitting/optimization.py`; the touched runtime file was formatted.
