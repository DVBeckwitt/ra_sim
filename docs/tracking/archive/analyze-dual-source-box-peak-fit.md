# Analyze dual-source box peak fit

Status: resolved
Type: feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-20

## Summary

Analyze mode now treats the selected integration box as the hard domain for
both background/measured and simulated peak discovery. The feature fixes the
old source-biased workflow where a visible background caked image could prevent
simulated peaks in the same box from being selected and fitted.

## Current state

- `Find Peaks in Box` discovers background peaks from the background caked
  image and simulated peaks from simulated peak records or simulated caked
  image fallback.
- Discovery runs inside selected-box caked image crops or selected-box record
  filters only; outside-box peaks do not consume candidate limits.
- Background and simulated peaks dedupe only against the same normalized
  source, so overlapping source-specific peaks both remain selected.
- Fit with no selected peaks auto-discovers both sources in the selected box
  before fitting.
- Radial and azimuth fits are grouped by source and model:
  `radial:background:<model>`, `radial:simulated:<model>`,
  `azimuth:background:<model>`, and `azimuth:simulated:<model>`.
- Source-specific fitting calls `_analysis_curve_data(..., allow_fallback=False)`;
  missing source curves emit failure entries instead of fitting against the
  other source.
- Fit curves still span the full selected radial/azimuth 1D integration
  windows only.
- Analyze markers and fit summaries show source-local labels such as `B1` and
  `S1`.

## Next actions

No known follow-up bugs from code review. Manual GUI smoke should still confirm
marker visibility and full-window fit curves with live experiment data.

## Validation

Validated on 2026-04-20 with:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gui_analysis_peak_tools.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py -q`
- `python -m py_compile ra_sim/gui/analysis_peak_tools.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/views.py ra_sim/gui/state.py tests/test_gui_runtime_import_safe.py tests/test_gui_views.py`

The targeted tests passed locally. `python -m ra_sim.dev check` was not green
because the repository already has formatter drift in unrelated files outside
this feature's staged scope.

## Links

- Runtime implementation: `ra_sim/gui/_runtime/runtime_session.py`
- Fit helper summaries: `ra_sim/gui/analysis_peak_tools.py`
- View wiring: `ra_sim/gui/views.py`
- Regression coverage: `tests/test_gui_runtime_import_safe.py`
- View coverage: `tests/test_gui_views.py`
- Tracking hub: [docs/tracking/index.md](../index.md)
