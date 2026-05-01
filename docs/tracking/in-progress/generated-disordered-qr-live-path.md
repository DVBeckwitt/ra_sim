# Generated Disordered Qr Live Path

Status: implemented, focused live-path validation green
Type: bug fix / GUI picker feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-05-01

## Summary

The live GUI could generate a HT-shifted disordered PbI2 CIF but still refresh
the Qr/Qz picker from primary-only cached rows. The reported symptom was
`w(p≈1)` nonzero with the same or fewer picker groups and no clickable
disordered-phase candidates.

## Current State

- Generated disordered Qr refs are controlled by a visible checkbox:
  `Include generated disordered-phase Qr refs`.
- The checkbox defaults on, persists in GUI state, and is independent from
  `Include packaged 6H Qr refs`.
- Runtime enable status now reports enabled/disabled reasons for generated
  disordered refs: checkbox off, stacking disorder off, zero disordered weight,
  or missing active primary CIF.
- The generated disordered source signature participates in Qr/Qz freshness,
  while the primary rendered-image cache uses a disabled disordered signature so
  Qr refs do not force a primary image redraw.
- Live Qr/Qz refresh evaluates the generated inventory from the active primary
  CIF, never from packaged `PbI2_6H.cif`.
- Hit-table-only updates schedule generated disordered collection when stored
  disordered rows are missing or stale, even when primary hit tables/images are
  reusable.
- Disordered hit-table collection runs with `accumulate_image=False`.
- Current-simulation Qr/Qz refresh publishes stored disordered rows into the
  active picker cache and keeps `source_label="disordered_phase"`.
- Ordered and disordered rows with the same numeric Qr/Qz remain separate by
  source label.

## Bug / Error / Feature Status

- Bug status: fixed for the user-reported primary-only picker cache reuse.
- Error status: no known generated-disordered runtime error remains in focused
  coverage.
- Feature status: implemented with explicit GUI control, live runtime logging,
  cache invalidation, hit-table scheduling, publishing, and picker-cache tests.
- Remaining risk: full `ra_sim.dev check` is still blocked by pre-existing
  formatting drift in `ra_sim/fitting/optimization.py`, outside this bug fix.

## Diagnostics

Every live Qr/Qz listing refresh that emits `Updated listed Qr/Qz peaks:` also
logs exactly one generated-disordered decision:

- `Disordered Qr refs enabled: true`
- `Disordered Qr refs skipped: <reason>`

Generated-disordered diagnostics also report:

- inventory source/generated CIF paths
- unavailable inventory skip reason
- collected hit-table count
- published group/peak count
- source counts in the Qr/Qz listing message, for example
  `sources: primary=22, disordered_phase=22`

## Validation

Focused validation:

- `python -m pytest tests/test_disordered_phase_user_report_live_path.py -q`
  - 1 passed.
- `python -m pytest tests/test_disordered_phase_live_q_group_refresh.py tests/test_disordered_phase_ui_enable.py tests/test_disordered_phase_inventory_live_path.py tests/test_disordered_phase_hit_table_scheduling.py tests/test_disordered_phase_current_refresh.py tests/test_disordered_phase_user_report_live_path.py -q`
  - 29 passed.
- `python -m pytest tests/test_disordered_phase_user_report_live_path.py tests/test_disordered_phase_q_group_cache.py tests/test_disordered_phase_hit_tables.py tests/test_disordered_phase_logging.py -q`
  - 15 passed.
- `python -m compileall ra_sim tests`
  - passed.
- `python -m ruff check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py tests/test_disordered_phase_user_report_live_path.py`
  - passed.
- `python -m ruff format --check ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/geometry_q_group_manager.py tests/test_disordered_phase_user_report_live_path.py`
  - passed.

Blocked validation:

- `python -m ra_sim.dev check`
  - blocked by existing `ra_sim/fitting/optimization.py` formatting drift.
