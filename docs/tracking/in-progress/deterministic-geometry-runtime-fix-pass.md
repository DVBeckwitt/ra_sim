# Deterministic Geometry Runtime Fix Pass

Status: in-progress
Type: bug
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-05-04

## Summary

Narrow runtime hardening pass for geometry-fit determinism. This pass fixes
helper order, optional preflight projection callbacks, exact caked trial
parameter authority, stale guard expected strings, and the invalid live-cache
source-row regression.

## Current state

- Bug fixed: `ra_sim/headless_geometry_fit.py` defines `_signature_numeric()`
  and `_signature_summary()` before caked payload signature callers.
- Error fixed: `scripts/debug/validate_geometry_preflight_rebind.py` tolerates
  missing `projection_callbacks` for non-caked validation paths. Caked
  grouping/projection paths now fail with explicit callback-required reasons.
- Bug fixed: `ra_sim/gui/geometry_fit.py` exact caked residual-only/trial
  projection uses call-time `local_params` for bundle resolution, projection
  signatures, and returned metadata.
- Bug fixed: stale caked guard test strings now match current
  projector-or-sim-visual diagnostics.
- Correctness regression fixed: invalid live runtime cache for required
  geometry-fit pairs is rejected instead of accepted as `live_runtime_cache`.
  Rebuild falls back to fresh source rows, records invalid live validation, and
  records dual-path diff metadata.
- Source-row status: collapsed `00l` coverage aliases remain valid, non-00l
  branch mismatches still fail, required callback payloads no longer leak
  internal `source_label`, and targeted full-simulation fallback diagnostics
  report unrelated scored rows.
- Bug fixed: live-cache canonical filtering is now scoped to trusted
  full-reflection required pairs. Q-group and disordered Q-group required pairs
  can reuse matching non-canonical live rows when existing source, Q group, HKL,
  branch, and finite-coordinate checks pass. Full-reflection pairs still reject
  untrusted matching candidates with `missing_canonical_candidate`.
- Validation invariant added: Qr branch cardinality is independent from
  canonical full-reflection filtering. Non-00l rows, where `(h, k) != (0, 0)`,
  keep two branch-specific points per Qr; 00l rows, where `h == 0` and
  `k == 0`, use one collapsed branch point per Qr.

## Validation

- `python -m pytest tests/test_gui_geometry_fit_workflow.py -k "exact_projector_uses_current_local_params or caked_point_reprojection or stale_caked_report or combined_caked_guard" -ra`
  - Result: `26 passed, 1 skipped`.
- `python -m pytest tests/test_gui_runtime_import_safe.py -k "headless_geometry_fit or caked_manual_fit or projection_callbacks" -ra`
  - Result: `2 passed`.
- `python -m pytest tests/test_cli_geometry_fit.py -ra`
  - Result: `24 passed`.
- `python -m pytest tests/test_geometry_fitter_cache_regression_gate_script.py -ra`
  - Result: `5 passed`.
- `python -m pytest tests/test_gui_geometry_fit_workflow.py::test_rebuild_geometry_fit_source_rows_rejects_invalid_live_cache_for_required_pair_and_records_dual_path_diff -ra`
  - Result: `1 passed`.
- Focused source-row/validator cluster
  - Result: `5 passed`.
- `python -m pytest tests/test_geometry_fit_live_cache_validation_acceptance.py -q`
  - Result: `16 passed`.
- `python -m compileall -q ra_sim/gui/geometry_fit.py tests/test_geometry_fit_live_cache_validation_acceptance.py tests/test_gui_geometry_fit_workflow.py`
  - Result: passed.
- `git ls-files --eol -- ra_sim/gui/geometry_fit.py tests/test_geometry_fit_live_cache_validation_acceptance.py tests/test_gui_geometry_fit_workflow.py`
  - Result: touched files are `i/lf w/lf`.
- `python -m pytest tests/test_geometry_fitting.py tests/test_gui_geometry_fit_workflow.py --tb=short -ra`
  - Result: `803 passed, 1 skipped, 6 failed`.
  - Remaining failures are outside this narrow pass: New4 sensitivity CLI
    smoke, dynamic reanchor stale peak index, three New4 full-beam finalizer
    tests, and New4 Rung 1 objective dry run.
- `python -m ra_sim.dev check`
  - Blocked by existing frozen formatter drift:
    `Would reformat: ra_sim\fitting\optimization.py`.
- `git diff --check`
  - Result: passed, with Windows CRLF checkout warnings only.

## Next actions

- Keep `ra_sim/fitting/optimization.py` branch-switch behavior frozen.
- Keep fit-space origin precedence frozen.
- Keep New4 production gate semantics frozen.
- Do not touch dynamic reanchor until source-row/validator failures stay
  reduced and remaining failures are scoped separately.
- Next pass should isolate the remaining New4/dynamic failures without folding
  them into this deterministic runtime helper pass.
