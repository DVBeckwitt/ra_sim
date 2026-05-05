# Cold-Start Caked Manual Fit Projector

Status: complete
Type: bug
Owner: -
Issue: #249
Priority: p1
Completed: 2026-05-05

## Summary

Cold-start manual geometry fitting in caked view could fail before the
optimizer started. The runtime reached `source_cache_full_cake_ready`, then
lost the exact caked fit-space projector during exact payload
hydration/storage and surfaced `exact caked projector unavailable` for the
selected background.

The fix splits geometric exact-caked projection metadata from display
intensity data. Exact projector hydration now uses detector shape, caked axes,
raw azimuth axis, row permutation, and `CakeTransformBundle`; display
background finiteness is validated only for image-facing paths such as caked
display storage and peak-refinement image use.

## Current state

- `geometry_fit.py` accepts projection-only exact caked payloads when
  `require_background=False`, while `require_background=True` remains strict
  for image use.
- `geometry_fit_caked_projection_payload` strips caked payloads down to the
  geometric fields needed for exact projector hydration.
- Caked display cache storage sanitizes only zero-support nonfinite density
  bins to `0.0`; supported-bin nonfinite values fail with
  `nonfinite_supported_caked_background`.
- Runtime caked preflight stores and hydrates
  `projection_payload_by_background[bg_idx]` before row projection or dataset
  build.
- Generated/noncurrent caked projection now distinguishes absent, invalid, and
  ready payload states. Empty skeleton payloads may use generated fallback;
  axes-only or detector-mismatched payloads fail closed.
- Worker row rebuild now has a hard caked-mode guard: it must receive a
  projection payload with a `CakeTransformBundle`, or it fails before source-row
  rebuild.
- Caked projection signatures are computed from projection payloads, not
  display caked images.
- Manual-fit dataset prep uses projection readiness for
  `fit_space_projector_kind=exact_caked_bundle` and keeps image readiness
  separate.
- Worker/manual fit code uses a separate caked projection accessor; the
  image-facing `geometry_manual_caked_view_for_index` accessor is not fed
  axes-only projection payloads.
- Targeted fresh simulation now emits timeout diagnostics and marks later
  ready/failed completion as late when a timeout fired.
- Headless saved manual caked fits now share the projection-only payload
  contract and default to direct fixed-pair least squares for saved Bi states.
- Active validation uses Bi2Se3 and Bi2Te3 saved GUI states from
  `user_data_root()`; New4 remains available as historical diagnostics but is
  no longer an active cache-regression gate.
- Final projection-token trust hardening stores private read-only exact-caked
  projection bundles, treats token-only payloads as absent, and lets warm
  caked pick-cache reuse consume only runtime-sourced verified tokens.

## Bug/error/feature status

- Bug status: fixed for the current-background projector hydration/cache split,
  generated/noncurrent fallback handling, and Bi-style saved manual caked row
  rebinding.
- Error status: `invalid_exact_caked_payload` is replaced at the touched
  storage/preflight sites by field-level statuses including
  `projection_payload_ready`, `missing_exact_caked_bundle`,
  `projection_payload_missing_axes`, `projection_payload_axis_mismatch`,
  `nonfinite_supported_caked_background`, and
  `empty_bin_nan_density_sanitized`.
- Feature status: cold-start caked manual fitting can build the exact
  fit-space projector without requiring a caked display image to be finite in
  undefined empty bins; warm caked manual picking skips row refinement on
  unchanged geometry signatures.
- Safety status: explicit invalid caked projection payloads are not replaced by
  generated fallback and do not fall back to detector/current-view projection.
- Validation status: structured Bi2Se3/Bi2Te3 progress reports confirm
  `seed_policy=direct`, `active_fit_mode=fixed_manual_pair_direct_least_squares`,
  `fit_space_projector_kind=exact_caked_bundle`, all fixed pairs matched, and
  direct RMS reduction.

## Validation

Passing checks:

- `python -m compileall -q ra_sim tests scripts`
- `python -m pytest tests/test_cli_geometry_fit.py tests/test_geometry_fit_quality_baseline.py -ra`
- `python -m pytest -q tests/test_manual_geometry_live_peak_cache.py tests/test_gui_runtime_import_safe.py tests/test_gui_geometry_fit_workflow.py -ra`
- `python -m pytest -q tests/test_geometry_fitting.py -k "exact_caked" tests/test_geometry_fitter_cache_regression_gate_script.py -ra`
- `python scripts/debug/run_geometry_fit_quality_baseline.py --output-root C:\asr_work\ra_sim_bi_quality_baseline_latest --state-timeout-seconds 700`
- `python -m ra_sim.dev check`

Real Bi saved-state results:

- Bi2Se3: accepted, 82/82 fixed manual pairs matched, missing 0, branch
  mismatch 0, direct RMS `34.5307 -> 31.078112`.
- Bi2Te3: accepted, 84/84 fixed manual pairs matched, missing 0, branch
  mismatch 0, direct RMS `36.8629 -> 34.394414`.

## Next actions

- Keep New4 ladder diagnostics opt-in and out of active gates unless a future
  task explicitly promotes them again.
- Treat future caked manual-fit regressions as projection-payload contract
  failures first: no detector fallback, no analytic fallback, no
  `projection_payload=None`.

## Links

- Related tracker: [new4-geometric-fitter-recovery-handoff.md](../in-progress/new4-geometric-fitter-recovery-handoff.md)
- Related tracker: [deterministic-geometry-runtime-fix-pass.md](../in-progress/deterministic-geometry-runtime-fix-pass.md)
