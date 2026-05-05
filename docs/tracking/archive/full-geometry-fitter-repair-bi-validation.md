# Full Geometry Fitter Repair and Bi Validation

Status: complete
Type: bug/feature
Owner: -
Issue: #249
Priority: p1
Completed: 2026-05-05

## Summary

The full geometry fitter repair is complete for active validation. The caked
manual-fit pipeline now treats exact caked projection metadata as geometry-only
state, separate from caked display/background images. Row projection requires a
ready projection payload with a `CakeTransformBundle`; invalid axes-only or
detector-mismatched payloads fail closed and do not generate over the error.

Manual caked picking now keys warm-cache refinement on stable source state,
simulation generation, verified projection content, and full-content caked
axis/permutation tokens. Display image churn, zero-support NaN sanitization,
and copied-but-equal axes do not invalidate the geometric cache; real axis,
permutation, projection-token, or bundle-content changes do.

Headless saved manual caked Bi states now default to `--seed-policy direct`.
Saved Bi rows with fixed source identity, HKL, Q-group key, and caked
coordinates rebind to provider-backed live rows without silent HKL fallback.
Accepted fits require all fixed pairs matched, no missing pairs, no branch
mismatches, and a lower final residual.

## Status

- Bug status: fixed for cold-start exact-caked projector hydration, caked row
  projection ordering, warm manual caked-pick cache churn, and Bi saved-row
  fixed-pair rebinding.
- Error status: caked manual paths fail closed on invalid projection payloads
  and no longer pass `projection_payload=None` into caked row rebuild.
- Feature status: active quality baseline validates Bi2Se3 and Bi2Te3 saved GUI
  states from `user_data_root()` with direct fixed-pair solves.
- Gate status: New4 is removed from active regression gates and default quality
  baseline state selection. New4 diagnostics remain historical/opt-in.

## Final Token-Trust Follow-Up

The final merge blocker was stale or malformed exact-caked projection tokens.
Runtime projection storage now recomputes `projection_content_token_v3` from
the stored projection fields and a private `CakeTransformBundle` copy, then
marks the token with
`projection_content_token_source="runtime_projection_storage_copy_v3"`.
Token-only payloads are classified as absent, so they do not block generated
fallback. Source-less or legacy projection tokens are not manual warm-cache
correctness keys.

Dense LUT arrays are copied into deterministic contiguous dtypes. Sparse LUTs
are canonicalized as sorted CSR content for hashing/copying. Only private
stored arrays are read-only; shared incoming bundles remain mutable and are not
silently frozen.

Bug/error/feature status after this follow-up:

- Bug status: fixed. Stored projection digests no longer trust stale incoming
  tokens, and mutating the original bundle after storage cannot change the
  stored projector digest.
- Error status: fixed. Token-only projection payloads are absent, explicit
  invalid axes-only payloads still fail closed, and caked row rebuilds require
  an exact `CakeTransformBundle`.
- Feature status: complete. Warm caked picking remains fast because repeated
  `get_pick_cache()` calls consume the trusted stored token instead of hashing
  LUT content in the click path.

## Validation

- `python -m compileall -q ra_sim tests scripts`
- `python -m pytest tests/test_cli_geometry_fit.py tests/test_geometry_fit_quality_baseline.py -ra`
- `python -m pytest -q tests/test_manual_geometry_live_peak_cache.py tests/test_gui_runtime_import_safe.py tests/test_gui_geometry_fit_workflow.py -ra`
- `python -m pytest -q tests/test_geometry_fitting.py -k "exact_caked" tests/test_geometry_fitter_cache_regression_gate_script.py -ra`
- `python scripts/debug/run_geometry_fit_quality_baseline.py --output-root C:\asr_work\ra_sim_bi_quality_baseline_latest --state-timeout-seconds 700`
- `python -m ra_sim.dev check`

Real Bi results:

- Bi2Se3: accepted, 82/82 fixed manual pairs matched, missing 0, branch
  mismatch 0, direct RMS `34.5307 -> 31.078112`,
  `seed_policy=direct`, `active_fit_mode=fixed_manual_pair_direct_least_squares`,
  `fit_space_projector_kind=exact_caked_bundle`.
- Bi2Te3: accepted, 84/84 fixed manual pairs matched, missing 0, branch
  mismatch 0, direct RMS `36.8629 -> 34.394414`,
  `seed_policy=direct`, `active_fit_mode=fixed_manual_pair_direct_least_squares`,
  `fit_space_projector_kind=exact_caked_bundle`.

## Compatibility

The public CLI adds `fit-geometry --seed-policy direct`. Saved GUI-state schema
and manual caked coordinate semantics are unchanged. Active validation defaults
change from New4 to Bi2Se3/Bi2Te3 under the user data root.
