# Qr/Qz Shape Sensitivity

Status: in-progress
Type: feature
Owner: -
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-04-22

## Summary

Extend the Qr/Qz peak sensitivity export from refined peak maxima to branch
center-of-mass and shape metrics. Ray-cloud COM is the default metric; refined
max remains available with `--metric refined_max`, and image-ROI COM is an
optional metric.

## Current state

Implemented in the peak sensitivity adapter and debug export script. The change
does not edit simulation internals, cache schema/invalidation, or
`SimulationRuntimeState`.

Added a headless transport-vs-full-recompute validation tool for Q-group peak
COM and shape metrics. The validator compares frozen detector/source points
against existing full recompute paths for `image_roi_com` and `ray_cloud_com`,
reports separate COM and shape decisions, and keeps same-ROI transport
transform-only. Review bugs around same-ROI recaking, COM decisions being
blocked by shape mismatch, and tautological transport tests were fixed.

## Next actions

- Review the transport validation artifacts before enabling any shortcut in
  production fitting. Current `new4.json` status is conservative: image-ROI
  same-ROI COM transport has zero transport error but remains blocked by
  baseline transport mismatch, while ray-cloud transport is blocked by
  insufficient source points. No parameter is cleared for background rebuild
  avoidance from this run.

## Validation

- `python -m pytest tests\test_peak_sensitivity.py -q`: 40 passed.
- `python -m ra_sim.dev format-check`: passed.
- `python -m ruff format --check ra_sim\gui\peak_sensitivity.py scripts\debug\run_q_group_peak_sensitivity.py tests\test_peak_sensitivity.py`: passed.
- `python -m ruff check ra_sim\gui\peak_sensitivity.py scripts\debug\run_q_group_peak_sensitivity.py tests\test_peak_sensitivity.py`: passed.
- `python -m ra_sim.dev check`: 239 fast tests passed; ruff and mypy subset passed.
- Acceptance: `python .\scripts\debug\run_q_group_peak_sensitivity.py --state .\artifacts\geometry_fit_gui_states\new4.json --group-key q_group,primary,1,5 --metric all --outdir .\artifacts\q_group_peak_sensitivity\new4_shape_com`.
  Baseline branch count was 2. Ray-cloud branch point counts were 1 and 2, so
  both branches reported `insufficient_cloud_points`. Image-ROI COM had 15 and
  14 positive-weight ROI points and reported `ok`.
- `python -m pytest tests\test_peak_sensitivity.py tests\test_peak_transport_validation.py -q`:
  50 passed.
- `python -m ruff check ra_sim\gui\peak_transport_validation.py scripts\debug\run_q_group_peak_transport_validation.py tests\test_peak_transport_validation.py`:
  passed.
- `python -m ruff format --check ra_sim\gui\peak_transport_validation.py scripts\debug\run_q_group_peak_transport_validation.py tests\test_peak_transport_validation.py`:
  passed.
- Acceptance: `python .\scripts\debug\run_q_group_peak_transport_validation.py --state "C:\Users\Kenpo\.local\share\ra_sim\new4.json" --group-key "q_group,primary,1,5" --params theta_initial,corto_detector --metric all --outdir ".\artifacts\q_group_peak_sensitivity\new4_transport_validation"`.
  Baseline branch count was 2. Same-ROI transport used no full recompute,
  refinement, direct simulation fallback, sparse source-row fallback, or
  `integrate2d`. `theta_initial` and `corto_detector` both remain blocked for
  image-ROI COM by baseline transport mismatch; ray-cloud COM remains blocked
  by insufficient points.

## Links

- Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
