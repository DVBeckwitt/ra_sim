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

## Next actions

- Review the acceptance artifacts and decide whether to use image-ROI COM as the
  practical fallback for sparse ray clouds on `new4.json`.

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

## Links

- Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
