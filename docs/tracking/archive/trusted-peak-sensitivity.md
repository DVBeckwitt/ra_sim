# Trusted Qr/Qz Peak Sensitivity

Status: completed
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-04-22

## Summary

Qr/Qz peak sensitivity produced finite-difference numbers, but all rows were
marked `identity_changed` because perturbed evaluations dropped trusted
reflection provenance. The sensitivity matrix existed but could not be treated
as trusted.

## Current state

Implemented baseline-provenance anchoring in the peak sensitivity evaluator.
Perturbed rows now restore `source_reflection_index`,
`source_reflection_namespace`, and `source_reflection_is_full` only when stable
identity fields match exactly: group key, HKL, source table index, source row
index, source branch index, and source peak index.

Trust restoration now rejects missing group/HKL, non-`full_reflection`
namespaces, non-full reflections, and mismatched stable identity fields. Rows
that cannot prove identity still remain `identity_changed`.

## Next actions

No implementation follow-up required for this bug. The remaining broad
`python -m ra_sim.dev check` failure is unrelated pre-existing formatting drift
outside this change.

## Validation

- `python -m pytest tests\test_peak_sensitivity.py -q`: 25 passed.
- `python -m ruff format --check ra_sim\gui\peak_sensitivity.py tests\test_peak_sensitivity.py`: passed.
- `python -m ruff check ra_sim\gui\peak_sensitivity.py tests\test_peak_sensitivity.py`: passed.
- `python .\scripts\debug\run_q_group_peak_sensitivity.py --state "C:\Users\Kenpo\.local\share\ra_sim\new4.json" --group-key "q_group,primary,1,5" --outdir ".\artifacts\q_group_peak_sensitivity\new4_local_run_after_fix"`: completed with 2 baseline peaks, 44 `ok` sensitivity rows, 0 non-ok rows, `baseline_coordinate_source = runtime_evaluator`, no sparse-image fallback, and no direct-simulation fallback.

## Links

- Issue: none
- Output: `artifacts/q_group_peak_sensitivity/new4_local_run_after_fix/`
