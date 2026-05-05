# GUI caked detector-count density

Status: fixed
Type: bug
Owner: -
Issue: none
Priority: p1
Last updated: 2026-05-05

## Summary

The main GUI `phi x 2theta` caked view was applying solid-angle correction
during detector-to-cake conversion. Flat detector-count backgrounds could ramp
upward at high `2theta` because density mode divided by solid-angle
normalization instead of detector pixel support.

## Current state

Fixed. GUI caked density now means:

```text
sum_signal / detector-pixel support normalization
```

The policy lives in `ra_sim/gui/caked_intensity_policy.py` as
`GUI_CAKED_VIEW_CORRECT_SOLID_ANGLE = False`. Runtime caking, geometry-fit
caked payloads, peak-sensitivity caked payloads, simulation/background cached
caking, and caked 1D profiles now share that convention. Cache signatures
include the resolved solid-angle policy.

Raw-sum mode still uses raw accumulated `sum_signal`. Q-space conversion and
fitting/background/peak-subtraction logic were not changed.

## Validation

- `python -m pytest tests/test_gui_runtime_import_safe.py -q`: 397 passed.
- `python -m pytest tests/test_exact_cake_portable.py -q`: 44 passed.
- `rg -n 'correctSolidAngle\s*[:=]\s*True' ra_sim/gui`: no hits.

## Links

- `README.md` detector-to-angle backend caked intensity convention.
- `CHANGELOG.md` Unreleased GUI and UX update entry.
