# Parratt Low-Q Stitch

Status: complete
Type: feature
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-27

## Summary

Added an off-by-default simulation option for finite-stack HT `00L` rods that
uses a scaled Parratt air/material/air slab curve near the total-reflection
region and stitches back to the existing Hendricks-Teller rod intensity.

## Current State

- GUI control: `Parratt low-Q stitch (00L)` in the simulation
  sampling/optics controls.
- Scope: active only for finite-stack HT curves with `(H,K)=(0,0)`.
- Material inputs: active CIF-derived optical constants, active wavelength,
  active lattice `c`, and stack thickness from `stack_layers * c`.
- Stitch policy: low-Q uses scaled Parratt reflectivity, the overlap blends
  back to HT intensity, and high-Q HT intensity remains unchanged.
- Saved state: the checkbox persists through the existing GUI-state variable
  path. Older states load with the feature disabled.
- Review follow-up: skipped the disabled-option copy path, preserved high-Q
  zero/negative HT values, and moved structure-model coverage out of QR
  grouping tests.

## Migration And Deprecation

No migration or deprecation is required. There are no new dependencies, CLI
flags, config keys, artifact fields, or saved-state schema changes. Integer
Bragg debug mode, non-`00L` rods, and default simulations retain existing
behavior.

## Shipping Status

Ready for review as a default-off feature. Rollout is controlled by the GUI
checkbox; rollback is a normal git revert of the feature commit. No CI workflow
or release-version change is required.

## Validation

- `python -m pytest tests/test_diffraction_constraints.py tests/test_gui_structure_model.py tests/test_qr_grouping.py tests/test_gui_views.py tests/test_gui_sim_signature.py tests/test_gui_state_io.py -k "parratt or sampling_optics_controls" -ra`
- `python -m ra_sim.dev check`

Both pass locally.

## Links

- User workflow note: `docs/gui-workflow.md`
- Validation status: `docs/testing-and-validation.md`
