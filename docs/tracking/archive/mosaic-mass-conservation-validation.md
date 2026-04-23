# Mosaic Mass Conservation Validation

Status: completed
Type: feature
Owner:
Issue: none
Priority: p1
Last updated: 2026-04-22

## Summary

Mosaic smearing now has a validation-facing normalization layer before detector
projection. The layer validates conservation of per-reflection mosaic mass with
synthetic reflection strengths only; it does not use VESTA TXT intensity values
because the Bi2Se3 reference intensity column is `NaN`.

The implementation adds `ra_sim/simulation/mosaic_normalization.py` and focused
coverage in `tests/test_mosaic_intensity_conservation.py`. The helper builds a
pre-detector mosaic distribution for one HKL, preserves `raw_mass`, normalizes
integrated quadrature weights only when raw mass is finite and positive, and
reports explicit statuses:

- `ok`
- `invalid_mass_zero`
- `invalid_mass_negative`
- `invalid_mass_nan`
- `invalid_mass_inf`

## Current state

The validation layer treats `solve_q(...)[0][:, 3]` as integrated quadrature
weight, not density. Tests keep a source-proof check for this convention:
uniform full-circle source stores `sigma_arr * ds`, and the adaptive source path
stores Simpson interval mass. In `integration_mode == "weights"`,
`integrate_mosaic_mass(...)` is exactly `sum(result.weights)`; no area or
Jacobian multiplication is applied.

The helper intentionally stops before detector projection, masks, clipping,
pixel binning, and final image scaling. It validates the mosaic-smearing layer
only.

The helper currently mirrors the uniform full-circle quadrature math in pure
Python because the local direct Numba/JIT path can crash. Drift risk is covered
by `test_mosaic_normalization_matches_solve_q_source_path`, which calls
`solve_q.py_func` with Python-source globals and compares sample count,
`qx/qy/qz`, integrated weights, raw mass, and normalized mass.

Branch IDs are a current-frame validation detail. They assume
`G_vec=[0, Gr, Gz]`; non-specular branches split on `qx` with an epsilon rule,
and near-zero `qx` samples are assigned deterministically to branch `0`.

Debug artifacts are quiet during normal test runs. They write under
`artifacts/mosaic_normalization/` only when
`RA_SIM_WRITE_MOSAIC_NORMALIZATION_DEBUG=1` or inside assertion-failure handling
before re-raising.

## Next actions

None for this validation layer. If production quadrature changes, update the
source-equivalence test first so the validation helper cannot silently drift.

A stronger future cleanup would refactor the shared pure quadrature core so the
JIT wrapper and validation helper call the same function directly.

## Validation

- `pytest tests/test_mosaic_intensity_conservation.py -q` passed: 14 passed.
- `pytest tests/test_bi2se3_vesta_reference_regression.py -q` passed: 5 passed.
- `pytest tests/test_raw_structure_factor_api.py -q` passed: 6 passed.
- `pytest tests/test_structure_factor_sites.py -q` passed: 5 passed.
- `pytest tests/test_bi2se3_vesta_geometry.py -q` passed: 3 passed.
- `ruff check` passed.
- `ruff format --check ra_sim/simulation/mosaic_normalization.py tests/test_mosaic_intensity_conservation.py` passed.
- `ruff format --check` failed on 144 pre-existing unrelated files; touched files
  were not listed.
- `python -X faulthandler -m pytest -q -x --tb=short -ra` failed normally, not
  by timeout, OOM, or LLVM/JIT crash. First failure was the pre-existing
  `tests/test_gui_geometry_fit_workflow.py::test_coordinate_diagnostic_optimizer_request_capture_failure_is_incomplete_not_frame_mismatch`
  assertion at line 2828 after 793 passed and 3 skipped.

## Links

- Issue: none
- Helper: `ra_sim/simulation/mosaic_normalization.py`
- Tests: `tests/test_mosaic_intensity_conservation.py`
