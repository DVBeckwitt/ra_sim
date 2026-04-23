# Bi2Se3 VESTA Structure-Factor Validation

## Purpose

This validation checks that RA-SIM reproduces VESTA-exported X-ray structure factors for Bi2Se3 when both use the same CIF, the same VESTA reference export, and the same factor convention.

The validation compares raw complex structure factors `F(hkl)` and derived `|F|`. It does not compare powder intensity, multiplicity-weighted intensity, Lorentz-polarization-corrected intensity, profile intensity, or any scaled diffraction pattern quantity.

## Reference Configuration

The reference was generated in VESTA using:

- Material: Bi2Se3
- Radiation: X-ray
- Line: Cu Kalpha1
- d(min): 0.7 Å
- CIF occupancy state: Se1 occupancy = 1.0000
- Reference file: `tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.txt`

The VESTA TXT contains:

- 206 total HKL rows
- 161 finite `2theta` rows
- 45 rows with non-finite `2theta`
- inferred wavelength near 1.540592925 Å

The non-finite `2theta` rows are retained for `|F|` comparison. They are excluded only from `2theta` comparison because those reflections are beyond the Bragg condition for the wavelength.

## Validated Quantities

The validation checks:

- HKL row matching
- d-spacing parity
- finite-row `2theta` parity
- raw complex `F(hkl)` computation
- `|F|` parity
- CIF and TXT fixture hashes
- Se1 occupancy provenance
- VESTA factor convention
- negative control for wrong Se1 occupancy

## Factor Convention

VESTA parity requires:

- raw complex `F(hkl)`
- CIF occupancies
- Waasmaier-Kirfel `f0`
- legacy Cu Kalpha1 anomalous terms
- no multiplicity scaling
- no intensity conversion
- no Lorentz-polarization correction
- no powder-profile correction
- no fitted scale factor

The relevant RA-SIM convention is the VESTA Cu Kalpha1 legacy structure-factor mode.

## Root Cause of the Original Discrepancy

The original mismatch had two causes.

First, the reference CIF and simulation CIF were not equivalent. The COD-derived CIF used in one path had `Se1 occupancy = 1.0`, while the earlier VESTA reference was generated from a CIF with `Se1 occupancy = 0.9000`.

This affected `|F|` but not peak geometry. For example, the corrected VESTA reference with `Se1 occupancy = 1.0` gives the `(0, 0, 3)` reflection as `d = 9.545333 Å`, `2theta = 9.25746 deg`, and `|F| = 105.976`. The earlier reference had the same `d` and `2theta`, but `|F| = 96.3389`.

Second, exact VESTA parity required matching VESTA's factor convention rather than using a generic package default.

## Final Validation Result

Final comparison result:

| Quantity | Result |
|---|---:|
| rows | 206 |
| finite `2theta` rows | 161 |
| NaN `2theta` rows | 45 |
| Se1 occupancy | 1.0 |
| max `|delta d|` | 5.23e-7 Å |
| max `|delta 2theta|` | 1.82e-5 deg |
| median relative `|F|` error | 0.000370 |
| max relative `|F|` error | 0.000616 |

Acceptance thresholds:

| Quantity | Threshold |
|---|---:|
| max `|delta d|` | < 1e-5 Å |
| max `|delta 2theta|` | < 1e-4 deg |
| median relative `|F|` error | < 0.001 |
| max relative `|F|` error | < 0.002 |

The validation passes with substantial margin.

## Fixture Provenance

Current fixture hashes:

```text
CIF SHA256:
2a02dfceb6b302810229076479c54515382b3adcd558e3d9f874192dcd6b539f

TXT SHA256:
b2f7e2de27db6bfee7497ad2c8e0ebf622d5db14c5da4240b7f334d6de89d379
```

The validation includes a negative control that modifies a temporary CIF copy to set `Se1 occupancy = 0.9000`. That modified CIF must fail parity against the current VESTA TXT generated from `Se1 occupancy = 1.0`.

This negative control confirms that the validation detects the original fixture-consistency failure mode.

## How to Rerun

Run the focused validation suite:

```bash
pytest tests/test_vesta_reference_parser.py -q
pytest tests/test_structure_factor_environment.py -q
pytest tests/test_structure_factor_sites.py -q
pytest tests/test_raw_structure_factor_api.py -q
pytest tests/test_bi2se3_vesta_geometry.py -q
pytest tests/test_bi2se3_vesta_reference_regression.py -q
```

Run the comparison command:

```bash
python -m ra_sim.tools.compare_bi2se3_reference --json
```

Run lint:

```bash
ruff check
```

Optional debug artifacts can be written with:

```bash
RA_SIM_WRITE_BI2SE3_DEBUG=1 python -m ra_sim.tools.compare_bi2se3_reference --json
```

Normal passing runs should not write debug artifacts.

## Known Repository-Wide Gate Status

The focused Bi2Se3 validation suite passes.

At the time this validation was added:

- `pytest -q` was blocked by an unrelated GUI ladder failure in `tests/test_gui_geometry_fit_workflow.py`
- repo-wide `ruff format --check` failed on pre-existing unrelated files
- touched files passed formatting checks
- `ruff check` passed

These issues are unrelated to the Bi2Se3 VESTA validation.

## Scope and Limitations

This validation proves parity with the specified VESTA export for this Bi2Se3 fixture and this factor convention.

It does not prove:

- general equivalence to all VESTA settings
- equivalence to all radiation types
- equivalence to arbitrary anomalous-factor tables
- powder intensity parity
- profile-shape parity
- GUI workflow correctness

The validated target is raw structure-factor parity for this controlled VESTA reference case.
