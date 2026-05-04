# Numba import fallback

Status: fixed
Type: bug
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-04

## Summary

RA-SIM no longer hard-fails while importing core simulation modules when Numba
is missing or when Numba itself fails during import. This covers dependency
mismatches such as Numba/Coverage import failures before any simulation code can
select a Python path.

## Root cause

Production modules imported Numba directly at module import time. The most
visible path was `ra_sim/simulation/diffraction.py`, which imported `njit`,
`prange`, `types`, and `numba.typed.List` before fallback runtime behavior could
run.

## Fix

Implemented in `ra_sim/utils/numba_compat.py`:

- real Numba is used unchanged when import succeeds;
- fallback `njit` supports direct, empty-call, keyword, signature, and
  call-returned decorator forms while preserving `.py_func`;
- fallback `prange` behaves like `range`;
- fallback `List` supports both `List()` and `List.empty_list(...)`;
- fallback `types` tolerates import-time signature expressions such as
  `types.float64[:, ::1]`, `types.Tuple(...)`, `types.UniTuple(...)`,
  `types.ListType(...)`, and `types.Array(...)`;
- `NUMBA_IMPORT_ERROR` preserves the original import failure for diagnostics.

All production Numba imports now route through the compatibility layer. The
exact-cake engine contract remains strict: `engine="numba"` still raises when
Numba is unavailable, while `engine="auto"` falls back to Python and
`engine="python"` stays on Python. `ra-sim-dev doctor` reports passive Numba
availability and import-error details without printing warnings during normal
imports.

## Status

- Bug status: fixed.
- Error status: fixed for import-time Numba failures in the known direct-import
  modules.
- Feature/API status: no public CLI/config/saved-state schema change; new
  internal compatibility module only.
- Performance status: real Numba fast paths are preserved when Numba imports
  successfully.
- Compatibility status: no dependency pins, algorithms, simulation parameters,
  weighted-event memory policy, fitting objective, or Qr resolver behavior were
  changed.

## Validation

Passed:

```powershell
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_numba_compat.py
python -m pytest -q tests/test_diffraction_weighted_events.py tests/test_diffraction_inner_loop_optimizations.py tests/test_ctr_fast_attenuation.py
python -m pytest -q tests/test_parallel_utils.py tests/test_exact_cake_portable.py
python -m ruff check .
python -c "import ra_sim.simulation.diffraction; import ra_sim.simulation.exact_cake; import ra_sim.StructureFactor.StructureFactor; import ra_sim.utils.calculations; import ra_sim.utils.parallel; print('ok')"
rg -n "from numba|import numba" ra_sim
git diff --check
```

Results:

- `tests/test_numba_compat.py`: 10 passed.
- targeted diffraction regression trio: 98 passed.
- parallel/exact-cake regression slice: 48 passed.
- production direct-import grep only reports `ra_sim/utils/numba_compat.py`.
