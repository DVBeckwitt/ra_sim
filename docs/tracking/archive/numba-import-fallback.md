# Numba import fallback

Status: fixed
Type: bug
Owner: -
Issue: none
Priority: p2
Last updated: 2026-05-23

## Summary

RA-SIM no longer hard-fails while importing core simulation modules when Numba
is missing or when Numba itself fails during import. This covers dependency
mismatches such as Numba/Coverage import failures before any simulation code can
select a Python path. Follow-up cleanup also handles `NUMBA_DISABLE_JIT=1`,
where Numba imports successfully but returns raw Python functions from `@njit`.

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

Disabled-JIT cleanup in `ra_sim/utils/numba_compat.py`:

- added `NUMBA_JIT_DISABLED` and `NUMBA_COMPILATION_AVAILABLE`;
- wrapped real `numba.njit` so raw Python functions returned under
  `NUMBA_DISABLE_JIT=1` get `.py_func`;
- preserved real Numba dispatcher behavior when JIT is enabled;
- kept the blocked-import fallback behavior unchanged.

Disabled-JIT cleanup in `ra_sim/simulation/exact_cake.py`:

- `engine="auto"` falls back to Python when compilation is unavailable;
- `engine="numba"` raises a clear JIT-disabled error when Numba imported but
  JIT is disabled;
- import failures and disabled-JIT cases report distinct errors.

Startup exact-cake warmup diagnostic follow-up:

- the optional background exact-cake Numba warmup now reports failures as
  fallback diagnostics instead of implying startup or simulation failure;
- the reported compiler signature
  `'Loc' object does not support the context manager protocol` is covered by a
  regression test;
- `engine="auto"` fallback, explicit `engine="numba"` strictness, public
  CLI/config/saved-state/artifact interfaces, and Numba fast paths are
  unchanged.

## Status

- Bug status: fixed.
- Error status: fixed for import-time Numba failures in the known direct-import
  modules.
- Feature/API status: no public CLI/config/saved-state schema change; new
  internal compatibility module only.
- Performance status: real Numba fast paths are preserved when Numba imports
  successfully.
- Startup warmup status: fixed. Optional exact-cake Numba warmup failures are
  latched once per process and logged as Python-fallback diagnostics.
- Disabled-JIT status: fixed. Decorated diffraction functions keep `.py_func`
  when `NUMBA_DISABLE_JIT=1`, and exact-cake no longer selects the Numba engine
  automatically when compilation is disabled.
- Compatibility status: no dependency pins, algorithms, simulation parameters,
  weighted-event memory policy, fitting objective, or Qr resolver behavior were
  changed.
- Migration/deprecation status: no migration, deprecation, feature flag, CI
  workflow, or release version change was required.

## Validation

Passed:

```powershell
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_numba_compat.py
python -m pytest -q tests/test_diffraction_weighted_events.py tests/test_diffraction_inner_loop_optimizations.py
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

2026-05-04 disabled-JIT cleanup validation:

```powershell
python -m compileall -q ra_sim tests
python -m pytest -q tests/test_numba_compat.py
$env:NUMBA_DISABLE_JIT='1'; python -m pytest -q tests/test_diffraction_weighted_events.py tests/test_diffraction_inner_loop_optimizations.py
$env:NUMBA_DISABLE_JIT='1'; python -m pytest -q tests/test_parallel_utils.py
python -m pytest -q tests/test_diffraction_weighted_events.py tests/test_diffraction_inner_loop_optimizations.py
python -m pytest -q tests/test_exact_cake_portable.py tests/test_parallel_utils.py
python -m ruff check ra_sim/utils/numba_compat.py ra_sim/simulation/exact_cake.py ra_sim/dev_doctor.py tests/test_numba_compat.py
rg -n "from numba|import numba" ra_sim tests
git diff --check -- ra_sim/utils/numba_compat.py ra_sim/simulation/exact_cake.py ra_sim/dev_doctor.py tests/test_numba_compat.py
```

Results:

- `tests/test_numba_compat.py`: 13 passed.
- disabled-JIT diffraction regression trio: 98 passed; no
  `AttributeError: 'function' object has no attribute 'py_func'`.
- disabled-JIT parallel utility tests: 4 passed.
- normal diffraction regression trio: 98 passed.
- normal exact-cake/parallel regression slice: 48 passed.
- disabled-JIT exact-cake full slice still has 3 expected strict-engine
  conflicts in tests that directly require Numba LUT/`engine="numba"` while the
  current contract requires strict `"numba"` requests to fail when JIT is
  disabled.

2026-05-23 startup warmup diagnostic validation:

```powershell
python -m pytest tests/test_exact_cake_portable.py -k "safe_numba_warmup" -ra
python -m pytest tests/test_exact_cake_portable.py -ra
python -m ra_sim.dev check
git diff --check
```

Results:

- exact-cake safe warmup slice: 2 passed.
- `tests/test_exact_cake_portable.py`: 44 passed.
- `python -m ra_sim.dev check`: formatting, ruff, fast tests, and mypy passed
  with 294 fast tests.
