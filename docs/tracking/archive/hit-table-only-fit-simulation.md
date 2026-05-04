# Hit-Table-Only Fit Simulation

Status: completed
Type: optimization
Owner:
Issue: none
Priority: p2
Last updated: 2026-05-04

## Summary

Geometry fitting paths that only consume hit tables or max positions no longer
allocate dense detector images. A shared helper now uses an intentional empty
`(0, 0)` simulation buffer, requests hit tables, and disables image
accumulation for those fit-only simulations.

## Current state

Feature status: implemented. Point matching, dynamic point matching, fixed
correspondence matching, `simulate_and_compare_hkl`, and full-beam polish now
share the hit-table-only simulation helper.

Bug/error status: fixed. The memory waste from allocating
`image_size x image_size` buffers in hit-table-only fitting paths is removed.
The locked-Qr point-only projection path still skips process calls and keeps
the empty buffer behavior so detector payloads can fall back to `image_size`
when shape is required.

Compatibility status: preserved. Image-consuming simulation paths still return
dense detector images and were not modified. The process wrapper strips
`accumulate_image` only inside the existing unexpected-keyword compatibility
branch for older monkeypatched process functions. The helper return annotation
now matches `hit_tables_to_max_positions(...)`, which returns an `np.ndarray`.

Regression status: clean. Targeted tests assert the empty buffer contract,
`collect_hit_tables=True`, `accumulate_image=False`, `prefer_python_runner`
forwarding, fallback stripping for older process-call signatures, and unchanged
numeric residual outputs for the targeted paths. CLI geometry-fit tests also
remain green.

Documentation status: complete. This archive note records the implemented
optimization and bug/error status; the tracking index lists it as completed.
The earlier validation concern about unrelated docs is resolved by keeping the
Numba fallback note in its separate commit.

## Next actions

None for this item. Future simulation-memory work should stay separate from
image-residual and image-cache paths unless the caller contract changes.

## Validation

- `python -m compileall -q ra_sim tests` passed.
- `python -m pytest -q tests/test_geometry_fitting.py` passed: 208 passed.
- `python -m pytest -q tests/test_cli_geometry_fit.py` passed: 24 passed.
- `rg -n "sim_buffer\s*=\s*np\.zeros\(\(image_size, image_size\)" ra_sim/fitting/optimization.py` returned no target-style dense buffer allocations.
- `rg -n "_process_peaks_parallel_safe\(" ra_sim/fitting/optimization.py` showed the shared hit-table helper plus existing image-consuming simulation paths.
- `ra_sim/fitting/optimization.py` was normalized back to CRLF-only line endings after the cleanup pass.

## Links

- Tracking hub: [docs/tracking/index.md](../index.md)
