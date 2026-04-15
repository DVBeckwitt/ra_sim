# Debug and cache guide

This page is the short operational guide for logging, debug output, and cache
retention. Use it when you need to understand where RA-SIM writes diagnostics
or why a debug path is noisy or silent.

See also:

- [docs index](index.md)
- [Troubleshooting guide](troubleshooting.md)
- [Canonical logging section](simulation_and_fitting.md#logging-debug-and-cache-controls)

## Main Control Surface

RA-SIM uses `config/debug.yaml` as the primary debug and logging configuration.

Important points:

- `debug.global.disable_all: true` is the master kill switch for user-facing debug/log output
- `debug.console.enabled` controls console debug printing
- geometry-fit and mosaic-fit log file creation have dedicated toggles
- optional retained caches are controlled separately from logging output

## Output Locations

The default output directories come from `config/dir_paths.yaml`.

Common outputs:

- runtime trace logs
- geometry-fit logs
- mosaic-shape fit logs
- projection-debug JSON
- diffraction debug CSV
- intersection-cache dump folders

## Cache Policy

The cache policy only affects optional retained caches. It does not turn off the
active runtime state needed for the current simulation or GUI session.

Retention modes:

- `never`: build on demand, then discard
- `auto`: retain when the active feature benefits from reuse
- `always`: retain whenever built

## Numba on-disk compilation cache

RA-SIM sets `NUMBA_CACHE_DIR` at package import time for stable startup behavior.
When `NUMBA_CACHE_DIR` is unset, RA-SIM defaults to `~/.cache/ra_sim/numba`.
If a value is already set, RA-SIM leaves it unchanged.

RA-SIM does not force a Numba CPU mode. `NUMBA_CPU_NAME=generic` is not
enabled by default.

To inspect cache activity:

1. `set NUMBA_DEBUG_CACHE=1` before launch (or `export NUMBA_DEBUG_CACHE=1` on
   macOS/Linux)
2. run one simulation (`python -m ra_sim simulate ...` or `ra-sim simulate ...`)
3. watch console output for cache write/hit events from the Numba cache loader

Known limitations:

- first import/first run of a new kernel still compiles once before reuse
- cache hit behavior can vary across hosts, Python versions, and Numba releases
- if the default path is not writable, RA-SIM keeps startup best-effort and may not use on-disk caching

Manual verification recipe:

1. remove `~/.cache/ra_sim/numba` (or platform equivalent configured in `NUMBA_CACHE_DIR`)
2. run one simulation with `NUMBA_DEBUG_CACHE=1`
3. confirm new cache artifacts appear in `NUMBA_CACHE_DIR`
4. rerun same simulation with `NUMBA_DEBUG_CACHE=1` and confirm cache read hits in output

## Developer Tool Caches

When you run RA-SIM development commands from the repository, tool caches stay
under the user cache root instead of the worktree.

- Python bytecode: `~/.cache/ra_sim/dev/pycache`
- `mypy`: `~/.cache/ra_sim/dev/mypy`
- `pytest`: `~/.cache/ra_sim/dev/pytest`
- `ruff`: `~/.cache/ra_sim/dev/ruff`

The tool-specific `mypy`/`pytest`/`ruff` cache dirs apply to both
`python -m ra_sim.dev ...` and direct `pytest`/`mypy`/`ruff` runs from the
repository root. Python bytecode redirection applies when the repo
`sitecustomize.py` is importable, which is guaranteed for `ra_sim.dev` and
`python -m ...` launches from the repository root.

Existing repo-local cache folders are not migrated or removed automatically.
It is safe to delete stale `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, and
`__pycache__/` directories manually when they are no longer needed.

## Good Debug Hygiene

- Prefer config-based toggles over ad hoc path edits.
- Keep machine-local output paths in ignored local config, not versioned files.
- If you add a new debug artifact, document its toggle and output location.
- When sharing logs or screenshots, avoid leaking local absolute paths or private data.

For the exact key-by-key reference and current defaults, use the canonical doc:

- [Logging, debug, and cache controls](simulation_and_fitting.md#logging-debug-and-cache-controls)
