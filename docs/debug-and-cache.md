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

## Good Debug Hygiene

- Prefer config-based toggles over ad hoc path edits.
- Keep machine-local output paths in ignored local config, not versioned files.
- If you add a new debug artifact, document its toggle and output location.
- When sharing logs or screenshots, avoid leaking local absolute paths or private data.

For the exact key-by-key reference and current defaults, use the canonical doc:

- [Logging, debug, and cache controls](simulation_and_fitting.md#logging-debug-and-cache-controls)
