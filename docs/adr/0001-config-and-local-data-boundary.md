# 0001: Config and local data boundary

Status: Accepted

Date: 2026-04-21

## Context

RA-SIM depends on experiment images, detector geometry, material files, measured
peaks, and output locations that are machine-local. The repository tracks
portable examples under `config/`, while `config/file_paths.yaml` and
`config/hbn_paths.yaml` are ignored local overrides.

The config loader resolves from `RA_SIM_CONFIG_DIR` when set, otherwise from the
repository `config/` directory. Relative file paths resolve from the repo root
for repo config and from the external config directory for `RA_SIM_CONFIG_DIR`.

## Decision

Local experiment paths stay in ignored local config files or in an external
`RA_SIM_CONFIG_DIR`. Portable example config stays tracked as the template for
new machines and agents.

## Consequences

Agents can inspect config shape without needing private data. Startup or
workflow commands may still warn or fail when local experiment files are absent.
Docs and examples should avoid private absolute paths and raw experiment data.

## Related docs/tests

- [Configuration](../../README.md#configuration)
- [Troubleshooting](../troubleshooting.md)
- [Debug and cache guide](../debug-and-cache.md)
- `tests/test_config_loader.py`
- `tests/test_hbn_path_resolution.py`
