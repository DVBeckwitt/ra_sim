# RA-SIM troubleshooting

This page collects common setup and workflow failures that can stall new
contributors or coding agents.

See also:

- [docs index](index.md)
- [GUI workflow guide](gui-workflow.md)
- [Debug and cache guide](debug-and-cache.md)
- [README configuration section](../README.md#configuration)

## Tkinter Missing

Symptom:

- `python -m ra_sim gui` or `python -m ra_sim calibrant` fails because Tkinter is unavailable

Fix:

- Install the system Tk package for your Python version, commonly `python3-tk` or `python3.11-tk` on Linux.

## Local Path Config Missing

Symptom:

- startup or CLI commands fail because detector images, `.poni`, CIFs, or measured peaks are not configured

Fix:

1. Copy `config/file_paths.example.yaml` to `config/file_paths.yaml`.
2. Copy `config/hbn_paths.example.yaml` to `config/hbn_paths.yaml` if you use the hBN CLI flow.
3. Update the local files for your machine and experiment.
4. Keep those overrides untracked.

## Config Outside The Repo

Symptom:

- you need different config per machine or dataset

Fix:

- Put the config files in another directory and set `RA_SIM_CONFIG_DIR` to that folder.

## hBN Bundle Or Paths Confusion

Symptom:

- `python -m ra_sim hbn-fit --load-bundle` cannot find a bundle
- startup tilt hint does not resolve

Fix:

- Pass `--load-bundle /path/to/bundle.npz`, or
- set `bundle` in local `config/hbn_paths.yaml`, or
- pass `--paths-file /path/to/custom_hbn_paths.yaml`

## Security Workflow Fails On Local Paths

Symptom:

- CI rejects a pull request because a tracked file contains `/Users/`, `/home/`, or `C:\Users\`

Fix:

- move machine-local values into ignored local config
- keep only portable examples in `config/*.example.yaml`
- scrub local absolute paths from docs or tracked JSON/YAML before pushing

## Need More Detail

Use these deeper references:

- [Canonical reference](simulation_and_fitting.md)
- [Debug and cache guide](debug-and-cache.md)
- [Contributing guide](../CONTRIBUTING.md)
