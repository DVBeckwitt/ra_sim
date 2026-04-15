# Contributing to RA-SIM

This repository is research software, but contributions should still be easy to
review, reproduce, and validate. Keep changes narrow, preserve existing
behavior unless the task explicitly changes it, and prefer simplifying existing
code over layering on new paths.

## Development Setup

RA-SIM supports Python 3.11+.

```bash
git clone https://github.com/DVBeckwitt/ra_sim.git
cd ra_sim
python -m ra_sim.dev bootstrap
```

Manual fallback:

```bash
python -m pip install --group dev -e .
# Older pip fallback:
# python -m pip install -e ".[dev]"
```

The GUI depends on Tkinter. Windows and macOS Python distributions usually
include it already. Some Linux environments require an extra system package
such as `python3-tk` or `python3.11-tk`.

Developer-tool caches now default to `~/.cache/ra_sim/dev/` so local `mypy`,
`pytest`, and `ruff` cache output stays out of the repo root. Python bytecode
redirection uses the same tree when the repo `sitecustomize.py` is importable,
including `ra_sim.dev` and `python -m ...` launches from the repo root. Older
repo-local cache folders are not migrated automatically.

## Local Configuration And Data

The repository does not bundle raw detector data or private experiment files.
Use the example templates, then keep your machine-local overrides untracked.

```bash
cp config/file_paths.example.yaml config/file_paths.yaml
cp config/hbn_paths.example.yaml config/hbn_paths.yaml
```

Rules:

- Do not commit machine-local paths, raw data, generated bundles, or secrets.
- Treat `config/*.example.yaml` as the versioned source of truth.
- Keep local overrides in ignored `config/file_paths.yaml` and
  `config/hbn_paths.yaml`.
- If your workflow needs a new configurable path, add it to the example file
  and document it.

## Making Changes

- Keep the scope tight. Avoid drive-by refactors unless they directly simplify
  the area you are touching.
- Prefer changing or removing existing code over adding new abstractions.
- Preserve behavior unless the task explicitly calls for a behavior change.
- If behavior changes, update or add tests close to the affected area.
- Update docs when commands, config flows, or user-visible workflows change.

Helpful repo entry points:

- `ra_sim/` for simulation, fitting, GUI, CLI, and IO code
- `tests/` for regression coverage
- `config/` for versioned defaults and example path templates
- `docs/` for workflow, architecture, and debug guidance

## Validation

Run the standard checks before opening a pull request:

```bash
python -m ra_sim.dev format-check
python -m ra_sim.dev check
python -m ra_sim.dev test-integration
```

Tier details:

- `python -m ra_sim.dev format-check`: `ruff format --check` on the maintained
  formatter frontier.
- `python -m ra_sim.dev check`: formatter check, `ruff`, the `fast` pytest
  tier, and the current mypy frontier.
- `python -m ra_sim.dev test-integration`: slower workflow-heavy tests marked
  `integration`.
- `python -m ra_sim.dev test-all`: full suite when you need every pytest target.
- `python -m ra_sim.dev hooks`: install the local `pre-commit` hooks from
  `.pre-commit-config.yaml`.
- `python -m ra_sim.dev lock`: refresh `pylock.toml` after dependency changes.

Choose tests based on change type:

- Logic/config changes: targeted unit tests plus the full suite if practical.
- Fitting/simulation changes: update the nearest regression tests.
- GUI workflow changes: cover the helper/controller path or the headless flow
  that exercises the same logic.
- Docs-only changes: no code checks required, but keep links and commands
  accurate.

## Pull Requests

- Summarize what changed and why.
- Note any behavior changes or migration steps.
- List the checks you ran.
- Call out known gaps if a full validation run was not possible.
- Keep unrelated local edits out of the pull request.

Before merge, confirm:

- no machine-local absolute paths were added outside example templates
- no secrets, credentials, or private experiment data were committed
- docs were updated for workflow or config changes
- CI and security workflows are green

## Documentation Map

Start with these docs depending on task:

- [README.md](README.md) for setup and top-level usage
- [docs/index.md](docs/index.md) for docs navigation
- [docs/gui-workflow.md](docs/gui-workflow.md) for operator workflow
- [docs/architecture.md](docs/architecture.md) for package layout and data flow
- [docs/debug-and-cache.md](docs/debug-and-cache.md) for logging and cache policy
- [docs/troubleshooting.md](docs/troubleshooting.md) for common setup and config failures
- [docs/simulation_and_fitting.md](docs/simulation_and_fitting.md) for the full canonical reference
