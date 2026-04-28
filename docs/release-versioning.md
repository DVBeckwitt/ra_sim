# RA-SIM release versioning

RA-SIM uses PEP 440 compatible semantic versioning for package releases.
The package version source of truth is:

```text
pyproject.toml
```

Git tags use a leading `v`, for example `v1.0.0`.

## 1.0 release sequence

Use this sequence for the path to the first stable release:

| Version | Meaning |
| --- | --- |
| `1.0.0.dev0` | Current active development toward version 1. |
| `1.0.0a1` | First internal alpha once major planned features are present but rough. |
| `1.0.0b1` | Beta once the feature set is mostly complete and workflows are testable. |
| `1.0.0rc1` | Release candidate once feature-complete and only release blockers remain. |
| `1.0.0` | First stable release. |

## Current status

As of 2026-04-28:

- Feature status: implemented. `pyproject.toml` is set to `1.0.0.dev0`, the main simulation GUI window title includes the resolved package version, and the Help tab Project section shows the same version.
- Documentation status: implemented. This page records the 1.0 release sequence and `CHANGELOG.md` records the user-facing version-display work under Unreleased.
- Validation status: `python -m pytest tests/test_gui_views.py -ra` and `python -m compileall ra_sim tests` passed.
- Open error status: `python -m pytest tests/test_gui_runtime_import_safe.py -ra` still fails in the existing dirty runtime-update worktree on update-prune/startup-overlay assertions; that failure is separate from the version-display feature.

## Rules

- Update `pyproject.toml` for every committed package version change.
- Update `CHANGELOG.md` for release-facing changes.
- Use `.devN`, `aN`, `bN`, `rcN`, and final versions exactly as PEP 440 forms.
- Do not commit local versions with `+metadata` to `pyproject.toml`.
- Use dates in changelog headings, release notes, and artifact manifests, not package versions.
