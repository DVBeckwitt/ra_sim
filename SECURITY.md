# Security Policy

## Supported branch

Security fixes land on the default branch. Report issues against the latest
`main` branch state before assuming an older checkout is supported.

## Reporting a vulnerability

Please do not open a public issue for suspected vulnerabilities, secrets, or
private data exposure.

Use GitHub Security Advisories for private reporting when available. If private
advisory reporting is unavailable for your environment, contact the repository
maintainers directly and wait for triage before public disclosure.

Include:

- affected commit, branch, or release
- reproduction steps
- impact assessment
- whether any credential, local path, or experiment data was exposed

## Repository-specific handling

- Keep machine-local overrides in ignored `config/file_paths.yaml` and
  `config/hbn_paths.yaml`.
- Commit only example templates under `config/*.example.yaml`.
- Do not commit raw detector data, private experiment bundles, API tokens,
  cloud credentials, or local absolute paths.
- Pull requests should pass CI and security workflows before merge.
