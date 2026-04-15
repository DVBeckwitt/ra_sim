"""Cross-platform developer entrypoints for bootstrap, checks, and locking."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from ra_sim.test_tiers import fast_test_paths, integration_test_paths
from ra_sim.user_paths import dev_cache_dir

FAST_MARKER = "fast"
INTEGRATION_MARKER = "integration"
LOCKFILE_NAME = "pylock.toml"
MYPY_FRONTIER_ARGS = ["--follow-imports=silent"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def typecheck_targets() -> list[str]:
    return [
        "ra_sim/config/__init__.py",
        "ra_sim/config/loader.py",
        "ra_sim/config/models.py",
        "ra_sim/config/validation.py",
        "ra_sim/dev.py",
        "ra_sim/fitting/optimization_mosaic_profiles.py",
        "ra_sim/fitting/optimization_runtime.py",
        "ra_sim/gui/_runtime/live_cache_helpers.py",
        "ra_sim/gui/_runtime/primary_cache_helpers.py",
        "ra_sim/gui/runtime_primary_cache.py",
        "ra_sim/test_tiers.py",
    ]


def format_targets() -> list[str]:
    return sorted(
        {
            *typecheck_targets(),
            "ra_sim/fitting/optimization.py",
            "ra_sim/gui/_runtime/runtime_session.py",
            "tests/test_dev_cli.py",
            "tests/test_gui_runtime_primary_cache.py",
            "tests/test_mosaic_shape_optimization.py",
        }
    )


def supports_pip_dependency_groups(help_text: str) -> bool:
    return "--group <[path:]group>" in help_text


def _python_module_command(*args: str) -> list[str]:
    return [sys.executable, *args]


def _subprocess_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    cache_defaults = {
        "PYTHONPYCACHEPREFIX": dev_cache_dir("pycache"),
        "MYPY_CACHE_DIR": dev_cache_dir("mypy"),
        "RUFF_CACHE_DIR": dev_cache_dir("ruff"),
    }
    for key, path in cache_defaults.items():
        if key in env:
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        env[key] = str(path)
    return env


def _run(command: Sequence[str], *, cwd: Path) -> int:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False, env=_subprocess_env())
    return int(completed.returncode)


def _run_all(commands: Sequence[Sequence[str]], *, cwd: Path) -> int:
    for command in commands:
        exit_code = _run(command, cwd=cwd)
        if exit_code != 0:
            return exit_code
    return 0


def _run_first_success(command_variants: Sequence[Sequence[str]], *, cwd: Path) -> int:
    last_exit_code = 1
    for command in command_variants:
        exit_code = _run(command, cwd=cwd)
        if exit_code == 0:
            return 0
        last_exit_code = exit_code
    return last_exit_code


def pip_install_command_variants() -> list[list[str]]:
    return [
        _python_module_command("-m", "pip", "install", "--group", "dev", "-e", "."),
        _python_module_command("-m", "pip", "install", "-e", ".[dev]"),
    ]


def pip_lock_command_variants(*, output: str) -> list[list[str]]:
    return [
        _python_module_command("-m", "pip", "lock", "-e", ".", "--group", "dev", "-o", output),
        _python_module_command("-m", "pip", "lock", "-e", ".[dev]", "-o", output),
    ]


def pre_commit_install_command() -> list[str]:
    return _python_module_command("-m", "pre_commit", "install")


def pytest_command_for_tier(tier: str) -> list[str]:
    root = repo_root()
    if tier == "fast":
        return _python_module_command(
            "-m",
            "pytest",
            "-q",
            *fast_test_paths(root),
            "-m",
            FAST_MARKER,
        )
    return _python_module_command(
        "-m",
        "pytest",
        "-q",
        *integration_test_paths(root),
        "-m",
        INTEGRATION_MARKER,
    )


def bootstrap() -> int:
    cwd = repo_root()
    exit_code = _run_all(
        [
            _python_module_command("-m", "pip", "install", "--upgrade", "pip"),
        ],
        cwd=cwd,
    )
    if exit_code != 0:
        return exit_code
    return _run_first_success(pip_install_command_variants(), cwd=cwd)


def lock(*, output: str) -> int:
    cwd = repo_root()
    return _run_first_success(
        pip_lock_command_variants(output=output),
        cwd=cwd,
    )


def lint() -> int:
    return _run(
        _python_module_command("-m", "ruff", "check", "."),
        cwd=repo_root(),
    )


def format_code() -> int:
    return _run(
        _python_module_command("-m", "ruff", "format", *format_targets()),
        cwd=repo_root(),
    )


def format_check() -> int:
    return _run(
        _python_module_command("-m", "ruff", "format", "--check", *format_targets()),
        cwd=repo_root(),
    )


def typecheck() -> int:
    return _run(
        _python_module_command("-m", "mypy", *MYPY_FRONTIER_ARGS, *typecheck_targets()),
        cwd=repo_root(),
    )


def test_fast() -> int:
    return _run(pytest_command_for_tier("fast"), cwd=repo_root())


def test_integration() -> int:
    return _run(pytest_command_for_tier("integration"), cwd=repo_root())


def test_all() -> int:
    return _run(
        _python_module_command("-m", "pytest", "-q"),
        cwd=repo_root(),
    )


def install_hooks() -> int:
    return _run(pre_commit_install_command(), cwd=repo_root())


def check() -> int:
    return _run_all(
        [
            _python_module_command("-m", "ruff", "format", "--check", *format_targets()),
            _python_module_command("-m", "ruff", "check", "."),
            *[pytest_command_for_tier("fast")],
            _python_module_command("-m", "mypy", *MYPY_FRONTIER_ARGS, *typecheck_targets()),
        ],
        cwd=repo_root(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap", help="Install editable package with dev tooling.")

    lock_parser = subparsers.add_parser("lock", help=f"Refresh {LOCKFILE_NAME}.")
    lock_parser.add_argument(
        "--output",
        default=LOCKFILE_NAME,
        help="Lockfile path to write.",
    )

    subparsers.add_parser("format", help="Format the current formatter frontier.")
    subparsers.add_parser("format-check", help="Check formatting on the formatter frontier.")
    subparsers.add_parser("hooks", help="Install local pre-commit hooks.")
    subparsers.add_parser("lint", help="Run ruff.")
    subparsers.add_parser("typecheck", help="Run current mypy frontier.")
    subparsers.add_parser("test-fast", help="Run fast pytest tier.")
    subparsers.add_parser("test-integration", help="Run integration pytest tier.")
    subparsers.add_parser("test-all", help="Run full pytest suite.")
    subparsers.add_parser("check", help="Run lint + fast tests + typecheck.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "bootstrap":
        return bootstrap()
    if args.command == "lock":
        return lock(output=str(args.output))
    if args.command == "format":
        return format_code()
    if args.command == "format-check":
        return format_check()
    if args.command == "hooks":
        return install_hooks()
    if args.command == "lint":
        return lint()
    if args.command == "typecheck":
        return typecheck()
    if args.command == "test-fast":
        return test_fast()
    if args.command == "test-integration":
        return test_integration()
    if args.command == "test-all":
        return test_all()
    if args.command == "check":
        return check()
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
