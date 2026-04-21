"""Warning-first setup checks for RA-SIM developer environments."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from ra_sim import config
from ra_sim.config.loader import FILE_PATHS_EXAMPLE_FILENAME, FILE_PATHS_FILENAME
from ra_sim.hbn_geometry import resolve_hbn_paths

REQUIRED_FILE_INPUT_KEYS = {
    "simulation_dark_osc_file",
    "simulation_background_osc_files",
    "geometry_poni",
    "cif_file",
    "measured_peaks",
    "gui_geometry_poni",
    "gui_background_image",
}
REQUIRED_HBN_INPUT_KEYS = {"osc", "dark"}
DEV_TOOL_MODULES = (
    ("pytest", "pytest"),
    ("mypy", "mypy"),
    ("pre_commit", "pre-commit"),
    ("coverage", "coverage"),
    ("pytest_cov", "pytest-cov"),
    ("build", "build"),
)


@dataclass(frozen=True)
class DoctorFinding:
    level: str
    label: str
    detail: str
    strict_failure: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _display_level(finding: DoctorFinding, *, strict: bool) -> str:
    if not strict and finding.level == "FAIL":
        return "WARN"
    if finding.strict_failure:
        return "FAIL" if strict else "WARN"
    return finding.level


def _finding_line(finding: DoctorFinding, *, strict: bool) -> str:
    return f"[{_display_level(finding, strict=strict)}] {finding.label}: {finding.detail}"


def _config_file_status(config_dir: Path, primary_name: str, example_name: str) -> DoctorFinding:
    primary_path = config_dir / primary_name
    example_path = config_dir / example_name
    if primary_path.exists():
        return DoctorFinding("OK", primary_name, f"using local {primary_path}")
    if example_path.exists():
        return DoctorFinding("WARN", primary_name, f"using example fallback {example_path}")
    return DoctorFinding("WARN", primary_name, f"missing {primary_path} and {example_path}")


def _iter_path_strings(value: Any):
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_path_strings(item)


def _path_exists(path_text: str) -> bool:
    return Path(os.path.expanduser(path_text)).exists()


def _check_file_paths(config_dir: Path, explicit_local_config: bool) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    bundle = config.get_config_bundle(config_dir)
    for key in sorted(bundle.file_paths):
        try:
            resolved = config.get_path(key, config_dir=config_dir)
        except Exception as exc:
            findings.append(DoctorFinding("WARN", f"file path {key}", str(exc)))
            continue
        values = list(_iter_path_strings(resolved))
        if not values:
            findings.append(DoctorFinding("WARN", f"file path {key}", "empty or non-path value"))
            continue
        for value in values:
            exists = _path_exists(value)
            is_strict_failure = (
                explicit_local_config and key in REQUIRED_FILE_INPUT_KEYS and not exists
            )
            level = "OK" if exists else "WARN"
            findings.append(
                DoctorFinding(
                    level,
                    f"file path {key}",
                    f"{value} ({'exists' if exists else 'missing'})",
                    strict_failure=is_strict_failure,
                )
            )
    return findings


def _check_hbn_paths(
    config_dir: Path,
    explicit_local_config: bool,
    *,
    use_active_resolver: bool,
) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    paths_file = None
    if not use_active_resolver:
        for candidate_name in ("hbn_paths.yaml", "hbn_paths.example.yaml"):
            candidate = config_dir / candidate_name
            if candidate.exists():
                paths_file = str(candidate)
                break
    try:
        resolved = resolve_hbn_paths(paths_file=paths_file)
    except Exception as exc:
        return [DoctorFinding("WARN", "hbn paths", str(exc))]

    paths_file = resolved.get("paths_file")
    if paths_file:
        findings.append(DoctorFinding("OK", "hbn paths file", str(paths_file)))
    else:
        findings.append(DoctorFinding("WARN", "hbn paths file", "no hBN paths file found"))

    for key in ("osc", "dark", "bundle", "click_profile", "fit_profile"):
        value = resolved.get(key)
        if not value:
            findings.append(DoctorFinding("WARN", f"hbn path {key}", "not configured"))
            continue
        exists = _path_exists(str(value))
        is_strict_failure = explicit_local_config and key in REQUIRED_HBN_INPUT_KEYS and not exists
        findings.append(
            DoctorFinding(
                "OK" if exists else "WARN",
                f"hbn path {key}",
                f"{value} ({'exists' if exists else 'missing'})",
                strict_failure=is_strict_failure,
            )
        )
    return findings


def _check_writable_dir(key: str, config_dir: Path) -> DoctorFinding:
    try:
        path = config.get_dir(key, config_dir=config_dir)
        probe = path / ".ra_sim_doctor_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return DoctorFinding("FAIL", f"dir {key}", str(exc), strict_failure=True)
    return DoctorFinding("OK", f"dir {key}", f"{path} writable")


def _check_dirs(config_dir: Path) -> list[DoctorFinding]:
    bundle = config.get_config_bundle(config_dir)
    keys = sorted(set(config.DEFAULT_DIRS) | set(bundle.dir_paths))
    return [_check_writable_dir(key, config_dir) for key in keys]


def _check_tkinter() -> DoctorFinding:
    if importlib.util.find_spec("tkinter") is None:
        return DoctorFinding("WARN", "tkinter", "not importable")
    return DoctorFinding("OK", "tkinter", "importable")


def check_dev_tools() -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    for module_name, label in DEV_TOOL_MODULES:
        if importlib.util.find_spec(module_name) is None:
            findings.append(DoctorFinding("FAIL", label, f"module {module_name!r} not found", True))
        else:
            findings.append(DoctorFinding("OK", label, f"module {module_name!r} found"))
    findings.append(_check_ruff())
    return findings


def _check_ruff() -> DoctorFinding:
    command = [sys.executable, "-m", "ruff", "--version"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return DoctorFinding("FAIL", "ruff", str(exc), strict_failure=True)
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        detail = output or f"exit {completed.returncode}"
        return DoctorFinding("FAIL", "ruff", detail, strict_failure=True)
    return DoctorFinding("OK", "ruff", output or "available")


def _check_python_version() -> DoctorFinding:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 11):
        return DoctorFinding("FAIL", "python", f"{version}; requires >= 3.11", True)
    return DoctorFinding("OK", "python", f"{version}; requires >= 3.11")


def build_report(
    *,
    config_dir: Path | None = None,
    strict: bool = False,
) -> tuple[list[str], bool]:
    use_active_resolver = config_dir is None
    active_config_dir = (config_dir or config.get_config_dir()).resolve()
    env_config_dir = os.environ.get(config.ENV_CONFIG_DIR)
    findings: list[DoctorFinding] = [
        _check_python_version(),
        DoctorFinding("OK", "repo root", str(repo_root())),
        DoctorFinding("OK", "config dir", str(active_config_dir)),
        DoctorFinding(
            "OK" if env_config_dir else "WARN",
            config.ENV_CONFIG_DIR,
            env_config_dir or "not set; using repository config",
        ),
        _config_file_status(
            active_config_dir,
            FILE_PATHS_FILENAME,
            FILE_PATHS_EXAMPLE_FILENAME,
        ),
        _config_file_status(active_config_dir, "hbn_paths.yaml", "hbn_paths.example.yaml"),
    ]

    file_paths_are_local = (active_config_dir / FILE_PATHS_FILENAME).exists()
    hbn_paths_are_local = (active_config_dir / "hbn_paths.yaml").exists()
    findings.extend(_check_file_paths(active_config_dir, file_paths_are_local))
    findings.extend(
        _check_hbn_paths(
            active_config_dir,
            hbn_paths_are_local,
            use_active_resolver=use_active_resolver,
        )
    )
    findings.extend(_check_dirs(active_config_dir))
    findings.append(_check_tkinter())
    findings.extend(check_dev_tools())

    strict_failed = any(finding.strict_failure for finding in findings)
    lines = [
        "RA-SIM setup doctor",
        *(_finding_line(finding, strict=strict) for finding in findings),
    ]
    return lines, strict_failed


def run_doctor(
    *,
    strict: bool = False,
    config_dir: Path | None = None,
    out: TextIO | None = None,
) -> int:
    stream = sys.stdout if out is None else out
    lines, strict_failed = build_report(config_dir=config_dir, strict=strict)
    for line in lines:
        print(line, file=stream)
    return 1 if strict and strict_failed else 0
