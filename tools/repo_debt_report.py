"""Read-only repository debt report for cleanup planning."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re


EXCLUDED_DIRS = {
    ".agents",
    ".git",
    ".github",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "temp",
}
ENV_FLAG_RE = re.compile(r"\bRA_SIM(?:_[A-Z0-9]+)+\b")
DIAGNOSTIC_STATUS_HEADINGS = {
    "maintained diagnostics": "maintained",
    "archived diagnostics": "archived",
    "delete candidates": "delete_candidate",
}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if not _is_excluded(path))


def _text_files(root: Path) -> list[Path]:
    suffixes = {".bat", ".cfg", ".json", ".md", ".ps1", ".py", ".sh", ".toml", ".yaml", ".yml"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not _is_excluded(path)
    )


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _line in handle)
    except UnicodeDecodeError:
        return 0


def top_files(root: Path, *, limit: int = 20) -> list[dict[str, object]]:
    rows = [
        {"path": _relative(path, root), "line_count": _line_count(path)}
        for path in _python_files(root)
    ]
    return sorted(rows, key=lambda row: (-int(row["line_count"]), str(row["path"])))[:limit]


def _iter_function_defs(path: Path) -> list[dict[str, int | str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    rows: list[dict[str, int | str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        line_number = node.lineno
        end_line = node.end_lineno or node.lineno
        rows.append(
            {
                "name": node.name,
                "line": line_number,
                "end_line": end_line,
                "line_count": end_line - line_number + 1,
            }
        )
    return rows


def top_functions(root: Path, *, limit: int = 20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _python_files(root):
        for node in _iter_function_defs(path):
            rows.append(
                {
                    "path": _relative(path, root),
                    "name": node["name"],
                    "line": node["line"],
                    "end_line": node["end_line"],
                    "line_count": node["line_count"],
                }
            )
    return sorted(rows, key=lambda row: (-int(row["line_count"]), str(row["path"]), str(row["name"])))[
        :limit
    ]


def _literal_string_sequence(value: ast.AST) -> list[str]:
    if not isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return []
    strings: list[str] = []
    for element in value.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            strings.append(element.value)
    return strings


def _tier_name(constant_name: str) -> str | None:
    if not constant_name.endswith("_TEST_FILES"):
        return None
    prefix = constant_name[: -len("_TEST_FILES")].lower()
    return prefix or None


def test_tier_report(root: Path) -> dict[str, object]:
    test_dir = root / "tests"
    top_level_tests = {path.name for path in test_dir.glob("test_*.py")}
    tier_file = root / "ra_sim" / "test_tiers.py"
    tiers: dict[str, list[str]] = {}
    benchmark_dirs: list[str] = []

    if tier_file.exists():
        tree = ast.parse(tier_file.read_text(encoding="utf-8"), filename=str(tier_file))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                tier = _tier_name(name)
                if tier is not None:
                    tiers[tier] = sorted(_literal_string_sequence(node.value))
                elif name == "BENCHMARK_TEST_DIR" and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        benchmark_dirs.append(node.value.value)

    classified_by_file: dict[str, list[str]] = {}
    for tier, names in tiers.items():
        for name in names:
            classified_by_file.setdefault(name, []).append(tier)

    benchmark_tests: set[str] = set()
    for benchmark_dir in benchmark_dirs:
        benchmark_path = root / benchmark_dir
        if benchmark_path.exists():
            tiers.setdefault("benchmark", [])
            for path in sorted(benchmark_path.glob("test_*.py")):
                rel_path = _relative(path, root)
                benchmark_tests.add(rel_path)
                tiers["benchmark"].append(rel_path)
                classified_by_file.setdefault(rel_path, []).append("benchmark")

    duplicates = {
        name: tier_names
        for name, tier_names in sorted(classified_by_file.items())
        if len(tier_names) > 1
    }
    all_tests = sorted(top_level_tests | benchmark_tests)
    unclassified = [name for name in all_tests if name not in classified_by_file]

    return {
        "total": len(all_tests),
        "classified": len(all_tests) - len(unclassified),
        "unclassified": unclassified,
        "duplicates": duplicates,
        "tiers": {tier: sorted(names) for tier, names in sorted(tiers.items())},
    }


def _parse_diagnostics_readme(root: Path) -> dict[str, list[str]]:
    readme_path = root / "scripts" / "diagnostics" / "README.md"
    by_status = {status: [] for status in DIAGNOSTIC_STATUS_HEADINGS.values()}
    if not readme_path.exists():
        return by_status

    current_status: str | None = None
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        heading = line.strip().lstrip("#").strip().lower()
        if heading in DIAGNOSTIC_STATUS_HEADINGS:
            current_status = DIAGNOSTIC_STATUS_HEADINGS[heading]
            continue
        if current_status is None:
            continue
        for filename in re.findall(r"`([^`]+\.py)`", line):
            by_status[current_status].append(filename)
    return {status: sorted(names) for status, names in by_status.items()}


def diagnostics_report(root: Path) -> dict[str, object]:
    diagnostics_dir = root / "scripts" / "diagnostics"
    scripts = sorted(path.name for path in diagnostics_dir.glob("*.py"))
    by_status = _parse_diagnostics_readme(root)
    seen: dict[str, list[str]] = {}
    for status, names in by_status.items():
        for name in names:
            seen.setdefault(name, []).append(status)

    duplicates = {
        name: statuses for name, statuses in sorted(seen.items()) if len(statuses) > 1
    }
    unclassified = [name for name in scripts if name not in seen]
    classified = [name for name in scripts if name in seen]
    return {
        "total": len(scripts),
        "classified": len(classified),
        "unclassified": unclassified,
        "duplicates": duplicates,
        "by_status": by_status,
    }


def duplicate_diagnostic_functions(root: Path) -> list[dict[str, object]]:
    diagnostics_dir = root / "scripts" / "diagnostics"
    function_files: dict[str, set[str]] = {}
    for path in sorted(diagnostics_dir.glob("*.py")):
        for node in _iter_function_defs(path):
            function_files.setdefault(str(node["name"]), set()).add(_relative(path, root))
    return [
        {"name": name, "files": sorted(files)}
        for name, files in sorted(function_files.items())
        if len(files) > 1
    ]


def env_flag_report(root: Path) -> dict[str, object]:
    observed: set[str] = set()
    declared: set[str] = set()
    for path in _text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        flags = set(ENV_FLAG_RE.findall(text))
        observed.update(flags)
        if path.as_posix().endswith("ra_sim/debug_flags.py"):
            declared.update(flags)
    return {
        "observed": sorted(observed),
        "declared": sorted(declared),
        "undeclared": sorted(observed - declared),
    }


def build_report(root: Path) -> dict[str, object]:
    root = root.resolve()
    return {
        "top_files": top_files(root),
        "top_functions": top_functions(root),
        "test_tiers": test_tier_report(root),
        "diagnostics": diagnostics_report(root),
        "duplicate_diagnostic_functions": duplicate_diagnostic_functions(root),
        "env_flags": env_flag_report(root),
    }


def _format_text_report(payload: dict[str, object]) -> str:
    lines = ["RA-SIM repository debt report"]
    lines.append("")
    lines.append("Top files:")
    for row in payload["top_files"][:10]:  # type: ignore[index]
        lines.append(f"- {row['line_count']:>6} {row['path']}")
    lines.append("")
    test_tiers = payload["test_tiers"]  # type: ignore[assignment]
    lines.append(
        "Test tiers: "
        f"{test_tiers['classified']}/{test_tiers['total']} classified, "
        f"{len(test_tiers['unclassified'])} unclassified"
    )
    diagnostics = payload["diagnostics"]  # type: ignore[assignment]
    lines.append(
        "Diagnostics: "
        f"{diagnostics['classified']}/{diagnostics['total']} classified, "
        f"{len(diagnostics['unclassified'])} unclassified"
    )
    env_flags = payload["env_flags"]  # type: ignore[assignment]
    lines.append(
        "Env flags: "
        f"{len(env_flags['observed'])} observed, {len(env_flags['undeclared'])} undeclared"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_report(args.root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
