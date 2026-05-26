from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "testing-and-validation.md"

PATH_PREFIXES = (
    ".agents/",
    ".github/",
    "artifacts/",
    "config/",
    "docs/",
    "ra_sim/",
    "scripts/",
    "tests/",
)
TOP_LEVEL_FILES = {
    ".pre-commit-config.yaml",
    "coverage.xml",
    "pyproject.toml",
}
GENERATED_OR_EXAMPLE_PATTERNS = (
    re.compile(r"^artifacts/"),
    re.compile(r"^coverage\.xml$"),
    re.compile(r"^scripts/debug/simulation\.npz$"),
)
PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:\.agents|\.github|artifacts|config|docs|ra_sim|scripts|tests)/[A-Za-z0-9_./{}<>*?-]+"
    r"|\.pre-commit-config\.yaml|pyproject\.toml|coverage\.xml)"
)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
TEST_INDEX_HEADER = "| Category | Path | Run with | What it validates | Notes |"
PLACEHOLDER_DESCRIPTION_RE = re.compile(r"\bRegression coverage for\b", re.IGNORECASE)


def _tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    deleted = subprocess.run(
        ["git", "ls-files", "--deleted"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    deleted_paths = set(deleted.stdout.splitlines())
    return [path for path in completed.stdout.splitlines() if path not in deleted_paths]


def _assert_contains_all(doc: str, paths: list[str]) -> None:
    missing = [path for path in paths if path not in doc]
    assert missing == []


def _expected_index_paths(tracked: list[str]) -> list[str]:
    expected: set[str] = set()
    expected.update(
        path
        for path in tracked
        if path.startswith("tests/") and Path(path).name.startswith("test_") and path.endswith(".py")
    )
    expected.update(path for path in tracked if path.startswith("tests/") and not path.endswith(".py"))
    expected.update(
        path
        for path in tracked
        if path.startswith("tests/")
        and path.endswith(".py")
        and not Path(path).name.startswith("test_")
        and path not in {"tests/conftest.py", "tests/__init__.py"}
    )
    expected.update(
        path for path in tracked if path.startswith("scripts/") and path.endswith(".py")
    )
    expected.update(
        path
        for path in tracked
        if path.startswith("ra_sim/tools/")
        and path.endswith(".py")
        and Path(path).name != "__init__.py"
    )
    expected.update(
        {
            "pyproject.toml",
            "tests/conftest.py",
            "ra_sim/dev.py",
            "ra_sim/dev_doctor.py",
            "ra_sim/test_tiers.py",
            "ra_sim/timing.py",
        }
    )
    if ".pre-commit-config.yaml" in tracked:
        expected.add(".pre-commit-config.yaml")
    expected.update(path for path in tracked if path.startswith(".github/workflows/"))
    expected.update(
        path for path in tracked if path.startswith(".agents/skills/") and path.endswith(".py")
    )
    expected.update(
        path
        for path in tracked
        if path.startswith("ra_sim/validation/") and path.endswith(".py")
    )
    expected.update(
        path
        for path in tracked
        if path.startswith("ra_sim/structure_factors/") and path.endswith(".py")
    )
    if "ra_sim/simulation/mosaic_normalization.py" in tracked:
        expected.add("ra_sim/simulation/mosaic_normalization.py")
    return sorted(expected)


def _table_cells(doc: str) -> list[str]:
    cells: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "|" not in stripped[1:]:
            continue
        cells.extend(cell.strip() for cell in stripped.strip("|").split("|"))
    return cells


def _first_check_for_change_type(doc: str, change_type: str) -> str:
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] == change_type:
            return cells[1].strip("`")
    raise AssertionError(f"Missing validation row for {change_type!r}")


def _candidate_path_tokens(doc: str) -> set[str]:
    search_units = [match.group(1) for match in CODE_SPAN_RE.finditer(doc)]
    search_units.extend(_table_cells(doc))
    candidates: set[str] = set()
    for unit in search_units:
        for match in PATH_TOKEN_RE.finditer(unit):
            token = match.group(1).rstrip(".,);]")
            if token:
                candidates.add(token)
    return candidates


def _test_file_index_rows(doc: str) -> list[list[str]]:
    rows: list[list[str]] = []
    in_index = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped == TEST_INDEX_HEADER:
            in_index = True
            continue
        if not in_index:
            continue
        if not stripped.startswith("|"):
            break
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 5 or set(cells[0]) <= {"-"}:
            continue
        path_cell = cells[1]
        if path_cell.startswith("`tests/") and path_cell.endswith(".py`"):
            rows.append(cells)
    return rows


def _is_placeholder_or_glob(path: str) -> bool:
    return any(marker in path for marker in ("<", ">", "{", "}", "*", "?"))


def _is_generated_or_example(path: str) -> bool:
    return any(pattern.search(path) for pattern in GENERATED_OR_EXAMPLE_PATTERNS)


def _looks_like_repo_path(path: str) -> bool:
    return path in TOP_LEVEL_FILES or path.startswith(PATH_PREFIXES)


def test_testing_validation_index_lists_tracked_validation_entrypoints() -> None:
    tracked = _tracked_files()
    doc = DOC_PATH.read_text(encoding="utf-8")

    _assert_contains_all(doc, _expected_index_paths(tracked))


def test_optics_mode_first_check_runs_cli_migration_guards_without_global_filter() -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")
    command = _first_check_for_change_type(doc, "Optics-mode compatibility or migration")

    assert " -k " not in f" {command} "
    assert (
        "tests/test_cli_geometry_fit.py::"
        "test_headless_geometry_fit_optics_defaults_to_exact_and_rejects_fast"
    ) in command
    assert (
        "tests/test_cli_geometry_fit.py::"
        "test_cli_saved_gui_optics_mode_migrates_stale_fast_to_exact"
    ) in command


def test_testing_validation_index_path_references_are_tracked_or_allowed() -> None:
    tracked = set(_tracked_files())
    doc = DOC_PATH.read_text(encoding="utf-8")
    bad_paths: list[str] = []
    for path in sorted(_candidate_path_tokens(doc)):
        if not _looks_like_repo_path(path):
            continue
        if _is_placeholder_or_glob(path) or _is_generated_or_example(path):
            continue
        if path in tracked:
            continue
        bad_paths.append(path)

    assert bad_paths == []


def test_testing_validation_index_test_rows_use_specific_descriptions() -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")

    vague_rows = [
        cells[1].strip("`")
        for cells in _test_file_index_rows(doc)
        if PLACEHOLDER_DESCRIPTION_RE.search(cells[3])
    ]

    assert vague_rows == []
