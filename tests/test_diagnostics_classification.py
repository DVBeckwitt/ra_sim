from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPO_ROOT / "scripts" / "diagnostics"
README_PATH = DIAGNOSTICS_DIR / "README.md"
STATUS_HEADINGS = {
    "maintained diagnostics": "maintained",
    "archived diagnostics": "archived",
    "delete candidates": "delete_candidate",
}


def _readme_classifications() -> dict[str, list[str]]:
    current_status: str | None = None
    classifications: dict[str, list[str]] = {}

    for line in README_PATH.read_text(encoding="utf-8").splitlines():
        heading = line.strip().lstrip("#").strip().lower()
        if heading in STATUS_HEADINGS:
            current_status = STATUS_HEADINGS[heading]
            continue
        if current_status is None:
            continue
        for filename in re.findall(r"`([^`]+\.py)`", line):
            classifications.setdefault(filename, []).append(current_status)

    return classifications


def test_every_diagnostics_script_is_classified_once() -> None:
    diagnostics_scripts = {path.name for path in DIAGNOSTICS_DIR.glob("*.py")}
    classifications = _readme_classifications()

    missing = sorted(diagnostics_scripts - set(classifications))
    unknown = sorted(set(classifications) - diagnostics_scripts)
    duplicates = {
        filename: statuses
        for filename, statuses in sorted(classifications.items())
        if len(statuses) != 1
    }

    assert not missing, f"Unclassified diagnostics scripts: {missing}"
    assert not unknown, f"README references missing diagnostics scripts: {unknown}"
    assert not duplicates, f"Diagnostics scripts classified more than once: {duplicates}"
