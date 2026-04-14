from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "ra_sim"
STDLIB_MODULES = set(sys.stdlib_module_names)
IMPORT_TO_DISTRIBUTION = {
    "ciffile": "pycifrw",
    "cv2": "opencv-python",
    "dans_diffraction": "dans_diffraction",
    "pil": "pillow",
    "pyqtgraph": "pyqtgraph",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
}
JUSTIFIED_DECLARED_ONLY = {
    "mosaic_sim",
    "openpyxl",
    "xlsxwriter",
}
OPTIONAL_RUNTIME_IMPORTS = {"pyqtgraph"}
FORBIDDEN_BASE_DEPENDENCIES = {"fabio", "pyqtgraph"}


def _normalized_distribution_name(requirement: str) -> str:
    text = requirement.split(";", 1)[0].strip()
    if " @ " in text:
        text = text.split(" @ ", 1)[0].strip()
    else:
        text = re.split(r"[<>=!~\[]", text, maxsplit=1)[0].strip()
    return text.lower()


def _declared_runtime_dependencies() -> set[str]:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    return {_normalized_distribution_name(requirement) for requirement in dependencies}


def _runtime_imported_distributions() -> set[str]:
    distributions: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or node.module is None:
                    continue
                names = [node.module]
            else:
                continue
            for name in names:
                top_level = name.split(".", 1)[0]
                normalized = top_level.lower()
                if normalized in STDLIB_MODULES or normalized == "ra_sim":
                    continue
                distributions.add(
                    IMPORT_TO_DISTRIBUTION.get(normalized, normalized)
                )
    return distributions


def test_project_dependencies_match_runtime_import_surface() -> None:
    declared = _declared_runtime_dependencies()
    imported = _runtime_imported_distributions()

    missing_from_declared = sorted(imported - declared - OPTIONAL_RUNTIME_IMPORTS)
    unjustified_declared = sorted(declared - imported - JUSTIFIED_DECLARED_ONLY)

    assert missing_from_declared == []
    assert unjustified_declared == []


def test_base_install_keeps_manual_optional_and_transitive_deps_out() -> None:
    declared = _declared_runtime_dependencies()

    assert JUSTIFIED_DECLARED_ONLY.issubset(declared)
    assert declared.isdisjoint(FORBIDDEN_BASE_DEPENDENCIES)
