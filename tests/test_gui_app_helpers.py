import ast
from pathlib import Path


def _load_app_functions(*names: str) -> dict[str, object]:
    source = Path("ra_sim/gui/app.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="ra_sim/gui/app.py")
    extracted: list[str] = []
    discovered = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)
    missing = sorted(set(names) - discovered)
    if missing:
        raise AssertionError(f"Failed to extract functions from app.py: {missing}")

    namespace: dict[str, object] = {}
    exec(
        "import numpy as np\n\n" + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_app_normalize_hkl_key_accepts_common_label_formats() -> None:
    namespace = _load_app_functions("_normalize_hkl_key")
    normalize = namespace["_normalize_hkl_key"]

    assert normalize("1,0,0") == (1, 0, 0)
    assert normalize("(1, 0, 0)") == (1, 0, 0)
    assert normalize("[1, 0, 0]") == (1, 0, 0)
    assert normalize((1.2, 0.2, 0.0)) == (1, 0, 0)
    assert normalize("bad-label") is None
