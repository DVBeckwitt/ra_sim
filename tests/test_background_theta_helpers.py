import ast
from pathlib import Path


def _load_main_functions(*names: str) -> dict[str, object]:
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="main.py")
    extracted: list[str] = []
    available = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - available)
    if missing:
        raise AssertionError(f"Failed to extract functions from main.py: {missing}")

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec(
        "import re\n"
        "import numpy as np\n"
        "from typing import Sequence\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_parse_background_theta_values_accepts_common_delimiters() -> None:
    namespace = _load_main_functions("_parse_background_theta_values")
    parse_values = namespace["_parse_background_theta_values"]

    assert parse_values("1.0, 2.5  3.25;4.0") == [1.0, 2.5, 3.25, 4.0]

    try:
        parse_values("1.0, bad", expected_count=2)
    except ValueError as exc:
        assert "Invalid theta value" in str(exc)
    else:
        raise AssertionError("Expected invalid theta token to raise ValueError")


def test_background_theta_for_index_adds_shared_offset_for_multiple_backgrounds() -> None:
    namespace = _load_main_functions(
        "_background_theta_default_value",
        "_parse_background_theta_values",
        "_geometry_fit_uses_shared_theta_offset",
        "_current_geometry_theta_offset",
        "_current_background_theta_values",
        "_background_theta_for_index",
    )

    class _Var:
        def __init__(self, value: str):
            self._value = value

        def get(self) -> str:
            return self._value

    namespace["theta_initial"] = 6.0
    namespace["defaults"] = {"theta_initial": 6.0}
    namespace["background_theta_list_var"] = _Var("4.0, 7.5")
    namespace["geometry_theta_offset_var"] = _Var("0.25")
    namespace["osc_files"] = ["bg0.osc", "bg1.osc"]

    theta_for_index = namespace["_background_theta_for_index"]

    assert theta_for_index(0, strict_count=True) == 4.25
    assert theta_for_index(1, strict_count=True) == 7.75
