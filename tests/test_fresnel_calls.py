"""Regression checks for Fresnel transmission call sites."""

from __future__ import annotations

import ast
from pathlib import Path


def test_fresnel_calls_use_boolean_direction_flag() -> None:
    source = Path("ra_sim/simulation/diffraction.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    string_direction_lines: list[int] = []
    bool_flags: list[bool] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "fresnel_transmission":
            continue
        if len(node.args) < 4:
            continue

        direction_arg = node.args[3]
        if isinstance(direction_arg, ast.Constant) and isinstance(direction_arg.value, str):
            string_direction_lines.append(node.lineno)
        if isinstance(direction_arg, ast.Constant) and isinstance(direction_arg.value, bool):
            bool_flags.append(direction_arg.value)

    assert not string_direction_lines, (
        "fresnel_transmission calls must pass a boolean direction flag "
        f"(found string literals at lines {string_direction_lines})."
    )
    assert True in bool_flags and False in bool_flags
