import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_APP_PATH = ROOT / "ra_sim" / "gui" / "app.py"


def _find_get_sim_signature_returns(path: Path) -> list[ast.Return]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    returns: list[ast.Return] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_sim_signature":
            returns.extend(
                child for child in ast.walk(node) if isinstance(child, ast.Return)
            )
    return returns


def _parse_file(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _number_value(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def _is_psi_z_round(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Name) or node.func.id != "round":
        return False
    if len(node.args) != 2:
        return False
    value_arg, precision_arg = node.args
    return (
        isinstance(value_arg, ast.Name)
        and value_arg.id == "psi_z_updated"
        and isinstance(precision_arg, ast.Constant)
        and precision_arg.value == 6
    )


def _assert_signature_includes_psi_z(path: Path) -> None:
    returns = _find_get_sim_signature_returns(path)
    assert returns, f"expected get_sim_signature() in {path}"

    for ret in returns:
        tuple_value = ret.value
        if not isinstance(tuple_value, ast.Tuple):
            continue
        if any(_is_psi_z_round(element) for element in tuple_value.elts):
            return

    raise AssertionError(f"psi_z_updated is missing from get_sim_signature() in {path}")


def _find_psi_z_slider_call(path: Path) -> ast.Call:
    tree = _parse_file(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Tuple):
                continue
            target_names = {
                elt.id for elt in target.elts if isinstance(elt, ast.Name)
            }
            if "psi_z_var" not in target_names:
                continue
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == "make_slider":
                    return node.value
    raise AssertionError(f"psi_z slider definition not found in {path}")


def _assert_psi_z_slider_is_limited(path: Path) -> None:
    call = _find_psi_z_slider_call(path)
    assert len(call.args) >= 3, f"psi_z slider call is malformed in {path}"
    min_value = _number_value(call.args[1])
    max_value = _number_value(call.args[2])
    assert min_value == -5.0, f"expected psi_z min -5 in {path}, found {min_value}"
    assert max_value == 5.0, f"expected psi_z max 5 in {path}, found {max_value}"
    assert all(
        keyword.arg != "allow_range_expand" for keyword in call.keywords
    ), f"psi_z slider unexpectedly allows range expansion in {path}"


def _assert_psi_z_trace_clamp(path: Path) -> None:
    tree = _parse_file(path)
    has_clamp_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "_clamp_psi_z_var"
        for node in ast.walk(tree)
    )
    assert has_clamp_fn, f"_clamp_psi_z_var() missing in {path}"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "trace_add":
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "psi_z_var":
            continue
        if len(node.args) < 2:
            continue
        if (
            isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "write"
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "_clamp_psi_z_var"
        ):
            return

    raise AssertionError(f"psi_z clamp trace is missing in {path}")


def test_packaged_gui_signature_includes_psi_z():
    _assert_signature_includes_psi_z(GUI_APP_PATH)


def test_packaged_gui_psi_z_slider_is_limited_and_clamped():
    _assert_psi_z_slider_is_limited(GUI_APP_PATH)
    _assert_psi_z_trace_clamp(GUI_APP_PATH)
