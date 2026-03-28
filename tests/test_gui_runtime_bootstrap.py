import ast
from pathlib import Path

RUNTIME_IMPL_PATH = Path("ra_sim/gui/_runtime/runtime_impl.py")


def _name_id(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        return node.args[0].id
    return None


def test_runtime_pruning_constants_are_defined_before_import_time_use() -> None:
    source = RUNTIME_IMPL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNTIME_IMPL_PATH))

    assignment_lines: dict[str, int] = {}
    bootstrap_use_lines: dict[str, int] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "SF_PRUNE_BIAS_MIN",
                "SF_PRUNE_BIAS_MAX",
            }:
                assignment_lines[target.id] = int(node.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "build_runtime_structure_factor_pruning_defaults":
            continue
        for keyword in node.keywords:
            name_id = _name_id(keyword.value)
            if keyword.arg == "prune_bias_minimum" and name_id == "SF_PRUNE_BIAS_MIN":
                bootstrap_use_lines["SF_PRUNE_BIAS_MIN"] = int(keyword.value.lineno)
            if keyword.arg == "prune_bias_maximum" and name_id == "SF_PRUNE_BIAS_MAX":
                bootstrap_use_lines["SF_PRUNE_BIAS_MAX"] = int(keyword.value.lineno)

    for name in ("SF_PRUNE_BIAS_MIN", "SF_PRUNE_BIAS_MAX"):
        assert name in assignment_lines, f"missing assignment for {name}"
        assert name in bootstrap_use_lines, f"missing bootstrap use for {name}"
        assert assignment_lines[name] < bootstrap_use_lines[name], (
            f"{name} must be assigned before runtime pruning defaults are built"
        )


def test_runtime_selected_peak_refresh_calls_use_maintenance_bundle() -> None:
    source = RUNTIME_IMPL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNTIME_IMPL_PATH))

    refresh_after_update_calls = 0
    apply_restored_target_calls = 0
    direct_reselect_calls = 0
    direct_hkl_refresh_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "refresh_after_simulation_update":
            refresh_after_update_calls += 1
        if func.attr == "apply_restored_selected_hkl_target":
            apply_restored_target_calls += 1
        if func.attr == "reselect_current_peak":
            direct_reselect_calls += 1
        if (
            func.attr == "refresh_controls"
            and isinstance(func.value, ast.Name)
            and func.value.id == "hkl_lookup_controls_runtime"
        ):
            direct_hkl_refresh_calls += 1

    assert refresh_after_update_calls == 1
    assert apply_restored_target_calls == 1
    assert direct_reselect_calls == 0
    assert direct_hkl_refresh_calls == 0


def test_runtime_geometry_fit_action_bootstrap_is_built_after_live_value_setup() -> None:
    source = RUNTIME_IMPL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNTIME_IMPL_PATH))

    assignment_lines: dict[str, int] = {}
    bootstrap_use_line = 0

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "theta_initial_var",
                "_geometry_fit_runtime_value_callbacks",
                "_geometry_fit_var_map",
            }:
                assignment_lines[target.id] = int(node.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_fit_action_workflow":
            bootstrap_use_line = int(node.lineno)

    assert bootstrap_use_line > 0
    assert assignment_lines["theta_initial_var"] < bootstrap_use_line
    assert assignment_lines["_geometry_fit_runtime_value_callbacks"] < bootstrap_use_line
    assert assignment_lines["_geometry_fit_var_map"] < bootstrap_use_line


