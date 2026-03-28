import ast
from pathlib import Path


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
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

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


def test_runtime_hkl_pick_constants_are_defined_before_binding_factory_use() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    assignment_lines: dict[str, int] = {}
    binding_use_lines: dict[str, int] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "HKL_PICK_MIN_SEPARATION_PX",
                "HKL_PICK_MAX_DISTANCE_PX",
            }:
                assignment_lines[target.id] = int(node.lineno)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "make_runtime_selected_peak_config_factories":
            continue
        for keyword in node.keywords:
            name_id = _name_id(keyword.value)
            if keyword.arg == "max_distance_px" and name_id == "HKL_PICK_MAX_DISTANCE_PX":
                binding_use_lines["HKL_PICK_MAX_DISTANCE_PX"] = int(keyword.value.lineno)
            if (
                keyword.arg == "min_separation_px"
                and name_id == "HKL_PICK_MIN_SEPARATION_PX"
            ):
                binding_use_lines["HKL_PICK_MIN_SEPARATION_PX"] = int(
                    keyword.value.lineno
                )

    for name in ("HKL_PICK_MIN_SEPARATION_PX", "HKL_PICK_MAX_DISTANCE_PX"):
        assert name in assignment_lines, f"missing assignment for {name}"
        assert name in binding_use_lines, f"missing binding-factory use for {name}"
        assert assignment_lines[name] < binding_use_lines[name], (
            f"{name} must be assigned before peak-selection bindings are built"
        )


def test_runtime_canvas_preview_callbacks_are_late_bound() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    lambda_keywords: dict[str, bool] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "build_runtime_canvas_interaction_bootstrap":
            continue
        for keyword in node.keywords:
            if keyword.arg in {
                "set_geometry_preview_exclude_mode",
                "toggle_live_geometry_preview_exclusion_at",
            }:
                lambda_keywords[keyword.arg] = isinstance(keyword.value, ast.Lambda)

    assert lambda_keywords.get("set_geometry_preview_exclude_mode") is True
    assert lambda_keywords.get("toggle_live_geometry_preview_exclusion_at") is True


def test_runtime_schedule_update_factories_are_late_bound() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    lambda_keywords: dict[tuple[str, str], bool] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {
            "build_runtime_background_bootstrap",
            "build_runtime_geometry_q_group_bootstrap",
        }:
            continue
        for keyword in node.keywords:
            if keyword.arg == "schedule_update_factory":
                lambda_keywords[(func.attr, keyword.arg)] = isinstance(
                    keyword.value,
                    ast.Lambda,
                )

    assert (
        lambda_keywords.get(
            ("build_runtime_background_bootstrap", "schedule_update_factory")
        )
        is True
    )
    assert (
        lambda_keywords.get(
            (
                "build_runtime_geometry_q_group_bootstrap",
                "schedule_update_factory",
            )
        )
        is True
    )
