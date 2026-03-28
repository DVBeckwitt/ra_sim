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


def test_runtime_hkl_pick_constants_are_defined_before_selected_peak_bootstrap() -> None:
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
        if func.attr != "build_runtime_selected_peak_bootstrap":
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


def test_runtime_structure_factor_pruning_controls_are_bootstrapped() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    direct_view_calls = 0
    direct_trace_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_structure_factor_pruning_controls_bootstrap":
            has_bootstrap_call = True
        if func.attr == "create_structure_factor_pruning_controls":
            direct_view_calls += 1
        if (
            func.attr == "trace_add"
            and isinstance(func.value, ast.Name)
            and func.value.id
            in {
                "sf_prune_bias_var",
                "solve_q_steps_var",
                "solve_q_rel_tol_var",
                "solve_q_mode_var",
            }
        ):
            direct_trace_calls += 1

    assert has_bootstrap_call is True
    assert direct_view_calls == 0
    assert direct_trace_calls == 0


def test_runtime_integration_range_controls_are_bootstrapped() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    direct_view_calls = {
        "create_integration_range_controls": 0,
        "create_analysis_view_controls": 0,
    }
    inline_defs = {
        "schedule_range_update": 0,
        "toggle_1d_plots": 0,
        "toggle_caked_2d": 0,
        "toggle_log_radial": 0,
        "toggle_log_azimuth": 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "build_runtime_integration_range_update_bootstrap":
                    has_bootstrap_call = True
                if func.attr in direct_view_calls:
                    direct_view_calls[func.attr] += 1
        if isinstance(node, ast.FunctionDef) and node.name in inline_defs:
            inline_defs[node.name] += 1

    assert has_bootstrap_call is True
    assert direct_view_calls["create_integration_range_controls"] == 0
    assert direct_view_calls["create_analysis_view_controls"] == 0
    assert inline_defs == {
        "schedule_range_update": 0,
        "toggle_1d_plots": 0,
        "toggle_caked_2d": 0,
        "toggle_log_radial": 0,
        "toggle_log_azimuth": 0,
    }


def test_runtime_background_controls_are_bootstrapped() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    direct_view_calls = {
        "create_background_file_controls": 0,
        "create_background_backend_debug_controls": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_background_controls_bootstrap":
            has_bootstrap_call = True
        if func.attr in direct_view_calls:
            direct_view_calls[func.attr] += 1

    assert has_bootstrap_call is True
    assert direct_view_calls == {
        "create_background_file_controls": 0,
        "create_background_backend_debug_controls": 0,
    }


def test_runtime_background_theta_callbacks_use_shared_bundle() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    inline_defs = {
        "_current_geometry_fit_background_indices": 0,
        "_geometry_fit_uses_shared_theta_offset": 0,
        "_current_geometry_theta_offset": 0,
        "_current_background_theta_values": 0,
        "_background_theta_for_index": 0,
        "_sync_background_theta_controls": 0,
        "_apply_background_theta_metadata": 0,
        "_apply_geometry_fit_background_selection": 0,
        "_sync_geometry_fit_background_selection": 0,
    }
    direct_helper_calls = {
        "current_geometry_fit_background_indices": 0,
        "geometry_fit_uses_shared_theta_offset": 0,
        "current_geometry_theta_offset": 0,
        "current_background_theta_values": 0,
        "background_theta_for_index": 0,
        "sync_background_theta_controls": 0,
        "apply_background_theta_metadata": 0,
        "apply_geometry_fit_background_selection": 0,
        "sync_geometry_fit_background_selection": 0,
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "build_runtime_background_theta_bootstrap":
                    has_bootstrap_call = True
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "gui_background_theta"
                    and func.attr in direct_helper_calls
                ):
                    direct_helper_calls[func.attr] += 1
        if isinstance(node, ast.FunctionDef) and node.name in inline_defs:
            inline_defs[node.name] += 1

    assert has_bootstrap_call is True
    assert inline_defs == {
        "_current_geometry_fit_background_indices": 0,
        "_geometry_fit_uses_shared_theta_offset": 0,
        "_current_geometry_theta_offset": 0,
        "_current_background_theta_values": 0,
        "_background_theta_for_index": 0,
        "_sync_background_theta_controls": 0,
        "_apply_background_theta_metadata": 0,
        "_apply_geometry_fit_background_selection": 0,
        "_sync_geometry_fit_background_selection": 0,
    }
    assert direct_helper_calls == {
        "current_geometry_fit_background_indices": 0,
        "geometry_fit_uses_shared_theta_offset": 0,
        "current_geometry_theta_offset": 0,
        "current_background_theta_values": 0,
        "background_theta_for_index": 0,
        "sync_background_theta_controls": 0,
        "apply_background_theta_metadata": 0,
        "apply_geometry_fit_background_selection": 0,
        "sync_geometry_fit_background_selection": 0,
    }


def test_runtime_background_status_refreshes_use_shared_helpers() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    helper_defs = {
        "_refresh_background_status": 0,
        "_refresh_background_backend_status": 0,
    }
    direct_runtime_status_refs: list[tuple[str, str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in helper_defs:
            helper_defs[node.name] += 1
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"refresh_status", "refresh_backend_status"}
            and isinstance(node.value, ast.Name)
            and node.value.id in {"background_runtime_callbacks", "background_controls_runtime"}
        ):
            direct_runtime_status_refs.append(
                (node.value.id, node.attr, int(node.lineno))
            )

    assert helper_defs == {
        "_refresh_background_status": 1,
        "_refresh_background_backend_status": 1,
    }
    assert direct_runtime_status_refs == []


def test_runtime_manual_geometry_cache_callbacks_use_shared_bootstrap() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    bootstrap_calls = 0
    direct_helper_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_manual_cache_bootstrap":
            bootstrap_calls += 1
        if (
            isinstance(func.value, ast.Name)
            and func.value.id == "gui_manual_geometry"
            and func.attr == "make_runtime_geometry_manual_cache_callbacks"
        ):
            direct_helper_calls += 1

    assert bootstrap_calls == 1
    assert direct_helper_calls == 0


def test_runtime_manual_geometry_projection_callbacks_use_shared_bootstrap() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    bootstrap_calls = 0
    direct_helper_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_manual_projection_bootstrap":
            bootstrap_calls += 1
        if (
            isinstance(func.value, ast.Name)
            and func.value.id == "gui_manual_geometry"
            and func.attr == "make_runtime_geometry_manual_projection_callbacks"
        ):
            direct_helper_calls += 1

    assert bootstrap_calls == 1
    assert direct_helper_calls == 0


def test_runtime_manual_geometry_callbacks_use_shared_bundle() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_helper_call = False
    direct_bootstrap_calls = 0
    direct_helper_calls = {
        "render_current_geometry_manual_pairs": 0,
        "geometry_manual_toggle_selection_at": 0,
        "geometry_manual_place_selection_at": 0,
        "geometry_manual_pick_preview_state": 0,
        "cancel_geometry_manual_pick_session": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_manual_bootstrap":
            has_helper_call = True
        if func.attr == "make_runtime_geometry_manual_callbacks":
            direct_bootstrap_calls += 1
        if (
            isinstance(func.value, ast.Name)
            and func.value.id == "gui_manual_geometry"
            and func.attr in direct_helper_calls
        ):
            direct_helper_calls[func.attr] += 1

    assert has_helper_call is True
    assert direct_helper_calls == {
        "render_current_geometry_manual_pairs": 0,
        "geometry_manual_toggle_selection_at": 0,
        "geometry_manual_place_selection_at": 0,
        "geometry_manual_pick_preview_state": 0,
        "cancel_geometry_manual_pick_session": 0,
    }
    assert direct_bootstrap_calls == 0


def test_runtime_geometry_tool_action_callbacks_use_shared_bootstrap() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    bootstrap_calls = 0
    direct_helper_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_tool_action_callbacks_bootstrap":
            bootstrap_calls += 1
        if (
            isinstance(func.value, ast.Name)
            and func.value.id == "gui_geometry_fit"
            and func.attr == "make_runtime_geometry_tool_action_callbacks"
        ):
            direct_helper_calls += 1

    assert bootstrap_calls == 1
    assert direct_helper_calls == 0


def test_runtime_hkl_lookup_controls_are_bootstrapped() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    direct_view_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_hkl_lookup_controls_bootstrap":
            has_bootstrap_call = True
        if func.attr == "create_hkl_lookup_controls":
            direct_view_calls += 1

    assert has_bootstrap_call is True
    assert direct_view_calls == 0


def test_runtime_geometry_tool_action_controls_are_bootstrapped() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    has_bootstrap_call = False
    direct_view_calls = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "build_runtime_geometry_tool_action_controls_bootstrap":
            has_bootstrap_call = True
        if func.attr == "create_geometry_tool_action_controls":
            direct_view_calls += 1

    assert has_bootstrap_call is True
    assert direct_view_calls == 0


def test_runtime_selected_peak_refresh_calls_use_maintenance_bundle() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

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


def test_runtime_geometry_fit_action_uses_factory_helpers() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

    helper_calls: dict[str, int] = {
        "build_runtime_geometry_fit_action_bootstrap": 0,
        "make_runtime_geometry_fit_action_bindings_factory": 0,
        "make_runtime_geometry_fit_action_callback": 0,
        "build_runtime_geometry_fit_action_bindings": 0,
        "run_runtime_geometry_fit_action": 0,
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr in helper_calls:
            helper_calls[func.attr] += 1

    assert helper_calls["build_runtime_geometry_fit_action_bootstrap"] == 1
    assert helper_calls["make_runtime_geometry_fit_action_bindings_factory"] == 0
    assert helper_calls["make_runtime_geometry_fit_action_callback"] == 0
    assert helper_calls["build_runtime_geometry_fit_action_bindings"] == 0
    assert helper_calls["run_runtime_geometry_fit_action"] == 0


def test_runtime_geometry_fit_action_bootstrap_is_built_after_live_value_setup() -> None:
    source = Path("ra_sim/gui/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="ra_sim/gui/runtime.py")

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
        if func.attr == "build_runtime_geometry_fit_action_bootstrap":
            bootstrap_use_line = int(node.lineno)

    assert bootstrap_use_line > 0
    assert assignment_lines["theta_initial_var"] < bootstrap_use_line
    assert assignment_lines["_geometry_fit_runtime_value_callbacks"] < bootstrap_use_line
    assert assignment_lines["_geometry_fit_var_map"] < bootstrap_use_line
