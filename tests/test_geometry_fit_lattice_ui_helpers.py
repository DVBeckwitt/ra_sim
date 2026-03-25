import ast
from pathlib import Path

GUI_APP_PATH = "ra_sim/gui/app.py"


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyRoot:
    def after_cancel(self, _token):
        return None


def _load_functions(path: str, *names: str) -> dict[str, object]:
    source = Path(path).read_text(encoding="utf-8")
    module = ast.parse(source, filename=path)
    extracted: list[str] = []
    discovered = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - discovered)
    if missing:
        raise AssertionError(f"Failed to extract functions from {path}: {missing}")

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec("import numpy as np\n\n" + "\n\n".join(extracted), namespace)
    return namespace


def _load_literal_assignment(path: str, name: str):
    source = Path(path).read_text(encoding="utf-8")
    module = ast.parse(source, filename=path)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"Failed to find literal assignment {name} in {path}")


def test_main_geometry_fit_param_order_includes_lattice_parameters() -> None:
    order = _load_literal_assignment(GUI_APP_PATH, "GEOMETRY_FIT_PARAM_ORDER")
    assert "a" in order
    assert "c" in order


def test_main_current_geometry_fit_var_names_includes_lattice_parameters() -> None:
    namespace = _load_functions(GUI_APP_PATH, "_current_geometry_fit_var_names")
    namespace["_geometry_fit_uses_shared_theta_offset"] = lambda *args, **kwargs: False
    namespace["fit_zb_var"] = _DummyVar(False)
    namespace["fit_zs_var"] = _DummyVar(False)
    namespace["fit_theta_var"] = _DummyVar(False)
    namespace["fit_psi_z_var"] = _DummyVar(False)
    namespace["fit_chi_var"] = _DummyVar(False)
    namespace["fit_cor_var"] = _DummyVar(False)
    namespace["fit_gamma_var"] = _DummyVar(False)
    namespace["fit_Gamma_var"] = _DummyVar(False)
    namespace["fit_dist_var"] = _DummyVar(False)
    namespace["fit_a_var"] = _DummyVar(True)
    namespace["fit_c_var"] = _DummyVar(True)
    namespace["fit_center_x_var"] = _DummyVar(False)
    namespace["fit_center_y_var"] = _DummyVar(False)

    var_names = namespace["_current_geometry_fit_var_names"]()

    assert var_names == ["a", "c"]


def test_main_geometry_fit_ui_params_capture_lattice_parameters() -> None:
    namespace = _load_functions(GUI_APP_PATH, "_current_geometry_fit_ui_params")
    namespace["zb_var"] = _DummyVar(0.0)
    namespace["zs_var"] = _DummyVar(0.0)
    namespace["theta_initial_var"] = _DummyVar(5.0)
    namespace["psi_z_var"] = _DummyVar(0.1)
    namespace["chi_var"] = _DummyVar(0.2)
    namespace["cor_angle_var"] = _DummyVar(0.3)
    namespace["gamma_var"] = _DummyVar(0.4)
    namespace["Gamma_var"] = _DummyVar(0.5)
    namespace["corto_detector_var"] = _DummyVar(0.06)
    namespace["a_var"] = _DummyVar(4.21)
    namespace["c_var"] = _DummyVar(32.4)
    namespace["center_x_var"] = _DummyVar(100.0)
    namespace["center_y_var"] = _DummyVar(200.0)
    namespace["geometry_theta_offset_var"] = None

    params = namespace["_current_geometry_fit_ui_params"]()

    assert params["a"] == 4.21
    assert params["c"] == 32.4


def test_main_restore_geometry_fit_undo_state_restores_lattice_parameters() -> None:
    namespace = _load_functions(
        GUI_APP_PATH,
        "_copy_geometry_fit_state_value",
        "_restore_geometry_fit_undo_state",
    )
    namespace["profile_cache"] = {}
    namespace["last_geometry_overlay_state"] = None
    namespace["last_simulation_signature"] = None
    namespace["update_pending"] = None
    namespace["root"] = _DummyRoot()
    namespace["do_update"] = lambda: None
    namespace["_draw_geometry_fit_overlay"] = lambda *args, **kwargs: None
    namespace["_draw_initial_geometry_pairs_overlay"] = lambda *args, **kwargs: None
    namespace["_set_background_file_status_text"] = lambda: None
    namespace["_update_geometry_manual_pick_button_label"] = lambda: None
    namespace["geometry_theta_offset_var"] = None
    namespace["zb_var"] = _DummyVar(0.0)
    namespace["zs_var"] = _DummyVar(0.0)
    namespace["theta_initial_var"] = _DummyVar(0.0)
    namespace["psi_z_var"] = _DummyVar(0.0)
    namespace["chi_var"] = _DummyVar(0.0)
    namespace["cor_angle_var"] = _DummyVar(0.0)
    namespace["gamma_var"] = _DummyVar(0.0)
    namespace["Gamma_var"] = _DummyVar(0.0)
    namespace["corto_detector_var"] = _DummyVar(0.0)
    namespace["a_var"] = _DummyVar(0.0)
    namespace["c_var"] = _DummyVar(0.0)
    namespace["center_x_var"] = _DummyVar(0.0)
    namespace["center_y_var"] = _DummyVar(0.0)

    namespace["_restore_geometry_fit_undo_state"](
        {
            "ui_params": {
                "a": 4.33,
                "c": 33.8,
            }
        }
    )

    assert namespace["a_var"].get() == 4.33
    assert namespace["c_var"].get() == 33.8
