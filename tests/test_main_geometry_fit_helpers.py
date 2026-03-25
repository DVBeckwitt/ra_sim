import ast
from pathlib import Path

GUI_APP_PATH = Path("ra_sim/gui/app.py")


def _load_main_functions(*names: str) -> dict[str, object]:
    source = GUI_APP_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(GUI_APP_PATH))
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
        raise AssertionError(
            f"Failed to extract functions from {GUI_APP_PATH}: {missing}"
        )

    namespace: dict[str, object] = {}
    exec(
        "import copy\nimport numpy as np\n\n" + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_main_build_geometry_fit_runtime_config_converts_ui_windows_to_absolute_bounds() -> None:
    namespace = _load_main_functions("_build_geometry_fit_runtime_config")
    build = namespace["_build_geometry_fit_runtime_config"]

    base_config = {
        "bounds": {
            "theta_initial": {"mode": "relative", "min": -0.5, "max": 0.5},
        },
        "priors": {
            "gamma": {"center": 1.0, "sigma": 0.25},
        },
    }
    current_params = {
        "theta_initial": 6.0,
        "gamma": 1.5,
    }
    control_settings = {
        "theta_initial": {"window": 0.4, "pull": 0.5},
        "gamma": {"window": 2.0, "pull": 0.0},
    }
    parameter_domains = {
        "theta_initial": (0.0, 30.0),
        "gamma": (-5.0, 5.0),
    }

    runtime_cfg = build(
        base_config,
        current_params,
        control_settings,
        parameter_domains,
    )

    assert runtime_cfg["bounds"]["theta_initial"] == [5.6, 6.4]
    assert runtime_cfg["bounds"]["gamma"] == [-0.5, 3.5]
    assert runtime_cfg["priors"]["theta_initial"]["center"] == 6.0
    assert abs(runtime_cfg["priors"]["theta_initial"]["sigma"] - 0.21) < 1e-9
    assert "gamma" not in runtime_cfg["priors"]
    assert base_config["bounds"]["theta_initial"] == {
        "mode": "relative",
        "min": -0.5,
        "max": 0.5,
    }


def test_main_build_geometry_fit_runtime_config_clamps_to_parameter_domain() -> None:
    namespace = _load_main_functions("_build_geometry_fit_runtime_config")
    build = namespace["_build_geometry_fit_runtime_config"]

    runtime_cfg = build(
        {},
        {"center_x": 98.0},
        {"center_x": {"window": 10.0, "pull": 1.0}},
        {"center_x": (0.0, 99.0)},
    )

    assert runtime_cfg["bounds"]["center_x"] == [88.0, 99.0]
    assert runtime_cfg["priors"]["center_x"]["center"] == 98.0
    assert abs(runtime_cfg["priors"]["center_x"]["sigma"] - 0.5) < 1e-12
