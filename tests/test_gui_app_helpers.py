from __future__ import annotations

from ra_sim.gui import app as gui_app


def test_app_normalize_hkl_key_accepts_common_label_formats() -> None:
    assert gui_app._normalize_hkl_key("1,0,0") == (1, 0, 0)
    assert gui_app._normalize_hkl_key("(1, 0, 0)") == (1, 0, 0)
    assert gui_app._normalize_hkl_key("[1, 0, 0]") == (1, 0, 0)
    assert gui_app._normalize_hkl_key((1.2, 0.2, 0.0)) == (1, 0, 0)
    assert gui_app._normalize_hkl_key("bad-label") is None


def test_app_build_geometry_fit_runtime_config_converts_ui_windows_to_absolute_bounds() -> None:
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

    runtime_cfg = gui_app._build_geometry_fit_runtime_config(
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


def test_app_build_geometry_fit_runtime_config_clamps_to_parameter_domain() -> None:
    runtime_cfg = gui_app._build_geometry_fit_runtime_config(
        {},
        {"center_x": 98.0},
        {"center_x": {"window": 10.0, "pull": 1.0}},
        {"center_x": (0.0, 99.0)},
    )

    assert runtime_cfg["bounds"]["center_x"] == [88.0, 99.0]
    assert runtime_cfg["priors"]["center_x"]["center"] == 98.0
    assert abs(runtime_cfg["priors"]["center_x"]["sigma"] - 0.5) < 1e-12
