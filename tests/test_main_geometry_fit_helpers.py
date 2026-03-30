from ra_sim.gui.geometry_fit import build_geometry_fit_runtime_config


def test_main_build_geometry_fit_runtime_config_converts_ui_windows_to_absolute_bounds() -> None:
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

    runtime_cfg = build_geometry_fit_runtime_config(
        base_config,
        current_params,
        control_settings,
        parameter_domains,
    )

    assert runtime_cfg["solver"] == {
        "workers": 1,
        "parallel_mode": "off",
        "worker_numba_threads": 1,
    }
    assert runtime_cfg["use_numba"] is False
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
    runtime_cfg = build_geometry_fit_runtime_config(
        {},
        {"center_x": 98.0},
        {"center_x": {"window": 10.0, "pull": 1.0}},
        {"center_x": (0.0, 99.0)},
    )

    assert runtime_cfg["solver"] == {
        "workers": 1,
        "parallel_mode": "off",
        "worker_numba_threads": 1,
    }
    assert runtime_cfg["use_numba"] is False
    assert runtime_cfg["bounds"]["center_x"] == [88.0, 99.0]
    assert runtime_cfg["priors"]["center_x"]["center"] == 98.0
    assert abs(runtime_cfg["priors"]["center_x"]["sigma"] - 0.5) < 1e-12


def test_main_build_geometry_fit_runtime_config_accepts_explicit_gui_solver_overrides() -> None:
    runtime_cfg = build_geometry_fit_runtime_config(
        {
            "solver": {
                "loss": "soft_l1",
                "gui_workers": 2,
                "gui_parallel_mode": "datasets",
                "gui_worker_numba_threads": 3,
            },
            "gui_use_numba": True,
        },
        {"gamma": 1.0},
        {"gamma": {"window": 0.2, "pull": 0.0}},
        {"gamma": (-5.0, 5.0)},
    )

    assert runtime_cfg["solver"] == {
        "loss": "soft_l1",
        "workers": 2,
        "parallel_mode": "datasets",
        "worker_numba_threads": 3,
    }
    assert runtime_cfg["use_numba"] is True
