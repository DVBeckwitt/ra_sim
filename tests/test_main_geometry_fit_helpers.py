from ra_sim.gui.geometry_fit import (
    build_geometry_fit_runtime_config,
    read_geometry_fit_caked_roi_config,
)


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
        "workers": "auto",
        "parallel_mode": "auto",
        "worker_numba_threads": 0,
    }
    assert runtime_cfg["use_numba"] is False
    assert runtime_cfg["bounds"]["theta_initial"] == [0.0, 30.0]
    assert runtime_cfg["bounds"]["gamma"] == [-5.0, 5.0]
    assert runtime_cfg["priors"] == {}
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
        "workers": "auto",
        "parallel_mode": "auto",
        "worker_numba_threads": 0,
    }
    assert runtime_cfg["use_numba"] is False
    assert runtime_cfg["bounds"]["center_x"] == [0.0, 99.0]
    assert runtime_cfg["priors"] == {}


def test_main_build_geometry_fit_runtime_config_ignores_unsafe_numba_opt_ins() -> None:
    runtime_cfg = build_geometry_fit_runtime_config(
        {
            "solver": {
                "loss": "soft_l1",
                "gui_workers": 2,
                "gui_parallel_mode": "datasets",
                "gui_worker_numba_threads": 3,
            },
            "gui_use_numba": True,
            "gui_allow_unsafe_runtime": True,
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
    assert runtime_cfg["allow_unsafe_runtime"] is False


def test_main_build_geometry_fit_runtime_config_keeps_parallel_settings_while_forcing_safe_runtime() -> None:
    runtime_cfg = build_geometry_fit_runtime_config(
        {
            "solver": {
                "loss": "soft_l1",
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
            },
            "gui_use_numba": True,
            "gui_allow_unsafe_runtime": True,
        },
        {"gamma": 1.0},
        {"gamma": {"window": 0.2, "pull": 0.0}},
        {"gamma": (-5.0, 5.0)},
    )

    assert runtime_cfg["solver"] == {
        "loss": "soft_l1",
        "workers": "auto",
        "parallel_mode": "auto",
        "worker_numba_threads": 0,
    }
    assert runtime_cfg["use_numba"] is True
    assert runtime_cfg["allow_unsafe_runtime"] is False


def test_main_geometry_fit_caked_roi_defaults_and_runtime_override() -> None:
    runtime_cfg = build_geometry_fit_runtime_config(
        {},
        {"gamma": 1.0},
        {"gamma": {"window": 0.2, "pull": 0.0}},
        {"gamma": (-5.0, 5.0)},
    )

    assert runtime_cfg["caked_roi"] == {
        "enabled": True,
        "half_width_px": 15.0,
        "max_detector_fraction": 0.35,
    }
    assert read_geometry_fit_caked_roi_config({}) == runtime_cfg["caked_roi"]

    disabled_runtime_cfg = build_geometry_fit_runtime_config(
        {"caked_roi": {"half_width_px": 4.0}},
        {"gamma": 1.0},
        {"gamma": {"window": 0.2, "pull": 0.0}},
        {"gamma": (-5.0, 5.0)},
        caked_roi_enabled=False,
    )

    assert disabled_runtime_cfg["caked_roi"] == {
        "enabled": False,
        "half_width_px": 4.0,
        "max_detector_fraction": 0.35,
    }
