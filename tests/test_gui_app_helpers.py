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

    assert runtime_cfg["bounds"]["theta_initial"] == [0.0, 30.0]
    assert runtime_cfg["bounds"]["gamma"] == [-5.0, 5.0]
    assert runtime_cfg["priors"] == {}
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

    assert runtime_cfg["bounds"]["center_x"] == [0.0, 99.0]
    assert runtime_cfg["priors"] == {}


def test_app_geometry_fit_value_helpers_delegate_to_shared_runtime_callbacks(
    monkeypatch,
) -> None:
    class _FakeCallbacks:
        var_map = {"gamma": "gamma-var"}

        def current_ui_params(self):
            return {"gamma": 1.5}

        def current_var_names(self):
            return ["gamma", "a"]

    calls: list[object] = []
    old_callbacks = gui_app._geometry_fit_runtime_value_callbacks
    gui_app._geometry_fit_runtime_value_callbacks = None

    monkeypatch.setattr(
        gui_app.gui_geometry_fit,
        "build_runtime_geometry_fit_value_callbacks",
        lambda bindings: (calls.append(bindings) or _FakeCallbacks()),
    )

    try:
        assert gui_app._current_geometry_fit_ui_params() == {"gamma": 1.5}
        assert gui_app._current_geometry_fit_var_names() == ["gamma", "a"]
        assert gui_app._geometry_fit_restore_var_map() == {"gamma": "gamma-var"}
    finally:
        gui_app._geometry_fit_runtime_value_callbacks = old_callbacks

    assert len(calls) == 1


def test_restore_geometry_fit_undo_state_uses_tk_after_cancel_helper(monkeypatch) -> None:
    """Undo restore should clear queued UI update callbacks through shared helper."""

    clear_calls: list[tuple[object, object]] = []
    run_calls: list[str] = []
    pending = object()

    def fake_clear(root: object, token: object) -> None:
        clear_calls.append((root, token))

    def fake_update() -> None:
        run_calls.append("do_update")

    old_clear = gui_app.gui_controllers.clear_tk_after_token
    old_update = gui_app.do_update
    old_update_pending = gui_app.update_pending
    old_profile_cache = gui_app.profile_cache
    old_overlay_state = gui_app.last_geometry_overlay_state
    old_last_sim_signature = gui_app.last_simulation_signature

    monkeypatch.setattr(gui_app.gui_controllers, "clear_tk_after_token", fake_clear)
    monkeypatch.setattr(gui_app, "do_update", fake_update)
    gui_app.update_pending = pending
    try:
        gui_app._restore_geometry_fit_undo_state({})
    finally:
        monkeypatch.setattr(gui_app.gui_controllers, "clear_tk_after_token", old_clear)
        monkeypatch.setattr(gui_app, "do_update", old_update)
        gui_app.update_pending = old_update_pending
        gui_app.profile_cache = old_profile_cache
        gui_app.last_geometry_overlay_state = old_overlay_state
        gui_app.last_simulation_signature = old_last_sim_signature

    assert clear_calls == [(gui_app.root, pending)]
    assert gui_app.update_pending is None
    assert run_calls == ["do_update"]


def test_restore_geometry_fit_undo_state_delegates_to_shared_runtime_helper(
    monkeypatch,
) -> None:
    calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def fake_restore_runtime_state(state: dict[str, object], **kwargs: object) -> dict[str, object]:
        calls.append((state, dict(kwargs)))
        return {}

    monkeypatch.setattr(
        gui_app.gui_geometry_fit,
        "restore_runtime_geometry_fit_undo_state",
        fake_restore_runtime_state,
    )

    restore_state = {"example": "value"}
    gui_app._restore_geometry_fit_undo_state(restore_state)

    assert len(calls) == 1
    passed_state, passed_kwargs = calls[0]
    assert passed_state is restore_state
    assert "replace_profile_cache" in passed_kwargs
    assert "set_last_overlay_state" in passed_kwargs
    assert "run_update" in passed_kwargs
