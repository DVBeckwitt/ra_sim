from ra_sim.gui import geometry_fit


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def test_geometry_fit_param_order_includes_lattice_parameters() -> None:
    order = geometry_fit.GEOMETRY_FIT_PARAM_ORDER
    assert "a" in order
    assert "c" in order


def test_current_geometry_fit_var_names_includes_lattice_parameters() -> None:
    var_names = geometry_fit.current_geometry_fit_var_names(
        fit_zb=False,
        fit_zs=False,
        fit_theta=False,
        fit_psi_z=False,
        fit_chi=False,
        fit_cor=False,
        fit_gamma=False,
        fit_Gamma=False,
        fit_dist=False,
        fit_a=True,
        fit_c=True,
        fit_center_x=False,
        fit_center_y=False,
    )

    assert var_names == ["a", "c"]


def test_current_geometry_fit_ui_params_capture_lattice_parameters() -> None:
    params = geometry_fit.current_geometry_fit_ui_params(
        zb=0.0,
        zs=0.0,
        theta_initial=5.0,
        psi_z=0.1,
        chi=0.2,
        cor_angle=0.3,
        gamma=0.4,
        Gamma=0.5,
        corto_detector=0.06,
        a=4.21,
        c=32.4,
        center_x=100.0,
        center_y=200.0,
    )

    assert params["a"] == 4.21
    assert params["c"] == 32.4


def test_apply_geometry_fit_undo_state_restores_lattice_parameters() -> None:
    a_var = _DummyVar(0.0)
    c_var = _DummyVar(0.0)

    restored = geometry_fit.apply_geometry_fit_undo_state(
        {
            "ui_params": {
                "a": 4.33,
                "c": 33.8,
            }
        },
        var_map={
            "a": a_var,
            "c": c_var,
        },
    )

    assert a_var.get() == 4.33
    assert c_var.get() == 33.8
    assert restored == {
        "profile_cache": {},
        "overlay_state": None,
    }
