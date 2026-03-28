from __future__ import annotations

from ra_sim.gui import app as gui_app


def test_packaged_gui_signature_includes_rounded_psi_z() -> None:
    previous_value = float(gui_app.psi_z_var.get())
    try:
        gui_app.psi_z_var.set(1.23456789)

        assert gui_app.get_sim_signature() == (1.234568,)
    finally:
        gui_app.psi_z_var.set(previous_value)


def test_packaged_gui_psi_z_slider_is_limited_and_clamped() -> None:
    previous_value = float(gui_app.psi_z_var.get())
    try:
        assert float(gui_app.psi_z_scale.cget("from")) == -5.0
        assert float(gui_app.psi_z_scale.cget("to")) == 5.0

        gui_app.psi_z_var.set(9.0)
        assert float(gui_app.psi_z_var.get()) == 5.0

        gui_app.psi_z_var.set(-9.0)
        assert float(gui_app.psi_z_var.get()) == -5.0
    finally:
        gui_app.psi_z_var.set(previous_value)
