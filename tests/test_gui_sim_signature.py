from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import app as gui_app
from ra_sim.gui._runtime import runtime_session


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


class _FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_geometry_source_signature_includes_events_per_beam_phase():
    params_a = {
        "events_per_beam_phase": 50,
        "solve_q_steps": 32,
        "solve_q_rel_tol": 1e-6,
        "solve_q_mode": 0,
        "beam_x_array": [0.0],
        "theta_array": [0.0],
        "_sampling_signature": (),
    }
    params_b = dict(params_a)
    params_b["events_per_beam_phase"] = 75

    sig_a = runtime_session._geometry_source_snapshot_signature_from_params(
        {},
        mosaic_params=params_a,
        optics_mode_component=0,
        sf_prune_bias=0.0,
        sf_prune_stats={},
        ordered_structure_scale=1.0,
        qr_cylinder_replace_requested=False,
        primary_source_signature=None,
        secondary_source_signature=None,
        secondary_a=0.0,
        secondary_c=0.0,
    )
    sig_b = runtime_session._geometry_source_snapshot_signature_from_params(
        {},
        mosaic_params=params_b,
        optics_mode_component=0,
        sf_prune_bias=0.0,
        sf_prune_stats={},
        ordered_structure_scale=1.0,
        qr_cylinder_replace_requested=False,
        primary_source_signature=None,
        secondary_source_signature=None,
        secondary_a=0.0,
        secondary_c=0.0,
    )

    assert sig_a != sig_b


def test_apply_events_per_beam_phase_invalidates_simulation_without_regenerating_mosaic(monkeypatch):
    runtime_session.DEFAULT_EVENTS_PER_BEAM_PHASE = 50
    runtime_session.MIN_EVENTS_PER_BEAM_PHASE = 1
    runtime_session.MAX_EVENTS_PER_BEAM_PHASE = 1000
    runtime_session.defaults = {"events_per_beam_phase": 50}
    runtime_session.sampling_optics_controls_view_state = SimpleNamespace(
        events_per_phase_var=_FakeVar(75),
        events_per_phase_independent_var=_FakeVar(True),
        events_per_phase_value_var=_FakeVar(""),
        events_per_phase_scale=None,
    )
    runtime_session.simulation_runtime_state = SimpleNamespace(
        last_sim_signature=("old",),
        last_simulation_signature=("old",),
        worker_ready_result=object(),
    )

    invalidated = {"count": 0}
    scheduled = {"count": 0}
    update_mosaic = {"count": 0}

    monkeypatch.setattr(runtime_session, "_refresh_resolution_display", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.__setitem__("count", invalidated["count"] + 1),
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: scheduled.__setitem__("count", scheduled["count"] + 1),
    )
    monkeypatch.setattr(
        runtime_session,
        "update_mosaic_cache",
        lambda: update_mosaic.__setitem__("count", update_mosaic["count"] + 1),
    )

    runtime_session._apply_events_per_beam_phase(trigger_update=True)

    assert runtime_session.defaults["events_per_beam_phase"] == 75
    assert invalidated["count"] == 1
    assert scheduled["count"] == 1
    assert update_mosaic["count"] == 0
    assert runtime_session.simulation_runtime_state.last_sim_signature is None
    assert runtime_session.simulation_runtime_state.last_simulation_signature is None
    assert runtime_session.simulation_runtime_state.worker_ready_result is None


def test_events_slider_control_cannot_produce_zero(monkeypatch):
    runtime_session.DEFAULT_EVENTS_PER_BEAM_PHASE = 50
    runtime_session.MIN_EVENTS_PER_BEAM_PHASE = 1
    runtime_session.MAX_EVENTS_PER_BEAM_PHASE = 1000
    runtime_session.defaults = {"events_per_beam_phase": 50}
    runtime_session.sampling_optics_controls_view_state = SimpleNamespace(
        events_per_phase_var=_FakeVar(0),
        events_per_phase_independent_var=_FakeVar(True),
        events_per_phase_value_var=_FakeVar(""),
        events_per_phase_scale=None,
    )
    monkeypatch.setattr(runtime_session, "_refresh_resolution_display", lambda: None)

    normalized = runtime_session._normalize_events_per_beam_phase_control(default=50)

    assert normalized == 1
    assert runtime_session.defaults["events_per_beam_phase"] == 1
    assert runtime_session.sampling_optics_controls_view_state.events_per_phase_var.get() == 1


def _install_sampling_control_test_state(
    *,
    sample_count: int,
    events_per_phase: int,
    independent: bool,
) -> None:
    runtime_session.DEFAULT_RANDOM_SAMPLE_COUNT = 75
    runtime_session.MIN_RANDOM_SAMPLE_COUNT = 1
    runtime_session.MAX_RANDOM_SAMPLE_COUNT = 5000
    runtime_session.DEFAULT_EVENTS_PER_BEAM_PHASE = 75
    runtime_session.MIN_EVENTS_PER_BEAM_PHASE = 1
    runtime_session.MAX_EVENTS_PER_BEAM_PHASE = 1000
    runtime_session.CUSTOM_SAMPLING_OPTION = "Custom"
    runtime_session.defaults = {
        "sampling_count": 75,
        "events_per_beam_phase": events_per_phase,
    }
    runtime_session.resolution_var = _FakeVar("Custom")
    runtime_session.sampling_optics_controls_view_state = SimpleNamespace(
        sample_count_var=_FakeVar(sample_count),
        sample_count_scale=None,
        sample_count_value_var=_FakeVar(""),
        events_per_phase_var=_FakeVar(events_per_phase),
        events_per_phase_independent_var=_FakeVar(independent),
        events_per_phase_independent_checkbutton=None,
        events_per_phase_value_var=_FakeVar(""),
        events_per_phase_scale=None,
    )
    runtime_session.simulation_runtime_state = SimpleNamespace(
        num_samples=75,
        last_sim_signature=("old",),
        last_simulation_signature=("old",),
        worker_ready_result=object(),
    )


def test_linked_events_per_beam_phase_tracks_sample_count(monkeypatch):
    _install_sampling_control_test_state(
        sample_count=125,
        events_per_phase=40,
        independent=False,
    )
    monkeypatch.setattr(runtime_session, "_refresh_resolution_display", lambda: None)

    normalized = runtime_session._normalize_random_sample_count_control(default=75)

    assert normalized == 125
    assert runtime_session.defaults["sampling_count"] == 125
    assert runtime_session.defaults["events_per_beam_phase"] == 125
    assert runtime_session.sampling_optics_controls_view_state.events_per_phase_var.get() == 125


def test_independent_events_per_beam_phase_stops_tracking_sample_count(monkeypatch):
    _install_sampling_control_test_state(
        sample_count=125,
        events_per_phase=40,
        independent=True,
    )
    monkeypatch.setattr(runtime_session, "_refresh_resolution_display", lambda: None)

    normalized = runtime_session._normalize_random_sample_count_control(default=75)

    assert normalized == 125
    assert runtime_session.defaults["sampling_count"] == 125
    assert runtime_session.defaults["events_per_beam_phase"] == 40
    assert runtime_session.sampling_optics_controls_view_state.events_per_phase_var.get() == 40


def test_turning_off_independent_events_relinks_and_invalidates(monkeypatch):
    _install_sampling_control_test_state(
        sample_count=125,
        events_per_phase=40,
        independent=False,
    )
    invalidated = {"count": 0}
    scheduled = {"count": 0}
    monkeypatch.setattr(runtime_session, "_refresh_resolution_display", lambda: None)
    monkeypatch.setattr(
        runtime_session,
        "_invalidate_geometry_manual_pick_cache",
        lambda: invalidated.__setitem__("count", invalidated["count"] + 1),
    )
    monkeypatch.setattr(
        runtime_session,
        "schedule_update",
        lambda: scheduled.__setitem__("count", scheduled["count"] + 1),
    )

    runtime_session._on_events_per_beam_phase_independent_change()

    assert runtime_session.defaults["events_per_beam_phase"] == 125
    assert runtime_session.sampling_optics_controls_view_state.events_per_phase_var.get() == 125
    assert invalidated["count"] == 1
    assert scheduled["count"] == 1
    assert runtime_session.simulation_runtime_state.last_sim_signature is None
    assert runtime_session.simulation_runtime_state.last_simulation_signature is None
    assert runtime_session.simulation_runtime_state.worker_ready_result is None
