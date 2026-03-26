from ra_sim.gui import controllers, state, structure_factor_pruning


class _FakeVar:
    def __init__(self, value=None):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _bindings(
    *,
    view_state=None,
    simulation_runtime_state=None,
    bragg_qr_manager_state=None,
    apply_filters=None,
    schedule_update=None,
    refresh_window=None,
):
    return structure_factor_pruning.StructureFactorPruningRuntimeBindings(
        view_state=(
            view_state
            if view_state is not None
            else state.StructureFactorPruningControlsViewState()
        ),
        simulation_runtime_state=(
            simulation_runtime_state
            if simulation_runtime_state is not None
            else state.SimulationRuntimeState()
        ),
        bragg_qr_manager_state=(
            bragg_qr_manager_state
            if bragg_qr_manager_state is not None
            else state.BraggQrManagerState()
        ),
        clip_prune_bias=lambda value: controllers.clip_structure_factor_prune_bias(
            value,
            fallback=0.0,
            minimum=-2.0,
            maximum=2.0,
        ),
        clip_solve_q_steps=lambda value: controllers.clip_solve_q_steps(
            value,
            fallback=32,
            minimum=4,
            maximum=128,
        ),
        clip_solve_q_rel_tol=lambda value: controllers.clip_solve_q_rel_tol(
            value,
            fallback=1.0e-3,
            minimum=1.0e-6,
            maximum=1.0e-2,
        ),
        normalize_solve_q_mode_label=controllers.normalize_solve_q_mode_label,
        apply_filters=apply_filters or controllers.apply_bragg_qr_filters,
        schedule_update=schedule_update,
        refresh_window=refresh_window,
    )


def test_structure_factor_pruning_runtime_current_value_helpers_and_status_text() -> None:
    view_state = state.StructureFactorPruningControlsViewState(
        sf_prune_bias_var=_FakeVar("0.25"),
        sf_prune_status_var=_FakeVar(""),
        solve_q_mode_var=_FakeVar("robust"),
        solve_q_steps_var=_FakeVar("12.6"),
        solve_q_rel_tol_var=_FakeVar("1e-4"),
    )
    simulation_runtime_state = state.SimulationRuntimeState(
        sf_prune_stats={
            "qr_total": 20,
            "qr_kept": 12,
            "hkl_primary_total": 50,
            "hkl_primary_kept": 30,
        }
    )
    bindings = _bindings(
        view_state=view_state,
        simulation_runtime_state=simulation_runtime_state,
    )

    assert structure_factor_pruning.current_runtime_sf_prune_bias(bindings) == 0.25
    assert structure_factor_pruning.current_runtime_solve_q_steps(bindings) == 13
    assert (
        structure_factor_pruning.current_runtime_solve_q_rel_tol(bindings) == 1.0e-4
    )
    assert (
        structure_factor_pruning.current_runtime_solve_q_mode_label(bindings)
        == "adaptive"
    )

    text = structure_factor_pruning.update_runtime_structure_factor_pruning_status(
        bindings
    )
    assert text == "SF pruning keeps 12/20 rod points (60.0%), bias=+0.25"
    assert view_state.sf_prune_status_var.get() == text


def test_apply_runtime_bragg_qr_filters_updates_status_invalidates_and_refreshes() -> None:
    view_state = state.StructureFactorPruningControlsViewState(
        sf_prune_bias_var=_FakeVar("0.5"),
        sf_prune_status_var=_FakeVar(""),
    )
    simulation_runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("sig", 1),
        stored_max_positions_local=[(1.0, 2.0)],
        stored_sim_image="sim",
        stored_peak_table_lattice=["lattice"],
        selected_peak_record={"hkl": (1, 0, 2)},
    )
    applied = []
    scheduled = []
    refreshed = []

    def fake_apply_filters(simulation_state, manager_state, *, prune_bias):
        simulation_state.sf_prune_stats = {
            "qr_total": 10,
            "qr_kept": 7,
            "hkl_primary_total": 0,
            "hkl_primary_kept": 0,
        }
        applied.append((simulation_state, manager_state, prune_bias))
        return dict(simulation_state.sf_prune_stats)

    bindings = _bindings(
        view_state=view_state,
        simulation_runtime_state=simulation_runtime_state,
        apply_filters=fake_apply_filters,
        schedule_update=lambda: scheduled.append(True),
        refresh_window=lambda: refreshed.append(True),
    )

    stats = structure_factor_pruning.apply_runtime_bragg_qr_filters(
        bindings,
        trigger_update=True,
    )

    assert stats == {
        "qr_total": 10,
        "qr_kept": 7,
        "hkl_primary_total": 0,
        "hkl_primary_kept": 0,
    }
    assert applied[0][0] is simulation_runtime_state
    assert applied[0][2] == 0.5
    assert simulation_runtime_state.last_simulation_signature is None
    assert simulation_runtime_state.stored_max_positions_local is None
    assert simulation_runtime_state.stored_sim_image is None
    assert simulation_runtime_state.stored_peak_table_lattice is None
    assert simulation_runtime_state.selected_peak_record is None
    assert scheduled == [True]
    assert refreshed == [True]
    assert view_state.sf_prune_status_var.get() == (
        "SF pruning keeps 7/10 rod points (70.0%), bias=+0.50"
    )


def test_structure_factor_pruning_runtime_bias_change_clips_then_applies(monkeypatch) -> None:
    view_state = state.StructureFactorPruningControlsViewState(
        sf_prune_bias_var=_FakeVar("99"),
        sf_prune_status_var=_FakeVar(""),
    )
    bindings = _bindings(view_state=view_state)
    calls = []

    monkeypatch.setattr(
        structure_factor_pruning,
        "apply_runtime_bragg_qr_filters",
        lambda bindings_arg, **kwargs: calls.append((bindings_arg, kwargs)) or {},
    )

    changed = structure_factor_pruning.on_runtime_sf_prune_bias_change(bindings)
    assert changed is False
    assert view_state.sf_prune_bias_var.get() == 2.0
    assert calls == []

    view_state.sf_prune_bias_var.set("0.25")
    changed = structure_factor_pruning.on_runtime_sf_prune_bias_change(bindings)
    assert changed is True
    assert calls == [(bindings, {"trigger_update": True})]


def test_structure_factor_pruning_runtime_steps_and_rel_tol_changes_clip_and_schedule() -> None:
    view_state = state.StructureFactorPruningControlsViewState(
        solve_q_steps_var=_FakeVar("999"),
        solve_q_rel_tol_var=_FakeVar("1.0"),
    )
    simulation_runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("sig", 1),
        stored_max_positions_local=[1],
        stored_sim_image="sim",
        stored_peak_table_lattice=["lattice"],
        selected_peak_record={"hkl": (1, 0, 2)},
    )
    scheduled = []
    bindings = _bindings(
        view_state=view_state,
        simulation_runtime_state=simulation_runtime_state,
        schedule_update=lambda: scheduled.append(True),
    )

    changed = structure_factor_pruning.on_runtime_solve_q_steps_change(bindings)
    assert changed is False
    assert view_state.solve_q_steps_var.get() == 128.0
    assert scheduled == []

    view_state.solve_q_steps_var.set("32")
    changed = structure_factor_pruning.on_runtime_solve_q_steps_change(bindings)
    assert changed is True
    assert simulation_runtime_state.last_simulation_signature is None
    assert simulation_runtime_state.stored_max_positions_local is None
    assert simulation_runtime_state.stored_sim_image is None
    assert simulation_runtime_state.stored_peak_table_lattice is None
    assert simulation_runtime_state.selected_peak_record is None
    assert scheduled == [True]

    simulation_runtime_state.last_simulation_signature = ("sig", 2)
    simulation_runtime_state.stored_max_positions_local = [1]
    simulation_runtime_state.stored_sim_image = "sim"
    simulation_runtime_state.stored_peak_table_lattice = ["lattice"]
    simulation_runtime_state.selected_peak_record = {"hkl": (1, 0, 2)}
    view_state.solve_q_rel_tol_var.set("1.0")

    changed = structure_factor_pruning.on_runtime_solve_q_rel_tol_change(bindings)
    assert changed is False
    assert view_state.solve_q_rel_tol_var.get() == 0.01
    assert scheduled == [True]

    view_state.solve_q_rel_tol_var.set("1e-4")
    changed = structure_factor_pruning.on_runtime_solve_q_rel_tol_change(bindings)
    assert changed is True
    assert simulation_runtime_state.last_simulation_signature is None
    assert simulation_runtime_state.stored_max_positions_local is None
    assert simulation_runtime_state.stored_sim_image is None
    assert simulation_runtime_state.stored_peak_table_lattice is None
    assert simulation_runtime_state.selected_peak_record is None
    assert scheduled == [True, True]


def test_structure_factor_pruning_runtime_mode_change_normalizes_and_syncs_view(
    monkeypatch,
) -> None:
    view_state = state.StructureFactorPruningControlsViewState(
        solve_q_mode_var=_FakeVar("robust"),
    )
    simulation_runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("sig", 1),
        stored_max_positions_local=[1],
        stored_sim_image="sim",
        stored_peak_table_lattice=["lattice"],
        selected_peak_record={"hkl": (1, 0, 2)},
    )
    scheduled = []
    enabled_calls = []
    bindings = _bindings(
        view_state=view_state,
        simulation_runtime_state=simulation_runtime_state,
        schedule_update=lambda: scheduled.append(True),
    )

    monkeypatch.setattr(
        structure_factor_pruning.gui_views,
        "set_structure_factor_pruning_rel_tol_enabled",
        lambda _view_state, *, enabled: enabled_calls.append(enabled),
    )

    changed = structure_factor_pruning.on_runtime_solve_q_mode_change(bindings)
    assert changed is False
    assert view_state.solve_q_mode_var.get() == "adaptive"
    assert enabled_calls == []
    assert scheduled == []

    changed = structure_factor_pruning.on_runtime_solve_q_mode_change(bindings)
    assert changed is True
    assert enabled_calls == [True]
    assert simulation_runtime_state.last_simulation_signature is None
    assert simulation_runtime_state.stored_max_positions_local is None
    assert simulation_runtime_state.stored_sim_image is None
    assert simulation_runtime_state.stored_peak_table_lattice is None
    assert simulation_runtime_state.selected_peak_record is None
    assert scheduled == [True]
