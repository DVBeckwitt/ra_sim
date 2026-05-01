from __future__ import annotations

from ra_sim.gui.state import SimulationRuntimeState


def test_runtime_state_has_disordered_phase_fields():
    state = SimulationRuntimeState()

    assert state.stored_disordered_phase_max_positions is None
    assert state.stored_disordered_phase_source_reflection_indices is None
    assert state.stored_disordered_phase_peak_table_lattice is None
    assert state.disordered_phase_inventory_cache is None
    assert state.generated_disordered_phase_cif_path is None
