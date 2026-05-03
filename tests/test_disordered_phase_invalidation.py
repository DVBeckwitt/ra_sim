from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ra_sim.gui.state import SimulationRuntimeState


class _Var:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def get(self) -> float:
        return float(self.value)


def _write_source(path: Path, marker: str) -> Path:
    path.write_text(
        "\n".join(
            [
                f"data_{marker}",
                "_cell_length_a 4.0",
                "_cell_length_b 4.0",
                "_cell_length_c 7.0",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 120",
                "loop_",
                "_atom_site_label",
                "_atom_site_occupancy",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_type_symbol",
                f"Pb1 1.0 0 0 0 Pb # {marker}",
                "I1 1.0 0.333333 0.666667 0.2675 I",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _patch_env(monkeypatch, tmp_path, source_path: Path):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = SimulationRuntimeState()
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "defaults",
        {"p0": 0.0, "p1": 1.0, "p2": 0.5, "w0": 0.0, "w1": 1.0, "w2": 0.0},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "p0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "p1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "p2_var", _Var(0.5), raising=False)
    monkeypatch.setattr(runtime_session, "w0_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "w1_var", _Var(100.0), raising=False)
    monkeypatch.setattr(runtime_session, "w2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "a_var", _Var(4.0), raising=False)
    monkeypatch.setattr(runtime_session, "c_var", _Var(7.0), raising=False)
    monkeypatch.setattr(runtime_session, "mx", 4, raising=False)
    monkeypatch.setattr(runtime_session, "lambda_", 1.5406, raising=False)
    monkeypatch.setattr(runtime_session, "two_theta_range", (0.0, 70.0), raising=False)
    monkeypatch.setattr(runtime_session, "intensity_threshold", 0.25, raising=False)
    monkeypatch.setattr(runtime_session, "_active_primary_cif_path", lambda: str(source_path))
    monkeypatch.setattr(runtime_session, "_occupancy_control_vars", lambda: [_Var(1.0)])
    monkeypatch.setattr(
        runtime_session,
        "_current_atom_site_fractional_values",
        lambda: [{"x": 0.0, "y": 0.0, "z": 0.0}],
    )
    monkeypatch.setattr(
        runtime_session,
        "_atom_site_fractional_signature",
        lambda values: tuple(
            (round(float(row["x"]), 9), round(float(row["y"]), 9), round(float(row["z"]), 9))
            for row in values
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_disordered_phase_cif_cache_dir",
        lambda: tmp_path / "generated-cache",
    )
    return runtime_session, runtime_state


def _patch_generator(monkeypatch, runtime_session, tmp_path):
    calls: list[tuple[object, object]] = []

    def fake_generate_pbii_ht_shifted_cif(*, source_cif, output_dir, mode):
        calls.append((source_cif, output_dir))
        generated_path = tmp_path / f"generated-{len(calls)}.cif"
        generated_path.write_text("data_generated\n", encoding="utf-8")
        return SimpleNamespace(cif_path=generated_path, a=4.0, c=21.0)

    monkeypatch.setattr(
        runtime_session,
        "generate_pbii_ht_shifted_cif",
        fake_generate_pbii_ht_shifted_cif,
    )
    monkeypatch.setattr(
        runtime_session,
        "miller_generator",
        lambda *args, **kwargs: (np.array([[1.0, 0.0, 0.0]]), np.array([1.0]), None, None),
    )
    return calls


def test_disordered_inventory_invalidates_when_primary_cif_changes(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "first")
    runtime_session, runtime_state = _patch_env(monkeypatch, tmp_path, source)
    calls = _patch_generator(monkeypatch, runtime_session, tmp_path)

    first = runtime_session._geometry_disordered_phase_inventory_payload()
    runtime_state.stored_disordered_phase_max_positions = [np.array([[1.0]])]
    runtime_state.geometry_q_group_entries_cache = [{"key": ("q_group", "disordered_phase", 1, 0)}]
    runtime_state.geometry_q_group_entries_cache_signature = ("stale",)
    _write_source(source, "second")
    second = runtime_session._geometry_disordered_phase_inventory_payload()

    assert first["signature"] != second["signature"]
    assert len(calls) == 2
    assert runtime_state.stored_disordered_phase_max_positions is None
    assert runtime_state.geometry_q_group_entries_cache == []
    assert runtime_state.geometry_q_group_entries_cache_signature is None


def test_disordered_inventory_invalidates_when_stacking_params_change(monkeypatch, tmp_path):
    source = _write_source(tmp_path / "active.cif", "first")
    runtime_session, runtime_state = _patch_env(monkeypatch, tmp_path, source)
    calls = _patch_generator(monkeypatch, runtime_session, tmp_path)

    first = runtime_session._geometry_disordered_phase_inventory_payload()
    runtime_state.stored_disordered_phase_peak_table_lattice = [(4.0, 21.0, "disordered_phase")]
    runtime_session.p1_var.value = 0.75
    second = runtime_session._geometry_disordered_phase_inventory_payload()

    assert first["signature"] != second["signature"]
    assert len(calls) == 2
    assert runtime_state.stored_disordered_phase_peak_table_lattice is None
