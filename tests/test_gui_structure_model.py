from types import SimpleNamespace

import numpy as np
from pathlib import Path

from ra_sim.gui import structure_model


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def test_extract_atom_site_fractional_metadata_handles_duplicate_labels_and_fraction_text() -> None:
    cif_block = {
        "_atom_site_label": ["I", "I", "Nb"],
        "_atom_site_fract_x": ["1/2", "0.125(2)", "0.75"],
        "_atom_site_fract_y": ["0.25", "0.5", "0.875"],
        "_atom_site_fract_z": ["0.0", "0.125", "1/4"],
    }

    rows = structure_model.extract_atom_site_fractional_metadata(cif_block)

    assert [row["label"] for row in rows] == ["I #1", "I #2", "Nb"]
    assert rows[0]["x"] == 0.5
    assert rows[1]["x"] == 0.125
    assert rows[2]["z"] == 0.25


def test_active_primary_cif_path_writes_override_and_reuses_cached_temp(tmp_path) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a 1",
                "_cell_length_b 1",
                "_cell_length_c 1",
                "loop_",
                "_atom_site_label",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "I1 0.1 0.2 0.3",
            ]
        ),
        encoding="utf-8",
    )
    state = structure_model.StructureModelState(
        cif_file=str(cif_path),
        cf=None,
        blk=None,
        atom_site_fractional_metadata=[
            {"row_index": 0, "label": "I1", "x": 0.1, "y": 0.2, "z": 0.3}
        ],
        atom_site_fract_vars=[
            {
                "x": _Var(0.15),
                "y": _Var(0.25),
                "z": _Var(0.35),
            }
        ],
    )
    override_state = SimpleNamespace(temp_path=None, source_path=None, signature=None)

    override_path = structure_model.active_primary_cif_path(state, override_state)
    override_path_2 = structure_model.active_primary_cif_path(state, override_state)

    assert override_path != str(cif_path)
    assert override_path == override_path_2
    assert "0.15" in Path(override_path).read_text(encoding="utf-8")


def test_build_initial_structure_model_state_initializes_single_cif(monkeypatch) -> None:
    monkeypatch.setattr(
        structure_model,
        "build_ht_cache",
        lambda *_args, **_kwargs: {
            "p": 0.1,
            "occ": (1.0,),
            "ht": {(1, 0): {"L": np.array([1.0]), "I": np.array([2.0])}},
            "qr": {1: {"L": np.array([1.0]), "I": np.array([2.0])}},
            "arrays": (
                np.array([[1.0, 0.0, 1.0]]),
                np.array([2.0]),
                np.array([1]),
                [["d"]],
            ),
            "two_theta_max": 1.0,
            "a": 1.0,
            "c": 2.0,
            "iodine_z": 0.2,
            "phi_l_divisor": 1.0,
            "phase_delta_expression": "0",
            "finite_stack": True,
            "stack_layers": 8,
            "cif_path": "primary.cif",
        },
    )
    monkeypatch.setattr(
        structure_model,
        "ht_dict_to_arrays",
        lambda _curves: (
            np.array([[1.0, 0.0, 1.0]]),
            np.array([2.0]),
            np.array([1]),
            [["d"]],
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "ht_dict_to_qr_dict",
        lambda _curves: {1: {"L": np.array([1.0]), "I": np.array([2.0])}},
    )

    state = structure_model.build_initial_structure_model_state(
        cif_file="primary.cif",
        cf=object(),
        blk={"dummy": True},
        cif_file2=None,
        occupancy_site_labels=["I1"],
        occupancy_site_expanded_map=[],
        occ=[1.0],
        atom_site_fractional_metadata=[],
        av=1.0,
        bv=1.0,
        cv=2.0,
        av2=None,
        cv2=None,
        defaults={
            "a": 1.0,
            "c": 2.0,
            "p0": 0.1,
            "p1": 0.2,
            "p2": 0.3,
            "w0": 50.0,
            "w1": 30.0,
            "w2": 20.0,
            "iodine_z": 0.2,
            "phi_l_divisor": 1.0,
            "phase_delta_expression": "0",
            "finite_stack": True,
            "stack_layers": 8,
        },
        mx=8,
        lambda_angstrom=1.54,
        intensity_threshold=1.0,
        two_theta_range=(0.0, 50.0),
        include_rods_flag=False,
        combine_weighted_intensities=lambda a, _b, **_kwargs: np.asarray(a, dtype=float),
        miller_generator=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected second CIF call")),
        inject_fractional_reflections=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rod injection")),
    )

    assert state.has_second_cif is False
    assert np.array_equal(state.miller, np.array([[1.0, 0.0, 1.0]]))
    assert np.array_equal(state.intensities, np.array([2.0]))
    assert set(state.sim_primary_qr_all) == {1}
    assert np.array_equal(state.sim_primary_qr_all[1]["L"], np.array([1.0]))
    assert np.array_equal(state.sim_primary_qr_all[1]["I"], np.array([2.0]))
    assert state.last_occ_for_ht == [1.0]


def test_rebuild_diffraction_inputs_updates_state_and_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        structure_model,
        "build_ht_cache",
        lambda *_args, **_kwargs: {
            "p": 0.1,
            "occ": (1.0,),
            "ht": {(1, 0): {"L": np.array([1.0]), "I": np.array([5.0])}},
            "qr": {1: {"L": np.array([1.0]), "I": np.array([5.0])}},
            "arrays": (),
            "two_theta_max": 10.0,
            "a": 1.0,
            "c": 2.0,
            "iodine_z": 0.2,
            "phi_l_divisor": 1.0,
            "phase_delta_expression": "0",
            "finite_stack": True,
            "stack_layers": 8,
            "cif_path": "primary.cif",
        },
    )
    monkeypatch.setattr(
        structure_model,
        "combine_ht_dicts",
        lambda _caches, _weights: {(1, 0): {"L": np.array([1.0]), "I": np.array([5.0])}},
    )
    monkeypatch.setattr(
        structure_model,
        "ht_dict_to_qr_dict",
        lambda _curves: {1: {"L": np.array([1.0]), "I": np.array([5.0])}},
    )
    monkeypatch.setattr(
        structure_model,
        "ht_dict_to_arrays",
        lambda _curves: (
            np.array([[1.0, 0.0, 1.0]]),
            np.array([5.0]),
            np.array([2]),
            [["detail"]],
        ),
    )

    state = structure_model.StructureModelState(
        cif_file="primary.cif",
        cf=None,
        blk=None,
        occ=[1.0],
        defaults={"a": 1.0, "c": 2.0},
        mx=8,
        lambda_angstrom=1.54,
        intensity_threshold=1.0,
        two_theta_range=(0.0, 10.0),
        has_second_cif=False,
    )
    runtime_state = SimpleNamespace(
        last_simulation_signature="old",
        sim_miller1_all=None,
        sim_intens1_all=None,
        sim_primary_qr_all=None,
        sim_miller2_all=None,
        sim_intens2_all=None,
    )
    override_state = SimpleNamespace(temp_path=None, source_path=None, signature=None)
    calls = []

    structure_model.rebuild_diffraction_inputs(
        state,
        new_occ=[1.0],
        p_vals=[0.1, 0.2, 0.3],
        weights=[0.5, 0.3, 0.2],
        a_axis=1.0,
        c_axis=2.0,
        finite_stack_flag=True,
        layers=8,
        phase_delta_expression_current="0",
        phi_l_divisor_current=1.0,
        atom_site_values=[],
        iodine_z_current=0.2,
        atom_site_override_state=override_state,
        simulation_runtime_state=runtime_state,
        combine_weighted_intensities=lambda a, _b, **_kwargs: np.asarray(a, dtype=float),
        build_intensity_dataframes=lambda *_args: ("summary", "details"),
        apply_bragg_qr_filters=lambda **kwargs: calls.append(("filters", kwargs)),
        schedule_update=lambda: calls.append(("schedule", None)),
        weight1=1.0,
        weight2=0.0,
        force=True,
        trigger_update=True,
    )

    assert np.array_equal(state.miller, np.array([[1.0, 0.0, 1.0]]))
    assert state.df_summary == "summary"
    assert runtime_state.last_simulation_signature is None
    assert np.array_equal(runtime_state.sim_miller1_all, np.array([[1.0, 0.0, 1.0]]))
    assert calls == [("filters", {"trigger_update": False}), ("schedule", None)]
