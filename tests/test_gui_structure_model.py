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


def _raise(exc):
    raise exc


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
        last_sim_signature="old-image",
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
    assert runtime_state.last_sim_signature is None
    assert runtime_state.last_simulation_signature is None
    assert np.array_equal(runtime_state.sim_miller1_all, np.array([[1.0, 0.0, 1.0]]))
    assert calls == [("filters", {"trigger_update": False}), ("schedule", None)]


def test_primary_cif_reload_helpers_apply_and_restore(monkeypatch, tmp_path) -> None:
    cif_path = tmp_path / "updated.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_updated",
                "_cell_length_a 3.5",
                "_cell_length_c 7.5",
            ]
        ),
        encoding="utf-8",
    )

    state = structure_model.StructureModelState(
        cif_file="original.cif",
        cf="old_cf",
        blk={"old": True},
        cf2="secondary_cf",
        blk2={"_cell_length_a": None, "_cell_length_c": None},
        occupancy_site_labels=["old_site"],
        occupancy_site_expanded_map=[0],
        occ=[0.8],
        occ_vars=["old_occ_var"],
        atom_site_fractional_metadata=[
            {"row_index": 0, "label": "old_site", "x": 0.1, "y": 0.2, "z": 0.3}
        ],
        atom_site_fract_vars=[{"x": "old_x", "y": "old_y", "z": "old_z"}],
        av=3.0,
        cv=9.0,
        av2=4.0,
        cv2=12.0,
        defaults={"a": 3.0, "c": 9.0, "iodine_z": 0.15},
        ht_cache_multi={"p0": {"cached": True}},
    )
    snapshot = structure_model.capture_primary_cif_reload_snapshot(
        state,
        current_occ_values=[0.7],
        current_atom_site_values=[(0.11, 0.22, 0.33)],
    )

    monkeypatch.setattr(
        structure_model,
        "extract_occupancy_site_metadata",
        lambda *_args, **_kwargs: (["I1", "Nb1"], [0, 1, 0]),
    )
    monkeypatch.setattr(
        structure_model,
        "extract_atom_site_fractional_metadata",
        lambda *_args, **_kwargs: [
            {"row_index": 0, "label": "I1", "x": 0.4, "y": 0.5, "z": 0.6}
        ],
    )
    monkeypatch.setattr(
        structure_model,
        "_infer_iodine_z_like_diffuse",
        lambda *_args, **_kwargs: 0.35,
    )

    plan = structure_model.prepare_primary_cif_reload_plan(
        state,
        str(cif_path),
        current_occ_values=snapshot.current_occ_values,
        clamp_site_occupancy_values=lambda values: [
            min(1.0, max(0.0, float(value))) for value in values
        ],
    )

    assert plan.candidate_path == str(cif_path)
    assert plan.occupancy_site_count == 2
    assert plan.occ == [0.7, 0.7]
    assert plan.atom_site_values == [(0.4, 0.5, 0.6)]
    assert plan.iodine_z == 0.35

    structure_model.apply_primary_cif_reload_plan(
        state,
        plan,
        occ_vars=["new_occ_1", "new_occ_2"],
        atom_site_fract_vars=[{"x": "new_x", "y": "new_y", "z": "new_z"}],
        has_second_cif=True,
    )

    assert state.cif_file == str(cif_path)
    assert state.occupancy_site_labels == ["I1", "Nb1"]
    assert state.occupancy_site_expanded_map == [0, 1, 0]
    assert state.occupancy_site_count == 2
    assert state.occ == [0.7, 0.7]
    assert state.atom_site_fractional_metadata == [
        {"row_index": 0, "label": "I1", "x": 0.4, "y": 0.5, "z": 0.6}
    ]
    assert state.av == 3.5
    assert state.cv == 7.5
    assert state.av2 == 3.5
    assert state.cv2 == 7.5
    assert state.defaults["iodine_z"] == 0.35
    assert state.ht_cache_multi == {}

    structure_model.restore_primary_cif_reload_snapshot(
        state,
        snapshot,
        occ_vars=["restored_occ"],
        atom_site_fract_vars=[{"x": "restored_x", "y": "restored_y", "z": "restored_z"}],
    )

    assert state.cif_file == "original.cif"
    assert state.cf == "old_cf"
    assert state.blk == {"old": True}
    assert state.cf2 == "secondary_cf"
    assert state.blk2 == {"_cell_length_a": None, "_cell_length_c": None}
    assert state.occupancy_site_labels == ["old_site"]
    assert state.occupancy_site_expanded_map == [0]
    assert state.occupancy_site_count == 1
    assert state.occ == [0.8]
    assert state.atom_site_fractional_metadata == [
        {"row_index": 0, "label": "old_site", "x": 0.1, "y": 0.2, "z": 0.3}
    ]
    assert state.av == 3.0
    assert state.cv == 9.0
    assert state.av2 == 4.0
    assert state.cv2 == 12.0
    assert state.defaults["a"] == 3.0
    assert state.defaults["c"] == 9.0
    assert state.defaults["iodine_z"] == 0.15
    assert state.ht_cache_multi == {"p0": {"cached": True}}


def test_build_diffuse_ht_request_packages_runtime_inputs(monkeypatch, tmp_path) -> None:
    cif_path = tmp_path / "primary.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_primary",
                "_cell_length_a 1",
                "_cell_length_c 1",
            ]
        ),
        encoding="utf-8",
    )
    state = structure_model.StructureModelState(
        cif_file=str(cif_path),
        cf=None,
        blk=None,
        occupancy_site_labels=["I1", "Nb1"],
        occupancy_site_expanded_map=[0, 1, 0],
        occ=[0.8, 0.6],
        occ_vars=[_Var(0.75), _Var(0.25)],
        defaults={"iodine_z": 0.1},
    )
    override_state = SimpleNamespace(temp_path=None, source_path=None, signature=None)

    monkeypatch.setattr(
        structure_model,
        "_infer_iodine_z_like_diffuse",
        lambda *_args, **_kwargs: 0.42,
    )

    request = structure_model.build_diffuse_ht_request(
        state,
        override_state,
        p_values=[0.1, 0.2, 0.3],
        w_values=[50.0, 30.0, 20.0],
        a_lattice=3.0,
        c_lattice=9.0,
        lambda_angstrom=1.54,
        mx=8,
        two_theta_range=(0.0, 25.0),
        finite_stack=True,
        stack_layers=5,
        phase_delta_expression="0",
        phi_l_divisor=2.0,
    )

    assert request.source_cif == str(cif_path)
    assert request.active_cif == str(cif_path)
    assert request.occ == [0.75, 0.25, 0.75]
    assert request.p_values == [0.1, 0.2, 0.3]
    assert request.w_values == [50.0, 30.0, 20.0]
    assert request.a_lattice == 3.0
    assert request.c_lattice == 9.0
    assert request.lambda_angstrom == 1.54
    assert request.mx == 8
    assert request.two_theta_max == 25.0
    assert request.finite_stack is True
    assert request.stack_layers == 5
    assert request.iodine_z == 0.42
    assert request.phase_delta_expression == "0"
    assert request.phi_l_divisor == 2.0


def test_primary_cif_dialog_helpers_choose_initial_dir_and_apply_selection(tmp_path) -> None:
    current_cif = tmp_path / "nested" / "sample.cif"
    current_cif.parent.mkdir()
    current_cif.write_text("data_test", encoding="utf-8")
    calls = []

    def _askopenfilename(**kwargs):
        calls.append(("dialog", kwargs))
        return str(tmp_path / "updated.cif")

    applied = structure_model.browse_primary_cif_with_dialog(
        current_cif_path=str(current_cif),
        file_dialog_dir=tmp_path / "fallback",
        askopenfilename=_askopenfilename,
        set_cif_path_text=lambda text: calls.append(("set", text)),
        apply_primary_cif_path=lambda text: calls.append(("apply", text)),
    )

    assert applied is True
    assert (
        structure_model.primary_cif_dialog_initial_dir(
            str(current_cif),
            tmp_path / "fallback",
        )
        == str(current_cif.parent)
    )
    assert calls == [
        (
            "dialog",
            {
                "title": "Select Primary CIF",
                "initialdir": str(current_cif.parent),
                "filetypes": [("CIF files", "*.cif *.CIF"), ("All files", "*.*")],
            },
        ),
        ("set", str(tmp_path / "updated.cif")),
        ("apply", str(tmp_path / "updated.cif")),
    ]


def test_open_diffuse_ht_view_with_status_reports_success_and_errors() -> None:
    request = structure_model.DiffuseHTRequest(
        source_cif="C:/data/primary.cif",
        active_cif="C:/data/primary.cif",
        occ=[1.0],
        p_values=[0.1, 0.2, 0.3],
        w_values=[50.0, 30.0, 20.0],
    )
    events = []

    ok = structure_model.open_diffuse_ht_view_with_status(
        build_request=lambda: request,
        open_view=lambda req: events.append(("open", req.source_cif)),
        set_status_text=lambda text: events.append(("status", text)),
    )

    assert ok is True
    assert events == [
        ("open", "C:/data/primary.cif"),
        ("status", "Opened diffuse HT viewer: primary.cif"),
    ]

    errors = []
    ok = structure_model.open_diffuse_ht_view_with_status(
        build_request=lambda: _raise(FileNotFoundError("missing.cif")),
        open_view=lambda req: None,
        set_status_text=lambda text: errors.append(text),
    )
    assert ok is False
    assert errors == ["missing.cif"]

    errors = []
    ok = structure_model.open_diffuse_ht_view_with_status(
        build_request=lambda: _raise(ValueError("bad inputs")),
        open_view=lambda req: None,
        set_status_text=lambda text: errors.append(text),
    )
    assert ok is False
    assert errors == ["Failed to read diffuse HT inputs: bad inputs"]

    errors = []
    ok = structure_model.open_diffuse_ht_view_with_status(
        build_request=lambda: request,
        open_view=lambda req: _raise(RuntimeError("viewer boom")),
        set_status_text=lambda text: errors.append(text),
    )
    assert ok is False
    assert errors == ["Failed to open diffuse HT viewer: viewer boom"]


def test_export_diffuse_ht_txt_with_dialog_reports_status_and_uses_fallback_dir(
    tmp_path,
) -> None:
    source_cif = tmp_path / "source.cif"
    source_cif.write_text("data_test", encoding="utf-8")
    request = structure_model.DiffuseHTRequest(
        source_cif=str(source_cif),
        active_cif=str(source_cif),
        occ=[1.0],
        p_values=[0.1, 0.2, 0.3],
        w_values=[50.0, 30.0, 20.0],
    )
    captured = {}
    messages = []

    def _asksaveasfilename(**kwargs):
        captured["dialog"] = kwargs
        return str(tmp_path / "out.txt")

    ok = structure_model.export_diffuse_ht_txt_with_dialog(
        build_request=lambda: request,
        get_download_dir=lambda: "downloads-dir",
        asksaveasfilename=_asksaveasfilename,
        export_table=lambda save_path, req: (
            captured.update({"save_path": save_path, "request": req}) or 7
        ),
        set_status_text=lambda text: messages.append(text),
    )

    assert ok is True
    assert captured["dialog"]["initialdir"] == "downloads-dir"
    assert captured["dialog"]["initialfile"] == "source_algebraic_ht.txt"
    assert captured["save_path"] == str(tmp_path / "out.txt")
    assert captured["request"] is request
    assert messages == ["Exported algebraic HT table (7 rows): out.txt"]

    def _fallback_save_dialog(**kwargs):
        fallback["dialog"] = kwargs
        return ""

    fallback = {}
    ok = structure_model.export_diffuse_ht_txt_with_dialog(
        build_request=lambda: request,
        get_download_dir=lambda: _raise(RuntimeError("no downloads")),
        asksaveasfilename=_fallback_save_dialog,
        export_table=lambda save_path, req: 0,
        set_status_text=lambda text: messages.append(text),
    )

    assert ok is False
    assert fallback["dialog"]["initialdir"] == str(source_cif.parent)

    errors = []
    ok = structure_model.export_diffuse_ht_txt_with_dialog(
        build_request=lambda: _raise(ValueError("bad export inputs")),
        get_download_dir=lambda: "downloads-dir",
        asksaveasfilename=lambda **kwargs: "",
        export_table=lambda save_path, req: 0,
        set_status_text=lambda text: errors.append(text),
    )

    assert ok is False
    assert errors == ["Failed to read algebraic HT export inputs: bad export inputs"]
