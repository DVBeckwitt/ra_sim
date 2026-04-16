from __future__ import annotations

from argparse import _SubParsersAction
from types import SimpleNamespace

import pytest

from ra_sim import cli


def _get_subparser_choices(parser):
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            return action.choices
    raise AssertionError("parser does not define any subcommands")


def _require_fit_geometry_command():
    cmd = getattr(cli, "_cmd_fit_geometry", None)
    if cmd is None:
        pytest.skip("fit-geometry CLI command is not available in this checkout")
    return cmd


def _patch_first_available(monkeypatch, candidates, replacement):
    for name in candidates:
        if hasattr(cli, name):
            monkeypatch.setattr(cli, name, replacement)
            return name
    pytest.skip("geometry-fit runner helper is not available in this checkout")


def test_cli_build_parser_includes_fit_geometry_command() -> None:
    if getattr(cli, "_cmd_fit_geometry", None) is None:
        pytest.skip("fit-geometry CLI command is not available in this checkout")

    parser = cli._build_parser()
    assert "fit-geometry" in _get_subparser_choices(parser)


def test_cli_build_parser_includes_fit_mosaic_shape_command() -> None:
    if getattr(cli, "_cmd_fit_mosaic_shape", None) is None:
        pytest.skip("fit-mosaic-shape CLI command is not available in this checkout")

    parser = cli._build_parser()
    choices = _get_subparser_choices(parser)
    assert "fit-mosaic-shape" in choices
    assert "fit-mosaic" in choices


def test_cmd_fit_geometry_loads_saved_state_runs_fit_and_saves(monkeypatch, tmp_path) -> None:
    cmd = _require_fit_geometry_command()

    input_path = tmp_path / "saved_gui_state.json"
    output_path = tmp_path / "fit_gui_state.json"
    loaded_payload = {
        "type": "ra_sim.gui_state",
        "state": {
            "files": {"background_files": ["bg0.osc"]},
            "geometry": {"manual_pairs": [{"background_index": 0, "entries": []}]},
        },
    }

    events: list[tuple[str, object]] = []
    saved_payload: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "load_gui_state_file",
        lambda path: events.append(("load", str(path))) or loaded_payload,
    )

    def _fake_runner(*args, **kwargs):
        events.append(("run", {"args": args, "kwargs": kwargs}))
        return {
            "type": "ra_sim.gui_state",
            "state": {
                "files": {"background_files": ["bg0.osc"]},
                "geometry": {
                    "manual_pairs": [{"background_index": 0, "entries": []}],
                    "fit_result": "ok",
                },
            },
        }

    _patch_first_available(
        monkeypatch,
        [
            "run_headless_geometry_fit",
            "run_geometry_fit_from_saved_state",
            "run_geometry_fit_from_state",
        ],
        _fake_runner,
    )

    def _fake_save(path, state, **kwargs):
        events.append(("save", str(path), kwargs))
        saved_payload["path"] = str(path)
        saved_payload["state"] = state
        saved_payload["kwargs"] = kwargs
        return {"path": str(path), "state": state}

    monkeypatch.setattr(cli, "save_gui_state_file", _fake_save)

    args = SimpleNamespace(
        state=str(input_path),
        input_state=str(input_path),
        gui_state=str(input_path),
        source_state=str(input_path),
        out_state=str(output_path),
        output_state=str(output_path),
        output=str(output_path),
        in_place=False,
        overwrite=False,
    )

    cmd(args)

    assert events[0] == ("load", str(input_path))
    assert events[1][0] == "run"
    assert events[2] == ("save", str(output_path), {})
    assert saved_payload["path"] == str(output_path)
    assert saved_payload["state"]["geometry"]["fit_result"] == "ok"


def test_cmd_fit_geometry_supports_in_place_saves(monkeypatch, tmp_path) -> None:
    cmd = _require_fit_geometry_command()

    input_path = tmp_path / "saved_gui_state.json"
    loaded_payload = {"type": "ra_sim.gui_state", "state": {"geometry": {}}}

    monkeypatch.setattr(
        cli,
        "load_gui_state_file",
        lambda path: loaded_payload,
    )

    _patch_first_available(
        monkeypatch,
        [
            "run_headless_geometry_fit",
            "run_geometry_fit_from_saved_state",
            "run_geometry_fit_from_state",
        ],
        lambda *args, **kwargs: loaded_payload,
    )

    save_calls: list[tuple[str, object]] = []

    def _fake_save(path, state, **kwargs):
        save_calls.append((str(path), state))
        return {"path": str(path), "state": state}

    monkeypatch.setattr(cli, "save_gui_state_file", _fake_save)

    args = SimpleNamespace(
        state=str(input_path),
        input_state=str(input_path),
        gui_state=str(input_path),
        source_state=str(input_path),
        out_state=None,
        output_state=None,
        output=None,
        in_place=True,
        overwrite=False,
    )

    cmd(args)

    assert save_calls == [(str(input_path), loaded_payload["state"])]


def test_run_headless_geometry_fit_delegates_to_shared_runner_for_geometry_only(
    monkeypatch,
    tmp_path,
) -> None:
    payload = {
        "type": "ra_sim.gui_state",
        "state": {
            "files": {"background_files": ["bg0.osc"]},
            "geometry": {"manual_pairs": [{"background_index": 0, "entries": []}]},
        },
    }
    calls: list[tuple[object, object, object]] = []

    def _fake_shared_runner(state_arg, *, state_path, downloads_dir):
        calls.append((state_arg, state_path, downloads_dir))
        return SimpleNamespace(
            state={
                "files": {"background_files": ["bg0.osc"]},
                "geometry": {"fit_result": "shared"},
            },
            log_path=tmp_path / "shared_geometry_fit.log",
            accepted=True,
            rejection_reason=None,
            rms_px=1.25,
        )

    monkeypatch.setattr(
        cli.shared_headless_geometry_fit,
        "run_headless_geometry_fit",
        _fake_shared_runner,
    )

    state_result, report = cli.run_headless_geometry_fit(
        payload,
        source_path=tmp_path / "state.json",
        output_dir=tmp_path / "artifacts",
    )

    assert calls == [
        (
            payload["state"],
            tmp_path / "state.json",
            tmp_path / "artifacts",
        )
    ]
    assert state_result["geometry"]["fit_result"] == "shared"
    assert report == {
        "accepted": True,
        "log_path": str(tmp_path / "shared_geometry_fit.log"),
        "matched_peaks_path": None,
        "rms_px": 1.25,
    }


def test_run_headless_mosaic_shape_fit_forwards_to_geometry_runner(monkeypatch) -> None:
    if getattr(cli, "run_headless_mosaic_shape_fit", None) is None:
        pytest.skip("headless mosaic-shape runner is not available in this checkout")

    payload = {"type": "ra_sim.gui_state", "state": {"geometry": {}}}
    calls: list[tuple[object, object, object, object]] = []

    def _fake_geometry_runner(
        payload_arg,
        *,
        source_path,
        output_dir,
        run_mosaic_shape_fit=False,
    ):
        calls.append((payload_arg, source_path, output_dir, run_mosaic_shape_fit))
        return payload_arg["state"], {"mosaic_shape_fit": {"accepted": True}}

    monkeypatch.setattr(cli, "run_headless_geometry_fit", _fake_geometry_runner)

    result = cli.run_headless_mosaic_shape_fit(
        payload,
        source_path="state.json",
        output_dir="artifacts",
    )

    assert result == (payload["state"], {"mosaic_shape_fit": {"accepted": True}})
    assert calls == [(payload, "state.json", "artifacts", True)]


def test_run_headless_geometry_fit_mosaic_uses_module_refraction_wrapper(
    monkeypatch,
    tmp_path,
) -> None:
    primary_cif_path = tmp_path / "primary.cif"
    primary_cif_path.write_text("data_test\n", encoding="utf-8")
    background_path = tmp_path / "bg0.osc"
    background_path.write_bytes(b"")
    payload = {
        "type": "ra_sim.gui_state",
        "state": {
            "files": {
                "background_files": [str(background_path)],
                "primary_cif_path": str(primary_cif_path),
            },
            "geometry": {"manual_pairs": [{"background_index": 0, "entries": []}]},
        },
    }

    class _StopAfterRefraction(RuntimeError):
        pass

    refraction_calls: list[tuple[float, str]] = []

    simulation_defaults = cli.HeadlessSimulationDefaults(
        out_path=str(tmp_path / "out.json"),
        image_size=8,
        samples=2,
        vmax=1.0,
        cif_file=str(primary_cif_path),
        geometry=SimpleNamespace(
            pixel_size_m=1.0e-4,
            lambda_angstrom=1.54,
            theta_initial_deg=6.0,
            cor_angle_deg=0.0,
            chi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            sample_width_m=0.0,
            sample_length_m=0.0,
            Gamma_deg=0.0,
            gamma_deg=0.0,
            distance_m=0.1,
            center=[4.0, 4.0],
        ),
        mosaic=SimpleNamespace(),
        debye_waller=SimpleNamespace(x=0.0, y=0.0),
        occ=(1.0,),
        p_values=(0.5,),
        weights=None,
        two_theta_max=1.0,
        ht_max_miller_index=1,
        ht_phase_delta_expression="0.0",
        ht_phi_l_divisor=1.0,
        ht_finite_stack=False,
        ht_stack_layers=1,
        divergence_sigma_rad=0.0,
        bandwidth_sigma=0.0,
        bandwidth_fraction=0.0,
        sample_depth_m=0.0,
    )

    geometry_modules = SimpleNamespace(
        gui_background=SimpleNamespace(),
        gui_background_theta=SimpleNamespace(
            format_background_theta_values=lambda values: "6.0",
            default_geometry_fit_background_selection=lambda *, osc_files: "current",
            apply_background_theta_metadata=lambda **kwargs: True,
            apply_geometry_fit_background_selection=lambda **kwargs: True,
        ),
        gui_controllers=SimpleNamespace(
            clamp_site_occupancy_values=lambda values, fallback_values=None: list(values),
            combine_cif_weighted_intensities=lambda *args, **kwargs: None,
            normalize_stacking_weight_values=lambda values: list(values),
        ),
        gui_geometry_fit=SimpleNamespace(),
        gui_geometry_overlay=SimpleNamespace(),
        gui_geometry_q_group_manager=SimpleNamespace(),
        gui_manual_geometry=SimpleNamespace(),
        gui_structure_model=SimpleNamespace(
            parse_cif_num=lambda value: float(value),
            extract_occupancy_site_metadata=lambda blk, path: ([], {}),
            extract_atom_site_fractional_metadata=lambda blk: [],
            build_initial_structure_model_state=lambda **kwargs: SimpleNamespace(),
            active_primary_cif_path=lambda *_args, **_kwargs: str(primary_cif_path),
            current_iodine_z=lambda *_args, **_kwargs: 0.0,
            rebuild_diffraction_inputs=lambda *args, **kwargs: None,
        ),
        AtomSiteOverrideState=lambda: SimpleNamespace(),
        SimulationRuntimeState=lambda *args, **kwargs: SimpleNamespace(),
    )

    monkeypatch.setattr(cli, "_load_cli_geometry_modules", lambda: geometry_modules)
    monkeypatch.setattr(
        cli,
        "_load_fitting_optimization",
        lambda: SimpleNamespace(
            fit_geometry_parameters=lambda *args, **kwargs: None,
            fit_mosaic_shape_parameters=lambda *args, **kwargs: None,
            simulate_and_compare_hkl=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_stacking_fault_module",
        lambda: SimpleNamespace(
            DEFAULT_PHI_L_DIVISOR=1.0,
            DEFAULT_PHASE_DELTA_EXPRESSION="0.0",
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_diffraction_tools_module",
        lambda: SimpleNamespace(detector_two_theta_max=lambda *args, **kwargs: 30.0),
    )
    monkeypatch.setattr(
        cli,
        "_load_simulation_modules",
        lambda: SimpleNamespace(
            diffraction=SimpleNamespace(
                hit_tables_to_max_positions=lambda *args, **kwargs: None,
                process_peaks_parallel=lambda *args, **kwargs: None,
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_tools_module",
        lambda: SimpleNamespace(
            build_intensity_dataframes=lambda *args, **kwargs: None,
            inject_fractional_reflections=lambda *args, **kwargs: None,
            miller_generator=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_calculation_module",
        lambda: SimpleNamespace(
            resolve_index_of_refraction=pytest.fail(
                "run_headless_geometry_fit() bypassed cli.resolve_index_of_refraction"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_headless_simulation_defaults",
        lambda *, out_path: simulation_defaults,
    )
    monkeypatch.setattr(
        cli,
        "get_instrument_config",
        lambda: {
            "instrument": {
                "detector": {"image_size": 8, "pixel_size_m": 1.0e-4, "intensity_threshold": 1.0},
                "beam": {"sigma_mosaic_fwhm_deg": 0.8, "gamma_mosaic_fwhm_deg": 0.7, "eta": 0.0},
                "sample_orientation": {
                    "theta_initial_deg": 6.0,
                    "cor_deg": 0.0,
                    "chi_deg": 0.0,
                    "psi_deg": 0.0,
                    "psi_z_deg": 0.0,
                    "zb": 0.0,
                    "zs": 0.0,
                    "width_m": 0.0,
                    "length_m": 0.0,
                    "depth_m": 0.0,
                },
                "debye_waller": {"x": 0.0, "y": 0.0},
                "hendricks_teller": {
                    "default_p": [0.5],
                    "default_w": [1.0],
                    "finite_stack": False,
                    "stack_layers": 1,
                    "max_miller_index": 1,
                    "include_rods": False,
                },
                "fit": {},
                "occupancies": {"default": [1.0]},
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "_load_cif_snapshot",
        lambda path: ({"path": path}, {"_cell_length_a": 4.0, "_cell_length_c": 7.0}),
    )
    monkeypatch.setattr(
        cli,
        "resolve_index_of_refraction",
        lambda lambda_m, *, cif_path: refraction_calls.append((float(lambda_m), str(cif_path)))
        or (1.0 + 0.0j),
    )
    monkeypatch.setattr(
        cli,
        "_build_headless_geometry_mosaic_params",
        lambda **kwargs: (_ for _ in ()).throw(_StopAfterRefraction()),
    )

    with pytest.raises(_StopAfterRefraction):
        cli.run_headless_geometry_fit(
            payload,
            source_path=tmp_path / "state.json",
            output_dir=tmp_path / "artifacts",
            run_mosaic_shape_fit=True,
        )

    assert refraction_calls == [(1.54e-10, str(primary_cif_path))]


def test_cmd_fit_mosaic_shape_loads_saved_state_runs_fit_and_saves(monkeypatch, tmp_path) -> None:
    cmd = getattr(cli, "_cmd_fit_mosaic_shape", None)
    if cmd is None:
        pytest.skip("fit-mosaic-shape CLI command is not available in this checkout")

    input_path = tmp_path / "saved_gui_state.json"
    output_path = tmp_path / "mosaic_fit_gui_state.json"
    loaded_payload = {
        "type": "ra_sim.gui_state",
        "state": {
            "files": {"background_files": ["bg0.osc"]},
            "geometry": {
                "manual_pairs": [{"background_index": 0, "entries": []}],
            },
        },
    }

    events: list[tuple[str, object]] = []
    saved_payload: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "load_gui_state_file",
        lambda path: events.append(("load", str(path))) or loaded_payload,
    )

    def _fake_runner(*args, **kwargs):
        events.append(("run", {"args": args, "kwargs": kwargs}))
        return (
            {
                "files": {"background_files": ["bg0.osc"]},
                "geometry": {
                    "manual_pairs": [{"background_index": 0, "entries": []}],
                    "fit_result": "ok",
                },
                "variables": {
                    "sigma_mosaic_var": 0.25,
                    "gamma_mosaic_var": 0.35,
                    "eta_var": 0.45,
                },
            },
            {
                "log_path": str(tmp_path / "geometry_fit.log"),
                "matched_peaks_path": str(tmp_path / "matched.csv"),
                "mosaic_shape_fit": {
                    "log_path": str(tmp_path / "mosaic_fit.log"),
                    "sigma_mosaic_deg": 0.25,
                    "gamma_mosaic_deg": 0.35,
                    "eta": 0.45,
                },
            },
        )

    monkeypatch.setattr(cli, "run_headless_mosaic_shape_fit", _fake_runner)

    def _fake_save(path, state, **kwargs):
        events.append(("save", str(path), kwargs))
        saved_payload["path"] = str(path)
        saved_payload["state"] = state
        saved_payload["kwargs"] = kwargs
        return {"path": str(path), "state": state}

    monkeypatch.setattr(cli, "save_gui_state_file", _fake_save)

    args = SimpleNamespace(
        state=str(input_path),
        input_state=str(input_path),
        gui_state=str(input_path),
        source_state=str(input_path),
        out_state=str(output_path),
        output_state=str(output_path),
        output=str(output_path),
        in_place=False,
        overwrite=False,
    )

    cmd(args)

    assert events[0] == ("load", str(input_path))
    assert events[1][0] == "run"
    assert events[2] == ("save", str(output_path), {})
    assert saved_payload["path"] == str(output_path)
    assert saved_payload["state"]["geometry"]["fit_result"] == "ok"
