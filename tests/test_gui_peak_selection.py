import numpy as np
from types import SimpleNamespace

from ra_sim.gui import peak_selection, state


class _FakeVar:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = str(value)


class _FakeMarker:
    def __init__(self):
        self.data = None
        self.visible = False

    def set_data(self, x_vals, y_vals):
        self.data = (list(x_vals), list(y_vals))

    def set_visible(self, visible):
        self.visible = bool(visible)


class _FakeManager:
    def __init__(self):
        self.title = None
        self.shown = False

    def set_window_title(self, title):
        self.title = title

    def show(self):
        self.shown = True


class _FakeCanvas:
    def __init__(self, manager=None):
        self.manager = manager


class _FakeFigure:
    def __init__(self, manager=None):
        self.canvas = _FakeCanvas(manager=manager)
        self.shown = False

    def show(self):
        self.shown = True


def _intersection_config() -> peak_selection.SelectedPeakIntersectionConfig:
    return peak_selection.build_selected_peak_intersection_config(
        image_size=512,
        center_col=255.5,
        center_row=256.5,
        distance_cor_to_detector=123.0,
        gamma_deg=1.5,
        Gamma_deg=2.5,
        chi_deg=3.5,
        psi_deg=4.5,
        psi_z_deg=5.5,
        zs=6.5,
        zb=7.5,
        theta_initial_deg=8.5,
        cor_angle_deg=9.5,
        sigma_mosaic_deg=0.25,
        gamma_mosaic_deg=0.5,
        eta=0.75,
        sample_width_m=0.012,
        sample_length_m=0.018,
        wavelength_angstrom=1.5406,
        solve_q_steps=321,
        solve_q_rel_tol=1.0e-4,
        solve_q_mode=2,
    )


def _canvas_pick_config(
    *,
    image_shape: tuple[int, ...] | None = (64, 64),
) -> peak_selection.SelectedPeakCanvasPickConfig:
    return peak_selection.build_selected_peak_canvas_pick_config(
        image_size=64,
        primary_a=5.0,
        primary_c=7.0,
        max_distance_px=12.0,
        min_separation_px=2.0,
        image_shape=image_shape,
    )


def _ideal_center_probe_config(
    *,
    lattice_a: float = 5.0,
    lattice_c: float = 7.0,
) -> peak_selection.SelectedPeakIdealCenterProbeConfig:
    return peak_selection.build_selected_peak_ideal_center_probe_config(
        image_size=64,
        lattice_a=lattice_a,
        lattice_c=lattice_c,
        wavelength=1.54,
        distance_cor_to_detector=123.0,
        gamma_deg=1.5,
        Gamma_deg=2.5,
        chi_deg=3.5,
        psi_deg=4.5,
        psi_z_deg=5.5,
        zs=6.5,
        zb=7.5,
        debye_x=0.1,
        debye_y=0.2,
        detector_center=(255.5, 256.5),
        theta_initial_deg=8.5,
        cor_angle_deg=9.5,
        optics_mode=2,
        solve_q_steps=321,
        solve_q_rel_tol=1.0e-4,
        solve_q_mode=1,
    )


def test_peak_selection_config_builders_normalize_values() -> None:
    canvas_cfg = peak_selection.build_selected_peak_canvas_pick_config(
        image_size="64",
        primary_a="5.0",
        primary_c=7,
        max_distance_px="12.5",
        min_separation_px=2,
        image_shape=[80, 96],
    )
    assert isinstance(canvas_cfg, peak_selection.SelectedPeakCanvasPickConfig)
    assert canvas_cfg.image_size == 64
    assert canvas_cfg.primary_a == 5.0
    assert canvas_cfg.primary_c == 7.0
    assert canvas_cfg.max_distance_px == 12.5
    assert canvas_cfg.min_separation_px == 2.0
    assert canvas_cfg.image_shape == (80, 96)

    intersection_cfg = peak_selection.build_selected_peak_intersection_config(
        image_size="512",
        center_col="255.5",
        center_row=256.5,
        distance_cor_to_detector="123.0",
        gamma_deg=1.5,
        Gamma_deg="2.5",
        chi_deg=3.5,
        psi_deg=4.5,
        psi_z_deg=5.5,
        zs=6.5,
        zb=7.5,
        theta_initial_deg=8.5,
        cor_angle_deg=9.5,
        sigma_mosaic_deg="0.25",
        gamma_mosaic_deg=0.5,
        eta="0.75",
        sample_width_m="0.012",
        sample_length_m=0.018,
        wavelength_angstrom="1.5406",
        solve_q_steps="321",
        solve_q_rel_tol="1.0e-4",
        solve_q_mode="2",
    )
    assert isinstance(intersection_cfg, peak_selection.SelectedPeakIntersectionConfig)
    assert intersection_cfg.image_size == 512
    assert intersection_cfg.center_col == 255.5
    assert intersection_cfg.solve_q_steps == 321
    assert intersection_cfg.solve_q_rel_tol == 1.0e-4
    assert intersection_cfg.solve_q_mode == 2
    assert intersection_cfg.sample_width_m == 0.012
    assert intersection_cfg.sample_length_m == 0.018
    assert intersection_cfg.wavelength_angstrom == 1.5406

    probe_cfg = peak_selection.build_selected_peak_ideal_center_probe_config(
        image_size="64",
        lattice_a="5.0",
        lattice_c=7,
        wavelength="1.54",
        distance_cor_to_detector=123,
        gamma_deg="1.5",
        Gamma_deg=2.5,
        chi_deg=3.5,
        psi_deg=4.5,
        psi_z_deg=5.5,
        zs=6.5,
        zb=7.5,
        debye_x="0.1",
        debye_y=0.2,
        detector_center=("255.5", 256.5),
        theta_initial_deg=8.5,
        cor_angle_deg=9.5,
        optics_mode="2",
        solve_q_steps="321",
        solve_q_rel_tol="1.0e-4",
        solve_q_mode="1",
        unit_x=[1, 0, 0],
        n_detector=[0, 1, 0],
    )
    assert isinstance(probe_cfg, peak_selection.SelectedPeakIdealCenterProbeConfig)
    assert probe_cfg.image_size == 64
    assert probe_cfg.lattice_a == 5.0
    assert probe_cfg.lattice_c == 7.0
    assert probe_cfg.detector_center == (255.5, 256.5)
    assert probe_cfg.unit_x == (1.0, 0.0, 0.0)
    assert probe_cfg.n_detector == (0.0, 1.0, 0.0)


def test_peak_selection_runtime_config_factories_read_live_values() -> None:
    live = {
        "primary_a": 5.0,
        "primary_c": 7.0,
        "image_shape": None,
        "center_col": 255.5,
        "center_row": 256.5,
        "distance": 123.0,
        "gamma": 1.5,
        "Gamma": 2.5,
        "chi": 3.5,
        "psi": 4.5,
        "psi_z": 5.5,
        "zs": 6.5,
        "zb": 7.5,
        "theta_initial": 8.5,
        "cor_angle": 9.5,
        "sigma_mosaic": 0.25,
        "gamma_mosaic": 0.5,
        "eta": 0.75,
        "sample_width": 0.012,
        "sample_length": 0.018,
        "pixel_size_m": 172.0e-6,
        "wavelength_angstrom": 1.5406,
        "solve_q": SimpleNamespace(steps=321, rel_tol=1.0e-4, mode_flag=2),
    }

    canvas_factory = peak_selection.make_runtime_selected_peak_canvas_pick_config_factory(
        image_size=64,
        primary_a_factory=lambda: live["primary_a"],
        primary_c_factory=lambda: live["primary_c"],
        max_distance_px=12.0,
        min_separation_px=2.0,
        image_shape_factory=lambda: live["image_shape"],
    )
    intersection_factory = (
        peak_selection.make_runtime_selected_peak_intersection_config_factory(
            image_size=512,
            center_col_factory=lambda: live["center_col"],
            center_row_factory=lambda: live["center_row"],
            distance_cor_to_detector_factory=lambda: live["distance"],
            gamma_deg_factory=lambda: live["gamma"],
            Gamma_deg_factory=lambda: live["Gamma"],
            chi_deg_factory=lambda: live["chi"],
            psi_deg_factory=lambda: live["psi"],
            psi_z_deg_factory=lambda: live["psi_z"],
            zs_factory=lambda: live["zs"],
            zb_factory=lambda: live["zb"],
            theta_initial_deg_factory=lambda: live["theta_initial"],
            cor_angle_deg_factory=lambda: live["cor_angle"],
            sigma_mosaic_deg_factory=lambda: live["sigma_mosaic"],
            gamma_mosaic_deg_factory=lambda: live["gamma_mosaic"],
            eta_factory=lambda: live["eta"],
            sample_width_m_factory=lambda: live["sample_width"],
            sample_length_m_factory=lambda: live["sample_length"],
            pixel_size_m_factory=lambda: live["pixel_size_m"],
            wavelength_angstrom_factory=lambda: live["wavelength_angstrom"],
            solve_q_values_factory=lambda: live["solve_q"],
        )
    )

    canvas_cfg = canvas_factory()
    intersection_cfg = intersection_factory()

    assert canvas_cfg.image_shape == (64, 64)
    assert canvas_cfg.primary_a == 5.0
    assert canvas_cfg.primary_c == 7.0
    assert intersection_cfg.center_col == 255.5
    assert intersection_cfg.center_row == 256.5
    assert intersection_cfg.solve_q_steps == 321
    assert intersection_cfg.solve_q_rel_tol == 1.0e-4
    assert intersection_cfg.solve_q_mode == 2
    assert intersection_cfg.sample_width_m == 0.012
    assert intersection_cfg.sample_length_m == 0.018
    assert intersection_cfg.pixel_size_m == 172.0e-6
    assert intersection_cfg.wavelength_angstrom == 1.5406

    live["primary_a"] = 6.0
    live["image_shape"] = (80, 96)
    live["pixel_size_m"] = 90.0e-6
    live["solve_q"] = SimpleNamespace(steps=123, rel_tol=2.5e-4, mode_flag=1)

    canvas_cfg = canvas_factory()
    intersection_cfg = intersection_factory()

    assert canvas_cfg.primary_a == 6.0
    assert canvas_cfg.image_shape == (80, 96)
    assert intersection_cfg.pixel_size_m == 90.0e-6
    assert intersection_cfg.solve_q_steps == 123
    assert intersection_cfg.solve_q_rel_tol == 2.5e-4
    assert intersection_cfg.solve_q_mode == 1


def test_peak_selection_runtime_ideal_center_factory_builds_live_probe_config(
    monkeypatch,
) -> None:
    live = {
        "wavelength": 1.54,
        "distance": 123.0,
        "gamma": 1.5,
        "Gamma": 2.5,
        "chi": 3.5,
        "psi": 4.5,
        "psi_z": 5.5,
        "zs": 6.5,
        "zb": 7.5,
        "debye_x": 0.1,
        "debye_y": 0.2,
        "detector_center": (255.5, 256.5),
        "theta_initial": 8.5,
        "cor_angle": 9.5,
        "optics_mode": 2,
        "solve_q": SimpleNamespace(steps=321, rel_tol=1.0e-4, mode_flag=1),
    }
    process_peaks = lambda *args, **kwargs: None
    captured = {}

    monkeypatch.setattr(
        peak_selection,
        "simulate_ideal_hkl_native_center",
        lambda runtime_state, h, k, l, *, config, n2, process_peaks_parallel: (
            captured.update(
                {
                    "runtime_state": runtime_state,
                    "hkl": (h, k, l),
                    "config": config,
                    "n2": n2,
                    "process_peaks_parallel": process_peaks_parallel,
                }
            ),
            (12.0, 34.0),
        )[-1],
    )

    factory = peak_selection.make_runtime_selected_peak_ideal_center_factory(
        simulation_runtime_state="runtime-state",
        image_size=64,
        wavelength_factory=lambda: live["wavelength"],
        distance_cor_to_detector_factory=lambda: live["distance"],
        gamma_deg_factory=lambda: live["gamma"],
        Gamma_deg_factory=lambda: live["Gamma"],
        chi_deg_factory=lambda: live["chi"],
        psi_deg_factory=lambda: live["psi"],
        psi_z_deg_factory=lambda: live["psi_z"],
        zs_factory=lambda: live["zs"],
        zb_factory=lambda: live["zb"],
        debye_x_factory=lambda: live["debye_x"],
        debye_y_factory=lambda: live["debye_y"],
        detector_center_factory=lambda: live["detector_center"],
        theta_initial_deg_factory=lambda: live["theta_initial"],
        cor_angle_deg_factory=lambda: live["cor_angle"],
        optics_mode_factory=lambda: live["optics_mode"],
        solve_q_values_factory=lambda: live["solve_q"],
        n2="n2",
        process_peaks_parallel=process_peaks,
    )

    result = factory(1.0, 0.0, 2.0, 4.5, 6.5)

    assert result == (12.0, 34.0)
    assert captured["runtime_state"] == "runtime-state"
    assert captured["hkl"] == (1.0, 0.0, 2.0)
    assert captured["n2"] == "n2"
    assert captured["process_peaks_parallel"] is process_peaks
    assert isinstance(captured["config"], peak_selection.SelectedPeakIdealCenterProbeConfig)
    assert captured["config"].lattice_a == 4.5
    assert captured["config"].lattice_c == 6.5
    assert captured["config"].detector_center == (255.5, 256.5)
    assert captured["config"].solve_q_steps == 321
    assert captured["config"].solve_q_rel_tol == 1.0e-4
    assert captured["config"].solve_q_mode == 1


def test_peak_selection_runtime_config_factory_bundle_delegates_to_helper_factories(
    monkeypatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        peak_selection,
        "make_runtime_selected_peak_canvas_pick_config_factory",
        lambda **kwargs: calls.append(("canvas", kwargs)) or "canvas-factory",
    )
    monkeypatch.setattr(
        peak_selection,
        "make_runtime_selected_peak_intersection_config_factory",
        lambda **kwargs: calls.append(("intersection", kwargs))
        or "intersection-factory",
    )
    monkeypatch.setattr(
        peak_selection,
        "make_runtime_selected_peak_ideal_center_factory",
        lambda **kwargs: calls.append(("ideal_center", kwargs))
        or "ideal-center-factory",
    )

    bundle = peak_selection.make_runtime_selected_peak_config_factories(
        simulation_runtime_state="runtime-state",
        image_size=64,
        primary_a_factory="primary-a",
        primary_c_factory="primary-c",
        max_distance_px=12.0,
        min_separation_px=2.0,
        image_shape_factory="image-shape",
        center_col_factory="center-col",
        center_row_factory="center-row",
        distance_cor_to_detector_factory="distance",
        gamma_deg_factory="gamma",
        Gamma_deg_factory="Gamma",
        chi_deg_factory="chi",
        psi_deg_factory="psi",
        psi_z_deg_factory="psi-z",
        zs_factory="zs",
        zb_factory="zb",
        theta_initial_deg_factory="theta-initial",
        cor_angle_deg_factory="cor-angle",
        sigma_mosaic_deg_factory="sigma-mosaic",
        gamma_mosaic_deg_factory="gamma-mosaic",
        eta_factory="eta",
        wavelength_factory="wavelength",
        sample_width_m_factory="sample-width",
        sample_length_m_factory="sample-length",
        pixel_size_m_factory="pixel-size",
        debye_x_factory="debye-x",
        debye_y_factory="debye-y",
        detector_center_factory="detector-center",
        optics_mode_factory="optics-mode",
        solve_q_values_factory="solve-q",
        n2="n2",
        process_peaks_parallel="process-peaks",
    )

    assert bundle.canvas_pick == "canvas-factory"
    assert bundle.intersection == "intersection-factory"
    assert bundle.ideal_center == "ideal-center-factory"
    assert calls[0] == (
        "canvas",
        {
            "image_size": 64,
            "primary_a_factory": "primary-a",
            "primary_c_factory": "primary-c",
            "max_distance_px": 12.0,
            "min_separation_px": 2.0,
            "image_shape_factory": "image-shape",
        },
    )
    assert calls[1][0] == "intersection"
    assert calls[1][1]["solve_q_values_factory"] == "solve-q"
    assert calls[1][1]["pixel_size_m_factory"] == "pixel-size"
    assert calls[2][0] == "ideal_center"
    assert calls[2][1]["simulation_runtime_state"] == "runtime-state"
    assert calls[2][1]["process_peaks_parallel"] == "process-peaks"


def test_peak_selection_runtime_peak_overlay_data_callback_delegates_to_helper(
    monkeypatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        peak_selection,
        "ensure_runtime_peak_overlay_data",
        lambda runtime_state, **kwargs: captured.update(
            {
                "runtime_state": runtime_state,
                "kwargs": kwargs,
            }
        )
        or True,
    )

    callback = peak_selection.make_runtime_peak_overlay_data_callback(
        simulation_runtime_state="runtime-state",
        primary_a_factory=lambda: 5.0,
        primary_c_factory=lambda: 7.0,
        native_sim_to_display_coords="coords",
        reflection_q_group_metadata="q-group",
        max_hits_per_reflection=lambda: 3,
        min_separation_px=2.5,
    )

    assert callback(force=True) is True
    assert captured["runtime_state"] == "runtime-state"
    assert captured["kwargs"] == {
        "primary_a": captured["kwargs"]["primary_a"],
        "primary_c": captured["kwargs"]["primary_c"],
        "native_sim_to_display_coords": "coords",
        "reflection_q_group_metadata": "q-group",
        "caked_view_enabled_factory": False,
        "native_detector_coords_to_caked_display_coords": None,
        "max_hits_per_reflection": captured["kwargs"]["max_hits_per_reflection"],
        "min_separation_px": 2.5,
        "force": True,
    }
    assert captured["kwargs"]["primary_a"]() == 5.0
    assert captured["kwargs"]["primary_c"]() == 7.0
    assert captured["kwargs"]["max_hits_per_reflection"]() == 3


def test_peak_selection_runtime_peak_overlay_data_builds_records_and_reuses_cache() -> None:
    runtime_state = state.SimulationRuntimeState(
        last_simulation_signature=("sig",),
        stored_max_positions_local=[
            np.asarray(
                [
                    [8.0, 10.0, 20.0, 0.125, 1.0, 0.0, 2.0],
                    [5.0, 11.0, 21.0, 0.25, 2.0, 0.0, 3.0],
                ],
                dtype=float,
            )
        ],
        stored_sim_image=np.zeros((64, 64), dtype=float),
        stored_peak_table_lattice=None,
    )
    coord_calls = []
    q_group_calls = []

    ok = peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        native_sim_to_display_coords=lambda col, row, image_shape: (
            coord_calls.append((col, row, image_shape)) or (col + 100.0, row + 200.0)
        ),
        reflection_q_group_metadata=lambda hkl_raw, **kwargs: (
            q_group_calls.append((tuple(hkl_raw), kwargs)) or ("group-key", None, 0.75)
        ),
        max_hits_per_reflection=1,
        min_separation_px=0.0,
    )

    assert ok is True
    assert runtime_state.peak_positions == [(110.0, 220.0)]
    assert runtime_state.peak_millers == [(1, 0, 2)]
    assert runtime_state.peak_intensities == [8.0]
    record = runtime_state.peak_records[0]
    assert record["display_col"] == 110.0
    assert record["display_row"] == 220.0
    assert record["native_col"] == 10.0
    assert record["native_row"] == 20.0
    assert record["hkl"] == (1, 0, 2)
    assert record["hkl_raw"] == (1.0, 0.0, 2.0)
    assert record["intensity"] == 8.0
    assert record["qz"] == 0.75
    assert record["q_group_key"] == "group-key"
    assert record["phi"] == 0.125
    assert np.isnan(record["two_theta_deg"])
    assert np.isnan(record["phi_deg"])
    assert record["source_table_index"] == 0
    assert record["source_row_index"] == 0
    assert record["source_label"] == "primary"
    assert record["av"] == 4.0
    assert record["cv"] == 6.0
    assert np.isclose(
        record["qr"],
        (2.0 * np.pi / 4.0) * np.sqrt(4.0 / 3.0),
    )
    assert coord_calls == [(10.0, 20.0, (64, 64))]
    assert q_group_calls == [
        (
            (1.0, 0.0, 2.0),
            {
                "source_label": "primary",
                "a_value": 4.0,
                "c_value": 6.0,
                "qr_value": runtime_state.peak_records[0]["qr"],
            },
        )
    ]

    runtime_state.peak_positions.clear()
    runtime_state.peak_millers.clear()
    runtime_state.peak_intensities.clear()
    runtime_state.peak_records.clear()

    ok = peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=99.0,
        primary_c=101.0,
        native_sim_to_display_coords=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache rebuild should not recalculate display coords")
        ),
        reflection_q_group_metadata=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache rebuild should not recalculate q-group metadata")
        ),
        max_hits_per_reflection=1,
        min_separation_px=0.0,
    )

    assert ok is True
    assert runtime_state.peak_positions == [(110.0, 220.0)]
    assert runtime_state.peak_millers == [(1, 0, 2)]
    assert runtime_state.peak_intensities == [8.0]
    assert runtime_state.peak_records[0]["source_label"] == "primary"
    assert runtime_state.peak_overlay_cache["sig"] is not None


def test_peak_selection_runtime_peak_overlay_data_prefers_intersection_cache_centers() -> None:
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=[
            np.asarray(
                [[50.0, 10.0, 20.0, 0.125, 1.0, 0.0, 2.0]],
                dtype=float,
            )
        ],
        stored_primary_intersection_cache=[
            np.asarray(
                [[np.nan, np.nan, 40.0, 50.0, 8.0, 0.375, 1.0, 0.0, 2.0]],
                dtype=float,
            )
        ],
        stored_sim_image=np.zeros((64, 64), dtype=float),
    )
    coord_calls = []
    q_group_calls = []

    ok = peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        native_sim_to_display_coords=lambda col, row, image_shape: (
            coord_calls.append((col, row, image_shape)) or (col + 100.0, row + 200.0)
        ),
        reflection_q_group_metadata=lambda hkl_raw, **kwargs: (
            q_group_calls.append((tuple(hkl_raw), kwargs)) or ("group-key", None, 0.75)
        ),
        max_hits_per_reflection=0,
        min_separation_px=0.0,
    )

    assert ok is True
    assert runtime_state.peak_positions == [(140.0, 250.0)]
    assert runtime_state.peak_millers == [(1, 0, 2)]
    assert runtime_state.peak_intensities == [8.0]
    assert coord_calls == [(40.0, 50.0, (64, 64))]
    assert q_group_calls == [
        (
            (1.0, 0.0, 2.0),
            {
                "source_label": "primary",
                "a_value": 4.0,
                "c_value": 6.0,
                "qr_value": runtime_state.peak_records[0]["qr"],
            },
        )
    ]
    record = runtime_state.peak_records[0]
    assert record["display_col"] == 140.0
    assert record["display_row"] == 250.0
    assert record["native_col"] == 40.0
    assert record["native_row"] == 50.0
    assert record["hkl"] == (1, 0, 2)
    assert record["hkl_raw"] == (1.0, 0.0, 2.0)
    assert record["intensity"] == 8.0
    assert record["qz"] == 0.75
    assert record["q_group_key"] == "group-key"
    assert record["phi"] == 0.375
    assert np.isnan(record["two_theta_deg"])
    assert np.isnan(record["phi_deg"])
    assert record["source_table_index"] == 0
    assert record["source_row_index"] == 0
    assert record["source_label"] == "primary"
    assert record["av"] == 4.0
    assert record["cv"] == 6.0
    assert np.isclose(
        record["qr"],
        (2.0 * np.pi / 4.0) * np.sqrt(4.0 / 3.0),
    )


def test_peak_selection_runtime_peak_overlay_data_uses_cached_caked_coords() -> None:
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=None,
        stored_primary_intersection_cache=[
            np.asarray(
                [
                    [
                        1.5,
                        2.5,
                        40.0,
                        50.0,
                        8.0,
                        0.375,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        17.5,
                        -32.0,
                    ]
                ],
                dtype=float,
            )
        ],
        stored_sim_image=np.zeros((64, 64), dtype=float),
    )

    ok = peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        native_sim_to_display_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("caked cache rows should not use raw display conversion")
        ),
        reflection_q_group_metadata=lambda *_args, **_kwargs: ("group-key", None, 99.0),
        caked_view_enabled_factory=True,
        native_detector_coords_to_caked_display_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("cached caked coordinates should be used directly")
        ),
    )

    assert ok is True
    assert runtime_state.peak_positions == [(17.5, -32.0)]
    assert runtime_state.peak_millers == [(1, 0, 2)]
    assert runtime_state.peak_intensities == [8.0]
    record = runtime_state.peak_records[0]
    assert record["display_col"] == 17.5
    assert record["display_row"] == -32.0
    assert record["native_col"] == 40.0
    assert record["native_row"] == 50.0
    assert record["hkl"] == (1, 0, 2)
    assert record["hkl_raw"] == (1.0, 0.0, 2.0)
    assert record["intensity"] == 8.0
    assert record["qr"] == 1.5
    assert record["qz"] == 2.5
    assert record["q_group_key"] == "group-key"
    assert record["phi"] == 0.375
    assert record["two_theta_deg"] == 17.5
    assert record["phi_deg"] == -32.0
    assert record["source_label"] == "primary"
    assert record["av"] == 4.0
    assert record["cv"] == 6.0
    assert "source_table_index" not in record
    assert "source_row_index" not in record


def test_peak_selection_runtime_peak_overlay_data_clears_state_when_unavailable() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(1, 0, 2)],
        peak_intensities=[3.0],
        peak_records=[{"hkl": (1, 0, 2)}],
        stored_max_positions_local=None,
        stored_sim_image=np.zeros((8, 8), dtype=float),
    )

    ok = peak_selection.ensure_runtime_peak_overlay_data(
        runtime_state,
        primary_a=4.0,
        primary_c=6.0,
        native_sim_to_display_coords=lambda *_args: (0.0, 0.0),
        reflection_q_group_metadata=lambda *_args, **_kwargs: ("group", None, 0.0),
    )

    assert ok is False
    assert runtime_state.peak_positions == []
    assert runtime_state.peak_millers == []
    assert runtime_state.peak_intensities == []
    assert runtime_state.peak_records == []


def test_peak_selection_ideal_center_helpers_handle_hit_tables_and_profile_fallback() -> None:
    assert peak_selection.brightest_hit_native_from_table([]) is None
    assert peak_selection.brightest_hit_native_from_table([[1.0, 5.0, 6.0]]) == (
        5.0,
        6.0,
    )
    assert peak_selection.brightest_hit_native_from_table(
        [[1.0, 5.0, 6.0], [3.0, 8.0, 9.0], [2.0, 7.0, 4.0]]
    ) == (8.0, 9.0)

    runtime_state = state.SimulationRuntimeState()
    strict_calls = []

    def fake_process_strict(*args, **kwargs):
        strict_calls.append(
            {
                "lattice_a": args[3],
                "lattice_c": args[4],
                "optics_mode": kwargs["optics_mode"],
                "single_sample_indices": kwargs["single_sample_indices"],
            }
        )
        return None, [np.array([[5.0, 12.5, 23.5]], dtype=float)]

    strict_center = peak_selection.simulate_ideal_hkl_native_center(
        runtime_state,
        1.0,
        0.0,
        2.0,
        config=_ideal_center_probe_config(lattice_a=4.5, lattice_c=6.5),
        n2="n2",
        process_peaks_parallel=fake_process_strict,
    )
    assert strict_center == (12.5, 23.5)
    assert strict_calls == [
        {
            "lattice_a": 4.5,
            "lattice_c": 6.5,
            "optics_mode": 2,
            "single_sample_indices": None,
        }
    ]

    runtime_state.profile_cache = {
        "beam_x_array": [0.3, 0.1],
        "beam_y_array": [0.4, 0.2],
        "theta_array": [0.8, 0.2],
        "phi_array": [0.7, 0.1],
        "wavelength_array": [1.55, 1.56],
    }
    fallback_calls = []

    def fake_process_fallback(*args, **kwargs):
        forced = kwargs["single_sample_indices"]
        fallback_calls.append(
            None if forced is None else tuple(int(v) for v in forced.tolist())
        )
        if forced is None:
            return None, []
        return None, [np.array([[1.0, 44.0, 55.0]], dtype=float)]

    fallback_center = peak_selection.simulate_ideal_hkl_native_center(
        runtime_state,
        1.0,
        0.0,
        2.0,
        config=_ideal_center_probe_config(),
        n2="n2",
        process_peaks_parallel=fake_process_fallback,
    )
    assert fallback_center == (44.0, 55.0)
    assert fallback_calls == [None, (1,)]


def test_peak_selection_degenerate_hkls_and_qr_helpers_use_source_tables() -> None:
    runtime_state = state.SimulationRuntimeState(
        sim_miller1=np.asarray(
            [
                [1.0, 0.0, 2.0],
                [-1.0, 1.0, 2.0],
                [-1.0, 1.0, 3.0],
                [1.0, 0.0, 3.0],
            ],
            dtype=float,
        ),
        sim_miller2=np.asarray([[2.0, 0.0, 1.0]], dtype=float),
    )

    assert peak_selection.hkl_pick_button_text(True) == "Pick HKL on Image (Armed)"
    assert peak_selection.nearest_integer_hkl(1.4, -0.6, 2.51) == (1, -1, 3)
    assert peak_selection.format_hkl_triplet(1, 0, 2) == "(1 0 2)"
    np.testing.assert_allclose(
        peak_selection.source_miller_for_label(runtime_state, "secondary"),
        [[2.0, 0.0, 1.0]],
    )

    deg = peak_selection.degenerate_hkls_for_qr(
        runtime_state,
        1,
        0,
        4,
        source_label="primary",
    )
    assert deg == [(-1, 1, 3), (1, 0, 3)]

    qr_val, deg = peak_selection.selected_peak_qr_and_degenerates(
        runtime_state,
        1.0,
        0.0,
        2.0,
        {"source_label": "primary", "av": 4.0},
        primary_a=5.0,
    )
    assert np.isclose(qr_val, (2.0 * np.pi / 4.0) * np.sqrt(4.0 / 3.0))
    assert deg == [(-1, 1, 2), (1, 0, 2)]


def test_select_peak_by_index_updates_state_marker_lookup_and_status() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(10.0, 12.0)],
        peak_millers=[(1, 0, 2)],
        peak_intensities=[42.0],
        peak_records=[
            {
                "hkl": (1, 0, 2),
                "hkl_raw": (1.0, 0.0, 2.0),
                "source_label": "primary",
                "av": 4.0,
                "native_col": 9.5,
                "native_row": 11.5,
            }
        ],
        sim_miller1=np.asarray([[1.0, 0.0, 2.0], [-1.0, 1.0, 2.0]], dtype=float),
    )
    peak_state = state.PeakSelectionState()
    view_state = state.HklLookupViewState(
        selected_h_var=_FakeVar(),
        selected_k_var=_FakeVar(),
        selected_l_var=_FakeVar(),
    )
    marker = _FakeMarker()
    status_messages = []
    synced = []
    drawn = []

    ok = peak_selection.select_peak_by_index(
        runtime_state,
        peak_state,
        view_state,
        marker,
        0,
        primary_a=5.0,
        sync_peak_selection_state=lambda: synced.append(True),
        set_status_text=status_messages.append,
        draw_idle=lambda: drawn.append(True),
        prefix="Nearest peak",
        clicked_display=(9.0, 11.0),
        clicked_native=(8.0, 10.0),
    )

    assert ok is True
    assert marker.visible is True
    assert marker.data == ([10.0], [12.0])
    assert peak_state.selected_hkl_target == (1, 0, 2)
    assert runtime_state.selected_peak_record["clicked_display_col"] == 9.0
    assert runtime_state.selected_peak_record["selected_native_col"] == 8.0
    assert runtime_state.selected_peak_record["degenerate_hkls"] == [(-1, 1, 2), (1, 0, 2)]
    assert view_state.selected_h_var.get() == "1"
    assert "Nearest peak: HKL=(1 0 2)" in status_messages[-1]
    assert "Qr=" in status_messages[-1]
    assert synced == [True]
    assert drawn == [True]


def test_select_peak_by_hkl_uses_rod_fallback_and_missing_path() -> None:
    runtime_state = state.SimulationRuntimeState(
        unscaled_image=np.ones((4, 4), dtype=float),
        peak_positions=[(5.0, 6.0), (8.0, 9.0)],
        peak_millers=[(1, 0, 2), (-1, 1, 2)],
        peak_intensities=[3.0, 7.0],
        peak_records=[
            {"hkl": (1, 0, 2), "native_col": 5.0, "native_row": 6.0},
            {"hkl": (-1, 1, 2), "native_col": 8.0, "native_row": 9.0},
        ],
        sim_miller1=np.asarray([[1.0, 0.0, 2.0], [-1.0, 1.0, 2.0]], dtype=float),
    )
    peak_state = state.PeakSelectionState()
    view_state = state.HklLookupViewState(
        selected_h_var=_FakeVar(),
        selected_k_var=_FakeVar(),
        selected_l_var=_FakeVar(),
    )
    marker = _FakeMarker()
    status_messages = []
    scheduled = []

    ok = peak_selection.select_peak_by_hkl(
        runtime_state,
        peak_state,
        view_state,
        marker,
        0,
        1,
        2,
        primary_a=5.0,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: scheduled.append(True),
        sync_peak_selection_state=lambda: None,
        set_status_text=status_messages.append,
        draw_idle=lambda: None,
    )

    assert ok is True
    assert peak_state.selected_hkl_target == (-1, 1, 2)
    assert "HKL=(-1 1 2)" in status_messages[-1]

    runtime_state.peak_positions = []
    status_messages.clear()
    ok = peak_selection.select_peak_by_hkl(
        runtime_state,
        peak_state,
        view_state,
        marker,
        1,
        0,
        2,
        primary_a=5.0,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: scheduled.append(True),
        sync_peak_selection_state=lambda: None,
        set_status_text=status_messages.append,
        draw_idle=lambda: None,
        silent_if_missing=False,
    )

    assert ok is False
    assert scheduled == [True]
    assert status_messages[-1] == "Preparing simulated peak map... try again after update."


def test_select_peak_from_hkl_controls_and_clear_selected_peak() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(2.0, 3.0)],
        peak_millers=[(1, 2, 3)],
        peak_intensities=[5.0],
        peak_records=[{"hkl": (1, 2, 3), "native_col": 2.0, "native_row": 3.0}],
        sim_miller1=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
    )
    peak_state = state.PeakSelectionState()
    view_state = state.HklLookupViewState(
        selected_h_var=_FakeVar("1"),
        selected_k_var=_FakeVar("2"),
        selected_l_var=_FakeVar("3"),
    )
    marker = _FakeMarker()
    status_messages = []
    synced = []

    ok = peak_selection.select_peak_from_hkl_controls(
        runtime_state,
        peak_state,
        view_state,
        marker,
        primary_a=5.0,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: None,
        sync_peak_selection_state=lambda: synced.append(True),
        set_status_text=status_messages.append,
        draw_idle=lambda: None,
    )

    assert ok is True
    assert peak_selection.selected_hkl_from_lookup_controls(view_state) == (1, 2, 3)

    view_state.selected_h_var.set("bad")
    assert peak_selection.selected_hkl_from_lookup_controls(view_state) is None

    peak_selection.clear_selected_peak(
        runtime_state,
        peak_state,
        marker,
        sync_peak_selection_state=lambda: synced.append(True),
        set_status_text=status_messages.append,
        draw_idle=lambda: None,
    )

    assert runtime_state.selected_peak_record is None
    assert peak_state.selected_hkl_target is None
    assert marker.visible is False
    assert status_messages[-1] == "Peak selection cleared."
    assert len(synced) >= 2


def test_open_selected_peak_intersection_figure_requires_selection() -> None:
    runtime_state = state.SimulationRuntimeState()
    status_messages = []

    ok = peak_selection.open_selected_peak_intersection_figure(
        runtime_state,
        config=_intersection_config(),
        n2=object(),
        set_status_text=status_messages.append,
    )

    assert ok is False
    assert status_messages == [
        "Select a Bragg peak first (arm Pick HKL on Image or use HKL controls)."
    ]


def test_open_selected_peak_intersection_figure_launches_seeded_specular_visualizer() -> None:
    runtime_state = state.SimulationRuntimeState(
        profile_cache={
            "beam_x_array": [1.0e-4, 2.0e-4],
            "beam_y_array": [3.0e-4, 4.0e-4],
            "theta_array": [1.0e-3, 2.0e-3],
            "phi_array": [3.0e-3, 4.0e-3],
            "wavelength_array": [1.5406, 1.5410],
        },
        selected_peak_record={
            "hkl": (1, 0, 2),
            "av": 4.1,
            "cv": 6.2,
            "source_label": "primary",
        },
    )
    status_messages = []
    captured = {}
    config = _intersection_config()

    def fake_launch_specular_visualizer(initial_state):
        captured["state"] = initial_state

    ok = peak_selection.open_selected_peak_intersection_figure(
        runtime_state,
        config=config,
        n2=object(),
        set_status_text=status_messages.append,
        launch_specular_visualizer=fake_launch_specular_visualizer,
    )

    assert ok is True
    specular = captured["state"]["specular-view"]
    assert specular["H"] == 1
    assert specular["K"] == 0
    assert specular["L"] == 2
    assert specular["sigma_deg"] == 0.25
    assert specular["mosaic_gamma_deg"] == 0.5
    assert specular["eta"] == 0.75
    assert specular["rays"] == 2
    assert specular["display_rays"] == 2
    assert specular["source_y"] == -20.0
    assert specular["sample_width"] == 12.0
    assert specular["sample_height"] == 18.0
    assert specular["distance"] == config.distance_cor_to_detector * 1000.0
    assert specular["beta"] == -config.gamma_deg
    assert specular["gamma"] == -config.Gamma_deg
    assert specular["chi"] == 0.0
    assert specular["pixel_u"] == config.pixel_size_m * 1000.0
    assert specular["pixel_v"] == config.pixel_size_m * 1000.0
    assert specular["i0"] == config.center_col
    assert specular["j0"] == config.center_row
    np.testing.assert_allclose(
        specular["beam_width_x"],
        np.std([1.0e-4, 2.0e-4]) * 1000.0,
    )
    np.testing.assert_allclose(
        specular["beam_width_z"],
        np.std([3.0e-4, 4.0e-4]) * 1000.0,
    )
    np.testing.assert_allclose(
        specular["divergence_x"],
        np.rad2deg(np.std([3.0e-3, 4.0e-3])),
    )
    np.testing.assert_allclose(
        specular["divergence_z"],
        np.rad2deg(np.std([1.0e-3, 2.0e-3])),
    )
    assert specular["theta_i"] == config.theta_initial_deg
    assert np.isfinite(specular["delta"])
    assert np.isfinite(specular["alpha"])
    assert np.isfinite(specular["psi"])
    np.testing.assert_allclose(specular["wavelength_m"], 1.5406e-10)
    np.testing.assert_allclose(specular["lattice_a_m"], 4.1e-10)
    np.testing.assert_allclose(specular["lattice_c_m"], 6.2e-10)
    assert status_messages[-1] == (
        "Opened 2D Mosaic specular view for HKL=(1 0 2) from source=primary."
    )


def test_build_selected_peak_specular_initial_state_uses_realistic_fallbacks() -> None:
    runtime_state = state.SimulationRuntimeState(
        selected_peak_record={
            "hkl": (1, 0, 2),
            "av": 4.1,
            "cv": 6.2,
        },
    )
    config = peak_selection.build_selected_peak_intersection_config(
        image_size=3000,
        center_col=1500.0,
        center_row=1500.0,
        distance_cor_to_detector=float("nan"),
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        sigma_mosaic_deg=0.8,
        gamma_mosaic_deg=0.7,
        eta=0.0,
        sample_width_m=0.0,
        sample_length_m=0.0,
        wavelength_angstrom=1.54,
        solve_q_steps=1000,
        solve_q_rel_tol=5.0e-4,
        solve_q_mode=1,
        pixel_size_m=0.0,
    )

    specular = peak_selection._build_selected_peak_specular_initial_state(
        runtime_state,
        config=config,
        selected_peak=runtime_state.selected_peak_record,
    )["specular-view"]

    assert specular["distance"] == 75.0
    assert specular["sample_width"] == 20.0
    assert specular["sample_height"] == 80.0
    assert specular["pixel_u"] == 0.1
    assert specular["pixel_v"] == 0.1
    assert specular["detector_width"] == 300.0
    assert specular["detector_height"] == 300.0


def test_open_selected_peak_intersection_figure_reports_failures() -> None:
    runtime_state = state.SimulationRuntimeState(
        selected_peak_record={
            "hkl": (1, 0, 2),
            "av": 4.1,
            "cv": 6.2,
        },
    )
    status_messages = []

    def raise_boom(_initial_state):
        raise ValueError("boom")

    ok = peak_selection.open_selected_peak_intersection_figure(
        runtime_state,
        config=_intersection_config(),
        n2=object(),
        set_status_text=status_messages.append,
        launch_specular_visualizer=raise_boom,
    )

    assert ok is False
    assert status_messages[-1] == "Specular visualizer launch failed for selected peak: boom"


def test_toggle_hkl_pick_mode_handles_ready_and_unready_paths() -> None:
    runtime_state = state.SimulationRuntimeState(
        unscaled_image=np.ones((4, 4), dtype=float),
    )
    peak_state = state.PeakSelectionState()
    status_messages = []
    scheduled = []
    pick_mode_calls = []

    peak_selection.toggle_hkl_pick_mode(
        runtime_state,
        peak_state,
        caked_view_enabled=True,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: scheduled.append(True),
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        set_status_text=status_messages.append,
    )

    assert pick_mode_calls == []
    assert status_messages[-1] == (
        "Switch off 2D caked view before picking HKL in the image."
    )

    peak_selection.toggle_hkl_pick_mode(
        runtime_state,
        peak_state,
        caked_view_enabled=False,
        ensure_peak_overlay_data=lambda **_kwargs: False,
        schedule_update=lambda: scheduled.append(True),
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        set_status_text=status_messages.append,
    )

    assert scheduled == [True]
    assert pick_mode_calls[-1] == (
        True,
        "Preparing simulated peak map for HKL picking... wait for the next update.",
    )

    runtime_state.peak_positions = [(1.0, 2.0)]
    peak_selection.toggle_hkl_pick_mode(
        runtime_state,
        peak_state,
        caked_view_enabled=False,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: scheduled.append(True),
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        set_status_text=status_messages.append,
    )

    assert pick_mode_calls[-1] == (
        True,
        "HKL image-pick armed: click near a Bragg peak in raw camera view.",
    )

    peak_state.hkl_pick_armed = True
    peak_selection.toggle_hkl_pick_mode(
        runtime_state,
        peak_state,
        caked_view_enabled=False,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: scheduled.append(True),
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        set_status_text=status_messages.append,
    )

    assert pick_mode_calls[-1] == (False, "HKL image-pick canceled.")


def test_select_peak_from_canvas_click_prepares_update_when_peak_map_missing() -> None:
    runtime_state = state.SimulationRuntimeState()
    peak_state = state.PeakSelectionState(hkl_pick_armed=True)
    status_messages = []
    scheduled = []
    ensured = []

    ok = peak_selection.select_peak_from_canvas_click(
        runtime_state,
        peak_state,
        9.3,
        11.7,
        config=_canvas_pick_config(),
        ensure_peak_overlay_data=lambda **kwargs: ensured.append(kwargs) or True,
        schedule_update=lambda: scheduled.append(True),
        display_to_native_sim_coords=lambda col, row, image_shape: (col, row),
        native_sim_to_display_coords=lambda col, row, image_shape: (col, row),
        simulate_ideal_hkl_native_center=lambda *_args: None,
        select_peak_by_index=lambda *_args, **_kwargs: True,
        set_pick_mode=lambda enabled, message=None: None,
        sync_peak_selection_state=lambda: None,
        set_status_text=status_messages.append,
    )

    assert ok is False
    assert ensured == [{"force": False}]
    assert scheduled == [True]
    assert status_messages[-1] == "Preparing simulated peak map... click again after update."


def test_select_peak_from_canvas_click_selects_nearest_cached_peak_without_reprobe() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(10.0, 12.0), (30.0, 40.0)],
        peak_millers=[(1, 0, 2), (2, 0, 1)],
        peak_intensities=[5.0, 8.0],
        peak_records=[
            {
                "hkl_raw": (1.1, -0.1, 2.0),
                "av": 4.5,
                "cv": 6.5,
                "native_col": 100.0,
                "native_row": 101.0,
            },
            {
                "hkl_raw": (2.0, 0.0, 1.0),
                "av": 4.5,
                "cv": 6.5,
                "native_col": 200.0,
                "native_row": 201.0,
            },
        ],
    )
    peak_state = state.PeakSelectionState(hkl_pick_armed=True)
    status_messages = []
    sync_calls = []
    pick_mode_calls = []
    select_calls = []

    def fake_select_peak_by_index(idx, **kwargs):
        select_calls.append((idx, kwargs))
        return True

    ok = peak_selection.select_peak_from_canvas_click(
        runtime_state,
        peak_state,
        9.8,
        12.4,
        config=_canvas_pick_config(image_shape=(80, 80)),
        ensure_peak_overlay_data=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("click selection should reuse the cached peak overlay")
        ),
        schedule_update=lambda: None,
        display_to_native_sim_coords=lambda col, row, image_shape: (
            float(col) + 0.25,
            float(row) + 0.75,
        ),
        native_sim_to_display_coords=lambda col, row, image_shape: (
            float(col) - 90.0,
            float(row) - 89.0,
        ),
        simulate_ideal_hkl_native_center=lambda *_args: (_ for _ in ()).throw(
            AssertionError("click selection should not re-probe ideal HKL centers")
        ),
        select_peak_by_index=fake_select_peak_by_index,
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        sync_peak_selection_state=lambda: sync_calls.append(True),
        set_status_text=status_messages.append,
    )

    assert ok is True
    assert len(select_calls) == 1
    idx, kwargs = select_calls[0]
    assert idx == 0
    assert kwargs["prefix"] == "Nearest peak (Δ=0.4px)"
    assert kwargs["clicked_display"] == (9.8, 12.4)
    assert kwargs["clicked_native"] == (10.25, 12.75)
    assert kwargs["selected_display"] is None
    assert kwargs["selected_native"] == (100.0, 101.0)
    assert kwargs["sync_hkl_vars"] is True
    assert pick_mode_calls == [(False, None)]
    assert peak_state.suppress_drag_press_once is True
    assert sync_calls == [True]
    assert status_messages == []


def test_select_peak_from_canvas_click_uses_100x100_square_search_window() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(50.0, 50.0)],
        peak_millers=[(1, 0, 2)],
        peak_intensities=[5.0],
        peak_records=[
            {
                "native_col": 50.0,
                "native_row": 50.0,
            }
        ],
    )
    peak_state = state.PeakSelectionState(hkl_pick_armed=True)
    pick_mode_calls = []
    select_calls = []

    ok = peak_selection.select_peak_from_canvas_click(
        runtime_state,
        peak_state,
        0.0,
        0.0,
        config=peak_selection.build_selected_peak_canvas_pick_config(
            image_size=128,
            primary_a=5.0,
            primary_c=7.0,
            max_distance_px=50.0,
            min_separation_px=2.0,
            image_shape=(128, 128),
        ),
        ensure_peak_overlay_data=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("click selection should reuse the cached peak overlay")
        ),
        schedule_update=lambda: None,
        display_to_native_sim_coords=lambda col, row, image_shape: (
            float(col),
            float(row),
        ),
        native_sim_to_display_coords=lambda col, row, image_shape: (
            float(col),
            float(row),
        ),
        simulate_ideal_hkl_native_center=lambda *_args: None,
        select_peak_by_index=lambda idx, **kwargs: select_calls.append((idx, kwargs)) or True,
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        sync_peak_selection_state=lambda: None,
        set_status_text=lambda _text: None,
    )

    assert ok is True
    assert len(select_calls) == 1
    idx, kwargs = select_calls[0]
    assert idx == 0
    assert kwargs["prefix"] == "Nearest peak (Δ=70.7px)"
    assert kwargs["selected_native"] == (50.0, 50.0)
    assert pick_mode_calls == [(False, None)]
    assert peak_state.suppress_drag_press_once is True


def test_select_peak_from_canvas_click_uses_intersection_cache_centers_for_nearest_hkl() -> None:
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=[
            np.asarray(
                [
                    [50.0, 10.0, 10.0, 0.125, 1.0, 0.0, 2.0],
                    [60.0, 40.0, 40.0, 0.250, 2.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        ],
        stored_primary_intersection_cache=[
            np.asarray(
                [
                    [np.nan, np.nan, 100.0, 100.0, 8.0, 0.125, 1.0, 0.0, 2.0],
                    [np.nan, np.nan, 30.0, 30.0, 9.0, 0.250, 2.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        ],
        stored_sim_image=np.zeros((128, 128), dtype=float),
    )
    peak_state = state.PeakSelectionState(hkl_pick_armed=True)
    pick_mode_calls = []
    select_calls = []
    status_messages = []
    sync_calls = []

    def ensure_peak_overlay_data(*, force: bool = False) -> bool:
        return peak_selection.ensure_runtime_peak_overlay_data(
            runtime_state,
            primary_a=5.0,
            primary_c=7.0,
            native_sim_to_display_coords=lambda col, row, image_shape: (
                float(col),
                float(row),
            ),
            reflection_q_group_metadata=lambda *_args, **_kwargs: (
                "group-key",
                None,
                0.0,
            ),
            max_hits_per_reflection=0,
            min_separation_px=0.0,
            force=force,
        )

    ok = peak_selection.select_peak_from_canvas_click(
        runtime_state,
        peak_state,
        96.0,
        98.0,
        config=_canvas_pick_config(image_shape=(128, 128)),
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        schedule_update=lambda: (_ for _ in ()).throw(
            AssertionError("cache-backed overlay should build without scheduling another frame")
        ),
        display_to_native_sim_coords=lambda col, row, image_shape: (
            float(col),
            float(row),
        ),
        native_sim_to_display_coords=lambda col, row, image_shape: (
            float(col),
            float(row),
        ),
        simulate_ideal_hkl_native_center=lambda *_args: None,
        select_peak_by_index=lambda idx, **kwargs: select_calls.append((idx, kwargs)) or True,
        set_pick_mode=lambda enabled, message=None: pick_mode_calls.append(
            (bool(enabled), message)
        ),
        sync_peak_selection_state=lambda: sync_calls.append(True),
        set_status_text=status_messages.append,
    )

    assert ok is True
    assert runtime_state.peak_positions == [(100.0, 100.0), (30.0, 30.0)]
    assert runtime_state.peak_millers == [(1, 0, 2), (2, 0, 1)]
    assert runtime_state.peak_records[0]["source_row_index"] == 0
    assert runtime_state.peak_records[1]["source_row_index"] == 1
    assert len(select_calls) == 1
    idx, kwargs = select_calls[0]
    assert idx == 0
    assert kwargs["prefix"] == "Nearest peak (Δ=4.5px)"
    assert kwargs["clicked_display"] == (96.0, 98.0)
    assert kwargs["clicked_native"] == (96.0, 98.0)
    assert kwargs["selected_display"] is None
    assert kwargs["selected_native"] == (100.0, 100.0)
    assert pick_mode_calls == [(False, None)]
    assert peak_state.suppress_drag_press_once is True
    assert sync_calls == [True]
    assert status_messages == []


def test_peak_selection_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"view": 0, "marker": 0, "status": 0, "schedule": 0, "draw": 0, "deactivate": 0}

    monkeypatch.setattr(
        peak_selection,
        "SelectedPeakRuntimeBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_view():
        counters["view"] += 1
        return f"view-{counters['view']}"

    def build_marker():
        counters["marker"] += 1
        return f"marker-{counters['marker']}"

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    def build_schedule():
        counters["schedule"] += 1
        idx = counters["schedule"]
        return lambda: f"schedule-{idx}"

    def build_draw():
        counters["draw"] += 1
        idx = counters["draw"]
        return lambda: f"draw-{idx}"

    def build_deactivate():
        counters["deactivate"] += 1
        idx = counters["deactivate"]
        return lambda: f"deactivate-{idx}"

    factory = peak_selection.make_runtime_peak_selection_bindings_factory(
        simulation_runtime_state="runtime-state",
        peak_selection_state="peak-state",
        hkl_lookup_view_state_factory=build_view,
        selected_peak_marker_factory=build_marker,
        current_primary_a_factory=lambda: 5.5,
        caked_view_enabled_factory=lambda: False,
        current_canvas_pick_config_factory=_canvas_pick_config,
        current_intersection_config_factory=_intersection_config,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        sync_peak_selection_state=lambda: None,
        schedule_update_factory=build_schedule,
        set_status_text_factory=build_status,
        draw_idle_factory=build_draw,
        display_to_native_sim_coords=lambda col, row, image_shape: (col, row),
        native_sim_to_display_coords=lambda col, row, image_shape: (col, row),
        simulate_ideal_hkl_native_center=lambda *_args: None,
        deactivate_conflicting_modes_factory=build_deactivate,
        n2="n2",
        tcl_error_types=(RuntimeError,),
    )

    assert factory()["simulation_runtime_state"] == "runtime-state"
    assert factory()["simulation_runtime_state"] == "runtime-state"
    assert calls[0]["peak_selection_state"] == "peak-state"
    assert calls[0]["hkl_lookup_view_state"] == "view-1"
    assert calls[1]["hkl_lookup_view_state"] == "view-2"
    assert calls[0]["selected_peak_marker"] == "marker-1"
    assert calls[1]["selected_peak_marker"] == "marker-2"
    assert callable(calls[0]["set_status_text"])
    assert callable(calls[0]["schedule_update"])
    assert callable(calls[0]["draw_idle"])
    assert callable(calls[0]["deactivate_conflicting_modes"])
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]
    assert calls[0]["schedule_update"] is not calls[1]["schedule_update"]
    assert calls[0]["draw_idle"] is not calls[1]["draw_idle"]
    assert calls[0]["deactivate_conflicting_modes"] is not calls[1]["deactivate_conflicting_modes"]
    assert calls[0]["tcl_error_types"] == (RuntimeError,)


def test_peak_selection_runtime_helpers_and_callback_bundle_delegate_live_bindings(
    monkeypatch,
) -> None:
    runtime_state = state.SimulationRuntimeState()
    peak_state = state.PeakSelectionState(selected_hkl_target=(1, 2, 3))
    view_state = state.HklLookupViewState(hkl_pick_button_var=_FakeVar())
    marker = _FakeMarker()
    messages = []
    sync_calls = []
    deactivate_calls = []
    mode_change_calls = []
    calls = []

    bindings = peak_selection.SelectedPeakRuntimeBindings(
        simulation_runtime_state=runtime_state,
        peak_selection_state=peak_state,
        hkl_lookup_view_state=view_state,
        selected_peak_marker=marker,
        current_primary_a_factory=lambda: 5.5,
        caked_view_enabled_factory=lambda: True,
        current_canvas_pick_config_factory=_canvas_pick_config,
        current_intersection_config_factory=_intersection_config,
        ensure_peak_overlay_data=lambda **_kwargs: True,
        sync_peak_selection_state=lambda: sync_calls.append(True),
        schedule_update=lambda: None,
        set_status_text=messages.append,
        draw_idle=lambda: None,
        display_to_native_sim_coords=lambda col, row, image_shape: (col, row),
        native_sim_to_display_coords=lambda col, row, image_shape: (col, row),
        simulate_ideal_hkl_native_center=lambda *_args: None,
        deactivate_conflicting_modes=lambda: deactivate_calls.append(True),
        on_hkl_pick_mode_changed=lambda enabled: mode_change_calls.append(bool(enabled)),
        n2="n2",
        tcl_error_types=(RuntimeError,),
    )

    peak_selection.update_runtime_hkl_pick_button_label(bindings)
    assert view_state.hkl_pick_button_var.get() == "Pick HKL on Image"

    peak_selection.set_runtime_hkl_pick_mode(bindings, True, message="armed")
    assert peak_state.hkl_pick_armed is True
    assert view_state.hkl_pick_button_var.get() == "Pick HKL on Image (Armed)"
    assert sync_calls == [True]
    assert deactivate_calls == [True]
    assert mode_change_calls == [True]
    assert messages == ["armed"]

    monkeypatch.setattr(
        peak_selection,
        "toggle_hkl_pick_mode",
        lambda runtime_state_arg, peak_state_arg, **kwargs: calls.append(
            ("toggle", runtime_state_arg, peak_state_arg, kwargs)
        ),
    )
    monkeypatch.setattr(
        peak_selection,
        "select_peak_by_hkl",
        lambda runtime_state_arg, peak_state_arg, view_state_arg, marker_arg, h, k, l, **kwargs: (
            calls.append(
                (
                    "select_hkl",
                    runtime_state_arg,
                    peak_state_arg,
                    view_state_arg,
                    marker_arg,
                    h,
                    k,
                    l,
                    kwargs,
                )
            ),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        peak_selection,
        "select_peak_from_hkl_controls",
        lambda runtime_state_arg, peak_state_arg, view_state_arg, marker_arg, **kwargs: (
            calls.append(
                (
                    "controls",
                    runtime_state_arg,
                    peak_state_arg,
                    view_state_arg,
                    marker_arg,
                    kwargs,
                )
            ),
            False,
        )[-1],
    )
    monkeypatch.setattr(
        peak_selection,
        "clear_selected_peak",
        lambda runtime_state_arg, peak_state_arg, marker_arg, **kwargs: calls.append(
            ("clear", runtime_state_arg, peak_state_arg, marker_arg, kwargs)
        ),
    )
    monkeypatch.setattr(
        peak_selection,
        "open_selected_peak_intersection_figure",
        lambda runtime_state_arg, *, config, n2, set_status_text: (
            calls.append(("open", runtime_state_arg, config, n2)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        peak_selection,
        "select_peak_from_canvas_click",
        lambda runtime_state_arg, peak_state_arg, click_col, click_row, **kwargs: (
            calls.append(
                ("click", runtime_state_arg, peak_state_arg, click_col, click_row, kwargs)
            ),
            True,
        )[-1],
    )

    peak_selection.toggle_runtime_hkl_pick_mode(bindings)
    toggle_call = calls[0]
    assert toggle_call[0] == "toggle"
    assert toggle_call[1] is runtime_state
    assert toggle_call[2] is peak_state
    assert toggle_call[3]["caked_view_enabled"] is True

    assert peak_selection.reselect_runtime_selected_peak(bindings) is True
    select_hkl_call = calls[1]
    assert select_hkl_call[0] == "select_hkl"
    assert select_hkl_call[1] is runtime_state
    assert select_hkl_call[2] is peak_state
    assert select_hkl_call[3] is view_state
    assert select_hkl_call[4] is marker
    assert select_hkl_call[5:8] == (1, 2, 3)
    assert select_hkl_call[8]["primary_a"] == 5.5
    assert select_hkl_call[8]["sync_hkl_vars"] is False
    assert select_hkl_call[8]["silent_if_missing"] is True

    assert peak_selection.select_peak_from_runtime_hkl_controls(bindings) is False
    controls_call = calls[2]
    assert controls_call[0] == "controls"
    assert controls_call[1] is runtime_state
    assert controls_call[2] is peak_state
    assert controls_call[3] is view_state
    assert controls_call[4] is marker
    assert controls_call[5]["primary_a"] == 5.5
    assert controls_call[5]["tcl_error_types"] == (RuntimeError,)

    peak_selection.clear_runtime_selected_peak(bindings)
    clear_call = calls[3]
    assert clear_call[0] == "clear"
    assert clear_call[1] is runtime_state
    assert clear_call[2] is peak_state
    assert clear_call[3] is marker

    assert peak_selection.open_runtime_selected_peak_intersection_figure(bindings) is True
    open_call = calls[4]
    assert open_call[0] == "open"
    assert open_call[1] is runtime_state
    assert isinstance(open_call[2], peak_selection.SelectedPeakIntersectionConfig)
    assert open_call[3] == "n2"

    assert peak_selection.select_peak_from_runtime_canvas_click(bindings, 9.5, 11.5) is True
    click_call = calls[5]
    assert click_call[0] == "click"
    assert click_call[1] is runtime_state
    assert click_call[2] is peak_state
    assert click_call[3:5] == (9.5, 11.5)
    assert isinstance(click_call[5]["config"], peak_selection.SelectedPeakCanvasPickConfig)
    assert callable(click_call[5]["select_peak_by_index"])
    assert callable(click_call[5]["set_pick_mode"])

    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        peak_selection,
        "update_runtime_hkl_pick_button_label",
        lambda bindings_arg: callback_calls.append(("label", bindings_arg)),
    )
    monkeypatch.setattr(
        peak_selection,
        "set_runtime_hkl_pick_mode",
        lambda bindings_arg, enabled, *, message=None: callback_calls.append(
            ("set", bindings_arg, enabled, message)
        ),
    )
    monkeypatch.setattr(
        peak_selection,
        "toggle_runtime_hkl_pick_mode",
        lambda bindings_arg: callback_calls.append(("toggle_cb", bindings_arg)),
    )
    monkeypatch.setattr(
        peak_selection,
        "reselect_runtime_selected_peak",
        lambda bindings_arg: callback_calls.append(("reselect", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        peak_selection,
        "select_peak_from_runtime_hkl_controls",
        lambda bindings_arg: callback_calls.append(("controls_cb", bindings_arg)) or False,
    )
    monkeypatch.setattr(
        peak_selection,
        "clear_runtime_selected_peak",
        lambda bindings_arg: callback_calls.append(("clear_cb", bindings_arg)),
    )
    monkeypatch.setattr(
        peak_selection,
        "open_runtime_selected_peak_intersection_figure",
        lambda bindings_arg: callback_calls.append(("open_cb", bindings_arg)) or True,
    )
    monkeypatch.setattr(
        peak_selection,
        "select_peak_from_runtime_canvas_click",
        lambda bindings_arg, click_col, click_row: callback_calls.append(
            ("click_cb", bindings_arg, click_col, click_row)
        )
        or False,
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = peak_selection.make_runtime_peak_selection_callbacks(build_bindings)

    callbacks.update_hkl_pick_button_label()
    callbacks.set_hkl_pick_mode(False, "off")
    callbacks.toggle_hkl_pick_mode()
    assert callbacks.reselect_current_peak() is True
    assert callbacks.select_peak_from_hkl_controls() is False
    callbacks.clear_selected_peak()
    assert callbacks.open_selected_peak_intersection_figure() is True
    assert callbacks.select_peak_from_canvas_click(3.0, 4.0) is False

    assert callback_calls == [
        ("label", "bindings-1"),
        ("set", "bindings-2", False, "off"),
        ("toggle_cb", "bindings-3"),
        ("reselect", "bindings-4"),
        ("controls_cb", "bindings-5"),
        ("clear_cb", "bindings-6"),
        ("open_cb", "bindings-7"),
        ("click_cb", "bindings-8", 3.0, 4.0),
    ]


def test_refresh_runtime_selected_peak_after_simulation_update_manages_overlay_state() -> None:
    runtime_state = state.SimulationRuntimeState(
        peak_positions=[(1.0, 2.0)],
        peak_millers=[(1, 0, 2)],
        peak_intensities=[3.0],
        peak_records=[{"hkl": (1, 0, 2)}],
    )
    peak_state = state.PeakSelectionState(selected_hkl_target=(1, 0, 2))
    events: list[object] = []

    bindings = peak_selection.SelectedPeakRuntimeBindings(
        simulation_runtime_state=runtime_state,
        peak_selection_state=peak_state,
        hkl_lookup_view_state=None,
        selected_peak_marker=None,
        current_primary_a_factory=lambda: 5.0,
        caked_view_enabled_factory=lambda: False,
        current_canvas_pick_config_factory=lambda: None,
        current_intersection_config_factory=lambda: None,
        ensure_peak_overlay_data=lambda *, force=False: events.append(("ensure", force)) or True,
    )

    handled = peak_selection.refresh_runtime_selected_peak_after_simulation_update(
        bindings,
        live_geometry_preview_enabled=False,
    )

    assert handled is True
    assert events == [("ensure", False)]

    events.clear()
    peak_state.selected_hkl_target = None
    runtime_state.selected_peak_record = None

    handled = peak_selection.refresh_runtime_selected_peak_after_simulation_update(
        bindings,
        live_geometry_preview_enabled=False,
    )

    assert handled is False
    assert events == []
    assert runtime_state.peak_positions == []
    assert runtime_state.peak_millers == []
    assert runtime_state.peak_intensities == []
    assert runtime_state.peak_records == []


def test_apply_runtime_restored_selected_hkl_target_normalizes_and_refreshes_label(
    monkeypatch,
) -> None:
    events: list[object] = []
    runtime_state = state.SimulationRuntimeState()
    peak_state = state.PeakSelectionState()

    monkeypatch.setattr(
        peak_selection,
        "update_runtime_hkl_pick_button_label",
        lambda bindings: events.append(("label", bindings)),
    )

    bindings = peak_selection.SelectedPeakRuntimeBindings(
        simulation_runtime_state=runtime_state,
        peak_selection_state=peak_state,
        hkl_lookup_view_state=None,
        selected_peak_marker=None,
        current_primary_a_factory=lambda: 5.0,
        caked_view_enabled_factory=lambda: False,
        current_canvas_pick_config_factory=lambda: None,
        current_intersection_config_factory=lambda: None,
        ensure_peak_overlay_data=lambda *, force=False: True,
        sync_peak_selection_state=lambda: events.append("sync"),
    )

    result = peak_selection.apply_runtime_restored_selected_hkl_target(
        bindings,
        [2, -1, 4],
    )
    assert result == (2, -1, 4)
    assert peak_state.selected_hkl_target == (2, -1, 4)
    assert events == ["sync", ("label", bindings)]

    events.clear()
    result = peak_selection.apply_runtime_restored_selected_hkl_target(
        bindings,
        "bad-target",
    )
    assert result is None
    assert peak_state.selected_hkl_target is None
    assert events == ["sync", ("label", bindings)]


def test_make_runtime_peak_selection_maintenance_callbacks_uses_fresh_bindings(
    monkeypatch,
) -> None:
    callback_calls: list[object] = []
    versions = {"count": 0}

    monkeypatch.setattr(
        peak_selection,
        "refresh_runtime_selected_peak_after_simulation_update",
        lambda bindings, *, live_geometry_preview_enabled: (
            callback_calls.append(("refresh", bindings, live_geometry_preview_enabled))
            or True
        ),
    )
    monkeypatch.setattr(
        peak_selection,
        "apply_runtime_restored_selected_hkl_target",
        lambda bindings, selected_hkl_target: (
            callback_calls.append(("restore", bindings, selected_hkl_target))
            or (1, 2, 3)
        ),
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = peak_selection.make_runtime_peak_selection_maintenance_callbacks(
        build_bindings
    )

    assert callbacks.refresh_after_simulation_update(True) is True
    assert callbacks.apply_restored_selected_hkl_target([1, 2, 3]) == (1, 2, 3)
    assert callback_calls == [
        ("refresh", "bindings-1", True),
        ("restore", "bindings-2", [1, 2, 3]),
    ]
