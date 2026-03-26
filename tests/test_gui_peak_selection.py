import numpy as np

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


def test_open_selected_peak_intersection_figure_builds_analysis_and_opens_window() -> None:
    runtime_state = state.SimulationRuntimeState(
        profile_cache={
            "beam_x_array": [1.0, 2.0],
            "beam_y_array": [3.0, 4.0],
            "theta_array": [5.0, 6.0],
            "phi_array": [7.0, 8.0],
            "wavelength_array": [9.0, 10.0],
        },
        selected_peak_record={
            "hkl": (1, 0, 2),
            "selected_native_col": 11.5,
            "selected_native_row": 12.5,
            "av": 4.1,
            "cv": 6.2,
            "source_label": "primary",
        },
    )
    status_messages = []
    captured = {}
    manager = _FakeManager()
    fig = _FakeFigure(manager=manager)
    n2 = object()

    def fake_geometry_factory(**kwargs):
        captured["geometry"] = kwargs
        return ("geometry", kwargs)

    def fake_beam_factory(**kwargs):
        captured["beam"] = kwargs
        return ("beam", kwargs)

    def fake_mosaic_factory(**kwargs):
        captured["mosaic"] = kwargs
        return ("mosaic", kwargs)

    def fake_analyze_intersection(**kwargs):
        captured["analyze"] = kwargs
        return {"analysis": "ok"}

    def fake_plot_intersection(analysis):
        captured["plot"] = analysis
        return fig

    ok = peak_selection.open_selected_peak_intersection_figure(
        runtime_state,
        config=_intersection_config(),
        n2=n2,
        set_status_text=status_messages.append,
        geometry_factory=fake_geometry_factory,
        beam_factory=fake_beam_factory,
        mosaic_factory=fake_mosaic_factory,
        analyze_intersection=fake_analyze_intersection,
        plot_intersection=fake_plot_intersection,
    )

    assert ok is True
    assert captured["analyze"]["h"] == 1
    assert captured["analyze"]["k"] == 0
    assert captured["analyze"]["l"] == 2
    assert captured["analyze"]["lattice_a"] == 4.1
    assert captured["analyze"]["lattice_c"] == 6.2
    assert captured["analyze"]["selected_native_col"] == 11.5
    assert captured["analyze"]["selected_native_row"] == 12.5
    assert captured["analyze"]["n2"] is n2
    np.testing.assert_allclose(captured["geometry"]["n_detector"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(captured["geometry"]["unit_x"], [1.0, 0.0, 0.0])
    assert captured["geometry"]["image_size"] == 512
    assert captured["geometry"]["pixel_size_m"] == 100e-6
    np.testing.assert_allclose(captured["beam"]["beam_x_array"], [1.0, 2.0])
    np.testing.assert_allclose(captured["beam"]["wavelength_array"], [9.0, 10.0])
    assert captured["mosaic"]["sigma_mosaic_deg"] == 0.25
    assert captured["mosaic"]["gamma_mosaic_deg"] == 0.5
    assert captured["mosaic"]["eta"] == 0.75
    assert captured["mosaic"]["solve_q_steps"] == 321
    assert captured["mosaic"]["solve_q_rel_tol"] == 1.0e-4
    assert captured["mosaic"]["solve_q_mode"] == 2
    assert captured["plot"] == {"analysis": "ok"}
    assert manager.title == "Bragg/Ewald HKL=(1,0,2)"
    assert manager.shown is True
    assert fig.shown is False
    assert status_messages[-1] == (
        "Opened Bragg/Ewald analysis for HKL=(1 0 2) from source=primary."
    )


def test_open_selected_peak_intersection_figure_reports_failures() -> None:
    runtime_state = state.SimulationRuntimeState(
        profile_cache={
            "beam_x_array": [1.0],
            "beam_y_array": [2.0],
            "theta_array": [3.0],
            "phi_array": [4.0],
            "wavelength_array": [5.0],
        },
        selected_peak_record={
            "hkl": (1, 0, 2),
            "native_col": 10.0,
            "native_row": 11.0,
            "av": 4.1,
            "cv": 6.2,
        },
    )
    status_messages = []

    def raise_boom(**_kwargs):
        raise ValueError("boom")

    ok = peak_selection.open_selected_peak_intersection_figure(
        runtime_state,
        config=_intersection_config(),
        n2=object(),
        set_status_text=status_messages.append,
        geometry_factory=lambda **kwargs: kwargs,
        beam_factory=lambda **kwargs: kwargs,
        mosaic_factory=lambda **kwargs: kwargs,
        analyze_intersection=raise_boom,
    )

    assert ok is False
    assert status_messages[-1] == "Intersection analysis failed for selected peak: boom"


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


def test_select_peak_from_canvas_click_selects_nearest_peak_and_snaps_to_ideal_center() -> None:
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
        ensure_peak_overlay_data=lambda **_kwargs: True,
        schedule_update=lambda: None,
        display_to_native_sim_coords=lambda col, row, image_shape: (
            float(col) + 0.25,
            float(row) + 0.75,
        ),
        native_sim_to_display_coords=lambda col, row, image_shape: (
            float(col) - 90.0,
            float(row) - 89.0,
        ),
        simulate_ideal_hkl_native_center=lambda h, k, l, av, cv: (100.5, 101.5),
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
    assert kwargs["prefix"].startswith("Ideal HKL center (click")
    assert kwargs["clicked_display"] == (9.8, 12.4)
    assert kwargs["clicked_native"] == (10.25, 12.75)
    assert kwargs["selected_display"] == (10.5, 12.5)
    assert kwargs["selected_native"] == (100.5, 101.5)
    assert kwargs["sync_hkl_vars"] is True
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
