from types import SimpleNamespace

import numpy as np

from ra_sim.gui import qr_cylinder_overlay


def _overlay_config(*, render_in_caked_space: bool) -> qr_cylinder_overlay.QrCylinderOverlayRenderConfig:
    return qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=render_in_caked_space,
        image_size=64,
        display_rotate_k=-1,
        center_col=10.0,
        center_row=11.0,
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
        pixel_size_m=1.0e-4,
        wavelength=1.54,
        n2=1.1 + 0.0j,
    )


def test_qr_cylinder_overlay_config_and_signature_normalize_values() -> None:
    config = qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=1,
        image_size="512",
        display_rotate_k="-1",
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
        pixel_size_m="0.0001",
        wavelength="1.54",
        n2=1.2 + 0.3j,
        phi_samples="361",
    )

    signature = qr_cylinder_overlay.build_qr_cylinder_overlay_signature(
        [
            {"source": "primary", "m": 1, "qr": 0.12345678901},
            {"source": "secondary", "m": 2, "qr": 0.5},
        ],
        config=config,
    )

    assert config.render_in_caked_space is True
    assert config.image_size == 512
    assert config.display_rotate_k == -1
    assert config.center_col == 255.5
    assert config.wavelength == 1.54
    assert config.phi_samples == 361
    assert signature[0] == (
        ("primary", 1, round(0.12345678901, 10)),
        ("secondary", 2, 0.5),
    )
    assert signature[1] is True
    assert signature[2] == 512
    assert signature[-2:] == (1.2, 0.3)


def test_qr_cylinder_overlay_path_builder_projects_detector_space_paths() -> None:
    calls = []
    config = qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=False,
        image_size=64,
        display_rotate_k=-1,
        center_col=10.0,
        center_row=11.0,
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
        pixel_size_m=1.0e-4,
        wavelength=1.54,
        n2=1.1 + 0.0j,
    )

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        calls.append((qr_value, geometry, wavelength, n2, phi_samples))
        return [
            SimpleNamespace(
                detector_col=np.asarray([1.0, 2.0, 3.0], dtype=float),
                detector_row=np.asarray([4.0, 5.0, 6.0], dtype=float),
                valid_mask=np.asarray([True, False, True], dtype=bool),
            ),
            SimpleNamespace(
                detector_col=np.asarray([9.0], dtype=float),
                detector_row=np.asarray([10.0], dtype=float),
                valid_mask=np.asarray([True], dtype=bool),
            ),
        ]

    paths = qr_cylinder_overlay.build_qr_cylinder_overlay_paths(
        [{"source": "primary", "m": 1, "qr": 0.25}],
        config=config,
        two_theta_map=None,
        phi_map_deg=None,
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        project_traces=_project_traces,
    )

    assert len(paths) == 1
    assert paths[0]["source"] == "primary"
    assert paths[0]["qr"] == 0.25
    np.testing.assert_allclose(paths[0]["cols"], [65.0, np.nan, 67.0], equal_nan=True)
    np.testing.assert_allclose(paths[0]["rows"], [68.0, np.nan, 70.0], equal_nan=True)
    assert calls[0][0] == 0.25
    assert calls[0][2] == 1.54
    assert calls[0][3] == 1.1 + 0.0j
    assert calls[0][4] == 721


def test_qr_cylinder_overlay_path_builder_projects_caked_space_paths() -> None:
    config = qr_cylinder_overlay.build_qr_cylinder_overlay_render_config(
        render_in_caked_space=True,
        image_size=4,
        display_rotate_k=-1,
        center_col=10.0,
        center_row=11.0,
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
        pixel_size_m=1.0e-4,
        wavelength=1.54,
        n2=1.1 + 0.0j,
    )
    two_theta_map = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=float,
    )
    phi_map = np.asarray(
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [90.0, 100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0, 160.0],
        ],
        dtype=float,
    )

    paths = qr_cylinder_overlay.build_qr_cylinder_overlay_paths(
        [{"source": "secondary", "m": 2, "qr": 0.5}],
        config=config,
        two_theta_map=two_theta_map,
        phi_map_deg=phi_map,
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        project_traces=lambda **_kwargs: [
            SimpleNamespace(
                detector_col=np.asarray([0.0, 1.0, 2.0], dtype=float),
                detector_row=np.asarray([0.0, 1.0, 2.0], dtype=float),
                valid_mask=np.asarray([True, True, True], dtype=bool),
            )
        ],
    )

    assert len(paths) == 1
    assert paths[0]["source"] == "secondary"
    np.testing.assert_allclose(paths[0]["cols"], [1.0, 6.0, 11.0], equal_nan=True)
    np.testing.assert_allclose(paths[0]["rows"], [10.0, 60.0, 110.0], equal_nan=True)


def test_refresh_runtime_qr_cylinder_overlay_clears_when_disabled(
    monkeypatch,
) -> None:
    clear_calls = []

    monkeypatch.setattr(
        qr_cylinder_overlay.gui_overlays,
        "clear_artists",
        lambda artists, *, draw_idle=None, redraw=True: clear_calls.append(
            (list(artists), draw_idle, redraw)
        ),
    )

    bindings = qr_cylinder_overlay.QrCylinderOverlayRuntimeBindings(
        ax=object(),
        overlay_artists=["artist"],
        overlay_cache={"signature": "sig", "paths": ["path"]},
        overlay_enabled_factory=lambda: False,
        get_active_entries=lambda: [],
        render_config_factory=lambda: None,
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (None, None),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=lambda _text: None,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(
        bindings,
        redraw=False,
        update_status=True,
    )

    assert clear_calls == [(["artist"], bindings.draw_idle, False)]
    assert bindings.overlay_cache == {"signature": "sig", "paths": ["path"]}


def test_refresh_runtime_qr_cylinder_overlay_resets_cache_for_empty_entries(
    monkeypatch,
) -> None:
    clear_calls = []
    status_messages = []

    monkeypatch.setattr(
        qr_cylinder_overlay.gui_overlays,
        "clear_artists",
        lambda artists, *, draw_idle=None, redraw=True: clear_calls.append(
            (list(artists), redraw)
        ),
    )

    bindings = qr_cylinder_overlay.QrCylinderOverlayRuntimeBindings(
        ax=object(),
        overlay_artists=["artist"],
        overlay_cache={"signature": "old", "paths": ["old-path"]},
        overlay_enabled_factory=lambda: True,
        get_active_entries=lambda: [],
        render_config_factory=lambda: _overlay_config(render_in_caked_space=False),
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (None, None),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=status_messages.append,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(
        bindings,
        redraw=True,
        update_status=True,
    )

    assert bindings.overlay_cache == {"signature": None, "paths": []}
    assert clear_calls == [(["artist"], True)]
    assert status_messages == [
        "Qr cylinder overlay unavailable: no active Bragg Qr groups."
    ]


def test_refresh_runtime_qr_cylinder_overlay_builds_and_reuses_cached_paths(
    monkeypatch,
) -> None:
    build_calls = []
    draw_calls = []
    status_messages = []
    entries = [{"source": "primary", "m": 1, "qr": 0.25}]
    fake_paths = [{"source": "primary", "qr": 0.25, "cols": [1.0], "rows": [2.0]}]

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "build_qr_cylinder_overlay_paths",
        lambda active_entries, **kwargs: build_calls.append((active_entries, kwargs))
        or fake_paths,
    )
    monkeypatch.setattr(
        qr_cylinder_overlay.gui_overlays,
        "draw_qr_cylinder_overlay_paths",
        lambda ax, paths, **kwargs: draw_calls.append((ax, paths, kwargs)),
    )

    bindings = qr_cylinder_overlay.QrCylinderOverlayRuntimeBindings(
        ax="axis",
        overlay_artists=[],
        overlay_cache={"signature": None, "paths": []},
        overlay_enabled_factory=lambda: True,
        get_active_entries=lambda: list(entries),
        render_config_factory=lambda: _overlay_config(render_in_caked_space=False),
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector maps should not be requested for detector view")
        ),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=status_messages.append,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(
        bindings,
        redraw=True,
        update_status=True,
    )
    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(
        bindings,
        redraw=False,
        update_status=False,
    )

    assert len(build_calls) == 1
    assert build_calls[0][0] == entries
    assert build_calls[0][1]["two_theta_map"] is None
    assert build_calls[0][1]["phi_map_deg"] is None
    assert bindings.overlay_cache["paths"] == fake_paths
    assert len(draw_calls) == 2
    assert draw_calls[0][0] == "axis"
    assert draw_calls[0][1] == fake_paths
    assert status_messages == [
        "Showing analytic Qr-cylinder traces for 1 active Qr groups."
    ]


def test_refresh_runtime_qr_cylinder_overlay_passes_detector_maps_for_caked_view(
    monkeypatch,
) -> None:
    build_calls = []
    draw_calls = []
    ai = object()
    two_theta_map = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    phi_map_deg = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "build_qr_cylinder_overlay_paths",
        lambda active_entries, **kwargs: build_calls.append(kwargs)
        or [{"source": "secondary", "qr": 0.5, "cols": [1.0], "rows": [2.0]}],
    )
    monkeypatch.setattr(
        qr_cylinder_overlay.gui_overlays,
        "draw_qr_cylinder_overlay_paths",
        lambda ax, paths, **kwargs: draw_calls.append((ax, paths)),
    )

    bindings = qr_cylinder_overlay.QrCylinderOverlayRuntimeBindings(
        ax="axis",
        overlay_artists=[],
        overlay_cache={"signature": None, "paths": []},
        overlay_enabled_factory=lambda: True,
        get_active_entries=lambda: [{"source": "secondary", "m": 2, "qr": 0.5}],
        render_config_factory=lambda: _overlay_config(render_in_caked_space=True),
        ai_factory=lambda: ai,
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta_map, phi_map_deg) if ai_arg is ai else (None, None)
        ),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=lambda _text: None,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(bindings, redraw=True)

    assert build_calls[0]["two_theta_map"] is two_theta_map
    assert build_calls[0]["phi_map_deg"] is phi_map_deg
    assert draw_calls[0][0] == "axis"


def test_qr_cylinder_overlay_toggle_and_callback_helpers_use_live_bindings(
    monkeypatch,
) -> None:
    clear_calls = []
    status_messages = []
    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        qr_cylinder_overlay.gui_overlays,
        "clear_artists",
        lambda artists, *, draw_idle=None, redraw=True: clear_calls.append(
            (list(artists), redraw)
        ),
    )

    bindings = qr_cylinder_overlay.QrCylinderOverlayRuntimeBindings(
        ax=object(),
        overlay_artists=["artist"],
        overlay_cache={"signature": None, "paths": []},
        overlay_enabled_factory=lambda: False,
        get_active_entries=lambda: [],
        render_config_factory=lambda: None,
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (None, None),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=status_messages.append,
    )

    qr_cylinder_overlay.toggle_runtime_qr_cylinder_overlay(bindings)

    assert clear_calls == [(["artist"], True)]
    assert status_messages == ["Qr cylinder overlay hidden."]

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "refresh_runtime_qr_cylinder_overlay",
        lambda bindings_arg, *, redraw=True, update_status=False: callback_calls.append(
            ("refresh", bindings_arg, redraw, update_status)
        ),
    )
    monkeypatch.setattr(
        qr_cylinder_overlay,
        "toggle_runtime_qr_cylinder_overlay",
        lambda bindings_arg: callback_calls.append(("toggle", bindings_arg)),
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    refresh_callback = qr_cylinder_overlay.make_runtime_qr_cylinder_overlay_refresh_callback(
        build_bindings
    )
    toggle_callback = qr_cylinder_overlay.make_runtime_qr_cylinder_overlay_toggle_callback(
        build_bindings
    )

    refresh_callback(redraw=False, update_status=True)
    toggle_callback()

    assert callback_calls == [
        ("refresh", "bindings-1", False, True),
        ("toggle", "bindings-2"),
    ]
