from types import SimpleNamespace

import numpy as np

from ra_sim.gui import qr_cylinder_overlay


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
