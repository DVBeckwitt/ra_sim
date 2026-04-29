from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from ra_sim.config import loader
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


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_config_dir(tmp_path: Path, *, debug: dict | None = None) -> Path:
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    _write_yaml(cfg / "file_paths.yaml", {})
    _write_yaml(cfg / "dir_paths.yaml", {})
    _write_yaml(cfg / "materials.yaml", {})
    _write_yaml(cfg / "instrument.yaml", {})
    if debug is not None:
        _write_yaml(cfg / "debug.yaml", debug)
    return cfg


@pytest.fixture(autouse=True)
def _reset_loader_cache() -> None:
    loader.clear_config_cache()
    yield
    loader.clear_config_cache()


def _caked_projection_context(
    *,
    detector_shape: tuple[int, int] = (4, 4),
) -> dict[str, object]:
    bundle = qr_cylinder_overlay.CakeTransformBundle(
        detector_shape=detector_shape,
        radial_deg=np.array([1.0, 2.0, 3.0], dtype=float),
        raw_azimuth_deg=np.array([-100.0, -90.0, -80.0], dtype=float),
        gui_azimuth_deg=np.array([10.0, 0.0, -10.0], dtype=float),
        lut=SimpleNamespace(),
    )
    return {
        "detector_shape": detector_shape,
        "radial_axis": np.array([1.0, 2.0, 3.0], dtype=float),
        "azimuth_axis": np.array([-10.0, 0.0, 10.0], dtype=float),
        "raw_azimuth_axis": np.array([-100.0, -90.0, -80.0], dtype=float),
        "transform_bundle": bundle,
    }


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
    assert signature[-3:-1] == (1.2, 0.3)
    assert signature[-1] is None


def test_qr_cylinder_overlay_render_config_factory_reads_live_values() -> None:
    values = {
        "render_in_caked_space": False,
        "center_col": 10.0,
        "center_row": 11.0,
    }

    factory = qr_cylinder_overlay.make_runtime_qr_cylinder_overlay_render_config_factory(
        render_in_caked_space_factory=lambda: values["render_in_caked_space"],
        image_size="64",
        display_rotate_k="-1",
        center_col_factory=lambda: values["center_col"],
        center_row_factory=lambda: values["center_row"],
        distance_cor_to_detector_factory=lambda: "123.0",
        gamma_deg_factory=lambda: "1.5",
        Gamma_deg_factory=lambda: 2.5,
        chi_deg_factory=lambda: 3.5,
        psi_deg_factory=lambda: 4.5,
        psi_z_deg_factory=lambda: 5.5,
        zs_factory=lambda: 6.5,
        zb_factory=lambda: 7.5,
        theta_initial_deg_factory=lambda: 8.5,
        cor_angle_deg_factory=lambda: 9.5,
        pixel_size_m="0.0001",
        wavelength="1.54",
        n2=1.1 + 0.0j,
        phi_samples="361",
    )

    values["render_in_caked_space"] = True
    values["center_col"] = 21.0
    values["center_row"] = 22.0
    config = factory()

    assert config.render_in_caked_space is True
    assert config.image_size == 64
    assert config.display_rotate_k == -1
    assert config.center_col == 21.0
    assert config.center_row == 22.0
    assert config.phi_samples == 361


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
        projection_context=None,
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
    context = _caked_projection_context()

    def _project_to_caked(bundle, col, row):
        assert bundle is context["transform_bundle"]
        if (float(col), float(row)) == (0.0, 0.0):
            return (1.0, 10.0)
        if (float(col), float(row)) == (1.0, 1.0):
            return (6.0, 60.0)
        if (float(col), float(row)) == (2.0, 2.0):
            return (11.0, 110.0)
        return (None, None)

    detector_pixel_to_caked_bin_original = qr_cylinder_overlay.detector_pixel_to_caked_bin
    qr_cylinder_overlay.detector_pixel_to_caked_bin = _project_to_caked

    try:
        paths = qr_cylinder_overlay.build_qr_cylinder_overlay_paths(
            [{"source": "secondary", "m": 2, "qr": 0.5}],
            config=config,
            projection_context=context,
            native_sim_to_display_coords=lambda col, row, shape: (col, row),
            project_traces=lambda **_kwargs: [
                SimpleNamespace(
                    detector_col=np.asarray([0.0, 1.0, 2.0], dtype=float),
                    detector_row=np.asarray([0.0, 1.0, 2.0], dtype=float),
                    valid_mask=np.asarray([True, True, True], dtype=bool),
                )
            ],
        )
    finally:
        qr_cylinder_overlay.detector_pixel_to_caked_bin = detector_pixel_to_caked_bin_original

    assert len(paths) == 1
    assert paths[0]["source"] == "secondary"
    np.testing.assert_allclose(paths[0]["cols"], [1.0, 6.0, 11.0], equal_nan=True)
    np.testing.assert_allclose(paths[0]["rows"], [10.0, 60.0, 110.0], equal_nan=True)


def test_build_qr_cylinder_caked_band_masks_clamps_specular_low_qr() -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()
    radial_axis = np.asarray([1.0, 2.0, 3.0], dtype=float)
    azimuth_axis = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    qr_values = []

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        qr_values.append(float(qr_value))
        return []

    result = qr_cylinder_overlay.build_qr_cylinder_caked_band_masks(
        [{"key": "rod-1", "source": "primary", "m": 1, "qr": 0.02}],
        config=config,
        projection_context=projection_context,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        delta_qr=0.05,
        project_traces=_project_traces,
    )

    assert qr_values == [0.0, 0.07]
    assert result is not None
    assert result["union_mask"].shape == (3, 3)
    assert not np.any(result["union_mask"])


def test_build_qr_cylinder_caked_band_masks_pairs_boundaries_by_branch_sign(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()
    interpolate_codes: list[int] = []

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        if float(qr_value) < 0.25:
            return [
                SimpleNamespace(
                    branch_sign=1,
                    detector_col=np.asarray([11.0, 11.5], dtype=float),
                    detector_row=np.asarray([0.0, 1.0], dtype=float),
                    valid_mask=np.asarray([True, True], dtype=bool),
                ),
                SimpleNamespace(
                    branch_sign=-1,
                    detector_col=np.asarray([99.0, 99.5], dtype=float),
                    detector_row=np.asarray([0.0, 1.0], dtype=float),
                    valid_mask=np.asarray([True, True], dtype=bool),
                ),
            ]
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([12.0, 12.5], dtype=float),
                detector_row=np.asarray([0.0, 1.0], dtype=float),
                valid_mask=np.asarray([True, True], dtype=bool),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        code = int(np.asarray(detector_cols, dtype=float)[0])
        interpolate_codes.append(code)
        if code == 11:
            return (
                np.asarray([1.0, 1.0], dtype=float),
                np.asarray([-1.0, 1.0], dtype=float),
            )
        if code == 12:
            return (
                np.asarray([3.0, 3.0], dtype=float),
                np.asarray([-1.0, 1.0], dtype=float),
            )
        raise AssertionError(f"unexpected branch code {code}")

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    result = qr_cylinder_overlay.build_qr_cylinder_caked_band_masks(
        [{"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25}],
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 2.0, 3.0], dtype=float),
        azimuth_axis=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        delta_qr=0.05,
        project_traces=_project_traces,
    )

    assert interpolate_codes == [11, 12]
    assert result is not None
    assert np.any(result["union_mask"])


def test_build_qr_cylinder_caked_band_masks_does_not_fill_across_phi_seam(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        code = 1.0 if float(qr_value) < 0.25 else 2.0
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([code, code + 0.1, code + 0.2], dtype=float),
                detector_row=np.asarray([0.0, 1.0, 2.0], dtype=float),
                valid_mask=np.asarray([True, True, True], dtype=bool),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        code = int(np.floor(np.asarray(detector_cols, dtype=float)[0]))
        if code == 1:
            return (
                np.asarray([1.0, 2.0, 3.0], dtype=float),
                np.asarray([178.0, 179.0, -179.0], dtype=float),
            )
        if code == 2:
            return (
                np.asarray([3.0, 4.0, 5.0], dtype=float),
                np.asarray([178.0, 179.0, -179.0], dtype=float),
            )
        raise AssertionError(f"unexpected seam code {code}")

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    result = qr_cylinder_overlay.build_qr_cylinder_caked_band_masks(
        [{"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25}],
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float),
        azimuth_axis=np.asarray([-179.0, 0.0, 179.0], dtype=float),
        delta_qr=0.05,
        project_traces=_project_traces,
    )

    assert result is not None
    assert not np.any(result["union_mask"][1])


def test_build_qr_cylinder_caked_band_masks_rasterizes_union_and_stamps_edges(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        code = 1.0 if float(qr_value) < 0.25 else 2.0
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([code, code + 0.1, code + 0.2], dtype=float),
                detector_row=np.asarray([0.0, 1.0, 2.0], dtype=float),
                valid_mask=np.asarray([True, True, True], dtype=bool),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        code = int(np.floor(np.asarray(detector_cols, dtype=float)[0]))
        if code == 1:
            return (
                np.asarray([1.0, 1.0, 1.0], dtype=float),
                np.asarray([-1.0, 0.0, 1.0], dtype=float),
            )
        if code == 2:
            return (
                np.asarray([3.0, 3.0, 3.0], dtype=float),
                np.asarray([-1.0, 0.0, 1.0], dtype=float),
            )
        raise AssertionError(f"unexpected rectangle code {code}")

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    result = qr_cylinder_overlay.build_qr_cylinder_caked_band_masks(
        [{"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25}],
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 2.0, 3.0], dtype=float),
        azimuth_axis=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        delta_qr=0.05,
        project_traces=_project_traces,
    )

    assert result is not None
    np.testing.assert_array_equal(
        result["union_mask"],
        result["masks_by_key"]["rod-1"],
    )
    assert result["union_mask"][1, 1]
    assert result["union_mask"][:, 0].any()
    assert result["union_mask"][:, 2].any()


def test_build_qr_cylinder_caked_band_masks_keeps_finite_points_around_isolated_nan_azimuth(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        code = 1.0 if float(qr_value) < 0.25 else 2.0
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([code, code + 0.1, code + 0.2, code + 0.3, code + 0.4]),
                detector_row=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                valid_mask=np.asarray([True, True, True, True, True], dtype=bool),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        code = int(np.floor(np.asarray(detector_cols, dtype=float)[0]))
        two_theta = 1.0 if code == 1 else 3.0
        return (
            np.asarray([two_theta, np.nan, two_theta, np.nan, two_theta], dtype=float),
            np.asarray([-90.0, np.nan, 0.0, np.nan, 90.0], dtype=float),
        )

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    result = qr_cylinder_overlay.build_qr_cylinder_caked_band_masks(
        [{"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25}],
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 2.0, 3.0], dtype=float),
        azimuth_axis=np.asarray([-90.0, 0.0, 90.0], dtype=float),
        delta_qr=0.05,
        project_traces=_project_traces,
    )

    assert result is not None
    mask = result["union_mask"]
    assert mask.shape == (3, 3)
    assert np.all(mask[:, 1])
    assert np.all(mask[:, 0])
    assert np.all(mask[:, 2])


def test_build_selected_qr_rod_qz_caked_mask_delta_qr_changes_sparse_trace_shape(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()
    qr0 = 0.25

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        qr_value = float(qr_value)
        two_theta = 2.0 + 10.0 * (qr_value - qr0)
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray(
                    [qr_value, qr_value + 0.1, qr_value + 0.2, qr_value + 0.3, qr_value + 0.4],
                    dtype=float,
                ),
                detector_row=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
                qz=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
                valid_mask=np.asarray([True, True, True, True, True], dtype=bool),
                two_theta_value=float(two_theta),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        qr_value = float(np.asarray(detector_cols, dtype=float)[0])
        two_theta = 2.0 + 10.0 * (qr_value - qr0)
        return (
            np.asarray([two_theta, np.nan, two_theta, np.nan, two_theta], dtype=float),
            np.asarray([-90.0, np.nan, 0.0, np.nan, 90.0], dtype=float),
        )

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    common_kwargs = dict(
        selected_entry={"key": "rod-1", "source": "primary", "m": 1, "qr": qr0},
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float),
        azimuth_axis=np.asarray([-90.0, 0.0, 90.0], dtype=float),
        qz_min=-1.0,
        qz_max=1.0,
        phi_min=-90.0,
        phi_max=90.0,
        project_traces=_project_traces,
    )

    narrow = qr_cylinder_overlay.build_selected_qr_rod_qz_caked_mask(
        **common_kwargs,
        delta_qr=0.05,
    )
    wide = qr_cylinder_overlay.build_selected_qr_rod_qz_caked_mask(
        **common_kwargs,
        delta_qr=0.10,
    )

    assert narrow is not None
    assert wide is not None
    assert narrow["signature"] != wide["signature"]
    assert int(np.count_nonzero(wide["mask"])) > int(np.count_nonzero(narrow["mask"]))
    assert not np.any(narrow["mask"][:, 0])
    assert not np.any(narrow["mask"][:, 4])
    assert np.all(wide["mask"][:, 0])
    assert np.all(wide["mask"][:, 4])


def test_build_selected_qr_rod_qz_caked_mask_uses_projected_qz_window(
    monkeypatch,
) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()

    def _project_traces(*, qr_value, geometry, wavelength, n2, phi_samples):
        if float(qr_value) < 0.25:
            code = 1.0
            qz = np.zeros(4, dtype=float)
        elif float(qr_value) > 0.25:
            code = 3.0
            qz = np.zeros(4, dtype=float)
        else:
            code = 2.0
            qz = np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=float)
        return [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([code, code + 0.1, code + 0.2, code + 0.3]),
                detector_row=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
                qz=qz,
                valid_mask=np.asarray([True, True, True, True], dtype=bool),
            )
        ]

    def _interpolate_trace_to_caked_coords(
        *,
        detector_cols,
        detector_rows,
        valid_mask,
        projection_context,
        two_theta_limits,
    ):
        code = int(np.floor(np.asarray(detector_cols, dtype=float)[0]))
        offsets = {1: 0.0, 2: 0.1, 3: 0.2}
        phi_values = (
            np.asarray([0.0, -2.0, 0.0, 0.0], dtype=float)
            if code == 2
            else np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)
        )
        return (
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float) + offsets[code],
            phi_values,
        )

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        _interpolate_trace_to_caked_coords,
    )

    result = qr_cylinder_overlay.build_selected_qr_rod_qz_caked_mask(
        selected_entry={"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25},
        config=config,
        projection_context=projection_context,
        radial_axis=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
        azimuth_axis=np.asarray([-1.0, 0.0, 1.0], dtype=float),
        delta_qr=0.05,
        qz_min=-1.0,
        qz_max=1.0,
        phi_min=-0.5,
        phi_max=0.5,
        project_traces=_project_traces,
    )

    assert result is not None
    mask = result["mask"]
    assert mask.shape == (3, 4)
    assert not mask[1, 1]
    assert mask[1, 2]
    assert not mask[1, 0]
    assert not mask[1, 3]


def test_project_selected_qr_rod_caked_samples_carries_physical_qz(monkeypatch) -> None:
    config = _overlay_config(render_in_caked_space=True)
    projection_context = _caked_projection_context()

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "interpolate_trace_to_caked_coords",
        lambda **_kwargs: (
            np.asarray([1.0, 2.0, np.nan], dtype=float),
            np.asarray([0.0, 1.0, np.nan], dtype=float),
        ),
    )

    result = qr_cylinder_overlay.project_selected_qr_rod_caked_samples(
        selected_entry={"key": "rod-1", "source": "primary", "m": 1, "qr": 0.25},
        config=config,
        projection_context=projection_context,
        project_traces=lambda **_kwargs: [
            SimpleNamespace(
                branch_sign=1,
                detector_col=np.asarray([1.0, 2.0, 3.0], dtype=float),
                detector_row=np.asarray([4.0, 5.0, 6.0], dtype=float),
                qz=np.asarray([-0.5, 0.75, 2.0], dtype=float),
                valid_mask=np.asarray([True, True, True], dtype=bool),
            )
        ],
    )

    assert result is not None
    np.testing.assert_allclose(result["two_theta"], [1.0, 2.0])
    np.testing.assert_allclose(result["phi"], [0.0, 1.0])
    np.testing.assert_allclose(result["qz"], [-0.5, 0.75])


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
    assert build_calls[0][1]["projection_context"] is None
    assert bindings.overlay_cache["paths"] == fake_paths
    assert len(draw_calls) == 2
    assert draw_calls[0][0] == "axis"
    assert draw_calls[0][1] == fake_paths
    assert status_messages == [
        "Showing analytic Qr-cylinder traces for 1 active Qr groups."
    ]


def test_refresh_runtime_qr_cylinder_overlay_can_disable_path_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={
            "debug": {
                "cache": {
                    "families": {
                        "qr_cylinder_overlay": "never",
                    }
                }
            }
        },
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    build_calls = []
    draw_calls = []
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
        get_detector_angular_maps=lambda _ai: (None, None),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=lambda _text: None,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(bindings, redraw=True)
    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(bindings, redraw=False)

    assert len(build_calls) == 2
    assert len(draw_calls) == 2
    assert bindings.overlay_cache == {"signature": None, "paths": []}


def test_refresh_runtime_qr_cylinder_overlay_passes_caked_projection_context_for_caked_view(
    monkeypatch,
) -> None:
    build_calls = []
    draw_calls = []
    projection_context = _caked_projection_context()

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
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("caked overlay should not request detector angle maps")
        ),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        get_caked_projection_context=lambda: projection_context,
        draw_idle=lambda: None,
        set_status_text=lambda _text: None,
    )

    qr_cylinder_overlay.refresh_runtime_qr_cylinder_overlay(bindings, redraw=True)

    assert build_calls[0]["projection_context"] is projection_context
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


def test_invalidate_runtime_qr_cylinder_overlay_cache_resets_paths_and_artists(
    monkeypatch,
) -> None:
    clear_calls = []

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
        overlay_cache={"signature": "cached", "paths": ["path"]},
        overlay_enabled_factory=lambda: True,
        get_active_entries=lambda: [],
        render_config_factory=lambda: None,
        ai_factory=lambda: None,
        get_detector_angular_maps=lambda _ai: (None, None),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        draw_idle=lambda: None,
        set_status_text=lambda _text: None,
    )

    qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(
        bindings,
        clear_artists=True,
        redraw=False,
    )

    assert bindings.overlay_cache == {"signature": None, "paths": []}
    assert clear_calls == [(["artist"], False)]
