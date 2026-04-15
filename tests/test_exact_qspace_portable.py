import numpy as np
import pytest

from ra_sim.simulation import exact_qspace
from ra_sim.simulation import exact_qspace_portable


@pytest.fixture(autouse=True)
def _clear_exact_qspace_process_caches() -> None:
    exact_qspace_portable._clear_shared_exact_qspace_caches()
    yield
    exact_qspace_portable._clear_shared_exact_qspace_caches()


def _base_geometry(**overrides: float) -> dict[str, float]:
    geometry = {
        "pixel_size_m": 1.0e-4,
        "distance_m": 0.5,
        "center_row_px": 32.0,
        "center_col_px": 32.0,
        "wavelength_m": 1.24e-10,
        "gamma_deg": 0.0,
        "Gamma_deg": 0.0,
        "chi_deg": 0.0,
        "psi_deg": 0.0,
        "psi_z_deg": 0.0,
        "theta_initial_deg": 0.0,
        "cor_angle_deg": 0.0,
        "zs": 0.0,
        "zb": 0.0,
    }
    geometry.update(overrides)
    return geometry


def _exact_geometry(**overrides: float) -> exact_qspace.DetectorQSpaceGeometry:
    return exact_qspace.DetectorQSpaceGeometry(**_base_geometry(**overrides))


def _convert_image_to_q_space(
    image: np.ndarray,
    *,
    method: str,
    npt_rad: int = 128,
    npt_azim: int = 128,
    **geometry_overrides: float,
) -> exact_qspace.DetectorQSpaceResult:
    return exact_qspace_portable.convert_image_to_q_space(
        np.asarray(image, dtype=np.float32),
        **_base_geometry(**geometry_overrides),
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        correct_solid_angle=False,
        method=method,
    )


def _build_low_qr_ridge_image(
    shape: tuple[int, int],
    *,
    chi_deg: float = 10.0,
) -> np.ndarray:
    geometry = _exact_geometry(
        center_row_px=float(shape[0]) / 2.0,
        center_col_px=float(shape[1]) / 2.0,
        chi_deg=chi_deg,
    )
    signed_qr_map, _, _ = exact_qspace.detector_corner_q_maps(shape, geometry)
    pixel_qr = np.mean(
        np.abs(
            np.stack(
                (
                    signed_qr_map[:-1, :-1],
                    signed_qr_map[1:, :-1],
                    signed_qr_map[1:, 1:],
                    signed_qr_map[:-1, 1:],
                ),
                axis=0,
            )
        ),
        axis=0,
    )
    ridge_cols = np.argmin(pixel_qr, axis=1)
    image = np.zeros(shape, dtype=np.float32)
    image[np.arange(shape[0], dtype=np.int64), ridge_cols] = 1.0
    return image


def test_detector_corner_direct_beam_maps_to_q_zero() -> None:
    geometry = _exact_geometry()
    signed_qr_map, qz_map, _ = exact_qspace.detector_corner_q_maps((64, 64), geometry)

    assert signed_qr_map[32, 32] == pytest.approx(0.0, abs=1.0e-15)
    assert qz_map[32, 32] == pytest.approx(0.0, abs=1.0e-15)
    assert np.min(np.hypot(np.abs(signed_qr_map), qz_map)) < 1.0e-12


def test_direct_exact_and_lut_paths_conserve_total_intensity() -> None:
    rng = np.random.default_rng(1234)
    image = rng.uniform(0.2, 5.0, size=(9, 11)).astype(np.float32)

    exact_result = _convert_image_to_q_space(image, method="exact", npt_rad=72, npt_azim=60)
    lut_result = _convert_image_to_q_space(image, method="lut", npt_rad=72, npt_azim=60)

    expected_signal = float(np.sum(image))
    expected_weight = float(image.size)

    assert float(np.sum(exact_result.sum_signal)) == pytest.approx(expected_signal, rel=1.0e-6)
    assert float(np.sum(lut_result.sum_signal)) == pytest.approx(expected_signal, rel=1.0e-6)
    assert float(np.sum(exact_result.sum_normalization)) == pytest.approx(expected_weight, rel=1.0e-6)
    assert float(np.sum(lut_result.sum_normalization)) == pytest.approx(expected_weight, rel=1.0e-6)
    assert float(np.sum(exact_result.count)) == pytest.approx(expected_weight, rel=1.0e-6)
    assert float(np.sum(lut_result.count)) == pytest.approx(expected_weight, rel=1.0e-6)
    np.testing.assert_allclose(exact_result.qr, lut_result.qr, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(exact_result.qz, lut_result.qz, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(exact_result.sum_signal, lut_result.sum_signal, rtol=5.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(exact_result.sum_normalization, lut_result.sum_normalization, rtol=5.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(exact_result.intensity, lut_result.intensity, rtol=5.0e-5, atol=1.0e-6)


def test_specular_ridge_stays_near_qr_zero() -> None:
    shape = (64, 64)
    ridge_image = _build_low_qr_ridge_image(shape)
    result = _convert_image_to_q_space(
        ridge_image,
        method="exact",
        npt_rad=128,
        npt_azim=128,
        chi_deg=10.0,
    )

    total_signal = float(np.sum(result.sum_signal))
    qr_profile = np.sum(result.sum_signal, axis=0) / total_signal
    qr_weighted_mean = float(np.sum(result.sum_signal * result.qr[None, :]) / total_signal)

    assert qr_profile[:3].sum() > 0.95
    assert qr_weighted_mean / float(result.qr[-1]) < 0.05


def test_qr_zero_seam_has_no_gap_or_spike() -> None:
    result = _convert_image_to_q_space(
        _build_low_qr_ridge_image((64, 64)),
        method="exact",
        npt_rad=128,
        npt_azim=128,
        chi_deg=10.0,
    )

    first_qr_profile = np.asarray(result.sum_signal[:, 0], dtype=np.float64)
    seam_index = int(np.argmin(np.abs(result.qz)))

    assert first_qr_profile[seam_index] > 0.0
    assert first_qr_profile[seam_index - 1] < first_qr_profile[seam_index]
    assert first_qr_profile[seam_index + 2] < first_qr_profile[seam_index + 1]
    assert first_qr_profile[seam_index] == pytest.approx(
        first_qr_profile[seam_index + 1],
        rel=1.0e-6,
        abs=1.0e-8,
    )
    assert first_qr_profile[seam_index - 1] == pytest.approx(
        first_qr_profile[seam_index + 2],
        rel=1.0e-6,
        abs=1.0e-8,
    )


def test_off_specular_seam_crossing_stays_off_qr_zero() -> None:
    shape = (64, 64)
    geometry = _exact_geometry(chi_deg=10.0, theta_initial_deg=0.5)
    qx_map, qy_map, _ = exact_qspace._detector_corner_sample_q_maps(shape, geometry)
    target: tuple[float, int, int] | None = None
    for row in range(shape[0]):
        for col in range(shape[1]):
            qy_corners = np.array(
                [
                    qy_map[row, col],
                    qy_map[row + 1, col],
                    qy_map[row + 1, col + 1],
                    qy_map[row, col + 1],
                ],
                dtype=np.float64,
            )
            if not (float(np.min(qy_corners)) < 0.0 < float(np.max(qy_corners))):
                continue
            qx_corners = np.array(
                [
                    qx_map[row, col],
                    qx_map[row + 1, col],
                    qx_map[row + 1, col + 1],
                    qx_map[row, col + 1],
                ],
                dtype=np.float64,
            )
            candidate = (float(np.mean(np.abs(qx_corners))), row, col)
            if target is None or candidate[0] > target[0]:
                target = candidate

    assert target is not None
    target_qr, row, col = target
    image = np.zeros(shape, dtype=np.float32)
    image[row, col] = 1.0

    for method in ("exact", "lut"):
        result = _convert_image_to_q_space(
            image,
            method=method,
            npt_rad=128,
            npt_azim=128,
            chi_deg=10.0,
            theta_initial_deg=0.5,
        )
        qr_profile = np.sum(np.asarray(result.sum_signal, dtype=np.float64), axis=0)
        peak_index = int(np.argmax(qr_profile))

        assert float(qr_profile[0]) < 0.05
        assert peak_index > 4
        assert float(result.qr[peak_index]) > 0.5 * float(target_qr)


def test_shared_geometry_reuses_detector_map_and_lut_cache(monkeypatch) -> None:
    map_calls = 0
    lut_calls = 0
    original_maps = exact_qspace_portable._detector_corner_sample_q_maps
    original_build_lut = exact_qspace_portable.build_detector_to_qspace_lut

    def _count_maps(*args, **kwargs):
        nonlocal map_calls
        map_calls += 1
        return original_maps(*args, **kwargs)

    def _count_lut(*args, **kwargs):
        nonlocal lut_calls
        lut_calls += 1
        return original_build_lut(*args, **kwargs)

    monkeypatch.setattr(exact_qspace_portable, "_detector_corner_sample_q_maps", _count_maps)
    monkeypatch.setattr(exact_qspace_portable, "build_detector_to_qspace_lut", _count_lut)

    sim_image = np.arange(64, dtype=np.float32).reshape(8, 8)
    bg_image = np.flipud(sim_image)
    sim_result = _convert_image_to_q_space(sim_image, method="lut", npt_rad=48, npt_azim=40)
    bg_result = _convert_image_to_q_space(bg_image, method="lut", npt_rad=48, npt_azim=40)

    assert map_calls == 1
    assert lut_calls == 1
    np.testing.assert_allclose(sim_result.qr, bg_result.qr, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(sim_result.qz, bg_result.qz, rtol=0.0, atol=1.0e-12)


def test_detector_distance_and_sample_tilt_move_q_map_in_expected_directions() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[40, 32] = 1.0

    near_result = _convert_image_to_q_space(image, method="exact", distance_m=0.25)
    far_result = _convert_image_to_q_space(image, method="exact", distance_m=1.0)
    untilted_result = _convert_image_to_q_space(image, method="exact", theta_initial_deg=0.0)
    tilted_result = _convert_image_to_q_space(image, method="exact", theta_initial_deg=0.5)

    untilted_qz_mean = float(
        np.sum(untilted_result.sum_signal * untilted_result.qz[:, None]) / np.sum(untilted_result.sum_signal)
    )
    tilted_qz_mean = float(
        np.sum(tilted_result.sum_signal * tilted_result.qz[:, None]) / np.sum(tilted_result.sum_signal)
    )

    assert float(far_result.qr[-1]) < float(near_result.qr[-1])
    assert float(abs(far_result.qz[0])) < float(abs(near_result.qz[0]))
    assert tilted_qz_mean > untilted_qz_mean + 1.0e-4
