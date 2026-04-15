import threading
import time

import numpy as np
import pytest

from ra_sim.simulation import exact_cake
from ra_sim.simulation import exact_cake_portable


@pytest.fixture(autouse=True)
def _clear_exact_cake_process_caches() -> None:
    exact_cake_portable._clear_shared_exact_cake_caches()
    yield
    exact_cake_portable._clear_shared_exact_cake_caches()


def _make_transform_bundle(
    *,
    detector_shape: tuple[int, int],
    radial_deg: np.ndarray,
    raw_azimuth_deg: np.ndarray,
    matrix: np.ndarray,
) -> exact_cake_portable.CakeTransformBundle:
    lut = exact_cake.DetectorCakeLUT(
        image_shape=tuple(int(v) for v in detector_shape),
        n_rad=int(len(radial_deg)),
        n_az=int(len(raw_azimuth_deg)),
        matrix=np.asarray(matrix, dtype=np.float32),
        count_flat=np.ones(int(len(radial_deg) * len(raw_azimuth_deg)), dtype=np.float64),
    )
    return exact_cake_portable.CakeTransformBundle(
        detector_shape=tuple(int(v) for v in detector_shape),
        radial_deg=np.asarray(radial_deg, dtype=np.float64),
        raw_azimuth_deg=np.asarray(raw_azimuth_deg, dtype=np.float64),
        gui_azimuth_deg=np.asarray(
            exact_cake_portable.raw_phi_to_gui_phi(raw_azimuth_deg),
            dtype=np.float64,
        ),
        lut=lut,
    )


def _make_identity_lut(
    image_shape: tuple[int, ...],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
) -> exact_cake.DetectorCakeLUT:
    detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
    return exact_cake.DetectorCakeLUT(
        image_shape=detector_shape,
        n_rad=int(len(radial_deg)),
        n_az=int(len(azimuthal_deg)),
        matrix=np.eye(int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(detector_shape)), dtype=np.float32),
        count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
    )


def test_fast_azimuthal_integrator_detector_maps_match_flat_geometry() -> None:
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=5.0,
        poni1=20.0,
        poni2=20.0,
        pixel1=2.0,
        pixel2=2.0,
    )

    two_theta = integrator.twoThetaArray(shape=(21, 21), unit="2th_deg")
    chi = integrator.chiArray(shape=(21, 21), unit="deg")

    rows = ((np.arange(21, dtype=np.float64) + 0.5) - 10.0) * 2.0
    cols = ((np.arange(21, dtype=np.float64) + 0.5) - 10.0) * 2.0
    yy = rows[:, None]
    xx = cols[None, :]
    expected_two_theta = np.degrees(np.arctan2(np.hypot(xx, yy), 5.0))
    expected_chi = np.degrees(np.arctan2(yy, xx))

    assert np.allclose(two_theta, expected_two_theta)
    assert np.allclose(chi, expected_chi)


def test_fast_azimuthal_integrator_detector_maps_reuse_readonly_cache() -> None:
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=5.0,
        poni1=20.0,
        poni2=20.0,
        pixel1=2.0,
        pixel2=2.0,
    )

    two_theta_first = integrator.twoThetaArray(shape=(21, 21), unit="2th_deg")
    chi_first = integrator.chiArray(shape=(21, 21), unit="deg")
    two_theta_second = integrator.twoThetaArray(shape=(21, 21), unit="2th_deg")
    chi_second = integrator.chiArray(shape=(21, 21), unit="deg")

    assert two_theta_first is two_theta_second
    assert chi_first is chi_second
    assert not two_theta_first.flags.writeable
    assert not chi_first.flags.writeable

    with pytest.raises(ValueError):
        two_theta_first[0, 0] = 0.0
    with pytest.raises(ValueError):
        chi_first[0, 0] = 0.0


def test_fast_azimuthal_integrator_detector_map_radian_requests_still_allocate() -> None:
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=5.0,
        poni1=20.0,
        poni2=20.0,
        pixel1=2.0,
        pixel2=2.0,
    )

    two_theta_deg = integrator.twoThetaArray(shape=(21, 21), unit="2th_deg")
    chi_deg = integrator.chiArray(shape=(21, 21), unit="deg")
    two_theta_rad = integrator.twoThetaArray(shape=(21, 21), unit="2th_rad")
    chi_rad = integrator.chiArray(shape=(21, 21), unit="rad")

    assert two_theta_rad is not two_theta_deg
    assert chi_rad is not chi_deg
    assert np.allclose(two_theta_rad, np.deg2rad(two_theta_deg))
    assert np.allclose(chi_rad, np.deg2rad(chi_deg))


def test_fast_azimuthal_integrator_detector_maps_cache_is_shared_across_instances(
    monkeypatch,
) -> None:
    map_calls: list[tuple[tuple[int, int], exact_cake_portable.PortableGeometry]] = []

    def _fake_maps(image_shape, geometry):
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        map_calls.append((detector_shape, geometry))
        return (
            np.full(detector_shape, 1.5, dtype=np.float64),
            np.full(detector_shape, -2.5, dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "detector_pixel_angular_maps", _fake_maps)

    integrator_a = exact_cake_portable.FastAzimuthalIntegrator(
        dist=5.0,
        poni1=20.0,
        poni2=20.0,
        pixel1=2.0,
        pixel2=2.0,
    )
    integrator_b = exact_cake_portable.FastAzimuthalIntegrator(
        dist=5.0,
        poni1=20.0,
        poni2=20.0,
        pixel1=2.0,
        pixel2=2.0,
    )

    two_theta_a = integrator_a.twoThetaArray(shape=(21, 21), unit="2th_deg")
    two_theta_b = integrator_b.twoThetaArray(shape=(21, 21), unit="2th_deg")

    assert len(map_calls) == 1
    assert two_theta_a is two_theta_b


def test_fast_azimuthal_integrator_warm_geometry_cache_prebuilds_common_caches(
    monkeypatch,
) -> None:
    map_calls: list[tuple[int, int]] = []
    norm_calls: list[tuple[int, int]] = []
    lut_calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_maps(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        map_calls.append(detector_shape)
        return (
            np.full(detector_shape, 1.0, dtype=np.float64),
            np.full(detector_shape, -1.0, dtype=np.float64),
        )

    def _fake_norm(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        norm_calls.append(detector_shape)
        return np.full(detector_shape, 2.0, dtype=np.float32)

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        lut_calls.append((detector_shape, len(radial_deg), len(azimuthal_deg)))
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(detector_shape)), dtype=np.float32),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "detector_pixel_angular_maps", _fake_maps)
    monkeypatch.setattr(exact_cake_portable, "flat_solid_angle_normalization", _fake_norm)
    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    integrator.warm_geometry_cache((4, 4), npt_rad=8, npt_azim=6)
    integrator.warm_geometry_cache((4, 4), npt_rad=8, npt_azim=6)

    assert map_calls == [(4, 4)]
    assert norm_calls == [(4, 4)]
    assert lut_calls == [((4, 4), 8, 6)]


def test_detector_pixel_to_caked_bin_bilinearly_blends_lut_projection() -> None:
    matrix = np.eye(4, dtype=np.float32)
    bundle = _make_transform_bundle(
        detector_shape=(2, 2),
        radial_deg=np.array([10.0, 30.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-170.0, 170.0], dtype=np.float64),
        matrix=matrix,
    )

    two_theta, phi = exact_cake_portable.detector_pixel_to_caked_bin(
        bundle,
        0.25,
        0.75,
    )

    pixel_weights = np.array([0.1875, 0.0625, 0.5625, 0.1875], dtype=np.float64)
    expected_two_theta = float(np.dot(np.array([10.0, 30.0, 10.0, 30.0]), pixel_weights))
    raw_angles = np.array([-170.0, -170.0, 170.0, 170.0], dtype=np.float64)
    radians = np.deg2rad(raw_angles)
    expected_raw_phi = float(
        np.degrees(
            np.arctan2(
                np.sum(np.sin(radians) * pixel_weights),
                np.sum(np.cos(radians) * pixel_weights),
            )
        )
    )
    expected_phi = float(exact_cake_portable.raw_phi_to_gui_phi(expected_raw_phi))

    assert two_theta == pytest.approx(expected_two_theta)
    assert phi == pytest.approx(expected_phi)


def test_detector_pixel_to_caked_bin_keeps_integer_pixel_exact() -> None:
    matrix = np.eye(4, dtype=np.float32)
    bundle = _make_transform_bundle(
        detector_shape=(2, 2),
        radial_deg=np.array([10.0, 30.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-170.0, 170.0], dtype=np.float64),
        matrix=matrix,
    )

    two_theta, phi = exact_cake_portable.detector_pixel_to_caked_bin(
        bundle,
        1.0,
        0.0,
    )

    assert two_theta == pytest.approx(30.0)
    assert phi == pytest.approx(float(exact_cake_portable.raw_phi_to_gui_phi(-170.0)))


def test_caked_point_to_detector_pixel_uses_display_axis_lut_centroid(
    monkeypatch,
) -> None:
    captured: dict[str, np.ndarray | tuple[int, int]] = {}
    matrix = np.zeros((6, 12), dtype=np.float32)
    matrix[2, 1] = 1.0
    matrix[2, 5] = 3.0

    def _fake_cached_lut(
        self,
        image_shape,
        radial_deg,
        azimuthal_deg,
        geometry,
        *,
        engine,
        workers,
    ):
        del self, geometry, engine, workers
        captured["shape"] = tuple(int(v) for v in tuple(image_shape)[:2])
        captured["radial_deg"] = np.asarray(radial_deg, dtype=np.float64)
        captured["azimuthal_deg"] = np.asarray(azimuthal_deg, dtype=np.float64)
        return exact_cake.DetectorCakeLUT(
            image_shape=(3, 4),
            n_rad=2,
            n_az=3,
            matrix=matrix,
            count_flat=np.ones(6, dtype=np.float64),
        )

    monkeypatch.setattr(
        exact_cake_portable.FastAzimuthalIntegrator,
        "_cached_cake_lut",
        _fake_cached_lut,
    )

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )

    col, row = exact_cake_portable.caked_point_to_detector_pixel(
        integrator,
        (3, 4),
        np.array([10.0, 20.0], dtype=float),
        np.array([-90.0, 0.0, 90.0], dtype=float),
        10.1,
        1.0,
    )

    assert (col, row) == pytest.approx((1.0, 0.75))
    assert captured["shape"] == (3, 4)
    assert np.allclose(captured["radial_deg"], np.array([10.0, 20.0], dtype=float))
    assert np.allclose(
        captured["azimuthal_deg"],
        np.array([-180.0, -90.0, 0.0], dtype=float),
    )


def test_resolve_cake_transform_bundle_reuses_matching_bundle() -> None:
    bundle = _make_transform_bundle(
        detector_shape=(3, 4),
        radial_deg=np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        matrix=np.eye(6, 12, dtype=np.float32),
    )
    conflicting_live_bundle = _make_transform_bundle(
        detector_shape=(3, 4),
        radial_deg=np.array([1.0, 2.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-30.0, 30.0], dtype=np.float64),
        matrix=np.eye(4, 12, dtype=np.float32),
    )
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )
    integrator._live_caked_transform_bundle = conflicting_live_bundle

    resolved = exact_cake_portable.resolve_cake_transform_bundle(
        integrator,
        (3, 4),
        np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        transform_bundle=bundle,
    )

    assert resolved is bundle


def test_resolve_cake_transform_bundle_rebuilds_mismatched_bundle_from_exact_axes(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_cached_lut(
        self,
        image_shape,
        radial_deg,
        azimuthal_deg,
        geometry,
        *,
        engine,
        workers,
    ):
        del self, geometry, engine, workers
        captured["shape"] = tuple(int(v) for v in tuple(image_shape)[:2])
        captured["radial_deg"] = np.asarray(radial_deg, dtype=np.float64)
        captured["azimuthal_deg"] = np.asarray(azimuthal_deg, dtype=np.float64)
        return exact_cake.DetectorCakeLUT(
            image_shape=(3, 4),
            n_rad=2,
            n_az=3,
            matrix=np.eye(6, 12, dtype=np.float32),
            count_flat=np.ones(6, dtype=np.float64),
        )

    monkeypatch.setattr(
        exact_cake_portable.FastAzimuthalIntegrator,
        "_cached_cake_lut",
        _fake_cached_lut,
    )
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )
    stale_bundle = _make_transform_bundle(
        detector_shape=(3, 4),
        radial_deg=np.array([1.0, 2.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-30.0, 30.0], dtype=np.float64),
        matrix=np.eye(4, 12, dtype=np.float32),
    )

    rebuilt = exact_cake_portable.resolve_cake_transform_bundle(
        integrator,
        (3, 4),
        np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        transform_bundle=stale_bundle,
    )

    assert isinstance(rebuilt, exact_cake_portable.CakeTransformBundle)
    assert rebuilt is not stale_bundle
    assert captured["shape"] == (3, 4)
    assert np.allclose(captured["radial_deg"], np.array([10.0, 20.0], dtype=np.float64))
    assert np.allclose(
        captured["azimuthal_deg"],
        np.array([-180.0, -90.0, 0.0], dtype=np.float64),
    )


def test_resolve_cake_transform_bundle_returns_none_when_exact_rebuild_is_unavailable() -> None:
    stale_bundle = _make_transform_bundle(
        detector_shape=(3, 4),
        radial_deg=np.array([1.0, 2.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-30.0, 30.0], dtype=np.float64),
        matrix=np.eye(4, 12, dtype=np.float32),
    )

    resolved = exact_cake_portable.resolve_cake_transform_bundle(
        None,
        (3, 4),
        np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        transform_bundle=stale_bundle,
    )

    assert resolved is None


def test_resolve_cake_transform_bundle_requires_exact_gui_display_axis_when_requested() -> None:
    bundle = _make_transform_bundle(
        detector_shape=(3, 4),
        radial_deg=np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        matrix=np.eye(6, 12, dtype=np.float32),
    )

    resolved = exact_cake_portable.resolve_cake_transform_bundle(
        None,
        (3, 4),
        np.array([10.0, 20.0], dtype=np.float64),
        gui_azimuth_deg=np.array([90.0, 0.0, -90.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, -90.0, 0.0], dtype=np.float64),
        transform_bundle=bundle,
        require_gui_display_match=True,
    )

    assert resolved is None


def test_caked_point_to_detector_pixel_bilinearly_blends_neighboring_cake_bins() -> None:
    bundle = _make_transform_bundle(
        detector_shape=(2, 2),
        radial_deg=np.array([10.0, 20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-180.0, 0.0], dtype=np.float64),
        matrix=np.eye(4, dtype=np.float32),
    )
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )

    col, row = exact_cake_portable.caked_point_to_detector_pixel(
        integrator,
        (2, 2),
        None,
        None,
        15.0,
        0.0,
        transform_bundle=bundle,
    )

    assert (col, row) == pytest.approx((0.5, 0.5))


def test_caked_point_to_detector_pixel_wraps_across_azimuth_seam() -> None:
    bundle = _make_transform_bundle(
        detector_shape=(1, 2),
        radial_deg=np.array([10.0], dtype=np.float64),
        raw_azimuth_deg=np.array([-170.0, 170.0], dtype=np.float64),
        matrix=np.eye(2, dtype=np.float32),
    )
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )

    col, row = exact_cake_portable.caked_point_to_detector_pixel(
        integrator,
        (1, 2),
        None,
        None,
        10.0,
        90.0,
        transform_bundle=bundle,
    )

    assert (col, row) == pytest.approx((0.5, 0.0))


def test_caked_point_to_detector_pixel_caches_fallback_lut_build(monkeypatch) -> None:
    build_calls: list[tuple[tuple[int, int], tuple[float, ...], tuple[float, ...]]] = []

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        build_calls.append(
            (
                detector_shape,
                tuple(float(v) for v in np.asarray(radial_deg, dtype=np.float64)),
                tuple(float(v) for v in np.asarray(azimuthal_deg, dtype=np.float64)),
            )
        )
        matrix = np.zeros((int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(detector_shape))), dtype=np.float32)
        matrix[0, 0] = 1.0
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=matrix,
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=1.0,
        poni1=1.0,
        poni2=1.0,
        pixel1=1.0,
        pixel2=1.0,
    )

    first = exact_cake_portable.caked_point_to_detector_pixel(
        integrator,
        (3, 4),
        np.array([10.0, 20.0], dtype=float),
        np.array([90.0, 0.0, -90.0], dtype=float),
        10.0,
        90.0,
        engine="python",
    )
    second = exact_cake_portable.caked_point_to_detector_pixel(
        integrator,
        (3, 4),
        np.array([10.0, 20.0], dtype=float),
        np.array([90.0, 0.0, -90.0], dtype=float),
        10.0,
        90.0,
        engine="python",
    )

    assert first == pytest.approx((0.0, 0.0))
    assert second == pytest.approx((0.0, 0.0))
    assert len(build_calls) == 1
    assert len(exact_cake_portable._PROCESS_CAKE_LUT_CACHE) == 1


def test_start_exact_cake_geometry_warmup_in_background_dedupes_requests(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_warm(self, detector_shape, *, npt_rad=1000, npt_azim=720, engine="auto", workers="auto"):
        del self, engine, workers
        calls.append((tuple(int(v) for v in tuple(detector_shape)[:2]), int(npt_rad), int(npt_azim)))
        entered.set()
        release.wait(timeout=1.0)

    monkeypatch.setattr(exact_cake_portable.FastAzimuthalIntegrator, "warm_geometry_cache", _fake_warm)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    assert exact_cake_portable.start_exact_cake_geometry_warmup_in_background(
        integrator,
        (4, 4),
        npt_rad=8,
        npt_azim=6,
    )
    assert entered.wait(timeout=1.0)
    assert not exact_cake_portable.start_exact_cake_geometry_warmup_in_background(
        integrator,
        (4, 4),
        npt_rad=8,
        npt_azim=6,
    )

    release.set()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if not exact_cake_portable._EXACT_CAKE_GEOMETRY_WARMUP_THREADS:
            break
        time.sleep(0.01)

    assert calls == [((4, 4), 8, 6)]
    assert not exact_cake_portable.start_exact_cake_geometry_warmup_in_background(
        integrator,
        (4, 4),
        npt_rad=8,
        npt_azim=6,
    )


def test_shared_cake_lut_single_flight_dedupes_warmup_and_analysis(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    analysis_started = threading.Event()
    analysis_done = threading.Event()
    analysis_results: list[exact_cake.DetectorCakeResult] = []
    analysis_errors: list[Exception] = []
    build_calls = 0

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        nonlocal build_calls
        build_calls += 1
        entered.set()
        if not release.wait(timeout=1.0):
            raise AssertionError("timed out waiting to release fake LUT build")
        return _make_identity_lut(image_shape, radial_deg, azimuthal_deg)

    def _fake_integrate_lut(image, radial_deg, azimuthal_deg, lut, *, normalization=None, mask=None):
        del image, normalization, mask
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray(radial_deg, dtype=np.float64),
            azimuthal_deg=np.asarray(azimuthal_deg, dtype=np.float64),
            intensity=np.zeros((lut.n_az, lut.n_rad), dtype=np.float32),
            sum_signal=np.zeros((lut.n_az, lut.n_rad), dtype=np.float64),
            sum_normalization=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
            count=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)

    integrator_a = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    integrator_b = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    assert exact_cake_portable.start_exact_cake_geometry_warmup_in_background(
        integrator_a,
        image.shape,
        npt_rad=8,
        npt_azim=6,
    )
    assert entered.wait(timeout=1.0)

    def _run_analysis() -> None:
        try:
            analysis_started.set()
            analysis_results.append(
                integrator_b.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")
            )
        except Exception as exc:
            analysis_errors.append(exc)
        finally:
            analysis_done.set()

    analysis_thread = threading.Thread(target=_run_analysis, name="test-exact-cake-analysis")
    analysis_thread.start()
    assert analysis_started.wait(timeout=1.0)
    assert not analysis_done.wait(timeout=0.05)

    release.set()
    analysis_thread.join(timeout=1.0)
    assert not analysis_thread.is_alive()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if not exact_cake_portable._EXACT_CAKE_GEOMETRY_WARMUP_THREADS:
            break
        time.sleep(0.01)

    assert build_calls == 1
    assert not exact_cake_portable._EXACT_CAKE_GEOMETRY_WARMUP_THREADS
    assert analysis_done.is_set()
    assert not analysis_errors
    assert len(analysis_results) == 1
    assert len(exact_cake_portable._PROCESS_CAKE_LUT_CACHE) == 1
    assert not exact_cake_portable._PROCESS_CAKE_LUT_IN_FLIGHT


def test_shared_cake_lut_single_flight_releases_waiters_on_build_error(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    analysis_started = threading.Event()
    analysis_done = threading.Event()
    analysis_errors: list[Exception] = []
    build_calls = 0

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del image_shape, radial_deg, azimuthal_deg, geometry, workers
        nonlocal build_calls
        build_calls += 1
        entered.set()
        if not release.wait(timeout=1.0):
            raise AssertionError("timed out waiting to release fake LUT build")
        raise RuntimeError("no lut in test")

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_lut",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LUT integration should not run on build failure")),
    )

    integrator_a = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    integrator_b = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    assert exact_cake_portable.start_exact_cake_geometry_warmup_in_background(
        integrator_a,
        image.shape,
        npt_rad=8,
        npt_azim=6,
    )
    assert entered.wait(timeout=1.0)

    def _run_analysis() -> None:
        try:
            analysis_started.set()
            integrator_b.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")
        except Exception as exc:
            analysis_errors.append(exc)
        finally:
            analysis_done.set()

    analysis_thread = threading.Thread(target=_run_analysis, name="test-exact-cake-analysis-error")
    analysis_thread.start()
    assert analysis_started.wait(timeout=1.0)
    assert not analysis_done.wait(timeout=0.05)

    release.set()
    analysis_thread.join(timeout=1.0)
    assert not analysis_thread.is_alive()

    deadline = time.time() + 1.0
    while time.time() < deadline:
        if not exact_cake_portable._EXACT_CAKE_GEOMETRY_WARMUP_THREADS:
            break
        time.sleep(0.01)

    assert build_calls == 1
    assert not exact_cake_portable._EXACT_CAKE_GEOMETRY_WARMUP_THREADS
    assert analysis_done.is_set()
    assert len(analysis_errors) == 1
    assert isinstance(analysis_errors[0], RuntimeError)
    assert str(analysis_errors[0]) == "no lut in test"
    assert not exact_cake_portable._PROCESS_CAKE_LUT_CACHE
    assert not exact_cake_portable._PROCESS_CAKE_LUT_IN_FLIGHT


def test_exact_cake_auto_workers_default_to_8(monkeypatch) -> None:
    monkeypatch.setattr(exact_cake, "system_cpu_worker_count", lambda: 32)

    assert exact_cake._resolve_workers("auto", 64, "numba") == 8


def test_exact_cake_workers_clamp_to_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr(exact_cake, "system_cpu_worker_count", lambda: 6)

    assert exact_cake._resolve_workers("auto", 64, "numba") == 6
    assert exact_cake._resolve_workers(24, 64, "numba") == 6


def test_exact_cake_numba_warmup_runs_once(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], str, int | str | None]] = []

    def _fake_integrate(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        *,
        normalization=None,
        mask=None,
        rows=None,
        cols=None,
        engine="auto",
        workers="auto",
    ):
        del radial_deg, azimuthal_deg, geometry, normalization, mask, rows, cols
        calls.append((tuple(np.asarray(image).shape), str(engine), workers))
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray([0.25, 0.75], dtype=np.float64),
            azimuthal_deg=np.asarray([-90.0, 90.0], dtype=np.float64),
            intensity=np.zeros((2, 2), dtype=np.float32),
            sum_signal=np.zeros((2, 2), dtype=np.float64),
            sum_normalization=np.ones((2, 2), dtype=np.float64),
            count=np.ones((2, 2), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake, "_HAS_NUMBA", True)
    monkeypatch.setattr(exact_cake, "_EXACT_CAKE_NUMBA_WARMED", False)
    monkeypatch.setattr(exact_cake, "_EXACT_CAKE_NUMBA_WARMUP_THREAD", None)
    monkeypatch.setattr(exact_cake, "integrate_detector_to_cake_exact", _fake_integrate)
    monkeypatch.setattr(exact_cake, "build_detector_to_cake_lut", lambda *args, **kwargs: None)

    assert exact_cake.warmup_exact_cake_numba() is True
    assert exact_cake.warmup_exact_cake_numba() is False
    assert calls == [((2, 2), "numba", 1)]


def test_fast_azimuthal_integrator_integrate2d_builds_lut_cake_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        captured["build_image_shape"] = tuple(int(v) for v in tuple(image_shape)[:2])
        captured["build_radial_deg"] = np.asarray(radial_deg, dtype=np.float64)
        captured["build_azimuthal_deg"] = np.asarray(azimuthal_deg, dtype=np.float64)
        captured["geometry"] = geometry
        captured["build_workers"] = workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(
                int(len(radial_deg) * len(azimuthal_deg)),
                int(np.prod(detector_shape)),
                dtype=np.float32,
            ),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        *,
        normalization=None,
        mask=None,
    ):
        captured["integrate_image_shape"] = tuple(np.asarray(image).shape)
        captured["integrate_radial_deg"] = np.asarray(radial_deg, dtype=np.float64)
        captured["integrate_azimuthal_deg"] = np.asarray(azimuthal_deg, dtype=np.float64)
        captured["integrate_normalization"] = (
            None if normalization is None else np.asarray(normalization)
        )
        captured["integrate_mask"] = mask
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray(radial_deg, dtype=np.float64),
            azimuthal_deg=np.asarray(azimuthal_deg, dtype=np.float64),
            intensity=np.zeros((lut.n_az, lut.n_rad), dtype=np.float32),
            sum_signal=np.zeros((lut.n_az, lut.n_rad), dtype=np.float64),
            sum_normalization=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
            count=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
        )

    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        _fake_build_lut,
    )
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_exact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact fallback should stay off live LUT path")
        ),
    )

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    result = integrator.integrate2d(
        np.arange(16, dtype=np.float64).reshape(4, 4),
        npt_rad=8,
        npt_azim=6,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg",
    )

    assert result.intensity.shape == (6, 8)
    assert captured["build_image_shape"] == (4, 4)
    assert captured["integrate_image_shape"] == (4, 4)
    assert captured["build_workers"] == "auto"
    assert captured["integrate_mask"] is None
    assert captured["integrate_normalization"] is not None
    assert captured["geometry"].center_row_px == 100.0
    assert captured["geometry"].center_col_px == 200.0


def test_fast_azimuthal_integrator_integrate2d_fails_closed_when_lut_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no lut in test")),
    )
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_exact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact fallback should stay off live LUT path")
        ),
    )

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    with pytest.raises(RuntimeError, match="no lut in test"):
        integrator.integrate2d(
            np.arange(16, dtype=np.float64).reshape(4, 4),
            npt_rad=8,
            npt_azim=6,
            method="lut",
            unit="2th_deg",
        )


def test_fast_azimuthal_integrator_integrate2d_maps_pixel_selection_into_lut_mask(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(
                int(len(radial_deg) * len(azimuthal_deg)),
                int(np.prod(detector_shape)),
                dtype=np.float32,
            ),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        *,
        normalization=None,
        mask=None,
    ):
        del image, radial_deg, azimuthal_deg, lut, normalization
        captured["mask"] = None if mask is None else np.asarray(mask, dtype=bool)
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray([1.0], dtype=np.float64),
            azimuthal_deg=np.asarray([0.0], dtype=np.float64),
            intensity=np.zeros((1, 1), dtype=np.float32),
            sum_signal=np.zeros((1, 1), dtype=np.float64),
            sum_normalization=np.ones((1, 1), dtype=np.float64),
            count=np.ones((1, 1), dtype=np.float64),
        )

    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        _fake_build_lut,
    )
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_exact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact fallback should stay off live LUT path")
        ),
    )

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    rows = np.asarray([0, 1, 2], dtype=np.int64)
    cols = np.asarray([3, 2, 1], dtype=np.int64)
    input_mask = np.zeros((4, 4), dtype=np.int8)
    input_mask[1, 2] = 1
    input_mask[3, 0] = 1
    integrator.integrate2d(
        np.arange(16, dtype=np.float64).reshape(4, 4),
        npt_rad=8,
        npt_azim=6,
        method="lut",
        unit="2th_deg",
        mask=input_mask,
        rows=rows,
        cols=cols,
    )

    expected_mask = np.ones((4, 4), dtype=bool)
    expected_mask[rows, cols] = False
    expected_mask |= input_mask.astype(bool)

    assert np.array_equal(captured["mask"], expected_mask)


def test_convert_image_to_angle_space_maps_pixel_selection_into_lut_mask(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(
                int(len(radial_deg) * len(azimuthal_deg)),
                int(np.prod(detector_shape)),
                dtype=np.float32,
            ),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        *,
        normalization=None,
        mask=None,
    ):
        del image, radial_deg, azimuthal_deg, lut, normalization
        captured["mask"] = None if mask is None else np.asarray(mask, dtype=bool)
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray([1.0], dtype=np.float64),
            azimuthal_deg=np.asarray([0.0], dtype=np.float64),
            intensity=np.zeros((1, 1), dtype=np.float32),
            sum_signal=np.zeros((1, 1), dtype=np.float64),
            sum_normalization=np.ones((1, 1), dtype=np.float64),
            count=np.ones((1, 1), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_exact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact fallback should stay off live LUT path")
        ),
    )

    rows = np.asarray([0, 2], dtype=np.int64)
    cols = np.asarray([1, 3], dtype=np.int64)
    exact_cake_portable.convert_image_to_angle_space(
        np.arange(16, dtype=np.float64).reshape(4, 4),
        pixel_size_m=1.0e-4,
        distance_m=0.3,
        center_row_px=100.0,
        center_col_px=200.0,
        rows=rows,
        cols=cols,
    )

    expected_mask = np.ones((4, 4), dtype=bool)
    expected_mask[rows, cols] = False

    assert np.array_equal(captured["mask"], expected_mask)


def test_convert_image_to_angle_space_fails_closed_when_lut_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no lut in test")),
    )
    monkeypatch.setattr(
        exact_cake_portable,
        "integrate_detector_to_cake_exact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Exact fallback should stay off live LUT path")
        ),
    )

    with pytest.raises(RuntimeError, match="no lut in test"):
        exact_cake_portable.convert_image_to_angle_space(
            np.arange(16, dtype=np.float64).reshape(4, 4),
            pixel_size_m=1.0e-4,
            distance_m=0.3,
            center_row_px=100.0,
            center_col_px=200.0,
        )


def test_fast_azimuthal_integrator_reuses_solid_angle_normalization(monkeypatch) -> None:
    norm_calls: list[tuple[int, int]] = []
    passed_normalizations: list[np.ndarray] = []

    def _fake_norm(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        norm_calls.append(detector_shape)
        return np.full(detector_shape, 7.0, dtype=np.float32)

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(
                int(len(radial_deg) * len(azimuthal_deg)),
                int(np.prod(detector_shape)),
                dtype=np.float32,
            ),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        *,
        normalization=None,
        mask=None,
    ):
        del image, radial_deg, azimuthal_deg, lut, mask
        passed_normalizations.append(normalization)
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray([1.0], dtype=np.float64),
            azimuthal_deg=np.asarray([0.0], dtype=np.float64),
            intensity=np.zeros((1, 1), dtype=np.float32),
            sum_signal=np.zeros((1, 1), dtype=np.float64),
            sum_normalization=np.ones((1, 1), dtype=np.float64),
            count=np.ones((1, 1), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "flat_solid_angle_normalization", _fake_norm)
    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    integrator.integrate2d(image, npt_rad=8, npt_azim=6, correctSolidAngle=True, method="lut", unit="2th_deg")
    integrator.integrate2d(image, npt_rad=8, npt_azim=6, correctSolidAngle=True, method="lut", unit="2th_deg")

    assert norm_calls == [(4, 4)]
    assert len(passed_normalizations) == 2
    assert passed_normalizations[0] is passed_normalizations[1]
    assert not passed_normalizations[0].flags.writeable


def test_exact_cake_lut_matches_direct_integration() -> None:
    image = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    geometry = exact_cake.DetectorCakeGeometry(
        pixel_size_m=1.0e-4,
        distance_m=0.075,
        center_row_px=2.0,
        center_col_px=2.0,
    )
    radial_deg = np.asarray([5.0, 15.0, 25.0], dtype=np.float64)
    azimuthal_deg = np.asarray([-90.0, 0.0, 90.0], dtype=np.float64)

    direct = exact_cake.integrate_detector_to_cake_exact(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        engine="numba",
        workers=1,
    )
    lut = exact_cake.build_detector_to_cake_lut(
        image.shape,
        radial_deg,
        azimuthal_deg,
        geometry,
        workers=1,
    )
    cached = exact_cake.integrate_detector_to_cake_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
    )

    assert np.allclose(cached.intensity, direct.intensity, atol=1.0e-5)
    assert np.allclose(cached.sum_signal, direct.sum_signal, atol=1.0e-5)
    assert np.allclose(cached.sum_normalization, direct.sum_normalization, atol=1.0e-5)
    assert np.allclose(cached.count, direct.count, atol=1.0e-5)


def test_default_exact_cake_lut_and_inverse_lut_agree_on_detector_pixels() -> None:
    image = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.075,
        poni1=2.5e-4,
        poni2=2.5e-4,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    result = integrator.integrate2d(
        image,
        npt_rad=64,
        npt_azim=180,
        unit="2th_deg",
        workers=1,
    )
    radial_axis = np.asarray(result.radial, dtype=np.float64)
    raw_azimuth_axis = np.asarray(result.azimuthal, dtype=np.float64)
    gui_phi_axis = (
        (
            exact_cake_portable.PHI_ZERO_OFFSET_DEGREES
            - raw_azimuth_axis
            + 180.0
        )
        % 360.0
    ) - 180.0
    lut = integrator._cached_cake_lut(
        image.shape,
        radial_axis,
        raw_azimuth_axis,
        exact_cake.DetectorCakeGeometry(
            pixel_size_m=float(integrator.geometry.pixel_size_m),
            distance_m=float(integrator.geometry.distance_m),
            center_row_px=float(integrator.geometry.center_row_px),
            center_col_px=float(integrator.geometry.center_col_px),
        ),
        engine="auto",
        workers=1,
    )

    assert lut is not None

    validated_bins = 0
    for flat_idx in range(int(lut.n_rad) * int(lut.n_az)):
        if hasattr(lut.matrix, "indptr") and hasattr(lut.matrix, "indices"):
            row_start = int(lut.matrix.indptr[flat_idx])
            row_stop = int(lut.matrix.indptr[flat_idx + 1])
            pixel_indices = np.asarray(lut.matrix.indices[row_start:row_stop], dtype=np.int64)
            weights = np.asarray(lut.matrix.data[row_start:row_stop], dtype=np.float64)
        else:
            row_weights = np.asarray(lut.matrix[flat_idx], dtype=np.float64).reshape(-1)
            pixel_indices = np.flatnonzero(np.isfinite(row_weights) & (row_weights > 0.0))
            weights = row_weights[pixel_indices]

        valid = np.isfinite(weights) & (weights > 0.0)
        pixel_indices = pixel_indices[valid]
        if pixel_indices.size != 1:
            continue

        pixel_index = int(pixel_indices[0])
        expected_col = float(pixel_index % image.shape[1])
        expected_row = float(pixel_index // image.shape[1])
        azimuth_idx = int(flat_idx // int(lut.n_rad))
        radial_idx = int(flat_idx % int(lut.n_rad))

        recovered_col, recovered_row = exact_cake_portable.caked_point_to_detector_pixel(
            integrator,
            image.shape,
            radial_axis,
            gui_phi_axis,
            float(radial_axis[radial_idx]),
            float(gui_phi_axis[azimuth_idx]),
            workers=1,
        )

        assert recovered_col == pytest.approx(expected_col, abs=1.0e-6)
        assert recovered_row == pytest.approx(expected_row, abs=1.0e-6)
        validated_bins += 1
        if validated_bins >= 3:
            break

    assert validated_bins >= 1


def test_exact_cake_pixel_selection_matches_equivalent_mask() -> None:
    image = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    geometry = exact_cake.DetectorCakeGeometry(
        pixel_size_m=1.0e-4,
        distance_m=0.075,
        center_row_px=2.0,
        center_col_px=2.0,
    )
    radial_deg = np.asarray([5.0, 15.0, 25.0], dtype=np.float64)
    azimuthal_deg = np.asarray([-90.0, 0.0, 90.0], dtype=np.float64)
    rows = np.asarray([0, 1, 2], dtype=np.int64)
    cols = np.asarray([1, 2, 3], dtype=np.int64)

    mask = np.ones_like(image, dtype=np.int8)
    mask[rows, cols] = 0

    selected = exact_cake.integrate_detector_to_cake_exact(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        rows=rows,
        cols=cols,
        engine="python",
        workers=1,
    )
    masked = exact_cake.integrate_detector_to_cake_exact(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        mask=mask,
        engine="python",
        workers=1,
    )

    assert np.allclose(selected.intensity, masked.intensity, atol=1.0e-5)
    assert np.allclose(selected.sum_signal, masked.sum_signal, atol=1.0e-5)
    assert np.allclose(selected.sum_normalization, masked.sum_normalization, atol=1.0e-5)
    assert np.allclose(selected.count, masked.count, atol=1.0e-5)


def test_fast_azimuthal_integrator_reuses_cake_lut(monkeypatch) -> None:
    lut_calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        lut_calls.append((tuple(int(v) for v in image_shape[:2]), len(radial_deg), len(azimuthal_deg)))
        return exact_cake.DetectorCakeLUT(
            image_shape=tuple(int(v) for v in image_shape[:2]),
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(image_shape[:2])), dtype=np.float32),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(image, radial_deg, azimuthal_deg, lut, *, normalization=None, mask=None):
        del image, normalization, mask
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray(radial_deg, dtype=np.float64),
            azimuthal_deg=np.asarray(azimuthal_deg, dtype=np.float64),
            intensity=np.zeros((lut.n_az, lut.n_rad), dtype=np.float32),
            sum_signal=np.zeros((lut.n_az, lut.n_rad), dtype=np.float64),
            sum_normalization=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
            count=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    integrator.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")
    integrator.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")

    assert lut_calls == [((4, 4), 8, 6)]


def test_fast_azimuthal_integrator_cake_lut_cache_key_uses_full_axis_contents(monkeypatch) -> None:
    lut_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        build_idx = len(lut_calls) + 1
        lut_calls.append(
            (
                np.asarray(radial_deg, dtype=np.float64).copy(),
                np.asarray(azimuthal_deg, dtype=np.float64).copy(),
            )
        )
        return exact_cake.DetectorCakeLUT(
            image_shape=tuple(int(v) for v in image_shape[:2]),
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.full(
                (int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(image_shape[:2]))),
                float(build_idx),
                dtype=np.float32,
            ),
            count_flat=np.full(int(len(radial_deg) * len(azimuthal_deg)), float(build_idx), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    geometry = exact_cake.DetectorCakeGeometry(
        pixel_size_m=float(integrator.geometry.pixel_size_m),
        distance_m=float(integrator.geometry.distance_m),
        center_row_px=float(integrator.geometry.center_row_px),
        center_col_px=float(integrator.geometry.center_col_px),
    )
    detector_shape = (4, 4)
    radial_a = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    radial_b = np.array([0.0, 2.0, 3.0], dtype=np.float64)
    azimuthal_a = np.array([-180.0, -20.0, 180.0], dtype=np.float64)
    azimuthal_b = np.array([-180.0, 40.0, 180.0], dtype=np.float64)

    lut_a = integrator._cached_cake_lut(
        detector_shape,
        radial_a,
        azimuthal_a,
        geometry,
        engine="auto",
        workers=1,
    )
    lut_b = integrator._cached_cake_lut(
        detector_shape,
        radial_b,
        azimuthal_b,
        geometry,
        engine="auto",
        workers=1,
    )
    lut_a_again = integrator._cached_cake_lut(
        detector_shape,
        radial_a.astype(np.float32, copy=True),
        azimuthal_a.astype(np.float32, copy=True),
        geometry,
        engine="auto",
        workers=1,
    )

    assert lut_a is not None
    assert lut_b is not None
    assert lut_a_again is not None
    assert lut_a is not lut_b
    assert lut_a_again is lut_a
    assert len(lut_calls) == 2
    assert np.array_equal(lut_calls[0][0], radial_a)
    assert np.array_equal(lut_calls[0][1], azimuthal_a)
    assert np.array_equal(lut_calls[1][0], radial_b)
    assert np.array_equal(lut_calls[1][1], azimuthal_b)


def test_fast_azimuthal_integrator_cake_lut_cache_is_shared_across_instances(monkeypatch) -> None:
    lut_calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        lut_calls.append((tuple(int(v) for v in image_shape[:2]), len(radial_deg), len(azimuthal_deg)))
        return exact_cake.DetectorCakeLUT(
            image_shape=tuple(int(v) for v in image_shape[:2]),
            n_rad=int(len(radial_deg)),
            n_az=int(len(azimuthal_deg)),
            matrix=np.eye(int(len(radial_deg) * len(azimuthal_deg)), int(np.prod(image_shape[:2])), dtype=np.float32),
            count_flat=np.ones(int(len(radial_deg) * len(azimuthal_deg)), dtype=np.float64),
        )

    def _fake_integrate_lut(image, radial_deg, azimuthal_deg, lut, *, normalization=None, mask=None):
        del image, normalization, mask
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray(radial_deg, dtype=np.float64),
            azimuthal_deg=np.asarray(azimuthal_deg, dtype=np.float64),
            intensity=np.zeros((lut.n_az, lut.n_rad), dtype=np.float32),
            sum_signal=np.zeros((lut.n_az, lut.n_rad), dtype=np.float64),
            sum_normalization=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
            count=np.ones((lut.n_az, lut.n_rad), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "build_detector_to_cake_lut", _fake_build_lut)
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_lut", _fake_integrate_lut)

    integrator_a = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    integrator_b = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)

    integrator_a.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")
    integrator_b.integrate2d(image, npt_rad=8, npt_azim=6, method="lut", unit="2th_deg")

    assert lut_calls == [((4, 4), 8, 6)]
