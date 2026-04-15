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
        9.9,
        1.0,
    )

    assert (col, row) == pytest.approx((1.0, 0.75))
    assert captured["shape"] == (3, 4)
    assert np.allclose(captured["radial_deg"], np.array([10.0, 20.0], dtype=float))
    assert np.allclose(
        captured["azimuthal_deg"],
        np.array([-180.0, -90.0, 0.0], dtype=float),
    )


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


def test_fast_azimuthal_integrator_integrate2d_builds_exact_cake_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_integrate(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        *,
        normalization=None,
        mask=None,
        engine="auto",
        workers="auto",
        **_kwargs,
    ):
        captured["image_shape"] = tuple(np.asarray(image).shape)
        captured["radial_deg"] = np.asarray(radial_deg, dtype=np.float64)
        captured["azimuthal_deg"] = np.asarray(azimuthal_deg, dtype=np.float64)
        captured["geometry"] = geometry
        captured["normalization"] = None if normalization is None else np.asarray(normalization)
        captured["mask"] = mask
        captured["engine"] = engine
        captured["workers"] = workers
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray(radial_deg, dtype=np.float64),
            azimuthal_deg=np.asarray(azimuthal_deg, dtype=np.float64),
            intensity=np.zeros((len(azimuthal_deg), len(radial_deg)), dtype=np.float32),
            sum_signal=np.zeros((len(azimuthal_deg), len(radial_deg)), dtype=np.float64),
            sum_normalization=np.ones((len(azimuthal_deg), len(radial_deg)), dtype=np.float64),
            count=np.ones((len(azimuthal_deg), len(radial_deg)), dtype=np.float64),
        )

    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no lut in test")),
    )
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fake_integrate)

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
    assert captured["image_shape"] == (4, 4)
    assert captured["workers"] == "auto"
    assert captured["engine"] == "auto"
    assert captured["mask"] is None
    assert captured["normalization"] is not None
    assert captured["geometry"].center_row_px == 100.0
    assert captured["geometry"].center_col_px == 200.0


def test_fast_azimuthal_integrator_integrate2d_forwards_pixel_selection(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

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
        **_kwargs,
    ):
        del image, radial_deg, azimuthal_deg, geometry, normalization, mask, engine, workers
        captured["rows"] = None if rows is None else np.asarray(rows, dtype=np.int64)
        captured["cols"] = None if cols is None else np.asarray(cols, dtype=np.int64)
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
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LUT path should be skipped")),
    )
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fake_integrate)

    integrator = exact_cake_portable.FastAzimuthalIntegrator(
        dist=0.3,
        poni1=0.01,
        poni2=0.02,
        pixel1=1.0e-4,
        pixel2=1.0e-4,
    )

    rows = np.asarray([0, 1, 2], dtype=np.int64)
    cols = np.asarray([3, 2, 1], dtype=np.int64)
    integrator.integrate2d(
        np.arange(16, dtype=np.float64).reshape(4, 4),
        npt_rad=8,
        npt_azim=6,
        method="lut",
        unit="2th_deg",
        rows=rows,
        cols=cols,
    )

    assert np.array_equal(captured["rows"], rows)
    assert np.array_equal(captured["cols"], cols)


def test_convert_image_to_angle_space_forwards_pixel_selection(monkeypatch) -> None:
    captured: dict[str, object] = {}

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
        del image, radial_deg, azimuthal_deg, geometry, normalization, mask, engine, workers
        captured["rows"] = None if rows is None else np.asarray(rows, dtype=np.int64)
        captured["cols"] = None if cols is None else np.asarray(cols, dtype=np.int64)
        return exact_cake.DetectorCakeResult(
            radial_deg=np.asarray([1.0], dtype=np.float64),
            azimuthal_deg=np.asarray([0.0], dtype=np.float64),
            intensity=np.zeros((1, 1), dtype=np.float32),
            sum_signal=np.zeros((1, 1), dtype=np.float64),
            sum_normalization=np.ones((1, 1), dtype=np.float64),
            count=np.ones((1, 1), dtype=np.float64),
        )

    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fake_integrate)

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

    assert np.array_equal(captured["rows"], rows)
    assert np.array_equal(captured["cols"], cols)


def test_fast_azimuthal_integrator_reuses_solid_angle_normalization(monkeypatch) -> None:
    norm_calls: list[tuple[int, int]] = []
    passed_normalizations: list[np.ndarray] = []

    def _fake_norm(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        norm_calls.append(detector_shape)
        return np.full(detector_shape, 7.0, dtype=np.float32)

    def _fake_integrate(
        image,
        radial_deg,
        azimuthal_deg,
        geometry,
        *,
        normalization=None,
        mask=None,
        engine="auto",
        workers="auto",
        **_kwargs,
    ):
        del image, radial_deg, azimuthal_deg, geometry, mask, engine, workers
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
    monkeypatch.setattr(
        exact_cake_portable,
        "build_detector_to_cake_lut",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no lut in test")),
    )
    monkeypatch.setattr(exact_cake_portable, "integrate_detector_to_cake_exact", _fake_integrate)

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
