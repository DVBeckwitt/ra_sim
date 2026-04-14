import numpy as np
import pytest

from ra_sim.simulation import exact_cake
from ra_sim.simulation import exact_cake_portable


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


def test_fast_azimuthal_integrator_detector_maps_reuse_shared_cache_across_instances(
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

    exact_cake_portable._SHARED_DETECTOR_MAP_CACHE.clear()
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
    assert not two_theta_a.flags.writeable

    exact_cake_portable._SHARED_DETECTOR_MAP_CACHE.clear()


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


def test_fast_azimuthal_integrator_warm_geometry_cache_primes_live_caches(monkeypatch) -> None:
    map_calls: list[tuple[int, int]] = []
    norm_calls: list[tuple[int, int]] = []
    lut_calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_maps(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        map_calls.append(detector_shape)
        return (
            np.full(detector_shape, 1.5, dtype=np.float64),
            np.full(detector_shape, -2.5, dtype=np.float64),
        )

    def _fake_norm(image_shape, geometry):
        del geometry
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        norm_calls.append(detector_shape)
        return np.full(detector_shape, 7.0, dtype=np.float32)

    def _fake_build_lut(image_shape, radial_deg, azimuthal_deg, geometry, *, workers="auto"):
        del geometry, workers
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        lut_calls.append((detector_shape, len(radial_deg), len(azimuthal_deg)))
        return exact_cake.DetectorCakeLUT(
            image_shape=detector_shape,
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

    exact_cake_portable._SHARED_DETECTOR_MAP_CACHE.clear()
    exact_cake_portable._SHARED_CAKE_LUT_CACHE.clear()
    monkeypatch.setattr(exact_cake_portable, "detector_pixel_angular_maps", _fake_maps)
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

    integrator.warm_geometry_cache(shape=image.shape, npt_rad=8, npt_azim=6)
    integrator.twoThetaArray(shape=image.shape, unit="2th_deg")
    integrator.integrate2d(
        image,
        npt_rad=8,
        npt_azim=6,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg",
    )

    assert map_calls == [(4, 4)]
    assert norm_calls == [(4, 4)]
    assert lut_calls == [((4, 4), 8, 6)]

    exact_cake_portable._SHARED_DETECTOR_MAP_CACHE.clear()
    exact_cake_portable._SHARED_CAKE_LUT_CACHE.clear()


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


def test_fast_azimuthal_integrator_reuses_shared_cake_lut_across_instances(monkeypatch) -> None:
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

    exact_cake_portable._SHARED_CAKE_LUT_CACHE.clear()
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

    exact_cake_portable._SHARED_CAKE_LUT_CACHE.clear()
