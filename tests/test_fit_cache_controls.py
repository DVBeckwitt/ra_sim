from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.optimize import OptimizeResult

from ra_sim.config import loader
from ra_sim.fitting import optimization
from ra_sim.fitting.optimization import SimulationCache
from ra_sim.utils.calculations import _legacy_kernel_n2_sample_array_from_angstrom


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


def test_simulation_kernel_kwargs_builds_legacy_managed_n2_cache_when_missing() -> None:
    wavelengths = np.array([1.0, np.nan], dtype=np.float64)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
    }

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 0.97 + 0.02j}, mosaic)

    expected = _legacy_kernel_n2_sample_array_from_angstrom(
        wavelengths,
        nominal_n2=0.97 + 0.02j,
        sample_count=2,
    )
    np.testing.assert_allclose(kwargs["n2_sample_array_override"], expected)
    np.testing.assert_allclose(mosaic["n2_sample_array"], expected)
    assert mosaic["_n2_sample_array_source"] == ("legacy_material", None)
    np.testing.assert_array_equal(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelengths,
    )


def test_simulation_kernel_kwargs_normalizes_cif_source_and_recomputes_in_meters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_sample\n", encoding="utf-8")
    wavelengths = np.array([1.0, 1.2], dtype=np.float64)
    returned = np.array([0.91 + 0.01j, 0.92 + 0.02j], dtype=np.complex128)
    captured: dict[str, object] = {}
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
        "n2_sample_array": np.array([1.0 + 0.0j], dtype=np.complex128),
        "_n2_sample_array_source": ["cif_path", str(cif_path)],
        "_n2_sample_array_wavelength_snapshot": np.array([1.5, 1.6], dtype=np.float64),
    }

    def fake_resolve(lambda_m_array, *, cif_path=None):
        captured["lambda_m_array"] = np.asarray(lambda_m_array, dtype=np.float64).copy()
        captured["cif_path"] = cif_path
        return returned.copy()

    monkeypatch.setattr(optimization, "resolve_index_of_refraction_array", fake_resolve)

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    np.testing.assert_allclose(captured["lambda_m_array"], wavelengths * 1.0e-10)
    assert captured["cif_path"] == str(cif_path.resolve())
    assert mosaic["_n2_sample_array_source"] == ("cif_path", str(cif_path.resolve()))
    np.testing.assert_allclose(kwargs["n2_sample_array_override"], returned)
    np.testing.assert_allclose(mosaic["n2_sample_array"], returned)
    np.testing.assert_allclose(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelengths,
    )


def test_simulation_kernel_kwargs_preserves_authoritative_array_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_sample\n", encoding="utf-8")
    supplied = np.array([0.95 + 0.05j, 0.96 + 0.06j], dtype=np.complex128)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": np.array([1.0, 1.1], dtype=np.float64),
        "n2_sample_array": supplied.copy(),
        "_n2_sample_array_source": ("cif_path", str(cif_path)),
    }

    monkeypatch.setattr(
        optimization,
        "resolve_index_of_refraction_array",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CIF recompute should not run without a snapshot")
        ),
    )
    monkeypatch.setattr(
        optimization,
        "_legacy_kernel_n2_sample_array_from_angstrom",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy helper should not run for authoritative arrays")
        ),
    )

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    np.testing.assert_allclose(kwargs["n2_sample_array_override"], supplied)
    np.testing.assert_allclose(mosaic["n2_sample_array"], supplied)


def test_simulation_kernel_kwargs_recomputes_wrong_length_array_when_source_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_sample\n", encoding="utf-8")
    wavelengths = np.array([1.0, 1.1], dtype=np.float64)
    returned = np.array([0.9 + 0.01j, 0.91 + 0.02j], dtype=np.complex128)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
        "n2_sample_array": np.array([1.0 + 0.0j], dtype=np.complex128),
        "_n2_sample_array_source": ("cif_path", str(cif_path)),
    }

    monkeypatch.setattr(
        optimization,
        "resolve_index_of_refraction_array",
        lambda *args, **kwargs: returned.copy(),
    )

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    np.testing.assert_allclose(kwargs["n2_sample_array_override"], returned)
    np.testing.assert_allclose(mosaic["n2_sample_array"], returned)
    np.testing.assert_allclose(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelengths,
    )


def test_simulation_kernel_kwargs_recomputes_malformed_array_when_source_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_sample\n", encoding="utf-8")
    wavelengths = np.array([1.0, 1.1], dtype=np.float64)
    returned = np.array([0.88 + 0.01j, 0.89 + 0.02j], dtype=np.complex128)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
        "n2_sample_array": ["bad-cache"],
        "_n2_sample_array_source": ("cif_path", str(cif_path)),
    }

    monkeypatch.setattr(
        optimization,
        "resolve_index_of_refraction_array",
        lambda *args, **kwargs: returned.copy(),
    )

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    np.testing.assert_allclose(kwargs["n2_sample_array_override"], returned)
    np.testing.assert_allclose(mosaic["n2_sample_array"], returned)
    np.testing.assert_allclose(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelengths,
    )


def test_simulation_kernel_kwargs_recomputes_malformed_snapshot_when_source_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_sample\n", encoding="utf-8")
    wavelengths = np.array([1.0, 1.1], dtype=np.float64)
    returned = np.array([0.87 + 0.01j, 0.88 + 0.02j], dtype=np.complex128)
    supplied = np.array([0.95 + 0.05j, 0.96 + 0.06j], dtype=np.complex128)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
        "n2_sample_array": supplied.copy(),
        "_n2_sample_array_source": ("cif_path", str(cif_path)),
        "_n2_sample_array_wavelength_snapshot": ["bad-snapshot"],
    }

    monkeypatch.setattr(
        optimization,
        "resolve_index_of_refraction_array",
        lambda *args, **kwargs: returned.copy(),
    )

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    np.testing.assert_allclose(kwargs["n2_sample_array_override"], returned)
    np.testing.assert_allclose(mosaic["n2_sample_array"], returned)
    np.testing.assert_allclose(
        mosaic["_n2_sample_array_wavelength_snapshot"],
        wavelengths,
    )


def test_simulation_kernel_kwargs_drops_wrong_length_authoritative_array_without_source() -> None:
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": np.array([1.0, 1.1], dtype=np.float64),
        "n2_sample_array": np.array([1.0 + 0.0j], dtype=np.complex128),
    }

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    assert "n2_sample_array_override" not in kwargs


def test_simulation_kernel_kwargs_drops_malformed_authoritative_array_without_source() -> None:
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": np.array([1.0, 1.1], dtype=np.float64),
        "n2_sample_array": ["bad-cache"],
    }

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 1.0}, mosaic)

    assert "n2_sample_array_override" not in kwargs


def test_simulation_kernel_kwargs_falls_back_to_legacy_when_cif_recompute_breaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cif_path = tmp_path / "missing.cif"
    wavelengths = np.array([1.0, np.nan], dtype=np.float64)
    mosaic = {
        "beam_x_array": np.array([0.0, 0.0], dtype=np.float64),
        "wavelength_array": wavelengths.copy(),
        "_n2_sample_array_source": ("cif_path", str(cif_path)),
    }

    monkeypatch.setattr(
        optimization,
        "resolve_index_of_refraction_array",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing cif")),
    )

    kwargs = optimization._simulation_kernel_kwargs({"lambda": 1.0, "n2": 0.97 + 0.02j}, mosaic)

    expected = _legacy_kernel_n2_sample_array_from_angstrom(
        wavelengths,
        nominal_n2=0.97 + 0.02j,
        sample_count=2,
    )
    np.testing.assert_allclose(kwargs["n2_sample_array_override"], expected)
    assert mosaic["_n2_sample_array_source"] == ("cif_path", str(cif_path.resolve()))
    assert "n2_sample_array" not in mosaic
    assert "_n2_sample_array_wavelength_snapshot" not in mosaic


def test_simulation_cache_retains_entries_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"cache": {"families": {"fit_simulation": "always"}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    cache = SimulationCache(keys=["gamma"])
    params = {"gamma": 1.0}
    image = np.ones((2, 2), dtype=np.float64)
    max_positions = np.ones((1, 7), dtype=np.float64)

    cache.store(params, image, max_positions)

    cached = cache.get(params)
    assert cached is not None
    np.testing.assert_allclose(cached[0], image)
    np.testing.assert_allclose(cached[1], max_positions)


def test_simulation_cache_discards_entries_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_config_dir(
        tmp_path,
        debug={"debug": {"cache": {"families": {"fit_simulation": "never"}}}},
    )
    monkeypatch.setenv(loader.ENV_CONFIG_DIR, str(cfg))
    cache = SimulationCache(keys=["gamma"])
    params = {"gamma": 1.0}

    cache.store(
        params,
        np.ones((2, 2), dtype=np.float64),
        np.ones((1, 7), dtype=np.float64),
    )

    assert cache.get(params) is None
    assert cache.images == {}
    assert cache.max_positions == {}

def test_retain_fit_simulation_cache_shim_delegates_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        optimization._runtime,
        "retain_fit_simulation_cache",
        lambda: calls.append("called") or False,
    )

    assert optimization._retain_fit_simulation_cache() is False
    assert calls == ["called"]


def _make_profile_roi(
    *,
    reflection_index: int,
    hkl: tuple[int, int, int],
    family: str,
    axis_name: str,
    measured_profile: np.ndarray,
) -> optimization.MosaicProfileROI:
    profile = np.asarray(measured_profile, dtype=np.float64)
    measured_area = float(np.sum(profile))
    return optimization.MosaicProfileROI(
        dataset_index=0,
        dataset_label="bg0",
        reflection_index=int(reflection_index),
        hkl=tuple(int(v) for v in hkl),
        family=str(family),
        axis_name=str(axis_name),
        center_row=0.5,
        center_col=0.5,
        row_bounds=(0, 2),
        col_bounds=(0, 2),
        flat_indices=np.array([0, 1], dtype=np.int64),
        axis_bin_indices=np.array([0, 1], dtype=np.int64),
        signal_mask=np.array([True, True], dtype=bool),
        side_mask=np.array([False, False], dtype=bool),
        signal_counts=np.array([1.0, 1.0], dtype=np.float64),
        side_counts=np.array([0.0, 0.0], dtype=np.float64),
        axis_bin_centers=np.array([-0.1, 0.1], dtype=np.float64),
        axis_half_span_deg=0.1,
        orthogonal_half_window_deg=0.1,
        measured_two_theta_deg=0.1,
        measured_phi_deg=0.1,
        measured_profile=profile,
        measured_area=measured_area,
        measured_shape_profile=profile / measured_area,
    )


def _make_prepared_mosaic_profile_dataset() -> optimization.MosaicProfileDatasetContext:
    in_plane_roi = _make_profile_roi(
        reflection_index=0,
        hkl=(1, 1, 0),
        family="in_plane",
        axis_name="phi",
        measured_profile=np.array([1.0, 3.0], dtype=np.float64),
    )
    specular_roi = _make_profile_roi(
        reflection_index=1,
        hkl=(0, 0, 1),
        family="specular",
        axis_name="two_theta",
        measured_profile=np.array([2.0, 4.0], dtype=np.float64),
    )
    return optimization.MosaicProfileDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        experimental_image=np.ones((2, 2), dtype=np.float64),
        miller=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        rois=[in_plane_roi, specular_roi],
        measured_peak_count=2,
        in_plane_roi_count=1,
        specular_roi_count=1,
    )


@pytest.mark.parametrize(
    ("retain_cache", "expected_process_calls"),
    [
        (True, 1),
        (False, 4),
    ],
)
def test_optimization_mosaic_image_cache_respects_retention_gate(
    monkeypatch: pytest.MonkeyPatch,
    retain_cache: bool,
    expected_process_calls: int,
) -> None:
    prepared_dataset = _make_prepared_mosaic_profile_dataset()
    process_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        optimization,
        "_build_mosaic_profile_dataset_contexts",
        lambda *args, **kwargs: ([prepared_dataset], []),
    )
    monkeypatch.setattr(
        optimization,
        "_retain_fit_simulation_cache",
        lambda: bool(retain_cache),
    )
    monkeypatch.setattr(
        optimization,
        "_extract_profile_from_flat_image",
        lambda flat_image, roi: np.asarray(
            [float(flat_image[0]), float(flat_image[1])],
            dtype=np.float64,
        ),
    )

    def fake_process_peaks_parallel_safe(*args, **kwargs):
        process_calls.append(dict(kwargs))
        return (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)

    monkeypatch.setattr(
        optimization,
        "_process_peaks_parallel_safe",
        fake_process_peaks_parallel_safe,
    )

    def fake_least_squares(fun, x0, **kwargs):
        x = np.asarray(x0, dtype=np.float64)
        fun(x)
        return OptimizeResult(
            x=x,
            fun=fun(x),
            success=True,
            message="stub-cache-gate",
            nfev=2,
        )

    monkeypatch.setattr(optimization, "least_squares", fake_least_squares)

    result = optimization._fit_mosaic_shape_parameters_profiles(
        miller=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        image_size=2,
        params={
            "a": 1.0,
            "c": 1.0,
            "lambda": 1.0,
            "corto_detector": 1.0,
            "gamma": 0.0,
            "Gamma": 0.0,
            "chi": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "n2": 1.0,
            "debye_x": 0.0,
            "debye_y": 0.0,
            "center": [0.0, 0.0],
            "pixel_size_m": 1.0e-4,
            "mosaic_params": {
                "beam_x_array": np.array([0.0], dtype=np.float64),
                "beam_y_array": np.array([0.0], dtype=np.float64),
                "theta_array": np.array([0.0], dtype=np.float64),
                "phi_array": np.array([0.0], dtype=np.float64),
                "wavelength_array": np.array([1.0], dtype=np.float64),
                "sigma_mosaic_deg": 0.4,
                "gamma_mosaic_deg": 0.3,
                "eta": 0.2,
            },
        },
        dataset_specs=[{"dataset_index": 0, "label": "bg0", "theta_initial": 0.0}],
        roi_half_width=1,
        min_total_rois=2,
        min_per_dataset_rois=2,
        fit_theta_i=False,
        max_restarts=0,
    )

    assert result.success is True
    assert len(process_calls) == expected_process_calls
    assert process_calls
    assert all(call.get("collect_hit_tables") is False for call in process_calls)


def test_geometry_objective_reuses_trial_source_rows_for_same_params_signature() -> None:
    builder_calls: list[dict[str, object]] = []

    def build_source_rows(*, local_params):
        builder_calls.append(dict(local_params))
        return {
            "rows": [
                {
                    "source_row_index": len(builder_calls),
                    "hkl": (1, 0, 0),
                }
            ],
            "source": "test-builder",
        }

    subset = optimization.ReflectionSimulationSubset(
        miller=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.asarray([10.0], dtype=np.float64),
        measured_entries=[],
        original_indices=np.asarray([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = optimization.GeometryFitDatasetContext(
        dataset_index=7,
        label="background-7",
        theta_initial=0.0,
        subset=subset,
        qr_fit_trial_source_rows_builder=build_source_rows,
        qr_fit_trial_source_rows_builder_kind="test-builder-kind",
    )
    fit_context: dict[str, object] = {"prediction_source_rows_cache": {}}

    first = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 1.0},
        params_signature="same-signature",
        fit_context=fit_context,
    )
    second = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 9.0},
        params_signature="same-signature",
        fit_context=fit_context,
    )
    third = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 9.0},
        params_signature="new-signature",
        fit_context=fit_context,
    )

    assert len(builder_calls) == 2
    assert first["source_rows_rebuilt_or_reused"] == "rebuilt_for_trial_params"
    assert first["prediction_source_rows_cache_size"] == 1
    assert first["prediction_source_rows_cache_max_entries"] is None
    assert first["prediction_source_rows_cache_eviction_count"] == 0
    assert second["rows"] == first["rows"]
    assert second["source_rows_rebuilt_or_reused"] == "reused_for_same_params_signature"
    assert second["reuse_valid_for_same_params_signature"] is True
    assert second["prediction_source_rows_cache_size"] == 1
    assert second["prediction_source_rows_cache_max_entries"] is None
    assert second["prediction_source_rows_cache_eviction_count"] == 0
    assert third["source_rows_rebuilt_or_reused"] == "rebuilt_for_trial_params"
    assert third["prediction_source_rows_cache_size"] == 2
    assert third["prediction_source_rows_cache_max_entries"] is None
    assert third["prediction_source_rows_cache_eviction_count"] == 0


def test_geometry_objective_trial_source_rows_cache_is_bounded() -> None:
    builder_calls: list[dict[str, object]] = []

    def build_source_rows(*, local_params):
        builder_calls.append(dict(local_params))
        return {
            "rows": [
                {
                    "source_row_index": len(builder_calls),
                    "hkl": (1, 0, 0),
                }
            ],
            "source": "test-builder",
        }

    subset = optimization.ReflectionSimulationSubset(
        miller=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.asarray([10.0], dtype=np.float64),
        measured_entries=[],
        original_indices=np.asarray([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = optimization.GeometryFitDatasetContext(
        dataset_index=7,
        label="background-7",
        theta_initial=0.0,
        subset=subset,
        qr_fit_trial_source_rows_builder=build_source_rows,
        qr_fit_trial_source_rows_builder_kind="test-builder-kind",
    )
    cache = optimization._BoundedPredictionSourceRowsCache(max_entries=3)
    fit_context: dict[str, object] = {"prediction_source_rows_cache": cache}

    results = [
        optimization._build_trial_qr_source_rows_payload(
            dataset_ctx,
            trial_params={"center_x": float(index)},
            params_signature=f"signature-{index}",
            fit_context=fit_context,
        )
        for index in range(5)
    ]

    assert len(builder_calls) == 5
    assert len(cache) == 3
    assert cache.eviction_count == 2
    assert results[-1]["prediction_source_rows_cache_size"] == 3
    assert results[-1]["prediction_source_rows_cache_max_entries"] == 3
    assert results[-1]["prediction_source_rows_cache_eviction_count"] == 2

    cached_call_count = len(builder_calls)
    cached = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 3.0},
        params_signature="signature-3",
        fit_context=fit_context,
    )

    assert len(builder_calls) == cached_call_count
    assert cached["source_rows_rebuilt_or_reused"] == "reused_for_same_params_signature"
    assert cached["objective_cache_hit"] is True
    assert cached["prediction_source_rows_cache_size"] == 3
    assert cached["prediction_source_rows_cache_max_entries"] == 3
    assert cached["prediction_source_rows_cache_eviction_count"] == 2

    evicted = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 0.0},
        params_signature="signature-0",
        fit_context=fit_context,
    )

    assert len(builder_calls) == cached_call_count + 1
    assert evicted["source_rows_rebuilt_or_reused"] == "rebuilt_for_trial_params"
    assert evicted["objective_cache_hit"] is False
    assert len(cache) == 3
    assert cache.eviction_count == 3
