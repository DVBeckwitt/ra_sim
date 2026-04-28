from __future__ import annotations

import numpy as np

from ra_sim.fitting.diffuse_background import (
    DiffuseBackgroundConfig,
    build_detector_valid_mask,
    build_peak_exclusion_mask,
    diffuse_background_config_from_mapping,
    estimate_phi_block_residual_background,
    fit_diffuse_background_native,
    subtract_diffuse_background,
)


def _synthetic_radial_scene(
    *,
    shape: tuple[int, int] = (128, 128),
    seed: int = 3,
) -> dict[str, np.ndarray | tuple[float, float]]:
    rng = np.random.default_rng(seed)
    rr, cc = np.indices(shape, dtype=np.float64)
    center = (62.0, 65.0)
    radius = np.hypot(rr - center[0], cc - center[1])
    two_theta = 0.035 * radius
    phi = np.degrees(np.arctan2(rr - center[0], cc - center[1]))
    background = 1050.0 * np.exp(-two_theta / 1.7) + 52.0 + 10.0 * np.sin(two_theta * 2.5)
    noise = rng.normal(0.0, 3.0, size=shape)

    peak_center = (84.4, 73.6)
    peak_sigma = 2.1
    peak = 510.0 * np.exp(
        -0.5
        * (
            ((rr - peak_center[0]) / peak_sigma) ** 2
            + ((cc - peak_center[1]) / peak_sigma) ** 2
        )
    )
    weak_peak = 180.0 * np.exp(
        -0.5
        * (
            ((rr - 43.2) / 2.8) ** 2
            + ((cc - 91.8) / 2.8) ** 2
        )
    )
    image = background + peak + weak_peak + noise
    return {
        "image": image,
        "background": background,
        "peak": peak,
        "two_theta": two_theta,
        "phi": phi,
        "center": center,
        "peak_center": peak_center,
    }


def test_mode_aliases_and_mapping_round_trip() -> None:
    cfg = diffuse_background_config_from_mapping(
        {
            "enabled": True,
            "mode": "radial-plus-caked-2d",
            "scale": "0.75",
            "radial_quantile": 2.0,
            "apply_to_display": "true",
        }
    )

    assert cfg.enabled is True
    assert cfg.mode == "radial_plus_caked_2d"
    assert cfg.scale == 0.75
    assert cfg.radial_quantile == 1.0
    assert cfg.apply_to_display is True
    assert diffuse_background_config_from_mapping({"mode": "radial+caked"}).mode == (
        "radial_plus_caked_2d"
    )
    assert diffuse_background_config_from_mapping({"mode": "radial-plus-phi-blocks"}).mode == (
        "radial_plus_phi_blocks"
    )
    assert diffuse_background_config_from_mapping({"mode": "radial+phi+caked"}).mode == (
        "radial_plus_phi_blocks_plus_caked_2d"
    )


def test_radial_halo_fit_masks_peaks_and_improves_peak_area() -> None:
    scene = _synthetic_radial_scene()
    image = np.asarray(scene["image"], dtype=np.float64)
    true_background = np.asarray(scene["background"], dtype=np.float64)
    true_peak = np.asarray(scene["peak"], dtype=np.float64)
    two_theta = np.asarray(scene["two_theta"], dtype=np.float64)
    peak_center = scene["peak_center"]
    assert isinstance(peak_center, tuple)

    cfg = DiffuseBackgroundConfig(
        enabled=True,
        mode="radial",
        radial_bin_width_deg=0.06,
        radial_quantile=0.35,
        radial_smooth_sigma_deg=0.18,
        peak_mask_sigma=3.0,
        peak_mask_radius_px=7.0,
        direct_beam_mask_radius_px=13.0,
    )
    result = fit_diffuse_background_native(
        image,
        two_theta_deg=two_theta,
        config=cfg,
        direct_beam_center_rc=scene["center"],
        simulated_peaks_rc=[peak_center],
    )

    model = np.asarray(result["model"], dtype=np.float64)
    corrected = np.asarray(result["corrected"], dtype=np.float64)
    exclusion = np.asarray(result["exclusion_mask"], dtype=bool)
    valid = np.asarray(result["valid_mask"], dtype=bool)
    outside_mask = valid & ~exclusion & (true_peak < 1.0)
    rel_error = np.abs(model[outside_mask] - true_background[outside_mask]) / np.maximum(
        true_background[outside_mask],
        1.0,
    )
    assert float(np.median(rel_error)) < 0.05

    pr, pc = peak_center
    rr, cc = np.indices(image.shape)
    peak_roi = (rr - pr) ** 2 + (cc - pc) ** 2 <= 5.0**2
    true_peak_area = float(np.sum(true_peak[peak_roi]))
    raw_area_error = abs(float(np.sum(image[peak_roi])) - true_peak_area)
    corrected_area_error = abs(float(np.sum(corrected[peak_roi])) - true_peak_area)
    assert corrected_area_error < raw_area_error * 0.15

    local = corrected[peak_roi]
    weights = np.clip(local - np.nanmedian(local), 0.0, None)
    centroid_r = float(np.sum(rr[peak_roi] * weights) / np.sum(weights))
    centroid_c = float(np.sum(cc[peak_roi] * weights) / np.sum(weights))
    assert abs(centroid_r - pr) < 0.25
    assert abs(centroid_c - pc) < 0.25

    peak_pixel = (int(round(pr)), int(round(pc)))
    assert exclusion[peak_pixel]
    assert abs(float(model[peak_pixel]) - float(true_background[peak_pixel])) < (
        0.08 * float(true_background[peak_pixel])
    )


def test_invalid_detector_support_pixels_are_ignored() -> None:
    scene = _synthetic_radial_scene()
    image = np.asarray(scene["image"], dtype=np.float64).copy()
    image[:, 48:54] = np.nan
    cfg = DiffuseBackgroundConfig(enabled=True, mode="radial")

    valid = build_detector_valid_mask(image, cfg)
    result = fit_diffuse_background_native(
        image,
        two_theta_deg=np.asarray(scene["two_theta"], dtype=np.float64),
        config=cfg,
    )

    assert not np.any(valid[:, 48:54])
    assert not np.any(np.asarray(result["valid_mask"], dtype=bool)[:, 48:54])
    assert np.all(np.isnan(np.asarray(result["model"], dtype=np.float64)[:, 48:54]))


def test_direct_beam_region_is_excluded() -> None:
    scene = _synthetic_radial_scene()
    image = np.asarray(scene["image"], dtype=np.float64)
    cfg = DiffuseBackgroundConfig(
        enabled=True,
        mode="radial",
        peak_mask_sigma=999.0,
        peak_mask_radius_px=0.0,
        direct_beam_mask_radius_px=9.0,
    )
    valid = build_detector_valid_mask(image, cfg)
    exclusion = build_peak_exclusion_mask(
        image,
        valid,
        cfg,
        direct_beam_center_rc=scene["center"],
    )

    center = scene["center"]
    assert isinstance(center, tuple)
    assert exclusion[int(round(center[0])), int(round(center[1]))]
    assert exclusion[int(round(center[0] + 8.0)), int(round(center[1]))]
    assert not exclusion[int(round(center[0] + 13.0)), int(round(center[1]))]


def test_subtract_preserves_negative_values_and_off_mode_is_raw() -> None:
    corrected = subtract_diffuse_background(
        np.array([[1.0, 5.0]]),
        np.array([[2.0, 3.0]]),
        DiffuseBackgroundConfig(enabled=True, mode="radial", scale=1.0),
    )
    assert corrected[0, 0] == -1.0

    scene = _synthetic_radial_scene()
    raw = np.asarray(scene["image"], dtype=np.float64)
    result = fit_diffuse_background_native(
        raw,
        two_theta_deg=np.asarray(scene["two_theta"], dtype=np.float64),
        config=DiffuseBackgroundConfig(enabled=False, mode="radial"),
    )
    np.testing.assert_allclose(result["corrected"], raw)
    np.testing.assert_allclose(result["model"], np.zeros_like(raw))


def _synthetic_phi_block_scene(
    *,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    azimuth_axis = np.linspace(-179.0, 179.0, 180)
    radial_axis = np.linspace(8.0, 52.0, 320)
    theta = radial_axis[None, :]
    phi = azimuth_axis[:, None]
    radial = 850.0 * np.exp(-theta / 26.0) + 55.0 + 12.0 * np.cos(theta / 3.0)
    sector = np.floor((azimuth_axis + 180.0) / 15.0).astype(int)
    sector_offsets = 14.0 * np.sin(0.9 * sector) + 8.0 * np.where(sector % 3 == 0, 1.0, -0.5)
    sector_gain = 0.9 + 0.35 * np.cos(0.7 * sector)
    block = (
        sector_offsets[:, None]
        + 42.0
        * sector_gain[:, None]
        * np.exp(-0.5 * ((theta - 22.0) / 1.2) ** 2)
        + 26.0
        * (1.0 - 0.25 * sector_gain[:, None])
        * np.exp(-0.5 * ((theta - 40.0) / 1.8) ** 2)
    )
    noise = rng.normal(0.0, 1.8, size=(azimuth_axis.size, radial_axis.size))
    image = radial + block + noise
    image[46:58, 120:148] = np.nan
    image[:, 260:268] = np.nan
    return {
        "image": image,
        "two_theta": np.broadcast_to(theta, image.shape).copy(),
        "phi": np.broadcast_to(phi, image.shape).copy(),
        "radial_axis": radial_axis,
        "azimuth_axis": azimuth_axis,
        "block": block,
    }


def _coarse_block_median_mad(
    values: np.ndarray,
    radial_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    *,
    theta_width: float = 1.0,
    phi_width: float = 15.0,
) -> float:
    theta_edges = np.arange(
        np.floor(float(np.nanmin(radial_axis)) / theta_width) * theta_width,
        np.ceil(float(np.nanmax(radial_axis)) / theta_width) * theta_width + theta_width,
        theta_width,
    )
    phi_edges = np.arange(
        np.floor(float(np.nanmin(azimuth_axis)) / phi_width) * phi_width,
        np.ceil(float(np.nanmax(azimuth_axis)) / phi_width) * phi_width + phi_width,
        phi_width,
    )
    medians: list[float] = []
    for pi in range(phi_edges.size - 1):
        phi_mask = (azimuth_axis >= phi_edges[pi]) & (azimuth_axis < phi_edges[pi + 1])
        for ti in range(theta_edges.size - 1):
            theta_mask = (radial_axis >= theta_edges[ti]) & (radial_axis < theta_edges[ti + 1])
            cell = values[np.ix_(phi_mask, theta_mask)]
            cell = cell[np.isfinite(cell)]
            if cell.size:
                medians.append(float(np.median(cell)))
    arr = np.asarray(medians, dtype=np.float64)
    center = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - center)))


def test_phi_block_mode_reduces_synthetic_block_residual_and_keeps_components() -> None:
    scene = _synthetic_phi_block_scene()
    cfg_common = dict(
        enabled=True,
        radial_bin_width_deg=0.20,
        radial_quantile=0.50,
        radial_smooth_sigma_deg=0.25,
        peak_mask_sigma=999.0,
        peak_mask_radius_px=0.0,
        direct_beam_mask_radius_px=0.0,
        phi_block_theta_bin_width_deg=1.0,
        phi_block_phi_bin_width_deg=15.0,
        phi_block_quantile=0.50,
        phi_block_min_pixels=4,
        phi_block_min_coverage=0.02,
        phi_block_smooth_theta_bins=0.10,
        phi_block_smooth_phi_bins=0.10,
        phi_block_outlier_sigma=8.0,
    )
    radial = fit_diffuse_background_native(
        scene["image"],
        two_theta_deg=scene["two_theta"],
        phi_deg=scene["phi"],
        caked_radial_axis_deg=scene["radial_axis"],
        caked_azimuth_axis_deg=scene["azimuth_axis"],
        config=DiffuseBackgroundConfig(mode="radial", **cfg_common),
    )
    phi_blocks = fit_diffuse_background_native(
        scene["image"],
        two_theta_deg=scene["two_theta"],
        phi_deg=scene["phi"],
        caked_radial_axis_deg=scene["radial_axis"],
        caked_azimuth_axis_deg=scene["azimuth_axis"],
        config=DiffuseBackgroundConfig(mode="radial_plus_phi_blocks", **cfg_common),
    )

    radial_mad = _coarse_block_median_mad(
        np.asarray(radial["corrected"], dtype=np.float64),
        scene["radial_axis"],
        scene["azimuth_axis"],
    )
    phi_mad = _coarse_block_median_mad(
        np.asarray(phi_blocks["corrected"], dtype=np.float64),
        scene["radial_axis"],
        scene["azimuth_axis"],
    )

    assert radial_mad > 3.0
    assert phi_mad < 0.4 * radial_mad
    assert np.nanmin(np.asarray(phi_blocks["corrected"], dtype=np.float64)) < 0.0
    assert np.all(np.isnan(np.asarray(phi_blocks["model"], dtype=np.float64)[:, 260:268]))
    components = phi_blocks["background_components"]
    assert isinstance(components, dict)
    assert components["phi_blocks_detector"] is phi_blocks["phi_block_model_detector"]
    assert components["phi_blocks_caked"] is phi_blocks["phi_block_model_caked"]
    diagnostics = phi_blocks["diagnostics"]
    assert diagnostics["phi_block_enabled"] is True
    assert diagnostics["phi_block_reduction_fraction"] > 0.3


def test_phi_block_plus_caked_refits_slow_residual_after_phi_blocks() -> None:
    scene = _synthetic_phi_block_scene(seed=9)
    cfg = DiffuseBackgroundConfig(
        enabled=True,
        mode="radial_plus_phi_blocks_plus_caked_2d",
        radial_bin_width_deg=0.20,
        radial_quantile=0.50,
        radial_smooth_sigma_deg=0.25,
        caked_theta_window_deg=2.0,
        caked_phi_window_deg=20.0,
        caked_quantile=0.50,
        peak_mask_sigma=999.0,
        peak_mask_radius_px=0.0,
        direct_beam_mask_radius_px=0.0,
        phi_block_theta_bin_width_deg=1.0,
        phi_block_phi_bin_width_deg=15.0,
        phi_block_min_pixels=4,
        phi_block_min_coverage=0.02,
    )
    result = fit_diffuse_background_native(
        scene["image"],
        two_theta_deg=scene["two_theta"],
        phi_deg=scene["phi"],
        caked_radial_axis_deg=scene["radial_axis"],
        caked_azimuth_axis_deg=scene["azimuth_axis"],
        config=cfg,
    )

    assert result["phi_block_model_caked"] is not None
    assert result["caked_residual_model"] is not None
    assert result["slow_caked_model_detector"] is not None
    assert result["after_radial_before_phi_blocks_caked"] is not None
    assert result["after_phi_blocks_caked"] is not None


def test_phi_block_interpolation_modes_preserve_or_smooth_edges() -> None:
    radial_axis = np.linspace(0.0, 5.0, 51)
    azimuth_axis = np.linspace(-30.0, 30.0, 121)
    phi = azimuth_axis[:, None]
    residual = np.where(phi < 0.0, -10.0, 10.0) * np.ones((1, radial_axis.size))
    valid = np.ones_like(residual, dtype=bool)
    exclusion = np.zeros_like(residual, dtype=bool)

    nearest = estimate_phi_block_residual_background(
        residual,
        radial_axis,
        azimuth_axis,
        valid,
        exclusion,
        DiffuseBackgroundConfig(
            enabled=True,
            mode="radial_plus_phi_blocks",
            phi_block_theta_bin_width_deg=1.0,
            phi_block_phi_bin_width_deg=15.0,
            phi_block_min_pixels=1,
            phi_block_min_coverage=0.0,
            phi_block_smooth_theta_bins=0.0,
            phi_block_smooth_phi_bins=0.0,
            phi_block_preserve_block_edges=True,
        ),
    )
    linear = estimate_phi_block_residual_background(
        residual,
        radial_axis,
        azimuth_axis,
        valid,
        exclusion,
        DiffuseBackgroundConfig(
            enabled=True,
            mode="radial_plus_phi_blocks",
            phi_block_theta_bin_width_deg=1.0,
            phi_block_phi_bin_width_deg=15.0,
            phi_block_min_pixels=1,
            phi_block_min_coverage=0.0,
            phi_block_smooth_theta_bins=0.0,
            phi_block_smooth_phi_bins=0.0,
            phi_block_interpolation="linear",
            phi_block_preserve_block_edges=False,
        ),
    )

    nearest_model = np.asarray(nearest["phi_block_model"], dtype=np.float64)
    linear_model = np.asarray(linear["phi_block_model"], dtype=np.float64)
    center_col = radial_axis.size // 2
    assert set(np.round(nearest_model[:, center_col], 6)) <= {-10.0, 10.0}
    boundary = np.argmin(np.abs(azimuth_axis))
    assert abs(float(linear_model[boundary, center_col])) < 9.0
