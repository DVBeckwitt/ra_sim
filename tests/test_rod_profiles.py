from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.fitting.rod_profiles import (
    binned_caked_mask_profile,
    caked_field_to_gui_phi,
    qz_profile_from_caked_mask,
)
from ra_sim.simulation.exact_cake_portable import prepare_gui_phi_display


def test_flat_caked_image_density_is_flat_despite_variable_pixel_count() -> None:
    image = np.full((3, 4), 5.0)
    model = np.full((3, 4), 7.0)
    qz_map = np.array(
        [
            [0.2, 1.2, 1.4, 2.1],
            [2.2, 2.3, 2.4, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    mask = np.isfinite(qz_map)

    profile = qz_profile_from_caked_mask(
        image=image,
        model=model,
        qz_map=qz_map,
        qz_edges=np.array([0.0, 1.0, 2.0, 3.0]),
        mask=mask,
    )

    np.testing.assert_array_equal(profile["pixel_count"], np.array([1, 2, 4]))
    np.testing.assert_allclose(profile["background_sum"], np.array([5.0, 10.0, 20.0]))
    np.testing.assert_allclose(profile["background_density"], np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(profile["fit_density"], np.array([7.0, 7.0, 7.0]))
    assert set(profile["acceptance_source"]) == {"pixel_count"}


def test_sum_signal_over_sum_normalization_path_is_used() -> None:
    image = np.array([[100.0, 200.0, 300.0]])
    model = np.array([[2.0, 4.0, 8.0]])
    coord = np.array([[0.2, 0.4, 0.8]])
    signal_sum = np.array([[10.0, 20.0, 30.0]])
    normalization_sum = np.array([[2.0, 3.0, 5.0]])

    profile = binned_caked_mask_profile(
        image=image,
        model=model,
        coord_map=coord,
        coord_edges=np.array([0.0, 1.0]),
        mask=np.ones_like(image, dtype=bool),
        signal_sum=signal_sum,
        normalization_sum=normalization_sum,
        coord_name="q",
    )

    assert profile["acceptance_source"][0] == "sum_normalization"
    assert profile["background_sum"][0] == 600.0
    assert profile["background_weighted_sum"][0] == 60.0
    assert profile["acceptance_sum"][0] == 10.0
    assert profile["background_density"][0] == 6.0
    assert profile["fit_weighted_sum"][0] == pytest.approx(2.0 * 2.0 + 4.0 * 3.0 + 8.0 * 5.0)
    assert profile["fit_density"][0] == pytest.approx(5.6)


def test_nonuniform_normalization_flat_signal_stays_flat() -> None:
    image = np.full((2, 4), 4.0)
    coord = np.array([[0.2, 0.3, 1.2, 1.3], [0.4, 0.5, 1.4, 1.5]])
    normalization_sum = np.array([[1.0, 10.0, 2.0, 20.0], [4.0, 40.0, 8.0, 80.0]])
    signal_sum = image * normalization_sum

    profile = qz_profile_from_caked_mask(
        image=image,
        qz_map=coord,
        qz_edges=np.array([0.0, 1.0, 2.0]),
        mask=np.ones_like(image, dtype=bool),
        signal_sum=signal_sum,
        normalization_sum=normalization_sum,
    )

    assert profile["acceptance_sum"][0] != profile["acceptance_sum"][1]
    np.testing.assert_allclose(profile["background_density"], np.array([4.0, 4.0]))


def test_acceptance_fallback_uses_weighted_sum_over_acceptance_sum() -> None:
    image = np.array([[2.0, 4.0, 8.0]])
    model = np.array([[1.0, 3.0, 5.0]])
    coord = np.array([[0.2, 0.4, 0.8]])
    acceptance = np.array([[1.0, 2.0, 3.0]])

    profile = binned_caked_mask_profile(
        image=image,
        model=model,
        coord_map=coord,
        coord_edges=np.array([0.0, 1.0]),
        mask=np.ones_like(image, dtype=bool),
        acceptance=acceptance,
        coord_name="q",
    )

    assert profile["acceptance_source"][0] == "acceptance"
    assert profile["acceptance_sum"][0] == 6.0
    assert profile["background_density"][0] == pytest.approx((2.0 + 8.0 + 24.0) / 6.0)
    assert profile["fit_density"][0] == pytest.approx((1.0 + 6.0 + 15.0) / 6.0)


def test_pixel_count_fallback_divides_raw_sum_by_selected_count() -> None:
    image = np.array([[2.0, 4.0, 8.0]])
    model = np.array([[1.0, 3.0, 5.0]])
    coord = np.array([[0.2, 0.4, 0.8]])

    profile = binned_caked_mask_profile(
        image=image,
        model=model,
        coord_map=coord,
        coord_edges=np.array([0.0, 1.0]),
        mask=np.ones_like(image, dtype=bool),
        coord_name="q",
    )

    assert profile["acceptance_source"][0] == "pixel_count"
    assert profile["pixel_count"][0] == 3
    assert profile["acceptance_sum"][0] == 3.0
    assert profile["background_density"][0] == profile["background_sum"][0] / 3.0
    assert profile["fit_density"][0] == profile["fit_sum"][0] / 3.0


def test_zero_acceptance_returns_nan_density() -> None:
    image = np.array([[2.0, 4.0]])
    coord = np.array([[0.2, 0.4]])
    signal_sum = np.array([[2.0, 4.0]])
    normalization_sum = np.array([[0.0, 0.0]])

    profile = binned_caked_mask_profile(
        image=image,
        coord_map=coord,
        coord_edges=np.array([0.0, 1.0]),
        mask=np.ones_like(image, dtype=bool),
        signal_sum=signal_sum,
        normalization_sum=normalization_sum,
        coord_name="q",
    )

    assert profile["acceptance_sum"][0] == 0.0
    assert np.isnan(profile["background_density"][0])
    assert np.isnan(profile["fit_density"][0])


def test_bad_optional_weight_shape_raises_value_error() -> None:
    image = np.ones((2, 2))
    with pytest.raises(ValueError, match="acceptance shape"):
        binned_caked_mask_profile(
            image=image,
            coord_map=np.ones_like(image),
            coord_edges=np.array([0.0, 1.0]),
            mask=np.ones_like(image, dtype=bool),
            acceptance=np.ones((2, 3)),
        )


def test_partial_signal_normalization_pair_raises_value_error() -> None:
    image = np.ones((2, 2))
    with pytest.raises(ValueError, match="signal_sum and normalization_sum"):
        binned_caked_mask_profile(
            image=image,
            coord_map=np.ones_like(image),
            coord_edges=np.array([0.0, 1.0]),
            mask=np.ones_like(image, dtype=bool),
            signal_sum=np.ones_like(image),
        )


def test_bad_coord_edges_raise_value_error() -> None:
    image = np.ones((2, 2))
    with pytest.raises(ValueError, match="coord_edges"):
        binned_caked_mask_profile(
            image=image,
            coord_map=np.ones_like(image),
            coord_edges=np.array([0.0, 1.0, 1.0]),
            mask=np.ones_like(image, dtype=bool),
        )


def test_model_none_returns_nan_fit_fields() -> None:
    image = np.ones((1, 2))
    profile = binned_caked_mask_profile(
        image=image,
        coord_map=np.array([[0.2, 0.4]]),
        coord_edges=np.array([0.0, 1.0]),
        mask=np.ones_like(image, dtype=bool),
    )

    for key in ("fit_sum", "fit_mean", "fit_weighted_sum", "fit_density"):
        assert key in profile
        assert np.isnan(profile[key][0])


def test_theta_map_stats_use_selected_support() -> None:
    image = np.ones((2, 2))
    coord = np.array([[0.2, 0.4], [0.6, 0.8]])
    theta = np.array([[10.0, 20.0], [np.nan, 40.0]])
    mask = np.array([[True, False], [True, True]])

    profile = binned_caked_mask_profile(
        image=image,
        coord_map=coord,
        coord_edges=np.array([0.0, 1.0]),
        mask=mask,
        theta_map=theta,
    )

    assert profile["two_theta_min"][0] == 10.0
    assert profile["two_theta_max"][0] == 40.0
    assert profile["two_theta_mean"][0] == 25.0


def test_caked_field_to_gui_phi_matches_prepare_gui_phi_display_order() -> None:
    raw_phi = np.array([90.0, 270.0, 0.0])
    radial = np.array([1.0, 2.0])
    field = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = SimpleNamespace(intensity=field, azimuthal_deg=raw_phi, radial_deg=radial)

    expected_field, expected_radial, expected_phi = prepare_gui_phi_display(result)
    actual_field, actual_radial, actual_phi = caked_field_to_gui_phi(field, raw_phi, radial)

    np.testing.assert_allclose(actual_field, expected_field)
    np.testing.assert_allclose(actual_radial, expected_radial)
    np.testing.assert_allclose(actual_phi, expected_phi)
