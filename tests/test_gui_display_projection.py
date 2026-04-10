import numpy as np

from ra_sim.gui import display_projection


def test_normalize_display_raster_size_limit_clamps_to_supported_range() -> None:
    assert display_projection.normalize_display_raster_size_limit(
        -5,
        fallback=100,
    ) == display_projection.MIN_DISPLAY_RASTER_SIZE
    assert display_projection.normalize_display_raster_size_limit(
        99999,
        fallback=100,
    ) == display_projection.MAX_DISPLAY_RASTER_SIZE
    assert display_projection.normalize_display_raster_size_limit(
        "250.6",
        fallback=100,
    ) == 251


def test_downsample_raster_for_display_returns_original_when_limit_is_large() -> None:
    image = np.arange(25, dtype=float).reshape(5, 5)

    projected = display_projection.downsample_raster_for_display(
        image,
        max_size=20,
    )

    assert projected is image


def test_downsample_raster_for_display_uses_centered_samples() -> None:
    image = np.arange(64, dtype=float).reshape(8, 8)

    projected = display_projection.downsample_raster_for_display(
        image,
        max_size=3,
    )

    assert projected.shape == (3, 3)
    assert np.array_equal(projected, image[np.ix_([1, 4, 6], [1, 4, 6])])


def test_downsample_raster_for_display_handles_rectangular_images() -> None:
    image = np.arange(40, dtype=float).reshape(10, 4)

    projected = display_projection.downsample_raster_for_display(
        image,
        max_size=4,
    )

    assert projected.shape == (4, 2)
    assert np.array_equal(projected, image[np.ix_([1, 3, 6, 8], [1, 3])])


def test_project_raster_to_view_crops_to_visible_window_with_overscan() -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)

    projection = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(4.0, 6.0),
        axis_ylim=(6.0, 4.0),
        max_size=100,
        bbox_width_px=1000,
        bbox_height_px=1000,
        overscan_fraction=0.0,
    )

    assert projection is not None
    assert projection.extent == (4.0, 6.0, 6.0, 4.0)
    assert np.array_equal(projection.image, image[4:6, 4:6])


def test_project_raster_to_view_limits_to_axes_pixel_budget() -> None:
    image = np.arange(300 * 300, dtype=float).reshape(300, 300)

    projection = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
    )

    assert projection is not None
    assert projection.image.shape == (50, 60)
    assert np.array_equal(
        projection.image,
        image[np.ix_(np.arange(50) * 6 + 3, np.arange(60) * 5 + 2)],
    )


def test_project_raster_to_view_preserves_zoomed_detail_when_budget_is_large() -> None:
    image = np.arange(256 * 256, dtype=float).reshape(256, 256)

    projection = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 256.0, 256.0, 0.0),
        axis_xlim=(120.0, 136.0),
        axis_ylim=(140.0, 124.0),
        max_size=3000,
        bbox_width_px=400,
        bbox_height_px=400,
        overscan_fraction=0.0,
    )

    assert projection is not None
    assert projection.image.shape == (16, 16)
    assert np.array_equal(projection.image, image[116:132, 120:136])
