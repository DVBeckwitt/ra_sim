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


def test_downsample_raster_for_display_uses_strided_view() -> None:
    image = np.arange(64, dtype=float).reshape(8, 8)

    projected = display_projection.downsample_raster_for_display(
        image,
        max_size=3,
    )

    assert projected.shape == (3, 3)
    assert np.shares_memory(projected, image)
    assert np.array_equal(projected, image[::3, ::3])


def test_downsample_raster_for_display_handles_rectangular_images() -> None:
    image = np.arange(40, dtype=float).reshape(10, 4)

    projected = display_projection.downsample_raster_for_display(
        image,
        max_size=4,
    )

    assert projected.shape == (4, 2)
    assert np.array_equal(projected, image[::3, ::3])
