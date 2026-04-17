import numpy as np

from ra_sim.gui import display_projection


def test_normalize_display_raster_size_limit_clamps_to_supported_range() -> None:
    assert (
        display_projection.normalize_display_raster_size_limit(
            -5,
            fallback=100,
        )
        == display_projection.MIN_DISPLAY_RASTER_SIZE
    )
    assert (
        display_projection.normalize_display_raster_size_limit(
            99999,
            fallback=100,
        )
        == display_projection.MAX_DISPLAY_RASTER_SIZE
    )
    assert (
        display_projection.normalize_display_raster_size_limit(
            "250.6",
            fallback=100,
        )
        == 251
    )


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


def test_project_raster_to_view_preserve_bright_features_keeps_sparse_peak_visible() -> None:
    image = np.zeros((300, 300), dtype=float)
    image[149, 151] = 1000.0

    projection = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
        preserve_bright_features=True,
    )

    assert projection is not None
    assert projection.image.shape == (50, 60)
    assert float(np.max(projection.image)) == 1000.0


def test_project_raster_to_view_without_bright_feature_preservation_can_drop_sparse_peak() -> None:
    image = np.zeros((300, 300), dtype=float)
    image[149, 151] = 1000.0

    projection = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
        preserve_bright_features=False,
    )

    assert projection is not None
    assert projection.image.shape == (50, 60)
    assert float(np.max(projection.image)) == 0.0


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


def test_project_raster_to_view_reuses_cached_projection_for_same_request(
    monkeypatch,
) -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)
    stride_calls: list[tuple[int, int]] = []
    display_projection._PROJECTION_CACHE.clear()

    original_stride_downsample = display_projection._stride_downsample

    def _record_stride_downsample(source, *, target_rows, target_cols):
        stride_calls.append((int(target_rows), int(target_cols)))
        return original_stride_downsample(
            source,
            target_rows=target_rows,
            target_cols=target_cols,
        )

    monkeypatch.setattr(
        display_projection,
        "_stride_downsample",
        _record_stride_downsample,
    )

    first = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    second = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )

    assert first is not None
    assert second is first
    assert stride_calls == [(6, 6)]


def test_project_raster_to_view_cache_distinguishes_bright_feature_preservation() -> None:
    image = np.zeros((300, 300), dtype=float)
    image[149, 151] = 1000.0
    display_projection._PROJECTION_CACHE.clear()

    first = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
        preserve_bright_features=False,
    )
    second = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
        preserve_bright_features=True,
    )
    third = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 300.0, 300.0, 0.0),
        axis_xlim=(0.0, 300.0),
        axis_ylim=(300.0, 0.0),
        max_size=3000,
        bbox_width_px=60,
        bbox_height_px=50,
        overscan_fraction=0.0,
        preserve_bright_features=True,
    )

    assert first is not None
    assert second is not None
    assert first is not second
    assert third is second
    assert len(display_projection._PROJECTION_CACHE) == 2
    assert float(np.max(first.image)) == 0.0
    assert float(np.max(second.image)) == 1000.0


def test_project_raster_to_view_busts_cache_when_source_signature_changes_after_in_place_update(
    monkeypatch,
) -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)
    stride_calls: list[tuple[int, int]] = []
    display_projection._PROJECTION_CACHE.clear()

    original_stride_downsample = display_projection._stride_downsample

    def _record_stride_downsample(source, *, target_rows, target_cols):
        stride_calls.append((int(target_rows), int(target_cols)))
        return original_stride_downsample(
            source,
            target_rows=target_rows,
            target_cols=target_cols,
        )

    monkeypatch.setattr(
        display_projection,
        "_stride_downsample",
        _record_stride_downsample,
    )

    first = display_projection.project_raster_to_view(
        image,
        source_signature=1,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=6,
        bbox_height_px=6,
        overscan_fraction=0.0,
    )

    image[:, :] = image[:, :] + 1000.0

    second = display_projection.project_raster_to_view(
        image,
        source_signature=2,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=6,
        bbox_height_px=6,
        overscan_fraction=0.0,
    )

    assert first is not None
    assert second is not None
    assert second is not first
    assert stride_calls == [(6, 6), (6, 6)]
    assert np.array_equal(second.image, first.image + 1000.0)


def test_project_raster_to_view_recomputes_detached_projection_after_in_place_update_without_explicit_signature() -> (
    None
):
    image = np.arange(100, dtype=float).reshape(10, 10)
    display_projection._PROJECTION_CACHE.clear()

    first = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=6,
        bbox_height_px=6,
        overscan_fraction=0.0,
    )

    image[:, :] = image[:, :] + 1000.0

    second = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=6,
        bbox_height_px=6,
        overscan_fraction=0.0,
    )

    assert first is not None
    assert second is not None
    assert second is not first
    assert np.array_equal(second.image, first.image + 1000.0)


def test_project_raster_to_view_keeps_shared_full_frame_projection_live_without_explicit_signature() -> (
    None
):
    image = np.arange(100, dtype=float).reshape(10, 10)
    display_projection._PROJECTION_CACHE.clear()

    first = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=100,
        bbox_height_px=100,
        overscan_fraction=0.0,
    )

    image[:, :] = -1.0

    second = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(0.0, 10.0),
        axis_ylim=(10.0, 0.0),
        max_size=100,
        bbox_width_px=100,
        bbox_height_px=100,
        overscan_fraction=0.0,
    )

    assert first is not None
    assert second is first
    assert np.array_equal(second.image, image)


def test_project_raster_to_view_busts_cache_when_projection_inputs_change(
    monkeypatch,
) -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)
    stride_calls: list[tuple[int, int]] = []
    display_projection._PROJECTION_CACHE.clear()

    original_stride_downsample = display_projection._stride_downsample

    def _record_stride_downsample(source, *, target_rows, target_cols):
        stride_calls.append((int(target_rows), int(target_cols)))
        return original_stride_downsample(
            source,
            target_rows=target_rows,
            target_cols=target_cols,
        )

    monkeypatch.setattr(
        display_projection,
        "_stride_downsample",
        _record_stride_downsample,
    )

    baseline = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    same_again = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    extent_changed = display_projection.project_raster_to_view(
        image,
        extent=(1.0, 11.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    window_changed = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(3.0, 7.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    budget_changed = display_projection.project_raster_to_view(
        image,
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=80,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )
    source_changed = display_projection.project_raster_to_view(
        image.copy(),
        extent=(0.0, 10.0, 10.0, 0.0),
        axis_xlim=(2.0, 8.0),
        axis_ylim=(8.0, 2.0),
        max_size=100,
        bbox_width_px=120,
        bbox_height_px=120,
        overscan_fraction=0.0,
    )

    assert baseline is not None
    assert same_again is baseline
    assert extent_changed is not baseline
    assert window_changed is not baseline
    assert budget_changed is not baseline
    assert source_changed is not baseline
    assert len(stride_calls) == 5
