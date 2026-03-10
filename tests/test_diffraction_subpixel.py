import numpy as np

from ra_sim.simulation import diffraction


def test_accumulate_bilinear_hit_preserves_total_intensity_and_centroid():
    image = np.zeros((5, 5), dtype=np.float64)

    deposited = diffraction._accumulate_bilinear_hit(image, 5, 1.25, 2.75, 8.0)

    assert deposited is True
    assert np.isclose(float(image.sum()), 8.0)

    rows, cols = np.indices(image.shape, dtype=np.float64)
    row_center = float(np.sum(image * rows) / np.sum(image))
    col_center = float(np.sum(image * cols) / np.sum(image))
    assert np.isclose(row_center, 1.25)
    assert np.isclose(col_center, 2.75)


def test_hit_tables_to_max_positions_merges_nearby_hits_into_subpixel_centroid():
    hit_tables = [
        np.array(
            [
                [4.0, 10.20, 20.10, 0.0, 1.0, 0.0, 0.0],
                [2.0, 10.80, 20.70, 0.0, 1.0, 0.0, 0.0],
                [1.5, 30.40, 40.60, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    maxpos = diffraction.hit_tables_to_max_positions(hit_tables)

    raw_centroid_col = (4.0 * 10.20 + 2.0 * 10.80) / 6.0
    raw_centroid_row = (4.0 * 20.10 + 2.0 * 20.70) / 6.0

    assert np.isclose(float(maxpos[0, 0]), 6.0)
    assert 10.20 <= float(maxpos[0, 1]) <= raw_centroid_col
    assert 20.10 <= float(maxpos[0, 2]) <= raw_centroid_row
    assert np.isclose(float(maxpos[0, 3]), 1.5)
    assert np.isclose(float(maxpos[0, 4]), 30.40)
    assert np.isclose(float(maxpos[0, 5]), 40.60)


def test_hit_tables_to_max_positions_prefers_local_peak_over_cluster_tail():
    hit_tables = [
        np.array(
            [
                [12.0, 10.00, 20.00, 0.0, 1.0, 0.0, 0.0],
                [9.0, 10.20, 20.10, 0.0, 1.0, 0.0, 0.0],
                [6.0, 10.30, 20.05, 0.0, 1.0, 0.0, 0.0],
                [5.0, 11.10, 20.05, 0.0, 1.0, 0.0, 0.0],
                [3.0, 11.25, 20.00, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    maxpos = diffraction.hit_tables_to_max_positions(hit_tables)

    raw_centroid_col = (
        12.0 * 10.00
        + 9.0 * 10.20
        + 6.0 * 10.30
        + 5.0 * 11.10
        + 3.0 * 11.25
    ) / 35.0

    assert np.isclose(float(maxpos[0, 0]), 35.0)
    assert float(maxpos[0, 1]) < raw_centroid_col - 0.1
    assert 10.05 < float(maxpos[0, 1]) < 10.25
    assert 19.95 < float(maxpos[0, 2]) < 20.15
