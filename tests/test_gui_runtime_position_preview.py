from __future__ import annotations

import numpy as np

from ra_sim.gui.runtime_position_preview import build_peak_position_preview_image


def test_build_peak_position_preview_image_marks_primary_and_secondary_centers() -> None:
    captured_hit_tables: list[object] = []

    def fake_hit_tables_to_max_positions(hit_tables):
        captured_hit_tables.extend(hit_tables)
        return np.array(
            [
                [2.0, 2.0, 3.0, 1.0, 5.0, 4.0],
                [0.0, 1.0, 1.0, 3.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )

    hit_tables = [object(), object()]
    image = build_peak_position_preview_image(
        hit_tables,
        image_size=8,
        hit_tables_to_max_positions=fake_hit_tables_to_max_positions,
    )

    assert captured_hit_tables == hit_tables
    assert image.shape == (8, 8)
    assert image[3, 2] > 0.0
    assert image[4, 5] > 0.0
    assert image[0, 6] > 0.0
    assert image[1, 1] == 0.0


def test_build_peak_position_preview_image_returns_blank_image_when_conversion_fails() -> None:
    image = build_peak_position_preview_image(
        [object()],
        image_size=5,
        hit_tables_to_max_positions=lambda _tables: (_ for _ in ()).throw(ValueError("boom")),
    )

    assert image.shape == (5, 5)
    assert not np.any(image)
