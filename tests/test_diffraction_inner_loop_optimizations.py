from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction


def test_local_pixel_cache_matches_direct_bilinear_after_flushes() -> None:
    image_size = 12
    direct = np.zeros((image_size, image_size), dtype=np.float64)
    cached = np.zeros_like(direct)
    cache_keys = np.empty(8, dtype=np.int64)
    cache_values = np.empty(8, dtype=np.float64)
    diffraction._clear_local_pixel_cache.py_func(cache_keys, cache_values)
    entry_count = 0

    hits = [
        (1.2, 2.4, 0.5),
        (1.8, 2.9, 1.2),
        (3.1, 4.7, 0.9),
        (3.6, 4.2, 1.5),
        (6.3, 1.1, 0.7),
        (6.9, 1.4, 1.1),
        (8.25, 9.75, 0.6),
        (8.4, 9.2, 1.4),
        (10.1, 5.5, 0.8),
    ]

    for row_f, col_f, value in hits:
        diffraction._accumulate_bilinear_hit.py_func(direct, image_size, row_f, col_f, value)
        deposited, needs_flush, entry_count = diffraction._accumulate_bilinear_cached.py_func(
            image_size,
            row_f,
            col_f,
            value,
            cache_keys,
            cache_values,
            entry_count,
            5,
        )
        if needs_flush:
            entry_count = diffraction._flush_local_pixel_cache.py_func(
                cached,
                image_size,
                cache_keys,
                cache_values,
            )
            deposited, needs_flush, entry_count = diffraction._accumulate_bilinear_cached.py_func(
                image_size,
                row_f,
                col_f,
                value,
                cache_keys,
                cache_values,
                entry_count,
                5,
            )
        assert deposited
        assert not needs_flush

    diffraction._flush_local_pixel_cache.py_func(
        cached,
        image_size,
        cache_keys,
        cache_values,
    )
    np.testing.assert_allclose(cached, direct, rtol=0.0, atol=1.0e-12)
