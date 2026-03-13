from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.utils.calculations import IndexofRefraction, fresnel_transmission


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


def test_fast_optics_lut_matches_direct_transport_terms() -> None:
    wavelength_angstrom = 1.54
    n2 = IndexofRefraction(wavelength_angstrom * 1.0e-10)
    n2_real = float(np.real(n2))
    k0 = 2.0 * np.pi / wavelength_angstrom
    lut = np.zeros(
        (diffraction._FAST_OPTICS_LUT_SIZE, diffraction._FAST_OPTICS_LUT_COLS),
        dtype=np.float64,
    )
    diffraction._build_fast_optics_lut_row.py_func(
        lut,
        k0,
        n2,
        n2_real,
        0.0,
    )

    for theta in np.linspace(0.0, 0.45 * np.pi, 15):
        tf2_lut, im_kz_lut, l_out_lut, out_angle_lut = diffraction._lookup_fast_optics_lut_row.py_func(
            lut,
            float(theta),
        )

        tf_s = fresnel_transmission(float(theta), n2, True, False)
        tf_p = fresnel_transmission(float(theta), n2, False, False)
        tf2_direct = diffraction._sanitize_transmission_power.py_func(
            0.5
            * (
                (np.real(tf_s) * np.real(tf_s) + np.imag(tf_s) * np.imag(tf_s))
                + (np.real(tf_p) * np.real(tf_p) + np.imag(tf_p) * np.imag(tf_p))
            )
        )
        _, im_kz_direct = diffraction.ktz_components.py_func(k0, n2, float(theta))
        l_out_direct = 1.0 / max(2.0 * float(im_kz_direct), 1.0e-30)
        out_angle_direct = np.arccos(np.clip(np.cos(theta) * n2_real, -1.0, 1.0))

        np.testing.assert_allclose(tf2_lut, tf2_direct, rtol=5.0e-3, atol=5.0e-6)
        np.testing.assert_allclose(im_kz_lut, im_kz_direct, rtol=5.0e-3, atol=5.0e-9)
        np.testing.assert_allclose(l_out_lut, l_out_direct, rtol=5.0e-3, atol=5.0e-6)
        np.testing.assert_allclose(out_angle_lut, out_angle_direct, rtol=5.0e-3, atol=5.0e-6)
