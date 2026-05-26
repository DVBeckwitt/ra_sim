"""Regression checks for Fresnel transmission call behavior."""

from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction


def test_sample_term_precompute_uses_exact_fresnel_boolean_polarization_flags(
    monkeypatch,
) -> None:
    t_calls: list[bool] = []
    power_calls: list[bool] = []

    def fake_fresnel_t_exact(
        _kz1: complex,
        _kz2: complex,
        _eps1: complex,
        _eps2: complex,
        is_s_polarized: bool,
    ) -> complex:
        assert type(is_s_polarized) is bool
        t_calls.append(is_s_polarized)
        return 1.0 + 0.0j

    def fake_fresnel_power_t_exact(
        _t: complex,
        _kz1: complex,
        _kz2: complex,
        _eps1: complex,
        _eps2: complex,
        is_s_polarized: bool,
    ) -> float:
        assert type(is_s_polarized) is bool
        power_calls.append(is_s_polarized)
        return 1.0

    monkeypatch.setattr(diffraction, "_fresnel_t_exact", fake_fresnel_t_exact)
    monkeypatch.setattr(diffraction, "_fresnel_power_t_exact", fake_fresnel_power_t_exact)
    monkeypatch.setattr(
        diffraction,
        "_build_sample_rotation",
        lambda *_args: (
            np.eye(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        diffraction,
        "intersect_line_plane",
        lambda *_args: (0.0, 0.0, 0.0, True),
    )
    monkeypatch.setattr(
        diffraction,
        "transmit_angle_grazing",
        lambda theta, _n2_samp: float(theta),
    )
    monkeypatch.setattr(
        diffraction,
        "ktz_components",
        lambda *_args: (1.0, 0.5),
    )
    monkeypatch.setattr(
        diffraction,
        "safe_path_length",
        lambda thickness, _theta: float(thickness),
    )

    diffraction._precompute_sample_terms.py_func(
        np.array([1.0], dtype=np.float64),
        1.0 + 0.0j,
        np.array([], dtype=np.complex128),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.0,
        1.0,
        0.0,
        0.0,
            diffraction.OPTICS_MODE_EXACT,
        0.0,
        0.0,
        0.0,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert t_calls == [True, False]
    assert power_calls == [True, False]
