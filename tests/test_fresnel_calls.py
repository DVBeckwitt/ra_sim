"""Regression checks for Fresnel transmission call behavior."""

from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction


def test_fast_optics_lut_builder_uses_boolean_direction_flags(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_fresnel_transmission(
        _theta: float,
        _n2_samp: complex,
        is_s_polarized: bool,
        is_incoming: bool,
    ) -> complex:
        assert type(is_s_polarized) is bool
        assert type(is_incoming) is bool
        calls.append((is_s_polarized, is_incoming))
        return 1.0 + 0.0j

    monkeypatch.setattr(diffraction, "fresnel_transmission", fake_fresnel_transmission)

    lut_row = np.zeros((3, diffraction._FAST_OPTICS_LUT_COLS), dtype=np.float64)
    diffraction._build_fast_optics_lut_row.py_func(
        lut_row,
        1.0,
        1.0 + 0.0j,
        1.0,
        0.0,
    )

    assert set(calls) == {(True, False), (False, False)}


def test_sample_term_precompute_uses_boolean_incident_flags(monkeypatch) -> None:
    calls: list[tuple[bool, bool]] = []

    def fake_fresnel_transmission(
        _theta: float,
        _n2_samp: complex,
        is_s_polarized: bool,
        is_incoming: bool,
    ) -> complex:
        assert type(is_s_polarized) is bool
        assert type(is_incoming) is bool
        calls.append((is_s_polarized, is_incoming))
        return 1.0 + 0.0j

    monkeypatch.setattr(diffraction, "fresnel_transmission", fake_fresnel_transmission)
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
        diffraction.OPTICS_MODE_FAST,
        0.0,
        0.0,
        0.0,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )

    assert calls == [(True, True), (False, True)]
