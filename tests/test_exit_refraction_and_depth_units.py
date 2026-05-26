import inspect

import numpy as np

from ra_sim.simulation import diffraction, projection_debug, simulation
from ra_sim.simulation.types import SimulationRequest


def _python_callable(func):
    return getattr(func, "py_func", func)


def test_exit_projection_defaults_to_external_air_wavevector():
    assert projection_debug.resolve_exit_projection_mode_flag(None) == (
        projection_debug.EXIT_PROJECTION_EXTERNAL
    )
    assert projection_debug.exit_projection_mode_label(None) == "external"
    assert (
        SimulationRequest.__dataclass_fields__["exit_projection_mode"].default
        == "external"
    )
    assert (
        inspect.signature(simulation._build_legacy_request)
        .parameters["exit_projection_mode"]
        .default
        == "external"
    )
    assert (
        inspect.signature(simulation.simulate_diffraction)
        .parameters["exit_projection_mode"]
        .default
        == "external"
    )
    assert (
        diffraction._PROCESS_PEAKS_PARALLEL_DEFAULTS["exit_projection_mode"]
        == diffraction.EXIT_PROJECTION_EXTERNAL
    )


def test_internal_projection_remains_explicit_legacy_mode():
    assert (
        projection_debug.resolve_exit_projection_mode_flag("internal")
        == projection_debug.EXIT_PROJECTION_INTERNAL
    )
    assert (
        projection_debug.resolve_exit_projection_mode_flag(
            projection_debug.EXIT_PROJECTION_INTERNAL
        )
        == projection_debug.EXIT_PROJECTION_INTERNAL
    )
    assert (
        projection_debug.resolve_exit_projection_mode_flag("external")
        == projection_debug.EXIT_PROJECTION_EXTERNAL
    )
    assert (
        projection_debug.resolve_exit_projection_mode_flag("refracted")
        == projection_debug.EXIT_PROJECTION_EXTERNAL
    )


def test_depth_m_is_converted_to_angstrom_before_exact_attenuation_terms():
    precompute_sample_terms = _python_callable(diffraction._precompute_sample_terms)

    _, sample_terms, _, _, _ = precompute_sample_terms(
        np.array([1.54], dtype=np.float64),
        1.0 + 1.0e-6j,
        np.array([1.0 + 1.0e-6j], dtype=np.complex128),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.05], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.0,
        50.0e-9,
        0.0,
        0.0,
        diffraction.OPTICS_MODE_EXACT,
        0.0,
        0.0,
        0.0,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )

    assert sample_terms[0, diffraction._SAMPLE_COL_L_IN] == 500.0


def test_mm_scale_depth_m_is_still_converted_to_angstrom():
    thickness_to_angstrom = _python_callable(diffraction._thickness_to_angstrom)

    assert thickness_to_angstrom(1.0e-3) == 1.0e7
