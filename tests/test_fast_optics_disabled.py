import pytest
import numpy as np

from ra_sim.simulation import diffraction


@pytest.mark.parametrize(
    "value",
    [
        diffraction.OPTICS_MODE_FAST,
        0,
        0.0,
        np.float64(0.0),
        "0",
        "fast",
        "approx",
        "fresnel_ctr_damping",
        "uncoupled fresnel + ctr damping (ufd)",
    ],
)
def test_fast_optics_mode_is_rejected(value):
    with pytest.raises(diffraction.FastOpticsDisabledError):
        diffraction.require_exact_optics_mode(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        diffraction.OPTICS_MODE_EXACT,
        1,
        1.0,
        np.float64(1.0),
        "1",
        "exact",
        "precise",
        "complex_k_dwba_slab",
        "complex-k dwba slab optics",
    ],
)
def test_exact_optics_mode_is_accepted(value):
    assert diffraction.require_exact_optics_mode(value) == diffraction.OPTICS_MODE_EXACT


@pytest.mark.parametrize(
    "value",
    [
        0.4,
        0.6,
        1.4,
        np.float64(0.51),
        np.nan,
        np.inf,
        2,
        "unknown",
    ],
)
def test_unsupported_optics_mode_is_rejected(value):
    with pytest.raises(ValueError, match="Unsupported optics_mode"):
        diffraction.require_exact_optics_mode(value)


def _peak_args():
    image_size = 4
    return (
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        image_size,
        4.0,
        7.0,
        1.54,
        np.zeros((image_size, image_size), dtype=np.float64),
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        0.2,
        0.1,
        0.05,
        np.array([1.54], dtype=np.float64),
        0.0,
        0.0,
        np.array([2.0, 2.0], dtype=np.float64),
        0.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        0,
    )


@pytest.mark.parametrize(
    "func",
    [
        diffraction.process_peaks_parallel,
        diffraction.process_peaks_parallel_safe,
        diffraction._process_peaks_parallel_impl,
        diffraction._process_peaks_parallel_weighted_events_python,
    ],
)
def test_peak_diffraction_entry_points_reject_fast_optics_before_compute(func):
    with pytest.raises(diffraction.FastOpticsDisabledError):
        func(*_peak_args(), optics_mode=diffraction.OPTICS_MODE_FAST)


@pytest.mark.parametrize(
    "func",
    [
        diffraction.process_qr_rods_parallel,
        diffraction.process_qr_rods_parallel_safe,
    ],
)
def test_qr_rod_diffraction_entry_points_reject_fast_optics_before_compute(func):
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}
    peak_args = _peak_args()
    rod_args = (qr_dict, *peak_args[2:])

    with pytest.raises(diffraction.FastOpticsDisabledError):
        func(*rod_args, optics_mode=diffraction.OPTICS_MODE_FAST)


def test_qr_rod_safe_wrapper_rejects_positional_fast_optics_before_compute():
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}
    peak_args = _peak_args()
    rod_args = (qr_dict, *peak_args[2:], False, 0.0, diffraction.OPTICS_MODE_FAST)

    with pytest.raises(diffraction.FastOpticsDisabledError):
        diffraction.process_qr_rods_parallel_safe(*rod_args)


def test_calculate_phi_rejects_fast_optics_before_compute():
    image_size = 4

    with pytest.raises(diffraction.FastOpticsDisabledError):
        diffraction.calculate_phi(
            1.0,
            0.0,
            0.0,
            4.0,
            7.0,
            np.array([1.54], dtype=np.float64),
            np.zeros((image_size, image_size), dtype=np.float64),
            image_size,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 + 0.0j,
            np.array([1.0 + 0.0j], dtype=np.complex128),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            1.0,
            0.2,
            0.1,
            0.05,
            0.0,
            0.0,
            np.array([2.0, 2.0], dtype=np.float64),
            0.0,
            0.0,
            np.eye(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.1, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.eye(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            0,
            np.empty((0, 0, 0), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            0,
            optics_mode=diffraction.OPTICS_MODE_FAST,
        )
