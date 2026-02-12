from __future__ import annotations

import numpy as np

from ra_sim.simulation.engine import simulate, simulate_qr_rods
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


def _build_request(image_size: int = 8) -> SimulationRequest:
    return SimulationRequest(
        miller=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        geometry=DetectorGeometry(
            image_size=image_size,
            av=4.0,
            cv=7.0,
            lambda_angstrom=1.54,
            distance_m=0.1,
            gamma_deg=0.0,
            Gamma_deg=0.0,
            chi_deg=0.0,
            psi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            center=np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64),
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ),
        beam=BeamSamples(
            beam_x_array=np.zeros(1, dtype=np.float64),
            beam_y_array=np.zeros(1, dtype=np.float64),
            theta_array=np.zeros(1, dtype=np.float64),
            phi_array=np.zeros(1, dtype=np.float64),
            wavelength_array=np.ones(1, dtype=np.float64),
        ),
        mosaic=MosaicParams(
            sigma_mosaic_deg=0.5,
            gamma_mosaic_deg=0.4,
            eta=0.2,
        ),
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=np.zeros((image_size, image_size), dtype=np.float64),
    )


def test_simulate_respects_typed_request_with_custom_runner() -> None:
    request = _build_request()

    def fake_runner(*args, **kwargs):
        image = np.array(args[6], copy=True)
        image += 2.0
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    result = simulate(request, peak_runner=fake_runner)
    assert np.allclose(result.image, 2.0)
    assert result.degeneracy is None
    assert result.q_data.shape == (1,)


def test_simulate_qr_rods_respects_typed_request_with_custom_runner() -> None:
    request = _build_request()
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}

    def fake_runner(*args, **kwargs):
        image = np.array(args[5], copy=True)
        image += 5.0
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
            np.array([1], dtype=np.int32),
        )

    result = simulate_qr_rods(qr_dict, request, peak_runner=fake_runner)
    assert np.allclose(result.image, 5.0)
    assert np.array_equal(result.degeneracy, np.array([1], dtype=np.int32))
