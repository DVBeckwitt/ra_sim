import numpy as np

from ra_sim.simulation import simulation as sim_mod


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_simulate_diffraction_reuses_supplied_profile_samples(monkeypatch):
    seen = {}

    def fake_generate_random_profiles(*args, **kwargs):
        raise AssertionError("generate_random_profiles should not be called")

    def fake_process_peaks_parallel_safe(*args, **kwargs):
        seen["beam_x_array"] = np.asarray(args[16], dtype=np.float64).copy()
        seen["beam_y_array"] = np.asarray(args[17], dtype=np.float64).copy()
        seen["theta_array"] = np.asarray(args[18], dtype=np.float64).copy()
        seen["phi_array"] = np.asarray(args[19], dtype=np.float64).copy()
        seen["wavelength_array"] = np.asarray(args[23], dtype=np.float64).copy()
        image = np.array(args[6], copy=True)
        image += 3.0
        return image, [], np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(sim_mod, "generate_random_profiles", fake_generate_random_profiles)
    monkeypatch.setattr(
        sim_mod,
        "process_peaks_parallel_safe",
        fake_process_peaks_parallel_safe,
    )

    profile_samples = {
        "beam_x_array": np.array([0.1, -0.2], dtype=np.float64),
        "beam_y_array": np.array([0.3, -0.4], dtype=np.float64),
        "theta_array": np.array([0.5, -0.6], dtype=np.float64),
        "phi_array": np.array([0.7, -0.8], dtype=np.float64),
        "wavelength_array": np.array([1.1, 1.2], dtype=np.float64),
    }

    image = sim_mod.simulate_diffraction(
        theta_initial=0.0,
        cor_angle=0.0,
        gamma=0.0,
        Gamma=0.0,
        chi=0.0,
        psi_z=0.0,
        zs=0.0,
        zb=0.0,
        debye_x_value=0.0,
        debye_y_value=0.0,
        corto_detector_value=0.1,
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        image_size=8,
        av=4.0,
        cv=7.0,
        lambda_=1.0,
        psi=0.0,
        n2=1.0,
        center=[4.0, 4.0],
        num_samples=2,
        divergence_sigma=0.1,
        bw_sigma=0.1,
        sigma_mosaic_var=_DummyVar(0.2),
        gamma_mosaic_var=_DummyVar(0.3),
        eta_var=_DummyVar(0.05),
        profile_samples=profile_samples,
    )

    assert np.allclose(image, 3.0)
    for key, expected in profile_samples.items():
        assert np.array_equal(seen[key], expected)
