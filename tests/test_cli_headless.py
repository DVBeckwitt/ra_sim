from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ra_sim import cli
from ra_sim.simulation.types import SimulationRequest


def test_run_headless_simulation_builds_typed_request(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        cli,
        "get_instrument_config",
        lambda: {
            "instrument": {
                "detector": {
                    "image_size": 8,
                    "monte_carlo_samples": 2,
                    "vmax": 10.0,
                    "pixel_size_m": 1.0e-4,
                },
                "geometry_defaults": {
                    "distance_m": 0.1,
                    "rot1": 0.0,
                    "rot2": 0.0,
                    "poni1_m": 0.0,
                    "poni2_m": 0.0,
                    "wavelength_m": 1.0e-10,
                },
                "beam": {
                    "wavelength_angstrom": 1.54,
                    "divergence_fwhm_deg": 0.05,
                    "bandwidth_sigma_fraction": 0.0,
                    "bandwidth_percent": 0.7,
                    "eta": 0.2,
                },
                "sample_orientation": {
                    "theta_initial_deg": 6.0,
                    "cor_deg": 0.0,
                    "chi_deg": 0.0,
                    "psi_deg": 0.0,
                    "psi_z_deg": 0.0,
                    "zb": 0.0,
                    "zs": 0.0,
                },
                "occupancies": {"default": [1.0, 1.0, 1.0]},
                "hendricks_teller": {
                    "max_miller_index": 3,
                    "default_p": [0.5],
                    "default_w": [1.0],
                    "finite_stack": False,
                    "stack_layers": 5,
                },
                "debye_waller": {"x": 0.0, "y": 0.0},
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "get_path",
        lambda key: str(tmp_path / f"{key}.dat"),
    )
    monkeypatch.setattr(
        cli,
        "parse_poni_file",
        lambda _path: {
            "Dist": 0.1,
            "Rot1": 0.0,
            "Rot2": 0.0,
            "Poni1": 0.0,
            "Poni2": 0.0,
            "Wavelength": 1.54e-10,
        },
    )
    monkeypatch.setattr(cli, "load_tilt_hint", lambda: None)
    monkeypatch.setattr(cli, "_parse_cif_cell_a_c", lambda _path: (4.0, 7.0))
    monkeypatch.setattr(
        cli,
        "ht_Iinf_dict",
        lambda **_kwargs: {"curves": True},
    )
    monkeypatch.setattr(
        cli,
        "ht_dict_to_qr_dict",
        lambda _curves: {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}},
    )
    monkeypatch.setattr(
        cli,
        "_combine_qr_dicts",
        lambda caches, _weights: caches[0]["qr"],
    )
    monkeypatch.setattr(
        cli,
        "generate_random_profiles",
        lambda *_args, **_kwargs: (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(cli, "IndexofRefraction", lambda _lambda_m: 1.0 + 0.0j)

    seen: dict[str, object] = {}

    def fake_simulate_qr_rods(qr_dict, request):
        seen["qr_dict"] = qr_dict
        seen["request"] = request
        return SimpleNamespace(image=np.ones((8, 8), dtype=np.float64))

    monkeypatch.setattr(cli, "simulate_qr_rods", fake_simulate_qr_rods)

    saved_paths: list[str] = []

    class _FakeImage:
        def save(self, path) -> None:
            saved_paths.append(str(path))

    monkeypatch.setattr(cli.Image, "fromarray", lambda *_args, **_kwargs: _FakeImage())

    out_path = tmp_path / "out.png"
    result = cli.run_headless_simulation(str(out_path), image_size=8, samples=2, vmax=5.0)

    assert result == str(out_path)
    assert saved_paths == [str(out_path)]
    assert 1 in seen["qr_dict"]
    assert seen["qr_dict"][1]["hk"] == (1, 0)
    assert np.array_equal(seen["qr_dict"][1]["L"], np.array([0.0], dtype=np.float64))
    assert np.array_equal(seen["qr_dict"][1]["I"], np.array([1.0], dtype=np.float64))
    assert seen["qr_dict"][1]["deg"] == 1

    request = seen["request"]
    assert isinstance(request, SimulationRequest)
    assert request.collect_hit_tables is False
    assert request.geometry.image_size == 8
    assert request.geometry.distance_m == 0.1
    assert request.mosaic.solve_q_steps == 1000
    assert np.array_equal(request.beam.wavelength_array, np.ones(2, dtype=np.float64))
