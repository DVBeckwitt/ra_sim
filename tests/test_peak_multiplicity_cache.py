import numpy as np

from ra_sim.simulation import diffraction


def _run_process(
    miller: np.ndarray,
    intensities: np.ndarray,
    *,
    n_samp: int = 2,
    events_per_beam_phase: int = 1,
):
    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)

    beam_x = np.zeros(n_samp, dtype=np.float64)
    beam_y = np.zeros(n_samp, dtype=np.float64)
    theta = np.zeros(n_samp, dtype=np.float64)
    phi = np.zeros(n_samp, dtype=np.float64)
    wavelength = np.full(n_samp, 1.0, dtype=np.float64)

    return diffraction.process_peaks_parallel.py_func(
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        beam_x,
        beam_y,
        theta,
        phi,
        0.5,
        0.5,
        0.0,
        wavelength,
        0.0,
        0.0,
        np.array([8.0, 8.0], dtype=np.float64),
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        events_per_beam_phase=events_per_beam_phase,
    )


def test_process_peaks_parallel_enumerates_duplicate_gr_gz_peaks_independently(monkeypatch):
    seen_calls: list[tuple[float, float, float, float]] = []

    def fake_solve_q(
        _k_in_crystal,
        _k_scat,
        _G_vec,
        _sigma,
        _gamma_pv,
        _eta_pv,
        H,
        K,
        L,
        *_args,
        **_kwargs,
    ):
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    monkeypatch.setattr(
        diffraction,
        "solve_q",
        fake_solve_q,
    )

    def fake_project_weighted_candidate(**kwargs):
        seen_calls.append(
            (
                float(kwargs["H"]),
                float(kwargs["K"]),
                float(kwargs["L"]),
                float(kwargs["reflection_intensity"]),
            )
        )
        return True, 5.0, 6.0, 0.2, float(kwargs["reflection_intensity"])

    monkeypatch.setattr(
        diffraction,
        "_project_weighted_candidate",
        fake_project_weighted_candidate,
    )

    # (1,0,1) and (0,1,1) have identical Gr and Gz for hexagonal metric.
    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    intensities = np.array([2.5, 2.5], dtype=np.float64)

    _run_process(miller, intensities, n_samp=1, events_per_beam_phase=1)

    assert seen_calls == [(1.0, 0.0, 1.0, 2.5), (0.0, 1.0, 1.0, 2.5)]


def test_process_peaks_parallel_uses_exact_original_row_intensities(monkeypatch):
    seen_intensities: list[float] = []

    def fake_solve_q(
        _k_in_crystal,
        _k_scat,
        _G_vec,
        _sigma,
        _gamma_pv,
        _eta_pv,
        H,
        K,
        L,
        *_args,
        **_kwargs,
    ):
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    monkeypatch.setattr(
        diffraction,
        "solve_q",
        fake_solve_q,
    )

    def fake_project_weighted_candidate(**kwargs):
        seen_intensities.append(float(kwargs["reflection_intensity"]))
        return True, 5.0, 6.0, 0.2, float(kwargs["reflection_intensity"])

    monkeypatch.setattr(
        diffraction,
        "_project_weighted_candidate",
        fake_project_weighted_candidate,
    )

    miller = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    intensities = np.array([2.0, 4.0], dtype=np.float64)

    _run_process(miller, intensities, n_samp=1, events_per_beam_phase=1)

    assert seen_intensities == [2.0, 4.0]


def test_process_peaks_parallel_visits_each_sample_without_group_reuse(monkeypatch):
    observed: list[tuple[float, float, float]] = []

    def fake_solve_q(
        _k_in_crystal,
        _k_scat,
        _G_vec,
        _sigma,
        _gamma_pv,
        _eta_pv,
        H,
        K,
        L,
        *_args,
        **_kwargs,
    ):
        return np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), 0

    monkeypatch.setattr(
        diffraction,
        "solve_q",
        fake_solve_q,
    )

    def fake_project_weighted_candidate(**kwargs):
        observed.append(
            (
                float(kwargs["sample_weight"]),
                float(kwargs["k_x_scat"]),
                float(kwargs["k_y_scat"]),
            )
        )
        return True, 5.0, 6.0, 0.2, 1.0

    monkeypatch.setattr(
        diffraction,
        "_project_weighted_candidate",
        fake_project_weighted_candidate,
    )

    image_size = 16
    image = np.zeros((image_size, image_size), dtype=np.float64)
    beam_x = np.array([0.0, 0.1, -0.1], dtype=np.float64)
    beam_y = np.zeros(3, dtype=np.float64)
    theta = np.zeros(3, dtype=np.float64)
    phi = np.zeros(3, dtype=np.float64)
    wavelength = np.ones(3, dtype=np.float64)

    diffraction.process_peaks_parallel.py_func(
        np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        image_size,
        1.0,
        1.0,
        1.0,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        beam_x,
        beam_y,
        theta,
        phi,
        0.5,
        0.5,
        0.0,
        wavelength,
        0.0,
        0.0,
        np.array([8.0, 8.0], dtype=np.float64),
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        events_per_beam_phase=1,
    )

    assert len(observed) == 3
