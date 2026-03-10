import math

import numpy as np

from ra_sim.fitting.background_peak_matching import (
    _refine_peak_center,
    match_simulated_peaks_to_background,
)


def _synthetic_background(
    shape: tuple[int, int],
    peaks: list[tuple[float, float, float, float]],
    *,
    noise_sigma: float = 0.02,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    image = np.zeros(shape, dtype=np.float64)
    for col, row, amplitude, sigma in peaks:
        image += amplitude * np.exp(
            -((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * sigma**2)
        )
    if noise_sigma > 0.0:
        rng = np.random.default_rng(42)
        image += rng.normal(0.0, noise_sigma, size=shape)
    return np.clip(image, 0.0, None)


def _match_config(search_radius_px: float) -> dict[str, float]:
    return {
        "search_radius_px": float(search_radius_px),
        "local_max_size_px": 3,
        "smooth_sigma_px": 2.0,
        "climb_sigma_px": 0.8,
        "min_prominence_sigma": 0.2,
        "min_match_prominence_sigma": 0.2,
        "max_candidate_peaks": 64,
        "k_neighbors": 6,
    }


def test_match_simulated_peaks_to_background_finds_corresponding_local_peaks():
    background = _synthetic_background(
        (64, 64),
        [
            (16.0, 18.0, 7.0, 1.1),
            (24.5, 18.2, 6.5, 1.1),
        ],
    )
    simulated = [
        {"hkl": (1, 0, 0), "label": "1,0,0", "sim_col": 15.4, "sim_row": 18.4, "weight": 5.0},
        {"hkl": (2, 0, 0), "label": "2,0,0", "sim_col": 24.0, "sim_row": 17.8, "weight": 4.5},
    ]

    matches, stats, _ = match_simulated_peaks_to_background(
        simulated,
        background,
        _match_config(search_radius_px=4.0),
    )

    assert len(matches) == 2
    assert math.isclose(float(stats["search_radius_px"]), 4.0)

    match_by_label = {str(entry["label"]): entry for entry in matches}
    assert math.hypot(float(match_by_label["1,0,0"]["x"]) - 16.0, float(match_by_label["1,0,0"]["y"]) - 18.0) < 1.0
    assert math.hypot(float(match_by_label["2,0,0"]["x"]) - 24.5, float(match_by_label["2,0,0"]["y"]) - 18.2) < 1.0


def test_match_simulated_peaks_to_background_respects_search_radius():
    background = _synthetic_background(
        (64, 64),
        [
            (16.0, 16.0, 4.0, 1.0),
            (22.0, 16.0, 12.0, 1.0),
        ],
    )
    simulated = [
        {"hkl": (1, 0, 0), "label": "1,0,0", "sim_col": 15.6, "sim_row": 16.1, "weight": 5.0},
    ]

    matches, stats, _ = match_simulated_peaks_to_background(
        simulated,
        background,
        _match_config(search_radius_px=3.5),
    )

    assert len(matches) == 1
    assert math.isclose(float(stats["search_radius_px"]), 3.5)
    assert math.hypot(float(matches[0]["x"]) - 16.0, float(matches[0]["y"]) - 16.0) < 1.0
    assert math.hypot(float(matches[0]["x"]) - 22.0, float(matches[0]["y"]) - 16.0) > 3.0


def test_match_simulated_peaks_to_background_keeps_one_to_one_pairs():
    background = _synthetic_background(
        (64, 64),
        [
            (18.0, 18.0, 8.0, 1.0),
            (22.0, 18.0, 8.0, 1.0),
        ],
    )
    simulated = [
        {"hkl": (1, 0, 0), "label": "1,0,0", "sim_col": 18.4, "sim_row": 18.0, "weight": 6.0},
        {"hkl": (2, 0, 0), "label": "2,0,0", "sim_col": 21.6, "sim_row": 18.1, "weight": 5.5},
    ]

    matches, _, _ = match_simulated_peaks_to_background(
        simulated,
        background,
        _match_config(search_radius_px=4.0),
    )

    assert len(matches) == 2
    xs = sorted(float(entry["x"]) for entry in matches)
    assert abs(xs[0] - xs[1]) > 2.0


def test_refine_peak_center_recovers_quadratic_subpixel_summit():
    yy, xx = np.mgrid[0:9, 0:9]
    true_col = 4.35
    true_row = 3.65
    peakness = (
        20.0
        - 1.8 * (xx - true_col) ** 2
        - 1.3 * (yy - true_row) ** 2
        - 0.25 * (xx - true_col) * (yy - true_row)
    )

    refined_col, refined_row = _refine_peak_center(
        peakness.astype(np.float64),
        peakness.astype(np.float64),
        row_idx=4,
        col_idx=4,
    )

    assert abs(refined_col - true_col) < 1.0e-3
    assert abs(refined_row - true_row) < 1.0e-3
