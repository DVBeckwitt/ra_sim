import json
import re
from pathlib import Path

import pytest


NOTEBOOK_PATH = Path("scripts/diagnostics/all_background_peak_fits.ipynb")


def _notebook_source() -> str:
    if not NOTEBOOK_PATH.exists():
        pytest.skip(f"{NOTEBOOK_PATH} is not present in this checkout")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_all_background_peak_fits_limits_qz_profiles_to_sixty_deg_two_theta() -> None:
    source = _notebook_source()

    assert "ROD_PROFILE_MAX_TWO_THETA_DEG = 60.0" in source
    assert "detector_two_theta_map=profile_detector_two_theta_map" in source
    assert "theta_map <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "two_theta_values <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "branch_mask = branch_mask & theta_region_within_profile_limit[None, :]" in source
    assert "ax.set_xlim(caked_region_theta_min, caked_region_theta_max)" in source


def test_background_peak_fits_notebook_uses_density_profiles() -> None:
    source = _notebook_source()

    for token in (
        "background_density",
        "fit_density",
        "caked_sum_signal",
        "caked_sum_normalization",
    ):
        assert token in source

    calls = re.findall(r"(?<!def )normalized_profile_pair\([^\)]*\)", source)
    assert calls
    assert not any("background_sum" in call or "fit_sum" in call for call in calls)


def test_background_peak_fits_notebook_uses_density_caked_figures() -> None:
    source = _notebook_source()

    for token in (
        'CAKED_FIGURE_INTENSITY_MODE = "density"',
        "def caked_image_for_intensity_mode(",
        '"caked_density_image"',
        '"caked_raw_sum_image"',
        '"caked_display_image"',
        '"caked_count"',
        'profile_bg.get("caked_display_image"',
    ):
        assert token in source

    assert 'caked_image = bg.get("caked_display_image"' in source
    assert 'caked_image = bg["caked_image"]' not in source
    assert 'caked_region_bg = caked_log_image(np.asarray(profile_bg["caked_image"]' not in source


def test_background_peak_fits_notebook_uses_rotated_gaussian_peak_fits() -> None:
    source = _notebook_source()

    for token in (
        "GAUSSIAN_TAIL_DISTANCE_WEIGHT = 1.25",
        "GAUSSIAN_CORE_SIGNAL_DOWNSCALE = 0.06",
        "GAUSSIAN_TAIL_OVERPREDICTION_START = 0.55",
        "GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT = 1.75",
        "def _rotated_gaussian_value_numba",
        "peak = math.exp(-0.5 * r2)",
        "_rotated_gaussian_residual_points_numba",
        "tail_overprediction_weight",
        "if residual > 0.0:",
        "residual *= tail_overprediction_weight[idx]",
        '"fit_model": "rotated_gaussian_plane"',
        "Fitted Gaussian peaks",
    ):
        assert token in source

    for removed in ("PSEUDO_VOIGT", "_pseudo_voigt", "Fitted pseudo-Voigt"):
        assert removed not in source
