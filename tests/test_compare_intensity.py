import pytest
import pandas as pd
from compare_intensity import compute_metrics
from ra_sim.tools import compare_intensity as packaged_compare_intensity


def test_root_compare_intensity_is_packaged_wrapper():
    assert compute_metrics is packaged_compare_intensity.compute_metrics

def test_compute_metrics_simple():
    df = pd.DataFrame({
        'Total_scaled': [10, 20],
        'Numeric_area': [12, 18]
    })
    metrics = compute_metrics(df)
    assert metrics['rmse'] == pytest.approx(2.0)
    mean_ratio = (12/10 + 18/20) / 2
    assert metrics['mean_ratio'] == pytest.approx(mean_ratio)
