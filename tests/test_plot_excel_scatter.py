import pandas as pd
from plot_excel_scatter import _find_intensity_columns

def test_find_intensity_columns_single():
    df = pd.DataFrame({
        'H': [0],
        'K': [0],
        'L': [1],
        'Intensity': [1.0],
    })
    cols = _find_intensity_columns(df, None)
    assert cols == ['Intensity']

def test_find_intensity_columns_multiple():
    df = pd.DataFrame({
        'h': [0],
        'k': [0],
        'l': [1],
        'A_scaled': [1.0],
        'B_scaled': [2.0],
    })
    cols = _find_intensity_columns(df, None)
    assert cols == ['A_scaled', 'B_scaled']


def test_find_intensity_columns_with_intensity_and_extra():
    df = pd.DataFrame({
        'h': [0],
        'k': [0],
        'l': [1],
        'Intensity': [1.0],
        'Intensity2': [2.0],
    })
    cols = _find_intensity_columns(df, None)
    assert cols == ['Intensity', 'Intensity2']
