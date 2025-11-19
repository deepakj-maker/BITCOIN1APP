# tests/test_features.py
import pandas as pd
import numpy as np
from features import compute_indicators

def test_compute_indicators_basic():
    # create a simple increasing price series
    data = {
        "Date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "Close": np.arange(10.0, 20.0),
        "High": np.arange(10.5, 20.5),
        "Low": np.arange(9.5, 19.5),
        "Volume": np.arange(100, 110)
    }
    df = pd.DataFrame(data)
    out = compute_indicators(df, close_col="Close", high_col="High", low_col="Low", vol_col="Volume")
    # Ensure some indicator columns exist
    assert "sma_10" in out.columns
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    # Values should be finite numbers (no NaN after fill)
    assert out["sma_10"].isnull().sum() == 0
    assert np.isfinite(out["rsi_14"]).all()
