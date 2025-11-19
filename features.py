"""
features.py

Lightweight feature builder used by app.py. Exposes:

- build_features_from_df(df, n_lags=60, use_extra_cols=True)

Behavior / return values:
- X : pandas.DataFrame (rows x features)
- y : numpy.ndarray (next-step target aligned with X)
- feat_names : list[str] (column names of X)
- enriched : pandas.DataFrame (original data with any computed indicator columns)

Notes:
- This implementation uses only pandas / numpy (no external TA libraries).
- The target `y` is the next-step Close price (shifted -1).
"""
from typing import Tuple, List

import numpy as np
import pandas as pd

def _ensure_close_series(df: pd.DataFrame) -> pd.Series:
    """Find a Close-like numeric series in df. Raises ValueError if none found."""
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce").reset_index(drop=True)
    # try last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        return pd.to_numeric(df[numeric_cols[-1]], errors="coerce").reset_index(drop=True)
    raise ValueError("Could not find a numeric Close column in the provided dataframe")

# --- simple technical indicators (pure pandas implementations) ---

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    # standard Wilder RSI implementation
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.diff(period)

def build_features_from_df(df: pd.DataFrame, n_lags: int = 60, use_extra_cols: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.DataFrame]:
    """Build lagged features + optional lightweight TA indicators from a price DataFrame.

    Args:
        df: input dataframe containing a Close-like column (or numeric column).
        n_lags: number of lag features to create (lag_1 .. lag_n_lags). lag_1 is previous timestep.
        use_extra_cols: if True, compute a small set of technical indicators and include them.

    Returns: (X, y, feat_names, enriched_df)
    """
    close = _ensure_close_series(df).astype(float)
    if len(close) < (n_lags + 2):
        raise ValueError(f"Not enough rows to produce {n_lags} lags (have {len(close)} rows)")

    enriched = pd.DataFrame({"Close": close})

    # optional extra columns
    if use_extra_cols:
        # short / long SMAs
        enriched["SMA_5"] = sma(close, 5)
        enriched["SMA_10"] = sma(close, 10)
        enriched["EMA_10"] = ema(close, 10)
        enriched["RSI_14"] = rsi(close, 14)
        enriched["MOM_10"] = momentum(close, 10)

    # build lag features: lag_1 is t-1, lag_n is t-n
    for i in range(1, int(n_lags) + 1):
        enriched[f"lag_{i}"] = close.shift(i)

    # target: next-step Close (shift -1)
    enriched["y_next"] = close.shift(-1)

    # drop rows with NA introduced by lags / indicators
    required_cols = [f"lag_{i}" for i in range(1, int(n_lags) + 1)] + ["y_next"]
    df_clean = enriched.dropna(subset=required_cols).reset_index(drop=True)

    # feature matrix
    feat_cols = [f"lag_{i}" for i in range(1, int(n_lags) + 1)]
    if use_extra_cols:
        extra_cols = [c for c in ["SMA_5", "SMA_10", "EMA_10", "RSI_14", "MOM_10"] if c in df_clean.columns]
        feat_cols += extra_cols

    X = df_clean.loc[:, feat_cols]
    y = df_clean.loc[:, "y_next"].to_numpy()

    feat_names = list(X.columns)

    return X, y, feat_names, enriched

# small test helper when running this file directly
if __name__ == "__main__":
    import pandas as pd
    s = pd.Series(np.linspace(1, 200, 200)) + np.random.normal(scale=0.5, size=200)
    df_test = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=len(s), freq="D"), "Close": s})
    X, y, fnames, enriched = build_features_from_df(df_test, n_lags=10, use_extra_cols=True)
    print("X.shape", X.shape)
    print("y.shape", y.shape)
    print("feat names", fnames[:10])
