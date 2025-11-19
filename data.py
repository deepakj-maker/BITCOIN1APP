# data.py
"""
Data fetching + single-file caching utilities for the Multi-Asset LSTM/GRU app.

- Public function:
    get_price_series(asset, source_preference='auto', interval='1d', lookback=1000)

- Caching:
    Single CSV file data_cache.csv next to this module stores rows for all asset+interval pairs.
    Columns: Date, Open, High, Low, Close, Volume, Source, Asset, Interval

- Sources supported:
    - binance (public REST klines)
    - yahoo (via yfinance)
    - av (AlphaVantage) - optional, requires ALPHAVANTAGE_API_KEY env var
"""

from __future__ import annotations
import os
import math
import time
from typing import Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# Lazy import yfinance
try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except Exception:
    _HAS_YFINANCE = False

# ----------------------
# Single-file cache (data_cache.csv)
# ----------------------
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data_cache.csv")
_CACHE_CSV_HEADER = ["Date", "Open", "High", "Low", "Close", "Volume", "Source", "Asset", "Interval"]

# ensure the cache file exists with header
if not os.path.exists(CACHE_FILE):
    try:
        import csv
        with open(CACHE_FILE, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(_CACHE_CSV_HEADER)
    except Exception:
        pd.DataFrame(columns=_CACHE_CSV_HEADER).to_csv(CACHE_FILE, index=False)


# ----------------------
# Helpers
# ----------------------
def _coerce_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert various OHLC formats into canonical columns and dtypes:
    Date, Open, High, Low, Close, Volume
    """
    df = df.copy()
    # map lower->orig for lookup
    cols_lower = {c.lower(): c for c in df.columns}

    def find_any(names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None

    date_col = find_any(["date", "time", "timestamp", "t", "index"])
    open_col = find_any(["open", "o"])
    high_col = find_any(["high", "h"])
    low_col = find_any(["low", "l"])
    close_col = find_any(["close", "c", "close_price", "adj close", "adj_close"])
    vol_col = find_any(["volume", "v", "vol", "q"])

    # If no date col but index is datetime-like, reset index and use it
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            date_col = df.columns[0]
        else:
            date_col = df.columns[0]

    rename_map = {}
    if date_col:
        rename_map[date_col] = "Date"
    if open_col:
        rename_map[open_col] = "Open"
    if high_col:
        rename_map[high_col] = "High"
    if low_col:
        rename_map[low_col] = "Low"
    if close_col:
        rename_map[close_col] = "Close"
    if vol_col:
        rename_map[vol_col] = "Volume"

    df = df.rename(columns=rename_map)

    # ensure columns exist
    for c in ["Date", "Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan

    # coerce Date
    try:
        # try ms epoch first
        df["Date"] = pd.to_datetime(df["Date"], unit="ms", errors="ignore")
    except Exception:
        pass
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # numeric columns
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)

    # sort ascending
    try:
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception:
        pass

    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# ----------------------
# Binance fetcher (public API)
# ----------------------
_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _fetch_binance_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch klines from Binance public API.
    Returns DataFrame with Date, Open, High, Low, Close, Volume
    """
    params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
    resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for row in data:
        open_time = int(row[0])
        o = float(row[1])
        h = float(row[2])
        l = float(row[3])
        c = float(row[4])
        v = float(row[5])
        rows.append([pd.to_datetime(open_time, unit="ms"), o, h, l, c, v])
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return df


# ----------------------
# Yahoo fetcher using yfinance
# ----------------------
def _fetch_yahoo(asset: str, interval: str, lookback: int) -> pd.DataFrame:
    """
    Fetch series from Yahoo via yfinance.
    """
    if not _HAS_YFINANCE:
        raise RuntimeError("yfinance is not installed. Install with `pip install yfinance` to use Yahoo source.")

    # Estimate period conservatively to cover 'lookback' rows
    if interval.endswith("m") and interval != "1h":
        minutes = int(interval[:-1])
        approx_minutes = minutes * lookback
        days = max(1, math.ceil(approx_minutes / (60 * 24)))
        period = f"{days}d"
    elif interval.endswith("h") or interval == "1h":
        hours = int(interval[:-1]) if interval.endswith("h") else 1
        approx_hours = hours * lookback
        days = max(1, math.ceil(approx_hours / 24))
        period = f"{days}d"
    else:
        period = f"{max(1, int(math.ceil(lookback)))}d"

    yf_interval = interval
    try:
        df = yf.download(tickers=asset, period=period, interval=yf_interval, progress=False, threads=False)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed for {asset} interval={interval} period={period}: {e}")

    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = df.reset_index()
    # rename to proper case
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df = _coerce_cols(df)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df


# ----------------------
# Alpha Vantage fetcher (optional)
# ----------------------
_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _fetch_alpha_vantage(symbol: str, interval: str, lookback: int, api_key: str) -> pd.DataFrame:
    if not api_key:
        raise RuntimeError("AlphaVantage API key required for AV source (set ALPHAVANTAGE_API_KEY env var).")

    if interval in ("1m", "5m", "15m", "30m", "60m"):
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full",
            "apikey": api_key
        }
    else:
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full", "apikey": api_key}

    resp = requests.get(_ALPHA_VANTAGE_URL, params=params, timeout=20)
    resp.raise_for_status()
    j = resp.json()

    ts_key = None
    for k in j.keys():
        if "Time Series" in k or "Time_Series" in k:
            ts_key = k
            break
    if ts_key is None:
        raise RuntimeError(f"AlphaVantage response did not contain time series data: {j}")

    series = j[ts_key]
    rows = []
    for dt_str, vals in series.items():
        try:
            dt = pd.to_datetime(dt_str)
            o = float(vals.get("1. open") or vals.get("open"))
            h = float(vals.get("2. high") or vals.get("high"))
            l = float(vals.get("3. low") or vals.get("low"))
            c = float(vals.get("4. close") or vals.get("close"))
            v = float(vals.get("5. volume") or vals.get("volume") or 0.0)
            rows.append([dt, o, h, l, c, v])
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) > lookback:
        df = df.iloc[-lookback:].reset_index(drop=True)
    return df


# ----------------------
# Cache helpers (single CSV)
# ----------------------
def _ensure_min_rows_and_write(df: pd.DataFrame, path: str, min_rows: int, asset: str, interval: str) -> None:
    """
    Update the single CSV cache at `path` with rows for `asset` + `interval`.
    Existing rows for that pair are removed and replaced by `df`.
    """
    out_df = df.copy().reset_index(drop=True)

    # ensure required columns
    for c in ["Date", "Open", "High", "Low", "Close", "Volume"]:
        if c not in out_df.columns:
            out_df[c] = np.nan

    # metadata
    out_df["Source"] = out_df.get("Source", None)
    out_df["Asset"] = asset
    out_df["Interval"] = interval

    # read existing cache
    try:
        existing = pd.read_csv(path, parse_dates=["Date"], infer_datetime_format=True)
    except Exception:
        existing = pd.DataFrame(columns=_CACHE_CSV_HEADER)

    # remove existing rows for this asset+interval
    if not existing.empty:
        mask_keep = ~((existing.get("Asset", "") == asset) & (existing.get("Interval", "") == interval))
        existing = existing.loc[mask_keep].copy().reset_index(drop=True)

    # normalize Date
    try:
        out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce")
    except Exception:
        pass

    # combine and keep canonical columns
    combined = pd.concat([existing, out_df[_CACHE_CSV_HEADER]], ignore_index=True, sort=False)
    combined = combined.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    try:
        combined = combined.sort_values("Date").reset_index(drop=True)
    except Exception:
        pass

    tmp_path = path + ".tmp"
    combined.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, path)
    except Exception:
        combined.to_csv(path, index=False)


# ----------------------
# Public API
# ----------------------
def get_price_series(asset: str, source_preference: Optional[str] = "auto", interval: str = "1d", lookback: int = 1000) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume, Source, Asset
    """
    asset = str(asset).strip()
    source_pref = (source_preference or "auto").lower()
    interval = str(interval)

    # choose source
    chosen = None
    up = asset.upper()
    if source_pref in ("binance", "b"):
        chosen = "binance"
    elif source_pref in ("yahoo", "yf", "y"):
        chosen = "yahoo"
    elif source_pref in ("av", "alphavantage", "alpha", "alpha_vantage"):
        chosen = "av"
    else:
        if up.endswith("USDT") or up.endswith("BUSD") or up.endswith("BTC") or up.endswith("ETH"):
            chosen = "binance"
        else:
            chosen = "yahoo"

    cache_path = CACHE_FILE

    # Try read from single-file cache for this asset+interval
    if os.path.exists(cache_path):
        try:
            df_cache_all = pd.read_csv(cache_path, parse_dates=["Date"], infer_datetime_format=True)
            df_cache = df_cache_all[
                (df_cache_all.get("Asset", "") == asset) & (df_cache_all.get("Interval", "") == interval)
            ].copy()
            if not df_cache.empty:
                df_cache = _coerce_cols(df_cache)
                if len(df_cache) >= int(lookback):
                    out = df_cache.iloc[-int(lookback):].copy().reset_index(drop=True)
                    out["Source"] = df_cache.get("Source", chosen)
                    out["Asset"] = asset
                    out["Interval"] = interval
                    return out[["Date", "Open", "High", "Low", "Close", "Volume", "Source", "Asset"]]
        except Exception:
            # corrupted cache; proceed to fetch fresh
            pass

    # Fetch fresh data
    df_result = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    fetch_error = None
    try:
        if chosen == "binance":
            remaining = int(lookback)
            # Binance limit is 1000; fetch min(1000, remaining)
            limit = min(1000, remaining)
            dfb = _fetch_binance_klines(asset, interval, limit=limit)
            df_result = _coerce_cols(dfb)
            if len(df_result) > remaining:
                df_result = df_result.iloc[-remaining:].reset_index(drop=True)
        elif chosen == "yahoo":
            dfy = _fetch_yahoo(asset, interval, int(lookback))
            df_result = _coerce_cols(dfy)
        elif chosen == "av":
            api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
            dfav = _fetch_alpha_vantage(asset, interval, int(lookback), api_key)
            df_result = _coerce_cols(dfav)
        else:
            raise RuntimeError(f"Unknown source chosen: {chosen}")
    except Exception as e:
        fetch_error = e

    # If fetch failed but cache has any rows for this asset+interval, return that as best-effort
    if (df_result is None or df_result.empty) and os.path.exists(cache_path):
        try:
            df_cache_all = pd.read_csv(cache_path, parse_dates=["Date"], infer_datetime_format=True)
            df_cache = df_cache_all[
                (df_cache_all.get("Asset", "") == asset) & (df_cache_all.get("Interval", "") == interval)
            ].copy()
            if not df_cache.empty:
                df_cache = _coerce_cols(df_cache)
                out = df_cache.copy().reset_index(drop=True)
                out["Source"] = df_cache.get("Source", chosen)
                out["Asset"] = asset
                return out[["Date", "Open", "High", "Low", "Close", "Volume", "Source", "Asset"]]
        except Exception:
            pass

    if fetch_error:
        # bubble up fetch error if nothing useful in cache
        raise fetch_error

    # attach metadata and save to cache (single-file)
    df_result["Source"] = chosen
    df_result["Asset"] = asset
    df_result["Interval"] = interval

    # Save cache even if shorter than lookback (replaces previous rows for this asset+interval)
    try:
        _ensure_min_rows_and_write(df_result, cache_path, int(lookback), asset=asset, interval=interval)
    except Exception:
        pass

    # Trim to requested lookback and return
    if len(df_result) > int(lookback):
        final = df_result.iloc[-int(lookback):].reset_index(drop=True)
    else:
        final = df_result.reset_index(drop=True)

    return final[["Date", "Open", "High", "Low", "Close", "Volume", "Source", "Asset"]]
# features.py