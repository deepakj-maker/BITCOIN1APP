"""
RandomForest vs XGBoost trainer and comparison utilities for the Streamlit app.
Replaces previous LSTM/GRU TensorFlow implementation with scikit-learn RandomForestRegressor
and XGBoost XGBRegressor. Function name `train_compare_lstm_gru` is kept for backward
compatibility with the rest of the app but now trains RF and XGB models.
"""

import math
import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional XGBoost - fall back to sklearn's GradientBoostingRegressor if not installed
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    _HAS_XGB = False

# Try to use user's features module if present
try:
    import features
except Exception:
    features = None

warnings.filterwarnings("ignore")


# -------------------------
# small helpers
# -------------------------
def _to_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _rmse(y_true, y_pred) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    if yt.size != yp.size:
        raise ValueError(f"_rmse: length mismatch y_true({yt.size}) vs y_pred({yp.size})")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _mape(y_true, y_pred) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    if yt.size == 0 or yt.size != yp.size:
        return float("nan")
    eps = 1e-8
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)


def _r2_score(y_true, y_pred) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    if yt.size == 0:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _direction_accuracy(y_true, y_pred, prev_values) -> float:
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    pv = _to_float_array(prev_values)
    L = min(yt.size, yp.size, pv.size)
    if L <= 0:
        return float("nan")
    true_dir = np.sign(yt[:L] - pv[:L])
    pred_dir = np.sign(yp[:L] - pv[:L])
    true_bin = np.where(true_dir > 0, 1, 0)
    pred_bin = np.where(pred_dir > 0, 1, 0)
    return float((true_bin == pred_bin).sum()) / float(L)


def _binary_direction_metrics_pct(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Convert regression true/pred arrays to up/down labels using forward difference,
    compute classification metrics: accuracy, precision, recall, f1 in percent.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    if yt.size < 2 or yp.size < 2:
        return {"accuracy_pct": float("nan"), "precision_pct": float("nan"),
                "recall_pct": float("nan"), "f1_pct": float("nan"), "confusion": None}
    true_dir = np.where(np.diff(yt) > 0, 1, 0)
    pred_dir = np.where(np.diff(yp) > 0, 1, 0)
    L = min(len(true_dir), len(pred_dir))
    if L <= 0:
        return {"accuracy_pct": float("nan"), "precision_pct": float("nan"),
                "recall_pct": float("nan"), "f1_pct": float("nan"), "confusion": None}
    t = true_dir[:L]
    p = pred_dir[:L]
    acc = accuracy_score(t, p)
    prec = precision_score(t, p, zero_division=0)
    rec = recall_score(t, p, zero_division=0)
    f1 = f1_score(t, p, zero_division=0)
    cm = confusion_matrix(t, p).tolist()
    return {
        "accuracy_pct": round(float(acc * 100.0), 3),
        "precision_pct": round(float(prec * 100.0), 3),
        "recall_pct": round(float(rec * 100.0), 3),
        "f1_pct": round(float(f1 * 100.0), 3),
        "confusion": cm
    }


def format_paper_table_df(rf_m: Dict[str, Any], xgb_m: Dict[str, Any], transformer_m: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Return a DataFrame with rows R2, MAE, MSE, RMSE and columns RandomForest, XGBoost, Transformer.
    Values formatted as floats with 6 decimals or '-' if missing.
    """
    rows = ["R2", "MAE", "MSE", "RMSE"]
    key_map = {
        "R2": ["r2", "r2_test", "r2_score"],
        "MAE": ["mae", "mae_test"],
        "MSE": ["mse", "mse_test"],
        "RMSE": ["rmse", "rmse_test"]
    }

    def _get_val(metric_name, mdict):
        if not mdict:
            return "-"
        for k in key_map[metric_name]:
            if k in mdict and mdict[k] is not None:
                try:
                    return f"{float(mdict[k]):.6f}"
                except Exception:
                    return str(mdict[k])
        return "-"

    transformer_m = transformer_m or {}
    df = pd.DataFrame({
        "Metric": rows,
        "RandomForest": [_get_val(r, rf_m) for r in rows],
        "XGBoost": [_get_val(r, xgb_m) for r in rows],
        "Transformer": [_get_val(r, transformer_m) for r in rows]
    }).set_index("Metric")
    return df


# -------------------------
# Main exported function (keeps original name for compatibility)
# -------------------------
def train_compare_lstm_gru(
    df: pd.DataFrame,
    n_lags: int = 60,
    test_size: float = 0.2,
    rf_n_estimators: int = 100,
    xgb_n_estimators: int = 100,
    random_seed: int = 42,
    verbose: int = 0
) -> Dict[str, Any]:
    """
    Train RandomForest and XGBoost on df['Close'] and compare. Returns a dict with:
      - metrics: per-model dicts (regression + classification)
      - preds: test/train/next predictions
      - models: trained model objects
      - scaler: MinMaxScaler instance
      - data_splits: info
      - histories: empty (kept for compatibility)
      - paper_table and paper_table_df
      - debug_shapes
    """
    np.random.seed(random_seed)

    if df is None or "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain 'Close' column.")

    # --- Prepare series ---
    series = pd.to_numeric(df["Close"], errors="coerce").dropna().values.flatten()
    if series.size == 0:
        raise ValueError("Close series is empty after coercion/dropping NA.")

    scaler = MinMaxScaler((0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # create tabular sequences (n_samples, n_lags)
    def _create_sequences_tabular(series_arr: np.ndarray, n_lags_local: int):
        X, y = [], []
        for i in range(len(series_arr) - n_lags_local):
            X.append(series_arr[i:i + n_lags_local])
            y.append(series_arr[i + n_lags_local])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        return X, y

    X, y = _create_sequences_tabular(series_scaled, int(n_lags))
    N = X.shape[0]
    if N <= 0:
        raise ValueError("Not enough data to create sequences with the chosen n_lags.")

    test_len = max(1, int(math.ceil(N * float(test_size))))
    train_len = N - test_len
    if train_len <= 0:
        raise ValueError("Train set length is zero. Reduce test_size or increase data length.")

    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    prev_train_scaled = X_train[:, -1] if X_train.shape[0] > 0 else np.array([])
    prev_test_scaled = X_test[:, -1] if X_test.shape[0] > 0 else np.array([])

    # --- Build models ---
    rf_model = RandomForestRegressor(n_estimators=int(rf_n_estimators), random_state=int(random_seed), n_jobs=-1)
    xgb_model = XGBRegressor(n_estimators=int(xgb_n_estimators), random_state=int(random_seed), verbosity=0)

    # --- Fit models ---
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # --- Predictions (scaled) ---
    train_pred_rf_s = rf_model.predict(X_train) if X_train.shape[0] > 0 else np.array([])
    test_pred_rf_s = rf_model.predict(X_test) if X_test.shape[0] > 0 else np.array([])
    train_pred_xgb_s = xgb_model.predict(X_train) if X_train.shape[0] > 0 else np.array([])
    test_pred_xgb_s = xgb_model.predict(X_test) if X_test.shape[0] > 0 else np.array([])

    def inv_scale(arr_scaled):
        arr = np.asarray(arr_scaled).reshape(-1, 1)
        if arr.size == 0:
            return np.array([])
        return scaler.inverse_transform(arr).flatten()

    y_train_inv = inv_scale(y_train)
    y_test_inv = inv_scale(y_test)
    train_pred_rf = inv_scale(train_pred_rf_s)
    test_pred_rf = inv_scale(test_pred_rf_s)
    train_pred_xgb = inv_scale(train_pred_xgb_s)
    test_pred_xgb = inv_scale(test_pred_xgb_s)

    # next-step prediction using last n_lags from scaled series
    last_seq = series_scaled[-int(n_lags):].reshape(1, -1)
    next_pred_rf_s = float(rf_model.predict(last_seq).ravel()[0])
    next_pred_xgb_s = float(xgb_model.predict(last_seq).ravel()[0])
    next_pred_rf = float(inv_scale([next_pred_rf_s])[0])
    next_pred_xgb = float(inv_scale([next_pred_xgb_s])[0])

    prev_test_real = inv_scale(prev_test_scaled)
    prev_train_real = inv_scale(prev_train_scaled)

    # --- Metrics assembly (regression + classification) ---
    def _assemble(y_true_inv, y_pred_inv, prev_inv, next_pred_val):
        yt = _to_float_array(y_true_inv)
        yp = _to_float_array(y_pred_inv)
        out = {}
        out["mae_test"] = float(mean_absolute_error(yt, yp)) if yt.size == yp.size and yt.size > 0 else float("nan")
        out["mse_test"] = float(mean_squared_error(yt, yp)) if yt.size == yp.size and yt.size > 0 else float("nan")
        try:
            out["rmse_test"] = float(_rmse(yt, yp))
        except Exception:
            out["rmse_test"] = float("nan")
        out["mape_test"] = float(_mape(yt, yp)) if yt.size == yp.size and yt.size > 0 else float("nan")
        out["r2"] = float(_r2_score(yt, yp)) if yt.size == yp.size and yt.size > 0 else float("nan")
        try:
            out["direction_acc_test"] = float(_direction_accuracy(yt, yp, prev_inv))
        except Exception:
            out["direction_acc_test"] = float("nan")
        out["next_pred"] = float(next_pred_val) if next_pred_val is not None else float("nan")

        # short keys for compatibility
        out["mae"] = out["mae_test"]
        out["mse"] = out["mse_test"]
        out["rmse"] = out["rmse_test"]
        out["mape"] = out["mape_test"]
        out["r2_test"] = out["r2"]

        # classification metrics based on up/down (percent)
        cls = _binary_direction_metrics_pct(y_true_inv, y_pred_inv)
        out["classification"] = cls

        return out

    rf_metrics = _assemble(y_test_inv, test_pred_rf, prev_test_real, next_pred_rf)
    xgb_metrics = _assemble(y_test_inv, test_pred_xgb, prev_test_real, next_pred_xgb)

    metrics = {"random_forest": rf_metrics, "xgboost": xgb_metrics}

    # Prepare paper-style table (DataFrame + string)
    paper_df = format_paper_table_df(rf_metrics, xgb_metrics, transformer_m=None)
    paper_table_str = paper_df.to_string()

    results = {
        "metrics": metrics,
        "preds": {
            "random_forest": {"train_pred": train_pred_rf, "test_pred": test_pred_rf, "next_pred": next_pred_rf},
            "xgboost": {"train_pred": train_pred_xgb, "test_pred": test_pred_xgb, "next_pred": next_pred_xgb},
            "y_test": y_test_inv
        },
        "models": {"random_forest": rf_model, "xgboost": xgb_model},
        "scaler": scaler,
        "data_splits": {
            "X_train_shape": X_train.shape,
            "X_test_shape": X_test.shape,
            "y_train_shape": y_train.shape,
            "y_test_shape": y_test.shape,
            "train_len": train_len,
            "test_len": test_len,
            "n_lags": int(n_lags)
        },
        # scikit-learn models do not provide epoch histories; kept for compatibility
        "histories": {"random_forest": {}, "xgboost": {}},
        "paper_table": paper_table_str,
        "paper_table_df": paper_df,
        "debug_shapes": {
            "series_len": int(len(series)),
            "N_sequences": int(N),
            "train_len": int(train_len),
            "test_len": int(test_len),
            "X_train_shape": tuple(X_train.shape),
            "X_test_shape": tuple(X_test.shape),
            "y_test_shape": tuple(y_test.shape),
            "test_pred_rf_shape": tuple(np.asarray(test_pred_rf).shape),
            "test_pred_xgb_shape": tuple(np.asarray(test_pred_xgb).shape),
            "y_test_inv_shape": tuple(np.asarray(y_test_inv).shape)
        }
    }

    return results
