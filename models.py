# models.py
"""
Robust LSTM vs GRU trainer and comparison utilities for the Streamlit app.
Replace previous train_compare_lstm_gru with this file.
"""

import math
import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models as kmodels
from tensorflow.keras import layers as klayers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


def _build_rnn_model(rnn_type: str, input_shape, units: int = 64, dropout: float = 0.0) -> kmodels.Model:
    if rnn_type not in ("lstm", "gru"):
        raise ValueError("rnn_type must be 'lstm' or 'gru'")
    model = kmodels.Sequential()
    model.add(klayers.Input(shape=input_shape))
    if rnn_type == "lstm":
        model.add(klayers.LSTM(units, activation="tanh"))
    else:
        model.add(klayers.GRU(units, activation="tanh"))
    if dropout and dropout > 0:
        model.add(klayers.Dropout(dropout))
    model.add(klayers.Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    return model


def format_paper_table_df(lstm_m: Dict[str, Any], gru_m: Dict[str, Any], transformer_m: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Return a DataFrame with rows R2, MAE, MSE, RMSE and columns LSTM, GRU, Transformer.
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
        "LSTM": [_get_val(r, lstm_m) for r in rows],
        "GRU": [_get_val(r, gru_m) for r in rows],
        "Transformer": [_get_val(r, transformer_m) for r in rows]
    }).set_index("Metric")
    return df


# -------------------------
# Main exported function
# -------------------------
def train_compare_lstm_gru(
    df: pd.DataFrame,
    n_lags: int = 60,
    test_size: float = 0.2,
    lstm_units: int = 64,
    gru_units: int = 64,
    dropout: float = 0.0,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    verbose: int = 0,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train LSTM and GRU on df['Close'] and compare. Returns a dict with:
      - metrics: per-model dicts (regression + classification)
      - preds: test/train/next predictions
      - models: keras model objects
      - scaler: MinMaxScaler instance
      - data_splits: info
      - histories: training histories
      - paper_table (string) and paper_table_df (DataFrame)
      - debug_shapes (helpful for diagnosing alignment)
    """
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    if df is None or "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain 'Close' column.")

    # --- Prepare series ---
    series = pd.to_numeric(df["Close"], errors="coerce").dropna().values.flatten()
    if series.size == 0:
        raise ValueError("Close series is empty after coercion/dropping NA.")

    scaler = MinMaxScaler((0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Try to use features.build_features_from_df if provided (must match expected output)
    # But for safety we use univariate sequences from close price
    def _create_sequences(series_arr: np.ndarray, n_lags_local: int):
        X, y = [], []
        for i in range(len(series_arr) - n_lags_local):
            X.append(series_arr[i:i + n_lags_local])
            y.append(series_arr[i + n_lags_local])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.size == 0:
            return np.empty((0, n_lags_local, 1)), np.empty((0,))
        return X.reshape((X.shape[0], X.shape[1], 1)), y

    X, y = _create_sequences(series_scaled, int(n_lags))
    N = X.shape[0]
    if N <= 0:
        raise ValueError("Not enough data to create sequences with the chosen n_lags.")

    test_len = max(1, int(math.ceil(N * float(test_size))))
    train_len = N - test_len
    if train_len <= 0:
        raise ValueError("Train set length is zero. Reduce test_size or increase data length.")

    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]

    prev_train_scaled = X_train[:, -1, 0] if X_train.shape[0] > 0 else np.array([])
    prev_test_scaled = X_test[:, -1, 0] if X_test.shape[0] > 0 else np.array([])

    # --- Build models ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = _build_rnn_model("lstm", input_shape, units=int(lstm_units), dropout=float(dropout))
    gru_model = _build_rnn_model("gru", input_shape, units=int(gru_units), dropout=float(dropout))

    es = EarlyStopping(monitor="val_loss", patience=int(patience), restore_best_weights=True, verbose=0)

    # --- Fit models ---
    history_lstm = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[es] if patience > 0 else [],
        verbose=int(verbose),
        shuffle=False
    )
    history_gru = gru_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[es] if patience > 0 else [],
        verbose=int(verbose),
        shuffle=False
    )

    # --- Predictions (scaled) ---
    train_pred_lstm_s = lstm_model.predict(X_train, batch_size=batch_size).ravel() if X_train.shape[0] > 0 else np.array([])
    test_pred_lstm_s = lstm_model.predict(X_test, batch_size=batch_size).ravel() if X_test.shape[0] > 0 else np.array([])
    train_pred_gru_s = gru_model.predict(X_train, batch_size=batch_size).ravel() if X_train.shape[0] > 0 else np.array([])
    test_pred_gru_s = gru_model.predict(X_test, batch_size=batch_size).ravel() if X_test.shape[0] > 0 else np.array([])

    def inv_scale(arr_scaled):
        arr = np.asarray(arr_scaled).reshape(-1, 1)
        if arr.size == 0:
            return np.array([])
        return scaler.inverse_transform(arr).flatten()

    y_train_inv = inv_scale(y_train)
    y_test_inv = inv_scale(y_test)
    train_pred_lstm = inv_scale(train_pred_lstm_s)
    test_pred_lstm = inv_scale(test_pred_lstm_s)
    train_pred_gru = inv_scale(train_pred_gru_s)
    test_pred_gru = inv_scale(test_pred_gru_s)

    # next-step prediction using last n_lags from scaled series
    last_seq = series_scaled[-int(n_lags):].reshape((1, int(n_lags), 1))
    next_pred_lstm_s = float(lstm_model.predict(last_seq).ravel()[0])
    next_pred_gru_s = float(gru_model.predict(last_seq).ravel()[0])
    next_pred_lstm = float(inv_scale([next_pred_lstm_s])[0])
    next_pred_gru = float(inv_scale([next_pred_gru_s])[0])

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

    lstm_metrics = _assemble(y_test_inv, test_pred_lstm, prev_test_real, next_pred_lstm)
    gru_metrics = _assemble(y_test_inv, test_pred_gru, prev_test_real, next_pred_gru)

    metrics = {"lstm": lstm_metrics, "gru": gru_metrics}

    # Prepare paper-style table (DataFrame + string)
    paper_df = format_paper_table_df(lstm_metrics, gru_metrics, transformer_m=None)
    # pretty string (simple markdown-like)
    paper_table_str = paper_df.to_string()

    results = {
        "metrics": metrics,
        "preds": {
            "lstm": {"train_pred": train_pred_lstm, "test_pred": test_pred_lstm, "next_pred": next_pred_lstm},
            "gru": {"train_pred": train_pred_gru, "test_pred": test_pred_gru, "next_pred": next_pred_gru},
            "y_test": y_test_inv
        },
        "models": {"lstm": lstm_model, "gru": gru_model},
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
        "histories": {"lstm": history_lstm.history, "gru": history_gru.history},
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
            "test_pred_lstm_shape": tuple(test_pred_lstm.shape),
            "test_pred_gru_shape": tuple(test_pred_gru.shape),
            "y_test_inv_shape": tuple(np.asarray(y_test_inv).shape)
        }
    }

    return results
