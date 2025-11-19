# run_experiment.py
"""
Run a simple end-to-end experiment for one asset.
Usage:
    python run_experiment.py --asset BTCUSDT --source auto --interval 1m --lookback 1000 --out results.json

This script:
 - fetches price series via data.get_price_series
 - builds lag features via features.build_features_from_df
 - trains RandomForest and XGBoost (if available)
 - evaluates using evaluate.py metrics on a single holdout (last 20% rows)
 - optionally calls HF LLM (if HUGGINGFACEHUB_API_TOKEN present) to get a single-step forecast
 - prints and saves results as JSON and a small plot (results_plot.png)
"""
import argparse
import json
import os
import sys
import math
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data
import features
import classical_models as cm
import evaluate
import models  # HF-only module

def split_train_test(X, y, test_frac=0.2):
    n = X.shape[0]
    split = int(math.floor(n * (1 - test_frac)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test

def run(asset="BTCUSDT", source="auto", interval="1m", lookback=1000, n_lags=12, use_extras=True, out_path="results.json", hf_model="gpt2"):
    results = {"asset": asset, "source": source, "models": {}, "notes": []}
    try:
        print(f"Fetching data for {asset}...")
        df = data.get_price_series(asset, source_preference=(None if source=="auto" else source), interval=interval, lookback=lookback)
        if df is None or df.empty:
            raise RuntimeError("No data returned.")
        print(f"Rows fetched: {len(df)}")
        X, y, feat_names, enriched = features.build_features_from_df(df, n_lags=n_lags, use_extra_cols=use_extras)
        if X.shape[0] == 0:
            raise RuntimeError("No feature rows produced.")
        # Train/test split
        X_train, X_test, y_train, y_test = split_train_test(X.values, y, test_frac=0.2)
        # RF
        print("Training RandomForest...")
        rf = cm.train_rf(X_train, y_train)
        yhat_rf = rf.predict(X_test)
        results["models"]["RF"] = {
            "mae": evaluate.mae(y_test, yhat_rf),
            "rmse": evaluate.rmse(y_test, yhat_rf),
            "mape": evaluate.mape(y_test, yhat_rf),
            "directional_accuracy": evaluate.directional_accuracy(y_test, yhat_rf)
        }
        # XGB if available
        try:
            print("Training XGBoost...")
            xgbm = cm.train_xgb(X_train, y_train)
            yhat_xgb = xgbm.predict(X_test)
            results["models"]["XGB"] = {
                "mae": evaluate.mae(y_test, yhat_xgb),
                "rmse": evaluate.rmse(y_test, yhat_xgb),
                "mape": evaluate.mape(y_test, yhat_xgb),
                "directional_accuracy": evaluate.directional_accuracy(y_test, yhat_xgb)
            }
        except Exception as e:
            results["notes"].append(f"XGB skipped: {e}")

        # Naive predictor (last-value)
        yhat_naive = np.array([float(X_test[i,0]) for i in range(X_test.shape[0])])  # lag_1 as naive predictor
        results["models"]["NaiveLag1"] = {
            "mae": evaluate.mae(y_test, yhat_naive),
            "rmse": evaluate.rmse(y_test, yhat_naive),
            "mape": evaluate.mape(y_test, yhat_naive),
            "directional_accuracy": evaluate.directional_accuracy(y_test, yhat_naive)
        }

        # ARIMA (attempt)
        try:
            # ARIMA expects a univariate series: use the last part of original series
            arima_series = df["Close"].astype(float).values
            arima_fit = cm.train_arima(arima_series, order=(5,1,0))
            # produce dynamic forecast for test length
            preds = arima_fit.predict(start=len(arima_series)-len(y_test), end=len(arima_series)-1 + len(y_test))
            preds = np.asarray(preds)[:len(y_test)]
            results["models"]["ARIMA"] = {
                "mae": evaluate.mae(y_test, preds),
                "rmse": evaluate.rmse(y_test, preds),
                "mape": evaluate.mape(y_test, preds),
                "directional_accuracy": evaluate.directional_accuracy(y_test, preds)
            }
        except Exception as e:
            results["notes"].append(f"ARIMA skipped: {e}")

        # LLM forecast (one-step) — only if HF token available
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            try:
                llm_val = models.get_llm_forecast_from_df(df, n_lags=n_lags, engine="hf_inference", hf_model=hf_model, hf_token=hf_token)
                # compare llm_val to first test y value (one-step horizon)
                y_true_first = float(y_test[0]) if len(y_test)>0 else None
                results["models"]["LLM_HF_inference"] = {"pred_first": float(llm_val), "first_truth": y_true_first}
            except Exception as e:
                results["notes"].append(f"LLM HF inference failed: {e}")
        else:
            results["notes"].append("HuggingFace token not found in env; skipping HF LLM forecast.")

        # Save results
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Plot actual vs RF & XGB (if available)
        try:
            plt.figure(figsize=(10,5))
            # define a time axis for the test period
            test_idx_start = len(X_train)
            times = pd.to_datetime(df["Date"].iloc[n_lags + test_idx_start : n_lags + test_idx_start + len(y_test)])
            plt.plot(times, y_test, label="Actual", linewidth=2)
            plt.plot(times, yhat_rf, label="RandomForest")
            if "XGB" in results["models"]:
                plt.plot(times, yhat_xgb, label="XGBoost")
            plt.legend()
            plt.title(f"{asset} - Test predictions")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.tight_layout()
            plt.savefig("results_plot.png")
            print("Saved plot results_plot.png")
        except Exception as e:
            print("Plotting failed:", e)

        print("Experiment finished. Results saved to", out_path)
    except Exception as e:
        print("Experiment failed:", e)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTCUSDT")
    parser.add_argument("--source", type=str, default="auto", choices=["auto","binance","yahoo","av"])
    parser.add_argument("--interval", type=str, default="1m")
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--n_lags", type=int, default=12)
    parser.add_argument("--out", type=str, default="results.json")
    parser.add_argument("--hf_model", type=str, default="gpt2")
    args = parser.parse_args()
    run(asset=args.asset, source=args.source, interval=args.interval, lookback=args.lookback, n_lags=args.n_lags, out_path=args.out, hf_model=args.hf_model)
