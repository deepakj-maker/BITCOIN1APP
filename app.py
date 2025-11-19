"""
Refined Streamlit UI (multi-asset) — RandomForest vs XGBoost predictor.
Place next to data.py, features.py, models.py (models.py must implement train_compare_lstm_gru that now trains RF/XGB).
Run: streamlit run app.py
"""
from typing import Optional
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import data
import features
import models  # models.py should implement train_compare_lstm_gru (now RF + XGB)

# --- helper to robustly extract a single close price series ---
def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if "Close" in df.columns:
        ser = df["Close"]
        if isinstance(ser, pd.Series):
            return ser.astype(float)
    if isinstance(df.columns, pd.MultiIndex):
        for col in df.columns:
            try:
                if any(str(part).lower() == "close" for part in col):
                    ser = df[col]
                    if isinstance(ser, pd.Series):
                        return ser.astype(float)
                    elif isinstance(ser, pd.DataFrame) and not ser.empty:
                        num_cols = ser.select_dtypes(include=[np.number]).columns
                        if len(num_cols) > 0:
                            return ser[num_cols[-1]].astype(float)
                        return ser.iloc[:, -1].astype(float)
            except Exception:
                continue
    for col in df.columns:
        try:
            if str(col).lower() == "close":
                ser = df[col]
                if isinstance(ser, pd.Series):
                    return ser.astype(float)
                elif isinstance(ser, pd.DataFrame) and not ser.empty:
                    num_cols = ser.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        return ser[num_cols[-1]].astype(float)
                    return ser.iloc[:, -1].astype(float)
        except Exception:
            continue
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        ser = df[numeric_cols[-1]]
        if isinstance(ser, pd.Series):
            return ser.astype(float)
        else:
            return ser.iloc[:, -1].astype(float)
    raise ValueError("No close or numeric column found in DataFrame")

# Page config
st.set_page_config(page_title="Multi-Asset — RF vs XGBoost", layout="wide", initial_sidebar_state="expanded")

# Header
st.title("Multi-Asset — RandomForest vs XGBoost Predictor")
st.markdown(
    "Use the left panel to select data & features. Click **Fetch** to load prices, **Build features** to create inputs, "
    "and **Predict** to train RandomForest and XGBoost and compare them using standard metrics."
)

# Layout
controls_col, output_col = st.columns([1, 2])

with controls_col:
    st.header("Controls")

    # Data form
    with st.expander("1) Data — select asset & source", expanded=True):
        with st.form(key="data_form"):
            asset = st.text_input("Asset ticker (ex: BTCUSDT, AAPL, SPY)", value="BTCUSDT")
            source_pref = st.selectbox(
                "Preferred source", options=["auto", "binance", "yahoo", "av"], index=0,
                help="Choose 'auto' for automatic selection, 'binance' for crypto realtime, 'yahoo' for equities/ETFs."
            )
            interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)
            lookback = st.number_input("Rows to fetch (lookback)", min_value=50, max_value=10000, value=1000, step=50)
            fetch_btn = st.form_submit_button("Fetch price data")

    # Features form
    with st.expander("2) Features — how to build inputs", expanded=False):
        with st.form(key="features_form"):
            n_lags = st.slider("Lag features (n)", 1, 240, 60)
            use_extras = st.checkbox("Use TA extras (technical indicators) when building features", value=True)
            build_btn = st.form_submit_button("Build features")

    # Prediction form (RandomForest / XGBoost)
    with st.expander("3) Predictor settings (RandomForest vs XGBoost)", expanded=True):
        with st.form(key="predict_form"):
            rf_n_estimators = st.number_input("RandomForest n_estimators", min_value=10, max_value=2000, value=100, step=10)
            xgb_n_estimators = st.number_input("XGBoost n_estimators", min_value=10, max_value=2000, value=100, step=10)
            random_seed = st.number_input("Random seed", min_value=0, max_value=2**31-1, value=42, step=1)
            retrain_btn = st.form_submit_button("Predict (Train & Compare RF/XGBoost)")

    st.markdown("---")
    if st.button("Clear cached preview"):
        st.session_state.pop("last_df", None)
        st.session_state.pop("last_features", None)
        st.session_state.pop("feat_names", None)
        st.success("Preview cleared — fetch again to repopulate.")

with output_col:
    status = st.empty()
    chart_area = st.container()
    table_area = st.expander("Data & features preview", expanded=True)
    result_area = st.expander("Prediction / Diagnostics", expanded=True)

    # session state defaults
    if "last_df" not in st.session_state:
        st.session_state["last_df"] = None
    if "last_features" not in st.session_state:
        st.session_state["last_features"] = None
    if "feat_names" not in st.session_state:
        st.session_state["feat_names"] = []

    # Fetch handling
    if fetch_btn:
        src = None if source_pref == "auto" else source_pref
        with status.container():
            st.info(f"Fetching {asset} from {src or 'auto'} — interval={interval}, lookback={lookback}...")
            try:
                df_src = data.get_price_series(asset, source_preference=src, interval=interval, lookback=lookback)
                if df_src is None or df_src.empty:
                    status.error(f"No data returned for {asset}. Check ticker / source and try again.")
                else:
                    st.session_state["last_df"] = df_src
                    status.success(f"Loaded {len(df_src)} rows for {asset}.")
                    with table_area:
                        st.subheader("Price data (tail)")
                        st.dataframe(df_src.tail(10))
                    # chart
                    with chart_area:
                        st.subheader("Price chart")
                        try:
                            import plotly.express as px
                            close_series = _extract_close_series(df_src)
                            if "Date" in df_src.columns:
                                times = pd.to_datetime(df_src["Date"])
                                x_vals = times
                            elif isinstance(close_series.index, pd.DatetimeIndex):
                                x_vals = close_series.index
                            else:
                                x_vals = pd.RangeIndex(len(close_series))
                            fig = px.line(x=x_vals, y=close_series.values, labels={"x": "Date", "y": "Close"}, title=f"{asset} Close")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            try:
                                numeric_cols = df_src.select_dtypes(include=[np.number]).columns
                                if "Date" in df_src.columns and len(numeric_cols) > 0:
                                    st.line_chart(df_src.set_index("Date")[numeric_cols[-1]].astype(float))
                                elif len(numeric_cols) > 0:
                                    st.line_chart(df_src[numeric_cols[-1]].astype(float))
                                else:
                                    st.write("Unable to render chart for this data frame.")
                            except Exception:
                                st.write("Unable to render chart for this data frame.")
            except Exception as e:
                st.session_state["last_df"] = None
                status.error(f"Failed to fetch: {e}")

    # Build features (robust with fallback)
    if build_btn:
        df_use = st.session_state.get("last_df")
        if df_use is None or df_use.empty:
            status.error("No data available — fetch first.")
        else:
            with status.container():
                st.info("Building features — this may take a few seconds...")
                try:
                    X, y, feat_names, enriched = features.build_features_from_df(df_use, n_lags=n_lags, use_extra_cols=use_extras)
                    st.session_state["feat_names"] = feat_names
                    st.session_state["last_features"] = (X, y, feat_names, enriched)
                    status.success(f"Features built: {X.shape[0]} rows x {X.shape[1]} cols")
                    with table_area:
                        st.subheader("Feature matrix (tail)")
                        st.dataframe(pd.DataFrame(X).tail(5))
                except Exception as primary_exc:
                    # fallback to single Close series
                    try:
                        close_ser = _extract_close_series(df_use)
                        if "Date" in df_use.columns:
                            dates = pd.to_datetime(df_use["Date"]).reset_index(drop=True)
                            df_min = pd.DataFrame({"Date": dates, "Close": close_ser.reset_index(drop=True)})
                        else:
                            df_min = pd.DataFrame({"Date": pd.RangeIndex(len(close_ser)), "Close": close_ser.reset_index(drop=True)})
                        X, y, feat_names, enriched = features.build_features_from_df(df_min, n_lags=n_lags, use_extra_cols=use_extras)
                        st.session_state["feat_names"] = feat_names
                        st.session_state["last_features"] = (X, y, feat_names, enriched)
                        status.success(f"Features built (fallback): {X.shape[0]} rows x {X.shape[1]} cols")
                        with table_area:
                            st.subheader("Feature matrix (tail) — fallback")
                            st.dataframe(pd.DataFrame(X).tail(5))
                    except Exception as fallback_exc:
                        status.error("Feature build failed: see debug info below.")
                        with st.expander("Feature build debug info", expanded=True):
                            st.write("Primary exception (first attempt):")
                            st.exception(primary_exc)
                            st.write("---")
                            st.write("Fallback exception (using single Close series):")
                            st.exception(fallback_exc)
                            st.write("---")
                            st.write("Original dataframe shape:", getattr(df_use, "shape", None))
                            st.write("Original dataframe columns:", list(df_use.columns))
                            try:
                                st.write("Original dataframe dtypes:", df_use.dtypes.to_dict())
                            except Exception:
                                st.write("Could not show dtypes")
                            try:
                                st.dataframe(df_use.head(10))
                            except Exception:
                                st.write("Could not render dataframe head")
                            try:
                                cs = _extract_close_series(df_use)
                                st.write("close_series.shape:", cs.shape)
                                st.write(list(cs.head().astype(float).values))
                            except Exception as e:
                                st.write("Could not extract Close series:", e)

    # Predict / retrain handling
    if retrain_btn:
        df_src = st.session_state.get("last_df")
        if df_src is None or df_src.empty:
            status.error("No data available to train on. Fetch first.")
        else:
            with st.spinner("Training RandomForest and XGBoost — this may take a while depending on data size..."):
                try:
                    # call the training/comparison function in models.py
                    res = models.train_compare_lstm_gru(
                        df=df_src,
                        n_lags=int(n_lags),
                        test_size=0.2,
                        rf_n_estimators=int(rf_n_estimators),
                        xgb_n_estimators=int(xgb_n_estimators),
                        random_seed=int(random_seed),
                        verbose=0
                    )
                except Exception as e:
                    status.error("Training failed — see exception below.")
                    with result_area:
                        st.exception(e)
                    res = None

            if res:
                status.success("Training completed — results ready.")
                metrics = res.get("metrics", {})
                preds = res.get("preds", {})
                scaler = res.get("scaler", None)
                data_splits = res.get("data_splits", {})
                histories = res.get("histories", {})

                # Display summary metrics
                with result_area:
                    st.subheader("Paper-style Comparison Table")

                    paper_df = res.get("paper_table_df")
                    if isinstance(paper_df, pd.DataFrame):
                        st.table(paper_df)
                    else:
                        paper_str = res.get("paper_table", None)
                        if paper_str:
                            st.text(paper_str)

                    # helper to safely get metric values supporting multiple naming conventions
                    def _get_metric(mdict, names, default=float("nan")):
                        """Return first available key in names from mdict, or default."""
                        if not mdict:
                            return default
                        for n in names:
                            if n in mdict and mdict[n] is not None:
                                return mdict[n]
                        return default

                    # retrieve model metric dicts
                    rf_m = metrics.get("random_forest", {}) or {}
                    xgb_m = metrics.get("xgboost", {}) or {}

                    # build small per-model tables
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**RandomForest**")
                        st.write(pd.DataFrame({
                            "metric": ["MAE (test)", "RMSE (test)", "MAPE (test)", "Direction Acc (test)", "Next prediction"],
                            "value": [
                                _get_metric(rf_m, ["mae_test", "mae"]),
                                _get_metric(rf_m, ["rmse_test", "rmse"]),
                                _get_metric(rf_m, ["mape_test", "mape"]),
                                _get_metric(rf_m, ["direction_acc_test", "direction_acc"]),
                                _get_metric(rf_m, ["next_pred", "next_prediction"])
                            ]
                        }))

                    with col2:
                        st.markdown("**XGBoost**")
                        st.write(pd.DataFrame({
                            "metric": ["MAE (test)", "RMSE (test)", "MAPE (test)", "Direction Acc (test)", "Next prediction"],
                            "value": [
                                _get_metric(xgb_m, ["mae_test", "mae"]),
                                _get_metric(xgb_m, ["rmse_test", "rmse"]),
                                _get_metric(xgb_m, ["mape_test", "mape"]),
                                _get_metric(xgb_m, ["direction_acc_test", "direction_acc"]),
                                _get_metric(xgb_m, ["next_pred", "next_prediction"])
                            ]
                        }))

                    # Show next-step predictions side-by-side (formatted)
                    nd_col1, nd_col2 = st.columns(2)
                    with nd_col1:
                        next_rf = _get_metric(rf_m, ["next_pred", "next_prediction"], default=float("nan"))
                        try:
                            nd_val = f"{float(next_rf):,.2f}"
                        except Exception:
                            nd_val = str(next_rf)
                        st.metric(label="RandomForest — next-step prediction", value=nd_val)
                    with nd_col2:
                        next_xgb = _get_metric(xgb_m, ["next_pred", "next_prediction"], default=float("nan"))
                        try:
                            nd_val2 = f"{float(next_xgb):,.2f}"
                        except Exception:
                            nd_val2 = str(next_xgb)
                        st.metric(label="XGBoost — next-step prediction", value=nd_val2)

                    # Optionally show classification (percent) metrics if available
                    try:
                        cls_rf = rf_m.get("classification") or rf_m.get("classification_metrics")
                        cls_xgb = xgb_m.get("classification") or xgb_m.get("classification_metrics")
                        if cls_rf or cls_xgb:
                            st.markdown("**Direction classification (percent)**")
                            cdf = pd.DataFrame({
                                "metric": ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"],
                                "RandomForest": [
                                    (cls_rf.get("accuracy_pct") if cls_rf else None),
                                    (cls_rf.get("precision_pct") if cls_rf else None),
                                    (cls_rf.get("recall_pct") if cls_rf else None),
                                    (cls_rf.get("f1_pct") if cls_rf else None)
                                ],
                                "XGBoost": [
                                    (cls_xgb.get("accuracy_pct") if cls_xgb else None),
                                    (cls_xgb.get("precision_pct") if cls_xgb else None),
                                    (cls_xgb.get("recall_pct") if cls_xgb else None),
                                    (cls_xgb.get("f1_pct") if cls_xgb else None)
                                ]
                            }).set_index("metric")
                            st.table(cdf.fillna("-"))
                    except Exception:
                        pass

                    # Plot test vs predictions (attempt robust alignment)
                    try:
                        n_lags_used = data_splits.get("n_lags", int(n_lags))
                        train_len = data_splits.get("train_len")
                        test_len = data_splits.get("test_len")

                        # true test series: prefer preds['y_test'] if available
                        y_test_inv = None
                        if isinstance(preds, dict) and "y_test" in preds:
                            y_test_inv = np.asarray(preds.get("y_test"))
                        else:
                            close_series = _extract_close_series(df_src).astype(float).reset_index(drop=True)
                            total_len = len(close_series)
                            if train_len is not None and test_len is not None:
                                start_idx = n_lags_used + train_len
                                end_idx = start_idx + test_len
                                if end_idx <= total_len:
                                    y_test_inv = close_series.iloc[start_idx:end_idx].values
                                else:
                                    L = len(res["preds"]["random_forest"]["test_pred"])
                                    y_test_inv = close_series.iloc[total_len - L: total_len].values
                            else:
                                L = len(res["preds"]["random_forest"]["test_pred"])
                                close_series = _extract_close_series(df_src).astype(float).reset_index(drop=True)
                                total_len = len(close_series)
                                y_test_inv = close_series.iloc[total_len - L: total_len].values

                        test_pred_rf = np.asarray(res["preds"]["random_forest"]["test_pred"]).ravel()
                        test_pred_xgb = np.asarray(res["preds"]["xgboost"]["test_pred"]).ravel()
                        y_test_inv = np.asarray(y_test_inv).ravel()

                        # align lengths: use min length among arrays
                        L = min(len(y_test_inv), len(test_pred_rf), len(test_pred_xgb))
                        if L <= 0:
                            raise ValueError("Not enough points to plot test vs predictions after alignment.")

                        # build index for plotting: try to reconstruct absolute indices if data_splits has train_len
                        if train_len is not None and test_len is not None and n_lags_used is not None:
                            start_idx = int(n_lags_used + train_len)
                            idx = list(range(start_idx, start_idx + L))
                        else:
                            total_len = len(_extract_close_series(df_src))
                            idx = list(range(total_len - L, total_len))

                        df_plot = pd.DataFrame({
                            "index": idx,
                            "actual": y_test_inv[:L],
                            "rf_pred": test_pred_rf[:L],
                            "xgb_pred": test_pred_xgb[:L]
                        })

                        # plot using plotly
                        try:
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_plot["index"], y=df_plot["actual"], mode="lines+markers", name="Actual (test)"))
                            fig.add_trace(go.Scatter(x=df_plot["index"], y=df_plot["rf_pred"], mode="lines+markers", name="RF Pred"))
                            fig.add_trace(go.Scatter(x=df_plot["index"], y=df_plot["xgb_pred"], mode="lines+markers", name="XGB Pred"))
                            fig.update_layout(title="Test set: actual vs predictions (indices)", xaxis_title="index (relative to full series)", yaxis_title="Price")
                            with chart_area:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            df_plot2 = df_plot.set_index("index")
                            with chart_area:
                                st.line_chart(df_plot2)
                    except Exception as e:
                        with result_area:
                            st.write("Could not build test-vs-pred plot. Showing last close instead.")
                            st.exception(e)
                            try:
                                with chart_area:
                                    st.line_chart(_extract_close_series(df_src).tail(200))
                            except Exception:
                                pass

                    # Plot training histories (empty for sklearn models but kept for compatibility)
                    try:
                        hist_rf = histories.get("random_forest", {})
                        hist_xgb = histories.get("xgboost", {})
                        if hist_rf or hist_xgb:
                            st.subheader("Training history")
                            cols = st.columns(2)
                            if hist_rf.get("loss"):
                                with cols[0]:
                                    st.markdown("**RandomForest training**")
                                    st.line_chart(pd.DataFrame({"loss": hist_rf.get("loss"), "val_loss": hist_rf.get("val_loss", [None]*len(hist_rf.get("loss")))}))
                            if hist_xgb.get("loss"):
                                with cols[1]:
                                    st.markdown("**XGBoost training**")
                                    st.line_chart(pd.DataFrame({"loss": hist_xgb.get("loss"), "val_loss": hist_xgb.get("val_loss", [None]*len(hist_xgb.get("loss")))}))
                    except Exception:
                        pass

                    # Expose a debug expander with full results dict and debug_shapes
                    with result_area:
                        with st.expander("Full results (debug)", expanded=False):
                            try:
                                st.write(res)
                                if "debug_shapes" in res:
                                    st.write("Debug shapes:", res["debug_shapes"])
                            except Exception:
                                st.write("Could not display full results.")

# Footer
st.markdown("---")
st.caption(
    "Hints: For equities/ETFs use Yahoo (SPY, VTI). For crypto real-time use Binance symbols (BTCUSDT).\n"
    "Ensure you have scikit-learn and xgboost installed in the environment (models.train_compare_lstm_gru uses them)."
)
