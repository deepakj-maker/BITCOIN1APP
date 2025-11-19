# classical_models.py
"""
Classical ML & statistical model helpers for time-series forecasting.

Provides:
 - train_rf: RandomForestRegressor trained on provided X_train, y_train
 - train_xgb: XGBoostRegressor trained on provided X_train, y_train
 - predict_naive_last: naive predictor returning last observed value (or mean)
 - train_arima: quick ARIMA fit (requires statsmodels), returns fitted model

Note: All training functions are small convenience wrappers intended for experiments.
"""
from typing import Tuple, Optional
import numpy as np

# sklearn / xgboost imports
from sklearn.ensemble import RandomForestRegressor
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ARIMA optional
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_ARIMA = True
except Exception:
    HAVE_ARIMA = False


def train_rf(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200, random_state: int = 42) -> RandomForestRegressor:
    """
    Train and return a RandomForestRegressor.
    X_train: 2D numpy array
    y_train: 1D numpy array
    """
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    m.fit(X_train, y_train)
    return m


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, params: Optional[dict] = None) -> "xgb.XGBRegressor":
    """
    Train and return an XGBoost regressor. Requires xgboost to be installed.
    If not available, raises RuntimeError.
    """
    if not HAVE_XGB:
        raise RuntimeError("XGBoost (xgboost) not installed. Install with `pip install xgboost`.")
    default_params = {"objective": "reg:squarederror", "n_estimators": 200, "random_state": 42, "n_jobs": -1}
    if params:
        default_params.update(params)
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


def predict_naive_last(series: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Naive forecast: repeat last value for horizon steps.
    series: 1D array-like of past values
    Returns numpy array length=horizon
    """
    if len(series) == 0:
        return np.zeros(horizon)
    last = float(series[-1])
    return np.array([last] * horizon)


def train_arima(series: np.ndarray, order: Tuple[int, int, int] = (5, 1, 0)):
    """
    Fit an ARIMA model to the provided univariate series and return the fitted results object.
    Requires statsmodels. If not available, raises RuntimeError.
    """
    if not HAVE_ARIMA:
        raise RuntimeError("statsmodels not installed for ARIMA. Install with `pip install statsmodels`.")
    # statsmodels expects a pandas Series or numpy array
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted
