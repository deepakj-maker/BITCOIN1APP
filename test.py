import json
import pandas as pd
import numpy as np
from features import compute_indicators, build_features_from_df
from models import craft_llm_prompt, parse_llm_json_response


def test_compute_indicators_basic():
    # simple synthetic series
    df = pd.DataFrame({"Close": np.arange(1.0, 21.0)})
    out = compute_indicators(df)
    # shape preserved
    assert "rsi_14" in out.columns
    assert not out["rsi_14"].isnull().any()
    assert "sma_10" in out.columns


def test_build_features_shape():
    df = pd.DataFrame({"Close": np.arange(1.0, 31.0)})
    X, y, feat_names, enriched = build_features_from_df(df, n_lags=5, use_extra_cols=True)
    assert X.shape[0] == len(df) - 5
    assert "lag_1" in X.columns


def test_craft_prompt_and_parse():
    prices = [100.0, 101.0, 100.5, 100.75]
    prompt = craft_llm_prompt(prices, extras={"sma_5": 100.56}, predict_horizon=1)
    assert "Prototype" in prompt
    # create a fake LLM response
    reply = json.dumps({"preds": [101.23]})
    parsed = parse_llm_json_response(reply)
    assert isinstance(parsed, list) and abs(parsed[0] - 101.23) < 1e-6