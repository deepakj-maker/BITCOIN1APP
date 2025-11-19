# tests/test_llm_prompt.py
from models import craft_llm_prompt

def test_craft_llm_prompt_structure():
    prices = [100.0, 101.0, 100.5, 102.0]
    prompt = craft_llm_prompt(prices, extras={"rsi_est": 55.0}, predict_horizon=1, method="pct")
    assert "Prototype" in prompt
    assert "Return JSON only" in prompt
    assert "rsi_est=55.0" in prompt or "rsi_est=55" in prompt
