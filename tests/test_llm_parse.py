# tests/test_llm_parse.py
from models import parse_llm_json_response

def test_parse_valid_json():
    txt = '{"preds":[12345.67]}'
    out = parse_llm_json_response(txt)
    assert out == [12345.67]

def test_parse_embedded_json():
    txt = "Here is my answer:\n{\"preds\": [9000.5]}\nThanks"
    out = parse_llm_json_response(txt)
    assert out == [9000.5]

def test_parse_plain_numbers():
    txt = "9000.5"
    out = parse_llm_json_response(txt)
    assert out == [9000.5]

def test_parse_multiple_numbers():
    txt = "I think preds: [100.0, 101.0]"
    out = parse_llm_json_response(txt)
    assert out == [100.0, 101.0]
