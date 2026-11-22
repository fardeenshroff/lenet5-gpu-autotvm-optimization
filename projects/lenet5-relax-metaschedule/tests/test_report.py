import json
import pytest
from relax_meta.report import summarize_ms, calculate_speedup, save_json


def test_summary():
    r = summarize_ms([1.0, 2.0, 3.0])
    assert r["median_ms"] == 2.0
    assert r["samples"] == 3
    assert r["mean_ms"] == 2.0


def test_speedup():
    assert calculate_speedup(5, 2) == 2.5
    with pytest.raises(ValueError):
        calculate_speedup(2, 0)


def test_invalid_samples():
    with pytest.raises(ValueError):
        summarize_ms([])
    with pytest.raises(ValueError):
        summarize_ms([-1])


def test_json_save(tmp_path):
    p = tmp_path / "a" / "b.json"
    save_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
