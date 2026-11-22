import ast
from pathlib import Path
import pytest
import numpy as np
from relax_meta.benchmark import build_parser, timed_forward, assert_close, count_records


def test_args_and_safe_default():
    a = build_parser().parse_args([])
    assert a.batch_sizes == [1, 128]
    assert a.trials > 0


def test_timing_samples_without_cuda():
    events = []
    def fn():
        events.append("call")
    def sync():
        events.append("sync")
    result = timed_forward(fn, sync, warmup=2, repeats=2, number=3)
    assert result["samples"] == 2
    assert events.count("call") == 8
    assert events.count("sync") == 5


def test_numerical_validation():
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    assert assert_close(a, a.copy(), 1e-3, 1e-3)["passed"]
    with pytest.raises(AssertionError):
        assert_close(a, np.ones((2, 2), dtype=np.float32), 1e-3, 1e-3)


def test_tuning_record_count(tmp_path):
    assert count_records(tmp_path) == 0
    (tmp_path / "database_tuning_record.json").write_text('{"a":1}\n\n{"b":2}\n')
    assert count_records(tmp_path) == 2
