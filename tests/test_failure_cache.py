from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from failure_cache import FailureCache


def test_failure_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "f.json"
    fc = FailureCache(cache_file)
    fc.record("h", "sig", "ERR")
    assert fc.get("h") == {"sig": "ERR"}
    fc2 = FailureCache(cache_file)
    assert fc2.get("h") == {"sig": "ERR"}
    fc.clear("h")
    assert fc.get("h") == {}


def test_failure_cache_global_ops(tmp_path):
    cache_file = tmp_path / "f.json"
    fc = FailureCache(cache_file)
    fc.record("h1", "sig1", "ERR1")
    fc.record("h2", "sig2", "ERR2")

    # inspect-all returns the entire dataset
    assert fc.inspect() == {
        "h1": {"sig1": "ERR1"},
        "h2": {"sig2": "ERR2"},
    }

    # clear-all removes every entry
    fc.clear()
    assert fc.inspect() == {}
