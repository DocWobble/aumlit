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
