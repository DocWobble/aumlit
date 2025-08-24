from pathlib import Path
import sys
import threading
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from failure_cache import FailureCache


def test_failure_cache_roundtrip(tmp_path):
    cache_file = tmp_path / "f.json"
    fc = FailureCache(cache_file)
    fc.record("h", "sig", "ERR", ints={"a": 1}, engine="torch", op_kind="mm", time_ms=1.2)
    expected = {
        "sig": {
            "error_cls": "ERR",
            "ints": {"a": 1},
            "engine": "torch",
            "op_kind": "mm",
            "time_ms": 1.2,
        }
    }
    assert fc.get("h") == expected
    fc2 = FailureCache(cache_file)
    assert fc2.get("h") == expected
    fc.clear("h")
    assert fc.get("h") == {}


def test_failure_cache_global_ops(tmp_path):
    cache_file = tmp_path / "f.json"
    fc = FailureCache(cache_file)
    fc.record("h1", "sig1", "ERR1", engine="e1")
    fc.record("h2", "sig2", "ERR2", engine="e2")

    # inspect-all returns the entire dataset
    assert fc.inspect() == {
        "h1": {
            "sig1": {
                "error_cls": "ERR1",
                "ints": {},
                "engine": "e1",
                "op_kind": None,
                "time_ms": None,
            }
        },
        "h2": {
            "sig2": {
                "error_cls": "ERR2",
                "ints": {},
                "engine": "e2",
                "op_kind": None,
                "time_ms": None,
            }
        },
    }

    # clear-all removes every entry
    fc.clear()
    assert fc.inspect() == {}


def test_failure_cache_concurrent_writes(tmp_path):
    cache_file = tmp_path / "f.json"

    def worker(idx: int) -> None:
        fc = FailureCache(cache_file)
        fc.record(f"h{idx}", f"sig{idx}", "ERR")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # File should remain valid JSON despite concurrent writes
    data = json.loads(cache_file.read_text())
    assert isinstance(data, dict)
