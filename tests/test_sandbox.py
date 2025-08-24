from pathlib import Path
import sys
import time
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sandbox import Sandbox, Limits
import headers


def test_sandbox_timeout():
    box = Sandbox(Limits(timeout=0.5))

    def sleepy():
        time.sleep(1)

    with pytest.raises(TimeoutError):
        box.try_forward(sleepy)


def test_sandbox_memory_cap():
    box = Sandbox(Limits(cpu_mem=10 * 1024 * 1024, timeout=2.0))

    def eater():
        _ = bytearray(20 * 1024 * 1024)
        return "done"

    with pytest.raises((RuntimeError, TimeoutError)):
        box.try_forward(eater)


def test_sandbox_cache_success(tmp_path):
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"hello")
    box = Sandbox()
    counter = artifact.with_suffix(".cnt")

    def worker(path: Path, counter_path: Path) -> int:
        with open(counter_path, "a") as fh:
            fh.write("x")
        return 10

    res, t_ms, eng = box.try_forward(worker, artifact, counter)
    assert res == 10
    assert eng == "worker"
    assert t_ms >= 0
    assert counter.read_text() == "x"
    res2, _, eng2 = box.try_forward(worker, artifact, counter)
    assert res2 == 10
    assert eng2 == "worker"
    assert counter.read_text() == "x"


def test_sandbox_cache_error(tmp_path):
    artifact = tmp_path / "b.bin"
    artifact.write_bytes(b"hello")
    box = Sandbox()
    counter = artifact.with_suffix(".err")

    def boom(path: Path, counter_path: Path) -> None:
        with open(counter_path, "a") as fh:
            fh.write("x")
        raise ValueError("boom")

    with pytest.raises(RuntimeError):
        box.try_forward(boom, artifact, counter)
    assert counter.read_text() == "x"
    with pytest.raises(RuntimeError):
        box.try_forward(boom, artifact, counter)
    assert counter.read_text() == "x"


def test_sandbox_disk_cache(tmp_path):
    artifact = tmp_path / "c.bin"
    artifact.write_bytes(b"hello")
    counter = artifact.with_suffix(".cnt")
    cache_dir = tmp_path / "cache"

    def worker(path: Path, counter_path: Path) -> int:
        with open(counter_path, "a") as fh:
            fh.write("x")
        return 5

    box = Sandbox(cache_dir=cache_dir)
    assert box.try_forward(worker, artifact, counter)[0] == 5
    assert counter.read_text() == "x"

    box2 = Sandbox(cache_dir=cache_dir)
    assert box2.try_forward(worker, artifact, counter)[0] == 5
    assert counter.read_text() == "x"


def test_sandbox_class_key_cache(tmp_path):
    artifact1 = tmp_path / "d1.bin"
    artifact1.write_bytes(b"one")
    artifact2 = tmp_path / "d2.bin"
    artifact2.write_bytes(b"two")
    box = Sandbox()

    meta = headers.Meta(tensors=[headers.TensorInfo("w", (1,))], hints={})
    key = headers.class_key(meta)

    def worker(path: Path) -> int:
        cnt = path.with_suffix(".cnt")
        with open(cnt, "a") as fh:
            fh.write("x")
        return 7

    assert box.try_forward(worker, artifact1, class_key=key)[0] == 7
    assert artifact1.with_suffix(".cnt").read_text() == "x"
    assert box.try_forward(worker, artifact2, class_key=key)[0] == 7
    assert not artifact2.with_suffix(".cnt").exists()


def test_validate_min_run_detects_nan():
    box = Sandbox()

    def bad():
        return float("nan")

    metrics = box.validate_min_run(bad)
    assert metrics["stable"] is False
    assert metrics["dtype_ok"] is True


class _DTypeFlip:
    def __init__(self) -> None:
        self.first = True

    def __call__(self):
        if self.first:
            self.first = False
            return FakeTensor(1.0, "f32")
        return FakeTensor(1.0, "f16")


class FakeTensor:
    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype

    def __eq__(self, other):  # type: ignore[override]
        if isinstance(other, FakeTensor):
            return self.data == other.data
        return NotImplemented


def test_validate_min_run_dtype_mismatch():
    box = Sandbox()
    fn = _DTypeFlip()
    metrics = box.validate_min_run(fn)
    assert metrics["dtype_ok"] is False
    assert metrics["stable"] is True


def test_validate_min_run_unstable():
    box = Sandbox()

    class Counter:
        def __init__(self) -> None:
            self.val = 0

        def __call__(self) -> int:
            self.val += 1
            return self.val

    fn = Counter()
    metrics = box.validate_min_run(fn)
    assert metrics["stable"] is False
