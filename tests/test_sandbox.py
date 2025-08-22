from pathlib import Path
import sys
import time
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sandbox import Sandbox, Limits


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
