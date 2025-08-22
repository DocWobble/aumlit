"""Resource-limited sandbox for engine probes.

This module exposes a tiny :class:`Sandbox` helper used by the probing
loop.  Each ``try_forward`` call runs the supplied engine helper in an
isolated subprocess with memory caps and a timeout.  The helper returns
the value produced by the engine or raises :class:`RuntimeError` on
failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, Callable, Mapping
import os
import resource


@dataclass
class Limits:
    """Execution limits for the sandbox."""

    timeout: float = 2.0
    """Maximum wall time for a probe in seconds."""

    cpu_mem: int | None = None
    """Optional cap on address space usage (bytes)."""

    vram: int | None = None
    """Optional cap on GPU memory (bytes)."""


class Sandbox:
    """Run engine helpers inside an isolated subprocess."""

    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()

    # Child process -------------------------------------------------
    def _worker(self, fn: Callable[..., Any], q: Queue, *args: Any, **kwargs: Any) -> None:
        """Invoke ``fn`` and communicate its result via ``q``."""
        if self.limits.cpu_mem:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (self.limits.cpu_mem, self.limits.cpu_mem))
            except Exception:  # pragma: no cover - platform may lack rlimit
                pass
        if self.limits.vram:
            mb = self.limits.vram // (1024 * 1024)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{mb}"
        try:
            q.put(("ok", fn(*args, **kwargs)))
        except Exception as e:  # pragma: no cover - error path
            q.put(("err", repr(e)))

    # Public API ----------------------------------------------------
    def try_forward(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` under resource limits.

        Parameters
        ----------
        fn:
            Engine-specific ``try_forward`` helper.
        *args, **kwargs:
            Passed directly to ``fn``.
        """
        q: Queue = Queue()
        p = Process(target=self._worker, args=(fn, q, *args), kwargs=kwargs)
        p.start()
        p.join(self.limits.timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("sandbox timed out")
        if q.empty():
            raise RuntimeError("sandbox produced no result")
        status, payload = q.get()
        if status == "ok":
            return payload
        raise RuntimeError(payload)

    def validate_min_run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Mapping[str, float]:
        """Run ``fn`` once more to collect timing and VRAM metrics."""

        q: Queue = Queue()

        def worker(q: Queue) -> None:
            if self.limits.cpu_mem:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (self.limits.cpu_mem, self.limits.cpu_mem))
                except Exception:  # pragma: no cover - platform may lack rlimit
                    pass
            if self.limits.vram:
                mb = self.limits.vram // (1024 * 1024)
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{mb}"

            import time
            start = time.perf_counter()
            try:
                import torch
                if torch.cuda.is_available():  # pragma: no cover - depends on torch
                    torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover - optional dependency
                pass

            try:
                fn(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000.0
                vram = 0.0
                try:
                    import torch
                    if torch.cuda.is_available():  # pragma: no cover - depends on torch
                        vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                except Exception:  # pragma: no cover - optional dependency
                    pass
                q.put(("ok", {"time_ms": duration, "vram_mb": vram}))
            except Exception as e:  # pragma: no cover - error path
                q.put(("err", repr(e)))

        p = Process(target=worker, args=(q,))
        p.start()
        p.join(self.limits.timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("sandbox timed out")
        if q.empty():
            raise RuntimeError("sandbox produced no result")
        status, payload = q.get()
        if status == "ok":
            return payload
        raise RuntimeError(payload)


# Factory -----------------------------------------------------------
def spawn_sandbox(limits: Mapping[str, int | float] | None = None) -> Sandbox:
    """Create a :class:`Sandbox` with ``limits``."""

    if limits is None:
        return Sandbox()
    return Sandbox(Limits(**limits))
