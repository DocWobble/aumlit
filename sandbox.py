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
from pathlib import Path
from typing import Any, Callable, Mapping
import hashlib
import json
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

    def __init__(self, limits: Limits | None = None, cache_dir: str | Path | None = None) -> None:
        self.limits = limits or Limits()
        self.cache: dict[tuple[str, str], tuple[str, Any]] = {}
        self.cache_file: Path | None = None
        if cache_dir is not None:
            self.cache_file = Path(cache_dir) / "sandbox_cache.json"
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.cache_file.exists():
                    data = json.loads(self.cache_file.read_text())
                    for k, v in data.items():
                        a, b = k.split(":", 1)
                        self.cache[(a, b)] = tuple(v)  # type: ignore[assignment]
            except Exception:  # pragma: no cover - corrupted cache
                pass

    def _save_cache(self) -> None:
        if not self.cache_file:
            return
        data = {f"{k[0]}:{k[1]}": list(v) for k, v in self.cache.items()}
        tmp = self.cache_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(self.cache_file)
        except Exception:  # pragma: no cover - write errors
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

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
    def try_forward(
        self,
        fn: Callable[..., Any],
        *args: Any,
        class_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn`` under resource limits.

        Parameters
        ----------
        fn:
            Engine-specific ``try_forward`` helper.
        *args, **kwargs:
            Passed directly to ``fn``.
        """
        key: tuple[str, str] | None = None
        base_key: str | None = class_key
        if base_key is None and args and isinstance(args[0], (str, os.PathLike, Path)):
            try:
                artifact_path = Path(args[0])
                hasher = hashlib.sha256()
                with artifact_path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        hasher.update(chunk)
                base_key = hasher.hexdigest()
            except Exception:
                base_key = None
        if base_key is not None:
            sig_src = repr((args[1:], sorted(kwargs.items())))
            probe_signature = hashlib.sha256(sig_src.encode()).hexdigest()
            key = (base_key, probe_signature)
            if key in self.cache:
                status, payload = self.cache[key]
                if status == "ok":
                    return payload
                raise RuntimeError(payload)

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
        if key is not None:
            self.cache[key] = (status, payload)
            self._save_cache()
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
def spawn_sandbox(
    limits: Mapping[str, int | float] | None = None, cache_dir: str | Path | None = None
) -> Sandbox:
    """Create a :class:`Sandbox` with ``limits`` and optional cache."""

    if limits is None:
        return Sandbox(cache_dir=cache_dir)
    return Sandbox(Limits(**limits), cache_dir=cache_dir)
