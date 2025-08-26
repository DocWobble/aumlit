from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Mapping
from filelock import FileLock


class FailureCache:
    """Store failed probe signatures per header hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # ``data`` maps header hashes to probe signatures and failure details
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text())
                # Coerce legacy formats to the new structure
                for hh, bucket in raw.items():
                    self.data[hh] = {}
                    for sig, info in bucket.items():
                        if isinstance(info, dict):
                            self.data[hh][sig] = info
                        else:  # legacy ``error_cls`` string
                            self.data[hh][sig] = {
                                "error_cls": info,
                                "ints": {},
                                "engine": None,
                                "op_kind": None,
                                "time_ms": None,
                            }
        except Exception:
            self.data = {}

    def get(self, header_hash: str) -> Dict[str, Dict[str, Any]]:
        """Return mapping of probe signatures to failure details."""
        return dict(self.data.get(header_hash, {}))

    def record(
        self,
        header_hash: str,
        probe_sig: str,
        error_cls: str,
        ints: Mapping[str, Any] | None = None,
        engine: str | None = None,
        op_kind: str | None = None,
        time_ms: float | None = None,
    ) -> None:
        """Record ``probe_sig`` with metadata under ``header_hash``."""
        bucket = self.data.setdefault(header_hash, {})
        bucket[probe_sig] = {
            "error_cls": error_cls,
            "ints": dict(ints or {}),
            "engine": engine,
            "op_kind": op_kind,
            "time_ms": time_ms,
        }
        self._save()

    def clear(self, header_hash: str | None = None) -> None:
        """Clear cache entries."""
        if header_hash is None:
            self.data.clear()
        else:
            self.data.pop(header_hash, None)
        self._save()

    def inspect(self, header_hash: str | None = None) -> Mapping[str, Dict[str, Dict[str, Any]]]:
        """Return current cache state or a single header entry."""
        if header_hash is None:
            return dict(self.data)
        return {header_hash: self.get(header_hash)}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(self.path) + ".lock")
            with lock:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data))
                tmp.replace(self.path)
        except Exception:
            pass


__all__ = ["FailureCache"]
