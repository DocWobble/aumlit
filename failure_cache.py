from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Mapping
from filelock import FileLock


class FailureCache:
    """Store failed probe signatures per header hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, str]] = {}
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
        except Exception:
            self.data = {}

    def get(self, header_hash: str) -> Dict[str, str]:
        """Return mapping of probe signatures to error classes."""
        return dict(self.data.get(header_hash, {}))

    def record(self, header_hash: str, probe_sig: str, error_cls: str) -> None:
        """Record ``probe_sig`` with ``error_cls`` under ``header_hash``."""
        bucket = self.data.setdefault(header_hash, {})
        bucket[probe_sig] = error_cls
        self._save()

    def clear(self, header_hash: str | None = None) -> None:
        """Clear cache entries."""
        if header_hash is None:
            self.data.clear()
        else:
            self.data.pop(header_hash, None)
        self._save()

    def inspect(self, header_hash: str | None = None) -> Mapping[str, Dict[str, str]]:
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
