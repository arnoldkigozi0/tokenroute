"""Answer cache — the cheapest token is the one never spent.

Pattern ported from Inquiline's memory layer (Phase 3): normalize the
prompt, look up a previous answer, return it for zero tokens and zero
latency. Persists to a JSON file so cache survives restarts; safe to
run without a path (in-memory only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def normalize(prompt: str) -> str:
    """Canonical form: casefold, collapse whitespace, strip edge punctuation."""
    text = prompt.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n.!?")
    return text


class AnswerCache:
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self._store: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._store = {}  # corrupt cache is not fatal; start clean

    def get(self, prompt: str) -> str | None:
        entry = self._store.get(normalize(prompt))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry["answer"]

    def put(self, prompt: str, answer: str, route: str = "?") -> None:
        if not answer.strip():
            return  # never cache empty answers
        self._store[normalize(prompt)] = {"answer": answer, "route": route}
        if self.path:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._store, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)  # atomic-ish: no torn cache file on crash

    def __len__(self) -> int:
        return len(self._store)
