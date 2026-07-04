"""Ollama model lifecycle — one brain at a time.

Pattern ported from Inquiline's brain manager: on constrained hardware
(one GPU, tight VRAM — exactly what a standardized scoring environment
looks like), load a model, use it, unload it before loading the next.
Unloading is done by generating with keep_alive=0, which is what the
`ollama stop` CLI does under the hood.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OllamaManager:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method)
        if body:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def loaded_models(self) -> list[str]:
        try:
            data = self._request("GET", "/api/ps")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []
        return [m.get("name", "") for m in data.get("models", [])]

    def unload(self, model: str) -> bool:
        try:
            self._request("POST", "/api/generate", {"model": model, "keep_alive": 0})
            return True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return False

    def swap_to(self, model: str) -> None:
        """Unload everything that isn't `model`. Call before heavy work."""
        for name in self.loaded_models():
            if name and name != model:
                self.unload(name)
