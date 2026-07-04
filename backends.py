"""Model backends: local (Ollama) and remote (Fireworks AI).

Stdlib only — urllib for HTTP. Every backend returns a Completion with
token counts so the router can account for cost. Local tokens score zero
in Track 1, but we still record them for diagnostics.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    backend: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a completion."""


def _estimate_tokens(text: str) -> int:
    # Fallback when the API returns no usage data: ~4 chars/token.
    return max(1, len(text) // 4)


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise BackendError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BackendError(f"request to {url} failed: {exc}") from exc


class OllamaBackend:
    """Local model served by Ollama. All tokens here are free for scoring."""

    is_local = True

    def __init__(self, model: str, base_url: str | None = None, timeout: float = 120.0):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        if system:
            payload["system"] = system
        data = _post_json(f"{self.base_url}/api/generate", payload, {}, self.timeout)
        text = data.get("response", "")
        return Completion(
            text=text,
            prompt_tokens=data.get("prompt_eval_count") or _estimate_tokens(prompt),
            completion_tokens=data.get("eval_count") or _estimate_tokens(text),
            backend=self.name,
        )


class FireworksBackend:
    """Remote model on Fireworks AI (OpenAI-compatible chat API). Tokens count."""

    is_local = False

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.model = model
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = (base_url or "https://api.fireworks.ai/inference/v1").rstrip("/")
        self.timeout = timeout
        self.name = f"fireworks:{model}"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> Completion:
        if not self.api_key:
            raise BackendError("FIREWORKS_API_KEY is not set")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        data = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise BackendError(f"unexpected response shape: {data}") from exc
        usage = data.get("usage", {})
        return Completion(
            text=text,
            prompt_tokens=usage.get("prompt_tokens") or _estimate_tokens(prompt),
            completion_tokens=usage.get("completion_tokens") or _estimate_tokens(text),
            backend=self.name,
        )


class StaticBackend:
    """Deterministic backend for tests and dry runs — no network."""

    def __init__(self, reply: str = "ok", is_local: bool = True, name: str = "static", fail: bool = False):
        self.reply = reply
        self.is_local = is_local
        self.name = name
        self.fail = fail
        self.calls = 0

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> Completion:
        self.calls += 1
        if self.fail:
            raise BackendError(f"{self.name} is configured to fail")
        return Completion(
            text=self.reply,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(self.reply),
            backend=self.name,
        )
