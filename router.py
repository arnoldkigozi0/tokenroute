"""Token-efficient routing: answer locally when possible, escalate when not.

Track 1 scoring counts only remote tokens, so the dominant strategy is
local-first: try the free local model, self-check the draft, and spend
remote tokens only when the check fails or the task looks too hard to
bother trying locally at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .backends import BackendError, Completion

# Cues that a task probably exceeds a small local model.
HARD_PATTERNS = [
    r"\bprove\b",
    r"\bderive\b",
    r"\btheorem\b",
    r"\brefactor\b",
    r"\bmulti[- ]step\b",
    r"step[- ]by[- ]step",
    r"```",  # embedded code blocks
    r"\bregex\b",
    r"\bsql\b",
]


def complexity_score(prompt: str) -> float:
    """Cheap 0..1 estimate of task difficulty. No model calls."""
    score = 0.0
    words = prompt.split()
    if len(words) > 150:
        score += 0.3
    elif len(words) > 60:
        score += 0.15
    lowered = prompt.lower()
    hits = sum(1 for pat in HARD_PATTERNS if re.search(pat, lowered))
    # one cue alone is weak evidence; three or more means skip local
    score += min(0.75, hits * 0.25)
    if prompt.count("\n") > 10:
        score += 0.2
    return min(1.0, score)


def looks_sane(draft: str, prompt: str) -> bool:
    """Zero-cost sanity check on a local draft before accepting it."""
    text = draft.strip()
    if not text:
        return False
    if len(text) < 2:
        return False
    lowered = text.lower()
    refusals = ("i cannot", "i can't", "as an ai", "i'm sorry", "i am sorry", "i don't know")
    if any(lowered.startswith(r) for r in refusals):
        return False
    # Degenerate repetition (a common small-model failure mode).
    tokens = lowered.split()
    if len(tokens) >= 12 and len(set(tokens)) < max(3, len(tokens) // 6):
        return False
    return True


SELF_CHECK_PROMPT = """You are a strict critic. A question and a proposed answer follow.
Question: {prompt}

Proposed answer: {draft}

Is the proposed answer correct, relevant and complete? Reply with exactly one word: YES or NO."""


@dataclass
class RouteResult:
    answer: str
    route: str  # "cache" | "local" | "remote" | "local->remote"
    local_tokens: int = 0
    remote_tokens: int = 0
    attempts: list = field(default_factory=list)

    @property
    def billable_tokens(self) -> int:
        """Tokens that count toward the Track 1 score: remote only."""
        return self.remote_tokens


class Router:
    """cache -> heuristic pre-screen -> local draft -> verify -> escalate.

    escalate_threshold: complexity above this skips the local attempt
    entirely (saves wall-clock, not tokens — local tokens are free, but a
    doomed local attempt still costs latency in the scored environment).

    self_check: additionally have the LOCAL model critique its own draft
    (Inquiline Critic pattern). Local tokens are free, so this buys
    accuracy at zero scored cost.
    """

    def __init__(self, local, remote, escalate_threshold: float = 0.75, verifier=looks_sane,
                 max_tokens: int = 512, cache=None, self_check: bool = False):
        self.local = local
        self.remote = remote
        self.escalate_threshold = escalate_threshold
        self.verifier = verifier
        self.max_tokens = max_tokens
        self.cache = cache
        self.self_check = self_check

    def _record(self, result: RouteResult, completion: Completion, is_local: bool) -> None:
        result.attempts.append(completion.backend)
        if is_local:
            result.local_tokens += completion.total_tokens
        else:
            result.remote_tokens += completion.total_tokens

    def _draft_passes(self, draft_text: str, prompt: str, result: RouteResult) -> bool:
        if not self.verifier(draft_text, prompt):
            return False
        if not self.self_check:
            return True
        try:
            check = self.local.complete(
                SELF_CHECK_PROMPT.format(prompt=prompt, draft=draft_text), max_tokens=8
            )
            self._record(result, check, is_local=True)
            return check.text.strip().upper().startswith("YES")
        except BackendError:
            return True  # critic unavailable: fall back to heuristic verdict

    def answer(self, prompt: str, system: str | None = None) -> RouteResult:
        if self.cache is not None:
            cached = self.cache.get(prompt)
            if cached is not None:
                return RouteResult(answer=cached, route="cache")

        result = RouteResult(answer="", route="local")
        skip_local = complexity_score(prompt) >= self.escalate_threshold

        if not skip_local:
            try:
                draft = self.local.complete(prompt, system=system, max_tokens=self.max_tokens)
                self._record(result, draft, is_local=True)
                if self._draft_passes(draft.text, prompt, result):
                    result.answer = draft.text
                    result.route = "local"
                    if self.cache is not None:
                        self.cache.put(prompt, result.answer, result.route)
                    return result
            except BackendError:
                pass  # fall through to remote

        final = self.remote.complete(prompt, system=system, max_tokens=self.max_tokens)
        self._record(result, final, is_local=False)
        result.answer = final.text
        result.route = "remote" if skip_local else "local->remote"
        if self.cache is not None:
            self.cache.put(prompt, result.answer, result.route)
        return result
