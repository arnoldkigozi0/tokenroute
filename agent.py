"""CLI entry point: route one task or a JSONL file of tasks.

Usage:
    python -m tokenroute.agent --task "What is 2+2?"
    python -m tokenroute.agent --tasks-file eval/tasks.sample.jsonl

Config via environment (all overridable by flags):
    LOCAL_MODEL     Ollama model tag        (default: qwen2.5:3b)
    REMOTE_MODEL    Fireworks model id      (default: placeholder — set at kickoff)
    OLLAMA_URL      Ollama base URL         (default: http://localhost:11434)
    FIREWORKS_API_KEY   required for remote calls
    ESCALATE_THRESHOLD  complexity cutoff 0..1 (default: 0.75)

Emits one JSON object per task on stdout:
    {"id", "answer", "route", "local_tokens", "remote_tokens", "billable_tokens"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import FireworksBackend, OllamaBackend, StaticBackend
from .cache import AnswerCache
from .router import Router

DEFAULT_LOCAL = "qwen2.5:3b"
DEFAULT_REMOTE = "accounts/fireworks/models/CHANGE-ME-AT-KICKOFF"


def build_router(args: argparse.Namespace) -> Router:
    if args.dry_run:
        local = StaticBackend(reply="dry-run local answer", is_local=True, name="static-local")
        remote = StaticBackend(reply="dry-run remote answer", is_local=False, name="static-remote")
    else:
        local = OllamaBackend(args.local_model)
        remote = FireworksBackend(args.remote_model)
    cache = AnswerCache(getattr(args, "cache_file", None)) if getattr(args, "use_cache", False) else None
    return Router(
        local,
        remote,
        escalate_threshold=args.escalate_threshold,
        cache=cache,
        self_check=getattr(args, "self_check", False),
    )


def run_task(router: Router, task_id: str, prompt: str, system: str | None = None) -> dict:
    result = router.answer(prompt, system=system)
    return {
        "id": task_id,
        "answer": result.answer,
        "route": result.route,
        "local_tokens": result.local_tokens,
        "remote_tokens": result.remote_tokens,
        "billable_tokens": result.billable_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenroute", description=__doc__)
    parser.add_argument("--task", help="single prompt to answer")
    parser.add_argument("--tasks-file", help="JSONL file: {id, prompt, [system]} per line")
    parser.add_argument("--local-model", default=os.environ.get("LOCAL_MODEL", DEFAULT_LOCAL))
    parser.add_argument("--remote-model", default=os.environ.get("REMOTE_MODEL", DEFAULT_REMOTE))
    parser.add_argument(
        "--escalate-threshold",
        type=float,
        default=float(os.environ.get("ESCALATE_THRESHOLD", "0.75")),
    )
    parser.add_argument("--dry-run", action="store_true", help="use static backends, no network")
    parser.add_argument("--use-cache", action="store_true", help="reuse answers for repeated prompts (zero tokens)")
    parser.add_argument("--cache-file", default=os.environ.get("CACHE_FILE"), help="persist cache to this JSON file")
    parser.add_argument("--self-check", action="store_true",
                        help="local model critiques its own draft before accepting (free tokens)")
    args = parser.parse_args(argv)

    if not args.task and not args.tasks_file:
        parser.error("provide --task or --tasks-file")

    router = build_router(args)
    if args.task:
        print(json.dumps(run_task(router, "cli", args.task)))
        return 0

    with open(args.tasks_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            record = run_task(router, str(task.get("id", "?")), task["prompt"], task.get("system"))
            print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
