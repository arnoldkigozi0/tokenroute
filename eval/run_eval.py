"""Local eval harness: accuracy + billable tokens over a JSONL task set.

The hackathon scores token count and output accuracy on a standardized
environment; this harness lets us measure both locally before submitting.

Task line format (JSONL):
    {"id": "t1", "prompt": "...", "expected": "...", "match": "contains"}

match modes: exact | contains | numeric (default: contains).

Usage:
    python eval/run_eval.py eval/tasks.sample.jsonl [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenroute.agent import build_router, run_task  # noqa: E402


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_last_number(text: str) -> float | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def is_correct(answer: str, expected: str, match: str) -> bool:
    if match == "exact":
        return normalize(answer) == normalize(expected)
    if match == "numeric":
        got = extract_last_number(answer)
        want = extract_last_number(expected)
        return got is not None and want is not None and abs(got - want) < 1e-6
    return normalize(expected) in normalize(answer)  # contains


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--remote-model", default=None)
    parser.add_argument("--escalate-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    from tokenroute.agent import DEFAULT_LOCAL, DEFAULT_REMOTE

    router = build_router(
        argparse.Namespace(
            dry_run=args.dry_run,
            local_model=args.local_model or DEFAULT_LOCAL,
            remote_model=args.remote_model or DEFAULT_REMOTE,
            escalate_threshold=args.escalate_threshold if args.escalate_threshold is not None else 0.75,
        )
    )

    total = correct = billable = 0
    rows = []
    with open(args.tasks_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            record = run_task(router, str(task["id"]), task["prompt"], task.get("system"))
            ok = is_correct(record["answer"], task.get("expected", ""), task.get("match", "contains"))
            total += 1
            correct += ok
            billable += record["billable_tokens"]
            rows.append((record["id"], "PASS" if ok else "FAIL", record["route"], record["billable_tokens"]))

    width = max((len(r[0]) for r in rows), default=2)
    for task_id, status, route, tokens in rows:
        print(f"{task_id:<{width}}  {status}  {route:<14} billable={tokens}")
    accuracy = correct / total if total else 0.0
    print(f"\naccuracy: {correct}/{total} = {accuracy:.1%}   billable tokens: {billable}")
    print(json.dumps({"accuracy": accuracy, "correct": correct, "total": total, "billable_tokens": billable}))
    return 0 if total and correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
