import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_agent_single_task_dry_run():
    proc = run([sys.executable, "-m", "tokenroute.agent", "--task", "What is 2+2?", "--dry-run"])
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout.strip())
    assert record["route"] == "local"
    assert record["billable_tokens"] == 0
    assert record["answer"]


def test_agent_tasks_file_dry_run():
    proc = run(
        [sys.executable, "-m", "tokenroute.agent", "--tasks-file", "eval/tasks.sample.jsonl", "--dry-run"]
    )
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(l) for l in proc.stdout.strip().splitlines()]
    assert len(lines) == 5
    routes = {r["id"]: r["route"] for r in lines}
    assert routes["hard-1"] == "remote"
    assert routes["arith-1"] == "local"
    assert all(r["billable_tokens"] == 0 for r in lines if r["route"] == "local")


def test_eval_harness_dry_run_reports_summary():
    proc = run([sys.executable, "eval/run_eval.py", "eval/tasks.sample.jsonl", "--dry-run"])
    summary = json.loads(proc.stdout.strip().splitlines()[-1])
    assert summary["total"] == 5
    assert "accuracy" in summary and "billable_tokens" in summary
