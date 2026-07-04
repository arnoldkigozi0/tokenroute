# tokenroute

**A token-efficient hybrid routing agent** — answers with a free local model whenever it can, and spends remote tokens only when it must.

Built solo by team **Inquiline** (Uganda) for the **AMD Developer Hackathon: ACT II — Track 1: Hybrid Token-Efficient Routing Agent**.

## The idea

Track 1 scores two things: **output accuracy** and **token count**, where *local tokens count as zero*. That makes the winning strategy routing intelligence, not raw compute:

1. **Cache first.** A repeated (or trivially re-worded) task is answered from the cache: zero tokens, zero latency (`--use-cache`).
2. **Pre-screen** each task with a zero-cost complexity heuristic. Tasks that clearly exceed the local model skip straight to the remote endpoint (saves latency, not tokens — a doomed local attempt is free but slow).
3. **Local first.** Everything else is drafted by the local model. Local tokens are free.
4. **Verify the draft** — zero-cost sanity checks (empty output, refusals, degenerate repetition), plus optionally the local model critiquing its own draft (`--self-check`): accuracy bought entirely with free tokens.
5. **Escalate only on failure.** Remote (Fireworks AI) is called only when the draft fails verification — those are the only tokens that ever hit the bill.

This mirrors the router at the heart of [Inquiline](https://github.com/arnoldkigozi0/inquiline), a local-first agent built to run on zero-budget infrastructure — the same constraint this track rewards.

## Layout

```
tokenroute/
  backends.py   # OllamaBackend (local, free), FireworksBackend (remote, billed), StaticBackend (tests)
  router.py     # cache → complexity pre-screen → local draft → verify/self-check → escalate
  cache.py      # normalized-prompt answer cache, JSON-persistent (Inquiline memory pattern)
  brains.py     # Ollama model lifecycle: one brain at a time on tight VRAM (Inquiline pattern)
  agent.py      # CLI: single task or JSONL batch, emits JSON per task
eval/
  run_eval.py           # accuracy + billable-token report over a task set
  tasks.sample.jsonl    # sample tasks until the real ones are revealed at kickoff
tests/          # pytest suite, runs fully offline
```

The routing core mirrors the architecture of [Inquiline](https://github.com/arnoldkigozi0/inquiline)
(private), a local-first AI agent: its Router→Coder→Critic squad, answer-memory
layer, and one-brain-at-a-time GPU discipline are re-implemented here as
standalone, dependency-free modules.

## Run it

```bash
# no network, static backends — proves the plumbing:
python -m tokenroute.agent --task "What is 2+2?" --dry-run

# real backends:
export FIREWORKS_API_KEY=...
export LOCAL_MODEL=qwen2.5:3b                  # any Ollama tag
export REMOTE_MODEL=accounts/fireworks/models/...   # set when models are revealed
python -m tokenroute.agent --tasks-file eval/tasks.sample.jsonl --use-cache --self-check
```

Each task emits one JSON line:

```json
{"id": "arith-1", "answer": "391", "route": "local", "local_tokens": 34, "remote_tokens": 0, "billable_tokens": 0}
```

## Evaluate before submitting

```bash
python eval/run_eval.py eval/tasks.sample.jsonl            # real backends
python eval/run_eval.py eval/tasks.sample.jsonl --dry-run  # plumbing check
```

Reports per-task pass/fail, the route taken, and total billable tokens — the two numbers the leaderboard scores.

## Docker

Submissions must be containerized:

```bash
docker build -t tokenroute .
docker run --rm --network host \
  -e FIREWORKS_API_KEY -e LOCAL_MODEL -e REMOTE_MODEL \
  tokenroute --tasks-file eval/tasks.sample.jsonl
```

## Tests

```bash
python -m pytest -q
```

Zero runtime dependencies — Python 3.10+ standard library only. `pytest` is needed only to run the test suite.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_MODEL` | `qwen2.5:3b` | Ollama model tag (free tokens) |
| `REMOTE_MODEL` | *set at kickoff* | Fireworks model id (billed tokens) |
| `OLLAMA_URL` | `http://localhost:11434` | local server |
| `FIREWORKS_API_KEY` | — | required for remote calls |
| `ESCALATE_THRESHOLD` | `0.75` | complexity ≥ this skips the local draft |
| `CACHE_FILE` | — | persist the answer cache to this JSON file |
