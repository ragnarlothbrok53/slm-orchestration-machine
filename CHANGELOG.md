# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.1.0] - 2026-10-09

First tagged release. It replaces the initial prototype: a FastAPI endpoint that started a
Temporal workflow which echoed its input, with no model, tools or tests.

### Added
- Durable execution engine (`engine.py`, `store.py`): journaled steps in SQLite (WAL,
  `synchronous=FULL` by default), replay on resume, claim-by-lease with heartbeats and fencing
  tokens, cancellation, graceful hand-back on shutdown, a cap on claims for poison runs, and
  detection of workflow code that changed under an in-flight run.
- Retry policies with classified errors, jittered exponential backoff and `Retry-After` as a
  floor.
- Agent workflow (`agent.py`): a bounded model → parse → tool loop with repair prompts, a
  repeat-call guard and a forced answer on loops or on the last step.
- Two prompt protocols: generic JSON (with prefill) and Qwen2.5's native tool-call format
  (matching its GGUF chat template).
- Lenient parsing of small-model output: code fences, Python dict literals, `arguments` vs `args`,
  answers cut off before their closing.
- Tools: a typed registry (`@tool` with a pydantic argument model), a safe AST-based calculator,
  `utc_now`, and `idempotency_key()` for side-effecting tools.
- Streaming client for slm-runtime / OpenAI-compatible `/v1/completions`: maps
  503/429/504/5xx/4xx, in-band errors and truncated streams to retry decisions; client-side
  concurrency cap; `X-Request-ID = run.seq.attempt` propagation.
- HTTP API: `POST /v1/runs` (idempotent `run_id`, optional `?wait=true`), `GET /v1/runs/{id}`
  with live step progress, `POST /v1/runs/{id}/cancel`, `/healthz`, `/readyz`, `/metrics`.
- CLI: `sof serve` (API and worker in one process), `sof api`, `sof worker`, `sof run`.
- Prometheus metrics for runs, steps, retries, replayed steps, LLM TTFT/tokens/errors, agent
  outcomes and tool calls. Structured logging in the same format as slm-runtime.
- Tests (124): unit tests, engine semantics, API contract, contract tests against the real
  slm-runtime v0.1.0 server, and crash recovery with a killed worker process. The tests were
  checked by mutation.
- Benchmarks with committed results: journal overhead, crash recovery, overload/backpressure
  against slm-runtime, and task success with Qwen2.5-0.5B.
- Dockerfile (non-root, one image for all roles), docker-compose with slm-runtime built from its
  `v0.1.0` tag, `.env.example`, GitHub Actions CI (lint, format, mypy strict, tests, compose
  smoke test).

### Removed
- Temporal and its dependency, the single-container nginx + supervisord + Temporal dev server
  setup (which also downloaded an arm64-only Temporal binary), the echo workflow, and the
  duplicated `Readme.md`.

### Fixed (relative to the prototype)
- A duplicate workflow ID returned `500` with the raw exception text; resubmitting a `run_id` is
  now idempotent, and reusing one with a different input returns `409`.
- `docker compose up` failed without a `.env` file, and the compose bind mount targeted a
  directory the image didn't use.
