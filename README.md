# SLOM — Small Language Orchestration Machine

SLOM runs tool-using agents on 0.5–3B models as **durable workflows**: every model call and tool
call is journaled to SQLite, so a crashed worker's runs resume elsewhere without paying for
finished model calls again. There is no orchestration server to run: it's a Python package plus
a SQLite file.

[![CI](https://github.com/ragnarlothbrok53/slm-orchestration-machine/actions/workflows/ci.yml/badge.svg)](https://github.com/ragnarlothbrok53/slm-orchestration-machine/actions/workflows/ci.yml)

SLOM is the orchestration layer above [**SLIM**](https://github.com/ragnarlothbrok53/slm-inference-machine),
a small inference server for local models. SLIM decides *whether* a request runs now
(admission control, deadlines, `503 + Retry-After`) and explicitly leaves retry policy to the
caller. SLOM is that caller.

## Why this exists

A 0.5B model on a CPU takes seconds per call and is unreliable in ways that compound over a
multi-step task:

- **It's wrong in the format, not just the content.** It emits JSON wrapped in prose, Python
  dicts instead of JSON, or repeats the same tool call forever.
- **Its server pushes back.** A busy inference server answers `503`; a model reload takes it
  away for a minute.
- **Agent runs are long and expensive to restart.** A crash at step 5 of 6 shouldn't cost five
  model calls again.

The usual answer to the last point is a workflow engine like Temporal, which is a cluster to run
before your first agent works. SLOM implements the part an agent loop needs (journaled steps,
leases, fenced recovery, classified retries) in about 800 lines, and backs each guarantee with a
test that breaks it on purpose and a measurement.

## Key features

- **Durable execution.** `await ctx.step(name, fn, arg)` journals each result. After a crash,
  finished steps return their recorded result (~0.03 ms) instead of running again. Leases with
  fencing tokens let a crashed or stalled worker's runs move to another worker, safely. Tested by
  killing a real worker process mid-call.
- **Failure-aware LLM client** for slim-serve and other OpenAI-compatible `/v1/completions`
  servers. It maps `503`/`429`/`504`/`5xx`/`4xx`, in-band stream errors and truncated streams to
  retry decisions, with jittered backoff that never undercuts `Retry-After`. A client-side
  concurrency cap matches server capacity. It always streams, so cancelling a run stops
  generation on the server.
- **An agent loop built for small models.** Bounded steps and repairs; a lenient parser that
  never raises on model output; a repeat-call guard; a forced answer when the model loops; two
  prompt protocols (generic JSON with prefill, and Qwen2.5's native tool-call format).
- **Typed tools.** A function plus a pydantic argument model. Bad arguments and tool errors go back
  to the model; unexpected exceptions are retried. Comes with a safe AST calculator and
  `idempotency_key()` for tools with side effects.
- **HTTP API and CLI:**
  - submit (idempotent `run_id`, optional `?wait=true`), inspect with live step progress, cancel;
  - `slom serve` runs the API and a worker in one process, `slom worker` scales out on one host;
  - `slom run "question"` runs a single task in the current process.
- **Observability.** Prometheus metrics that separate live work from replayed steps, structured
  logs, and `X-Request-ID = run.step.attempt` propagated into slim-serve's logs.
- **Evidence.**
  - 124 tests: unit, engine semantics, API, contract tests against the real slim-serve v0.1.0,
    and a process-kill recovery test. They were checked by mutation (9 deliberately broken
    guarantees, all caught).
  - Benchmarks with committed raw results.

## Architecture

```mermaid
flowchart LR
    client([Client]) -->|"POST /v1/runs · GET /v1/runs/id"| api

    subgraph host["one host"]
        api["API<br/>validate · submit · read"]
        db[("SQLite journal (WAL)<br/>runs · steps · leases")]
        subgraph w["worker process × N"]
            loop["claim loop<br/>lease · heartbeat · cancel"]
            wf["agent workflow<br/>render → model → parse → tool"]
            llmc["LlmClient<br/>stream · retry class · cap"]
            tools["tools"]
        end
        api --> db
        loop -->|"claim / heartbeat / finish<br/>(fenced by lease token)"| db
        wf -->|"ctx.step → journal"| db
        loop --> wf --> llmc
        wf --> tools
    end

    llmc -->|"SSE /v1/completions<br/>X-Request-ID"| slm["slim-serve<br/>llama.cpp · MLX"]
    slm -.->|"503 Retry-After · 504 · 400"| llmc
    prom([Prometheus]) -.->|"/metrics"| api
    prom -.->|":9464/metrics"| loop
```

- **Request path.** The API validates a run and inserts it as `pending`; it never executes
  anything. A worker claims the oldest claimable run inside a `BEGIN IMMEDIATE` transaction (so
  two workers can't both claim it), sets a lease and bumps the lease token. It then runs the
  workflow. Each `ctx.step` either returns a journaled result or executes, retrying as its policy
  says, and writes the outcome, fenced by the token, before returning.
- **Interfaces.**
  - A workflow is `async def wf(ctx, input) -> output`. A step function is
    `async def f(arg) -> T` with dataclass or plain types, since results are journaled as JSON.
  - A tool is `@tool def f(args: SomeModel) -> str`.
  - The LLM boundary is an OpenAI-style `/v1/completions` stream.
- **Failure boundaries.**
  - Bad model output, tool failure: the error goes back to the model.
  - Overload or LLM outage: the step retries, then the run fails.
  - Worker crash: other workers take over that worker's runs.
  - A run that keeps killing workers: the run fails after N claims.
  - Changed workflow code: the run fails loudly.
  - Host loss takes everything down: SLOM is single-host by design.
- **State.** Two tables, `runs` and `steps`. Nothing else is persisted.
- **External dependencies.** An LLM server. That's all.

The sequence diagrams (a run, crash recovery), the thread model and the full failure table are in
[docs/architecture.md](docs/architecture.md).

## Technical design

| Decision | Why |
|---|---|
| Own durable engine, not Temporal / DBOS | The guarantees an agent loop needs fit in a journal and leases; no server to run. The concepts map 1:1 if you outgrow it. |
| SQLite (WAL), not Postgres / DuckDB | Multi-process OLTP on one host, in the standard library. DuckDB allows one writing process only. Postgres is the 10x path. |
| Replay by re-execution | The agent stays plain Python. The price is determinism between steps, which is enforced (`NonDeterminismError`). |
| Leases + fencing tokens | A stalled ("zombie") worker can't overwrite the run that a new owner is executing. |
| At-least-once steps | Exactly-once across systems is impossible; tools get an idempotency key instead. |
| Failures are journaled too | Replay takes the same branch the workflow already took. |
| Jittered backoff, `Retry-After` as floor | Measured: without jitter, synchronized retries left an overloaded server idle (4–6× slower batch). |
| Client concurrency cap ≤ server capacity | Measured: oversubscribing costs 2.2–2.3× makespan, even with good retries. |
| Streaming client | Cancellation reaches the model; client-side TTFT is visible. |
| Two prompt protocols | Which format a 0.5B model follows best is an empirical question, so it's measured. |

Each is argued with alternatives and limits in
[docs/design-decisions.md](docs/design-decisions.md).

## Performance / Evaluation

Everything below was **measured** with the scripts in [`benchmarks/`](benchmarks/), at the commit
recorded in each `summary.json`. Host: a Windows 11 laptop (Intel Core Ultra 7 165U, 14 logical
CPUs, 32 GB, CPU only) running normal background software. Methodology and caveats are in
[docs/benchmarking.md](docs/benchmarking.md).

### 1. The orchestration makes a 0.5B model succeed more often

Qwen2.5-0.5B-Instruct (Q4_K_M) on slim-serve, greedy decoding, 42 arithmetic word problems
(6 templates × 7, exact integer answers). The question is whether SLOM gets a tiny model to
*use a tool and finish*. It's a narrow test by design, not a general agent benchmark.

| Mode | Correct | Held-out (39)¹ | Runs that answered | Model calls / run | p50 time / run |
|---|---|---|---|---|---|
| CoT baseline (1 call, no tools, "think step by step") | 16/42 (38%) | 16/39 | 42/42 | 1.0 | 5.7 s |
| SLOM agent, generic JSON protocol | 20/42 (48%) | 19/39 | 42/42 | 3.24 | 5.8 s |
| SLOM agent, Qwen-native protocol | **28/42 (67%)** | **26/39** | 42/42 | 2.0 | 5.0 s |

¹ Three problems were inspected while designing the repeat guard and forced answer.

- **Significant:** agent-qwen vs the CoT baseline. On the same problems it fixed 15 and broke 3;
  exact McNemar test p = 0.008. agent-qwen vs agent-json: 8 fixed, 0 broken, p = 0.008.
- **Not significant:** agent-json vs CoT (8 fixed vs 4 broken, p = 0.39). I'm not claiming the
  JSON protocol beats the baseline.
- **What the orchestration fixed:** all 84 agent runs ended with an answer. None crashed, failed
  to parse or ran out of steps. Before the forced answer was added, the JSON protocol ended all 3
  development problems with no answer at all, re-issuing the same tool call until its repair
  budget ran out.
- **What it can't fix:** every one of the 36 wrong agent answers comes from a wrong formula. No
  tool call in those runs computed the right value. Both agents scored 0/7 on two templates: they
  subtract the delivery fee in "price × qty + fee", and drop the multiplication in
  "boxes × size − sold". In 23 of the 36 the answer is exactly what the model's own expression
  computes; in the other 13 the model also mis-reported the tool's result (e.g. dropped a minus
  sign).
- **Protocol matters:** the format the model was trained on beat a generic one by 8 problems,
  in fewer calls. That's measured on one model; it's the default worth testing, not a law.

Raw rows, transcripts and the analysis are in
[`benchmarks/results/eval-agent/`](benchmarks/results/eval-agent/) (`analyze_eval.py` reproduces
the tests from `rows.csv`).

### 2. Crash recovery: no lost steps, recovery time set by the lease

A worker process is hard-killed at a random moment during a 10-step run, and a replacement starts
immediately. 10 trials per lease length:

| Lease | Runs completed | Steps lost | Kill → progress again (p50 / max) | Side effects executed twice |
|---|---|---|---|---|
| 2 s | 10/10 | 0 | 1.96 s / 2.34 s | 0 of 10 kills |
| 5 s | 10/10 | 0 | 4.67 s / 5.01 s | 1 of 10 kills |

The single duplicate is at-least-once semantics, visible: that kill landed between a step's side
effect and its journal commit. This is the window `idempotency_key()` exists for. Finished steps
were never re-executed in any trial (also asserted by `tests/test_crash.py`).

### 3. Backpressure: match the client to the server, and jitter the retries

slim-serve with the fake engine (2 replicas, queue of 2, 0.2 s per call), 40 runs × 3 calls
submitted at once; ideal makespan 12.0 s. Two interleaved repetitions:

| SLOM LLM concurrency | Makespan | p95 run latency | Server 503s | Every run completed |
|---|---|---|---|---|
| 2 (= replicas) | 14.2–14.7 s | 14.0–14.4 s | 0 | yes |
| 4 (= replicas + queue) | **12.9 s** | 12.7 s | 0 | yes |
| 16 (oversubscribed), jittered backoff | 29.0–29.7 s | 15.0–16.3 s | 122–125 | yes |
| 16, *before jitter was added* | 123–182 s | 122–153 s | 162–192 | yes |

- With the cap at replicas + queue, the batch finished about 8% over ideal.
- Oversubscribing is absorbed (every 503 retried, every run done) but costs 2.2–2.3× the
  makespan.
- The first version of the retry policy had no jitter. Clients rejected together retried
  together, and the batch ran 4–6× slower still. That's why `RetryPolicy` jitters and treats
  `Retry-After` as a floor. The pre-fix result is kept in
  [`results/overload-nojitter/`](benchmarks/results/overload-nojitter/).

### 4. What durability costs

No model; no-op steps, so this is engine plus journal only. Ranges are over three separate
invocations:

| | `SLOM_DB_SYNCHRONOUS=FULL` (default) | `NORMAL` |
|---|---|---|
| Journaled step, p50 / p95 | 3.0–3.8 ms / 4.2–4.7 ms | 0.52–0.69 ms / 0.86–1.05 ms |
| Replayed step (recovery), p50 | 0.014–0.034 ms | 0.022–0.025 ms |
| Steps/s, 1 worker, 8 runs in flight | 367–417 | 1,235–1,546 |

`FULL` is `fsync`-bound: a few hundred step commits per second per journal, flat with
concurrency because SQLite has a single writer. A model call on this CPU takes 0.5–10 s, so the
journal adds well under 1% to a model-bound step.

## Failure handling

| Situation | What happens | Mechanism |
|---|---|---|
| Model output unparseable | repair prompt with the reason; after `max_repairs`, the run completes as `max_repairs` | `parsing.py` (never raises), `agent.py` |
| Model repeats a tool call | not executed; next turn is forced to answer | repeat guard + answer prefill |
| Tool gets bad args / raises `ToolError` | the error text is the tool result the model sees | `Tool.invoke` |
| Tool raises anything else | retried (3×), then shown to the model | step retry policy |
| LLM `503` / `429` | retry after ≥ `Retry-After` + jitter | `LlmClient` → `RetryableError(retry_after)` |
| LLM `504` / `5xx` / connection error / truncated stream | retry with jittered exponential backoff; 8 attempts ≈ 45–90 s of outage | `RetryPolicy` |
| LLM `4xx` (e.g. `max_tokens` over the server limit) | run fails at once with the server's message | `PermanentError` |
| Model still loading | workers don't claim runs until `/readyz` is 200; runs wait as `pending` | `wait_for_llm` |
| Worker killed | its runs resume on another worker after the lease (default 15 s) | lease expiry + journal replay |
| Worker stalled past its lease | its writes are rejected; it abandons the run | fencing token |
| Run crashes every worker | failed after `SLOM_MAX_RUN_ATTEMPTS` (5) claims | `attempts` counter |
| `SIGTERM` | stop claiming; in-flight runs get `SLOM_SHUTDOWN_GRACE_S`, then are handed back (not failed) | `Worker._drain` + `Store.release` |
| Cancel | pending: immediate. Running: within `lease_s/3`; the HTTP stream closes, so slim-serve stops generating | heartbeat reads the cancel flag |
| Duplicate submit | same `run_id` + same input returns the same run; different input returns `409` | `Store.submit` |
| Embedded worker dies (`slom serve`) | logged; `/readyz` turns `503` | done-callback |

## Observability

**Metrics** (Prometheus; worker on `:9464`, API on `/metrics`):

| Metric | Answers |
|---|---|
| `slom_runs{status}` | How many runs are pending / running / done? (sampled from the journal) |
| `slom_runs_active`, `slom_run_claims_total{kind=new\|resumed}` | Is this worker busy? How often are runs recovered from another worker? |
| `slom_runs_finished_total{outcome}`, `slom_run_duration_seconds` | Outcomes; submit-to-done latency including queueing and recovery |
| `slom_steps_total{step,outcome}`, `slom_step_duration_seconds{step}` | Which step is slow or failing? (live executions only) |
| `slom_step_retries_total{step,error}` | Where are retries going, and why? |
| `slom_steps_replayed_total{step}` | Recovery at work: steps answered from the journal |
| `slom_leases_lost_total` | Workers stalling past their lease |
| `slom_llm_ttft_seconds`, `slom_llm_tokens_total{kind}`, `slom_llm_errors_total{cause}` | Model latency as SLOM sees it; overload vs timeouts vs bad requests |
| `slom_agent_runs_total{status}`, `slom_agent_invalid_outputs_total`, `slom_tool_calls_total{tool,ok}` | Is the model following the protocol? Are tools succeeding? |

Replayed steps never touch the latency, token or error metrics, so a recovery doesn't look like
a burst of very fast model calls.

**Logs.** Structured, in the same format as slim-serve (`SLOM_LOG_JSON=true` for JSON lines).
Run-scoped lines carry `run_id`.

**Correlation.** Every model call carries `X-Request-ID: <run_id>.<step>.<attempt>`. It's stored
in the run's result (`steps[].llm_request_id`) and logged by slim-serve on its access line, so
`grep <run_id>` works across both services. The test suite asserts that the two lists match.

**Live progress.** `GET /v1/runs/{id}` returns the steps journaled so far, with attempts and
durations, while the run is still going.

## Quick start

Requires Python 3.11+, plus a running slim-serve (or another OpenAI-compatible
`/v1/completions` server).

```bash
# 1. The model server (https://github.com/ragnarlothbrok53/slm-inference-machine)
pip install "slim-serve[llama]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
mkdir -p models && curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
slim-serve serve                            # :8000

# 2. SLOM
pip install slom
slom run "A shop sells pens at 17 rupees each. Priya buys 23 pens and pays a 4 rupee delivery fee. How much does she pay?" --protocol qwen
```

From source: `git clone https://github.com/ragnarlothbrok53/slm-orchestration-machine.git`, then
`uv sync` and prefix the commands with `uv run`.

Real output of that command (Qwen2.5-0.5B on a laptop CPU):

```
model> <tool_call>
{{"name": "calculator", "arguments": {"expression": "23 * 17 - 4"}}
</tool_call>
tool> 387
model> Priya pays 387 rupees for the pens.

answered: Priya pays 387 rupees for the pens.
[2 LLM calls, 1 tool calls, 706+40 tokens, 1.8s in the model]
```

The orchestration did its job: one well-formed tool call, the result, a clean answer, in 2 calls.
The *answer is wrong*: the model subtracted the delivery fee. In the evaluation it wrote
`price * qty - fee` on all 7 problems of this template, with both protocols. No amount of
orchestration fixes a wrong formula that is then computed faithfully. See [Performance](#performance--evaluation).

**Docker** (the whole stack; slim-serve is built from its GitHub tag):

```bash
mkdir -p models   # put the GGUF here, as above
docker compose up --build                  # API on :8080, model server on :8000
SLM_ENGINE=fake docker compose up --build  # no model: checks the wiring only
docker compose up --scale worker=3         # more workers on the same journal
```

## Usage

```bash
uv run slom serve        # API + worker, http://127.0.0.1:8080

# Submit and wait for the result
curl -s 'localhost:8080/v1/runs?wait=true' -H 'content-type: application/json' \
  -d '{"question": "A crate holds 48 bottles. How many bottles are in 26 crates?", "protocol": "qwen"}'

# Submit without waiting (idempotent on run_id), then poll and cancel
curl -s localhost:8080/v1/runs -H 'content-type: application/json' \
  -d '{"question": "...", "run_id": "order-42"}'
curl -s localhost:8080/v1/runs/order-42          # status, answer, steps so far
curl -s -X POST localhost:8080/v1/runs/order-42/cancel
```

`POST /v1/runs` fields: `question`; `tools` (a subset of the worker's tools, null = all);
`max_steps` (1–20, default 6); `max_repairs` (0–5, default 2); `max_tokens` (per call);
`temperature`; `protocol` (`json` | `qwen`); `run_id` (an idempotency key).

**Adding tools.** Build your own worker with them:

```python
import asyncio
from pydantic import BaseModel, Field
from slom import BUILTIN_TOOLS, ToolError, idempotency_key, tool
from slom.config import Settings
from slom.worker import serve_worker


class RefundArgs(BaseModel):
    order_id: str = Field(description="e.g. A-1001")
    amount: float


@tool(description="Refund an order. Returns the refund ID.")
async def refund(args: RefundArgs) -> str:
    if args.amount <= 0:
        raise ToolError("amount must be positive")  # the model sees this and can retry
    # `payments` is your client; the key is stable across retries and crash recovery.
    return await payments.refund(args.order_id, args.amount, idempotency_key=idempotency_key())


asyncio.run(serve_worker(Settings(), tools=[*BUILTIN_TOOLS, refund]))
```

**Writing other workflows.** The engine isn't agent-specific:

```python
from slom import RunContext, Store, Worker, Workflow


async def pipeline(ctx: RunContext, doc_id: str) -> str:
    text = await ctx.step("fetch", fetch_document, doc_id)
    summary = await ctx.step("summarize", summarize, text)  # not repaid after a crash
    return await ctx.step("store", save_summary, summary)


worker = Worker(Store("slom.db"), [Workflow.define("pipeline", pipeline)])
```

## Configuration

Environment variables (prefix `SLOM_`), a `.env` file (see [`.env.example`](.env.example)), or CLI
flags (`--db`, `--llm-url`, `--host`, `--port`), which take precedence. Full definitions, types
and bounds are in [`config.py`](src/slom/config.py).

| Variable | Default | Meaning |
|---|---|---|
| `SLOM_DB_PATH` | `slom.db` | The SQLite journal, shared by the API and workers on this host |
| `SLOM_DB_SYNCHRONOUS` | `FULL` | `FULL`: finished steps survive power loss. `NORMAL`: survive crashes only; cheaper commits |
| `SLOM_LLM_BASE_URL` | `http://127.0.0.1:8000` | slim-serve or another OpenAI-compatible server |
| `SLOM_LLM_CONCURRENCY` | `1` | Model calls in flight per worker. Set to slim-serve's `SLM_REPLICAS`; never above replicas + queue |
| `SLOM_LLM_TIMEOUT_S` | `130` | Per attempt; keep it above slim-serve's 120 s deadline |
| `SLOM_LLM_MAX_ATTEMPTS` | `8` | Per model call; about 45–90 s of tolerated outage |
| `SLOM_MAX_CONCURRENT_RUNS` | `8` | Runs one worker executes at once |
| `SLOM_LEASE_S` | `15` | Recovery time after a crash vs heartbeat write rate |
| `SLOM_MAX_RUN_ATTEMPTS` | `5` | Claims before a run is declared poison |
| `SLOM_SHUTDOWN_GRACE_S` | `10` | Time in-flight runs get on SIGTERM before being handed back |
| `SLOM_POLL_INTERVAL_S` | `0.2` | Idle claim polling |
| `SLOM_API_HOST`, `SLOM_API_PORT` | `127.0.0.1`, `8080` | API bind address |
| `SLOM_METRICS_PORT` | `9464` | Worker Prometheus port (0 disables) |
| `SLOM_LOG_LEVEL`, `SLOM_LOG_JSON` | `INFO`, `false` | Logging |

## Testing

```bash
uv sync                      # dev deps include slim-serve v0.1.0 for the contract tests
uv run pytest                # 124 tests, ~30 s
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

What the tests pin down ([`tests/`](tests/)):

- **Engine** (`test_engine.py`, `test_store.py`):
  - finished steps aren't re-executed on resume;
  - four separate connections racing for 20 runs claim each exactly once;
  - expired leases are reclaimed, and the old owner is fenced;
  - a zombie worker stops itself;
  - retries are bounded and honour `Retry-After`; permanent errors fail fast;
  - failures replay identically; changed workflow code is detected;
  - poison runs are abandoned; cancellation stops the in-flight step;
  - graceful shutdown hands runs back; the concurrency limit holds;
  - the documented retry budget (45–90 s) holds.
- **Agent** (`test_agent.py`):
  - tool then answer; repair after invalid output; bounded repairs;
  - the repeat guard and the forced answer; the step budget;
  - tool errors are shown to the model; LLM failures (transient retried, permanent fatal);
  - both protocols;
  - a resumed run makes exactly one more model call, with a byte-identical prompt.
- **LLM client** (`test_llm.py`): SSE parsing, the status-code → retry mapping, in-band errors,
  truncated streams, total deadline, concurrency cap, readiness.
- **Contract with slim-serve** (`test_integration.py`, the real v0.1.0 server over a socket):
  - an end-to-end agent run, with request IDs matching one-to-one between the journal and
    slim-serve's logs;
  - a server with no queue, oversubscribed 4×: every 503 is retried, every run completes, and no
    model call runs twice;
  - cancelling a run stops generation on the server.
- **Crash** (`test_crash.py`): a worker *process* is killed during a model call, and a new
  process finishes the run. The finished model call ran once; the interrupted one ran twice, with
  an identical prompt.

**Mutation check.** Each of these deliberate bugs makes at least one test fail:
- disabling journal replay;
- removing write fencing;
- a deferred (not `IMMEDIATE`) claim transaction;
- not retrying 503;
- ignoring the cancel flag;
- ignoring `Retry-After`;
- not checking the lease on heartbeat;
- removing the repeat guard;
- a non-streaming client.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, format, strict mypy and all
tests, and brings up the compose stack (slim-serve with the fake engine) to push a run through
API → journal → worker → slim-serve and back.

## Benchmarking

```bash
uv run python benchmarks/bench_engine.py      # journal overhead, FULL vs NORMAL  (~1 min, no model)
uv run python benchmarks/bench_recovery.py    # kill workers, measure recovery   (~3 min, no model)
uv run python benchmarks/bench_overload.py    # vs a real slim-serve, fake engine (~4 min)
uv run python benchmarks/eval_agent.py --llm-url http://127.0.0.1:8000   # real model (~40 min on CPU)
```

Each writes `benchmarks/results/<name>/summary.json` (environment, commit, config, summary) and
`rows.csv` (raw rows). Details and caveats are in [docs/benchmarking.md](docs/benchmarking.md).

## Design tradeoffs

**Why not Temporal?** The previous version of this repo used it. Temporal is the right tool for
multi-host workers, long timers, signals, versioned workflows and an ops UI. SLOM needs none of
these for its target use case, a few workers next to a local model, and each of them would add a
cluster to run first. If you need them, the concepts map directly: steps are activities, the
journal is the event history, the lease is the heartbeat timeout.

**Why SQLite?** Several processes making small transactional writes on one host is exactly what
SQLite in WAL mode does well, and it ships with Python. DuckDB was considered and rejected: one
writing process per file. Postgres is the scale-out path (see below).

**Where it breaks down:**
- **One host.** SQLite's locking doesn't hold on network filesystems, so all workers must share a
  disk (or a VM under Docker Desktop).
- **One writer.** With `FULL`, a journal sustains a few hundred step commits per second. That's
  irrelevant at CPU model speeds (one call per second or more), but a ceiling for fast GPU
  backends with many workers.
- **Recovery waits for the lease.** That's 15 s by default: a shorter lease recovers faster and
  costs more heartbeat writes.
- **No workflow versioning.** Changing the agent's step sequence fails in-flight runs (loudly).
  Drain workers before deploying such changes.
- **No retention.** The journal grows; there's no pruning yet.
- **The model is the weak link.** The orchestration can make a 0.5B model use a tool and stop
  looping. It can't make it write the right expression: the remaining errors in the eval are
  mostly wrong formulas, faithfully computed.

**Current limitations:**
- ChatML models only, via raw completions (the chat template is applied client-side).
- No constrained decoding (slim-serve has no grammar support).
- No auth or multi-tenancy on the API.
- `?wait=true` and idle workers poll the journal.
- The MLX backend path is untested here, as in slim-serve.

**At 10x scale I would change:**
1. **Storage.** Move `store.py` to Postgres: `FOR UPDATE SKIP LOCKED` claims, the same fencing
   token, and `LISTEN/NOTIFY` instead of polling. That allows workers on several hosts. The
   engine, agent and API don't change.
2. **Backpressure.** Use a global LLM concurrency budget (a token bucket in the database, or a
   gateway) instead of a per-worker cap, since workers × cap must stay within the servers'
   replicas + queue. Route model calls by queue depth (`slm_queue_depth`) across several
   slim-serve instances.
3. **Output constraints.** Use constrained decoding (a GBNF grammar from the tool schemas) in
   slim-serve instead of prefill-and-repair. It removes invalid output by construction.
4. **Deploys.** Add workflow versioning (a version on each run, old code kept until the run
   drains) and retention for finished runs.
5. **Tracing.** Use OpenTelemetry once tools call other services; the step request ID is where the
   trace context would go.

## Roadmap

- `grammar` / `response_format` support in slim-serve, and a constrained-decoding protocol here
- Postgres store behind the same interface
- `slom prune --older-than` (journal retention)
- A version field on runs, for safe deploys of changed workflows
- Evaluation beyond arithmetic word problems (multi-tool tasks), and on 1.5B / 3B models

## Project structure

```
src/slom/      engine.py · store.py · agent.py · prompts.py · parsing.py · llm.py · tools.py
              api.py · worker.py · cli.py · config.py · contracts.py · metrics.py · logs.py
tests/        engine, store, agent, API, LLM client, slim-serve contract, process-kill recovery
benchmarks/   bench_engine · bench_recovery · bench_overload · eval_agent · data/ · results/
docs/         architecture.md · design-decisions.md · benchmarking.md
```

## License

MIT, see [LICENSE](LICENSE).
