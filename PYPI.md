# slom

**SLOM — Small Language Orchestration Machine.** Tool-using agents for 0.5–3B models as durable
workflows: every model and tool call is journaled to SQLite, so a crashed worker's runs resume
elsewhere without repeating finished calls. No orchestration server, just a package and a file.

[GitHub](https://github.com/ragnarlothbrok53/slm-orchestration-machine) ·
[Full README](https://github.com/ragnarlothbrok53/slm-orchestration-machine#readme) ·
[Architecture](https://github.com/ragnarlothbrok53/slm-orchestration-machine/blob/main/docs/architecture.md) ·
[Benchmarks](https://github.com/ragnarlothbrok53/slm-orchestration-machine/blob/main/docs/benchmarking.md) ·
[Changelog](https://github.com/ragnarlothbrok53/slm-orchestration-machine/blob/main/CHANGELOG.md)

## Install

```bash
pip install slom
```

SLOM needs an OpenAI-compatible `/v1/completions` server. It's built and tested against
[slim-serve](https://pypi.org/project/slim-serve/):

```bash
pip install "slim-serve[llama]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
slim-serve serve --model qwen2.5-0.5b-instruct-q4_k_m.gguf      # :8000
```

## Quick start

```bash
slom run "A crate holds 48 bottles. How many bottles are in 26 crates?" --protocol qwen

slom serve      # HTTP API + worker on :8080
curl -s 'localhost:8080/v1/runs?wait=true' -H 'content-type: application/json' \
  -d '{"question": "A crate holds 48 bottles. How many bottles are in 26 crates?", "protocol": "qwen"}'
```

## Your own tools

A tool is a function plus a pydantic argument model. Errors you raise as `ToolError` go back to the
model; `idempotency_key()` stays stable across retries and crash recovery.

```python
import asyncio
from pydantic import BaseModel
from slom import BUILTIN_TOOLS, ToolError, idempotency_key, tool
from slom.config import Settings
from slom.worker import serve_worker

class RefundArgs(BaseModel):
    order_id: str
    amount: float

@tool(description="Refund an order. Returns the refund ID.")
async def refund(args: RefundArgs) -> str:
    if args.amount <= 0:
        raise ToolError("amount must be positive")
    return await payments.refund(args.order_id, args.amount, idempotency_key=idempotency_key())

asyncio.run(serve_worker(Settings(), tools=[*BUILTIN_TOOLS, refund]))
```

## Your own workflows

The engine isn't agent-specific. Each `ctx.step` is journaled, so finished steps aren't repeated
after a crash:

```python
from slom import RunContext, Store, Worker, Workflow

async def pipeline(ctx: RunContext, doc_id: str) -> str:
    text = await ctx.step("fetch", fetch_document, doc_id)
    summary = await ctx.step("summarize", summarize, text)
    return await ctx.step("store", save_summary, summary)

worker = Worker(Store("slom.db"), [Workflow.define("pipeline", pipeline)])
```

## Measured

- **Task success:** with Qwen2.5-0.5B on 42 arithmetic word problems, the agent got 62% right,
  against 38% for a chain-of-thought baseline (p = 0.03).
- **Crash recovery:** workers hard-killed mid-run, 20 trials: every run completed, no finished
  step was lost.
- **Durability cost:** about 3 ms per journaled step with full `fsync`, well under 1% of a model call.

[Methodology, caveats and raw data](https://github.com/ragnarlothbrok53/slm-orchestration-machine#performance--evaluation)

## Configuration

`SLOM_*` environment variables, a `.env` file, or CLI flags (which take precedence):

| Variable | Default | Meaning |
|---|---|---|
| `SLOM_LLM_BASE_URL` | `http://127.0.0.1:8000` | slim-serve or another OpenAI-compatible server |
| `SLOM_LLM_CONCURRENCY` | `1` | Model calls in flight per worker; match the server's replicas |
| `SLOM_DB_PATH` | `slom.db` | SQLite journal, shared by the API and workers on one host |
| `SLOM_MAX_CONCURRENT_RUNS` | `8` | Runs one worker executes at once |
| `SLOM_LEASE_S` | `15` | How soon a crashed worker's runs move elsewhere |
| `SLOM_API_HOST`, `SLOM_API_PORT` | `127.0.0.1`, `8080` | API bind address |

[All settings](https://github.com/ragnarlothbrok53/slm-orchestration-machine#configuration)

## Learn more

- [Failure handling](https://github.com/ragnarlothbrok53/slm-orchestration-machine#failure-handling) and [metrics](https://github.com/ragnarlothbrok53/slm-orchestration-machine#observability)
- [Design decisions](https://github.com/ragnarlothbrok53/slm-orchestration-machine/blob/main/docs/design-decisions.md): why not Temporal, why SQLite, leases and fencing
- [Limitations and the 10x plan](https://github.com/ragnarlothbrok53/slm-orchestration-machine#design-tradeoffs)

MIT licensed.
