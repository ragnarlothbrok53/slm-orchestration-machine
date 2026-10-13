# Benchmarking

Four benchmarks, each answering one question. All of them write
`benchmarks/results/<name>/summary.json` (environment, git commit, configuration, summary) and
`rows.csv` (one row per measurement), so results can be re-analysed without re-running.

| Script | Question | Needs |
|---|---|---|
| `bench_engine.py` | What does durability cost per step? | nothing |
| `bench_recovery.py` | When a worker dies, how long until its run moves again, and what runs twice? | nothing |
| `bench_overload.py` | How should SLOM's LLM concurrency relate to the server's capacity? | slim-serve (dev dependency; fake engine) |
| `eval_agent.py` | Does the orchestration make a small model succeed at a tool task? | slim-serve with a real model |

## Environment

The committed results come from one machine: a Windows 11 laptop (Intel Core Ultra 7 165U,
14 logical CPUs, 32 GB RAM, Samsung NVMe SSD, CPU only), with normal background software
running. Journals were written to `%TEMP%` on that SSD. Absolute numbers will differ on other
machines. Compare the *ratios* within a benchmark, which were stable.

## 1. Journal overhead (`bench_engine.py`)

**Workload.** A workflow of 20 no-op steps, so everything measured is engine and journal.
- *Latency:* 30 runs, one at a time. Each `ctx.step` is timed as workflow code sees it, which
  includes the fenced journal commit.
- *Replay:* a run with 200 pre-journaled steps is resumed, and each replayed step is timed.
- *Throughput:* 60 runs × 20 steps, with up to 1, 8 or 32 runs in flight in one worker, repeated
  3 times. Reported as steps per second.

Both `SLOM_DB_SYNCHRONOUS` modes are measured.

**Caveat.** Sub-millisecond timings on this host vary between sessions: an earlier session
(before benchmark logging was turned down to WARNING) measured `NORMAL` at 0.24–0.86 ms p50.
The three back-to-back invocations reported here agree within 1.3×. Their raw results are in
`results/engine/run{1,2,3}/`.

**Results** (commit `6ddad67`, ranges over the three invocations):

| | `FULL` | `NORMAL` |
|---|---|---|
| Live step p50 / p95 / p99 | 3.0–3.8 / 4.2–4.7 / 5.9–6.2 ms | 0.52–0.69 / 0.86–1.05 / 1.0–1.2 ms |
| Replayed step p50 | 0.014–0.034 ms | 0.022–0.025 ms |
| Steps/s, 1 run in flight | 235–366 | 905–1,156 |
| Steps/s, 8 in flight | 367–417 | 1,235–1,546 |
| Steps/s, 32 in flight | 257–363 | 1,382–1,728 |

**Reading it.** With `FULL`, every step costs an `fsync` (~3 ms on this SSD), and commits
serialize on SQLite's single writer, so concurrency barely moves throughput. `NORMAL` skips the
per-commit `fsync` in WAL mode; it is about 5× cheaper and scales a little with concurrency
because commits batch better. Replay is essentially free (tens of microseconds per step, mostly
pydantic decoding).

## 2. Crash recovery (`bench_recovery.py`)

**Workload.** One run of 10 steps of 0.3 s each. Every *execution* of a step appends an
fsync-ed line to a side file, so duplicates are visible.

**Procedure.** A worker process starts the run. After its first step, at a uniformly random
moment within the run, it is killed hard (`TerminateProcess` on Windows, `SIGKILL` elsewhere),
with no cleanup and no lease release. A replacement worker is started immediately, as a
supervisor would. 10 trials per lease length.

**Measured.**
- `recovery_s`: time from the kill to the first step journaled by the replacement.
- `re_executed`: step executions beyond 10.
- `lost`: steps missing from the final journal.

**Results** (commit `6ddad67`):

| Lease | Completed | Lost steps | Recovery p50 | p95 | max | Duplicate executions |
|---|---|---|---|---|---|---|
| 2 s | 10/10 | 0 | 1.96 s | 2.30 s | 2.34 s | 0 in 10 trials |
| 5 s | 10/10 | 0 | 4.67 s | 4.92 s | 5.01 s | 1 in 10 trials |

An earlier session (commit `367c8c7` plus the then-uncommitted benchmark code) gave the same picture: 20/20 completed, 0 lost, p50 1.90 s
and 4.54 s, and 1 duplicate in 20.

**Reading it.**
- *Recovery time.* It is roughly the time left on the dead worker's lease, plus up to one poll
  interval, plus the replacement's start-up. A uniformly random kill time makes it vary between
  ~0 and the lease. Shorter leases recover faster; the cost is one heartbeat write per run every
  `lease_s / 3`.
- *Duplicates.* A duplicate occurs only when the kill lands after a step's side effect and before
  its journal commit. Here that window (an fsync-ed append, then a ~3 ms `FULL` commit) is a few
  percent of each 0.3 s step, so roughly 1 duplicate in 20 kills is the expected order of
  magnitude.

## 3. Overload and backpressure (`bench_overload.py`)

**Workload.**
- slim-serve runs as a subprocess with the `fake` engine: 2 replicas, a queue of 2,
  50 ms TTFT and 10 ms per token, 16 tokens per call, so a call costs 0.2 s.
- 40 runs are submitted at once, each a chain of 3 model calls (120 calls in total). One SLOM
  worker with up to 40 runs in flight sends them, capped at C concurrent model calls.
- C is 2 (= replicas), 4 (= replicas + queue) and 16 (oversubscribed).
- Configurations are interleaved and repeated twice.
- **Ideal makespan:** 120 calls × 0.2 s / 2 replicas = **12.0 s**.

**Measured.**
- makespan;
- run latency p50 and p95;
- the server's `slm_rejected_total` (503s), scraped from its `/metrics`;
- SLOM's retry counter.

Every run must complete in every configuration.

**Results** (commit `6ddad67`, two interleaved repetitions):

| Cap | Makespan | Run p50 | Run p95 | Server 503s | SLOM retries | Completed |
|---|---|---|---|---|---|---|
| 2 | 14.2–14.7 s | 12.0–12.5 s | 14.0–14.4 s | 0 | 0 | 40/40 |
| 4 | 12.9 s | 10.9–11.0 s | 12.7 s | 0 | 0 | 40/40 |
| 16 | 29.0–29.7 s | 7.8–8.0 s | 15.0–16.3 s | 122–125 | 122–125 | 40/40 |

And before jitter was added to `RetryPolicy` (`results/overload-nojitter/`, otherwise the same
setup):

| Cap | Makespan | Run p95 | Server 503s |
|---|---|---|---|
| 16 | 123–182 s | 122–153 s | 162–192 |

**Reading it.**
- *Cap = replicas.* The server is idle while each client round-trip completes, so the run takes
  18–22% over ideal.
- *Cap = replicas + queue.* The server's queue hides that gap: ~8% over ideal, and nothing is
  rejected.
- *Oversubscribed.* The server sheds about one request for every one it serves. Each rejection
  costs at least `Retry-After` (1 s) on that call. Lucky runs finish *earlier* (p50 8 s) and
  unlucky ones much later: shedding is unfair as well as slower.
- *Without jitter.* Rejected clients retried in synchronized waves, collided again, and backed off
  exponentially while the server sat idle. That's the reason for the jitter, and why
  `Retry-After` is treated as a floor with jitter only added above it.
- *Every row:* no run failed. The retry budget absorbed the load either way.

## 4. Task success with a real model (`eval_agent.py`)

**Model and server.** Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (469 MB), served by slim-serve v0.1.0
on the CPU above (llama.cpp backend, one replica). Decoding is greedy (temperature 0).

**Data.** `benchmarks/data/arith.jsonl`: 42 arithmetic word problems, 7 for each of 6 templates
(price × quantity + fee, give away and buy bags, two-leg distance, boxes minus sold, percentage
discount, weekly savings minus spending). Each problem needs 2–3 operations on 2–3 digit numbers
and has an exact integer answer. The file is regenerated deterministically by
`benchmarks/data/make_arith.py` (seed 7).

**What this measures, and what it doesn't.** This task is chosen so that a tool can help: the
question is whether the orchestration gets a 0.5B model to *use* the tool and finish cleanly.
It says nothing about general reasoning, multi-tool planning or other models.

**Modes.** Questions are interleaved across modes, so host drift affects all of them alike.

| Mode | What runs |
|---|---|
| `cot` | One call, no tools. The system prompt asks for step-by-step reasoning and a final `Answer: <number>` line (max 384 tokens). This is the model on its own, with the standard prompting trick. |
| `agent-json` | The SLOM agent, generic JSON protocol (prefill `{`, repair prompts, repeat guard, forced answer), calculator tool, up to 6 model calls. |
| `agent-qwen` | The same, with Qwen2.5's native `<tool_call>` format. |

Agent runs go through the full stack (journal, worker, retries, streaming client), not a
shortcut.

**Grading.** The last number in the final answer (or after `Answer:` for `cot`) must equal the
expected integer.

**Development contamination.** The repeat guard and the forced answer were designed after reading
transcripts of failing runs on three problems (`arith-00`–`arith-02`). Accuracy is reported both
on all 42 and on the 39 held-out problems.

**Results** (commit `4cb8401`, recorded in `summary.json`):

| Mode | Correct (42) | Held-out (39) | Answered | Calls / run | p50 / p95 time per run |
|---|---|---|---|---|---|
| `cot` | 16 (38%) | 16 | 42 | 1.0 | 6.8 / 15.9 s |
| `agent-json` | 20 (48%) | 19 | 42 | 3.24 | 6.1 / 11.5 s |
| `agent-qwen` | 26 (62%) | 24 | 42 | 2.0 | 5.1 / 9.6 s |

**Paired comparisons** (same problems, exact McNemar; `analyze_eval.py`):

| | only first right | only second right | p |
|---|---|---|---|
| cot vs agent-qwen | 4 | 14 | 0.031 |
| agent-json vs agent-qwen | 1 | 7 | 0.070 |
| cot vs agent-json | 4 | 8 | 0.39 |

**By template** (7 problems each):

| Template | cot | agent-json | agent-qwen |
|---|---|---|---|
| price × qty + fee | 2 | 0 | 0 |
| give away, buy bags | 1 | 0 | 7 |
| two-leg distance | 3 | 6 | 6 |
| boxes × size − sold | 1 | 0 | 0 |
| percent discount | 7 | 7 | 7 |
| weekly savings − gift | 2 | 7 | 6 |

**Where the agents fail.**
- In 37 of the 38 wrong agent answers, no calculator call computed the expected value: the model
  wrote a wrong formula.
- In 24 of the 38, the final answer equals the value of the model's own expression. In most of the
  others, it also mis-reported the tool result, e.g. answering 627 when the calculator returned
  −627.
- The remaining one (`arith-41`, agent-qwen) is a grading artifact: "Kabir has 232 rupees left …
  spending 68 rupees on a gift" states the right answer but ends on 68. It is counted as wrong, as
  the grading rule says.
- An earlier run (commit `813bcd5`) had a bug in the Qwen system prompt: the tool-call example
  showed doubled braces (`{{"name": ...}}`). With it, agent-qwen scored 28/42. CoT and agent-json,
  which don't use that prompt, gave identical results in both runs.
- On "price × qty + fee", both agents wrote `price * qty - fee` on all 7 problems.
- Using a tool doesn't help on what the model already does reliably (percent discount: 7/7 in
  every mode).
- It helps most where CoT arithmetic breaks down: multi-digit products, as in "weekly savings":
  2/7 for CoT against 6–7/7 with a tool.

## Reproducing

```bash
uv sync
uv run python benchmarks/bench_engine.py
uv run python benchmarks/bench_recovery.py --trials 10 --lease 2 5
uv run python benchmarks/bench_overload.py
# real model: start slim-serve first (see the README), then
uv run python benchmarks/eval_agent.py --llm-url http://127.0.0.1:8000
```

## Known gaps

- One machine, Windows, CPU only. Not measured: Linux, GPUs, or a journal on a different disk
  class.
- Recovery was measured with one run at a time; many runs moving at once after a crash will
  contend on the single SQLite writer.
- The overload benchmark is closed-loop (a fixed burst). Open-loop arrival (Poisson) would show
  behaviour under sustained overload better.
- The eval is narrow (one task family, 42 problems, one 0.5B model). Differences of a few problems
  between modes are within noise at this size.
