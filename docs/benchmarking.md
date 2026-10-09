# Benchmarking

Four benchmarks, each answering one question. All of them write
`benchmarks/results/<name>/summary.json` (environment, git commit, configuration, summary) and
`rows.csv` (one row per measurement), so results can be re-analysed without re-running.

| Script | Question | Needs |
|---|---|---|
| `bench_engine.py` | What does durability cost per step? | nothing |
| `bench_recovery.py` | When a worker dies, how long until its run moves again, and what runs twice? | nothing |
| `bench_overload.py` | How should SOF's LLM concurrency relate to the server's capacity? | slm-runtime (dev dependency; fake engine) |
| `eval_agent.py` | Does the orchestration make a small model succeed at a tool task? | slm-runtime with a real model |

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

Both `SOF_DB_SYNCHRONOUS` modes are measured.

**Caveat.** Run-to-run variation on this host is large for sub-millisecond operations: whole
invocations differed by up to 3.5× for `NORMAL`. The results below therefore give ranges over
several invocations, not one run.

<!-- ENGINE -->

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

<!-- RECOVERY -->

## 3. Overload and backpressure (`bench_overload.py`)

**Workload.**
- slm-runtime runs as a subprocess with the `fake` engine: 2 replicas, a queue of 2,
  50 ms TTFT and 10 ms per token, 16 tokens per call, so a call costs 0.2 s.
- 40 runs are submitted at once, each a chain of 3 model calls (120 calls in total). One SOF
  worker with up to 40 runs in flight sends them, capped at C concurrent model calls.
- C is 2 (= replicas), 4 (= replicas + queue) and 16 (oversubscribed).
- Configurations are interleaved and repeated twice.
- **Ideal makespan:** 120 calls × 0.2 s / 2 replicas = **12.0 s**.

**Measured.**
- makespan;
- run latency p50 and p95;
- the server's `slm_rejected_total` (503s), scraped from its `/metrics`;
- SOF's retry counter.

Every run must complete in every configuration.

<!-- OVERLOAD -->

## 4. Task success with a real model (`eval_agent.py`)

**Model and server.** Qwen2.5-0.5B-Instruct, Q4_K_M GGUF (469 MB), served by slm-runtime v0.1.0
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
| `agent-json` | The SOF agent, generic JSON protocol (prefill `{`, repair prompts, repeat guard, forced answer), calculator tool, up to 6 model calls. |
| `agent-qwen` | The same, with Qwen2.5's native `<tool_call>` format. |

Agent runs go through the full stack (journal, worker, retries, streaming client), not a
shortcut.

**Grading.** The last number in the final answer (or after `Answer:` for `cot`) must equal the
expected integer.

**Development contamination.** The repeat guard and the forced answer were designed after reading
transcripts of failing runs on three problems (`arith-00`–`arith-02`). Accuracy is reported both
on all 42 and on the 39 held-out problems.

<!-- EVAL -->

## Reproducing

```bash
uv sync
uv run python benchmarks/bench_engine.py
uv run python benchmarks/bench_recovery.py --trials 10 --lease 2 5
uv run python benchmarks/bench_overload.py
# real model: start slm-runtime first (see the README), then
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
