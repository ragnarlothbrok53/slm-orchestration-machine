# Design decisions

Each entry: the decision, the alternatives, why, and when it stops being the right call.
Numbers come from [`benchmarks/results/`](../benchmarks/results/); how they were measured is in
[benchmarking.md](benchmarking.md).

## 1. A small durable-execution engine instead of Temporal / DBOS / Restate

**Decision.** SLOM implements durable execution itself, in two modules: `store.py`, about 350 lines,
and `engine.py`, about 450, including docstrings.

**Alternatives.**
- *Temporal*, which the first prototype used, is the industry standard. It brings timers, signals,
  workflow versioning, multi-region support and a UI. The cost is a separate server, and a library
  that needs a cluster can't be adopted with `uv add`.
- *DBOS Transact* is library-only and keeps checkpoints in Postgres. It's the closest in spirit.
- *Restate* is a single-binary server, lighter than Temporal, but still a server.

**Why.** The project's goal is a lightweight package that turns a 0.5–3B model into a reliable
agent on one machine. For that use case, a separate orchestration service costs more to run than
the problem it solves. The guarantees an agent loop actually needs are few. Finished model calls
must not be repaid after a crash, a crashed worker's runs must resume elsewhere, and retries must
be bounded and observable. Those fit in a journal plus leases. Owning the mechanism also makes its
semantics explicit and testable, for example the real process-kill test in `tests/test_crash.py`.

**When to switch.** Choose Temporal (or DBOS on Postgres) when you need any of the following:
workers on several hosts; long sleeps or timers (days); signals into running workflows;
versioned workflow code for in-flight runs; or an operations UI. The concepts map one to one.
`ctx.step` is an activity, the journal is the event history, a lease is the activity heartbeat
timeout, and `NonDeterminismError` is Temporal's non-determinism error. Moving would mostly mean
rewriting `engine.py` against Temporal's SDK; the agent, tools, protocols and LLM client stay.

## 2. SQLite (WAL) as the journal, not Postgres or DuckDB

**Decision.** One SQLite file shared by the API and all workers on a host. WAL mode,
`synchronous=FULL` by default, every write in a `BEGIN IMMEDIATE` transaction.

**Why.** The workload is small transactional writes from several processes: insert a run, claim
with a compare-and-set, append a step, heartbeat. WAL gives concurrent readers and one serialized
writer through file locks, which is exactly the coordination claim-by-lease needs. It ships with
Python, so the package needs no infrastructure.

**Why not DuckDB.** DuckDB is a columnar analytics engine. Only one process may open a database
file for writing; other processes get read-only access. The API and every worker write, so DuckDB
would need a single writer process in front of it, which is the server we're avoiding. Frequent
single-row updates (heartbeats) are also its weak spot. It *is* a good tool for analysing the
journal afterwards: its `sqlite` extension reads the file directly.

**Measured cost** (no model, so engine and journal only; ranges over repeated runs are in
[benchmarking.md](benchmarking.md)):
- A journaled step costs 3.0–3.8 ms p50 with `FULL`, dominated by `fsync`, and 0.5–0.7 ms with
  `NORMAL`.
- Replaying a journaled step costs 0.01–0.03 ms.
- Throughput: with `FULL`, one journal sustains 235–417 steps/s, because commits serialize on
  SQLite's single writer, so adding concurrency barely raises it. With `NORMAL`, it's
  900–1,700 steps/s.
- In context: a model call on the benchmark CPU takes 0.5–10 s, so the journal adds well under 1%
  to a model-bound step.

**`FULL` vs `NORMAL`.** `NORMAL` survives process crashes but may lose the last commits on power
loss. `FULL` is the default because "finished steps are never repaid" should also hold across a
power cut; `SLOM_DB_SYNCHRONOUS=NORMAL` trades that for cheaper commits.

**Where it breaks.**
- *One host only:* SQLite's locking is unreliable on network filesystems, so workers must share
  the host (or the VM, under Docker Desktop).
- *One writer:* the ceiling above is per journal.
- *No cleanup:* runs and steps accumulate; there is no retention policy yet.

At 10x, move `store.py` to Postgres: `SELECT ... FOR UPDATE SKIP LOCKED` for claims, the same
fencing token, and `LISTEN/NOTIFY` instead of polling. Everything else stays.

## 3. Replay by re-execution (journal), not state snapshots

**Decision.** After a crash the workflow function runs again from the top. Each `ctx.step` call
returns the journaled result if one exists, so execution effectively resumes at the first
unfinished step.

**Alternative.** Serialize the agent's state (history, counters) after every step and restore it.

**Why.** The agent loop stays ordinary Python with local variables: no state class to keep in sync
with the code, and nothing to migrate when it changes. The price is a determinism requirement:
code between steps must make the same decisions given the same step results. Prompt rendering and
parsing are pure functions for this reason (`prompts.py`, `parsing.py`). Violations are detected:
if the step name at position *n* differs from the journal, the run fails with
`NonDeterminismError` instead of continuing on corrupted assumptions.

**Limitation.** There's no workflow versioning. Changing the sequence of steps the agent takes
breaks runs that are in flight during a deploy (they fail loudly). Drain workers before such
changes, or add a version field to the run. That's on the roadmap.

## 4. Leases with fencing tokens

**Decision.**
- A worker claims a run by setting `lease_until = now + SLOM_LEASE_S` and incrementing
  `lease_token`, then renews the lease every `lease_s / 3`.
- Every write a worker makes (step, finish, heartbeat) checks that `lease_token` is still its own,
  in the same transaction.
- If another worker has claimed the run since, the write fails and the first worker abandons the
  run.

**Why the token, not the worker ID.** A worker can stall without dying: a GC pause, a suspended
laptop, a frozen VM. When it wakes up, its lease may have expired and another worker may own the
run. Comparing worker IDs wouldn't catch this if the same worker re-claimed the run later; a
monotonically increasing token does. This is tested in
`test_zombie_worker_stops_when_its_lease_is_taken` and
`test_expired_lease_is_reclaimed_and_old_owner_is_fenced`.

**Measured.** With a hard kill and an immediate replacement worker, the replacement made progress
again after a p50 of 1.96 s with a 2 s lease, and 4.67 s with a 5 s lease, over 10 trials each. All
20 runs completed and no steps were lost. Recovery time is set by the lease, which is the main
setting to tune: shorter means faster recovery and more heartbeat writes.

**Poison runs.** `attempts` counts claims that didn't end in a clean hand-back. After
`SLOM_MAX_RUN_ATTEMPTS` the run is failed instead of being handed to yet another worker it might
also kill (e.g. a tool that segfaults).

## 5. At-least-once steps, with an idempotency key

**Decision.** A step whose result was journaled never runs again. A step that was in flight when a
worker died runs again on the next worker. Tools get `idempotency_key()`, which stays the same
across retries and recovery, to pass to systems they modify.

**Why not exactly-once.** It's impossible in general: the side effect (an HTTP call, a payment)
and the journal write can't be atomic across systems. Temporal makes the same promise for the same
reason.

**Measured.** In the recovery benchmark, 1 of 20 kills landed between a step's side effect and its
journal commit, so that side effect happened twice. This is the window an idempotency key closes.

## 6. Step failures are journaled too

**Decision.** When a step fails permanently (non-retryable, or out of attempts), the failure is
recorded like a result and re-raised identically on replay.

**Why.** The workflow may have reacted to the failure. The agent, for example, shows a failed tool
call to the model. If replay re-ran the failed step and it succeeded this time, the code would take
a different branch than the one whose later steps are already in the journal. Recording failures
keeps replay deterministic.

## 7. Retries: classified errors, jittered backoff, `Retry-After` as a floor

**Decision.**
- `PermanentError` fails immediately.
- `RetryableError` and unknown exceptions retry with exponential backoff and ±50% jitter.
- A server's `Retry-After` is a lower bound, and jitter is only ever added on top of it.

**Why, measured.** The first version had no jitter. In `bench_overload`, 40 runs whose
oversubscribed model calls were rejected together retried together, were rejected together again,
and backed off further while the server sat idle. The batch took 123–182 s, against 12.9 s when
not oversubscribed. With jitter it took 29–30 s. The pre-jitter result is kept in
`benchmarks/results/overload-nojitter/`.

**Mapping of slim-serve's contract** (`llm.py`):

| Response | Action |
|---|---|
| 503 (queue full / model loading), 429 | retry, wait ≥ `Retry-After` |
| 504 (server deadline), 5xx, 408 | retry with backoff |
| other 4xx (e.g. `max_tokens` over limit) | fail the step |
| connection error, client timeout | retry with backoff |
| in-band `data: {"error": ...}`, stream ends without `finish_reason` | retry |

`SLOM_LLM_MAX_ATTEMPTS=8` gives 7 waits that add up to 45–90 s, enough to ride out a model-server
restart. This is pinned by a test.

## 8. Client-side LLM concurrency matched to server capacity

**Decision.** Each worker caps its in-flight model calls (`SLOM_LLM_CONCURRENCY`, default 1, the
slim-serve default replica count). Calls beyond the cap wait in the worker. The per-attempt
deadline starts only once a slot is acquired.

**Why, measured** (`bench_overload`: 2 replicas, queue of 2, 40 runs × 3 calls, ideal makespan
12.0 s):

| cap | makespan | 503s | |
|---|---|---|---|
| 2 (= replicas) | 14.2–14.7 s | 0 | the replica idles during each client round-trip |
| 4 (= replicas + queue) | 12.9 s | 0 | the queue hides the round-trip; ~8% over ideal |
| 16 (oversubscribed) | 29.0–29.7 s | 122–125 | retries work but waste capacity and spread latency (p95 15–16 s, p50 8 s) |

The server's admission control (503) is a safety net, not a scheduler. It fails fast, but every
rejection costs at least `Retry-After`. Keep the cap at or below replicas + queue. Note that
slim-serve's deadline (`SLM_REQUEST_TIMEOUT_S`) includes time in its queue, so a deep server queue
also risks 504s; waiting on the client side doesn't.

## 9. The LLM client always streams

**Why.**
1. **Cancellation reaches the model.** Cancelling a run closes the HTTP stream, slim-serve
   detects the disconnect and stops generating within about one token. A non-streaming request
   would keep a replica busy until `max_tokens`. This is tested against the real server in
   `test_cancelling_a_run_stops_generation_on_the_server`.
2. **Client-side TTFT is observable** (`slom_llm_ttft_seconds`), which separates time queued plus
   prefill from decode.

## 10. Two prompt protocols, chosen by measurement

**Decision.** slim-serve serves raw completions, so SLOM owns the chat template (ChatML). There
are two protocols:
- **`json`** is generic: `{"tool": ..., "args": ...}` or `{"answer": ...}`, with the assistant turn
  prefilled with `{`.
- **`qwen`** is Qwen2.5's native tool-call format, rendered byte for byte as its GGUF chat template
  does, including the template's literal double braces.

Results are in [benchmarking.md](benchmarking.md#4-task-success-with-a-real-model).

**Loop policies**, found by reading transcripts of failing runs on three development problems
(excluded from the held-out score):
- **Repeat guard.** If the model re-issues a call it has already made, the call isn't executed
  again; the model is told it already has the result.
- **Forced answer.** After a repeat, or on the last allowed step, the `json` protocol prefills
  `{"answer": "`, leaving the model no option but to answer. Greedy decoding plus in-context
  copying otherwise made the 0.5B model repeat its previous turn verbatim until `max_steps`.

**Why not constrained decoding (grammars).** It would be the stronger tool, but slim-serve
v0.1.0 has no grammar support. Adding a `grammar` / `response_format` parameter there is on the
roadmap.

## 11. Tool failures are observations, not run failures

**Decision.**
- Invalid arguments and `ToolError` go back to the model as the tool's result, so it can correct
  itself.
- Unexpected exceptions are retried as transient (3 attempts). If retries run out, that also goes
  back to the model.
- The run fails only for infrastructure problems: the LLM is unreachable after its retries, a
  permanent LLM error, or an unknown tool requested in the run's input.

## 12. Readiness, polling, metrics

- **API readiness means "the journal answers"**, not "a model is loaded". Runs can be accepted
  while the model loads; they stay `pending`. Workers wait for the LLM's `/readyz` before claiming
  anything, so they don't burn retries during a model load.
- **Polling.** Idle workers poll for claimable runs every `SLOM_POLL_INTERVAL_S` (0.2 s), which
  costs one indexed query. Postgres would use `LISTEN/NOTIFY`; for SQLite on one host, polling is
  the simple correct option. `?wait=true` on the API also polls the journal.
- **Metrics only count live work.** Steps served from the journal increment
  `slom_steps_replayed_total` and nothing else, so recovery doesn't inflate latency, token or error
  metrics. Workflow-level metrics (invalid outputs, agent outcomes) are guarded by
  `ctx.replaying`.
- **No distributed tracing yet.** Instead, the journal's step identity is propagated as
  `X-Request-ID = run.seq.attempt`, which slim-serve logs, so one ID finds a model call in both
  services' logs. This is tested end to end. OpenTelemetry becomes worth it with more hops (tools
  calling services), and the request ID is where a trace ID would go.
