"""Backpressure against a real slim-serve server: what ``SLOM_LLM_CONCURRENCY`` should be.

slim-serve (started here as a subprocess, ``fake`` engine with a fixed per-call cost) admits at
most ``replicas + queue`` requests and answers ``503 + Retry-After`` beyond that. SLOM submits a
burst of runs, each a chain of model calls, and caps its own in-flight calls at C:

* C = replicas: nothing waits in the server's queue; no 503s expected.
* C = replicas + queue: the queue is used; still no 503s expected.
* C > replicas + queue: the server sheds load; SLOM retries (backoff >= Retry-After = 1s).

Every configuration must complete every run; the question is what the excess costs.

    uv run python benchmarks/bench_overload.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from common import summarize, write_results
from workloads import StepsInput, llm_chain

from slom.engine import RetryPolicy, Worker
from slom.llm import LlmClient
from slom.logs import configure_logging
from slom.metrics import Metrics
from slom.store import TERMINAL, Store


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_server(
    port: int, replicas: int, queue: int, ttft_s: float, token_s: float
) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "SLM_ENGINE": "fake",
        "SLM_REPLICAS": str(replicas),
        "SLM_MAX_QUEUE_SIZE": str(queue),
        "SLM_FAKE_TTFT_S": str(ttft_s),
        "SLM_FAKE_TOKEN_LATENCY_S": str(token_s),
        "SLM_LOG_LEVEL": "WARNING",
    }
    cmd = [
        sys.executable,
        "-c",
        "from slim_serve.cli import app; app()",
        "serve",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/readyz").status_code == 200:
                return proc
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError("slim-serve did not start")


def server_counter(metrics_text: str, name: str) -> float:
    return sum(float(v) for v in re.findall(rf"^{name}(?:{{[^}}]*}})? (\S+)$", metrics_text, re.M))


async def one(args: argparse.Namespace, concurrency: int, tmp: Path, rep: int) -> dict[str, object]:
    port = free_port()
    server = start_server(port, args.replicas, args.queue, args.ttft, args.token_latency)
    base = f"http://127.0.0.1:{port}"
    store = Store(tmp / f"c{concurrency}-{rep}.db")
    metrics = Metrics()
    llm = LlmClient(base, concurrency=concurrency, metrics=metrics)
    retry = RetryPolicy(max_attempts=20, initial_s=1.0, max_s=30.0)  # the agent's LLM policy
    wf = llm_chain(llm, retry)
    try:
        inp = wf.encode_input(StepsInput(steps=args.calls, max_tokens=args.tokens))
        ids = [f"r{i}" for i in range(args.runs)]
        for rid in ids:
            await store.submit(rid, wf.name, inp)
        worker = Worker(
            store, [wf], metrics=metrics, max_concurrent_runs=args.runs, poll_interval_s=0.01
        )
        t0 = time.time()
        task = asyncio.create_task(worker.run())
        done = {}
        while len(done) < len(ids):
            for rid in ids:
                if rid not in done and (r := await store.get(rid)) and r.status in TERMINAL:
                    done[rid] = r
            await asyncio.sleep(0.02)
        makespan = time.time() - t0
        worker.stop()
        await task
        server_metrics = httpx.get(f"{base}/metrics").text
    finally:
        await llm.aclose()
        await store.close()
        server.terminate()
        server.wait()

    latencies = [r.updated_at - r.created_at for r in done.values()]
    retries = (
        metrics.registry.get_sample_value(
            "slom_step_retries_total", {"step": "llm", "error": "RetryableError"}
        )
        or 0.0
    )
    return {
        "llm_concurrency": concurrency,
        "repeat": rep,
        "completed": sum(r.status == "completed" for r in done.values()),
        "runs": len(ids),
        "makespan_s": round(makespan, 2),
        "run_latency_p50_s": round(summarize(latencies)["p50"], 2),
        "run_latency_p95_s": round(summarize(latencies)["p95"], 2),
        "server_503s": server_counter(server_metrics, "slm_rejected_total"),
        "slom_retries": retries,
        "server_ok_requests": server_counter(
            server_metrics, 'slm_requests_total\\{outcome="ok"\\}'
        ),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--calls", type=int, default=3, help="model calls per run")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--queue", type=int, default=2)
    ap.add_argument("--ttft", type=float, default=0.05)
    ap.add_argument("--token-latency", type=float, default=0.01)
    ap.add_argument("--concurrency", type=int, nargs="+", default=None)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    configure_logging("ERROR")
    levels = args.concurrency or [args.replicas, args.replicas + args.queue, 16]

    rows = []
    with tempfile.TemporaryDirectory() as d:
        for rep in range(args.repeat):  # interleaved, so drift affects every level alike
            for c in levels:
                row = await one(args, c, Path(d), rep)
                rows.append(row)
                print(row)
    per_call_s = args.ttft + (args.tokens - 1) * args.token_latency
    ideal = args.runs * args.calls * per_call_s / args.replicas
    summary = {
        "config": {**vars(args), "levels": levels},
        "model_cost_per_call_s": round(per_call_s, 3),
        "ideal_makespan_s": round(ideal, 2),
        "rows": rows,
    }
    print("ideal makespan (server busy 100%, zero overhead):", round(ideal, 2), "s")
    print("wrote", write_results("overload", summary, rows))


if __name__ == "__main__":
    asyncio.run(main())
