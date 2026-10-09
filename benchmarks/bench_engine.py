"""What durability costs: per-step latency, replay speed, and throughput of the SQLite journal.

No model is involved: steps are no-ops, so everything measured is engine + journal overhead.
Run for both ``synchronous`` modes:

    uv run python benchmarks/bench_engine.py
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Literal

from common import summarize, write_results
from workloads import STEPS, TIMINGS, StepsInput

from sof.engine import Worker
from sof.logs import configure_logging
from sof.metrics import Metrics
from sof.store import TERMINAL, StepRecord, Store

Sync = Literal["FULL", "NORMAL"]


async def _drive(store: Store, run_ids: list[str], max_concurrent: int) -> float:
    worker = Worker(
        store, [STEPS], metrics=Metrics(), max_concurrent_runs=max_concurrent, poll_interval_s=0.001
    )
    task = asyncio.create_task(worker.run())
    t0 = time.perf_counter()
    pending = set(run_ids)
    while pending:
        for rid in list(pending):
            run = await store.get(rid)
            if run and run.status in TERMINAL:
                assert run.status == "completed", run.error
                pending.discard(rid)
        await asyncio.sleep(0.001)
    elapsed = time.perf_counter() - t0
    worker.stop()
    await task
    return elapsed


async def latency(sync: Sync, runs: int, steps: int, tmp: Path) -> dict[str, object]:
    """One run at a time: the cost of a journaled step as seen by workflow code."""
    store = Store(tmp / f"lat-{sync}.db", synchronous=sync)
    TIMINGS.step_s.clear()
    TIMINGS.replayed_s.clear()
    inp = STEPS.encode_input(StepsInput(steps=steps))
    for i in range(runs):
        await store.submit(f"lat{i}", "steps", inp)
        await _drive(store, [f"lat{i}"], 1)

    # Replay: a run whose whole journal exists already (as after a crash at the last step).
    replay_steps = steps * 10
    await store.submit("replay", "steps", STEPS.encode_input(StepsInput(steps=replay_steps)))
    claimed = await store.claim("dead-worker", lease_s=0.0)  # expires at once
    assert claimed is not None
    for seq in range(replay_steps):
        rec = StepRecord(seq, "work", True, "1", None, 1, 0.0, 0.0)
        await store.record_step("replay", claimed.lease_token, rec)
    await _drive(store, ["replay"], 1)
    await store.close()
    return {
        "live_step_ms": summarize(TIMINGS.step_s, 1000),
        "replayed_step_ms": summarize(TIMINGS.replayed_s, 1000),
    }


async def throughput(
    sync: Sync, runs: int, steps: int, concurrency: int, tmp: Path, rep: int
) -> dict[str, float]:
    store = Store(tmp / f"tput-{sync}-{concurrency}-{rep}.db", synchronous=sync)
    inp = STEPS.encode_input(StepsInput(steps=steps))
    ids = [f"t{i}" for i in range(runs)]
    for rid in ids:
        await store.submit(rid, "steps", inp)
    elapsed = await _drive(store, ids, concurrency)
    await store.close()
    return {
        "elapsed_s": round(elapsed, 3),
        "steps_per_s": round(runs * steps / elapsed, 1),
        "runs_per_s": round(runs / elapsed, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--repeat", type=int, default=3, help="throughput repetitions per config")
    args = ap.parse_args()
    configure_logging("WARNING")

    summary: dict[str, object] = {"config": vars(args)}
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for sync in ("FULL", "NORMAL"):
            lat = await latency(sync, args.runs, args.steps, tmp)
            summary[f"latency_{sync}"] = lat
            print(f"[{sync}] live step ms {lat['live_step_ms']}")
            print(f"[{sync}] replayed step ms {lat['replayed_step_ms']}")
            for c in args.concurrency:
                rates = []
                for rep in range(args.repeat):
                    tp = await throughput(sync, args.runs * 2, args.steps, c, tmp, rep)
                    rows.append({"synchronous": sync, "concurrency": c, "repeat": rep, **tp})
                    rates.append(tp["steps_per_s"])
                rates.sort()
                print(
                    f"[{sync}] concurrency {c}: steps/s median {rates[len(rates) // 2]} "
                    f"[{rates[0]}-{rates[-1]}]"
                )
    summary["throughput"] = rows
    print("wrote", write_results("engine", summary, rows))


if __name__ == "__main__":
    asyncio.run(main())
