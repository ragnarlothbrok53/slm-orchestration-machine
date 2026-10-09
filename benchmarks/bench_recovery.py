"""Crash recovery: kill a worker process at a random moment, start a replacement, measure.

Each trial runs one ``steps`` workflow (N steps of S seconds; every *execution* of a step is
fsync-logged to a side file). The worker is killed hard (SIGKILL / TerminateProcess) at a uniform
random time inside the run, and a replacement worker starts immediately, as a supervisor would
restart it. Measured per trial:

* ``recovery_s``: kill -> first new step journaled by the replacement. Expected to be bounded by
  the lease (the dead worker's claim must expire) plus process start-up.
* ``re_executed``: step executions beyond N. Expected: 0 or 1 (only the step in flight at the kill).
* ``lost``: journaled steps missing at the end. Expected: always 0.

    uv run python benchmarks/bench_recovery.py --trials 10 --lease 2 5
"""

from __future__ import annotations

import argparse
import asyncio
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from common import summarize, write_results
from workloads import STEPS, StepsInput

from sof.store import TERMINAL, Store

HERE = Path(__file__).parent


def spawn(db: Path, lease_s: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, str(HERE / "_bench_worker.py"), str(db), str(lease_s)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def trial(tmp: Path, i: int, lease_s: float, steps: int, step_s: float) -> dict[str, float]:
    db, log = tmp / f"t{lease_s}-{i}.db", tmp / f"t{lease_s}-{i}.log"
    store = Store(db)
    inp = StepsInput(steps=steps, sleep_s=step_s, log_path=str(log))
    await store.submit("run", "steps", STEPS.encode_input(inp))

    first = spawn(db, lease_s)
    # Wait for the first step so process start-up isn't part of the kill window.
    while not await store.steps("run"):
        await asyncio.sleep(0.005)
    await asyncio.sleep(random.uniform(0, steps * step_s * 0.8))
    first.kill()
    first.wait()
    killed_at = time.time()
    journaled_at_kill = len(await store.steps("run"))

    second = spawn(db, lease_s)
    try:
        while True:
            run = await store.get("run")
            assert run is not None
            if run.status in TERMINAL:
                break
            await asyncio.sleep(0.01)
    finally:
        second.terminate()
        second.wait()
    journal = await store.steps("run")
    await store.close()

    after = [s.finished_at for s in journal[journaled_at_kill:]]
    executions = Counter(line.split()[0] for line in log.read_text().splitlines())
    return {
        "lease_s": lease_s,
        "trial": i,
        "status_ok": int(run.status == "completed" and run.output == str(steps)),
        "journaled_at_kill": journaled_at_kill,
        "recovery_s": round(min(after) - killed_at, 3) if after else float("nan"),
        "re_executed": sum(executions.values()) - steps,
        "lost": steps - len(journal),
        "claims": run.attempts,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--lease", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--step-s", type=float, default=0.3)
    args = ap.parse_args()

    rows = []
    summary: dict[str, object] = {"config": vars(args)}
    with tempfile.TemporaryDirectory() as d:
        for lease_s in args.lease:
            for i in range(args.trials):
                row = await trial(Path(d), i, lease_s, args.steps, args.step_s)
                rows.append(row)
                print(row)
            mine = [r for r in rows if r["lease_s"] == lease_s]
            summary[f"lease_{lease_s}"] = {
                "completed": f"{sum(r['status_ok'] for r in mine)}/{len(mine)}",
                "recovery_s": summarize([r["recovery_s"] for r in mine]),
                "re_executed_steps": dict(Counter(r["re_executed"] for r in mine)),
                "lost_steps": sum(r["lost"] for r in mine),
            }
            print(summary[f"lease_{lease_s}"])
    print("wrote", write_results("recovery", summary, rows))


if __name__ == "__main__":
    asyncio.run(main())
