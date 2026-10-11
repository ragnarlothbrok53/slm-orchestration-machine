"""A worker process for the synthetic ``steps`` workflow (used by bench_recovery.py).

python benchmarks/_bench_worker.py DB_PATH LEASE_S
"""

from __future__ import annotations

import asyncio
import sys

from workloads import STEPS

from sof.engine import Worker
from sof.logs import configure_logging
from sof.metrics import Metrics
from sof.store import Store


async def main(db: str, lease_s: float) -> None:
    configure_logging("WARNING")
    store = Store(db)
    worker = Worker(store, [STEPS], metrics=Metrics(), lease_s=lease_s, poll_interval_s=0.05)
    try:
        await worker.run()
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], float(sys.argv[2])))
