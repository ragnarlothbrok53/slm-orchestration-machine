"""A worker process for the synthetic ``steps`` workflow (used by bench_recovery.py).

python benchmarks/_bench_worker.py DB_PATH LEASE_S
"""

from __future__ import annotations

import asyncio
import sys

from workloads import STEPS

from slom.engine import Worker
from slom.logs import configure_logging
from slom.metrics import Metrics
from slom.store import Store


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
