"""Durable-execution semantics, tested on small purpose-built workflows (no LLM involved)."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass

import pytest
from conftest import FakeClock
from helpers import sample_value, wait_for

from sof.engine import (
    PermanentError,
    RetryableError,
    RetryPolicy,
    RunContext,
    StepFailedError,
    Worker,
    Workflow,
    step_info,
)
from sof.metrics import Metrics
from sof.store import StepRecord, Store

pytestmark = pytest.mark.anyio

FAST = RetryPolicy(max_attempts=3, initial_s=0.01, max_s=0.05)


@dataclass
class Point:
    x: int
    label: str


class Probe:
    """Step functions that count how often they really run."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.cancelled = False
        self.failures_left = 0
        self.seen: list[tuple[str, str]] = []

    async def double(self, n: int) -> int:
        self.calls["double"] += 1
        self.seen.append((step_info().request_id, step_info().idempotency_key))
        return n * 2

    async def point(self, n: int) -> Point:
        self.calls["point"] += 1
        return Point(n, f"p{n}")

    async def flaky(self, n: int) -> int:
        self.calls["flaky"] += 1
        if self.failures_left:
            self.failures_left -= 1
            raise RetryableError("try again", retry_after=0.1)
        return n

    async def broken(self, n: int) -> int:
        self.calls["broken"] += 1
        raise PermanentError("bad input")

    async def block(self, n: int) -> int:
        self.calls["block"] += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return n


def workflows(probe: Probe) -> list[Workflow[int, int]]:
    async def three_steps(ctx: RunContext, n: int) -> int:
        a = await ctx.step("double", probe.double, n)
        p = await ctx.step("point", probe.point, a)
        assert isinstance(p, Point)  # journaled dataclasses come back as dataclasses
        return await ctx.step("double", probe.double, p.x)

    async def with_flaky(ctx: RunContext, n: int) -> int:
        return await ctx.step("flaky", probe.flaky, n, retry=FAST)

    async def handles_failure(ctx: RunContext, n: int) -> int:
        try:
            await ctx.step("broken", probe.broken, n, retry=FAST)
        except StepFailedError as exc:
            assert exc.error_type == "PermanentError" and exc.message == "bad input"
            return -1
        return n

    async def fails(ctx: RunContext, n: int) -> int:
        return await ctx.step("broken", probe.broken, n, retry=FAST)

    async def blocking(ctx: RunContext, n: int) -> int:
        await ctx.step("double", probe.double, n)
        return await ctx.step("block", probe.block, n)

    return [
        Workflow.define("three_steps", three_steps),
        Workflow.define("flaky", with_flaky),
        Workflow.define("handles_failure", handles_failure),
        Workflow.define("fails", fails),
        Workflow.define("blocking", blocking),
    ]


def make(store: Store, probe: Probe, **kw: object) -> Worker:
    opts: dict[str, object] = {"poll_interval_s": 0.01, "lease_s": 3.0, "metrics": Metrics(), **kw}
    return Worker(store, workflows(probe), **opts)  # type: ignore[arg-type]


async def run_all(worker: Worker, store: Store, *run_ids: str) -> None:
    task = asyncio.create_task(worker.run())
    try:
        for rid in run_ids:
            await wait_for(store, rid)
    finally:
        worker.stop()
        await task


async def test_completes_and_journals_every_step(store: Store) -> None:
    probe = Probe()
    await store.submit("r1", "three_steps", "5")
    await run_all(make(store, probe), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "completed" and run.output == "20"
    assert [s.name for s in await store.steps("r1")] == ["double", "point", "double"]
    assert probe.calls == {"double": 2, "point": 1}
    # request IDs are unique per attempt; idempotency keys per step
    assert probe.seen == [("r1.0.1", "r1.0"), ("r1.2.1", "r1.2")]


async def test_resume_after_crash_replays_finished_steps_without_rerunning(
    store: Store, clock: FakeClock
) -> None:
    """A worker died after journaling step 0. The next worker must not run step 0 again."""
    await store.submit("r1", "three_steps", "5")
    dead = await store.claim("dead-worker", lease_s=3.0)
    assert dead is not None
    await store.record_step(
        "r1", dead.lease_token, StepRecord(0, "double", True, "10", None, 1, 0, 0)
    )
    clock.advance(4)  # the dead worker's lease runs out

    probe = Probe()
    worker = make(store, probe)
    await run_all(worker, store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "completed" and run.output == "20"
    assert probe.calls == {"point": 1, "double": 1}  # step 0 came from the journal
    m = worker.metrics
    assert sample_value(m, "sof_steps_replayed_total", {"step": "double"}) == 1
    assert sample_value(m, "sof_run_claims_total", {"kind": "resumed"}) == 1


async def test_retryable_errors_are_retried_honouring_retry_after(store: Store) -> None:
    probe = Probe()
    probe.failures_left = 2
    await store.submit("r1", "flaky", "7")
    worker = make(store, probe)
    t0 = time.perf_counter()
    await run_all(worker, store, "r1")
    elapsed = time.perf_counter() - t0
    run = await store.get("r1")
    assert run is not None and run.output == "7"
    (rec,) = await store.steps("r1")
    assert rec.attempts == 3 and probe.calls["flaky"] == 3
    assert elapsed >= 0.2  # two waits of Retry-After=0.1s, although the backoff says 0.01s
    assert (
        sample_value(
            worker.metrics, "sof_step_retries_total", {"step": "flaky", "error": "RetryableError"}
        )
        == 2
    )


async def test_retries_are_bounded(store: Store) -> None:
    probe = Probe()
    probe.failures_left = 10
    await store.submit("r1", "flaky", "7")
    await run_all(make(store, probe), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "failed" and "RetryableError" in (run.error or "")
    assert probe.calls["flaky"] == FAST.max_attempts


async def test_permanent_errors_fail_fast_and_are_catchable(store: Store) -> None:
    probe = Probe()
    await store.submit("r1", "handles_failure", "1")
    await store.submit("r2", "fails", "1")
    await run_all(make(store, probe), store, "r1", "r2")
    r1, r2 = await store.get("r1"), await store.get("r2")
    assert r1 is not None and r1.status == "completed" and r1.output == "-1"
    assert r2 is not None and r2.status == "failed" and "bad input" in (r2.error or "")
    assert probe.calls["broken"] == 2  # once per run: no retries
    (rec,) = await store.steps("r1")
    assert not rec.ok and rec.attempts == 1


async def test_recorded_failure_replays_as_the_same_failure(store: Store, clock: FakeClock) -> None:
    await store.submit("r1", "handles_failure", "1")
    dead = await store.claim("dead", lease_s=3.0)
    assert dead is not None
    err = '{"type": "PermanentError", "message": "bad input"}'
    await store.record_step(
        "r1", dead.lease_token, StepRecord(0, "broken", False, None, err, 1, 0, 0)
    )
    clock.advance(4)
    probe = Probe()
    await run_all(make(store, probe), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.output == "-1" and probe.calls["broken"] == 0


async def test_changed_workflow_code_is_detected(store: Store, clock: FakeClock) -> None:
    await store.submit("r1", "three_steps", "5")
    dead = await store.claim("dead", lease_s=3.0)
    assert dead is not None
    await store.record_step(
        "r1", dead.lease_token, StepRecord(0, "renamed", True, "1", None, 1, 0, 0)
    )
    clock.advance(4)
    probe = Probe()
    await run_all(make(store, probe), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "failed"
    assert "NonDeterminismError" in (run.error or "") and not probe.calls


async def test_a_run_that_keeps_killing_workers_is_abandoned(
    store: Store, clock: FakeClock
) -> None:
    await store.submit("r1", "three_steps", "5")
    for _ in range(2):  # two workers claimed it and died
        assert await store.claim("dead", lease_s=3.0) is not None
        clock.advance(4)
    probe = Probe()
    await run_all(make(store, probe, max_run_attempts=2), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "failed" and "abandoned after 2" in (run.error or "")
    assert not probe.calls


async def test_cancel_stops_the_in_flight_step(store: Store) -> None:
    probe = Probe()
    await store.submit("r1", "blocking", "1")
    worker = make(store, probe, lease_s=0.3)  # heartbeat (and cancel check) every 0.1s
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(probe.started.wait(), 5)
        await store.request_cancel("r1")
        run = await wait_for(store, "r1")
    finally:
        worker.stop()
        await task
    assert run.status == "cancelled" and probe.cancelled
    assert [s.name for s in await store.steps("r1")] == ["double"]  # in-flight step not recorded


async def test_zombie_worker_stops_when_its_lease_is_taken(store: Store, clock: FakeClock) -> None:
    probe = Probe()
    await store.submit("r1", "blocking", "1")
    zombie = make(store, probe, lease_s=0.3)
    task = asyncio.create_task(zombie.run())
    try:
        await asyncio.wait_for(probe.started.wait(), 5)
        clock.advance(1)  # e.g. the zombie's host was suspended past its lease
        usurper = await store.claim("w2", lease_s=60)
        assert usurper is not None and usurper.lease_token == 2
        deadline = time.monotonic() + 5
        while not probe.cancelled and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
    finally:
        zombie.stop()
        await task
    assert probe.cancelled
    assert sample_value(zombie.metrics, "sof_leases_lost_total") == 1
    run = await store.get("r1")
    assert run is not None and run.status == "running" and run.lease_token == 2  # untouched


async def test_graceful_shutdown_hands_the_run_back(store: Store) -> None:
    probe = Probe()
    await store.submit("r1", "blocking", "1")
    worker = make(store, probe, shutdown_grace_s=0.05)
    task = asyncio.create_task(worker.run())
    await asyncio.wait_for(probe.started.wait(), 5)
    worker.stop()
    await task
    run = await store.get("r1")
    assert run is not None and run.status == "running" and run.attempts == 0
    # Another worker picks it up immediately (no lease wait) and replays step 0.
    probe2 = Probe()
    probe2.release.set()
    await run_all(make(store, probe2), store, "r1")
    done = await store.get("r1")
    assert done is not None and done.status == "completed"
    assert probe2.calls == {"block": 1}


async def test_concurrency_limit(store: Store) -> None:
    probe = Probe()
    for i in range(5):
        await store.submit(f"r{i}", "blocking", str(i))
    worker = make(store, probe, max_concurrent_runs=2)
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(probe.started.wait(), 5)
        await asyncio.sleep(0.2)
        assert probe.calls["block"] == 2
        assert sample_value(worker.metrics, "sof_runs_active") == 2
        probe.release.set()
        for i in range(5):
            await wait_for(store, f"r{i}")
    finally:
        worker.stop()
        await task
    assert probe.calls["block"] == 5


async def test_unknown_workflow_fails_the_run(store: Store) -> None:
    await store.submit("r1", "nope", "1")
    await run_all(make(store, Probe()), store, "r1")
    run = await store.get("r1")
    assert run is not None and run.status == "failed" and "no workflow named" in (run.error or "")


def test_workflow_and_step_definitions_are_checked() -> None:
    async def untyped(ctx, n):  # type: ignore[no-untyped-def]
        return n

    with pytest.raises(TypeError, match="must be async def"):
        Workflow.define("x", untyped)
    assert RetryPolicy(initial_s=1, backoff=2, max_s=5).delay(10, None) == 5
    assert RetryPolicy(initial_s=1).delay(1, retry_after=3) == 3
