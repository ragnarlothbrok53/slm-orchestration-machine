"""The journal's concurrency contract: exclusive claims, lease expiry, fencing, idempotency."""

import asyncio
from pathlib import Path

import pytest
from conftest import FakeClock

from sof.store import LeaseLostError, RunConflictError, StepRecord, Store

pytestmark = pytest.mark.anyio


def step(seq: int, name: str = "s") -> StepRecord:
    return StepRecord(seq, name, True, '"x"', None, 1, 0.0, 0.0)


async def test_submit_is_idempotent_and_rejects_conflicting_reuse(store: Store) -> None:
    run, created = await store.submit("r1", "agent", '{"q": 1}')
    assert created and run.status == "pending"
    again, created = await store.submit("r1", "agent", '{"q": 1}')
    assert not created and again == run
    with pytest.raises(RunConflictError):
        await store.submit("r1", "agent", '{"q": 2}')


async def test_claims_are_exclusive_and_fifo(store: Store, clock: FakeClock) -> None:
    for i in range(3):
        await store.submit(f"r{i}", "agent", "{}")
        clock.advance(1)
    first = await store.claim("w1", lease_s=10)
    second = await store.claim("w2", lease_s=10)
    assert first is not None and second is not None
    assert (first.id, second.id) == ("r0", "r1")
    assert first.status == "running" and first.attempts == 1 and first.lease_token == 1


async def test_two_processes_never_claim_the_same_run(tmp_path: Path) -> None:
    """Separate connections (as separate processes would have) racing for 20 runs."""
    path = tmp_path / "race.db"
    stores = [Store(path) for _ in range(4)]
    try:
        for i in range(20):
            await stores[0].submit(f"r{i:02}", "agent", "{}")

        async def drain(s: Store, owner: str) -> list[str]:
            got = []
            while (run := await s.claim(owner, lease_s=60)) is not None:
                got.append(run.id)
            return got

        results = await asyncio.gather(*(drain(s, f"w{i}") for i, s in enumerate(stores)))
        claimed = [rid for r in results for rid in r]
        assert sorted(claimed) == [f"r{i:02}" for i in range(20)]  # each exactly once
    finally:
        for s in stores:
            await s.close()


async def test_expired_lease_is_reclaimed_and_old_owner_is_fenced(
    store: Store, clock: FakeClock
) -> None:
    await store.submit("r1", "agent", "{}")
    old = await store.claim("w1", lease_s=10)
    assert old is not None
    await store.record_step("r1", old.lease_token, step(0))

    assert await store.claim("w2", lease_s=10) is None  # lease still valid
    clock.advance(11)
    new = await store.claim("w2", lease_s=10)
    assert new is not None and new.lease_token == old.lease_token + 1 and new.attempts == 2

    # The stalled first worker wakes up: every write it attempts is rejected.
    with pytest.raises(LeaseLostError):
        await store.heartbeat("r1", old.lease_token, 10)
    with pytest.raises(LeaseLostError):
        await store.record_step("r1", old.lease_token, step(1))
    with pytest.raises(LeaseLostError):
        await store.finish("r1", old.lease_token, "completed", output="{}")

    await store.record_step("r1", new.lease_token, step(1))
    await store.finish("r1", new.lease_token, "completed", output='"done"')
    assert [s.seq for s in await store.steps("r1")] == [0, 1]
    final = await store.get("r1")
    assert final is not None and final.status == "completed"


async def test_heartbeat_extends_lease(store: Store, clock: FakeClock) -> None:
    await store.submit("r1", "agent", "{}")
    run = await store.claim("w1", lease_s=10)
    assert run is not None
    clock.advance(8)
    assert await store.heartbeat("r1", run.lease_token, 10) is False
    clock.advance(8)  # 16s after the claim, but only 8s after the heartbeat
    assert await store.claim("w2", lease_s=10) is None


async def test_duplicate_step_write_is_rejected(store: Store) -> None:
    await store.submit("r1", "agent", "{}")
    run = await store.claim("w1", lease_s=10)
    assert run is not None
    await store.record_step("r1", run.lease_token, step(0))
    with pytest.raises(LeaseLostError):
        await store.record_step("r1", run.lease_token, step(0))


async def test_cancel_pending_is_immediate_running_is_flagged(store: Store) -> None:
    await store.submit("pending", "agent", "{}")
    cancelled = await store.request_cancel("pending")
    assert cancelled is not None and cancelled.status == "cancelled"

    await store.submit("running", "agent", "{}")
    run = await store.claim("w1", lease_s=10)
    assert run is not None and run.id == "running"
    flagged = await store.request_cancel("running")
    assert flagged is not None and flagged.status == "running" and flagged.cancel_requested
    assert await store.heartbeat("running", run.lease_token, 10) is True

    assert await store.request_cancel("missing") is None


async def test_release_hands_back_without_counting_an_attempt(store: Store) -> None:
    await store.submit("r1", "agent", "{}")
    run = await store.claim("w1", lease_s=60)
    assert run is not None
    await store.release("r1", run.lease_token)
    again = await store.claim("w2", lease_s=60)  # immediately claimable, despite the long lease
    assert again is not None and again.attempts == 1


async def test_journal_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "persist.db"
    s1 = Store(path)
    await s1.submit("r1", "agent", "{}")
    run = await s1.claim("w1", lease_s=10)
    assert run is not None
    await s1.record_step("r1", run.lease_token, step(0, "llm"))
    await s1.close()

    s2 = Store(path)
    try:
        assert [(s.seq, s.name) for s in await s2.steps("r1")] == [(0, "llm")]
        assert await s2.count_by_status() == {"running": 1}
    finally:
        await s2.close()
