"""Crash recovery with real processes: a worker is killed (no cleanup, no lease release) in the
middle of a model call, and a second worker process finishes the run.

This is the claim the project rests on, so it is tested the hard way rather than simulated.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from helpers import wait_for
from pydantic import TypeAdapter
from slm_server import ScriptedEngine, slim_serve

from slom.agent import WORKFLOW_NAME
from slom.contracts import AgentInput, AgentResult
from slom.store import Store

pytestmark = pytest.mark.anyio

LEASE_S = 1.0


def start_worker(db: Path, llm_url: str, log: Path) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "SLOM_DB_PATH": str(db),
        "SLOM_LLM_BASE_URL": llm_url,
        "SLOM_LEASE_S": str(LEASE_S),
        "SLOM_POLL_INTERVAL_S": "0.05",
        "SLOM_METRICS_PORT": "0",
    }
    out = log.open("wb")
    return subprocess.Popen(
        [sys.executable, "-m", "slom", "worker"], env=env, stdout=out, stderr=subprocess.STDOUT
    )


async def test_killed_worker_is_replaced_without_repeating_finished_steps(tmp_path: Path) -> None:
    db = tmp_path / "slom.db"
    engine = ScriptedEngine(token_latency_s=0.3)  # the answer call takes ~0.6s: time to kill
    with slim_serve(engine) as base:
        store = Store(db)
        inp = AgentInput(question="17 pens at 23 each plus a 4 fee?")
        await store.submit("crash", WORKFLOW_NAME, TypeAdapter(AgentInput).dump_json(inp).decode())

        first = start_worker(db, base, tmp_path / "worker1.log")
        try:
            # describe_tools, llm, tool:calculator are journaled => the 2nd model call is running
            await wait_for(store, "crash", timeout_s=60, until=lambda r: r.status == "running")
            deadline = time.monotonic() + 60
            while engine.calls < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert [s.name for s in await store.steps("crash")] == [
                "describe_tools",
                "llm",
                "tool:calculator",
            ]
        finally:
            first.kill()  # SIGKILL / TerminateProcess: no finally blocks, no lease release
            first.wait(10)
        killed_at = time.monotonic()

        second = start_worker(db, base, tmp_path / "worker2.log")
        try:
            run = await wait_for(store, "crash", timeout_s=60)
        finally:
            second.terminate()
            second.wait(10)
        recovery_s = time.monotonic() - killed_at
        steps = await store.steps("crash")
        await store.close()

    assert run.status == "completed", (tmp_path / "worker2.log").read_text()
    assert run.attempts == 2
    result = TypeAdapter(AgentResult).validate_json(run.output or "")
    assert (result.status, result.answer) == ("answered", "395")
    assert [s.name for s in steps] == ["describe_tools", "llm", "tool:calculator", "llm"]

    # The finished model call ran once; only the one in flight at the kill ran twice.
    first_call = [p for p in engine.prompts if "Result of calculator" not in p]
    second_call = [p for p in engine.prompts if "Result of calculator" in p]
    assert len(first_call) == 1 and len(second_call) == 2
    assert second_call[0] == second_call[1]  # the resumed worker rebuilt the identical prompt
    assert recovery_s >= LEASE_S  # it had to wait for the dead worker's lease to expire
