"""SLOM against the real slim-serve server (v0.1.0) over a real socket.

slim-serve runs in-process with a scripted engine, so everything except the model is real: its
scheduler, admission control, SSE framing, cancellation and metrics. These tests pin down the
contract between the two services, not just each side's idea of it.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import httpx
import pytest
from helpers import sample_value, wait_for
from loguru import logger
from pydantic import TypeAdapter
from slm_server import ScriptedEngine, slim_serve

from slom.agent import WORKFLOW_NAME
from slom.config import Settings
from slom.contracts import AgentInput, AgentResult
from slom.metrics import Metrics
from slom.store import Store
from slom.worker import agent_worker

pytestmark = pytest.mark.anyio


async def submit(store: Store, run_id: str, **kw: object) -> None:
    inp = AgentInput(question="17 pens at 23 each plus a 4 fee?", **kw)  # type: ignore[arg-type]
    await store.submit(run_id, WORKFLOW_NAME, TypeAdapter(AgentInput).dump_json(inp).decode())


def sof_settings(tmp_path: Path, base: str, **kw: object) -> Settings:
    return Settings(db_path=str(tmp_path / "slom.db"), llm_base_url=base, poll_interval_s=0.01, **kw)  # type: ignore[arg-type]


async def test_agent_end_to_end_with_correlated_request_ids(tmp_path: Path) -> None:
    server_request_ids: list[str] = []
    sink = logger.add(
        lambda m: server_request_ids.append(m.record["extra"]["request_id"]),
        filter=lambda r: r["message"].startswith("POST /v1/completions"),
    )
    engine = ScriptedEngine()
    try:
        with slim_serve(engine) as base:
            settings = sof_settings(tmp_path, base)
            store = Store(settings.db_path)
            await submit(store, "e2e")
            async with agent_worker(settings, store, Metrics()) as worker:
                task = asyncio.create_task(worker.run())
                run = await wait_for(store, "e2e")
                worker.stop()
                await task
            await store.close()
    finally:
        logger.remove(sink)

    assert run.status == "completed", run.error
    result = TypeAdapter(AgentResult).validate_json(run.output or "")
    assert (result.status, result.answer) == ("answered", "395")
    assert result.usage.prompt_tokens > 0 and result.usage.completion_tokens > 0
    # Every model call in the journal can be found in slim-serve's access log, and vice versa.
    journal_ids = [s.llm_request_id for s in result.steps]
    assert journal_ids == ["e2e.1.1", "e2e.3.1"]
    assert server_request_ids == journal_ids


async def test_overload_503s_are_absorbed_by_retries(tmp_path: Path) -> None:
    """Server capacity: 1 replica, no queue. SLOM oversubscribes it 4x on purpose."""
    engine = ScriptedEngine(token_latency_s=0.02)
    metrics = Metrics()
    with slim_serve(engine, replicas=1, max_queue_size=0) as base:
        settings = sof_settings(tmp_path, base, llm_concurrency=4, max_concurrent_runs=4)
        store = Store(settings.db_path)
        for i in range(4):
            await submit(store, f"r{i}")
        async with agent_worker(settings, store, metrics) as worker:
            task = asyncio.create_task(worker.run())
            runs = [await wait_for(store, f"r{i}", timeout_s=60) for i in range(4)]
            worker.stop()
            await task
        server_metrics = httpx.get(f"{base}/metrics").text
        await store.close()

    assert [r.status for r in runs] == ["completed"] * 4
    rejected = re.search(r'slm_rejected_total\{reason="queue_full"\} (\S+)', server_metrics)
    assert rejected is not None and float(rejected.group(1)) > 0  # the server did shed load
    retries = sample_value(
        metrics, "sof_step_retries_total", {"step": "llm", "error": "RetryableError"}
    )
    assert retries == float(rejected.group(1))  # ...and every rejection was retried
    assert engine.calls == 8  # but each model call ran exactly once


async def test_cancelling_a_run_stops_generation_on_the_server(tmp_path: Path) -> None:
    engine = ScriptedEngine(token_latency_s=0.01, max_text=" x")  # 1000 tokens = ~10s
    with slim_serve(engine) as base:
        settings = sof_settings(tmp_path, base, lease_s=0.3)
        store = Store(settings.db_path)
        await submit(store, "c1", max_tokens=1000)
        async with agent_worker(settings, store, Metrics()) as worker:
            task = asyncio.create_task(worker.run())
            while engine.tokens < 5:
                await asyncio.sleep(0.01)
            await store.request_cancel("c1")
            run = await wait_for(store, "c1")
            t_cancelled = time.monotonic()
            worker.stop()
            await task
        # slim-serve notices the closed stream and frees the replica.
        while time.monotonic() - t_cancelled < 3:
            m = httpx.get(f"{base}/metrics").text
            if 'slm_requests_total{outcome="cancelled"} 1.0' in m:
                break
            await asyncio.sleep(0.05)
        await store.close()

    assert run.status == "cancelled"
    assert 'slm_requests_total{outcome="cancelled"} 1.0' in m
    assert engine.tokens < 300  # nowhere near the 1000 requested
