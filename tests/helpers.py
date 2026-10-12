"""Test doubles shared across modules."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence

from slom.agent import AgentSteps, agent_workflow
from slom.contracts import LlmRequest, LlmResponse
from slom.engine import RetryPolicy, Worker, step_info
from slom.llm import LlmClient
from slom.metrics import Metrics
from slom.store import TERMINAL, Run, Store
from slom.tools import BUILTIN_TOOLS, ToolRegistry

FAST_RETRY = RetryPolicy(max_attempts=3, initial_s=0.01, max_s=0.05)


class ScriptedLlm(LlmClient):
    """Returns canned replies in order; records every prompt it was sent."""

    def __init__(self, replies: Sequence[str | Exception | asyncio.Event]) -> None:
        # No super().__init__(): there is no HTTP client to build (its SSL setup is slow).
        self.base_url = "scripted"
        self.metrics = Metrics()
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def ready(self) -> bool:
        return True

    async def complete(self, req: LlmRequest) -> LlmResponse:
        self.prompts.append(req.prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, asyncio.Event):  # block until set (or cancelled)
            await reply.wait()
            reply = "{}"
        return LlmResponse(reply, "stop", 10, 5, 0.001, 0.002, step_info().request_id)


def make_worker(
    store: Store,
    llm: LlmClient,
    *,
    metrics: Metrics | None = None,
    tools: ToolRegistry | None = None,
    **worker_kwargs: object,
) -> Worker:
    metrics = metrics or Metrics()
    steps = AgentSteps(llm, tools or ToolRegistry(BUILTIN_TOOLS), metrics)
    wf = agent_workflow(steps, llm_retry=FAST_RETRY, tool_retry=FAST_RETRY)
    opts: dict[str, object] = {"poll_interval_s": 0.01, "lease_s": 3.0, **worker_kwargs}
    return Worker(store, [wf], metrics=metrics, **opts)  # type: ignore[arg-type]


async def wait_for(
    store: Store,
    run_id: str,
    until: Callable[[Run], bool] = lambda r: r.status in TERMINAL,
    timeout_s: float = 10.0,
) -> Run:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = await store.get(run_id)
        if run is not None and until(run):
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach the expected state; last: {run}")


async def run_to_completion(worker: Worker, store: Store, run_id: str) -> Run:
    task = asyncio.create_task(worker.run())
    try:
        return await wait_for(store, run_id)
    finally:
        worker.stop()
        await task


def sample_value(metrics: Metrics, name: str, labels: dict[str, str] | None = None) -> float:
    value = metrics.registry.get_sample_value(name, labels or {})
    return 0.0 if value is None else value
