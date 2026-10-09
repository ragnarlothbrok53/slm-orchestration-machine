"""Synthetic workflows with known costs, so measurements isolate the engine from the model."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

from sof.contracts import LlmRequest, LlmResponse
from sof.engine import RetryPolicy, RunContext, Workflow, step_info
from sof.llm import LlmClient


@dataclass
class StepsInput:
    steps: int
    sleep_s: float = 0.0
    log_path: str = ""  # if set, every real execution of a step appends a line here
    max_tokens: int = 16


@dataclass
class Timings:
    """Filled in-process by the instrumented workflow (benchmark-only side channel)."""

    step_s: list[float] = field(default_factory=list)
    replayed_s: list[float] = field(default_factory=list)


TIMINGS = Timings()


async def _work(inp: StepsInput) -> int:
    if inp.sleep_s:
        await asyncio.sleep(inp.sleep_s)
    if inp.log_path:
        # One line per *execution* (not per journaled step): duplicates reveal re-execution.
        line = f"{step_info().idempotency_key} {step_info().attempt} {time.time():.6f}\n"
        fd = os.open(inp.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    return 1


async def steps_workflow(ctx: RunContext, inp: StepsInput) -> int:
    total = 0
    for _ in range(inp.steps):
        t0 = time.perf_counter()
        total += await ctx.step("work", _work, inp)
        (TIMINGS.replayed_s if ctx.replaying else TIMINGS.step_s).append(time.perf_counter() - t0)
    return total


STEPS = Workflow.define("steps", steps_workflow)


def llm_chain(llm: LlmClient, retry: RetryPolicy) -> Workflow[StepsInput, int]:
    """``steps`` sequential model calls per run: a model-bound workload with no parsing."""

    async def complete(req: LlmRequest) -> LlmResponse:
        return await llm.complete(req)

    async def chain(ctx: RunContext, inp: StepsInput) -> int:
        tokens = 0
        for i in range(inp.steps):
            req = LlmRequest(prompt=f"{ctx.run_id} step {i}", max_tokens=inp.max_tokens)
            resp = await ctx.step("llm", complete, req, retry=retry)
            tokens += resp.completion_tokens
        return tokens

    return Workflow.define("llm_chain", chain)
