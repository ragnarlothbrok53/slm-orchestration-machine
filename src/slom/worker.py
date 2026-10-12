"""Assemble a worker from settings, and run one as a process (``slom worker``)."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

from loguru import logger
from prometheus_client import start_http_server

from .agent import AgentSteps, agent_workflow
from .config import Settings
from .engine import RetryPolicy, Worker
from .llm import LlmClient
from .logs import configure_logging
from .metrics import Metrics
from .store import Store
from .tools import BUILTIN_TOOLS, Tool, ToolRegistry


@asynccontextmanager
async def agent_worker(
    settings: Settings,
    store: Store,
    metrics: Metrics,
    tools: Iterable[Tool] = BUILTIN_TOOLS,
) -> AsyncIterator[Worker]:
    """A worker that runs the agent workflow, with its LLM client closed on exit."""
    llm = LlmClient(
        settings.llm_base_url,
        timeout_s=settings.llm_timeout_s,
        concurrency=settings.llm_concurrency,
        metrics=metrics,
    )
    steps = AgentSteps(llm, ToolRegistry(tools), metrics)
    llm_retry = RetryPolicy(max_attempts=settings.llm_max_attempts, initial_s=1.0, max_s=30.0)
    worker = Worker(
        store,
        [agent_workflow(steps, llm_retry=llm_retry)],
        metrics=metrics,
        max_concurrent_runs=settings.max_concurrent_runs,
        lease_s=settings.lease_s,
        poll_interval_s=settings.poll_interval_s,
        max_run_attempts=settings.max_run_attempts,
        shutdown_grace_s=settings.shutdown_grace_s,
    )
    try:
        await wait_for_llm(llm)
        yield worker
    finally:
        await llm.aclose()


async def wait_for_llm(llm: LlmClient, log_every_s: float = 10.0) -> None:
    """Don't claim runs before the model is loaded: they would only burn retries meanwhile."""
    waited = 0.0
    while not await llm.ready():
        if waited % log_every_s == 0:
            logger.info("waiting for the LLM server to be ready", url=llm.base_url)
        await asyncio.sleep(1.0)
        waited += 1.0


def install_stop_handlers(worker: Worker) -> None:
    """SIGINT/SIGTERM stop claiming and drain. ``signal.signal`` works on Windows too."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: loop.call_soon_threadsafe(worker.stop))


async def serve_worker(settings: Settings, tools: Iterable[Tool] = BUILTIN_TOOLS) -> None:
    configure_logging(settings.log_level, settings.log_json)
    metrics = Metrics()
    if settings.metrics_port:
        start_http_server(settings.metrics_port, registry=metrics.registry)
        logger.info("metrics on :{port}/metrics", port=settings.metrics_port)
    store = Store(settings.db_path, synchronous=settings.db_synchronous)
    try:
        async with agent_worker(settings, store, metrics, tools) as worker:
            install_stop_handlers(worker)
            await worker.run()
    finally:
        await store.close()
